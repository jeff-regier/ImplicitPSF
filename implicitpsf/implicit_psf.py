import math

import pytorch_lightning as pl
import torch
from einops import rearrange
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm as apply_spectral_norm

from implicitpsf.blend import blend_chi2, sample_grid


def stamp_offsets(positions, patch_size, oversample=1):
    """Continuous (du, dv) offsets of stamp sample centers from each star's center.

    Stamps are extracted with corner = round(center) - patch_size // 2, so the sample
    at native pixel (i, j) has center offset (j - half - fx, i - half - fy), where
    (fx, fy) = center - round(center). With oversample > 1, each native pixel is
    subdivided into oversample^2 samples covering the same extent.

    Args:
        positions: (batch, n_stars, 2) star centers (x, y) in CCD pixel coordinates
        patch_size: native stamp width in pixels
        oversample: samples per native pixel along each axis

    Returns:
        offsets: (batch, n_stars, (patch_size * oversample) ** 2, 2) offsets in pixels
    """
    grid = sample_grid(patch_size, oversample, positions.device, positions.dtype)
    subpixel = positions - positions.round()
    return grid - subpixel.unsqueeze(2)


class Sin(nn.Module):
    """SIREN sinusoidal activation sin(omega_0 * x) (Sitzmann et al. 2020)."""

    def __init__(self, omega_0):
        super().__init__()
        self.omega_0 = omega_0

    def forward(self, x):
        return torch.sin(self.omega_0 * x)


def siren_init_(linear, in_features, omega_0, first):
    """Sitzmann SIREN weight/bias init for a Linear feeding a Sin activation."""
    bound = 1.0 / in_features if first else (6.0 / in_features) ** 0.5 / omega_0
    with torch.no_grad():
        linear.weight.uniform_(-bound, bound)
        if linear.bias is not None:
            linear.bias.uniform_(-bound, bound)


class PSFDecoder(nn.Module):
    """Implicit PSF decoder: intensity at continuous offsets from a star center.

    With film=True, per-star features modulate every hidden layer multiplicatively
    (FiLM). Additive conditioning alone cannot produce angle-dependent profiles —
    elliptical PSFs need the feature vector to gate coordinate channels.
    """

    N_HARMONICS = 5  # angular harmonics: 1, cos(t), sin(t), cos(2t), sin(2t)

    def __init__(  # noqa: C901 — flat setup of the decoder layers
        self,
        feature_dim,
        patch_size,
        decoder_dim=128,
        n_freqs=8,
        use_features=True,
        film=False,
        diagonal_coords=False,
        polar_coords=False,
        siren=False,
        siren_omega=30.0,
        rff_sigma=None,
        analytic_core=False,
        activation="relu",
        spectral_norm=False,
        decoder_residual=False,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.activation = activation
        self.decoder_residual = decoder_residual
        self.n_freqs = n_freqs
        self.use_features = use_features
        self.film = film
        self.diagonal_coords = diagonal_coords
        self.polar_coords = polar_coords
        self.siren = siren
        self.siren_omega = siren_omega
        self.rff_sigma = rff_sigma
        if rff_sigma is not None:
            # tuned-sigma random Fourier features (Tancik et al. 2020): Gaussian radial
            # frequencies B ~ N(0, sigma^2) replace the fixed dyadic ladder. One bandwidth
            # knob sigma sets the spectral content directly, avoiding both core
            # under-resolution (small n_freqs) and aliasing (large dyadic max-frequency).
            self.register_buffer("rff_freqs", torch.randn(n_freqs) * rff_sigma)
        self.analytic_core = analytic_core
        if analytic_core:
            if not use_features:
                raise ValueError("analytic_core requires use_features (core params from context)")
            # Gaussian core (log-width, log-amplitude) predicted from the attended context,
            # added to the decoder output: the sharp central cusp the MLP under-resolves is
            # then correct by construction and the network models only the broader residual.
            self.core_head = nn.Linear(feature_dim, 2)
            # Start the core NEGLIGIBLE (amp ~ e^-4 ~ 0.02, width ~1 px) so the model begins
            # at the no-core (baseline) solution and grows a small core, rather than starting
            # with a large core it must tame -- a large initial core optimizes very poorly
            # (val started ~12 and plateaued ~3.2, vs ~2.2 for the other candidates).
            self.core_head.weight.data.mul_(0.1)
            self.core_head.bias.data[0] = 0.0  # log_sigma -> ~1 px core width
            self.core_head.bias.data[1] = -4.0  # log_amp   -> tiny initial core

        if polar_coords:
            # radial Fourier features times angular harmonics: ellipticity is spin-2,
            # so cos(2t)/sin(2t) terms make BOTH e1 and e2 first-order learnable
            # (cartesian-separable features learn only the axis-aligned component)
            coord_dim = 2 * n_freqs * self.N_HARMONICS
        else:
            # sin/cos per axis per frequency; with diagonal_coords the 45-degree-rotated
            # axes are featurized too — separable u/v features make e1 (axis-aligned
            # elongation) learnable but starve e2, which needs u*v cross-terms
            n_axes = 4 if diagonal_coords else 2
            coord_dim = 2 * n_axes * n_freqs
        def make_act():  # GeLU, ReLU, or SIREN sine — one factory for every hidden activation
            if siren:
                return Sin(siren_omega)
            return nn.GELU() if activation == "gelu" else nn.ReLU(inplace=True)

        def lin(in_dim, out_dim):  # spectral-normalized Linear when enabled (stabilizes training)
            layer = nn.Linear(in_dim, out_dim)
            return apply_spectral_norm(layer) if spectral_norm else layer

        self.coord_embedding = nn.Sequential(
            lin(coord_dim, decoder_dim),
            make_act(),
            lin(decoder_dim, decoder_dim),
        )
        if siren:  # SIREN init: first layer spreads input freqs, hidden keeps sin in range
            siren_init_(self.coord_embedding[0], coord_dim, siren_omega, first=True)
            siren_init_(self.coord_embedding[2], decoder_dim, siren_omega, first=False)
        if use_features:
            self.feature_projection = lin(feature_dim, decoder_dim)

        if film and use_features:
            self.film_hidden = nn.ModuleList(lin(decoder_dim, decoder_dim) for _ in range(3))
            self.film_generators = nn.ModuleList(
                lin(feature_dim, 2 * decoder_dim) for _ in range(3)
            )
            self.film_head = nn.Sequential(lin(decoder_dim, 1), nn.Softplus())
            if siren:
                for layer in self.film_hidden:
                    siren_init_(layer, decoder_dim, siren_omega, first=False)
        else:
            self.pixel_mlp = nn.Sequential(
                lin(decoder_dim, decoder_dim),
                make_act(),
                lin(decoder_dim, decoder_dim),
                make_act(),
                lin(decoder_dim, 1),
                nn.Softplus(),  # strictly positive intensity
            )
            if siren:
                siren_init_(self.pixel_mlp[0], decoder_dim, siren_omega, first=False)
                siren_init_(self.pixel_mlp[2], decoder_dim, siren_omega, first=False)

    def _radial_freqs(self, device, dtype):
        """Frequency vector for the coordinate encoding: a fixed Gaussian RFF buffer when
        rff_sigma is set, else the dyadic ladder 2^k."""
        if self.rff_sigma is not None:
            return self.rff_freqs.to(device=device, dtype=dtype)
        return 2.0 ** torch.arange(self.n_freqs, device=device, dtype=dtype)

    def _fourier_features(self, offsets):
        normalized = offsets / (self.patch_size // 2)
        if self.polar_coords:
            u, v = normalized[..., 0], normalized[..., 1]
            radius = torch.sqrt(u * u + v * v + 1e-12)
            theta = torch.atan2(v, u)
            freqs = self._radial_freqs(offsets.device, offsets.dtype)
            radial = radius.unsqueeze(-1) * (torch.pi * freqs)
            radial = torch.cat([torch.sin(radial), torch.cos(radial)], dim=-1)
            harmonics = torch.stack(
                [
                    torch.ones_like(theta),
                    torch.cos(theta),
                    torch.sin(theta),
                    torch.cos(2 * theta),
                    torch.sin(2 * theta),
                ],
                dim=-1,
            )
            features = radial.unsqueeze(-1) * harmonics.unsqueeze(-2)
            return rearrange(features, "b s n f h -> b s n (f h)")
        if self.diagonal_coords:
            u, v = normalized[..., 0], normalized[..., 1]
            rotated = torch.stack([(u + v), (u - v)], dim=-1) / math.sqrt(2.0)
            normalized = torch.cat([normalized, rotated], dim=-1)
        freqs = self._radial_freqs(offsets.device, offsets.dtype)
        angles = normalized.unsqueeze(-1) * (torch.pi * freqs)
        angles = rearrange(angles, "b s n c f -> b s n (c f)")
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

    def forward(self, offsets, features):
        """Evaluate PSF intensity at continuous offsets.

        Args:
            offsets: (batch, n_stars, n_samples, 2) offsets from star centers in pixels
            features: (batch, n_stars, feature_dim) per-star conditioning, or None

        Returns:
            intensities: (batch, n_stars, n_samples) strictly positive raw intensities
        """
        hidden = self.coord_embedding(self._fourier_features(offsets))
        if self.use_features:
            hidden = hidden + self.feature_projection(features).unsqueeze(2)

        if self.film and self.use_features:
            for layer, generator in zip(self.film_hidden, self.film_generators, strict=True):
                scale, shift = generator(features).unsqueeze(2).chunk(2, dim=-1)
                pre = layer(hidden)
                if self.siren:
                    activated = torch.sin(self.siren_omega * pre)
                elif self.activation == "gelu":
                    activated = nn.functional.gelu(pre)
                else:
                    activated = torch.relu(pre)
                out = activated * (1.0 + scale) + shift
                hidden = hidden + out if self.decoder_residual else out
            base = self.film_head(hidden).squeeze(-1)
        else:
            base = self.pixel_mlp(hidden).squeeze(-1)
        if self.analytic_core:
            return base + self._gaussian_core(offsets, features)
        return base

    def _gaussian_core(self, offsets, features):
        """Context-predicted Gaussian core intensity I(r) = amp * exp(-r^2 / 2 sigma^2),
        with sigma (pixels) clamped to a compact range so it captures only the sharp peak."""
        log_sigma = self.core_head(features)[..., 0:1]  # (b, s, 1)
        log_amp = self.core_head(features)[..., 1:2]
        sigma = log_sigma.exp().clamp(0.3, 3.0)
        r2 = (offsets**2).sum(dim=-1)  # (b, s, n)
        return log_amp.exp() * torch.exp(-0.5 * r2 / sigma**2)


class ImplicitPSF(pl.LightningModule):
    """Attention-based implicit PSF field for single-CCD exposures.

    Every star is a query (position, color, flux); keys and values combine position
    encodings with encoded cutouts of the context stars. The decoder renders the PSF
    at arbitrary continuous offsets, so stamps can be produced at any resolution.
    """

    def __init__(
        self,
        patch_size,
        ccd_width=2048.0,
        ccd_height=4096.0,
        hidden_dim=256,
        decoder_dim=128,
        n_heads=8,
        learning_rate=1e-4,
        weight_decay=1e-6,
        use_attention=True,
        context_dropout_max=0.5,
        n_attn_layers=1,
        decoder_film=False,
        diagonal_coords=False,
        polar_coords=False,
        n_freqs=8,
        siren=False,
        siren_omega=30.0,
        rff_sigma=None,
        analytic_core=False,
        activation="relu",
        spectral_norm=False,
        decoder_residual=False,
        loss_mode="single",
        blend_radius=22.0,
        blend_k_max=4,
        galaxy_mode="exclude",
        chi2_cap=None,
        blend_max_targets=None,
        point_source_context=True,
        cnn_encoder=False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.point_source_context = point_source_context
        self.cnn_encoder = cnn_encoder

        self.patch_size = patch_size
        self.ccd_size = (ccd_width, ccd_height)
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.use_attention = use_attention
        self.context_dropout_max = context_dropout_max
        self.n_attn_layers = n_attn_layers
        self.loss_mode = loss_mode
        self.blend_radius = blend_radius
        self.blend_k_max = blend_k_max
        self.galaxy_mode = galaxy_mode
        self.chi2_cap = chi2_cap
        self.blend_max_targets = blend_max_targets
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # color and log-flux of the queried object (flux enables brighter-fatter)
        self.scalar_embedding = nn.Linear(2, hidden_dim)

        if cnn_encoder:  # spatial inductive bias for filtering local contamination from the stamps
            self.image_encoder = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 3, padding=1, stride=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 96, 3, padding=1, stride=2),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(96, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
        else:
            self.image_encoder = nn.Sequential(
                nn.Linear(patch_size * patch_size, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.LayerNorm(hidden_dim),
            )

        # n_attn_layers == 1 is the original single-layer additive architecture and
        # must keep its module names so earlier checkpoints load; >= 2 stacks
        # residual cross-attention blocks with feed-forward layers (much stronger
        # at extracting per-exposure seeing from context stamps)
        if self.use_attention and n_attn_layers == 1:
            self.attention_layer = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
            self.attention_norm = nn.LayerNorm(hidden_dim)
        elif self.use_attention:
            self.attention_layers = nn.ModuleList(
                nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
                for _ in range(n_attn_layers)
            )
            self.attention_norms = nn.ModuleList(
                nn.LayerNorm(hidden_dim) for _ in range(n_attn_layers)
            )
            self.ffn_layers = nn.ModuleList(
                nn.Sequential(
                    nn.Linear(hidden_dim, 2 * hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(2 * hidden_dim, hidden_dim),
                )
                for _ in range(n_attn_layers)
            )
            self.ffn_norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(n_attn_layers))

        self.psf_decoder = PSFDecoder(
            hidden_dim,
            patch_size,
            decoder_dim=decoder_dim,
            n_freqs=n_freqs,
            use_features=self.use_attention,
            film=decoder_film,
            diagonal_coords=diagonal_coords,
            polar_coords=polar_coords,
            siren=siren,
            siren_omega=siren_omega,
            rff_sigma=rff_sigma,
            analytic_core=analytic_core,
            activation=activation,
            spectral_norm=spectral_norm,
            decoder_residual=decoder_residual,
        )

    def _sinusoidal_position_encoding(self, positions):
        """Sinusoidal encodings of 2D CCD positions, normalized per axis.

        Args:
            positions: (batch, n_stars, 2) positions (x, y) in CCD pixel coordinates

        Returns:
            encodings: (batch, n_stars, hidden_dim)
        """
        ccd_size = torch.tensor(self.ccd_size, device=positions.device, dtype=positions.dtype)
        normalized_pos = positions / ccd_size

        num_bands = self.hidden_dim // 4  # sin/cos for x and y per band
        band_idx = torch.arange(num_bands, device=positions.device, dtype=positions.dtype)
        freq_bands = torch.pow(1000.0, -band_idx / num_bands)

        x_scaled = normalized_pos[:, :, 0:1] * freq_bands
        y_scaled = normalized_pos[:, :, 1:2] * freq_bands

        return torch.cat(
            [torch.sin(x_scaled), torch.cos(x_scaled), torch.sin(y_scaled), torch.cos(y_scaled)],
            dim=-1,
        )

    def attend(self, cutouts, positions, colors, fluxes, context_mask):
        """Attention features for every star, attending only to context stars.

        Args:
            cutouts: (batch, n_stars, patch, patch) observed stamps
            positions: (batch, n_stars, 2) star centers in CCD coordinates
            colors: (batch, n_stars) per-star colors
            fluxes: (batch, n_stars) per-star fluxes
            context_mask: (batch, n_stars) bool, True = may be attended to as a key

        Returns:
            features: (batch, n_stars, hidden_dim)
        """
        n_stars = cutouts.shape[1]
        if not context_mask.any(dim=1).all():
            raise ValueError("every exposure needs at least one context star")

        pos_encodings = self._sinusoidal_position_encoding(positions)
        scalars = torch.stack([colors, torch.log1p(fluxes.clamp(min=0.0))], dim=-1)
        scalar_features = self.scalar_embedding(scalars)

        # scale-normalize stamps; absolute brightness enters via the log-flux scalar
        cutouts_flat = rearrange(cutouts, "b s h w -> b s (h w)")
        scale = cutouts_flat.abs().sum(dim=-1, keepdim=True) + 1e-8
        normed = cutouts_flat / scale
        if self.cnn_encoder:
            b, s = cutouts.shape[:2]
            images = normed.reshape(b * s, 1, self.patch_size, self.patch_size)
            image_features = self.image_encoder(images).reshape(b, s, self.hidden_dim)
        else:
            image_features = self.image_encoder(normed)

        queries = pos_encodings + scalar_features
        keys = pos_encodings + scalar_features + image_features
        self_mask = torch.eye(n_stars, device=cutouts.device, dtype=torch.bool)

        if self.n_attn_layers == 1:
            attn_output, _ = self.attention_layer(
                queries,
                keys,
                keys,
                attn_mask=self_mask,
                key_padding_mask=~context_mask,
                need_weights=False,
            )
            return self.attention_norm(attn_output)

        hidden = queries
        layers = zip(
            self.attention_layers,
            self.attention_norms,
            self.ffn_layers,
            self.ffn_norms,
            strict=True,
        )
        for attention, attn_norm, ffn, ffn_norm in layers:
            attn_output, _ = attention(
                hidden,
                keys,
                keys,
                attn_mask=self_mask,
                key_padding_mask=~context_mask,
                need_weights=False,
            )
            hidden = attn_norm(hidden + attn_output)
            hidden = ffn_norm(hidden + ffn(hidden))
        return hidden

    def forward(self, cutouts, positions, colors, fluxes, context_mask, oversample=1):
        """Render unit-sum PSF stamps at every star's position.

        Args:
            cutouts: (batch, n_stars, patch, patch) observed stamps (context information)
            positions: (batch, n_stars, 2) star centers in CCD coordinates
            colors: (batch, n_stars) per-star colors (0.0 where unknown)
            fluxes: (batch, n_stars) per-star fluxes (0.0 marks padding)
            context_mask: (batch, n_stars) bool, True = may be attended to as a key
            oversample: samples per native pixel along each axis

        Returns:
            psfs: (batch, n_stars, patch * oversample, patch * oversample) stamps,
                normalized to unit sum at native resolution
        """
        features = None
        if self.use_attention:
            features = self.attend(cutouts, positions, colors, fluxes, context_mask)

        offsets = stamp_offsets(positions, self.patch_size, oversample)
        intensities = self.psf_decoder(offsets, features)
        normalized = intensities / intensities.sum(dim=-1, keepdim=True) * oversample**2
        side = self.patch_size * oversample
        return rearrange(normalized, "b s (h w) -> b s h w", h=side, w=side)

    def get_loss(self, batch):
        """Weighted chi-square loss; dispatches on loss_mode.

        "single": clean isolated stars only, scalar amplitude per star (original).
        "blend": targets include blended stars; each cutout is a GLS-amplitude sum of
        the target plus detected star neighbors (see blend.py).
        """
        cutouts = batch["cutouts"]
        fluxes = batch["flux"]
        star_types = batch["star_types"]

        context_mask = fluxes > 0
        if self.point_source_context:  # only point sources (stars) inform the PSF
            is_point = (star_types == 0) | (star_types == 1) | (star_types == 5)
            context_mask = context_mask & is_point
        if self.training and self.context_dropout_max > 0:
            gated = context_mask  # pre-dropout context (gating may shrink it to a few)
            keep_frac = 1.0 - torch.rand(1, device=cutouts.device) * self.context_dropout_max
            kept = torch.rand_like(fluxes) < keep_frac
            context_mask = context_mask & kept
            empty = ~context_mask.any(dim=1)  # dropout must not empty an exposure's context
            context_mask[empty] = gated[empty]

        if self.loss_mode == "blend":
            chi2 = blend_chi2(
                self,
                batch,
                context_mask,
                self.blend_radius,
                self.blend_k_max,
                self.galaxy_mode,
                self.chi2_cap,
                self.blend_max_targets,
            )
            return chi2.mean()

        if self.loss_mode == "clean":
            return self._clean_target_loss(cutouts, batch, fluxes, star_types, context_mask)

        clean_mask = (star_types == 0) & (fluxes > 0)
        if not clean_mask.any():
            raise ValueError("batch contains no clean stars")

        pred_psfs = self(cutouts, batch["positions"], batch["colors"], fluxes, context_mask)

        weights = batch["valid_pixels"].float() / batch["variance"]
        observed = rearrange(cutouts[clean_mask], "n h w -> n (h w)")
        model = rearrange(pred_psfs[clean_mask], "n h w -> n (h w)")
        w = rearrange(weights[clean_mask], "n h w -> n (h w)")

        n_valid = w.count_nonzero(dim=-1).float()
        if (n_valid == 0).any():
            raise ValueError("clean star with no valid pixels")

        amplitude = (w * observed * model).sum(dim=-1) / (w * model.square()).sum(dim=-1)
        residuals = observed - amplitude.unsqueeze(-1) * model
        chi2_per_star = (w * residuals.square()).sum(dim=-1) / n_valid
        return chi2_per_star.mean()

    def _clean_target_loss(self, cutouts, batch, fluxes, star_types, context_mask):
        """Contamination correction: supervise the predicted PSF against the KNOWN clean truth PSF
        (batch['clean_psf'], from add_clean_psf_target.py) instead of the contaminated cutout, so the
        net learns to output the clean PSF from contaminated context. Flat valid-pixel weighting: the
        squared error already emphasizes the bright core, and full weight on the wings stops
        over-concentration via dropped wing flux (intensity-weighting zeroed the wings -> EE@r2 +25%)."""
        clean_mask = (star_types == 0) & (fluxes > 0)
        if not clean_mask.any():
            raise ValueError("batch contains no clean stars")
        pred_psfs = self(cutouts, batch["positions"], batch["colors"], fluxes, context_mask)
        pred = rearrange(pred_psfs[clean_mask], "n h w -> n (h w)")
        pred = pred / pred.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        target = rearrange(batch["clean_psf"][clean_mask], "n h w -> n (h w)")
        w = rearrange(batch["valid_pixels"].float()[clean_mask], "n h w -> n (h w)")
        denom = w.sum(dim=-1)
        if (denom == 0).any():
            raise ValueError("clean star with no valid pixels")
        chi2_per_star = (w * (pred - target).square()).sum(dim=-1) / denom
        return chi2_per_star.mean()
