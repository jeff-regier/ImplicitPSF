import math

import pytorch_lightning as pl
import torch
from einops import rearrange
from torch import nn


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
    half = patch_size // 2
    n_samples = patch_size * oversample
    step = 1.0 / oversample
    grid_1d = torch.arange(n_samples, device=positions.device, dtype=positions.dtype)
    grid_1d = grid_1d * step + (step - 1.0) / 2.0 - half
    grid_v, grid_u = torch.meshgrid(grid_1d, grid_1d, indexing="ij")
    grid = torch.stack([grid_u, grid_v], dim=-1).reshape(-1, 2)

    subpixel = positions - positions.round()
    return grid - subpixel.unsqueeze(2)


class PSFDecoder(nn.Module):
    """Implicit PSF decoder: intensity at continuous offsets from a star center.

    With film=True, per-star features modulate every hidden layer multiplicatively
    (FiLM). Additive conditioning alone cannot produce angle-dependent profiles —
    elliptical PSFs need the feature vector to gate coordinate channels.
    """

    N_HARMONICS = 5  # angular harmonics: 1, cos(t), sin(t), cos(2t), sin(2t)

    def __init__(
        self,
        feature_dim,
        patch_size,
        decoder_dim=128,
        n_freqs=8,
        use_features=True,
        film=False,
        diagonal_coords=False,
        polar_coords=False,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.n_freqs = n_freqs
        self.use_features = use_features
        self.film = film
        self.diagonal_coords = diagonal_coords
        self.polar_coords = polar_coords

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
        self.coord_embedding = nn.Sequential(
            nn.Linear(coord_dim, decoder_dim),
            nn.ReLU(inplace=True),
            nn.Linear(decoder_dim, decoder_dim),
        )
        if use_features:
            self.feature_projection = nn.Linear(feature_dim, decoder_dim)

        if film and use_features:
            self.film_hidden = nn.ModuleList(nn.Linear(decoder_dim, decoder_dim) for _ in range(3))
            self.film_generators = nn.ModuleList(
                nn.Linear(feature_dim, 2 * decoder_dim) for _ in range(3)
            )
            self.film_head = nn.Sequential(nn.Linear(decoder_dim, 1), nn.Softplus())
        else:
            self.pixel_mlp = nn.Sequential(
                nn.Linear(decoder_dim, decoder_dim),
                nn.ReLU(inplace=True),
                nn.Linear(decoder_dim, decoder_dim),
                nn.ReLU(inplace=True),
                nn.Linear(decoder_dim, 1),
                nn.Softplus(),  # strictly positive intensity
            )

    def _fourier_features(self, offsets):
        normalized = offsets / (self.patch_size // 2)
        if self.polar_coords:
            u, v = normalized[..., 0], normalized[..., 1]
            radius = torch.sqrt(u * u + v * v + 1e-12)
            theta = torch.atan2(v, u)
            freqs = 2.0 ** torch.arange(self.n_freqs, device=offsets.device, dtype=offsets.dtype)
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
        freqs = 2.0 ** torch.arange(self.n_freqs, device=offsets.device, dtype=offsets.dtype)
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
                hidden = torch.relu(layer(hidden)) * (1.0 + scale) + shift
            return self.film_head(hidden).squeeze(-1)
        return self.pixel_mlp(hidden).squeeze(-1)


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
    ):
        super().__init__()
        self.save_hyperparameters()

        self.patch_size = patch_size
        self.ccd_size = (ccd_width, ccd_height)
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.use_attention = use_attention
        self.context_dropout_max = context_dropout_max
        self.n_attn_layers = n_attn_layers
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # color and log-flux of the queried object (flux enables brighter-fatter)
        self.scalar_embedding = nn.Linear(2, hidden_dim)

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
            use_features=self.use_attention,
            film=decoder_film,
            diagonal_coords=diagonal_coords,
            polar_coords=polar_coords,
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

    def _attend(self, cutouts, positions, colors, fluxes, context_mask):
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
        image_features = self.image_encoder(cutouts_flat / scale)

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
            features = self._attend(cutouts, positions, colors, fluxes, context_mask)

        offsets = stamp_offsets(positions, self.patch_size, oversample)
        intensities = self.psf_decoder(offsets, features)
        normalized = intensities / intensities.sum(dim=-1, keepdim=True) * oversample**2
        side = self.patch_size * oversample
        return rearrange(normalized, "b s (h w) -> b s h w", h=side, w=side)

    def get_loss(self, batch):
        """Weighted chi-square loss over clean stars, amplitude fit per star.

        Batch keys: cutouts, variance, valid_pixels, flux, positions, colors, star_types.
        Cutouts are background-subtracted; the per-star amplitude is solved in closed
        form by weighted least squares, so the loss measures shape, not flux calibration.
        """
        cutouts = batch["cutouts"]
        fluxes = batch["flux"]
        star_types = batch["star_types"]

        context_mask = fluxes > 0
        if self.training and self.context_dropout_max > 0:
            keep_frac = 1.0 - torch.rand(1, device=cutouts.device) * self.context_dropout_max
            kept = torch.rand_like(fluxes) < keep_frac
            context_mask = context_mask & kept

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
