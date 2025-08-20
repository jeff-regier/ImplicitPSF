import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class PSFDecoder(nn.Module):
    """Decodes attention features and star centers into PSF images."""

    def __init__(self, hidden_dim, image_size, use_features=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.image_size = image_size
        self.use_features = use_features

        mlp_input_size = hidden_dim * 2 if use_features else hidden_dim

        self.coord_embedding = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # MLP to generate PSF values
        self.pixel_mlp = nn.Sequential(
            nn.Linear(mlp_input_size, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, image_size**2),
            nn.Softplus(),  # Ensure nonnegative flux
        )

    def forward(self, star_centers, attn_features):
        """Generate PSF images from attention features and star centers."""
        subpixel_offset = (star_centers + 0.5) % 1 - 0.5  # varies within [-0.5, 0.5]
        mlp_input = self.coord_embedding(subpixel_offset)

        if attn_features is not None:
            mlp_input = torch.cat([mlp_input, attn_features], dim=-1)

        pred_psf_flat = self.pixel_mlp(mlp_input)
        return rearrange(
            pred_psf_flat, "b s (h w) -> b s h w", h=self.image_size, w=self.image_size
        )


class ImplicitPSF(pl.LightningModule):
    """Attention-based PSF modeling with PyTorch Lightning.

    Uses multi-head attention to predict each star's PSF based on all other stars in the field.
    """

    def __init__(
        self,
        patch_size,
        image_size,  # Size of full survey image for position encoding
        background_level=0.0,
        hidden_dim=256,
        n_heads=8,
        learning_rate=1e-4,
        weight_decay=1e-6,
        use_attention=True,  # Flag to enable/disable attention
    ):
        super().__init__()
        self.save_hyperparameters()

        # Model parameters
        self.patch_size = patch_size
        self.image_size = image_size
        self.background_level = background_level
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.use_attention = use_attention

        # Sinusoidal position encoding for full image coordinates
        self.pos_encoding_dim = hidden_dim  # Full hidden_dim for position only

        # Image encoder - encode 8x8 star images to hidden dimension
        self.image_encoder = nn.Sequential(
            nn.Flatten(),  # 8x8 = 64 pixels
            nn.Linear(patch_size * patch_size, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
        )

        # Single multi-head attention layer (only if using attention)
        if self.use_attention:
            self.attention_layer = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
            self.attention_norm = nn.LayerNorm(hidden_dim)

        # PSF decoder module
        self.psf_decoder = PSFDecoder(hidden_dim, patch_size, use_features=self.use_attention)

        # Training parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def _sinusoidal_position_encoding(self, positions):
        """Create sinusoidal position encodings for 2D positions.

        Args:
            positions: (batch_size, n_stars, 2) positions in full image coordinates

        Returns:
            pos_encodings: (batch_size, n_stars, pos_encoding_dim) sinusoidal encodings
        """
        # Normalize positions to [0, 1] range
        normalized_pos = positions / self.image_size

        # Create frequency bands
        num_bands = self.pos_encoding_dim // 4  # 4 terms per band (sin/cos for x/y)
        freq_bands = torch.arange(num_bands, device=positions.device, dtype=torch.float32)
        # Scale frequencies appropriately for 10000x10000 images
        # Lower frequencies for better spatial resolution
        freq_bands = torch.pow(1000.0, -freq_bands / num_bands)  # Reduced base from 10000 to 1000

        # Apply frequencies to x and y coordinates
        x_pos = normalized_pos[:, :, 0:1]  # (batch_size, n_stars, 1)
        y_pos = normalized_pos[:, :, 1:2]  # (batch_size, n_stars, 1)

        # Broadcast positions with frequency bands
        x_scaled = x_pos * freq_bands  # (batch_size, n_stars, num_bands)
        y_scaled = y_pos * freq_bands  # (batch_size, n_stars, num_bands)

        # Create sin/cos encodings
        encodings = torch.cat(
            [
                torch.sin(x_scaled),  # (batch_size, n_stars, num_bands)
                torch.cos(x_scaled),  # (batch_size, n_stars, num_bands)
                torch.sin(y_scaled),  # (batch_size, n_stars, num_bands)
                torch.cos(y_scaled),  # (batch_size, n_stars, num_bands)
            ],
            dim=-1,
        )  # (batch_size, n_stars, pos_encoding_dim)

        return encodings

    def forward(self, star_images, star_centers, nonzero_mask=None, clean_mask=None):
        """Forward pass using attention."""
        batch_size, n_stars = star_images.shape[:2]

        if self.use_attention:
            # Attention-based approach: stars attend to each other using image features
            pos_encodings = self._sinusoidal_position_encoding(star_centers)

            # Encode star images to get features
            star_images_flat = rearrange(star_images, "b s h w -> (b s) (h w)")
            image_features = self.image_encoder(
                star_images_flat
            )  # (batch_size * n_stars, hidden_dim)
            image_features = rearrange(image_features, "(b s) d -> b s d", b=batch_size)

            # Combine position and image information for keys and values
            # Queries are still just positions (what each star is asking for)
            # Keys combine position + image (where each star is + what it looks like)
            # Values combine position + image (the information to aggregate)
            combined_features = pos_encodings + image_features  # (batch_size, n_stars, hidden_dim)

            queries = pos_encodings  # (batch_size, n_stars, hidden_dim) - where we're asking from
            keys = combined_features  # (batch_size, n_stars, hidden_dim) - what's available
            values = combined_features  # (batch_size, n_stars, hidden_dim) - what we get

            # Create mask to prevent self-attention and attention to padding sources
            mask = torch.eye(
                n_stars, device=pos_encodings.device, dtype=torch.bool
            )  # (n_stars, n_stars)

            # Also mask out padding sources (zero flux) from being attended to
            if nonzero_mask is not None:
                # Expand nonzero_mask to attention shape: (batch_size, n_stars) -> (n_stars, n_stars)
                # We want to mask out columns (keys) where flux is zero
                padding_mask = ~nonzero_mask[0]  # Assuming same mask across batch
                mask = mask | padding_mask.unsqueeze(0)  # Broadcast to (n_stars, n_stars)

            # Single attention layer with mask
            attn_output, _ = self.attention_layer(queries, keys, values, attn_mask=mask)
            attended_features = self.attention_norm(attn_output)
        else:
            # Non-attention approach: no learned features, just use coordinate-based MLP
            attended_features = None

        # Use PSF decoder to generate images
        return self.psf_decoder(star_centers, attended_features)

    def _generic_step(self, batch, batch_idx, stage):
        star_images, star_fluxes, star_centers, star_types = batch
        nonzero_mask = star_fluxes > 0
        clean_mask = (star_types == 0) & nonzero_mask  # Only clean stars with nonzero flux

        pred_psfs = self(star_images, star_centers, nonzero_mask, clean_mask)
        pred_images = pred_psfs * star_fluxes.unsqueeze(-1).unsqueeze(-1) + self.background_level

        # Only compute loss for clean stars (all other stars are predictors only)
        if clean_mask.any():
            loss = F.mse_loss(pred_images[clean_mask], star_images[clean_mask])
        else:
            loss = torch.zeros(1, device=star_images.device, requires_grad=True)

        # No logging needed - handled by custom training loop
        return loss

    def training_step(self, batch, batch_idx):
        return self._generic_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._generic_step(batch, batch_idx, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        return {"optimizer": optimizer, "monitor": "val_loss"}
