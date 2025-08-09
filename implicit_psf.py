import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class PSFDecoder(nn.Module):
    """Decodes attention features and star centers into PSF images."""

    def __init__(self, hidden_dim, image_size, use_features=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.image_size = image_size
        self.use_features = use_features

        # MLP input size depends on whether we use features
        if use_features:
            mlp_input_size = hidden_dim + 2 + 2  # attention + center + pixel_coords
        else:
            mlp_input_size = 2 + 2  # center + pixel_coords only

        # MLP to generate PSF values
        self.pixel_mlp = nn.Sequential(
            nn.Linear(mlp_input_size, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        # Precompute pixel coordinate grid
        pixel_coords = torch.arange(image_size, dtype=torch.float32)
        y_coords, x_coords = torch.meshgrid(pixel_coords, pixel_coords, indexing="ij")
        coords_grid = torch.stack([x_coords, y_coords], dim=-1)
        self.register_buffer("coords_grid", coords_grid)

    def _create_pixel_coordinates(self, star_positions):
        """Create pixel coordinates relative to star positions."""
        batch_size = star_positions.shape[0]
        coords = repeat(self.coords_grid, "h w xy -> b h w xy", b=batch_size)
        star_array_coords = rearrange(star_positions, "b 1 xy -> b 1 1 xy")
        relative_coords = coords - star_array_coords
        return relative_coords

    def forward(self, target_features, target_centers):
        """Generate PSF images from attention features and star centers."""
        target_positions_for_coords = target_centers.unsqueeze(1)
        coords = self._create_pixel_coordinates(target_positions_for_coords)
        coords_flat = rearrange(coords, "b h w d -> (b h w) d")

        target_centers_expanded = repeat(
            target_centers, "b d -> (b h w) d", h=self.image_size, w=self.image_size
        )

        if target_features is not None:
            # Include target features (attention case)
            target_features_expanded = repeat(
                target_features, "b d -> (b h w) d", h=self.image_size, w=self.image_size
            )
            psf_input = torch.cat(
                [target_features_expanded, target_centers_expanded, coords_flat], dim=-1
            )
            batch_size = target_features.shape[0]
        else:
            # No target features (coordinate-only case)
            psf_input = torch.cat([target_centers_expanded, coords_flat], dim=-1)
            batch_size = target_centers.shape[0]

        pred_values = self.pixel_mlp(psf_input)
        pred_values = F.softplus(pred_values)

        pred_psfs = rearrange(
            pred_values, "(b h w) 1 -> b h w", b=batch_size, h=self.image_size, w=self.image_size
        )

        return pred_psfs


class ImplicitPSF(pl.LightningModule):
    """Attention-based PSF modeling with PyTorch Lightning.

    Uses multi-head attention to predict each star's PSF based on all other stars in the field.
    """

    def __init__(
        self,
        image_size,
        full_image_size=1000,  # Size of full survey image for position encoding
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
        self.image_size = image_size
        self.full_image_size = full_image_size
        self.background_level = background_level
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.use_attention = use_attention

        # Sinusoidal position encoding for full image coordinates
        self.pos_encoding_dim = hidden_dim  # Full hidden_dim for position only

        # Image encoder - encode 8x8 star images to hidden dimension
        self.image_encoder = nn.Sequential(
            nn.Flatten(),  # 8x8 = 64 pixels
            nn.Linear(image_size * image_size, hidden_dim),
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
        self.psf_decoder = PSFDecoder(hidden_dim, image_size, use_features=self.use_attention)

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
        normalized_pos = positions / self.full_image_size

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

    def forward(self, star_images, star_centers):
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

            # Create mask to prevent self-attention (star attending to itself)
            device = pos_encodings.device
            mask = torch.eye(n_stars, device=device, dtype=torch.bool)  # (n_stars, n_stars)

            # Single attention layer with mask
            attn_output, _ = self.attention_layer(queries, keys, values, attn_mask=mask)
            attended_features = self.attention_norm(attn_output)

            # Flatten for PSF decoder
            target_features = rearrange(attended_features, "b s d -> (b s) d")
        else:
            # Non-attention approach: no learned features, just use coordinate-based MLP
            target_features = None

        # Get target centers - just flatten the star_centers
        target_centers = rearrange(star_centers, "b s d -> (b s) d")

        # Use PSF decoder to generate images
        pred_psfs_flat = self.psf_decoder(target_features, target_centers)
        pred_psfs = rearrange(pred_psfs_flat, "(b s) h w -> b s h w", b=batch_size, s=n_stars)

        return pred_psfs

    def _generic_step(self, batch, batch_idx, stage):
        star_images, star_fluxes, star_centers = batch
        pred_psfs = self(star_images, star_centers)
        pred_images = pred_psfs * star_fluxes.unsqueeze(-1).unsqueeze(-1) + self.background_level
        loss = F.mse_loss(pred_images, star_images)
        self.log(f"{stage}_loss", loss, prog_bar=True)
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
