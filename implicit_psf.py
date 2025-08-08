import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class ImplicitPSF(pl.LightningModule):
    """Local Implicit Image Function for PSF modeling with PyTorch Lightning.

    Maps (x,y) coordinates to pixel-integrated PSF values.
    """

    def __init__(
        self,
        image_size,
        background_level=0.0,
        hidden_dim=256,
        n_layers=8,
        learning_rate=1e-4,
        weight_decay=1e-6,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Model parameters
        self.image_size = image_size
        self.background_level = background_level
        input_dim = 2  # just original coords

        # Build MLP layers
        layers = []
        current_dim = input_dim

        for i in range(n_layers):
            next_dim = hidden_dim if i < n_layers - 1 else 1
            layers.append(nn.Linear(current_dim, next_dim))
            if i < n_layers - 1:
                layers.append(nn.ReLU(inplace=True))
            current_dim = next_dim

        self.mlp = nn.Sequential(*layers)

        # Training parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Precompute pixel coordinate grid (image_size, image_size, 2)
        pixel_coords = torch.arange(self.image_size, dtype=torch.float32)
        y_coords, x_coords = torch.meshgrid(pixel_coords, pixel_coords, indexing="ij")
        coords_grid = torch.stack([x_coords, y_coords], dim=-1)
        self.register_buffer("coords_grid", coords_grid)

    def _create_pixel_coordinates(self, star_positions):
        """Create pixel center coordinates for each image in the batch.

        Args:
            star_positions: (batch_size, 1, 2) star positions

        Returns:
            coords: (batch_size, image_size, image_size, 2) pixel coordinates relative to star
        """
        batch_size = star_positions.shape[0]

        # Expand precomputed coordinate grid for batch dimension
        coords = repeat(self.coords_grid, "h w xy -> b h w xy", b=batch_size)

        # star_positions are already in astronomical coords
        # No conversion needed - they're already in the right coordinate system
        star_array_coords = rearrange(star_positions, "b 1 xy -> b 1 1 xy")

        # Relative coordinates from star position
        relative_coords = coords - star_array_coords

        return relative_coords

    def forward(self, positions):
        """Forward pass.

        Args:
            positions: (batch_size, 1, 2) star positions

        Returns:
            (batch_size, image_size, image_size) tensor of predicted PSF images (without background)
        """
        batch_size = positions.shape[0]

        # Create coordinate grid relative to star positions
        coords = self._create_pixel_coordinates(positions)

        # Flatten coordinates for network input
        coords_input = rearrange(coords, "b h w d -> (b h w) d")

        # Get predictions from MLP
        pred_values = self.mlp(coords_input)

        # Ensure positivity using softplus
        pred_values = F.softplus(pred_values)

        # Reshape predictions back to image format
        pred_images = rearrange(
            pred_values, "(b h w) 1 -> b h w", b=batch_size, h=self.image_size, w=self.image_size
        )

        return pred_images

    def _generic_step(self, batch, batch_idx, stage):
        """Generic step for both training and validation.

        Args:
            batch: Input batch
            batch_idx: Batch index
            stage: Either 'train' or 'val'

        Returns:
            loss: Computed MSE loss
        """
        images, fluxes, positions = batch

        # Get predictions from LIIF
        pred_images = self(positions)

        # Scale predictions by star flux
        star_fluxes = rearrange(fluxes, "b 1 -> b")  # (batch_size,)
        pred_images = pred_images * rearrange(star_fluxes, "b -> b 1 1")

        # Add background level to predictions
        pred_images = pred_images + self.background_level

        # Target images (remove batch and channel dimensions)
        target_images = rearrange(images, "b 1 h w -> b h w")

        # L2 loss
        loss = F.mse_loss(pred_images, target_images)

        # Logging
        self.log(f"{stage}_loss", loss, prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx):
        """Training step."""
        return self._generic_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        return self._generic_step(batch, batch_idx, "val")

    def configure_optimizers(self):
        """Configure optimizer."""
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        return {"optimizer": optimizer, "monitor": "val_loss"}
