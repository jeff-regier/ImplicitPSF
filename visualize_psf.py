#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from einops import rearrange

from data_generation import Catalog, StarDataModule, StarImageDistribution
from implicit_psf import ImplicitPSF


def visualize_predictions(model, datamodule, device, num_examples=4):
    """Visualize model predictions on validation data."""
    model.eval()

    # Get a batch from validation data
    val_dataloader = datamodule.val_dataloader()
    batch = next(iter(val_dataloader))
    images, fluxes, positions = batch

    # Move to device
    images = images.to(device)
    fluxes = fluxes.to(device)
    positions = positions.to(device)

    # Get predictions (without background)
    with torch.no_grad():
        pred_images = model(positions)
        star_fluxes = rearrange(fluxes, "b 1 -> b")
        pred_images = pred_images * rearrange(star_fluxes, "b -> b 1 1")

    # Move to CPU for plotting
    images_np = images.cpu().numpy()
    pred_images_np = pred_images.cpu().numpy()

    # Subtract background from target images to match model output
    if hasattr(model, "background_level"):
        images_np = images_np - model.background_level

    # Calculate shared vmin/vmax per column (original and predicted share same scale)
    batch_size = images.shape[0]
    num_show = min(num_examples, batch_size)

    # Plot comparisons with colorbars
    fig, axes = plt.subplots(3, num_examples, figsize=(num_examples * 4, 10))

    for i in range(num_show):
        # Calculate MSE for this prediction
        mse = np.mean((images_np[i, 0] - pred_images_np[i]) ** 2)

        # Calculate vmin/vmax for this column (original and predicted share same scale)
        col_vmin = min(images_np[i, 0].min(), pred_images_np[i].min())
        col_vmax = max(images_np[i, 0].max(), pred_images_np[i].max())

        # Calculate residual vmin/vmax for this column
        residual = images_np[i, 0] - pred_images_np[i]
        resid_vmax = np.abs(residual).max()
        resid_vmin = -resid_vmax

        # Original image
        im0 = axes[0, i].imshow(
            images_np[i, 0], cmap="Blues_r", origin="lower", vmin=col_vmin, vmax=col_vmax
        )
        axes[0, i].set_title(f"Original {i+1}")
        axes[0, i].axis("off")
        plt.colorbar(im0, ax=axes[0, i], fraction=0.046, pad=0.04)

        # Predicted image
        im1 = axes[1, i].imshow(
            pred_images_np[i], cmap="Blues_r", origin="lower", vmin=col_vmin, vmax=col_vmax
        )
        axes[1, i].set_title(f"Predicted {i+1}\nMSE: {mse:.2e}")
        axes[1, i].axis("off")
        plt.colorbar(im1, ax=axes[1, i], fraction=0.046, pad=0.04)

        # Residual
        im2 = axes[2, i].imshow(
            residual, cmap="RdBu", origin="lower", vmin=resid_vmin, vmax=resid_vmax
        )
        axes[2, i].set_title(f"Residual {i+1}")
        axes[2, i].axis("off")
        plt.colorbar(im2, ax=axes[2, i], fraction=0.046, pad=0.04)

    plt.tight_layout()
    return fig


def visualize_psf_positions(model, device, image_size=8, sigma=0.8):
    """Visualize PSF at different positions: y=0, x from -0.25 to 0.5."""
    model.eval()

    # Define test positions: y=4 (center), x from 3.75 to 4.5 (within training range)
    # Training data has positions around [3.5-4.5, 3.5-4.5]
    x_positions = [3.75, 4.0, 4.25, 4.5]  # Within training distribution
    y_position = 4.0  # Center

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for i, x_pos in enumerate(x_positions):
        # Create position tensor
        position = torch.tensor([[x_pos, y_position]], dtype=torch.float32).unsqueeze(
            0
        )  # (1, 1, 2)
        position = position.to(device)

        # Get model prediction (without background)
        with torch.no_grad():
            pred_psf = model(position).cpu().numpy()[0]  # (image_size, image_size)

        # Get ground truth PSF
        # Create catalog for this position
        # StarImageDistribution expects positions in image coordinates (0 to image_size)
        # Our positions are already in the right coordinate system [3.5-4.5] for 8x8 images
        img_x = x_pos  # Already in image coordinates
        img_y = y_position  # Already in image coordinates

        # Catalog expects fluxes: (batch, n_stars) and positions: (batch, n_stars, 2)
        star_positions = torch.tensor([[[img_x, img_y]]], dtype=torch.float32)  # (1, 1, 2)
        star_fluxes = torch.tensor([[1.0]], dtype=torch.float32)  # (1, 1)
        catalog = Catalog(fluxes=star_fluxes, positions=star_positions)

        # Create StarImageDistribution for ground truth
        star_dist = StarImageDistribution(
            catalog=catalog,
            image_size=image_size,
            patch_size=image_size,  # Use full image as patch
            sigma=sigma,
            background_intensity=0.0,  # No background for clean PSF visualization
            shot_noise=False,
        )

        images, _patches = star_dist.sample()  # Sample images from the distribution
        true_psf = images.numpy()[0, 0]  # (image_size, image_size) - clean PSF without background

        # Calculate shared scale for this position
        vmin = min(pred_psf.min(), true_psf.min())
        vmax = max(pred_psf.max(), true_psf.max())

        # Plot true PSF
        im_true = axes[0, i].imshow(true_psf, cmap="Blues_r", origin="lower", vmin=vmin, vmax=vmax)
        axes[0, i].set_title(f"True PSF\nx={x_pos:.2f}, y={y_position:.2f}")
        axes[0, i].axis("off")
        plt.colorbar(im_true, ax=axes[0, i], fraction=0.046, pad=0.04)

        # Plot predicted PSF
        im_pred = axes[1, i].imshow(pred_psf, cmap="Blues_r", origin="lower", vmin=vmin, vmax=vmax)
        mse = np.mean((true_psf - pred_psf) ** 2)
        axes[1, i].set_title(f"Predicted PSF\nMSE: {mse:.2e}")
        axes[1, i].axis("off")
        plt.colorbar(im_pred, ax=axes[1, i], fraction=0.046, pad=0.04)

    plt.tight_layout()
    return fig


def load_trained_model(model_path="trained_psf_model.ckpt"):
    """Load a trained model or create a new one for testing."""
    import os

    if model_path and os.path.exists(model_path):
        # Load checkpoint if provided
        model = ImplicitPSF.load_from_checkpoint(model_path)
        print(f"Loaded model from {model_path}")
    else:
        print(
            f"Model file {model_path} not found. Using untrained model (for testing visualization)"
        )
        model = ImplicitPSF(
            image_size=8,
            background_level=1.0,  # Match training configuration
            hidden_dim=256,
            n_layers=6,
            learning_rate=3e-4,
        )

    return model


def main():
    """Main visualization script."""
    pl.seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create data module for getting validation examples
    datamodule = StarDataModule(
        n_samples=2048 * 4,  # Smaller for quick testing
        image_size=8,
        max_sources=1,
        mean_sources=1,
        min_sources=1,
        center_sources=True,
        batch_size=512,
        sigma=0.8,
        seed=42,
        background_intensity=0.0,
        shot_noise=False,
    )

    # Setup datamodule to create datasets
    datamodule.setup()

    # Load model (you can provide a checkpoint path here)
    model = load_trained_model()
    model = model.to(device)

    print("Generating validation predictions visualization...")
    fig1 = visualize_predictions(model, datamodule, device, num_examples=4)
    plt.savefig("validation_predictions.png", dpi=150, bbox_inches="tight")
    print("Saved validation predictions to validation_predictions.png")
    plt.close(fig1)

    print("Generating PSF position comparison...")
    fig2 = visualize_psf_positions(model, device, image_size=8, sigma=0.8)
    plt.savefig("psf_position_comparison.png", dpi=150, bbox_inches="tight")
    print("Saved PSF position comparison to psf_position_comparison.png")
    plt.close(fig2)


if __name__ == "__main__":
    main()
