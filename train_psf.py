#!/usr/bin/env python3

import pytorch_lightning as pl
import torch

from data_generation import StarDataModule
from implicit_psf import ImplicitPSF


def main():
    pl.seed_everything(42)

    # Create data module for PSF training (single centered stars)
    datamodule = StarDataModule(
        n_samples=4096 * 4,
        image_size=10000,
        max_sources=256,
        mean_sources=256,
        min_sources=256,
        patch_size=8,
        as_patches=True,
        batch_size=4,
        sigma=0.8,
        seed=42,
        background_intensity=1.0,
        shot_noise=False,
        variable_psf=True,
    )

    # Create model
    model = ImplicitPSF(
        image_size=8,
        full_image_size=10000,  # Full survey image size for position encoding
        background_level=1.0,
        hidden_dim=256,
        n_heads=4,
        learning_rate=3e-4,
        use_attention=True,  # Test attention mode
    )

    # Trainer with checkpointing enabled
    trainer = pl.Trainer(
        max_epochs=20,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=False,
        enable_checkpointing=True,
        default_root_dir="./checkpoints",
        log_every_n_steps=10,
    )

    # Train
    print("Starting training...")
    trainer.fit(model, datamodule)
    print("Training complete!")

    # Save the final model
    final_model_path = "trained_psf_model.ckpt"
    trainer.save_checkpoint(final_model_path)
    print(f"Model saved to {final_model_path}")
    print("Run 'python visualize_psf.py' to generate visualizations with the trained model.")


if __name__ == "__main__":
    main()
