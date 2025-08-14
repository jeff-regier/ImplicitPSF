#!/usr/bin/env python3
"""
Visualize DES single epoch PSF data from HDF5 file
"""

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def visualize_psf_data(h5_file_path):
    """Create visual summary of DES PSF data"""

    with h5py.File(h5_file_path, "r") as f:
        # Load data
        cutouts = f["cutouts"][:]
        magnitudes = f["magnitude"][:]
        ra = f["ra"][:]
        dec = f["dec"][:]
        bandpass = [b.decode() if isinstance(b, bytes) else str(b) for b in f["bandpass"][:]]

        # Get metadata
        field_name = f.attrs["field_name"]
        num_stars = f.attrs["num_stars"]
        patch_size = f.attrs["patch_size"]

        print(f"Visualizing {field_name}: {num_stars} stars, {patch_size}x{patch_size} cutouts")

        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))

        # 1. Overview: Show first 16 PSF cutouts
        plt.subplot(2, 3, 1)
        n_show = min(16, len(cutouts))
        grid_size = int(np.ceil(np.sqrt(n_show)))

        combined = np.zeros((grid_size * patch_size, grid_size * patch_size))

        for i in range(n_show):
            row = i // grid_size
            col = i % grid_size
            y_start = row * patch_size
            y_end = (row + 1) * patch_size
            x_start = col * patch_size
            x_end = (col + 1) * patch_size

            # Normalize each cutout for display
            cutout = cutouts[i]
            if cutout.max() > cutout.min():
                cutout_norm = (cutout - cutout.min()) / (cutout.max() - cutout.min())
            else:
                cutout_norm = cutout

            combined[y_start:y_end, x_start:x_end] = cutout_norm

        plt.imshow(combined, cmap="viridis", origin="lower")
        plt.title(f"{field_name}: First {n_show} PSF Cutouts\n(Normalized for display)")
        plt.colorbar(shrink=0.6)

        # 2. Magnitude distribution
        plt.subplot(2, 3, 2)
        plt.hist(magnitudes, bins=20, alpha=0.7, edgecolor="black")
        plt.xlabel("Magnitude (r-band)")
        plt.ylabel("Count")
        plt.title(f"Magnitude Distribution\nMean: {magnitudes.mean():.2f}")
        plt.grid(True, alpha=0.3)

        # 3. Sky position
        plt.subplot(2, 3, 3)
        plt.scatter(ra, dec, c=magnitudes, cmap="plasma", alpha=0.8, s=50)
        plt.xlabel("RA (degrees)")
        plt.ylabel("Dec (degrees)")
        plt.title("Sky Positions")
        cb = plt.colorbar()
        cb.set_label("Magnitude")
        plt.grid(True, alpha=0.3)

        # 4. Band distribution
        plt.subplot(2, 3, 4)
        unique_bands = list(set(bandpass))
        band_counts = [bandpass.count(band) for band in unique_bands]
        plt.bar(
            range(len(unique_bands)),
            band_counts,
            color=["red", "green", "blue", "orange", "purple"][: len(unique_bands)],
        )
        plt.xlabel("Band")
        plt.ylabel("Count")
        plt.title("Observations by Band")
        plt.xticks(
            range(len(unique_bands)),
            [band[:10] + "..." if len(band) > 10 else band for band in unique_bands],
            rotation=45,
        )

        # 5. Flux distribution
        plt.subplot(2, 3, 5)
        mean_flux = np.mean(cutouts, axis=(1, 2))
        peak_flux = np.max(cutouts, axis=(1, 2))

        plt.scatter(magnitudes, mean_flux, alpha=0.6, label="Mean flux", s=30)
        plt.scatter(magnitudes, peak_flux, alpha=0.6, label="Peak flux", s=30)
        plt.xlabel("Magnitude")
        plt.ylabel("Flux (counts)")
        plt.title("Flux vs Magnitude")
        plt.legend()
        plt.yscale("log")
        plt.grid(True, alpha=0.3)

        # 6. Individual PSF examples
        plt.subplot(2, 3, 6)
        # Show 4 example PSFs
        examples = np.linspace(0, len(cutouts) - 1, 4, dtype=int)

        fig2, axes = plt.subplots(2, 2, figsize=(8, 8))
        axes = axes.flatten()

        for i, idx in enumerate(examples):
            ax = axes[i]
            cutout = cutouts[idx]

            # Normalize for display
            if cutout.max() > cutout.min():
                cutout_norm = (cutout - cutout.min()) / (cutout.max() - cutout.min())
            else:
                cutout_norm = cutout

            im = ax.imshow(cutout_norm, cmap="viridis", origin="lower")
            ax.set_title(f"Star {idx}\nMag: {magnitudes[idx]:.2f}\nBand: {bandpass[idx][:15]}")
            plt.colorbar(im, ax=ax, shrink=0.6)

        plt.tight_layout()

        # Save individual PSF examples
        output_dir = Path("psf_visualizations")
        output_dir.mkdir(exist_ok=True)

        psf_file = output_dir / f"{field_name}_individual_psfs.png"
        plt.savefig(psf_file, dpi=150, bbox_inches="tight")
        print(f"Saved individual PSFs: {psf_file}")
        plt.close()

        # Back to main figure
        plt.figure(fig.number)

        # Remove the empty subplot 6
        plt.subplot(2, 3, 6)
        plt.text(
            0.5,
            0.5,
            f"✅ {field_name}\n\n{num_stars} single epoch\nPSF cutouts\n\nReal DES data\nfrom SIA service\n\nSaved to PNG files",
            ha="center",
            va="center",
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
        )
        plt.axis("off")

        plt.tight_layout()

        # Save main figure
        main_file = output_dir / f"{field_name}_overview.png"
        plt.savefig(main_file, dpi=150, bbox_inches="tight")
        print(f"Saved overview: {main_file}")
        plt.close()

        return main_file, psf_file


# Main execution
if __name__ == "__main__":
    print("=" * 50)
    print("DES SINGLE EPOCH PSF VISUALIZATION")
    print("=" * 50)

    h5_dir = Path("des_single_epoch_psf_data")
    h5_files = list(h5_dir.glob("*.h5"))

    if len(h5_files) == 0:
        print("No HDF5 files found in des_single_epoch_psf_data/")
        exit(1)

    print(f"Found {len(h5_files)} HDF5 files to visualize")

    output_dir = Path("psf_visualizations")
    output_dir.mkdir(exist_ok=True)

    for h5_file in h5_files:
        print(f"\nProcessing: {h5_file.name}")
        try:
            main_file, psf_file = visualize_psf_data(h5_file)
            print(f"  ✅ Created visualizations")
        except Exception as e:
            print(f"  ❌ Error: {e}")

    print(f"\n📁 All visualizations saved to: psf_visualizations/")
    print("🎉 DES single epoch PSF visualization complete!")
