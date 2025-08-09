#!/usr/bin/env python3
"""
Example script showing how to load and use PSF star data from HDF5 files.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np


def load_psf_stars(filepath: Path) -> Tuple[Dict, List[Dict]]:
    """
    Load PSF star data from HDF5 file.

    Returns:
        metadata: File-level metadata dict
        stars: List of star dictionaries with cutouts and metadata
    """
    stars = []

    with h5py.File(filepath, "r") as f:
        # Load file metadata
        metadata = dict(f.attrs)

        # Load each star
        star_groups = [key for key in f.keys() if key.startswith("star_")]
        for star_key in sorted(star_groups):
            star_group = f[star_key]

            # Load star metadata
            star_data = dict(star_group.attrs)

            # Load cutouts for each band
            cutouts = {}
            for dataset_key in star_group.keys():
                if dataset_key.startswith("cutout_"):
                    band = dataset_key.replace("cutout_", "")
                    cutouts[band] = star_group[dataset_key][:]

            star_data["cutouts"] = cutouts
            stars.append(star_data)

    return metadata, stars


def demonstrate_usage():
    """Demonstrate how to use the PSF star data."""
    # Load data from the first available file
    psf_dir = Path("psf_stars")
    hdf5_files = list(psf_dir.glob("*.h5"))

    if not hdf5_files:
        print("No HDF5 files found in psf_stars/ directory")
        return

    filepath = hdf5_files[0]
    print(f"Loading PSF stars from: {filepath}")

    metadata, stars = load_psf_stars(filepath)

    print(f"\nFile metadata:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")

    print(f"\nLoaded {len(stars)} stars")

    # Show some example usage
    if stars:
        star = stars[0]
        print(f"\nFirst star:")
        print(f"  Position: ({star['x_center']:.1f}, {star['y_center']:.1f})")
        print(f"  Magnitude: {star['magnitude']:.2f}")
        print(f"  Available bands: {list(star['cutouts'].keys())}")

        # Show cutout shapes and value ranges
        for band, cutout in star["cutouts"].items():
            print(f"  {band}-band: {cutout.shape}, range [{cutout.min():.1f}, {cutout.max():.1f}]")

    # Create a visualization showing stars by magnitude
    if len(stars) >= 12:
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        fig.suptitle(f"PSF Stars by Magnitude (r-band) - {filepath.stem}", fontsize=16)

        # Sort stars by magnitude
        sorted_stars = sorted(stars, key=lambda s: s["magnitude"])

        for i in range(12):
            star = sorted_stars[i]
            row = i // 4
            col = i % 4
            ax = axes[row, col]

            if "r" in star["cutouts"]:
                cutout = star["cutouts"]["r"]
                im = ax.imshow(cutout, origin="lower", cmap="viridis")

                ax.set_title(
                    f"mag={star['magnitude']:.2f}\n"
                    f"({star['x_center']:.0f}, {star['y_center']:.0f})",
                    fontsize=10,
                )
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            else:
                ax.text(0.5, 0.5, "No r-band", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"mag={star['magnitude']:.2f}")

            ax.axis("off")

        plt.tight_layout()
        output_name = f"{filepath.stem}_by_magnitude.png"
        plt.savefig(output_name, dpi=150, bbox_inches="tight")
        print(f"\nSaved magnitude visualization to: {output_name}")

    # Demonstrate creating a multi-band RGB image
    if stars and all(band in stars[0]["cutouts"] for band in ["g", "r", "i"]):
        print(f"\nCreating RGB composite for brightest star...")

        # Use the brightest star (first in sorted list)
        star = min(stars, key=lambda s: s["magnitude"])

        # Get g, r, i cutouts
        g_band = star["cutouts"]["g"]
        r_band = star["cutouts"]["r"]
        i_band = star["cutouts"]["i"]

        # Create RGB composite (simple linear scaling)
        def scale_image(img, percentiles=(1, 99)):
            """Scale image to 0-1 range using percentile clipping."""
            p_low, p_high = np.percentile(img, percentiles)
            img_scaled = np.clip((img - p_low) / (p_high - p_low), 0, 1)
            return img_scaled

        rgb_image = np.stack(
            [
                scale_image(i_band),  # Red channel
                scale_image(r_band),  # Green channel
                scale_image(g_band),  # Blue channel
            ],
            axis=2,
        )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        # Show r-band grayscale
        ax1.imshow(r_band, origin="lower", cmap="viridis")
        ax1.set_title(f'r-band (mag={star["magnitude"]:.2f})')
        ax1.axis("off")

        # Show RGB composite
        ax2.imshow(rgb_image, origin="lower")
        ax2.set_title("RGB Composite (i-r-g)")
        ax2.axis("off")

        plt.suptitle(f"Brightest PSF Star from {filepath.stem}", fontsize=14)
        plt.tight_layout()

        output_name = f"{filepath.stem}_rgb_example.png"
        plt.savefig(output_name, dpi=150, bbox_inches="tight")
        print(f"Saved RGB composite to: {output_name}")

    # Print some summary statistics
    magnitudes = [star["magnitude"] for star in stars]
    positions = [(star["x_center"], star["y_center"]) for star in stars]

    print(f"\nSummary statistics:")
    print(f"  Magnitude range: {min(magnitudes):.2f} to {max(magnitudes):.2f}")
    print(f"  Median magnitude: {np.median(magnitudes):.2f}")
    print(
        f"  Spatial coverage: x=[{min(pos[0] for pos in positions):.0f}, {max(pos[0] for pos in positions):.0f}], "
        f"y=[{min(pos[1] for pos in positions):.0f}, {max(pos[1] for pos in positions):.0f}]"
    )


if __name__ == "__main__":
    demonstrate_usage()
