#!/usr/bin/env python3
"""
Extract clean star cutouts from DES tiles for PSF fitting.

This script:
1. Discovers available DES tiles in the cache directory
2. For each tile, loads the catalog and filters for good PSF stars
3. Extracts 32x32 pixel cutouts around selected stars
4. Saves cutouts and metadata to HDF5 files (one per tile)

Filtering criteria:
- Brightness: mag < mag_limit (default 19.0)
- Star classification: CLASS_STAR > 0.9 or EXTENDED_CLASS_COADD == 0
- Morphology: |SPREAD_MODEL| < 0.005 (compact objects)
- Signal-to-noise: SNR > 20
- Not saturated: FLAGS == 0 (no bad flags)
- Isolation: min_separation pixels from other bright objects
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
from astropy.io import fits
from scipy.spatial.distance import cdist


class DESStarExtractor:
    def __init__(self, cache_dir: str = "/data/scratch/des", output_dir: str = "./psf_stars"):
        self.cache_dir = Path(cache_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Filtering parameters
        self.mag_limit = 19.0
        self.min_separation = 64.0  # pixels
        self.patch_size = 32
        self.snr_threshold = 20.0
        self.spread_limit = 0.005
        self.class_star_threshold = 0.9

        # Image bands to process
        self.bands = ["g", "r", "i", "z", "Y"]

    def discover_tiles(self) -> List[str]:
        """Discover all available DES tiles in the cache directory."""
        if not self.cache_dir.exists():
            raise FileNotFoundError(f"Cache directory {self.cache_dir} does not exist")

        tiles = set()

        # Look for FITS files and extract tile names
        for fits_file in self.cache_dir.rglob("*.fits*"):
            filename = fits_file.name

            # DES tile format: DESXXXX-YYYY_rNNNNpNN_BAND.fits.fz
            if filename.startswith("DES") and "_r" in filename:
                # Extract tile name (everything before the first underscore)
                tile_name = filename.split("_")[0]
                tiles.add(tile_name)

        return sorted(list(tiles))

    def load_catalog(self, tile_name: str) -> Optional[np.ndarray]:
        """Load the main catalog for a tile."""
        # Try different catalog types and locations
        for catalog_type in ["main", "magnitude"]:
            patterns = [f"{tile_name}_dr2_{catalog_type}.fits", f"dr2_{catalog_type}.fits"]

            for pattern in patterns:
                # Check in tile subdirectory first, then root cache dir
                cache_paths = [self.cache_dir / tile_name / pattern, self.cache_dir / pattern]

                for cache_path in cache_paths:
                    if cache_path.exists():
                        try:
                            with fits.open(cache_path) as hdul:
                                catalog_data = hdul[1].data if len(hdul) > 1 else hdul[0].data
                            return catalog_data
                        except Exception as e:
                            print(f"    Error loading catalog {cache_path}: {e}")
                            continue

        return None

    def load_image(self, tile_name: str, band: str) -> Optional[np.ndarray]:
        """
        Load image data for a tile and band.

        Note: This method explicitly excludes "_nobkg" files and uses the regular
        DES coadd images, which are the primary data products from the DES pipeline.
        """
        # Try different processing versions and locations
        for version in ["r4931p01", "r4939p01", "r4931p02", "r4907p01"]:
            filename = f"{tile_name}_{version}_{band}.fits.fz"
            # Check in tile subdirectory first, then root cache dir
            cache_paths = [self.cache_dir / tile_name / filename, self.cache_dir / filename]

            for cache_path in cache_paths:
                # Skip _nobkg files - counterintuitively, they contain background!
                if "_nobkg" in cache_path.name:
                    continue

                if cache_path.exists():
                    try:
                        with fits.open(cache_path) as hdul:
                            # DES coadd images are typically in the first extension for .fits.fz files
                            image_data = hdul[1].data if len(hdul) > 1 else hdul[0].data
                        return image_data
                    except Exception as e:
                        print(f"    Error loading image {cache_path}: {e}")
                        continue

        # If we get here, no image was found - provide helpful debug info
        print(f"    Debug: Could not find {band}-band image for {tile_name}")
        print(f"    Tried versions: {['r4931p01', 'r4939p01', 'r4931p02', 'r4907p01']}")

        # Check what files actually exist for this tile
        tile_dir = self.cache_dir / tile_name
        if tile_dir.exists():
            # Show all band files, highlighting which ones we'd use
            actual_files = [f.name for f in tile_dir.glob(f"*_{band}.fits*")]
            if actual_files:
                clean_files = [f for f in actual_files if "_nobkg" not in f]
                print(f"    Found {band}-band files: {actual_files}")
                print(f"    Would use (non-nobkg): {clean_files}")
            else:
                print(f"    No {band}-band files found in {tile_dir}")

        return None

    def filter_psf_stars(self, catalog: np.ndarray) -> np.ndarray:
        """Filter catalog for good PSF stars."""
        print(f"    Starting with {len(catalog)} catalog objects")

        # Start with all objects
        mask = np.ones(len(catalog), dtype=bool)

        # Filter by magnitude (using r-band)
        mag_field = "MAG_AUTO_R" if "MAG_AUTO_R" in catalog.dtype.names else "WAVG_MAG_PSF_R"
        if mag_field in catalog.dtype.names:
            mag_mask = (catalog[mag_field] < self.mag_limit) & np.isfinite(catalog[mag_field])
            mask &= mag_mask
            print(
                f"    After magnitude filter ({mag_field} < {self.mag_limit}): {np.sum(mask)} objects"
            )

        # Filter for star-like objects
        if "CLASS_STAR_R" in catalog.dtype.names:
            star_mask = catalog["CLASS_STAR_R"] > self.class_star_threshold
            mask &= star_mask
            print(
                f"    After star classification (CLASS_STAR_R > {self.class_star_threshold}): {np.sum(mask)} objects"
            )
        elif "EXTENDED_CLASS_COADD" in catalog.dtype.names:
            star_mask = catalog["EXTENDED_CLASS_COADD"] == 0
            mask &= star_mask
            print(
                f"    After extended classification (EXTENDED_CLASS_COADD == 0): {np.sum(mask)} objects"
            )

        # Filter for compact morphology
        if "SPREAD_MODEL_R" in catalog.dtype.names:
            spread_mask = np.abs(catalog["SPREAD_MODEL_R"]) < self.spread_limit
            mask &= spread_mask
            print(
                f"    After morphology filter (|SPREAD_MODEL_R| < {self.spread_limit}): {np.sum(mask)} objects"
            )

        # Filter for high signal-to-noise
        flux_field = "FLUX_AUTO_R" if "FLUX_AUTO_R" in catalog.dtype.names else "WAVG_FLUX_PSF_R"
        fluxerr_field = (
            "FLUXERR_AUTO_R" if "FLUXERR_AUTO_R" in catalog.dtype.names else "WAVG_FLUXERR_PSF_R"
        )

        if flux_field in catalog.dtype.names and fluxerr_field in catalog.dtype.names:
            # Avoid division by zero and ensure positive flux
            flux_ok = (catalog[fluxerr_field] > 0) & (catalog[flux_field] > 0)
            mask &= flux_ok

            snr = np.where(flux_ok, catalog[flux_field] / catalog[fluxerr_field], 0)
            snr_mask = snr > self.snr_threshold
            mask &= snr_mask
            print(f"    After SNR filter (SNR > {self.snr_threshold}): {np.sum(mask)} objects")

        # Filter for clean photometry (no bad flags)
        if "FLAGS_R" in catalog.dtype.names:
            clean_mask = catalog["FLAGS_R"] == 0
            mask &= clean_mask
            print(f"    After flags filter (FLAGS_R == 0): {np.sum(mask)} objects")

        # Apply initial filters
        filtered_stars = catalog[mask]

        if len(filtered_stars) == 0:
            print("    No stars passed initial filters")
            return filtered_stars

        # Apply isolation filter
        isolated_stars = self.apply_isolation_filter(filtered_stars)
        print(
            f"    After isolation filter (min_sep > {self.min_separation} px): {len(isolated_stars)} stars"
        )

        return isolated_stars

    def apply_isolation_filter(self, stars: np.ndarray) -> np.ndarray:
        """Remove stars that are too close to each other."""
        if len(stars) <= 1:
            return stars

        # Get positions
        x_coords = stars["XWIN_IMAGE"]
        y_coords = stars["YWIN_IMAGE"]
        positions = np.column_stack([x_coords, y_coords])

        # Get magnitudes for prioritization
        mag_field = "MAG_AUTO_R" if "MAG_AUTO_R" in stars.dtype.names else "WAVG_MAG_PSF_R"
        magnitudes = stars[mag_field]

        # Sort by magnitude (brightest first)
        sort_indices = np.argsort(magnitudes)
        sorted_stars = stars[sort_indices]
        sorted_positions = positions[sort_indices]

        # Keep track of which stars to keep
        keep_mask = np.ones(len(sorted_stars), dtype=bool)

        # For each star, check if it's too close to any brighter star we're keeping
        for i in range(len(sorted_stars)):
            if not keep_mask[i]:
                continue

            # Calculate distances to all other stars
            distances = cdist([sorted_positions[i]], sorted_positions[i + 1 :])[0]

            # Mark nearby fainter stars for removal
            close_indices = np.where(distances < self.min_separation)[0]
            keep_mask[i + 1 :][close_indices] = False

        return sorted_stars[keep_mask]

    def extract_cutout(
        self, image: np.ndarray, x_center: float, y_center: float
    ) -> Optional[np.ndarray]:
        """Extract a 32x32 cutout around the given center coordinates."""
        half_size = self.patch_size // 2

        # Convert to integer pixel coordinates
        x_int = int(round(x_center))
        y_int = int(round(y_center))

        # Calculate patch boundaries
        x_min = x_int - half_size
        x_max = x_int + half_size
        y_min = y_int - half_size
        y_max = y_int + half_size

        # Check if patch is mostly within image bounds
        img_h, img_w = image.shape
        if (
            x_min < -half_size // 2
            or x_max > img_w + half_size // 2
            or y_min < -half_size // 2
            or y_max > img_h + half_size // 2
        ):
            return None  # Too close to edge

        # Create output patch
        patch = np.zeros((self.patch_size, self.patch_size), dtype=image.dtype)

        # Calculate overlap region
        patch_x_start = max(0, -x_min)
        patch_x_end = patch_x_start + min(img_w, x_max) - max(0, x_min)
        patch_y_start = max(0, -y_min)
        patch_y_end = patch_y_start + min(img_h, y_max) - max(0, y_min)

        img_x_start = max(0, x_min)
        img_x_end = min(img_w, x_max)
        img_y_start = max(0, y_min)
        img_y_end = min(img_h, y_max)

        # Copy the overlapping region
        if (
            patch_x_end > patch_x_start
            and patch_y_end > patch_y_start
            and img_x_end > img_x_start
            and img_y_end > img_y_start
        ):
            patch[patch_y_start:patch_y_end, patch_x_start:patch_x_end] = image[
                img_y_start:img_y_end, img_x_start:img_x_end
            ]

        return patch

    def process_tile(self, tile_name: str) -> bool:
        """Process a single tile and save star cutouts."""
        print(f"\nProcessing tile: {tile_name}")

        # Check if output already exists
        output_file = self.output_dir / f"{tile_name}_psf_stars.h5"
        if output_file.exists():
            print(f"  Output file already exists: {output_file}")
            return True

        # Load catalog
        catalog = self.load_catalog(tile_name)
        if catalog is None:
            print(f"  No catalog found for tile {tile_name}")
            return False

        # Filter for PSF stars
        psf_stars = self.filter_psf_stars(catalog)
        if len(psf_stars) == 0:
            print(f"  No suitable PSF stars found in {tile_name}")
            return False

        # Load images for all bands
        images = {}
        for band in self.bands:
            image = self.load_image(tile_name, band)
            if image is not None:
                images[band] = image
                print(f"  Loaded {band}-band image: {image.shape}")
            else:
                print(f"  Warning: Could not load {band}-band image")

        if not images:
            print(f"  No images loaded for tile {tile_name}")
            return False

        # Extract cutouts
        star_cutouts = []
        star_metadata = []

        for i, star in enumerate(psf_stars):
            x_center = star["XWIN_IMAGE"]
            y_center = star["YWIN_IMAGE"]

            # Extract cutouts for all available bands
            cutouts_this_star = {}
            for band, image in images.items():
                cutout = self.extract_cutout(image, x_center, y_center)
                if cutout is not None:
                    cutouts_this_star[band] = cutout

            # Only keep stars where we got cutouts for at least 3 bands
            if len(cutouts_this_star) >= 3:
                star_cutouts.append(cutouts_this_star)

                # Store metadata
                mag_field = (
                    "MAG_AUTO_R" if "MAG_AUTO_R" in psf_stars.dtype.names else "WAVG_MAG_PSF_R"
                )
                metadata = {
                    "star_id": i,
                    "x_center": x_center,
                    "y_center": y_center,
                    "magnitude": star[mag_field],
                    "ra": star["RA"],
                    "dec": star["DEC"],
                }

                # Add additional fields if available
                if "CLASS_STAR_R" in psf_stars.dtype.names:
                    metadata["class_star"] = star["CLASS_STAR_R"]
                if "SPREAD_MODEL_R" in psf_stars.dtype.names:
                    metadata["spread_model"] = star["SPREAD_MODEL_R"]

                star_metadata.append(metadata)

        print(f"  Successfully extracted cutouts for {len(star_cutouts)} stars")

        if not star_cutouts:
            print(f"  No valid cutouts extracted for {tile_name}")
            return False

        # Save to HDF5 file
        self.save_star_data(tile_name, star_cutouts, star_metadata, output_file)
        print(f"  Saved star data to: {output_file}")

        return True

    def save_star_data(
        self, tile_name: str, star_cutouts: List[Dict], star_metadata: List[Dict], output_file: Path
    ):
        """Save star cutouts and metadata to HDF5 file."""
        with h5py.File(output_file, "w") as f:
            # Save tile-level metadata
            f.attrs["tile_name"] = tile_name
            f.attrs["num_stars"] = len(star_cutouts)
            f.attrs["patch_size"] = self.patch_size
            f.attrs["bands"] = [b.encode("utf-8") for b in self.bands]
            f.attrs["creation_time"] = time.time()

            # Save filtering parameters
            f.attrs["mag_limit"] = self.mag_limit
            f.attrs["min_separation"] = self.min_separation
            f.attrs["snr_threshold"] = self.snr_threshold
            f.attrs["spread_limit"] = self.spread_limit

            # Create groups for each star
            for i, (cutouts, metadata) in enumerate(zip(star_cutouts, star_metadata)):
                star_group = f.create_group(f"star_{i:04d}")

                # Save metadata
                for key, value in metadata.items():
                    star_group.attrs[key] = value

                # Save cutouts for each band
                for band, cutout in cutouts.items():
                    star_group.create_dataset(
                        f"cutout_{band}", data=cutout, compression="gzip", compression_opts=6
                    )

    def run(self, tiles: Optional[List[str]] = None, max_tiles: Optional[int] = None):
        """Process all or specified tiles."""
        if tiles is None:
            tiles = self.discover_tiles()

        if max_tiles is not None:
            tiles = tiles[:max_tiles]

        print(f"Found {len(tiles)} tiles to process")
        print(f"Output directory: {self.output_dir}")
        print(f"Filtering parameters:")
        print(f"  Magnitude limit: {self.mag_limit}")
        print(f"  Minimum separation: {self.min_separation} pixels")
        print(f"  Patch size: {self.patch_size}x{self.patch_size}")
        print(f"  SNR threshold: {self.snr_threshold}")
        print(f"  Spread limit: {self.spread_limit}")

        successful = 0
        failed = 0
        start_time = time.time()

        for i, tile_name in enumerate(tiles):
            print(f"\n[{i+1}/{len(tiles)}] Processing {tile_name}...")

            try:
                if self.process_tile(tile_name):
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"  Error processing {tile_name}: {e}")
                failed += 1

            # Progress update
            elapsed = time.time() - start_time
            if i > 0:
                avg_time = elapsed / (i + 1)
                remaining = avg_time * (len(tiles) - i - 1)
                print(f"  Progress: {i+1}/{len(tiles)} tiles, " f"ETA: {remaining/60:.1f} minutes")

        total_time = time.time() - start_time
        print(f"\nProcessing complete!")
        print(f"  Successful: {successful} tiles")
        print(f"  Failed: {failed} tiles")
        print(f"  Total time: {total_time/60:.1f} minutes")


def main():
    parser = argparse.ArgumentParser(description="Extract PSF stars from DES tiles")
    parser.add_argument("--cache-dir", default="/data/scratch/des", help="DES cache directory")
    parser.add_argument(
        "--output-dir", default="./psf_stars", help="Output directory for HDF5 files"
    )
    parser.add_argument(
        "--mag-limit", type=float, default=19.0, help="Magnitude limit for star selection"
    )
    parser.add_argument(
        "--min-separation",
        type=float,
        default=64.0,
        help="Minimum separation between stars (pixels)",
    )
    parser.add_argument(
        "--max-tiles", type=int, help="Maximum number of tiles to process (for testing)"
    )
    parser.add_argument("--tiles", nargs="+", help="Specific tiles to process")

    args = parser.parse_args()

    # Create extractor
    extractor = DESStarExtractor(cache_dir=args.cache_dir, output_dir=args.output_dir)
    extractor.mag_limit = args.mag_limit
    extractor.min_separation = args.min_separation

    # Run processing
    extractor.run(tiles=args.tiles, max_tiles=args.max_tiles)


if __name__ == "__main__":
    main()
