#!/usr/bin/env python3
"""
Extract PSF star cutouts from locally downloaded DES FITS files using SEP source detection
Processes all downloaded CCD 31 files and uses SEP to identify stars instead of catalog queries
"""

import multiprocessing as mp
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import astroscrappy
import h5py
import numpy as np
import pandas as pd
import sep
import torch
from astropy.io import fits
from astropy.wcs import WCS

# Set SEP pixel stack size to handle large images
sep.set_extract_pixstack(1000000)

# Configuration
CONFIG = {
    "patch_size": 32,
    "stars_per_exposure": 512,  # Fixed number of stars per exposure
    "local_data_dir": "/nfs/turbo/lsa-regier/des",
    "num_processes": 8,  # Number of worker processes
    "verbose": False,  # Global verbosity control
    # SEP source detection parameters
    "detection_threshold": 2.5,  # SNR threshold for detection
    "min_area": 2,  # Minimum area in pixels
    "deblend_nthresh": 32,
    "deblend_cont": 0.005,
    # Star classification parameters
    "fwhm_min": 0.8,  # Minimum FWHM in pixels
    "fwhm_max": 8.0,  # Maximum FWHM in pixels
    "ellipticity_max": 0.3,  # Maximum ellipticity for stars
    "flux_min": 100.0,  # Minimum flux for reliable detection
    "snr_min": 10.0,  # Minimum SNR
    # Blending/isolation parameters
    "min_separation": 16.0,  # Minimum separation between stars (pixels) - relaxed from 24
    "crowding_radius": 12.0,  # Radius to check for nearby sources (pixels) - relaxed from 16
    # Quality filters
    "edge_buffer": 32,  # Pixels from edge to avoid
    "saturation_threshold": 50000,  # Rough saturation level
    "companion_detection_threshold": 4.0,  # Threshold for companion detection in cutouts - relaxed from 3.0
    # Cosmic ray detection parameters
    "cosmic_ray_detection": True,  # Enable cosmic ray detection
    "cosmic_ray_sigclip": 5.0,  # Detection threshold (conservative)
    "cosmic_ray_sigfrac": 0.3,  # Neighboring pixel threshold
    "cosmic_ray_objlim": 5.0,  # Object detection limit
    "cosmic_ray_niter": 4,  # Number of iterations
}


def load_directories_list(directories_file: str = "des_directories.txt") -> list[str]:
    """
    Load directory list and convert to local paths
    """
    print(f"📋 Loading directories from {directories_file}...")

    directories = []
    with open(directories_file) as f:
        for line in f:
            line = line.strip()
            if line and line.startswith("des_dr2://"):
                # Convert des_dr2:// path to local path
                # des_dr2://dr2_se/finalcut/Y5A1/r3515/... -> /nfs/turbo/lsa-regier/des/Y5A1/r3515/...
                # Remove the des_dr2://dr2_se/finalcut/ prefix and keep the rest
                path_suffix = line.replace("des_dr2://dr2_se/finalcut/", "")
                local_path = os.path.join(CONFIG["local_data_dir"], path_suffix)
                directories.append(local_path)

    print(f"✅ Loaded {len(directories)} directories")
    return directories


def process_safe_print(*args, **kwargs):
    """Process-safe print function - multiprocessing handles this automatically"""
    print(*args, **kwargs)


def worker_process_directories(process_id: int, directories_subset: list) -> dict:
    """
    Worker process function to process directories
    Each process processes a subset of directories and writes to its own PyTorch files
    """
    # Disable verbosity for worker processes to reduce output noise
    CONFIG["verbose"] = False

    # Create output directory if it doesn't exist
    output_dir = Path("/data/scratch/regier/sep_des_stars")
    output_dir.mkdir(parents=True, exist_ok=True)

    process_safe_print(
        f"🔥 Process {process_id} started with {len(directories_subset)} directories"
    )

    process_star_data = []
    process_processed_files = 0
    process_successful_files = 0
    process_processed_dirs = 0

    # Target 1GB files (~25,000-30,000 stars based on 32x32 float32 patches + metadata)
    # Each star: 32x32x4 bytes (image) + ~200 bytes (metadata) ≈ 4.3KB per star
    # 1GB / 4.3KB ≈ 250,000 stars, but we'll be conservative and use 200,000
    target_file_size_gb = 1.0
    approx_bytes_per_star = 4300  # Conservative estimate
    target_stars_per_file = int(target_file_size_gb * 1024**3 / approx_bytes_per_star)

    # Track current exposure to avoid splitting
    current_exposure_stars = []
    current_exposure_id = None

    for directory in directories_subset:
        try:
            # Process all FITS files in this directory
            if not os.path.exists(directory):
                continue

            try:
                files = os.listdir(directory)
                dir_fits_files = [f for f in files if f.endswith(".fits.fz") and "_c31_" in f]

                # Removed individual directory processing messages

                for filename in dir_fits_files:
                    fits_path = os.path.join(directory, filename)
                    star_cutouts = process_fits_file_quiet(fits_path)

                    if len(star_cutouts) > 0:
                        # Get exposure ID from first star (all stars from same file have same exposure)
                        file_exposure_id = star_cutouts[0]["exposure_id"]

                        # Check if this is a new exposure
                        if current_exposure_id is None:
                            current_exposure_id = file_exposure_id
                        elif current_exposure_id != file_exposure_id:
                            # New exposure detected - finish current exposure first
                            if len(current_exposure_stars) > 0:
                                process_star_data.extend(current_exposure_stars)
                                current_exposure_stars = []

                            # Check if we should save the current batch
                            if len(process_star_data) >= target_stars_per_file:
                                # Use process_id and sequential batch counter to ensure uniqueness
                                batch_counter = len(
                                    list(
                                        output_dir.glob(
                                            f"des_psf_stars_process_{process_id:02d}_*.h5"
                                        )
                                    )
                                )
                                # Count unique exposures in this batch
                                unique_exposures = set(
                                    star["exposure_id"] for star in process_star_data
                                )
                                exposure_count = len(unique_exposures)
                                output_file = (
                                    output_dir
                                    / f"desstars_process{process_id:02d}_file{batch_counter:03d}_exposures{exposure_count:03d}.pt"
                                )
                                process_safe_print(
                                    f"💾 Process {process_id}: Saving 1GB batch with {len(process_star_data)} stars ({exposure_count} exposures) to {output_file}"
                                )
                                save_batch_pytorch(process_star_data, str(output_file))
                                process_star_data = []

                            # Start new exposure
                            current_exposure_id = file_exposure_id

                        # Add stars to current exposure buffer
                        current_exposure_stars.extend(star_cutouts)
                        process_successful_files += 1

                        # Removed individual file processing messages

                    process_processed_files += 1

                    # Update local counters (reduce lock frequency)
                    # We'll update global stats less frequently

                process_processed_dirs += 1

                # Report progress every 10 directories
                if process_processed_dirs % 10 == 0:
                    progress_pct = (process_processed_dirs / len(directories_subset)) * 100
                    total_buffered = len(process_star_data) + len(current_exposure_stars)
                    buffer_mb = total_buffered * approx_bytes_per_star / (1024**2)
                    process_safe_print(
                        f"📈 Process {process_id}: {progress_pct:.1f}% complete ({process_processed_dirs}/{len(directories_subset)} dirs, {total_buffered} stars buffered, {buffer_mb:.0f}MB)"
                    )

            except (OSError, PermissionError):
                process_safe_print(
                    f"⚠️  Process {process_id}: Cannot access directory {os.path.relpath(directory, CONFIG['local_data_dir'])}"
                )

        except Exception as e:
            process_safe_print(f"❌ Process {process_id} error processing {directory}: {e}")
            continue

    # Save final batches (finish current exposure and any remaining data)
    if len(current_exposure_stars) > 0:
        process_star_data.extend(current_exposure_stars)

    if len(process_star_data) > 0:
        batch_counter = len(list(output_dir.glob(f"desstars_process{process_id:02d}_*.pt")))
        # Count unique exposures in final batch
        unique_exposures = set(star["exposure_id"] for star in process_star_data)
        exposure_count = len(unique_exposures)
        output_file = (
            output_dir
            / f"desstars_process{process_id:02d}_file{batch_counter:03d}_exposures{exposure_count:03d}.pt"
        )
        process_safe_print(
            f"💾 Process {process_id}: Saving final batch with {len(process_star_data)} stars to {output_file}"
        )
        save_batch_pytorch(process_star_data, str(output_file))

    final_progress_pct = (process_processed_dirs / len(directories_subset)) * 100
    process_safe_print(
        f"🏁 Process {process_id} finished: {final_progress_pct:.1f}% complete ({process_processed_dirs}/{len(directories_subset)} dirs, {process_processed_files} files, {process_successful_files} successful)"
    )

    # Return stats for aggregation
    return {
        "process_id": process_id,
        "processed_dirs": process_processed_dirs,
        "processed_files": process_processed_files,
        "successful_files": process_successful_files,
    }


def load_fits_image(fits_path: str) -> tuple[np.ndarray, WCS, dict] | None:
    """
    Load FITS image and extract metadata
    Returns: (image_data, wcs, metadata)
    """
    try:
        # Suppress FITS warnings for cleaner output in multithreaded mode
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*truncated.*")
            warnings.filterwarnings("ignore", message=".*smaller than.*expected.*")

            with fits.open(fits_path) as hdul:
                # DES single epoch files typically have data in extension 1
                if len(hdul) > 1:
                    # Try to load the image data - this will catch buffer size issues
                    try:
                        image_data = hdul[1].data.astype(np.float32)
                        header = hdul[1].header
                    except (OSError, ValueError, MemoryError) as data_error:
                        # Handle specific data corruption errors
                        if "buffer is too small" in str(data_error).lower():
                            print(
                                f"    ❌ Error loading {fits_path}: Corrupted FITS file (buffer size mismatch)"
                            )
                        elif "truncated" in str(data_error).lower():
                            print(f"    ❌ Error loading {fits_path}: Truncated FITS file")
                        else:
                            print(f"    ❌ Error loading {fits_path}: {data_error!s}")
                        return None
                else:
                    try:
                        image_data = hdul[0].data.astype(np.float32)
                        header = hdul[0].header
                    except (OSError, ValueError, MemoryError) as data_error:
                        print(f"    ❌ Error loading {fits_path}: {data_error!s}")
                        return None

            # Extract WCS
            try:
                wcs = WCS(header)
            except Exception as e:
                print(f"    ⚠️ WCS parsing failed: {e}")
                wcs = None

            # Extract metadata from filename and header
            filename = Path(fits_path).name

            # Parse DES filename: D00681214_g_c31_r3516p01_immasked.fits.fz
            match = re.search(r"D(\d{8})_([grizyY])_c(\d+)_r(\d+)p(\d+)", filename)
            if match:
                exposure_id = match.group(1)
                band = match.group(2)
                ccd_num = int(match.group(3))
                run = match.group(4)
                proc_num = match.group(5)
            else:
                exposure_id = "unknown"
                band = "unknown"
                ccd_num = 31
                run = "unknown"
                proc_num = "unknown"

            metadata = {
                "filename": filename,
                "filepath": fits_path,
                "exposure_id": exposure_id,
                "band": band,
                "ccd_num": ccd_num,
                "run": run,
                "proc_num": proc_num,
                "exptime": header.get("EXPTIME", 0.0),
                "seeing": header.get("FWHM", header.get("SEEING", 0.0)),
                "airmass": header.get("AIRMASS", 0.0),
                "mjd_obs": header.get("MJD-OBS", 0.0),
                "filter": header.get("FILTER", band),
                "image_shape": image_data.shape,
            }

            return image_data, wcs, metadata

    except Exception as e:
        print(f"    ❌ Error loading {fits_path}: {e}")
        return None


def detect_sources_sep(image_data: np.ndarray) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Use SEP to detect sources in the image
    Returns: (background_subtracted_image, sources_catalog)
    """
    # Ensure image is in the right byte order for SEP
    if not image_data.flags["C_CONTIGUOUS"]:
        image_data = np.ascontiguousarray(image_data)
    if image_data.dtype != np.float32:
        image_data = image_data.astype(np.float32)

    # Apply cosmic ray detection to the full image before source detection
    if CONFIG["cosmic_ray_detection"]:
        try:
            # Estimate noise for cosmic ray detection
            # Use a robust background estimate for noise calculation
            median_val = np.median(image_data)
            mad = np.median(np.abs(image_data - median_val))
            noise_estimate = 1.4826 * mad  # Convert MAD to standard deviation

            if CONFIG["verbose"]:
                print(
                    f"    🌟 Running cosmic ray detection (noise estimate: {noise_estimate:.2f})..."
                )

            cosmic_ray_mask, image_data = astroscrappy.detect_cosmics(
                image_data,
                sigclip=CONFIG["cosmic_ray_sigclip"],
                sigfrac=CONFIG["cosmic_ray_sigfrac"],
                objlim=CONFIG["cosmic_ray_objlim"],
                gain=1.0,  # Assume data is already in electrons or equivalent
                readnoise=noise_estimate,
                niter=CONFIG["cosmic_ray_niter"],
                sepmed=True,  # Use separable median (faster)
                cleantype="meanmask",  # Use masked mean for cleaning
                verbose=False,
            )

            n_cosmic_rays = np.sum(cosmic_ray_mask)
            cosmic_ray_fraction = n_cosmic_rays / cosmic_ray_mask.size

            if CONFIG["verbose"]:
                print(
                    f"    ✅ Cosmic ray detection: {n_cosmic_rays} pixels ({cosmic_ray_fraction * 100:.2f}%) flagged"
                )

        except Exception as e:
            if CONFIG["verbose"]:
                print(f"    ⚠️  Cosmic ray detection failed: {e}")
            # Continue without cosmic ray cleaning if it fails

    # Estimate and subtract background with more robust handling
    try:
        bkg = sep.Background(image_data, bw=64, bh=64)  # Larger background mesh
        image_sub = image_data - bkg
        bkg_rms = bkg.globalrms
        if CONFIG["verbose"]:
            print(f"    ✅ SEP background: median={np.median(bkg.back()):.1f}, RMS={bkg_rms:.1f}")
    except Exception:
        # Fallback to simple background subtraction
        median_bkg = np.median(image_data)
        mad = np.median(np.abs(image_data - median_bkg))
        image_sub = image_data - median_bkg
        bkg_rms = 1.4826 * mad

    # Adaptive detection threshold based on background RMS
    detection_threshold = max(
        CONFIG["detection_threshold"],
        (
            3.0 * bkg_rms / np.median(np.abs(image_sub[image_sub != 0]))
            if np.any(image_sub != 0)
            else CONFIG["detection_threshold"]
        ),
    )

    # Detect sources with more conservative settings
    try:
        sources = sep.extract(
            image_sub,
            detection_threshold,
            err=bkg_rms,
            minarea=max(CONFIG["min_area"], 9),  # Larger minimum area
            deblend_nthresh=16,  # Less aggressive deblending
            deblend_cont=0.01,  # Higher deblend contrast
            clean=True,
            clean_param=1.0,
        )

        # Convert to pandas DataFrame for easier handling
        sources_df = pd.DataFrame(
            {
                "x": sources["x"],
                "y": sources["y"],
                "flux": sources["flux"],
                "a": sources["a"],  # Semi-major axis
                "b": sources["b"],  # Semi-minor axis
                "theta": sources["theta"],  # Position angle
                "cxx": sources["cxx"],  # Shape parameters
                "cyy": sources["cyy"],
                "cxy": sources["cxy"],
                "flag": sources["flag"],
                "npix": sources["npix"],
                "tnpix": sources["npix"],  # Use npix for now, tnpix may not be available
            }
        )

        # Add computed quantities
        sources_df["fluxerr"] = bkg_rms * np.sqrt(sources_df["npix"])
        sources_df["fwhm"] = 2.0 * np.sqrt(sources_df["a"] * sources_df["b"])
        sources_df["ellipticity"] = 1.0 - (sources_df["b"] / sources_df["a"])
        sources_df["snr"] = sources_df["flux"] / sources_df["fluxerr"]

        # Calculate magnitudes safely
        flux_values = sources_df["flux"].values
        flux_positive = flux_values > 0
        sources_df["magnitude"] = np.where(
            flux_positive, -2.5 * np.log10(np.maximum(flux_values, 1e-10)) + 25.0, 99.0
        )

        if CONFIG["verbose"]:
            print(f"    ✅ SEP detected {len(sources_df)} sources")

        return image_sub, sources_df
    except Exception:
        return image_sub, pd.DataFrame()


def detect_cosmic_rays_in_cutout(cutout: np.ndarray, threshold: float = 10.0) -> bool:
    """
    Detect cosmic rays in a cutout using very conservative criteria.
    Returns True if obvious cosmic rays are detected.
    """
    if cutout.size == 0:
        return False

    # Calculate robust statistics
    median_val = np.median(cutout)
    mad = np.median(np.abs(cutout - median_val))
    robust_std = 1.4826 * mad

    if robust_std <= 0:
        return False

    # Very conservative cosmic ray detection
    # Look for extremely sharp spikes that are much brighter than surroundings
    max_val = np.max(cutout)

    # Only flag if there's an extremely bright spike (>15 sigma)
    if max_val > median_val + 15 * robust_std:
        # Check if the peak is isolated (cosmic ray signature)
        peak_indices = np.where(cutout == max_val)
        if len(peak_indices[0]) > 0:
            # Check if neighboring pixels are much dimmer
            y_peak, x_peak = peak_indices[0][0], peak_indices[1][0]

            # Sample neighborhood around peak
            y_min, y_max = max(0, y_peak - 2), min(cutout.shape[0], y_peak + 3)
            x_min, x_max = max(0, x_peak - 2), min(cutout.shape[1], x_peak + 3)
            neighborhood = cutout[y_min:y_max, x_min:x_max]

            # If peak is >5x brighter than neighborhood median, likely cosmic ray
            neighborhood_median = np.median(neighborhood[neighborhood != max_val])
            if max_val > 5 * neighborhood_median and neighborhood_median > 0:
                return True

    return False


def check_cutout_isolation(cutout: np.ndarray, strict: bool = True) -> bool:
    """
    Check if a cutout contains a truly isolated source.

    Parameters:
    - cutout: 32x32 numpy array
    - strict: If True, use very strict criteria for training targets

    Returns:
    - True if isolated, False if has close neighbors
    """
    if cutout.size == 0:
        return False

    # Calculate robust background statistics
    median_val = np.median(cutout)
    mad = np.median(np.abs(cutout - median_val))
    robust_std = 1.4826 * mad

    if robust_std <= 0:
        return False

    # Find the main peak
    peak_y, peak_x = np.unravel_index(np.argmax(cutout), cutout.shape)

    # For strict isolation (clean training targets), use very conservative criteria
    if strict:
        threshold_factor = 3.5  # 3.5σ threshold for neighbors (more sensitive)
        min_neighbor_distance = 10.0  # Must be >10 pixels from any neighbor (relaxed from 12)
        max_neighbor_strength = 4.0  # Neighbors must be <4σ (stricter)
        max_neighbor_pixels = 3  # Maximum 3 neighbor pixels allowed
    else:
        threshold_factor = 3.0  # 3σ threshold for neighbors
        min_neighbor_distance = 8.0  # Must be >8 pixels from any neighbor
        max_neighbor_strength = 6.0  # Neighbors must be <6σ
        max_neighbor_pixels = 10  # Maximum 10 neighbor pixels allowed

    # Create signal mask
    threshold = median_val + threshold_factor * robust_std
    signal_mask = cutout > threshold

    # Mask out the central region around the main peak
    y_indices, x_indices = np.ogrid[:32, :32]
    center_dist = np.sqrt((x_indices - peak_x) ** 2 + (y_indices - peak_y) ** 2)
    center_mask = center_dist <= min_neighbor_distance

    # Look for signal outside the exclusion zone
    neighbor_mask = signal_mask & ~center_mask
    neighbor_pixels = np.sum(neighbor_mask)

    if neighbor_pixels == 0:
        return True  # No neighbors detected - isolated!

    # Check strength of brightest neighbor
    if neighbor_pixels > 0:
        neighbor_coords = np.where(neighbor_mask)
        brightest_neighbor = np.max(cutout[neighbor_mask])
        neighbor_significance = (brightest_neighbor - median_val) / robust_std

        # Check if neighbor is too bright
        if neighbor_significance > max_neighbor_strength:
            return False

        # Check minimum distance to brightest neighbor
        brightest_idx = np.argmax(cutout[neighbor_mask])
        neighbor_y = neighbor_coords[0][brightest_idx]
        neighbor_x = neighbor_coords[1][brightest_idx]
        distance = np.sqrt((neighbor_x - peak_x) ** 2 + (neighbor_y - peak_y) ** 2)

        if distance < min_neighbor_distance:
            return False

        # Check maximum allowed neighbor pixels
        max_pixels = max_neighbor_pixels if strict else 10
        if neighbor_pixels > max_pixels:
            return False

    return True


def detect_edge_contamination(cutout: np.ndarray) -> bool:
    """
    Detect edge contamination/artifacts in cutouts.
    Returns True if edge contamination is detected.
    """
    if cutout.size == 0:
        return False

    # Check for unusual edge patterns
    edge_width = 2  # Check 2-pixel border

    # Get edge pixels
    top_edge = cutout[:edge_width, :].flatten()
    bottom_edge = cutout[-edge_width:, :].flatten()
    left_edge = cutout[:, :edge_width].flatten()
    right_edge = cutout[:, -edge_width:].flatten()

    all_edges = np.concatenate([top_edge, bottom_edge, left_edge, right_edge])
    center_region = cutout[edge_width:-edge_width, edge_width:-edge_width]

    if center_region.size == 0:
        return True  # Too small, suspicious

    # Calculate statistics
    np.median(all_edges)
    np.median(center_region)
    edge_std = np.std(all_edges)
    center_std = np.std(center_region)

    # Flag if edges are much more variable than center (edge artifacts)
    if edge_std > 1.5 * center_std and edge_std > 5:
        return True

    # Flag if edges have extreme values
    edge_range = np.max(all_edges) - np.min(all_edges)
    center_range = np.max(center_region) - np.min(center_region)
    if edge_range > 2 * center_range and edge_range > 50:
        return True

    # Flag if edge pixels have systematic pattern (like edge artifacts)
    edge_mean = np.mean(all_edges)
    center_mean = np.mean(center_region)
    return bool(abs(edge_mean - center_mean) > 2 * center_std and abs(edge_mean - center_mean) > 10)


def detect_background_gradient(cutout: np.ndarray) -> bool:
    """
    Detect significant background gradients that indicate contamination.
    Returns True if a significant gradient is detected.
    """
    if cutout.size == 0:
        return False

    # Create coordinate grids
    y_coords, x_coords = np.meshgrid(
        np.arange(cutout.shape[0]), np.arange(cutout.shape[1]), indexing="ij"
    )
    y_coords = y_coords.flatten()
    x_coords = x_coords.flatten()
    cutout_flat = cutout.flatten()

    # Mask out the central star region (avoid including the star in gradient fit)
    center_y, center_x = cutout.shape[0] // 2, cutout.shape[1] // 2
    center_mask = ((y_coords - center_y) ** 2 + (x_coords - center_x) ** 2) > 6**2  # 6-pixel radius

    if np.sum(center_mask) < 10:  # Need enough background pixels
        return False

    background_pixels = cutout_flat[center_mask]
    background_y = y_coords[center_mask]
    background_x = x_coords[center_mask]

    try:
        # Fit a plane to the background: z = a*x + b*y + c
        A = np.column_stack([background_x, background_y, np.ones(len(background_x))])
        coeffs, _residuals, rank, _ = np.linalg.lstsq(A, background_pixels, rcond=None)

        if rank < 3:  # Singular matrix
            return False

        # Calculate gradient magnitude
        gradient_x, gradient_y, _ = coeffs
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

        # Calculate background variation
        background_std = np.std(background_pixels)

        # Flag if gradient is significant - use absolute thresholds for robust detection
        # Gradient of >0.5 count per pixel is significant regardless of noise
        if gradient_magnitude > 0.5:
            return True

        # Also check for systematic variation across the cutout
        gradient_variation = abs(np.max(A @ coeffs) - np.min(A @ coeffs))
        # Absolute threshold: >20 count variation across cutout is significant
        if gradient_variation > 20:
            return True

        # Relative threshold for smaller gradients in low-noise regions
        if gradient_magnitude > 0.3 and gradient_magnitude * 32 > 1.5 * background_std:
            return True

    except (np.linalg.LinAlgError, ValueError):
        return False

    return False


def detect_extended_source(cutout: np.ndarray, psf_fwhm: float | None = None) -> bool:
    """
    Detect extended sources (galaxies) based on cutout analysis.
    Uses PSF-adaptive thresholds when PSF FWHM is available.
    Returns True if the source appears to be extended/galaxy-like.
    """
    if cutout.size == 0:
        return False

    # Calculate robust background statistics
    median_val = np.median(cutout)
    mad = np.median(np.abs(cutout - median_val))
    robust_std = 1.4826 * mad

    if robust_std <= 0:
        return False

    # Find peak
    peak_y, peak_x = np.unravel_index(np.argmax(cutout), cutout.shape)

    # Check for extended structure
    threshold = median_val + 3 * robust_std
    signal_mask = cutout > threshold
    signal_pixels = np.sum(signal_mask)

    # PSF-adaptive thresholds
    if psf_fwhm is not None and psf_fwhm > 0:
        # Scale thresholds based on PSF size
        # For typical seeing: FWHM ~1.5-3.0 pixels
        # For good seeing: FWHM ~1.0-2.0 pixels
        # For poor seeing: FWHM ~3.0-6.0 pixels

        # Expected area scales roughly as FWHM^2
        psf_area_factor = (psf_fwhm / 2.0) ** 2  # Normalize to FWHM=2.0
        max_signal_pixels = int(40 * psf_area_factor)  # Base: 40 pixels for FWHM=2.0
        max_signal_pixels = max(30, min(max_signal_pixels, 120))  # Clamp to reasonable range

        # Expected radius scales roughly as FWHM
        max_effective_radius = psf_fwhm * 1.2  # Stars should be ~1.2x FWHM radius
        max_effective_radius = max(2.0, min(max_effective_radius, 6.0))  # Reasonable bounds

    else:
        # Fallback to fixed thresholds when no PSF info available
        max_signal_pixels = 70
        max_effective_radius = 3.5

    # Extended source indicators:

    # 1. Too many signal pixels (galaxies are extended)
    if signal_pixels > max_signal_pixels:
        return True

    # 2. Large effective radius
    if signal_pixels > 0:
        signal_coords = np.where(signal_mask)
        distances = np.sqrt((signal_coords[0] - peak_y) ** 2 + (signal_coords[1] - peak_x) ** 2)
        effective_radius = np.median(distances)

        if effective_radius > max_effective_radius:
            return True

    # 3. Asymmetric/elliptical structure
    if signal_pixels > 10:
        signal_coords = np.where(signal_mask)
        y_coords = signal_coords[0] - peak_y
        x_coords = signal_coords[1] - peak_x

        # Calculate second moments
        if len(x_coords) > 5:
            Ixx = np.sum(x_coords**2) / len(x_coords)
            Iyy = np.sum(y_coords**2) / len(y_coords)
            Ixy = np.sum(x_coords * y_coords) / len(x_coords)

            # Calculate ellipticity
            trace = Ixx + Iyy
            det = Ixx * Iyy - Ixy**2
            if det > 0 and trace > 0:
                ellipticity = 1 - 2 * np.sqrt(det) / trace
                if ellipticity > 0.30:  # Very elliptical = galaxy (tightened from 0.35)
                    return True

    # 4. Multiple bright peaks (galaxy substructure)
    bright_threshold = median_val + 5 * robust_std
    bright_pixels = cutout > bright_threshold

    # Label connected components of bright pixels
    from scipy import ndimage

    labeled, n_components = ndimage.label(bright_pixels)

    if n_components > 1:
        # Check if components are well-separated
        component_centers = ndimage.center_of_mass(
            bright_pixels, labeled, range(1, n_components + 1)
        )

        for i in range(len(component_centers)):
            for j in range(i + 1, len(component_centers)):
                dist = np.sqrt(
                    (component_centers[i][0] - component_centers[j][0]) ** 2
                    + (component_centers[i][1] - component_centers[j][1]) ** 2
                )
                if dist > 4:  # Well-separated bright regions = complex galaxy
                    return True

    # 5. Very bright sources (likely saturated stars or bright galaxies)
    max_val = np.max(cutout)
    if max_val > median_val + 50 * robust_std:  # Extremely bright (more sensitive)
        return True

    return False


def reclassify_star_based_on_cutout_isolation(
    cutout: np.ndarray, current_type: str, psf_fwhm: float | None = None
) -> str:
    """
    Reclassify a star based on cutout isolation analysis.

    Parameters:
    - cutout: 32x32 numpy array
    - current_type: Current classification (clean, predictor, etc.)
    - psf_fwhm: Estimated PSF FWHM for this exposure (adaptive thresholds)

    Returns:
    - New classification based on isolation
    """
    # Skip non-stellar classifications
    if current_type in ["cosmic_ray", "galaxy", "padding"]:
        return current_type

    # First check if it's an extended source (galaxy) with PSF-adaptive thresholds
    if detect_extended_source(cutout, psf_fwhm):
        return "galaxy"  # Extended sources are galaxies, not stars

    # Check for edge contamination
    if detect_edge_contamination(cutout):
        return "predictor"  # Edge contamination makes it unsuitable for clean training

    # Check for background gradient contamination
    if detect_background_gradient(cutout):
        return "predictor"  # Background gradients make it unsuitable for clean training

    # Check for very strict isolation (clean training targets)
    if check_cutout_isolation(cutout, strict=True):
        return "clean"

    # Check for moderate isolation (useful predictors)
    elif check_cutout_isolation(cutout, strict=False):
        return "predictor"

    # Poor isolation - might be blended/contaminated
    else:
        return "predictor"  # Still useful for context but not for training


def has_companions_in_cutout(cutout: np.ndarray, threshold: float | None = None) -> bool:
    """
    Detect companions in a cutout using connected component analysis.
    Returns True if multiple well-separated sources are detected.
    """
    if cutout.size == 0:
        return False

    if threshold is None:
        threshold = CONFIG["companion_detection_threshold"]

    # Calculate robust background statistics
    median_val = np.median(cutout)
    mad = np.median(np.abs(cutout - median_val))
    robust_std = 1.4826 * mad

    if robust_std <= 0:
        return False

    # Use a very high threshold to find only clear secondary sources
    # This prevents noise in the PSF core from being detected as companions
    companion_threshold = threshold + 2.0  # Much higher threshold
    signal_mask = cutout > median_val + companion_threshold * robust_std

    if np.sum(signal_mask) < 5:  # Require at least 5 pixels for a companion
        return False

    # Use connected component analysis to find distinct sources
    from scipy import ndimage

    # Define structure for 8-connected components
    structure = ndimage.generate_binary_structure(2, 2)
    labeled, n_components = ndimage.label(signal_mask, structure=structure)

    if n_components <= 1:
        return False  # Single component = single source

    # Check if components are well-separated and significant
    component_centers = ndimage.center_of_mass(signal_mask, labeled, range(1, n_components + 1))
    component_sizes = [(labeled == i).sum() for i in range(1, n_components + 1)]

    # Only consider components with reasonable size (at least 3 pixels)
    significant_components = []
    for i, (center, size) in enumerate(zip(component_centers, component_sizes, strict=False)):
        if size >= 3:  # Must have at least 3 connected pixels
            significant_components.append((center, size, i + 1))

    if len(significant_components) <= 1:
        return False

    # Check separation between significant components
    for i in range(len(significant_components)):
        for j in range(i + 1, len(significant_components)):
            center1, size1, _label1 = significant_components[i]
            center2, size2, _label2 = significant_components[j]

            # Calculate distance between centers
            dist = np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)

            # Require separation of at least 6 pixels for true companions
            if dist > 6:
                # Both components should be reasonably sized
                if size1 >= 3 and size2 >= 3:
                    return True

    return False


def compute_star_quality_metrics(sources_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute quality metrics for star classification.
    """
    if len(sources_df) == 0:
        return pd.DataFrame()

    sources_df = sources_df.copy()

    # Calculate pairwise distances for isolation metric
    from scipy.spatial.distance import cdist

    positions = sources_df[["x", "y"]].values
    distances = cdist(positions, positions)
    np.fill_diagonal(distances, np.inf)
    min_dist_to_neighbor = np.min(distances, axis=1)

    # Star-like quality score (0-1, higher is better)
    # Combines morphology, SNR, and other factors
    star_score = np.zeros(len(sources_df))

    # FWHM score: prefer stars with reasonable FWHM
    fwhm_ideal = 2.5  # Typical seeing-limited star
    fwhm_penalty = np.abs(sources_df["fwhm"] - fwhm_ideal) / fwhm_ideal
    fwhm_score = np.exp(-(fwhm_penalty**2))

    # Ellipticity score: prefer round sources
    ellip_score = np.exp(-5.0 * sources_df["ellipticity"] ** 2)

    # SNR score: logarithmic preference for higher SNR
    snr_score = np.log10(np.maximum(sources_df["snr"], 1.0)) / 2.0  # Normalize to ~0-1
    snr_score = np.minimum(snr_score, 1.0)

    # Isolation score: prefer isolated sources
    isolation_score = np.minimum(min_dist_to_neighbor / CONFIG["min_separation"], 1.0)

    # Combine scores
    star_score = 0.3 * fwhm_score + 0.2 * ellip_score + 0.2 * snr_score + 0.3 * isolation_score
    sources_df["star_quality"] = star_score

    # Enhanced isolation check for truly clean training data
    # Basic distance requirement (24 pixel minimum)
    distance_isolated = min_dist_to_neighbor >= CONFIG["min_separation"]

    # SEP flag check: exclude sources with any extraction problems
    flag_check = sources_df["flag"] == 0

    # Multi-tier neighbor checks for ultra-clean stars
    clean_check = np.ones(len(sources_df), dtype=bool)
    for i in range(len(sources_df)):
        # Check 1: No bright neighbors within cutout (32 pixels)
        near_mask = (distances[i] < 32) & (distances[i] > 0)
        if np.any(near_mask):
            neighbor_fluxes = sources_df["flux"].iloc[near_mask]
            # Reject if any neighbor is >10% as bright (contamination from bright galaxies)
            bright_neighbors = neighbor_fluxes > 0.1 * sources_df["flux"].iloc[i]
            if np.any(bright_neighbors):
                clean_check[i] = False
                continue

        # Check 2: No very bright neighbors within extended region (64 pixels)
        # This catches contamination from nearby bright galaxies
        extended_mask = (distances[i] < 64) & (distances[i] > 0)
        if np.any(extended_mask):
            neighbor_fluxes = sources_df["flux"].iloc[extended_mask]
            # Reject if any neighbor is much brighter (bright galaxy contamination)
            very_bright_neighbors = neighbor_fluxes > 2.0 * sources_df["flux"].iloc[i]
            if np.any(very_bright_neighbors):
                clean_check[i] = False
                continue

            # Also reject if there are multiple moderately bright neighbors
            moderate_neighbors = neighbor_fluxes > 0.2 * sources_df["flux"].iloc[i]
            if np.sum(moderate_neighbors) >= 3:  # 3+ moderate neighbors = crowded field
                clean_check[i] = False

    # Additional morphological checks for clean stars
    morphology_check = (
        (sources_df["ellipticity"] <= 0.15)  # Very round
        & (sources_df["fwhm"] >= 2.0)
        & (sources_df["fwhm"] <= 6.0)  # Reasonable FWHM
        & (sources_df["snr"] >= 20.0)  # High SNR for reliable morphology
        & (sources_df["npix"] >= 10)  # Sufficient pixel count for reliable measurement
    )

    # Stars must pass all checks: distance, flags, neighbors, and morphology
    sources_df["is_clean"] = distance_isolated & flag_check & clean_check & morphology_check

    # Predictor usefulness score (bright, confident stars for PSF prediction)
    # Weight by flux and star quality
    predictor_score = np.log10(sources_df["flux"]) * star_score
    sources_df["predictor_score"] = predictor_score

    return sources_df


def select_exposure_stars(sources_df: pd.DataFrame, image_shape: tuple[int, int]) -> pd.DataFrame:
    """
    Select exactly 1024 sources per exposure, prioritizing brightness over quality.
    Fill all slots aggressively - galaxies occasionally sneaking in is fine.
    """
    if len(sources_df) == 0:
        return pd.DataFrame()

    height, width = image_shape
    target_count = CONFIG["stars_per_exposure"]

    # Very minimal filtering - be aggressive about filling slots
    minimal_mask = (
        # Position criteria (avoid edges)
        (sources_df["x"] >= CONFIG["edge_buffer"])
        & (sources_df["x"] <= width - CONFIG["edge_buffer"])
        & (sources_df["y"] >= CONFIG["edge_buffer"])
        & (sources_df["y"] <= height - CONFIG["edge_buffer"])
        &
        # Only exclude truly problematic extractions
        (sources_df["flag"] <= 1)  # Allow minor extraction flags
        &
        # Very basic flux filter to avoid negative flux issues
        (sources_df["flux"] > 0)
        &
        # More permissive saturation threshold
        (sources_df["flux"] <= CONFIG["saturation_threshold"] * 2)  # Allow somewhat saturated
    )

    candidates = sources_df[minimal_mask].copy()

    if len(candidates) == 0:
        return pd.DataFrame()

    # Compute quality metrics for classification
    candidates = compute_star_quality_metrics(candidates)

    # Sort ALL candidates by brightness (flux) - this is our primary criterion
    candidates = candidates.sort_values("flux", ascending=False)

    # Take the brightest target_count sources
    if len(candidates) >= target_count:
        final_stars = candidates.head(target_count).copy()
    else:
        final_stars = candidates.copy()

    # Classify the selected sources
    final_stars["star_type"] = "predictor"  # Default
    final_stars.loc[final_stars["is_clean"], "star_type"] = "clean"

    # More aggressive classification - only mark very extended sources as galaxies
    # Keep more sources as predictors since they can still help with PSF prediction
    very_extended_mask = (
        (final_stars["ellipticity"] > 0.6)  # Very elliptical
        | (final_stars["fwhm"] > 10.0)  # Very extended
        | (final_stars["snr"] < 5.0)  # Very low SNR
    )

    # Mark only very extended/poor sources as galaxies
    final_stars.loc[very_extended_mask, "star_type"] = "galaxy"

    # Keep the rest as predictors - they may not be perfect stars but can help PSF prediction

    if CONFIG["verbose"]:
        n_clean = (final_stars["star_type"] == "clean").sum()
        n_predictor = (final_stars["star_type"] == "predictor").sum()
        n_galaxy = (final_stars["star_type"] == "galaxy").sum()
        print(
            f"    ⭐ Selected {len(final_stars)} sources: {n_clean} clean, {n_predictor} predictor, {n_galaxy} galaxy"
        )

    return final_stars


def estimate_psf_fwhm(stars_df: pd.DataFrame) -> float:
    """
    Estimate PSF FWHM for this exposure from the stellar sources.
    Returns median FWHM of likely stellar sources.
    """
    if len(stars_df) == 0:
        return 2.5  # Default fallback

    # Filter to likely stellar sources for PSF estimation
    stellar_mask = (
        (stars_df["ellipticity"] < 0.3)  # Round sources
        & (stars_df["snr"] > 15)  # High SNR for reliable measurement
        & (stars_df["fwhm"] > 1.0)  # Reasonable FWHM bounds
        & (stars_df["fwhm"] < 8.0)
        & (stars_df["flux"] > 0)  # Valid flux
    )

    stellar_sources = stars_df[stellar_mask]

    if len(stellar_sources) < 5:
        # Fallback to all sources if too few stellar candidates
        stellar_sources = stars_df[
            (stars_df["fwhm"] > 1.0) & (stars_df["fwhm"] < 8.0) & (stars_df["flux"] > 0)
        ]

    if len(stellar_sources) == 0:
        return 2.5  # Default fallback

    # Use median FWHM as robust estimator
    psf_fwhm = np.median(stellar_sources["fwhm"])

    # Sanity check and bounds
    psf_fwhm = max(1.0, min(psf_fwhm, 6.0))

    if CONFIG["verbose"]:
        print(
            f"    🔍 Estimated PSF FWHM: {psf_fwhm:.2f} pixels (from {len(stellar_sources)} stellar sources)"
        )

    return psf_fwhm


def extract_star_cutouts(
    image_data: np.ndarray, stars_df: pd.DataFrame, metadata: dict, patch_size: int = 32
) -> list[dict]:
    """
    Extract exactly 1024 cutouts per exposure, padding with zeros if necessary.
    """
    target_count = CONFIG["stars_per_exposure"]
    cutouts = []
    half_patch = patch_size // 2

    # Estimate PSF FWHM for this exposure (for adaptive galaxy detection)
    psf_fwhm = estimate_psf_fwhm(stars_df)

    # Extract real star cutouts
    valid_stars = 0
    for _, star in stars_df.iterrows():
        x_center = round(star["x"])
        y_center = round(star["y"])

        # Extract cutout
        y_start = y_center - half_patch
        y_end = y_center + half_patch
        x_start = x_center - half_patch
        x_end = x_center + half_patch

        # Check bounds
        if (
            y_start < 0
            or y_end >= image_data.shape[0]
            or x_start < 0
            or x_end >= image_data.shape[1]
        ):
            # Create zero cutout for out-of-bounds star
            cutout = np.zeros((patch_size, patch_size), dtype=np.float32)
            flux = 0.0
            flux_err = 0.0
            x_pixel = 0.0
            y_pixel = 0.0
            star_type = "padding"
        else:
            cutout = image_data[y_start:y_end, x_start:x_end]

            if cutout.shape != (patch_size, patch_size):
                # Pad to correct size if needed
                padded_cutout = np.zeros((patch_size, patch_size), dtype=np.float32)
                h, w = cutout.shape
                padded_cutout[:h, :w] = cutout
                cutout = padded_cutout
                flux = 0.0
                flux_err = 0.0
                x_pixel = 0.0
                y_pixel = 0.0
                star_type = "padding"
            else:
                # Check for cosmic ray contamination in clean stars
                original_star_type = star.get("star_type", "unknown")

                # Apply contamination checks to all star types
                star_type = original_star_type

                # Check for cosmic rays FIRST (applies to all star types)
                if detect_cosmic_rays_in_cutout(cutout):
                    star_type = "cosmic_ray"
                    if CONFIG["verbose"]:
                        print(
                            f"      🌟 Cosmic ray detected in {original_star_type} star at ({star['x']:.0f},{star['y']:.0f}), demoting to cosmic_ray"
                        )

                # Apply improved cutout-based isolation check to all stars (if not already cosmic ray)
                elif star_type != "cosmic_ray":
                    # Reclassify based on actual cutout isolation with PSF-adaptive thresholds
                    star_type = reclassify_star_based_on_cutout_isolation(
                        cutout, original_star_type, psf_fwhm
                    )

                    if CONFIG["verbose"] and star_type != original_star_type:
                        print(
                            f"      🔍 Isolation check: {original_star_type} → {star_type} for star at ({star['x']:.0f},{star['y']:.0f})"
                        )

                flux = float(star["flux"])
                flux_err = float(star["fluxerr"])
                x_pixel = float(star["x"])
                y_pixel = float(star["y"])
                valid_stars += 1

        star_data = {
            "cutout": cutout,
            "x_pixel": x_pixel,
            "y_pixel": y_pixel,
            "flux": flux,
            "flux_err": flux_err,
            "star_type": star_type,  # clean, predictor, or padding
            "band": metadata["band"],
            "exposure_id": metadata["exposure_id"],
            "ccd_num": metadata["ccd_num"],
            "run": metadata["run"],
            "proc_num": metadata["proc_num"],
        }

        cutouts.append(star_data)

    # Pad with zero cutouts to reach exactly target_count
    while len(cutouts) < target_count:
        padding_data = {
            "cutout": np.zeros((patch_size, patch_size), dtype=np.float32),
            "x_pixel": 0.0,
            "y_pixel": 0.0,
            "flux": 0.0,
            "flux_err": 0.0,
            "star_type": "padding",
            "band": metadata["band"],
            "exposure_id": metadata["exposure_id"],
            "ccd_num": metadata["ccd_num"],
            "run": metadata["run"],
            "proc_num": metadata["proc_num"],
        }
        cutouts.append(padding_data)

    # Ensure we have exactly target_count cutouts
    cutouts = cutouts[:target_count]

    if CONFIG["verbose"]:
        n_clean = sum(1 for c in cutouts if c["star_type"] == "clean")
        n_predictor = sum(1 for c in cutouts if c["star_type"] == "predictor")
        n_padding = sum(1 for c in cutouts if c["star_type"] == "padding")
        print(
            f"    📦 Extracted {len(cutouts)} cutouts: {n_clean} clean, {n_predictor} predictor, {n_padding} padding"
        )

    return cutouts


def process_fits_file(fits_path: str) -> list[dict]:
    """
    Process a single FITS file and extract star cutouts
    """
    print(f"\n📁 Processing: {Path(fits_path).name}")

    # Load image
    result = load_fits_image(fits_path)
    if result is None:
        return []

    image_data, _wcs, metadata = result
    print(f"  📐 Image shape: {image_data.shape}")
    print(f"  🔭 Exposure: {metadata['exposure_id']}, Band: {metadata['band']}")
    print(f'  ⏱️  ExpTime: {metadata["exptime"]:.1f}s, Seeing: {metadata["seeing"]:.2f}"')

    # Detect sources with SEP
    image_sub, sources_df = detect_sources_sep(image_data)
    if len(sources_df) == 0:
        print("  ❌ No sources detected")
        return []

    # Select stars using new two-tier approach
    stars_df = select_exposure_stars(sources_df, image_data.shape)
    if len(stars_df) == 0:
        print("  ❌ No stars selected")
        return []

    # Extract cutouts
    star_cutouts = extract_star_cutouts(image_sub, stars_df, metadata, CONFIG["patch_size"])

    if len(star_cutouts) == 0:
        print("  ❌ No cutouts extracted")
        return []

    print(f"  ✅ Extracted {len(star_cutouts)} star cutouts")
    return star_cutouts


def process_fits_file_quiet(fits_path: str) -> list[dict]:
    """
    Process a single FITS file and extract star cutouts (quiet version for threading)
    """
    try:
        # Load image
        result = load_fits_image(fits_path)
        if result is None:
            return []

        image_data, _wcs, metadata = result

        # Detect sources with SEP
        image_sub, sources_df = detect_sources_sep(image_data)
        if len(sources_df) == 0:
            return []

        # Select exactly 1024 stars using two-tier approach
        stars_df = select_exposure_stars(sources_df, image_data.shape)
        if len(stars_df) == 0:
            return []

        # Extract cutouts
        star_cutouts = extract_star_cutouts(image_sub, stars_df, metadata, CONFIG["patch_size"])

        return star_cutouts

    except Exception as e:
        # Handle corrupted FITS files, truncation warnings, etc.
        filename = os.path.basename(fits_path)
        error_msg = str(e).lower()

        if "buffer is too small" in error_msg:
            process_safe_print(f"  🗑️ Corrupted file (buffer size): {filename}")
        elif "truncated" in error_msg or "smaller than expected" in error_msg:
            process_safe_print(f"  🗑️ Corrupted file (truncated): {filename}")
        elif "no such file" in error_msg or "not found" in error_msg:
            process_safe_print(f"  ❓ File not found: {filename}")
        else:
            process_safe_print(f"  ❌ Error processing {filename}: {str(e)[:100]}")
        return []


def save_batch_pytorch(all_star_data: list[dict], output_file: str):
    """
    Save all extracted star data to PyTorch file with exposure-based tensors (512, 32, 32)
    """
    if len(all_star_data) == 0:
        print("❌ No star data to save")
        return

    # Group stars by exposure
    exposure_groups = {}
    for star in all_star_data:
        exp_id = star["exposure_id"]
        if exp_id not in exposure_groups:
            exposure_groups[exp_id] = []
        exposure_groups[exp_id].append(star)

    # Verify each exposure has exactly 512 stars
    target_count = CONFIG["stars_per_exposure"]
    for exp_id, stars in exposure_groups.items():
        if len(stars) != target_count:
            print(f"⚠️ Warning: Exposure {exp_id} has {len(stars)} stars, expected {target_count}")

    n_exposures = len(exposure_groups)
    sorted_exp_ids = sorted(exposure_groups.keys())

    # Create PyTorch tensors directly
    patch_size = CONFIG["patch_size"]

    # Initialize tensors with shape (n_exposures, 512, ...)
    cutouts_tensor = torch.zeros(
        (n_exposures, target_count, patch_size, patch_size), dtype=torch.float32
    )
    x_pixel_tensor = torch.zeros((n_exposures, target_count), dtype=torch.float32)
    y_pixel_tensor = torch.zeros((n_exposures, target_count), dtype=torch.float32)
    flux_tensor = torch.zeros((n_exposures, target_count), dtype=torch.float32)
    ccd_num_tensor = torch.zeros((n_exposures, target_count), dtype=torch.uint8)

    # String arrays for metadata (keep as numpy)
    band_array = np.full((n_exposures, target_count), "", dtype="U1")
    # Star types as integers: 0=clean, 1=predictor, 2=galaxy, 3=cosmic_ray, 4=padding
    star_type_array = np.full((n_exposures, target_count), 4, dtype=np.uint8)  # Default to padding

    # Exposure metadata arrays
    exposure_ids = []
    runs = []
    proc_nums = []

    # Fill tensors directly
    for exp_idx, exp_id in enumerate(sorted_exp_ids):
        stars = exposure_groups[exp_id]

        # Store exposure metadata (from first star)
        first_star = stars[0]
        exposure_ids.append(first_star["exposure_id"])
        runs.append(first_star["run"])
        proc_nums.append(first_star["proc_num"])

        # Fill star data directly into tensors (up to target_count)
        star_type_encoding = {
            "clean": 0,
            "predictor": 1,
            "galaxy": 2,
            "cosmic_ray": 3,
            "padding": 4,
        }
        for star_idx, star in enumerate(stars[:target_count]):
            cutouts_tensor[exp_idx, star_idx] = torch.from_numpy(star["cutout"])
            x_pixel_tensor[exp_idx, star_idx] = star["x_pixel"]
            y_pixel_tensor[exp_idx, star_idx] = star["y_pixel"]
            flux_tensor[exp_idx, star_idx] = star["flux"]
            band_array[exp_idx, star_idx] = star["band"]
            star_type_array[exp_idx, star_idx] = star_type_encoding.get(
                star["star_type"], 4
            )  # Convert to int
            ccd_num_tensor[exp_idx, star_idx] = star["ccd_num"]

    # Create final data structure
    data = {
        "cutouts": cutouts_tensor,
        "x_pixel": x_pixel_tensor,
        "y_pixel": y_pixel_tensor,
        "flux": flux_tensor,
        "band": band_array,  # Keep as numpy array for strings
        "star_type": torch.from_numpy(star_type_array),  # Convert to tensor of uint8
        "ccd_num": ccd_num_tensor,
        "exposure_id": np.array(exposure_ids, dtype="U20"),
        "run": np.array(runs, dtype="U10"),
        "proc_num": np.array(proc_nums, dtype="U10"),
        # Metadata
        "metadata": {
            "num_exposures": n_exposures,
            "stars_per_exposure": target_count,
            "patch_size": CONFIG["patch_size"],
            "creation_time": time.time(),
            "data_source": "DES_Local_SEP_Detection",
            "detection_method": "SEP_Source_Extractor",
            "config": {k: v for k, v in CONFIG.items() if isinstance(v, (int, float, str))},
        },
    }

    # Save to PyTorch file
    torch.save(data, output_file)

    file_size_mb = Path(output_file).stat().st_size / (1024 * 1024)
    total_stars = n_exposures * target_count
    if CONFIG["verbose"]:
        print(f"💾 Saved: {output_file}")
        print(f"    {n_exposures} exposures × {target_count} stars = {total_stars} total entries")
    else:
        process_safe_print(
            f"💾 Saved: {output_file} ({n_exposures} exposures, {file_size_mb:.1f} MB)"
        )


def save_batch_hdf5(all_star_data: list[dict], output_file: str):
    """
    Save all extracted star data to HDF5 file with exposure-based arrays (512, 32, 32)
    """
    if len(all_star_data) == 0:
        print("❌ No star data to save")
        return

    # Group stars by exposure
    exposure_groups = {}
    for star in all_star_data:
        exp_id = star["exposure_id"]
        if exp_id not in exposure_groups:
            exposure_groups[exp_id] = []
        exposure_groups[exp_id].append(star)

    # Verify each exposure has exactly 512 stars
    target_count = CONFIG["stars_per_exposure"]
    for exp_id, stars in exposure_groups.items():
        if len(stars) != target_count:
            print(f"⚠️ Warning: Exposure {exp_id} has {len(stars)} stars, expected {target_count}")

    n_exposures = len(exposure_groups)
    sorted_exp_ids = sorted(exposure_groups.keys())

    with h5py.File(output_file, "w") as f:
        # Global metadata
        f.attrs["num_exposures"] = n_exposures
        f.attrs["stars_per_exposure"] = target_count
        f.attrs["patch_size"] = CONFIG["patch_size"]
        f.attrs["creation_time"] = time.time()
        f.attrs["data_source"] = "DES_Local_SEP_Detection"
        f.attrs["detection_method"] = "SEP_Source_Extractor"

        # Save configuration
        for key, value in CONFIG.items():
            if isinstance(value, (int, float, str)):
                f.attrs[f"config_{key}"] = value

        # Create exposure-based arrays
        patch_size = CONFIG["patch_size"]

        # Arrays with shape (n_exposures, 512, ...)
        cutouts_array = np.zeros(
            (n_exposures, target_count, patch_size, patch_size), dtype=np.float32
        )
        x_pixel_array = np.zeros((n_exposures, target_count), dtype=np.float32)
        y_pixel_array = np.zeros((n_exposures, target_count), dtype=np.float32)
        flux_array = np.zeros((n_exposures, target_count), dtype=np.float32)

        # String arrays need careful handling
        band_array = np.full((n_exposures, target_count), b"", dtype="S1")
        star_type_array = np.full((n_exposures, target_count), b"padding", dtype="S10")
        ccd_num_array = np.zeros((n_exposures, target_count), dtype=np.uint8)

        # Exposure metadata arrays
        exposure_ids = []
        runs = []
        proc_nums = []

        # Fill arrays
        for exp_idx, exp_id in enumerate(sorted_exp_ids):
            stars = exposure_groups[exp_id]

            # Store exposure metadata (from first star)
            first_star = stars[0]
            exposure_ids.append(first_star["exposure_id"])
            runs.append(first_star["run"])
            proc_nums.append(first_star["proc_num"])

            # Fill star data (up to target_count)
            for star_idx, star in enumerate(stars[:target_count]):
                cutouts_array[exp_idx, star_idx] = star["cutout"]
                x_pixel_array[exp_idx, star_idx] = star["x_pixel"]
                y_pixel_array[exp_idx, star_idx] = star["y_pixel"]
                flux_array[exp_idx, star_idx] = star["flux"]
                band_array[exp_idx, star_idx] = star["band"].encode("utf-8")
                star_type_array[exp_idx, star_idx] = star["star_type"].encode("utf-8")
                ccd_num_array[exp_idx, star_idx] = star["ccd_num"]

        # Save exposure-based datasets
        f.create_dataset("cutouts", data=cutouts_array, compression="gzip", compression_opts=6)
        f.create_dataset("x_pixel", data=x_pixel_array, compression="gzip", compression_opts=6)
        f.create_dataset("y_pixel", data=y_pixel_array, compression="gzip", compression_opts=6)
        f.create_dataset("flux", data=flux_array, compression="gzip", compression_opts=6)
        f.create_dataset("band", data=band_array, compression="gzip", compression_opts=6)
        f.create_dataset("star_type", data=star_type_array, compression="gzip", compression_opts=6)
        f.create_dataset("ccd_num", data=ccd_num_array, compression="gzip", compression_opts=6)

        # Save exposure metadata
        f.create_dataset(
            "exposure_id",
            data=np.array([exp_id.encode("utf-8") for exp_id in exposure_ids], dtype="S20"),
            compression="gzip",
            compression_opts=6,
        )
        f.create_dataset(
            "run",
            data=np.array([run.encode("utf-8") for run in runs], dtype="S10"),
            compression="gzip",
            compression_opts=6,
        )
        f.create_dataset(
            "proc_num",
            data=np.array([proc_num.encode("utf-8") for proc_num in proc_nums], dtype="S10"),
            compression="gzip",
            compression_opts=6,
        )

        # Statistics
        bands = set()
        for stars in exposure_groups.values():
            for star in stars:
                bands.add(star["band"])
        f.attrs["num_bands"] = len(bands)
        f.attrs["bands"] = ",".join(sorted(bands))

        file_size_mb = Path(output_file).stat().st_size / (1024 * 1024)
        total_stars = n_exposures * target_count
        if CONFIG["verbose"]:
            print(f"💾 Saved: {output_file}")
            print(
                f"    {n_exposures} exposures × {target_count} stars = {total_stars} total entries"
            )
            print(f"    {file_size_mb:.1f} MB, {len(bands)} bands: {sorted(bands)}")
            print(f"    Array shapes: cutouts {cutouts_array.shape}, flux {flux_array.shape}")
        else:
            process_safe_print(
                f"💾 Saved: {output_file} ({n_exposures} exposures, {file_size_mb:.1f} MB)"
            )


def main():
    """
    Main processing function with directory-based multiprocessing
    """
    # Set verbose mode for main function
    CONFIG["verbose"] = True

    print("🔧 Configuration:")
    print(f"  Data directory: {CONFIG['local_data_dir']}")
    print(f"  Patch size: {CONFIG['patch_size']}x{CONFIG['patch_size']}")
    print(f"  Stars per exposure: {CONFIG['stars_per_exposure']} (fixed count)")
    print("  Two-tier selection: Clean stars + bright predictor stars")
    print(f"  Number of processes: {CONFIG['num_processes']}")
    print("  Target file size: ~1GB per HDF5 file")
    print("  Exposures never split across files")
    print(f"  Detection threshold: {CONFIG['detection_threshold']} σ")
    print(f"  FWHM range: {CONFIG['fwhm_min']:.1f} - {CONFIG['fwhm_max']:.1f} pixels")
    print(f"  Max ellipticity: {CONFIG['ellipticity_max']:.2f}")
    print(f"  Min SNR: {CONFIG['snr_min']:.1f}")
    print(f"  Min separation: {CONFIG['min_separation']:.1f} pixels (isolation filter)")
    print(f"  Crowding radius: {CONFIG['crowding_radius']:.1f} pixels (crowding filter)")
    print(f"  Verbose mode: {'ON' if CONFIG['verbose'] else 'OFF'}")

    print("\\n🔇 Note: Verbose mode disabled during multiprocessing to reduce output noise")

    # Load directories from des_directories.txt
    directories = load_directories_list("des_directories.txt")

    print(f"\n🚀 Starting directory-based processing with {CONFIG['num_processes']} processes...")
    total_dirs = len(directories)

    # Partition directories among processes
    dirs_per_process = total_dirs // CONFIG["num_processes"]
    directory_partitions = []

    for i in range(CONFIG["num_processes"]):
        start_idx = i * dirs_per_process
        if i == CONFIG["num_processes"] - 1:  # Last process gets any remainder
            end_idx = total_dirs
        else:
            end_idx = (i + 1) * dirs_per_process

        partition = directories[start_idx:end_idx]
        directory_partitions.append(partition)
        print(
            f"📂 Process {i}: assigned {len(partition)} directories (indices {start_idx}-{end_idx - 1})"
        )

    print(
        f"📊 Total: {total_dirs} directories partitioned across {CONFIG['num_processes']} processes"
    )

    start_time = time.time()

    # Start worker processes using ProcessPoolExecutor
    print(f"🚀 Starting {CONFIG['num_processes']} worker processes...")

    with ProcessPoolExecutor(max_workers=CONFIG["num_processes"]) as executor:
        # Submit all processes
        futures = []
        for i, partition in enumerate(directory_partitions):
            future = executor.submit(worker_process_directories, i, partition)
            futures.append(future)

        print(f"✅ All {CONFIG['num_processes']} processes launched successfully")
        print("⚡ Processes running independently...")
        print(f"🎯 Target: Process {total_dirs} directories across all processes")
        print()

        # Wait for all processes to complete and collect results
        results = []
        for future in futures:
            try:
                result = future.result()  # This will block until the process completes
                results.append(result)
                print(
                    f"✅ Process {result['process_id']} completed: {result['processed_dirs']} dirs, {result['successful_files']} successful"
                )
            except Exception as e:
                print(f"❌ Process failed with error: {e}")

    total_time = time.time() - start_time

    # Aggregate results
    total_processed_dirs = sum(r["processed_dirs"] for r in results)
    total_processed_files = sum(r["processed_files"] for r in results)
    total_successful_files = sum(r["successful_files"] for r in results)

    print(f"\n{'=' * 70}")
    print("MULTIPROCESSING EXTRACTION COMPLETE")
    print(f"{'=' * 70}")
    print(f"✅ Total directories processed: {total_processed_dirs}/{total_dirs}")
    print(f"✅ Total files processed: {total_processed_files}")
    print(f"✅ Successful files: {total_successful_files}")
    print(f"⏱️  Total time: {total_time / 60:.1f} minutes")
    if total_time > 0:
        print(f"📈 Processing rate: {total_processed_files / total_time * 60:.1f} files/minute")

    # List all output files
    output_dir = Path("/data/scratch/regier/sep_des_stars")
    process_files = list(output_dir.glob("des_psf_stars_*.h5")) if output_dir.exists() else []
    # Also check current directory for backward compatibility
    process_files.extend(list(Path(".").glob("des_psf_stars_*.h5")))
    if process_files:
        total_stars = 0
        total_size = 0

        print("\n📁 Output files:")
        for f in sorted(process_files):
            size_mb = f.stat().st_size / (1024**2)
            total_size += size_mb

            # Try to read star count from file
            try:
                with h5py.File(f, "r") as hf:
                    num_stars = hf.attrs["num_stars"]
                    total_stars += num_stars
                print(f"    {f.name}: {num_stars} stars, {size_mb:.1f} MB")
            except Exception:
                print(f"    {f.name}: {size_mb:.1f} MB")

        print("\n📊 Summary:")
        print(f"    📁 Total output files: {len(process_files)}")
        print(f"    ⭐ Total stars extracted: {total_stars}")
        print(f"    💾 Total output size: {total_size:.1f} MB")
        if total_time > 0:
            print(f"    📈 Stars per minute: {total_stars / total_time * 60:.0f}")
    else:
        print("\n❌ No output files found!")


if __name__ == "__main__":
    # Required for multiprocessing on Windows and some Unix systems
    mp.set_start_method("spawn", force=True)
    main()
