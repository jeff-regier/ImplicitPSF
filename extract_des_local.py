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
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import sep
from astropy.io import fits
from astropy.wcs import WCS

# Set SEP pixel stack size to handle large images
sep.set_extract_pixstack(1000000)

# Configuration
CONFIG = {
    "patch_size": 32,
    "max_stars_per_image": 512,
    "local_data_dir": "/nfs/turbo/lsa-regier/des",
    "num_processes": 8,  # Number of worker processes
    # SEP source detection parameters
    "detection_threshold": 5.0,  # SNR threshold for detection
    "min_area": 5,  # Minimum area in pixels
    "deblend_nthresh": 32,
    "deblend_cont": 0.005,
    # Star classification parameters
    "fwhm_min": 0.8,  # Minimum FWHM in pixels
    "fwhm_max": 8.0,  # Maximum FWHM in pixels
    "ellipticity_max": 0.3,  # Maximum ellipticity for stars
    "flux_min": 100.0,  # Minimum flux for reliable detection
    "snr_min": 10.0,  # Minimum SNR
    # Quality filters
    "edge_buffer": 50,  # Pixels from edge to avoid
    "saturation_threshold": 50000,  # Rough saturation level
}


def load_directories_list(directories_file: str = "des_directories.txt") -> List[str]:
    """
    Load directory list and convert to local paths
    """
    print(f"📋 Loading directories from {directories_file}...")

    directories = []
    with open(directories_file, "r") as f:
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
    Each process processes a subset of directories and writes to its own HDF5 files
    """
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
                                batch_num = process_successful_files // 100
                                output_file = f"des_psf_stars_process_{process_id:02d}_batch_{batch_num:03d}.h5"
                                process_safe_print(
                                    f"💾 Process {process_id}: Saving 1GB batch with {len(process_star_data)} stars to {output_file}"
                                )
                                save_batch_hdf5_quiet(process_star_data, output_file)
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

            except (OSError, PermissionError) as e:
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
        batch_num = process_successful_files // 100
        output_file = f"des_psf_stars_process_{process_id:02d}_final_{batch_num:03d}.h5"
        process_safe_print(
            f"💾 Process {process_id}: Saving final batch with {len(process_star_data)} stars to {output_file}"
        )
        save_batch_hdf5_quiet(process_star_data, output_file)

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


def load_fits_image(fits_path: str) -> Optional[Tuple[np.ndarray, WCS, Dict]]:
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
                            print(f"    ❌ Error loading {fits_path}: {str(data_error)}")
                        return None
                else:
                    try:
                        image_data = hdul[0].data.astype(np.float32)
                        header = hdul[0].header
                    except (OSError, ValueError, MemoryError) as data_error:
                        print(f"    ❌ Error loading {fits_path}: {str(data_error)}")
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


def detect_sources_sep(image_data: np.ndarray) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Use SEP to detect sources in the image
    Returns: (background_subtracted_image, sources_catalog)
    """
    # Ensure image is in the right byte order for SEP
    if not image_data.flags["C_CONTIGUOUS"]:
        image_data = np.ascontiguousarray(image_data)
    if image_data.dtype != np.float32:
        image_data = image_data.astype(np.float32)

    # Estimate and subtract background with more robust handling
    bkg = None
    image_sub = None

    bkg = sep.Background(image_data, bw=64, bh=64)  # Larger background mesh
    image_sub = image_data - bkg
    bkg_rms = bkg.globalrms
    print(f"    ✅ SEP background: median={np.median(bkg.back()):.1f}, RMS={bkg_rms:.1f}")

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
        }
    )

    # Compute flux errors if not provided
    if "fluxerr" in sources.dtype.names:
        sources_df["fluxerr"] = sources["fluxerr"]
    else:
        # Estimate flux errors from background RMS and aperture
        sources_df["fluxerr"] = bkg_rms * np.sqrt(sources_df["npix"])

    # Add derived quantities
    sources_df["fwhm"] = 2.0 * np.sqrt(sources_df["a"] * sources_df["b"])  # Approximate FWHM
    sources_df["ellipticity"] = 1.0 - (sources_df["b"] / sources_df["a"])
    sources_df["snr"] = sources_df["flux"] / sources_df["fluxerr"]

    # Calculate magnitudes, handling zero/negative flux
    flux_values = sources_df["flux"].values
    flux_positive = flux_values > 0
    sources_df["magnitude"] = np.where(
        flux_positive,
        -2.5 * np.log10(np.maximum(flux_values, 1e-10)) + 25.0,
        99.0,  # Sentinel value for invalid magnitudes
    )

    print(f"    ✅ SEP detected {len(sources_df)} sources")

    return image_sub, sources_df


def classify_stars(sources_df: pd.DataFrame, image_shape: Tuple[int, int]) -> pd.DataFrame:
    """
    Classify sources as likely stars based on morphological criteria
    """
    if len(sources_df) == 0:
        return pd.DataFrame()

    height, width = image_shape

    # Apply star selection criteria
    mask = (
        # Shape criteria
        (sources_df["fwhm"] >= CONFIG["fwhm_min"])
        & (sources_df["fwhm"] <= CONFIG["fwhm_max"])
        & (sources_df["ellipticity"] <= CONFIG["ellipticity_max"])
        &
        # Flux/SNR criteria
        (sources_df["flux"] >= CONFIG["flux_min"])
        & (sources_df["snr"] >= CONFIG["snr_min"])
        & (sources_df["flux"] <= CONFIG["saturation_threshold"])
        &
        # Position criteria (avoid edges)
        (sources_df["x"] >= CONFIG["edge_buffer"])
        & (sources_df["x"] <= width - CONFIG["edge_buffer"])
        & (sources_df["y"] >= CONFIG["edge_buffer"])
        & (sources_df["y"] <= height - CONFIG["edge_buffer"])
        &
        # Quality flags
        (sources_df["flag"] == 0)  # No extraction flags
    )

    stars_df = sources_df[mask].copy()

    # Sort by flux (brightest first) and limit
    stars_df = stars_df.sort_values("flux", ascending=False)
    if len(stars_df) > CONFIG["max_stars_per_image"]:
        stars_df = stars_df.head(CONFIG["max_stars_per_image"])

    stars_df = stars_df.reset_index(drop=True)

    print(f"    ⭐ Classified {len(stars_df)} likely stars")

    return stars_df


def extract_star_cutouts(
    image_data: np.ndarray, wcs: WCS, stars_df: pd.DataFrame, metadata: Dict, patch_size: int = 32
) -> List[Dict]:
    """
    Extract cutouts around detected stars
    """
    if len(stars_df) == 0:
        return []

    cutouts = []
    half_patch = patch_size // 2

    for idx, star in stars_df.iterrows():
        x_center = int(round(star["x"]))
        y_center = int(round(star["y"]))

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
            continue

        cutout = image_data[y_start:y_end, x_start:x_end]

        if cutout.shape != (patch_size, patch_size):
            continue

        # Skip RA/Dec conversion - only using pixel coordinates

        star_data = {
            "cutout": cutout,
            "x_pixel": float(star["x"]),
            "y_pixel": float(star["y"]),
            "flux": float(star["flux"]),
            "flux_err": float(star["fluxerr"]),
            "band": metadata["band"],  # Band is per-star, not per-exposure
            # Reference to exposure (will be index into exposure table)
            "exposure_id": metadata["exposure_id"],  # Keep for now to build exposure table
            "ccd_num": metadata["ccd_num"],
            "run": metadata["run"],
            "proc_num": metadata["proc_num"],
        }

        cutouts.append(star_data)

    return cutouts


def process_fits_file(fits_path: str) -> List[Dict]:
    """
    Process a single FITS file and extract star cutouts
    """
    print(f"\n📁 Processing: {Path(fits_path).name}")

    # Load image
    result = load_fits_image(fits_path)
    if result is None:
        return []

    image_data, wcs, metadata = result
    print(f"  📐 Image shape: {image_data.shape}")
    print(f"  🔭 Exposure: {metadata['exposure_id']}, Band: {metadata['band']}")
    print(f"  ⏱️  ExpTime: {metadata['exptime']:.1f}s, Seeing: {metadata['seeing']:.2f}\"")

    # Detect sources with SEP
    image_sub, sources_df = detect_sources_sep(image_data)
    if len(sources_df) == 0:
        print("  ❌ No sources detected")
        return []

    # Classify stars
    stars_df = classify_stars(sources_df, image_data.shape)
    if len(stars_df) == 0:
        print("  ❌ No stars classified")
        return []

    # Extract cutouts
    star_cutouts = extract_star_cutouts(image_sub, wcs, stars_df, metadata, CONFIG["patch_size"])

    if len(star_cutouts) == 0:
        print("  ❌ No cutouts extracted")
        return []

    print(f"  ✅ Extracted {len(star_cutouts)} star cutouts")
    return star_cutouts


def process_fits_file_quiet(fits_path: str) -> List[Dict]:
    """
    Process a single FITS file and extract star cutouts (quiet version for threading)
    """
    try:
        # Load image
        result = load_fits_image(fits_path)
        if result is None:
            return []

        image_data, wcs, metadata = result

        # Detect sources with SEP (suppress output)
        image_sub, sources_df = detect_sources_sep_quiet(image_data)
        if len(sources_df) == 0:
            return []

        # Classify stars
        stars_df = classify_stars_quiet(sources_df, image_data.shape)
        if len(stars_df) == 0:
            return []

        # Extract cutouts
        star_cutouts = extract_star_cutouts(
            image_sub, wcs, stars_df, metadata, CONFIG["patch_size"]
        )

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


def detect_sources_sep_quiet(image_data: np.ndarray) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Quiet version of detect_sources_sep for threading
    """
    # Ensure image is in the right byte order for SEP
    if not image_data.flags["C_CONTIGUOUS"]:
        image_data = np.ascontiguousarray(image_data)
    if image_data.dtype != np.float32:
        image_data = image_data.astype(np.float32)

    # Estimate and subtract background with more robust handling
    try:
        bkg = sep.Background(image_data, bw=64, bh=64)
        image_sub = image_data - bkg
        bkg_rms = bkg.globalrms
    except:
        # Fallback to simple background subtraction
        median_bkg = np.median(image_data)
        mad = np.median(np.abs(image_data - median_bkg))
        image_sub = image_data - median_bkg
        bkg_rms = 1.4826 * mad

    # Adaptive detection threshold
    detection_threshold = max(
        CONFIG["detection_threshold"],
        (
            3.0 * bkg_rms / np.median(np.abs(image_sub[image_sub != 0]))
            if np.any(image_sub != 0)
            else CONFIG["detection_threshold"]
        ),
    )

    # Detect sources
    try:
        sources = sep.extract(
            image_sub,
            detection_threshold,
            err=bkg_rms,
            minarea=max(CONFIG["min_area"], 9),
            deblend_nthresh=16,
            deblend_cont=0.01,
            clean=True,
            clean_param=1.0,
        )

        # Convert to DataFrame
        sources_df = pd.DataFrame(
            {
                "x": sources["x"],
                "y": sources["y"],
                "flux": sources["flux"],
                "a": sources["a"],
                "b": sources["b"],
                "theta": sources["theta"],
                "cxx": sources["cxx"],
                "cyy": sources["cyy"],
                "cxy": sources["cxy"],
                "flag": sources["flag"],
                "npix": sources["npix"],
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

        return image_sub, sources_df
    except:
        return image_sub, pd.DataFrame()


def classify_stars_quiet(sources_df: pd.DataFrame, image_shape: Tuple[int, int]) -> pd.DataFrame:
    """
    Quiet version of classify_stars for threading
    """
    if len(sources_df) == 0:
        return pd.DataFrame()

    height, width = image_shape

    # Apply selection criteria
    mask = (
        (sources_df["fwhm"] >= CONFIG["fwhm_min"])
        & (sources_df["fwhm"] <= CONFIG["fwhm_max"])
        & (sources_df["ellipticity"] <= CONFIG["ellipticity_max"])
        & (sources_df["flux"] >= CONFIG["flux_min"])
        & (sources_df["snr"] >= CONFIG["snr_min"])
        & (sources_df["flux"] <= CONFIG["saturation_threshold"])
        & (sources_df["x"] >= CONFIG["edge_buffer"])
        & (sources_df["x"] <= width - CONFIG["edge_buffer"])
        & (sources_df["y"] >= CONFIG["edge_buffer"])
        & (sources_df["y"] <= height - CONFIG["edge_buffer"])
        & (sources_df["flag"] == 0)
    )

    stars_df = sources_df[mask].copy()

    # Sort by flux and limit
    stars_df = stars_df.sort_values("flux", ascending=False)
    if len(stars_df) > CONFIG["max_stars_per_image"]:
        stars_df = stars_df.head(CONFIG["max_stars_per_image"])

    return stars_df.reset_index(drop=True)


def save_batch_hdf5_quiet(all_star_data: List[Dict], output_file: str):
    """
    Quiet version of save_batch_hdf5 for threading
    """
    if len(all_star_data) == 0:
        return

    with h5py.File(output_file, "w") as f:
        # Global metadata
        f.attrs["num_stars"] = len(all_star_data)
        f.attrs["patch_size"] = CONFIG["patch_size"]
        f.attrs["creation_time"] = time.time()
        f.attrs["data_source"] = "DES_Local_SEP_Detection"
        f.attrs["detection_method"] = "SEP_Source_Extractor"

        # Save configuration
        for key, value in CONFIG.items():
            if isinstance(value, (int, float, str)):
                f.attrs[f"config_{key}"] = value

        # Build exposure table - one row per unique exposure (not per CCD)
        exposure_data = {}
        for star in all_star_data:
            exp_id = star["exposure_id"]
            if exp_id not in exposure_data:
                exposure_data[exp_id] = {
                    "exposure_id": exp_id,
                    "run": star["run"],
                    "proc_num": star["proc_num"],
                }

        # Create exposure index mapping
        sorted_exposures = sorted(exposure_data.keys())
        exposure_to_index = {exp_id: idx for idx, exp_id in enumerate(sorted_exposures)}

        # Statistics
        f.attrs["num_exposures"] = len(sorted_exposures)
        bands = set(star["band"] for star in all_star_data)  # Bands from stars, not exposures
        f.attrs["num_bands"] = len(bands)
        f.attrs["bands"] = ",".join(sorted(bands))

        # Create and save arrays (same as original function)
        n_stars = len(all_star_data)
        cutouts = np.array([star["cutout"] for star in all_star_data], dtype=np.float32)

        # Save star data with exposure indices
        f.create_dataset("cutouts", data=cutouts, compression="gzip", compression_opts=6)
        f.create_dataset(
            "x_pixel",
            data=np.array([star["x_pixel"] for star in all_star_data], dtype=np.float32),
            compression="gzip",
            compression_opts=6,
        )
        f.create_dataset(
            "y_pixel",
            data=np.array([star["y_pixel"] for star in all_star_data], dtype=np.float32),
            compression="gzip",
            compression_opts=6,
        )
        f.create_dataset(
            "flux",
            data=np.array([star["flux"] for star in all_star_data], dtype=np.float32),
            compression="gzip",
            compression_opts=6,
        )
        f.create_dataset(
            "band",
            data=np.array([star["band"].encode("utf-8") for star in all_star_data], dtype="S1"),
            compression="gzip",
            compression_opts=6,
        )
        f.create_dataset(
            "ccd_num",
            data=np.array([star["ccd_num"] for star in all_star_data], dtype=np.uint8),
            compression="gzip",
            compression_opts=6,
        )

        # Store exposure index for each star (references exposure table)
        exposure_indices = np.array(
            [exposure_to_index[star["exposure_id"]] for star in all_star_data], dtype=np.uint32
        )
        f.create_dataset(
            "exposure_idx", data=exposure_indices, compression="gzip", compression_opts=6
        )

        # Create exposure table - one row per unique exposure
        exp_table = f.create_group("exposures")
        exp_table.create_dataset(
            "exposure_id",
            data=np.array(
                [exposure_data[exp]["exposure_id"].encode("utf-8") for exp in sorted_exposures],
                dtype="S20",
            ),
            compression="gzip",
            compression_opts=6,
        )
        exp_table.create_dataset(
            "run",
            data=np.array(
                [exposure_data[exp]["run"].encode("utf-8") for exp in sorted_exposures], dtype="S10"
            ),
            compression="gzip",
            compression_opts=6,
        )
        exp_table.create_dataset(
            "proc_num",
            data=np.array(
                [exposure_data[exp]["proc_num"].encode("utf-8") for exp in sorted_exposures],
                dtype="S10",
            ),
            compression="gzip",
            compression_opts=6,
        )

    file_size_mb = Path(output_file).stat().st_size / (1024 * 1024)
    process_safe_print(f"💾 Saved: {output_file} ({n_stars} stars, {file_size_mb:.1f} MB)")


def save_batch_hdf5(all_star_data: List[Dict], output_file: str):
    """
    Save all extracted star data to HDF5 file
    """
    if len(all_star_data) == 0:
        print("❌ No star data to save")
        return

    with h5py.File(output_file, "w") as f:
        # Global metadata
        f.attrs["num_stars"] = len(all_star_data)
        f.attrs["patch_size"] = CONFIG["patch_size"]
        f.attrs["creation_time"] = time.time()
        f.attrs["data_source"] = "DES_Local_SEP_Detection"
        f.attrs["detection_method"] = "SEP_Source_Extractor"

        # Save configuration
        for key, value in CONFIG.items():
            if isinstance(value, (int, float, str)):
                f.attrs[f"config_{key}"] = value

        # Build exposure table - one row per unique exposure (not per CCD)
        exposure_data = {}
        for star in all_star_data:
            exp_id = star["exposure_id"]
            if exp_id not in exposure_data:
                exposure_data[exp_id] = {
                    "exposure_id": exp_id,
                    "run": star["run"],
                    "proc_num": star["proc_num"],
                }

        # Create exposure index mapping
        sorted_exposures = sorted(exposure_data.keys())
        exposure_to_index = {exp_id: idx for idx, exp_id in enumerate(sorted_exposures)}

        # Statistics
        f.attrs["num_exposures"] = len(sorted_exposures)
        bands = set(star["band"] for star in all_star_data)  # Bands from stars, not exposures
        f.attrs["num_bands"] = len(bands)
        f.attrs["bands"] = ",".join(sorted(bands))

        # Create essential arrays only
        n_stars = len(all_star_data)
        cutouts = np.array([star["cutout"] for star in all_star_data], dtype=np.float32)

        # Coordinate arrays (pixel coordinates only)
        x_array = np.array([star["x_pixel"] for star in all_star_data], dtype=np.float32)
        y_array = np.array([star["y_pixel"] for star in all_star_data], dtype=np.float32)

        # Photometric arrays
        flux_array = np.array([star["flux"] for star in all_star_data], dtype=np.float32)

        # Exposure index array (reference to exposure table)
        exposure_indices = np.array(
            [exposure_to_index[star["exposure_id"]] for star in all_star_data], dtype=np.uint16
        )

        # Save star data with exposure indices
        f.create_dataset("cutouts", data=cutouts, compression="gzip", compression_opts=6)
        f.create_dataset("x_pixel", data=x_array, compression="gzip", compression_opts=6)
        f.create_dataset("y_pixel", data=y_array, compression="gzip", compression_opts=6)
        f.create_dataset("flux", data=flux_array, compression="gzip", compression_opts=6)
        f.create_dataset(
            "band",
            data=np.array([star["band"].encode("utf-8") for star in all_star_data], dtype="S1"),
            compression="gzip",
            compression_opts=6,
        )
        f.create_dataset(
            "ccd_num",
            data=np.array([star["ccd_num"] for star in all_star_data], dtype=np.uint8),
            compression="gzip",
            compression_opts=6,
        )
        f.create_dataset(
            "exposure_idx", data=exposure_indices, compression="gzip", compression_opts=6
        )

        # Create exposure table - one row per unique exposure
        exp_table = f.create_group("exposures")
        exp_table.create_dataset(
            "exposure_id",
            data=np.array(
                [exposure_data[exp]["exposure_id"].encode("utf-8") for exp in sorted_exposures],
                dtype="S20",
            ),
            compression="gzip",
            compression_opts=6,
        )
        exp_table.create_dataset(
            "run",
            data=np.array(
                [exposure_data[exp]["run"].encode("utf-8") for exp in sorted_exposures], dtype="S10"
            ),
            compression="gzip",
            compression_opts=6,
        )
        exp_table.create_dataset(
            "proc_num",
            data=np.array(
                [exposure_data[exp]["proc_num"].encode("utf-8") for exp in sorted_exposures],
                dtype="S10",
            ),
            compression="gzip",
            compression_opts=6,
        )

        file_size_mb = Path(output_file).stat().st_size / (1024 * 1024)
        print(f"💾 Saved: {output_file}")
        print(f"    {n_stars} stars, {file_size_mb:.1f} MB")
        print(f"    {len(sorted_exposures)} exposures, {len(bands)} bands: {sorted(bands)}")


def main():
    """
    Main processing function with directory-based multiprocessing
    """
    print(f"🔧 Configuration:")
    print(f"  Data directory: {CONFIG['local_data_dir']}")
    print(f"  Patch size: {CONFIG['patch_size']}x{CONFIG['patch_size']}")
    print(f"  Max stars per image: {CONFIG['max_stars_per_image']}")
    print(f"  Number of processes: {CONFIG['num_processes']}")
    print(f"  Target file size: ~1GB per HDF5 file")
    print(f"  Exposures never split across files")
    print(f"  Detection threshold: {CONFIG['detection_threshold']} σ")
    print(f"  FWHM range: {CONFIG['fwhm_min']:.1f} - {CONFIG['fwhm_max']:.1f} pixels")
    print(f"  Max ellipticity: {CONFIG['ellipticity_max']:.2f}")
    print(f"  Min SNR: {CONFIG['snr_min']:.1f}")

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
            f"📂 Process {i}: assigned {len(partition)} directories (indices {start_idx}-{end_idx-1})"
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
        print(f"⚡ Processes running independently...")
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

    print(f"\n{'='*70}")
    print("MULTIPROCESSING EXTRACTION COMPLETE")
    print(f"{'='*70}")
    print(f"✅ Total directories processed: {total_processed_dirs}/{total_dirs}")
    print(f"✅ Total files processed: {total_processed_files}")
    print(f"✅ Successful files: {total_successful_files}")
    print(f"⏱️  Total time: {total_time/60:.1f} minutes")
    if total_time > 0:
        print(f"📈 Processing rate: {total_processed_files/total_time*60:.1f} files/minute")

    # List all output files
    process_files = list(Path(".").glob("des_psf_stars_*.h5"))
    if process_files:
        total_stars = 0
        total_size = 0

        print(f"\n📁 Output files:")
        for f in sorted(process_files):
            size_mb = f.stat().st_size / (1024**2)
            total_size += size_mb

            # Try to read star count from file
            try:
                with h5py.File(f, "r") as hf:
                    num_stars = hf.attrs["num_stars"]
                    total_stars += num_stars
                print(f"    {f.name}: {num_stars} stars, {size_mb:.1f} MB")
            except:
                print(f"    {f.name}: {size_mb:.1f} MB")

        print(f"\n📊 Summary:")
        print(f"    📁 Total output files: {len(process_files)}")
        print(f"    ⭐ Total stars extracted: {total_stars}")
        print(f"    💾 Total output size: {total_size:.1f} MB")
        if total_time > 0:
            print(f"    📈 Stars per minute: {total_stars/total_time*60:.0f}")
    else:
        print("\n❌ No output files found!")


if __name__ == "__main__":
    # Required for multiprocessing on Windows and some Unix systems
    mp.set_start_method("spawn", force=True)
    main()
