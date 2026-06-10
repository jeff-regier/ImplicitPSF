#!/usr/bin/env python3
"""
DES DR2 Bulk Download
Download files from directories discovered by des_file_discovery.py
"""

import argparse
import contextlib
import json
import os
import re
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from astropy.io import fits
from dl import authClient as ac
from dl import storeClient as sc

print("=" * 70)
print("DES DR2 BULK DOWNLOAD")
print("=" * 70)
print(f"Authenticated as: {ac.whoAmI()}")


def validate_fits_file(file_path: Path, verbose=False) -> bool:
    """
    Validate a FITS file by checking if it can be opened and read properly
    Returns True if valid, False if corrupted
    """
    try:
        # First check actual file size - basic sanity check
        actual_file_size = file_path.stat().st_size

        # Very basic size check - FITS files should be at least a few KB
        if actual_file_size < 10240:  # 10KB minimum
            if verbose:
                print(f"    ❌ FITS validation failed: File too small ({actual_file_size} bytes)")
            return False

        with warnings.catch_warnings():
            # Suppress warnings during validation to avoid spam
            warnings.filterwarnings("ignore", message=".*truncated.*")
            warnings.filterwarnings("ignore", message=".*smaller than.*expected.*")

            with fits.open(str(file_path)) as hdul:
                # Check if file has extensions and can read basic header
                if len(hdul) == 0:
                    if verbose:
                        print("    ❌ FITS validation failed: No extensions found")
                    return False

                # Try to access the header of the primary extension
                _ = hdul[0].header

                # Skip complex size validation for compressed files (.fits.fz)
                # Compression ratios can vary dramatically (5:1 to 20:1 or more)
                # Instead, just verify we can access the data structure

                # For DES files, just check that we have the expected structure
                # Don't access .data at all as this can trigger decompression issues
                if len(hdul) > 1:
                    try:
                        # Just check that we can read the header of the image extension
                        # This doesn't trigger data decompression
                        image_header = hdul[1].header

                        # Basic header sanity checks without accessing data
                        if "NAXIS" in image_header and image_header["NAXIS"] >= 2:
                            # File structure looks reasonable
                            pass
                        else:
                            if verbose:
                                print(
                                    "    ❌ FITS validation failed: Invalid image header structure"
                                )
                            return False

                    except Exception as header_error:
                        if verbose:
                            print(
                                f"    ❌ FITS validation failed: Cannot read image header: {str(header_error)[:100]}"
                            )
                        return False

                # If we get here, the file seems valid
                if verbose:
                    print(
                        f"    ✅ FITS validation passed: {actual_file_size} bytes, {len(hdul)} HDUs"
                    )
                return True

    except Exception as e:
        # Catch all other exceptions including "buffer is too small for requested array"
        if verbose:
            print(f"    ❌ FITS validation failed: {str(e)[:100]}")
        return False


def download_directory(
    dir_path: str, local_base_dir: str, verbose=False, ccd_filter=None, max_retries=3
) -> dict:
    """
    Download all FITS files from a single directory
    Returns: {'directory': dir_path, 'downloaded': int, 'failed': int, 'size_mb': float, 'errors': list}
    """
    try:
        # Announce start of directory processing
        print(f"🔄 Starting: {dir_path}")

        # Get all FITS files from directory (with retry for truncated responses)
        dir_contents = None
        for attempt in range(3):  # Try up to 3 times
            dir_contents = sc.ls(dir_path)
            if dir_contents and "does not exist" not in dir_contents:
                # Check if response seems complete (heuristic: should have multiple commas for full directories)
                comma_count = dir_contents.count(",")
                if (
                    comma_count > 10 or len(dir_contents.strip()) < 100
                ):  # Either many files OR genuinely small directory
                    break
                # Response seems truncated, retry
                if attempt < 2:
                    import time

                    time.sleep(1.0)  # Longer wait for server recovery
            elif attempt < 2:  # Don't sleep on last attempt
                import time

                time.sleep(0.5)  # Brief wait before retry

        if not dir_contents or "does not exist" in dir_contents:
            return {
                "directory": dir_path,
                "downloaded": 0,
                "failed": 0,
                "size_mb": 0.0,
                "errors": ["Directory does not exist"],
            }

        files = [f.strip() for f in dir_contents.split(",")]
        fits_files = [f for f in files if f.endswith((".fits.fz", ".fits"))]

        # Track exposure ID for informative messages
        exposure_id = dir_path.split("/")[-4]

        # Apply CCD filtering if specified
        original_count = len(fits_files)
        if ccd_filter:
            filtered_files = []
            # Debug: Track what CCDs we find vs what we're looking for
            found_ccds = set()
            for fits_file in fits_files:
                # Extract CCD number from filename (e.g., D00667559_r_c01_r3515p02_immasked.fits.fz -> 1)
                ccd_match = re.search(r"_c(\d+)_", fits_file)
                if ccd_match:
                    ccd_num = int(ccd_match.group(1))
                    found_ccds.add(ccd_num)
                    if ccd_num in ccd_filter:
                        filtered_files.append(fits_file)
                else:
                    # If no CCD number found, include file (safety fallback)
                    filtered_files.append(fits_file)

            fits_files = filtered_files
            if len(fits_files) == 0 and original_count > 0:
                if original_count == 1:
                    # For single-file directories, show the specific CCD
                    found_ccd = sorted(found_ccds)[0] if found_ccds else "unknown"
                    print(
                        f"  ⏭️ {exposure_id}: Skipped (has CCD {found_ccd}, need CCD {sorted(ccd_filter)[0]})"
                    )
                else:
                    print(
                        f"  ⏭️ {exposure_id}: Skipped ({original_count} files, no CCD {sorted(ccd_filter)[0]} found)"
                    )

        downloaded = 0
        failed = 0
        total_size_mb = 0.0
        errors = []

        # Download each file in this directory
        for fits_file in fits_files:
            file_path = f"{dir_path}/{fits_file}"
            result = download_file(
                file_path, local_base_dir, verbose=verbose, max_retries=max_retries
            )

            if result["success"]:
                downloaded += 1
                total_size_mb += result["size_mb"]
            else:
                failed += 1
                errors.append(result["error"])

        # Announce completion with more details
        if downloaded > 0:
            print(f"✅ Completed: {dir_path} ({downloaded} files, {total_size_mb:.1f} MB)")
        # Don't print redundant skip messages - already printed above with exposure ID
        elif not (ccd_filter and original_count > 0 and len(fits_files) == 0):
            print(f"⚠️  {exposure_id}: No files downloaded (found {len(fits_files)} FITS files)")

        return {
            "directory": dir_path,
            "downloaded": downloaded,
            "failed": failed,
            "size_mb": total_size_mb,
            "errors": errors,
        }

    except Exception as e:
        print(f"❌ Error: {dir_path} - {e}")
        return {
            "directory": dir_path,
            "downloaded": 0,
            "failed": 0,
            "size_mb": 0.0,
            "errors": [str(e)],
        }


def download_file(file_path: str, local_base_dir: str, verbose=False, max_retries=3) -> dict:
    """
    Download a single file with retry logic
    Returns: {'file': file_path, 'success': bool, 'size_mb': float, 'error': str}
    """
    import time

    # Extract relative path for local storage
    # des_dr2://dr2_se/finalcut/Y5A1/r3515/20170815/D00667559/p01/red/immask/file.fits.fz
    # -> Y5A1/r3515/20170815/D00667559/p01/red/immask/file.fits.fz
    parts = file_path.split("/")
    if "finalcut" in parts:
        finalcut_idx = parts.index("finalcut")
        relative_path = "/".join(parts[finalcut_idx + 1 :])
    else:
        relative_path = parts[-1]  # Just filename as fallback

    local_path = Path(local_base_dir) / relative_path
    local_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if file already exists and is valid
    if local_path.exists():
        try:
            file_size = local_path.stat().st_size
            if file_size > 0:  # File exists and has content
                # Always validate FITS file integrity for .fits.fz files
                if parts[-1].endswith(".fits.fz"):
                    if verbose:
                        print(f"  🔍 Validating existing FITS file: {parts[-1]}")
                    is_valid = validate_fits_file(local_path, verbose=verbose)
                    if not is_valid:
                        # File is corrupted, remove and re-download
                        local_path.unlink()
                        if verbose:
                            print(f"  🗑️ Removed corrupted FITS file: {parts[-1]}")
                        # Fall through to download section
                    else:
                        file_size_mb = file_size / (1024 * 1024)
                        if verbose:
                            print(
                                f"  ✅ Already exists and valid: {parts[-1]} ({file_size_mb:.1f} MB)"
                            )
                        return {
                            "file": file_path,
                            "success": True,
                            "size_mb": file_size_mb,
                            "local_path": str(local_path),
                            "error": None,
                        }
                else:
                    # Non-FITS file, just check size
                    file_size_mb = file_size / (1024 * 1024)
                    if verbose:
                        print(f"  ✅ Already exists: {parts[-1]} ({file_size_mb:.1f} MB)")
                    return {
                        "file": file_path,
                        "success": True,
                        "size_mb": file_size_mb,
                        "local_path": str(local_path),
                        "error": None,
                    }
            else:
                # File exists but is empty, remove it and download
                local_path.unlink()
                if verbose:
                    print(f"  🗑️ Removed empty file: {parts[-1]}")
        except Exception:
            # If we can't check the file, remove it and download
            with contextlib.suppress(BaseException):
                local_path.unlink()
            if verbose:
                print(f"  ⚠️ Removed corrupted file: {parts[-1]}")

    last_error = None

    for attempt in range(max_retries):
        try:
            if attempt > 0:
                # Wait with exponential backoff: 1s, 2s, 4s
                wait_time = 2 ** (attempt - 1)
                if verbose:
                    print(
                        f"  🔄 Retry {attempt}/{max_retries - 1} for {parts[-1]} (waiting {wait_time}s)"
                    )
                time.sleep(wait_time)
            elif verbose:
                print(f"  📥 Downloading {parts[-1]} -> {local_path}")

            # Remove partial download if it exists
            if local_path.exists():
                local_path.unlink()

            # Download the file
            result = sc.get(file_path, str(local_path), verbose=False)

            if result and local_path.exists():
                file_size = local_path.stat().st_size / (1024 * 1024)  # MB

                # Validate FITS file after download
                if parts[-1].endswith(".fits.fz"):
                    is_valid = validate_fits_file(local_path, verbose=verbose)
                    if not is_valid:
                        # Downloaded file is corrupted, remove and continue to retry
                        local_path.unlink()
                        last_error = "Downloaded file failed FITS validation (corrupted)"
                        if verbose:
                            print(f"  ❌ Downloaded file is corrupted: {parts[-1]}")
                        continue

                if attempt > 0 and verbose:
                    print(f"  ✅ Retry successful for {parts[-1]}")
                elif verbose and parts[-1].endswith(".fits.fz"):
                    print(f"  ✅ Downloaded and validated: {parts[-1]}")

                return {
                    "file": file_path,
                    "success": True,
                    "size_mb": file_size,
                    "local_path": str(local_path),
                    "error": None,
                }
            else:
                last_error = "Download failed or file not created"
                continue

        except Exception as e:
            last_error = str(e)
            if verbose and attempt < max_retries - 1:
                print(f"  ❌ Attempt {attempt + 1} failed: {e}")
            continue

    # All retries failed
    return {
        "file": file_path,
        "success": False,
        "size_mb": 0,
        "local_path": str(local_path) if "local_path" in locals() else "",
        "error": f"Failed after {max_retries} attempts: {last_error}",
    }


def download_directories(
    directory_list_path: str,
    local_dir: str,
    max_workers=16,
    progress_file="download_progress.json",
    resume=False,
    ccd_filter=None,
    max_retries=3,
):
    """
    Download all files from directories with progress tracking and resume capability
    """

    # Load directory list
    print(f"📋 Loading directory list from {directory_list_path}")
    with open(directory_list_path) as f:
        all_directories = [line.strip() for line in f if line.strip()]

    print(f"📊 Total directories to process: {len(all_directories):,}")

    # Load or initialize progress
    completed_directories = set()
    failed_directories = set()
    stats = {
        "directories_completed": 0,
        "directories_failed": 0,
        "files_downloaded": 0,
        "files_failed": 0,
        "total_size_mb": 0.0,
        "start_time": time.time(),
    }

    if resume and os.path.exists(progress_file):
        print(f"📊 Loading progress from {progress_file}")
        with open(progress_file) as f:
            progress = json.load(f)
        completed_directories = set(progress.get("completed_directories", []))
        failed_directories = set(progress.get("failed_directories", []))
        stats = progress.get("stats", stats)
        print(
            f"📊 Resume: {len(completed_directories):,} directories completed, {len(failed_directories):,} failed"
        )

        # Clean up inconsistent state: failed directories shouldn't be in completed list
        if failed_directories:
            overlap = completed_directories & failed_directories
            if overlap:
                completed_directories = completed_directories - failed_directories
                print(
                    f"📊 Fixed inconsistent state: removed {len(overlap)} failed directories from completed list"
                )
            print(f"📊 Will retry {len(failed_directories)} previously failed directories")

    # Filter out already completed directories but include failed ones for retry
    remaining_directories = [d for d in all_directories if d not in completed_directories]
    print(f"📊 Remaining directories: {len(remaining_directories):,}")

    # Keep failed directories for retry tracking but reset the failed count
    # This ensures previously failed directories are included in remaining_directories
    if resume:
        stats["directories_failed"] = 0

    if len(remaining_directories) == 0:
        print("✅ All directories already processed!")
        return True

    # Create local directory
    Path(local_dir).mkdir(parents=True, exist_ok=True)

    print(f"📁 Download directory: {local_dir}")
    print(f"⚡ Max workers: {max_workers}")
    if ccd_filter:
        ccd_list = sorted(ccd_filter)
        if len(ccd_list) <= 10:
            print(f"🔧 CCD filter: {ccd_list}")
        else:
            print(f"🔧 CCD filter: {len(ccd_list)} CCDs ({min(ccd_list)}-{max(ccd_list)})")
    else:
        print("🔧 CCD filter: All CCDs (1-62)")

    start_time = time.time()

    # Process directories in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all directory download tasks
        future_to_dir = {
            executor.submit(
                download_directory,
                dir_path,
                local_dir,
                verbose=True,
                ccd_filter=ccd_filter,
                max_retries=max_retries,
            ): dir_path
            for dir_path in remaining_directories
        }

        processed = 0
        for future in as_completed(future_to_dir):
            dir_path = future_to_dir[future]
            try:
                result = future.result()
                processed += 1

                if result["downloaded"] > 0:
                    completed_directories.add(dir_path)
                    stats["directories_completed"] += 1
                    stats["files_downloaded"] += result["downloaded"]
                    stats["files_failed"] += result["failed"]
                    stats["total_size_mb"] += result["size_mb"]

                    if result["failed"] > 0:
                        print(
                            f"  ⚠️  {dir_path}: {result['downloaded']} downloaded, {result['failed']} failed"
                        )
                        # Show first few errors for debugging
                        for error in result["errors"][:3]:
                            print(f"    Error: {error}")

                # Check if this was a CCD filtering skip vs a real failure
                elif len(result["errors"]) == 0:
                    # No errors means CCD filtering excluded all files (expected)
                    # Mark as completed so it doesn't get reprocessed on resume
                    completed_directories.add(dir_path)
                    stats["directories_completed"] += 1
                else:
                    # Real download failures with errors
                    failed_directories.add(dir_path)
                    stats["directories_failed"] += 1
                    print(f"  ❌ {dir_path}: Failed to download any files")
                    # Show the actual errors
                    print(f"    Errors: {result['errors'][:3]}")  # Show first 3 errors

                # Progress update every 10 directories
                if processed % 10 == 0:
                    print(
                        f"📊 Progress: {processed}/{len(remaining_directories)} directories - {stats['files_downloaded']:,} files downloaded"
                    )

                # Save progress every 50 directories
                if processed % 50 == 0:
                    progress = {
                        "completed_directories": list(completed_directories),
                        "failed_directories": list(failed_directories),
                        "stats": stats,
                        "timestamp": time.time(),
                    }
                    with open(progress_file, "w") as f:
                        json.dump(progress, f)

            except Exception as e:
                failed_directories.add(dir_path)
                stats["directories_failed"] += 1
                print(f"  ❌ {dir_path}: Exception - {e}")

    total_time = time.time() - start_time

    print(f"\n{'=' * 50}")
    print("DOWNLOAD COMPLETE")
    print(f"{'=' * 50}")
    print(f"✅ Directories completed: {stats['directories_completed']:,}")
    print(f"❌ Directories failed: {stats['directories_failed']:,}")
    print(f"✅ Files downloaded: {stats['files_downloaded']:,}")
    print(f"❌ Files failed: {stats['files_failed']:,}")
    print(
        f"💾 Total data downloaded: {stats['total_size_mb']:.1f} MB ({stats['total_size_mb'] / 1024:.2f} GB)"
    )
    print(f"⏱️  Total time: {total_time / 60:.1f} minutes")
    if stats["files_downloaded"] > 0:
        print(f"⚡ Average rate: {stats['files_downloaded'] / (total_time / 60):.1f} files/minute")
        print(f"⚡ Average speed: {stats['total_size_mb'] / (total_time / 60):.1f} MB/minute")

    # Save final progress
    final_progress = {
        "completed_directories": list(completed_directories),
        "failed_directories": list(failed_directories),
        "stats": stats,
        "timestamp": time.time(),
        "status": "completed",
    }
    with open(progress_file, "w") as f:
        json.dump(final_progress, f)

    return stats["directories_failed"] == 0


def main():
    parser = argparse.ArgumentParser(
        description="Download DES DR2 files from directories discovered by des_file_discovery.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "file_list", help="File containing list of directory paths from des_file_discovery.py"
    )
    parser.add_argument(
        "--local-dir", default="./des_dr2_data", help="Local directory for downloaded data"
    )
    parser.add_argument(
        "--max-workers", type=int, default=16, help="Number of concurrent downloads"
    )
    parser.add_argument(
        "--progress-file",
        default="download_progress.json",
        help="Progress file for resumable downloads",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from previous progress")
    parser.add_argument(
        "--ccds",
        type=str,
        default=None,
        help='Comma-separated list of CCD numbers to download (1-62). Example: "1,2,3" or "1-10" or "1,5-8,15"',
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=4,
        help="Maximum number of retry attempts for failed downloads",
    )

    args = parser.parse_args()

    # Parse CCD filter
    ccd_filter = None
    if args.ccds:
        ccd_filter = set()
        for part in args.ccds.split(","):
            part = part.strip()
            if "-" in part:
                # Handle range like "1-10"
                start, end = map(int, part.split("-"))
                ccd_filter.update(range(start, end + 1))
            else:
                # Handle single number like "5"
                ccd_filter.add(int(part))

        # Validate CCD numbers
        invalid_ccds = [ccd for ccd in ccd_filter if ccd < 1 or ccd > 62]
        if invalid_ccds:
            print(f"❌ Invalid CCD numbers: {invalid_ccds}. Valid range is 1-62.")
            sys.exit(1)

    if not os.path.exists(args.file_list):
        print(f"❌ Directory list not found: {args.file_list}")
        print("💡 Run des_file_discovery.py first to generate directory list")
        sys.exit(1)

    print(f"\n📋 Directory list: {args.file_list}")
    print(f"📁 Local directory: {args.local_dir}")
    if args.resume:
        print("🔄 Resume mode enabled")

    success = download_directories(
        directory_list_path=args.file_list,
        local_dir=args.local_dir,
        max_workers=args.max_workers,
        progress_file=args.progress_file,
        resume=args.resume,
        ccd_filter=ccd_filter,
        max_retries=args.max_retries,
    )

    if success:
        print("\n✅ Download completed successfully!")
    else:
        print("\n⚠️  Download completed with some failures")
        print("💡 Use --resume to retry failed downloads")


if __name__ == "__main__":
    main()
