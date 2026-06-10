#!/usr/bin/env python3
"""
DES DR2 Directory Discovery
Traverse all DES directories and output immask directory paths for bulk download
"""

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from dl import authClient as ac
from dl import storeClient as sc

print("=" * 70)
print("DES DR2 FILE DISCOVERY")
print("=" * 70)
print(f"Authenticated as: {ac.whoAmI()}")


def discover_directories(
    output_file="des_directories.txt",
    progress_file="discovery_progress.json",
    resume=False,
    single_run=None,
):
    """
    Discover all immask directories in DES DR2 Y5A1 and write to output file
    """

    # Load or initialize progress
    discovered_files = []
    processed_runs = set()
    stats = {"total_files": 0, "total_exposures": 0, "total_runs": 0}

    if resume and os.path.exists(progress_file):
        print(f"📊 Loading progress from {progress_file}")
        with open(progress_file) as f:
            progress = json.load(f)
        discovered_files = progress.get("discovered_files", [])
        processed_runs = set(progress.get("processed_runs", []))
        stats = progress.get("stats", stats)
        print(
            f"📊 Resuming: {len(discovered_files):,} files, {len(processed_runs):,} runs processed"
        )

    start_time = time.time()

    try:
        # Fixed to Y5A1 only
        survey = "Y5A1"
        base_path = f"des_dr2://dr2_se/finalcut/{survey}"
        print(f"\n🔍 Discovering files in {survey}")

        # Get all run directories
        survey_str = sc.ls(base_path)
        if not survey_str:
            print(f"  No data found in {survey}")
            return False

        survey_items = [item.strip() for item in survey_str.split(",")]
        all_runs = [item for item in survey_items if re.match(r"(\d{8}-r\d+|r\d+)", item)]

        # Filter runs based on single_run parameter
        if single_run:
            if single_run in all_runs:
                run_dirs = [single_run]
                print(f"  Processing single run: {single_run}")
            else:
                print(f"  ❌ Requested run '{single_run}' not found")
                print(f"  Available runs: {all_runs[:10]}...")  # Show first 10
                return False
        else:
            run_dirs = all_runs
            print(f"  Processing all {len(run_dirs)} runs")

        for i, run_dir in enumerate(run_dirs):
            if run_dir in processed_runs:
                continue

            print(
                f"    [{i + 1}/{len(run_dirs)}] Scanning {run_dir} (Directories: {len(discovered_files):,})"
            )

            run_path = f"{base_path}/{run_dir}"

            try:
                # Get run directory contents
                run_str = sc.ls(run_path)
                if not run_str:
                    continue

                run_items = [item.strip() for item in run_str.split(",")]

                # Y5A1 structure: date directories containing exposures
                date_dirs = [
                    item
                    for item in run_items
                    if re.match(r"\d{8}", item) and not item.startswith("D")
                ]

                run_files = 0
                total_exposures = 0

                for date_idx, date_dir in enumerate(date_dirs):
                    date_path = f"{run_path}/{date_dir}"

                    try:
                        date_str = sc.ls(date_path)
                        if date_str:
                            date_items = [item.strip() for item in date_str.split(",")]
                            date_exposures = [
                                item for item in date_items if re.match(r"D\d{8}", item)
                            ]

                            # Fast discovery: find correct processing version and collect immask directory paths
                            def process_exposure(exp_dir):
                                """Check if exposure has immask directory and return the directory path"""
                                exp_path = f"{date_path}/{exp_dir}"
                                try:
                                    # Check what processing versions are available (p01, p02, etc.)
                                    exp_contents = sc.ls(exp_path)
                                    if not exp_contents or "does not exist" in exp_contents:
                                        return [], 0

                                    # Find processing versions (p01, p02, p03, etc.)
                                    proc_versions = [
                                        p.strip()
                                        for p in exp_contents.split(",")
                                        if p.strip().startswith("p")
                                    ]

                                    # Try each processing version to find immask directory
                                    for proc_version in sorted(
                                        proc_versions, reverse=True
                                    ):  # Try newest first (p02, p01)
                                        immask_path = f"{exp_path}/{proc_version}/red/immask"
                                        try:
                                            immask_str = sc.ls(immask_path)
                                            if immask_str and "does not exist" not in immask_str:
                                                return [
                                                    immask_path
                                                ], 1  # Return directory path and exposure count
                                        except Exception:
                                            continue

                                    return [], 0  # No valid immask directory found
                                except Exception:
                                    return [], 0

                            # Use ThreadPoolExecutor for parallel processing
                            batch_dirs = []
                            with ThreadPoolExecutor(max_workers=8) as executor:
                                # Submit all exposure processing tasks
                                future_to_exp = {
                                    executor.submit(process_exposure, exp_dir): exp_dir
                                    for exp_dir in date_exposures
                                }

                                for future in as_completed(future_to_exp):
                                    try:
                                        dir_paths, exp_count = future.result()
                                        batch_dirs.extend(dir_paths)
                                        total_exposures += exp_count
                                    except Exception:
                                        continue

                            # Add all collected directory paths
                            if batch_dirs:
                                discovered_files.extend(batch_dirs)
                                run_files += len(batch_dirs)

                            # Report progress after completing each night
                            print(
                                f"        {date_dir} ({date_idx + 1}/{len(date_dirs)}): {len(batch_dirs)} directories, {len(date_exposures)} exposures - Total: {len(discovered_files):,}"
                            )

                    except Exception:
                        continue

                if run_files > 0:
                    print(f"      Added {run_files} directories from {run_dir}")
                    stats["total_files"] += run_files
                    stats["total_exposures"] += total_exposures

                processed_runs.add(run_dir)
                stats["total_runs"] += 1

                # Save progress every 10 runs
                if len(processed_runs) % 10 == 0:
                    progress = {
                        "discovered_files": discovered_files,
                        "processed_runs": list(processed_runs),
                        "stats": stats,
                        "timestamp": time.time(),
                    }
                    with open(progress_file, "w") as f:
                        json.dump(progress, f)
                    print(f"      💾 Progress saved: {len(discovered_files):,} directories")

            except Exception as e:
                print(f"      ❌ Error processing {run_dir}: {e}")
                continue

        # Write final directory list
        print(f"\n💾 Writing directory list to {output_file}")
        with open(output_file, "w") as f:
            for dir_path in discovered_files:
                f.write(f"{dir_path}\n")

        # Save final progress
        final_progress = {
            "discovered_files": discovered_files,
            "processed_runs": list(processed_runs),
            "stats": stats,
            "timestamp": time.time(),
            "status": "completed",
        }
        with open(progress_file, "w") as f:
            json.dump(final_progress, f)

        total_time = time.time() - start_time

        print(f"\n{'=' * 50}")
        print("DISCOVERY COMPLETE")
        print(f"{'=' * 50}")
        print(f"📊 Total directories discovered: {len(discovered_files):,}")
        print(f"📊 Total exposures: {stats['total_exposures']:,}")
        print(f"📊 Total runs: {stats['total_runs']:,}")
        print(f"📁 Directory list saved to: {output_file}")
        print(f"⏱️  Total time: {total_time / 60:.1f} minutes")
        print(f"⚡ Rate: {len(discovered_files) / (total_time / 60):.0f} directories/minute")

        # Estimate total files that would be in these directories
        estimated_files = len(discovered_files) * 61  # ~61 files per immask directory
        print(f"💡 Estimated total FITS files: {estimated_files:,} (~61 files per directory)")
        print("💡 To expand to full file paths, you can use the known DES naming pattern")

    except Exception as e:
        print(f"❌ Discovery failed: {e}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Discover all FITS files in DES DR2 surveys",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--output", default="des_directories.txt", help="Output file for discovered directory paths"
    )
    parser.add_argument(
        "--progress-file",
        default="discovery_progress.json",
        help="Progress file for resumable discovery",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from previous progress")
    parser.add_argument(
        "--run", type=str, default=None, help="Process only a specific run (e.g., r3515)"
    )

    args = parser.parse_args()

    print("\n🎯 Target survey: Y5A1 (hardcoded)")
    if args.run:
        print(f"🎯 Target run: {args.run} (single run mode)")
    print(f"📁 Output file: {args.output}")
    print(f"📊 Progress file: {args.progress_file}")

    if args.resume:
        print("🔄 Resume mode enabled")

    success = discover_directories(
        output_file=args.output,
        progress_file=args.progress_file,
        resume=args.resume,
        single_run=args.run,
    )

    if success:
        print("\n✅ Discovery completed successfully!")
        print("📋 Next step: Use des_bulk_download.py to download files")
    else:
        print("\n❌ Discovery failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
