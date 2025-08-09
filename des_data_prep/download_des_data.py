#!/usr/bin/env python3
"""
Script to download 10 random DES DR2 tiles with all bands and catalogs.
Stores data in /data/scratch/des
"""

import argparse
import re
import urllib.request
from pathlib import Path
from typing import List
from urllib.error import HTTPError


def get_known_working_tiles() -> List[str]:
    """Get list of tiles we know work from previous downloads."""
    return ["DES0000+0209", "DES0001-0458", "DES0002+0000", "DES0052+0500", "DES0003-0458"]


def get_all_available_tiles() -> List[str]:
    """Get a comprehensive list of potential DES tiles."""
    # Start with known working tiles
    tiles = get_known_working_tiles()

    # Add systematic variations around known working coordinates
    # Focus on small offsets from known working tiles

    base_tiles = [(0, 209), (1, -458), (2, 0), (52, 500), (3, -458)]

    for base_ra, base_dec in base_tiles:
        # Small RA offsets
        for ra_offset in range(-2, 3):
            # Small Dec offsets
            for dec_offset in range(-2, 3):
                if ra_offset == 0 and dec_offset == 0:
                    continue  # Skip the base tile itself

                ra = base_ra + ra_offset
                dec = base_dec + dec_offset

                if ra < 0:
                    continue

                if dec >= 0:
                    tile = f"DES{ra:04d}+{dec:04d}"
                else:
                    tile = f"DES{ra:04d}{dec:05d}"
                tiles.append(tile)

    # Add some additional systematic patterns
    for ra in range(0, 10):
        for dec in [0, 209, 500]:
            tile = f"DES{ra:04d}+{dec:04d}"
            tiles.append(tile)

    for ra in range(0, 5):
        for dec in [-458, -207]:
            tile = f"DES{ra:04d}{dec:05d}"
            tiles.append(tile)

    return list(set(tiles))  # Remove duplicates


class DESDownloader:
    """Self-contained DES data downloader."""

    def __init__(self, cache_dir: str = "/data/scratch/des"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = "https://desdr-server.ncsa.illinois.edu/despublic/dr2_tiles"

    def _get_directory_listing(self, tile_name: str) -> List[str]:
        """Get list of files in a tile directory from the web server."""
        url = f"{self.base_url}/{tile_name}/"

        try:
            with urllib.request.urlopen(url) as response:
                html_content = response.read().decode("utf-8")

            # Parse HTML to extract file links
            # Look for patterns like: href="filename.ext"
            file_pattern = r'href="([^"]+\.[a-z.]+)"'
            matches = re.findall(file_pattern, html_content, re.IGNORECASE)

            # Filter to only keep actual files (not directories or parent links)
            files = []
            for match in matches:
                if not match.startswith("../") and not match.endswith("/"):
                    files.append(match)

            return files

        except HTTPError as e:
            if e.code == 404:
                print(f"  Directory {tile_name} not found on server")
                return []
            else:
                raise
        except Exception as e:
            print(f"  Failed to get directory listing for {tile_name}: {e}")
            return []

    def _download_file(self, tile_name: str, filename: str) -> bool:
        """Download a single file for a tile, preserving directory structure."""
        # Create tile-specific directory
        tile_dir = self.cache_dir / tile_name
        tile_dir.mkdir(exist_ok=True)

        cache_path = tile_dir / filename

        # Skip if already exists
        if cache_path.exists():
            print(f"  ✓ {filename} already cached")
            return True

        url = f"{self.base_url}/{tile_name}/{filename}"

        print(f"  Downloading {filename}...")
        try:
            urllib.request.urlretrieve(url, cache_path)
            print(f"  ✓ {filename} downloaded successfully")
            return True
        except Exception as e:
            print(f"  ✗ Failed to download {filename}: {e}")
            return False

    def download_tile_directory(self, tile_name: str, file_patterns: List[str] = None) -> bool:
        """
        Download entire directory for a tile from the DES server.

        Args:
            tile_name: DES tile name (e.g., 'DES0310-5748')
            file_patterns: List of regex patterns to match files to download.
                          If None, downloads all files.

        Returns:
            bool: True if successful, False if tile doesn't exist
        """
        print(f"\n=== Downloading tile directory {tile_name} ===")

        # Get directory listing
        print(f"Getting directory listing for {tile_name}...")
        files = self._get_directory_listing(tile_name)

        if not files:
            print(f"✗ No files found for tile {tile_name}")
            return False

        print(f"Found {len(files)} files in directory")

        # Filter files if patterns specified
        if file_patterns:
            filtered_files = []
            for file in files:
                for pattern in file_patterns:
                    if re.search(pattern, file):
                        filtered_files.append(file)
                        break
            files = filtered_files
            print(f"Filtered to {len(files)} files matching patterns")

        if not files:
            print(f"✗ No files match the specified patterns")
            return False

        # Download all files
        print(f"Downloading {len(files)} files...")
        successful = 0
        failed = 0

        for filename in files:
            success = self._download_file(tile_name, filename)
            if success:
                successful += 1
            else:
                failed += 1

        print(f"Download complete: {successful} successful, {failed} failed")

        if successful > 0:
            print(f"✓ Tile {tile_name} completed successfully")
            return True
        else:
            print(f"✗ Failed to download any files for tile {tile_name}")
            return False

    def download_tile(self, tile_name: str) -> bool:
        """
        Download entire directory for a tile.

        Returns:
            bool: True if successful, False if tile doesn't exist
        """
        return self.download_tile_directory(tile_name, file_patterns=None)


def download_tile_data(tile_name: str, cache_dir: str = "/data/scratch/des") -> bool:
    """Download entire directory for a tile using DESDownloader."""
    downloader = DESDownloader(cache_dir)
    return downloader.download_tile(tile_name)


def main():
    parser = argparse.ArgumentParser(description="Download DES DR2 tiles (full directories)")
    parser.add_argument(
        "--cache-dir", type=str, default="/data/scratch/des", help="Cache directory for DES data"
    )
    parser.add_argument("tiles", nargs="+", help="Tile names to download (e.g., DES0310-5748)")

    args = parser.parse_args()

    selected_tiles = args.tiles
    print(f"Downloading {len(selected_tiles)} tiles: {selected_tiles}")

    # Create cache directory
    cache_path = Path(args.cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    print(f"Cache directory: {cache_path.absolute()}")

    # Download tiles
    successful_tiles = []
    failed_tiles = []

    print(f"\n{'='*60}")
    print(f"Starting download of {len(selected_tiles)} tiles")
    print(f"{'='*60}")

    for i, tile_name in enumerate(selected_tiles, 1):
        print(f"\nProgress: {i}/{len(selected_tiles)}")

        success = download_tile_data(tile_name=tile_name, cache_dir=args.cache_dir)

        if success:
            successful_tiles.append(tile_name)
        else:
            failed_tiles.append(tile_name)

        print(f"Running totals: {len(successful_tiles)} success, {len(failed_tiles)} failed")

    # Summary
    print(f"\n{'='*60}")
    print(f"DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    print(f"Successful downloads: {len(successful_tiles)}")
    print(f"Failed downloads: {len(failed_tiles)}")

    if successful_tiles:
        print(f"\nSuccessful tiles:")
        for tile in successful_tiles:
            print(f"  ✓ {tile}")

    if failed_tiles:
        print(f"\nFailed tiles:")
        for tile in failed_tiles:
            print(f"  ✗ {tile}")

    print(f"\nData cached in: {cache_path.absolute()}")


if __name__ == "__main__":
    main()
