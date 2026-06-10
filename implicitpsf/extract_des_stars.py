"""Extract star cutouts from DES single-epoch images (v2 schema, see datasets.py).

For each immasked FITS file: detect sources with SEP, cross-match to the DES DR2
coadd catalog (colors, star/galaxy labels, persistent IDs), classify, and save
per-exposure tensors including variance and pixel-validity cutouts.
"""

import argparse
import multiprocessing as mp
import os
import subprocess
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sep
import torch
from astromatch.preprocessing.utils import retry_with_backoff
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning
from astropy.wcs import WCS
from dl import queryClient
from scipy.spatial import cKDTree

warnings.simplefilter("ignore", AstropyWarning)

CONFIG = {
    "patch_size": 32,
    "stars_per_exposure": 512,
    "exposures_per_file": 128,
    "detection_threshold": 2.5,  # sigma above background
    "min_area": 5,  # pixels
    "snr_min_clean": 10.0,
    "isolation_radius": 16.0,  # pixels; neighbor search radius for clean stars
    "isolation_flux_ratio": 0.1,  # neighbors brighter than this fraction break isolation
    "match_radius_arcsec": 1.0,
    "pixel_scale_arcsec": 0.263,
    "spread_model_star_max": 0.003,  # DES star/galaxy cut on SPREAD_MODEL_R
    "catalog_grid_deg": 0.1,  # cone centers snapped to this grid for caching
    "catalog_radius_deg": 0.35,  # covers a CCD from any pointing within a grid cell
}

# DESDM finalcut MSK bits
BPM, SATURATE, INTERP, BADAMP, CRAY, STAR = 1, 2, 4, 8, 16, 32
TRAIL, EDGEBLEED, SSXTALK, EDGE, STREAK = 64, 128, 256, 512, 1024
INVALID_BITS = BPM | SATURATE | INTERP | BADAMP | CRAY | TRAIL | EDGEBLEED | SSXTALK | EDGE | STREAK
SATURATION_BITS = SATURATE | TRAIL | EDGEBLEED

# star_type taxonomy (uint8)
TYPE_CLEAN = 0  # DR2 star, isolated, high SNR, fully valid core: contributes to the loss
TYPE_STAR_CONTEXT = 1  # DR2 star failing clean cuts (blended, low SNR, partly masked)
TYPE_GALAXY = 2  # DR2 source with SPREAD_MODEL_R above the star cut
TYPE_UNMATCHED = 3  # no DR2 counterpart within the match radius
TYPE_PADDING = 4  # zero-flux slot filler
TYPE_SATURATED = 5  # DR2 star with saturation/bleed bits in its stamp


def read_exposure(fits_path):
    """Read SCI/MSK/WGT planes and header metadata from an immasked DES file."""
    with fits.open(fits_path) as hdul:
        sci = hdul["SCI"].data.astype(np.float32)
        msk = hdul["MSK"].data.astype(np.int32)
        wgt = hdul["WGT"].data.astype(np.float32)
        header = hdul["SCI"].header
    return sci, msk, wgt, header


def detect_sources(sci, msk, wgt):
    """SEP detection on the background-subtracted image; winpos centroids.

    Returns (image_sub, sources DataFrame with x, y, flux, flux_err, snr).
    """
    bad = (msk & INVALID_BITS) > 0
    bkg = sep.Background(sci, mask=bad, bw=64, bh=64)
    image_sub = sci - bkg.back()

    err = np.full_like(sci, 1e10)  # effectively infinite noise where the weight map is zero
    ok = wgt > 0
    err[ok] = 1.0 / np.sqrt(wgt[ok])

    objects = sep.extract(
        image_sub,
        CONFIG["detection_threshold"],
        err=err,
        mask=bad,
        minarea=CONFIG["min_area"],
        deblend_nthresh=32,
        deblend_cont=0.005,
    )

    x_win, y_win, win_flags = sep.winpos(image_sub, objects["x"], objects["y"], objects["a"])
    radius = np.full(len(objects), 8.0)
    flux, flux_err, _ = sep.sum_circle(image_sub, x_win, y_win, radius, err=err)

    sources = pd.DataFrame(
        {
            "x": x_win,
            "y": y_win,
            "flux": flux,
            "flux_err": flux_err,
            "snr": np.where(flux_err > 0, flux, 0.0) / np.where(flux_err > 0, flux_err, 1.0),
        }
    )
    usable = (win_flags == 0) & np.isfinite(flux) & np.isfinite(flux_err) & (flux > 0)
    return image_sub, sources[usable].reset_index(drop=True)


def catalog_cache_path(cache_dir, ra, dec):
    """Cone-search cache file for a pointing, snapped to the catalog grid."""
    grid = CONFIG["catalog_grid_deg"]
    ra_snap = round(ra / grid) * grid
    dec_snap = round(dec / grid) * grid
    return Path(cache_dir) / f"dr2_{ra_snap:07.2f}_{dec_snap:+06.2f}.parquet"


def download_dr2_cone(ra, dec, radius_deg, output_path):
    """Cone-search des_dr2.main for the columns the extraction needs; write parquet.

    wavg_flux_psf_* is the multi-epoch weighted PSF photometry — the most reliable
    stellar fluxes in DR2, used here to form g-i colors. Uses the Data Lab query
    service (raw SQL) because its TAP/ADQL translator rejects q3c predicates.
    """
    sql = f"""
    SELECT coadd_object_id, ra, dec, spread_model_r, flags_r,
           wavg_flux_psf_g, wavg_flux_psf_i
    FROM des_dr2.main
    WHERE q3c_radial_query(ra, dec, {ra}, {dec}, {radius_deg})
    """
    catalog = retry_with_backoff(
        lambda: queryClient.query(sql=sql, fmt="pandas", timeout=600),
        error_prefix="Data Lab query failed ",
    )
    catalog.to_parquet(output_path)


def fetch_dr2_catalog(wcs, shape, cache_dir):
    """DES DR2 coadd catalog covering this exposure, cached on disk per pointing."""
    ra_center, dec_center = wcs.pixel_to_world_values((shape[1] - 1) / 2, (shape[0] - 1) / 2)
    cache_file = catalog_cache_path(cache_dir, float(ra_center), float(dec_center))
    if not cache_file.exists():
        grid = CONFIG["catalog_grid_deg"]
        ra_snap = round(float(ra_center) / grid) * grid
        dec_snap = round(float(dec_center) / grid) * grid
        # download to a process-unique path, then atomically publish (workers race here)
        tmp_file = cache_file.with_suffix(f".pid{os.getpid()}.parquet")
        download_dr2_cone(ra_snap, dec_snap, CONFIG["catalog_radius_deg"], tmp_file)
        tmp_file.replace(cache_file)
    return pd.read_parquet(cache_file)


def match_to_dr2(sources, catalog, wcs):
    """Annotate detected sources with DR2 color, star/galaxy label, and object ID."""
    annotated = sources.copy()
    annotated["matched"] = False
    annotated["is_star"] = False
    annotated["color"] = 0.0
    annotated["star_id"] = -1

    # the cone cache covers far more sky than the CCD; the TPV inverse diverges for
    # distant coordinates, so cut to the CCD neighborhood before world_to_pixel
    ra_center, dec_center = wcs.pixel_to_world_values(1024.0, 2048.0)
    cos_dec = np.cos(np.deg2rad(float(dec_center)))
    raw_delta_ra = (catalog["ra"].values - float(ra_center) + 180.0) % 360.0 - 180.0
    delta_ra = raw_delta_ra * cos_dec  # RA wraps at 0/360 within the DES footprint
    delta_dec = catalog["dec"].values - float(dec_center)
    catalog = catalog[np.hypot(delta_ra, delta_dec) < 0.2].reset_index(drop=True)
    if len(catalog) == 0:
        return annotated

    cat_x, cat_y = wcs.world_to_pixel_values(catalog["ra"].values, catalog["dec"].values)
    tree = cKDTree(np.column_stack([cat_x, cat_y]))
    radius_px = CONFIG["match_radius_arcsec"] / CONFIG["pixel_scale_arcsec"]
    distance, index = tree.query(np.column_stack([sources["x"], sources["y"]]))
    matched = distance < radius_px

    hit = catalog.iloc[index[matched]]
    spread = hit["spread_model_r"].values
    flux_g = hit["wavg_flux_psf_g"].values
    flux_i = hit["wavg_flux_psf_i"].values
    # Data Lab encodes missing band photometry as +inf, which passes a bare > 0 cut
    color_ok = np.isfinite(flux_g) & np.isfinite(flux_i) & (flux_g > 0) & (flux_i > 0)
    ratio = np.where(color_ok, flux_g, 1.0) / np.where(color_ok, flux_i, 1.0)
    color = np.where(color_ok, -2.5 * np.log10(ratio), 0.0)

    annotated.loc[matched, "matched"] = True
    annotated.loc[matched, "is_star"] = np.abs(spread) < CONFIG["spread_model_star_max"]
    annotated.loc[matched, "color"] = color
    annotated.loc[matched, "star_id"] = hit["coadd_object_id"].values
    return annotated


def compute_isolation(sources):
    """True where no neighbor within the isolation radius is bright enough to matter."""
    tree = cKDTree(np.column_stack([sources["x"], sources["y"]]))
    pairs = tree.query_pairs(CONFIG["isolation_radius"], output_type="ndarray")
    isolated = np.ones(len(sources), dtype=bool)
    flux = sources["flux"].values
    ratio = CONFIG["isolation_flux_ratio"]
    if len(pairs) > 0:
        first, second = pairs[:, 0], pairs[:, 1]
        np.logical_and.at(isolated, first, ~(flux[second] > ratio * flux[first]))
        np.logical_and.at(isolated, second, ~(flux[first] > ratio * flux[second]))
    return isolated


def classify_sources(sources, msk):
    """Assign a star_type to every in-bounds detected source (vectorized)."""
    half = CONFIG["patch_size"] // 2
    height, width = msk.shape

    isolated = compute_isolation(sources)  # before the bounds cut: edge neighbors still count

    x_int = sources["x"].round().astype(int)
    y_int = sources["y"].round().astype(int)
    in_bounds = (
        (x_int >= half)
        & (x_int <= width - half - 1)
        & (y_int >= half)
        & (y_int <= height - half - 1)
    ).values
    sources = sources[in_bounds].reset_index(drop=True)
    isolated = isolated[in_bounds]
    x_int, y_int = x_int[in_bounds].values, y_int[in_bounds].values

    windows = np.lib.stride_tricks.sliding_window_view(
        msk, (CONFIG["patch_size"], CONFIG["patch_size"])
    )
    stamps = windows[y_int - half, x_int - half]
    saturated = ((stamps & SATURATION_BITS) > 0).any(axis=(1, 2))
    core = stamps[:, half - 4 : half + 4, half - 4 : half + 4]
    core_valid = ((core & INVALID_BITS) == 0).all(axis=(1, 2))

    star = sources["is_star"].values
    clean = (
        star
        & ~saturated
        & core_valid
        & isolated
        & (sources["snr"].values >= CONFIG["snr_min_clean"])
    )

    star_type = np.full(len(sources), TYPE_UNMATCHED, dtype=np.uint8)
    star_type[sources["matched"].values & ~star] = TYPE_GALAXY
    star_type[star] = TYPE_STAR_CONTEXT
    star_type[star & saturated] = TYPE_SATURATED
    star_type[clean] = TYPE_CLEAN

    sources = sources.copy()
    sources["star_type"] = star_type
    return sources


def assemble_exposure(image_sub, msk, wgt, sources, header, fits_path):
    """Build the fixed-size per-exposure record from classified sources."""
    n_slots = CONFIG["stars_per_exposure"]
    patch = CONFIG["patch_size"]
    half = patch // 2

    selected = sources.nlargest(n_slots, "flux").reset_index(drop=True)
    n_real = len(selected)
    rows = selected["y"].round().astype(int).values - half
    cols = selected["x"].round().astype(int).values - half

    def stamps_at(plane):
        windows = np.lib.stride_tricks.sliding_window_view(plane, (patch, patch))
        return windows[rows, cols]

    cutouts = np.zeros((n_slots, patch, patch), dtype=np.float32)
    variance = np.full((n_slots, patch, patch), np.inf, dtype=np.float32)
    valid = np.zeros((n_slots, patch, patch), dtype=bool)
    cutouts[:n_real] = stamps_at(image_sub)
    wgt_stamps = stamps_at(wgt)
    ok = (wgt_stamps > 0) & ((stamps_at(msk) & INVALID_BITS) == 0)
    variance[:n_real][ok] = 1.0 / wgt_stamps[ok]
    valid[:n_real] = ok

    def padded(values, dtype, fill=0):
        if not np.isfinite(np.asarray(values, dtype=np.float64)).all():
            raise ValueError(f"non-finite per-star value in {fits_path}")
        column = np.full(n_slots, fill, dtype=dtype)
        column[:n_real] = values
        return column

    return {
        "cutouts": cutouts,
        "variance": variance,
        "valid_pixels": valid,
        "flux": padded(selected["flux"].values, np.float32),
        "flux_err": padded(selected["flux_err"].values, np.float32),
        "snr": padded(selected["snr"].values, np.float32),
        "sky": padded(np.zeros(n_real), np.float32),  # cutouts are background-subtracted
        "x_pixel": padded(selected["x"].values, np.float32),
        "y_pixel": padded(selected["y"].values, np.float32),
        "color": padded(selected["color"].values, np.float32),
        "star_type": padded(selected["star_type"].values, np.uint8, fill=TYPE_PADDING),
        "star_id": padded(selected["star_id"].values, np.int64, fill=-1),
        "exposure_id": f"D{header['EXPNUM']:08d}",
        "band": header["BAND"].strip(),
        "fits_path": str(fits_path),
        "mjd": float(header["MJD-OBS"]),
        "night": str(header["NITE"]),
        "fwhm": float(header["FWHM"]),
        "skysigma": float(header["SKYSIGMA"]),
        "gain": (float(header["GAINA"]) + float(header["GAINB"])) / 2,
        "airmass": float(header["AIRMASS"]),
        "wcs_header": header.tostring(),
    }


def process_fits_file(fits_path, cache_dir):
    """Full pipeline for one exposure; returns the per-exposure record or None."""
    sci, msk, wgt, header = read_exposure(fits_path)
    image_sub, sources = detect_sources(sci, msk, wgt)
    if len(sources) == 0:
        return None

    wcs = WCS(header)
    catalog = fetch_dr2_catalog(wcs, sci.shape, cache_dir)
    annotated = match_to_dr2(sources, catalog, wcs)
    classified = classify_sources(annotated, msk)
    if (classified["star_type"] == TYPE_CLEAN).sum() == 0:
        return None
    return assemble_exposure(image_sub, msk, wgt, classified, header, fits_path)


def save_chunk(records, out_path):
    """Stack per-exposure records into the v2 .pt schema."""
    tensor_keys = {
        "cutouts": torch.float32,
        "variance": torch.float32,
        "valid_pixels": torch.bool,
        "flux": torch.float32,
        "flux_err": torch.float32,
        "snr": torch.float32,
        "sky": torch.float32,
        "x_pixel": torch.float32,
        "y_pixel": torch.float32,
        "color": torch.float32,
        "star_type": torch.uint8,
        "star_id": torch.int64,
    }
    array_keys = (
        "exposure_id",
        "band",
        "fits_path",
        "mjd",
        "night",
        "fwhm",
        "skysigma",
        "gain",
        "airmass",
        "wcs_header",
    )
    data = {
        key: torch.from_numpy(np.stack([record[key] for record in records])).to(dtype)
        for key, dtype in tensor_keys.items()
    }
    for key in array_keys:
        data[key] = np.array([record[key] for record in records])
    data["config"] = dict(CONFIG)
    torch.save(data, out_path)


def worker(worker_id, fits_paths, out_dir, cache_dir):
    """Process a list of FITS files, writing .pt chunks as exposures accumulate."""
    records, n_chunks, n_failed = [], 0, 0
    for fits_path in fits_paths:
        try:
            record = process_fits_file(fits_path, cache_dir)
        except Exception:
            n_failed += 1
            print(f"[worker {worker_id}] FAILED {fits_path}\n{traceback.format_exc()}")
            continue
        if record is not None:
            records.append(record)
        if len(records) == CONFIG["exposures_per_file"]:
            out_path = Path(out_dir) / f"desstars_w{worker_id:02d}_{n_chunks:04d}.pt"
            save_chunk(records, out_path)
            print(f"[worker {worker_id}] wrote {out_path}")
            records, n_chunks = [], n_chunks + 1
    if records:
        out_path = Path(out_dir) / f"desstars_w{worker_id:02d}_{n_chunks:04d}.pt"
        save_chunk(records, out_path)
    print(f"[worker {worker_id}] done: {n_chunks + 1} chunks, {n_failed} failures")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="/nfs/turbo/lsa-regier/des")
    parser.add_argument("--out-dir", default="/data/scratch/regier/sep_des_stars_v2")
    parser.add_argument("--cache-dir", default="/data/scratch/regier/des_dr2_catalogs")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--band", default=None, help="restrict to one band, e.g. r")
    args = parser.parse_args()

    # find(1) is dramatically faster than Path.rglob on this NFS tree
    listing = subprocess.run(
        ["find", args.data_dir, "-name", "*immasked.fits.fz"],
        capture_output=True,
        text=True,
        check=True,
    )
    fits_paths = sorted(Path(line) for line in listing.stdout.splitlines())
    if args.band is not None:
        fits_paths = [path for path in fits_paths if path.name.split("_")[1] == args.band]
    if args.max_files is not None:
        fits_paths = fits_paths[: args.max_files]
    print(f"{len(fits_paths)} FITS files to process")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)

    chunks = [fits_paths[i :: args.num_workers] for i in range(args.num_workers)]
    processes = [
        mp.Process(target=worker, args=(i, chunk, args.out_dir, args.cache_dir))
        for i, chunk in enumerate(chunks)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join()


if __name__ == "__main__":
    main()
