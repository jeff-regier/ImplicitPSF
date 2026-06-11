"""Simulated single-CCD exposures with a known spatially varying PSF.

Each exposure draws a Moffat PSF field whose FWHM and shear (g1, g2) vary as
random second-order polynomials of CCD position — representable by PIFF/PSFEx's
second-order interpolation, so the comparison is fair. Output is twofold:
FITS files (SCI/MSK/WGT + TAN WCS) for the baseline fitters, and v2-schema .pt
chunks (see datasets.py) for ImplicitPSF, so the entire real-data pipeline
(splits -> training -> run_eval) runs unchanged. Ground-truth PSF parameters are
stored per star, and true_psf_params evaluates the field anywhere for field-error
maps at star-free positions.
"""

import argparse
import multiprocessing as mp
from pathlib import Path

import galsim
import numpy as np
import torch
from astropy.io import fits
from scipy.spatial import cKDTree

from implicitpsf.splits import assign_split

PIXEL_SCALE = 0.263
WIDTH, HEIGHT = 1024, 2048
PATCH = 32
NOISE_SIGMA = 1.0
MOFFAT_BETA = 2.5
N_STARS_RANGE = (120, 240)
FLUX_RANGE = (2e3, 1e5)
FWHM_BASE_RANGE = (2.6, 5.4)  # pixels, per-exposure seeing
FWHM_VARIATION = 0.15  # fractional field variation
SHEAR_SCALE = 0.04
COLOR_MEAN, COLOR_SCATTER = 1.0, 0.6  # g-i distribution of simulated stars
CHROMATIC_FWHM_SLOPE = -0.03  # fractional FWHM change per mag of g-i (DCR/seeing-like)
ISOLATION_RADIUS = 16.0  # match the real extraction: blended stars are context-only
ISOLATION_FLUX_RATIO = 0.1


def random_poly2(rng, scale):
    """Random 2nd-order polynomial coefficients over normalized coords in [-1, 1]."""
    coeffs = rng.normal(0.0, scale, 6)
    coeffs[0] = 0.0  # constant term handled separately
    return coeffs


def eval_poly2(coeffs, x, y):
    u = 2.0 * x / WIDTH - 1.0
    v = 2.0 * y / HEIGHT - 1.0
    return (
        coeffs[0]
        + coeffs[1] * u
        + coeffs[2] * v
        + coeffs[3] * u * u
        + coeffs[4] * u * v
        + coeffs[5] * v * v
    )


def true_psf_params(field, x, y, color=0.0):
    """Ground-truth (fwhm_pixels, g1, g2) of the PSF field at any position and color.

    In chromatic fields the FWHM shifts fractionally with the object's g-i color
    (bluer -> broader, like differential chromatic refraction plus seeing's
    wavelength dependence); achromatic fields ignore color entirely.
    """
    fwhm = field["fwhm_base"] * (1.0 + eval_poly2(field["fwhm_poly"], x, y))
    if field["chromatic"]:
        fwhm = fwhm * (1.0 + CHROMATIC_FWHM_SLOPE * (color - COLOR_MEAN))
    g1 = eval_poly2(field["g1_poly"], x, y)
    g2 = eval_poly2(field["g2_poly"], x, y)
    return fwhm, g1, g2


def sample_field(rng, chromatic=False):
    return {
        "fwhm_base": rng.uniform(*FWHM_BASE_RANGE),
        "fwhm_poly": random_poly2(rng, FWHM_VARIATION / 2),
        "g1_poly": np.concatenate(
            [[rng.normal(0, SHEAR_SCALE)], random_poly2(rng, SHEAR_SCALE)[1:]]
        ),
        "g2_poly": np.concatenate(
            [[rng.normal(0, SHEAR_SCALE)], random_poly2(rng, SHEAR_SCALE)[1:]]
        ),
        "chromatic": chromatic,
    }


def render_star(image, field, x, y, flux, color):
    fwhm, g1, g2 = true_psf_params(field, x, y, color)
    profile = galsim.Moffat(beta=MOFFAT_BETA, fwhm=fwhm * PIXEL_SCALE, flux=flux)
    profile = profile.shear(g1=g1, g2=g2)
    stamp = profile.drawImage(
        nx=PATCH * 2,
        ny=PATCH * 2,
        scale=PIXEL_SCALE,
        center=galsim.PositionD(x + 1.0, y + 1.0),  # galsim is 1-based
    )
    bounds = stamp.bounds & image.bounds
    image[bounds] += stamp[bounds]


def wcs_header(exposure_seed):
    header = fits.Header()
    header["CTYPE1"], header["CTYPE2"] = "RA---TAN", "DEC--TAN"
    header["CRVAL1"] = 10.0 + (exposure_seed % 360) * 0.5
    header["CRVAL2"] = -30.0
    header["CRPIX1"], header["CRPIX2"] = WIDTH / 2, HEIGHT / 2
    scale_deg = PIXEL_SCALE / 3600
    header["CD1_1"], header["CD1_2"] = -scale_deg, 0.0
    header["CD2_1"], header["CD2_2"] = 0.0, scale_deg
    header["SATURATE"] = 1e9
    return header


def simulate_exposure(exposure_seed, fits_dir, chromatic=False):
    """Render one exposure; returns the v2-schema per-exposure record."""
    rng = np.random.default_rng(exposure_seed)
    field = sample_field(rng, chromatic=chromatic)
    n_stars = rng.integers(*N_STARS_RANGE)

    half = PATCH // 2
    margin = half + 2
    x = rng.uniform(margin, WIDTH - margin, n_stars)
    y = rng.uniform(margin, HEIGHT - margin, n_stars)
    flux = np.exp(rng.uniform(np.log(FLUX_RANGE[0]), np.log(FLUX_RANGE[1]), n_stars))
    color = np.clip(rng.normal(COLOR_MEAN, COLOR_SCATTER, n_stars), -0.5, 3.0)

    image = galsim.Image(WIDTH, HEIGHT, scale=PIXEL_SCALE, dtype=np.float32)
    for x0, y0, flux0, color0 in zip(x, y, flux, color, strict=True):
        render_star(image, field, x0, y0, flux0, color0)
    sci = image.array + rng.normal(0, NOISE_SIGMA, image.array.shape).astype(np.float32)

    header = wcs_header(exposure_seed)
    header["FWHM"] = field["fwhm_base"]
    exposure_id = f"S{exposure_seed:08d}"
    night = f"sim{exposure_seed // 8:05d}"  # 8 exposures share a "night"

    # PIFF/PSFEx run only on val/test exposures; skip the 25 MB FITS for train ones.
    # Split args must match build_manifest's (seed=0, frac_val=0.1, frac_test=0.1).
    split = assign_split(night, seed=0, frac_val=0.1, frac_test=0.1)
    fits_path = ""
    if split != "train":
        fits_path = str(Path(fits_dir) / f"{exposure_id}.fits")
        hdus = [
            fits.PrimaryHDU(),
            fits.ImageHDU(sci, header=header, name="SCI"),
            fits.ImageHDU(np.zeros(sci.shape, dtype=np.int16), header=header, name="MSK"),
            fits.ImageHDU(
                np.full(sci.shape, 1.0 / NOISE_SIGMA**2, dtype=np.float32),
                header=header,
                name="WGT",
            ),
        ]
        fits.HDUList(hdus).writeto(fits_path, overwrite=True)

    windows = np.lib.stride_tricks.sliding_window_view(sci, (PATCH, PATCH))
    rows = np.round(y).astype(int) - half
    cols = np.round(x).astype(int) - half
    cutouts = windows[rows, cols].astype(np.float32)

    # blended stars poison a single-star likelihood: like the real extraction,
    # only isolated stars are clean (type 0); the rest are context-only (type 1)
    tree = cKDTree(np.column_stack([x, y]))
    pairs = tree.query_pairs(ISOLATION_RADIUS, output_type="ndarray")
    isolated = np.ones(n_stars, dtype=bool)
    if len(pairs) > 0:
        first, second = pairs[:, 0], pairs[:, 1]
        np.logical_and.at(isolated, first, ~(flux[second] > ISOLATION_FLUX_RATIO * flux[first]))
        np.logical_and.at(isolated, second, ~(flux[first] > ISOLATION_FLUX_RATIO * flux[second]))
    star_type = np.where(isolated, 0, 1).astype(np.uint8)

    n_slots = n_stars  # variable star counts; padding handled by fixed-slot packing
    true_fwhm, true_g1, true_g2 = true_psf_params(field, x, y, color)
    return {
        "n_stars": int(n_slots),
        "cutouts": cutouts,
        "variance": np.full(cutouts.shape, NOISE_SIGMA**2, dtype=np.float32),
        "valid_pixels": np.ones(cutouts.shape, dtype=bool),
        "flux": flux.astype(np.float32),
        "flux_err": np.full(n_stars, NOISE_SIGMA * PATCH, dtype=np.float32),
        "snr": (flux / (NOISE_SIGMA * PATCH)).astype(np.float32),
        "sky": np.zeros(n_stars, dtype=np.float32),
        "x_pixel": x.astype(np.float32),
        "y_pixel": y.astype(np.float32),
        "color": color.astype(np.float32),
        "star_type": star_type,
        "star_id": exposure_seed * 100_000 + np.arange(n_stars, dtype=np.int64),
        "true_fwhm": true_fwhm.astype(np.float32),
        "true_g1": true_g1.astype(np.float32),
        "true_g2": true_g2.astype(np.float32),
        "exposure_id": exposure_id,
        "band": "r",
        "fits_path": fits_path,
        "mjd": 56000.0 + exposure_seed,
        "night": night,
        "fwhm": float(field["fwhm_base"]),
        "skysigma": NOISE_SIGMA,
        "gain": 1.0,
        "airmass": 1.0,
        "wcs_header": header.tostring(),
        "field": field,
    }


def pack_records(records, n_slots):
    """Stack variable-length exposures into fixed-slot v2 tensors."""
    per_star_image = ("cutouts", "variance", "valid_pixels")
    per_star_scalar = (
        "flux",
        "flux_err",
        "snr",
        "sky",
        "x_pixel",
        "y_pixel",
        "color",
        "true_fwhm",
        "true_g1",
        "true_g2",
    )
    per_exposure = (
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

    data = {}
    n_exp = len(records)
    for key in per_star_image:
        sample = records[0][key]
        stack = np.zeros((n_exp, n_slots, *sample.shape[1:]), dtype=sample.dtype)
        for i, record in enumerate(records):
            stack[i, : record["n_stars"]] = record[key]
        data[key] = torch.from_numpy(stack)
    for key in per_star_scalar:
        stack = np.zeros((n_exp, n_slots), dtype=np.float32)
        for i, record in enumerate(records):
            stack[i, : record["n_stars"]] = record[key]
        data[key] = torch.from_numpy(stack)

    star_type = np.full((n_exp, n_slots), 4, dtype=np.uint8)  # padding
    star_id = np.full((n_exp, n_slots), -1, dtype=np.int64)
    for i, record in enumerate(records):
        star_type[i, : record["n_stars"]] = record["star_type"]
        star_id[i, : record["n_stars"]] = record["star_id"]
    data["star_type"] = torch.from_numpy(star_type)
    data["star_id"] = torch.from_numpy(star_id)

    for key in per_exposure:
        data[key] = np.array([record[key] for record in records])
    data["true_field"] = [record["field"] for record in records]
    return data


def worker(worker_id, seeds, out_dir, fits_dir, exposures_per_file, chromatic):
    records = []
    n_chunks = 0
    for seed in seeds:
        records.append(simulate_exposure(seed, fits_dir, chromatic=chromatic))
        if len(records) == exposures_per_file:
            out_path = Path(out_dir) / f"desstars_sim_w{worker_id:02d}_{n_chunks:04d}.pt"
            torch.save(pack_records(records, max(N_STARS_RANGE)), out_path)
            print(f"[worker {worker_id}] wrote {out_path}")
            records, n_chunks = [], n_chunks + 1
    if records:
        out_path = Path(out_dir) / f"desstars_sim_w{worker_id:02d}_{n_chunks:04d}.pt"
        torch.save(pack_records(records, max(N_STARS_RANGE)), out_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="/data/scratch/regier/sim_psf_stars")
    parser.add_argument("--fits-dir", default="/data/scratch/regier/sim_psf_fits")
    parser.add_argument("--n-exposures", type=int, default=6000)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--exposures-per-file", type=int, default=32)
    parser.add_argument(
        "--chromatic", action="store_true", help="PSF FWHM depends on star color (DCR-like)"
    )
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    Path(args.fits_dir).mkdir(parents=True, exist_ok=True)

    seeds = list(range(args.n_exposures))
    chunks = [seeds[i :: args.num_workers] for i in range(args.num_workers)]
    processes = [
        mp.Process(
            target=worker,
            args=(i, chunk, args.out_dir, args.fits_dir, args.exposures_per_file, args.chromatic),
        )
        for i, chunk in enumerate(chunks)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    print(f"simulated {args.n_exposures} exposures")


if __name__ == "__main__":
    main()
