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
WIDTH, HEIGHT = 2048, 4096  # real DES CCD 31 dimensions (was a 1024x2048 quarter-frame)
PATCH = 32
NOISE_SIGMA = 20.0  # realistic sky noise: gives star SNR p50~54, galaxy SNR p50~14 (match DES)
MOFFAT_BETA = 2.5
_PSF_MODEL = {"name": "moffat"}  # mutable holder (fork workers inherit it); avoids a global stmt
N_STARS_RANGE = (90, 150)  # ~99 clean stars/exposure to match real density over the full CCD


def set_psf_model(name):
    """Select the atmospheric PSF model ('moffat' or 'kolmogorov'); call before generation
    AND in any truth eval (sim_truth/galaxy_recovery) so the truth PSF matches the sim."""
    assert name in ("moffat", "kolmogorov")
    _PSF_MODEL["name"] = name


def psf_profile(fwhm_pixels, flux=1.0):
    """Atmospheric PSF at the given FWHM. Moffat is a soft-cored fitting function (too soft
    to exhibit the real PSF-core under-concentration deficit); Kolmogorov is the physical
    ground-based-turbulence model, sharper-cored, used to reproduce that deficit on a clean
    simulation truth grid (W3 sharp-core testbed)."""
    fwhm_arcsec = fwhm_pixels * PIXEL_SCALE
    if _PSF_MODEL["name"] == "kolmogorov":
        return galsim.Kolmogorov(fwhm=fwhm_arcsec, flux=flux)
    return galsim.Moffat(beta=MOFFAT_BETA, fwhm=fwhm_arcsec, flux=flux)


FLUX_RANGE = (2e3, 6e5)  # real clean-star flux p10-p90 ~7e3-5e5 (bright tail matters)
FWHM_BASE_RANGE = (2.6, 5.4)  # pixels, per-exposure seeing
FWHM_VARIATION = 0.15  # fractional field variation
SHEAR_SCALE = 0.04
COLOR_MEAN, COLOR_SCATTER = 1.0, 0.6  # g-i distribution of simulated stars
CHROMATIC_FWHM_SLOPE = -0.03  # fractional FWHM change per mag of g-i (DCR/seeing-like)
ISOLATION_RADIUS = 16.0  # match the real extraction: blended stars are context-only
ISOLATION_FLUX_RATIO = 0.1
GALAXY_RE_RANGE = (1.5, 6.0)  # pixels, half-light radius of injected galaxy detections
GALAXY_SERSIC_RANGE = (0.5, 4.0)
GALAXY_SHEAR_MAX = 0.5  # max |reduced shear| of an injected galaxy
GALAXY_FLUX_RANGE = (1e3, 8e4)  # real galaxies are ~4x fainter than stars (p10-p90 2e3-6e4)
TYPE_CLEAN, TYPE_STAR_CONTEXT, TYPE_GALAXY = 0, 1, 2


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
    profile = psf_profile(fwhm, flux).shear(g1=g1, g2=g2)
    stamp = profile.drawImage(
        nx=PATCH * 2,
        ny=PATCH * 2,
        scale=PIXEL_SCALE,
        center=galsim.PositionD(x + 1.0, y + 1.0),  # galsim is 1-based
    )
    bounds = stamp.bounds & image.bounds
    image[bounds] += stamp[bounds]


def render_galaxy(image, field, x, y, flux, color, re_pix, sersic_n, g1_gal, g2_gal):
    """Render a Sersic galaxy convolved with the local Moffat PSF into the image.

    Galaxies are extended detections (star_type=2): the blend likelihood models them as
    misspecified point sources, so they let the galaxy-handling modes (exclude/mask/
    component) be selected on simulations where the star truth is known exactly.
    """
    fwhm, g1_psf, g2_psf = true_psf_params(field, x, y, color)
    psf = psf_profile(fwhm).shear(g1=g1_psf, g2=g2_psf)
    galaxy = galsim.Sersic(n=sersic_n, half_light_radius=re_pix * PIXEL_SCALE, flux=flux)
    galaxy = galaxy.shear(g1=g1_gal, g2=g2_gal)
    profile = galsim.Convolve([galaxy, psf])
    stamp = profile.drawImage(
        nx=PATCH * 2,
        ny=PATCH * 2,
        scale=PIXEL_SCALE,
        center=galsim.PositionD(x + 1.0, y + 1.0),  # galsim is 1-based
    )
    bounds = stamp.bounds & image.bounds
    image[bounds] += stamp[bounds]


def sample_galaxies(rng, n_galaxies):
    """Galaxy detection properties: position, flux, color, size, Sersic index, shape."""
    half = PATCH // 2
    margin = half + 2
    shear_g = rng.uniform(0.0, GALAXY_SHEAR_MAX, n_galaxies)
    shear_phi = rng.uniform(0.0, np.pi, n_galaxies)
    return {
        "x": rng.uniform(margin, WIDTH - margin, n_galaxies),
        "y": rng.uniform(margin, HEIGHT - margin, n_galaxies),
        "flux": np.exp(
            rng.uniform(np.log(GALAXY_FLUX_RANGE[0]), np.log(GALAXY_FLUX_RANGE[1]), n_galaxies)
        ),
        "color": np.clip(rng.normal(COLOR_MEAN, COLOR_SCATTER, n_galaxies), -0.5, 3.0),
        "re_pix": rng.uniform(*GALAXY_RE_RANGE, n_galaxies),
        "sersic_n": rng.uniform(*GALAXY_SERSIC_RANGE, n_galaxies),
        "g1": shear_g * np.cos(2.0 * shear_phi),
        "g2": shear_g * np.sin(2.0 * shear_phi),
    }


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


def simulate_exposure(exposure_seed, fits_dir, chromatic=False, galaxy_fraction=0.0):
    """Render one exposure; returns the v2-schema per-exposure record.

    galaxy_fraction injects that many galaxy detections per star (star_type=2); they
    render into the image (contaminating nearby star cutouts) and carry their own
    cutouts so the blend likelihood's galaxy modes can be selected on simulations.
    """
    rng = np.random.default_rng(exposure_seed)
    field = sample_field(rng, chromatic=chromatic)
    n_stars = rng.integers(*N_STARS_RANGE)
    n_galaxies = round(galaxy_fraction * float(n_stars))

    half = PATCH // 2
    margin = half + 2
    x = rng.uniform(margin, WIDTH - margin, n_stars)
    y = rng.uniform(margin, HEIGHT - margin, n_stars)
    flux = np.exp(rng.uniform(np.log(FLUX_RANGE[0]), np.log(FLUX_RANGE[1]), n_stars))
    color = np.clip(rng.normal(COLOR_MEAN, COLOR_SCATTER, n_stars), -0.5, 3.0)

    image = galsim.Image(WIDTH, HEIGHT, scale=PIXEL_SCALE, dtype=np.float32)
    for x0, y0, flux0, color0 in zip(x, y, flux, color, strict=True):
        render_star(image, field, x0, y0, flux0, color0)

    gal = sample_galaxies(rng, n_galaxies)
    gal_cols = zip(
        gal["x"],
        gal["y"],
        gal["flux"],
        gal["color"],
        gal["re_pix"],
        gal["sersic_n"],
        gal["g1"],
        gal["g2"],
        strict=True,
    )
    for gx, gy, gf, gc, gre, gn, gg1, gg2 in gal_cols:
        render_galaxy(image, field, gx, gy, gf, gc, gre, gn, gg1, gg2)
    sci = image.array + rng.normal(0, NOISE_SIGMA, image.array.shape).astype(np.float32)

    n_det = n_stars + n_galaxies
    x = np.concatenate([x, gal["x"]])
    y = np.concatenate([y, gal["y"]])
    flux = np.concatenate([flux, gal["flux"]])
    color = np.concatenate([color, gal["color"]])

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

    # blended stars poison a single-star likelihood: like the real extraction, isolated
    # stars are clean (0), blended stars are context-only (1), galaxies are 2. Isolation
    # is computed over stars only — galaxy contamination is handled by the blend
    # likelihood's galaxy_mode, not by relabeling the star.
    star_flux = flux[:n_stars]
    tree = cKDTree(np.column_stack([x[:n_stars], y[:n_stars]]))
    pairs = tree.query_pairs(ISOLATION_RADIUS, output_type="ndarray")
    isolated = np.ones(n_stars, dtype=bool)
    if len(pairs) > 0:
        first, second = pairs[:, 0], pairs[:, 1]
        keep_first = ~(star_flux[second] > ISOLATION_FLUX_RATIO * star_flux[first])
        keep_second = ~(star_flux[first] > ISOLATION_FLUX_RATIO * star_flux[second])
        np.logical_and.at(isolated, first, keep_first)
        np.logical_and.at(isolated, second, keep_second)
    star_type = np.full(n_det, TYPE_GALAXY, dtype=np.uint8)
    star_type[:n_stars] = np.where(isolated, TYPE_CLEAN, TYPE_STAR_CONTEXT)

    n_slots = n_det  # stars + galaxies; padding handled by fixed-slot packing
    true_fwhm, true_g1, true_g2 = true_psf_params(field, x, y, color)
    return {
        "n_stars": int(n_slots),
        "cutouts": cutouts,
        "variance": np.full(cutouts.shape, NOISE_SIGMA**2, dtype=np.float32),
        "valid_pixels": np.ones(cutouts.shape, dtype=bool),
        "flux": flux.astype(np.float32),
        "flux_err": np.full(n_det, NOISE_SIGMA * PATCH, dtype=np.float32),
        "snr": (flux / (NOISE_SIGMA * PATCH)).astype(np.float32),
        "sky": np.zeros(n_det, dtype=np.float32),
        "x_pixel": x.astype(np.float32),
        "y_pixel": y.astype(np.float32),
        "color": color.astype(np.float32),
        "star_type": star_type,
        "star_id": exposure_seed * 100_000 + np.arange(n_det, dtype=np.int64),
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


def save_chunk(records, out_dir, worker_id, n_chunks):
    """Pack a chunk into fixed slots sized to its largest exposure (stars + galaxies)."""
    n_slots = max(record["n_stars"] for record in records)
    out_path = Path(out_dir) / f"desstars_sim_w{worker_id:02d}_{n_chunks:04d}.pt"
    torch.save(pack_records(records, n_slots), out_path)
    print(f"[worker {worker_id}] wrote {out_path}")


def worker(worker_id, seeds, out_dir, fits_dir, exposures_per_file, chromatic, galaxy_fraction):
    records = []
    n_chunks = 0
    for seed in seeds:
        record = simulate_exposure(
            seed, fits_dir, chromatic=chromatic, galaxy_fraction=galaxy_fraction
        )
        records.append(record)
        if len(records) == exposures_per_file:
            save_chunk(records, out_dir, worker_id, n_chunks)
            records, n_chunks = [], n_chunks + 1
    if records:
        save_chunk(records, out_dir, worker_id, n_chunks)


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
    parser.add_argument(
        "--galaxy-fraction",
        type=float,
        default=0.0,
        help="inject this many galaxy detections (star_type=2) per star",
    )
    parser.add_argument(
        "--psf-model",
        default="moffat",
        choices=["moffat", "kolmogorov", "realistic"],
        help="atmospheric PSF model; kolmogorov is sharper-cored (W3 sharp-core testbed)",
    )
    args = parser.parse_args()
    set_psf_model(args.psf_model)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    Path(args.fits_dir).mkdir(parents=True, exist_ok=True)

    seeds = list(range(args.n_exposures))
    chunks = [seeds[i :: args.num_workers] for i in range(args.num_workers)]
    processes = [
        mp.Process(
            target=worker,
            args=(
                i,
                chunk,
                args.out_dir,
                args.fits_dir,
                args.exposures_per_file,
                args.chromatic,
                args.galaxy_fraction,
            ),
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
