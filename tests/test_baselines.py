"""End-to-end baseline harness test on a synthetic exposure with known PSF.

Builds a small CCD image of Gaussian stars at random subpixel phases, fits PIFF and
PSFEx through the same code paths used in evaluation, renders models at held-out star
positions, and checks moments and chi^2. This validates every coordinate convention
(SEP 0-based vs FITS 1-based, stamp corners, subpixel offsets) in one place.
"""

import galsim
import numpy as np
import pytest
from astropy.io import fits

from implicitpsf.baselines.catalogs import write_piff_catalog, write_psfex_ldac
from implicitpsf.baselines.piff_runner import fit_piff, render_piff
from implicitpsf.baselines.psfex_runner import fit_psfex, render_psfex
from implicitpsf.evaluation.chi2 import reduced_chi2
from implicitpsf.evaluation.moments import PIXEL_SCALE, hsm_moments

WIDTH, HEIGHT = 1024, 2048
PATCH = 32
SIGMA_PIXELS = 2.0
NOISE_SIGMA = 0.5
N_FIT, N_HELD_OUT = 90, 24


def synthetic_exposure(tmp_path):
    rng = np.random.default_rng(7)
    grid_x, grid_y = np.meshgrid(np.arange(8), np.arange(16))
    x = (grid_x.ravel() + 0.5) * (WIDTH / 8) + rng.uniform(-20, 20, 128)
    y = (grid_y.ravel() + 0.5) * (HEIGHT / 16) + rng.uniform(-20, 20, 128)
    flux = rng.uniform(2e3, 2e4, 128)  # realistic star SNR; at SNR ~1e4 no model reaches chi2=1

    image = galsim.Image(WIDTH, HEIGHT, scale=PIXEL_SCALE, dtype=np.float32)
    psf = galsim.Gaussian(sigma=SIGMA_PIXELS * PIXEL_SCALE)
    for x0, y0, flux0 in zip(x, y, flux, strict=True):
        stamp = (psf * flux0).drawImage(
            nx=PATCH,
            ny=PATCH,
            scale=PIXEL_SCALE,
            center=galsim.PositionD(x0 + 1.0, y0 + 1.0),  # galsim is 1-based
        )
        bounds = stamp.bounds & image.bounds
        image[bounds] += stamp[bounds]
    sci = image.array + rng.normal(0, NOISE_SIGMA, image.array.shape).astype(np.float32)

    header = fits.Header()
    header["CTYPE1"], header["CTYPE2"] = "RA---TAN", "DEC--TAN"
    header["CRVAL1"], header["CRVAL2"] = 30.0, -30.0
    header["CRPIX1"], header["CRPIX2"] = WIDTH / 2, HEIGHT / 2
    scale_deg = PIXEL_SCALE / 3600
    header["CD1_1"], header["CD1_2"] = -scale_deg, 0.0
    header["CD2_1"], header["CD2_2"] = 0.0, scale_deg
    header["FWHM"] = SIGMA_PIXELS * 2.355

    sci_hdu = fits.ImageHDU(sci, header=header, name="SCI")
    msk_hdu = fits.ImageHDU(np.zeros(sci.shape, dtype=np.int16), header=header, name="MSK")
    wgt = np.full(sci.shape, 1.0 / NOISE_SIGMA**2, dtype=np.float32)
    wgt_hdu = fits.ImageHDU(wgt, header=header, name="WGT")
    fits_path = tmp_path / "synthetic.fits"
    fits.HDUList([fits.PrimaryHDU(), sci_hdu, msk_hdu, wgt_hdu]).writeto(fits_path)

    return fits_path, sci, x, y, flux


def data_stamps(sci, x, y):
    half = PATCH // 2
    stamps = np.zeros((len(x), PATCH, PATCH))
    for index, (x0, y0) in enumerate(zip(x, y, strict=True)):
        col, row = round(float(x0)) - half, round(float(y0)) - half
        stamps[index] = sci[row : row + PATCH, col : col + PATCH]
    return stamps


def assert_model_quality(model_stamps, star_stamps):
    star_moments = hsm_moments(star_stamps)
    model_moments = hsm_moments(model_stamps)
    assert (star_moments["flag"] == 0).all()
    assert (model_moments["flag"] == 0).all()

    t_frac = (star_moments["T"] - model_moments["T"]) / star_moments["T"]
    assert np.abs(np.median(t_frac)) < 0.02
    assert np.abs(model_moments["e1"] - star_moments["e1"]).max() < 0.02
    assert np.abs(model_moments["e2"] - star_moments["e2"]).max() < 0.02

    # model centroid must land where the star does (subpixel conventions)
    assert np.abs(model_moments["centroid_x"] - star_moments["centroid_x"]).max() < 0.1
    assert np.abs(model_moments["centroid_y"] - star_moments["centroid_y"]).max() < 0.1

    variance = np.full(star_stamps.shape, NOISE_SIGMA**2)
    valid = np.ones(star_stamps.shape, dtype=bool)
    result = reduced_chi2(star_stamps, model_stamps, variance, valid)
    assert 0.5 < np.median(result["chi2"]) < 1.5


@pytest.fixture(scope="module")
def exposure(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("synthetic_exposure")
    fits_path, sci, x, y, flux = synthetic_exposure(tmp_path)
    held_out = slice(N_FIT, N_FIT + N_HELD_OUT)
    return {
        "tmp_path": tmp_path,
        "fits_path": fits_path,
        "sci": sci,
        "fit": (x[:N_FIT], y[:N_FIT], flux[:N_FIT]),
        "held_out": (x[held_out], y[held_out]),
    }


def test_piff_round_trip(exposure):
    x_fit, y_fit, flux_fit = exposure["fit"]
    cat_path = exposure["tmp_path"] / "piff_cat.fits"
    write_piff_catalog(x_fit, y_fit, flux_fit, cat_path)

    psf = fit_piff(exposure["fits_path"], cat_path, exposure["tmp_path"] / "model.piff")
    x_test, y_test = exposure["held_out"]
    model_stamps = render_piff(psf, x_test, y_test, patch_size=PATCH)
    assert_model_quality(model_stamps, data_stamps(exposure["sci"], x_test, y_test))


def test_psfex_round_trip(exposure):
    x_fit, y_fit, flux_fit = exposure["fit"]
    stamps = data_stamps(exposure["sci"], x_fit, y_fit)
    snr = flux_fit / (NOISE_SIGMA * PATCH)
    with fits.open(exposure["fits_path"]) as hdul:
        image_header = hdul["SCI"].header

    ldac_path = exposure["tmp_path"] / "psfex_cat.fits"
    write_psfex_ldac(
        stamps,
        np.ones(stamps.shape, dtype=bool),
        x_fit,
        y_fit,
        flux_fit,
        np.full(len(x_fit), NOISE_SIGMA * PATCH),
        snr,
        SIGMA_PIXELS * 2.355,
        image_header,
        ldac_path,
        background_dev=NOISE_SIGMA,
    )

    psf_path = fit_psfex(ldac_path, exposure["tmp_path"] / "psfex_out")
    x_test, y_test = exposure["held_out"]
    model_stamps = render_psfex(psf_path, exposure["fits_path"], x_test, y_test, PATCH)
    assert_model_quality(model_stamps, data_stamps(exposure["sci"], x_test, y_test))
