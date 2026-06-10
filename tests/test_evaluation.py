import galsim
import numpy as np
import pytest

from implicitpsf.evaluation.chi2 import reduced_chi2
from implicitpsf.evaluation.moments import PIXEL_SCALE, hsm_moments
from implicitpsf.evaluation.rho_stats import rho_statistics

PATCH = 32


def gaussian_stamp(sigma_arcsec, e1=0.0, e2=0.0, offset=(0.0, 0.0), flux=1.0):
    profile = galsim.Gaussian(sigma=sigma_arcsec, flux=flux)
    shear = galsim.Shear(e1=e1, e2=e2)
    image = profile.shear(shear).drawImage(
        nx=PATCH, ny=PATCH, scale=PIXEL_SCALE, offset=offset, method="no_pixel"
    )
    return image.array


def test_hsm_recovers_gaussian_size_and_shape():
    sigma = 2.0 * PIXEL_SCALE  # 2-pixel sigma
    stamps = np.stack(
        [
            gaussian_stamp(sigma),
            gaussian_stamp(sigma, e1=0.05),
            gaussian_stamp(sigma, e2=-0.04, offset=(0.3, -0.2)),
        ]
    )
    moments = hsm_moments(stamps)
    assert (moments["flag"] == 0).all()
    np.testing.assert_allclose(moments["T"], 2 * sigma**2, rtol=0.01)
    np.testing.assert_allclose(moments["e1"], [0.0, 0.05, 0.0], atol=0.005)
    np.testing.assert_allclose(moments["e2"], [0.0, 0.0, -0.04], atol=0.005)
    # third stamp offset by (0.3, -0.2) pixels from the image center
    center = (PATCH - 1) / 2
    np.testing.assert_allclose(moments["centroid_x"][2], center + 0.3, atol=0.02)
    np.testing.assert_allclose(moments["centroid_y"][2], center - 0.2, atol=0.02)


def test_reduced_chi2_perfect_model_is_zero_and_amplitude_recovered():
    model = gaussian_stamp(2 * PIXEL_SCALE)[None]
    observed = 137.0 * model
    variance = np.ones_like(model)
    valid = np.ones_like(model, dtype=bool)
    result = reduced_chi2(observed, model, variance, valid)
    np.testing.assert_allclose(result["chi2"], 0.0, atol=1e-12)
    np.testing.assert_allclose(result["amplitude"], 137.0, rtol=1e-6)


def test_reduced_chi2_is_calibrated_for_pure_noise():
    rng = np.random.default_rng(0)
    model = np.tile(gaussian_stamp(2 * PIXEL_SCALE), (200, 1, 1))
    sigma = 0.01
    observed = 50.0 * model + rng.normal(0, sigma, model.shape)
    variance = np.full(model.shape, sigma**2)
    valid = np.ones(model.shape, dtype=bool)
    result = reduced_chi2(observed, model, variance, valid)
    assert 0.95 < result["chi2"].mean() < 1.05


def make_exposure_table(rng, n_stars, de1=0.0, de2=0.0, noise=0.0):
    return {
        "x_pixel": rng.uniform(0, 2048, n_stars),
        "y_pixel": rng.uniform(0, 4096, n_stars),
        "e1": rng.normal(0, 0.02, n_stars),
        "e2": rng.normal(0, 0.02, n_stars),
        "de1": de1 + rng.normal(0, noise, n_stars),
        "de2": de2 + rng.normal(0, noise, n_stars),
        "t_frac": rng.normal(0.01, 0.005, n_stars),
    }


def test_rho1_constant_residual_field():
    rng = np.random.default_rng(0)
    tables = [make_exposure_table(rng, 200, de1=0.03, de2=0.04) for _ in range(10)]
    rho = rho_statistics(tables)
    expected = 0.03**2 + 0.04**2  # <de* de> of a constant field
    np.testing.assert_allclose(rho["rho1"]["xip"], expected, rtol=1e-6)
    assert (rho["rho1"]["npairs"] > 0).all()


def test_rho1_uncorrelated_residuals_near_zero():
    rng = np.random.default_rng(1)
    tables = [make_exposure_table(rng, 300, noise=0.05) for _ in range(20)]
    rho = rho_statistics(tables)
    # 5-sigma shot-noise bound: std(xip) = sqrt(2) * sigma_component^2 / sqrt(npairs)
    bound = 5 * np.sqrt(2) * 0.05**2 / np.sqrt(rho["rho1"]["npairs"])
    assert (np.abs(rho["rho1"]["xip"]) < bound).all()


def test_rho_statistics_requires_exposures():
    with pytest.raises(ValueError, match="no exposures"):
        rho_statistics([])
