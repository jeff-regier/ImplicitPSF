"""Blocking validation for the galaxy-fitting convolution math (M8 gates).

The delta test and the galsim cross-check must pass before any injection-recovery
experiment runs: they pin the fine-lattice phase bookkeeping that everything rests on.
"""

import galsim
import numpy as np
import torch

from implicitpsf.blend import sample_grid
from implicitpsf.evaluation.galaxy_fit import (
    OVERSAMPLE,
    convolve_with_epsf,
    fine_sersic_samples,
    fit_galaxies,
)
from implicitpsf.evaluation.moments import PIXEL_SCALE

PATCH = 32
S = OVERSAMPLE
FINE = PATCH * S


def moffat_epsf_fine(fwhm_pixels, g1=0.0, g2=0.0, subpixel=(0.0, 0.0), beta=2.5):
    """Effective (pixel-convolved) Moffat sampled on our fine lattice.

    Returns fine samples normalized to native unit sum (sum = S^2), matching
    render_at's convention, for a PSF centered at `subpixel` from round(position).
    """
    profile = galsim.Convolve(
        galsim.Moffat(beta=beta, fwhm=fwhm_pixels * PIXEL_SCALE).shear(g1=g1, g2=g2),
        galsim.Pixel(PIXEL_SCALE),
    )
    grid = sample_grid(PATCH, S, torch.device("cpu"), torch.float64).numpy()
    shifted = grid - np.asarray(subpixel)
    vals = np.array(
        [profile.xValue(galsim.PositionD(p[0] * PIXEL_SCALE, p[1] * PIXEL_SCALE)) for p in shifted]
    ).reshape(FINE, FINE)
    vals = vals / vals.sum() * S * S
    return torch.tensor(vals)


def test_delta_galaxy_reproduces_kernel_at_native_centers():
    """A lattice-aligned discrete delta convolved with the kernel must reproduce the
    kernel at native pixel centers exactly (index bookkeeping, no quantization)."""
    kernel = moffat_epsf_fine(4.0, g1=0.03, g2=-0.02)
    # build the delta directly on the fine lattice at offset (1/S, -1/S)
    half = PATCH // 2
    delta = torch.zeros(1, FINE, FINE, dtype=kernel.dtype)
    n_x = S * (half) + (S - 1) // 2 + 1  # lattice index of offset +1/S
    n_y = S * (half) + (S - 1) // 2 - 1  # lattice index of offset -1/S
    delta[0, n_y, n_x] = 1.0  # unit-flux discrete delta (rows = y)
    out = convolve_with_epsf(delta, kernel.unsqueeze(0), PATCH)[0]

    # expected: ePSF point samples at native centers shifted by (-1/S, +1/S)
    profile = galsim.Convolve(
        galsim.Moffat(beta=2.5, fwhm=4.0 * PIXEL_SCALE).shear(g1=0.03, g2=-0.02),
        galsim.Pixel(PIXEL_SCALE),
    )
    grid_native = sample_grid(PATCH, 1, torch.device("cpu"), torch.float64).numpy()
    expected = np.array(
        [
            profile.xValue(
                galsim.PositionD((p[0] - 1 / S) * PIXEL_SCALE, (p[1] + 1 / S) * PIXEL_SCALE)
            )
            for p in grid_native
        ]
    ).reshape(PATCH, PATCH)
    got = out.numpy() / out.numpy().sum()
    expected = expected / expected.sum()
    assert np.abs(got - expected).max() < 1e-6


def test_galsim_cross_check_sersic_times_psf():
    """Full pipeline vs galsim.Convolve(Sersic, Moffat, Pixel) drawn no_pixel.

    beta=4.5 keeps the Moffat's out-of-stamp flux ~2e-4: our kernels are stamp-sum
    normalized (render_at convention), so a wingy PSF (beta=2.5 leaves ~1% outside
    32 px) would mismatch the untruncated galsim reference by construction.
    """
    fwhm, re, q, beta = 4.0, 3.0, 0.6, 4.5
    pa_rad = 0.5
    dx, dy = 0.3, -0.4

    kernel = moffat_epsf_fine(fwhm, beta=beta)
    eta = -np.log(q)  # eta1/eta2 parameterization: |eta| = -log q at angle pa
    eta1 = torch.tensor([eta * np.cos(2 * pa_rad)])
    eta2 = torch.tensor([eta * np.sin(2 * pa_rad)])
    gal_fine = fine_sersic_samples(
        torch.tensor([dx]),
        torch.tensor([dy]),
        torch.tensor([1.0]),
        torch.tensor([re], dtype=torch.float32),
        eta1.float(),
        eta2.float(),
        PATCH,
    )
    ours = convolve_with_epsf(gal_fine, kernel.unsqueeze(0), PATCH)[0].numpy()

    # galsim reference: same definitions — half-light radius along the geometric mean
    gal = galsim.Sersic(n=1.0, half_light_radius=re * PIXEL_SCALE, flux=1.0)
    gal = gal.shear(q=q, beta=pa_rad * galsim.radians)
    obj = galsim.Convolve(
        gal, galsim.Moffat(beta=beta, fwhm=fwhm * PIXEL_SCALE), galsim.Pixel(PIXEL_SCALE)
    )
    half = PATCH // 2
    bounds = galsim.BoundsI(1, PATCH, 1, PATCH)
    image = galsim.Image(bounds, scale=PIXEL_SCALE)
    center = galsim.PositionD(half + 1 + dx, half + 1 + dy)
    obj.drawImage(image=image, center=center, method="no_pixel")
    reference = image.array

    peak = reference.max()
    # measured s=3 quadrature floor (kernel midpoint error within fine cells): 1.0e-3
    assert np.abs(ours - reference).max() < 2e-3 * peak


def test_fit_recovers_injected_galaxy():
    rng = np.random.default_rng(0)
    kernel = moffat_epsf_fine(4.0)
    true = dict(flux=5000.0, dx=0.2, dy=-0.3, re=2.5, eta1=0.3, eta2=-0.2)
    gal = fine_sersic_samples(
        torch.tensor([true["dx"]]),
        torch.tensor([true["dy"]]),
        torch.tensor([1.0]),
        torch.tensor([true["re"]]),
        torch.tensor([true["eta1"]]),
        torch.tensor([true["eta2"]]),
        PATCH,
    )
    clean = true["flux"] * convolve_with_epsf(gal, kernel.unsqueeze(0), PATCH)[0]
    noisy = clean + torch.tensor(rng.normal(0, 1.0, clean.shape), dtype=clean.dtype)

    result = fit_galaxies(
        noisy.unsqueeze(0).float(),
        torch.ones(1, PATCH, PATCH),
        torch.ones(1, PATCH, PATCH, dtype=torch.bool),
        kernel.unsqueeze(0).float(),
        sersic_n=torch.tensor([1.0]),
        init_flux=torch.tensor([4000.0]),
        init_re=torch.tensor([3.5]),
        steps=400,
    )
    assert abs(result["flux"][0].item() / true["flux"] - 1) < 0.05
    assert abs(result["re"][0].item() / true["re"] - 1) < 0.05
    assert abs(result["eta1"][0].item() - true["eta1"]) < 0.05
    assert abs(result["eta2"][0].item() - true["eta2"]) < 0.05
    assert result["chi2"][0].item() < 1.2
