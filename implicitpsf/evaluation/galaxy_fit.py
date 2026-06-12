"""Differentiable cutout-level galaxy fitting through a pixel-convolved PSF.

The model for a pixel is (continuous galaxy profile CONV effective PSF) at the pixel
center — no second pixel integration, since the kernels are already pixel-convolved.
Discretization is exact by construction: with odd oversampling s, the sample lattice
g_n = n/s + (1/s - 1)/2 - half contains every native pixel center, and querying the
PSF kernel at the INTEGER part of the galaxy position (the sub-pixel offset lives in
the galaxy profile instead) makes the zero-padded discrete convolution produce native
centers at output indices s*j + (s*half + s - half - 1) with no interpolation.

The galaxy profile enters as fine-cell averages via a scaling trick around
AstroMatch's evaluate_sersic_profile: patch_size = s*patch with re and offsets
scaled by s makes its "pixels" our fine cells, and its block-averaging integrates
each cell — exact for the cuspy galaxy; no second native-pixel integration occurs.

Flux convention: kernels are stamp-sum normalized (render_at), so PSF flux outside
the stamp is redistributed inside (~1% for a beta=2.5 Moffat at FWHM=4 on 32 px).
Fitted fluxes are therefore relative to the in-stamp PSF — identical convention for
every PSF arm in the recovery experiment, so comparisons are unaffected.
"""

import torch
from astromatch.simulator.sersic_profile import (
    eta1eta2_to_q_pa,
    evaluate_sersic_profile,
    sersic_analytic_flux,
)

OVERSAMPLE = 3  # must be odd: the fine lattice then contains native pixel centers
SUBINTEGRATE = 4  # even sub-grid per fine cell for galaxy integration (cusp safety)


def fine_sersic_samples(
    dx, dy, sersic_n, sersic_re, eta1, eta2, patch_size, s=OVERSAMPLE, sub=SUBINTEGRATE
):
    """Fine-lattice cell averages of the continuous unit-flux Sersic profile.

    Each fine cell is integrated on a sub x sub grid (even, so a cusp centered on a
    cell is straddled, never point-sampled): high-n Sersic cores are far too cuspy
    for point sampling at any practical s. Cell-averaging makes the galaxy factor of
    the convolution quadrature exact; the remaining error is the kernel's smooth
    variation within a cell.

    Args:
        dx, dy: (n,) galaxy center offsets from round(position), in native pixels
        sersic_n, sersic_re, eta1, eta2: (n,) profile parameters (re in native pixels)
        patch_size: native stamp width
        s: oversampling factor
        sub: per-cell integration factor (even)

    Returns:
        (n, s*patch, s*patch) cell-averaged samples, normalized so the continuous
        profile has unit flux (values are mean surface brightness times the
        fine-cell area 1/s^2)
    """
    half = patch_size // 2
    fine = patch_size * s
    # fine-lattice index of a galaxy at native offset (d) from round(position):
    # lattice g_n = n/s + (1/s-1)/2 - half  =>  n(d) = s*(d + half) + (s-1)/2
    x_pos = s * (dx + half) + (s - 1) / 2.0
    y_pos = s * (dy + half) + (s - 1) / 2.0

    axis_ratio, position_angle = eta1eta2_to_q_pa(eta1, eta2)
    zeros = torch.zeros_like(dx)
    profile = evaluate_sersic_profile(
        x_pos,
        y_pos,
        sersic_n,
        sersic_re * s,
        axis_ratio,
        position_angle,
        zeros,
        zeros,
        patch_size=fine,
        oversample=sub,
    )
    flux_norm = sersic_analytic_flux(sersic_n, sersic_re * s, zeros)
    return profile / flux_norm.reshape(-1, 1, 1)


def convolve_with_epsf(galaxy_fine, kernel_fine, patch_size, s=OVERSAMPLE):
    """Convolve fine galaxy samples with a pixel-convolved PSF kernel; read native pixels.

    Args:
        galaxy_fine: (n, s*patch, s*patch) from fine_sersic_samples (unit flux)
        kernel_fine: (n, s*patch, s*patch) effective-PSF samples from
            render_at(..., oversample=s) queried at the INTEGER galaxy position
            (unit native sum: fine samples sum to s^2)
        patch_size: native stamp width

    Returns:
        (n, patch, patch) model stamps with unit total flux (up to stamp truncation)
    """
    fine = patch_size * s
    full = 2 * fine - 1
    gal_f = torch.fft.rfft2(galaxy_fine, s=(full, full))
    ker_f = torch.fft.rfft2(kernel_fine, s=(full, full))
    conv = torch.fft.irfft2(gal_f * ker_f, s=(full, full))
    # native center j sits at full-convolution index s*j + offset on each axis:
    # x_j - g_n = g_m requires m + n = s*j + s*half + s - 1 (lattice algebra)
    half = patch_size // 2
    offset = s * half + s - 1
    idx = offset + s * torch.arange(patch_size, device=conv.device)
    # flux bookkeeping: fine_sersic_samples normalizes by the analytic flux computed
    # in FINE pixels, which already carries the 1/s^2 cell area; the kernel's
    # native-sum convention supplies the rest — no further factor here
    return conv[:, idx][:, :, idx]


def fit_galaxies(
    cutouts,
    variance,
    valid,
    kernels_fine,
    sersic_n,
    init_flux,
    init_re,
    patch_size=32,
    steps=300,
    lr=0.03,
):
    """Batched gradient fit of (flux, dx, dy, re, eta1, eta2) with fixed Sersic n.

    Args:
        cutouts, variance: (n, patch, patch) data and noise
        valid: (n, patch, patch) bool
        kernels_fine: (n, s*patch, s*patch) effective-PSF kernels (integer-position)
        sersic_n: (n,) fixed Sersic indices
        init_flux, init_re: (n,) initial values

    Returns:
        dict of fitted (n,) tensors: flux, dx, dy, re, eta1, eta2, chi2
    """
    n_gal = len(cutouts)
    log_flux = torch.log(init_flux.clamp(min=1.0)).clone().requires_grad_(True)
    dx = torch.zeros(n_gal, requires_grad=True)
    dy = torch.zeros(n_gal, requires_grad=True)
    log_re = torch.log(init_re.clamp(min=0.3)).clone().requires_grad_(True)
    eta1 = torch.zeros(n_gal, requires_grad=True)
    eta2 = torch.zeros(n_gal, requires_grad=True)

    weights = valid.float() / variance
    optimizer = torch.optim.Adam([log_flux, dx, dy, log_re, eta1, eta2], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=lr / 100)
    for _ in range(steps):
        optimizer.zero_grad()
        profile = fine_sersic_samples(
            dx, dy, sersic_n, log_re.exp().clamp(0.3, 20.0), eta1, eta2, patch_size
        )
        stamps = convolve_with_epsf(profile, kernels_fine, patch_size)
        model = log_flux.exp().reshape(-1, 1, 1) * stamps
        chi2 = (weights * (cutouts - model).square()).sum(dim=(-2, -1))
        chi2.sum().backward()
        optimizer.step()
        scheduler.step()

    with torch.no_grad():
        n_valid = valid.sum(dim=(-2, -1)).clamp(min=7)
        final_chi2 = chi2.detach() / (n_valid - 6)
    return {
        "flux": log_flux.exp().detach(),
        "dx": dx.detach(),
        "dy": dy.detach(),
        "re": log_re.exp().detach(),
        "eta1": eta1.detach(),
        "eta2": eta2.detach(),
        "chi2": final_chi2,
    }
