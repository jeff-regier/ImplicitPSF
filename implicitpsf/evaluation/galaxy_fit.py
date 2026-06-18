"""Differentiable cutout-level galaxy fitting through a pixel-convolved PSF.

The model for a pixel is (continuous galaxy profile CONV effective PSF) at the pixel
center — no second pixel integration, since the kernels are already pixel-convolved.
Discretization is exact by construction: with odd oversampling s, the sample lattice
g_n = n/s + (1/s - 1)/2 - half contains every native pixel center, and querying the
PSF kernel at the INTEGER part of the galaxy position (the sub-pixel offset lives in
the galaxy profile instead) makes the zero-padded discrete convolution produce native
centers at output indices s*j + (s*half + s - half - 1) with no interpolation.

The galaxy profile enters as fine-cell averages from _sersic_cell_averages (sub-grid
block averaging per fine cell — exact for the cuspy galaxy; no second native-pixel
integration occurs), with the ellipse written as a smooth quadratic form in eta.

Flux convention: kernels are stamp-sum normalized (render_at), so PSF flux outside
the stamp is redistributed inside (~1% for a beta=2.5 Moffat at FWHM=4 on 32 px).
Fitted fluxes are therefore relative to the in-stamp PSF — identical convention for
every PSF arm in the recovery experiment, so comparisons are unaffected.
"""

import torch

OVERSAMPLE = 3  # must be odd: the fine lattice then contains native pixel centers
SUBINTEGRATE = 4  # even sub-grid per fine cell for galaxy integration (cusp safety)


def sersic_bn(n):
    """Sersic b_n (intensity-at-Re normalization constant), differentiable in n.

    b_n satisfies the half-light condition gamma(2n, b_n)/Gamma(2n) = 1/2; we use the
    Ciotti & Bertin (1999) asymptotic expansion (error < 1e-4 for n > 0.36), which is a
    closed form in torch (the catalogue n range is well inside this). Reimplemented here
    so the codebase does not depend on the (unstable) astromatch package.
    """
    return (
        2.0 * n - 1.0 / 3.0 + 4.0 / (405.0 * n) + 46.0 / (25515.0 * n**2)
        + 131.0 / (1148175.0 * n**3) - 2194697.0 / (30690717750.0 * n**4)
    )


def sersic_total_flux(n, re):
    """Total flux of a round Sersic with unit intensity at Re (Ie=1):
    F = 2*pi*n*Re^2 * exp(b_n) * Gamma(2n) / b_n^(2n). Only the (n, re) dependence
    matters here -- the per-galaxy fit amplitude absorbs the overall scale."""
    bn = sersic_bn(n)
    return 2.0 * torch.pi * n * re**2 * torch.exp(bn + torch.lgamma(2.0 * n)) / bn ** (2.0 * n)


def _sersic_cell_averages(x_pos, y_pos, sersic_n, sersic_re, eta1, eta2, n_cells, sub):
    """Cell-averaged Sersic intensities with the ellipse as a quadratic form in eta.

    AstroMatch's (q, PA) route computes PA = atan2(eta2, eta1 + eps), whose gradient
    is eta1 / |eta|^2 — it explodes at the round-galaxy origin where every fit starts,
    flooding Adam's second-moment buffer and freezing eta2 (eta1's branch is exactly
    zero there, which is why only eta2 broke). The area-preserving elliptical radius
    is instead written directly as R^2 = M11 x^2 + 2 M12 x y + M22 y^2 with
    M = cosh|eta| I - sinh|eta| [[cos2PA, sin2PA], [sin2PA, -cos2PA]] — analytic in
    (eta1, eta2) everywhere, identical values to the (q, PA) form.
    """
    coords = (torch.arange(n_cells * sub, dtype=x_pos.dtype) + 0.5) / sub - 0.5
    xx = coords.unsqueeze(0) - x_pos[:, None]  # (n, fine*sub)
    yy = coords.unsqueeze(0) - y_pos[:, None]

    m = torch.sqrt(eta1**2 + eta2**2 + 1e-12)
    sinch = torch.sinh(m) / m  # m >= 1e-6, and sinch -> 1 smoothly
    m11 = (torch.cosh(m) - eta1 * sinch)[:, None, None]
    m22 = (torch.cosh(m) + eta1 * sinch)[:, None, None]
    m12 = (-eta2 * sinch)[:, None, None]

    x2 = (xx**2).unsqueeze(1)  # (n, 1, fine*sub) — broadcast outer product
    y2 = (yy**2).unsqueeze(2)
    xy = yy.unsqueeze(2) * xx.unsqueeze(1)
    r = torch.sqrt((m11 * x2 + m22 * y2 + 2.0 * m12 * xy).clamp(min=1e-12))

    bn = sersic_bn(sersic_n)[:, None, None]
    re = sersic_re[:, None, None]
    n_inv = (1.0 / sersic_n)[:, None, None]
    profile = torch.exp(-bn * (torch.pow(r.clamp(min=1e-6) / re, n_inv) - 1.0))
    blocks = profile.reshape(-1, n_cells, sub, n_cells, sub)
    return blocks.mean(dim=(2, 4))


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

    profile = _sersic_cell_averages(x_pos, y_pos, sersic_n, sersic_re * s, eta1, eta2, fine, sub)
    flux_norm = sersic_total_flux(sersic_n, sersic_re * s)
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
    fit_n=False,
):
    """Batched gradient fit of (flux, dx, dy, re, eta1, eta2) and optionally Sersic n.

    Args:
        cutouts, variance: (n, patch, patch) data and noise
        valid: (n, patch, patch) bool
        kernels_fine: (n, s*patch, s*patch) effective-PSF kernels (integer-position)
        sersic_n: (n,) Sersic indices — fixed values, or the init when fit_n is True
        init_flux, init_re: (n,) initial values
        fit_n: free the Sersic index (clamped to [0.4, 5.5])

    Returns:
        dict of fitted (n,) tensors: flux, dx, dy, re, eta1, eta2, n, chi2
    """
    n_gal = len(cutouts)
    log_flux = torch.log(init_flux.clamp(min=1.0)).clone().requires_grad_(True)
    dx = torch.zeros(n_gal, requires_grad=True)
    dy = torch.zeros(n_gal, requires_grad=True)
    log_re = torch.log(init_re.clamp(min=0.3)).clone().requires_grad_(True)
    eta1 = torch.zeros(n_gal, requires_grad=True)
    eta2 = torch.zeros(n_gal, requires_grad=True)
    log_n = torch.log(sersic_n).clone().requires_grad_(fit_n)

    params = [log_flux, dx, dy, log_re, eta1, eta2] + ([log_n] if fit_n else [])
    weights = valid.float() / variance
    optimizer = torch.optim.Adam(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=lr / 100)
    for _ in range(steps):
        optimizer.zero_grad()
        profile = fine_sersic_samples(
            dx,
            dy,
            log_n.exp().clamp(0.4, 5.5),
            log_re.exp().clamp(0.3, 20.0),
            eta1,
            eta2,
            patch_size,
        )
        stamps = convolve_with_epsf(profile, kernels_fine, patch_size)
        model = log_flux.exp().reshape(-1, 1, 1) * stamps
        chi2 = (weights * (cutouts - model).square()).sum(dim=(-2, -1))
        chi2.sum().backward()
        optimizer.step()
        scheduler.step()

    with torch.no_grad():
        n_free = 7 if fit_n else 6
        n_valid = valid.sum(dim=(-2, -1)).clamp(min=n_free + 1)
        final_chi2 = chi2.detach() / (n_valid - n_free)
    return {
        "flux": log_flux.exp().detach(),
        "dx": dx.detach(),
        "dy": dy.detach(),
        "re": log_re.exp().detach(),
        "eta1": eta1.detach(),
        "eta2": eta2.detach(),
        "n": log_n.exp().clamp(0.4, 5.5).detach(),
        "chi2": final_chi2,
    }
