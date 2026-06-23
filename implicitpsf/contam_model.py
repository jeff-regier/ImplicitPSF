"""Shared per-pixel-Bernoulli contamination model — the substrate for all inference methods.

Detection is ONE BERNOULLI PER GRID CELL (at most one contaminant per cell), so the model is
fixed-dimensional: no transdimensional reversible jump (which was painful to tune). A stamp is the
bright central source (the current PSF estimate times central_flux) plus, in each "on" cell, a
contaminant (a Gaussian at the cell centre, matched or variable FWHM) scaled by a power-law flux,
plus noise. MCMC (Gibbs), SMC, VI, and the flow-matching detector all operate on THIS representation
and differ only in how they infer the posterior over the detection grid + marks.

State (per stamp): detect (n_cells,) bool [z, <=1 source/cell]; flux (n_cells,) [0 where off];
central_flux float. Core cells (under the bright star) are excluded -- a contaminant there is
degenerate with the star. Reuses the likelihood/prior helpers from contam_mcmc / contam_sampler.
"""

import numpy as np

from implicitpsf.contam_mcmc import log_powerlaw, sample_powerlaw
from implicitpsf.contam_sampler import gaussian_source


def cell_centers(size, grid, core_radius):
    """Sub-pixel centres of a grid x grid tiling of a size x size stamp, excluding cells within
    core_radius px of the centre (a contaminant there is degenerate with the star). (n_cells, 2)."""
    step = size / grid
    coords = (np.arange(grid) + 0.5) * step - 0.5
    xx, yy = np.meshgrid(coords, coords)
    centers = np.column_stack([xx.ravel(), yy.ravel()])
    c = (size - 1) / 2.0
    keep = np.hypot(centers[:, 0] - c, centers[:, 1] - c) >= core_radius
    return centers[keep]


def design_columns(centers, fwhm, size):
    """(n_cells, n_pix) unit-flux Gaussian per cell centre -- the per-cell contaminant kernel."""
    return np.stack([gaussian_source(x, y, fwhm, size).ravel() for x, y in centers])


def render_contam(detect, flux, columns):
    """Contaminant image (n_pix,) = sum over on-cells of flux x kernel."""
    if not detect.any():
        return np.zeros(columns.shape[1])
    return flux[detect] @ columns[detect]


def loglike(data, weight, central_flux, central_g, detect, flux, columns):
    """Gaussian inverse-variance log-likelihood of the full model (central + contaminants)."""
    model = central_flux * central_g + render_contam(detect, flux, columns)
    return -0.5 * float((weight * (data - model) ** 2).sum())


def logprior(detect, flux, prior, n_cells):
    """Per-cell Bernoulli(p=lambda/n_cells) detection (Poisson mean-field) + power-law flux."""
    p = prior["lam"] / n_cells
    k = int(detect.sum())
    lp = k * np.log(p) + (n_cells - k) * np.log1p(-p)
    if k:
        lo, hi, a = prior["flux_lo"], prior["flux_hi"], prior["alpha"]
        lp += float(log_powerlaw(flux[detect], lo, hi, a).sum())
    return lp


def sample_prior(rng, prior, n_cells):
    """Draw a contamination state (detect, flux) from the prior -- for sim-based calibration."""
    p = prior["lam"] / n_cells
    detect = rng.uniform(size=n_cells) < p
    flux = np.zeros(n_cells)
    k = int(detect.sum())
    if k:
        flux[detect] = sample_powerlaw(rng, prior["flux_lo"], prior["flux_hi"], prior["alpha"], k)
    return detect, flux


def make_stamp(rng, central_flux, central_g, columns, detect, flux, noise_sigma):
    """Noisy stamp from a contamination state -- the SBC/test generator. Returns (data, weight)."""
    clean = central_flux * central_g + render_contam(detect, flux, columns)
    data = clean + rng.normal(0, noise_sigma, size=clean.shape)
    weight = np.full_like(data, 1.0 / noise_sigma**2)
    return data, weight


def total_contam_flux(detect, flux):
    """The EM-relevant scalar: total contaminating light (what the M-step subtracts)."""
    return float(flux[detect].sum())


def _self_test():
    """Smoke test: render -> likelihood -> prior -> prior-sample on a 32x32 stamp."""
    rng = np.random.default_rng(0)
    size, grid, core_radius, fwhm = 32, 32, 2.5, 4.0
    centers = cell_centers(size, grid, core_radius)
    columns = design_columns(centers, fwhm, size)
    central_g = gaussian_source((size - 1) / 2.0, (size - 1) / 2.0, fwhm, size).ravel()
    prior = {"lam": 1.0, "flux_lo": 100.0, "flux_hi": 2000.0, "alpha": 1.5}
    n_cells = len(centers)
    detect, flux = sample_prior(rng, prior, n_cells)
    data, weight = make_stamp(rng, 1.0e5, central_g, columns, detect, flux, 30.0)
    ll = loglike(data, weight, 1.0e5, central_g, detect, flux, columns)
    lp = logprior(detect, flux, prior, n_cells)
    print(f"n_cells {n_cells} of {grid * grid} (core r{core_radius} excluded); "
          f"injected {int(detect.sum())}, total flux {total_contam_flux(detect, flux):.0f}")
    print(f"loglike {ll:.1f} ({np.isfinite(ll)}), logprior {lp:.1f} ({np.isfinite(lp)}), "
          f"columns {columns.shape}")


if __name__ == "__main__":
    _self_test()
