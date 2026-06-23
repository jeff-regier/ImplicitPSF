"""Method A — collapsed Gibbs sampler on the per-pixel-Bernoulli substrate (contam_model).

Each sweep visits every cell and resamples (detect_c, flux_c) JOINTLY, with the flux handled on a
shared log-spaced grid against the power-law prior: the cell's "on" evidence is the prior-weighted
Gaussian-in-flux integral over the grid, vs "off" evidence 1; detect_c ~ Bernoulli of that, and if
on, flux_c is drawn from that grid posterior. No transdimensional jump (fixed grid) -> mixes far
better than the RJMCMC. The central amplitude is a Gibbs draw (flat prior, Gaussian-linear). Returns
posterior-mean contamination image (the EM E-step) plus per-sweep count/total-flux for diagnostics.
Validated by simulation-based calibration (sbc_coverage).
"""

import argparse

import numpy as np

from implicitpsf.contam_mcmc import log_powerlaw
from implicitpsf.contam_model import (
    cell_centers,
    design_columns,
    make_stamp,
    sample_prior,
    total_contam_flux,
)
from implicitpsf.contam_sampler import gaussian_source

N_FLUX_GRID = 64


def flux_grid(prior):
    """Shared log-spaced flux grid over [lo, hi] and its normalized power-law log-prior."""
    grid = np.geomspace(prior["flux_lo"], prior["flux_hi"], N_FLUX_GRID)
    log_w = log_powerlaw(grid, prior["flux_lo"], prior["flux_hi"], prior["alpha"])
    log_w = log_w + np.log(np.gradient(grid))  # trapezoid-ish weight for the integral over the grid
    return grid, log_w


def _gibbs_cell(resid, weight, col, b_c, grid, log_w, log_odds_prior, rng):
    """Resample one cell given the residual EXCLUDING it. Returns (on, flux, contribution)."""
    a = float((weight * resid * col).sum())  # data projection onto this cell's kernel
    # likelihood-ratio(on/off) in flux = exp(0.5 a^2/b) * exp(-0.5 b (f - a/b)^2); x prior, integrate
    log_post = -0.5 * b_c * (grid - a / b_c) ** 2 + log_w
    m = log_post.max()
    log_slab = 0.5 * a * a / b_c + m + np.log(np.exp(log_post - m).sum())  # evidence vs "off"=1
    log_on = log_odds_prior + log_slab
    # Bernoulli(sigmoid(log_on)): z=1 iff log(u) < log p(on) = -softplus(-log_on) (stable, no overflow)
    if np.log(rng.uniform()) < -np.logaddexp(0.0, -log_on):
        p = np.exp(log_post - m)
        f = float(rng.choice(grid, p=p / p.sum()))
        return True, f, f * col
    return False, 0.0, np.zeros_like(col)


def run(data, weight, central_g, columns, prior, n_sweeps, rng, burn=0.3):
    """Collapsed Gibbs; returns (posterior-mean contam image, counts, totals) post-burn."""
    n_cells = len(columns)
    b = (weight * columns**2).sum(axis=1)  # per-cell design norm (precomputed)
    cg_norm = float((weight * central_g**2).sum())
    grid, log_w = flux_grid(prior)
    log_odds_prior = np.log(prior["lam"] / n_cells) - np.log1p(-prior["lam"] / n_cells)

    detect = np.zeros(n_cells, dtype=bool)
    flux = np.zeros(n_cells)
    central_flux = float((data * weight * central_g).sum() / cg_norm)
    contam = np.zeros_like(data)

    accum, counts, totals, n_acc = np.zeros_like(data), [], [], 0
    burn_steps = int(burn * n_sweeps)
    for sweep in range(n_sweeps):
        central_flux = (weight * (data - contam) * central_g).sum() / cg_norm
        central_flux += rng.normal(0, 1.0 / np.sqrt(cg_norm))  # Gibbs draw (flat prior)
        resid_full = data - central_flux * central_g - contam
        for c in range(n_cells):
            resid_wo = resid_full + (flux[c] * columns[c] if detect[c] else 0.0)
            on, f, contrib = _gibbs_cell(resid_wo, weight, columns[c], b[c], grid, log_w,
                                         log_odds_prior, rng)
            contam += contrib - (flux[c] * columns[c] if detect[c] else 0.0)
            resid_full = data - central_flux * central_g - contam
            detect[c], flux[c] = on, f
        if sweep >= burn_steps:
            accum += contam
            counts.append(int(detect.sum()))
            totals.append(float(flux[detect].sum()))
            n_acc += 1
    return accum / max(n_acc, 1), np.array(counts), np.array(totals)


def sbc_coverage(rng, prior, n_draws, n_sweeps, size=32, grid_n=32, core_radius=2.5, fwhm=4.0,
                 noise=30.0, central_flux=1.0e5):
    """Simulation-based calibration: draw from the prior, infer, check 90% CI coverage of the count
    and total contaminant flux (the EM-relevant quantity). Calibrated => ~0.90."""
    centers = cell_centers(size, grid_n, core_radius)
    columns = design_columns(centers, fwhm, size)
    central_g = gaussian_source((size - 1) / 2.0, (size - 1) / 2.0, fwhm, size).ravel()
    n_cells = len(centers)
    flux_in, flux_cov, count_cov = 0, 0, 0
    for _ in range(n_draws):
        detect_t, flux_t = sample_prior(rng, prior, n_cells)
        data, weight = make_stamp(rng, central_flux, central_g, columns, detect_t, flux_t, noise)
        _, counts, totals = run(data, weight, central_g, columns, prior, n_sweeps, rng)
        true_total = total_contam_flux(detect_t, flux_t)
        tflo, tfhi = np.percentile(totals, [5, 95])
        clo, chi = np.percentile(counts, [5, 95])
        flux_cov += tflo <= true_total <= tfhi
        count_cov += clo <= int(detect_t.sum()) <= chi
        flux_in += 1
    return flux_cov / flux_in, count_cov / flux_in


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["recover", "sbc"], default="recover")
    parser.add_argument("--n-sweeps", type=int, default=60)
    parser.add_argument("--n-draws", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    prior = {"lam": 2.0, "flux_lo": 100.0, "flux_hi": 3000.0, "alpha": 1.5}

    if args.mode == "sbc":
        fcov, ccov = sbc_coverage(rng, prior, args.n_draws, args.n_sweeps)
        print(f"Gibbs SBC ({args.n_draws} draws, {args.n_sweeps} sweeps): "
              f"flux 90% coverage {fcov:.2f}, count {ccov:.2f}  (want ~0.90)")
        return
    size, grid_n, fwhm = 32, 32, 4.0
    centers = cell_centers(size, grid_n, 2.5)
    columns = design_columns(centers, fwhm, size)
    central_g = gaussian_source((size - 1) / 2.0, (size - 1) / 2.0, fwhm, size).ravel()
    detect_t, flux_t = sample_prior(rng, {**prior, "lam": 3.0}, len(centers))
    data, weight = make_stamp(rng, 1.0e5, central_g, columns, detect_t, flux_t, 30.0)
    _, counts, totals = run(data, weight, central_g, columns, prior, args.n_sweeps, rng)
    tot = total_contam_flux(detect_t, flux_t)
    print(f"injected {int(detect_t.sum())} contaminants, total flux {tot:.0f}")
    print(f"posterior count {counts.mean():.2f} +/- {counts.std():.2f}; "
          f"total flux {totals.mean():.0f} (90% CI [{np.percentile(totals, 5):.0f}, "
          f"{np.percentile(totals, 95):.0f}])")


if __name__ == "__main__":
    main()
