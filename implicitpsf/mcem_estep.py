"""Hierarchical E-step: per-star contamination + GLOBAL (lambda, alpha) shared across stars.

The contamination prior's rate lambda and Pareto slope alpha are unknown on real data, so we
treat them as global random variables (manuscript sec:mcem) and infer them jointly from the
many-star aggregate: per star, Gibbs-sample the contaminants given the current (lambda, alpha,
PSF); pool the detection counts and fluxes; draw (lambda, alpha) from their conditionals (Gamma
rate, griddy-Gibbs slope). recover() is the IDENTIFIABILITY GATE on synthetic data: with
(lambda, alpha) unknown and started wrong, the aggregate must recover the generator's values --
otherwise the PSF is free to absorb the contamination and the whole correction is unidentified.
"""

import argparse

import numpy as np

from implicitpsf.contam_mcmc import log_powerlaw
from implicitpsf.contam_model import cell_centers
from implicitpsf.mcem_sampler import (
    cov_columns,
    gaussian_cov,
    make_stamp,
    psf_covariance,
    run,
    sample_prior,
)


def update_lambda(total_count, n_stars, rng, a0=1.0, b0=1.0):
    """Draw lambda | counts: Poisson(lambda)/star, Gamma(a0,b0) hyperprior -> Gamma conjugate."""
    return float(rng.gamma(a0 + total_count, 1.0 / (b0 + n_stars)))


def update_alpha(fluxes, lo, hi, rng, grid=None, c0=2.0, d0=1.0):
    """Draw alpha | fluxes (truncated power-law) by griddy-Gibbs; Gamma(c0,d0) hyperprior."""
    if grid is None:
        grid = np.linspace(1.1, 3.5, 30)
    if len(fluxes) == 0:
        return float(rng.choice(grid))
    logp = np.array([log_powerlaw(fluxes, lo, hi, a).sum() for a in grid])
    logp = logp + (c0 - 1) * np.log(grid) - d0 * grid
    p = np.exp(logp - logp.max())
    return float(rng.choice(grid, p=p / p.sum()))


def recover(rng, n_stars=40, n_sweeps=15, outer=6, lam_true=1.5, alpha_true=1.8):
    """Identifiability gate: recover unknown (lambda, alpha) from a synthetic star aggregate."""
    size, grid_n, core, lo, hi = 32, 16, 2.5, 100.0, 2000.0
    centers = cell_centers(size, grid_n, core)
    central_g = gaussian_cov((size - 1) / 2, (size - 1) / 2, 2.5 * np.eye(2), size).ravel()
    cols = cov_columns(centers, psf_covariance(central_g, size), size)
    truth = {"lam": lam_true, "flux_lo": lo, "flux_hi": hi, "alpha": alpha_true}
    stars = []
    for _ in range(n_stars):
        dt, ci, fl = sample_prior(rng, truth, len(centers), cols.shape[1])
        stars.append(make_stamp(rng, 1e5, central_g, cols, dt, ci, fl, 30.0))
    lam, alpha = 1.0, 2.8  # deliberately wrong start
    print(f"true lam={lam_true} alpha={alpha_true}; init lam={lam} alpha={alpha}")
    for t in range(outer):
        prior = {"lam": lam, "flux_lo": lo, "flux_hi": hi, "alpha": alpha}
        tot_count, all_flux = 0.0, []
        for data, w in stars:
            _, counts, _, pooled = run(data, w, central_g, cols, prior, n_sweeps, rng, n_keep=3)
            tot_count += counts.mean()
            all_flux.extend(pooled.tolist())
        lam = update_lambda(tot_count, n_stars, rng)
        alpha = update_alpha(np.array(all_flux), lo, hi, rng)
        print(f"  iter {t}: lam={lam:.2f} alpha={alpha:.2f}  (pooled fluxes {len(all_flux)})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["recover"], default="recover")
    parser.add_argument("--n-stars", type=int, default=40)
    parser.add_argument("--n-sweeps", type=int, default=15)
    parser.add_argument("--outer", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    recover(np.random.default_rng(args.seed), args.n_stars, args.n_sweeps, args.outer)


if __name__ == "__main__":
    main()
