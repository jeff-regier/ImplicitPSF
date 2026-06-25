"""Full hierarchical E-step: per-star contaminants + GLOBAL (lambda, alpha), inferred jointly.

The contamination rate lambda and Pareto slope alpha are unknown on real data and are treated as
global random variables (manuscript sec:mcem). The contaminants we care about are largely
UNRESOLVABLE (sub-threshold), so we do NOT restrict to well-detected sources: the data constrains
(lambda, alpha) through the aggregate. hier_gibbs() is the proper coupled sampler — each sweep
updates every star's contaminants given the current (lambda, alpha), then draws (lambda, alpha)
from their conditionals (Gamma rate; griddy-Gibbs slope) from the pooled configs, with the flux
grid and detection log-odds recomputed as (lambda, alpha) move. hier_sbc() validates it: draw
(lambda, alpha) from the SAME hyperprior used for inference, generate a star aggregate, infer,
and require nominal credible-interval coverage. A wide-but-calibrated posterior is acceptable (we
marginalize it in the EM); a biased one is not.
"""

import argparse

import numpy as np

from implicitpsf.contam_gibbs import flux_grid
from implicitpsf.contam_mcmc import log_powerlaw
from implicitpsf.contam_model import cell_centers
from implicitpsf.mcem_sampler import (
    _cell,
    cov_columns,
    gaussian_cov,
    make_stamp,
    psf_covariance,
    rhat_ess,
    sample_prior,
)

# matched hyperpriors (generation == inference, for valid SBC)
LAM_A0, LAM_B0 = 1.0, 1.0  # lambda ~ Gamma(a0, b0)
ALPHA_GRID = np.linspace(1.1, 3.5, 30)
ALPHA_C0, ALPHA_D0 = 2.0, 1.0  # alpha prior ~ grid weighted by Gamma(c0, d0)


def _alpha_logprior(grid):
    return (ALPHA_C0 - 1) * np.log(grid) - ALPHA_D0 * grid


def draw_lambda_prior(rng):
    return float(rng.gamma(LAM_A0, 1.0 / LAM_B0))


def draw_alpha_prior(rng):
    p = np.exp(_alpha_logprior(ALPHA_GRID) - _alpha_logprior(ALPHA_GRID).max())
    return float(rng.choice(ALPHA_GRID, p=p / p.sum()))


def update_lambda(total_count, n_stars, rng):
    """Draw lambda | counts: Poisson(lambda)/star, Gamma conjugate."""
    return float(rng.gamma(LAM_A0 + total_count, 1.0 / (LAM_B0 + n_stars)))


def update_alpha(fluxes, lo, hi, rng):
    """Draw alpha | fluxes (truncated power-law) by griddy-Gibbs with the matched hyperprior."""
    if len(fluxes) == 0:
        return draw_alpha_prior(rng)
    logp = np.array([log_powerlaw(fluxes, lo, hi, a).sum() for a in ALPHA_GRID])
    logp = logp + _alpha_logprior(ALPHA_GRID)
    p = np.exp(logp - logp.max())
    return float(rng.choice(ALPHA_GRID, p=p / p.sum()))


def _collapsed_log_odds(total, di_c, n_stars, n_cells):
    """Per-cell detection log-odds with lambda MARGINALIZED (Gamma-Poisson predictive rate from the
    running total count, less this cell) -- decouples detections from a stale sampled lambda."""
    rate = (LAM_A0 + total - int(di_c)) / (LAM_B0 + n_stars)  # E[lambda | counts elsewhere]
    p = rate / n_cells
    return np.log(p) - np.log1p(-p)


def hier_gibbs(datas, weight, central_g, columns, lo, hi, n_sweeps, rng, burn=0.4):
    """Hierarchical Gibbs over many stars with lambda COLLAPSED out of the detection update: cells
    share a running total count via the Gamma-Poisson predictive rate, so the chain is not trapped
    by a stale sampled lambda (the un-collapsed version had R-hat 1.23 / ESS 13). The reported
    lambda chain is drawn from lambda | z each sweep. Returns the post-burn (lambda,alpha) chain."""
    n_cells, n_cov = columns.shape[0], columns.shape[1]
    b = (weight * columns**2).sum(axis=2)
    cg_norm = float((weight * central_g**2).sum())
    log_ncov = -np.log(n_cov)
    n_stars = len(datas)
    detect = np.zeros((n_stars, n_cells), dtype=bool)
    cov_idx = np.zeros((n_stars, n_cells), dtype=int)
    flux = np.zeros((n_stars, n_cells))
    contam = [np.zeros_like(d) for d in datas]
    alpha = draw_alpha_prior(rng)
    total = 0  # running total detections across all stars/cells (lambda integrated out)
    chain = []
    burn_steps = int(burn * n_sweeps)
    for sweep in range(n_sweeps):
        grid, log_w = flux_grid({"flux_lo": lo, "flux_hi": hi, "alpha": alpha})
        pooled_flux = []
        for i, data in enumerate(datas):
            di, ki, fi, cm = detect[i], cov_idx[i], flux[i], contam[i]
            cf = (weight * (data - cm) * central_g).sum() / cg_norm
            resid = data - cf * central_g - cm
            for c in range(n_cells):
                cur = fi[c] * columns[c, ki[c]] if di[c] else 0.0
                log_odds = _collapsed_log_odds(total, di[c], n_stars, n_cells)
                on, k, f, contrib = _cell(resid + cur, weight, columns[c], b[c], grid, log_w,
                                          log_odds, log_ncov, rng)
                cm += contrib - cur
                resid = data - cf * central_g - cm
                total += int(on) - int(di[c])
                di[c], ki[c], fi[c] = on, k, f
            pooled_flux.extend(fi[di].tolist())
        lam = update_lambda(total, n_stars, rng)
        alpha = update_alpha(np.array(pooled_flux), lo, hi, rng)
        if sweep >= burn_steps:
            chain.append((lam, alpha))
    return np.array(chain)


def _setup(size=32, grid_n=16, core=2.5):
    centers = cell_centers(size, grid_n, core)
    central_g = gaussian_cov((size - 1) / 2, (size - 1) / 2, 2.5 * np.eye(2), size).ravel()
    cols = cov_columns(centers, psf_covariance(central_g, size), size)
    return centers, central_g, cols


def hier_sbc(rng, n_draws=10, n_stars=25, n_sweeps=60, lo=100.0, hi=2000.0, noise=30.0, grid_n=16):
    """Hierarchical SBC: draw (lambda, alpha) from the hyperprior, generate, infer, check 90% CI
    coverage of both. Calibrated => ~0.90 (wide-but-calibrated is fine; biased is not)."""
    centers, central_g, cols = _setup(grid_n=grid_n)
    weight = np.full(central_g.shape, 1.0 / noise**2)
    lcov = acov = 0
    for _ in range(n_draws):
        lam_t, alpha_t = draw_lambda_prior(rng), draw_alpha_prior(rng)
        truth = {"lam": lam_t, "flux_lo": lo, "flux_hi": hi, "alpha": alpha_t}
        datas = []
        for _ in range(n_stars):
            dt, ci, fl = sample_prior(rng, truth, len(centers), cols.shape[1])
            data, _ = make_stamp(rng, 1e5, central_g, cols, dt, ci, fl, noise)
            datas.append(data)
        chain = hier_gibbs(datas, weight, central_g, cols, lo, hi, n_sweeps, rng)
        llo, lhi = np.percentile(chain[:, 0], [5, 95])
        alo, ahi = np.percentile(chain[:, 1], [5, 95])
        lcov += llo <= lam_t <= lhi
        acov += alo <= alpha_t <= ahi
        print(f"  truth lam={lam_t:.2f} alpha={alpha_t:.2f}; post lam[{llo:.2f},{lhi:.2f}] "
              f"alpha[{alo:.2f},{ahi:.2f}]")
    return lcov / n_draws, acov / n_draws


def hier_mixing(rng, lam_t=1.0, alpha_t=1.8, n_stars=25, n_sweeps=120, n_chains=4,
                lo=100.0, hi=2000.0, noise=30.0, grid_n=16):
    """Multi-chain Gelman-Rubin R-hat + ESS on the GLOBAL (lambda, alpha) chain for ONE generated
    star aggregate. Over-dispersed inits (independent prior draws per chain) -> R-hat ~1 means the
    coupled (lambda, detections) Gibbs mixes; R-hat >> 1 flags the self-consistent-mode trap."""
    centers, central_g, cols = _setup(grid_n=grid_n)
    weight = np.full(central_g.shape, 1.0 / noise**2)
    truth = {"lam": lam_t, "flux_lo": lo, "flux_hi": hi, "alpha": alpha_t}
    datas = []
    for _ in range(n_stars):
        dt, ci, fl = sample_prior(rng, truth, len(centers), cols.shape[1])
        data, _ = make_stamp(rng, 1e5, central_g, cols, dt, ci, fl, noise)
        datas.append(data)
    lam_chains, alpha_chains = [], []
    for _ in range(n_chains):
        chain = hier_gibbs(datas, weight, central_g, cols, lo, hi, n_sweeps, rng)
        lam_chains.append(chain[:, 0])
        alpha_chains.append(chain[:, 1])
    lam_r, lam_ess = rhat_ess(np.array(lam_chains))
    alpha_r, alpha_ess = rhat_ess(np.array(alpha_chains))
    for j, ch in enumerate(lam_chains):
        print(f"  chain {j}: lam mean {ch.mean():.3f} (init {ch[0]:.3f}); "
              f"alpha mean {alpha_chains[j].mean():.3f}")
    pooled_lam = np.concatenate(lam_chains)
    pooled_alpha = np.concatenate(alpha_chains)
    llo, lhi = np.percentile(pooled_lam, [5, 95])
    alo, ahi = np.percentile(pooled_alpha, [5, 95])
    print(f"R-hat lam {lam_r:.3f} ESS {lam_ess:.0f}; R-hat alpha {alpha_r:.3f} "
          f"ESS {alpha_ess:.0f}  (want R-hat<1.1, ESS>=100)")
    print(f"truth lam={lam_t} in [{llo:.3f},{lhi:.3f}]? {llo <= lam_t <= lhi}; "
          f"truth alpha={alpha_t} in [{alo:.3f},{ahi:.3f}]? {alo <= alpha_t <= ahi}")
    return lam_r, lam_ess, alpha_r, alpha_ess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sbc", "mixing"], default="sbc")
    parser.add_argument("--n-draws", type=int, default=10)
    parser.add_argument("--n-stars", type=int, default=25)
    parser.add_argument("--n-sweeps", type=int, default=60)
    parser.add_argument("--grid-n", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    if args.mode == "mixing":
        hier_mixing(rng, n_stars=args.n_stars, n_sweeps=args.n_sweeps, grid_n=args.grid_n)
        return
    lc, ac = hier_sbc(rng, args.n_draws, args.n_stars, args.n_sweeps, grid_n=args.grid_n)
    print(f"hierarchical SBC ({args.n_draws} draws): lambda coverage {lc:.2f}, "
          f"alpha coverage {ac:.2f}  (want ~0.90)")


if __name__ == "__main__":
    main()
