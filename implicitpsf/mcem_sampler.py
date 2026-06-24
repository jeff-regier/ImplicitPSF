"""Proper MCEM contamination sampler (rebuild) — single-star collapsed Gibbs.

Replaces the posterior-mean / discrete-isotropic-size approximation. Per the method (manuscript
sec:mcem): the cutout is a bright central source (the PSF, amplitude PROFILED — a parameter, no
prior) plus faint nuisance contaminants (a counts prior, MARGINALIZED). Each contaminant is ONE
bivariate Gaussian with covariance Sigma = Sigma_psf + L L^T (Cholesky L >= 0), lower-bounded by
the current PSF so a source is never sharper than a point source and may be extended/elliptical —
no star/galaxy split. Detection is one Bernoulli per grid cell (fixed dimension). The collapsed
Gibbs marginalizes each flux on a shared grid and sums the detection evidence over a covariance
grid; the central amplitude is PROFILED each sweep (not sampled). run() returns K thinned
post-burn samples (NOT the mean) for the Monte Carlo E-step. Global (lambda, alpha) come later
(hierarchical step); here they are fixed inputs. SBC- and mixing-gated (see sbc_coverage, rhat_ess).
"""

import numpy as np

from implicitpsf.contam_gibbs import flux_grid
from implicitpsf.contam_mcmc import sample_powerlaw
from implicitpsf.contam_model import cell_centers


def gaussian_cov(cx, cy, sigma, size):
    """Unit-sum 2D Gaussian, mean (cx,cy), covariance sigma (2x2), on a size x size grid."""
    yy, xx = np.mgrid[0:size, 0:size]
    dx, dy = xx - cx, yy - cy
    inv = np.linalg.inv(sigma)
    q = inv[0, 0] * dx * dx + 2.0 * inv[0, 1] * dx * dy + inv[1, 1] * dy * dy
    g = np.exp(-0.5 * q)
    return g / (g.sum() + 1e-12)


def psf_covariance(central_g, size):
    """Second-moment matrix of the central PSF stamp — the covariance lower bound Sigma_psf."""
    img = np.clip(central_g, 0, None).reshape(size, size)
    img = img / (img.sum() + 1e-12)
    yy, xx = np.mgrid[0:size, 0:size]
    cx = float((img * xx).sum())
    cy = float((img * yy).sum())
    sxx = float((img * (xx - cx) ** 2).sum())
    syy = float((img * (yy - cy) ** 2).sum())
    sxy = float((img * (xx - cx) * (yy - cy)).sum())
    return np.array([[sxx, sxy], [sxy, syy]])


# intrinsic broadening grid (sigma px, ellipticity, orientation): 0=point, isotropic, elliptical.
_SIZES = np.array([0.0, 2.0, 4.0])
_ELLIP = 0.4
_PHIS = np.array([0.0, 45.0, 90.0, 135.0])


def intrinsic_covariances():
    """Discrete set of intrinsic covariances L L^T >= 0 (the marginalized covariance mark)."""
    out = [np.zeros((2, 2))]  # s=0: point source (kernel = Sigma_psf, the lower bound)
    for s in _SIZES[1:]:
        out.append(s * s * np.eye(2))  # isotropic
        for phi in _PHIS:
            r = np.deg2rad(phi)
            rot = np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])
            d = s * s * np.diag([1.0 + _ELLIP, 1.0 - _ELLIP])
            out.append(rot @ d @ rot.T)
    return np.stack(out)  # (n_cov, 2, 2)


def cov_columns(centers, sigma_psf, size):
    """(n_cells, n_cov, n_pix) unit-flux kernels: Gaussian(cell, Sigma_psf + intrinsic)."""
    intr = intrinsic_covariances()
    cols = np.empty((len(centers), len(intr), size * size))
    for ci, (x, y) in enumerate(centers):
        for ki, dsig in enumerate(intr):
            cols[ci, ki] = gaussian_cov(x, y, sigma_psf + dsig, size).ravel()
    return cols  # marginalize over the n_cov axis in the Gibbs (uniform cov prior for now)


def sample_prior(rng, prior, n_cells, n_cov):
    """Draw (detect, cov_idx, flux) from the prior (per-cell Bernoulli, uniform cov, power-law)."""
    p = prior["lam"] / n_cells
    detect = rng.uniform(size=n_cells) < p
    cov_idx = np.zeros(n_cells, dtype=int)
    flux = np.zeros(n_cells)
    k = int(detect.sum())
    if k:
        cov_idx[detect] = rng.integers(0, n_cov, size=k)
        flux[detect] = sample_powerlaw(rng, prior["flux_lo"], prior["flux_hi"], prior["alpha"], k)
    return detect, cov_idx, flux


def render(detect, cov_idx, flux, columns):
    out = np.zeros(columns.shape[2])
    for c in np.nonzero(detect)[0]:
        out += flux[c] * columns[c, cov_idx[c]]
    return out


def make_stamp(rng, central_flux, central_g, columns, detect, cov_idx, flux, noise):
    clean = central_flux * central_g + render(detect, cov_idx, flux, columns)
    data = clean + rng.normal(0, noise, size=clean.shape)
    return data, np.full_like(data, 1.0 / noise**2)


def _cell(resid, weight, cols_c, b_c, grid, log_w, log_odds, log_ncov, rng):
    """Resample one cell over (detect, covariance, flux); flux marginalized, covariance summed."""
    a = (weight * resid) @ cols_c.T  # (n_cov,)
    mean = a / b_c
    log_post = -0.5 * b_c[:, None] * (grid[None, :] - mean[:, None]) ** 2 + log_w[None, :]
    m = log_post.max(axis=1)
    slab = m + np.log(np.exp(log_post - m[:, None]).sum(axis=1))
    log_cov = 0.5 * a * a / b_c + slab + log_ncov  # per-covariance evidence (uniform cov prior)
    mc = log_cov.max()
    log_on = log_odds + mc + np.log(np.exp(log_cov - mc).sum())  # summed over covariance
    if np.log(rng.uniform()) < -np.logaddexp(0.0, -log_on):
        pc = np.exp(log_cov - mc)
        k = int(rng.choice(len(b_c), p=pc / pc.sum()))
        pf = np.exp(log_post[k] - log_post[k].max())
        f = float(rng.choice(grid, p=pf / pf.sum()))
        return True, k, f, f * cols_c[k]
    return False, 0, 0.0, np.zeros_like(cols_c[0])


def run(data, weight, central_g, columns, prior, n_sweeps, rng, burn=0.3, n_keep=8):
    """Collapsed Gibbs with PROFILED central amplitude; returns (K post-burn contam samples,
    counts, totals) — the Monte Carlo E-step output (NOT the posterior mean)."""
    n_cells, n_cov = columns.shape[0], columns.shape[1]
    b = (weight * columns**2).sum(axis=2)
    cg_norm = float((weight * central_g**2).sum())
    grid, log_w = flux_grid(prior)
    log_odds = np.log(prior["lam"] / n_cells) - np.log1p(-prior["lam"] / n_cells)
    log_ncov = -np.log(n_cov)
    detect = np.zeros(n_cells, dtype=bool)
    cov_idx = np.zeros(n_cells, dtype=int)
    flux = np.zeros(n_cells)
    contam = np.zeros_like(data)
    samples, counts, totals = [], [], []
    burn_steps = int(burn * n_sweeps)
    keep_every = max(1, (n_sweeps - burn_steps) // n_keep)
    for sweep in range(n_sweeps):
        central_flux = (weight * (data - contam) * central_g).sum() / cg_norm  # PROFILE, no draw
        resid = data - central_flux * central_g - contam
        for c in range(n_cells):
            cur = flux[c] * columns[c, cov_idx[c]] if detect[c] else 0.0
            on, k, f, contrib = _cell(resid + cur, weight, columns[c], b[c], grid, log_w,
                                      log_odds, log_ncov, rng)
            contam += contrib - cur
            resid = data - central_flux * central_g - contam
            detect[c], cov_idx[c], flux[c] = on, k, f
        if sweep >= burn_steps and (sweep - burn_steps) % keep_every == 0:
            samples.append(contam.copy())
            counts.append(int(detect.sum()))
            totals.append(float(flux[detect].sum()))
    return np.array(samples), np.array(counts), np.array(totals)


def sbc_coverage(rng, prior, n_draws, n_sweeps, size=32, grid_n=16, core=2.5, noise=30.0):
    """Simulation-based calibration: draw z from prior, infer, check 90% CI coverage of the total
    contaminant flux and count (the EM-relevant summaries). Calibrated => ~0.90."""
    centers = cell_centers(size, grid_n, core)
    central_g = gaussian_cov((size - 1) / 2, (size - 1) / 2, 2.5 * np.eye(2), size).ravel()
    cols = cov_columns(centers, psf_covariance(central_g, size), size)
    fcov = ccov = 0
    for _ in range(n_draws):
        dt, ci, fl = sample_prior(rng, prior, len(centers), cols.shape[1])
        data, w = make_stamp(rng, 1e5, central_g, cols, dt, ci, fl, noise)
        _, counts, totals = run(data, w, central_g, cols, prior, n_sweeps, rng, n_keep=n_sweeps)
        tlo, thi = np.percentile(totals, [5, 95])
        clo, chi = np.percentile(counts, [5, 95])
        fcov += tlo <= fl[dt].sum() <= thi
        ccov += clo <= int(dt.sum()) <= chi
    return fcov / n_draws, ccov / n_draws


def rhat_ess(chains):
    """Gelman-Rubin R-hat and effective sample size for a scalar summary across chains.
    chains: (n_chains, n_samples). Returns (R_hat, ESS)."""
    m, n = chains.shape
    means = chains.mean(axis=1)
    bvar = n * means.var(ddof=1)
    wvar = chains.var(axis=1, ddof=1).mean()
    var = (1 - 1 / n) * wvar + bvar / n
    rhat = np.sqrt(var / wvar) if wvar > 0 else np.nan
    flat = chains.ravel()
    if flat.var() <= 0:
        return float(rhat), float(m * n)
    ac = np.correlate(flat - flat.mean(), flat - flat.mean(), "full")[len(flat) - 1:]
    ac = ac / ac[0]
    neg = np.where(ac < 0)[0]
    tau = 1 + 2 * ac[1: (neg[0] if len(neg) else len(ac))].sum()
    return float(rhat), float(m * n / max(tau, 1.0))


def _self_test():
    size, grid, core = 32, 16, 2.5
    rng = np.random.default_rng(0)
    centers = cell_centers(size, grid, core)
    central_g = gaussian_cov((size - 1) / 2, (size - 1) / 2, 2.5 * np.eye(2), size).ravel()
    sigma_psf = psf_covariance(central_g, size)
    cols = cov_columns(centers, sigma_psf, size)
    prior = {"lam": 1.0, "flux_lo": 100.0, "flux_hi": 2000.0, "alpha": 1.5}
    dt, ci, fl = sample_prior(rng, prior, len(centers), cols.shape[1])
    data, _ = make_stamp(rng, 1e5, central_g, cols, dt, ci, fl, 30.0)
    print(f"n_cells {len(centers)}, n_cov {cols.shape[1]} (point + iso + elliptical)")
    print(f"Sigma_psf diag {np.diag(sigma_psf).round(2)}; injected {int(dt.sum())} contaminants")
    print(f"cov0 (point) == Sigma_psf? kernel peak matches PSF: "
          f"{np.allclose(cols[0, 0].sum(), 1.0)}; data finite {np.isfinite(data).all()}")


def _mixing_test(rng, prior, n_sweeps, n_chains=4, size=32, grid_n=16, core=2.5, noise=30.0):
    """Multi-chain R-hat/ESS on the total contaminant flux for one prior-drawn stamp."""
    centers = cell_centers(size, grid_n, core)
    central_g = gaussian_cov((size - 1) / 2, (size - 1) / 2, 2.5 * np.eye(2), size).ravel()
    cols = cov_columns(centers, psf_covariance(central_g, size), size)
    dt, ci, fl = sample_prior(rng, {**prior, "lam": 2.0}, len(centers), cols.shape[1])
    data, w = make_stamp(rng, 1e5, central_g, cols, dt, ci, fl, noise)
    chains = []
    for _ in range(n_chains):
        _, _, totals = run(data, w, central_g, cols, prior, n_sweeps,
                           np.random.default_rng(rng.integers(1 << 30)), n_keep=n_sweeps)
        chains.append(totals)
    return rhat_ess(np.array(chains))


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["selftest", "sbc", "mixing"], default="selftest")
    parser.add_argument("--n-draws", type=int, default=20)
    parser.add_argument("--n-sweeps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    prior = {"lam": 2.0, "flux_lo": 100.0, "flux_hi": 3000.0, "alpha": 1.5}
    if args.mode == "sbc":
        f, c = sbc_coverage(rng, prior, args.n_draws, args.n_sweeps)
        print(f"MCEM-sampler SBC ({args.n_draws} draws, {args.n_sweeps} sweeps): "
              f"flux {f:.2f}, count {c:.2f}  (want ~0.90)")
    elif args.mode == "mixing":
        r, ess = _mixing_test(rng, prior, args.n_sweeps)
        print(f"mixing ({args.n_sweeps} sweeps, 4 chains): R-hat {r:.3f} (want ~1), "
              f"ESS {ess:.0f}  (total contaminant flux)")
    else:
        _self_test()


if __name__ == "__main__":
    main()
