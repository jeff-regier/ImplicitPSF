"""Step 6: extended/variable-size contaminants on the per-pixel-Bernoulli substrate.

The point-source Gibbs cleans only ~13% of real-sim contamination because 58% of it is EXTENDED
galaxies (re 1.5-6px) a matched-FWHM point source can't absorb. Here each contaminant also chooses a
SIZE from a discrete set (point ... galaxy-scale), sampled jointly with detect+flux in the collapsed
Gibbs cell update (the flux is still grid-marginalized per size). The decisive test: does modeling
extended contaminants lift the bridge cleaning above ~13%? Reuses the substrate (contam_model) + the
flux grid (contam_gibbs); SBC- and bridge-validated like the point-source version.
"""

import argparse

import numpy as np

from implicitpsf.contam_bridge import bridge
from implicitpsf.contam_gibbs import flux_grid
from implicitpsf.contam_mcmc import sample_powerlaw
from implicitpsf.contam_model import cell_centers
from implicitpsf.contam_sampler import gaussian_source

SIZES = np.array([4.0, 7.0, 11.0, 16.0])  # contaminant FWHMs: point + extended (galaxy-scale)
SIZE_PROBS = np.array([0.42, 0.58 / 3, 0.58 / 3, 0.58 / 3])  # 42% star / 58% galaxy (extended)


def multisize_columns(centers, size):
    """(n_cells, n_sizes, n_pix) unit-flux Gaussian per (cell, size)."""
    return np.stack(
        [np.stack([gaussian_source(x, y, fw, size).ravel() for x, y in centers]) for fw in SIZES],
        axis=1,
    )


def sample_prior_sized(rng, prior, n_cells):
    """Draw (detect, size_idx, flux) from the prior + the size distribution -- for SBC."""
    p = prior["lam"] / n_cells
    detect = rng.uniform(size=n_cells) < p
    size_idx = np.zeros(n_cells, dtype=int)
    flux = np.zeros(n_cells)
    k = int(detect.sum())
    if k:
        size_idx[detect] = rng.choice(len(SIZES), size=k, p=SIZE_PROBS)
        flux[detect] = sample_powerlaw(rng, prior["flux_lo"], prior["flux_hi"], prior["alpha"], k)
    return detect, size_idx, flux


def render_sized(detect, size_idx, flux, columns):
    """Contaminant image from a sized state: sum over on-cells of flux × kernel(cell, size)."""
    out = np.zeros(columns.shape[2])
    for c in np.nonzero(detect)[0]:
        out += flux[c] * columns[c, size_idx[c]]
    return out


def make_stamp_sized(rng, central_flux, central_g, columns, detect, size_idx, flux, noise):
    clean = central_flux * central_g + render_sized(detect, size_idx, flux, columns)
    data = clean + rng.normal(0, noise, size=clean.shape)
    return data, np.full_like(data, 1.0 / noise**2)


def _cell_sized(resid, weight, cols_c, b_c, grid, log_w, log_odds_prior, size_logprior, rng):
    """Resample one cell over (detect, size, flux); flux grid-marginalized per size.

    Vectorized over the size set (cols_c is (n_sizes, n_pix), b_c (n_sizes,)) — identical math to
    the per-size loop, just batched, so the SBC coverage is unchanged."""
    a = (weight * resid) @ cols_c.T  # (n_sizes,) data projection per size
    mean = a / b_c
    log_post = -0.5 * b_c[:, None] * (grid[None, :] - mean[:, None]) ** 2 + log_w[None, :]
    m = log_post.max(axis=1)  # (n_sizes,)
    slab = m + np.log(np.exp(log_post - m[:, None]).sum(axis=1))
    log_size = size_logprior + 0.5 * a * a / b_c + slab  # (n_sizes,)
    msz = log_size.max()
    log_on = log_odds_prior + msz + np.log(np.exp(log_size - msz).sum())  # summed over sizes
    if np.log(rng.uniform()) < -np.logaddexp(0.0, -log_on):
        ps = np.exp(log_size - msz)
        s = int(rng.choice(len(SIZES), p=ps / ps.sum()))
        p = np.exp(log_post[s] - log_post[s].max())
        f = float(rng.choice(grid, p=p / p.sum()))
        return True, s, f, f * cols_c[s]
    return False, 0, 0.0, np.zeros_like(cols_c[0])


def run_sized(data, weight, central_g, columns, prior, n_sweeps, rng, burn=0.3):
    """Size-aware collapsed Gibbs; returns (posterior-mean contam image, counts, totals)."""
    n_cells = columns.shape[0]
    b = (weight * columns**2).sum(axis=2)  # (n_cells, n_sizes) design norms
    cg_norm = float((weight * central_g**2).sum())
    grid, log_w = flux_grid(prior)
    log_odds_prior = np.log(prior["lam"] / n_cells) - np.log1p(-prior["lam"] / n_cells)
    size_logprior = np.log(SIZE_PROBS)

    detect = np.zeros(n_cells, dtype=bool)
    size_idx = np.zeros(n_cells, dtype=int)
    flux = np.zeros(n_cells)
    central_flux = float((data * weight * central_g).sum() / cg_norm)
    contam = np.zeros_like(data)
    accum, counts, totals, n_acc = np.zeros_like(data), [], [], 0
    burn_steps = int(burn * n_sweeps)
    for sweep in range(n_sweeps):
        central_flux = (weight * (data - contam) * central_g).sum() / cg_norm
        central_flux += rng.normal(0, 1.0 / np.sqrt(cg_norm))
        resid = data - central_flux * central_g - contam
        for c in range(n_cells):
            cur = flux[c] * columns[c, size_idx[c]] if detect[c] else 0.0
            on, s, f, contrib = _cell_sized(resid + cur, weight, columns[c], b[c], grid, log_w,
                                            log_odds_prior, size_logprior, rng)
            contam += contrib - cur
            resid = data - central_flux * central_g - contam
            detect[c], size_idx[c], flux[c] = on, s, f
        if sweep >= burn_steps:
            accum += contam
            counts.append(int(detect.sum()))
            totals.append(float(flux[detect].sum()))
            n_acc += 1
    return accum / max(n_acc, 1), np.array(counts), np.array(totals)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sbc", "bridge"], default="sbc")
    parser.add_argument("--n-draws", type=int, default=20)
    parser.add_argument("--n-sweeps", type=int, default=60)
    parser.add_argument("--n-exposures", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grid-n", type=int, default=32, help="detection grid (16 = 4x faster)")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    size = 32
    prior = {"lam": 2.0, "flux_lo": 100.0, "flux_hi": 3000.0, "alpha": 1.5}
    centers = cell_centers(size, args.grid_n, 2.5)
    columns = multisize_columns(centers, size)
    central_g = gaussian_source((size - 1) / 2.0, (size - 1) / 2.0, 4.0, size).ravel()
    n_cells = len(centers)

    if args.mode == "sbc":
        fcov = ccov = 0
        for _ in range(args.n_draws):
            dt, si, ft = sample_prior_sized(rng, prior, n_cells)
            data, w = make_stamp_sized(rng, 1.0e5, central_g, columns, dt, si, ft, 30.0)
            _, counts, totals = run_sized(data, w, central_g, columns, prior, args.n_sweeps, rng)
            tlo, thi = np.percentile(totals, [5, 95])
            clo, chi = np.percentile(counts, [5, 95])
            fcov += tlo <= ft[dt].sum() <= thi
            ccov += clo <= int(dt.sum()) <= chi
        print(f"sized-Gibbs SBC ({args.n_draws} draws, {args.n_sweeps} sweeps): "
              f"flux {fcov / args.n_draws:.2f}, count {ccov / args.n_draws:.2f}  (want ~0.90)")
        return

    real_prior = {"lam": 1.0, "flux_lo": 100.0, "flux_hi": 2000.0, "alpha": 1.5}
    real_columns = multisize_columns(cell_centers(size, size, 2.5), size)
    brng = np.random.default_rng(0)

    def clean_fn(d, w, cg):
        return run_sized(d, w, cg, real_columns, real_prior, args.n_sweeps, brng)[0]

    mean, sem, n = bridge(clean_fn, "/data/scratch/regier/sim_contamreal_stars", args.n_exposures)
    print(f"sized-Gibbs bridge on {n} real stars: cleaning {mean:+.5f} +/- {sem:.5f} "
          f"(~{100 * mean / 0.0071:.0f}% of contamination; point-source Gibbs was ~13%)")


if __name__ == "__main__":
    main()
