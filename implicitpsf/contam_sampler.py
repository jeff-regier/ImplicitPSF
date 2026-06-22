"""Single-stamp contamination sampler — first unit of the generative-EM PSF correction.

A 'clean' star stamp is modeled as a bright central source plus a faint Poisson population of
nearly-undetectable contaminants. Each source is a unit-flux bivariate Gaussian at the current PSF
FWHM (the faint sources sit near the noise, so a matched-FWHM Gaussian is indistinguishable from the
real PSF and is smooth/cheap to place; the bright central source can later swap in the real PSF).
Given the source positions the stamp is LINEAR in the fluxes (non-negative least squares), so the
reversible-jump MCMC samples only the NUMBER and POSITIONS of the faint sources. The PSF is held
fixed (its FWHM); the prior lives entirely on the contamination — Poisson rate, faint power-law
flux, uniform position — never on the PSF. Validated on a synthetic stamp with known injected
contaminants before wiring the M-step (global PSF refit) around it.
"""

import argparse

import numpy as np
from scipy.optimize import nnls
from scipy.special import erf

FWHM_TO_SIGMA = 1.0 / 2.3548


def gaussian_source(x0, y0, fwhm, size):
    """Pixel-integrated unit-flux 2D Gaussian centred at (x0, y0) on a size x size grid."""
    sigma = fwhm * FWHM_TO_SIGMA
    edges = np.arange(size + 1) - 0.5
    cdf_x = 0.5 * (1.0 + erf((edges - x0) / (sigma * np.sqrt(2.0))))
    cdf_y = 0.5 * (1.0 + erf((edges - y0) / (sigma * np.sqrt(2.0))))
    return np.outer(np.diff(cdf_y), np.diff(cdf_x))  # (size, size), sums to ~1


def design_matrix(central, positions, fwhm, size):
    """Columns = flattened unit-flux Gaussians: the central source then each contaminant."""
    cols = [gaussian_source(central[0], central[1], fwhm, size).ravel()]
    cols += [gaussian_source(x, y, fwhm, size).ravel() for x, y in positions]
    return np.stack(cols, axis=1)  # (n_pix, 1 + n_contam)


def fit_fluxes(design, data, weight):
    """Non-negative weighted least-squares source fluxes (profiles the linear amplitudes out)."""
    sw = np.sqrt(weight)
    flux, _ = nnls(design * sw[:, None], data * sw)
    return flux


def log_target(positions, data, weight, central, fwhm, size, prior):
    """log[ likelihood(profile fluxes) x prior ] for a contaminant configuration."""
    design = design_matrix(central, positions, fwhm, size)
    flux = fit_fluxes(design, data, weight)
    resid = data - design @ flux
    log_like = -0.5 * float((weight * resid**2).sum())
    n = len(positions)
    log_pois = n * np.log(prior["lam"]) - prior["lam"]  # Poisson(n) up to const
    contam_flux = np.clip(flux[1:], prior["flux_lo"], prior["flux_hi"])
    log_flux = float((-prior["alpha"] * np.log(contam_flux)).sum())  # power-law S^-alpha
    return log_like + log_pois + log_flux, flux


def _birth(positions, rng, size):
    new = rng.uniform(0, size, size=2)
    return np.vstack([positions, new]) if len(positions) else new[None, :]


def _death(positions, rng):
    drop = rng.integers(len(positions))
    return np.delete(positions, drop, axis=0)


def _move(positions, rng, step):
    out = positions.copy()
    k = rng.integers(len(positions))
    out[k] = out[k] + rng.normal(0, step, size=2)
    return out


def sampler_step(positions, cur_logp, data, weight, central, fwhm, size, prior, rng, step):
    """One reversible-jump step: birth / death / move, with Poisson-process acceptance ratios."""
    n = len(positions)
    kind = rng.choice(["birth", "death", "move"]) if n > 0 else "birth"
    if kind == "birth":
        prop = _birth(positions, rng, size)
        log_jac = np.log(prior["lam"]) - np.log(n + 1)  # birth: lam/(n+1) dimension factor
    elif kind == "death":
        prop = _death(positions, rng)
        log_jac = np.log(n) - np.log(prior["lam"])  # death: reverse of birth
    else:
        prop = _move(positions, rng, step)
        log_jac = 0.0
    prop_logp, _ = log_target(prop, data, weight, central, fwhm, size, prior)
    if np.log(rng.uniform()) < (prop_logp - cur_logp + log_jac):
        return prop, prop_logp
    return positions, cur_logp


def run_sampler(data, weight, central, fwhm, size, prior, n_steps, rng, step=1.0, burn=0.3):
    """Run the RJMCMC; return the post-burn-in counts and the visited configurations."""
    positions = np.empty((0, 2))
    cur_logp, _ = log_target(positions, data, weight, central, fwhm, size, prior)
    counts, samples = [], []
    burn_steps = int(burn * n_steps)
    for i in range(n_steps):
        positions, cur_logp = sampler_step(
            positions, cur_logp, data, weight, central, fwhm, size, prior, rng, step
        )
        if i >= burn_steps:
            counts.append(len(positions))
            samples.append(positions.copy())
    return np.array(counts), samples


def synthetic_stamp(rng, size, fwhm, central_flux, contam, noise_sigma):
    """Central source + injected faint contaminants + Gaussian noise; returns stamp + truth."""
    central = np.array([size / 2.0, size / 2.0])
    img = central_flux * gaussian_source(central[0], central[1], fwhm, size)
    true_pos = contam["pos"]
    for (x, y), f in zip(true_pos, contam["flux"], strict=True):
        img = img + f * gaussian_source(x, y, fwhm, size)
    data = img + rng.normal(0, noise_sigma, size=(size, size))
    return data.ravel(), central, true_pos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--fwhm", type=float, default=4.0)
    parser.add_argument("--n-true", type=int, default=3)
    parser.add_argument("--inject-flux", type=float, default=1000.0, help="mean injected flux")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    central_flux = 1.0e5
    noise_sigma = 30.0
    true_pos = rng.uniform(6, args.size - 6, size=(args.n_true, 2))
    lo, hi = args.inject_flux * 0.7, args.inject_flux * 1.3
    true_flux = rng.uniform(lo, hi, size=args.n_true)  # faint: near-noise contaminants
    data, central, _ = synthetic_stamp(
        rng, args.size, args.fwhm, central_flux,
        {"pos": true_pos, "flux": true_flux}, noise_sigma,
    )
    weight = np.full(args.size * args.size, 1.0 / noise_sigma**2)
    prior = {"lam": 2.0, "flux_lo": 100.0, "flux_hi": 3000.0, "alpha": 1.5}

    counts, samples = run_sampler(
        data, weight, central, args.fwhm, args.size, prior, args.steps, rng
    )
    print(f"injected {args.n_true} faint sources (flux {true_flux.round().astype(int)}, "
          f"central {central_flux:.0e}, noise {noise_sigma:.0f})")
    vals, freq = np.unique(counts, return_counts=True)
    post = {int(v): round(float(c) / len(counts), 3) for v, c in zip(vals, freq, strict=True)}
    print(f"posterior over # contaminants: {post}")
    print(f"posterior mean count = {counts.mean():.2f}  (injected {args.n_true})")
    if samples and len(samples[-1]):
        last = samples[-1]
        print(f"a sampled configuration ({len(last)} sources):")
        for x, y in last:
            d = np.hypot(true_pos[:, 0] - x, true_pos[:, 1] - y).min()
            print(f"  ({x:5.1f},{y:5.1f})  nearest injected source: {d:.1f} px")


if __name__ == "__main__":
    main()
