"""Calibrated flux-sampling contamination MCMC (replaces the point-estimate NNLS profiling).

The point-estimate sampler (contam_sampler.py) profiles fluxes by NNLS, which biases the count
posterior toward fewer sources (no Occam factor on the flux dimension, hard non-negativity).
The EM MARGINALIZES over contamination, so the posterior must be CALIBRATED, not conservative. Here
the fluxes are SAMPLED (true truncated power-law prior, positivity from the prior) jointly with the
number and positions by reversible-jump MCMC (births propose marks from the prior, so the continuous
parameters cancel and the count factor is just lambda/(K+1)). Validated by recovery + posterior
coverage; simulation-based calibration is the acceptance test.
"""

import argparse

import numpy as np

from implicitpsf.contam_sampler import gaussian_source


def sample_powerlaw(rng, lo, hi, alpha, n):
    """Inverse-CDF sample from p(S) ∝ S^-alpha truncated to [lo, hi]."""
    u = rng.uniform(size=n)
    a = 1.0 - alpha
    if abs(a) < 1e-9:
        return lo * (hi / lo) ** u
    return (lo**a + u * (hi**a - lo**a)) ** (1.0 / a)


def log_powerlaw(flux, lo, hi, alpha):
    """Normalized log p(S) ∝ S^-alpha on [lo, hi]; -inf outside (array in, array out)."""
    a = 1.0 - alpha
    norm = (hi**a - lo**a) / a if abs(a) > 1e-9 else np.log(hi / lo)
    inside = (flux >= lo) & (flux <= hi)
    return np.where(inside, -alpha * np.log(np.where(inside, flux, 1.0)) - np.log(norm), -np.inf)


def model_loglike(data, weight, central_flux, central_g, positions, fluxes, fwhm, size):
    """Gaussian log-likelihood for the full additive model (central + sampled contaminants)."""
    model = central_flux * central_g
    for (x, y), f in zip(positions, fluxes, strict=True):
        model = model + f * gaussian_source(x, y, fwhm, size).ravel()
    return -0.5 * float((weight * (data - model) ** 2).sum())


def log_post(state, data, weight, central_g, fwhm, size, prior):
    """log posterior up to const: likelihood + Poisson(K) + power-law(contaminant fluxes)."""
    cf, pos, fl = state
    ll = model_loglike(data, weight, cf, central_g, pos, fl, fwhm, size)
    k = len(fl)
    lp = k * np.log(prior["lam"]) - prior["lam"]
    if k:
        lp += float(log_powerlaw(fl, prior["flux_lo"], prior["flux_hi"], prior["alpha"]).sum())
    return ll + lp


def _birth(state, rng, size, prior):
    cf, pos, fl = state
    xy = rng.uniform(0, size, size=2)
    f = float(sample_powerlaw(rng, prior["flux_lo"], prior["flux_hi"], prior["alpha"], 1)[0])
    return cf, np.vstack([pos, xy]) if len(pos) else xy[None, :], np.append(fl, f)


def _death(state, rng):
    cf, pos, fl = state
    j = rng.integers(len(fl))
    return cf, np.delete(pos, j, axis=0), np.delete(fl, j)


def _move_pos(state, rng, step):
    cf, pos, fl = state
    pos = pos.copy()
    j = rng.integers(len(fl))
    pos[j] = pos[j] + rng.normal(0, step, size=2)
    return cf, pos, fl


def _move_flux(state, rng, log_step, prior):
    """Log-space random walk on one flux (central or contaminant); returns state + log Jac."""
    cf, pos, fl = state
    fl = fl.copy()
    n = len(fl) + 1
    j = rng.integers(n)
    if j == 0:
        old = cf
        cf = cf * np.exp(rng.normal(0, log_step))
        return (cf, pos, fl), np.log(cf / old)  # central: flat prior, Jacobian only
    old = fl[j - 1]
    fl[j - 1] = old * np.exp(rng.normal(0, log_step))
    lp = float(log_powerlaw(np.array([fl[j - 1]]), prior["flux_lo"], prior["flux_hi"],
                            prior["alpha"])[0])
    lp -= float(log_powerlaw(np.array([old]), prior["flux_lo"], prior["flux_hi"],
                             prior["alpha"])[0])
    return (cf, pos, fl), lp + np.log(fl[j - 1] / old)


def step(state, cur_lp, data, weight, central_g, fwhm, size, prior, rng, pos_step, log_step):
    k = len(state[2])
    move = rng.choice(["birth", "death", "pos", "flux"]) if k else rng.choice(["birth", "flux"])
    log_corr = 0.0
    if move == "birth":
        prop = _birth(state, rng, size, prior)
        log_corr = np.log(prior["lam"]) - np.log(k + 1)
    elif move == "death":
        prop = _death(state, rng)
        log_corr = np.log(k) - np.log(prior["lam"])
    elif move == "pos":
        prop = _move_pos(state, rng, pos_step)
    else:
        prop, log_corr = _move_flux(state, rng, log_step, prior)
    prop_lp = log_post(prop, data, weight, central_g, fwhm, size, prior)
    if np.log(rng.uniform()) < (prop_lp - cur_lp + log_corr):
        return prop, prop_lp
    return state, cur_lp


def run(data, weight, central_g, fwhm, size, prior, n_steps, rng, pos_step=1.0, log_step=0.3,
        burn=0.3):
    """RJMCMC; returns post-burn-in (count, total contaminant flux) per retained step."""
    cf0 = float((data * central_g).sum() / (central_g**2).sum())  # quick central init
    state = (cf0, np.empty((0, 2)), np.empty(0))
    cur_lp = log_post(state, data, weight, central_g, fwhm, size, prior)
    counts, totals = [], []
    burn_steps = int(burn * n_steps)
    for i in range(n_steps):
        state, cur_lp = step(state, cur_lp, data, weight, central_g, fwhm, size, prior, rng,
                             pos_step, log_step)
        if i >= burn_steps:
            counts.append(len(state[2]))
            totals.append(float(state[2].sum()))
    return np.array(counts), np.array(totals)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--fwhm", type=float, default=4.0)
    parser.add_argument("--n-true", type=int, default=3)
    parser.add_argument("--inject-flux", type=float, default=1000.0)
    parser.add_argument("--steps", type=int, default=40000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    central_flux, noise = 1.0e5, 30.0
    central_g = gaussian_source((args.size - 1) / 2.0, (args.size - 1) / 2.0, args.fwhm, args.size)
    img = central_flux * central_g
    true_pos = rng.uniform(6, args.size - 6, size=(args.n_true, 2))
    true_flux = rng.uniform(args.inject_flux * 0.7, args.inject_flux * 1.3, size=args.n_true)
    for (x, y), f in zip(true_pos, true_flux, strict=True):
        img = img + f * gaussian_source(x, y, args.fwhm, args.size)
    data = (img + rng.normal(0, noise, size=(args.size, args.size))).ravel()
    weight = np.full(args.size**2, 1.0 / noise**2)
    central_g = central_g.ravel()
    prior = {"lam": 2.0, "flux_lo": 100.0, "flux_hi": 3000.0, "alpha": 1.5}

    counts, totals = run(data, weight, central_g, args.fwhm, args.size, prior, args.steps, rng)
    vals, freq = np.unique(counts, return_counts=True)
    post = {int(v): round(float(c) / len(counts), 3) for v, c in zip(vals, freq, strict=True)}
    print(f"injected {args.n_true} sources, fluxes {true_flux.round().astype(int)} "
          f"(total {true_flux.sum():.0f}), noise {noise:.0f}")
    print(f"posterior over # contaminants: {post}")
    print(f"posterior count mean = {counts.mean():.2f}  (injected {args.n_true})")
    lo_t, hi_t = np.percentile(totals, [5, 95])
    print(f"total contaminant flux: posterior mean {totals.mean():.0f}, "
          f"90% CI [{lo_t:.0f}, {hi_t:.0f}]"
          f"  (true {true_flux.sum():.0f})")


if __name__ == "__main__":
    main()
