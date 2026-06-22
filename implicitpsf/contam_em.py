"""Point-estimate EM loop: iteratively clean contamination and re-fit the PSF, on a synthetic field.

Demonstrates the core of the generative correction WITHOUT the neural net. Many stars share one true
PSF; each = the central source (the PSF) + a faint Poisson population of contaminants (matched-FWHM
Gaussians) + noise. Scattered over stars the contamination averages into a broad faint halo, so the
naive stack is too wide -- the deficit. Each EM round: E-step greedily fits + subtracts the per-star
contamination (point estimate; prior = detection threshold + an off-core search region) using the
current PSF as the central template; M-step re-stacks the cleaned stars into a sharper PSF. We track
the PSF's encircled energy at r=2 toward the truth -- the metric the deficit is measured in. The NN
M-step replaces the stack later; here we check the loop sharpens the PSF and stops at the truth (not
past it -- the threshold/off-core prior is what keeps it from over-subtracting the PSF's own wings).
"""

import argparse

import numpy as np

from implicitpsf.contam_mcmc import posterior_mean_contam
from implicitpsf.contam_sampler import FWHM_TO_SIGMA, fit_fluxes, gaussian_source


def psf_fwhm(stamp):
    """Second-moment FWHM (px) of a non-negative PSF stamp."""
    size = stamp.shape[0]
    c = (size - 1) / 2.0
    yy, xx = np.mgrid[0:size, 0:size]
    w = np.clip(stamp, 0, None)
    second_moment = (((xx - c) ** 2 + (yy - c) ** 2) * w).sum() / w.sum()  # = 2 sigma^2
    return np.sqrt(second_moment / 2.0) / FWHM_TO_SIGMA


def ee_at_r(stamp, r):
    """Encircled energy within r px of the stamp centre (unit-sum stamp)."""
    size = stamp.shape[0]
    c = (size - 1) / 2.0
    yy, xx = np.mgrid[0:size, 0:size]
    rr = np.hypot(xx - c, yy - c)
    return float(stamp[rr <= r].sum() / stamp.sum())


def _off_core_peak(resid, center, min_radius):
    """Brightest residual pixel outside min_radius of the centre; returns (x, y, value)."""
    size = resid.shape[0]
    yy, xx = np.mgrid[0:size, 0:size]
    rr = np.hypot(xx - center, yy - center)
    masked = np.where(rr >= min_radius, resid, -np.inf)
    py, px = np.unravel_index(np.argmax(masked), resid.shape)
    return px, py, masked[py, px]


def clean_star(stamp, weight, psf, fwhm, noise, threshold, max_sources, min_radius):
    """Greedy point-estimate E-step: central = current PSF template, add faint Gaussian contaminants
    at off-core residual peaks above `threshold` sigma. Returns (contaminant model, central amp)."""
    size = int(np.sqrt(stamp.size))
    central = psf.ravel()
    positions = []
    center = (size - 1) / 2.0
    for _ in range(max_sources):
        design = np.column_stack([central] + [gaussian_source(x, y, fwhm, size).ravel()
                                              for x, y in positions])
        flux = fit_fluxes(design, stamp, weight)
        resid = (stamp - design @ flux).reshape(size, size)
        px, py, peak = _off_core_peak(resid, center, min_radius)
        if peak < threshold * noise:
            break
        positions.append((px, py))
    design = np.column_stack([central] + [gaussian_source(x, y, fwhm, size).ravel()
                                          for x, y in positions])
    flux = fit_fluxes(design, stamp, weight)
    contam = (design[:, 1:] @ flux[1:]) if positions else np.zeros_like(stamp)
    return contam.reshape(size, size), float(flux[0])


def em_round(stars, weights, psf, noise, threshold, max_sources, min_radius):
    """One EM iteration: clean every star with the current PSF, re-stack into a new PSF."""
    size = psf.shape[0]
    fwhm = psf_fwhm(psf)
    cleaned = []
    for d, w in zip(stars, weights, strict=True):
        contam, amp = clean_star(d.ravel(), w.ravel(), psf, fwhm, noise, threshold,
                                 max_sources, min_radius)
        cleaned.append((d - contam) / max(amp, 1e-8))
    new_psf = np.clip(np.mean(cleaned, axis=0), 0, None)
    return (new_psf / new_psf.sum()).reshape(size, size)


def em_round_mcmc(stars, weights, psf, prior, rng, n_steps):
    """Calibrated EM round: E-step subtracts each star's POSTERIOR-MEAN contamination (marginalized,
    faint sources included by probability) via the calibrated MCMC; M-step re-stacks."""
    size = psf.shape[0]
    fwhm = psf_fwhm(psf)
    central_g = psf.ravel()
    cleaned = []
    for d, w in zip(stars, weights, strict=True):
        contam = posterior_mean_contam(d.ravel(), w.ravel(), central_g, fwhm, size, prior,
                                       n_steps, rng)
        c = d.ravel() - contam
        cleaned.append((c / max(c.sum(), 1e-8)).reshape(size, size))
    new_psf = np.clip(np.mean(cleaned, axis=0), 0, None)
    return new_psf / new_psf.sum()


def make_field(rng, n_stars, size, fwhm, central_flux, lam, flux_lo, flux_hi, noise_sigma):
    """Synthetic field: each star = central PSF (Gaussian) + Poisson faint Gaussian contaminants."""
    true_psf = gaussian_source((size - 1) / 2.0, (size - 1) / 2.0, fwhm, size)
    stars = []
    for _ in range(n_stars):
        img = central_flux * true_psf.copy()
        k = rng.poisson(lam)
        for _ in range(k):
            x, y = rng.uniform(4, size - 4, size=2)
            img = img + rng.uniform(flux_lo, flux_hi) * gaussian_source(x, y, fwhm, size)
        stars.append(img + rng.normal(0, noise_sigma, size=(size, size)))
    return np.array(stars), true_psf


def normalized_stack(stars):
    """Naive stack: each star normalized by its total flux, then averaged (the contaminated PSF)."""
    norm = np.array([s / s.sum() for s in stars])
    psf = norm.mean(axis=0)
    return psf / psf.sum()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-stars", type=int, default=300)
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--fwhm", type=float, default=4.0)
    parser.add_argument("--lam", type=float, default=2.0)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--threshold", type=float, default=3.0)
    parser.add_argument("--calibrated", action="store_true", help="use the calibrated MCMC E-step")
    parser.add_argument("--mcmc-steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    central_flux, noise_sigma = 1.0e5, 30.0
    stars, true_psf = make_field(rng, args.n_stars, args.size, args.fwhm, central_flux,
                                 args.lam, 500.0, 2000.0, noise_sigma)
    weights = np.full_like(stars, 1.0 / noise_sigma**2)
    prior = {"lam": args.lam, "flux_lo": 300.0, "flux_hi": 2500.0, "alpha": 1.0}

    true_ee = ee_at_r(true_psf, 2.0)
    psf = normalized_stack(stars)
    e_step = "calibrated MCMC" if args.calibrated else "greedy point-estimate"
    print(f"{args.n_stars} stars, true FWHM {args.fwhm}, lam {args.lam}, E-step: {e_step}")
    print(f"  truth          EE@r2 = {true_ee:.4f}")
    print(f"  round 0 (stack) EE@r2 = {ee_at_r(psf, 2.0):.4f}  "
          f"(deficit {ee_at_r(psf, 2.0) - true_ee:+.4f})")
    for r in range(1, args.rounds + 1):
        if args.calibrated:
            psf = em_round_mcmc(stars, weights, psf, prior, rng, args.mcmc_steps)
        else:
            psf = em_round(stars, weights, psf, noise_sigma, args.threshold, 6, min_radius=2.5)
        ee = ee_at_r(psf, 2.0)
        print(f"  EM round {r}      EE@r2 = {ee:.4f}  (deficit {ee - true_ee:+.4f})")


if __name__ == "__main__":
    main()
