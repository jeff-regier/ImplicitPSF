"""K-imputation E-step on sim/real data: per-star Gibbs samples -> K cleaned cutouts to retrain on.

Given the global (lambda, alpha) (inferred separately by mcem_estep.hier_gibbs over a subset), this
cleans every clean star in parallel: render the current PSF as the central template, build the
covariance design once per exposure (Sigma_psf lower bound from the field PSF), and for each star
draw K thinned post-burn contamination samples (mcem_sampler.run) -> K cleaned cutouts y - C^(k).
The M-step trains on the K imputations (random one per minibatch) so the contaminant posterior
UNCERTAINTY propagates -- no posterior mean. Reuses add_gibbs_cleaned's loader/central render and
mcem_sampler's calibrated, well-mixing per-star sampler.
"""

import argparse
import glob

import numpy as np

from implicitpsf.add_gibbs_cleaned import central_psf_stamps
from implicitpsf.baselines.implicit_runner import load_model
from implicitpsf.contam_model import cell_centers
from implicitpsf.datasets import load_exposure_file
from implicitpsf.mcem_sampler import cov_columns, psf_covariance, run
from implicitpsf.simulate import PATCH


def clean_exposure_kimpute(model, data, index, prior, n_keep, n_sweeps, rng, grid_n=16,
                           max_stars=10**9):
    """Return (clean star indices, (n_clean, K, n_pix) cleaned imputations) for one exposure."""
    st = data["star_type"][index].numpy()
    clean = np.nonzero(st == 0)[0][:max_stars]
    if len(clean) == 0:
        return clean, np.zeros((0, n_keep, PATCH * PATCH))
    centrals = central_psf_stamps(model, data, index, clean)  # (n_clean, PATCH, PATCH)
    centers = cell_centers(PATCH, grid_n, 2.5)
    rep = np.clip(centrals.mean(0), 0, None).ravel()
    rep = rep / (rep.sum() + 1e-12)
    cols = cov_columns(centers, psf_covariance(rep, PATCH), PATCH)  # once per exposure
    cut = data["cutouts"][index].numpy()
    var = data["variance"][index].numpy()
    val = data["valid_pixels"][index].numpy()
    out = np.empty((len(clean), n_keep, PATCH * PATCH))
    for n, (j, cen) in enumerate(zip(clean, centrals)):
        cg = np.clip(cen, 0, None).ravel()
        cg = cg / (cg.sum() + 1e-12)
        w = (val[j].ravel() > 0) / np.clip(var[j].ravel(), 1e-6, None)
        samples, _, _, _ = run(cut[j].ravel(), w, cg, cols, prior, n_sweeps, rng, n_keep=n_keep)
        out[n] = cut[j].ravel()[None, :] - samples
    return clean, out


def _smoke(checkpoint, data_dir, n_keep, n_sweeps):
    rng = np.random.default_rng(0)
    model = load_model(checkpoint)
    data = load_exposure_file(sorted(glob.glob(f"{data_dir}/*.pt"))[0])
    prior = {"lam": 1.0, "flux_lo": 100.0, "flux_hi": 2000.0, "alpha": 1.5}
    clean, imp = clean_exposure_kimpute(model, data, 0, prior, n_keep, n_sweeps, rng, max_stars=4)

    def ee2(v):
        s = np.clip(v, 0, None).reshape(PATCH, PATCH)
        c = (PATCH - 1) / 2.0
        yy, xx = np.mgrid[0:PATCH, 0:PATCH]
        return s[np.hypot(xx - c, yy - c) <= 2].sum() / (s.sum() + 1e-12)

    cut = data["cutouts"][0].numpy()
    for n, j in enumerate(clean):
        obs = ee2(cut[j].ravel())
        ees = [ee2(imp[n, k]) for k in range(n_keep)]
        print(f"star {j}: obs EE@r2 {obs:.4f} -> {n_keep} imputations EE@r2 "
              f"mean {np.mean(ees):.4f} spread {np.std(ees):.4f}")
    print(f"shape {imp.shape} (n_clean, K, n_pix); imputations vary (spread>0) = the posterior")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/sim_contamreal_nfreq8/best.pt")
    parser.add_argument("--data-dir", default="/data/scratch/regier/sim_contamreal_stars")
    parser.add_argument("--n-keep", type=int, default=4)
    parser.add_argument("--n-sweeps", type=int, default=40)
    args = parser.parse_args()
    _smoke(args.checkpoint, args.data_dir, args.n_keep, args.n_sweeps)


if __name__ == "__main__":
    main()
