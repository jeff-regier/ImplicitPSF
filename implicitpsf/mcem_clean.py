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
from pathlib import Path

import numpy as np
import torch

from implicitpsf.add_gibbs_cleaned import central_psf_stamps
from implicitpsf.baselines.implicit_runner import load_model
from implicitpsf.contam_model import cell_centers
from implicitpsf.datasets import load_exposure_file
from implicitpsf.mcem_sampler import cov_columns, psf_covariance, run_batch_gpu
from implicitpsf.simulate import PATCH


def build_columns(model, data, grid_n=16):
    """Covariance design built ONCE from a representative field PSF (Sigma_psf ~ constant)."""
    st = data["star_type"][0].numpy()
    rep_idx = np.nonzero(st == 0)[0][:8]
    centrals = central_psf_stamps(model, data, 0, rep_idx)
    rep = np.clip(centrals.mean(0), 0, None).ravel()
    rep = rep / (rep.sum() + 1e-12)
    centers = cell_centers(PATCH, grid_n, 2.5)
    return cov_columns(centers, psf_covariance(rep, PATCH), PATCH)


def _gather_clean_stars(model, data, cols_npix):
    """Stack every clean star in a file into (B, npix) data/weight/central arrays + their (exp,star)
    index. Centrals are the NN-rendered PSF per star (per exposure); weight = inverse-variance."""
    st = data["star_type"].numpy()  # (E, S)
    cut = data["cutouts"].numpy().reshape(st.shape[0], st.shape[1], -1)
    var = data["variance"].numpy().reshape(cut.shape)
    val = data["valid_pixels"].numpy().reshape(cut.shape)
    datas, weights, centrals, idx = [], [], [], []
    for i in range(st.shape[0]):
        clean = np.nonzero(st[i] == 0)[0]
        if len(clean) == 0:
            continue
        cen = central_psf_stamps(model, data, i, clean).reshape(len(clean), cols_npix)
        cg = np.clip(cen, 0, None)
        cg = cg / (cg.sum(axis=1, keepdims=True) + 1e-12)
        w = (val[i, clean] > 0) / np.clip(var[i, clean], 1e-6, None)
        datas.append(cut[i, clean])
        weights.append(w)
        centrals.append(cg)
        idx.extend((i, int(j)) for j in clean)
    if not datas:
        return None
    return np.concatenate(datas), np.concatenate(weights), np.concatenate(centrals), idx


def clean_file_kimpute(model, data, cols, prior, n_keep, n_sweeps, rng, device):
    """K-imputation cutouts for a WHOLE file: batch every clean star into one run_batch_gpu call
    (B ~ thousands) -> (E, S, K, PATCH, PATCH) imputation tensor (non-clean = repeated original)."""
    cut = data["cutouts"].numpy()  # (E, S, PATCH, PATCH)
    imp = np.repeat(cut[:, :, None], n_keep, axis=2)
    gathered = _gather_clean_stars(model, data, PATCH * PATCH)
    if gathered is None:
        return imp, 0
    datas, weights, centrals, idx = gathered
    samples, _, _ = run_batch_gpu(
        datas, weights, centrals, cols, prior, n_sweeps, rng, n_keep=n_keep, device=device
    )
    cleaned = datas[:, None, :] - samples  # (B, K, npix)
    for n, (i, j) in enumerate(idx):
        imp[i, j] = cleaned[n].reshape(n_keep, PATCH, PATCH)
    return imp, len(idx)


def write_kimpute(
    checkpoint, data_dir, out_dir, prior, n_keep, n_sweeps, limit, offset, device="cuda"
):
    """Write a K-imputation cleaned dataset (cutout_imp field) for the MCEM M-step."""
    model = load_model(checkpoint)
    rng = np.random.default_rng(0)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(f"{data_dir}/*.pt"))[offset:]
    if limit is not None:
        files = files[:limit]
    for path in files:
        data = load_exposure_file(path)
        cols = build_columns(model, data)  # Sigma_psf design built ONCE per file (field-constant)
        imp, n_clean = clean_file_kimpute(model, data, cols, prior, n_keep, n_sweeps, rng, device)
        data["cutout_imp"] = torch.from_numpy(imp).float()
        torch.save(data, Path(out_dir) / Path(path).name)
        print(f"{Path(path).name}: {n_clean} clean stars x {n_keep} imputations -> cutout_imp")


def _smoke(checkpoint, data_dir, n_keep, n_sweeps, device):
    rng = np.random.default_rng(0)
    model = load_model(checkpoint)
    data = load_exposure_file(sorted(glob.glob(f"{data_dir}/*.pt"))[0])
    prior = {"lam": 1.0, "flux_lo": 100.0, "flux_hi": 2000.0, "alpha": 1.5}
    cols = build_columns(model, data)
    imp, n_clean = clean_file_kimpute(model, data, cols, prior, n_keep, n_sweeps, rng, device)

    def ee2(v):
        s = np.clip(v, 0, None).reshape(PATCH, PATCH)
        c = (PATCH - 1) / 2.0
        yy, xx = np.mgrid[0:PATCH, 0:PATCH]
        return s[np.hypot(xx - c, yy - c) <= 2].sum() / (s.sum() + 1e-12)

    st = data["star_type"][0].numpy()
    clean = np.nonzero(st == 0)[0][:4]
    cut = data["cutouts"][0].numpy()
    for j in clean:
        obs = ee2(cut[j].ravel())
        ees = [ee2(imp[0, j, k].ravel()) for k in range(n_keep)]
        print(
            f"star {j}: obs EE@r2 {obs:.4f} -> {n_keep} imputations EE@r2 "
            f"mean {np.mean(ees):.4f} spread {np.std(ees):.4f}"
        )
    print(
        f"file: {n_clean} clean stars cleaned; imp shape {imp.shape} (E,S,K,P,P); "
        f"spread>0 = the posterior"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "write"], default="smoke")
    parser.add_argument("--checkpoint", default="checkpoints/sim_contamreal_nfreq8/best.pt")
    parser.add_argument("--data-dir", default="/data/scratch/regier/sim_contamreal_stars")
    parser.add_argument("--out-dir", default="/data/scratch/regier/sim_contamreal_kimpute")
    parser.add_argument("--n-keep", type=int, default=8)
    parser.add_argument("--n-sweeps", type=int, default=40)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--prior-lam", type=float, default=1.0)
    parser.add_argument("--prior-alpha", type=float, default=1.5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    prior = {"lam": args.prior_lam, "flux_lo": 100.0, "flux_hi": 2000.0, "alpha": args.prior_alpha}
    if args.mode == "write":
        write_kimpute(
            args.checkpoint,
            args.data_dir,
            args.out_dir,
            prior,
            args.n_keep,
            args.n_sweeps,
            args.limit,
            args.offset,
            args.device,
        )
    else:
        _smoke(args.checkpoint, args.data_dir, args.n_keep, args.n_sweeps, args.device)


if __name__ == "__main__":
    main()
