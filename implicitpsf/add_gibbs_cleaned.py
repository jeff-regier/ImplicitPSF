"""Real-data EM M-step prep: Gibbs-clean the contam-sim stars -> cleaned cutouts to retrain on.

For each clean star, render the CURRENT PSF estimate (the NN) at its position as the central
template, run the sized Gibbs (contam_sized) to infer the posterior-mean contamination, subtract ->
cleaned cutout. Writes a copy of the data with the clean stars' cutouts replaced by the cleaned
versions. Retraining the PSF on the cleaned data (normal blend loss) is the EM M-step; iterate
with the sharpened PSF as the next central template. This is the end-to-end test: does Gibbs-
cleaning shrink the real (simulated) PSF/galaxy-size deficit? Per-star Gibbs is slow -> use
--limit / --max-stars for tests.
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import torch

from implicitpsf.baselines.implicit_runner import load_model
from implicitpsf.contam_model import cell_centers
from implicitpsf.contam_sized import multisize_columns, run_sized
from implicitpsf.datasets import load_exposure_file, make_batch
from implicitpsf.render import render_at
from implicitpsf.simulate import PATCH

PRIOR = {"lam": 1.0, "flux_lo": 100.0, "flux_hi": 2000.0, "alpha": 1.5}


def central_psf_stamps(model, data, index, star_idx):
    """Render the NN's predicted PSF at each star position (the Gibbs central template)."""
    batch = dict(make_batch(data, [index]))
    x = data["x_pixel"][index].numpy()[star_idx]
    y = data["y_pixel"][index].numpy()[star_idx]
    q = torch.tensor(np.column_stack([np.round(x), np.round(y)]), dtype=torch.float32)
    return render_at(model, batch, q, torch.zeros(len(star_idx)), oversample=1).numpy()


def clean_exposure(model, data, index, columns, n_sweeps, rng, max_stars, prior):
    """Replace clean stars' cutouts with Gibbs-cleaned versions; returns how many were cleaned."""
    st = data["star_type"][index].numpy()
    clean = np.nonzero(st == 0)[0][:max_stars]
    if len(clean) == 0:
        return 0
    centrals = central_psf_stamps(model, data, index, clean)
    cut = data["cutouts"][index].numpy()
    var = data["variance"][index].numpy()
    val = data["valid_pixels"][index].numpy()
    for k, j in enumerate(clean):
        cg = np.clip(centrals[k], 0, None).ravel()
        cg = cg / (cg.sum() + 1e-12)
        w = (val[j].ravel() > 0) / np.clip(var[j].ravel(), 1e-6, None)
        contam, _, _ = run_sized(cut[j].ravel(), w, cg, columns, prior, n_sweeps, rng)
        data["cutouts"][index][j] = torch.from_numpy(
            (cut[j].ravel() - contam).reshape(PATCH, PATCH)
        )
    return len(clean)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", required=True, help="current PSF estimate (central template)"
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-sweeps", type=int, default=40)
    parser.add_argument("--limit", type=int, default=None, help="process only N files")
    parser.add_argument("--offset", type=int, default=0, help="skip the first OFFSET files")
    parser.add_argument("--max-stars", type=int, default=10**9, help="clean stars per exposure cap")
    parser.add_argument("--grid-n", type=int, default=PATCH, help="detection grid (16 = 4x faster)")
    parser.add_argument("--prior-lam", type=float, default=PRIOR["lam"], help="contam rate (lower=less)")
    parser.add_argument("--prior-flux-hi", type=float, default=PRIOR["flux_hi"], help="flux ceiling")
    args = parser.parse_args()

    prior = {**PRIOR, "lam": args.prior_lam, "flux_hi": args.prior_flux_hi}
    model = load_model(args.checkpoint)
    centers = cell_centers(PATCH, args.grid_n, 2.5)
    columns = multisize_columns(centers, PATCH)
    rng = np.random.default_rng(0)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(f"{args.data_dir}/*.pt"))[args.offset :]
    if args.limit is not None:
        files = files[: args.limit]
    total = 0
    for path in files:
        data = load_exposure_file(path)
        n = sum(
            clean_exposure(model, data, i, columns, args.n_sweeps, rng, args.max_stars, prior)
            for i in range(len(data["cutouts"]))
        )
        torch.save(data, Path(args.out_dir) / Path(path).name)
        total += n
        print(f"{Path(path).name}: cleaned {n} stars")
    print(f"done: {total} stars cleaned over {len(files)} files -> {args.out_dir}")


if __name__ == "__main__":
    main()
