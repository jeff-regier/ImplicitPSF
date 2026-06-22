"""Precompute clean-PSF supervision targets for the contamination-correction experiment.

For each star in a contaminated sim exposure, render the KNOWN clean truth PSF (from true_field) at
its position, unit-normalized — the target the corrected network should predict instead of the
contaminated cutout. Adds a `clean_psf` (n_stars, PATCH, PATCH) field and writes to a new directory
(originals untouched). The clean-target loss (blend.py, --loss-mode clean) supervises the decoded
target PSF against this, so the network learns to output the clean PSF from contaminated context.
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import torch

from implicitpsf.datasets import load_exposure_file
from implicitpsf.evaluation.sim_truth import truth_stamps
from implicitpsf.simulate import COLOR_MEAN, set_psf_model


def clean_targets(data, index):
    """Unit-normalized clean truth-PSF stamps at every star position in one exposure."""
    field = {"chromatic": False, **data["true_field"][index]}
    ref_color = COLOR_MEAN if field["chromatic"] else 0.0
    x = data["x_pixel"][index].numpy()
    y = data["y_pixel"][index].numpy()
    stamps = truth_stamps(field, x, y, ref_color)  # (n_stars, P, P), noiseless clean PSF
    norm = stamps.reshape(len(stamps), -1).sum(axis=1).clip(min=1e-12)
    return (stamps / norm[:, None, None]).astype(np.float32)


def process_file(path, out_dir):
    data = load_exposure_file(path)
    n_exp = len(data["cutouts"])
    clean = np.stack([clean_targets(data, i) for i in range(n_exp)])  # (n_exp, n_stars, P, P)
    data["clean_psf"] = torch.from_numpy(clean)
    out_path = Path(out_dir) / Path(path).name
    torch.save(data, out_path)
    return out_path, clean.shape


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--psf-model", default="realistic")
    parser.add_argument("--limit", type=int, default=None, help="process only N files (smoke test)")
    args = parser.parse_args()
    set_psf_model(args.psf_model)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(f"{args.data_dir}/*.pt"))
    if args.limit is not None:
        files = files[: args.limit]
    for path in files:
        out_path, shape = process_file(path, args.out_dir)
        print(f"{Path(path).name}: clean_psf {shape} -> {out_path}")


if __name__ == "__main__":
    main()
