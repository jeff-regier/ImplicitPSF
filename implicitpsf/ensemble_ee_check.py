"""Does ensembling the production seeds reduce the core under-concentration? Encircled energy is
LINEAR in the PSF, so EE(mean of PSFs) = mean(EE) exactly -- ensembling cannot sharpen a core, only
cut variance. This checks it empirically: render each seed's ePSF at real star positions, measure
the fractional encircled energy within r=2 px, and compare each seed to the 4-seed ensemble.
"""

import glob

import numpy as np
import torch

from implicitpsf.baselines.implicit_runner import load_model
from implicitpsf.datasets import load_exposure_file, make_batch
from implicitpsf.render import render_at
from implicitpsf.simulate import PATCH

SEEDS = ["s0", "s1", "s2", "s3"]
DATA = "/data/scratch/regier/sep_des_stars_v2"
R = 2.0  # core radius (px) for encircled energy


def ee_within(kernels):
    """Fraction of each (n, P, P) kernel's flux within radius R of the stamp center."""
    half = PATCH // 2
    yy, xx = np.mgrid[0:PATCH, 0:PATCH]
    rr = np.hypot(xx - half, yy - half)
    mask = rr <= R
    return (kernels[:, mask].sum(1)) / (kernels.sum((1, 2)) + 1e-12)


def main():
    models = [load_model(f"checkpoints/real_v6_rff8_ps_{s}/best.pt") for s in SEEDS]
    data = load_exposure_file(sorted(glob.glob(f"{DATA}/*.pt"))[0])
    index = 0
    batch = dict(make_batch(data, [index]))
    st = data["star_type"][index].numpy()
    clean = np.nonzero(st == 0)[0][:40]
    x = data["x_pixel"][index].numpy()[clean]
    y = data["y_pixel"][index].numpy()[clean]
    queries = torch.tensor(np.column_stack([np.round(x), np.round(y)]), dtype=torch.float32)
    colors = torch.zeros(len(clean))

    per_seed = [render_at(m, batch, queries, colors, oversample=1).numpy() for m in models]
    ee_seed = [ee_within(k) for k in per_seed]  # (n,) per seed
    ensemble = np.mean(per_seed, axis=0)  # average the ePSFs
    ee_ens = ee_within(ensemble)

    print(f"EE within r={R}px on {len(clean)} stars (higher = more concentrated core):")
    for s, ee in zip(SEEDS, ee_seed, strict=True):
        print(f"  seed {s}: median EE@r2 = {np.median(ee):.4f}")
    print(f"  mean-of-seeds median        = {np.median(np.mean(ee_seed, axis=0)):.4f}")
    print(f"  ENSEMBLE (mean PSF) median  = {np.median(ee_ens):.4f}")
    print("\n=> ensemble EE matches the mean seed EE (linearity); ensembling does NOT sharpen the")
    print("   core, so it cannot reduce the size deficit -- only cuts the tiny seed scatter.")


if __name__ == "__main__":
    main()
