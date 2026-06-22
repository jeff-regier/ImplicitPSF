"""Fast core-concentration pre-screen for bake-off variants: model-side encircled energy at r=2 px.

The galaxy-size deficit is the model's ePSF being under-concentrated (too little flux in the core).
EE@r2 measures exactly that and renders in seconds, so it ranks variants without the ~2 h galaxy-
recovery screen. Higher EE@r2 = more concentrated = expected smaller deficit. We render each model's
ePSF at the same real clean-star positions (oversample=1) and also measure the empirical stars
(background-subtracted) as the target. All within r=2 px of the flux-weighted centroid.
"""

import argparse
import glob

import numpy as np
import torch

from implicitpsf.baselines.implicit_runner import load_model
from implicitpsf.datasets import load_exposure_file, make_batch
from implicitpsf.render import render_at
from implicitpsf.simulate import PATCH

R_CORE = 2.0


def ee_at_r(stamps):
    """Median flux fraction within R_CORE px of each stamp's flux-weighted centroid."""
    yy, xx = np.mgrid[0:PATCH, 0:PATCH]
    out = []
    for s in stamps:
        pos = np.clip(s, 0, None)
        tot = pos.sum() + 1e-12
        cx = (xx * pos).sum() / tot
        cy = (yy * pos).sum() / tot
        rr = np.hypot(xx - cx, yy - cy)
        out.append(s[rr <= R_CORE].sum() / (s.sum() + 1e-12))
    return np.array(out)


def star_stamps(data, index, clean):
    """Background-subtracted (edge median) empirical star stamps for the clean stars."""
    cut = data["cutouts"][index].numpy()
    val = data["valid_pixels"][index].numpy()
    out = []
    for j in clean:
        s = cut[j].astype(np.float64)
        edge = np.concatenate([s[0], s[-1], s[:, 0], s[:, -1]])
        out.append((s - np.median(edge)) * val[j])
    return np.array(out)


def collect(model, data, indices, snr_min):
    """Render model ePSF + gather star stamps at bright clean stars over several exposures."""
    ee_model, ee_star = [], []
    for index in indices:
        st = data["star_type"][index].numpy()
        snr = data["snr"][index].numpy()
        clean = np.nonzero((st == 0) & (snr >= snr_min))[0]
        if clean.size < 5:
            continue
        batch = dict(make_batch(data, [index]))
        x = data["x_pixel"][index].numpy()[clean]
        y = data["y_pixel"][index].numpy()[clean]
        q = torch.tensor(np.column_stack([np.round(x), np.round(y)]), dtype=torch.float32)
        rendered = render_at(model, batch, q, torch.zeros(len(clean)), oversample=1).numpy()
        ee_model.append(ee_at_r(rendered))
        ee_star.append(ee_at_r(star_stamps(data, index, clean)))
    return np.concatenate(ee_model), np.concatenate(ee_star)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/data/scratch/regier/sep_des_stars_v2")
    parser.add_argument("--n-exposures", type=int, default=4)
    parser.add_argument("--snr-min", type=float, default=50.0)
    parser.add_argument("--checkpoints", nargs="+", required=True, help="label=path pairs")
    args = parser.parse_args()

    data = load_exposure_file(sorted(glob.glob(f"{args.data_dir}/*.pt"))[0])
    indices = list(range(min(args.n_exposures, len(data["cutouts"]))))
    print(f"EE@r{R_CORE:.0f}, real clean stars snr>={args.snr_min:.0f} (higher=more concentrated):")
    star_ref = None
    for pair in args.checkpoints:
        label, path = pair.split("=", 1)
        model = load_model(path)
        ee_m, ee_s = collect(model, data, indices, args.snr_min)
        star_ref = ee_s
        print(f"  {label:24} EE@r2 = {np.median(ee_m):.4f}   (n={len(ee_m)} stars)")
    if star_ref is not None:
        print(f"  {'EMPIRICAL STARS (target)':24} EE@r2 = {np.median(star_ref):.4f}")


if __name__ == "__main__":
    main()
