"""ePSF-native validation figures: the stacked normalized pixel residual.

For a sample of test exposures with on-disk FITS, render each method's PSF at the
reserved stars, normalize observed and model stamps to unit sum (the effective-PSF
"fraction of light per pixel"), and stack the mean residual. A model that has captured
the effective PSF leaves a structureless residual; coherent structure is a shape error
that second-moment metrics can miss. Outputs a vector PDF to paper/figures/.
"""

import glob
import os
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from implicitpsf.baselines.implicit_runner import load_model
from implicitpsf.evaluation.run_eval import (
    exposure_masks,
    implicit_stamps,
    piff_stamps,
    psfex_stamps,
)
from implicitpsf.splits import load_manifest, reserved_star_ids

plt.rcParams.update(
    {"font.size": 9, "font.family": "serif", "figure.dpi": 150, "savefig.bbox": "tight"}
)

FIGDIR = "paper/figures"
DATA_DIR = "/data/scratch/regier/sep_des_stars_v2"
MANIFEST = "manifests/split_v1.json"
CHECKPOINT = "checkpoints/real_v6_blend/best.pt"
HALF = 10  # central (2*HALF) x (2*HALF) window
N_EXPOSURES = 12


def unit_norm_crop(stamp):
    c = stamp.shape[0] // 2
    s = stamp[c - HALF : c + HALF, c - HALF : c + HALF].astype(np.float64)
    total = s.sum()
    return s / total if total > 0 else None


def accumulate(stamps_by_method, star_cutouts, acc, count):
    """Add unit-normalized (star - model) residuals for each method into acc/count."""
    for j in range(star_cutouts.shape[0]):
        star = unit_norm_crop(star_cutouts[j])
        if star is None:
            continue
        for method, stamps in stamps_by_method.items():
            model = unit_norm_crop(stamps[j])
            if model is None:
                continue
            acc[method] += star - model
            count[method] += 1


def collect_residual_stacks():
    manifest = load_manifest(MANIFEST)
    model = load_model(CHECKPOINT)
    methods = ["implicit", "piff", "psfex"]
    acc = {m: np.zeros((2 * HALF, 2 * HALF)) for m in methods}
    count = {m: 0 for m in methods}

    n_used = 0
    for path in sorted(glob.glob(f"{DATA_DIR}/*.pt")):
        if n_used >= N_EXPOSURES:
            break
        data = torch.load(path, map_location="cpu", weights_only=False)
        fps = np.array(data["fits_path"])
        for idx, fp in enumerate(fps):
            if n_used >= N_EXPOSURES:
                break
            if not (str(fp) and os.path.exists(str(fp))):
                continue
            reserved_ids = reserved_star_ids(manifest, data["exposure_id"][idx])
            if not reserved_ids:
                continue
            clean, reserved = exposure_masks(data, idx, reserved_ids)
            fit_mask = clean & ~reserved
            if reserved.sum() < 3 or fit_mask.sum() < 10:
                continue
            star_cutouts = data["cutouts"][idx].numpy()[reserved]
            impl = implicit_stamps(model, data, idx, reserved)
            with tempfile.TemporaryDirectory() as wd:
                piff = piff_stamps(data, idx, fit_mask, reserved, Path(wd))
                psfex = psfex_stamps(data, idx, fit_mask, reserved, Path(wd))
            stamps = {"implicit": impl, "piff": piff, "psfex": psfex}
            accumulate(stamps, star_cutouts, acc, count)
            n_used += 1
            print(f"exposure {n_used}/{N_EXPOSURES}: {reserved.sum()} reserved stars")
    means = {m: acc[m] / max(count[m], 1) for m in methods}
    return means, count


def fig_residual_stack():
    means, count = collect_residual_stacks()
    labels = {"implicit": "Neural PSF", "piff": "PIFF", "psfex": "PSFEx"}
    lim = 0.004  # common symmetric color scale (fraction of unit-sum light per pixel)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), layout="constrained")
    for ax, m in zip(axes, ["implicit", "piff", "psfex"], strict=True):
        rms = np.sqrt(np.mean(means[m] ** 2))
        im = ax.imshow(means[m], cmap="RdBu_r", vmin=-lim, vmax=lim, origin="lower")
        ax.set_title(f"{labels[m]}\nRMS$=${rms:.1e}", fontsize=8.5)
        ax.set_xticks([])
        ax.set_yticks([])
    # All panels share the symmetric scale, so one colorbar serves the row (per-panel
    # colorbars overlapped the neighboring panels).
    fig.colorbar(im, ax=axes, fraction=0.046, pad=0.02, label="fraction of unit-sum light")
    n = count["implicit"]
    fig.suptitle(
        f"Stacked normalized pixel residual (star $-$ model), {n} reserved stars",
        fontsize=9,
    )
    fig.savefig(f"{FIGDIR}/fig_residual_stack.pdf")
    plt.close(fig)
    print(f"wrote {FIGDIR}/fig_residual_stack.pdf (n={n})")


if __name__ == "__main__":
    os.makedirs(FIGDIR, exist_ok=True)
    fig_residual_stack()
