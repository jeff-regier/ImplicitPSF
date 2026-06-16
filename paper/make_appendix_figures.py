"""Appendix figures: (A1) a real DES image with the star-selection boxes overlaid, and
(A2) one validation star shown beside the PSFEx, PIFF, and ImplicitPSF models at its
position. Outputs vector PDFs to paper/figures/.
"""

import glob
import os
import tempfile
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from astropy.io import fits

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


def robust_std(x):
    return 1.4826 * np.median(np.abs(x - np.median(x))) + 1e-9


def load_exposure_with_fits():
    """First test exposure whose on-disk FITS exists."""
    manifest = load_manifest(MANIFEST)
    for path in sorted(glob.glob(f"{DATA_DIR}/*.pt")):
        data = torch.load(path, map_location="cpu", weights_only=False)
        fps = np.array(data["fits_path"])
        for i, fp in enumerate(fps):
            if str(fp) and os.path.exists(str(fp)):
                exp_id = data["exposure_id"][i]
                if reserved_star_ids(manifest, exp_id):
                    return data, i, manifest
    raise RuntimeError("no test exposure with FITS + reserved stars found")


def read_sci(fits_path):
    with fits.open(fits_path) as hdul:
        for hdu in hdul:
            if hdu.data is not None and hdu.data.ndim == 2 and hdu.data.shape[0] > 1000:
                return hdu.data.astype(np.float32)
    raise RuntimeError("no SCI image plane found")


def fig_des_selection():
    data, idx, manifest = load_exposure_with_fits()
    sci = read_sci(str(np.array(data["fits_path"])[idx]))
    x = data["x_pixel"][idx].numpy()
    y = data["y_pixel"][idx].numpy()
    st = data["star_type"][idx].numpy()
    flux = data["flux"][idx].numpy()
    reserved_ids = reserved_star_ids(manifest, data["exposure_id"][idx])
    clean, reserved = exposure_masks(data, idx, reserved_ids)
    context_only = (st == 1) & (flux > 0)
    fit = clean & ~reserved

    # crop a window around a clean-star-dense region for visible detail
    cy, cx = int(np.median(y[clean])), int(np.median(x[clean]))
    half = 380
    y0, y1 = max(0, cy - half), min(sci.shape[0], cy + half)
    x0, x1 = max(0, cx - half), min(sci.shape[1], cx + half)
    crop = sci[y0:y1, x0:x1]
    vmed = np.median(crop)
    show = np.arcsinh((crop - vmed) / robust_std(crop.ravel()))

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.imshow(
        show,
        cmap="gray_r",
        origin="lower",
        extent=[x0, x1, y0, y1],
        vmin=np.percentile(show, 5),
        vmax=np.percentile(show, 99.5),
    )
    s = 16
    styles = [
        (fit, "limegreen", "clean star (fit/score)"),
        (reserved, "deepskyblue", "reserved (held out)"),
        (context_only, "orange", "context-only star"),
    ]
    for mask, color, label in styles:
        inwin = mask & (x > x0) & (x < x1) & (y > y0) & (y < y1)
        for xi, yi in zip(x[inwin], y[inwin], strict=True):
            ax.add_patch(
                mpatches.Rectangle(
                    (xi - s, yi - s), 2 * s, 2 * s, fill=False, edgecolor=color, lw=1.2
                )
            )
        ax.plot([], [], "s", mfc="none", mec=color, label=label)
    ax.legend(loc="upper right", fontsize=7.5, framealpha=0.9)
    ax.set_xlabel("CCD $x$ (px)")
    ax.set_ylabel("CCD $y$ (px)")
    ax.set_title("DES single-epoch $r$-band (CCD 31), star selection")
    fig.savefig(f"{FIGDIR}/fig_des_selection.pdf")
    plt.close(fig)


def fig_psf_comparison():
    data, idx, manifest = load_exposure_with_fits()
    reserved_ids = reserved_star_ids(manifest, data["exposure_id"][idx])
    clean, reserved = exposure_masks(data, idx, reserved_ids)
    fit_mask = clean & ~reserved
    model = load_model(CHECKPOINT)

    with tempfile.TemporaryDirectory() as wd:
        piff = piff_stamps(data, idx, fit_mask, reserved, Path(wd))
        psfex = psfex_stamps(data, idx, fit_mask, reserved, Path(wd))
    impl = implicit_stamps(model, data, idx, reserved)
    cut = data["cutouts"][idx].numpy()[reserved]
    snr = data["snr"][idx].numpy()[reserved]
    j = int(np.argmax(snr))  # a high-S/N reserved star, clean to look at

    def norm(s):
        s = s.astype(np.float32)
        return s / s.sum()

    panels = [
        (norm(cut[j]), "observed star"),
        (norm(psfex[j]), "PSFEx"),
        (norm(piff[j]), "PIFF"),
        (norm(impl[j]), "ImplicitPSF (ours)"),
    ]
    vmax = max(p[0].max() for p in panels)
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.0))
    for ax, (img, label) in zip(axes, panels, strict=True):
        c = img.shape[0] // 2
        ax.imshow(img[c - 12 : c + 12, c - 12 : c + 12], cmap="magma", vmin=0, vmax=vmax)
        ax.set_title(label, fontsize=8.5)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Same reserved test star: observed versus modeled PSF", fontsize=9, y=1.04)
    fig.savefig(f"{FIGDIR}/fig_psf_comparison.pdf")
    plt.close(fig)


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    fig_des_selection()
    print(f"wrote {FIGDIR}/fig_des_selection.pdf")
    fig_psf_comparison()
    print(f"wrote {FIGDIR}/fig_psf_comparison.pdf")


if __name__ == "__main__":
    main()
