"""Encircled-energy deficit of the model PSF vs the empirical star, on reserved stars.

The W1 galaxy-recovery test traced the ~8% recovered-size deficit to the model PSF being
slightly *under-concentrated* in the central few pixels (a profile-shape effect, not a
second-moment size error). This diagnostic quantifies that directly: for reserved test
stars it renders the model PSF and the (background-subtracted) empirical star stamp, both
unit-normalized, and compares their encircled energy EE(r) = (flux within radius r of the
flux-weighted centroid) / (total flux). A negative deficit EE_model(r) - EE_star(r) at
small r is the core under-concentration. This is the headline metric for the W3
architecture bake-off (we want the deficit at r=2 px driven to ~0 with no reserved-star
ellipticity regression).

Runs on real data (the sim does not reproduce the deficit). CPU-friendly for a few hundred
stars; never share a GPU with a live training.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from implicitpsf.baselines.implicit_runner import load_model, render_implicit
from implicitpsf.datasets import load_exposure_file, make_batch
from implicitpsf.evaluation.run_eval import exposure_masks
from implicitpsf.provenance import write_result
from implicitpsf.splits import load_manifest, reserved_star_ids

DEFAULT_RADII = (1.0, 2.0, 3.0, 5.0)


NORM_RADIUS = 8.0  # aperture (px) the encircled energy is normalized to


def encircled_energy(stamps, radii, r_norm=NORM_RADIUS):
    """(n, len(radii)) encircled energy normalized to the flux within r_norm of the centroid.

    EE(r) = flux(<r) / flux(<r_norm). Normalizing to a fixed central aperture (not the full
    stamp) makes the denominator stable and positive -- the noisy outer pixels, which can
    drive a single star's total near zero and blow up the ratio, are excluded. Radii must be
    <= r_norm. The centroid uses non-negative flux for robustness.
    """
    flux = stamps.astype(np.float64)
    patch = flux.shape[-1]
    ys, xs = np.mgrid[0:patch, 0:patch]
    positive = np.clip(flux, 0.0, None)
    weight = positive / positive.sum(axis=(1, 2))[:, None, None]
    cx = (weight * xs).sum(axis=(1, 2))
    cy = (weight * ys).sum(axis=(1, 2))
    dist = np.sqrt((xs[None] - cx[:, None, None]) ** 2 + (ys[None] - cy[:, None, None]) ** 2)
    norm = (flux * (dist <= r_norm)).sum(axis=(1, 2))
    return np.stack([(flux * (dist <= r)).sum(axis=(1, 2)) / norm for r in radii], axis=1)


def background_subtract(stamps):
    """Subtract the per-stamp edge median (the local sky) from empirical star cutouts."""
    edges = np.concatenate([stamps[:, 0], stamps[:, -1], stamps[:, :, 0], stamps[:, :, -1]], axis=1)
    return stamps - np.median(edges, axis=1)[:, None, None]


def exposure_stamps(model, data, index, reserved_ids):
    """Unit-sum-normalized (model, star) reserved-star stamps for one exposure. Normalizing
    each stamp before stacking keeps bright stars from dominating; stacking then averages
    down the per-star noise so the encircled energy of the *stack* is a clean profile."""
    _, reserved = exposure_masks(data, index, reserved_ids)
    if reserved.sum() < 5:
        return None
    batch = make_batch(data, [index])
    reserved_t = torch.from_numpy(reserved).unsqueeze(0)
    model_stamps = render_implicit(model, batch, reserved_t)[0][reserved_t[0]].numpy()
    star_stamps = background_subtract(data["cutouts"][index].numpy()[reserved])
    model_stamps = _core_normalize(model_stamps)
    star_stamps = _core_normalize(star_stamps)
    keep = np.isfinite(model_stamps).all(axis=(1, 2)) & np.isfinite(star_stamps).all(axis=(1, 2))
    return model_stamps[keep], star_stamps[keep]


def _core_normalize(stamps):
    """Divide each stamp by its central-box flux (the bright, reliably-positive core), so
    stacking weights stars by core brightness rather than by a noise-dominated total."""
    half = stamps.shape[-1] // 2
    core = stamps[:, half - 4 : half + 4, half - 4 : half + 4].sum(axis=(1, 2), keepdims=True)
    return stamps / np.where(core > 0, core, np.nan)


def stacked_deficit(model_stamps, star_stamps, radii, n_boot=200):
    """EE of the stacked model and star, and the deficit with a bootstrap-over-stars error."""
    ee_model = encircled_energy(model_stamps.mean(axis=0)[None], radii)[0]
    ee_star = encircled_energy(star_stamps.mean(axis=0)[None], radii)[0]
    n = len(model_stamps)
    rng = np.random.default_rng(0)
    boot = np.array(
        [
            encircled_energy(model_stamps[idx].mean(axis=0)[None], radii)[0]
            - encircled_energy(star_stamps[idx].mean(axis=0)[None], radii)[0]
            for idx in (rng.integers(0, n, n) for _ in range(n_boot))
        ]
    )
    return ee_model, ee_star, boot.std(axis=0)


def run(checkpoint, manifest_path, data_dir, max_exposures, radii):
    model = load_model(checkpoint)
    manifest = load_manifest(manifest_path)
    selected = [
        (name, info)
        for name, info in sorted(manifest["exposures"].items())
        if info["split"] == "test"
    ]
    if max_exposures is not None:
        selected = selected[:max_exposures]
    model_stamps, star_stamps = [], []
    by_file = {}
    for name, info in selected:
        by_file.setdefault(info["file"], []).append(info["index"])
    for file_name, indices in sorted(by_file.items()):
        data = load_exposure_file(Path(data_dir) / file_name)
        for index in indices:
            reserved_ids = reserved_star_ids(manifest, data["exposure_id"][index])
            if not reserved_ids:
                continue
            result = exposure_stamps(model, data, index, reserved_ids)
            if result is not None:
                model_stamps.append(result[0])
                star_stamps.append(result[1])
    model_stamps = np.concatenate(model_stamps)
    star_stamps = np.concatenate(star_stamps)
    ee_model, ee_star, deficit_err = stacked_deficit(model_stamps, star_stamps, radii)
    return pd.DataFrame(
        {
            "radius": list(radii),
            "ee_model": ee_model,
            "ee_star": ee_star,
            "deficit": ee_model - ee_star,
            "deficit_boot_err": deficit_err,
            "n_stars": len(model_stamps),
        }
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default="manifests/split_v1.json")
    parser.add_argument("--data-dir", default="/data/scratch/regier/sep_des_stars_v2")
    parser.add_argument("--max-exposures", type=int, default=20)
    parser.add_argument("--out", default="results/encircled_energy.parquet")
    args = parser.parse_args()

    table = run(args.checkpoint, args.manifest, args.data_dir, args.max_exposures, DEFAULT_RADII)
    write_result(
        table,
        args.out,
        checkpoint=args.checkpoint,
        source="encircled_energy",
        purpose=f"EE deficit {args.checkpoint}",
    )
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
