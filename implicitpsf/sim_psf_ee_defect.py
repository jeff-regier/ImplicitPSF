"""Direct PSF-level encircled-energy defect on SIMULATED data, where the truth PSF is known
exactly (no noise floor, unlike real stars). For each test exposure we render the model ePSF and
the TRUE PSF at the same star-free grid positions, with identical stamp centering, and compare the
fraction of flux within r=2 px. Because both stamps share the exact sub-pixel center and carry no
noise, dEE = EE_model - EE_truth is a clean statement of whether the learned PSF is under-
concentrated in the sim -- the PSF-level cause behind the galaxy-size discrepancy.

Run on a CLEAN-sim model and a CONTAMINATED-sim model to see whether the defect is present, and
whether contamination of the training stars is what produces it.
"""

import argparse

import numpy as np

from implicitpsf.baselines.implicit_runner import load_model
from implicitpsf.datasets import load_exposure_file
from implicitpsf.evaluation.run_eval import exposure_masks
from implicitpsf.evaluation.sim_truth import grid_positions, implicit_grid_stamps, truth_stamps
from implicitpsf.simulate import COLOR_MEAN, HEIGHT, PATCH, WIDTH, set_psf_model
from implicitpsf.splits import load_manifest, reserved_star_ids

R_CORE = 2.0  # encircled-energy radius (px)


def ee_within(stamps, center):
    """Flux fraction within R_CORE px of `center` for each (n, P, P) stamp."""
    yy, xx = np.mgrid[0:PATCH, 0:PATCH]
    rr = np.hypot(xx - center[0], yy - center[1])
    mask = rr <= R_CORE
    flux = stamps.reshape(len(stamps), -1).sum(1) + 1e-12
    return stamps[:, mask].sum(1) / flux


def exposure_deficit(model, data, index, reserved_ids):
    """Per-position (EE_model, EE_truth) at star-free grid points for one exposure."""
    clean, reserved = exposure_masks(data, index, reserved_ids)
    fit_mask = clean & ~reserved
    if fit_mask.sum() < 5:
        return None
    x, y = grid_positions(WIDTH, HEIGHT)
    field = {"chromatic": False, **data["true_field"][index]}
    ref_color = COLOR_MEAN if field["chromatic"] else 0.0
    truth = truth_stamps(field, x, y, ref_color)
    model_stamps = implicit_grid_stamps(model, data, index, fit_mask, x, y, ref_color)
    ee_m, ee_t = [], []
    for k in range(len(x)):
        t = truth[k]
        pos = np.clip(t, 0, None)
        tot = pos.sum() + 1e-12
        yy, xx = np.mgrid[0:PATCH, 0:PATCH]
        center = ((xx * pos).sum() / tot, (yy * pos).sum() / tot)  # truth centroid, shared
        ee_t.append(ee_within(t[None], center)[0])
        ee_m.append(ee_within(model_stamps[k][None], center)[0])
    return np.array(ee_m), np.array(ee_t)


def run(checkpoint, manifest_path, data_dir, psf_model, max_exposures):
    set_psf_model(psf_model)
    model = load_model(checkpoint)
    manifest = load_manifest(manifest_path)
    test = [
        (eid, info)
        for eid, info in sorted(manifest["exposures"].items())
        if info["split"] == "test"
    ][:max_exposures]
    ee_m, ee_t = [], []
    for exposure_id, info in test:
        data = load_exposure_file(f"{data_dir}/{info['file']}")
        reserved_ids = reserved_star_ids(manifest, exposure_id)
        out = exposure_deficit(model, data, info["index"], reserved_ids)
        if out is None:
            continue
        ee_m.append(out[0])
        ee_t.append(out[1])
    ee_m = np.concatenate(ee_m)
    ee_t = np.concatenate(ee_t)
    dee = ee_m - ee_t
    print(f"checkpoint: {checkpoint}")
    print(f"star-free grid positions: {len(ee_m)}  (psf-model={psf_model})")
    print(f"  EE@r2 model  median = {np.median(ee_m):.4f}")
    print(f"  EE@r2 truth  median = {np.median(ee_t):.4f}")
    print(f"  dEE@r2 (model-truth) median = {np.median(dee):+.4f}", end="")
    print(f"  ({100 * np.median(dee) / np.median(ee_t):+.1f}% of truth)")
    print(f"  dEE@r2 mean +/- sem = {dee.mean():+.4f} +/- {dee.std() / np.sqrt(len(dee)):.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--psf-model", default="realistic")
    parser.add_argument("--max-exposures", type=int, default=8)
    args = parser.parse_args()
    run(args.checkpoint, args.manifest, args.data_dir, args.psf_model, args.max_exposures)


if __name__ == "__main__":
    main()
