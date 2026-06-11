"""Compare PSF models against the known simulated truth at star-free positions.

For each simulated test exposure, every method fits using the same non-reserved
stars, then renders the PSF on a uniform grid of star-free positions. The truth
stamp at each grid point comes from the stored field polynomials, so this measures
true field error everywhere — the claim reserved stars can only sample.
"""

import argparse
import multiprocessing as mp
import tempfile
import traceback
from pathlib import Path

import galsim
import numpy as np
import pandas as pd
import torch
from astropy.io import fits

from implicitpsf.baselines.catalogs import write_piff_catalog, write_psfex_ldac
from implicitpsf.baselines.implicit_runner import load_model
from implicitpsf.baselines.piff_runner import fit_piff, render_piff
from implicitpsf.baselines.psfex_runner import fit_psfex, render_psfex
from implicitpsf.datasets import load_exposure_file, make_batch
from implicitpsf.evaluation.moments import PIXEL_SCALE, hsm_moments
from implicitpsf.evaluation.run_eval import exposure_masks
from implicitpsf.render import render_at
from implicitpsf.simulate import COLOR_MEAN, HEIGHT, MOFFAT_BETA, PATCH, WIDTH, true_psf_params
from implicitpsf.splits import load_manifest, reserved_star_ids


def grid_positions(width, height, nx=6, ny=12, margin=40):
    x = np.linspace(margin, width - margin, nx)
    y = np.linspace(margin, height - margin, ny)
    grid_x, grid_y = np.meshgrid(x, y)
    return grid_x.ravel(), grid_y.ravel()


def truth_stamps(field, x, y, color):
    """Noiseless pixel-convolved truth stamps on the data stamp grid.

    Renders onto explicit bounds matching the corner = round(center) - half
    convention: galsim's automatic bounds for even stamps with center= can land
    one pixel off depending on the fractional part of the position.
    """
    half = PATCH // 2
    stamps = np.zeros((len(x), PATCH, PATCH))
    for index, (x0, y0) in enumerate(zip(x, y, strict=True)):
        fwhm, g1, g2 = true_psf_params(field, float(x0), float(y0), color)
        profile = galsim.Moffat(beta=MOFFAT_BETA, fwhm=fwhm * PIXEL_SCALE)
        profile = profile.shear(g1=g1, g2=g2)
        corner_x = round(float(x0)) - half + 1  # 1-based stamp corner
        corner_y = round(float(y0)) - half + 1
        bounds = galsim.BoundsI(corner_x, corner_x + PATCH - 1, corner_y, corner_y + PATCH - 1)
        image = galsim.Image(bounds, scale=PIXEL_SCALE)
        profile.drawImage(
            image=image,
            center=galsim.PositionD(float(x0) + 1.0, float(y0) + 1.0),
            add_to_image=True,
        )
        stamps[index] = image.array
    return stamps


def implicit_grid_stamps(model, data, index, fit_mask, x, y, ref_color):
    batch = make_batch(data, [index])
    # restrict context to exactly the stars the baselines fit on
    batch = dict(batch)
    keep = torch.from_numpy(fit_mask)
    batch["flux"] = batch["flux"] * keep.unsqueeze(0)
    queries = torch.from_numpy(np.column_stack([x, y])).float()
    colors = torch.full((len(x),), ref_color)
    return render_at(model, batch, queries, colors).numpy()


def evaluate_exposure(model, data, index, reserved_ids, workdir):
    clean, reserved = exposure_masks(data, index, reserved_ids)
    fit_mask = clean & ~reserved

    x, y = grid_positions(WIDTH, HEIGHT)
    field = {"chromatic": False, **data["true_field"][index]}  # migrate pre-chromatic files
    # grid truth is evaluated at a reference color: the training-data mean for
    # chromatic fields, 0 for achromatic ones (whose stars carry color 0)
    ref_color = COLOR_MEAN if field["chromatic"] else 0.0
    truth = truth_stamps(field, x, y, ref_color)
    truth_moments = hsm_moments(truth)

    cat_x = data["x_pixel"][index].numpy()
    workdir = Path(workdir)
    renderers = {
        "implicit": lambda: implicit_grid_stamps(model, data, index, fit_mask, x, y, ref_color)
    }

    def piff_grid():
        cat_path = workdir / "piff_cat.fits"
        write_piff_catalog(
            cat_x[fit_mask],
            data["y_pixel"][index].numpy()[fit_mask],
            data["flux"][index].numpy()[fit_mask],
            cat_path,
        )
        psf = fit_piff(data["fits_path"][index], cat_path, workdir / "model.piff")
        return render_piff(psf, x, y)

    def psfex_grid():
        with fits.open(data["fits_path"][index]) as hdul:
            image_header = hdul["SCI"].header
        ldac_path = workdir / "psfex_cat.fits"
        write_psfex_ldac(
            data["cutouts"][index].numpy()[fit_mask],
            data["valid_pixels"][index].numpy()[fit_mask],
            cat_x[fit_mask],
            data["y_pixel"][index].numpy()[fit_mask],
            data["flux"][index].numpy()[fit_mask],
            data["flux_err"][index].numpy()[fit_mask],
            data["snr"][index].numpy()[fit_mask],
            float(data["fwhm"][index]),
            image_header,
            ldac_path,
            background_dev=float(data["skysigma"][index]),
        )
        psf_path = fit_psfex(ldac_path, workdir / "psfex_out")
        return render_psfex(psf_path, data["fits_path"][index], x, y)

    renderers["piff"] = piff_grid
    renderers["psfex"] = psfex_grid

    frames = []
    for method, renderer in renderers.items():
        moments = hsm_moments(renderer())
        frames.append(
            pd.DataFrame(
                {
                    "exposure_id": data["exposure_id"][index],
                    "method": method,
                    "x": x,
                    "y": y,
                    "T_true": truth_moments["T"],
                    "e1_true": truth_moments["e1"],
                    "e2_true": truth_moments["e2"],
                    "T_model": moments["T"],
                    "e1_model": moments["e1"],
                    "e2_model": moments["e2"],
                    "flag_true": truth_moments["flag"],
                    "flag_model": moments["flag"],
                    "n_fit_stars": int(fit_mask.sum()),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def eval_file_group(args, file_name, exposures):
    """Evaluate one data file's test exposures (one worker task)."""
    torch.set_num_threads(1)
    manifest = load_manifest(args.manifest)
    model = load_model(args.checkpoint)
    data = load_exposure_file(Path(args.data_dir) / file_name)

    frames, n_failed = [], 0
    for exposure_id, index in exposures:
        try:
            with tempfile.TemporaryDirectory() as tmp:
                frame = evaluate_exposure(
                    model, data, index, reserved_star_ids(manifest, exposure_id), tmp
                )
        except Exception:
            n_failed += 1
            print(f"FAILED {exposure_id}\n{traceback.format_exc()}")
            continue
        frames.append(frame)
    return frames, n_failed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="manifests/sim_split_v1.json")
    parser.add_argument("--data-dir", default="/data/scratch/regier/sim_psf_stars")
    parser.add_argument("--checkpoint", default="checkpoints/sim_run/best.pt")
    parser.add_argument("--out", default="results/sim_truth.parquet")
    parser.add_argument("--max-exposures", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    selected = [
        (exposure_id, info)
        for exposure_id, info in sorted(manifest["exposures"].items())
        if info["split"] == "test"
    ]
    if args.max_exposures is not None:
        selected = selected[: args.max_exposures]
    print(f"{len(selected)} sim test exposures")

    groups = {}
    for exposure_id, info in selected:
        groups.setdefault(info["file"], []).append((exposure_id, info["index"]))
    tasks = [(args, file_name, exposures) for file_name, exposures in sorted(groups.items())]
    with mp.Pool(args.num_workers) as pool:
        outputs = pool.starmap(eval_file_group, tasks)
    results = [frame for frames, _ in outputs for frame in frames]
    n_failed = sum(failed for _, failed in outputs)

    if not results:
        raise RuntimeError("no exposures evaluated")
    table = pd.concat(results, ignore_index=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out_path)
    ok = (table["flag_model"] == 0) & (table["flag_true"] == 0)
    summary = (
        table[ok]
        .assign(dT_frac=lambda t: (t.T_model - t.T_true) / t.T_true)
        .groupby("method")["dT_frac"]
        .agg(["median", "std"])
    )
    print(summary)
    print(f"wrote {len(table)} rows ({n_failed} failed) -> {out_path}")


if __name__ == "__main__":
    main()
