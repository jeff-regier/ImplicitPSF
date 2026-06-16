"""Evaluate ImplicitPSF, PIFF, and PSFEx on reserved stars of the frozen test split.

For every test exposure: all methods fit/attend using the identical non-reserved
clean stars, all are scored on the identical reserved stars, on the same data pixel
grids, with the same HSM measurement and chi^2 code. Output is one tidy parquet with
a row per (exposure, reserved star, method).
"""

import argparse
import multiprocessing as mp
import tempfile
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from astropy.io import fits

from implicitpsf.baselines.catalogs import write_piff_catalog, write_psfex_ldac
from implicitpsf.baselines.implicit_runner import load_model, render_implicit
from implicitpsf.baselines.piff_runner import fit_piff, render_piff
from implicitpsf.baselines.psfex_runner import fit_psfex, render_psfex
from implicitpsf.datasets import load_exposure_file, make_batch, stable_seed
from implicitpsf.evaluation.chi2 import reduced_chi2
from implicitpsf.evaluation.moments import hsm_moments
from implicitpsf.splits import load_manifest, reserved_star_ids

TYPE_CLEAN = 0


def exposure_masks(data, index, reserved_ids):
    """Boolean star masks (numpy) for one exposure of a loaded file."""
    flux = data["flux"][index].numpy()
    star_type = data["star_type"][index].numpy()
    star_id = data["star_id"][index].numpy()
    clean = (star_type == TYPE_CLEAN) & (flux > 0)
    reserved = clean & np.isin(star_id, list(reserved_ids))
    return clean, reserved


def piff_stamps(data, index, fit_mask, reserved_mask, workdir):
    x = data["x_pixel"][index].numpy()
    y = data["y_pixel"][index].numpy()
    cat_path = workdir / "piff_cat.fits"
    write_piff_catalog(x[fit_mask], y[fit_mask], data["flux"][index].numpy()[fit_mask], cat_path)
    psf = fit_piff(data["fits_path"][index], cat_path, workdir / "model.piff")
    return render_piff(psf, x[reserved_mask], y[reserved_mask])


def psfex_stamps(data, index, fit_mask, reserved_mask, workdir):
    fits_path = data["fits_path"][index]
    with fits.open(fits_path) as hdul:
        image_header = hdul["SCI"].header

    ldac_path = workdir / "psfex_cat.fits"
    write_psfex_ldac(
        data["cutouts"][index].numpy()[fit_mask],
        data["valid_pixels"][index].numpy()[fit_mask],
        data["x_pixel"][index].numpy()[fit_mask],
        data["y_pixel"][index].numpy()[fit_mask],
        data["flux"][index].numpy()[fit_mask],
        data["flux_err"][index].numpy()[fit_mask],
        data["snr"][index].numpy()[fit_mask],
        float(data["fwhm"][index]),
        image_header,
        ldac_path,
        gain=float(data["gain"][index]),
        background_dev=float(data["skysigma"][index]),
        saturation=float(image_header["SATURATE"]),
    )
    psf_path = fit_psfex(ldac_path, workdir / "psfex_out")
    x = data["x_pixel"][index].numpy()
    y = data["y_pixel"][index].numpy()
    return render_psfex(psf_path, fits_path, x[reserved_mask], y[reserved_mask])


def implicit_stamps(model, data, index, reserved_mask, zero_color=False, context_mask=None):
    batch = make_batch(data, [index])
    if zero_color:  # models trained with --zero-color must be queried the same way
        batch["colors"] = torch.zeros_like(batch["colors"])
    if context_mask is not None:
        # strict same-information mode: attend only to the given fit stars
        batch["flux"] = batch["flux"] * torch.from_numpy(context_mask).unsqueeze(0)
    reserved = torch.from_numpy(reserved_mask).unsqueeze(0)
    stamps = render_implicit(model, batch, reserved)
    return stamps[0][torch.from_numpy(reserved_mask)].numpy()


def star_rows(data, index, reserved_mask):
    """Per-reserved-star metadata and data-stamp moments shared by all methods."""
    stamps = data["cutouts"][index].numpy()[reserved_mask]
    valid = data["valid_pixels"][index].numpy()[reserved_mask]
    moments = hsm_moments(stamps, valid_pixels=valid)
    return pd.DataFrame(
        {
            "exposure_id": data["exposure_id"][index],
            "band": data["band"][index],
            "night": data["night"][index],
            "star_id": data["star_id"][index].numpy()[reserved_mask],
            "x_pixel": data["x_pixel"][index].numpy()[reserved_mask],
            "y_pixel": data["y_pixel"][index].numpy()[reserved_mask],
            "flux": data["flux"][index].numpy()[reserved_mask],
            "snr": data["snr"][index].numpy()[reserved_mask],
            "color": data["color"][index].numpy()[reserved_mask],
            "T_star": moments["T"],
            "e1_star": moments["e1"],
            "e2_star": moments["e2"],
            "cx_star": moments["centroid_x"],
            "cy_star": moments["centroid_y"],
            "flag_star": moments["flag"],
            "valid_frac": valid.reshape(len(stamps), -1).mean(axis=1),
        }
    )


def method_columns(model_stamps, data, index, reserved_mask):
    """Model moments and chi^2 against the reserved data stamps.

    Model moments use the star's valid-pixel mask so both sides of every
    star-minus-model difference are measured over identical pixels.
    """
    observed = data["cutouts"][index].numpy()[reserved_mask]
    variance = data["variance"][index].numpy()[reserved_mask]
    valid = data["valid_pixels"][index].numpy()[reserved_mask]
    moments = hsm_moments(model_stamps, valid_pixels=valid)
    fit = reduced_chi2(observed, model_stamps, variance, valid)
    return {
        "T_model": moments["T"],
        "e1_model": moments["e1"],
        "e2_model": moments["e2"],
        "cx_model": moments["centroid_x"],
        "cy_model": moments["centroid_y"],
        "flag_model": moments["flag"],
        "chi2": fit["chi2"],
        "amplitude": fit["amplitude"],
    }


def evaluate_exposure(
    model, data, index, reserved_ids, methods, workdir, zero_color=False, fit_subset=None
):
    clean, reserved = exposure_masks(data, index, reserved_ids)
    fit_mask = clean & ~reserved
    if reserved.sum() < 5 or fit_mask.sum() < 20:
        return None

    context_mask = None
    if fit_subset is not None:
        # same k fit stars for every method (sample-efficiency sweep)
        candidates = np.flatnonzero(fit_mask)
        rng = np.random.default_rng(stable_seed("fit-subset", data["exposure_id"][index]))
        keep = rng.choice(candidates, size=min(fit_subset, len(candidates)), replace=False)
        fit_mask = np.zeros_like(fit_mask)
        fit_mask[keep] = True
        context_mask = fit_mask

    base = star_rows(data, index, reserved)
    renderers = {
        "implicit": lambda: implicit_stamps(model, data, index, reserved, zero_color, context_mask),
        "piff": lambda: piff_stamps(data, index, fit_mask, reserved, workdir),
        "psfex": lambda: psfex_stamps(data, index, fit_mask, reserved, workdir),
    }
    frames = []
    for method in methods:
        stamps = renderers[method]()
        frame = base.copy()
        frame["method"] = method
        for column, values in method_columns(stamps, data, index, reserved).items():
            frame[column] = values
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="manifests/split_v1.json")
    parser.add_argument("--data-dir", default="/data/scratch/regier/sep_des_stars_v2")
    parser.add_argument("--checkpoint", default="checkpoints/run/best.pt")
    parser.add_argument("--out", default="results/test_eval.parquet")
    parser.add_argument("--split", default="test", choices=["test", "val"])
    parser.add_argument("--band", default=None)
    parser.add_argument("--max-exposures", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--zero-color",
        action="store_true",
        help="query the implicit model with zeroed colors (ablation)",
    )
    parser.add_argument(
        "--fit-subset",
        type=int,
        default=None,
        help="restrict every method to k randomly chosen fit stars",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["implicit", "piff", "psfex"],
        choices=["implicit", "piff", "psfex"],
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    return parser.parse_args()


def eval_file_group(args, file_name, exposures):
    """Evaluate one data file's exposures (one worker task); returns (frames, n_failed)."""
    torch.set_num_threads(1)  # workers parallelize across exposures, not BLAS threads
    manifest = load_manifest(args.manifest)
    model = load_model(args.checkpoint, device=args.device) if "implicit" in args.methods else None
    data = load_exposure_file(Path(args.data_dir) / file_name)

    frames, n_failed = [], 0
    for exposure_id, index in exposures:
        reserved_ids = reserved_star_ids(manifest, exposure_id)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                frame = evaluate_exposure(
                    model,
                    data,
                    index,
                    reserved_ids,
                    args.methods,
                    Path(tmp),
                    zero_color=args.zero_color,
                    fit_subset=args.fit_subset,
                )
        except Exception:
            n_failed += 1
            print(f"FAILED {exposure_id}\n{traceback.format_exc()}")
            continue
        if frame is not None:
            frames.append(frame)
    return frames, n_failed


def main():
    args = parse_args()
    manifest = load_manifest(args.manifest)

    selected = [
        (exposure_id, info)
        for exposure_id, info in sorted(manifest["exposures"].items())
        if info["split"] == args.split and (args.band is None or info["band"] == args.band)
    ]
    if args.max_exposures is not None:
        selected = selected[: args.max_exposures]
    print(f"{len(selected)} {args.split} exposures, methods: {args.methods}")

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
    print(f"wrote {len(table)} rows for {table['exposure_id'].nunique()} exposures")
    print(f"({n_failed} exposures failed) -> {out_path}")


if __name__ == "__main__":
    main()
