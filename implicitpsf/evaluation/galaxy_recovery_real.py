"""Star-anchored galaxy injection-recovery on REAL test exposures (WS4).

Real data has no known true PSF, so we anchor on stars: a bright, strictly-isolated
reserved star's pixel image IS an empirical sample of the true effective-PSF at its CCD
location (the same object run_eval treats as ground truth on reserved stars). We inject a
galsim Sersic galaxy convolved with that star's image (an InterpolatedImage, so it
already carries the pixel response) plus the star's own noise, then fit it through PSF
arms that PREDICT the PSF at the anchor position from the OTHER (non-reserved) stars:
the anchor's own image (truth/floor), the implicit model via render_at, and PIFF. The
truth arm recovers the injected galaxy with its own kernel, so its bias must be ~0 — that
is the build-correctness gate before any cross-method comparison is trusted.

Unlike the sim version this measures interpolation/amortization error on REAL PSFs, the
exact quantity a galaxy-shape pipeline cares about. PSFEx is added once the truth-arm
null test passes.
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
from implicitpsf.blend import chebyshev_distances, sample_grid
from implicitpsf.datasets import load_exposure_file, make_batch, stable_seed
from implicitpsf.evaluation.galaxy_fit import OVERSAMPLE, fit_galaxies
from implicitpsf.evaluation.galaxy_recovery import lattice_kernel, sample_galaxies
from implicitpsf.evaluation.moments import PIXEL_SCALE
from implicitpsf.evaluation.run_eval import exposure_masks
from implicitpsf.provenance import write_result
from implicitpsf.render import render_at
from implicitpsf.simulate import PATCH
from implicitpsf.splits import load_manifest, reserved_star_ids

ANCHOR_CLEARANCE = 24.0  # Chebyshev px to any other detection (strict — anchor is truth)
ANCHOR_SNR_MIN = 200.0
ANCHOR_VALID_MIN = 0.98  # valid-pixel fraction over the stamp
AMP_MARGIN = 64.0  # px from the CCD mid-column (dead-amp B guard)
CCD_MID_X = 1024.0  # CCD 31 is 2048 wide; amp boundary at the middle column
GALS_PER_ANCHOR = 4


def select_anchors(data, index, reserved_ids):
    """Indices of reserved clean stars that pass strict anchor cuts (their image is the
    truth PSF, so isolation must be far tighter than the 16 px extraction cut)."""
    _clean, reserved = exposure_masks(data, index, reserved_ids)
    x = data["x_pixel"][index].numpy()
    y = data["y_pixel"][index].numpy()
    flux = data["flux"][index].numpy()
    snr = data["snr"][index].numpy()
    valid = data["valid_pixels"][index].numpy()  # (n_stars, patch, patch)
    valid_frac = valid.reshape(valid.shape[0], -1).mean(axis=1)

    positions = torch.from_numpy(np.column_stack([x, y])).unsqueeze(0)
    cheb = chebyshev_distances(positions)[0].numpy()  # (n_stars, n_stars)
    real = flux > 0
    np.fill_diagonal(cheb, np.inf)
    cheb[:, ~real] = np.inf  # only real detections count as neighbors
    nearest = cheb.min(axis=1)

    ok = (
        reserved
        & (snr >= ANCHOR_SNR_MIN)
        & (valid_frac >= ANCHOR_VALID_MIN)
        & (nearest >= ANCHOR_CLEARANCE)
        & (np.abs(x - CCD_MID_X) >= AMP_MARGIN)
    )
    return np.nonzero(ok)[0]


def anchor_kernel(stamp, valid):
    """Empirical-PSF galsim profile from a star stamp: background-subtracted (edge median),
    unit-normalized over valid pixels, as an InterpolatedImage carrying the pixel response."""
    stamp = stamp.astype(np.float64).copy()
    edge = np.concatenate([stamp[0], stamp[-1], stamp[:, 0], stamp[:, -1]])
    stamp = stamp - np.median(edge)
    stamp = stamp * valid  # zero masked pixels
    stamp = stamp / stamp.sum()
    image = galsim.Image(stamp, scale=PIXEL_SCALE)
    return galsim.InterpolatedImage(image, x_interpolant="lanczos15", normalization="flux")


def inject_anchor_stamp(rng, kernel, x, y, gal, noise_sigma):
    """Sersic galaxy convolved with the anchor's empirical PSF, drawn no_pixel (the
    anchor image already includes the pixel response) plus the anchor's noise."""
    q = np.exp(-np.hypot(gal["eta1"], gal["eta2"]))
    beta = 0.5 * np.arctan2(gal["eta2"], gal["eta1"])
    profile = galsim.Sersic(
        n=float(gal["n"]), half_light_radius=float(gal["re"]) * PIXEL_SCALE, flux=float(gal["flux"])
    ).shear(q=float(q), beta=float(beta) * galsim.radians)
    obj = galsim.Convolve(profile, kernel)

    half = PATCH // 2
    corner_x = round(float(x)) - half + 1  # 1-based stamp corner
    corner_y = round(float(y)) - half + 1
    bounds = galsim.BoundsI(corner_x, corner_x + PATCH - 1, corner_y, corner_y + PATCH - 1)
    image = galsim.Image(bounds, scale=PIXEL_SCALE)
    obj.drawImage(image=image, center=galsim.PositionD(x + 1.0, y + 1.0), method="no_pixel")
    return image.array + rng.normal(0, noise_sigma, image.array.shape)


def implicit_kernels(model, data, index, fit_mask, x, y):
    """Implicit-model PSF at the anchor positions, conditioned only on fit stars."""
    batch = dict(make_batch(data, [index]))
    batch["flux"] = batch["flux"] * torch.from_numpy(fit_mask).unsqueeze(0)
    queries = torch.from_numpy(np.column_stack([np.round(x), np.round(y)]).astype(np.float32))
    colors = torch.zeros(len(x))
    return render_at(model, batch, queries, colors, oversample=OVERSAMPLE).numpy()


def implicit_interp_kernels(model, data, index, fit_mask, x, y, grid):
    """The implicit ePSF rendered through the SAME native-sample -> InterpolatedImage(lanczos15) ->
    fine-lattice path as the truth/PIFF arms (oversample=1), instead of render_at's direct fine
    evaluation. Isolates whether the size deficit is an artifact of the render_at rendering path:
    same model, rendered apples-to-apples with PIFF. If this arm matches `implicit`, the path is not
    the cause; if it matches PIFF (~0), the deficit was the rendering."""
    batch = dict(make_batch(data, [index]))
    batch["flux"] = batch["flux"] * torch.from_numpy(fit_mask).unsqueeze(0)
    queries = torch.from_numpy(np.column_stack([np.round(x), np.round(y)]).astype(np.float32))
    colors = torch.zeros(len(x))
    native = render_at(model, batch, queries, colors, oversample=1).numpy()  # (n, PATCH, PATCH)
    kernels = []
    for stamp in native:
        image = galsim.Image(np.ascontiguousarray(stamp), scale=PIXEL_SCALE)
        profile = galsim.InterpolatedImage(image, x_interpolant="lanczos15", normalization="flux")
        kernels.append(lattice_kernel(profile, grid))
    return np.stack(kernels)


def piff_kernels(psf, x, y, grid):
    """PIFF kernels in the data pixel frame, resampled on the fine lattice exactly as the
    empirical truth arm. We draw each model WCS-aware (``render_piff``, identical to the
    reserved-star path) and wrap the pixel-frame stamp as an InterpolatedImage, instead of
    sampling ``psf.get_profile()`` on a bare pixel lattice: get_profile returns the PSF in
    PIFF's sky frame, which the DECam WCS transposes relative to the pixel axes, corrupting
    e1 only (e2 invariant) and biasing recovered galaxy ellipticity by ~-0.14."""
    stamps = render_piff(psf, x, y, patch_size=PATCH)  # (n, PATCH, PATCH), WCS applied
    kernels = []
    for stamp in stamps:
        image = galsim.Image(np.ascontiguousarray(stamp), scale=PIXEL_SCALE)
        profile = galsim.InterpolatedImage(image, x_interpolant="lanczos15", normalization="flux")
        kernels.append(lattice_kernel(profile, grid))
    return np.stack(kernels)


def psfex_kernels(psf_path, fits_path, x, y, grid):
    """PSFEx kernels in the data pixel frame, resampled on the fine lattice exactly as the
    truth/PIFF arms. ``render_psfex`` draws WCS-aware via galsim DES_PSFEx (``no_pixel``); we
    wrap each pixel-frame stamp as an InterpolatedImage rather than sampling on a bare lattice,
    identical to ``piff_kernels`` (so PSFEx cannot reintroduce the transpose/sampling bug)."""
    stamps = render_psfex(psf_path, fits_path, x, y, patch_size=PATCH)  # (n, PATCH, PATCH)
    kernels = []
    for stamp in stamps:
        image = galsim.Image(np.ascontiguousarray(stamp), scale=PIXEL_SCALE)
        profile = galsim.InterpolatedImage(image, x_interpolant="lanczos15", normalization="flux")
        kernels.append(lattice_kernel(profile, grid))
    return np.stack(kernels)


def truth_kernels(anchor_profiles, grid):
    """Lattice kernels from each anchor's own empirical PSF (the validation arm)."""
    return np.stack([lattice_kernel(p, grid) for p in anchor_profiles])


def evaluate_exposure(model, data, index, reserved_ids, workdir, free_n, with_psfex):
    exposure_id = data["exposure_id"][index]
    anchors = select_anchors(data, index, reserved_ids)
    if len(anchors) == 0:
        return None
    clean, reserved = exposure_masks(data, index, reserved_ids)
    fit_mask = clean & ~reserved  # PIFF/implicit see only non-reserved clean stars

    star_x = data["x_pixel"][index].numpy()
    star_y = data["y_pixel"][index].numpy()
    cutouts = data["cutouts"][index].numpy()
    valid = data["valid_pixels"][index].numpy()
    variance = data["variance"][index].numpy()

    rng = np.random.default_rng(stable_seed("galaxy_recovery_real", exposure_id))
    n = len(anchors) * GALS_PER_ANCHOR
    gals = sample_galaxies(rng, n)
    ax = np.repeat(star_x[anchors], GALS_PER_ANCHOR)
    ay = np.repeat(star_y[anchors], GALS_PER_ANCHOR)
    anchor_ids = np.repeat(data["star_id"][index].numpy()[anchors], GALS_PER_ANCHOR)

    profiles, stamps, noises = [], [], []
    for j, a in enumerate(anchors):
        kernel = anchor_kernel(cutouts[a], valid[a])
        noise_sigma = float(np.sqrt(np.median(variance[a][valid[a] > 0])))
        for k in range(GALS_PER_ANCHOR):
            i = j * GALS_PER_ANCHOR + k
            gal = {key: v[i] for key, v in gals.items()}
            stamps.append(inject_anchor_stamp(rng, kernel, ax[i], ay[i], gal, noise_sigma))
            profiles.append(kernel)
            noises.append(noise_sigma)
    stamps = torch.tensor(np.stack(stamps), dtype=torch.float32)
    noises = np.array(noises)

    grid = sample_grid(PATCH, OVERSAMPLE, torch.device("cpu"), torch.float64).numpy()
    fits_path = data["fits_path"][index]
    cat_path = Path(workdir) / "piff_cat.fits"
    write_piff_catalog(
        star_x[fit_mask], star_y[fit_mask], data["flux"][index].numpy()[fit_mask], cat_path
    )
    piff_psf = fit_piff(fits_path, cat_path, Path(workdir) / "model.piff")

    arms = {
        "truth": truth_kernels(profiles, grid),
        "implicit": implicit_kernels(model, data, index, fit_mask, ax, ay),
        "implicit_interp": implicit_interp_kernels(model, data, index, fit_mask, ax, ay, grid),
        "piff": piff_kernels(piff_psf, ax, ay, grid),
    }
    if with_psfex:  # PSFEx fit + DES_PSFEx render is ~10 min/exposure; opt in for the final table
        with fits.open(fits_path) as hdul:
            image_header = hdul["SCI"].header
        ldac_path = Path(workdir) / "psfex_cat.fits"
        write_psfex_ldac(
            cutouts[fit_mask],
            valid[fit_mask],
            star_x[fit_mask],
            star_y[fit_mask],
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
        psfex_path = fit_psfex(ldac_path, Path(workdir) / "psfex_out")
        arms["psfex"] = psfex_kernels(psfex_path, fits_path, ax, ay, grid)

    arm_names = list(arms)
    kernels = torch.tensor(np.concatenate([arms[a] for a in arm_names]), dtype=torch.float32)
    rep = len(arm_names)
    cut = stamps.repeat(rep, 1, 1)
    var = torch.tensor(np.tile(noises**2, rep), dtype=torch.float32)[:, None, None]
    var = var.expand_as(cut).contiguous()
    valid_fit = torch.ones_like(cut, dtype=torch.bool)
    n_arg = torch.full((n * rep,), 1.5) if free_n else torch.tensor(np.tile(gals["n"], rep))
    init_flux = cut.sum(dim=(-2, -1)).clamp(min=100.0)
    init_re = torch.full((n * rep,), 3.0)
    result = fit_galaxies(cut, var, valid_fit, kernels, n_arg, init_flux, init_re, fit_n=free_n)

    frames = []
    for a, arm in enumerate(arm_names):
        sl = slice(a * n, (a + 1) * n)
        frames.append(
            pd.DataFrame(
                {
                    "exposure_id": exposure_id,
                    "band": data["band"][index],
                    "arm": arm,
                    "anchor_star_id": anchor_ids,
                    "x": ax,
                    "y": ay,
                    "n_true": gals["n"],
                    "flux_true": gals["flux"],
                    "re_true": gals["re"],
                    "eta1_true": gals["eta1"],
                    "eta2_true": gals["eta2"],
                    "flux_fit": result["flux"][sl].numpy(),
                    "re_fit": result["re"][sl].numpy(),
                    "n_fit": result["n"][sl].numpy(),
                    "eta1_fit": result["eta1"][sl].numpy(),
                    "eta2_fit": result["eta2"][sl].numpy(),
                    "chi2": result["chi2"][sl].numpy(),
                    "n_fit_stars": int(fit_mask.sum()),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def eval_file_group(args, file_name, exposures):
    torch.set_num_threads(1)
    manifest = load_manifest(args.manifest)
    model = load_model(args.checkpoint)
    data = load_exposure_file(Path(args.data_dir) / file_name)

    frames = []
    for exposure_name, index in exposures:
        exposure_id = data["exposure_id"][index]
        reserved_ids = reserved_star_ids(manifest, exposure_id)
        with tempfile.TemporaryDirectory() as workdir:
            try:
                frame = evaluate_exposure(
                    model, data, index, reserved_ids, workdir, args.free_n, args.with_psfex
                )
                if frame is not None:
                    frames.append(frame)
            except Exception:
                print(f"FAILED {exposure_name}:\n{traceback.format_exc()}")
    return frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="manifests/split_v1.json")
    parser.add_argument("--data-dir", default="/data/scratch/regier/sep_des_stars_v2")
    parser.add_argument("--checkpoint", default="checkpoints/real_v6_blend/best.pt")
    parser.add_argument("--out", default="results/galaxy_recovery_real.parquet")
    parser.add_argument("--split", default="test", choices=["test", "val"])
    parser.add_argument("--band", default=None)
    parser.add_argument("--max-exposures", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--free-n", action="store_true")
    parser.add_argument(
        "--with-psfex",
        action="store_true",
        help="add the PSFEx arm (4th method); ~10 min/exposure, so use few exposures",
    )
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    selected = [
        (name, info)
        for name, info in sorted(manifest["exposures"].items())
        if info["split"] == args.split and (args.band is None or info.get("band") == args.band)
    ]
    if args.max_exposures is not None:
        selected = selected[: args.max_exposures]
    by_file = {}
    for name, info in selected:
        by_file.setdefault(info["file"], []).append((name, info["index"]))

    tasks = [(args, file_name, exposures) for file_name, exposures in sorted(by_file.items())]
    if args.num_workers <= 1:
        groups = [eval_file_group(*task) for task in tasks]  # serial: no pool, always works
    else:
        # fork pool, as in run_eval/sim_truth (a 'spawn' pool re-imports this module per
        # worker, which was both slow and fragile to third-party import behavior)
        with mp.Pool(args.num_workers) as pool:
            groups = pool.starmap(eval_file_group, tasks)

    frames = [frame for group in groups for frame in group]
    table = pd.concat(frames, ignore_index=True)
    write_result(
        table, args.out, checkpoint=args.checkpoint, source="galaxy_recovery_real", purpose=args.out
    )
    n_exp = table["exposure_id"].nunique()
    print(
        f"wrote {args.out}: {len(table)} rows, {n_exp} exposures, arms {sorted(table.arm.unique())}"
    )


if __name__ == "__main__":
    main()
