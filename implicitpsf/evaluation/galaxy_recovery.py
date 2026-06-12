"""Galaxy injection-recovery on simulated test exposures (M8 headline experiment).

Galsim-rendered Sersic galaxies (an independent rendering path) are injected at
star-free positions of sim test exposures. Each is fit with fit_galaxies through
four PSF arms built on the same fine lattice: the true Moffat ePSF (floor), the
implicit model via render_at, a per-exposure PIFF fit via get_profile, and a round
Moffat at the header FWHM (the classical fallback). Sersic n is fixed to truth.
All kernels share the stamp-sum normalization convention, so arm comparisons are
exact; the headline is re and ellipticity bias/scatter vs re/FWHM per arm.
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

from implicitpsf.baselines.catalogs import write_piff_catalog
from implicitpsf.baselines.implicit_runner import load_model
from implicitpsf.baselines.piff_runner import fit_piff
from implicitpsf.blend import sample_grid
from implicitpsf.datasets import load_exposure_file, make_batch, stable_seed
from implicitpsf.evaluation.galaxy_fit import OVERSAMPLE, fit_galaxies
from implicitpsf.evaluation.moments import PIXEL_SCALE
from implicitpsf.evaluation.run_eval import exposure_masks
from implicitpsf.render import render_at
from implicitpsf.simulate import (
    HEIGHT,
    MOFFAT_BETA,
    NOISE_SIGMA,
    PATCH,
    WIDTH,
    true_psf_params,
)
from implicitpsf.splits import load_manifest, reserved_star_ids

MARGIN = 40
STAR_CLEARANCE = 24.0  # Chebyshev pixels between an injection and any real star
RE_RANGE = (1.0, 8.0)
N_RANGE = (0.8, 4.0)
ETA_RANGE = (-0.4, 0.4)
FLUX_RANGE = (5e3, 5e4)


def starfree_positions(rng, star_x, star_y, n_wanted):
    """Uniform positions at least STAR_CLEARANCE (Chebyshev) from every star."""
    cand_x = rng.uniform(MARGIN, WIDTH - MARGIN, 600)
    cand_y = rng.uniform(MARGIN, HEIGHT - MARGIN, 600)
    cheb = np.maximum(
        np.abs(cand_x[:, None] - star_x[None, :]), np.abs(cand_y[:, None] - star_y[None, :])
    ).min(axis=1)
    keep = np.nonzero(cheb > STAR_CLEARANCE)[0][:n_wanted]
    if len(keep) < n_wanted:
        raise RuntimeError(f"only {len(keep)} star-free positions found")
    return cand_x[keep], cand_y[keep]


def sample_galaxies(rng, n_gal):
    log_re = rng.uniform(np.log(RE_RANGE[0]), np.log(RE_RANGE[1]), n_gal)
    return {
        "re": np.exp(log_re),
        "n": rng.uniform(*N_RANGE, n_gal),
        "eta1": rng.uniform(*ETA_RANGE, n_gal),
        "eta2": rng.uniform(*ETA_RANGE, n_gal),
        "flux": np.exp(rng.uniform(np.log(FLUX_RANGE[0]), np.log(FLUX_RANGE[1]), n_gal)),
    }


def inject_stamp(rng, field, x, y, gal):
    """Noisy galaxy stamp rendered entirely by galsim (independent of our fitter)."""
    fwhm, g1, g2 = true_psf_params(field, float(x), float(y), 0.0)
    psf = galsim.Moffat(beta=MOFFAT_BETA, fwhm=fwhm * PIXEL_SCALE).shear(g1=g1, g2=g2)
    q = np.exp(-np.hypot(gal["eta1"], gal["eta2"]))
    beta = 0.5 * np.arctan2(gal["eta2"], gal["eta1"])
    profile = galsim.Sersic(
        n=float(gal["n"]), half_light_radius=float(gal["re"]) * PIXEL_SCALE, flux=float(gal["flux"])
    ).shear(q=float(q), beta=float(beta) * galsim.radians)
    obj = galsim.Convolve(profile, psf, galsim.Pixel(PIXEL_SCALE))

    half = PATCH // 2
    corner_x = round(float(x)) - half + 1  # 1-based stamp corner
    corner_y = round(float(y)) - half + 1
    bounds = galsim.BoundsI(corner_x, corner_x + PATCH - 1, corner_y, corner_y + PATCH - 1)
    image = galsim.Image(bounds, scale=PIXEL_SCALE)
    obj.drawImage(image=image, center=galsim.PositionD(x + 1.0, y + 1.0), method="no_pixel")
    noisy = image.array + rng.normal(0, NOISE_SIGMA, image.array.shape)
    return noisy, fwhm


def lattice_kernel(profile, grid):
    """Stamp-sum-normalized fine-lattice point samples of a pixel-convolved profile."""
    vals = np.array(
        [profile.xValue(galsim.PositionD(p[0] * PIXEL_SCALE, p[1] * PIXEL_SCALE)) for p in grid]
    )
    fine = PATCH * OVERSAMPLE
    vals = vals.reshape(fine, fine)
    return vals / vals.sum() * OVERSAMPLE**2


def truth_kernels(field, x, y, grid):
    kernels = []
    for x0, y0 in zip(x, y, strict=True):
        fwhm, g1, g2 = true_psf_params(field, float(x0), float(y0), 0.0)
        profile = galsim.Convolve(
            galsim.Moffat(beta=MOFFAT_BETA, fwhm=fwhm * PIXEL_SCALE).shear(g1=g1, g2=g2),
            galsim.Pixel(PIXEL_SCALE),
        )
        kernels.append(lattice_kernel(profile, grid))
    return np.stack(kernels)


def piff_kernels(psf, x, y, grid):
    """PIFF kernels via get_profile; the returned method says whether the profile
    already includes the pixel response ('no_pixel') or not ('auto')."""
    kernels = []
    for x0, y0 in zip(x, y, strict=True):
        profile, method = psf.get_profile(x=round(float(x0)) + 1.0, y=round(float(y0)) + 1.0)
        if method == "auto":
            profile = galsim.Convolve(profile, galsim.Pixel(PIXEL_SCALE))
        kernels.append(lattice_kernel(profile, grid))
    return np.stack(kernels)


def moffat_kernels(field, n_gal, grid):
    profile = galsim.Convolve(
        galsim.Moffat(beta=MOFFAT_BETA, fwhm=field["fwhm_base"] * PIXEL_SCALE),
        galsim.Pixel(PIXEL_SCALE),
    )
    return np.tile(lattice_kernel(profile, grid), (n_gal, 1, 1))


def implicit_kernels(model, data, index, fit_mask, x, y):
    batch = dict(make_batch(data, [index]))
    batch["flux"] = batch["flux"] * torch.from_numpy(fit_mask).unsqueeze(0)
    queries = torch.from_numpy(np.column_stack([np.round(x), np.round(y)]).astype(np.float32))
    colors = torch.zeros(len(x))
    return render_at(model, batch, queries, colors, oversample=OVERSAMPLE).numpy()


def evaluate_exposure(model, data, index, reserved_ids, workdir, n_gal, free_n):
    exposure_id = data["exposure_id"][index]
    clean, reserved = exposure_masks(data, index, reserved_ids)
    fit_mask = clean & ~reserved
    star_x = data["x_pixel"][index].numpy()
    star_y = data["y_pixel"][index].numpy()
    field = {"chromatic": False, **data["true_field"][index]}

    rng = np.random.default_rng(stable_seed("galaxy_recovery", exposure_id))
    x, y = starfree_positions(rng, star_x, star_y, n_gal)
    gals = sample_galaxies(rng, n_gal)

    stamps, fwhms = [], []
    for i in range(n_gal):
        stamp, fwhm = inject_stamp(rng, field, x[i], y[i], {k: v[i] for k, v in gals.items()})
        stamps.append(stamp)
        fwhms.append(fwhm)
    stamps = torch.tensor(np.stack(stamps), dtype=torch.float32)
    fwhms = np.array(fwhms)

    grid = sample_grid(PATCH, OVERSAMPLE, torch.device("cpu"), torch.float64).numpy()
    cat_path = Path(workdir) / "piff_cat.fits"
    fit_fluxes = data["flux"][index].numpy()[fit_mask]
    write_piff_catalog(star_x[fit_mask], star_y[fit_mask], fit_fluxes, cat_path)
    piff_psf = fit_piff(data["fits_path"][index], cat_path, Path(workdir) / "model.piff")
    arms = {
        "truth": truth_kernels(field, x, y, grid),
        "implicit": implicit_kernels(model, data, index, fit_mask, x, y),
        "piff": piff_kernels(piff_psf, x, y, grid),
        "moffat_header": moffat_kernels(field, n_gal, grid),
    }

    # one batched fit across all arms: identical data, per-arm kernels
    arm_names = list(arms)
    kernels = torch.tensor(np.concatenate([arms[a] for a in arm_names]), dtype=torch.float32)
    rep = len(arm_names)
    cutouts = stamps.repeat(rep, 1, 1)
    variance = torch.full_like(cutouts, NOISE_SIGMA**2)
    valid = torch.ones_like(cutouts, dtype=torch.bool)
    if free_n:  # the fitter must not see the true index; init at a neutral value
        n_arg = torch.full((n_gal * rep,), 1.5)
    else:
        n_arg = torch.tensor(np.tile(gals["n"], rep), dtype=torch.float32)
    init_flux = cutouts.sum(dim=(-2, -1)).clamp(min=100.0)
    init_re = torch.full((n_gal * rep,), 3.0)
    result = fit_galaxies(
        cutouts, variance, valid, kernels, n_arg, init_flux, init_re, fit_n=free_n
    )

    frames = []
    for a, arm in enumerate(arm_names):
        sl = slice(a * n_gal, (a + 1) * n_gal)
        frames.append(
            pd.DataFrame(
                {
                    "exposure_id": exposure_id,
                    "arm": arm,
                    "x": x,
                    "y": y,
                    "fwhm_true": fwhms,
                    "n_true": gals["n"],
                    "flux_true": gals["flux"],
                    "re_true": gals["re"],
                    "eta1_true": gals["eta1"],
                    "eta2_true": gals["eta2"],
                    "flux_fit": result["flux"][sl].numpy(),
                    "dx_err": result["dx"][sl].numpy() - (x - np.round(x)),
                    "dy_err": result["dy"][sl].numpy() - (y - np.round(y)),
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
    """Evaluate one data file's selected test exposures (one worker task)."""
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
                frames.append(
                    evaluate_exposure(
                        model,
                        data,
                        index,
                        reserved_ids,
                        workdir,
                        args.galaxies_per_exposure,
                        args.free_n,
                    )
                )
            except Exception:
                print(f"FAILED {exposure_name}:\n{traceback.format_exc()}")
    return frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="manifests/sim_split_v1.json")
    parser.add_argument("--data-dir", default="/data/scratch/regier/sim_psf_stars")
    parser.add_argument("--checkpoint", default="checkpoints/sim_v5_long/best.pt")
    parser.add_argument("--out", default="results/galaxy_recovery_sim.parquet")
    parser.add_argument("--max-exposures", type=int, default=75)
    parser.add_argument("--galaxies-per-exposure", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument(
        "--free-n", action="store_true", help="fit the Sersic index instead of fixing to truth"
    )
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    selected = [
        (name, info)
        for name, info in sorted(manifest["exposures"].items())
        if info["split"] == "test"
    ][: args.max_exposures]
    by_file = {}
    for name, info in selected:
        by_file.setdefault(info["file"], []).append((name, info["index"]))

    tasks = [(args, file_name, exposures) for file_name, exposures in sorted(by_file.items())]
    with mp.get_context("spawn").Pool(args.num_workers) as pool:
        groups = pool.starmap(eval_file_group, tasks)

    frames = [frame for group in groups for frame in group]
    table = pd.concat(frames, ignore_index=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out)
    n_exp = table["exposure_id"].nunique()
    arms = sorted(table["arm"].unique())
    print(f"wrote {out}: {len(table)} rows, {n_exp} exposures, arms {arms}")


if __name__ == "__main__":
    main()
