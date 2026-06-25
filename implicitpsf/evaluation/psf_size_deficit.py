"""Direct PSF-size deficit: HSM adaptive size (T = 2 sigma^2) of the model PSF vs the empirical
star, on BRIGHT clean reserved stars. This is the DIRECT instrument for the under-concentration
(Jeff): galaxy-size recovery is an indirect symptom, and held-out RMSE/chi2 is dominated by
near-noise pixels so it barely constrains the FWHM -- a model can win on RMSE yet carry a size
bias. We focus on BRIGHT stars: their size is measurable above noise (the encircled_energy stack
is noise-limited on typical stars) AND they are the least fractionally contaminated, so they are
the cleanest available proxy for the true PSF.

delta-T = T_model - T_star (as % of T_star); positive => model PSF too BIG (under-concentrated).
The model is rendered at each star's OWN position/color/flux (BF- and conditioning-matched). A
flux sweep (--flux-sweep) renders instead at fixed query fluxes to trace the clean window: the
model PSF sharpens with flux as contamination fades, until brighter-fatter widens it again.
On sim (--psf-model), the truth PSF is the reference instead of the noisy star.
"""

import argparse
import glob

import numpy as np
import torch

from implicitpsf.baselines.implicit_runner import load_model
from implicitpsf.datasets import load_exposure_file, make_batch
from implicitpsf.evaluation.moments import hsm_moments
from implicitpsf.render import render_at
from implicitpsf.splits import load_manifest, reserved_star_ids


def bright_clean_reserved(data, index, reserved_ids, snr_min):
    """Clean reserved stars with snr >= snr_min (bright + held-out + least contaminated)."""
    st = data["star_type"][index].numpy()
    snr = data["snr"][index].numpy()
    sid = data["star_id"][index].numpy()
    reserved = np.isin(sid, list(reserved_ids))
    return np.nonzero((st == 0) & (snr >= snr_min) & reserved)[0]


def star_stamps(data, index, idx):
    """Background-subtracted empirical star stamps + their valid-pixel masks."""
    cut = data["cutouts"][index].numpy()
    val = data["valid_pixels"][index].numpy()
    stamps, masks = [], []
    for j in idx:
        s = cut[j].astype(np.float64)
        edge = np.concatenate([s[0], s[-1], s[:, 0], s[:, -1]])
        stamps.append(s - np.median(edge))
        masks.append(val[j] > 0)
    return np.array(stamps), np.array(masks)


def model_psf(model, data, index, idx, flux_override):
    """Render the model PSF at the stars' positions/colors; flux = own (None) or a fixed value."""
    batch = dict(make_batch(data, [index]))
    x = data["x_pixel"][index].numpy()[idx]
    y = data["y_pixel"][index].numpy()[idx]
    q = torch.tensor(np.column_stack([np.round(x), np.round(y)]), dtype=torch.float32)
    colors = data["color"][index][idx].float()
    if flux_override is None:  # match each star's OWN flux (BF + cleanliness matched)
        qf = data["flux"][index][idx].float()
    elif flux_override == "median":  # what galaxy_recovery actually uses (render_at default)
        qf = None
    else:
        qf = torch.full((len(idx),), float(flux_override))
    return render_at(model, batch, q, colors, query_fluxes=qf, oversample=1).numpy()


def measure(model, data, indices, manifest, eid_of, snr_min, flux_override):
    """delta-T (model - star) over bright clean reserved stars, as fraction of T_star."""
    dt_frac, t_star_all = [], []
    for index in indices:
        eid = eid_of[index]
        idx = bright_clean_reserved(data, index, reserved_star_ids(manifest, eid), snr_min)
        if idx.size < 3:
            continue
        stamps, masks = star_stamps(data, index, idx)
        t_star = hsm_moments(stamps, valid_pixels=masks)["T"]
        t_model = hsm_moments(model_psf(model, data, index, idx, flux_override))["T"]
        ok = np.isfinite(t_star) & np.isfinite(t_model) & (t_star > 0)
        dt_frac.append((t_model[ok] - t_star[ok]) / t_star[ok])
        t_star_all.append(t_star[ok])
    if not dt_frac:
        return np.array([]), np.array([])
    return np.concatenate(dt_frac), np.concatenate(t_star_all)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-dir", default="/data/scratch/regier/sep_des_stars_v2")
    parser.add_argument("--split", default="test")
    parser.add_argument("--snr-min", type=float, default=200.0)
    parser.add_argument("--max-exposures", type=int, default=20)
    parser.add_argument("--flux-sweep", action="store_true", help="render at fixed fluxes vs own")
    parser.add_argument("--sweep-fluxes", nargs="+", type=float, default=None,
                        help="explicit fixed query fluxes to sweep (e.g. find the BF turn-up)")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    model = load_model(args.checkpoint, device="cuda")  # else renders on CPU (slow)
    files = sorted(glob.glob(f"{args.data_dir}/*.pt"))
    want = {e: i for e, i in manifest["exposures"].items() if i["split"] == args.split}
    by_file = {}
    for eid, info in want.items():
        by_file.setdefault(info["file"], []).append((info["index"], eid))

    if args.sweep_fluxes:
        fluxes = [None, "median", *args.sweep_fluxes]
    elif args.flux_sweep:
        fluxes = [None, "median", 1e4, 1e5, 1e6]
    else:
        fluxes = [None, "median"]
    n_exp = 0
    per_flux = {f: ([], []) for f in fluxes}
    for path in files:
        name = path.split("/")[-1]
        if name not in by_file or n_exp >= args.max_exposures:
            continue
        data = load_exposure_file(path)
        idx_eid = by_file[name][: args.max_exposures - n_exp]
        eid_of = {i: e for i, e in idx_eid}
        indices = [i for i, _ in idx_eid]
        for f in fluxes:
            dt, ts = measure(model, data, indices, manifest, eid_of, args.snr_min, f)
            per_flux[f][0].append(dt)
            per_flux[f][1].append(ts)
        n_exp += len(indices)

    print(f"DIRECT PSF-size deficit (HSM T), bright clean reserved stars snr>={args.snr_min:.0f}, "
          f"{args.split} split, {n_exp} exposures:")
    print(f"{'render flux':>14} {'median dT%':>11} {'mean dT% +/- sem':>20}  n")
    for f in fluxes:
        dt = np.concatenate(per_flux[f][0])
        label = {None: "own (matched)", "median": "median (galrec)"}.get(f, f"{f}")
        sem = dt.std() / np.sqrt(len(dt))
        print(f"{label:>14} {100*np.median(dt):>+10.2f}% {100*dt.mean():>+9.2f} +/- {100*sem:>4.2f}"
              f"   {len(dt)}")
    print("dT%>0 => model PSF too BIG (under-concentrated) vs the bright clean star")


if __name__ == "__main__":
    main()
