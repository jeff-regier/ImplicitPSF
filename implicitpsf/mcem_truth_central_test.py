"""Decisive diagnostic: is the MCEM sampler unbiased GIVEN the true PSF as the central template?

The EM over-corrects (delta-EE plateaus over-sharp). Two candidate causes: (a) the sampler/prior
over-cleans intrinsically, or (b) the EM central-mismatch dynamics overshoot (the loop never has the
true PSF as the central). SBC says the sampler is calibrated on synthetic stamps whose central IS
the truth; this checks the same on REAL sim stars. We render the TRUE PSF at each clean star's
position (same corner convention as the cutout), use it as the cleaning central, run the batched
Gibbs, and compare stacked EE@r2 of {contaminated cutout, cleaned cutout, true PSF} at the centroid.

  cleaned EE@r2 ~ truth => sampler UNBIASED given the central (with --model-central this caught the
  sub-pixel-centering over-cleaning bug); cleaned > truth => over-cleans (sampler/prior/centering).
"""

import argparse

import numpy as np

from implicitpsf.add_gibbs_cleaned import central_psf_stamps
from implicitpsf.baselines.implicit_runner import load_model
from implicitpsf.contam_model import cell_centers
from implicitpsf.datasets import load_exposure_file
from implicitpsf.evaluation.run_eval import exposure_masks
from implicitpsf.evaluation.sim_truth import truth_stamps
from implicitpsf.mcem_sampler import cov_columns, psf_covariance, run_batch_gpu
from implicitpsf.simulate import COLOR_MEAN, PATCH, set_psf_model
from implicitpsf.splits import load_manifest, reserved_star_ids

R_CORE = 2.0


def ee_at(stamp, center):
    """Flux fraction within R_CORE px of center for one (P,P) stamp (clipped, normalized)."""
    yy, xx = np.mgrid[0:PATCH, 0:PATCH]
    mask = np.hypot(xx - center[0], yy - center[1]) <= R_CORE
    s = stamp.reshape(PATCH, PATCH)
    return s[mask].sum() / (s.sum() + 1e-12)


def exposure_test(data, index, reserved_ids, prior, n_sweeps, rng, device, model=None):
    """Clean the exposure's clean stars; central = TRUE PSF, or the NN model's PSF if `model` given
    (what the EM actually uses). Return per-star (contaminated, cleaned, truth) EE@r2 triples."""
    clean, reserved = exposure_masks(data, index, reserved_ids)
    idx = np.nonzero(clean & ~reserved)[0]
    if len(idx) < 5:
        return None
    x = data["x_pixel"][index].numpy()[idx]
    y = data["y_pixel"][index].numpy()[idx]
    field = {"chromatic": False, **data["true_field"][index]}
    ref_color = COLOR_MEAN if field["chromatic"] else 0.0
    truecen = truth_stamps(field, x, y, ref_color).reshape(len(idx), -1)  # true PSF, star frame
    raw = central_psf_stamps(model, data, index, idx) if model is not None else truecen
    cg = np.clip(raw.reshape(len(idx), -1), 0, None)
    cg = cg / (cg.sum(1, keepdims=True) + 1e-12)
    cut = data["cutouts"][index].numpy()[idx].reshape(len(idx), -1)
    var = data["variance"][index].numpy()[idx].reshape(len(idx), -1)
    val = data["valid_pixels"][index].numpy()[idx].reshape(len(idx), -1)
    w = (val > 0) / np.clip(var, 1e-6, None)
    sigma_psf = psf_covariance(cg.mean(0), PATCH)
    cols = cov_columns(cell_centers(PATCH, 16, 2.5), sigma_psf, PATCH)
    samples, _, _ = run_batch_gpu(cut, w, cg, cols, prior, n_sweeps, rng, n_keep=4, device=device)
    cleaned = (cut[:, None, :] - samples).mean(1)  # posterior-mean cleaned, for the EE comparison
    rows = []
    yy, xx = np.mgrid[0:PATCH, 0:PATCH]
    for j in range(len(idx)):
        t = np.clip(truecen[j], 0, None).reshape(PATCH, PATCH)
        tot = t.sum() + 1e-12
        c = ((xx * t).sum() / tot, (yy * t).sum() / tot)  # truth centroid, shared center
        rows.append((ee_at(cut[j], c), ee_at(cleaned[j], c), ee_at(truecen[j], c)))
    return np.array(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="manifests/sim_contamreal_sub118.json")
    parser.add_argument("--data-dir", default="/data/scratch/regier/sim_contamreal_stars")
    parser.add_argument("--psf-model", default="realistic")
    parser.add_argument("--max-exposures", type=int, default=8)
    parser.add_argument("--n-sweeps", type=int, default=40)
    parser.add_argument("--prior-lam", type=float, default=1.0)
    parser.add_argument("--prior-alpha", type=float, default=1.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-central", default=None, help="NN ckpt for central (else truth)")
    args = parser.parse_args()
    set_psf_model(args.psf_model)
    prior = {"lam": args.prior_lam, "flux_lo": 100.0, "flux_hi": 2000.0, "alpha": args.prior_alpha}
    rng = np.random.default_rng(0)
    model = load_model(args.model_central, device=args.device) if args.model_central else None
    manifest = load_manifest(args.manifest)
    test = [(e, i) for e, i in sorted(manifest["exposures"].items()) if i["split"] == "test"]
    rows = []
    for eid, info in test[: args.max_exposures]:
        data = load_exposure_file(f"{args.data_dir}/{info['file']}")
        out = exposure_test(data, info["index"], reserved_star_ids(manifest, eid), prior,
                            args.n_sweeps, rng, args.device, model)
        if out is not None:
            rows.append(out)
    r = np.concatenate(rows)
    cut_ee, cln_ee, tru_ee = r[:, 0], r[:, 1], r[:, 2]
    central = args.model_central or "TRUE"
    print(f"n stars: {len(r)}  (central={central}, lam={args.prior_lam})")
    print(f"  EE@r2 contaminated mean = {cut_ee.mean():.4f}")
    print(f"  EE@r2 cleaned      mean = {cln_ee.mean():.4f}")
    print(f"  EE@r2 truth        mean = {tru_ee.mean():.4f}")
    d = cln_ee - tru_ee
    print(f"  cleaned - truth mean +/- sem = {d.mean():+.4f} +/- {d.std() / np.sqrt(len(d)):.4f}")
    print("  (>0 => over-cleans given this central; ~0 => unbiased cleaning)")


if __name__ == "__main__":
    main()
