"""De-confounded bridge metric: compare contamination-inference methods on REAL contam-sim stars.

The metric is the per-star ΔEE@r2(cleaned − contaminated): each star is compared to ITSELF before/
after the method subtracts its inferred contamination, so the stacking/noise/centering artifact
cancels (an absolute stack-vs-truth deficit does NOT — clean sim stars show ~-0.015 of it; see plan,
that confound burned an earlier conclusion). A positive ΔEE@r2 = the method removed contaminating
light and sharpened the star. Any method plugs in via clean_fn(data, weight, central_g) -> the
contamination image to subtract (the posterior-mean contamination = the EM E-step). Reports median /
mean ΔEE@r2 ± sem and the rough fraction of the ~+0.0071 contamination effect recovered. The central
template is the KNOWN clean truth PSF (optimistic) -- swap for the NN estimate in the real EM.
"""

import argparse
import glob

import numpy as np

from implicitpsf.contam_gibbs import run as gibbs_run
from implicitpsf.contam_model import cell_centers, design_columns
from implicitpsf.datasets import load_exposure_file
from implicitpsf.evaluation.sim_truth import truth_stamps
from implicitpsf.simulate import COLOR_MEAN, PATCH, set_psf_model

CONTAM_EFFECT = 0.0071  # measured contam-vs-clean-sim deficit difference (the signal to recover)


def ee2(v):
    """Encircled energy within r=2 px of the stamp centre (1D or 2D input)."""
    s = np.clip(v, 0, None).reshape(PATCH, PATCH)
    c = (PATCH - 1) / 2.0
    yy, xx = np.mgrid[0:PATCH, 0:PATCH]
    return s[np.hypot(xx - c, yy - c) <= 2].sum() / (s.sum() + 1e-12)


def bridge(clean_fn, data_dir, n_exposures=8, snr_min=80.0, max_per_exp=12):
    """Run clean_fn on real contam-sim clean stars; return (mean, sem, n) of the per-star cleaning."""
    set_psf_model("realistic")
    data = load_exposure_file(sorted(glob.glob(f"{data_dir}/*.pt"))[0])
    deltas = []
    for idx in range(min(n_exposures, len(data["cutouts"]))):
        st = data["star_type"][idx].numpy()
        snr = data["snr"][idx].numpy()
        clean = np.nonzero((st == 0) & (snr >= snr_min))[0][:max_per_exp]
        field = {"chromatic": False, **data["true_field"][idx]}
        x = data["x_pixel"][idx].numpy()
        y = data["y_pixel"][idx].numpy()
        cut = data["cutouts"][idx].numpy()
        var = data["variance"][idx].numpy()
        val = data["valid_pixels"][idx].numpy()
        for j in clean:
            color = COLOR_MEAN if field["chromatic"] else 0.0
            truth = truth_stamps(field, [x[j]], [y[j]], color)[0]
            cg = np.clip(truth, 0, None).ravel()
            cg = cg / cg.sum()
            w = (val[j].ravel() > 0) / np.clip(var[j].ravel(), 1e-6, None)
            contam = clean_fn(cut[j].ravel(), w, cg)
            deltas.append(ee2(cut[j].ravel() - contam) - ee2(cut[j].ravel()))
    deltas = np.array(deltas)
    return deltas.mean(), deltas.std() / np.sqrt(len(deltas)), len(deltas)


def gibbs_clean_fn(prior, fwhm, grid_n, core_radius, n_sweeps, seed):
    """A clean_fn that runs the Gibbs E-step (posterior-mean contamination) for the bridge."""
    centers = cell_centers(PATCH, grid_n, core_radius)
    columns = design_columns(centers, fwhm, PATCH)
    rng = np.random.default_rng(seed)

    def clean_fn(data, weight, central_g):
        mean_contam, _, _ = gibbs_run(data, weight, central_g, columns, prior, n_sweeps, rng)
        return mean_contam

    return clean_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["gibbs"], default="gibbs")
    parser.add_argument("--data-dir", default="/data/scratch/regier/sim_contamreal_stars")
    parser.add_argument("--n-exposures", type=int, default=6)
    parser.add_argument("--n-sweeps", type=int, default=60)
    args = parser.parse_args()
    prior = {"lam": 1.0, "flux_lo": 100.0, "flux_hi": 2000.0, "alpha": 1.5}
    clean_fn = gibbs_clean_fn(prior, 4.0, PATCH, 2.5, args.n_sweeps, seed=0)
    mean, sem, n = bridge(clean_fn, args.data_dir, args.n_exposures)
    print(f"{args.method} bridge on {n} real contam-sim stars:")
    print(f"  cleaning ΔEE@r2 = {mean:+.5f} +/- {sem:.5f}  "
          f"(~{100 * mean / CONTAM_EFFECT:.0f}% of the {CONTAM_EFFECT:+.4f} contamination effect)")


if __name__ == "__main__":
    main()
