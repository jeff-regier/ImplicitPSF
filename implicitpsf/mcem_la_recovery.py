"""(lambda, alpha) recovery on REAL simulated stars -- the open test beyond SBC.

SBC validates the hierarchical (lambda, alpha) inference on data generated FROM its own prior. This
runs the same inference on the ACTUAL simulated star cutouts, whose contamination comes from a
different, realistic process (simulate.inject_contaminants: a spatial Poisson rate, truncated-Pareto
fluxes, a star/galaxy mix, and SEP selection that drops stars with a detectable neighbour). We use
the TRUE PSF as each star's central template (sub-pixel, the centering-bug-free path) so the
residual is contamination, build the same grid_n=8 / flux_lo=200 design, and infer (lambda, alpha)
by the generalized hier_gibbs. Truth: surviving clean stars carry contaminants in [floor,
FLAG_LIMIT], so the effective rate of inference-range [200, 2000] sources per stamp is
contam_rate * P(200<=S<=2000) under the S^-1.5 generator (=0.555 for the truncation [100, 6000]);
the slope is alpha=1.5. A recovery near those values is the realistic-data confirmation that the
inference is not merely self-consistent (vs the closed-loop SBC).
"""

import argparse

import numpy as np

from implicitpsf.contam_model import cell_centers
from implicitpsf.datasets import load_exposure_file
from implicitpsf.evaluation.run_eval import exposure_masks
from implicitpsf.evaluation.sim_truth import truth_stamps
from implicitpsf.mcem_estep import hier_gibbs
from implicitpsf.mcem_sampler import cov_columns, psf_covariance
from implicitpsf.simulate import (
    COLOR_MEAN,
    CONTAM_FLUX_CEIL,
    CONTAM_FLUX_EXPONENT,
    CONTAM_FLUX_FLOOR,
    PATCH,
    set_psf_model,
)
from implicitpsf.splits import load_manifest, reserved_star_ids


def true_effective_rate(lo, hi, contam_rate):
    """Generator rate of inference-range [lo,hi] contaminants per stamp (Poisson thinning of the
    S^-(1+b) power law over the full [floor,ceil] range)."""
    b = CONTAM_FLUX_EXPONENT
    frac = (lo**-b - hi**-b) / (CONTAM_FLUX_FLOOR**-b - CONTAM_FLUX_CEIL**-b)
    return contam_rate * frac, 1.0 + b  # (lambda_true, alpha_true)


def gather_stars(manifest, data_dir, max_exposures):
    """Collect clean stars across exposures: cutout, inverse-variance weight, TRUE per-star PSF."""
    test = [(e, i) for e, i in sorted(manifest["exposures"].items()) if i["split"] == "test"]
    datas, weights, centrals = [], [], []
    for eid, info in test[:max_exposures]:
        data = load_exposure_file(f"{data_dir}/{info['file']}")
        index = info["index"]
        clean, reserved = exposure_masks(data, index, reserved_star_ids(manifest, eid))
        idx = np.nonzero(clean & ~reserved)[0]
        if len(idx) < 5:
            continue
        x = data["x_pixel"][index].numpy()[idx]
        y = data["y_pixel"][index].numpy()[idx]
        field = {"chromatic": False, **data["true_field"][index]}
        ref = COLOR_MEAN if field["chromatic"] else 0.0
        tc = truth_stamps(field, x, y, ref).reshape(len(idx), -1)
        cg = np.clip(tc, 0, None)
        cg /= cg.sum(1, keepdims=True) + 1e-12
        cut = data["cutouts"][index].numpy()[idx].reshape(len(idx), -1)
        var = data["variance"][index].numpy()[idx].reshape(len(idx), -1)
        val = data["valid_pixels"][index].numpy()[idx].reshape(len(idx), -1)
        w = (val > 0) / np.clip(var, 1e-6, None)
        datas.extend(list(cut))
        weights.append(w)
        centrals.append(cg)
    return datas, np.concatenate(weights), np.concatenate(centrals)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="manifests/sim_contamreal_sub118.json")
    parser.add_argument("--data-dir", default="/data/scratch/regier/sim_contamreal_stars")
    parser.add_argument("--psf-model", default="realistic")
    parser.add_argument("--max-exposures", type=int, default=12)
    parser.add_argument("--n-sweeps", type=int, default=250)
    parser.add_argument("--grid-n", type=int, default=8)
    parser.add_argument("--flux-lo", type=float, default=200.0)
    parser.add_argument("--flux-hi", type=float, default=2000.0)
    parser.add_argument("--contam-rate", type=float, default=1.0)
    args = parser.parse_args()
    set_psf_model(args.psf_model)
    rng = np.random.default_rng(0)
    manifest = load_manifest(args.manifest)
    datas, weight, central = gather_stars(manifest, args.data_dir, args.max_exposures)
    centers = cell_centers(PATCH, args.grid_n, 2.5)
    cols = cov_columns(centers, psf_covariance(central.mean(0), PATCH), PATCH)
    chain = hier_gibbs(datas, weight, central, cols, args.flux_lo, args.flux_hi, args.n_sweeps, rng)
    lam_t, alpha_t = true_effective_rate(args.flux_lo, args.flux_hi, args.contam_rate)
    llo, lmed, lhi = np.percentile(chain[:, 0], [5, 50, 95])
    alo, amed, ahi = np.percentile(chain[:, 1], [5, 50, 95])
    print(f"(lambda,alpha) recovery on {len(datas)} real sim clean stars "
          f"(grid_n={args.grid_n}, flux [{args.flux_lo:.0f},{args.flux_hi:.0f}])")
    print(f"  lambda: truth {lam_t:.3f} | posterior median {lmed:.3f}  90% CI [{llo:.3f},{lhi:.3f}]"
          f"  covers={llo <= lam_t <= lhi}")
    print(f"  alpha:  truth {alpha_t:.3f} | posterior median {amed:.3f} "
          f"90% CI [{alo:.3f},{ahi:.3f}]  covers={alo <= alpha_t <= ahi}")


if __name__ == "__main__":
    main()
