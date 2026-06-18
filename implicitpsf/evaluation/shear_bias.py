"""Translate PSF-model residuals into weak-lensing shear-bias estimates.

A PSF modeling error propagates into the inferred galaxy shear two ways:

* **Additive** (delta xi_+): the spatially coherent PSF-modeling error sources an additive
  contribution to the shear two-point function. This is the noise-immune quantity (per-star
  ellipticity noise cancels in the cross-correlations), built from the rho-statistics
  (rho_stats.py) via the standard quadratic form (Jarvis et al. 2016, eq. 3.6; Gatti et al.
  2021). With the PSF-systematic model
      de_sys = alpha * e_psf + beta * de_psf + eta * (e_psf * dT/T_psf),
  the additive bias is
      delta xi_+ = alpha^2 rho0 + beta^2 rho1 + eta^2 rho3
                   + 2 alpha beta rho2 + 2 beta eta rho4 + 2 alpha eta rho5,
  with rho0 = <e_psf e_psf> (PSF ellipticity auto-correlation), rho1 = <de de>,
  rho2 = <e_psf de>, rho3 = <(e_psf dT/T)^2>, rho4 = <de (e_psf dT/T)>,
  rho5 = <e_psf (e_psf dT/T)>. We omit the alpha^2 rho0 term (rho0 not in the rho parquet;
  alpha^2 ~ 1e-3 makes it sub-dominant for the small leakage measured here) and note it.
  For beta=1 (model error propagates directly) the bias is dominated by rho1.

  We deliberately do NOT estimate the leakage alpha by per-star regression of de on e_psf:
  e_psf is the *noisy* single-star ellipticity, so that slope is biased by measurement noise
  (regression dilution) and comes out the same for every method -- it measures the e_psf
  noise, not the model. alpha must be set at the 2-point level (fiducial here; CLI override).

* **Multiplicative** (m): an error in the modeled PSF *size* mis-calibrates the galaxy size.
  To leading order m ~ (T_psf / T_gal) * <dT/T> (Hirata & Seljak 2003; Massey et al. 2013):
  a galaxy twice the PSF size inherits about half the fractional PSF size error. <dT/T> is a
  mean, so unlike the additive slope it is noise-unbiased. We report both.

Comparison is against a fiducial Stage-III tolerance (|m| < 0.03; |delta xi_+| < 2e-7),
both CLI-overridable.
"""

import argparse

import numpy as np
import pandas as pd

from implicitpsf.provenance import write_result

FIDUCIAL_TGAL_OVER_TPSF = 2.0
# fiducial PSF-systematic leakage coefficients (DES Y3-scale; set at the 2-point level)
FIDUCIAL_ALPHA = 0.02
FIDUCIAL_BETA = 1.0
FIDUCIAL_ETA = 1.0
REQUIREMENT_M = 0.03
REQUIREMENT_DXI = 2.0e-7


def multiplicative_bias(df, tgal_over_tpsf):
    """Per-method multiplicative bias m from the mean fractional PSF size residual."""
    rows = []
    for method, g in df.groupby("method"):
        dt_over_t = (g["T_star"].to_numpy() - g["T_model"].to_numpy()) / g["T_star"].to_numpy()
        mean_dt = float(np.nanmean(dt_over_t))
        rows.append(
            {
                "method": method,
                "mean_dT_over_T": mean_dt,
                "m": mean_dt / tgal_over_tpsf,
                "n_stars": int(len(g)),
            }
        )
    return pd.DataFrame(rows)


def additive_dxi_plus(rho_df, alpha, beta, eta):
    """Per-method, per-scale additive contamination delta xi_+(theta) from rho-statistics.

    rho_df is the long rho parquet (method, rho, theta, xip, npairs). Returns a tidy frame
    (method, bin, theta, dxi_plus) using the Jarvis+2016 quadratic form (alpha^2 rho0
    omitted). We pivot on the theta-BIN rank, not the float theta: treecorr's mean-radius
    differs in its last digits between correlations, so an exact-theta pivot would split each
    bin across rows and yield all-NaN combinations.
    """
    rho_df = rho_df.copy()
    rho_df["bin"] = rho_df.groupby(["method", "rho"])["theta"].rank(method="first").astype(int)
    wide = rho_df.pivot_table(index=["method", "bin"], columns="rho", values="xip")
    theta = rho_df.groupby(["method", "bin"])["theta"].mean()
    dxi = (
        beta**2 * wide["rho1"]
        + eta**2 * wide["rho3"]
        + 2 * alpha * beta * wide["rho2"]
        + 2 * beta * eta * wide["rho4"]
        + 2 * alpha * eta * wide["rho5"]
    )
    out = dxi.reset_index(name="dxi_plus")
    out["theta"] = theta.reset_index(drop=True)
    return out


def shear_bias_summary(
    df,
    rho_df,
    tgal_over_tpsf=FIDUCIAL_TGAL_OVER_TPSF,
    alpha=FIDUCIAL_ALPHA,
    beta=FIDUCIAL_BETA,
    eta=FIDUCIAL_ETA,
):
    """Combine multiplicative m and the peak additive |delta xi_+| per method."""
    mult = multiplicative_bias(df, tgal_over_tpsf).set_index("method")
    dxi = additive_dxi_plus(rho_df, alpha, beta, eta)
    peak = dxi.assign(absdxi=dxi["dxi_plus"].abs()).groupby("method")["absdxi"].max()
    rows = []
    for method in mult.index:
        m = float(mult.loc[method, "m"])
        max_dxi = float(peak.get(method, np.nan))
        rows.append(
            {
                "method": method,
                "mean_dT_over_T": float(mult.loc[method, "mean_dT_over_T"]),
                "m": m,
                "max_abs_dxi_plus": max_dxi,
                "n_stars": int(mult.loc[method, "n_stars"]),
                "meets_m_req": bool(abs(m) < REQUIREMENT_M),
                "meets_dxi_req": bool(max_dxi < REQUIREMENT_DXI),
                "alpha": alpha,
                "beta": beta,
                "eta": eta,
                "tgal_over_tpsf": tgal_over_tpsf,
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", default="results/real_test_v6_blend_allmethods.parquet")
    parser.add_argument("--rho-parquet", default="results/rho_test_v6_blend_recompute.parquet")
    parser.add_argument("--out", default="results/shear_bias_v6_blend.parquet")
    parser.add_argument("--tgal-over-tpsf", type=float, default=FIDUCIAL_TGAL_OVER_TPSF)
    parser.add_argument("--alpha", type=float, default=FIDUCIAL_ALPHA)
    parser.add_argument("--beta", type=float, default=FIDUCIAL_BETA)
    parser.add_argument("--eta", type=float, default=FIDUCIAL_ETA)
    parser.add_argument("--band", default=None)
    args = parser.parse_args()

    df = pd.read_parquet(args.parquet)
    if args.band is not None:
        df = df[df["band"] == args.band]
    rho_df = pd.read_parquet(args.rho_parquet)
    table = shear_bias_summary(df, rho_df, args.tgal_over_tpsf, args.alpha, args.beta, args.eta)
    write_result(table, args.out, source="shear_bias", purpose=f"WL shear bias from {args.parquet}")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
