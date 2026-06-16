"""Benchmark-standard PSF validation figures (matching PIFF, Jarvis+2021): spatial
residual maps, residuals vs magnitude (brighter-fatter), residuals vs color (DCR),
residual-ellipticity whiskers, and the size-magnitude stellar-locus star selection.
All from the frozen real-data results; outputs vector PDFs to paper/figures/.
"""

import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import binned_statistic, binned_statistic_2d

plt.rcParams.update(
    {"font.size": 9, "font.family": "serif", "figure.dpi": 150, "savefig.bbox": "tight"}
)

FIGDIR = "paper/figures"
HEADLINE = "results/real_test_v6_blend_allmethods.parquet"
DATA_DIR = "/data/scratch/regier/sep_des_stars_v2"
METHOD_LABEL = {"implicit": "This work", "piff": "PIFF", "psfex": "PSFEx"}
METHOD_COLOR = {"implicit": "C3", "piff": "C0", "psfex": "C1"}


def robust_std(x):
    return 1.4826 * np.median(np.abs(x - np.median(x))) + 1e-12


def clean(d, method):
    return d[(d.method == method) & (d.flag_star == 0) & (d.flag_model == 0)]


def fig_spatial_residuals():
    """Mean residual maps across the CCD for our model (cf. PIFF Fig. 9)."""
    a = clean(pd.read_parquet(HEADLINE), "implicit")
    quants = [
        ("$\\langle\\delta T/T\\rangle$", (a.T_model - a.T_star) / a.T_star, 0.04),
        ("$\\langle\\delta e_1\\rangle$", a.e1_model - a.e1_star, 0.02),
        ("$\\langle\\delta e_2\\rangle$", a.e2_model - a.e2_star, 0.02),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7))
    for ax, (label, vals, lim) in zip(axes, quants, strict=True):
        stat, xe, ye, _ = binned_statistic_2d(
            a.x_pixel, a.y_pixel, vals, statistic="median", bins=[8, 16]
        )
        im = ax.imshow(
            stat.T,
            origin="lower",
            aspect="auto",
            cmap="RdBu_r",
            vmin=-lim,
            vmax=lim,
            extent=[xe[0], xe[-1], ye[0], ye[-1]],
        )
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("CCD $x$")
        fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    axes[0].set_ylabel("CCD $y$")
    fig.suptitle("ImplicitPSF residual maps across the CCD", fontsize=9, y=1.02)
    fig.savefig(f"{FIGDIR}/fig_spatial_residuals.pdf")
    plt.close(fig)


def _binned(x, y, bins):
    med, edges, _ = binned_statistic(x, y, statistic="median", bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, med


def fig_resid_vs_mag():
    """Residuals vs instrumental magnitude, per method (brighter-fatter; cf. PIFF Fig.10)."""
    d = pd.read_parquet(HEADLINE)
    fig, axes = plt.subplots(3, 1, figsize=(3.4, 5.4), sharex=True)
    rows = [
        ("$\\delta T/T$", lambda a: (a.T_model - a.T_star) / a.T_star),
        ("$\\delta e_1$", lambda a: a.e1_model - a.e1_star),
        ("$\\delta e_2$", lambda a: a.e2_model - a.e2_star),
    ]
    for ax, (label, func) in zip(axes, rows, strict=True):
        for m in ["implicit", "piff", "psfex"]:
            a = clean(d, m)
            mag = -2.5 * np.log10(np.clip(a.flux, 1, None))
            bins = np.linspace(np.percentile(mag, 2), np.percentile(mag, 98), 12)
            c, med = _binned(mag, func(a), bins)
            ax.plot(c, med, "o-", ms=3, color=METHOD_COLOR[m], label=METHOD_LABEL[m])
        ax.axhline(0, color="k", lw=0.6, ls=":")
        ax.set_ylabel(label)
    axes[0].legend(fontsize=7, frameon=False)
    axes[-1].set_xlabel("instrumental magnitude $-2.5\\log_{10} f$")
    axes[0].set_title("Residuals vs magnitude (brighter-fatter)", fontsize=9)
    fig.savefig(f"{FIGDIR}/fig_resid_vs_mag.pdf")
    plt.close(fig)


def fig_resid_vs_color():
    """Size residual vs g-i color, per method (chromatic/DCR; cf. PIFF Fig. 11)."""
    d = pd.read_parquet(HEADLINE)
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    for m in ["implicit", "piff", "psfex"]:
        a = clean(d, m)
        a = a[np.isfinite(a.color) & (a.color != 0)]
        bins = np.linspace(np.percentile(a.color, 5), np.percentile(a.color, 95), 12)
        c, med = _binned(a.color, (a.T_model - a.T_star) / a.T_star, bins)
        ax.plot(c, med, "o-", ms=3, color=METHOD_COLOR[m], label=METHOD_LABEL[m])
    ax.axhline(0, color="k", lw=0.6, ls=":")
    ax.set_xlabel("$g-i$ color")
    ax.set_ylabel("$\\delta T/T$")
    ax.legend(fontsize=7.5, frameon=False)
    ax.set_title("Size residual vs color (chromatic PSF)", fontsize=9)
    fig.savefig(f"{FIGDIR}/fig_resid_vs_color.pdf")
    plt.close(fig)


def fig_whisker_resid():
    """Residual-ellipticity whiskers across the CCD: ours vs PIFF (cf. PIFF Fig. 2)."""
    d = pd.read_parquet(HEADLINE)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
    for ax, m in zip(axes, ["implicit", "piff"], strict=True):
        a = clean(d, m)
        de1 = a.e1_model - a.e1_star
        de2 = a.e2_model - a.e2_star
        s1, xe, ye, _ = binned_statistic_2d(a.x_pixel, a.y_pixel, de1, "median", bins=[6, 12])
        s2, _, _, _ = binned_statistic_2d(a.x_pixel, a.y_pixel, de2, "median", bins=[6, 12])
        xc = 0.5 * (xe[:-1] + xe[1:])
        yc = 0.5 * (ye[:-1] + ye[1:])
        gx, gy = np.meshgrid(xc, yc, indexing="ij")
        de = np.hypot(s1, s2)
        ang = 0.5 * np.arctan2(s2, s1)
        scale = 9000.0
        ax.quiver(
            gx,
            gy,
            scale * de * np.cos(ang),
            scale * de * np.sin(ang),
            angles="xy",
            scale_units="xy",
            scale=1,
            headwidth=1,
            headlength=0,
            pivot="mid",
            width=0.006,
        )
        ax.set_title(f"{METHOD_LABEL[m]} residual $e$", fontsize=9)
        ax.set_xlabel("CCD $x$")
        ax.set_aspect("equal")
    axes[0].set_ylabel("CCD $y$")
    fig.suptitle("Residual PSF ellipticity whiskers", fontsize=9, y=1.0)
    fig.savefig(f"{FIGDIR}/fig_whisker_resid.pdf")
    plt.close(fig)


def _second_moment_size(stamp):
    s = stamp - np.median(stamp)
    s = np.clip(s, 0, None)
    tot = s.sum()
    if tot <= 0:
        return np.nan
    p = stamp.shape[0]
    yy, xx = np.mgrid[0:p, 0:p]
    cx = (s * xx).sum() / tot
    cy = (s * yy).sum() / tot
    return (s * ((xx - cx) ** 2 + (yy - cy) ** 2)).sum() / tot


def fig_size_mag():
    """Size-magnitude stellar locus used for star selection (cf. PIFF Fig. 3)."""
    files = sorted(glob.glob(f"{DATA_DIR}/*.pt"))[:1]
    sizes, mags, kinds = [], [], []
    for f in files:
        data = torch.load(f, map_location="cpu", weights_only=False)
        for i in range(min(20, data["cutouts"].shape[0])):
            st = data["star_type"][i].numpy()
            cut = data["cutouts"][i].numpy()
            flux = data["flux"][i].numpy()
            sel = np.argwhere(np.isin(st, [0, 2]) & (flux > 0)).ravel()[:120]
            for j in sel:
                t = _second_moment_size(cut[j])
                if np.isfinite(t) and t < 60:
                    sizes.append(t)
                    mags.append(-2.5 * np.log10(flux[j]))
                    kinds.append(st[j])
    sizes, mags, kinds = np.array(sizes), np.array(mags), np.array(kinds)
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    ax.plot(
        mags[kinds == 2], sizes[kinds == 2], ".", ms=2, alpha=0.3, color="gray", label="galaxies"
    )
    ax.plot(
        mags[kinds == 0], sizes[kinds == 0], ".", ms=3, alpha=0.6, color="C3", label="clean stars"
    )
    ax.set_xlabel("instrumental magnitude")
    ax.set_ylabel("size $T$ (px$^2$)")
    ax.set_ylim(0, 60)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("Size-magnitude star selection", fontsize=9)
    fig.savefig(f"{FIGDIR}/fig_size_mag.pdf")
    plt.close(fig)


RHO = "results/rho_allband.parquet"
RHO_LABEL = {
    "rho1": r"$\rho_1=\langle\delta e\,\delta e\rangle$",
    "rho2": r"$\rho_2=\langle e\,\delta e\rangle$",
    "rho3": r"$\rho_3=\langle e\delta_T\, e\delta_T\rangle$",
    "rho4": r"$\rho_4=\langle\delta e\, e\delta_T\rangle$",
    "rho5": r"$\rho_5=\langle e\, e\delta_T\rangle$",
}


def fig_rho_stats():
    """rho1-rho5 vs angular scale, all three methods (cf. PIFF Jarvis+2021 Fig. 12)."""
    d = pd.read_parquet(RHO)
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6), sharex=True)
    order = ["rho1", "rho2", "rho3", "rho4", "rho5"]
    for ax, r in zip(axes.ravel()[: len(order)], order, strict=True):
        for m in ["implicit", "piff", "psfex"]:
            a = d[(d.method == m) & (d.rho == r)].sort_values("theta")
            ax.plot(
                a.theta,
                np.abs(a.xip),
                "o-",
                ms=2.5,
                color=METHOD_COLOR[m],
                label=METHOD_LABEL[m],
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(RHO_LABEL[r], fontsize=8.5)
        ax.set_ylim(1e-7, 2e-3)
    axes[1, 0].set_xlabel(r"$\theta$ (arcmin)")
    axes[1, 1].set_xlabel(r"$\theta$ (arcmin)")
    axes[0, 0].set_ylabel(r"$|\xi_+(\theta)|$")
    axes[1, 0].set_ylabel(r"$|\xi_+(\theta)|$")
    axes[0, 0].legend(fontsize=7, frameon=False)
    axes[1, 2].axis("off")
    fig.suptitle("PSF residual rho-statistics (reserved stars, all bands)", fontsize=9, y=1.0)
    fig.savefig(f"{FIGDIR}/fig_rho_stats.pdf")
    plt.close(fig)


def main():
    for name, fn in [
        ("rho_stats", fig_rho_stats),
        ("spatial_residuals", fig_spatial_residuals),
        ("resid_vs_mag", fig_resid_vs_mag),
        ("resid_vs_color", fig_resid_vs_color),
        ("whisker_resid", fig_whisker_resid),
        ("size_mag", fig_size_mag),
    ]:
        fn()
        print(f"wrote {FIGDIR}/fig_{name}.pdf")


if __name__ == "__main__":
    main()
