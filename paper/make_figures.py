"""Generate the AJ manuscript figures from the frozen results and simulation data.

Outputs vector PDFs to paper/figures/. Each figure is a standalone function so the
set can be regenerated against a new checkpoint by re-running this script.
"""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

plt.rcParams.update(
    {
        "font.size": 9,
        "font.family": "serif",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
    }
)

FIGDIR = "paper/figures"
SIM_PT = sorted(glob.glob("/data/scratch/regier/sim_realdens6k_stars/*.pt"))[0]
HEADLINE = "results/real_test_v6_blend_allmethods.parquet"
GALREC = "results/galaxy_recovery_real_v6_blend.parquet"
METHOD_LABEL = {"implicit": "This work", "piff": "PIFF", "psfex": "PSFEx"}
METHOD_COLOR = {"implicit": "C3", "piff": "C0", "psfex": "C1"}


def robust_std(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def fig_simdata():
    """Example synthetic cutouts: clean stars (top) and injected galaxies (bottom)."""
    data = torch.load(SIM_PT, map_location="cpu", weights_only=False)
    cut = data["cutouts"][0].numpy()
    st = data["star_type"][0].numpy()
    flux = data["flux"][0].numpy()
    stars = np.argwhere((st == 0) & (flux > 0)).ravel()[:8]
    gals = np.argwhere((st == 2) & (flux > 0)).ravel()[:8]

    fig, axes = plt.subplots(2, 8, figsize=(7.2, 2.0))
    for col, idx in enumerate(stars):
        stamp = cut[idx]
        axes[0, col].imshow(np.arcsinh(stamp / robust_std(stamp.ravel())), cmap="gray_r")
    for col, idx in enumerate(gals):
        stamp = cut[idx]
        axes[1, col].imshow(np.arcsinh(stamp / robust_std(stamp.ravel())), cmap="gray_r")
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    axes[0, 0].set_ylabel("clean\nstars", fontsize=9)
    axes[1, 0].set_ylabel("galaxies", fontsize=9)
    fig.suptitle(
        "Example simulated cutouts ($32\\times32$ px, arcsinh stretch)", fontsize=9, y=1.02
    )
    fig.savefig(f"{FIGDIR}/fig_simdata.pdf")
    plt.close(fig)


def fig_psffield():
    """Whisker plot of the true (spatially varying) PSF ellipticity field in simulation."""
    d = pd.read_parquet("results/truth_sim_realdens6k_single_s7.parquet")
    d = d[(d.method == "implicit") & (d.flag_true == 0)].iloc[:600]
    e = np.hypot(d.e1_true, d.e2_true)
    ang = 0.5 * np.arctan2(d.e2_true, d.e1_true)
    scale = 220.0
    dx = scale * e * np.cos(ang)
    dy = scale * e * np.sin(ang)
    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    ax.quiver(
        d.x,
        d.y,
        dx,
        dy,
        e,
        angles="xy",
        scale_units="xy",
        scale=1,
        headwidth=1,
        headlength=0,
        pivot="mid",
        cmap="viridis",
        width=0.004,
    )
    ax.set_xlabel("CCD $x$ (px)")
    ax.set_ylabel("CCD $y$ (px)")
    ax.set_title("True PSF ellipticity field (simulation)")
    ax.set_aspect("equal")
    fig.savefig(f"{FIGDIR}/fig_psffield.pdf")
    plt.close(fig)


def fig_residuals():
    """Real-data reserved-star residual histograms for the three methods."""
    d = pd.read_parquet(HEADLINE)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
    panels = [
        ("$\\delta T/T$", lambda a: (a.T_model - a.T_star) / a.T_star, (-0.15, 0.15)),
        ("$\\delta e_1$", lambda a: a.e1_model - a.e1_star, (-0.06, 0.06)),
        ("$\\delta e_2$", lambda a: a.e2_model - a.e2_star, (-0.06, 0.06)),
    ]
    for ax, (label, func, lim) in zip(axes, panels, strict=True):
        for m in ["implicit", "piff", "psfex"]:
            a = d[(d.method == m) & (d.flag_star == 0) & (d.flag_model == 0)]
            vals = np.clip(func(a), *lim)
            ax.hist(
                vals,
                bins=80,
                range=lim,
                histtype="step",
                density=True,
                color=METHOD_COLOR[m],
                label=METHOD_LABEL[m],
                lw=1.3,
            )
        ax.axvline(0, color="k", lw=0.6, ls=":")
        ax.set_xlabel(label)
        ax.set_yticks([])
    axes[0].legend(fontsize=7, frameon=False)
    axes[0].set_ylabel("density")
    fig.suptitle("Reserved-star residuals on the DES test split", fontsize=9, y=1.02)
    fig.savefig(f"{FIGDIR}/fig_residuals.pdf")
    plt.close(fig)


def _corr(a, comp):
    return np.corrcoef(a[f"{comp}_true"], a[f"{comp}_model"])[0, 1]


def fig_galaxy_context():
    """The spin-2 finding: galaxies-in-context kill one e-component; point-source fixes it."""
    base = pd.read_parquet("results/truth_sim_realdens6k_single_s7.parquet")
    fix = pd.read_parquet("results/truth_sim_realdens6k_psctx.parquet")
    base = base[(base.method == "implicit") & (base.flag_true == 0) & (base.flag_model == 0)]
    fix = fix[(fix.method == "implicit") & (fix.flag_true == 0) & (fix.flag_model == 0)]

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    for ax, a, title in [
        (axes[0], base, f"polar + galaxies\ncorr$(e_1)$={_corr(base, 'e1'):.2f}"),
        (axes[1], fix, f"point-source context\ncorr$(e_1)$={_corr(fix, 'e1'):.2f}"),
    ]:
        ax.plot(a.e1_true, a.e1_model, ".", ms=1.5, alpha=0.3, color="C3")
        lim = (-0.4, 0.4)
        ax.plot(lim, lim, "k-", lw=0.7)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel("true $e_1$")
        ax.set_ylabel("model $e_1$")
        ax.set_title(title, fontsize=8)
        ax.set_aspect("equal")

    configs = [
        ("polar+gal", base, "C3"),
        ("diagonal", pd.read_parquet("results/truth_sim_realdens6k_diagonal.parquet"), "C1"),
        ("pt-source", fix, "C2"),
        ("no galaxy", pd.read_parquet("results/truth_nogal.parquet"), "C0"),
    ]
    names, c1s, c2s, cols = [], [], [], []
    for name, df, col in configs:
        a = df[(df.method == "implicit") & (df.flag_true == 0) & (df.flag_model == 0)]
        names.append(name)
        c1s.append(_corr(a, "e1"))
        c2s.append(_corr(a, "e2"))
        cols.append(col)
    xpos = np.arange(len(names))
    axes[2].bar(xpos - 0.2, c1s, 0.4, label="$e_1$", color="C3")
    axes[2].bar(xpos + 0.2, c2s, 0.4, label="$e_2$", color="C0")
    axes[2].set_xticks(xpos)
    axes[2].set_xticklabels(names, rotation=30, ha="right", fontsize=7)
    axes[2].set_ylabel("shape correlation")
    axes[2].set_ylim(0, 1)
    axes[2].legend(fontsize=7, frameon=False)
    axes[2].set_title("recovery vs context", fontsize=8)
    fig.savefig(f"{FIGDIR}/fig_galaxy_context.pdf")
    plt.close(fig)


def fig_sampeff():
    """Size-residual scatter vs number of fit stars k, per method."""
    ks = [5, 10, 25, 50, 100]
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    for m in ["implicit", "piff", "psfex"]:
        scat = []
        for k in ks:
            d = pd.read_parquet(f"results/ksweep_k{k}.parquet")
            a = d[(d.method == m) & (d.flag_star == 0) & (d.flag_model == 0)]
            scat.append(robust_std((a.T_model - a.T_star) / a.T_star))
        ax.plot(ks, scat, "o-", color=METHOD_COLOR[m], label=METHOD_LABEL[m], ms=4)
    ax.set_xscale("log")
    ax.set_xlabel("fit stars per exposure $k$")
    ax.set_ylabel("size residual scatter")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("Sample efficiency")
    fig.savefig(f"{FIGDIR}/fig_sampeff.pdf")
    plt.close(fig)


def fig_galrec():
    """Galaxy ellipticity recovery bias per arm (star-anchored injection)."""
    d = pd.read_parquet(GALREC)
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    arms = [
        ("truth", "k", "truth (validation)"),
        ("implicit", "C3", "This work"),
        ("piff", "C0", "PIFF"),
    ]
    for arm, col, label in arms:
        a = d[d.arm == arm]
        de1 = np.clip(a.eta1_fit - a.eta1_true, -0.4, 0.4)
        ax.hist(
            de1,
            bins=50,
            range=(-0.4, 0.4),
            histtype="step",
            density=True,
            color=col,
            label=label,
            lw=1.4,
        )
    ax.axvline(0, color="k", lw=0.6, ls=":")
    ax.set_xlabel("recovered $-$ true galaxy $e_1$")
    ax.set_yticks([])
    ax.set_ylabel("density")
    ax.legend(fontsize=7.5, frameon=False)
    ax.set_title("Galaxy ellipticity recovery (real data)")
    fig.savefig(f"{FIGDIR}/fig_galrec.pdf")
    plt.close(fig)


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    for name, fn in [
        ("simdata", fig_simdata),
        ("psffield", fig_psffield),
        ("residuals", fig_residuals),
        ("galaxy_context", fig_galaxy_context),
        ("sampeff", fig_sampeff),
        ("galrec", fig_galrec),
    ]:
        fn()
        print(f"wrote {FIGDIR}/fig_{name}.pdf")


if __name__ == "__main__":
    main()
