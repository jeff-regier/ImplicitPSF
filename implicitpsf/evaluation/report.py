"""Turn evaluation parquets into a markdown report with tables and figures.

Inputs are the tidy tables from run_eval.py (reserved-star metrics, any number of
labeled files for ablations) and optionally sim_truth.py (truth at star-free grid
positions). Statistics follow the paper protocol: the unit of inference is the
exposure, and paired method differences get bootstrap-over-exposures intervals.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from implicitpsf.evaluation.rho_stats import rho_statistics

BOOTSTRAP_SAMPLES = 2000


def load_eval(spec):
    """Load 'label=path' (or bare path, labeled by stem) into one long table."""
    label, _, path = spec.rpartition("=")
    label = label or Path(path).stem
    table = pd.read_parquet(path)
    table["run"] = label
    return table


def with_residuals(table):
    ok = (table["flag_star"] == 0) & (table["flag_model"] == 0)
    table = table[ok].copy()
    table["t_frac"] = (table["T_star"] - table["T_model"]) / table["T_star"]
    table["de1"] = table["e1_star"] - table["e1_model"]
    table["de2"] = table["e2_star"] - table["e2_model"]
    return table


def robust_scatter(values):
    return 1.4826 * np.median(np.abs(values - np.median(values)))


def summary_table(table):
    rows = []
    for (run, method), group in table.groupby(["run", "method"]):
        rows.append(
            {
                "run": run,
                "method": method,
                "n_stars": len(group),
                "n_exposures": group["exposure_id"].nunique(),
                "t_frac_median": np.median(group["t_frac"]),
                "t_frac_scatter": robust_scatter(group["t_frac"]),
                "de1_median": np.median(group["de1"]),
                "de2_median": np.median(group["de2"]),
                "de_scatter": robust_scatter(np.hypot(group["de1"], group["de2"])),
                "chi2_median": np.median(group["chi2"]),
            }
        )
    return pd.DataFrame(rows)


def paired_bootstrap(table, metric, baseline="piff"):
    """Per-exposure paired differences vs the baseline with bootstrap 95% CIs."""
    per_exposure = (
        table.assign(abs_metric=table[metric].abs())
        .groupby(["run", "method", "exposure_id"])["abs_metric"]
        .mean()
        .unstack(["run", "method"])
    )
    rows = []
    rng = np.random.default_rng(0)
    for run, method in per_exposure.columns:
        if method == baseline:
            continue
        base_col = (run, baseline)
        if base_col not in per_exposure.columns:
            continue
        diff = (per_exposure[(run, method)] - per_exposure[base_col]).dropna().values
        if len(diff) < 5:
            continue
        samples = rng.choice(diff, size=(BOOTSTRAP_SAMPLES, len(diff)), replace=True)
        lo, hi = np.percentile(samples.mean(axis=1), [2.5, 97.5])
        rows.append(
            {
                "run": run,
                "method": method,
                "metric": f"mean |{metric}| - {baseline}",
                "difference": diff.mean(),
                "ci_low": lo,
                "ci_high": hi,
                "n_exposures": len(diff),
            }
        )
    return pd.DataFrame(rows)


def rho_section(table, out_dir):
    """rho1/rho2 per (run, method), accumulated over exposures; returns figure path."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for (run, method), group in table.groupby(["run", "method"]):
        tables = [
            {
                "x_pixel": exposure["x_pixel"].values,
                "y_pixel": exposure["y_pixel"].values,
                "e1": exposure["e1_star"].values,
                "e2": exposure["e2_star"].values,
                "de1": exposure["de1"].values,
                "de2": exposure["de2"].values,
                "t_frac": exposure["t_frac"].values,
            }
            for _, exposure in group.groupby("exposure_id")
            if len(exposure) >= 2
        ]
        if not tables:
            continue
        rho = rho_statistics(tables)
        label = f"{run}:{method}"
        axes[0].loglog(rho["rho1"]["theta"], np.abs(rho["rho1"]["xip"]), "o-", label=label)
        axes[1].loglog(rho["rho2"]["theta"], np.abs(rho["rho2"]["xip"]), "o-", label=label)
    for axis, name in zip(axes, ["rho1", "rho2"], strict=True):
        axis.set_xlabel("theta [arcmin]")
        axis.set_title(f"|{name}|")
    axes[0].legend(fontsize=7)
    path = out_dir / "rho_stats.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def residual_histograms(table, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    for (run, method), group in table.groupby(["run", "method"]):
        label = f"{run}:{method}"
        for axis, column, lim in zip(
            axes, ["t_frac", "de1", "de2"], [0.1, 0.05, 0.05], strict=True
        ):
            axis.hist(
                group[column].clip(-lim, lim),
                bins=60,
                histtype="step",
                density=True,
                label=label,
            )
            axis.set_xlabel(column)
    axes[0].legend(fontsize=7)
    path = out_dir / "residual_histograms.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def spatial_maps(table, out_dir, ccd_width, ccd_height):
    pairs = sorted(table.groupby(["run", "method"]).groups)
    fig, axes = plt.subplots(1, len(pairs), figsize=(3.2 * len(pairs), 5), squeeze=False)
    for axis, (run, method) in zip(axes[0], pairs, strict=True):
        group = table[(table["run"] == run) & (table["method"] == method)]
        stat, _, _, _ = _binned_2d(
            group["x_pixel"], group["y_pixel"], group["t_frac"], ccd_width, ccd_height
        )
        image = axis.imshow(
            stat.T,
            origin="lower",
            extent=[0, ccd_width, 0, ccd_height],
            vmin=-0.03,
            vmax=0.03,
            cmap="RdBu_r",
            aspect="auto",
        )
        axis.set_title(f"{run}:{method}", fontsize=8)
    fig.colorbar(image, ax=axes[0], label="median t_frac", shrink=0.8)
    path = out_dir / "spatial_t_frac.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def _binned_2d(x, y, values, width, height, nx=8, ny=16):
    x_edges = np.linspace(0, width, nx + 1)
    y_edges = np.linspace(0, height, ny + 1)
    x_idx = np.clip(np.digitize(x, x_edges) - 1, 0, nx - 1)
    y_idx = np.clip(np.digitize(y, y_edges) - 1, 0, ny - 1)
    stat = np.full((nx, ny), np.nan)
    for i in range(nx):
        for j in range(ny):
            cell = values[(x_idx == i) & (y_idx == j)]
            if len(cell) > 0:
                stat[i, j] = np.median(cell)
    return stat, x_edges, y_edges, (x_idx, y_idx)


def truth_summary(truth_path):
    table = pd.read_parquet(truth_path)
    ok = (table["flag_true"] == 0) & (table["flag_model"] == 0)
    table = table[ok].copy()
    table["t_frac"] = (table["T_model"] - table["T_true"]) / table["T_true"]
    table["de1"] = table["e1_model"] - table["e1_true"]
    table["de2"] = table["e2_model"] - table["e2_true"]
    rows = []
    for method, group in table.groupby("method"):
        rows.append(
            {
                "method": method,
                "n_grid_points": len(group),
                "t_frac_median": np.median(group["t_frac"]),
                "t_frac_scatter": robust_scatter(group["t_frac"]),
                "de1_median": np.median(group["de1"]),
                "de2_median": np.median(group["de2"]),
                "de_scatter": robust_scatter(np.hypot(group["de1"], group["de2"])),
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", nargs="+", required=True, help="label=path parquet specs")
    parser.add_argument("--sim-truth", default=None)
    parser.add_argument("--out", default="results/report")
    parser.add_argument("--ccd-width", type=float, default=2048.0)
    parser.add_argument("--ccd-height", type=float, default=4096.0)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    table = with_residuals(pd.concat([load_eval(spec) for spec in args.eval]))
    summary = summary_table(table)
    paired = pd.concat(
        [paired_bootstrap(table, metric) for metric in ["t_frac", "de1", "de2"]],
        ignore_index=True,
    )

    hist_path = residual_histograms(table, out_dir)
    map_path = spatial_maps(table, out_dir, args.ccd_width, args.ccd_height)
    rho_path = rho_section(table, out_dir)

    lines = [
        "# PSF model comparison report",
        "",
        "## Reserved-star metrics (per run and method)",
        "",
        summary.to_markdown(index=False, floatfmt=".5f"),
        "",
        "## Paired differences vs PIFF (bootstrap over exposures, 95% CI)",
        "",
        paired.to_markdown(index=False, floatfmt=".6f") if len(paired) else "(piff absent)",
        "",
        f"![residuals]({hist_path.name})",
        f"![spatial]({map_path.name})",
        f"![rho]({rho_path.name})",
    ]
    if args.sim_truth is not None:
        truth = truth_summary(args.sim_truth)
        lines += [
            "",
            "## Truth-grid metrics (star-free positions, simulation only)",
            "",
            truth.to_markdown(index=False, floatfmt=".5f"),
        ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report -> {out_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
