#!/bin/bash
# Waits for the chromatic-sim training runs, evaluates the C2 experiment, and
# appends it to the morning report.
set -uo pipefail
cd /home/regier/ImplicitPSF

for log in train_chrom_color train_chrom_nocolor; do
  until grep -qE "done; best val loss|Traceback" "/data/scratch/regier/$log.log" 2>/dev/null; do
    sleep 300
  done
done

echo "=== chromatic eval: color-conditioned model, all methods ==="
uv run python -m implicitpsf.evaluation.run_eval \
  --manifest manifests/sim_chrom_split_v1.json \
  --data-dir /data/scratch/regier/sim_chrom_stars \
  --checkpoint checkpoints/chrom_color/best.pt \
  --split test --num-workers 12 \
  --out results/chrom_eval_color.parquet

echo "=== chromatic eval: zero-color ablation, implicit only ==="
uv run python -m implicitpsf.evaluation.run_eval \
  --manifest manifests/sim_chrom_split_v1.json \
  --data-dir /data/scratch/regier/sim_chrom_stars \
  --checkpoint checkpoints/chrom_nocolor/best.pt \
  --split test --methods implicit --num-workers 12 --zero-color \
  --out results/chrom_eval_nocolor.parquet

uv run python -m implicitpsf.evaluation.report \
  --eval chrom_color=results/chrom_eval_color.parquet \
         chrom_nocolor=results/chrom_eval_nocolor.parquet \
  --ccd-width 1024 --ccd-height 2048 \
  --out results/chromatic_report

{
  echo ""
  echo "## Chromatic simulation (C2): color-dependent PSF, color conditioning on/off"
  echo ""
  tail -n +2 results/chromatic_report/REPORT.md
} >> results/morning_report/REPORT.md 2>/dev/null || true

echo "=== chromatic chain done ==="
