#!/bin/bash
# Waits for the sim training runs, then assembles the morning report:
# reserved-star eval (3 methods), truth-grid eval, ablation table, pilot gate.
set -uo pipefail
cd /home/regier/ImplicitPSF

wait_for_run() {
  local log=$1 limit_minutes=$2 waited=0
  until grep -qE "done; best val loss" "$log" 2>/dev/null; do
    if grep -qE "Traceback" "$log" 2>/dev/null; then
      echo "WARNING: $log shows a traceback; continuing with best checkpoint so far"
      return
    fi
    sleep 120
    waited=$((waited + 2))
    if [ "$waited" -ge "$limit_minutes" ]; then
      echo "WARNING: $log still running after ${limit_minutes}m; using best checkpoint so far"
      return
    fi
  done
}

echo "=== waiting for sim training runs ==="
wait_for_run /data/scratch/regier/train_sim.log 240
wait_for_run /data/scratch/regier/train_sim_big.log 300
wait_for_run /data/scratch/regier/train_sim_seed1.log 300
wait_for_run /data/scratch/regier/train_sim_noattn.log 300

echo "=== reserved-star eval: main model, all three methods ==="
uv run python -m implicitpsf.evaluation.run_eval \
  --manifest manifests/sim_split_v1.json \
  --data-dir /data/scratch/regier/sim_psf_stars \
  --checkpoint checkpoints/sim_run/best.pt \
  --split test --num-workers 12 \
  --out results/sim_eval_main.parquet

echo "=== truth-grid eval: main model, all three methods ==="
uv run python -m implicitpsf.evaluation.sim_truth \
  --manifest manifests/sim_split_v1.json \
  --data-dir /data/scratch/regier/sim_psf_stars \
  --checkpoint checkpoints/sim_run/best.pt \
  --num-workers 12 \
  --out results/sim_truth_main.parquet

echo "=== reserved-star eval: ablation checkpoints, implicit only ==="
for run in sim_big sim_seed1 sim_noattn; do
  uv run python -m implicitpsf.evaluation.run_eval \
    --manifest manifests/sim_split_v1.json \
    --data-dir /data/scratch/regier/sim_psf_stars \
    --checkpoint checkpoints/$run/best.pt \
    --split test --methods implicit --num-workers 12 \
    --out results/sim_eval_$run.parquet
done

echo "=== assembling morning report ==="
uv run python -m implicitpsf.evaluation.report \
  --eval main=results/sim_eval_main.parquet \
         big=results/sim_eval_sim_big.parquet \
         seed1=results/sim_eval_sim_seed1.parquet \
         noattn=results/sim_eval_sim_noattn.parquet \
  --sim-truth results/sim_truth_main.parquet \
  --ccd-width 1024 --ccd-height 2048 \
  --out results/morning_report

{
  echo ""
  echo "## Real-data training progress (overnight)"
  echo ""
  echo '```'
  tail -5 checkpoints/real_run/train_log.csv 2>/dev/null || echo "no real log yet"
  echo '```'
  echo ""
  echo "## Real-data pilot gate (PIFF/PSFEx on 25 val exposures)"
  echo ""
  cat results/pilot_report/REPORT.md 2>/dev/null | tail -n +2 || echo "pilot pending"
} >> results/morning_report/REPORT.md

echo "=== morning report ready: results/morning_report/REPORT.md ==="
