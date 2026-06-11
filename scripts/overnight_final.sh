#!/bin/bash
# Deadline-aware final chain: evals + morning report assembled by ~08:30 with the
# best checkpoints available, regardless of which trainings have fully converged.
set -uo pipefail
cd /home/regier/ImplicitPSF

wait_done() {
  local log=$1 limit_minutes=$2 waited=0
  until grep -aqE "done; best val loss" "$log" 2>/dev/null; do
    grep -aq "Traceback" "$log" 2>/dev/null && { echo "WARNING: $log crashed"; return; }
    sleep 120; waited=$((waited + 2))
    [ "$waited" -ge "$limit_minutes" ] && { echo "NOTE: $log still running, using best-so-far"; return; }
  done
}

echo "=== waiting for v1c (cap 50m) ==="
wait_done /data/scratch/regier/train_sim_v1c.log 50
echo "=== launching chrom_v3nocolor on GPU 1 ==="
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 nohup uv run python -u \
  -m implicitpsf.train_psf --data-dir /data/scratch/regier/sim_chrom_stars \
  --manifest manifests/sim_chrom_split_v1.json --out-dir checkpoints/chrom_v3nocolor \
  --max-epochs 45 --patience 10 --batch-size 6 --n-attn-layers 2 --decoder-film \
  --diagonal-coords --zero-color \
  > /data/scratch/regier/train_chrom_v3nocolor.log 2>&1 &

echo "=== waiting for v3c (cap 70m) ==="
wait_done /data/scratch/regier/train_sim_v3c.log 70

echo "=== base-sim evals ==="
uv run python -m implicitpsf.evaluation.run_eval \
  --manifest manifests/sim_split_v1.json --data-dir /data/scratch/regier/sim_psf_stars \
  --checkpoint checkpoints/sim_v3c/best.pt --split test --num-workers 12 \
  --out results/eval_sim_v3c_full.parquet
uv run python -m implicitpsf.evaluation.run_eval \
  --manifest manifests/sim_split_v1.json --data-dir /data/scratch/regier/sim_psf_stars \
  --checkpoint checkpoints/sim_v1c/best.pt --split test --methods implicit --num-workers 12 \
  --out results/eval_sim_v1c.parquet
uv run python -m implicitpsf.evaluation.sim_truth \
  --manifest manifests/sim_split_v1.json --data-dir /data/scratch/regier/sim_psf_stars \
  --checkpoint checkpoints/sim_v3c/best.pt --num-workers 12 \
  --out results/truth_sim_v3c.parquet

echo "=== chromatic color eval ==="
uv run python -m implicitpsf.evaluation.run_eval \
  --manifest manifests/sim_chrom_split_v1.json --data-dir /data/scratch/regier/sim_chrom_stars \
  --checkpoint checkpoints/chrom_v3color/best.pt --split test --num-workers 12 \
  --out results/eval_chrom_color.parquet

echo "=== early evals of in-flight runs (best-so-far checkpoints) ==="
if [ -f checkpoints/sim_v4/best.pt ]; then
  uv run python -m implicitpsf.evaluation.run_eval \
    --manifest manifests/sim_split_v1.json --data-dir /data/scratch/regier/sim_psf_stars \
    --checkpoint checkpoints/sim_v4/best.pt --split test --methods implicit \
    --max-exposures 150 --num-workers 12 --out results/eval_sim_v4_early.parquet
fi
if [ -f checkpoints/chrom_v3nocolor/best.pt ]; then
  uv run python -m implicitpsf.evaluation.run_eval \
    --manifest manifests/sim_chrom_split_v1.json --data-dir /data/scratch/regier/sim_chrom_stars \
    --checkpoint checkpoints/chrom_v3nocolor/best.pt --split test --methods implicit \
    --max-exposures 150 --num-workers 12 --zero-color \
    --out results/eval_chrom_nocolor_early.parquet
fi

echo "=== assembling morning report ==="
EVALS="clean_v3=results/eval_sim_v3c_full.parquet clean_v1=results/eval_sim_v1c.parquet"
EVALS="$EVALS blended_v1=results/sim_eval_v1_blended.parquet"
[ -f results/eval_sim_v4_early.parquet ] && EVALS="$EVALS v4_early=results/eval_sim_v4_early.parquet"
uv run python -m implicitpsf.evaluation.report \
  --eval $EVALS --sim-truth results/truth_sim_v3c.parquet \
  --ccd-width 1024 --ccd-height 2048 --out results/morning_report

CHROM="chrom_color=results/eval_chrom_color.parquet"
[ -f results/eval_chrom_nocolor_early.parquet ] && CHROM="$CHROM chrom_nocolor_early=results/eval_chrom_nocolor_early.parquet"
uv run python -m implicitpsf.evaluation.report \
  --eval $CHROM --ccd-width 1024 --ccd-height 2048 --out results/chromatic_report

{
  echo ""
  echo "## Chromatic simulation (C2)"
  tail -n +2 results/chromatic_report/REPORT.md 2>/dev/null
  echo ""
  echo "## Real-data training progress (overnight, ongoing)"
  echo '```'
  for f in real_run real_rband real_noattn; do
    echo "$f: $(tail -1 checkpoints/$f/train_log.csv 2>/dev/null)"
  done
  echo '```'
  echo ""
  echo "## Real-data baselines at scale"
  echo "pilot (25 exp): results/pilot_report/ | dress rehearsal (297 exp): results/real_val_dress.parquet"
  echo "PIFF: dT/T med +0.005, chi2 1.07 | PSFEx: +0.010, 1.05 (real val, all bands)"
} >> results/morning_report/REPORT.md
echo "=== MORNING REPORT READY ==="
