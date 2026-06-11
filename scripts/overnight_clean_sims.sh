#!/bin/bash
# Rest-of-night chain after the isolation-label fix: retrain on clean sims,
# evaluate, and assemble the final morning report.
set -uo pipefail
cd /home/regier/ImplicitPSF

SIM_ARGS="--data-dir /data/scratch/regier/sim_psf_stars \
  --manifest manifests/sim_split_v1.json --max-epochs 60 --patience 12"
CHROM_ARGS="--data-dir /data/scratch/regier/sim_chrom_stars \
  --manifest manifests/sim_chrom_split_v1.json --max-epochs 60 --patience 12"

train() {
  local gpu=$1 name=$2
  shift 2
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu nohup uv run python -u \
    -m implicitpsf.train_psf --out-dir "checkpoints/$name" "$@" \
    > "/data/scratch/regier/train_$name.log" 2>&1 &
}

wait_done() {
  local log=$1 limit_minutes=$2 waited=0
  until grep -aqE "done; best val loss" "$log" 2>/dev/null; do
    grep -aq "Traceback" "$log" 2>/dev/null && { echo "WARNING: $log crashed"; return; }
    sleep 120; waited=$((waited + 2))
    [ "$waited" -ge "$limit_minutes" ] && { echo "WARNING: $log timeout"; return; }
  done
}

until grep -q "CLEAN SIMS READY" /data/scratch/regier/sim_regen.log 2>/dev/null; do sleep 120; done
echo "=== launching clean-sim trainings ==="
train 1 sim_v1c $SIM_ARGS --batch-size 8
train 2 sim_v3c $SIM_ARGS --batch-size 6 --n-attn-layers 2 --decoder-film
train 6 chrom_v3color $CHROM_ARGS --batch-size 6 --n-attn-layers 2 --decoder-film

wait_done /data/scratch/regier/train_sim_v1c.log 270
wait_done /data/scratch/regier/train_sim_v3c.log 270
# chrom_nocolor reuses GPU 1 or 2 once free; simplest: wait for v1c then launch there
train 1 chrom_v3nocolor $CHROM_ARGS --batch-size 6 --n-attn-layers 2 --decoder-film --zero-color

echo "=== evals: clean base sim ==="
for run in sim_v1c sim_v3c; do
  uv run python -m implicitpsf.evaluation.run_eval \
    --manifest manifests/sim_split_v1.json --data-dir /data/scratch/regier/sim_psf_stars \
    --checkpoint checkpoints/$run/best.pt --split test --num-workers 12 \
    $( [ "$run" = "sim_v3c" ] && echo "" || echo "--methods implicit" ) \
    --out results/eval_$run.parquet
done
uv run python -m implicitpsf.evaluation.run_eval \
  --manifest manifests/sim_split_v1.json --data-dir /data/scratch/regier/sim_psf_stars \
  --checkpoint checkpoints/sim_v3c/best.pt --split test --num-workers 12 \
  --out results/eval_sim_v3c_full.parquet
uv run python -m implicitpsf.evaluation.sim_truth \
  --manifest manifests/sim_split_v1.json --data-dir /data/scratch/regier/sim_psf_stars \
  --checkpoint checkpoints/sim_v3c/best.pt --num-workers 12 \
  --out results/truth_sim_v3c.parquet

echo "=== evals: chromatic ==="
wait_done /data/scratch/regier/train_chrom_v3color.log 240
wait_done /data/scratch/regier/train_chrom_v3nocolor.log 240
uv run python -m implicitpsf.evaluation.run_eval \
  --manifest manifests/sim_chrom_split_v1.json --data-dir /data/scratch/regier/sim_chrom_stars \
  --checkpoint checkpoints/chrom_v3color/best.pt --split test --num-workers 12 \
  --out results/eval_chrom_color.parquet
uv run python -m implicitpsf.evaluation.run_eval \
  --manifest manifests/sim_chrom_split_v1.json --data-dir /data/scratch/regier/sim_chrom_stars \
  --checkpoint checkpoints/chrom_v3nocolor/best.pt --split test --methods implicit \
  --num-workers 12 --zero-color --out results/eval_chrom_nocolor.parquet

echo "=== final morning report ==="
uv run python -m implicitpsf.evaluation.report \
  --eval clean_v3=results/eval_sim_v3c_full.parquet \
         clean_v1=results/eval_sim_v1c.parquet \
         blended_v1=results/sim_eval_v1_blended.parquet \
  --sim-truth results/truth_sim_v3c.parquet \
  --ccd-width 1024 --ccd-height 2048 --out results/morning_report
uv run python -m implicitpsf.evaluation.report \
  --eval chrom_color=results/eval_chrom_color.parquet \
         chrom_nocolor=results/eval_chrom_nocolor.parquet \
  --ccd-width 1024 --ccd-height 2048 --out results/chromatic_report
{
  echo ""
  echo "## Chromatic simulation (C2)"
  tail -n +2 results/chromatic_report/REPORT.md
  echo ""
  echo "## Real-data training progress"
  echo '```'
  for f in real_run real_rband real_noattn; do
    echo "$f: $(tail -1 checkpoints/$f/train_log.csv 2>/dev/null)"
  done
  echo '```'
  echo ""
  echo "## Real-data baseline dress rehearsal (297 val exposures)"
  echo "see results/real_val_dress.parquet; pilot report: results/pilot_report/"
} >> results/morning_report/REPORT.md
echo "=== MORNING REPORT READY ==="
