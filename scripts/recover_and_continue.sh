#!/bin/bash
# Regenerate hook-corrupted parquets from data+checkpoints, then start the
# real-data test-split baseline evaluation (the half that needs no trained model).
set -uo pipefail
cd /home/regier/ImplicitPSF

SIM="--manifest manifests/sim_split_v1.json --data-dir /data/scratch/regier/sim_psf_stars"
CHROM="--manifest manifests/sim_chrom_split_v1.json --data-dir /data/scratch/regier/sim_chrom_stars"
REAL="--manifest manifests/split_v1.json --data-dir /data/scratch/regier/sep_des_stars_v2"

echo "=== sim ladder evals (full test split) ==="
uv run python -m implicitpsf.evaluation.run_eval $SIM \
  --checkpoint checkpoints/sim_v3c/best.pt --split test --num-workers 8 \
  --out results/eval_sim_v3c_full.parquet
uv run python -m implicitpsf.evaluation.run_eval $SIM \
  --checkpoint checkpoints/sim_v1c/best.pt --split test --methods implicit --num-workers 8 \
  --out results/eval_sim_v1c.parquet
uv run python -m implicitpsf.evaluation.run_eval $SIM \
  --checkpoint checkpoints/sim_v4/best.pt --split test --methods implicit --num-workers 8 \
  --out results/eval_sim_v4_full.parquet
uv run python -m implicitpsf.evaluation.run_eval $SIM \
  --checkpoint checkpoints/sim_v5/best.pt --split test --methods implicit --num-workers 8 \
  --out results/eval_sim_v5_full.parquet
uv run python -m implicitpsf.evaluation.sim_truth $SIM \
  --checkpoint checkpoints/sim_v4/best.pt --num-workers 8 \
  --out results/truth_sim_v4.parquet

echo "=== chromatic pair ==="
uv run python -m implicitpsf.evaluation.run_eval $CHROM \
  --checkpoint checkpoints/chrom_v3color/best.pt --split test --num-workers 8 \
  --out results/eval_chrom_color.parquet
uv run python -m implicitpsf.evaluation.run_eval $CHROM \
  --checkpoint checkpoints/chrom_v3nocolor/best.pt --split test --methods implicit \
  --num-workers 8 --zero-color --out results/eval_chrom_nocolor.parquet

echo "=== real pilot + dress rehearsal ==="
uv run python -m implicitpsf.evaluation.run_eval $REAL \
  --split val --max-exposures 25 --methods piff psfex --num-workers 8 \
  --out results/pilot_val.parquet
uv run python -m implicitpsf.evaluation.run_eval $REAL \
  --split val --max-exposures 300 --methods piff psfex --num-workers 8 \
  --out results/real_val_dress.parquet

echo "=== blended-era resurrection (pre-isolation simulate.py from git) ==="
git show c90e786:implicitpsf/simulate.py > /tmp/simulate_blended.py
mkdir -p /tmp/blended_pkg && cp /tmp/simulate_blended.py /tmp/blended_pkg/simulate_blended.py
PYTHONPATH=/tmp/blended_pkg uv run python -u -c "
import sys; sys.argv = ['x', '--n-exposures', '6000', '--num-workers', '8',
  '--out-dir', '/data/scratch/regier/sim_blended_stars',
  '--fits-dir', '/data/scratch/regier/sim_blended_fits']
import simulate_blended; simulate_blended.main()
"
uv run python -c "
from implicitpsf.splits import build_manifest, write_manifest
manifest = build_manifest('/data/scratch/regier/sim_blended_stars', seed=0)
write_manifest(manifest, 'manifests/sim_blended_split_v1.json')
print('blended manifest rebuilt')
"
uv run python -m implicitpsf.evaluation.run_eval \
  --manifest manifests/sim_blended_split_v1.json \
  --data-dir /data/scratch/regier/sim_blended_stars \
  --checkpoint checkpoints/sim_run/best.pt --split test --num-workers 8 \
  --out results/sim_eval_v1_blended.parquet

echo "=== real test split: baseline halves (no model needed) ==="
uv run python -m implicitpsf.evaluation.run_eval $REAL \
  --split test --methods piff psfex --num-workers 10 \
  --out results/real_test_baselines.parquet

echo "=== RECOVERY AND BASELINES DONE ==="
