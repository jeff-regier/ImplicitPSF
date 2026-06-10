#!/bin/bash
# Overnight chain for the real-data pipeline: manifest -> training -> pilot gate.
set -euo pipefail
cd /home/regier/ImplicitPSF

echo "=== building real-data manifest ==="
uv run python -c "
from implicitpsf.splits import build_manifest, write_manifest
from collections import Counter
manifest = build_manifest('/data/scratch/regier/sep_des_stars_v2', seed=0)
write_manifest(manifest, 'manifests/split_v1.json')
counts = Counter(info['split'] for info in manifest['exposures'].values())
bands = Counter(info['band'] for info in manifest['exposures'].values())
print('splits:', dict(counts))
print('bands:', dict(bands))
print('reserved stars:', sum(len(v) for v in manifest['reserved'].values()))
"

echo "=== launching real training on GPU 4 ==="
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 nohup uv run python -u \
  -m implicitpsf.train_psf \
  --data-dir /data/scratch/regier/sep_des_stars_v2 \
  --manifest manifests/split_v1.json \
  --out-dir checkpoints/real_run \
  --max-epochs 60 --patience 10 --batch-size 8 \
  > /data/scratch/regier/train_real.log 2>&1 &

echo "=== PIFF/PSFEx pilot gate: 25 val exposures ==="
uv run python -m implicitpsf.evaluation.run_eval \
  --manifest manifests/split_v1.json \
  --data-dir /data/scratch/regier/sep_des_stars_v2 \
  --split val --max-exposures 25 --methods piff psfex --num-workers 8 \
  --out results/pilot_val.parquet

uv run python -m implicitpsf.evaluation.report \
  --eval pilot=results/pilot_val.parquet --out results/pilot_report

echo "=== overnight_real done ==="
