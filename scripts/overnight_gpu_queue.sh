#!/bin/bash
# Launch queued training runs as GPUs free up overnight. Chromatic-sim runs first
# (highest marginal value per the overnight doctrine), then real-data ablations.
# Claims only fully-free GPUs among nvidia-smi indices 1,2,3,6.
set -uo pipefail
cd /home/regier/ImplicitPSF

# real exposures have 512 star slots (sim: 240); batch 8 OOMs an 11 GB card
CHROM_ARGS="--data-dir /data/scratch/regier/sim_chrom_stars \
  --manifest manifests/sim_chrom_split_v1.json --max-epochs 80 --patience 20 --batch-size 8"
REAL_ARGS="--data-dir /data/scratch/regier/sep_des_stars_v2 \
  --manifest manifests/split_v1.json --max-epochs 60 --patience 10 --batch-size 4"

free_gpu() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F', ' '$1==1 || $1==2 || $1==3 || $1==6 {if ($2 < 500) {print $1; exit}}'
}

launch_when_free() {
  local name=$1
  shift
  while true; do
    gpu=$(free_gpu)
    if [ -n "${gpu:-}" ]; then
      CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu nohup uv run python -u \
        -m implicitpsf.train_psf --out-dir "checkpoints/$name" "$@" \
        > "/data/scratch/regier/train_$name.log" 2>&1 &
      echo "launched $name on GPU $gpu"
      sleep 90  # let it allocate before the next probe
      return
    fi
    sleep 300
  done
}

SIM_ARGS="--data-dir /data/scratch/regier/sim_psf_stars \
  --manifest manifests/sim_split_v1.json --max-epochs 80 --patience 15 --batch-size 6"

# v3 (FiLM decoder + 2-layer attention) is the architecture-fix bet and outranks
# further v1-architecture runs; chrom_color (v1) is already training separately
launch_when_free sim_v3 $SIM_ARGS --n-attn-layers 2 --decoder-film
launch_when_free real_noattn $REAL_ARGS --no-attention
launch_when_free chrom_nocolor $CHROM_ARGS --zero-color
launch_when_free real_zerocolor $REAL_ARGS --zero-color
echo "gpu queue drained"
