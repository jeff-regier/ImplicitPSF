#!/bin/bash
# First real-data comparison: r-band test split, ImplicitPSF (v4 and v1 arch)
# vs PIFF vs PSFEx on identical reserved stars.
set -euo pipefail
cd /home/regier/ImplicitPSF

uv run python -c "
import pandas as pd
baselines = pd.read_parquet('results/real_test_baselines.parquet')
rband = baselines[baselines.band == 'r']
print('r-band baseline rows:', len(rband), 'exposures:', rband.exposure_id.nunique())
for arch in ['v4', 'v1']:
    implicit = pd.read_parquet(f'results/real_test_implicit_{arch}_rband.parquet')
    merged = pd.concat([implicit, rband[rband.exposure_id.isin(implicit.exposure_id)]],
                       ignore_index=True)
    merged.to_parquet(f'results/real_rband_{arch}_merged.parquet')
    print(arch, 'merged rows:', len(merged))
"

uv run python -m implicitpsf.evaluation.report \
  --eval rband_v4=results/real_rband_v4_merged.parquet \
         rband_v1=results/real_rband_v1_merged.parquet \
  --ccd-width 2048 --ccd-height 4096 \
  --out results/real_rband_report

echo "=== RBAND COMPARISON ASSEMBLED ==="
