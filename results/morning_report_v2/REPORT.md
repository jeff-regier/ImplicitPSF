# Morning report v2 — June 12 (corrected measurements)

**Headline: on the all-band real-data test (1,200 frozen test exposures, well-measured
reserved stars), the amortized model — polar architecture, best-so-far checkpoint at
epoch ~30, no per-exposure fitting — reaches parity-to-better with PIFF across every
metric:** size scatter 0.0272 vs PIFF 0.0314 / PSFEx 0.0294 (best); ellipticity
correlations (+0.85, +0.83) vs PIFF (+0.82, +0.82); |de| 0.0247 vs 0.0222/0.0211
(~11% gap); chi2/dof 1.089 between the baselines. Per-band size scatter uniform
across grizY.

Also tonight: all real-data moment tables were regenerated after visual inspection
(Jeff) exposed measurement corruption — stamps near the dead amplifier were measured
including masked garbage; moments now use the valid-pixel mask, and a labeled
"well-measured subset" handles the remaining contaminated tail (neighbors just outside
the isolation radius; faint misclassified galaxies). The r-band architecture story:
v4 (27 epochs) corr(e2)=+0.04 -> polar (80 epochs) corr(e1)=+0.75, corr(e2)=+0.39 ->
all-band polar (+0.85, +0.83): the simulation-derived ladder transferred to real data
and the residual r-band deficit was a data-volume effect.

## C2 on real data (r-band)

- paired slope difference (color - zerocolor): **+0.00237/mag**, CI [+0.0016, +0.0030]
- residual chromatic slope: color model -0.0057/mag vs PIFF -0.0076, PSFEx -0.0079,
  zero-color twin -0.0080 — the baselines structurally retain the systematic.

## Density stratification (paired mean |dT/T| - PIFF, r-band)

| q         |   size |     mean |
|:----------|-------:|---------:|
| q1_sparse |    177 | -0.01568 |
| q2        |    167 |  0.00215 |
| q3        |    169 |  0.00435 |
| q4_dense  |    170 |  0.00471 |

Negative = ImplicitPSF better. Sparse-field advantage as pre-registered.

## Sample efficiency (same k fit stars for all methods)

| k | implicit scat / chi2 | piff scat / chi2 | psfex scat / chi2 |
|---|---|---|---|
| 5 | 0.078 / 1.11 | 0.182 / 5.14 | 0.059 / 1.27 |
| 10 | 0.070 / 1.10 | 0.075 / 1.84 | 0.059 / 1.28 |
| 25 | 0.060 / 1.10 | 0.046 / 1.17 | 0.040 / 1.07 |
| 50 | 0.055 / 1.10 | 0.039 / 1.10 | 0.037 / 1.05 |
| 100 | 0.056 / 1.10 | 0.038 / 1.08 | 0.038 / 1.05 |

ImplicitPSF is flat in k; PIFF collapses below k~25 (chi2 5.1 at k=5).

## Sim ladder (final)

chi2/dof on 608 sim test exposures: blended 31.3 -> clean v1 16.6 -> FiLM 8.9 ->
diagonal 4.96 -> **polar 4.41** (PIFF 2.67, verified floor 1.0).
polar: corr(e1,e2) = 0.95/0.97, |de| = 0.027.

## All-band real comparison

see allband/REPORT.md (tables and figures)

## Polar vs v4 on real r-band

see rband_polar/REPORT.md (tables and figures)

## All-band color-conditioning test (appended 08:40)

Paired slope difference d(dT/T)/d(color), color minus zero-color, identical stars
(note: zero-color twin is the v4 architecture at epoch ~27 vs polar at epoch ~30,
so this mixes architecture and color effects — the clean same-architecture pair
is queued for today): +0.00762 [+0.00722, +0.00810] n=28358 (1185 exposures)
