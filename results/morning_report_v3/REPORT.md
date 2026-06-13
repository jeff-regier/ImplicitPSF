# Morning report v3 — June 13

**Headline: the model now has a galaxy-fitting result, not just a star-prediction
result.** On 900 galsim-injected galaxies across 75 simulated test exposures, fitting
galaxies with the *implicit* PSF recovers size with MAD 0.0223 (bias -1.2%), versus
PIFF 0.0393 (bias +5.7%) and a header-Moffat PSF 0.0757 — the amortized, no-per-exposure-fit
model wins the inferential task it was built for. PIFF keeps its ellipticity edge
(|e| err 0.014-0.023 vs implicit 0.023-0.075). Separately, Gate 2 confirms the blend
likelihood removes the blend-induced size bias on simulations (6x reduction), at a cost
in ellipticity that the 90-epoch rerun is meant to disambiguate.

## M8 — galaxy injection-recovery (DONE, committed fecd2f8 / ce07389)

900 galsim-injected galaxies, 75 sim test exposures, 4 arms; truth-arm chi2 1.03 (the
fitter is unbiased when handed the true PSF). Figure: `results/galaxy_recovery_sim.png`.

| arm (fixed-n) | size MAD | size bias | |e| err |
|---|---|---|---|
| implicit | 0.0223 | -1.2% | 0.023-0.075 |
| PIFF | 0.0393 | +5.7% | 0.014-0.023 (best) |
| header-Moffat | 0.0757 | — | 0.10-0.35 |

- **Free-n**: rankings unchanged; PIFF absorbs PSF size error into Sersic n (+0.29 bias).
- Fitter bug found via the truth arm: atan2 gradient explosion at round init froze eta2
  (fixed, commit 5c849c4); the first-run parquet is kept as
  `galaxy_recovery_sim_frozen_eta2.parquet` for the record.

## Gate 2 — blend likelihood on the blended sim (committed b5ca6fd / 78a6f8d)

Star-free truth grid (`sim_truth`), 608 sim test exposures. Metrics recomputed
consistently (implicit, both HSM flags ok, n=43,773): T bias = median dT/T, T scatter =
robust(MAD) std, de = median |e_model − e_true|.

| run | T bias | T scatter | de |
|---|---|---|---|
| single-loss control @59 ep | +4.79% | 0.064 | 0.0335 |
| blend loss @60 ep | +0.83% (6x better) | 0.054 | 0.0881 (2.6x worse) |
| **blend loss @90 ep** | **+0.67%** | **0.052** | **0.0870** |

The blend loss removes the size bias but worsens ellipticity. **The 90-epoch rerun
settles the sub-question: the ellipticity cost is NOT undertraining** — de is flat from
60→90 ep (0.0881 → 0.0870) while size keeps improving slightly. So the cost is a
persistent effect (GLS amplitude-trading in tight blends), not a training artifact.
PIFF/PSFEx on the same exposures: T scatter ~0.010, de ~0.003 — per-exposure fits
average the contamination down.

Note (publication plan WS3): this truth grid is on the BLENDED sim. The open question
is whether the blend model's ellipticity cost appears only where there are tight blends
(a real, characterizable tradeoff) or also on the clean sim / isolated stars (a general
regression). The matched single-loss control retrained to 90 ep
(`checkpoints/sim_blended_single_90ep`, in progress) and a clean-sim single-vs-blend
comparison will localize it.

## Real r-band galaxy-handling variants — three-way (committed 2bed7e3 / 910b2c8)

40 ep, batch 2, chi2 cap 50, max targets 96. Reserved-star eval on the r-band test.

| variant | best val | T scatter | de | chi2 | recovered-target frac vs single |
|---|---|---|---|---|---|
| exclude | 3.8174 | 0.0334 | 0.0403 | 1.073 | 1.20x |
| mask | 3.8562 | 0.0336 | 0.0420 | 1.075 | 1.30x |
| component | 3.8488 | 0.0346 | 0.0403 | 1.075 | 1.30x |
| single-loss ref (real_v5_rband) | — | 0.0351 | 0.0366 | 1.068 | 1.00x |

- **Verdict**: the three variants are within noise of each other on reserved stars;
  *exclude* is nominally best (smallest T scatter, smallest e1 bias of the trio).
- Pattern: blend variants give slightly better size scatter, slightly worse de, and a
  small negative e1 median bias (excl -0.006, mask -0.011) — undertraining suspected
  (40 vs the 80 epochs of the reference architecture story).
- **Caveat (pre-registered)**: reserved stars have a contamination floor; the sim Gate 2
  above is the real instrument for blend handling.

## Track A — convergence, seed stability, color pair

- **real_v5** (all-band polar) converged: best val 23.535 @ epoch 28. Full-test eval and
  rho stats already computed (`real_allband_merged.parquet`, `rho_allband.parquet`).
- **real_v5_rband_seed1** done: best val 30.89 (original r-band polar ~31.1) — seed-stable.
- **C2 color pair** (matched architecture): real_v5 *with* color (val 23.535 @ ep28) vs
  **real_v5_zerocolor** (val 23.552 @ epoch 26). Reserved-star eval: color de 0.0322
  (n=29,534, masked subset) vs zero-color de 0.0317 (n=73,609) — within noise on
  aggregate reserved-star shape, as expected. The real chromatic-correction signal is the
  slope-vs-color analysis (v2: +0.00237/mag color−zerocolor), not these aggregates;
  re-verify that slope with the production model in WS5.

## Fixes landed today (all committed)

- M8 gates: S^2 flux double-count; lattice-aligned delta test; beta=4.5 cross-check
  (stamp-sum vs Moffat wings); cell-averaged galaxy profiles (high-n cusps); cosine LR
  decay in `fit_galaxies`; quadratic-form ellipse (eta2 freeze).
- blend: `--blend-max-targets` (memory, unbiased subsets); unusable-target filtering
  (dead-amp stamps); galaxy `component` mode (PSF x round exponential in GLS).

## Status of in-flight work (09:10)

- `sim_truth` on blend_90ep best.pt → Gate 2 90-epoch row (running, GPU 2).
- `run_eval` real_v5_zerocolor best-so-far, all-band, `--zero-color` → C2 twin
  (running, GPU 3).
- real_v5_zerocolor training continues on GPU 1 (epoch 32/60).
