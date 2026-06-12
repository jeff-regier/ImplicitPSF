# Raw material for morning report v3 (assembled June 13 ~07:00-08:30)

## M8 galaxy injection-recovery (DONE, committed fecd2f8 / ce07389)
- 900 galsim-injected galaxies, 75 sim test exposures, 4 arms; truth-arm chi2 1.03.
- Fixed-n: implicit re MAD 0.0223 (bias -1.2%) vs PIFF 0.0393 (bias +5.7%) vs
  header-Moffat 0.0757; PIFF best ellipticity (|e| err 0.014-0.023) vs implicit
  (0.023-0.075), Moffat far behind (0.10-0.35). Figure: results/galaxy_recovery_sim.png.
- Free-n: rankings unchanged; PIFF absorbs PSF size error into n (+0.29 bias).
- Fitter bug found via truth arm: atan2 gradient explosion at round init froze eta2
  (commit 5c849c4); first-run parquet kept as galaxy_recovery_sim_frozen_eta2.parquet.

## Gate 2: blend likelihood on blended sim (committed b5ca6fd / 78a6f8d)
- Single loss control (60 ep): star-free T bias +4.8%, T scatter 0.064, de 0.0212.
- Blend loss (60 ep): T bias +0.8% (6x reduction), T scatter 0.054, de 0.0638 (3x WORSE).
- Blend run hit 60-epoch cap still improving; 90-epoch rerun on GPU 3
  (checkpoints/sim_blended_blend_90ep) to separate undertraining from GLS
  amplitude-trading in tight pairs. Eval best-so-far at 07:00.
- PIFF/PSFEx on same exposures: T scatter ~0.010, de ~0.003 (per-exposure fits average
  contamination down; the amortized model learned it — and blend loss fixes the size part).

## Real r-band galaxy-variant trainings (40 ep, batch 2, chi2 cap 50, max targets 96)
- exclude: best val 3.8174; reserved-star eval: T scatter 0.0334, de 0.0403, chi2 1.073
- mask:    best val 3.8562; eval: T scatter 0.0336, de 0.0420, chi2 1.075
- component: best val 3.8488; eval running (results/real_test_implicit_blend_component.parquet)
- single-loss reference (real_v5_rband): T scatter 0.0351, de 0.0366, chi2 1.068
- Caveat (pre-registered): reserved stars have a contamination floor; sim Gate 2 is the instrument.
- Pattern: blend variants slightly better size scatter, slightly worse de, small negative
  e1 median bias (excl -0.006, mask -0.011) — undertraining suspected (40 vs 80 epochs).
- Recovered-target fraction vs single-loss clean stars (30-file r-band train sample):
  exclude 1.20x, mask/component 1.30x.

## Track A
- real_v5 (all-band polar) converged: best val 23.535 at epoch 28; full-test eval
  (real_test_implicit_allband_masked.parquet, merged real_allband_merged.parquet) and
  rho stats (rho_allband.parquet) already computed on this checkpoint.
- real_v5_rband_seed1 done: best val 30.89 (original r-band polar ~31.1) — seed-stable.
- real_v5_zerocolor (all-band): still training (epoch ~12/60 at ~41 min/epoch);
  best-so-far eval at 07:00 for the clean same-architecture C2 color pair.

## Fixes landed today (all committed)
- M8 gates: S^2 flux double-count; lattice-aligned delta test; beta=4.5 cross-check
  (stamp-sum convention vs Moffat wings explained); cell-averaged galaxy profiles
  (high-n cusps); cosine lr decay in fit_galaxies; quadratic-form ellipse (eta2 freeze).
- blend: --blend-max-targets (memory, unbiased subsets); unusable-target filtering
  (dead-amp stamps); galaxy 'component' mode (PSF x round exponential in GLS).

## 07:00 queue
1. Eval sim_blended_blend_90ep best-so-far via sim_truth (star-free) -> Gate 2 update.
2. Eval real_v5_zerocolor best-so-far (run_eval implicit, all-band test, --zero-color)
   -> C2 pair with real_v5.
3. Component-variant numbers into the three-way table.
4. Assemble REPORT.md, commit by 08:30, CronDelete loop 9abc22f7.
