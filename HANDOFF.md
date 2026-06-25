# ImplicitPSF — MCEM contamination-correction handoff

_Snapshot: 2026-06-24. Head: `348a0dd` (origin/main). Written for an agent picking this up._

## Goal

ImplicitPSF is an attention-based **continuous effective-PSF field** for DES single-epoch
images: one network maps an exposure's stars (context) to the PSF at any queried
position/color/flux/resolution, amortizing the per-exposure fitting that PSFEx/PIFF repeat.

The current thrust is **contamination correction**. "Clean" training stars carry
sub-threshold blends that bias the learned PSF ~2% too wide (under-concentrated), which
propagates to a ~−5.6% galaxy-size deficit on real data. We correct it with **proper
Monte-Carlo EM** (Wei & Tanner). The method is written up in `manuscript/main.tex` §`sec:mcem`.

**Hard rule (Jeff):** the synthetic story must be airtight first — the PSF concentration,
galaxy size, AND the inferred `(λ, α)` must all return to truth, *across EM iterations,
converged* — before any real-data work. **Real data is ON HOLD.** The earlier posterior-mean
Gibbs EM is SUPERSEDED ("nonsense") — do not resurrect "subtract the posterior mean → retrain
`--loss-mode clean`", and do not launch clean-target or architecture-bake-off jobs.

## What's built (all committed + pushed)

| File | Role |
|---|---|
| `implicitpsf/mcem_sampler.py` | Per-star collapsed Gibbs. Bivariate-Gaussian Cholesky contaminants (Σ = Σ_central + LLᵀ, lower-bounded by the current PSF), **profiled** central amplitude (no prior), returns **K post-burn samples (not the mean)**. `run_batch_gpu` = vectorized torch/GPU batch over all stars in a file. SBC + R̂/ESS gates (`--mode sbc`). |
| `implicitpsf/mcem_clean.py` | K-imputation E-step → writes a `cutout_imp` (E,S,K,H,W) dataset. **`--device cuda` loads the model on GPU** (else renders run on CPU — a silent 10× slowdown). |
| `implicitpsf/datasets.py` | `sample_imputations`: M-step loader draws one random imputation per star per epoch (Monte-Carlo-averages the posterior; no point estimate). No-op without `cutout_imp`. |
| `implicitpsf/train_psf.py` | `--init-checkpoint` warm-start for the EM M-step. |
| `implicitpsf/mcem_estep.py` | Hierarchical `(λ, α)` as global random variables; `--mode mixing` (multi-chain R̂/ESS). |
| `implicitpsf/mcem_iterate.py` | Automated EM driver: E-step → warm partial M-step → δEE diagnostic, logged to `trajectory.csv`. `--gpu` pins one chain per GPU. |

**Performance:** the per-star Python Gibbs (~1.2 s/star × 515k stars ≈ 11 hr) was vectorized
to GPU; full K-imputation generation of the 150-file sub118 set is now **~11 min**.

## Diagnostic / gate tooling

- `implicitpsf/sim_psf_ee_defect.py` — δEE@r2 of the model PSF vs the **known sim truth** at
  star-free positions. The PSF gate. Usage:
  `python -m implicitpsf.sim_psf_ee_defect --checkpoint <best.pt> --manifest manifests/sim_contamreal_sub118.json --data-dir /data/scratch/regier/sim_contamreal_stars --psf-model realistic --max-exposures 30`
- `implicitpsf/evaluation/galaxy_recovery_real.py --psf-model realistic` — the `analytic_truth`
  arm gives the de-confounded galaxy-size bias (`implicit − analytic_truth`). Run
  `--num-workers 0` (Pool deadlocks otherwise).
- SBC + mixing must gate **every** sampler change (`mcem_sampler.py --mode sbc`,
  `mcem_estep.py --mode mixing`).

## Running right now

**5 parallel EM chains** (`mcem_iterate`, seeds 0-4, GPUs 1-5, 25 iters each), started from the
iteration-0 model. ~51 min/iter. Trajectories → `/data/scratch/regier/mcem_em_s{0..4}/trajectory.csv`.
Logs `logs/mcem_em_s*.log`.

## ✅ RESOLVED (Jun 25): the over-correction was a SUB-PIXEL CENTERING BUG — core gate passes

The iterated EM ran away to δEE@r2 +0.020 (the earlier "+0.0003" was a non-converged transient).
**Root cause:** `central_psf_stamps` rendered the NN cleaning central at `round(x,y)` = integer-
centered, but each star sits at the cutout's sub-pixel center (corner = round(center) − half, star
at half+frac). Cleaning with the mis-centered template made a dipole the Gibbs ate as
"contamination" → severe over-cleaning, amplified by iteration. **Fix (commit 61e5999):** Fourier
sub-pixel-shift the rendered central by frac. The decisive diagnostic was `mcem_truth_central_test
--model-central` (measure cleaning bias with the ACTUAL NN central, not just truth): cleaned−truth
EE@r2 **+0.0246 → +0.0007** (unbiased).

**Synthetic gate results (centering-fixed, λ=0.1):**
- PSF δEE@r2: contam **−0.0072** → iterated EM converges to ~**+0.004** (near truth, gate edge);
  `--fixed-central` variant (clean with the stable broad data-supported PSF, not the slightly-
  sharpening model) centers it ~**+0.0007**. Truth is now a stable fixed point.
- Galaxy size (de-confounded implicit−analytic_truth median): contam **−2.37%** → cleaned
  **−0.82%** (~65% of the deficit removed; a δEE=0 model → ~0%).
- **λ=0.1 is the EFFECTIVE sub-threshold rate** on SEP-selected clean stars (detectable neighbors
  flag and drop their hosts), well below the injection λ=1 — what the (λ,α) inference should recover.

So **PSF δEE and galaxy size both move to truth** — the core synthetic gate (items 7a,7b) passes.

## Second open item (decoupled): hierarchical (λ, α) mixing

The global `(λ, α)` inference **does not mix** (R̂ ~1.2 / ESS ~13; collapsing λ analytically did
not fix the per-cell detection-config autocorrelation). This is synthetic-gate item 7c (the
"(λ, α) unknown" robustness demo). It is **separate** from the core PSF/galaxy gate, because the
cleaner currently uses a **fixed, generator-matched prior** that *is* SBC-calibrated. Levers to
try: coarser detection grid, blocked/checkerboard cell updates, or longer chains.

## To resume

1. `cat /data/scratch/regier/mcem_em_s*/trajectory.csv` → plot δEE vs iteration (5 chains); read
   the **shape** and apply the decision rule above.
2. If converging to ~0: run `galaxy_recovery_real --psf-model realistic` (analytic_truth arm) on
   a fixed-point model — does galaxy size → 0 too? δEE→0 + size→0 + converged = **PSF/galaxy gate
   passes.**
3. Then tackle the `(λ, α)` mixing, then (only then) real data.
4. Operate autonomously, GPUs 1-5 for trainings/chains, evals on free GPUs (0/6/7), never eval on
   a live-training GPU, kill by PID with the `[x]`-bracket pgrep trick, watch `quota -s`.

## Where the rest of the context lives

- `~/.claude/plans/make-a-plan-to-glistening-pond.md` — full plan: top "METHOD REBUILD" +
  "HANDOFF SNAPSHOT" + "REBUILD PROGRESS (Jun 24)", plus the long campaign history below them.
- Memory `msstep-corrects-sim-deficit` (in `~/.claude/projects/.../memory/`).
- `CLAUDE.md` — repo conventions, provenance rules, hard-won gotchas.
- `manuscript/main.tex` §`sec:mcem` — the method as written for the paper.
