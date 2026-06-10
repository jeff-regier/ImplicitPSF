# CLAUDE.md

## Project Overview

**ImplicitPSF**: an attention-based *continuous PSF field* for DES single-epoch images.
Given an exposure's stars as context (multi-head attention over position/color/flux
encodings and cutouts), the model predicts the PSF at **any queried position and
resolution** (per-pixel coordinate decoder, `implicitpsf/implicit_psf.py`). Predicting
held-out stars is the training objective and validation instrument; the *goal* is the
PSF at star-free positions, rendered (oversampled) for convolution with galaxies being
fit (`implicitpsf/render.py: render_at`). Baselines: PIFF and PSFEx, run by us.

## Commands

```bash
uv run pytest                                        # run tests
uv run ruff check --fix . && uv run ruff format .    # lint + format
uv run python -m implicitpsf.extract_des_stars       # SEP + DR2 extraction -> v2 .pt
uv run python -m implicitpsf.simulate                # simulated exposures, known PSF field
uv run python -m implicitpsf.train_psf --manifest manifests/split_v1.json
uv run python -m implicitpsf.evaluation.run_eval     # reserved-star eval, 3 methods
uv run python -m implicitpsf.evaluation.sim_truth    # sim-only: truth at star-free grid
```

## Data Flow

1. `des_file_discovery.py` / `download_des_exposures.py`: DES single-epoch FITS
   (`/nfs/turbo/lsa-regier/des`, CCD 31 — note: amp B is dead, MSK bit 8 covers half)
2. `extract_des_stars.py`: SEP detection + DES DR2 cross-match (colors, SPREAD_MODEL
   star/galaxy labels, COADD_OBJECT_ID star ids) -> v2 `.pt` files at
   `/data/scratch/regier/sep_des_stars_v2`; DR2 cones cached at
   `/data/scratch/regier/des_dr2_catalogs`
3. `splits.py`: night-level train/val/test by deterministic hash + reserved-star lists,
   frozen in `manifests/*.json` (build once, commit, never regenerate after test eval)
4. `train_psf.py`: producer/consumer queue, weighted chi^2 loss, per-epoch checkpoints,
   best-on-val selection, cosine LR + warmup, `--patience` early stopping
5. `evaluation/run_eval.py`: per test exposure, all methods fit on identical
   non-reserved clean stars, scored on identical reserved stars -> tidy parquet
6. `simulate.py`: Moffat field with polynomial FWHM/g1/g2 variation; FITS written only
   for val/test exposures (split computed at generation time, must match manifest args)

## Schema and conventions

- v2 batch dict (owned by `datasets.py`): `cutouts`, `variance`, `valid_pixels`,
  `flux`, `positions`, `colors`, `star_types`, `star_ids`
- `star_type`: 0 clean (in loss), 1 star context-only, 2 galaxy, 3 unmatched,
  4 padding (zero flux), 5 saturated
- Coordinates are 0-based SEP pixel centers; FITS/PIFF/PSFEx are 1-based (+1 on write)
- Stamp convention: corner = round(center) - patch//2; offsets in `stamp_offsets`
- Loss = inverse-variance chi^2 with closed-form per-star amplitude (MSE = variance 1)
- The model conditions on color (DR2 g-i; 0 = unknown) and log-flux (brighter-fatter)

## Hard-won gotchas

- psfex segfaults unless LDAC_IMHEAD bytes are space-padded with an `END` card
  (astropy null-pads strings; `baselines/catalogs.py` patches bytes after writeto)
- PSFEx needs `PSF_SAMPLING 0.5` (super-resolved); at 1.0 its model is ~3% small in T;
  render via galsim DES_PSFEx with `method='no_pixel'`
- Data Lab encodes missing photometry as `+inf` (passes `> 0` cuts — check isfinite)
- AstroMatch's `download_des_catalog` is broken against des_dr2.main (flux_g vs
  flux_auto_g); we query via `dl.queryClient` with raw SQL (TAP rejects q3c)
- The batch producer must outlive its queued tensors (fd-shared); shutdown via
  done_event after draining the queue
- `pkill -f` patterns must use the `[x]` bracket trick or they kill the calling shell
