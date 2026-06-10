# CLAUDE.md

## Project Overview

**ImplicitPSF**: Attention-based implicit PSF modeling for DES single-epoch images.
Each star's PSF is predicted from all other stars in the field via multi-head attention
(`implicitpsf/implicit_psf.py`); a coordinate-MLP decoder captures subpixel structure.

## Commands

```bash
uv run pytest                                        # run tests
uv run ruff check --fix . && uv run ruff format .    # lint + format
uv run python -m implicitpsf.train_psf               # train on extracted DES stars
uv run python -m implicitpsf.extract_des_stars       # SEP star extraction -> .pt files
```

## Data Flow

1. `des_file_discovery.py` lists DES exposures via Astro Data Lab
2. `download_des_exposures.py` bulk-downloads FITS files
3. `extract_des_stars.py` extracts star cutouts with SEP into `.pt` files
   (default location: `/data/scratch/regier/sep_des_stars`)
4. `train_psf.py` trains with a multiprocessing producer/consumer batch queue
   (no Lightning Trainer; `ImplicitPSF` subclasses `LightningModule` for hparams only)

## Conventions

- Batches are dicts with keys `cutouts`, `flux`, `positions`, `star_types`
- `star_types == 0` marks clean stars; only those contribute to the loss
- Zero flux marks padding stars; they are masked out of attention
- The DES acquisition scripts have ruff per-file-ignores pending refactor (see pyproject.toml)
