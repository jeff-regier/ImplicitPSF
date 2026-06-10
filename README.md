# ImplicitPSF

Attention-based implicit PSF modeling for DES single-epoch images. Each star's PSF is
predicted from all other stars in the field via multi-head attention, with a
coordinate-MLP decoder that captures subpixel structure.

## Setup

```bash
uv sync
uv run prek install
```

## Pipeline

```bash
uv run python -m implicitpsf.des_file_discovery      # list DES exposures via Data Lab
uv run python -m implicitpsf.download_des_exposures  # bulk download FITS files
uv run python -m implicitpsf.extract_des_stars       # SEP star extraction -> .pt files
uv run python -m implicitpsf.train_psf               # train the PSF model
```

## Development

```bash
uv run pytest
uv run ruff check --fix . && uv run ruff format .
```
