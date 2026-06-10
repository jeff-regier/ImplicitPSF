"""Fit and render PSFEx models via the psfex binary and galsim.des.DES_PSFEx.

PSFEx internal sample selection is disabled as far as its config allows: the LDAC
catalog we synthesize already contains exactly the fitting stars.
"""

import shutil
import subprocess
from pathlib import Path

import galsim
import galsim.des
import numpy as np

# DES-style super-resolved basis: at native sampling (PSF_SAMPLING 1.0) the fitted
# model is systematically ~3% too small in T (interpolation-kernel sharpening)
PSFEX_CONFIG = """
BASIS_TYPE      PIXEL_AUTO
PSF_SIZE        45,45
PSF_SAMPLING    0.5
PSFVAR_KEYS     X_IMAGE,Y_IMAGE
PSFVAR_GROUPS   1,1
PSFVAR_DEGREES  2
SAMPLE_AUTOSELECT  N
SAMPLE_MINSN       5
SAMPLE_MAXELLIP    1.0
SAMPLE_FWHMRANGE   0.5,30.0
SAMPLE_VARIABILITY 1.0
CHECKPLOT_TYPE  NONE
CHECKIMAGE_TYPE NONE
WRITE_XML       N
VERBOSE_TYPE    QUIET
"""


def fit_psfex(ldac_path, out_dir):
    """Run psfex on a synthesized LDAC catalog; returns the .psf model path."""
    binary = shutil.which("psfex")
    if binary is None:
        raise RuntimeError("psfex binary not found; install it (e.g. conda-forge psfex)")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = out_dir / "psfex.conf"
    config_path.write_text(PSFEX_CONFIG)

    subprocess.run(
        [binary, str(ldac_path), "-c", str(config_path), "-PSF_DIR", str(out_dir)],
        check=True,
        capture_output=True,
    )
    psf_path = out_dir / (Path(ldac_path).stem + ".psf")
    if not psf_path.exists():
        raise FileNotFoundError(f"psfex did not produce {psf_path}")
    return psf_path


def render_psfex(psf_path, fits_path, x_pixel, y_pixel, patch_size=32):
    """Render unit-flux PSFEx model stamps on the data stamp grid of each star.

    galsim.des.DES_PSFEx with the image file returns world-coordinate profiles that
    already include the pixel response, hence method='no_pixel' on draw.
    """
    des_psfex = galsim.des.DES_PSFEx(str(psf_path), image_file_name=str(fits_path))

    stamps = np.zeros((len(x_pixel), patch_size, patch_size))
    for index, (x_raw, y_raw) in enumerate(zip(x_pixel, y_pixel, strict=True)):
        x0, y0 = float(x_raw), float(y_raw)
        position = galsim.PositionD(x0 + 1.0, y0 + 1.0)
        profile = des_psfex.getPSF(position)
        # star lands at 1-based stamp coord 17 + frac; image true center is 16.5
        offset = (x0 - round(x0) + 0.5, y0 - round(y0) + 0.5)
        image = profile.drawImage(
            nx=patch_size,
            ny=patch_size,
            wcs=des_psfex.getLocalWCS(position),
            method="no_pixel",
            offset=offset,
        )
        stamps[index] = image.array / image.array.sum()
    return stamps
