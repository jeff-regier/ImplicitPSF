"""Fit and render PIFF PSF models for single-CCD exposures.

Configuration mirrors DES Y3 production (Jarvis et al. 2021): per-CCD PixelGrid
model with second-order BasisPolynomial interpolation and chi-square outlier
rejection. Stars come exclusively from our pre-selected catalog.
"""

import galsim
import numpy as np
import piff

PIXEL_SCALE = 0.263


def fit_piff(fits_path, cat_path, out_path, order=2, stamp_size=25, grid_size=17):
    """Fit a PIFF model; returns the fitted piff.PSF (also written to out_path)."""
    config = {
        "input": {
            "image_file_name": str(fits_path),
            "image_hdu": 1,
            "weight_hdu": 3,
            "badpix_hdu": 2,
            "cat_file_name": str(cat_path),
            "cat_hdu": 1,
            "x_col": "x",
            "y_col": "y",
            "stamp_size": stamp_size,
        },
        "psf": {
            "model": {"type": "PixelGrid", "scale": PIXEL_SCALE, "size": grid_size},
            "interp": {"type": "BasisPolynomial", "order": order},
            "outliers": {"type": "Chisq", "nsigma": 4.0, "max_remove": 0.05},
        },
        "verbose": 0,
    }
    psf = piff.process(config)
    psf.write(str(out_path))
    return psf


def render_piff(psf, x_pixel, y_pixel, patch_size=32):
    """Render unit-flux model stamps on the data stamp grid of each star.

    Args:
        psf: a fitted piff.PSF
        x_pixel, y_pixel: (n,) 0-based star centroids
        patch_size: stamp width; grid corner = round(center) - patch_size // 2

    Returns:
        (n, patch_size, patch_size) float64 stamps
    """
    half = patch_size // 2
    stamps = np.zeros((len(x_pixel), patch_size, patch_size))
    for index, (x0, y0) in enumerate(zip(x_pixel, y_pixel, strict=True)):
        x_fits, y_fits = float(x0) + 1.0, float(y0) + 1.0
        corner_x = round(float(x0)) - half + 1  # 1-based stamp corner
        corner_y = round(float(y0)) - half + 1
        bounds = galsim.BoundsI(
            corner_x, corner_x + patch_size - 1, corner_y, corner_y + patch_size - 1
        )
        image = galsim.Image(bounds, wcs=psf.wcs[0])
        psf.draw(x=x_fits, y=y_fits, image=image, flux=1.0)
        stamps[index] = image.array
    return stamps
