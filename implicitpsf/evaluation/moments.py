"""HSM adaptive-moment measurement of star and PSF-model stamps.

Conventions: T = 2 * sigma^2 in arcsec^2 (DES Y3 currency); e1, e2 are distortions
(galsim observed_shape.e1/e2), matching the rho-statistics definitions in
Jarvis et al. (2021). Centroids are returned in 0-based stamp pixel coordinates.
"""

import galsim
import numpy as np

PIXEL_SCALE = 0.263  # arcsec per DECam pixel

MOMENT_FIELDS = ("T", "e1", "e2", "centroid_x", "centroid_y", "flag")


def hsm_moments(stamps, pixel_scale=PIXEL_SCALE, valid_pixels=None):
    """Adaptive moments for a stack of stamps.

    Args:
        stamps: (n, k, k) array of star or model images
        pixel_scale: arcsec per pixel
        valid_pixels: optional (n, k, k) bool; False pixels (masked regions, e.g.
            the dead amplifier) are excluded from the fit — measuring raw stamps
            that contain masked garbage produces wildly corrupted shapes

    Returns:
        dict of (n,) arrays: T (arcsec^2), e1, e2 (distortion), centroid_x,
        centroid_y (0-based stamp pixels), flag (0 = success)
    """
    stamps = np.asarray(stamps, dtype=np.float64)
    results = {field: np.full(len(stamps), np.nan) for field in MOMENT_FIELDS}
    results["flag"] = np.zeros(len(stamps), dtype=np.int32)

    for index, stamp in enumerate(stamps):
        image = galsim.Image(np.ascontiguousarray(stamp), scale=pixel_scale)
        badpix = None
        if valid_pixels is not None:
            bad = np.ascontiguousarray((~valid_pixels[index]).astype(np.int32))
            badpix = galsim.Image(bad, scale=pixel_scale)
        shape_data = galsim.hsm.FindAdaptiveMom(image, badpix=badpix, strict=False)
        if shape_data.error_message != "":
            results["flag"][index] = 1
            continue
        sigma_arcsec = shape_data.moments_sigma * pixel_scale
        results["T"][index] = 2 * sigma_arcsec**2
        results["e1"][index] = shape_data.observed_shape.e1
        results["e2"][index] = shape_data.observed_shape.e2
        # galsim images are 1-based; convert to 0-based stamp indices
        results["centroid_x"][index] = shape_data.moments_centroid.x - 1.0
        results["centroid_y"][index] = shape_data.moments_centroid.y - 1.0

    return results
