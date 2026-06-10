"""Pixel-level goodness of fit between data stamps and unit-flux model stamps.

The per-star amplitude is solved in closed form by weighted least squares (the same
treatment as the training loss and as PIFF's flux fit), so chi^2 measures shape.
"""

import numpy as np


def reduced_chi2(observed, model, variance, valid):
    """Per-star reduced chi^2 with closed-form amplitude.

    Args:
        observed: (n, k, k) background-subtracted data stamps
        model: (n, k, k) unit-flux model stamps on the same pixel grid
        variance: (n, k, k) per-pixel variance
        valid: (n, k, k) bool, True where pixels are usable

    Returns:
        dict with chi2 (n,), amplitude (n,), n_valid (n,)
    """
    observed = np.asarray(observed, dtype=np.float64).reshape(len(observed), -1)
    model = np.asarray(model, dtype=np.float64).reshape(len(model), -1)
    variance = np.asarray(variance, dtype=np.float64).reshape(len(variance), -1)
    valid = np.asarray(valid, dtype=bool).reshape(len(valid), -1)

    weights = np.where(valid, 1.0 / variance, 0.0)
    n_valid = valid.sum(axis=1)
    if (n_valid == 0).any():
        raise ValueError("star with no valid pixels")

    amplitude = (weights * observed * model).sum(axis=1)
    amplitude = amplitude / (weights * model**2).sum(axis=1)
    residuals = observed - amplitude[:, None] * model
    # one dof consumed by the amplitude fit
    chi2 = (weights * residuals**2).sum(axis=1) / (n_valid - 1)
    return {"chi2": chi2, "amplitude": amplitude, "n_valid": n_valid}
