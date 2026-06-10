"""Rho statistics (Rowe 2010; Jarvis et al. 2016) of PSF residuals with treecorr.

Each test exposure contributes an independent flat-sky catalog (CCD x/y in arcmin);
pair counts accumulate within exposures only — never across, since different
exposures share CCD coordinates but not sky. Quantities follow DES conventions:
e is the distortion, de = e_star - e_model, t_frac = (T_star - T_model) / T_star.

  rho1 = <de de>     rho2 = <e de>      rho3 = <(e t_frac)(e t_frac)>
  rho4 = <de (e t_frac)>                rho5 = <e (e t_frac)>
"""

import numpy as np
import treecorr

ARCMIN_PER_PIXEL = 0.263 / 60.0

RHO_PAIRS = {
    "rho1": ("de", "de"),
    "rho2": ("e", "de"),
    "rho3": ("etf", "etf"),
    "rho4": ("de", "etf"),
    "rho5": ("e", "etf"),
}


def exposure_catalogs(table):
    """Per-exposure treecorr catalogs for the three residual fields.

    Args:
        table: dict with x_pixel, y_pixel, e1, e2, de1, de2, t_frac arrays
    """
    x = np.asarray(table["x_pixel"]) * ARCMIN_PER_PIXEL
    y = np.asarray(table["y_pixel"]) * ARCMIN_PER_PIXEL

    def catalog(g1, g2):
        return treecorr.Catalog(x=x, y=y, g1=g1, g2=g2, x_units="arcmin", y_units="arcmin")

    e1, e2 = np.asarray(table["e1"]), np.asarray(table["e2"])
    de1, de2 = np.asarray(table["de1"]), np.asarray(table["de2"])
    t_frac = np.asarray(table["t_frac"])
    return {
        "e": catalog(e1, e2),
        "de": catalog(de1, de2),
        "etf": catalog(e1 * t_frac, e2 * t_frac),
    }


def rho_statistics(tables, min_sep=0.3, max_sep=15.0, nbins=12):
    """Accumulate rho1-rho5 over a sequence of per-exposure tables.

    Returns dict rho_name -> dict(theta, xip, xim, npairs); theta in arcmin.
    """
    correlations = {
        name: treecorr.GGCorrelation(
            min_sep=min_sep, max_sep=max_sep, nbins=nbins, sep_units="arcmin"
        )
        for name in RHO_PAIRS
    }

    catalogs = [exposure_catalogs(table) for table in tables]
    if not catalogs:
        raise ValueError("no exposures supplied")

    for name, (first, second) in RHO_PAIRS.items():
        gg = correlations[name]
        for index, cats in enumerate(catalogs):
            gg.process(
                cats[first],
                None if first == second else cats[second],
                initialize=(index == 0),
                finalize=(index == len(catalogs) - 1),
            )

    return {
        name: {"theta": gg.meanr, "xip": gg.xip, "xim": gg.xim, "npairs": gg.npairs}
        for name, gg in correlations.items()
    }
