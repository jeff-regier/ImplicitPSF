"""Fast test of the contamination hypothesis: split the existing galaxy-recovery size deficit
by whether each anchor 'clean' star is flagged contaminated by Gaia (a sub-arcsecond companion
DES blended in, or RUWE/non_single_star indicating an unresolved binary). If the -7.6% deficit
concentrates in Gaia-contaminated anchors, contamination drives it -- not decoder spectral bias.

No retrain: uses results/galaxy_recovery_real_v6_blend.parquet + per-exposure WCS + Gaia DR3.
"""

import numpy as np
import pandas as pd
import torch
from astropy.io import fits
from astropy.wcs import WCS
from dl import queryClient

from implicitpsf.splits import load_manifest

STAMP_RADIUS_ARCSEC = 4.0  # half the 32 px x 0.263 stamp; a Gaia source here blends the star
SELF_MATCH_ARCSEC = 1.5  # the anchor's own Gaia match
RUWE_CUT = 1.4  # Gaia astrometric-binary / blend indicator
DATA_DIR = "/data/scratch/regier/sep_des_stars_v2"


def wcs_for(batch, index):
    cards = "".join(str(x) for x in np.atleast_1d(batch["wcs_header"][index]).ravel())
    return WCS(fits.Header.fromstring(cards))


def gaia_cone(ra, dec, radius_deg=0.2):
    sql = (
        "SELECT ra, dec, phot_g_mean_mag, ruwe, non_single_star "
        "FROM gaia_dr3.gaia_source "
        f"WHERE q3c_radial_query(ra, dec, {ra}, {dec}, {radius_deg})"
    )
    return queryClient.query(sql=sql, fmt="pandas", timeout=280)


def flag_anchors(anchor_ra, anchor_dec, gaia):
    """Per anchor: contaminated if a 2nd Gaia source within the stamp, or its match is a binary."""
    gr = np.radians(gaia.dec.to_numpy())
    np.cos(gr)
    flags = []
    for ra0, dec0 in zip(anchor_ra, anchor_dec, strict=True):
        ddec = (gaia.dec.to_numpy() - dec0) * 3600.0
        dra = (gaia.ra.to_numpy() - ra0) * 3600.0 * np.cos(np.radians(dec0))
        sep = np.hypot(dra, ddec)
        within_stamp = np.sum(sep < STAMP_RADIUS_ARCSEC)
        match = sep < SELF_MATCH_ARCSEC
        bad_match = bool(
            np.any(
                match & ((gaia.ruwe.to_numpy() > RUWE_CUT) | (gaia.non_single_star.to_numpy() > 0))
            )
        )
        companion = within_stamp >= 2  # the self-match plus at least one more
        no_gaia = within_stamp == 0  # not even a self-match (Gaia-faint; can't vet)
        flags.append({"contaminated": bool(companion or bad_match), "no_gaia": no_gaia})
    return pd.DataFrame(flags)


def main():
    d = pd.read_parquet("results/galaxy_recovery_real_v6_blend.parquet")
    d = d[d.arm == "implicit"].copy()
    man = load_manifest("manifests/split_v1.json")
    rows = []
    for eid, g in d.groupby("exposure_id"):
        info = man["exposures"].get(str(eid))
        if info is None:
            continue
        batch = torch.load(f"{DATA_DIR}/{info['file']}", map_location="cpu", weights_only=False)
        wcs = wcs_for(batch, info["index"])
        sky = wcs.all_pix2world(np.c_[g.x.to_numpy() + 1, g.y.to_numpy() + 1], 1)
        gaia = gaia_cone(float(np.median(sky[:, 0])), float(np.median(sky[:, 1])))
        flags = flag_anchors(sky[:, 0], sky[:, 1], gaia).reset_index(drop=True)
        gg = g.reset_index(drop=True)
        gg["contaminated"] = flags["contaminated"]
        gg["no_gaia"] = flags["no_gaia"]
        rows.append(gg)
        print(
            f"  {eid}: {len(gg)} anchors, {int(flags.contaminated.sum())} contaminated, "
            f"{int(flags.no_gaia.sum())} Gaia-faint, {len(gaia)} Gaia in field"
        )
    out = pd.concat(rows, ignore_index=True)
    out["sb"] = 100 * (out.re_fit - out.re_true) / out.re_true
    vet = out[~out.no_gaia]  # only anchors Gaia can vet
    print("\n=== DEFICIT SPLIT BY GAIA CONTAMINATION (implicit arm) ===")
    print(f"all anchors:            {np.median(out.sb):+.1f}%  (n={len(out)})")
    print(
        f"Gaia-CLEAN anchors:     {np.median(vet[~vet.contaminated].sb):+.1f}%  (n={int((~vet.contaminated).sum())})"
    )
    print(
        f"Gaia-CONTAMINATED:      {np.median(vet[vet.contaminated].sb):+.1f}%  (n={int(vet.contaminated.sum())})"
    )
    print(
        f"Gaia-faint (unvettable):{np.median(out[out.no_gaia].sb):+.1f}%  (n={int(out.no_gaia.sum())})"
    )
    out.to_parquet("results/gaia_split_test.parquet")


if __name__ == "__main__":
    main()
