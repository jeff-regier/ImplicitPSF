"""Empirical test of Jeff's background-misestimation hypothesis (Jun 21): blending wings from
bright/saturated/large objects elevate the local background under 'clean' stars, leaving a
constant pedestal that broadens the fitted PSF and biases recovered galaxy sizes small. Pure
real-data diagnostic -- no model needed for the first cut. For each clean star measure (a) the
background pedestal (outer-frame median of its cutout, sky-subtracted so ~0 if well-estimated),
(b) a smooth-fit residual (poorly-fit proxy), then correlate both with proximity to the nearest
bright and nearest saturated object in the same exposure.
"""

import glob

import numpy as np
import torch
from scipy.spatial import cKDTree

DATA = "/data/scratch/regier/sep_des_stars_v2"
PATCH = 32
BORDER = 4  # outer frame width (pixels) used as the local-background estimate


def border_mask():
    m = np.zeros((PATCH, PATCH), dtype=bool)
    m[:BORDER] = m[-BORDER:] = m[:, :BORDER] = m[:, -BORDER:] = True
    return m


def per_exposure(data, i, frame):
    """Per-clean-star (chi2, pedestal, dist_bright, dist_sat, flux) for one exposure.

    chi2 = goodness of fit against the exposure's flux-scaled mean PSF, variance-weighted (so it
    is ~1 for a well-fit star regardless of brightness; >>1 for stars with extra structure).
    """
    st = data["star_type"][i].numpy()
    clean = np.nonzero(st == 0)[0]
    if clean.size < 8:
        return None
    cut = data["cutouts"][i].numpy()
    var = data["variance"][i].numpy()
    vp = data["valid_pixels"][i].numpy().astype(bool)
    x = data["x_pixel"][i].numpy()
    y = data["y_pixel"][i].numpy()
    flux = data["flux"][i].numpy()
    det = st != 4
    bright = det & (flux > np.percentile(flux[det], 90))
    sat = st == 5
    pos = np.column_stack([x, y])
    tree_b = cKDTree(pos[bright]) if bright.sum() else None
    tree_s = cKDTree(pos[sat]) if sat.sum() else None
    # unit-flux empirical PSF template = median over clean stars of cutout/flux
    fpos = flux[clean] > 0
    template = np.median(cut[clean[fpos]] / flux[clean[fpos], None, None], axis=0)
    out = []
    for j in clean:
        m = frame & vp[j]
        good = vp[j] & (var[j] > 0)
        if m.sum() < 20 or good.sum() < 100 or flux[j] <= 0:
            continue
        model = flux[j] * template
        chi2 = float(np.mean((cut[j][good] - model[good]) ** 2 / var[j][good]))
        pedestal = float(np.median(cut[j][m]))
        db = _nearest_excluding_self(tree_b, pos[j])  # nearest OTHER bright object
        ds = _nearest_excluding_self(tree_s, pos[j])
        out.append((chi2, pedestal, db, ds, float(flux[j])))
    return out


def _nearest_excluding_self(tree, p):
    """Distance to the nearest catalog source that is not p itself (drops the 0-distance self)."""
    if tree is None:
        return np.inf
    dd, _ = tree.query(p, k=min(2, tree.n))
    dd = np.atleast_1d(dd)
    nonself = dd[dd > 0.5]
    return float(nonself[0]) if nonself.size else np.inf


def main():
    frame = border_mask()
    files = sorted(glob.glob(f"{DATA}/*.pt"))[:4]
    rows = []
    for f in files:
        data = torch.load(f, weights_only=False, map_location="cpu")
        for i in range(data["star_type"].shape[0]):
            r = per_exposure(data, i, frame)
            if r:
                rows.extend(r)
    a = np.array(rows)  # chi2, pedestal, dist_bright(excl self), dist_sat, flux
    chi2, pedestal, db, ds, flux = a.T
    print(f"clean stars analyzed: {len(a)}  (chi2 vs mean PSF: p50 {np.median(chi2):.2f})")
    print("\n=== brightness-controlled: within each flux quartile, near vs far from a bright "
          "neighbour (excl. self) ===")
    qedges = np.percentile(flux, [0, 25, 50, 75, 100])
    print(f"{'flux bin':>20} | {'near<60px chi2':>16} (n) | {'far>=200px chi2':>16} (n)")
    for k in range(4):
        binsel = (flux >= qedges[k]) & (flux < qedges[k + 1] + (k == 3))
        near = binsel & (db < 60)
        far = binsel & (db >= 200)
        if near.sum() > 20 and far.sum() > 20:
            print(f"[{qedges[k]:8.0f},{qedges[k+1]:8.0f}] | "
                  f"{np.median(chi2[near]):8.2f}  ({near.sum():5d}) | "
                  f"{np.median(chi2[far]):8.2f}  ({far.sum():5d})")
    print("\n=== brightness-controlled: near a SATURATED star (within 150px) vs far ===")
    for k in range(4):
        binsel = (flux >= qedges[k]) & (flux < qedges[k + 1] + (k == 3))
        near = binsel & (ds < 150)
        far = binsel & (ds >= 400)
        if near.sum() > 20 and far.sum() > 20:
            print(f"[{qedges[k]:8.0f},{qedges[k+1]:8.0f}] | sat-near "
                  f"{np.median(chi2[near]):6.2f} ({near.sum():4d}) | far "
                  f"{np.median(chi2[far]):6.2f} ({far.sum():5d})")


if __name__ == "__main__":
    main()
