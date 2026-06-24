"""Do real 'clean' training stars get broader near artifacts (saturated-star bleed, masked regions,
amp boundary)? If so, under-masked artifacts broaden the learned PSF -- a real-data-only contributor
the contamination sim lacks, explaining real deficit > sim deficit. We measure each clean star's
second-moment size T (background-subtracted, over valid pixels) and its outer-wing flux fraction,
then bin by brightness (brighter-fatter is a real effect, so control for it) and ask whether size /
wings rise with: distance to nearest saturated star, masked-pixel fraction, distance to amp boundary.
"""

import glob

import numpy as np

from implicitpsf.datasets import load_exposure_file
from implicitpsf.simulate import PATCH

DATA = "/data/scratch/regier/sep_des_stars_v2"
CCD_MID_X = 1024.0
N_EXPOSURES = 8


def star_size_and_wing(cut, val):
    """Background-subtracted second-moment size T (px^2) and outer-annulus (4-8px) flux fraction."""
    s = cut.astype(np.float64)
    edge = np.concatenate([s[0], s[-1], s[:, 0], s[:, -1]])
    s = (s - np.median(edge)) * val
    yy, xx = np.mgrid[0:PATCH, 0:PATCH]
    pos = np.clip(s, 0, None)
    tot = pos.sum() + 1e-12
    cx = (xx * pos).sum() / tot
    cy = (yy * pos).sum() / tot
    rr = np.hypot(xx - cx, yy - cy)
    flux = s.sum() + 1e-12
    T = ((rr**2) * pos).sum() / tot
    wing = s[(rr >= 4) & (rr <= 8)].sum() / flux
    return T, wing


def nearest_saturated(x, y, sat_x, sat_y):
    if len(sat_x) == 0:
        return np.full(len(x), np.inf)
    d = np.hypot(x[:, None] - sat_x[None, :], y[:, None] - sat_y[None, :])
    return d.min(axis=1)


def collect():
    rows = []
    for path in sorted(glob.glob(f"{DATA}/*.pt"))[:1]:
        data = load_exposure_file(path)
        for index in range(min(N_EXPOSURES, len(data["cutouts"]))):
            st = data["star_type"][index].numpy()
            x = data["x_pixel"][index].numpy()
            y = data["y_pixel"][index].numpy()
            flux = data["flux"][index].numpy()
            cut = data["cutouts"][index].numpy()
            val = data["valid_pixels"][index].numpy()
            clean = np.nonzero(st == 0)[0]
            sat = np.nonzero(st == 5)[0]
            d_sat = nearest_saturated(x[clean], y[clean], x[sat], y[sat])
            for k, j in enumerate(clean):
                T, wing = star_size_and_wing(cut[j], val[j])
                masked = 1.0 - val[j].mean()
                rows.append((flux[j], T, wing, masked, d_sat[k], abs(x[j] - CCD_MID_X)))
    return np.array(rows)


def report(rows):
    flux, T, wing, masked, dsat, damp = rows.T
    fb = np.quantile(flux, [0, 0.25, 0.5, 0.75, 1.0])
    print(f"{len(rows)} clean stars over {N_EXPOSURES} exposures. Size T (px^2) by flux quartile,")
    print("split by proximity to a saturated star (near <100px vs far >=100px):")
    print(f"  {'flux quartile':16} {'T near-sat':>11} {'T far-sat':>11} {'wing near':>10} {'wing far':>10}")
    for q in range(4):
        m = (flux >= fb[q]) & (flux < fb[q + 1] if q < 3 else flux <= fb[q + 1])
        near = m & (dsat < 100)
        far = m & (dsat >= 100)
        tn = np.median(T[near]) if near.sum() else np.nan
        tf = np.median(T[far]) if far.sum() else np.nan
        wn = np.median(wing[near]) if near.sum() else np.nan
        wf = np.median(wing[far]) if far.sum() else np.nan
        print(f"  Q{q + 1} n={m.sum():<11} {tn:>11.3f} {tf:>11.3f} {wn:>10.4f} {wf:>10.4f}")
    print(f"\nmasked-pixel fraction: median {np.median(masked):.3f}, "
          f"90th pct {np.quantile(masked, 0.9):.3f}, max {masked.max():.3f}")
    hi = masked > np.quantile(masked, 0.75)
    print(f"size T: low-masked median {np.median(T[~hi]):.3f} vs high-masked {np.median(T[hi]):.3f}")
    print(f"stars within 100px of a saturated star: {(dsat < 100).mean() * 100:.1f}%")


if __name__ == "__main__":
    report(collect())
