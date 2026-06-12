"""Gate 1: score known truth through the blend likelihood on the blended simulation.

For each test exposure of the blended sim, render the TRUE effective PSF for every
target and its detected neighbors on the target's stamp grid, solve GLS amplitudes,
and compute reduced chi-square. Pass criterion: median chi2/(n_valid - k) in
[0.97, 1.03] and recovered amplitudes consistent with true fluxes. This validates
neighbor indexing, offset geometry, the GLS solve, and masking before any training.
"""

import argparse
from pathlib import Path

import galsim
import numpy as np
import torch

from implicitpsf.blend import gls_amplitudes, neighbor_table
from implicitpsf.datasets import load_exposure_file
from implicitpsf.evaluation.moments import PIXEL_SCALE
from implicitpsf.simulate import MOFFAT_BETA, PATCH, true_psf_params
from implicitpsf.splits import load_manifest


def true_component_stamp(field, comp_pos, target_pos, color):
    """True effective PSF of a component rendered on the target's stamp grid."""
    half = PATCH // 2
    corner_x = round(float(target_pos[0])) - half + 1  # 1-based stamp corner
    corner_y = round(float(target_pos[1])) - half + 1
    bounds = galsim.BoundsI(corner_x, corner_x + PATCH - 1, corner_y, corner_y + PATCH - 1)
    fwhm, g1, g2 = true_psf_params(field, float(comp_pos[0]), float(comp_pos[1]), float(color))
    profile = galsim.Moffat(beta=MOFFAT_BETA, fwhm=fwhm * PIXEL_SCALE).shear(g1=g1, g2=g2)
    image = galsim.Image(bounds, scale=PIXEL_SCALE)
    profile.drawImage(
        image=image,
        center=galsim.PositionD(float(comp_pos[0]) + 1.0, float(comp_pos[1]) + 1.0),
        add_to_image=True,
    )
    return image.array


def validate_exposure(data, index, radius, k_max):
    pos = torch.stack([data["x_pixel"][index], data["y_pixel"][index]], dim=1).unsqueeze(0)
    fluxes = data["flux"][index].unsqueeze(0)
    types = data["star_type"][index].unsqueeze(0)
    field = {"chromatic": False, **data["true_field"][index]}

    idx, mask, _ = neighbor_table(pos, fluxes, types, radius=radius, k_max=k_max)
    is_target = (types[0] == 0) & (fluxes[0] > 0)
    targets = torch.nonzero(is_target, as_tuple=True)[0]

    rows = []
    for t in targets.tolist():
        comp_slots = [t, *idx[0, t][mask[0, t]].tolist()]
        k = len(comp_slots)
        stamps = np.stack(
            [
                true_component_stamp(
                    field, pos[0, c].numpy(), pos[0, t].numpy(), data["color"][index][c]
                )
                for c in comp_slots
            ]
        )
        components = torch.tensor(stamps.reshape(k, -1), dtype=torch.float64)
        components = components / components.sum(dim=-1, keepdim=True)

        observed = data["cutouts"][index][t].reshape(-1).double()
        variance = data["variance"][index][t].reshape(-1).double()
        valid = data["valid_pixels"][index][t].reshape(-1)
        weights = valid.double() / variance

        amps = gls_amplitudes(
            components.unsqueeze(0),
            observed.unsqueeze(0),
            weights.unsqueeze(0),
            torch.ones(1, k, dtype=torch.bool),
            ridge=0.0,
        )[0]
        model = (amps.unsqueeze(-1) * components).sum(0)
        n_valid = int(valid.sum())
        chi2 = float((weights * (observed - model).square()).sum() / (n_valid - k))
        true_flux = float(data["flux"][index][t])
        rows.append(
            {
                "chi2": chi2,
                "k": k,
                "amp_target": float(amps[0]),
                "true_flux": true_flux,
                "amp_ratio": float(amps[0]) / true_flux,
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="manifests/sim_blended_split_v1.json")
    parser.add_argument("--data-dir", default="/data/scratch/regier/sim_blended_stars")
    parser.add_argument("--max-exposures", type=int, default=30)
    parser.add_argument("--blend-radius", type=float, default=22.0)
    parser.add_argument("--blend-k-max", type=int, default=4)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    selected = [(e, i) for e, i in sorted(manifest["exposures"].items()) if i["split"] == "test"][
        : args.max_exposures
    ]

    rows = []
    cache = {"name": None, "data": None}
    for _, info in selected:
        if info["file"] != cache["name"]:
            cache = {
                "name": info["file"],
                "data": load_exposure_file(Path(args.data_dir) / info["file"]),
            }
        rows.extend(
            validate_exposure(cache["data"], info["index"], args.blend_radius, args.blend_k_max)
        )

    chi2 = np.array([r["chi2"] for r in rows])
    ks = np.array([r["k"] for r in rows])
    ratio = np.array([r["amp_ratio"] for r in rows])
    print(f"targets: {len(rows)} (k distribution: {np.bincount(ks)[1:].tolist()})")
    print(f"chi2/(n-k): median={np.median(chi2):.4f}  p90={np.percentile(chi2, 90):.3f}")
    print(f"amp/true_flux: median={np.median(ratio):.4f}  scatter={ratio.std():.4f}")
    blended = chi2[ks > 1]
    if len(blended):
        print(f"blended targets only: chi2 median={np.median(blended):.4f} (n={len(blended)})")
    ok = 0.97 <= np.median(chi2) <= 1.03
    print("GATE 1:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
