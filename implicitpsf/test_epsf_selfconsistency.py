"""Self-consistency test (Jeff, Jun 21): render a galaxy through our frozen-weights ePSF with NO
shot noise, then recover it with the SAME ePSF. With no PSF mismatch and no noise, a correct
forward model must return the input size to the discretization floor (~-1%). A larger bias is a
pure artifact of how we USE the ePSF (render_at oversampling / convolution), not a real PSF error
-- which would mean the -5% galaxy-size deficit is a recovery artifact, not an under-concentrated
PSF. We recover the SAME injected galaxy two ways: (A) the implicit render_at fine-grid path, and
(B) the truth/PIFF InterpolatedImage->lattice path. B reuses the injection's own representation so
it must hit the floor; the question is whether A agrees.
"""

import glob

import galsim
import numpy as np
import torch

from implicitpsf.baselines.implicit_runner import load_model
from implicitpsf.blend import sample_grid
from implicitpsf.datasets import load_exposure_file, make_batch
from implicitpsf.evaluation.galaxy_fit import OVERSAMPLE, fit_galaxies
from implicitpsf.evaluation.galaxy_recovery import lattice_kernel
from implicitpsf.evaluation.galaxy_recovery_real import inject_anchor_stamp
from implicitpsf.evaluation.moments import PIXEL_SCALE
from implicitpsf.render import render_at
from implicitpsf.simulate import PATCH

CKPT = "checkpoints/real_v6_rff8_ps_s0/best.pt"
DATA = "/data/scratch/regier/sep_des_stars_v2"
RE_TRUE = np.array([1.5, 2.5, 4.0, 6.0])  # pixels


def main():
    model = load_model(CKPT)
    data = load_exposure_file(sorted(glob.glob(f"{DATA}/*.pt"))[0])
    index = 0
    batch = dict(make_batch(data, [index]))
    st = data["star_type"][index].numpy()
    j = int(np.nonzero(st == 0)[0][0])
    x = float(data["x_pixel"][index][j])
    y = float(data["y_pixel"][index][j])
    queries = torch.tensor([[round(x), round(y)]], dtype=torch.float32)
    colors = torch.zeros(1)

    # our ePSF at this position, frozen weights: native (for galsim) + fine (render_at recovery)
    native = render_at(model, batch, queries, colors, oversample=1).numpy()[0]
    fine = render_at(model, batch, queries, colors, oversample=OVERSAMPLE).numpy()[0]
    image = galsim.Image(np.ascontiguousarray(native.astype(np.float64)), scale=PIXEL_SCALE)
    epsf = galsim.InterpolatedImage(image, x_interpolant="lanczos15", normalization="flux")
    grid = sample_grid(PATCH, OVERSAMPLE, torch.device("cpu"), torch.float64).numpy()
    kernel_b = lattice_kernel(epsf, grid)

    rng = np.random.default_rng(0)
    stamps = []
    for re in RE_TRUE:
        gal = {"n": 1.0, "re": float(re), "flux": 1e5, "eta1": 0.0, "eta2": 0.0}
        stamps.append(inject_anchor_stamp(rng, epsf, x, y, gal, noise_sigma=0.0))  # NO noise
    stamps = torch.tensor(np.stack(stamps), dtype=torch.float32)
    var = torch.ones_like(stamps)
    valid = torch.ones_like(stamps, dtype=torch.bool)
    n_arg = torch.ones(len(RE_TRUE))
    init_flux = stamps.sum(dim=(-2, -1)).clamp(min=100.0)
    init_re = torch.full((len(RE_TRUE),), 3.0)

    print(f"query=({x:.0f},{y:.0f}); ePSF sums native={native.sum():.4f} fine={fine.sum():.4f}")
    print("re_true (px):", RE_TRUE)
    arms = [("A render_at (implicit path)", fine), ("B InterpImage (truth/PIFF path)", kernel_b)]
    for name, kern in arms:
        kt = torch.tensor(np.tile(kern, (len(RE_TRUE), 1, 1)), dtype=torch.float32)
        res = fit_galaxies(stamps, var, valid, kt, n_arg, init_flux, init_re)
        re_fit = res["re"].numpy()
        bias = 100.0 * (re_fit - RE_TRUE) / RE_TRUE
        print(f"{name:34s} re_fit={np.round(re_fit,3)}  bias%={np.round(bias,2)}")
    print("\nfloor ~= -1%; if A shows the ~-5% real-data deficit here (no PSF mismatch, no noise),")
    print("the deficit is a render_at forward-model artifact, not an under-concentrated PSF.")


if __name__ == "__main__":
    main()
