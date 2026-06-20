"""Generate an empirical-PSF simulation: synthetic CLEAN stars/galaxies rendered through real
fitted PSFEx models (true DECam morphology), to test whether the implicit decoder under-fits real
PSF structure -- the leading explanation for the -7.6% real-data galaxy deficit that does NOT
appear on smooth analytic sims (and is not Gaia-resolvable contamination).

Each sim exposure uses one real exposure's PSFEx model (cycled across a pool fit once up front).
The PSFEx model + its FITS path are stored in true_field so galaxy_recovery's empirical truth arm
renders the same PSF. Reuses simulate.simulate_exposure (now in 'empirical' mode) + pack_records.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from astropy.io import fits

from implicitpsf.baselines.catalogs import write_psfex_ldac
from implicitpsf.baselines.psfex_runner import fit_psfex
from implicitpsf.evaluation.run_eval import exposure_masks
from implicitpsf.simulate import (
    pack_records,
    save_chunk,
    set_empirical_psf,
    set_psf_model,
    simulate_exposure,
)
from implicitpsf.splits import load_manifest, reserved_star_ids

REAL_DATA_DIR = "/data/scratch/regier/sep_des_stars_v2"


def fit_psfex_for_exposure(eid, info, manifest, psf_dir):
    """Fit PSFEx on a real exposure's clean stars; return (psf_path, fits_path). Cached."""
    out = Path(psf_dir) / eid
    cached = list(out.glob("*.psf"))
    batch = torch.load(f"{REAL_DATA_DIR}/{info['file']}", map_location="cpu", weights_only=False)
    idx = info["index"]
    fits_path = str(batch["fits_path"][idx])
    if cached:
        return str(cached[0]), fits_path
    clean, reserved = exposure_masks(batch, idx, reserved_star_ids(manifest, eid))
    fit = clean & ~reserved
    with fits.open(fits_path) as hdul:
        header = hdul["SCI"].header
    out.mkdir(parents=True, exist_ok=True)
    ldac = out / "cat.fits"
    write_psfex_ldac(
        batch["cutouts"][idx].numpy()[fit],
        batch["valid_pixels"][idx].numpy()[fit],
        batch["x_pixel"][idx].numpy()[fit],
        batch["y_pixel"][idx].numpy()[fit],
        batch["flux"][idx].numpy()[fit],
        batch["flux_err"][idx].numpy()[fit],
        batch["snr"][idx].numpy()[fit],
        float(batch["fwhm"][idx]),
        header,
        ldac,
        gain=float(batch["gain"][idx]),
        background_dev=float(batch["skysigma"][idx]),
        saturation=float(header["SATURATE"]),
    )
    return str(fit_psfex(ldac, out)), fits_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="/data/scratch/regier/sim_empirical_stars")
    parser.add_argument("--fits-dir", default="/data/scratch/regier/sim_empirical_fits")
    parser.add_argument("--psf-dir", default="/data/scratch/regier/empirical_psf_models")
    parser.add_argument("--manifest", default="manifests/split_v1.json")
    parser.add_argument("--n-psf-models", type=int, default=40)
    parser.add_argument("--n-exposures", type=int, default=1200)
    parser.add_argument("--exposures-per-file", type=int, default=32)
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    Path(args.fits_dir).mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(args.manifest)
    reals = [
        (eid, info)
        for eid, info in sorted(manifest["exposures"].items())
        if info.get("band") == "r"
    ][: args.n_psf_models]

    print(f"fitting PSFEx on {len(reals)} real exposures...")
    models = []
    for eid, info in reals:
        try:
            models.append(fit_psfex_for_exposure(eid, info, manifest, args.psf_dir))
        except Exception as exc:  # noqa: BLE001 -- skip a bad exposure, keep the pool
            print(f"  PSFEx failed for {eid}: {exc}")
    print(f"got {len(models)} PSFEx models; generating {args.n_exposures} empirical exposures")

    set_psf_model("empirical")
    records, n_chunks = [], 0
    for seed in range(args.n_exposures):
        psf_path, real_fits = models[seed % len(models)]
        set_empirical_psf(psf_path, real_fits)
        record = simulate_exposure(seed, args.fits_dir, galaxy_fraction=0.0)
        record["field"] = {**record["field"], "psfex_path": psf_path, "psfex_fits": real_fits}
        records.append(record)
        if len(records) == args.exposures_per_file:
            save_chunk(records, args.out_dir, 0, n_chunks)
            records, n_chunks = [], n_chunks + 1
    if records:
        save_chunk(records, args.out_dir, 0, n_chunks)
    print("done")


if __name__ == "__main__":
    main()
