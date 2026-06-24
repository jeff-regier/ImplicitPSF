"""Automated EM-iteration driver for the proper MCEM contamination correction.

EM is iterative -- it takes tens of iterations to reach a fixed point, so a single E+M step (which
over-corrects, since iteration 0 cleans with the still-contaminated central) is NOT the answer. Each
iteration here:
  E-step    -- clean the training stars with the CURRENT PSF model as the central template
               (mcem_clean.write_kimpute on GPU, ~minutes) -> a fresh cutout_imp dataset, written to
               ONE reused directory (disk stays bounded).
  M-step    -- warm-start train_psf from the current model and partially update on the new
               imputations (generalized/SAEM EM: increase Q, fast from a warm start) -> next model.
  diagnostic -- sim_psf_ee_defect delta-EE@r2 vs the known sim truth, appended to a trajectory CSV.
The cleaning central sharpens each round; the delta-EE trajectory should settle to a fixed point
near truth (cleaning with the true PSF removes exactly the contamination -> SBC-calibrated). We run
it unattended and read the trajectory, not any single iteration.
"""

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

N_FILES = 150


def _env(gpu):
    e = dict(os.environ)
    e["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    e["CUDA_VISIBLE_DEVICES"] = gpu
    return e


def repoint_manifest(base_manifest, data_dir, out_path):
    """Copy the manifest with its data_dir repointed at the freshly-cleaned dataset."""
    with open(base_manifest) as f:
        m = json.load(f)
    m["data_dir"] = data_dir
    with open(out_path, "w") as f:
        json.dump(m, f)
    return out_path


def e_step(central, base_data_dir, clean_dir, n_keep, n_sweeps, gpu):
    """Clean every file with the current central on this chain's single GPU (fast GPU path)."""
    Path(clean_dir).mkdir(parents=True, exist_ok=True)
    cmd = [
        "python",
        "-m",
        "implicitpsf.mcem_clean",
        "--mode",
        "write",
        "--offset",
        "0",
        "--limit",
        str(N_FILES),
        "--n-keep",
        str(n_keep),
        "--n-sweeps",
        str(n_sweeps),
        "--data-dir",
        base_data_dir,
        "--checkpoint",
        central,
        "--out-dir",
        clean_dir,
        "--device",
        "cuda",
    ]
    if subprocess.run(cmd, env=_env(gpu), check=False).returncode != 0:
        raise RuntimeError("e_step cleaner failed")


def m_step(init_ckpt, data_dir, manifest, out_dir, epochs, seed, gpu):
    """Warm-start from init_ckpt and partially fit the new imputations (the M-step)."""
    cmd = [
        "python",
        "-m",
        "implicitpsf.train_psf",
        "--data-dir",
        data_dir,
        "--manifest",
        manifest,
        "--hidden-dim",
        "256",
        "--decoder-dim",
        "128",
        "--n-heads",
        "8",
        "--n-attn-layers",
        "2",
        "--decoder-film",
        "--polar-coords",
        "--n-freqs",
        "8",
        "--galaxy-mode",
        "exclude",
        "--loss-mode",
        "single",
        "--max-epochs",
        str(epochs),
        "--patience",
        str(epochs),
        "--init-checkpoint",
        init_ckpt,
        "--seed",
        str(seed),
        "--out-dir",
        out_dir,
    ]
    subprocess.run(cmd, env=_env(gpu), check=True)
    return str(Path(out_dir) / "best.pt")


def evaluate(ckpt, manifest, data_dir, max_exposures, gpu):
    """delta-EE@r2 of the model PSF vs sim truth at star-free positions."""
    cmd = [
        "python",
        "-m",
        "implicitpsf.sim_psf_ee_defect",
        "--checkpoint",
        ckpt,
        "--manifest",
        manifest,
        "--data-dir",
        data_dir,
        "--psf-model",
        "realistic",
        "--max-exposures",
        str(max_exposures),
    ]
    out = subprocess.run(cmd, env=_env(gpu), check=True, capture_output=True, text=True)
    match = re.search(r"dEE@r2 mean.*=\s*([+-][\d.]+)", out.stdout)
    if not match:
        raise RuntimeError(f"could not parse delta-EE from:\n{out.stdout[-500:]}")
    return float(match.group(1))


def one_iteration(it, central, args, clean_dir, log_path):
    """Run E-step + M-step + diagnostic for one EM iteration; return the new central checkpoint."""
    e_step(central, args.base_data_dir, clean_dir, args.n_keep, args.n_sweeps, args.gpu)
    manifest = repoint_manifest(args.base_manifest, clean_dir, f"{args.work_dir}/manifest.json")
    out_dir = f"{args.work_dir}/model_it{it}"
    new_central = m_step(central, clean_dir, manifest, out_dir, args.epochs, args.seed, args.gpu)
    dee = evaluate(
        new_central, args.base_manifest, args.base_data_dir, args.max_exposures, args.gpu
    )
    with open(log_path, "a") as f:
        f.write(f"{it},{new_central},{dee:+.5f}\n")
    print(f"[EM it{it}] delta-EE = {dee:+.5f}  (central -> {new_central})", flush=True)
    return new_central


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-checkpoint", required=True, help="iteration-0 model (the start)")
    parser.add_argument("--base-data-dir", default="/data/scratch/regier/sim_contamreal_stars")
    parser.add_argument("--base-manifest", default="manifests/sim_contamreal_sub118.json")
    parser.add_argument("--work-dir", default="/data/scratch/regier/mcem_em")
    parser.add_argument("--n-iter", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=12)  # warm-started partial M-step
    parser.add_argument("--n-keep", type=int, default=4)
    parser.add_argument("--n-sweeps", type=int, default=40)
    parser.add_argument("--max-exposures", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", default="1", help="single GPU for this chain (clean+train+eval)")
    args = parser.parse_args()
    Path(args.work_dir).mkdir(parents=True, exist_ok=True)
    clean_dir = f"{args.work_dir}/clean"  # reused every iteration (overwritten) -> bounded disk
    log_path = f"{args.work_dir}/trajectory.csv"
    with open(log_path, "w") as f:
        f.write("iter,checkpoint,dEE_r2\n")
    central = args.init_checkpoint
    for it in range(args.n_iter):
        central = one_iteration(it, central, args, clean_dir, log_path)


if __name__ == "__main__":
    main()
