"""Train ImplicitPSF on extracted DES stars with a producer/consumer batch queue.

Exposure selection is driven by the frozen split manifest (see splits.py). One
producer process loads .pt files and emits ready batches through a bounded queue;
the consumer (this process) trains, validates, checkpoints, and logs per epoch.
"""

import argparse
import csv
import math
import multiprocessing as mp
import random
import shutil
import time
from pathlib import Path

import torch
from tqdm import tqdm

from implicitpsf.datasets import (
    BATCH_KEYS,
    load_exposure_file,
    make_batch,
    sample_imputations,
    stable_seed,
)
from implicitpsf.implicit_psf import ImplicitPSF
from implicitpsf.provenance import checkpoint_provenance
from implicitpsf.splits import files_for_split, load_manifest

QUEUE_TIMEOUT_SECONDS = 600


class EMA:
    """Exponential moving average of model weights; a shadow copy updated every step."""

    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for key, value in model.state_dict().items():
            shadow = self.shadow[key]
            if value.dtype.is_floating_point:
                shadow.mul_(self.decay).add_(value.detach().float(), alpha=1.0 - self.decay)
            else:
                shadow.copy_(value)

    def state_for(self, model):
        """Shadow weights cast back to the model's parameter dtypes."""
        reference = model.state_dict()
        return {key: value.to(reference[key].dtype) for key, value in self.shadow.items()}


def split_file_indices(manifest, split, band):
    """Map file name -> exposure indices for a split, optionally one band."""
    files = files_for_split(manifest, split)
    if band is None:
        return files
    by_band = {}
    for info in manifest["exposures"].values():
        if info["split"] == split and info["band"] == band:
            by_band.setdefault(info["file"], []).append(info["index"])
    return {name: sorted(indices) for name, indices in sorted(by_band.items())}


def file_batches(data_dir, file_name, indices, batch_size, phase, epoch, shuffle):
    """Batches from one file, restricted to manifest-selected exposures."""
    data = load_exposure_file(Path(data_dir) / file_name)
    sample_imputations(data, stable_seed("imp", file_name, epoch))  # MCEM: fresh imputation/epoch
    indices = list(indices)
    if shuffle:
        random.Random(stable_seed("exposures", file_name, epoch)).shuffle(indices)

    batches = []
    for start in range(0, len(indices), batch_size):
        batch = make_batch(data, indices[start : start + batch_size])
        batch["phase"] = phase
        batch["epoch"] = epoch
        batches.append(batch)
    return batches


def count_batches(files, batch_size):
    return sum(math.ceil(len(indices) / batch_size) for indices in files.values())


def batch_producer(
    data_dir, train_files, val_files, batch_size, max_epochs, batch_queue, done_event
):
    """Producer process: emits all epochs' batches through the bounded queue.

    Stays alive until the consumer sets done_event: queued tensors are shared by
    file descriptor and become invalid if this process exits while they are in flight.
    """
    for epoch in range(max_epochs):
        file_names = list(train_files)
        random.Random(stable_seed("files", epoch)).shuffle(file_names)
        for file_name in file_names:
            batches = file_batches(
                data_dir, file_name, train_files[file_name], batch_size, "train", epoch, True
            )
            for batch in batches:
                batch_queue.put(batch)
        for file_name, indices in val_files.items():
            for batch in file_batches(
                data_dir, file_name, indices, batch_size, "val", epoch, False
            ):
                batch_queue.put(batch)
    done_event.wait()


def run_epoch_phase(
    model, optimizer, batch_queue, n_batches, phase, epoch, device, zero_color=False, ema=None
):
    """Consume one epoch's worth of batches for a phase; train if an optimizer is given."""
    losses = []
    with tqdm(total=n_batches, desc=f"{phase} {epoch}", ncols=120) as pbar:
        while len(losses) < n_batches:
            batch_data = batch_queue.get(timeout=QUEUE_TIMEOUT_SECONDS)
            if batch_data["phase"] != phase or batch_data["epoch"] != epoch:
                raise RuntimeError(f"out-of-order batch in {phase} epoch {epoch}")

            batch = {key: batch_data[key].to(device) for key in BATCH_KEYS}
            if "clean_psf" in batch_data:  # contamination-correction target (sim only)
                batch["clean_psf"] = batch_data["clean_psf"].to(device)
            if zero_color:
                batch["colors"] = torch.zeros_like(batch["colors"])
            loss = model.get_loss(batch)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if ema is not None:
                    ema.update(model)

            losses.append(loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.3e}", "queue": batch_queue.qsize()})
            pbar.update(1)
    return sum(losses) / len(losses)


def make_scheduler(optimizer, max_epochs, warmup_epochs=1):
    """Cosine decay with linear warmup, stepped once per epoch."""

    def schedule(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def save_checkpoint(model, optimizer, epoch, val_loss, out_dir, provenance):
    # overwrite a single rolling checkpoint; per-epoch files blow past the home quota
    path = Path(out_dir) / "last.pt"
    torch.save(
        {
            "epoch": epoch,
            "val_loss": val_loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "hyperparameters": dict(model.hparams),
            "provenance": provenance,
        },
        path,
    )
    return path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="/data/scratch/regier/sep_des_stars_v2")
    parser.add_argument("--manifest", default="manifests/split_v1.json")
    parser.add_argument("--out-dir", default="checkpoints/run")
    parser.add_argument("--band", default=None, help="restrict training to one band")
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--decoder-dim", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-attn-layers", type=int, default=1)
    parser.add_argument("--decoder-film", action="store_true")
    parser.add_argument("--diagonal-coords", action="store_true")
    parser.add_argument("--polar-coords", action="store_true")
    parser.add_argument("--n-freqs", type=int, default=8, help="decoder Fourier frequencies")
    parser.add_argument("--siren", action="store_true", help="SIREN (sin) decoder activations")
    parser.add_argument("--siren-omega", type=float, default=30.0, help="SIREN omega_0")
    parser.add_argument(
        "--rff-sigma", type=float, default=None, help="tuned-sigma random Fourier features"
    )
    parser.add_argument(
        "--analytic-core", action="store_true", help="add a context-predicted Gaussian core"
    )
    parser.add_argument(
        "--activation", default="relu", choices=["relu", "gelu"], help="decoder hidden activation"
    )
    parser.add_argument(
        "--spectral-norm", action="store_true", help="spectral-normalize the decoder Linears"
    )
    parser.add_argument(
        "--decoder-residual", action="store_true", help="skip connections in the FiLM decoder"
    )
    parser.add_argument("--loss-mode", default="single", choices=["single", "blend", "clean"])
    parser.add_argument("--blend-radius", type=float, default=22.0)
    parser.add_argument("--blend-k-max", type=int, default=4)
    parser.add_argument(
        "--galaxy-mode", default="exclude", choices=["exclude", "mask", "component"]
    )
    parser.add_argument("--chi2-cap", type=float, default=None)
    parser.add_argument(
        "--blend-max-targets",
        type=int,
        default=None,
        help="cap scored targets per step (memory); random subsets are unbiased",
    )
    parser.add_argument(
        "--point-source-context",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="attend only to point sources (stars) as context; gate out galaxies "
        "(production design; --no-point-source-context restores the galaxies-in-context ablation)",
    )
    parser.add_argument(
        "--cnn-encoder",
        action="store_true",
        help="encode context stamps with a CNN (spatial inductive bias) instead of a flat MLP; "
        "better at filtering local contamination for the clean-target correction",
    )
    parser.add_argument("--context-dropout-max", type=float, default=0.5)
    parser.add_argument("--no-attention", action="store_true")
    parser.add_argument(
        "--zero-color", action="store_true", help="ablation: erase color conditioning"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=25,
        help="stop after this many epochs without val improvement",
    )
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=None,
        help="if set, validate and checkpoint exponential-moving-average weights",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--queue-size", type=int, default=64)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    manifest = load_manifest(args.manifest)
    train_files = split_file_indices(manifest, "train", args.band)
    val_files = split_file_indices(manifest, "val", args.band)
    train_batches = count_batches(train_files, args.batch_size)
    val_batches = count_batches(val_files, args.batch_size)
    print(f"train: {len(train_files)} files, {train_batches} batches/epoch")
    print(f"val:   {len(val_files)} files, {val_batches} batches/epoch")
    if train_batches == 0 or val_batches == 0:
        raise RuntimeError("a split has no batches; check the manifest and band filter")

    batch_queue = mp.Queue(maxsize=args.queue_size)
    done_event = mp.Event()
    producer = mp.Process(
        target=batch_producer,
        args=(
            args.data_dir,
            train_files,
            val_files,
            args.batch_size,
            args.max_epochs,
            batch_queue,
            done_event,
        ),
        daemon=True,
    )
    producer.start()

    model = ImplicitPSF(
        patch_size=32,
        hidden_dim=args.hidden_dim,
        decoder_dim=args.decoder_dim,
        n_heads=args.n_heads,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        use_attention=not args.no_attention,
        context_dropout_max=args.context_dropout_max,
        n_attn_layers=args.n_attn_layers,
        decoder_film=args.decoder_film,
        diagonal_coords=args.diagonal_coords,
        polar_coords=args.polar_coords,
        n_freqs=args.n_freqs,
        siren=args.siren,
        siren_omega=args.siren_omega,
        rff_sigma=args.rff_sigma,
        analytic_core=args.analytic_core,
        activation=args.activation,
        spectral_norm=args.spectral_norm,
        decoder_residual=args.decoder_residual,
        loss_mode=args.loss_mode,
        blend_radius=args.blend_radius,
        blend_k_max=args.blend_k_max,
        galaxy_mode=args.galaxy_mode,
        chi2_cap=args.chi2_cap,
        blend_max_targets=args.blend_max_targets,
        point_source_context=args.point_source_context,
        cnn_encoder=args.cnn_encoder,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"device: {device}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = make_scheduler(optimizer, args.max_epochs)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(["epoch", "train_loss", "val_loss", "lr", "seconds"])

    ema = EMA(model, args.ema_decay) if args.ema_decay is not None else None
    provenance = checkpoint_provenance(manifest=args.manifest)
    best_val = float("inf")
    epochs_since_best = 0
    for epoch in range(args.max_epochs):
        start = time.time()
        model.train()
        train_loss = run_epoch_phase(
            model,
            optimizer,
            batch_queue,
            train_batches,
            "train",
            epoch,
            device,
            zero_color=args.zero_color,
            ema=ema,
        )
        # validate and checkpoint the EMA weights; keep raw weights for the next epoch
        raw_state = None
        if ema is not None:
            raw_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            model.load_state_dict(ema.state_for(model))
        model.eval()
        with torch.no_grad():
            val_loss = run_epoch_phase(
                model,
                None,
                batch_queue,
                val_batches,
                "val",
                epoch,
                device,
                zero_color=args.zero_color,
            )

        lr = scheduler.get_last_lr()[0]
        scheduler.step()
        seconds = time.time() - start
        print(f"epoch {epoch}: train={train_loss:.4e} val={val_loss:.4e} lr={lr:.2e}")
        with open(log_path, "a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow([epoch, train_loss, val_loss, lr, round(seconds, 1)])

        checkpoint_path = save_checkpoint(model, optimizer, epoch, val_loss, out_dir, provenance)
        if raw_state is not None:
            model.load_state_dict(raw_state)  # resume training from raw, not EMA, weights
        if val_loss < best_val:
            best_val = val_loss
            epochs_since_best = 0
            shutil.copy(checkpoint_path, out_dir / "best.pt")
        else:
            epochs_since_best += 1
            if epochs_since_best >= args.patience:
                print(f"early stop at epoch {epoch}: no val improvement in {args.patience}")
                break

    # unblock a producer stuck on put() before signalling shutdown
    while not batch_queue.empty():
        batch_queue.get_nowait()
    done_event.set()
    producer.join(timeout=30)
    print(f"done; best val loss {best_val:.4e}; checkpoints in {out_dir}")


if __name__ == "__main__":
    main()
