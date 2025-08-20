#!/usr/bin/env python3

import glob
import multiprocessing as mp
import queue
import random
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from implicit_psf import ImplicitPSF

warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*IterableDataset.*has.*__len__.*defined.*")


def get_file_exposure_count(file_path):
    """Get actual exposure count from file data (filename pattern is unreliable)"""
    try:
        data = torch.load(file_path, weights_only=False)
        return data["cutouts"].shape[0]  # First dimension is number of exposures
    except Exception as e:
        raise ValueError(f"Could not read exposure count from {file_path}: {e}")


def compute_dataset_length(file_paths, batch_size):
    """
    Compute total number of batches by processing each file individually.
    batch_size refers to number of exposures per batch.
    """
    total_batches = 0
    for file_path in file_paths:
        exposure_count = get_file_exposure_count(file_path)
        file_batches = exposure_count // batch_size  # Only full batches
        total_batches += file_batches
    return total_batches


def process_file_batches(file_path, batch_size, phase, epoch, shuffle=False):
    """Process a file into batches"""
    # Load entire file - tensors are already in correct format
    data = torch.load(file_path, weights_only=False)

    # Get tensor data directly
    cutouts = data["cutouts"]  # Shape: (n_exposures, 512, 32, 32)
    flux = data["flux"]  # Shape: (n_exposures, 512)
    x_pixel = data["x_pixel"]  # Shape: (n_exposures, 512)
    y_pixel = data["y_pixel"]  # Shape: (n_exposures, 512)

    # Create positions tensor
    positions = torch.stack([x_pixel, y_pixel], dim=2)  # Shape: (n_exposures, 512, 2)

    # Handle star types - should already be uint8 integers
    star_types = data["star_type"]  # Already a tensor of uint8

    n_exposures = cutouts.shape[0]

    # Handle shuffling for training
    if shuffle:
        epoch_seed = hash((file_path, epoch)) % (2**32)
        exposure_random = random.Random(epoch_seed)
        exposure_indices = list(range(n_exposures))
        exposure_random.shuffle(exposure_indices)
    else:
        exposure_indices = list(range(n_exposures))

    # Create batches by slicing tensors
    batches = []
    for batch_start in range(0, n_exposures, batch_size):
        batch_end = min(batch_start + batch_size, n_exposures)

        if batch_end - batch_start == batch_size:  # Only full batches
            # Get indices for this batch
            batch_indices = exposure_indices[batch_start:batch_end]

            # Convert tensors to numpy arrays for safer multiprocessing serialization
            batch_serialized = {
                "phase": phase,
                "epoch": epoch,
                "data": {
                    "cutouts": cutouts[batch_indices].cpu().numpy(),  # (batch_size, 512, 32, 32)
                    "flux": flux[batch_indices].cpu().numpy(),  # (batch_size, 512)
                    "positions": positions[batch_indices].cpu().numpy(),  # (batch_size, 512, 2)
                    "star_types": star_types[batch_indices].cpu().numpy(),  # (batch_size, 512)
                },
            }
            batches.append(batch_serialized)

    return batches


def batch_worker(train_files, val_files, batch_size, max_epochs, batch_queue):
    """Predictive worker with pipelined batch processing using single queue"""
    # Generate complete training schedule for all epochs
    for epoch in range(max_epochs):

        # TRAINING PHASE - put batches in queue
        # Training: randomize file order each epoch
        train_files_epoch = train_files.copy()
        epoch_seed = hash((tuple(train_files), epoch)) % (2**32)
        file_random = random.Random(epoch_seed)
        file_random.shuffle(train_files_epoch)

        for file_path in train_files_epoch:
            # Process training file with shuffling
            train_batches = process_file_batches(
                file_path, batch_size, "train", epoch, shuffle=True
            )

            # Put all batches from this file into single queue
            for batch in train_batches:
                batch_queue.put(batch)

        # VALIDATION PHASE - put batches in same queue (pipelined)
        for file_path in val_files:
            # Process validation file without shuffling
            val_batches = process_file_batches(file_path, batch_size, "val", epoch, shuffle=False)

            # Put all batches from this file into same queue
            for batch in val_batches:
                batch_queue.put(batch)


def convert_batch_to_tensors(batch_data, device):
    """Convert numpy batch data to tensors and move to device"""
    data = batch_data["data"]
    cutouts = torch.from_numpy(data["cutouts"]).to(device)
    flux = torch.from_numpy(data["flux"]).to(device)
    positions = torch.from_numpy(data["positions"]).to(device)
    star_types = torch.from_numpy(data["star_types"]).to(device)
    return (cutouts, flux, positions, star_types)


def setup_data_files(data_dir, file_pattern, max_files, train_split, seed):
    """Set up train and validation file lists"""
    print(f"Setting up DES data from {data_dir}...")

    # Find PyTorch files with the specified pattern
    pattern_path = str(Path(data_dir) / file_pattern)
    all_files = sorted(glob.glob(pattern_path))

    if not all_files:
        raise FileNotFoundError(
            f"No PyTorch files found in {data_dir} with pattern '{file_pattern}'"
        )

    if max_files:
        all_files = all_files[:max_files]

    print(f"Found {len(all_files)} PyTorch files")

    # Split files between train and validation
    np.random.seed(seed)
    shuffled_files = np.random.permutation(all_files)

    n_train_files = int(len(shuffled_files) * train_split)
    train_files = shuffled_files[:n_train_files].tolist()
    val_files = shuffled_files[n_train_files:].tolist()

    print(f"Using {len(train_files)} files for training, {len(val_files)} files for validation")

    return train_files, val_files


def main():
    torch.manual_seed(42)

    # Configuration
    image_size = 10000  # DES survey image size
    patch_size = 32  # DES cutout size
    max_epochs = 20
    batch_size = 8  # Real DES files have many exposures each
    seed = 42
    data_dir = "/data/scratch/regier/sep_des_stars"
    file_pattern = "desstars_*.pt"  # Only real DES files, not test files
    train_split = 0.9
    queue_size = 16

    # Set up data files
    train_files, val_files = setup_data_files(data_dir, file_pattern, None, train_split, seed)

    # Compute dataset lengths for progress bars
    train_batches_per_epoch = compute_dataset_length(train_files, batch_size)
    val_batches_per_epoch = compute_dataset_length(val_files, batch_size)

    # Create and start background worker
    print(f"Creating predictive worker for {max_epochs} epochs...")
    batch_queue = mp.Queue(maxsize=queue_size)

    worker_process = mp.Process(
        target=batch_worker,
        args=(train_files, val_files, batch_size, max_epochs, batch_queue),
        daemon=True,
    )
    worker_process.start()

    # Create model
    model = ImplicitPSF(
        patch_size=patch_size,
        image_size=image_size,  # Full survey image size for position encoding
        background_level=1.0,
        hidden_dim=256,
        n_heads=4,
        learning_rate=1e-4,
        use_attention=True,  # Test attention mode
    )

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")

    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=model.learning_rate, weight_decay=model.weight_decay
    )

    print("Starting training...")
    start_time = time.time()

    # Training loop
    train_losses = []
    val_losses = []

    for epoch in range(max_epochs):
        print(f"\nEpoch {epoch + 1}/{max_epochs}")

        # Training phase
        model.train()
        train_epoch_losses = []

        with tqdm(total=train_batches_per_epoch, desc="Training", ncols=120) as pbar:
            train_batch_count = 0

            while train_batch_count < train_batches_per_epoch:
                try:
                    # Get batch from queue with timeout
                    batch_data = batch_queue.get(timeout=10)

                    # Skip if not training batch for this epoch
                    if batch_data.get("phase") != "train" or batch_data.get("epoch") != epoch:
                        continue

                    # Convert numpy back to tensors and move to device
                    batch = convert_batch_to_tensors(batch_data, device)

                    # Forward pass
                    loss = model._generic_step(batch, train_batch_count, "train")

                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Track loss
                    train_epoch_losses.append(loss.item())
                    train_batch_count += 1

                    # Update progress bar
                    pbar.set_postfix({"loss": f"{loss.item():.1e}", "queue": batch_queue.qsize()})
                    pbar.update(1)

                except queue.Empty:
                    print(f"⚠️ Queue timeout during training epoch {epoch}")
                    break

        # Validation phase
        model.eval()
        val_epoch_losses = []

        with tqdm(total=val_batches_per_epoch, desc="Validation", ncols=120) as pbar:
            val_batch_count = 0

            with torch.no_grad():
                while val_batch_count < val_batches_per_epoch:
                    try:
                        # Get batch from queue with timeout
                        batch_data = batch_queue.get(timeout=10)

                        # Skip if not validation batch for this epoch
                        if batch_data.get("phase") != "val" or batch_data.get("epoch") != epoch:
                            continue

                        # Convert numpy back to tensors and move to device
                        batch = convert_batch_to_tensors(batch_data, device)

                        # Forward pass only
                        loss = model._generic_step(batch, val_batch_count, "val")

                        # Track loss
                        val_epoch_losses.append(loss.item())
                        val_batch_count += 1

                        # Update progress bar
                        pbar.set_postfix(
                            {"loss": f"{loss.item():.1e}", "queue": batch_queue.qsize()}
                        )
                        pbar.update(1)

                    except queue.Empty:
                        print(f"⚠️ Queue timeout during validation epoch {epoch}")
                        break

        # Compute and display epoch averages
        if train_epoch_losses:
            train_avg = sum(train_epoch_losses) / len(train_epoch_losses)
            train_losses.append(train_avg)

        if val_epoch_losses:
            val_avg = sum(val_epoch_losses) / len(val_epoch_losses)
            val_losses.append(val_avg)
            print(f"Epoch {epoch + 1}: train_loss={train_avg:.2e}, val_loss={val_avg:.2e}")

    elapsed_time = time.time() - start_time
    print(f"\nTraining complete! Total time: {elapsed_time:.1f}s")

    # Save the final model
    final_model_path = "trained_psf_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_losses": train_losses,
            "val_losses": val_losses,
            "hyperparameters": model.hparams,
        },
        final_model_path,
    )
    print(f"Model saved to {final_model_path}")

    # Clean up worker process
    if worker_process.is_alive():
        print("Cleaning up predictive worker...")
        try:
            # Wait for graceful shutdown (worker should complete autonomously)
            worker_process.join(timeout=5)

            if worker_process.is_alive():
                print("Terminating predictive worker process...")
                worker_process.terminate()
                worker_process.join(timeout=2)

        except Exception as e:
            print(f"Error during predictive worker cleanup: {e}")

        # Close queue
        try:
            batch_queue.close()
        except Exception:
            pass  # Ignore queue cleanup errors


if __name__ == "__main__":
    main()
