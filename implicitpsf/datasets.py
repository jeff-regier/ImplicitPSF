"""Loading and batching of extracted DES star files (v2 schema).

Each .pt file holds per-exposure tensors produced by extract_des_stars. The batch
dict convention consumed by ImplicitPSF.get_loss is defined here and nowhere else.
"""

import hashlib

import torch

PER_STAR_IMAGE_KEYS = ("cutouts", "variance", "valid_pixels")
PER_STAR_SCALAR_KEYS = ("flux", "flux_err", "snr", "sky", "x_pixel", "y_pixel", "color")
PER_STAR_ID_KEYS = ("star_type", "star_id")
PER_EXPOSURE_KEYS = ("exposure_id", "band", "fits_path", "mjd", "night")

BATCH_KEYS = (
    "cutouts",
    "variance",
    "valid_pixels",
    "flux",
    "positions",
    "colors",
    "star_types",
    "star_ids",
)


def load_exposure_file(path):
    """Load one extracted .pt file and validate the v2 schema.

    Returns the raw dict with per-star tensors of shape (n_exposures, n_stars, ...)
    and per-exposure arrays of length n_exposures.
    """
    data = torch.load(path, weights_only=False)

    required = PER_STAR_IMAGE_KEYS + PER_STAR_SCALAR_KEYS + PER_STAR_ID_KEYS
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"{path} is missing keys {missing}; re-extract with the v2 pipeline")

    n_exposures = data["cutouts"].shape[0]
    for key in PER_EXPOSURE_KEYS:
        if len(data[key]) != n_exposures:
            raise ValueError(f"{path}: {key} has length {len(data[key])} != {n_exposures}")

    real_stars = data["flux"] > 0
    if not real_stars.any(dim=1).all():
        raise ValueError(f"{path} contains an exposure with no real stars")

    return data


def make_batch(data, exposure_indices):
    """Assemble the model batch dict for the given exposures of a loaded file."""
    index = torch.as_tensor(exposure_indices, dtype=torch.long)
    positions = torch.stack([data["x_pixel"][index], data["y_pixel"][index]], dim=2)
    batch = {
        "cutouts": data["cutouts"][index],
        "variance": data["variance"][index],
        "valid_pixels": data["valid_pixels"][index],
        "flux": data["flux"][index],
        "positions": positions,
        "colors": data["color"][index],
        "star_types": data["star_type"][index],
        "star_ids": data["star_id"][index],
    }
    if "clean_psf" in data:  # contamination-correction target (sim only; add_clean_psf_target.py)
        batch["clean_psf"] = data["clean_psf"][index]
    return batch


def sample_imputations(data, seed):
    """MCEM M-step: if K-imputation cutouts (cutout_imp, shape (E,S,K,H,W)) are present, replace
    cutouts with ONE random imputation per star (epoch-seeded). Over epochs the SGD M-step then
    Monte-Carlo-averages the loss over the contaminant posterior -- no posterior mean."""
    if "cutout_imp" not in data:
        return
    imp = data["cutout_imp"]
    n_exp, n_star, n_imp = imp.shape[0], imp.shape[1], imp.shape[2]
    gen = torch.Generator().manual_seed(int(seed))
    choice = torch.randint(0, n_imp, (n_exp, n_star), generator=gen)
    ei = torch.arange(n_exp)[:, None]
    si = torch.arange(n_star)[None, :]
    data["cutouts"] = imp[ei, si, choice]


def batch_to_device(batch, device):
    """Move every tensor in a batch dict to a device."""
    return {key: value.to(device) for key, value in batch.items()}


def stable_seed(*parts):
    """Deterministic 32-bit seed from arbitrary parts (unlike salted builtin hash)."""
    text = ":".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")
