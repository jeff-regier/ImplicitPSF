"""Night-level train/val/test splits and reserved-star lists, frozen in a manifest.

The split unit is the observing night: consecutive exposures share atmospheric state,
so exposure-level splits would leak. Assignment is a deterministic hash of the night,
so the split is stable under re-extraction and additional data. Within every test
exposure, a deterministic hash of (exposure_id, star_id) reserves ~20% of clean stars;
reserved stars are excluded from every method's fit/context and used only for scoring.
"""

import hashlib
import json
from pathlib import Path

import torch

TYPE_CLEAN = 0
RESERVED_FRACTION = 0.2


def _unit_hash(text):
    """Deterministic uniform [0, 1) value from a string (stable across processes)."""
    digest = hashlib.sha256(text.encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def assign_split(night, seed, frac_val, frac_test):
    """Assign an observing night to train/val/test by deterministic hash."""
    value = _unit_hash(f"night-split:{seed}:{night}")
    if value < frac_test:
        return "test"
    if value < frac_test + frac_val:
        return "val"
    return "train"


def is_reserved(exposure_id, star_id, seed):
    """Whether a clean star is reserved for evaluation (test exposures only)."""
    return _unit_hash(f"reserve:{seed}:{exposure_id}:{star_id}") < RESERVED_FRACTION


def build_manifest(data_dir, seed=0, frac_val=0.1, frac_test=0.1):
    """Scan extracted .pt files and freeze the split and reserved-star lists."""
    paths = sorted(Path(data_dir).glob("desstars_*.pt"))
    if not paths:
        raise FileNotFoundError(f"no extracted files in {data_dir}")

    exposures = {}
    reserved = {}
    for path in paths:
        data = torch.load(path, weights_only=False)
        clean = (data["star_type"] == TYPE_CLEAN) & (data["flux"] > 0)
        for index, exposure_id in enumerate(data["exposure_id"]):
            if exposure_id in exposures:
                raise ValueError(f"duplicate exposure {exposure_id} in {path}")
            night = str(data["night"][index])
            split = assign_split(night, seed, frac_val, frac_test)
            exposures[exposure_id] = {
                "split": split,
                "file": path.name,
                "index": index,
                "night": night,
                "band": str(data["band"][index]),
                "n_clean": int(clean[index].sum()),
            }
            if split in ("val", "test"):  # val reservations make dress rehearsals possible
                star_ids = data["star_id"][index][clean[index]].tolist()
                reserved[exposure_id] = sorted(
                    star_id for star_id in star_ids if is_reserved(exposure_id, star_id, seed)
                )

    return {
        "seed": seed,
        "frac_val": frac_val,
        "frac_test": frac_test,
        "reserved_fraction": RESERVED_FRACTION,
        "data_dir": str(data_dir),
        "exposures": exposures,
        "reserved": reserved,
    }


def write_manifest(manifest, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=1, sort_keys=True)


def load_manifest(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def files_for_split(manifest, split):
    """Map file name -> sorted exposure indices belonging to a split."""
    files = {}
    for info in manifest["exposures"].values():
        if info["split"] == split:
            files.setdefault(info["file"], []).append(info["index"])
    return {name: sorted(indices) for name, indices in sorted(files.items())}


def reserved_star_ids(manifest, exposure_id):
    """Reserved star IDs for a test exposure (empty for train/val exposures)."""
    return set(manifest["reserved"].get(exposure_id, []))
