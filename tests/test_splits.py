import numpy as np
import pytest
import torch

from implicitpsf.splits import (
    assign_split,
    build_manifest,
    files_for_split,
    is_reserved,
    load_manifest,
    reserved_star_ids,
    write_manifest,
)

N_EXPOSURES = 40
N_STARS = 30


@pytest.fixture
def data_dir(tmp_path):
    generator = torch.Generator().manual_seed(0)
    for chunk in range(2):
        flux = torch.rand(N_EXPOSURES, N_STARS, generator=generator) * 100 + 1
        star_id = torch.arange(N_EXPOSURES * N_STARS).reshape(N_EXPOSURES, N_STARS)
        offset = chunk * N_EXPOSURES
        data = {
            "flux": flux,
            "star_type": torch.zeros(N_EXPOSURES, N_STARS, dtype=torch.uint8),
            "star_id": star_id + offset * N_STARS,
            "exposure_id": np.array([f"D{offset + i:08d}" for i in range(N_EXPOSURES)]),
            "band": np.array(["r"] * N_EXPOSURES),
            "night": np.array([f"201309{(offset + i) % 20 + 1:02d}" for i in range(N_EXPOSURES)]),
        }
        torch.save(data, tmp_path / f"desstars_w{chunk:02d}_0000.pt")
    return tmp_path


def test_split_assignment_deterministic_and_disjoint():
    nights = [f"2013{m:02d}{d:02d}" for m in range(1, 13) for d in range(1, 29)]
    splits = [assign_split(night, seed=0, frac_val=0.1, frac_test=0.1) for night in nights]
    splits_again = [assign_split(night, seed=0, frac_val=0.1, frac_test=0.1) for night in nights]
    assert splits == splits_again
    fractions = {name: splits.count(name) / len(splits) for name in ("train", "val", "test")}
    assert 0.05 < fractions["test"] < 0.16
    assert 0.05 < fractions["val"] < 0.16
    assert fractions["train"] > 0.7


def test_reserved_fraction_approximate():
    flags = [is_reserved("D00000001", star_id, seed=0) for star_id in range(5000)]
    assert 0.17 < np.mean(flags) < 0.23


def test_manifest_round_trip_and_consistency(data_dir, tmp_path):
    manifest = build_manifest(data_dir, seed=0)
    manifest_path = tmp_path / "manifest.json"
    write_manifest(manifest, manifest_path)
    loaded = load_manifest(manifest_path)

    assert loaded["exposures"].keys() == manifest["exposures"].keys()
    splits = {info["split"] for info in loaded["exposures"].values()}
    assert splits <= {"train", "val", "test"}

    # same night -> same split, across files
    night_splits = {}
    for info in loaded["exposures"].values():
        night_splits.setdefault(info["night"], set()).add(info["split"])
    assert all(len(values) == 1 for values in night_splits.values())

    # every val/test exposure has a reserved list; train has none
    for exposure_id, info in loaded["exposures"].items():
        expected = info["split"] in ("val", "test")
        assert (exposure_id in loaded["reserved"]) == expected

    # files_for_split covers every exposure exactly once
    total = sum(
        len(idx)
        for split in ("train", "val", "test")
        for idx in files_for_split(loaded, split).values()
    )
    assert total == len(loaded["exposures"])


def test_reserved_star_ids_subset_of_clean(data_dir):
    manifest = build_manifest(data_dir, seed=0)
    test_ids = [eid for eid, info in manifest["exposures"].items() if info["split"] == "test"]
    assert len(test_ids) > 0
    for exposure_id in test_ids:
        info = manifest["exposures"][exposure_id]
        assert len(reserved_star_ids(manifest, exposure_id)) <= info["n_clean"]


def test_duplicate_exposure_fails_loudly(data_dir):
    duplicate = torch.load(data_dir / "desstars_w00_0000.pt", weights_only=False)
    torch.save(duplicate, data_dir / "desstars_w99_0000.pt")
    with pytest.raises(ValueError, match="duplicate exposure"):
        build_manifest(data_dir, seed=0)
