import numpy as np
import pytest
import torch

from implicitpsf.datasets import BATCH_KEYS, load_exposure_file, make_batch

N_EXPOSURES = 3
N_STARS = 4
PATCH = 8


@pytest.fixture
def exposure_file(tmp_path):
    generator = torch.Generator().manual_seed(0)
    flux = torch.rand(N_EXPOSURES, N_STARS, generator=generator) * 100 + 1
    data = {
        "cutouts": torch.rand(N_EXPOSURES, N_STARS, PATCH, PATCH, generator=generator),
        "variance": torch.ones(N_EXPOSURES, N_STARS, PATCH, PATCH),
        "valid_pixels": torch.ones(N_EXPOSURES, N_STARS, PATCH, PATCH, dtype=torch.bool),
        "flux": flux,
        "flux_err": flux.sqrt(),
        "snr": flux.sqrt(),
        "sky": torch.zeros(N_EXPOSURES, N_STARS),
        "x_pixel": torch.rand(N_EXPOSURES, N_STARS, generator=generator) * 2048,
        "y_pixel": torch.rand(N_EXPOSURES, N_STARS, generator=generator) * 4096,
        "color": torch.rand(N_EXPOSURES, N_STARS, generator=generator),
        "star_type": torch.zeros(N_EXPOSURES, N_STARS, dtype=torch.uint8),
        "star_id": torch.arange(N_EXPOSURES * N_STARS).reshape(N_EXPOSURES, N_STARS),
        "exposure_id": np.array([f"D0012345{i}" for i in range(N_EXPOSURES)]),
        "band": np.array(["r"] * N_EXPOSURES),
        "fits_path": np.array([f"/data/exp{i}.fits.fz" for i in range(N_EXPOSURES)]),
        "mjd": np.arange(N_EXPOSURES, dtype=np.float64) + 56000.0,
        "night": np.array(["20130901"] * N_EXPOSURES),
    }
    path = tmp_path / "desstars_v2_test.pt"
    torch.save(data, path)
    return path


def test_load_and_batch_round_trip(exposure_file):
    data = load_exposure_file(exposure_file)
    batch = make_batch(data, [0, 2])

    assert set(batch.keys()) == set(BATCH_KEYS)
    assert batch["cutouts"].shape == (2, N_STARS, PATCH, PATCH)
    assert batch["positions"].shape == (2, N_STARS, 2)
    assert torch.equal(batch["positions"][..., 0], data["x_pixel"][[0, 2]])
    assert torch.equal(batch["star_ids"], data["star_id"][[0, 2]])


def test_missing_key_fails_loudly(exposure_file, tmp_path):
    data = torch.load(exposure_file, weights_only=False)
    del data["variance"]
    bad_path = tmp_path / "bad.pt"
    torch.save(data, bad_path)
    with pytest.raises(KeyError, match="variance"):
        load_exposure_file(bad_path)


def test_exposure_with_no_real_stars_fails_loudly(exposure_file, tmp_path):
    data = torch.load(exposure_file, weights_only=False)
    data["flux"][1] = 0.0
    bad_path = tmp_path / "empty.pt"
    torch.save(data, bad_path)
    with pytest.raises(ValueError, match="no real stars"):
        load_exposure_file(bad_path)
