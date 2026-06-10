import pytest
import torch

from implicitpsf.implicit_psf import ImplicitPSF, PSFDecoder

PATCH_SIZE = 8
IMAGE_SIZE = 1024
HIDDEN_DIM = 32
N_HEADS = 4
BATCH_SIZE = 2
N_STARS = 5


@pytest.fixture
def batch():
    generator = torch.Generator().manual_seed(0)
    cutouts = torch.rand(BATCH_SIZE, N_STARS, PATCH_SIZE, PATCH_SIZE, generator=generator)
    flux = torch.rand(BATCH_SIZE, N_STARS, generator=generator) * 100 + 1
    positions = torch.rand(BATCH_SIZE, N_STARS, 2, generator=generator) * IMAGE_SIZE
    star_types = torch.zeros(BATCH_SIZE, N_STARS, dtype=torch.uint8)
    return {
        "cutouts": cutouts,
        "flux": flux,
        "positions": positions,
        "star_types": star_types,
    }


@pytest.fixture
def model():
    return ImplicitPSF(
        patch_size=PATCH_SIZE,
        image_size=IMAGE_SIZE,
        hidden_dim=HIDDEN_DIM,
        n_heads=N_HEADS,
    )


def test_forward_shape_and_nonnegativity(model, batch):
    pred_psfs = model(batch["cutouts"], batch["positions"])
    assert pred_psfs.shape == (BATCH_SIZE, N_STARS, PATCH_SIZE, PATCH_SIZE)
    assert (pred_psfs >= 0).all()


def test_forward_without_attention(batch):
    model = ImplicitPSF(
        patch_size=PATCH_SIZE,
        image_size=IMAGE_SIZE,
        hidden_dim=HIDDEN_DIM,
        n_heads=N_HEADS,
        use_attention=False,
    )
    pred_psfs = model(batch["cutouts"], batch["positions"])
    assert pred_psfs.shape == (BATCH_SIZE, N_STARS, PATCH_SIZE, PATCH_SIZE)


def test_position_encoding_shape(model, batch):
    encodings = model._sinusoidal_position_encoding(batch["positions"])
    assert encodings.shape == (BATCH_SIZE, N_STARS, HIDDEN_DIM)
    assert (encodings >= -1).all()
    assert (encodings <= 1).all()


def test_loss_is_finite_scalar_with_gradient(model, batch):
    loss = model.get_loss(batch)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0


def test_loss_ignores_contaminated_stars(model, batch):
    loss_clean = model.get_loss(batch)

    contaminated = {**batch, "cutouts": batch["cutouts"].clone()}
    contaminated["star_types"] = batch["star_types"].clone()
    contaminated["star_types"][:, -1] = 1
    contaminated["cutouts"][:, -1] += 1000.0

    torch.manual_seed(0)
    loss_contaminated = model.get_loss(contaminated)
    assert torch.isfinite(loss_contaminated)
    assert loss_contaminated < loss_clean + 1000.0


def test_padding_stars_masked_from_attention(model, batch):
    padded = {key: value.clone() for key, value in batch.items()}
    padded["flux"][:, -1] = 0.0
    loss = model.get_loss(padded)
    assert torch.isfinite(loss)


def test_decoder_subpixel_offset_invariance():
    decoder = PSFDecoder(hidden_dim=HIDDEN_DIM, image_size=PATCH_SIZE, use_features=False)
    centers = torch.tensor([[[10.25, 20.75]]])
    shifted_centers = centers + 3.0  # whole-pixel shift preserves subpixel offset
    psf = decoder(centers, None)
    psf_shifted = decoder(shifted_centers, None)
    assert torch.allclose(psf, psf_shifted)
