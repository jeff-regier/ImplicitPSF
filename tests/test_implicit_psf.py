import pytest
import torch

from implicitpsf.implicit_psf import ImplicitPSF, PSFDecoder, stamp_offsets

PATCH_SIZE = 8
CCD_WIDTH = 512.0
CCD_HEIGHT = 1024.0
HIDDEN_DIM = 32
N_HEADS = 4
BATCH_SIZE = 2
N_STARS = 5


@pytest.fixture
def batch():
    generator = torch.Generator().manual_seed(0)
    cutouts = torch.rand(BATCH_SIZE, N_STARS, PATCH_SIZE, PATCH_SIZE, generator=generator)
    flux = torch.rand(BATCH_SIZE, N_STARS, generator=generator) * 100 + 1
    x_pixel = torch.rand(BATCH_SIZE, N_STARS, generator=generator) * CCD_WIDTH
    y_pixel = torch.rand(BATCH_SIZE, N_STARS, generator=generator) * CCD_HEIGHT
    return {
        "cutouts": cutouts,
        "variance": torch.ones(BATCH_SIZE, N_STARS, PATCH_SIZE, PATCH_SIZE),
        "valid_pixels": torch.ones(BATCH_SIZE, N_STARS, PATCH_SIZE, PATCH_SIZE, dtype=torch.bool),
        "flux": flux,
        "positions": torch.stack([x_pixel, y_pixel], dim=2),
        "colors": torch.rand(BATCH_SIZE, N_STARS, generator=generator) * 3,
        "star_types": torch.zeros(BATCH_SIZE, N_STARS, dtype=torch.uint8),
        "star_ids": torch.arange(BATCH_SIZE * N_STARS).reshape(BATCH_SIZE, N_STARS),
    }


@pytest.fixture
def model():
    torch.manual_seed(0)
    net = ImplicitPSF(
        patch_size=PATCH_SIZE,
        ccd_width=CCD_WIDTH,
        ccd_height=CCD_HEIGHT,
        hidden_dim=HIDDEN_DIM,
        n_heads=N_HEADS,
    )
    return net.eval()


def forward_batch(net, batch, **kwargs):
    context_mask = batch["flux"] > 0
    return net(
        batch["cutouts"],
        batch["positions"],
        batch["colors"],
        batch["flux"],
        context_mask,
        **kwargs,
    )


def test_forward_shape_nonnegativity_unit_sum(model, batch):
    pred_psfs = forward_batch(model, batch)
    assert pred_psfs.shape == (BATCH_SIZE, N_STARS, PATCH_SIZE, PATCH_SIZE)
    assert (pred_psfs >= 0).all()
    sums = pred_psfs.sum(dim=(-2, -1))
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_forward_without_attention(batch):
    torch.manual_seed(0)
    net = ImplicitPSF(
        patch_size=PATCH_SIZE,
        ccd_width=CCD_WIDTH,
        ccd_height=CCD_HEIGHT,
        hidden_dim=HIDDEN_DIM,
        n_heads=N_HEADS,
        use_attention=False,
    ).eval()
    pred_psfs = forward_batch(net, batch)
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


def test_loss_excludes_contaminated_stars_from_target(model, batch):
    contaminated = {**batch, "cutouts": batch["cutouts"].clone()}
    contaminated["star_types"] = batch["star_types"].clone()
    contaminated["star_types"][:, -1] = 1
    contaminated["cutouts"][:, -1] += 1000.0
    loss = model.get_loss(contaminated)
    assert torch.isfinite(loss)


def test_loss_invariant_to_flux_scale(model, batch):
    loss = model.get_loss(batch)
    rescaled = {**batch, "cutouts": batch["cutouts"] * 10, "variance": batch["variance"] * 100}
    loss_rescaled = model.get_loss(rescaled)
    assert torch.allclose(loss, loss_rescaled, rtol=1e-4)


def test_heterogeneous_padding_isolated_per_exposure(model, batch):
    padded = {key: value.clone() for key, value in batch.items()}
    padded["flux"][0, -2:] = 0.0  # exposure 0 padding differs from exposure 1

    full = forward_batch(model, padded)
    alone = {key: value[1:2] for key, value in padded.items()}
    alone_pred = forward_batch(model, alone)
    assert torch.allclose(full[1], alone_pred[0], atol=1e-6)


def test_all_context_masked_raises(model, batch):
    bad = {key: value.clone() for key, value in batch.items()}
    bad["flux"][0] = 0.0
    bad["star_types"][0] = 4
    with pytest.raises(ValueError, match="context star"):
        model.get_loss(bad)


def test_stamp_offsets_convention():
    positions = torch.tensor([[[10.25, 20.75]]])
    offsets = stamp_offsets(positions, patch_size=4)
    assert offsets.shape == (1, 1, 16, 2)
    # corner = round(center) - 2; first sample center offset = -2 - (center - round(center))
    assert torch.allclose(offsets[0, 0, 0], torch.tensor([-2.25, -1.75]))


def test_stamp_offsets_oversample_centers():
    positions = torch.tensor([[[10.0, 20.0]]])
    native = stamp_offsets(positions, patch_size=4, oversample=1)
    fine = stamp_offsets(positions, patch_size=4, oversample=2)
    assert fine.shape == (1, 1, 64, 2)
    # each native pixel's 2x2 sub-sample offsets average to the native offset
    fine_grid = fine.reshape(1, 1, 4, 2, 4, 2, 2)
    block_means = fine_grid.mean(dim=(3, 5)).reshape(1, 1, 16, 2)
    assert torch.allclose(block_means, native, atol=1e-6)


def test_decoder_whole_pixel_shift_invariance():
    torch.manual_seed(0)
    decoder = PSFDecoder(HIDDEN_DIM, PATCH_SIZE, use_features=False).eval()
    positions = torch.tensor([[[10.25, 20.75]]])
    shifted = positions + 3.0  # whole-pixel shift preserves subpixel phase
    psf = decoder(stamp_offsets(positions, PATCH_SIZE), None)
    psf_shifted = decoder(stamp_offsets(shifted, PATCH_SIZE), None)
    assert torch.allclose(psf, psf_shifted, atol=1e-6)


def test_oversampled_render_consistent_with_native(model, batch):
    native = forward_batch(model, batch)
    fine = forward_batch(model, batch, oversample=2)
    side = PATCH_SIZE * 2
    assert fine.shape == (BATCH_SIZE, N_STARS, side, side)
    blocks = fine.reshape(BATCH_SIZE, N_STARS, PATCH_SIZE, 2, PATCH_SIZE, 2)
    downsampled = blocks.mean(dim=(3, 5))
    assert torch.allclose(downsampled, native, atol=0.02)
