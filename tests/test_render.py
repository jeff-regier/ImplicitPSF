import pytest
import torch

from implicitpsf.implicit_psf import ImplicitPSF
from implicitpsf.render import render_at

PATCH = 8
N_STARS = 6


@pytest.fixture
def setup():
    generator = torch.Generator().manual_seed(0)
    torch.manual_seed(0)
    model = ImplicitPSF(
        patch_size=PATCH,
        ccd_width=512.0,
        ccd_height=1024.0,
        hidden_dim=32,
        n_heads=4,
    ).eval()
    flux = torch.rand(1, N_STARS, generator=generator) * 100 + 1
    x = torch.rand(1, N_STARS, generator=generator) * 512
    y = torch.rand(1, N_STARS, generator=generator) * 1024
    batch = {
        "cutouts": torch.rand(1, N_STARS, PATCH, PATCH, generator=generator),
        "flux": flux,
        "positions": torch.stack([x, y], dim=2),
        "colors": torch.rand(1, N_STARS, generator=generator),
    }
    return model, batch


def test_render_at_shape_and_normalization(setup):
    model, batch = setup
    queries = torch.tensor([[100.25, 700.5], [400.75, 200.0]])
    colors = torch.tensor([0.5, 1.5])
    kernels = render_at(model, batch, queries, colors)
    assert kernels.shape == (2, PATCH, PATCH)
    sums = kernels.sum(dim=(-2, -1))
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_render_at_oversampled_flux_conserved(setup):
    model, batch = setup
    queries = torch.tensor([[100.25, 700.5]])
    colors = torch.tensor([0.5])
    fine = render_at(model, batch, queries, colors, oversample=4)
    assert fine.shape == (1, PATCH * 4, PATCH * 4)
    # unit native-pixel sum: oversampled values sum to oversample^2
    assert torch.allclose(fine.sum(), torch.tensor(16.0), atol=1e-3)


def test_render_at_matches_reserved_star_protocol(setup):
    model, batch = setup
    last = N_STARS - 1
    reserved = torch.zeros(1, N_STARS, dtype=torch.bool)
    reserved[0, last] = True
    context_mask = (batch["flux"] > 0) & ~reserved
    with torch.no_grad():
        direct = model(
            batch["cutouts"],
            batch["positions"],
            batch["colors"],
            batch["flux"],
            context_mask,
        )[0, last]

    # render_at treats every real star in its batch as context, so drop the reserved
    # star from the batch to reproduce the same context the direct path used
    truncated = {key: value[:, :last] for key, value in batch.items()}
    via_query = render_at(
        model,
        truncated,
        batch["positions"][0, last : last + 1],
        batch["colors"][0, last : last + 1],
        query_fluxes=batch["flux"][0, last : last + 1],
    )[0]
    assert torch.allclose(direct, via_query, atol=1e-6)


def test_render_at_continuity(setup):
    model, batch = setup
    queries = torch.tensor([[250.0, 500.0], [250.5, 500.5]])
    colors = torch.tensor([1.0, 1.0])
    kernels = render_at(model, batch, queries, colors)
    # same subpixel phase, nearly the same field position -> nearly identical kernels
    assert (kernels[0] - kernels[1]).abs().max() < 0.05 * kernels[0].max()
