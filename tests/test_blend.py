import pytest
import torch

from implicitpsf.blend import (
    chebyshev_distances,
    component_offsets,
    gls_amplitudes,
    neighbor_table,
    sample_grid,
)
from implicitpsf.implicit_psf import ImplicitPSF, stamp_offsets

PATCH = 8


def gaussian_stamp_flat(grid, center_shift, sigma=1.5):
    """Unit-sum Gaussian evaluated on a flat (n_pix, 2) sample grid."""
    d2 = ((grid - center_shift) ** 2).sum(-1)
    g = torch.exp(-0.5 * d2 / sigma**2)
    return g / g.sum()


def test_neighbor_table_geometry_and_types():
    positions = torch.tensor([[[50.0, 50.0], [60.0, 50.0], [50.0, 70.0], [55.0, 55.0]]])
    fluxes = torch.tensor([[100.0, 50.0, 200.0, 10.0]])
    types = torch.tensor([[0, 1, 2, 5]], dtype=torch.uint8)  # star, star, GALAXY, saturated

    idx, mask, is_gal, galaxy_near = neighbor_table(positions, fluxes, types, radius=22.0, k_max=4)
    # target 0: neighbors within Chebyshev 22 are slots 1 (star) and 3 (saturated);
    # slot 2 is a galaxy (excluded from point components but flags galaxy_near)
    neighbors_0 = set(idx[0, 0][mask[0, 0]].tolist())
    assert neighbors_0 == {1, 3}
    assert not is_gal[0, 0].any()
    assert bool(galaxy_near[0, 0])
    # brightest first
    assert idx[0, 0, 0].item() == 1

    # with include_galaxies the (bright) galaxy joins, flagged, still brightest-first
    idx_g, mask_g, is_gal_g, _ = neighbor_table(
        positions, fluxes, types, radius=22.0, k_max=4, include_galaxies=True
    )
    neighbors_g = idx_g[0, 0][mask_g[0, 0]].tolist()
    assert neighbors_g == [2, 1, 3]
    assert is_gal_g[0, 0].tolist() == [True, False, False]


def test_gls_recovers_exact_amplitudes_noiseless():
    grid = sample_grid(PATCH, 1, torch.device("cpu"), torch.float64)
    comp_a = gaussian_stamp_flat(grid, torch.tensor([0.2, -0.1], dtype=torch.float64))
    comp_b = gaussian_stamp_flat(grid, torch.tensor([3.0, 2.0], dtype=torch.float64))
    components = torch.stack([comp_a, comp_b]).unsqueeze(0)  # (1, 2, n_pix)
    true_amps = torch.tensor([[1000.0, 250.0]], dtype=torch.float64)
    observed = (true_amps.unsqueeze(-1) * components).sum(1)
    weights = torch.ones_like(observed)
    mask = torch.ones(1, 2, dtype=torch.bool)

    amps = gls_amplitudes(components, observed, weights, mask, ridge=0.0)
    torch.testing.assert_close(amps, true_amps, rtol=1e-8, atol=1e-6)


def test_gls_masked_component_is_inert():
    grid = sample_grid(PATCH, 1, torch.device("cpu"), torch.float64)
    comp = gaussian_stamp_flat(grid, torch.zeros(2, dtype=torch.float64))
    junk = torch.rand(grid.shape[0], dtype=torch.float64)
    components = torch.stack([comp, junk]).unsqueeze(0)
    observed = (777.0 * comp).unsqueeze(0)
    weights = torch.ones_like(observed)
    mask = torch.tensor([[True, False]])

    amps = gls_amplitudes(components, observed, weights, mask, ridge=0.0)
    torch.testing.assert_close(amps[0, 0], torch.tensor(777.0, dtype=torch.float64))
    assert amps[0, 1].abs().item() < 1e-8


def test_gls_near_coincident_components_stay_finite():
    grid = sample_grid(PATCH, 1, torch.device("cpu"), torch.float64)
    comp_a = gaussian_stamp_flat(grid, torch.tensor([0.0, 0.0], dtype=torch.float64))
    comp_b = gaussian_stamp_flat(grid, torch.tensor([0.05, 0.0], dtype=torch.float64))
    components = torch.stack([comp_a, comp_b]).unsqueeze(0)
    observed = (500.0 * comp_a + 500.0 * comp_b).unsqueeze(0)
    weights = torch.ones_like(observed)
    mask = torch.ones(1, 2, dtype=torch.bool)

    amps = gls_amplitudes(components, observed, weights, mask)
    assert torch.isfinite(amps).all()
    total = (amps.unsqueeze(-1) * components).sum(1)
    assert (total - observed).abs().max() < 1.0


def test_component_offsets_match_stamp_offsets_for_self():
    positions = torch.tensor([[[100.37, 80.62], [104.0, 82.0]]])
    grid = sample_grid(PATCH, 1, positions.device, positions.dtype)
    batch_idx = torch.tensor([0])
    target_idx = torch.tensor([0])
    comp_idx = torch.tensor([[0, 1]])
    offsets = component_offsets(positions, batch_idx, target_idx, comp_idx, grid)

    self_offsets = stamp_offsets(positions, PATCH)[0, 0]
    torch.testing.assert_close(offsets[0, 0], self_offsets)
    # neighbor offsets: shifted by its position relative to the target's stamp origin
    expected_shift = positions[0, 1] - positions[0, 0].round()
    torch.testing.assert_close(offsets[0, 1], grid - expected_shift)


@pytest.fixture
def blend_batch():
    generator = torch.Generator().manual_seed(0)
    n_stars = 6
    flux = torch.rand(1, n_stars, generator=generator) * 100 + 1
    x = torch.rand(1, n_stars, generator=generator) * 400 + 50
    y = torch.rand(1, n_stars, generator=generator) * 800 + 50
    return {
        "cutouts": torch.rand(1, n_stars, PATCH, PATCH, generator=generator),
        "variance": torch.ones(1, n_stars, PATCH, PATCH),
        "valid_pixels": torch.ones(1, n_stars, PATCH, PATCH, dtype=torch.bool),
        "flux": flux,
        "positions": torch.stack([x, y], dim=2),
        "colors": torch.rand(1, n_stars, generator=generator),
        "star_types": torch.zeros(1, n_stars, dtype=torch.uint8),
        "star_ids": torch.arange(n_stars).reshape(1, n_stars),
    }


def test_blend_loss_equals_single_loss_when_isolated(blend_batch):
    # spread stars far apart -> no neighbors -> blend mode must equal single mode
    blend_batch["positions"][0, :, 0] = torch.arange(6) * 100.0 + 50.37
    blend_batch["positions"][0, :, 1] = torch.arange(6) * 150.0 + 60.62

    torch.manual_seed(0)
    single = ImplicitPSF(
        patch_size=PATCH, ccd_width=700.0, ccd_height=1000.0, hidden_dim=32, n_heads=4
    ).eval()
    torch.manual_seed(0)
    blend = ImplicitPSF(
        patch_size=PATCH,
        ccd_width=700.0,
        ccd_height=1000.0,
        hidden_dim=32,
        n_heads=4,
        loss_mode="blend",
    ).eval()
    blend.load_state_dict(single.state_dict())

    loss_single = single.get_loss(blend_batch)
    loss_blend = blend.get_loss(blend_batch)
    torch.testing.assert_close(loss_blend, loss_single, rtol=1e-5, atol=1e-7)


def test_blend_loss_runs_with_neighbors_and_galaxy_modes(blend_batch):
    # cluster stars so neighbors exist; make one a galaxy
    blend_batch["positions"][0, :, 0] = torch.tensor([100.2, 108.0, 115.5, 300.0, 310.0, 480.0])
    blend_batch["positions"][0, :, 1] = torch.tensor([100.7, 106.0, 98.0, 300.0, 305.0, 480.0])
    blend_batch["star_types"][0, 4] = 2  # galaxy near slot 3

    for mode in ["exclude", "mask", "component"]:
        torch.manual_seed(0)
        model = ImplicitPSF(
            patch_size=PATCH,
            ccd_width=700.0,
            ccd_height=1000.0,
            hidden_dim=32,
            n_heads=4,
            loss_mode="blend",
            galaxy_mode=mode,
        ).eval()
        loss = model.get_loss(blend_batch)
        assert torch.isfinite(loss)
        loss.backward()


def test_chebyshev_distances():
    positions = torch.tensor([[[0.0, 0.0], [3.0, 4.0]]])
    d = chebyshev_distances(positions)
    assert d[0, 0, 1].item() == 4.0
