"""Cutout-level blend likelihood: target + detected star neighbors, GLS amplitudes.

Each target star's cutout is modeled as a sum of point-source components — the target
plus up to k_max-1 detected star neighbors — all rendered by the same decoder on the
TARGET's pixel grid. Component amplitudes are linear, so they are solved in closed
form by generalized least squares inside the loss (a k-vector generalization of the
single-star amplitude solve). Galaxies are misspecified by point sources; how to
handle them is an empirical question with three modes: exclude targets near them,
mask their pixels, or (later) add a crude profile component.

Conventions: stamp corner = round(target center) - patch//2; a component at CCD
position p contributes at stamp offsets grid - (p - round(p_target)). With zero
neighbors this reduces exactly to the existing single-star loss (regression-tested).
"""

import torch


def sample_grid(patch_size, oversample, device, dtype):
    """(n_samples, 2) stamp sample-center coordinates relative to the stamp's
    round(center) origin: corner = round(center) - patch_size // 2."""
    half = patch_size // 2
    n_samples = patch_size * oversample
    step = 1.0 / oversample
    grid_1d = torch.arange(n_samples, device=device, dtype=dtype)
    grid_1d = grid_1d * step + (step - 1.0) / 2.0 - half
    grid_v, grid_u = torch.meshgrid(grid_1d, grid_1d, indexing="ij")
    return torch.stack([grid_u, grid_v], dim=-1).reshape(-1, 2)


STAR_TYPES_POINT = (0, 1, 5)  # clean, star-context, saturated: all genuine point sources
STAR_TYPES_GALAXY = (2, 3)  # galaxy, unmatched: misspecification risk
GALAXY_MASK_RADIUS = 6.0  # pixels zeroed around a galaxy detection in 'mask' mode
GALAXY_COMPONENT_RE = 2.5  # pixels; the crude round-exponential galaxy model's radius
SERSIC_B1 = 1.67834699  # exact b_n for n=1


def chebyshev_distances(positions):
    """(b, s, s) pairwise L-infinity distances; square stamps make this the metric."""
    diff = positions.unsqueeze(2) - positions.unsqueeze(1)
    return diff.abs().amax(dim=-1)


def neighbor_table(positions, fluxes, star_types, radius=22.0, k_max=4, include_galaxies=False):
    """Per star, the indices of its brightest detected neighbors within radius.

    Args:
        include_galaxies: admit galaxy/unmatched detections as components (brightest-
            first alongside stars) instead of only flagging them

    Returns:
        idx: (b, s, k_max - 1) neighbor slot indices (0 where masked)
        mask: (b, s, k_max - 1) bool, True = real neighbor
        is_gal: (b, s, k_max - 1) bool, True where the neighbor is a galaxy/unmatched
        galaxy_near: (b, s) bool, True if a galaxy/unmatched detection is within radius
    """
    cheb = chebyshev_distances(positions)
    is_point = torch.zeros_like(fluxes, dtype=torch.bool)
    for t in STAR_TYPES_POINT:
        is_point |= star_types == t
    is_point &= fluxes > 0
    is_galaxy = torch.zeros_like(fluxes, dtype=torch.bool)
    for t in STAR_TYPES_GALAXY:
        is_galaxy |= star_types == t
    is_galaxy &= fluxes > 0

    n_stars = positions.shape[1]
    eye = torch.eye(n_stars, device=positions.device, dtype=torch.bool)
    near = (cheb < radius) & ~eye
    component_ok = is_point | is_galaxy if include_galaxies else is_point
    eligible = near & component_ok.unsqueeze(1)

    score = torch.where(eligible, fluxes.unsqueeze(1).expand_as(eligible), -torch.inf)
    top = score.topk(k_max - 1, dim=-1)
    mask = torch.isfinite(top.values)
    idx = torch.where(mask, top.indices, torch.zeros_like(top.indices))
    is_gal = is_galaxy.unsqueeze(1).expand(-1, n_stars, -1).gather(2, idx) & mask

    galaxy_near = (near & is_galaxy.unsqueeze(1)).any(dim=-1)
    return idx, mask, is_gal, galaxy_near


def component_offsets(positions, batch_idx, target_idx, comp_idx, grid):
    """Offsets of the target's stamp samples from each component's center.

    Args:
        positions: (b, s, 2) CCD positions
        batch_idx: (N,) exposure index per target
        target_idx: (N,) target slot per target
        comp_idx: (N, K) component slot indices (column 0 = the target itself)
        grid: (n_samples, 2) stamp sample grid from sample_grid()

    Returns:
        (N, K, n_samples, 2)
    """
    target_pos = positions[batch_idx, target_idx]  # (N, 2)
    comp_pos = positions[batch_idx.unsqueeze(1), comp_idx]  # (N, K, 2)
    shift = comp_pos - target_pos.round().unsqueeze(1)  # (N, K, 2)
    return grid.unsqueeze(0).unsqueeze(0) - shift.unsqueeze(2)


def gls_amplitudes(components, observed, weights, comp_mask, ridge=1e-4):
    """Closed-form weighted least-squares amplitudes for K components per target.

    Args:
        components: (N, K, n_pix) unit-ish component stamps (masked columns arbitrary)
        observed: (N, n_pix) data stamps
        weights: (N, n_pix) inverse-variance times validity
        comp_mask: (N, K) bool, True = real component
        ridge: relative Tikhonov factor (guards near-coincident components)

    Returns:
        amplitudes: (N, K), zero at masked slots; differentiable throughout
    """
    masked = components * comp_mask.unsqueeze(-1)
    normal = torch.einsum("nkp,np,nlp->nkl", masked, weights, masked)
    rhs = torch.einsum("nkp,np,np->nk", masked, weights, observed)
    diag = torch.diagonal(normal, dim1=-2, dim2=-1)
    regular = torch.diag_embed(ridge * diag + (~comp_mask).float())
    return torch.linalg.solve(normal + regular, rhs)


def blend_chi2(
    model, batch, context_mask, radius, k_max, galaxy_mode, chi2_cap=None, max_targets=None
):
    """Per-target reduced chi-square under the multi-component blend model.

    galaxy_mode: 'exclude' drops targets with a galaxy detection within radius;
    'mask' keeps them but zeroes weights near the galaxy; 'component' admits galaxy
    detections as GLS components, modeled crudely as the decoded PSF convolved with
    a fixed round exponential (linear amplitude, so the solve is unchanged).

    max_targets caps how many targets are scored per call (memory: each target
    decodes k_max stamps). Training draws a fresh random subset each step — an
    unbiased estimate of the per-target mean; eval subsets are deterministic.
    """
    positions = batch["positions"]
    fluxes = batch["flux"]
    star_types = batch["star_types"]
    patch = model.patch_size

    features = (
        model.attend(batch["cutouts"], positions, batch["colors"], fluxes, context_mask)
        if model.use_attention
        else None
    )

    idx, mask, neighbor_is_gal, galaxy_near = neighbor_table(
        positions, fluxes, star_types, radius, k_max, include_galaxies=galaxy_mode == "component"
    )

    is_target = ((star_types == 0) | (star_types == 1)) & (fluxes > 0)
    is_target &= batch["valid_pixels"].any(dim=-1).any(dim=-1)  # dead-amp/masked stamps
    if galaxy_mode == "exclude":
        is_target &= ~galaxy_near
    if not is_target.any():
        raise ValueError("no blend targets in batch")

    batch_idx, target_idx = torch.nonzero(is_target, as_tuple=True)
    batch_idx, target_idx = _subsample_targets(batch_idx, target_idx, max_targets, model.training)
    n_targets = len(batch_idx)
    comp_idx = torch.cat(
        [target_idx.unsqueeze(1), idx[batch_idx, target_idx]], dim=1
    )  # (N, K): column 0 is the target itself
    comp_mask = torch.cat(
        [
            torch.ones(n_targets, 1, dtype=torch.bool, device=positions.device),
            mask[batch_idx, target_idx],
        ],
        dim=1,
    )

    grid = sample_grid(patch, 1, positions.device, positions.dtype)
    offsets = component_offsets(positions, batch_idx, target_idx, comp_idx, grid)

    comp_features = None
    if features is not None:
        comp_features = features[batch_idx.unsqueeze(1), comp_idx]  # (N, K, hidden)
    k_eff = comp_idx.shape[1]
    flat_offsets = offsets.reshape(1, n_targets * k_eff, -1, 2)
    flat_features = None
    if comp_features is not None:
        flat_features = comp_features.reshape(1, n_targets * k_eff, -1)
    intensities = model.psf_decoder(flat_offsets, flat_features)
    components = intensities.reshape(n_targets, k_eff, -1)
    components = components / components.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    if galaxy_mode == "component":
        is_gal_comp = torch.cat(
            [
                torch.zeros(n_targets, 1, dtype=torch.bool, device=positions.device),
                neighbor_is_gal[batch_idx, target_idx],
            ],
            dim=1,
        )
        components = _spread_galaxy_components(components, is_gal_comp, patch)

    observed = batch["cutouts"][batch_idx, target_idx].reshape(n_targets, -1)
    variance = batch["variance"][batch_idx, target_idx].reshape(n_targets, -1)
    valid = batch["valid_pixels"][batch_idx, target_idx].reshape(n_targets, -1)
    weights = valid.float() / variance

    if galaxy_mode == "mask":
        weights = weights * _galaxy_pixel_mask(
            positions, star_types, fluxes, batch_idx, target_idx, grid, radius
        )

    n_valid = weights.gt(0).sum(dim=-1).float()
    usable = n_valid > 0  # galaxy masking can zero out a target's remaining pixels
    if not usable.any():
        raise ValueError("no blend target with usable pixels")

    amplitudes = gls_amplitudes(components, observed, weights, comp_mask)
    model_stamps = torch.einsum("nk,nkp->np", amplitudes, components)
    residual_chi2 = (weights * (observed - model_stamps).square()).sum(dim=-1)
    chi2 = residual_chi2[usable] / n_valid[usable]
    if chi2_cap is not None:
        chi2 = chi2.clamp(max=chi2_cap)
    return chi2


def _subsample_targets(batch_idx, target_idx, max_targets, training):
    """Cap scored targets: fresh random subsets while training (unbiased estimate of
    the per-target mean), deterministic subsets in eval (comparable val curves)."""
    if max_targets is None or len(batch_idx) <= max_targets:
        return batch_idx, target_idx
    generator = None
    if not training:
        generator = torch.Generator(device=batch_idx.device)
        generator.manual_seed(len(batch_idx))
    keep = torch.randperm(len(batch_idx), device=batch_idx.device, generator=generator)
    keep = keep[:max_targets]
    return batch_idx[keep], target_idx[keep]


def _spread_galaxy_components(components, is_gal_comp, patch):
    """Convolve galaxy components' decoded PSF stamps with a fixed round exponential.

    The exponential (n=1 Sersic, re=GALAXY_COMPONENT_RE) is deliberately crude: the
    GLS amplitude absorbs total flux, and the convolution only needs to spread the
    light enough that the galaxy stops biasing its neighbors' point-source fits.
    """
    flat_gal = is_gal_comp.reshape(-1)
    if not flat_gal.any():
        return components

    half = patch // 2
    grid = sample_grid(patch, 1, components.device, components.dtype)
    radii = grid.norm(dim=-1)
    kernel = torch.exp(-SERSIC_B1 * radii / GALAXY_COMPONENT_RE).reshape(patch, patch)
    kernel = kernel / kernel.sum()

    stamps = components.reshape(-1, patch, patch)[flat_gal]
    full = 2 * patch - 1
    conv = torch.fft.irfft2(
        torch.fft.rfft2(stamps, s=(full, full)) * torch.fft.rfft2(kernel, s=(full, full)),
        s=(full, full),
    )
    # kernel center sits at grid index (half, half): aligned crop starts there
    spread = conv[:, half : half + patch, half : half + patch]
    spread = spread / spread.sum(dim=(-2, -1), keepdim=True).clamp(min=1e-8)

    out = components.reshape(-1, patch, patch).clone()
    out[flat_gal] = spread
    return out.reshape(components.shape)


def _galaxy_pixel_mask(positions, star_types, fluxes, batch_idx, target_idx, grid, radius):
    """(N, n_pix) multiplier zeroing pixels within GALAXY_MASK_RADIUS of any galaxy."""
    is_galaxy = torch.zeros_like(fluxes, dtype=torch.bool)
    for t in STAR_TYPES_GALAXY:
        is_galaxy |= star_types == t
    is_galaxy &= fluxes > 0

    cheb = chebyshev_distances(positions)
    near_gal = (cheb < radius) & is_galaxy.unsqueeze(1)  # (b, s_target, s_other)

    n_targets = len(batch_idx)
    keep = torch.ones(n_targets, grid.shape[0], device=positions.device)
    rows = near_gal[batch_idx, target_idx]  # (N, s)
    target_pos = positions[batch_idx, target_idx]
    for n in range(n_targets):  # galaxies near a target are rare; sparse loop is cheap
        gal_slots = torch.nonzero(rows[n], as_tuple=True)[0]
        for g in gal_slots:
            shift = positions[batch_idx[n], g] - target_pos[n].round()
            dist = (grid - shift).abs().amax(dim=-1)
            keep[n] = keep[n] * (dist > GALAXY_MASK_RADIUS).float()
    return keep
