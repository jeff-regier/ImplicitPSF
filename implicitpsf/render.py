"""Query the trained PSF field at arbitrary positions — the project's deliverable API.

A query is a "virtual star": appended to the exposure's star list with an empty
cutout, excluded from the attention keys (context_mask False), but still a query.
What the model returns is the PIXEL-CONVOLVED PSF (the "effective PSF"): each value
is the expected flux of a 1-pixel-square region centered at the sample point, because
that is what the training targets are. `oversample` evaluates that same function on a
finer grid of offsets — it does NOT undo the one-pixel blur; no sample represents the
unconvolved PSF. This is exactly what galaxy fitting needs: a pixel of a galaxy image
is (galaxy * PSF * pixel) = (galaxy * effectivePSF) at the pixel center, so render
the galaxy profile at the same fine sampling, FFT-convolve with these kernels, and
read off native pixel centers WITHOUT a second pixel integration (galsim's no_pixel
convention). If the unconvolved PSF is ever required (deconvolution, undersampled
imagers), the extension is to move the pixel integration inside the training loss.
"""

import torch


def render_at(model, batch, query_positions, query_colors, query_fluxes=None, oversample=1):
    """Render PSF kernels at arbitrary positions of a single exposure.

    Args:
        model: trained ImplicitPSF in eval mode
        batch: single-exposure batch dict (see datasets.make_batch), batch size 1
        query_positions: (m, 2) CCD pixel coordinates (x, y)
        query_colors: (m,) g-i colors of the objects whose PSF is wanted
        query_fluxes: (m,) fluxes for brighter-fatter conditioning; defaults to the
            median context-star flux
        oversample: samples per native pixel along each axis

    Returns:
        (m, patch * oversample, patch * oversample) kernels, unit native-pixel sum
    """
    flux = batch["flux"]
    if flux.shape[0] != 1:
        raise ValueError("render_at expects a single-exposure batch")
    real = flux[0] > 0
    if query_fluxes is None:
        query_fluxes = flux[0][real].median().expand(len(query_positions))

    n_queries = len(query_positions)
    patch = model.patch_size
    cutouts = torch.cat([batch["cutouts"], torch.zeros(1, n_queries, patch, patch)], dim=1)
    positions = torch.cat([batch["positions"], query_positions.unsqueeze(0)], dim=1)
    colors = torch.cat([batch["colors"], query_colors.unsqueeze(0)], dim=1)
    fluxes = torch.cat([flux, query_fluxes.unsqueeze(0)], dim=1)
    context = real
    if model.point_source_context:  # match training: only point sources as context keys
        st = batch["star_types"][0]
        context = real & ((st == 0) | (st == 1) | (st == 5))
    context_mask = torch.cat(
        [context.unsqueeze(0), torch.zeros(1, n_queries, dtype=torch.bool)], dim=1
    )

    with torch.no_grad():
        stamps = model(cutouts, positions, colors, fluxes, context_mask, oversample=oversample)
    return stamps[0, -n_queries:]
