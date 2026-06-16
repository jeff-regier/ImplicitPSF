"""Render ImplicitPSF model stamps for evaluation, mirroring the baseline runners.

Reserved stars are excluded from the attention context (key_padding_mask) but still
queried, which is the model's native reserved-star protocol.
"""

import torch

from implicitpsf.implicit_psf import ImplicitPSF


def load_model(checkpoint_path, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = ImplicitPSF(**checkpoint["hyperparameters"])
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval()


def render_implicit(model, batch, reserved_mask, oversample=1):
    """Unit-flux model stamps for every star, reserved stars out of context.

    Args:
        model: a trained ImplicitPSF in eval mode
        batch: single-exposure batch dict (see datasets.make_batch), batch size 1
        reserved_mask: (1, n_stars) bool, True = reserved for evaluation
        oversample: samples per native pixel

    Returns:
        (1, n_stars, k, k) stamps on each star's data pixel grid
    """
    device = next(model.parameters()).device
    context_mask = (batch["flux"] > 0) & ~reserved_mask
    with torch.no_grad():
        stamps = model(
            batch["cutouts"].to(device),
            batch["positions"].to(device),
            batch["colors"].to(device),
            batch["flux"].to(device),
            context_mask.to(device),
            oversample=oversample,
        )
    return stamps.cpu()
