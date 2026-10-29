"""Pure output adaptation, tested without model weights."""
import numpy as np


def as_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def merge_refined_masks(cached, refined):
    """Merge tracker refinements without discarding them when cache is absent.

    ``None`` and empty mappings mean no objects. A present object with a None
    mask is invalid; an all-zero mask is retained for downstream filtering.
    Inputs are not mutated.
    """
    result = dict(cached) if cached is not None else {}
    for obj_id, mask in (refined or {}).items():
        if mask is None:
            raise ValueError(f"Refinement for object {obj_id} has no mask")
        result[obj_id] = mask
    return result


def diagnostic_tpos_maximum(original_maximum, fixed_maximum=None, *, dummy=False):
    """Override only the temporal-encoding maximum in a diagnostic probe.

    This does not change the video's frame count or which object pointers are
    selected. Dummy/no-maximum encodings are left untouched.
    """
    if dummy or original_maximum is None or fixed_maximum is None:
        return original_maximum
    if original_maximum <= 1 or fixed_maximum <= 1:
        raise ValueError("Temporal encoding maxima must be greater than one")
    return fixed_maximum


def unique_mask(outputs, shape):
    """Accept exactly one nonempty instance; never silently union multiple objects.

    Stateless SAM calls do not establish identity among identical instances.
    Ambiguous prompts are marked missing and must be refined or manually initialized.
    """
    if "out_binary_masks" not in outputs:
        raise ValueError("SAM output lacks out_binary_masks; inspect installed API, do not guess")
    masks = as_numpy(outputs["out_binary_masks"])
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim != 3 or tuple(masks.shape[1:]) != tuple(shape):
        raise ValueError(f"Expected original-resolution [N,H,W] masks; received {masks.shape}")
    masks = masks.astype(bool)
    nonempty = masks.reshape(len(masks), -1).any(axis=1) if len(masks) else np.zeros(0, bool)
    masks = masks[nonempty]
    if len(masks) != 1:
        return np.zeros(shape, bool), "missing" if not len(masks) else "ambiguous"
    return masks[0], "observed"
