"""Experimental SAM 3.1 point-path fixes; not a certified online adapter.

The context modifies model-wide methods and must not be shared concurrently
between sessions. Callers remain responsible for real frame availability.
"""
from contextlib import contextmanager

from .perception import diagnostic_tpos_maximum, merge_refined_masks


@contextmanager
def experimental_point_adapter(predictor, *, disable_detector_prefetch=False):
    """Install missing-cache merge, fixed pointer scale and past-only pointers.

    The scale is derived from the configured pointer maximum. No sequence length,
    frame-selection bound, or spatial-memory selection is changed.
    """
    model = predictor.model
    tracker = model.tracker.model
    maximum = int(tracker.max_obj_ptrs_in_encoder)
    if maximum <= 1 or not tracker.use_obj_ptrs_in_encoder or not tracker.add_tpos_enc_to_obj_ptrs:
        raise ValueError("Point adapter requires enabled temporal object pointers with maximum > 1")
    original_merge = model._build_sam2_output
    original_tpos = tracker._get_tpos_enc
    original_past_only = tracker.only_obj_ptrs_in_the_past_for_eval
    original_prepare = model._prepare_backbone_feats if disable_detector_prefetch else None
    original_chunk = model.detector._build_multigpu_buffer_next_chunk if disable_detector_prefetch else None
    active_frame = {"index": None}

    def merge(state, frame_idx, refined_obj_id_to_mask=None):
        return merge_refined_masks(state["cached_frame_outputs"].get(frame_idx), refined_obj_id_to_mask)

    def tpos(rel_pos_list, device, max_abs_pos=None, dummy=False):
        scale = diagnostic_tpos_maximum(max_abs_pos, maximum, dummy=dummy)
        return original_tpos(rel_pos_list, device, max_abs_pos=scale, dummy=dummy)

    def prepare(state, frame_idx, reverse):
        if reverse:
            raise ValueError("The experimental prefetch guard supports forward tracking only")
        active_frame["index"] = int(frame_idx)
        try:
            return original_prepare(state, frame_idx, reverse)
        finally:
            active_frame["index"] = None

    def chunk(*args, **kwargs):
        if active_frame["index"] is None:
            raise RuntimeError("Detector chunk requested outside a bounded frame operation")
        if args:
            raise RuntimeError("Unexpected positional detector chunk API; cannot audit frame bounds")
        start, end = kwargs["frame_idx_begin"], kwargs["frame_idx_end"]
        if start > active_frame["index"]:
            return None  # Optional upstream lookahead; current chunk remains available.
        if end > active_frame["index"] + 1:
            raise RuntimeError("Detector chunk includes unavailable future frames")
        return original_chunk(*args, **kwargs)

    model._build_sam2_output = merge
    tracker._get_tpos_enc = tpos
    tracker.only_obj_ptrs_in_the_past_for_eval = True
    if disable_detector_prefetch:
        model._prepare_backbone_feats = prepare
        model.detector._build_multigpu_buffer_next_chunk = chunk
    try:
        yield {"experimental": True, "initialization": "explicit point required",
               "missing_cache_merge": True, "pointer_scale_max": maximum,
               "pointer_temporal_denominator": maximum - 1, "past_only_pointers": True,
               "detector_prefetch_disabled": disable_detector_prefetch,
               "real_sequence_length_preserved": True, "validated_online": False}
    finally:
        model._build_sam2_output = original_merge
        tracker._get_tpos_enc = original_tpos
        tracker.only_obj_ptrs_in_the_past_for_eval = original_past_only
        if disable_detector_prefetch:
            model._prepare_backbone_feats = original_prepare
            model.detector._build_multigpu_buffer_next_chunk = original_chunk
