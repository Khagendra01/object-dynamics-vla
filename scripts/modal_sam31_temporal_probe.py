"""Diagnose sequence-length-dependent pointer temporal encoding on DAVIS.

The fixed-normalizer intervention changes only _get_tpos_enc's max_abs_pos;
the real video length and upstream pointer selection/bounds remain untouched.
"""
import modal
from scripts.modal_davis_human_init import inference_image, volume

image = (inference_image.add_local_file("scripts/__init__.py", "/root/scripts/__init__.py")
         .add_local_file("scripts/modal_davis_human_init.py", "/root/scripts/modal_davis_human_init.py"))
app = modal.App("odvla-sam31-temporal-probe")
OUT = "/data/eval/sam31_temporal_probe_v1"
LENGTHS = (11, 12, 15, 16, 17, 69)


@app.function(image=image, gpu="A100-80GB", volumes={"/data": volume}, timeout=1800,
              single_use_containers=True)
def run_case(length: int, fixed: bool, repeat: int = 1):
    import hashlib
    import inspect
    import json
    import shutil
    import sys
    import tempfile
    from pathlib import Path
    import numpy as np
    import torch
    from PIL import Image
    from odvla.perception import diagnostic_tpos_maximum, merge_refined_masks
    from sam3.model_builder import build_sam3_multiplex_video_predictor
    sys.path.insert(0, "/root")
    from extract_sam31 import _patch_multiplex_init_state

    if length not in LENGTHS or repeat not in (1, 2):
        raise ValueError("Unexpected length or repeat")
    root = Path("/data/davis2017/human_init_probe")
    metadata = json.loads((root / "metadata.json").read_text())
    if length > metadata["frames"]:
        raise ValueError("Requested more frames than exist")
    first = np.asarray(Image.open(root / "masks/000000.png"))
    ids = sorted(int(x) for x in np.unique(first) if x not in (0, 255))
    ys, xs = np.where(first == ids[0])
    nearest = np.argmin((xs - xs.mean()) ** 2 + (ys - ys.mean()) ** 2)
    point = [float((xs[nearest] + .5) / first.shape[1]), float((ys[nearest] + .5) / first.shape[0])]
    torch.manual_seed(2468)
    torch.cuda.manual_seed_all(2468)
    np.random.seed(2468)
    torch.backends.cudnn.benchmark = False
    predictor = build_sam3_multiplex_video_predictor(
        checkpoint_path="/data/checkpoints/sam3.1/sam3.1_multiplex.pt",
        use_fa3=False, compile=False, warm_up=False, async_loading_frames=False)
    _patch_multiplex_init_state(predictor)
    constructor_flags = {"matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                         "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32)}
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tracker = predictor.model.tracker.model
    original_tpos = tracker._get_tpos_enc
    original_merge = predictor.model._build_sam2_output
    original_propagate = predictor.model._propogate_tracker_one_frame_local_gpu
    events, processed, captured = [], [], {}

    def tpos(rel_pos_list, device, max_abs_pos=None, dummy=False):
        caller = inspect.currentframe().f_back
        local = caller.f_locals
        frame = int(local.get("frame_idx", -1))
        maximum = diagnostic_tpos_maximum(max_abs_pos, 16 if fixed else None, dummy=dummy)
        output_dict = local.get("output_dict", {})
        lookup = {}
        for group in ("cond_frame_outputs", "non_cond_frame_outputs"):
            for index, value in output_dict.get(group, {}).items():
                lookup.setdefault(id(value), []).append({"frame": int(index), "source": group})
        selected = []
        for distance, value, is_cond in local.get("filtered_data", []):
            selected.append({"relative_position": float(distance), "is_conditioning": bool(is_cond),
                             "source_matches": lookup.get(id(value), [])})
        events.append({"frame": frame, "implementation": caller.f_code.co_name,
            "num_frames": int(local["num_frames"]) if local.get("num_frames") is not None else None,
            "pointer_budget": int(max_abs_pos) if max_abs_pos is not None else None,
            "normalizer_max": int(maximum) if maximum is not None else None,
            "normalization_denominator": int(maximum - 1) if maximum is not None and not dummy else None,
            "selected_pointers": selected, "dummy": bool(dummy)})
        if fixed and frame <= 10:
            frames = [match["frame"] for item in selected for match in item["source_matches"]]
            if len(frames) != len(selected) or any(index > frame for index in frames):
                raise RuntimeError("Could not verify that selected pointers are available at the current frame")
        return original_tpos(rel_pos_list, device, max_abs_pos=maximum, dummy=dummy)

    def merge(state, frame_idx, refined_obj_id_to_mask=None):
        return merge_refined_masks(state["cached_frame_outputs"].get(frame_idx), refined_obj_id_to_mask)

    def propagate(*args, **kwargs):
        result = original_propagate(*args, **kwargs)
        frame = int(kwargs["frame_idx"])
        processed.append(frame)
        if frame == 10:
            obj_ids, logits, _ = result
            matches = [i for i, obj_id in enumerate(obj_ids) if int(obj_id) == 1]
            if len(matches) == 1:
                captured["logits"] = logits[matches[0]].detach().float().cpu().numpy()
        return result

    hashes = []
    with tempfile.TemporaryDirectory() as folder:
        for i in range(length):
            source = root / "frames" / f"{i:06}.jpg"
            hashes.append(hashlib.sha256(source.read_bytes()).hexdigest())
            shutil.copyfile(source, Path(folder) / f"{i:06}.jpg")
        sid = predictor.handle_request({"type": "start_session", "resource_path": folder,
                                        "offload_video_to_cpu": True})["session_id"]
        try:
            predictor.handle_request({"type": "add_prompt", "session_id": sid, "frame_index": 0,
                                      "points": [point], "point_labels": [1], "obj_id": 1})
            predictor.model._build_sam2_output = merge
            predictor.model._propogate_tracker_one_frame_local_gpu = propagate
            tracker._get_tpos_enc = tpos
            iterator = predictor.handle_stream_request({"type": "propagate_in_video", "session_id": sid,
                        "propagation_direction": "forward", "start_frame_index": 0,
                        "output_prob_thresh": 0.0})
            for item in iterator:
                if int(item["frame_index"]) == 10:
                    ids_out = np.asarray(item["outputs"]["out_obj_ids"])
                    found = np.flatnonzero(ids_out == 1)
                    if len(found) != 1 or "logits" not in captured:
                        raise RuntimeError("Frame 10 lacks target or pre-threshold logits")
                    captured["mask"] = np.asarray(item["outputs"]["out_binary_masks"])[found[0]]
                    break
            iterator.close()
            if "mask" not in captured:
                raise RuntimeError("Frame 10 was not emitted")
        finally:
            tracker._get_tpos_enc = original_tpos
            predictor.model._build_sam2_output = original_merge
            predictor.model._propogate_tracker_one_frame_local_gpu = original_propagate
            predictor.handle_request({"type": "close_session", "session_id": sid})
    at_frame = [event for event in events if event["frame"] == 10]
    if not at_frame or not any(event["selected_pointers"] for event in at_frame):
        raise RuntimeError("No active pointer temporal encoding observed at frame 10")
    out = Path(OUT)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{'fixed' if fixed else 'baseline'}_{length}_{repeat}"
    if (out / (stem + ".json")).exists():
        raise FileExistsError(f"Refusing to overwrite {stem}")
    np.savez_compressed(out / (stem + ".npz"), **captured)
    report = {"video": metadata["video"], "length": length, "fixed_normalizer": fixed, "repeat": repeat,
        "point_xy": point, "seed": 2468, "checkpoint_sha256": "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6",
        "input_prefix_sha256": hashlib.sha256("".join(hashes[:11]).encode()).hexdigest(),
        "tracker_class": f"{type(tracker).__module__}.{type(tracker).__qualname__}",
        "memory_attention_class": f"{type(tracker.transformer).__module__}.{type(tracker.transformer).__qualname__}",
        "memory_encoder_class": f"{type(tracker.transformer.encoder).__module__}.{type(tracker.transformer.encoder).__qualname__}",
        "memory_layer_class": f"{type(tracker.transformer.encoder.layers[0]).__module__}.{type(tracker.transformer.encoder.layers[0]).__qualname__}",
        "memory_cross_attention_class": f"{type(tracker.transformer.encoder.layers[0].cross_attention_rope).__module__}.{type(tracker.transformer.encoder.layers[0].cross_attention_rope).__qualname__}",
        "pointer_config": {"use_obj_ptrs": bool(tracker.use_obj_ptrs_in_encoder),
                           "add_tpos": bool(tracker.add_tpos_enc_to_obj_ptrs),
                           "max_obj_ptrs": int(tracker.max_obj_ptrs_in_encoder),
                           "past_only": bool(tracker.only_obj_ptrs_in_the_past_for_eval),
                           "memory_selection": bool(tracker.use_memory_selection)},
        "tf32_after_constructor": constructor_flags,
        "tf32_at_emission": {"matmul": bool(torch.backends.cuda.matmul.allow_tf32),
                             "cudnn": bool(torch.backends.cudnn.allow_tf32)},
        "num_frames_in_session": length, "frames_processed_at_emission": len(processed),
        "last_processed_frame": processed[-1], "tpos_events": events,
        "mask_area": int(captured["mask"].sum()), "logit_shape": list(captured["logits"].shape)}
    (out / (stem + ".json")).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    volume.commit()
    return {k: v for k, v in report.items() if k != "tpos_events"} | {"frame10_tpos": at_frame}


@app.function(image=image, volumes={"/data": volume}, timeout=600)
def summarize():
    import json
    from pathlib import Path
    import numpy as np
    root = Path(OUT)
    cases = {}
    arrays = {}
    for mode in ("baseline", "fixed"):
        for length in LENGTHS:
            name = f"{mode}_{length}_1"
            cases[name] = json.loads((root / f"{name}.json").read_text())
            arrays[name] = dict(np.load(root / f"{name}.npz"))
        for length in (11, 16):
            name = f"{mode}_{length}_2"
            cases[name] = json.loads((root / f"{name}.json").read_text())
            arrays[name] = dict(np.load(root / f"{name}.npz"))
    pairs = []
    for mode in ("baseline", "fixed"):
        pairs.extend((f"{mode}_{length}_1", f"{mode}_16_1") for length in LENGTHS)
        pairs.extend((f"{mode}_{length}_1", f"{mode}_{length}_2") for length in (11, 16))
    pairs.extend((f"baseline_{length}_1", f"fixed_{length}_1") for length in LENGTHS)
    comparisons = []
    for left, right in pairs:
        a, b = arrays[left], arrays[right]
        diff = np.abs(a["logits"] - b["logits"])
        left_ptr = [e["selected_pointers"] for e in cases[left]["tpos_events"] if e["frame"] == 10]
        right_ptr = [e["selected_pointers"] for e in cases[right]["tpos_events"] if e["frame"] == 10]
        left_history = [(e["frame"], e["selected_pointers"]) for e in cases[left]["tpos_events"] if 0 <= e["frame"] <= 10]
        right_history = [(e["frame"], e["selected_pointers"]) for e in cases[right]["tpos_events"] if 0 <= e["frame"] <= 10]
        comparisons.append({"left": left, "right": right, "pointer_selection_equal": left_ptr == right_ptr,
            "pointer_selection_equal_through_frame10": left_history == right_history,
            "logit_max_abs_diff": float(diff.max()), "logit_mean_abs_diff": float(diff.mean()),
            "logit_sign_flips": int(np.count_nonzero((a["logits"] > 0) != (b["logits"] > 0))),
            "mask_different_pixels": int(np.count_nonzero(a["mask"] != b["mask"]))})
    result = {"cases": cases, "comparisons": comparisons,
              "scope": "one human-annotation-initialized DAVIS point track; diagnostic tpos argument override only"}
    (root / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    volume.commit()
    return {"path": str(root / "summary.json"), "comparisons": comparisons}


@app.function(image=image, gpu="A100-80GB", volumes={"/data": volume}, timeout=600,
              single_use_containers=True)
def inspect_model():
    import json
    from pathlib import Path
    import torch
    from sam3.model_builder import build_sam3_multiplex_video_predictor
    predictor = build_sam3_multiplex_video_predictor(
        checkpoint_path="/data/checkpoints/sam3.1/sam3.1_multiplex.pt",
        use_fa3=False, compile=False, warm_up=False, async_loading_frames=False)
    tracker = predictor.model.tracker.model
    encoder = tracker.transformer.encoder
    layer = encoder.layers[0]
    def name(obj):
        return f"{type(obj).__module__}.{type(obj).__qualname__}"
    report = {"tracker": name(tracker), "wrapper": name(tracker.transformer),
              "memory_encoder": name(encoder), "memory_layer": name(layer),
              "memory_cross_attention": name(layer.cross_attention_rope),
              "layers": len(encoder.layers), "use_fa3": False, "compile": False,
              "constructor_tf32": {"matmul": bool(torch.backends.cuda.matmul.allow_tf32),
                                    "cudnn": bool(torch.backends.cudnn.allow_tf32)}}
    path = Path(OUT) / "memory_implementation.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    volume.commit()
    return report


@app.local_entrypoint()
def main(summary_only: bool = False, inspect_only: bool = False):
    if inspect_only:
        print(inspect_model.remote())
        return
    if summary_only:
        print(summarize.remote())
        return
    for fixed in (False, True):
        for length in LENGTHS:
            print(run_case.remote(length, fixed, 1))
        for length in (16, 11):
            print(run_case.remote(length, fixed, 2))
    print(summarize.remote())
