"""Trace DAVIS box/point propagation stages without modifying upstream SAM.

The local raw JSON trace is gitignored. Instrumentation wraps pinned upstream
methods and records IDs/areas/scores, filtering, emission order and collection.
"""
import modal

app = modal.App("odvla-sam31-propagation-trace")
volume = modal.Volume.from_name("object-dynamics-vla-data")
image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("git")
         .pip_install("torch==2.10.0", "torchvision", index_url="https://download.pytorch.org/whl/cu128")
         .run_commands("git clone https://github.com/facebookresearch/sam3.git /opt/sam3",
                       "cd /opt/sam3 && git checkout f1647e8d0c1d280991f4d1b7d974c2704537fe66")
         .pip_install("/opt/sam3")
         .pip_install("numpy<2", "Pillow", "opencv-python-headless", "setuptools", "einops", "pycocotools", "psutil")
         .add_local_python_source("odvla")
         .add_local_file("scripts/extract_sam31.py", "/root/extract_sam31.py"))


@app.function(image=image, gpu="A100-80GB", volumes={"/data": volume}, timeout=3600)
def trace():
    import inspect
    import json
    import sys
    from pathlib import Path
    import numpy as np
    import torch
    from PIL import Image
    from odvla.perception import merge_refined_masks
    from sam3.model_builder import build_sam3_multiplex_video_predictor
    sys.path.insert(0, "/root")
    from extract_sam31 import _patch_multiplex_init_state

    root = Path("/data/davis2017/human_init_probe")
    metadata = json.loads((root / "metadata.json").read_text())
    gt = np.asarray(Image.open(root / "masks/000000.png"))
    h, w = gt.shape
    object_id = min(int(i) for i in np.unique(gt) if i not in (0, 255))
    ys, xs = np.where(gt == object_id)
    box = [float(xs.min() / w), float(ys.min() / h),
           float((xs.max() - xs.min() + 1) / w), float((ys.max() - ys.min() + 1) / h)]
    nearest = np.argmin((xs - xs.mean()) ** 2 + (ys - ys.mean()) ** 2)
    point = [float((xs[nearest] + .5) / w), float((ys[nearest] + .5) / h)]
    predictor = build_sam3_multiplex_video_predictor(
        checkpoint_path="/data/checkpoints/sam3.1/sam3.1_multiplex.pt",
        use_fa3=False, compile=False, warm_up=False, async_loading_frames=False)
    _patch_multiplex_init_state(predictor)
    model = predictor.model
    underlying_signature = str(inspect.signature(model.propagate_in_video))
    explicitly_accepts_force = "force_tracker_propagation" in inspect.signature(model.propagate_in_video).parameters
    cases = []
    for prompt_kind in ("box", "point"):
        paths = ("official_wrapper", "direct_model", "wrapper_missing_cache_workaround") if prompt_kind == "point" else ("official_wrapper", "direct_model")
        for path in paths:
            events, output_frames, out_to_frame = [], [], {}
            progress = {"processed": 0, "latest_frame": None, "phase": "initialization", "in_full": False}
            original_run = model._run_single_frame_inference
            original_partial = model._propogate_tracker_one_frame_local_gpu
            original_batch = model._postprocess_output_batched
            original_post = model._postprocess_output
            original_parse = model.parse_action_history_for_propagation
            original_build = model._build_sam2_output

            def scalar(value):
                if value is None:
                    return None
                if hasattr(value, "detach"):
                    value = value.detach().float().cpu().numpy()
                a = np.asarray(value)
                return float(a.reshape(-1)[0]) if a.size else None

            def values(value):
                if value is None:
                    return None
                if hasattr(value, "detach"):
                    value = value.detach().cpu().numpy()
                return [int(x) for x in list(value)]

            def raw(out):
                result = []
                for oid, mask in out.get("obj_id_to_mask", {}).items():
                    area = int(mask.sum().item()) if hasattr(mask, "sum") else None
                    result.append({"id": int(oid), "mask_area": area,
                        "detection_score": scalar(out.get("obj_id_to_score", {}).get(oid)),
                        "tracker_score": scalar(out.get("obj_id_to_sam2_score", {}).get(oid))})
                return result

            def public(out):
                return [{"id": int(oid), "area": int(mask.sum()), "score": float(score)}
                        for oid, mask, score in zip(out["out_obj_ids"], out["out_binary_masks"], out["out_probs"])]

            def run_wrap(*args, **kwargs):
                frame = int(kwargs.get("frame_idx", args[1] if len(args) > 1 else -1))
                progress["processed"] += 1
                progress["latest_frame"] = frame
                progress["in_full"] = True
                try:
                    result = original_run(*args, **kwargs)
                finally:
                    progress["in_full"] = False
                state = kwargs.get("inference_state", args[0] if args else {})
                metadata = state.get("tracker_metadata", {})
                out_to_frame[id(result)] = frame
                events.append({"stage": "internal_full", "phase": progress["phase"], "frame": frame, "processed": progress["processed"],
                    "objects": raw(result), "removed": values(result.get("removed_obj_ids")),
                    "suppressed": values(result.get("suppressed_obj_ids")),
                    "unconfirmed": values(result.get("unconfirmed_obj_ids")),
                    "active_tracker_ids": values(metadata.get("obj_ids_all_gpu"))})
                return result

            def partial_wrap(*args, **kwargs):
                result = original_partial(*args, **kwargs)
                frame = int(kwargs.get("frame_idx", -1))
                if not progress["in_full"]:
                    progress["processed"] += 1
                progress["latest_frame"] = frame
                ids, masks, scores = result
                objects = []
                for i, oid in enumerate(ids):
                    mask = masks[i]
                    objects.append({"id": int(oid), "positive_logit_area": int((mask > 0).sum().item()),
                                    "mask_shape": list(mask.shape), "tracker_score": scalar(scores[i])})
                events.append({"stage": "internal_partial", "phase": progress["phase"], "frame": frame,
                               "processed": progress["processed"], "objects": objects})
                return result

            def batch_wrap(height, width, batch):
                result = original_batch(height, width, batch)
                for (out, removed, suppressed, unconfirmed), response in zip(batch, result):
                    events.append({"stage": "filter_batched", "phase": progress["phase"], "frame": out_to_frame.get(id(out)),
                        "processed": progress["processed"], "raw": raw(out),
                        "removed": values(removed), "suppressed": values(suppressed),
                        "unconfirmed": values(unconfirmed), "public": public(response)})
                return result

            def post_wrap(state, out, removed_obj_ids=None, suppressed_obj_ids=None, unconfirmed_obj_ids=None):
                result = original_post(state, out, removed_obj_ids, suppressed_obj_ids, unconfirmed_obj_ids)
                events.append({"stage": "filter_single", "phase": progress["phase"], "frame": out_to_frame.get(id(out), progress["latest_frame"]),
                    "processed": progress["processed"], "raw": raw(out), "removed": values(removed_obj_ids),
                    "suppressed": values(suppressed_obj_ids), "unconfirmed": values(unconfirmed_obj_ids),
                    "public": public(result)})
                return result

            def parse_wrap(state):
                kind, ids = original_parse(state)
                events.append({"stage": "propagation_type", "type": kind, "object_ids": values(ids)})
                return kind, ids

            def build_wrap(state, frame_idx, refined_obj_id_to_mask=None):
                cache_present = frame_idx in state["cached_frame_outputs"]
                refined_ids = sorted(int(x) for x in (refined_obj_id_to_mask or {}))
                workaround = (path == "wrapper_missing_cache_workaround" and
                              progress["phase"] == "propagation" and not cache_present and
                              refined_obj_id_to_mask is not None)
                if path == "wrapper_missing_cache_workaround" and progress["phase"] == "propagation":
                    result = merge_refined_masks(state["cached_frame_outputs"].get(frame_idx), refined_obj_id_to_mask)
                else:
                    result = original_build(state, frame_idx, refined_obj_id_to_mask)
                events.append({"stage": "cache_merge", "phase": progress["phase"], "frame": int(frame_idx),
                               "processed": progress["processed"], "cache_present": cache_present,
                               "refined_ids": refined_ids, "output_ids": sorted(int(x) for x in result),
                               "workaround_applied": workaround})
                return result

            model._run_single_frame_inference = run_wrap
            model._propogate_tracker_one_frame_local_gpu = partial_wrap
            model._postprocess_output_batched = batch_wrap
            model._postprocess_output = post_wrap
            model.parse_action_history_for_propagation = parse_wrap
            model._build_sam2_output = build_wrap
            sid = predictor.handle_request({"type": "start_session", "resource_path": str(root / "frames"),
                                            "offload_video_to_cpu": True})["session_id"]
            try:
                request = {"type": "add_prompt", "session_id": sid, "frame_index": 0, "output_prob_thresh": 0.0}
                if prompt_kind == "box":
                    request.update(bounding_boxes=[box], bounding_box_labels=[1])
                else:
                    request.update(points=[point], point_labels=[1], obj_id=1)
                initial = predictor.handle_request(request)["outputs"]
                ids = np.asarray(initial["out_obj_ids"])
                masks = np.asarray(initial["out_binary_masks"])
                reference = gt == object_id
                overlaps = []
                for mask in masks:
                    union = np.logical_or(mask, reference).sum()
                    overlaps.append(float(np.logical_and(mask, reference).sum() / union) if union else None)
                target = int(ids[0]) if len(ids) == 1 else (1 if prompt_kind == "point" else None)
                progress.update(processed=0, latest_frame=None, phase="propagation")
                state = predictor._get_session(sid)["state"]
                request_args = {"start_frame_idx": 0, "max_frame_num_to_track": None,
                                "reverse": False, "output_prob_thresh": 0.0}
                if path != "direct_model":
                    iterator = predictor.handle_stream_request({"type": "propagate_in_video", "session_id": sid,
                        "propagation_direction": "forward", "start_frame_index": 0,
                        "max_frame_num_to_track": None, "output_prob_thresh": 0.0,
                        "force_tracker_propagation": True})
                else:
                    iterator = ({"frame_index": frame, "outputs": out} for frame, out in
                                model.propagate_in_video(state, **request_args))
                for item in iterator:
                    frame = int(item["frame_index"])
                    response = public(item["outputs"])
                    matched = [entry for entry in response if entry["id"] == target]
                    first_iou = None
                    if frame == 0 and len(matched) == 1:
                        ix = list(item["outputs"]["out_obj_ids"]).index(target)
                        mask = item["outputs"]["out_binary_masks"][ix]
                        union = np.logical_or(mask, reference).sum()
                        first_iou = float(np.logical_and(mask, reference).sum() / union) if union else None
                    output_frames.append({"frame": frame, "processed": progress["processed"],
                        "latest_processed_frame": progress["latest_frame"], "response": response,
                        "collector_target": target, "collector_matches": len(matched),
                        "collector_stored_area": matched[0]["area"] if len(matched) == 1 else 0,
                        "frame0_iou": first_iou})
                cases.append({"prompt": prompt_kind, "path": path, "box": box if prompt_kind == "box" else None,
                    "point": point if prompt_kind == "point" else None, "initial": public(initial),
                    "initial_mask_iou": overlaps, "target_id": target,
                    "request": {**request_args, "force_requested": path != "direct_model",
                                "force_forwarded_by_wrapper": explicitly_accepts_force if path != "direct_model" else False},
                    "events": events, "emitted": output_frames})
            finally:
                predictor.handle_request({"type": "close_session", "session_id": sid})
                model._run_single_frame_inference = original_run
                model._propogate_tracker_one_frame_local_gpu = original_partial
                model._postprocess_output_batched = original_batch
                model._postprocess_output = original_post
                model.parse_action_history_for_propagation = original_parse
                model._build_sam2_output = original_build
    report = {"source": metadata, "image_width": w, "image_height": h, "annotation_object_id": object_id,
              "model_signature": underlying_signature, "force_explicitly_accepted": explicitly_accepts_force,
              "hotstart_delay": model.hotstart_delay, "postprocess_batch_size": model.postprocess_batch_size,
              "cases": cases}
    output = Path("/data/eval/davis_sam31_stage_trace_v2.json")
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    volume.commit()
    return {"path": str(output), "signature": underlying_signature,
            "force_explicitly_accepted": explicitly_accepts_force,
            "hotstart_delay": model.hotstart_delay, "postprocess_batch_size": model.postprocess_batch_size,
            "cases": [{"prompt": c["prompt"], "path": c["path"], "initial_iou": c["initial_mask_iou"],
                       "type": [e for e in c["events"] if e["stage"] == "propagation_type"],
                       "first_emissions": c["emitted"][:11]} for c in cases]}


@app.local_entrypoint()
def main():
    print(trace.remote())
