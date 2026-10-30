"""Bounded prefix audit and initialized RGB tracking pilot, not a VLA policy.

DAVIS points are derived from human annotations. LIBERO's one milk point is
agent-selected from the native RGB first frame. Ground truth is read only after
tracking, for scoring, and never as a policy input or prompt.
"""
import modal
from scripts.modal_davis_human_init import inference_image, volume

image = (inference_image.add_local_file("scripts/__init__.py", "/root/scripts/__init__.py")
         .add_local_file("scripts/modal_davis_human_init.py", "/root/scripts/modal_davis_human_init.py"))
app = modal.App("odvla-sam31-point-pilot")


@app.function(image=image, gpu="A100-80GB", volumes={"/data": volume}, timeout=7200,
              single_use_containers=True)
def pilot(dataset: str, clip_index: int = 0, run_name: str = "v3"):
    import inspect
    import json
    import re
    import shutil
    import sys
    import tempfile
    from pathlib import Path
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    from odvla.experimental_sam31 import experimental_point_adapter
    from odvla.measurements import mask_metrics
    from sam3.model_builder import build_sam3_multiplex_video_predictor
    sys.path.insert(0, "/root")
    from extract_sam31 import _patch_multiplex_init_state

    if dataset == "davis":
        if clip_index not in (0, 1, 2):
            raise ValueError("DAVIS clip_index must be 0, 1, or 2")
        root = Path("/data/davis2017/human_init_probe" if clip_index == 0 else
                    f"/data/davis2017/human_init_extra_{clip_index}")
        meta = json.loads((root / "metadata.json").read_text())
        first_gt = np.asarray(Image.open(root / "masks/000000.png"))
        object_id = min(int(x) for x in np.unique(first_gt) if x not in (0, 255))
        ys, xs = np.where(first_gt == object_id)
        closest = np.argmin((xs - xs.mean()) ** 2 + (ys - ys.mean()) ** 2)
        point = [(float(xs[closest]) + .5) / first_gt.shape[1],
                 (float(ys[closest]) + .5) / first_gt.shape[0]]
        frames = root / "frames"
        count = meta["frames"]
        label = meta["video"]
        init = "point derived from human-annotated first frame; not autonomous grounding"
    elif dataset == "libero" and clip_index == 0:
        frames = Path("/data/eval/milk_native_512_demo_21_full")
        meta = json.loads((frames / "provenance.json").read_text())
        count = len(meta["frames"])
        point = [128.0 / 512, 277.0 / 512]
        label = "milk_episode_0014_demo_21"
        init = "agent-selected point from first RGB frame; not human annotation or autonomous grounding"
    else:
        raise ValueError("Use dataset davis or libero")
    if re.fullmatch(r"[A-Za-z0-9_-]+", run_name) is None:
        raise ValueError("run_name must contain only letters, digits, underscores or hyphens")
    out = Path(f"/data/eval/sam31_point_pilot_{run_name}")
    stem = f"{dataset}_{label}"
    if (out / f"{stem}.json").exists():
        raise FileExistsError(f"Refusing to overwrite {out / (stem + '.json')}")
    checks = sorted(set([0, 1, 5, 14, 15, 16, 20, count - 1]))
    checks = [x for x in checks if x < count]
    torch.manual_seed(2468)
    np.random.seed(2468)
    predictor = build_sam3_multiplex_video_predictor(
        checkpoint_path="/data/checkpoints/sam3.1/sam3.1_multiplex.pt",
        use_fa3=False, compile=False, warm_up=False, async_loading_frames=False)
    _patch_multiplex_init_state(predictor)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tracker = predictor.model.tracker.model
    full_masks = None
    runs = []

    def session(length, target_frames):
        memory_events, cache_events, detector_events, availability_events, emitted = [], [], [], [], {}
        current = {"frame": 0, "processed": 0}
        with tempfile.TemporaryDirectory() as directory:
            for i in range(length):
                shutil.copyfile(frames / f"{i:06}.jpg", Path(directory) / f"{i:06}.jpg")
            sid = predictor.handle_request({"type": "start_session", "resource_path": directory,
                                            "offload_video_to_cpu": True})["session_id"]
            original_tpos = tracker._get_tpos_enc
            original_image_feature = tracker._get_image_feature
            original_prepare = predictor.model._prepare_backbone_feats
            original_chunk = predictor.model.detector._build_multigpu_buffer_next_chunk
            try:
                with experimental_point_adapter(predictor, disable_detector_prefetch=True) as adapter:
                    guarded_tpos = tracker._get_tpos_enc
                    guarded_prepare = predictor.model._prepare_backbone_feats
                    guarded_chunk = predictor.model.detector._build_multigpu_buffer_next_chunk

                    def prepare(state, frame_idx, reverse):
                        current["frame"] = int(frame_idx)
                        result = guarded_prepare(state, frame_idx, reverse)
                        feature_cache = state["feature_cache"]
                        numeric = sorted(int(k) for k in feature_cache if isinstance(k, (int, np.integer)))
                        buffer = feature_cache.get("multigpu_buffer", {})
                        buffered = sorted(int(k) for k in buffer if isinstance(k, (int, np.integer)))
                        if any(k > frame_idx for k in numeric + buffered):
                            raise RuntimeError("Future frame features present in an inspected cache")
                        cache_events.append({"frame": int(frame_idx), "feature_cache_frames": numeric,
                                             "detector_buffer_frames": buffered})
                        return result

                    def chunk(*args, **kwargs):
                        start, end = int(kwargs["frame_idx_begin"]), int(kwargs["frame_idx_end"])
                        detector_events.append({"current_frame": current["frame"], "start": start, "end_exclusive": end,
                                                "blocked_future_prefetch": start > current["frame"]})
                        return guarded_chunk(*args, **kwargs)

                    def image_feature(state, frame_idx, batch_size):
                        keys = sorted(int(k) for k in state["cached_features"] if isinstance(k, (int, np.integer)))
                        if int(frame_idx) > current["frame"] or any(k > current["frame"] for k in keys):
                            raise RuntimeError("Future cached image feature requested or present")
                        cache_events.append({"frame": current["frame"], "requested_image_feature": int(frame_idx),
                                             "cached_feature_frames": keys})
                        return original_image_feature(state, frame_idx, batch_size)

                    def tpos(positions, device, max_abs_pos=None, dummy=False):
                        local = inspect.currentframe().f_back.f_locals
                        frame_idx = int(local["frame_idx"])
                        output = local.get("output_dict", {})
                        lookup = {}
                        for group in ("cond_frame_outputs", "non_cond_frame_outputs"):
                            for frame, value in output.get(group, {}).items():
                                lookup.setdefault(id(value), []).append(int(frame))
                        pointer_sources = [lookup.get(id(item[1]), []) for item in local.get("filtered_data", [])]
                        spatial_sources = []
                        for _, prev, _ in local.get("t_pos_and_prevs", []):
                            if prev is None or prev.get("maskmem_features") is None:
                                continue
                            if prev["maskmem_features"].shape[0] == 0:
                                continue
                            spatial_sources.append(lookup.get(id(prev), []))
                        if any(len(indices) != 1 or indices[0] > frame_idx for indices in pointer_sources + spatial_sources):
                            raise RuntimeError("Unresolved or future pointer/spatial memory source")
                        memory_events.append({"frame": frame_idx, "pointers": [x[0] for x in pointer_sources],
                                              "spatial_memory": [x[0] for x in spatial_sources],
                                              "upstream_pointer_budget": max_abs_pos,
                                              "fixed_temporal_max": adapter["pointer_scale_max"]})
                        return guarded_tpos(positions, device, max_abs_pos=max_abs_pos, dummy=dummy)

                    predictor.model._prepare_backbone_feats = prepare
                    predictor.model.detector._build_multigpu_buffer_next_chunk = chunk
                    tracker._get_image_feature = image_feature
                    tracker._get_tpos_enc = tpos
                    try:
                        predictor.handle_request({"type": "add_prompt", "session_id": sid, "frame_index": 0,
                                                  "points": [point], "point_labels": [1], "obj_id": 1})
                        stream = predictor.handle_stream_request({"type": "propagate_in_video", "session_id": sid,
                                  "propagation_direction": "forward", "start_frame_index": 0,
                                  "output_prob_thresh": 0.0})
                        for item in stream:
                            frame = int(item["frame_index"])
                            availability_events.append({"described_frame": frame,
                                "latest_processed_frame_at_emission": current["frame"],
                                "latency_frames": current["frame"] - frame})
                            ids_out = np.asarray(item["outputs"]["out_obj_ids"])
                            matches = np.flatnonzero(ids_out == 1)
                            if len(matches) == 1:
                                emitted[frame] = np.asarray(item["outputs"]["out_binary_masks"])[matches[0]].copy()
                            current["processed"] = max(current["processed"], frame + 1)
                            if frame >= max(target_frames):
                                break
                        stream.close()
                    finally:
                        tracker._get_tpos_enc = guarded_tpos
                        tracker._get_image_feature = original_image_feature
                        predictor.model._prepare_backbone_feats = guarded_prepare
                        predictor.model.detector._build_multigpu_buffer_next_chunk = guarded_chunk
            finally:
                # The scoped adapter restores the original methods even after errors.
                predictor.handle_request({"type": "close_session", "session_id": sid})
            return emitted, {"length": length, "adapter": adapter, "memory": memory_events,
                             "feature_cache": cache_events, "detector_chunks": detector_events,
                             "availability": availability_events}

    full_masks, full_audit = session(count, [count - 1])
    comparisons = []
    for target in checks:
        prefix_masks, audit = session(target + 1, [target])
        a, b = prefix_masks.get(target), full_masks.get(target)
        comparisons.append({"frame": target, "prefix_present": a is not None and bool(a.any()),
            "full_present": b is not None and bool(b.any()),
            "different_pixels": int(np.count_nonzero(a != b)) if a is not None and b is not None else None,
            "audited_memory_frames": sorted(set(e["frame"] for e in audit["memory"])),
            "prefix_length": target + 1})
    if dataset == "davis":
        labels = np.stack([np.asarray(Image.open(root / "masks" / f"{i:06}.png")) for i in range(count)])
        truth = labels == object_id
    else:
        truth = np.load("/data/libero/milk_masks_oracle/episode_0014.npz")["masks"][:, 0]
    # Full ground truth is first opened here, after tracking; DAVIS frame 0
    # annotation was already used solely to initialize its point.
    h, w = truth.shape[1:]
    masks = np.stack([np.asarray(Image.fromarray(full_masks[i].astype(np.uint8)).resize((w, h), Image.Resampling.NEAREST), bool)
                      if i in full_masks else np.zeros((h, w), bool) for i in range(count)])
    if dataset == "davis":
        masks &= labels != 255  # DAVIS void pixels are excluded from scoring.
    pred_features, pred_valid = mask_metrics(masks[:, None])
    true_features, true_valid = mask_metrics(truth[:, None].astype(bool))
    valid = pred_valid & true_valid
    pair_valid = valid[1:] & valid[:-1]
    residual = np.abs(pred_features - true_features)
    pred_delta = np.diff(pred_features, axis=0)
    true_delta = np.diff(true_features, axis=0)
    delta_residual = pred_delta - true_delta
    def metric_average(values, mask):
        return [float(values[..., j][mask[..., j]].mean()) if mask[..., j].any() else None for j in range(5)]
    ious = [float(np.logical_and(a, b).sum() / max(np.logical_or(a, b).sum(), 1)) for a, b in zip(masks, truth)]
    memory_frames = sorted(set(event["frame"] for event in full_audit["memory"]))
    result = {"dataset": dataset, "episode": label, "frames": count, "initialization": init,
              "initial_point_normalized_xy": point, "perception": "initialized camera RGB; simulator masks scoring only" if dataset == "libero" else "human-annotation-initialized camera RGB",
              "adapter": full_audit["adapter"], "prefix_checks": comparisons,
              "mean_iou_all_frames": float(np.mean(ious)), "minimum_iou": float(np.min(ious)),
              "p05_iou": float(np.quantile(ious, .05)), "per_frame_iou": ious,
              "nonempty_predictions": int(sum(m.any() for m in masks)),
              "metric_names": ["centroid_u", "centroid_v", "bbox_width", "bbox_height", "visible_area"],
              "measurement_mae_image_fraction": metric_average(residual, valid),
              "frame_delta_error_mae_image_fraction": metric_average(np.abs(delta_residual), pair_valid),
              "predicted_frame_delta_abs_mean_image_fraction": metric_average(np.abs(pred_delta), pair_valid),
              "ground_truth_frame_delta_abs_mean_image_fraction": metric_average(np.abs(true_delta), pair_valid),
              "frame_delta_error_std_image_fraction": [float(delta_residual[..., j][pair_valid[..., j]].std())
                  if pair_valid[..., j].any() else None for j in range(5)],
              "valid_object_frames": int(valid[:, 0, 0].sum()),
              "valid_consecutive_pairs": int(pair_valid[:, 0, 0].sum()),
              "memory_audited_frames": memory_frames,
              "memory_frames_without_tpos_hook": [i for i in range(1, count) if i not in memory_frames],
              "audit_scope": "point path: selected pointer/spatial identities and requested cache/chunk indices; no claim about unaudited internals or appendable live input",
              "full_audit": full_audit}
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{stem}.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(out / f"{stem}_masks.npz", masks=masks, predicted_features=pred_features,
                        reference_features=true_features, valid=valid)
    if dataset == "libero":
        samples = sorted(set([0, count // 3, 2 * count // 3, count - 1]))
        montage = Image.new("RGB", (512 * len(samples), 540), "white")
        draw = ImageDraw.Draw(montage)
        for column, frame in enumerate(samples):
            rgb = np.asarray(Image.open(frames / f"{frame:06}.jpg").convert("RGB")).copy()
            pred = np.asarray(Image.fromarray(masks[frame].astype(np.uint8)).resize((512, 512), Image.Resampling.NEAREST), bool)
            gt = np.asarray(Image.fromarray(truth[frame].astype(np.uint8)).resize((512, 512), Image.Resampling.NEAREST), bool)
            rgb[pred] = (.5 * rgb[pred] + .5 * np.array([255, 0, 0])).astype(np.uint8)
            rgb[gt] = (.5 * rgb[gt] + .5 * np.array([0, 255, 255])).astype(np.uint8)
            montage.paste(Image.fromarray(rgb), (512 * column, 28))
            draw.text((512 * column + 5, 5), f"frame {frame}: red pred / cyan oracle", fill="black")
        montage.save(out / f"{stem}_review.png")
    volume.commit()
    return {k: v for k, v in result.items() if k != "full_audit"}


@app.local_entrypoint()
def main(dataset: str = "davis", clip_index: int = 0, run_name: str = "v3"):
    print(pilot.remote(dataset, clip_index, run_name))
