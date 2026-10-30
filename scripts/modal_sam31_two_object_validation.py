"""RGB-point-initialized, two-object LIBERO validation; oracle masks score only."""
import modal
from scripts.modal_davis_human_init import inference_image, volume

image = (inference_image.add_local_file("scripts/__init__.py", "/root/scripts/__init__.py")
         .add_local_file("scripts/modal_davis_human_init.py", "/root/scripts/modal_davis_human_init.py"))
app = modal.App("odvla-sam31-two-object-validation")
EPISODES = {"episode_0014": "demo_21", "episode_0020": "demo_27", "episode_0022": "demo_29"}
# Selected by inspecting each native first RGB frame, before opening oracle masks.
POINTS = {"milk": (128 / 512, 277 / 512), "basket": (410 / 512, 210 / 512)}


@app.function(image=image, gpu="A100-80GB", volumes={"/data": volume},
              timeout=7200, single_use_containers=True)
def validate(episode: str):
    import json
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

    if episode not in EPISODES:
        raise ValueError("Only the three preselected validation episodes are supported")
    demo = EPISODES[episode]
    frames = Path(f"/data/eval/milk_native_512_{demo}_full")
    count = len(json.loads((frames / "provenance.json").read_text())["frames"])
    out = Path("/data/eval/sam31_two_object_validation_v1")
    if (out / f"{episode}.json").exists():
        raise FileExistsError(f"Result already exists for {episode}")
    torch.manual_seed(2468)
    np.random.seed(2468)
    predictor = build_sam3_multiplex_video_predictor(
        checkpoint_path="/data/checkpoints/sam3.1/sam3.1_multiplex.pt",
        use_fa3=False, compile=False, warm_up=False, async_loading_frames=False)
    _patch_multiplex_init_state(predictor)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    def session(length):
        output = {}
        ids_by_frame = {}
        with tempfile.TemporaryDirectory() as directory:
            for i in range(length):
                shutil.copyfile(frames / f"{i:06}.jpg", Path(directory) / f"{i:06}.jpg")
            sid = predictor.handle_request({"type": "start_session", "resource_path": directory,
                                            "offload_video_to_cpu": True})["session_id"]
            try:
                with experimental_point_adapter(predictor, disable_detector_prefetch=True) as adapter:
                    for obj_id, point in enumerate(POINTS.values(), 1):
                        predictor.handle_request({"type": "add_prompt", "session_id": sid,
                            "frame_index": 0, "points": [point], "point_labels": [1], "obj_id": obj_id})
                    stream = predictor.handle_stream_request({"type": "propagate_in_video", "session_id": sid,
                        "propagation_direction": "forward", "start_frame_index": 0, "output_prob_thresh": 0.0})
                    for item in stream:
                        frame = int(item["frame_index"])
                        ids = np.asarray(item["outputs"]["out_obj_ids"])
                        masks = np.asarray(item["outputs"]["out_binary_masks"])
                        ids_by_frame[frame] = ids.tolist()
                        output[frame] = {}
                        for obj_id in (1, 2):
                            matches = np.flatnonzero(ids == obj_id)
                            if len(matches) == 1:
                                output[frame][obj_id] = masks[int(matches[0])].copy()
                    stream.close()
            finally:
                predictor.handle_request({"type": "close_session", "session_id": sid})
        return output, ids_by_frame, adapter

    full, ids, adapter = session(count)
    prefix, _, _ = session(17)
    prefix_checks = []
    for frame in (0, 1, 5, 14, 15, 16):
        for obj_id, name in enumerate(POINTS, 1):
            a, b = prefix.get(frame, {}).get(obj_id), full.get(frame, {}).get(obj_id)
            prefix_checks.append({"frame": frame, "object": name,
                "prefix_nonempty": bool(a is not None and a.any()),
                "full_nonempty": bool(b is not None and b.any()),
                "different_pixels": int(np.count_nonzero(a != b)) if a is not None and b is not None else None})

    # Open privileged simulator masks only after all tracking and prefix checks.
    truth = np.load(f"/data/libero/milk_masks_oracle/{episode}.npz")["masks"].astype(bool)
    if truth.shape[0] != count or truth.shape[1] != 2:
        raise ValueError("Oracle reference dimensions do not match rendered episode")
    h, w = truth.shape[-2:]
    masks = np.zeros_like(truth)
    for frame, objects in full.items():
        for obj_id, mask in objects.items():
            masks[frame, obj_id - 1] = np.asarray(Image.fromarray(mask.astype(np.uint8)).resize(
                (w, h), Image.Resampling.NEAREST), bool)
    features, valid = mask_metrics(masks)
    reference, ref_valid = mask_metrics(truth)
    pair = valid & ref_valid
    consecutive = pair[1:] & pair[:-1]
    residual = np.abs(features - reference)
    delta = np.diff(features, axis=0) - np.diff(reference, axis=0)
    results = {}
    cross = np.zeros((count, 2, 2))
    for t in range(count):
        for i in range(2):
            for j in range(2):
                union = np.logical_or(masks[t, i], truth[t, j]).sum()
                cross[t, i, j] = np.logical_and(masks[t, i], truth[t, j]).sum() / max(union, 1)
    for i, name in enumerate(POINTS):
        def mean(values, mask):
            return [float(values[:, i, j][mask[:, i, j]].mean()) if mask[:, i, j].any() else None
                    for j in range(5)]
        results[name] = {"id": i + 1, "nonempty_frames": int(masks[:, i].any(axis=(1, 2)).sum()),
            "valid_consecutive_pairs": int(consecutive[:, i, 0].sum()),
            "mean_iou": float(cross[:, i, i].mean()), "p05_iou": float(np.quantile(cross[:, i, i], .05)),
            "min_iou": float(cross[:, i, i].min()), "mean_iou_other_object": float(cross[:, i, 1-i].mean()),
            "other_object_higher_iou_frames": int((cross[:, i, 1-i] > cross[:, i, i]).sum()),
            "measurement_mae": mean(residual, pair), "delta_error_mae": mean(np.abs(delta), consecutive)}
    result = {"episode": episode, "demo": demo, "frames": count, "points_normalized_xy": POINTS,
        "initialization": "agent-selected first RGB points; not autonomous grounding",
        "oracle_use": "simulator masks opened after tracking and prefix checks, for scoring only",
        "adapter": adapter, "preloaded_full_session": True, "objects": results,
        "output_ids_by_frame": ids, "prefix_length": 17, "prefix_checks": prefix_checks,
        "metric_names": ["centroid_u", "centroid_v", "bbox_width", "bbox_height", "visible_area"]}
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{episode}_masks.npz", masks=masks, cross_iou=cross)
    samples = sorted(set([0, count // 3, 2 * count // 3, count - 1]))
    montage = Image.new("RGB", (512 * len(samples), 2 * 540), "white")
    draw = ImageDraw.Draw(montage)
    for row, name in enumerate(POINTS):
        for col, frame in enumerate(samples):
            rgb = np.asarray(Image.open(frames / f"{frame:06}.jpg").convert("RGB")).copy()
            pred = np.asarray(Image.fromarray(masks[frame, row].astype(np.uint8)).resize((512, 512), Image.Resampling.NEAREST), bool)
            gt = np.asarray(Image.fromarray(truth[frame, row].astype(np.uint8)).resize((512, 512), Image.Resampling.NEAREST), bool)
            rgb[pred] = (.5 * rgb[pred] + .5 * np.array([255, 0, 0])).astype(np.uint8)
            rgb[gt] = (.5 * rgb[gt] + .5 * np.array([0, 255, 255])).astype(np.uint8)
            montage.paste(Image.fromarray(rgb), (512 * col, 540 * row + 28))
            draw.text((512 * col + 5, 540 * row + 5), f"{name} frame {frame}: red prediction / cyan oracle", fill="black")
    montage.save(out / f"{episode}_review.png")
    (out / f"{episode}.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    volume.commit()
    return result


@app.local_entrypoint()
def main(episode: str = "episode_0014"):
    print(validate.remote(episode))
