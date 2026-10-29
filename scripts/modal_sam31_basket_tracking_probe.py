"""Test agent-box-initialized SAM tracking and prefix consistency on milk/basket.

The box is selected by visual inspection of the initial RGB; simulator masks are
used only to score outputs. This is not autonomous text grounding or milk identity.
"""
import modal
from scripts.modal_sam31 import image as sam_image, volume

image = (sam_image.add_local_file("scripts/__init__.py", "/root/scripts/__init__.py")
         .add_local_file("scripts/modal_sam31.py", "/root/scripts/modal_sam31.py"))
app = modal.App("odvla-sam31-basket-tracking-probe")


@app.function(image=image, gpu="A100-80GB", volumes={"/data": volume}, timeout=3600)
def probe(native: bool = False, slot: str = "basket"):
    import json
    import shutil
    import sys
    import tempfile
    from pathlib import Path
    import numpy as np
    import torch
    from PIL import Image
    from odvla.data import load_episode, load_manifest
    from sam3.model_builder import build_sam3_multiplex_video_predictor
    sys.path.insert(0, "/root")
    from extract_sam31 import _patch_multiplex_init_state

    dataset = "/data/libero/milk_raw"
    manifest = load_manifest(dataset)
    episode = load_episode(dataset, manifest["episodes"][0], manifest)
    if slot not in ("milk", "basket") or (slot == "milk" and not native):
        raise ValueError("Use milk or basket; the milk box is defined on native frames")
    oracle = np.load("/data/libero/milk_masks_oracle/episode_0000.npz")["masks"][:, 0 if slot == "milk" else 1]
    box = ([110 / 512, 239 / 512, 40 / 512, 76 / 512] if slot == "milk" else
           [350 / 512, 183 / 512, 120 / 512, 117 / 512] if native else
           [86 / 128, 45 / 128, 33 / 128, 31 / 128])
    model = build_sam3_multiplex_video_predictor(
        checkpoint_path="/data/checkpoints/sam3.1/sam3.1_multiplex.pt",
        use_fa3=False, compile=False, warm_up=False, async_loading_frames=False)
    _patch_multiplex_init_state(model)
    sequences = {}
    with tempfile.TemporaryDirectory() as temp:
        for length in (51, 101, len(episode["rgb"])):
            folder = Path(temp) / str(length)
            folder.mkdir()
            for i, frame in enumerate(episode["rgb"][:length]):
                if native:
                    shutil.copyfile(f"/data/eval/milk_native_512_full/{i:06d}.jpg", folder / f"{i:06d}.jpg")
                else:
                    Image.fromarray(frame).save(folder / f"{i:06d}.jpg", quality=98)
            sid = model.handle_request({"type": "start_session", "resource_path": str(folder),
                                        "offload_video_to_cpu": True})["session_id"]
            try:
                initial = model.handle_request({"type": "add_prompt", "session_id": sid, "frame_index": 0,
                    "bounding_boxes": [box],
                    "bounding_box_labels": [1], "output_prob_thresh": 0.0})["outputs"]
                initial_ids = np.asarray(initial["out_obj_ids"])
                if len(initial_ids) != 1:
                    raise ValueError(f"Expected one initialized basket, got IDs {initial_ids.tolist()}")
                target_id = int(initial_ids[0])
                masks = {}
                instance_counts = {}
                for item in model.handle_stream_request({"type": "propagate_in_video", "session_id": sid,
                        "propagation_direction": "forward", "output_prob_thresh": 0.0,
                        "force_tracker_propagation": True}):
                    predictions = np.asarray(item["outputs"]["out_binary_masks"])
                    ids = np.asarray(item["outputs"]["out_obj_ids"])
                    matches = np.flatnonzero(ids == target_id)
                    frame_index = int(item["frame_index"])
                    instance_counts[frame_index] = int(len(ids))
                    if len(matches) == 1:
                        mask = predictions[matches[0]]
                        masks[frame_index] = np.asarray(Image.fromarray(mask.astype(np.uint8)).resize(
                            (128, 128), Image.Resampling.NEAREST), dtype=bool)
                    else:
                        masks[frame_index] = np.zeros((128, 128), bool)
                sequences[length] = {"masks": masks, "instance_counts": instance_counts, "target_id": target_id}
            finally:
                model.handle_request({"type": "close_session", "session_id": sid})
    full_info = sequences[len(episode["rgb"])]
    full = full_info["masks"]
    ious = []
    for frame, mask in sorted(full.items()):
        ious.append(float(np.logical_and(mask, oracle[frame]).sum() /
                          max(np.logical_or(mask, oracle[frame]).sum(), 1)))
    checks = {}
    for length in (51, 101):
        frame = length - 1
        prefix = sequences[length]["masks"]
        if frame not in prefix or frame not in full:
            checks[str(frame)] = {"available": False}
        else:
            checks[str(frame)] = {"available": True,
                "different_pixels": int(np.count_nonzero(prefix[frame] != full[frame])),
                "prefix_target_present": bool(prefix[frame].any()),
                "full_target_present": bool(full[frame].any())}
    result = {"dataset": dataset, "episode": "episode_0000", "slot": slot,
              "image_source": "native 512x512 render" if native else "recorded 128x128 RGB",
              "initialization": "agent-selected RGB bounding box; not human annotation or text grounding",
              "oracle_use": "post-hoc scoring only", "tracked_frames": len(full),
              "force_tracker_propagation": True,
              "expected_frames": len(episode["rgb"]), "mean_iou": float(np.mean(ious)),
              "minimum_iou": float(np.min(ious)),
              "target_id": full_info["target_id"],
              "target_present_frames": sum(bool(m.any()) for m in full.values()),
              "max_instances": max(full_info["instance_counts"].values()),
              "iou_by_frame": ious, "prefix_checks": checks}
    path = Path(f"/data/eval/sam31_milk_{slot}_{'native' if native else '128'}_tracking_probe.json")
    path.write_text(json.dumps(result, indent=2) + "\n")
    volume.commit()
    return result


@app.local_entrypoint()
def main(native: bool = False, slot: str = "basket"):
    print(probe.remote(native, slot))
