"""Extract the frozen two-point SAM cache; never reads simulator reference masks."""
import modal
from scripts.modal_davis_human_init import inference_image, volume

image = (inference_image.add_local_file("scripts/__init__.py", "/root/scripts/__init__.py")
         .add_local_file("scripts/modal_davis_human_init.py", "/root/scripts/modal_davis_human_init.py")
         .add_local_file("sam31_milk_development_config.json", "/root/sam31_milk_development_config.json"))
app = modal.App("odvla-milk-sam-cache")


@app.function(image=image, gpu="A100-80GB", volumes={"/data": volume},
              timeout=7200, single_use_containers=True)
def extract():
    import hashlib
    import json
    import sys
    import time
    from pathlib import Path
    import numpy as np
    import torch
    from PIL import Image
    from odvla.experimental_sam31 import experimental_point_adapter
    from odvla.measurements import mask_metrics
    from sam3.model_builder import build_sam3_multiplex_video_predictor
    sys.path.insert(0, "/root")
    from extract_sam31 import _patch_multiplex_init_state

    config_path = Path("/root/sam31_milk_development_config.json")
    config = json.loads(config_path.read_text())
    checkpoint = Path("/data/checkpoints/sam3.1/sam3.1_multiplex.pt")
    digest = hashlib.sha256()
    with checkpoint.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != config["checkpoint_sha256"]:
        raise ValueError("SAM checkpoint hash mismatch")
    render_root = Path("/data/libero/milk_sam_native_v1")
    render = json.loads((render_root / "provenance.json").read_text())
    splits = json.loads(Path("/data/libero/milk_oracle/splits.json").read_text())
    expected = set(splits["train"] + splits["val"])
    if {e["episode"] for e in render["entries"]} != expected:
        raise ValueError("Rendered episode list does not match train/validation split")
    output = Path("/data/libero/milk_sam_masks_v1")
    output.mkdir(parents=True, exist_ok=True)
    saved_config = output / "configuration.json"
    progress_path = output / "progress.json"
    if (output / "provenance.json").exists():
        raise FileExistsError("Cache already complete; refusing to overwrite")
    if saved_config.exists():
        if saved_config.read_bytes() != config_path.read_bytes() or not progress_path.exists():
            raise ValueError("Partial cache configuration/progress mismatch")
        results = json.loads(progress_path.read_text())
        if [e["episode"] for e in results] != [e["episode"] for e in render["entries"][:len(results)]]:
            raise ValueError("Partial cache is not a contiguous prefix of the frozen episode list")
        for item in results:
            with np.load(output / (item["episode"] + ".npz"), allow_pickle=False) as saved:
                if saved["masks"].shape != (item["frames"], 2, 128, 128):
                    raise ValueError("Partial cache has incompatible output")
    else:
        if any(output.iterdir()):
            raise ValueError("Unrecognized existing cache directory")
        saved_config.write_bytes(config_path.read_bytes())
        results = []
    resumed_after = len(results)
    torch.manual_seed(config["settings"]["seed"])
    np.random.seed(config["settings"]["seed"])
    predictor = build_sam3_multiplex_video_predictor(checkpoint_path=str(checkpoint),
        use_fa3=False, compile=False, warm_up=False, async_loading_frames=False)
    _patch_multiplex_init_state(predictor)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for entry in render["entries"][resumed_after:]:
        name, count = entry["episode"], entry["frames"]
        with np.load(f"/data/libero/milk_raw/episodes/{name}.npz", allow_pickle=False) as raw:
            timestamps = raw["timestamps"].copy()
        if len(timestamps) != count or not np.allclose(np.diff(timestamps), 1 / config["fps"]):
            raise ValueError(f"{name}: timestamp/frame mismatch")
        masks = np.zeros((count, 2, 128, 128), bool)
        id_status = np.zeros((count, 2), np.uint8)  # 0 missing, 1 unique, 2 duplicate
        sid = predictor.handle_request({"type": "start_session", "resource_path": str(render_root / name),
                                        "offload_video_to_cpu": True})["session_id"]
        started = time.monotonic()
        try:
            with experimental_point_adapter(predictor, disable_detector_prefetch=True) as adapter:
                for obj_id, point in zip(config["object_ids"], config["first_frame_points_xy_pixels_512"]):
                    predictor.handle_request({"type": "add_prompt", "session_id": sid, "frame_index": 0,
                        "points": [[point[0] / 512, point[1] / 512]], "point_labels": [1], "obj_id": obj_id})
                stream = predictor.handle_stream_request({"type": "propagate_in_video", "session_id": sid,
                    "propagation_direction": "forward", "start_frame_index": 0, "output_prob_thresh": 0.0})
                seen = set()
                for item in stream:
                    t = int(item["frame_index"])
                    if t in seen or not 0 <= t < count:
                        raise ValueError(f"Unexpected output frame {t}")
                    seen.add(t)
                    ids = np.asarray(item["outputs"]["out_obj_ids"])
                    emitted = np.asarray(item["outputs"]["out_binary_masks"])
                    for slot, obj_id in enumerate(config["object_ids"]):
                        matches = np.flatnonzero(ids == obj_id)
                        id_status[t, slot] = min(len(matches), 2)
                        if len(matches) == 1:
                            mask = emitted[int(matches[0])].astype(np.uint8)
                            masks[t, slot] = np.asarray(Image.fromarray(mask).resize((128, 128), Image.Resampling.NEAREST), bool)
                stream.close()
                if len(seen) != count:
                    raise ValueError(f"{name}: only {len(seen)}/{count} frames emitted")
        finally:
            predictor.handle_request({"type": "close_session", "session_id": sid})
        features, valid = mask_metrics(masks)
        np.savez_compressed(output / f"{name}.npz", masks=masks, timestamps=timestamps,
                            features=features, valid=valid, id_status=id_status)
        results.append({"episode": name, "demo": entry["demo"], "split": entry["split"],
            "frames": count, "nonempty_per_slot": masks.any(axis=(2, 3)).sum(axis=0).tolist(),
            "unique_id_frames": (id_status == 1).sum(axis=0).tolist(),
            "seconds": round(time.monotonic() - started, 2), "rgb_sequence_sha256": entry["jpeg_sequence_sha256"]})
        progress_path.write_text(json.dumps(results, indent=2) + "\n")
        volume.commit()
    summary = {"configuration_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
               "adapter": adapter, "entries": results, "complete": True,
               "reference_masks_opened": False, "test_episodes_used": False,
               "last_invocation_resumed_after_episodes": resumed_after,
               "rng_seed_reset_at_invocation": config["settings"]["seed"]}
    (output / "provenance.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    volume.commit()
    return {"episodes": len(results), "frames": sum(x["frames"] for x in results), "output": str(output)}


@app.local_entrypoint()
def main():
    print(extract.remote())
