"""Experimental streaming SAM probe versus exact prefix replay; not a rollout adapter."""
import modal
from scripts.modal_muse_sam_paired_rollout import image

volume = modal.Volume.from_name("object-dynamics-vla-data", create_if_missing=False)
image = (image.add_local_file("scripts/__init__.py", "/root/scripts/__init__.py")
         .add_local_file("scripts/modal_muse_sam_paired_rollout.py", "/root/scripts/modal_muse_sam_paired_rollout.py")
         .add_local_file("scripts/modal_muse_motor_rollout.py", "/root/scripts/modal_muse_motor_rollout.py"))
app = modal.App("odvla-sam-incremental-probe")


@app.function(image=image, gpu="A100-80GB", cpu=4, memory=32768, volumes={"/data": volume}, timeout=900)
def run(variant: str = "M2"):
    import json
    import sys
    import tempfile
    import time
    from pathlib import Path
    import numpy as np
    import torch
    from PIL import Image
    from sam3.model.io_utils import load_image_as_single_frame_video
    from sam3.model_builder import build_sam3_multiplex_video_predictor
    from odvla.data import file_hash, read_json
    from odvla.experimental_sam31 import experimental_point_adapter
    from odvla.sam_prefix import replay_current
    sys.path.insert(0, "/root")
    from extract_sam31 import _patch_multiplex_init_state
    cfg = read_json("/root/sam31_milk_development_config.json")
    checkpoint = Path("/data/checkpoints/sam3.1/sam3.1_multiplex.pt")
    if file_hash(checkpoint) != cfg["checkpoint_sha256"]: raise ValueError("SAM checkpoint mismatch")
    if variant not in ("M1", "M2"): raise ValueError("Expected M1 or M2")
    dest = Path(f"/data/eval/sam_incremental_probe_{variant}_70_capacity200_v1.json")
    if dest.exists(): raise FileExistsError(dest)
    frames = Path(f"/data/eval/multitask_muse_sam_paired_pilot70/{variant}/frames")
    count = 70; torch.manual_seed(2468); torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    predictor = build_sam3_multiplex_video_predictor(checkpoint_path=str(checkpoint), use_fa3=False,
        compile=False, warm_up=False, async_loading_frames=False)
    _patch_multiplex_init_state(predictor)
    rows = []; error = None; sid = None
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        # Placeholder images are copies of the first observed frame, never future RGB.
        for t in range(200): (folder / f"{t:06}.jpg").write_bytes((frames / "000000.jpg").read_bytes())
        try:
            sid = predictor.handle_request({"type": "start_session", "resource_path": str(folder), "offload_video_to_cpu": True})["session_id"]
            state = predictor._all_inference_states[sid]["state"]
            state["num_frames"] = 1
            with experimental_point_adapter(predictor, disable_detector_prefetch=True):
                for oid, point in zip(cfg["object_ids"], cfg["first_frame_points_xy_pixels_512"]):
                    predictor.handle_request({"type": "add_prompt", "session_id": sid, "frame_index": 0,
                        "points": [[point[0] / 512, point[1] / 512]], "point_labels": [1], "obj_id": oid})
                for t in range(count):
                    if t:
                        tensor, _, _ = load_image_as_single_frame_video(str(frames / f"{t:06}.jpg"),
                            image_size=predictor.model.image_size, offload_video_to_cpu=True,
                            img_mean=predictor.model.image_mean, img_std=predictor.model.image_std)
                        images = state["input_batch"].img_batch
                        with torch.inference_mode(): images.tensors[t].copy_(tensor[0].to(images.device))
                        state["num_frames"] = t + 1
                        for tracker_state in state["sam2_inference_states"]:
                            tracker_state["num_frames"] = t + 1
                        # Experimental internal dispatch marker: request tracker-only
                        # propagation for existing objects; no new prompt is supplied.
                        predictor.model.add_action_history(state, "refine", frame_idx=0, obj_ids=cfg["object_ids"])
                    clock = time.monotonic()
                    stream = predictor.handle_stream_request({"type": "propagate_in_video", "session_id": sid,
                        "propagation_direction": "forward", "start_frame_index": t, "max_frame_num_to_track": 0,
                        "output_prob_thresh": 0.0, "force_tracker_propagation": True})
                    outputs = list(stream)
                    incremental_time = time.monotonic() - clock
                    latest = next(v["outputs"] for v in outputs if v["frame_index"] == t)
                    ids = np.asarray(latest["out_obj_ids"]); masks = np.asarray(latest["out_binary_masks"])
                    actual = np.zeros((2, 128, 128), bool)
                    for i, oid in enumerate(cfg["object_ids"]):
                        matches = np.flatnonzero(ids == oid)
                        if len(matches) != 1: raise ValueError(f"Expected one object {oid}, found {len(matches)}")
                        actual[i] = np.asarray(Image.fromarray(masks[matches[0]].astype(np.uint8)).resize((128, 128), Image.Resampling.NEAREST), bool)
                    # Reference gets exactly observed prefix; no placeholder/future images.
                    reference_dir = folder / "reference"; reference_dir.mkdir(exist_ok=True)
                    (reference_dir / f"{t:06}.jpg").write_bytes((frames / f"{t:06}.jpg").read_bytes())
                    # Separate reference session cannot run while monkeypatch context is active.
                    rows.append({"frame": t, "incremental_seconds": incremental_time, "masks": actual})
            # The earlier 70-step rollout saved each current-frame prefix-replay
            # mask; load them only after all incremental outputs are complete.
            with np.load(f"/data/eval/multitask_muse_sam_paired_pilot70/{variant}/masks.npz", allow_pickle=False) as z:
                expected = z["masks"].copy()
            if len(expected) != count: raise ValueError("Expected 70 stored prefix masks")
            for row in rows:
                row["different_pixels"] = (row.pop("masks") != expected[row["frame"]]).sum((1, 2)).tolist()
        except Exception as exc:
            import traceback
            error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            for row in rows: row.pop("masks", None)
        finally:
            if sid is not None: predictor.handle_request({"type": "close_session", "session_id": sid})
    result = {"status": "experimental; no live control", "variant": variant, "error": error, "frames_compared": len(rows),
              "all_exact": error is None and len(rows) == count and all(r["different_pixels"] == [0, 0] for r in rows), "rows": rows}
    dest.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n"); volume.commit(); return result


@app.local_entrypoint()
def main(variant: str = "M2"): print(run.remote(variant))
