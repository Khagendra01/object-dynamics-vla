"""Frozen SAM previews and causal training measurements for ten LIBERO tasks."""
import modal
from scripts.modal_muse_smolvla_live import vision_image, volume

app = modal.App("odvla-libero-object-ten-task-sam")
image = vision_image.add_local_file("configs/libero_object_10_integration_points.json", "/root/ten_task_points.json")


@app.function(image=image, gpu="A100-80GB", cpu=4, memory=131072, volumes={"/data": volume},
              timeout=7200, max_containers=1, single_use_containers=True)
def extract(preview_only: bool = True, max_seconds: int = 3600):
    import hashlib
    import json
    import os
    import tempfile
    import sys
    import time
    from pathlib import Path
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    from odvla.data import file_hash, read_json, write_json
    from odvla.measurements import mask_metrics
    from odvla.sam_incremental import IncrementalPointTracker
    from odvla.sam_prefix import replay_current
    from sam3.model_builder import build_sam3_multiplex_video_predictor
    sys.path.insert(0, "/root")
    from extract_sam31 import _patch_multiplex_init_state

    if not 60 <= max_seconds <= 6000:
        raise ValueError("SAM runtime budget must be 60..6000 seconds")
    base = Path("/data/libero/ten_task_integration_v1")
    points = read_json("/root/ten_task_points.json")
    if len(points["tasks"]) != 10:
        raise ValueError("Expected ten explicit task-specific RGB point annotations")
    checkpoint = Path("/data/checkpoints/sam3.1/sam3.1_multiplex.pt")
    if file_hash(checkpoint) != "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6":
        raise ValueError("SAM checkpoint checksum mismatch")
    if not preview_only:
        renders = read_json(base / "renders.json")
        if renders.get("status") != "complete" or len(renders["entries"]) != 10:
            raise ValueError("Full ten-task training-episode renders are incomplete")
    torch.manual_seed(2468)
    np.random.seed(2468)
    predictor = build_sam3_multiplex_video_predictor(checkpoint_path=str(checkpoint), use_fa3=False,
        compile=False, warm_up=False, async_loading_frames=False)
    _patch_multiplex_init_state(predictor)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.monotonic()
    sheet = Image.new("RGB", (1024, 5 * 548), "white")
    draw = ImageDraw.Draw(sheet)
    results = []
    for index, (task_id, annotation) in enumerate(points["tasks"].items()):
        folder = base / task_id / annotation["episode"]
        first = folder / "rgb/000000.jpg"
        if hashlib.sha256(first.read_bytes()).hexdigest() != annotation["rgb_sha256"]:
            raise ValueError(f"Annotated first frame changed: {task_id}")
        destination = folder / "sam_measurements.npz"
        if not preview_only and destination.exists():
            with np.load(destination, allow_pickle=False) as saved:
                if str(saved["first_rgb_sha256"].item()) != annotation["rgb_sha256"]:
                    raise ValueError(f"Existing SAM cache identity changed: {task_id}")
                results.append({"task_id": task_id, "frames": len(saved["features"]), "reused": True})
            continue
        paths = [first] if preview_only else sorted((folder / "rgb").glob("*.jpg"))
        if not 1 <= len(paths) <= 200:
            raise ValueError(f"Expected 1..200 observed frames: {task_id}")
        tracker = IncrementalPointTracker(predictor, first, annotation["points"], points["object_ids"], capacity=len(paths))
        masks = np.zeros((len(paths), 2, 128, 128), dtype=bool)
        status = np.zeros((len(paths), 2), dtype=np.uint8)
        parity_frames = []
        try:
            for frame, path in enumerate(paths):
                masks[frame], status[frame] = tracker.step(path)
                if not preview_only and frame in (0, 9):
                    with tempfile.TemporaryDirectory(prefix="odvla-ten-prefix-") as temp:
                        for prefix_path in paths[:frame + 1]:
                            os.symlink(prefix_path, Path(temp) / prefix_path.name)
                        ref_masks, ref_status = replay_current(predictor, temp, frame + 1,
                                                               annotation["points"], points["object_ids"])
                    if not np.array_equal(masks[frame], ref_masks) or not np.array_equal(status[frame], ref_status):
                        raise ValueError(f"Incremental SAM prefix mismatch: {task_id} frame {frame}")
                    parity_frames.append(frame)
                if frame == 0:
                    original = np.asarray(Image.open(first).convert("RGB")).copy()
                    for slot, color in enumerate(((255, 70, 70), (70, 230, 120))):
                        overlay = np.asarray(Image.fromarray(masks[frame, slot].astype(np.uint8)).resize((512, 512), Image.Resampling.NEAREST), dtype=bool)
                        original[overlay] = (original[overlay] * .55 + np.asarray(color) * .45).astype(np.uint8)
                    x, y = (index % 2) * 512, (index // 2) * 548
                    sheet.paste(Image.fromarray(original), (x, y + 36))
                    draw.text((x + 4, y + 8), f"{task_id} / red target / green basket", fill="black")
                if not preview_only and time.monotonic() - started > max_seconds:
                    raise TimeoutError("SAM budget reached; complete episodes remain reusable")
        finally:
            tracker.close()
        features, valid = mask_metrics(masks)
        valid &= status[:, :, None] == 1
        record = {"task_id": task_id, "episode": annotation["episode"], "frames": len(paths),
                  "first_frame_unique": status[0].tolist(), "visible_per_slot": masks.any(axis=(2, 3)).sum(axis=0).tolist(),
                  "prefix_parity_frames": parity_frames}
        if not preview_only:
            np.savez_compressed(destination, features=features, valid=valid, id_status=status,
                                first_rgb_sha256=np.asarray(annotation["rgb_sha256"]))
            volume.commit()
        results.append(record)
        print(json.dumps(record), flush=True)
    if preview_only:
        sheet.save(base / "sam_initialization_preview.png")
        write_json(base / "sam_preview.json", {"status": "complete", "entries": results,
            "points_source": points["initialization_method"], "manually_reviewed": False})
    else:
        write_json(base / "sam_measurements_manifest.json", {"status": "complete", "entries": results,
            "points_source": points["initialization_method"], "tracking": "observed-prefix incremental SAM"})
    volume.commit()
    return {"status": "complete", "preview_only": preview_only, "tasks": len(results)}


@app.local_entrypoint()
def main(preview_only: bool = True, max_seconds: int = 3600, background: bool = False):
    if background:
        print(f"Spawned SAM extraction: {extract.spawn(preview_only, max_seconds).object_id}")
    else:
        print(extract.remote(preview_only, max_seconds))
