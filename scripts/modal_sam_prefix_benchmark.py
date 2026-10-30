"""Compare prefix-only replay with frozen SAM cache at preselected frames."""
import modal
from scripts.modal_milk_sam_cache import image, volume

image = (image.add_local_file("scripts/__init__.py", "/root/scripts/__init__.py")
         .add_local_file("scripts/modal_milk_sam_cache.py", "/root/scripts/modal_milk_sam_cache.py")
         .add_local_file("scripts/modal_davis_human_init.py", "/root/scripts/modal_davis_human_init.py"))
app = modal.App("odvla-sam-prefix-benchmark")


@app.function(image=image, gpu="A100-80GB", volumes={"/data": volume}, timeout=3600)
def benchmark():
    import hashlib
    import json
    import shutil
    import sys
    import tempfile
    import time
    from pathlib import Path
    import numpy as np
    import torch
    from odvla.sam_prefix import replay_current
    from sam3.model_builder import build_sam3_multiplex_video_predictor
    sys.path.insert(0, "/root")
    from extract_sam31 import _patch_multiplex_init_state
    cfg = json.loads(Path("/root/sam31_milk_development_config.json").read_text())
    checkpoint = Path("/data/checkpoints/sam3.1/sam3.1_multiplex.pt")
    with checkpoint.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    if digest != cfg["checkpoint_sha256"]:
        raise ValueError("Checkpoint mismatch")
    dest = Path("/data/eval/milk_sam_prefix_benchmark_v1.json")
    if dest.exists():
        raise FileExistsError("Refusing to overwrite benchmark")
    torch.manual_seed(2468); np.random.seed(2468)
    predictor = build_sam3_multiplex_video_predictor(checkpoint_path=str(checkpoint), use_fa3=False,
        compile=False, warm_up=False, async_loading_frames=False)
    _patch_multiplex_init_state(predictor)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    episode = "episode_0014"
    frames = Path("/data/libero/milk_sam_native_v1") / episode
    with np.load(f"/data/libero/milk_sam_masks_v1/{episode}.npz", allow_pickle=False) as z:
        cached, cached_status = z["masks"], z["id_status"]
    checks = [0, 1, 5, 10, 15, 16, 20, 30, 50]
    rows = []
    with tempfile.TemporaryDirectory() as temp:
        folder = Path(temp)
        copied = 0
        for frame in checks:
            for i in range(copied, frame + 1):
                shutil.copyfile(frames / f"{i:06}.jpg", folder / f"{i:06}.jpg")
            copied = frame + 1
            started = time.monotonic()
            masks, status = replay_current(predictor, folder, copied, cfg["first_frame_points_xy_pixels_512"], cfg["object_ids"])
            elapsed = time.monotonic() - started
            rows.append({"frame": frame, "prefix_length": copied, "seconds": elapsed,
                "different_pixels_per_slot": np.count_nonzero(masks != cached[frame], axis=(1, 2)).tolist(),
                "nonempty_per_slot": masks.any(axis=(1, 2)).tolist(), "id_status": status.tolist(),
                "cached_id_status": cached_status[frame].tolist()})
    result = {"episode": episode, "source": "development recording only", "checkpoint_sha256": digest,
        "all_matched": all(r["different_pixels_per_slot"] == [0, 0] and r["id_status"] == [1, 1]
                           and all(r["nonempty_per_slot"]) for r in rows), "checks": rows,
        "timing_scope": "session load, prompt, propagation, resizing and close; excludes simulator rendering and policy inference",
        "not_realtime": True}
    dest.write_text(json.dumps(result, indent=2) + "\n"); volume.commit()
    return result


@app.local_entrypoint()
def main():
    print(benchmark.remote())
