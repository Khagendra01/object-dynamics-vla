"""Resumable frozen Muse development cache with per-episode commits and time cap."""
import modal
from scripts.modal_muse_smoke import inference_image, volume

image = (inference_image.add_local_file("scripts/__init__.py", "/root/scripts/__init__.py")
         .add_local_file("scripts/modal_muse_smoke.py", "/root/scripts/modal_muse_smoke.py"))
app = modal.App("odvla-muse-full-cache")


@app.function(image=image, volumes={"/data": volume}, gpu="A100-80GB", cpu=4, memory=131072, timeout=5400)
def cache(max_runtime_seconds: int = 3600):
    import hashlib
    import json
    import time
    from pathlib import Path
    import numpy as np
    import torch
    import transformers
    from PIL import Image
    from odvla.vla.muse_context import FrozenMuseTokens
    if not 60 <= max_runtime_seconds <= 4500:
        raise ValueError("Runtime cap must be 60..4500 seconds")
    parent = Path("/data/libero/milk_sam_development_v1")
    manifest = json.loads((parent / "manifest.json").read_text())
    splits = json.loads((parent / "splits.json").read_text())
    expected_fingerprint = "b7bae449415728262307c262677def270cd50380406183d2455540351a1c72c4"
    if splits["fingerprint"] != expected_fingerprint or splits["test"]:
        raise ValueError("Unexpected development split")
    root = Path("/data/libero/milk_sam_native_v1")
    renders = {x["episode"]: x for x in json.loads((root / "provenance.json").read_text())["entries"]}
    out = Path("/data/libero/muse_middle_tokens_development_v1")
    out.mkdir(parents=True, exist_ok=True)
    identity = {"dataset_fingerprint": expected_fingerprint, "muse_revision": "cec94fcfd6512f2cfe708138c32533b346ea4f69",
        "layer_index": 26, "processor_backend": "Transformers 5.21.0 torchvision 0.25.0: LANCZOS falls back to BICUBIC",
        "dtype": "BF16 bit pattern stored as uint16", "fps": 20, "test_used": False}
    identity_path = out / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Existing cache has different provenance")
    identity_path.write_text(json.dumps(identity, indent=2) + "\n")
    started = time.monotonic()
    model = FrozenMuseTokens("/data/checkpoints/muse-glimmer-30b/cec94fcfd6512f2cfe708138c32533b346ea4f69", 26)
    completed = []
    try:
        for entry in manifest["episodes"]:
            episode = entry["id"]
            folder = out / episode
            episode_manifest = folder / "manifest.json"
            if episode_manifest.exists():
                prev = json.loads(episode_manifest.read_text())
                if prev["frames"] != renders[episode]["frames"] or prev["instruction_sha256"] != hashlib.sha256(entry["instruction"].encode()).hexdigest():
                    raise ValueError("Existing cache episode differs")
                completed.append(episode)
                continue
            if time.monotonic() - started > max_runtime_seconds:
                break
            folder.mkdir(exist_ok=True)
            files = sorted((root / episode).glob("*.jpg"))
            if len(files) != renders[episode]["frames"]:
                raise ValueError("Render frame count mismatch")
            records = []
            for i, image_path in enumerate(files):
                output = folder / f"{i:06}.npz"
                if output.exists():
                    raise FileExistsError(f"Uncommitted partial episode exists: {output}")
                item = model.encode(Image.open(image_path), entry["instruction"], current_date="2026-10-30")
                np.savez(output, **item)
                records.append({"frame": i, "timestamp": i / 20, "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                    "tokens": len(item["attention_mask"]), "bytes": output.stat().st_size,
                    "sha256": hashlib.sha256(output.read_bytes()).hexdigest()})
            record = {"episode": episode, "frames": len(files), "instruction_sha256": hashlib.sha256(entry["instruction"].encode()).hexdigest(),
                "processor_config_sha256": model.processor_hash, "records": records}
            episode_manifest.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
            completed.append(episode)
            (out / "progress.json").write_text(json.dumps({"episodes": completed}, indent=2) + "\n")
            volume.commit()
            print(json.dumps({"episode": episode, "completed": len(completed), "elapsed": time.monotonic() - started}), flush=True)
        result = dict(identity, complete=len(completed) == len(manifest["episodes"]), episodes=completed,
            elapsed_seconds=time.monotonic() - started, processor_config_sha256=model.processor_hash,
            transformers=transformers.__version__, torch=torch.__version__, chat_current_date="2026-10-30", peak_gpu_gib=torch.cuda.max_memory_allocated() / 1024 ** 3)
        (out / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
        volume.commit()
        return result
    finally:
        model.close()


@app.local_entrypoint()
def main(max_runtime_seconds: int = 3600):
    print(cache.remote(max_runtime_seconds))
