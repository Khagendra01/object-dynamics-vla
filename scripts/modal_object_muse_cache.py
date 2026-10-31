"""Frozen Muse middle-token cache for alphabet-soup train/validation frames."""
import modal
from scripts.modal_muse_smoke import inference_image

volume = modal.Volume.from_name("object-dynamics-vla-data", create_if_missing=False)
image = (inference_image.add_local_file("scripts/__init__.py", "/root/scripts/__init__.py")
         .add_local_file("scripts/modal_muse_smoke.py", "/root/scripts/modal_muse_smoke.py"))
app = modal.App("odvla-soup-muse-cache")


@app.function(image=image, volumes={"/data": volume}, gpu="A100-80GB", cpu=4, memory=131072,
              timeout=4500, max_containers=1, single_use_containers=True)
def cache(full: bool = False):
    import hashlib
    import json
    import time
    from pathlib import Path
    import numpy as np
    import torch
    import transformers
    from PIL import Image
    from odvla.data import read_json, validate_splits
    from odvla.vla.muse_context import FrozenMuseTokens
    root = Path("/data/libero/object_raw"); splits = read_json(root / "splits.json")
    manifest = validate_splits(root, splits)
    renders = Path("/data/libero/object_native_512_v1")
    render_info = read_json(renders / "provenance.json")
    if render_info["dataset_fingerprint"] != splits["fingerprint"] or render_info["test_episodes_rendered"]:
        raise ValueError("Unexpected render provenance")
    selected = set(splits["train"] + splits["val"])
    entries = [e for e in manifest["episodes"] if e["id"] in selected]
    if not full: entries = entries[:1]
    out = Path("/data/libero/object_muse_middle_tokens_v1" if full else "/data/eval/object_muse_token_pilot_v1")
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists(): raise FileExistsError("Completed cache exists")
    model = FrozenMuseTokens("/data/checkpoints/muse-glimmer-30b/cec94fcfd6512f2cfe708138c32533b346ea4f69", 26)
    started = time.monotonic(); frame_count = 0; byte_count = 0
    try:
        reference = read_json("/data/libero/muse_middle_tokens_development_v1/manifest.json")
        if model.processor_hash != reference["processor_config_sha256"]:
            raise ValueError("Muse processor differs from milk cache")
        render_by_id = {r["episode"]: r for r in render_info["entries"]}
        for entry in entries:
            eid = entry["id"]; folder = out / eid; folder.mkdir(exist_ok=True)
            images = sorted((renders / eid).glob("*.jpg"))
            expected = render_by_id[eid]
            digest = hashlib.sha256()
            for path in images: digest.update(path.read_bytes())
            if len(images) != expected["frames"] or digest.hexdigest() != expected["jpeg_sequence_sha256"]:
                raise ValueError("Rendered frames do not match provenance")
            episode_manifest = folder / "manifest.json"
            if episode_manifest.exists():
                prior = read_json(episode_manifest)
                if prior["frames"] != len(images) or prior["instruction_sha256"] != hashlib.sha256(entry["instruction"].encode()).hexdigest():
                    raise ValueError("Existing cache metadata mismatch")
                for rec in prior["records"]:
                    path = folder / f'{rec["frame"]:06}.npz'
                    if hashlib.sha256(path.read_bytes()).hexdigest() != rec["sha256"]:
                        raise ValueError("Existing cached tokens changed")
                    byte_count += path.stat().st_size
                frame_count += len(images)
                continue
            records = []
            for t, image_path in enumerate(images):
                path = folder / f"{t:06}.npz"
                item = model.encode(Image.open(image_path), entry["instruction"])
                if path.exists():
                    with np.load(path, allow_pickle=False) as old:
                        if set(old.files) != set(item) or any(not np.array_equal(old[k], item[k]) for k in item):
                            raise ValueError("Partial cache token mismatch")
                else:
                    np.savez(path, **item)
                records.append({"frame": t, "timestamp": t / 20, "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                                "tokens": len(item["attention_mask"]), "bytes": path.stat().st_size,
                                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
                byte_count += path.stat().st_size; frame_count += 1
            info = {"episode": eid, "frames": len(images), "instruction_sha256": hashlib.sha256(entry["instruction"].encode()).hexdigest(),
                    "processor_config_sha256": model.processor_hash, "records": records}
            episode_manifest.write_text(json.dumps(info, indent=2) + "\n")
            volume.commit()
            print(json.dumps({"episode": eid, "frames": frame_count, "seconds": time.monotonic() - started}), flush=True)
        result = {"dataset_fingerprint": splits["fingerprint"], "muse_revision": reference["muse_revision"],
                  "layer_index": 26, "processor_backend": reference["processor_backend"], "dtype": reference["dtype"],
                  "fps": 20, "test_used": False, "complete": full, "episodes": [e["id"] for e in entries],
                  "processor_config_sha256": model.processor_hash, "transformers": transformers.__version__,
                  "torch": str(torch.__version__), "peak_gpu_gib": torch.cuda.max_memory_allocated() / 1024**3,
                  "frames": frame_count, "bytes": byte_count, "elapsed_seconds": time.monotonic() - started}
        (out / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
        volume.commit()
        return result
    finally:
        model.close()


@app.local_entrypoint()
def main(full: bool = False): print(cache.remote(full))
