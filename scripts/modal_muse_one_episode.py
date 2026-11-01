"""Cache one development episode's unprojected BF16 Muse tokens, with parity check."""
import modal
from scripts.modal_muse_smoke import inference_image, volume

image = (inference_image.add_local_file("scripts/__init__.py", "/root/scripts/__init__.py")
         .add_local_file("scripts/modal_muse_smoke.py", "/root/scripts/modal_muse_smoke.py"))
app = modal.App("odvla-muse-one-episode")


@app.function(image=image, volumes={"/data": volume}, gpu="A100-80GB", cpu=4, memory=131072, timeout=1800)
def cache():
    import hashlib
    import json
    import time
    from pathlib import Path
    import numpy as np
    import torch
    import transformers
    from PIL import Image
    from odvla.vla.muse_context import FrozenMuseTokens
    root = Path("/data/libero/milk_sam_native_v1/episode_0014")
    out = Path("/data/libero/muse_middle_tokens_episode0014_v1")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a fresh output directory")
    out.mkdir(parents=True, exist_ok=True)
    instruction = "pick up the milk and place it in the basket"
    files = sorted(root.glob("*.jpg"))
    if len(files) != 137 or [p.name for p in files] != [f"{i:06}.jpg" for i in range(len(files))]:
        raise ValueError("Unexpected episode render")
    model = FrozenMuseTokens("/data/checkpoints/muse-glimmer-30b/cec94fcfd6512f2cfe708138c32533b346ea4f69", 26)
    records = []
    started = time.monotonic()
    try:
        for t, path in enumerate(files):
            clock = time.monotonic()
            item = model.encode(Image.open(path), instruction, current_date="2026-10-30")
            output = out / f"{t:06}.npz"
            np.savez(output, **item)
            records.append({"frame": t, "timestamp": t / 20, "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "file": output.name, "shape": list(item["features_bfloat16_bits"].shape),
                "valid_tokens": int(item["attention_mask"].sum()), "bytes": output.stat().st_size,
                "seconds": time.monotonic() - clock})
            if (t + 1) % 20 == 0:
                (out / "progress.json").write_text(json.dumps(records, indent=2) + "\n"); volume.commit()
        replay = model.encode(Image.open(files[0]), instruction, current_date="2026-10-30")
        with np.load(out / "000000.npz", allow_pickle=False) as saved:
            parity = {k: bool(np.array_equal(saved[k], replay[k])) for k in replay}
        result = {"complete": True, "episode": "episode_0014", "frames": len(files),
            "dataset_fingerprint": "b7bae449415728262307c262677def270cd50380406183d2455540351a1c72c4",
            "revision": "cec94fcfd6512f2cfe708138c32533b346ea4f69", "layer_index": model.layer_index,
            "module_name": model.module_name, "processor_config_sha256": model.processor_hash,
            "processor_backend": "Transformers 5.21.0 torchvision 0.25.0: configured LANCZOS falls back to BICUBIC", "chat_current_date": "2026-10-30",
            "transformers": transformers.__version__, "torch": torch.__version__, "dtype": "BF16 bit pattern in uint16",
            "instruction_sha256": hashlib.sha256(instruction.encode()).hexdigest(), "frames_metadata": records,
            "cache_online_same_backend_parity_first_frame": parity, "frozen": all(not p.requires_grad for p in model.model.parameters()),
            "elapsed_seconds_after_model_load": time.monotonic() - started,
            "peak_gpu_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
            "reference_masks_opened": False}
        (out / "manifest.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        volume.commit()
        return {k: v for k, v in result.items() if k != "frames_metadata"}
    finally:
        model.close()


@app.local_entrypoint()
def main():
    print(cache.remote())
