"""Diagnose runtime/cache processor token IDs on identical recorded pixels."""
import modal
from scripts.modal_muse_motor_rollout import image as base_image

image = (base_image.add_local_file("scripts/__init__.py", "/root/scripts/__init__.py")
         .add_local_file("scripts/modal_muse_motor_rollout.py", "/root/scripts/modal_muse_motor_rollout.py"))
volume = modal.Volume.from_name("object-dynamics-vla-data", create_if_missing=False)
app = modal.App("odvla-muse-processor-parity")


@app.function(image=image, volumes={"/data": volume}, cpu=2, memory=8192, timeout=600)
def run():
    import hashlib
    import json
    from pathlib import Path
    import numpy as np
    import tokenizers
    import transformers
    from PIL import Image
    path = Path("/data/libero/milk_sam_native_v1/episode_0014/000000.jpg")
    model = "/data/checkpoints/muse-glimmer-30b/cec94fcfd6512f2cfe708138c32533b346ea4f69"
    processor = transformers.AutoProcessor.from_pretrained(model, trust_remote_code=False)
    messages = [{"role": "user", "content": [{"type": "image", "image": Image.open(path).convert("RGB")},
        {"type": "text", "text": "pick up the milk and place it in the basket"}]}]
    batch = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
    with np.load("/data/libero/muse_middle_tokens_development_v1/episode_0014/000000.npz", allow_pickle=False) as z:
        old = z["input_ids"].tolist()
    new = batch["input_ids"][0].tolist()
    differences = [{"index": i, "cache": a, "runtime": b} for i, (a, b) in enumerate(zip(old, new)) if a != b]
    result = {"image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "old_length": len(old), "new_length": len(new),
              "token_differences": differences, "tokenizers": tokenizers.__version__, "transformers": transformers.__version__,
              "runtime_last_tokens": new[-40:], "cache_last_tokens": old[-40:],
              "runtime_prefix_decoded": processor.tokenizer.decode(new[:60]),
              "cache_prefix_decoded": processor.tokenizer.decode(old[:60])}
    cache_dates = {}
    for name, base, episode in (("milk", "muse_middle_tokens_development_v1", "episode_0014"),
                                ("soup", "object_muse_middle_tokens_v1", "episode_0000")):
        for idx in (0, 50, 100):
            with np.load(f"/data/libero/{base}/{episode}/{idx:06}.npz", allow_pickle=False) as z:
                cache_dates[f"{name}_{idx}"] = processor.tokenizer.decode(z["input_ids"][:45].tolist())
    result["cached_prompt_prefixes"] = cache_dates
    import re
    summary = {}
    for name, base in (("milk", "muse_middle_tokens_development_v1"), ("soup", "object_muse_middle_tokens_v1")):
        root = Path("/data/libero") / base
        manifest = json.loads((root / "manifest.json").read_text())
        dates = {}
        for episode in manifest["episodes"]:
            info = json.loads((root / episode / "manifest.json").read_text())
            for t in {0, info["frames"] - 1}:
                with np.load(root / episode / f"{t:06}.npz", allow_pickle=False) as z:
                    prefix = processor.tokenizer.decode(z["input_ids"][:45].tolist())
                match = re.search(r"Current date: ([0-9-]+)", prefix)
                date = match.group(1) if match else "missing"
                dates[date] = dates.get(date, 0) + 1
        summary[name] = dates
    result["episode_first_last_date_counts"] = summary
    dest = Path("/data/eval/muse_processor_parity_debug_v4.json")
    if dest.exists(): raise FileExistsError(dest)
    dest.write_text(json.dumps(result, indent=2) + "\n"); volume.commit(); return result


@app.local_entrypoint()
def main(): print(run.remote())
