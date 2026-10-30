"""Benchmark five real RGB frames using the frozen Muse middle decoder block."""
import modal
from scripts.modal_muse_smoke import inference_image, volume

image = (inference_image.add_local_file("scripts/__init__.py", "/root/scripts/__init__.py")
         .add_local_file("scripts/modal_muse_smoke.py", "/root/scripts/modal_muse_smoke.py"))
app = modal.App("odvla-muse-token-benchmark")


@app.function(image=image, volumes={"/data": volume}, gpu="A100-80GB", cpu=4, memory=131072, timeout=1800)
def benchmark():
    import hashlib
    import inspect
    import json
    import logging
    import time
    import warnings
    from pathlib import Path
    import numpy as np
    import torch
    import transformers
    from PIL import Image
    source = Path("/data/checkpoints/muse-glimmer-30b/cec94fcfd6512f2cfe708138c32533b346ea4f69")
    dest = Path("/data/eval/muse_middle_token_benchmark_v1.json")
    if dest.exists():
        raise FileExistsError("Refusing to overwrite benchmark")
    processor = transformers.AutoProcessor.from_pretrained(str(source), trust_remote_code=False)
    factory = getattr(transformers, "AutoModelForMultimodalLM", None) or transformers.AutoModelForImageTextToText
    started = time.perf_counter()
    model = factory.from_pretrained(str(source), dtype=torch.bfloat16, device_map="auto", trust_remote_code=False)
    model.eval().requires_grad_(False)
    loaded_seconds = time.perf_counter() - started
    expected = model.config.text_config.num_hidden_layers
    candidates = [(name, module) for name, module in model.named_modules()
                  if isinstance(module, torch.nn.ModuleList) and len(module) == expected and "vision" not in name]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one text layer list of length {expected}, got {[n for n, _ in candidates]}")
    layer_name, layers = candidates[0]
    index = expected // 2
    captured = {}
    def hook(module, args, result):
        value = result[0] if isinstance(result, tuple) else result
        if not torch.is_tensor(value) or value.ndim != 3:
            raise ValueError("Selected decoder block did not return [batch,tokens,width]")
        captured["value"] = value.detach().clone()
    handle = layers[index].register_forward_hook(hook)
    frame_dir = Path("/data/libero/milk_sam_native_v1/episode_0014")
    instruction = "pick up the milk and place it in the basket"
    rows = []
    try:
        for frame in (0, 30, 60, 90, 120):
            path = frame_dir / f"{frame:06}.jpg"
            raw = path.read_bytes()
            rgb = Image.open(path).convert("RGB")
            messages = [{"role": "user", "content": [{"type": "image", "image": rgb}, {"type": "text", "text": instruction}]}]
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                batch = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                    return_dict=True, return_tensors="pt")
            batch = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in batch.items()}
            torch.cuda.synchronize(); start = time.perf_counter()
            kwargs = {"use_cache": False, "return_dict": True}
            if "logits_to_keep" in inspect.signature(model.forward).parameters:
                kwargs["logits_to_keep"] = 1
            with torch.inference_mode():
                outputs = model(**batch, **kwargs)
            torch.cuda.synchronize(); elapsed = time.perf_counter() - start
            hidden = captured.pop("value")
            mask = batch["attention_mask"]
            if hidden.shape[:2] != mask.shape:
                raise ValueError(f"Token mask shape mismatch: {hidden.shape} / {mask.shape}")
            rows.append({"frame": frame, "image_sha256": hashlib.sha256(raw).hexdigest(),
                "shape": list(hidden.shape), "dtype": str(hidden.dtype), "valid_tokens": int(mask.sum().item()),
                "seconds_forward": elapsed, "bytes_bfloat16": hidden.numel() * 2,
                "finite": bool(torch.isfinite(hidden).all().item()),
                "warnings": [str(x.message) for x in caught]})
            del hidden, outputs, batch
    finally:
        handle.remove()
    result = {"model_revision": "cec94fcfd6512f2cfe708138c32533b346ea4f69", "transformers": transformers.__version__,
        "torch": torch.__version__, "decoder_layers": expected, "selected_layer_zero_based": index,
        "module_name": layer_name, "processor_class": type(processor).__name__,
        "processor_config_sha256": hashlib.sha256((source / "processor_config.json").read_bytes()).hexdigest(),
        "processor_resample_configured": json.loads((source / "processor_config.json").read_text())["image_processor"]["resample"],
        "preprocessing": "pinned processor, native 512 RGB; torchvision 0.25 logs LANCZOS-to-BICUBIC fallback",
        "frames": rows, "peak_gpu_gib": torch.cuda.max_memory_allocated() / (1024 ** 3),
        "load_seconds": loaded_seconds, "projected_full_cache_bytes": int(np.mean([r["bytes_bfloat16"] for r in rows]) * 6098),
        "projected_forward_seconds": float(np.mean([r["seconds_forward"] for r in rows]) * 6098),
        "reference_masks_opened": False, "frozen": all(not p.requires_grad for p in model.parameters())}
    dest.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n"); volume.commit()
    return result


@app.local_entrypoint()
def main():
    print(benchmark.remote())
