"""Download pinned public Muse weights and test one real multimodal forward pass.

The download is intentionally a separate CPU function so GPU billing does not
continue while the approximately 60 GB BF16 checkpoint is being fetched.
"""
import modal

app = modal.App("object-dynamics-vla-muse-smoke")
volume = modal.Volume.from_name("object-dynamics-vla-data")
REVISION = "cec94fcfd6512f2cfe708138c32533b346ea4f69"
MODEL = "meta-models/Muse-Glimmer-30B"
CHECKPOINT = "/data/checkpoints/muse-glimmer-30b/" + REVISION
download_image = modal.Image.debian_slim(python_version="3.12").pip_install("huggingface-hub>=0.34")
inference_image = (modal.Image.debian_slim(python_version="3.12")
                   .pip_install("torch==2.10.0", "torchvision", index_url="https://download.pytorch.org/whl/cu128")
                   .pip_install("transformers==5.21.0", "accelerate", "safetensors", "pillow", "numpy<2")
                   .add_local_python_source("odvla")
                   .add_local_file("scripts/cache_muse.py", "/root/cache_muse.py"))


@app.function(image=download_image, volumes={"/data": volume}, cpu=4, memory=8192, timeout=7200)
def download():
    from huggingface_hub import snapshot_download
    path = snapshot_download(repo_id=MODEL, revision=REVISION, local_dir=CHECKPOINT)
    volume.commit()
    return path


@app.function(image=inference_image, volumes={"/data": volume}, gpu="A100-80GB",
              cpu=4, memory=131072, timeout=3600)
def forward():
    import sys
    import time
    import torch
    import transformers
    from PIL import Image
    from odvla.data import load_manifest, load_episode

    sys.path.insert(0, "/root")
    from cache_muse import pick_last_hidden

    dataset = "/data/libero/milk_oracle"
    manifest = load_manifest(dataset)
    episode = load_episode(dataset, manifest["episodes"][0], manifest)
    processor = transformers.AutoProcessor.from_pretrained(CHECKPOINT, trust_remote_code=False)
    factory = getattr(transformers, "AutoModelForMultimodalLM", None)
    if factory is None:
        factory = transformers.AutoModelForImageTextToText
    model = factory.from_pretrained(CHECKPOINT, dtype=torch.bfloat16, device_map="auto", trust_remote_code=False)
    model.eval().requires_grad_(False)
    image = Image.fromarray(episode["rgb"][0])
    messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                                  {"type": "text", "text": manifest["episodes"][0]["instruction"]}]}]
    batch = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                          return_dict=True, return_tensors="pt")
    batch = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in batch.items()}
    timings = []
    vectors = []
    for _ in range(3):
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            outputs = model(**batch, output_hidden_states=True, use_cache=False, return_dict=True)
            hidden = pick_last_hidden(outputs)
            vectors.append(hidden[0, -1].float().cpu())
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - started)
        del outputs
    return {"revision": REVISION, "shape": list(hidden.shape),
            "finite": bool(torch.isfinite(hidden).all().item()),
            "peak_gpu_gib": torch.cuda.max_memory_allocated() / (1024 ** 3),
            "forward_seconds": timings,
            "repeat_max_abs_difference": float((vectors[1] - vectors[2]).abs().max().item()),
            "transformers_version": transformers.__version__}


@app.function(image=inference_image, volumes={"/data": volume}, gpu="A100-80GB",
              cpu=4, memory=131072, timeout=7200)
def cache(dataset: str, out: str, stride: int):
    import subprocess
    import sys

    command = [sys.executable, "/root/cache_muse.py", "--dataset", dataset,
               "--out", out, "--model", MODEL, "--model-path", CHECKPOINT,
               "--revision", REVISION, "--stride", str(stride)]
    result = subprocess.run(command, capture_output=True, text=True)
    volume.commit()
    return {"returncode": result.returncode, "stdout": result.stdout[-4000:],
            "stderr": result.stderr[-4000:]}


@app.local_entrypoint()
def main(make_cache: bool = False, dataset: str = "/data/libero/milk_oracle",
         out: str = "/data/libero/milk_oracle_muse", stride: int = 4):
    print("Checkpoint:", download.remote())
    if make_cache:
        result = cache.remote(dataset, out, stride)
        print(result)
        if result["returncode"]:
            raise SystemExit(result["returncode"])
    else:
        print("Forward:", forward.remote())
