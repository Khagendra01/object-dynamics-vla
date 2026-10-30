"""Seed-7 matched suite with resident cache and bounded billable runtime."""
import modal
from scripts.modal_real_ablations import image as base_image, volume

image = (base_image.add_local_file("scripts/__init__.py", "/root/scripts/__init__.py")
         .add_local_file("scripts/modal_real_ablations.py", "/root/scripts/modal_real_ablations.py")
         .add_local_file("scripts/train_muse_vla.py", "/root/train_muse_vla.py"))
app = modal.App("odvla-muse-seed7-suite")


@app.function(image=image, gpu="T4", cpu=4, memory=65536, volumes={"/data": volume}, timeout=19800,
              single_use_containers=True)
def run(max_seconds: int = 18000):
    import json
    import sys
    import time
    from pathlib import Path
    import torch
    from odvla.data import fit_stats, read_json
    from odvla.vla.dataset import CachedEpisodeDataset
    sys.path.insert(0, "/root")
    from train_muse_vla import train
    if not 600 <= max_seconds <= 18000:
        raise ValueError("Runtime cap must be 600..18000 seconds")
    root = Path("/data/libero/milk_sam_development_v1")
    cache = Path("/data/libero/muse_middle_tokens_development_v1")
    out = Path("/data/runs/muse_vla_seed7_v2")
    if out.exists(): raise FileExistsError("Refusing to overwrite run suite")
    out.mkdir(parents=True)
    started = time.monotonic()
    stats = fit_stats(root, read_json(root / "splits.json"))
    trainset = CachedEpisodeDataset(root, cache, "train", stats)
    valset = CachedEpisodeDataset(root, cache, "val", stats)
    # Read, hash-check and retain each frozen token sample once; no projector is cached.
    def preload(dataset, name):
        result = []
        for i in range(len(dataset)):
            if time.monotonic() - started > max_seconds:
                raise TimeoutError("Stopped during preload to preserve cloud budget")
            result.append(dataset[i])
            if (i + 1) % 500 == 0:
                print(json.dumps({"preload": name, "items": i + 1, "elapsed": time.monotonic() - started}), flush=True)
        return result
    train_data = preload(trainset, "train"); val_data = preload(valset, "val")
    prepared = (stats, train_data, val_data)
    print(json.dumps({"preload_complete_seconds": time.monotonic() - started,
                      "train_samples": len(train_data), "val_samples": len(val_data)}), flush=True)
    results = []
    def commit_and_limit():
        volume.commit()
        if time.monotonic() - started > max_seconds:
            raise TimeoutError("Budget runtime cap reached; latest checkpoint is preserved")
    try:
        for variant in ("M0", "M1", "M2"):
            result = train(root, cache, out / variant, variant, 7, "cuda", steps=5000, allow_scratch=True,
                           batch_size=64, accumulation=1, prepared=prepared, checkpoint_hook=commit_and_limit)
            results.append({"variant": variant, **result})
            torch.cuda.empty_cache()
    except TimeoutError as exc:
        result = {"complete": False, "reason": str(exc), "completed_arms": results,
                  "elapsed_seconds": time.monotonic() - started}
    else:
        result = {"complete": True, "completed_arms": results, "elapsed_seconds": time.monotonic() - started}
    (out / "suite.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    volume.commit()
    return result


@app.local_entrypoint()
def main(max_seconds: int = 18000): print(run.remote(max_seconds))
