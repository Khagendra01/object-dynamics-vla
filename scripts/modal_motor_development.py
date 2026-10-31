"""Bounded milk-only motor-learning run using the audited frozen token cache."""
import modal
from scripts.modal_real_ablations import image as base_image

volume = modal.Volume.from_name("object-dynamics-vla-data", create_if_missing=False)
image = (base_image.add_local_file("scripts/__init__.py", "/root/scripts/__init__.py")
         .add_local_file("scripts/modal_real_ablations.py", "/root/scripts/modal_real_ablations.py")
         .add_local_file("scripts/train_muse_vla.py", "/root/train_muse_vla.py"))
app = modal.App("odvla-motor-development")


@app.function(image=image, gpu="T4", cpu=4, memory=65536, volumes={"/data": volume},
              timeout=4000, max_containers=1, single_use_containers=True)
def run():
    import json
    import sys
    import time
    from pathlib import Path
    from odvla.data import fit_stats, read_json
    from odvla.vla.dataset import CachedEpisodeDataset
    sys.path.insert(0, "/root")
    from train_muse_vla import train

    started = time.monotonic()
    root = Path("/data/libero/milk_sam_development_v1")
    cache = Path("/data/libero/muse_middle_tokens_development_v1")
    out = Path("/data/runs/muse_motor_milk_seed7_500_v1")
    if out.exists():
        raise FileExistsError("Refusing to overwrite an existing run")
    stats = fit_stats(root, read_json(root / "splits.json"))
    def preload(split):
        dataset = CachedEpisodeDataset(root, cache, split, stats)
        items = []
        for i in range(len(dataset)):
            if time.monotonic() - started > 3000:
                raise TimeoutError("Preload exceeded runtime budget")
            items.append(dataset[i])
            if (i + 1) % 500 == 0:
                print(json.dumps({"preload": split, "items": i + 1,
                                  "elapsed": time.monotonic() - started}), flush=True)
        return items
    training, validation = preload("train"), preload("val")
    print(json.dumps({"train_samples": len(training), "val_samples": len(validation),
                      "preload_seconds": time.monotonic() - started}), flush=True)
    def checkpoint():
        volume.commit()
        if time.monotonic() - started > 3000:
            raise TimeoutError("Runtime budget reached after preserving checkpoint")
    try:
        result = train(root, cache, out, "motor", 7, "cuda", steps=500,
                       batch_size=64, accumulation=1,
                       prepared=(stats, training, validation), checkpoint_hook=checkpoint)
    finally:
        volume.commit()
    return {**result, "total_seconds": time.monotonic() - started}


@app.local_entrypoint()
def main():
    print(run.remote())
