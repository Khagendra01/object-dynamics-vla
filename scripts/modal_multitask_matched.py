"""Matched milk M0/M1/M2 adaptations from the audited two-task motor checkpoint."""
import modal
from scripts.modal_real_ablations import image as base_image

volume = modal.Volume.from_name("object-dynamics-vla-data", create_if_missing=False)
image = (base_image.add_local_file("scripts/__init__.py", "/root/scripts/__init__.py")
         .add_local_file("scripts/modal_real_ablations.py", "/root/scripts/modal_real_ablations.py")
         .add_local_file("scripts/train_muse_vla.py", "/root/train_muse_vla.py"))
app = modal.App("odvla-multitask-matched-milk")


@app.function(image=image, gpu="T4", cpu=4, memory=65536, volumes={"/data": volume},
              timeout=3600, max_containers=1, single_use_containers=True)
def run(seed: int = 7):
    import json
    import sys
    import time
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path
    import numpy as np
    from odvla.vla.dataset import CachedEpisodeDataset
    sys.path.insert(0, "/root")
    from train_muse_vla import train
    if seed not in (7, 19, 31): raise ValueError("Undeclared seed")
    root = Path("/data/libero/milk_sam_development_v1")
    cache = Path("/data/libero/muse_middle_tokens_development_v1")
    motor = Path("/data/runs/muse_multitask_milk_soup_motor_seed7_v1")
    checkpoint = motor / "weights_best.pt"
    audit = Path("/data/eval/milk_transfer_contract_v1.json")
    out = Path(f"/data/runs/multitask_milk_matched_seed{seed}_v1")
    if out.exists(): raise FileExistsError(out)
    with np.load(motor / "stats.npz", allow_pickle=False) as z: stats = {k: z[k].copy() for k in z.files}
    started = time.monotonic(); subsets = []
    for split in ("train", "val"):
        dataset = CachedEpisodeDataset(root, cache, split, stats)
        with ThreadPoolExecutor(max_workers=8) as pool:
            subsets.append(list(pool.map(dataset.__getitem__, range(len(dataset)))))
        print(json.dumps({"preload": split, "items": len(subsets[-1]), "seconds": time.monotonic() - started}), flush=True)
    out.mkdir(parents=True); rows = []
    def commit():
        volume.commit()
        if time.monotonic() - started > 3000: raise TimeoutError("Budget cap; saved checkpoint preserved")
    try:
        for variant in ("M0", "M1", "M2"):
            result = train(root, cache, out / variant, variant, seed, "cuda", steps=500, batch_size=64, accumulation=1,
                           prepared=(stats, subsets[0], subsets[1]), motor_checkpoint=checkpoint,
                           dataset_audit=audit, checkpoint_hook=commit)
            rows.append({"variant": variant, **result})
            (out / "results.json").write_text(json.dumps(rows, indent=2) + "\n"); volume.commit()
    finally:
        volume.commit()
    return {"seed": seed, "results": rows, "seconds": time.monotonic() - started}


@app.local_entrypoint()
def main(seed: int = 7): print(run.remote(seed))
