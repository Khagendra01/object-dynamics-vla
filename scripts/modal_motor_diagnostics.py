"""Gripper class baselines and M2 train/validation forecast breakdown."""
import modal
from scripts.modal_real_ablations import image as base_image

volume = modal.Volume.from_name("object-dynamics-vla-data", create_if_missing=False)
image = (base_image.add_local_file("scripts/__init__.py", "/root/scripts/__init__.py")
         .add_local_file("scripts/modal_real_ablations.py", "/root/scripts/modal_real_ablations.py")
         .add_local_file("scripts/train_muse_vla.py", "/root/train_muse_vla.py"))
app = modal.App("odvla-motor-diagnostics")


@app.function(image=image, gpu="T4", cpu=4, memory=16384, volumes={"/data": volume},
              timeout=1800, max_containers=1, single_use_containers=True)
def run():
    import json
    import sys
    from pathlib import Path
    import numpy as np
    import torch
    from odvla.data import fit_stats, read_json
    from odvla.vla.action_expert import MuseActionExpert
    from odvla.vla.dataset import CachedEpisodeDataset
    sys.path.insert(0, "/root")
    from train_muse_vla import context, make_batch
    root = Path("/data/libero/milk_sam_development_v1")
    cache = Path("/data/libero/muse_middle_tokens_development_v1")
    runs = Path("/data/runs/muse_motor_milk_comparison_seed7_v1")
    dest = Path("/data/eval/muse_motor_milk_diagnostics_seed7_v1.json")
    if dest.exists(): raise FileExistsError(dest)
    stats = fit_stats(root, read_json(root / "splits.json"))
    results = {}
    for split in ("train", "val"):
        dataset = CachedEpisodeDataset(root, cache, split, stats)
        variants = ("M2",) if split == "train" else ("M0", "M1", "M2")
        for variant in variants:
            model = MuseActionExpert().cuda().eval()
            model.load_state_dict(torch.load(runs / variant / "weights_best.pt", map_location="cuda", weights_only=True))
            rng = np.random.default_rng(921)
            # confusion rows are target -1, +1; columns prediction -1, +1, zero.
            confusion = np.zeros((2, 3), dtype=np.int64)
            groups = {k: {"squared": np.zeros(5), "persistence": np.zeros(5), "count": np.zeros(5)}
                      for k in ("all", "moving", "stationary")}
            with torch.no_grad():
                for start in range(0, len(dataset), 8):
                    b = make_batch([dataset[i] for i in range(start, min(start + 8, len(dataset)))], "cuda")
                    actions = b["actions"]
                    ctx, cmask = context(model, b, variant != "M0")
                    if split == "val":
                        noise = torch.from_numpy(rng.normal(size=tuple(actions.shape)).astype(np.float32)).cuda()
                        pred = model.sample(ctx, cmask, noise, steps=10)[:, 0, -1].cpu().numpy()
                        target = actions[:, 0, -1].cpu().numpy()
                        pred = pred * stats["actions_std"][-1] + stats["actions_mean"][-1]
                        target = target * stats["actions_std"][-1] + stats["actions_mean"][-1]
                        for truth, estimate in zip(np.sign(target), np.sign(pred)):
                            if truth not in (-1, 1): raise ValueError("Unexpected zero gripper target")
                            confusion[int(truth > 0), {-1: 0, 1: 1, 0: 2}[int(estimate)]] += 1
                    if variant == "M2":
                        forecast = model.dynamics(ctx, cmask, actions[:, 0]).cpu().numpy()
                        change, valid = b["change"].cpu().numpy(), b["change_valid"].cpu().numpy()
                        # An object is moving when any observed normalized metric changes by >0.1.
                        moving = (np.where(valid, np.abs(change), 0) > 0.1).any(axis=-1, keepdims=True)
                        for name, selector in (("all", np.ones_like(moving)), ("moving", moving), ("stationary", ~moving)):
                            m = valid & selector
                            groups[name]["squared"] += np.where(m, (forecast - change) ** 2, 0).sum((0, 1))
                            groups[name]["persistence"] += np.where(m, change ** 2, 0).sum((0, 1))
                            groups[name]["count"] += m.sum((0, 1))
            row = {}
            if split == "val":
                counts = confusion.sum(axis=1)
                row["gripper"] = {"target_by_prediction_minus_plus_zero": confusion.tolist(),
                                  "always_minus_accuracy": float(counts[0] / counts.sum()),
                                  "always_plus_accuracy": float(counts[1] / counts.sum()),
                                  "class_recall_minus_plus": [float(confusion[i, i] / counts[i]) if counts[i] else None for i in range(2)]}
            if variant == "M2":
                row["dynamics"] = {name: {"count": v["count"].tolist(),
                    "mse": [float(x / n) if n else None for x, n in zip(v["squared"], v["count"])],
                    "persistence_mse": [float(x / n) if n else None for x, n in zip(v["persistence"], v["count"])]}
                    for name, v in groups.items()}
            results[f"{split}_{variant}"] = row
            print(json.dumps({f"{split}_{variant}": row}, allow_nan=False), flush=True)
            del model
            torch.cuda.empty_cache()
    payload = {"movement": "any valid normalized metric |delta| > 0.1 per object", "results": results}
    dest.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    volume.commit()
    return payload


@app.local_entrypoint()
def main(): print(run.remote())
