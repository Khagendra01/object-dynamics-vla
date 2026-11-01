"""Paired sampled-action and M2 forecasting diagnostics for three seeds."""
import modal
from scripts.modal_real_ablations import image as base_image

volume = modal.Volume.from_name("object-dynamics-vla-data", create_if_missing=False)
image = (base_image.add_local_file("scripts/__init__.py", "/root/scripts/__init__.py")
         .add_local_file("scripts/modal_real_ablations.py", "/root/scripts/modal_real_ablations.py")
         .add_local_file("scripts/train_muse_vla.py", "/root/train_muse_vla.py"))
app = modal.App("odvla-multitask-offline")


@app.function(image=image, gpu="T4", cpu=4, memory=16384, volumes={"/data": volume}, timeout=2700, max_containers=1)
def run():
    import json
    import sys
    import time
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path
    import numpy as np
    import torch
    from odvla.data import read_json
    from odvla.vla.action_expert import MuseActionExpert
    from odvla.vla.dataset import CachedEpisodeDataset
    sys.path.insert(0, "/root")
    from train_muse_vla import make_batch, context
    dest = Path("/data/eval/multitask_milk_offline_3seed_v1.json")
    if dest.exists(): raise FileExistsError(dest)
    root = Path("/data/libero/milk_sam_development_v1"); cache = Path("/data/libero/muse_middle_tokens_development_v1")
    with np.load("/data/runs/muse_multitask_milk_soup_motor_seed7_v1/stats.npz", allow_pickle=False) as z:
        stats = {k: z[k].copy() for k in z.files}
    dataset = CachedEpisodeDataset(root, cache, "val", stats)
    with ThreadPoolExecutor(max_workers=8) as pool: samples = list(pool.map(dataset.__getitem__, range(len(dataset))))
    started = time.monotonic(); rows = []
    for seed in (7, 19, 31):
        for variant in ("M0", "M1", "M2"):
            path = Path(f"/data/runs/multitask_milk_matched_seed{seed}_v1") / variant
            model = MuseActionExpert().cuda().eval()
            model.load_state_dict(torch.load(path / "weights_best.pt", map_location="cuda", weights_only=True))
            rng = np.random.default_rng(921); squared = np.zeros(7); absolute = np.zeros(7); confusion = np.zeros((2, 3), int)
            groups = {k: {"squared": np.zeros(5), "persistence": np.zeros(5), "count": np.zeros(5)}
                      for k in ("all", "moving", "stationary")}
            with torch.no_grad():
                for start in range(0, len(samples), 8):
                    b = make_batch(samples[start:start + 8], "cuda"); actions = b["actions"]
                    noise = torch.from_numpy(rng.normal(size=tuple(actions.shape)).astype(np.float32)).cuda()
                    ctx, mask = context(model, b, variant != "M0")
                    pred = model.sample(ctx, mask, noise, steps=10)[:, 0].cpu().numpy(); target = actions[:, 0].cpu().numpy()
                    delta = pred - target; squared += (delta**2).sum(0); absolute += (np.abs(delta) * stats["actions_std"]).sum(0)
                    native_pred = pred * stats["actions_std"] + stats["actions_mean"]
                    native_target = target * stats["actions_std"] + stats["actions_mean"]
                    for truth, estimate in zip(np.sign(native_target[:, -1]), np.sign(native_pred[:, -1])):
                        if truth not in (-1, 1): raise ValueError("Unexpected zero gripper target")
                        confusion[int(truth > 0), {-1: 0, 1: 1, 0: 2}[int(estimate)]] += 1
                    if variant == "M2":
                        forecast = model.dynamics(ctx, mask, actions[:, 0]).cpu().numpy()
                        change, valid = b["change"].cpu().numpy(), b["change_valid"].cpu().numpy()
                        moving = (np.where(valid, np.abs(change), 0) > .1).any(-1, keepdims=True)
                        for name, selector in (("all", np.ones_like(moving)), ("moving", moving), ("stationary", ~moving)):
                            m = valid & selector
                            groups[name]["squared"] += np.where(m, (forecast - change)**2, 0).sum((0, 1))
                            groups[name]["persistence"] += np.where(m, change**2, 0).sum((0, 1))
                            groups[name]["count"] += m.sum((0, 1))
            n = len(samples); count = confusion.sum(axis=1)
            row = {"seed": seed, "variant": variant, "windows": n,
                   "selected_validation_flow_mse": read_json(path / "best_validation.json")["flow_mse"],
                   "normalized_mse_by_channel": (squared / n).tolist(), "native_mae_by_channel": (absolute / n).tolist(),
                   "gripper_confusion_minus_plus_zero": confusion.tolist(), "gripper_sign_agreement": float((confusion[0,0] + confusion[1,1]) / n),
                   "always_minus_accuracy": float(count[0] / n), "always_plus_accuracy": float(count[1] / n),
                   "gripper_class_recall": [float(confusion[i, i] / count[i]) if count[i] else None for i in range(2)],
                   "initial_shared_gradient": read_json(path / "shared_gradient_diagnostic.json")}
            if variant == "M2":
                row["dynamics"] = {name: {"count": v["count"].tolist(),
                    "mse": [float(x / c) if c else None for x, c in zip(v["squared"], v["count"])],
                    "persistence_mse": [float(x / c) if c else None for x, c in zip(v["persistence"], v["count"])]}
                    for name, v in groups.items()}
            rows.append(row); print(json.dumps(row, allow_nan=False), flush=True)
            del model; torch.cuda.empty_cache()
    result = {"split": "reused validation; also checkpoint selection", "sampling": "seed 921 reset for every arm; 10 Euler steps",
              "channels": ["x", "y", "z", "rx", "ry", "rz", "gripper"],
              "metrics": ["centroid_u", "centroid_v", "bbox_width", "bbox_height", "visible_area"],
              "movement": "any valid normalized metric absolute delta > .1 per object",
              "seconds_excluding_preload": time.monotonic() - started, "rows": rows}
    dest.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n"); volume.commit(); return result


@app.local_entrypoint()
def main(): print(run.remote())
