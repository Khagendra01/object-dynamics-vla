"""Score provisional Muse motor variants on reused milk validation windows."""
import modal
from scripts.modal_real_ablations import image as base_image

volume = modal.Volume.from_name("object-dynamics-vla-data", create_if_missing=False)
image = (base_image.add_local_file("scripts/__init__.py", "/root/scripts/__init__.py")
         .add_local_file("scripts/modal_real_ablations.py", "/root/scripts/modal_real_ablations.py")
         .add_local_file("scripts/train_muse_vla.py", "/root/train_muse_vla.py"))
app = modal.App("odvla-motor-offline-eval")


@app.function(image=image, gpu="T4", cpu=4, memory=16384, volumes={"/data": volume},
              timeout=1800, max_containers=1, single_use_containers=True)
def run():
    import json
    import sys
    import time
    from pathlib import Path
    import numpy as np
    import torch
    from odvla.data import fit_stats, read_json
    from odvla.vla.action_expert import MuseActionExpert
    from odvla.vla.dataset import CachedEpisodeDataset
    sys.path.insert(0, "/root")
    from train_muse_vla import make_batch, context

    root = Path("/data/libero/milk_sam_development_v1")
    cache = Path("/data/libero/muse_middle_tokens_development_v1")
    runs = Path("/data/runs/muse_motor_milk_comparison_seed7_v1")
    dest = Path("/data/eval/muse_motor_milk_offline_seed7_v1.json")
    if dest.exists():
        raise FileExistsError("Refusing to overwrite existing evaluation")
    stats = fit_stats(root, read_json(root / "splits.json"))
    dataset = CachedEpisodeDataset(root, cache, "val", stats)
    samples = [dataset[i] for i in range(len(dataset))]
    results = []
    start_time = time.monotonic()
    for variant in ("M0", "M1", "M2"):
        model = MuseActionExpert().cuda().eval()
        model.load_state_dict(torch.load(runs / variant / "weights_best.pt", map_location="cuda", weights_only=True))
        rng = np.random.default_rng(921)
        sum_sq = np.zeros(7); sum_abs = np.zeros(7); sign_count = 0
        dyn_sq = np.zeros(5); pers_sq = np.zeros(5); dyn_count = np.zeros(5)
        example = None
        with torch.no_grad():
            for start in range(0, len(samples), 8):
                b = make_batch(samples[start:start + 8], "cuda")
                a = b["actions"]
                noise = torch.from_numpy(rng.normal(size=tuple(a.shape)).astype(np.float32)).cuda()
                ctx, mask = context(model, b, variant != "M0")
                pred = model.sample(ctx, mask, noise, steps=10)[:, 0].cpu().numpy()
                target = a[:, 0].cpu().numpy()
                diff = pred - target
                sum_sq += (diff ** 2).sum(axis=0)
                sum_abs += (np.abs(diff) * stats["actions_std"]).sum(axis=0)
                native_pred = pred * stats["actions_std"] + stats["actions_mean"]
                native_target = target * stats["actions_std"] + stats["actions_mean"]
                sign_count += int((np.sign(native_pred[:, -1]) == np.sign(native_target[:, -1])).sum())
                forecast = model.dynamics(ctx, mask, a[:, 0]).cpu().numpy()
                change, valid = b["change"].cpu().numpy(), b["change_valid"].cpu().numpy()
                dyn_sq += np.where(valid, (forecast - change) ** 2, 0).sum(axis=(0, 1))
                pers_sq += np.where(valid, change ** 2, 0).sum(axis=(0, 1))
                dyn_count += valid.sum(axis=(0, 1))
                if example is None:
                    example = {"episode": samples[start]["episode"], "frame": samples[start]["t"],
                               "target_action_source": native_target[0].tolist(),
                               "sampled_action_source": native_pred[0].tolist(),
                               "change_normalized": change[0].tolist(),
                               "change_valid": valid[0].tolist(),
                               "forecast_normalized": forecast[0].tolist()}
        n = len(samples)
        results.append({"variant": variant, "windows": n,
                       "first_action_normalized_mse_by_channel": (sum_sq / n).tolist(),
                       "first_action_native_mae_by_channel": (sum_abs / n).tolist(),
                       "gripper_sign_agreement": sign_count / n,
                       "dynamics_normalized_mse_by_metric": (dyn_sq / np.maximum(dyn_count, 1)).tolist(),
                       "persistence_normalized_mse_by_metric": (pers_sq / np.maximum(dyn_count, 1)).tolist(),
                       "valid_counts_by_metric": dyn_count.tolist(), "example": example})
        print(json.dumps(results[-1], allow_nan=False), flush=True)
        del model
        torch.cuda.empty_cache()
    result = {"split": "reused validation; also used for checkpoint selection",
              "sampling": "seed 921 independently reset for each variant, 10 Euler steps",
              "channels": ["x", "y", "z", "rx", "ry", "rz", "gripper"],
              "metrics": ["centroid_u", "centroid_v", "bbox_width", "bbox_height", "visible_area"],
              "dynamics_warning": "Only M2 has a trained dynamics head; forecasts use demonstrated current action.",
              "evaluation_seconds_excluding_preload": time.monotonic() - start_time, "results": results}
    dest.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    volume.commit()
    return {"output": str(dest), "results": results}


@app.local_entrypoint()
def main():
    print(run.remote())
