"""Evaluate frozen milk policies' first sampled action on development windows."""
import modal
from scripts.modal_real_ablations import image as base_image, volume

image = (base_image.add_local_file("scripts/__init__.py", "/root/scripts/__init__.py")
         .add_local_file("scripts/modal_real_ablations.py", "/root/scripts/modal_real_ablations.py"))
app = modal.App("odvla-first-action-eval")


@app.function(image=image, gpu="T4", volumes={"/data": volume}, timeout=3600)
def evaluate():
    import json
    from pathlib import Path
    import numpy as np
    import torch
    from odvla.data import build_windows, validate_splits
    from odvla.experiment import load_run
    ds = Path("/data/libero/milk_sam_development_v1")
    root = Path("/data/runs/milk_sam_matched_policy_v1")
    dest = Path("/data/eval/milk_sam_first_action_v1.json")
    if dest.exists():
        raise FileExistsError("Refusing to overwrite")
    rows = []
    for seed in (7, 19, 31):
        for variant in ("action_only", "joint"):
            run = root / f"{variant}_seed{seed}"
            model, cfg, stats = load_run(run, "cuda")
            splits = json.loads((run / "splits.json").read_text())
            validate_splits(ds, splits)
            if splits["fingerprint"] != cfg["dataset_fingerprint"]:
                raise ValueError("Checkpoint and data fingerprint differ")
            data = build_windows(ds, splits["val"], cfg["history"], cfg["horizon"], stats)
            rng = np.random.default_rng(921)
            sampled = []
            with torch.no_grad():
                for start in range(0, len(data["x"]), 256):
                    stop = min(start + 256, len(data["x"]))
                    x = torch.from_numpy(data["x"][start:stop]).to("cuda")
                    a = data["a"][start:stop]
                    noise = torch.from_numpy(rng.normal(size=a.shape).astype(np.float32)).to("cuda")
                    rng.uniform(size=(len(a), 1))  # Match evaluate_arrays' flow-time draws.
                    h, _ = model.encode(x)
                    sampled.append(model.sample_actions(h, noise, cfg["flow_steps"]).cpu().numpy())
            pred = np.concatenate(sampled).reshape(-1, cfg["horizon"], cfg["action_channels"])[:, 0]
            target = data["a"].reshape(-1, cfg["horizon"], cfg["action_channels"])[:, 0]
            diff = pred - target
            rows.append({"seed": seed, "variant": variant, "windows": len(pred),
                "first_action_normalized_mse_by_channel": (diff ** 2).mean(axis=0).tolist(),
                "first_action_mae_source_units_by_channel": (np.abs(diff) * stats["actions_std"]).mean(axis=0).tolist(),
                "first_action_normalized_mse": float(np.mean(diff ** 2))})
    result = {"split": "reused validation", "sampling": "seed 921, 20 steps; same noise schedule as frozen offline evaluation",
              "channels": ["translation_x", "translation_y", "translation_z", "rotation_x", "rotation_y", "rotation_z", "gripper"],
              "rows": rows, "test_accessed": False}
    dest.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n"); volume.commit()
    return result


@app.local_entrypoint()
def main():
    print(evaluate.remote())
