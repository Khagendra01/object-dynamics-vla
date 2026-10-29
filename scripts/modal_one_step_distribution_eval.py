"""Compare one-step head and persistence on validation demos and policy traces."""
import modal

app = modal.App("odvla-one-step-distribution-eval")
volume = modal.Volume.from_name("object-dynamics-vla-data")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch>=2.7", "numpy>=1.24,<3", "Pillow>=10")
         .add_local_python_source("odvla"))


@app.function(image=image, cpu=4, memory=16384, timeout=1800, volumes={"/data": volume})
def evaluate():
    import json
    from pathlib import Path
    import numpy as np
    import torch
    from odvla.data import build_windows, read_json, validate_splits
    from odvla.experiment import load_run
    from odvla.one_step import load_one_step

    torch.set_num_threads(2)
    dataset = Path("/data/libero/milk_oracle")
    run = Path("/data/runs/libero_milk_oracle_ablations/objects_joint_seed7")
    encoder, config, stats = load_run(run, "cpu")
    head = load_one_step("/data/runs/milk_one_step_joint_seed7", run, config)
    splits = read_json(run / "splits.json")
    validate_splits(dataset, splits)
    val = build_windows(dataset, splits["val"], config["history"], config["horizon"], stats)
    objects, features = config["objects"], config["features"]
    width = objects * features
    scale = np.asarray(stats["feature_std"], dtype=np.float64)

    def compute(pred, observed, valid):
        pred, observed = np.asarray(pred, np.float64), np.asarray(observed, np.float64)
        valid = np.asarray(valid, bool)
        if pred.shape != observed.shape or pred.shape != valid.shape or pred.shape[1:] != (objects, features):
            raise ValueError("Expected matching [steps, objects, features] arrays")
        # Threshold is an exploratory, predeclared visible-image-fraction motion proxy.
        moving = np.max(np.where(valid, np.abs(observed), 0), axis=-1) > 0.002
        summaries = {}
        for name, selection in (("all", np.ones_like(moving)), ("moving_objects", moving),
                                ("stationary_objects", ~moving)):
            mask = valid & selection[..., None]
            count = int(mask.sum())
            error = np.where(mask, pred - observed, 0)
            base = np.where(mask, observed, 0)
            denom = max(count, 1)
            summaries[name] = {"valid_elements": count, "object_steps": int(selection.sum()),
                "head_native_mae": float(np.abs(error).sum() / denom) if count else None,
                "persistence_native_mae": float(np.abs(base).sum() / denom) if count else None,
                "head_normalized_mse": float(((error / scale) ** 2).sum() / denom) if count else None,
                "persistence_normalized_mse": float(((base / scale) ** 2).sum() / denom) if count else None}
        return summaries

    predictions = []
    with torch.no_grad():
        for start in range(0, len(val["x"]), 256):
            end = min(start + 256, len(val["x"]))
            h, _ = encoder.encode(torch.from_numpy(val["x"][start:end]))
            action = torch.from_numpy(val["a"][start:end, :config["action_channels"]])
            predictions.append(head(h, action).numpy().reshape(-1, objects, features) * scale)
    demo_prediction = np.concatenate(predictions)
    demo_observed = val["y"][:, :width].reshape(-1, objects, features) * scale
    demo_valid = val["mask"][:, :width].reshape(-1, objects, features).astype(bool)
    ids = np.array([ref["episode"] for ref in val["refs"]])
    demo_per_episode = {episode: compute(demo_prediction[ids == episode], demo_observed[ids == episode],
                                         demo_valid[ids == episode]) for episode in sorted(set(ids))}
    policy = {}
    for seed in (7, 19):
        path = Path(f"/data/eval/milk_objects_joint_seed7_val_sample{seed}_one_step_diagnostics.json")
        trace = read_json(path)
        samples = [step for episode in trace["results"] for step in episode["trace"]]
        if any(step["predicted_delta"] is None for step in samples):
            raise ValueError("Policy trace must contain trained one-step monitor predictions")
        per_episode = {}
        for episode in trace["results"]:
            rows = episode["trace"]
            per_episode[episode["episode"]] = {"success": episode["success"],
                "metrics": compute([r["predicted_delta"] for r in rows],
                                   [r["observed_delta"] for r in rows], [r["valid"] for r in rows])}
        policy[str(seed)] = {"pooled": compute([r["predicted_delta"] for r in samples],
            [r["observed_delta"] for r in samples], [r["valid"] for r in samples]), "per_episode": per_episode}
    report = {"split": "validation", "warning": "validation selected this monitor; no untouched test or grasp/slip labels",
              "motion_rule": "object moving if any valid one-step image-fraction delta exceeds 0.002",
              "demonstrations": {"pooled": compute(demo_prediction, demo_observed, demo_valid),
                                 "per_episode": demo_per_episode}, "policy_sampling_seeds": policy}
    dest = Path("/data/eval/milk_one_step_distribution_val.json")
    dest.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    volume.commit()
    return {"path": str(dest), "motion_rule": report["motion_rule"],
            "demonstrations": report["demonstrations"]["pooled"],
            "policy": {k: v["pooled"] for k, v in policy.items()}}


@app.local_entrypoint()
def main():
    print(evaluate.remote())
