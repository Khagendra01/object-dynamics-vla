"""Reproducible training/evaluation; held-out test is never loaded by train()."""
from __future__ import annotations
import json
import platform
import time
from pathlib import Path
import numpy as np
from .data import (read_json, write_json, load_manifest, load_episode, validate_splits,
                   fit_stats, build_windows)


def get_torch():
    try:
        import torch
        return torch
    except ImportError as exc:
        raise RuntimeError("Install PyTorch first; see README GPU setup. CPU smoke uses CPU PyTorch.") from exc


def _metrics(pred, data, horizon, objects, features, scale):
    shape = (len(pred), horizon, objects, features)
    pred, target, mask = pred.reshape(shape), data["y"].reshape(shape), data["mask"].reshape(shape)
    error2 = (pred - target) ** 2
    moving = np.max(np.where(mask.astype(bool), np.abs(target), 0), axis=(1, 2, 3)) > 0.01
    def subset_mse(selection):
        count = mask[selection].sum()
        return float((error2[selection] * mask[selection]).sum() / count) if count else None
    result = {"normalized_mse": subset_mse(slice(None)), "moving_normalized_mse": subset_mse(moving),
              "stationary_normalized_mse": subset_mse(~moving),
              "moving_windows": int(moving.sum()), "windows": len(pred),
              "valid_target_elements": int(mask.sum()),
              "per_horizon_normalized_mse": [], "per_metric_mae_native_units": []}
    for i in range(horizon):
        count = mask[:, i].sum()
        result["per_horizon_normalized_mse"].append(float((error2[:, i] * mask[:, i]).sum() / count) if count else None)
    absolute = np.abs(pred - target) * np.asarray(scale)[None, None, None, :]
    for j in range(features):
        count = mask[..., j].sum()
        result["per_metric_mae_native_units"].append(float((absolute[..., j] * mask[..., j]).sum() / count) if count else None)
    return result


def evaluate_arrays(model, data, device, config, stats, sample_actions=False):
    torch = get_torch()
    model.eval()
    preds, action_preds, gates = [], [], []
    flow_sum, flow_count = 0.0, 0
    rng = np.random.default_rng(921)
    with torch.no_grad():
        for start in range(0, len(data["x"]), 256):
            stop = start + 256
            x = torch.from_numpy(data["x"][start:stop]).to(device)
            a = torch.from_numpy(data["a"][start:stop]).to(device)
            h, gate = model.encode(x)
            predicted = model.dynamics(h, a, config["no_dynamics_actions"])
            preds.append(predicted.cpu().numpy())
            gates.append(gate.cpu().numpy())
            noise = torch.from_numpy(rng.normal(size=tuple(a.shape)).astype(np.float32)).to(device)
            tau = torch.from_numpy(rng.uniform(size=(len(a), 1)).astype(np.float32)).to(device)
            v = model.velocity(h, (1 - tau) * noise + tau * a, tau)
            flow_sum += float((v - (a - noise)).square().sum().item())
            flow_count += a.numel()
            if sample_actions:
                action_preds.append(model.sample_actions(h, noise, config["flow_steps"]).cpu().numpy())
    pred = np.concatenate(preds)
    args = (config["horizon"], config["objects"], config["features"], stats["feature_std"])
    result = {"dynamics": _metrics(pred, data, *args),
              "persistence": _metrics(np.zeros_like(pred), data, *args),
              "constant_velocity": _metrics(data["cv"], data, *args),
              "flow_mse": flow_sum / max(flow_count, 1),
              "mean_measurement_gates": np.concatenate(gates).mean(axis=0).tolist()}
    if action_preds:
        sampled = np.concatenate(action_preds)
        result["sampled_action_normalized_mse"] = float(np.mean((sampled - data["a"]) ** 2))
        error = np.abs(sampled - data["a"]).reshape(-1, config["horizon"], config["action_channels"])
        result["sampled_action_mae_source_units"] = (error * stats["actions_std"]).mean(axis=(0, 1)).tolist()
    # Episode-level diagnostics; windows from the same episode are correlated.
    result["per_episode_dynamics"] = {}
    ids = np.array([r["episode"] for r in data["refs"]])
    for eid in sorted(set(ids)):
        take = ids == eid
        small = {"y": data["y"][take], "mask": data["mask"][take]}
        result["per_episode_dynamics"][eid] = _metrics(pred[take], small, *args)["normalized_mse"]
    return result, pred


def train(dataset, split_file, output, *, epochs=20, history=4, horizon=4, hidden=128,
          batch_size=64, lr=1e-3, dynamics_weight=1.0, no_objects=False,
          no_dynamics_actions=False, seed=7, device="cpu", max_per_episode=0, threads=2,
          allow_experimental_perception=False, checkpoint_selection="combined"):
    torch = get_torch()
    from .model import ObjectDynamicsPolicy, masked_mse
    if epochs < 1 or history < 2 or horizon < 1 or batch_size < 1 or dynamics_weight < 0 or hidden < 1 or lr <= 0:
        raise ValueError("Invalid training configuration; history>=2, epochs/horizon/batch/hidden>=1, lr>0, weight>=0")
    if no_objects and dynamics_weight != 0:
        raise ValueError("no_objects baseline requires dynamics_weight=0; compare matched action baselines")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite an existing run; use a new --out")
    splits = read_json(split_file)
    manifest = validate_splits(dataset, splits)
    if checkpoint_selection not in ("combined", "flow"):
        raise ValueError("checkpoint_selection must be combined or flow")
    if manifest.get("perception", {}).get("causal") is not True and not (
            allow_experimental_perception and manifest.get("perception", {}).get("experimental_prefix_audited") is True):
        raise ValueError("Perception causality must be audited and declared before training")
    stats = fit_stats(dataset, splits)
    tr = build_windows(dataset, splits["train"], history, horizon, stats, no_objects, max_per_episode)
    va = build_windows(dataset, splits["val"], history, horizon, stats, no_objects, max_per_episode)
    first = load_episode(dataset, manifest["episodes"][0], manifest, include_images=False)
    objects, features = first["features"].shape[1:]
    model_config = {"input_dim": tr["x"].shape[1], "object_input_dim": 4 * history * objects * features,
                    "metric_dim": features, "context_dim": first["context"].shape[1],
                    "action_dim": tr["a"].shape[1], "target_dim": tr["y"].shape[1], "hidden": hidden}
    config = {"model": model_config, "history": history, "horizon": horizon,
              "objects": objects, "features": features, "action_channels": first["actions"].shape[1],
              "epochs": epochs, "batch_size": batch_size, "lr": lr, "dynamics_weight": dynamics_weight,
              "no_objects": no_objects, "no_dynamics_actions": no_dynamics_actions, "seed": seed,
              "flow_steps": 20, "max_per_episode": max_per_episode,
              "dataset_fingerprint": splits["fingerprint"], "synthetic": manifest.get("synthetic", False),
              "metric_names": manifest["metric_names"], "metric_units": manifest["metric_units"],
              "torch_version": str(torch.__version__), "python_version": platform.python_version(),
              "device": device, "context_provenance": manifest.get("context_provenance", None),
              "checkpoint_selection": checkpoint_selection,
              "allow_experimental_perception": allow_experimental_perception,
              "claim": "Offline experiment only. No robot task success measured."}
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Install compatible CUDA-enabled PyTorch or use --device cpu")
    model = ObjectDynamicsPolicy(**model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config)
    write_json(output / "splits.json", splits)
    np.savez(output / "stats.npz", **stats)
    write_json(output / "data_summary.json", {"train_windows": len(tr["x"]), "val_windows": len(va["x"]),
                                             "test_accessed": False, "train_episodes": splits["train"],
                                             "val_episodes": splits["val"]})
    best = float("inf")
    with (output / "train.jsonl").open("w", encoding="utf-8") as log:
        for epoch in range(1, epochs + 1):
            model.train()
            totals = np.zeros(3)
            n_batches = 0
            started = time.monotonic()
            for indices in np.array_split(rng.permutation(len(tr["x"])), max(1, int(np.ceil(len(tr["x"]) / batch_size)))):
                x, a, target, mask = [torch.from_numpy(tr[k][indices]).to(device) for k in ("x", "a", "y", "mask")]
                h, _ = model.encode(x)
                noise, tau = torch.randn_like(a), torch.rand((len(a), 1), device=device)
                flow = model.velocity(h, (1 - tau) * noise + tau * a, tau)
                action_loss = (flow - (a - noise)).square().mean()
                predicted = model.dynamics(h, a, no_dynamics_actions)
                dynamics_loss = masked_mse(predicted, target, mask)
                loss = action_loss + dynamics_weight * dynamics_loss
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite loss; inspect units, inputs, and learning rate")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                totals += [loss.item(), action_loss.item(), dynamics_loss.item()]
                n_batches += 1
            metrics, _ = evaluate_arrays(model, va, device, config, stats)
            val_score = metrics["flow_mse"] + (dynamics_weight * metrics["dynamics"]["normalized_mse"] if checkpoint_selection == "combined" else 0)
            row = {"epoch": epoch, "train_loss": float(totals[0] / n_batches),
                   "train_flow_mse": float(totals[1] / n_batches),
                   "train_dynamics_mse": float(totals[2] / n_batches),
                   "val_flow_mse": metrics["flow_mse"],
                   "val_dynamics_mse": metrics["dynamics"]["normalized_mse"],
                   "val_persistence_mse": metrics["persistence"]["normalized_mse"],
                   "val_constant_velocity_mse": metrics["constant_velocity"]["normalized_mse"],
                   "seconds": round(time.monotonic() - started, 3)}
            log.write(json.dumps(row, allow_nan=False) + "\n")
            log.flush()
            print(json.dumps(row, allow_nan=False), flush=True)
            if val_score < best:
                best = val_score
                torch.save(model.state_dict(), output / "weights.pt")
                write_json(output / "best_validation.json", {"epoch": epoch, "selection_objective": val_score, "metrics": metrics})
    return config


def load_run(run, device="cpu"):
    torch = get_torch()
    from .model import ObjectDynamicsPolicy
    run = Path(run)
    config = read_json(run / "config.json")
    with np.load(run / "stats.npz", allow_pickle=False) as z:
        stats = {k: z[k] for k in z.files}
    model = ObjectDynamicsPolicy(**config["model"]).to(device)
    model.load_state_dict(torch.load(run / "weights.pt", map_location=device, weights_only=True))
    model.eval()
    return model, config, stats


def evaluate(dataset, run, split="test", device="cpu", output=None):
    run = Path(run)
    model, config, stats = load_run(run, device)
    splits = read_json(run / "splits.json")
    validate_splits(dataset, splits)
    if config["dataset_fingerprint"] != splits["fingerprint"]:
        raise ValueError("Checkpoint/split fingerprint mismatch")
    data = build_windows(dataset, splits[split], config["history"], config["horizon"], stats,
                         config["no_objects"], config["max_per_episode"])
    metrics, pred = evaluate_arrays(model, data, device, config, stats, sample_actions=True)
    metrics.update(split=split, synthetic=config["synthetic"],
                   dataset_fingerprint=splits["fingerprint"],
                   metric_names=config["metric_names"], metric_units=config["metric_units"],
                   limitations=["Offline teacher-forced dynamics uses recorded future actions.",
                                "Sampled action error is not closed-loop task success.",
                                "Episode-disjoint does not imply unseen-task or unseen-scene generalization."])
    dest = Path(output) if output else run / (split + "_metrics.json")
    write_json(dest, metrics)
    # One inspectable example with native-unit deltas and its observation index.
    yshape = (config["horizon"], config["objects"], config["features"])
    scale = stats["feature_std"][None, None, :]
    trace = {"sample": data["refs"][0], "metric_names": config["metric_names"],
             "predicted_delta": (pred[0].reshape(yshape) * scale).tolist(),
             "observed_delta": (data["y"][0].reshape(yshape) * scale).tolist(),
             "valid": data["mask"][0].reshape(yshape).astype(bool).tolist(),
             "recorded_actions": (data["a"][0].reshape(config["horizon"], -1) * stats["actions_std"] + stats["actions_mean"]).tolist()}
    write_json(dest.with_name(dest.stem + "_sample.json"), trace)
    return metrics
