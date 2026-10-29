"""Transparent action and one-step prediction diagnostics for offline rollouts."""
import numpy as np


def summarize_actions(values):
    """Channel-wise empirical summaries; values are [steps, channels]."""
    a = np.asarray(values, dtype=np.float64)
    if a.ndim != 2 or not len(a) or not np.isfinite(a).all():
        raise ValueError("Expected a nonempty finite [steps, channels] array")
    return {"count": len(a), "mean": a.mean(0).tolist(), "std": a.std(0).tolist(),
            "min": a.min(0).tolist(), "max": a.max(0).tolist(),
            "p01": np.quantile(a, .01, axis=0).tolist(),
            "p50": np.quantile(a, .5, axis=0).tolist(),
            "p99": np.quantile(a, .99, axis=0).tolist()}


def accepted_chunk_forecast(prefix, actions, model, config, stats, device="cpu"):
    """Historical chunk-conditioned diagnostic, unsuitable for one-step monitoring.

    The head expects the whole future action chunk; only its first element is
    actually executed before the observation. Later actions are hypothetical.
    """
    import torch
    from .data import encode_history
    chunk = np.asarray(actions, dtype=np.float32)
    expected = (config["horizon"], config["action_channels"])
    if chunk.shape != expected or not np.isfinite(chunk).all():
        raise ValueError(f"Expected finite action chunk {expected}")
    normalized = (chunk - stats["actions_mean"]) / stats["actions_std"]
    x = encode_history(prefix, len(prefix["timestamps"]) - 1, config["history"],
                       stats, config["no_objects"])
    with torch.no_grad():
        h, _ = model.encode(torch.from_numpy(x[None]).to(device))
        pred = model.dynamics(h, torch.from_numpy(normalized.reshape(1, -1)).to(device),
                              config["no_dynamics_actions"])
    delta = pred.cpu().numpy().reshape(config["horizon"], config["objects"], config["features"])
    return delta[0] * stats["feature_std"][None, :]


def one_step_forecast(prefix, executed_action, encoder, head, config, stats, device="cpu"):
    """Forecast a one-step delta from history and exactly one executed action."""
    import torch
    from .data import encode_history
    action = np.asarray(executed_action, dtype=np.float32)
    if action.shape != (config["action_channels"],) or not np.isfinite(action).all():
        raise ValueError("Expected exactly one finite executed action")
    normalized = (action - stats["actions_mean"]) / stats["actions_std"]
    x = encode_history(prefix, len(prefix["timestamps"]) - 1, config["history"],
                       stats, config["no_objects"])
    with torch.no_grad():
        h, _ = encoder.encode(torch.from_numpy(x[None]).to(device))
        delta = head(h, torch.from_numpy(normalized[None].astype(np.float32)).to(device))
    return delta.cpu().numpy().reshape(config["objects"], config["features"]) * stats["feature_std"][None, :]
