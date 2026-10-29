"""Offline policy sampling. Robot-specific command validation belongs in a later adapter."""
import numpy as np
from .data import encode_history


def predict(prefix, model, config, stats, device="cpu", seed=0):
    """No future frames/actions required. Prefix contains observations through current t."""
    import torch
    t = len(prefix["timestamps"]) - 1
    x = encode_history(prefix, t, config["history"], stats, config["no_objects"])
    rng = np.random.default_rng(seed)
    with torch.no_grad():
        h, gates = model.encode(torch.from_numpy(x[None]).to(device))
        noise = torch.from_numpy(rng.normal(size=(1, config["model"]["action_dim"])).astype(np.float32)).to(device)
        normalized = model.sample_actions(h, noise, config["flow_steps"])
        change = model.dynamics(h, normalized, config["no_dynamics_actions"])
    actions = normalized.cpu().numpy().reshape(config["horizon"], -1)
    actions = actions * stats["actions_std"] + stats["actions_mean"]
    delta = change.cpu().numpy().reshape(config["horizon"], config["objects"], config["features"])
    delta *= stats["feature_std"][None, None, :]
    return {"actions_source_units": actions.tolist(), "predicted_numeric_delta": delta.tolist(),
            "measurement_gates": gates.cpu().numpy()[0].tolist(),
            "forecast_condition": "sampled action chunk before robot-specific validation",
            "execution_status": "not executed; no robot driver included"}
