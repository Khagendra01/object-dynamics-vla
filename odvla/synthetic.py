"""Synthetic integration fixture, deliberately unrelated to real robotics performance."""
from pathlib import Path
import numpy as np
from .data import write_json
from .measurements import mask_metrics, METRIC_NAMES, METRIC_UNITS


def generate(root, episodes=30, steps=64, seed=123):
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty directory: {root}")
    if episodes < 5 or steps < 12:
        raise ValueError("Use >=5 episodes and >=12 steps")
    (root / "episodes").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    entries = []
    for i in range(episodes):
        masks = np.zeros((steps, 2, 64, 64), dtype=bool)
        rgb = np.full((steps, 64, 64, 3), 225, dtype=np.uint8)
        action = rng.uniform(-1, 1, (steps, 2)).astype(np.float32)
        # Variable action blocks: next movements cannot be perfectly inferred from history.
        for j in range(0, steps, 4):
            action[j:j + 4] = action[j]
        pos = rng.uniform(15, 25, 2)
        robot = np.zeros((steps, 3), np.float32)
        for t in range(steps):
            if t:
                pos = np.clip(pos + 2.0 * action[t - 1], 6, 56)
            x, y = np.rint(pos).astype(int)
            masks[t, 0, y - 3:y + 3, x - 3:x + 3] = True
            masks[t, 1, 43:55, 43:55] = True
            # Occasional missing observation; no future fill-in.
            if t % 23 == 15:
                masks[t, 0] = False
            rgb[t, masks[t, 0]] = [190, 65, 65]
            rgb[t, masks[t, 1]] = [70, 110, 190]
            robot[t] = [0.25, 0.30, 1.0]  # Deliberately no hidden copy of object position.
        features, valid = mask_metrics(masks)
        name = f"synthetic_{i:04d}"
        np.savez_compressed(root / "episodes" / (name + ".npz"),
                            features=features, valid=valid, robot=robot, actions=action,
                            timestamps=np.arange(steps, dtype=np.float64) / 20, rgb=rgb)
        entries.append({"id": name, "file": f"episodes/{name}.npz", "group": name,
                        "instruction": "Move the red square toward the blue region.",
                        "source": "synthetic_fixture_not_robot_demonstrations"})
    manifest = {"schema_version": 1, "synthetic": True, "metric_names": METRIC_NAMES,
                "metric_units": METRIC_UNITS, "object_slots": ["red square", "blue destination"],
                "action_convention": "Fixture commands: signed 2D image motion, not robot meters",
                "perception": {"source": "exact synthetic masks", "causal": True}, "episodes": entries}
    write_json(root / "manifest.json", manifest)
    return manifest
