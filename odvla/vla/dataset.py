"""Causal window contract for padded LIBERO observations and action chunks."""
import numpy as np
from pathlib import Path
import hashlib

from ..data import file_hash, load_episode, load_manifest, read_json, validate_splits


def observation_window(ep, t, stats, history=4, horizon=50):
    """Build a window ending at t; padding is explicitly marked and never a target."""
    n = len(ep["timestamps"])
    if not 0 <= t < n or history < 1 or horizon < 1:
        raise ValueError("Invalid observation index, history or horizon")
    indices = np.arange(t - history + 1, t + 1)
    present = indices >= 0
    indices = np.maximum(indices, 0)
    raw = ep["features"][indices]
    valid = ep["valid"][indices] & present[:, None, None]
    values = np.where(valid, (raw - stats["feature_mean"]) / stats["feature_std"], 0).astype(np.float32)
    delta = np.zeros_like(values)
    pair = valid[1:] & valid[:-1]
    delta[1:] = np.where(pair, (raw[1:] - raw[:-1]) / stats["feature_std"], 0)
    age = np.full_like(raw, 10, dtype=np.float32)
    seen = np.full(raw.shape[1:], -np.inf)
    for index in range(t + 1):
        seen = np.where(ep["valid"][index], ep["timestamps"][index], seen)
        if index >= t - history + 1:
            pos = index - (t - history + 1)
            age[pos] = np.minimum(ep["timestamps"][index] - seen, 10)
    age = np.where(present[:, None, None], age, 0)
    robot = ((ep["robot"][indices] - stats["robot_mean"]) / stats["robot_std"]).astype(np.float32)
    actions = np.zeros((horizon, ep["actions"].shape[1]), np.float32)
    count = min(horizon, n - t)
    actions[:count] = (ep["actions"][t:t + count] - stats["actions_mean"]) / stats["actions_std"]
    action_valid = np.arange(horizon) < count
    change = np.zeros_like(ep["features"][t], dtype=np.float32)
    change_valid = np.zeros_like(ep["valid"][t])
    if t + 1 < n:
        change_valid = ep["valid"][t] & ep["valid"][t + 1]
        change = np.where(change_valid, (ep["features"][t + 1] - ep["features"][t]) / stats["feature_std"], 0).astype(np.float32)
    return {"robot": robot, "history_valid": present, "history_timestamps": ep["timestamps"][indices].copy(),
            "sam_values": values, "sam_deltas": delta, "sam_valid": valid,
            "sam_age": age, "actions": actions, "action_valid": action_valid,
            "change": change, "change_valid": change_valid, "timestamp": float(ep["timestamps"][t])}


class CachedEpisodeDataset:
    """Development split adapter. Cached features are checked against frame hashes."""
    def __init__(self, dataset, cache, split, stats, history=4, horizon=50):
        self.root, self.cache = Path(dataset), Path(cache)
        self.splits = read_json(self.root / "splits.json")
        self.manifest = validate_splits(self.root, self.splits)
        provenance = read_json(self.cache / "manifest.json")
        if not provenance.get("complete") or provenance.get("dataset_fingerprint") != self.splits["fingerprint"]:
            raise ValueError("Missing or mismatched complete Muse cache")
        if split not in ("train", "val"):
            raise ValueError("Only train and validation are supported; no untouched test bank")
        self.entries = {e["id"]: e for e in self.manifest["episodes"] if e["id"] in self.splits[split]}
        self.episodes = {name: load_episode(self.root, entry, self.manifest, include_images=False)
                         for name, entry in self.entries.items()}
        self.index = [(name, t) for name, episode in self.episodes.items() for t in range(len(episode["timestamps"]))]
        self.stats, self.history, self.horizon = stats, history, horizon
        self.cache_manifests = {name: read_json(self.cache / name / "manifest.json") for name in self.entries}
        for name, record in self.cache_manifests.items():
            instruction_hash = hashlib.sha256(self.entries[name]["instruction"].encode()).hexdigest()
            if record["episode"] != name or record["instruction_sha256"] != instruction_hash or record["frames"] != len(self.episodes[name]["timestamps"]):
                raise ValueError("Cached episode/instruction mismatch")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        name, t = self.index[index]
        path = self.cache / name / f"{t:06}.npz"
        record = self.cache_manifests[name]["records"][t]
        if record["frame"] != t or abs(record["timestamp"] - self.episodes[name]["timestamps"][t]) > 1e-6:
            raise ValueError("Cached token frame/timestamp mismatch")
        if file_hash(path) != record["sha256"]:
            raise ValueError("Cached token checksum mismatch")
        with np.load(path, allow_pickle=False) as z:
            features = z["features_bfloat16_bits"].copy()
            token_mask = z["attention_mask"].copy()
            ids, positions = z["input_ids"].copy(), z["positions"].copy()
        if features.dtype != np.uint16 or features.ndim != 2 or len(token_mask) != len(features):
            raise ValueError("Invalid BF16 token cache")
        sample = observation_window(self.episodes[name], t, self.stats, self.history, self.horizon)
        sample.update(episode=name, t=t, instruction=self.entries[name]["instruction"],
                      muse_bits=features, muse_valid=token_mask, input_ids=ids, token_positions=positions)
        return sample
