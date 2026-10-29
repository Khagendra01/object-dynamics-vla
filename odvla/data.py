"""Canonical episode IO, group splits, train-only statistics, causal windows."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def safe_path(root, relative):
    root = Path(root).resolve()
    p = (root / relative).resolve()
    if not p.is_relative_to(root):
        raise ValueError(f"Path escapes dataset directory: {relative}")
    return p


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(root):
    m = read_json(Path(root) / "manifest.json")
    if m.get("schema_version") != 1 or not m.get("episodes"):
        raise ValueError("Expected schema_version=1 and nonempty episodes")
    for key in ("metric_names", "metric_units", "object_slots", "action_convention"):
        if not m.get(key):
            raise ValueError(f"Missing manifest field: {key}")
    if len(m["metric_names"]) != len(m["metric_units"]):
        raise ValueError("metric_names and metric_units must have equal lengths")
    for key in ("metric_names", "object_slots"):
        if len(set(m[key])) != len(m[key]):
            raise ValueError(f"Duplicate {key}")
    ids = [e["id"] for e in m["episodes"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate episode IDs")
    return m


def load_episode(root, entry, manifest=None, include_images=True):
    manifest = manifest or load_manifest(root)
    with np.load(safe_path(root, entry["file"]), allow_pickle=False) as z:
        ep = {k: z[k] for k in z.files if include_images or k not in ("rgb", "depth", "masks")}
    required = ("features", "valid", "robot", "actions", "timestamps")
    if not all(k in ep for k in required):
        raise ValueError(f"{entry['id']}: required arrays are {required}")
    x, valid = ep["features"], ep["valid"]
    if x.ndim != 3 or valid.shape != x.shape or valid.dtype != np.bool_:
        raise ValueError("features must be [T,O,F]; valid must be same-shaped bool")
    t, o, f = x.shape
    if t < 3 or o != len(manifest["object_slots"]) or f != len(manifest["metric_names"]):
        raise ValueError("Episode dimensions disagree with manifest, or T<3")
    if not np.isfinite(x[valid]).all():
        raise ValueError("Observed feature values must be finite")
    for key in ("robot", "actions"):
        if ep[key].ndim != 2 or ep[key].shape[0] != t or not np.isfinite(ep[key]).all():
            raise ValueError(f"{key} must be a finite [T,D] array")
    times = ep["timestamps"]
    if times.shape != (t,) or not np.isfinite(times).all() or not (np.diff(times) > 0).all():
        raise ValueError("timestamps must be finite, strictly increasing seconds")
    if "context" not in ep:
        ep["context"] = np.zeros((t, 0), dtype=np.float32)
    if ep["context"].ndim != 2 or ep["context"].shape[0] != t or not np.isfinite(ep["context"]).all():
        raise ValueError("context must be finite [T,C]")
    if "rgb" in ep:
        rgb = ep["rgb"]
        if rgb.ndim != 4 or rgb.shape[0] != t or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
            raise ValueError("rgb must be uint8 [T,H,W,3]")
    return ep


def fingerprint(root, manifest=None):
    m = manifest or load_manifest(root)
    digest = hashlib.sha256(json.dumps(m, sort_keys=True).encode())
    for e in sorted(m["episodes"], key=lambda v: v["id"]):
        digest.update(e["id"].encode())
        digest.update(file_hash(safe_path(root, e["file"])).encode())
    return digest.hexdigest()


def make_splits(root, seed=17):
    m = load_manifest(root)
    groups = sorted({e.get("group", e["id"]) for e in m["episodes"]})
    if len(groups) < 5:
        raise ValueError("Need at least five independent episode groups for train/val/test")
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    n_val = max(1, int(round(len(groups) * 0.15)))
    n_test = n_val
    assignment = {g: "train" for g in groups}
    assignment.update({g: "val" for g in groups[:n_val]})
    assignment.update({g: "test" for g in groups[n_val:n_val + n_test]})
    result = {k: [e["id"] for e in m["episodes"] if assignment[e.get("group", e["id"])] == k]
              for k in ("train", "val", "test")}
    result.update(schema_version=1, seed=seed, fingerprint=fingerprint(root, m))
    return result


def validate_splits(root, splits):
    m = load_manifest(root)
    if splits["fingerprint"] != fingerprint(root, m):
        raise ValueError("Dataset changed after splitting. Regenerate splits and retrain.")
    all_ids = set()
    all_groups = set()
    all_payloads = set()
    by_id = {e["id"]: e for e in m["episodes"]}
    for split in ("train", "val", "test"):
        ids = splits[split]
        if not ids or len(ids) != len(set(ids)) or all_ids.intersection(ids):
            raise ValueError("Empty, duplicated, or overlapping episode split")
        if not set(ids) <= by_id.keys():
            raise ValueError("Unknown episode ID in split")
        groups = {by_id[i].get("group", i) for i in ids}
        hashes = {file_hash(safe_path(root, by_id[i]["file"])) for i in ids}
        if all_groups.intersection(groups) or all_payloads.intersection(hashes):
            raise ValueError("Related episode groups or identical files cross splits")
        all_ids.update(ids)
        all_groups.update(groups)
        all_payloads.update(hashes)
    if all_ids != set(by_id):
        raise ValueError("Every episode must belong to one split")
    return m


def _moments(values, valid=None):
    values = np.concatenate(values, axis=0).astype(np.float64)
    mask = np.ones_like(values, dtype=bool) if valid is None else np.concatenate(valid, axis=0)
    count = mask.sum(axis=0)
    mean = np.where(mask, values, 0).sum(axis=0) / np.maximum(count, 1)
    variance = np.where(mask, (values - mean) ** 2, 0).sum(axis=0) / np.maximum(count, 1)
    std = np.maximum(np.sqrt(variance), 1e-3)
    return mean.astype(np.float32), std.astype(np.float32)


def fit_stats(root, splits):
    m = load_manifest(root)
    selected = [load_episode(root, e, m, include_images=False) for e in m["episodes"] if e["id"] in splits["train"]]
    stat = {}
    f = len(m["metric_names"])
    stat["feature_mean"], stat["feature_std"] = _moments(
        [e["features"].reshape(-1, f) for e in selected],
        [e["valid"].reshape(-1, f) for e in selected])
    for key in ("robot", "actions", "context"):
        stat[key + "_mean"], stat[key + "_std"] = _moments([e[key] for e in selected])
    return stat


def encode_history(ep, t, history, stats, no_objects=False):
    """Encode observations only; works with a prefix ending at t and no action labels."""
    if t < history - 1 or t >= len(ep["timestamps"]):
        raise ValueError("Insufficient observation history")
    sl = slice(t - history + 1, t + 1)
    values, valid = ep["features"][sl], ep["valid"][sl]
    z = np.where(valid, (values - stats["feature_mean"]) / stats["feature_std"], 0)
    delta = np.zeros_like(z)
    delta[1:] = np.where(valid[1:] & valid[:-1], z[1:] - z[:-1], 0)
    # Prefix-only age: no backward filling and no future visibility.
    age = np.full_like(values, 10.0)
    last_seen = np.full(values.shape[1:], -np.inf)
    for i in range(t + 1):
        last_seen = np.where(ep["valid"][i], ep["timestamps"][i], last_seen)
        if i >= t - history + 1:
            age[i - (t - history + 1)] = np.minimum(ep["timestamps"][i] - last_seen, 10)
    obj = np.concatenate([z.ravel(), valid.astype(np.float32).ravel(), delta.ravel(), np.log1p(age).ravel()])
    if no_objects:
        obj = np.zeros_like(obj)
    robot = (ep["robot"][sl] - stats["robot_mean"]) / stats["robot_std"]
    context = (ep["context"][t] - stats["context_mean"]) / stats["context_std"]
    x = np.concatenate([obj, robot.ravel(), np.diff(ep["timestamps"][sl]), context]).astype(np.float32)
    return x


def sample_window(ep, t, history, horizon, stats, no_objects=False):
    """Input ends at state t; a[t] acts on state t; targets are states t+1..t+H."""
    if t < history - 1 or t + horizon >= len(ep["timestamps"]):
        raise ValueError("Window lacks full observation history or future targets")
    x = encode_history(ep, t, history, stats, no_objects)
    a = (ep["actions"][t:t + horizon] - stats["actions_mean"]) / stats["actions_std"]
    fut = ep["features"][t + 1:t + horizon + 1]
    mask = ep["valid"][t + 1:t + horizon + 1] & ep["valid"][t]
    y = np.where(mask, (fut - ep["features"][t]) / stats["feature_std"], 0)
    # Last-two-observation velocity extrapolation in normalized feature units.
    cv = np.zeros_like(y)
    if t > 0:
        pair = ep["valid"][t] & ep["valid"][t - 1]
        velocity = np.where(pair, (ep["features"][t] - ep["features"][t - 1]) / stats["feature_std"], 0)
        dt = ep["timestamps"][t] - ep["timestamps"][t - 1]
        elapsed = ep["timestamps"][t + 1:t + horizon + 1] - ep["timestamps"][t]
        cv = elapsed[:, None, None] / dt * velocity[None]
    return {"x": x, "a": a.ravel().astype(np.float32), "y": y.ravel().astype(np.float32),
            "mask": mask.ravel().astype(np.float32), "cv": cv.ravel().astype(np.float32)}


def build_windows(root, ids, history, horizon, stats, no_objects=False, max_per_episode=0):
    m = load_manifest(root)
    samples, refs = [], []
    for e in m["episodes"]:
        if e["id"] not in ids:
            continue
        ep = load_episode(root, e, m, include_images=False)
        starts = list(range(history - 1, len(ep["timestamps"]) - horizon))
        if max_per_episode and len(starts) > max_per_episode:
            starts = [starts[i] for i in np.linspace(0, len(starts) - 1, max_per_episode, dtype=int)]
        for t in starts:
            samples.append(sample_window(ep, t, history, horizon, stats, no_objects))
            refs.append({"episode": e["id"], "t": t})
    if not samples:
        raise ValueError("No complete windows. Reduce history/horizon or check episode length.")
    result = {key: np.stack([s[key] for s in samples]) for key in samples[0]}
    if result["mask"].sum() == 0:
        raise ValueError("No observed numerical targets. Extract/attach masks before training.")
    result["refs"] = refs
    return result
