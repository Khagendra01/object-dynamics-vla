"""Validated, training-only multi-demonstration episode views."""
from pathlib import Path

import numpy as np

from odvla.data import file_hash, load_episode, read_json, safe_path, validate_splits


def pad_object_slots(features, valid, slots=2):
    """Pad zero, one or two tracked objects; never silently drop extras."""
    features, valid = np.asarray(features), np.asarray(valid)
    if (features.ndim != 3 or features.shape[2] != 5 or valid.shape != features.shape
            or valid.dtype != np.bool_ or not 0 <= features.shape[1] <= slots):
        raise ValueError("Object measurements must be [T,O,5] bool-masked with O within the adapter's slot limit")
    result = np.zeros((features.shape[0], slots, 5), dtype=features.dtype)
    mask = np.zeros(result.shape, dtype=bool)
    result[:, :features.shape[1]] = features
    mask[:, :features.shape[1]] = valid
    return result, mask


def load_multidemo(data_root, demos_per_task=3, expected_tasks=None, retain_milk=True,
                   prepared_dir="libero/ten_task_multidemo_v1", registry_dir="libero/ten_task_sources"):
    """Load aligned/SAM arrays, retaining all original milk train episodes.

    Images remain on disk. Evaluation and development episodes are not loaded.
    The unverified chocolate-pudding target is explicitly marked missing.
    """
    root = Path(data_root)
    base = safe_path(root, prepared_dir)
    registry_root = safe_path(root, registry_dir)
    renders = read_json(base / "renders.json")
    sam = read_json(base / "sam_extract.json")
    registry = read_json(registry_root / "registry.json")
    audit = read_json(registry_root / "controller_audit.json")
    if (renders.get("status") != "complete" or renders.get("demos_per_new_task") != demos_per_task
            or sam.get("status") != "complete" or registry.get("status") != "sources_prepared"
            or audit.get("status") != "passed"):
        raise ValueError("Prepared renders, SAM, registry or controller audit is incomplete")
    tasks = {item["task_id"]: item for item in registry["tasks"]}
    if (not tasks or len(tasks) != len(registry["tasks"]) or audit.get("task_count") != len(tasks)
            or (expected_tasks is not None and len(tasks) != expected_tasks)):
        raise ValueError("Task registry/controller-audit count mismatch")
    sam_entries = {(r["task_id"], r["episode"]): r for r in sam["entries"]}
    if len(sam_entries) != len(sam["entries"]) or len(sam_entries) != len(renders["entries"]):
        raise ValueError("SAM and render episode identities differ")
    episodes = {task: {} for task in tasks}
    lengths = {task: {} for task in tasks}
    provenance = {task: [] for task in tasks}
    for record in renders["entries"]:
        task, episode = record["task_id"], record["episode"]
        if (task not in tasks or episode in episodes[task] or record["split"] != "train"
                or episode not in tasks[task]["episodes"]["train"]
                or record["instruction"] != tasks[task]["instruction"]
                or record["source_sha256"] != tasks[task]["source_sha256"]):
            raise ValueError(f"Untrusted, duplicate or non-training episode: {task}/{episode}")
        if (task, episode) not in sam_entries:
            raise ValueError(f"Missing SAM measurements: {task}/{episode}")
        folder = safe_path(root, record["folder"])
        sam_path = safe_path(root, sam_entries[(task, episode)]["sam_path"])
        if sam_path.parent != folder:
            raise ValueError(f"SAM path is not scoped to the episode: {task}/{episode}")
        with np.load(folder / "state_actions.npz", allow_pickle=False) as saved:
            episode_data = {k: saved[k].copy() for k in saved.files}
        with np.load(sam_path, allow_pickle=False) as saved:
            features, valid = pad_object_slots(saved["features"], saved["valid"])
            episode_data.update(features=features, valid=valid)
        count = len(episode_data["timestamps"])
        if (count != record["frames"] or episode_data["robot"].shape != (count, 15)
                or episode_data["actions"].shape != (count, 7) or episode_data["features"].shape != (count, 2, 5)
                or episode_data["valid"].shape != (count, 2, 5) or episode_data["valid"].dtype != np.bool_
                or not np.isfinite(episode_data["robot"]).all() or not np.isfinite(episode_data["actions"]).all()
                or not np.isfinite(episode_data["features"][episode_data["valid"]]).all()
                or not np.isfinite(episode_data["timestamps"]).all()
                or not np.all(np.diff(episode_data["timestamps"]) > 0)):
            raise ValueError(f"Invalid aligned training arrays: {task}/{episode}")
        if task == "libero_object_chocolate_pudding":
            episode_data["valid"][:, 0, :] = False
        episodes[task][episode] = {"episode": episode, "instruction": record["instruction"],
                                   "data": episode_data, "rgb_dir": folder / "rgb"}
        lengths[task][episode] = count
        provenance[task].append({"episode": episode, "frames": count, "source_sha256": record["source_sha256"],
            "rgb_sequence_sha256": record["rgb_sequence_sha256"], "state_actions_sha256": file_hash(folder / "state_actions.npz"),
            "sam_sha256": file_hash(sam_path), "sam_target_masked_unverified": task == "libero_object_chocolate_pudding"})
    for task in tasks:
        expected = 1 if retain_milk and task == "libero_object_milk" else demos_per_task
        if len(episodes[task]) != expected:
            raise ValueError(f"Expected {expected} prepared episodes for {task}")
    if not retain_milk:
        return episodes, lengths, provenance
    if "libero_object_milk" not in tasks:
        raise ValueError("Milk retention was requested but milk is absent from the registry")
    milk_root = root / "libero/muse_aligned_development_v1"
    splits = read_json(milk_root / "splits.json")
    manifest = validate_splits(milk_root, splits)
    entries = {entry["id"]: entry for entry in manifest["episodes"]}
    for episode_id in splits["train"]:
        entry = entries[episode_id]
        episode = entry["source_demo"]
        if episode in episodes["libero_object_milk"]:
            continue
        ep = load_episode(milk_root, entry, manifest, include_images=False)
        episodes["libero_object_milk"][episode] = {"episode": episode, "instruction": entry["instruction"],
            "data": ep, "rgb_dir": root / "libero/milk_sam_native_v1" / episode_id}
        lengths["libero_object_milk"][episode] = len(ep["timestamps"])
        provenance["libero_object_milk"].append({"episode": episode, "frames": len(ep["timestamps"]),
                                                   "episode_sha256": file_hash(milk_root / entry["file"])})
    if set(episodes["libero_object_milk"]) != set(tasks["libero_object_milk"]["episodes"]["train"]):
        raise ValueError("Retained milk episodes do not match its audited training split")
    return episodes, lengths, provenance
