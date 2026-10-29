"""Existing-dataset audit/import and external mask attachment."""
from __future__ import annotations
import copy
from pathlib import Path
import numpy as np
from .data import load_manifest, load_episode, read_json, safe_path, write_json
from .measurements import METRIC_NAMES, METRIC_UNITS, mask_metrics


def require_h5py():
    try:
        import h5py
        return h5py
    except ImportError as exc:
        raise RuntimeError("Install dataset support: pip install -e '.[datasets]'") from exc


def audit(path):
    path = Path(path)
    if path.is_dir() and (path / "manifest.json").exists():
        manifest = load_manifest(path)
        result = {"type": "canonical", "synthetic": manifest.get("synthetic", False),
                  "metrics": manifest["metric_names"], "objects": manifest["object_slots"],
                  "perception": manifest.get("perception", {}), "episodes": []}
        for entry in manifest["episodes"]:
            ep = load_episode(path, entry, manifest)
            result["episodes"].append({"id": entry["id"], "group": entry.get("group", entry["id"]),
                                       "frames": len(ep["timestamps"]),
                                       "observed_fraction": float(ep["valid"].mean()),
                                       "shapes": {k: list(v.shape) for k, v in ep.items()},
                                       "median_interval_s": float(np.median(np.diff(ep["timestamps"])))})
        return result
    if path.suffix.lower() in (".h5", ".hdf5"):
        h5py = require_h5py()
        result = {"type": "HDF5", "file": path.name, "datasets": {}, "attributes": {}}
        with h5py.File(path, "r") as f:
            def visitor(name, value):
                if isinstance(value, h5py.Dataset):
                    result["datasets"][name] = {"shape": list(value.shape), "dtype": str(value.dtype)}
            f.visititems(visitor)
            for k, v in f.attrs.items():
                result["attributes"][k] = str(v)[:2000]
        return result
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as z:
            return {"type": "NPZ", "arrays": {k: {"shape": list(z[k].shape), "dtype": str(z[k].dtype)} for k in z.files}}
    raise ValueError("Audit a canonical directory, HDF5 demonstration file, or NPZ")


def import_libero(source, output, objects, fps, instruction, robot_keys, image_key="agentview_rgb", flip_vertical=False):
    """Conservative import. Explicit instruction/FPS avoids guessing metadata or control timing."""
    h5py = require_h5py()
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing nonempty output: {output}")
    if fps <= 0 or not instruction.strip():
        raise ValueError("Positive --fps and an explicit --instruction are required")
    (output / "episodes").mkdir(parents=True, exist_ok=True)
    entries = []
    with h5py.File(source, "r") as f:
        if "data" not in f:
            raise ValueError("Expected LIBERO/robomimic-style /data/demo_*/obs and actions")
        for name in sorted(f["data"].keys()):
            demo = f["data"][name]
            if not isinstance(demo, h5py.Group) or "actions" not in demo or "obs" not in demo:
                continue
            obs = demo["obs"]
            missing = [k for k in [image_key, *robot_keys] if k not in obs]
            if missing:
                raise ValueError(f"{name}: missing {missing}; run audit and set --robot-keys/--image-key")
            actions = np.asarray(demo["actions"], dtype=np.float32)
            t = len(actions)
            rgb = np.asarray(obs[image_key])
            robot = np.concatenate([np.asarray(obs[k]).reshape(t, -1) for k in robot_keys], axis=1).astype(np.float32)
            if len(rgb) != t:
                raise ValueError("Image/action lengths differ. Inspect alignment instead of silently truncating.")
            if flip_vertical:
                rgb = rgb[:, ::-1].copy()
            ep_id = f"episode_{len(entries):04d}"
            np.savez_compressed(output / "episodes" / (ep_id + ".npz"),
                                features=np.zeros((t, len(objects), 5), np.float32),
                                valid=np.zeros((t, len(objects), 5), bool), robot=robot, actions=actions,
                                rgb=rgb, timestamps=np.arange(t, dtype=np.float64) / fps)
            entries.append({"id": ep_id, "file": f"episodes/{ep_id}.npz", "group": ep_id,
                            "instruction": instruction, "source_demo": name})
    if not entries:
        raise ValueError("No demonstrations found")
    manifest = {"schema_version": 1, "synthetic": False, "metric_names": METRIC_NAMES,
                "metric_units": METRIC_UNITS, "object_slots": objects,
                "action_convention": "Copied source commands unchanged. Verify source controller units/frame before rollouts.",
                "import": {"source_name": Path(source).name, "fps_user_supplied": fps,
                           "robot_keys": robot_keys, "image_key": image_key,
                           "flip_vertical": flip_vertical,
                           "alignment_assumption": "obs[t] precedes executed actions[t]; verify against source/replay"},
                "perception": {"source": "not extracted", "causal": False}, "episodes": entries}
    write_json(output / "manifest.json", manifest)
    return manifest


def attach_masks(dataset, mask_dir, output):
    m = copy.deepcopy(load_manifest(dataset))
    provenance = read_json(Path(mask_dir) / "provenance.json")
    if provenance.get("causal") is not True:
        raise ValueError("Mask provenance must explicitly declare causal=true. Audit prefix consistency first.")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new output dataset directory")
    (output / "episodes").mkdir(parents=True, exist_ok=True)
    for entry in m["episodes"]:
        ep = load_episode(dataset, entry, m)
        with np.load(Path(mask_dir) / (entry["id"] + ".npz"), allow_pickle=False) as z:
            masks = z["masks"]
        if masks.shape[:2] != ep["features"].shape[:2]:
            raise ValueError(f"{entry['id']}: masks must match all episode frames and ordered slots")
        if "rgb" in ep and masks.shape[2:] != ep["rgb"].shape[1:3]:
            raise ValueError("Masks must use the image's original resolution")
        ep["features"], ep["valid"] = mask_metrics(masks)
        destination = safe_path(output, entry["file"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination, **ep)
    m["metric_names"], m["metric_units"] = METRIC_NAMES, METRIC_UNITS
    m["perception"] = provenance
    write_json(output / "manifest.json", m)
    return m
