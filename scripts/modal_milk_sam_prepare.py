"""Build a shared SAM dataset, then separately score against simulator references."""
import modal
from scripts.modal_odvla import image as base_image, volume

image = (base_image.add_local_file("scripts/__init__.py", "/root/scripts/__init__.py")
         .add_local_file("scripts/modal_odvla.py", "/root/scripts/modal_odvla.py"))
app = modal.App("odvla-milk-sam-prepare")


@app.function(image=image, volumes={"/data": volume}, cpu=4, memory=8192, timeout=3600)
def prepare():
    import copy
    import hashlib
    import json
    from pathlib import Path
    import numpy as np
    from odvla.data import fingerprint, load_manifest, load_episode, write_json
    from odvla.measurements import mask_metrics
    raw_root = Path("/data/libero/milk_raw")
    cache = Path("/data/libero/milk_sam_masks_v1")
    out = Path("/data/libero/milk_sam_development_v1")
    report = Path("/data/eval/milk_sam_measurement_audit_v1.json")
    if out.exists() or report.exists():
        raise FileExistsError("Use fresh dataset and audit output paths")
    provenance = json.loads((cache / "provenance.json").read_text())
    if not provenance["complete"] or provenance["reference_masks_opened"]:
        raise ValueError("Cache must be complete and independent of references")
    prior = json.loads(Path("/data/libero/milk_oracle/splits.json").read_text())
    selected = set(prior["train"] + prior["val"])
    source_manifest = load_manifest(raw_root)
    manifest = copy.deepcopy(source_manifest)
    manifest["episodes"] = [e for e in manifest["episodes"] if e["id"] in selected]
    if selected != {e["id"] for e in manifest["episodes"]}:
        raise ValueError("Missing selected episode")
    manifest["perception"] = {"source": "SAM3.1 initialized native 512 camera RGB, nearest-neighbor masks on 128 grid",
        "causal": False, "experimental_prefix_audited": True,
        "scope": "offline preloaded video with bounded point-path prefix checks; no general or live causality guarantee",
        "configuration_sha256": provenance["configuration_sha256"], "oracle": False,
        "cached_masks": str(cache), "test_accessed": False}
    (out / "episodes").mkdir(parents=True)
    hashes = {}
    for entry in manifest["episodes"]:
        episode = load_episode(raw_root, entry, source_manifest)
        with np.load(cache / f"{entry['id']}.npz", allow_pickle=False) as z:
            masks, timestamps, valid, features = z["masks"], z["timestamps"], z["valid"], z["features"]
            if not np.array_equal(timestamps, episode["timestamps"]):
                raise ValueError("Cache timestamps do not match raw data")
            check_features, check_valid = mask_metrics(masks)
            if not np.array_equal(valid, check_valid) or not np.allclose(features, check_features):
                raise ValueError("Cached metrics do not agree with masks")
            episode["features"], episode["valid"] = features, valid
        path = out / entry["file"]
        np.savez_compressed(path, **episode)
        hashes[entry["id"]] = hashlib.sha256((cache / f"{entry['id']}.npz").read_bytes()).hexdigest()
    write_json(out / "manifest.json", manifest)
    split = {"train": prior["train"], "val": prior["val"], "test": [], "allow_empty_test": True,
             "schema_version": 1, "seed": prior["seed"], "fingerprint": fingerprint(out, manifest)}
    write_json(out / "splits.json", split)
    write_json(out / "cache_hashes.json", hashes)
    # Separate post-hoc scoring only; no references are written to the dataset.
    measurements = {f"{s}/{o}/{m}": {k: [] for k in ("all", "moving", "stationary")}
                    for s in ("train", "val") for o in ("milk", "basket")
                    for m in manifest["metric_names"]}
    ious = {s: {o: [] for o in ("milk", "basket")} for s in ("train", "val")}
    for entry in manifest["episodes"]:
        name = entry["id"]
        split_name = "train" if name in split["train"] else "val"
        with np.load(cache / f"{name}.npz", allow_pickle=False) as z:
            masks, features, valid = z["masks"], z["features"], z["valid"]
        with np.load(f"/data/libero/milk_masks_oracle/{name}.npz", allow_pickle=False) as z:
            truth = z["masks"].astype(bool)
        reference, ref_valid = mask_metrics(truth)
        paired = valid & ref_valid
        consecutive = paired[1:] & paired[:-1]
        ref_delta = np.diff(reference, axis=0)
        error = np.abs(np.diff(features, axis=0) - ref_delta)
        for i, obj in enumerate(("milk", "basket")):
            intersection = np.logical_and(masks[:, i], truth[:, i]).sum(axis=(1, 2))
            union = np.logical_or(masks[:, i], truth[:, i]).sum(axis=(1, 2))
            ious[split_name][obj].extend((intersection / np.maximum(union, 1)).tolist())
            for j, metric in enumerate(manifest["metric_names"]):
                is_valid = consecutive[:, i, j]
                moving = np.abs(ref_delta[:, i, j]) > 0.002
                target = measurements[f"{split_name}/{obj}/{metric}"]
                for key, selection in (("all", is_valid), ("moving", is_valid & moving), ("stationary", is_valid & ~moving)):
                    target[key].extend(error[:, i, j][selection].tolist())
    summaries = {key: {k: {"count": len(v), "delta_error_mae": float(np.mean(v)) if v else None}
                       for k, v in subsets.items()} for key, subsets in measurements.items()}
    result = {"dataset": str(out), "fingerprint": split["fingerprint"], "episodes": len(manifest["episodes"]),
        "moving_definition": "per-object, per-metric absolute oracle reference change > 0.002 image fraction; scoring only",
        "measurement_delta_error": summaries,
        "mean_iou": {s: {o: float(np.mean(v)) for o, v in objects.items()} for s, objects in ious.items()},
        "reference_masks_used_only_for_scoring": True, "test_episodes_used": False}
    write_json(report, result)
    volume.commit()
    return result


@app.local_entrypoint()
def main():
    print(prepare.remote())
