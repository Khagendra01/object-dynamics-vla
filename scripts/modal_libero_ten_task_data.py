"""Download and inspect ten pinned LIBERO-Object tasks without GPU use."""
import modal

app = modal.App("odvla-libero-object-ten-task-data")
volume = modal.Volume.from_name("object-dynamics-vla-data", create_if_missing=False)
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("h5py", "numpy<2")
         .add_local_python_source("odvla"))


@app.function(image=image, volumes={"/data": volume}, cpu=2, memory=8192, timeout=300)
def metadata():
    import h5py
    import json
    from pathlib import Path
    from odvla.data import read_json, write_json
    root = Path("/data")
    data = read_json(root / "libero/ten_task_sources/registry.json")
    result, reference = {}, None
    for task in data["tasks"]:
        with h5py.File(root / task["source_path"], "r") as source:
            group = source["data"]
            args = json.loads(group.attrs["env_args"])
            info = json.loads(group.attrs["problem_info"])
            kwargs = args["env_kwargs"]
            contract = {"robots": kwargs["robots"], "controller_configs": kwargs["controller_configs"],
                        "control_freq": kwargs["control_freq"], "camera_names": kwargs["camera_names"],
                        "macros_image_convention": str(group.attrs["macros_image_convention"])}
            if info["language_instruction"].lower() != task["instruction"]:
                raise ValueError(f"Instruction mismatch: {task['task_id']}")
            if reference is not None and contract != reference:
                raise ValueError(f"Controller/observation metadata differs: {task['task_id']}")
            reference = contract
            result[task["task_id"]] = {"instruction": info["language_instruction"], "contract": contract}
    report = {"status": "passed", "tasks": result, "task_count": len(result),
              "scope": "HDF5 declared robot/controller/camera metadata; not action-replay validation"}
    path = root / "libero/ten_task_sources/controller_audit.json"
    write_json(path, report)
    volume.commit()
    return {"status": report["status"], "task_count": report["task_count"], "output": str(path)}


@app.function(image=image, volumes={"/data": volume}, cpu=2, memory=8192, timeout=600)
def audit_initial_states():
    """Check initial-state identity across demo-ID splits without executing rollouts."""
    import hashlib
    import h5py
    import numpy as np
    from pathlib import Path
    from odvla.data import read_json, write_json
    root = Path("/data")
    registry = read_json(root / "libero/ten_task_sources/registry.json")
    tasks = []
    for task in registry["tasks"]:
        with h5py.File(root / task["source_path"], "r") as source:
            hashes = {}
            for split, demos in task["episodes"].items():
                hashes[split] = {hashlib.sha256(np.asarray(source["data"][demo]["states"][0]).tobytes()).hexdigest()
                                 for demo in demos}
        tasks.append({"task_id": task["task_id"], "unique_initial_states": {k: len(v) for k, v in hashes.items()},
            "train_development_overlap": len(hashes["train"] & hashes["development"]),
            "train_evaluation_overlap": len(hashes["train"] & hashes["evaluation"]),
            "development_evaluation_overlap": len(hashes["development"] & hashes["evaluation"])})
    result = {"status": "completed", "scope": "SHA-256 of source simulator states[0] only; no evaluation actions or rollouts",
              "tasks": tasks}
    output = root / "libero/ten_task_sources/initial_state_audit.json"
    write_json(output, result)
    volume.commit()
    return result


@app.function(image=image, volumes={"/data": volume}, cpu=2, memory=8192,
              timeout=7200, max_containers=1, single_use_containers=True)
def prepare():
    import hashlib
    import json
    import os
    import urllib.request
    from pathlib import Path
    import h5py
    import numpy as np
    from odvla.data import file_hash, read_json, validate_splits, write_json
    from odvla.vla.libero_object_tasks import registry

    root = Path("/data")
    dest = root / "libero/ten_task_sources"
    dest.mkdir(parents=True, exist_ok=True)
    result = registry()
    result.update(status="preparing", source="official LIBERO Object demonstration HDF5 files",
                  test_accessed=False, images_rendered=False, sam_executed=False, model_trained=False)
    output = dest / "registry.json"
    for task in result["tasks"]:
        slug = task["task_id"].removeprefix("libero_object_")
        existing_milk = root / "libero/source/milk_demo.hdf5"
        path = existing_milk if slug == "milk" and existing_milk.exists() else root / task["source_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if file_hash(path) != task["source_sha256"]:
                raise ValueError(f"Existing source checksum mismatch: {path}")
        else:
            temporary = path.with_suffix(".hdf5.partial")
            digest = hashlib.sha256()
            with urllib.request.urlopen(task["source_url"], timeout=120) as src, temporary.open("wb") as out:
                while chunk := src.read(8 * 1024 * 1024):
                    digest.update(chunk)
                    out.write(chunk)
            if digest.hexdigest() != task["source_sha256"]:
                temporary.unlink(missing_ok=True)
                raise ValueError(f"Source download checksum mismatch: {task['task_id']}")
            os.replace(temporary, path)
            volume.commit()
        task["source_path"] = str(path.relative_to(root))
        with h5py.File(path, "r") as source:
            demos = sorted(source["data"].keys(), key=lambda x: int(x.split("_")[-1]))
            if len(demos) < 15:
                raise ValueError(f"Expected >=15 independent demonstration IDs for {task['task_id']}")
            if slug == "milk":
                aligned = root / "libero/muse_aligned_development_v1"
                splits = read_json(aligned / "splits.json")
                manifest = validate_splits(aligned, splits)
                entries = {e["id"]: e for e in manifest["episodes"]}
                train = [entries[e]["source_demo"] for e in splits["train"]]
                dev = [entries[e]["source_demo"] for e in splits["val"]]
                evaluation = [d for d in demos if d not in set(train + dev)]
                task["evaluation_history"] = "previously used milk initializations; not an untouched test set"
            else:
                rng = np.random.default_rng(int.from_bytes(hashlib.sha256(task["task_id"].encode()).digest()[:8], "big"))
                order = list(np.asarray(demos)[rng.permutation(len(demos))])
                evaluation, dev, train = order[:5], order[5:10], order[10:]
                task["evaluation_history"] = "reserved demonstration IDs; initialization uniqueness not yet audited"
            if not train or set(train) & set(dev) or set(train) & set(evaluation) or set(dev) & set(evaluation):
                raise ValueError(f"Overlapping or empty splits for {task['task_id']}")
            first = source["data"][train[0]]
            actions = first["actions"]
            required = ("agentview_rgb", "joint_states", "ee_states", "gripper_states")
            if actions.ndim != 2 or actions.shape[1] != 7 or not all(key in first["obs"] for key in required):
                raise ValueError(f"Unexpected source action/observation schema for {task['task_id']}")
            if first["states"].shape[0] != actions.shape[0]:
                raise ValueError(f"State/action timing length mismatch: {task['task_id']}")
            task["episodes"] = {"train": train, "development": dev, "evaluation": evaluation}
            task["source_schema"] = {"demonstrations": len(demos), "example_train_demo": train[0],
                "actions_shape": list(actions.shape), "states_shape": list(first["states"].shape),
                "observations": {key: list(first["obs"][key].shape) for key in required}}
        task["source_verified"] = True
        write_json(output, result)
        volume.commit()
        print(json.dumps({"task": task["task_id"], "train": len(train), "development": len(dev),
                          "evaluation": len(evaluation), "source_verified": True}), flush=True)
    result["status"] = "sources_prepared"
    write_json(output, result)
    volume.commit()
    return {"status": result["status"], "tasks": len(result["tasks"]), "registry": str(output),
            "note": "Source preparation only; no robot policy trained or evaluated."}


@app.local_entrypoint()
def main(background: bool = False, stage: str = "prepare"):
    if stage == "metadata":
        print(metadata.remote())
        return
    if stage == "initial-state-audit":
        print(audit_initial_states.remote())
        return
    if stage != "prepare":
        raise ValueError("Use prepare or metadata")
    if background:
        print(f"Spawned source preparation: {prepare.spawn().object_id}")
    else:
        print(prepare.remote())
