"""Bounded two-task motor pretraining from audited LIBERO and Muse caches."""
import modal
from scripts.modal_real_ablations import image as base_image

volume = modal.Volume.from_name("object-dynamics-vla-data", create_if_missing=False)
image = (base_image.add_local_file("scripts/__init__.py", "/root/scripts/__init__.py")
         .add_local_file("scripts/modal_real_ablations.py", "/root/scripts/modal_real_ablations.py")
         .add_local_file("scripts/train_muse_vla.py", "/root/train_muse_vla.py"))
app = modal.App("odvla-multitask-motor")


@app.function(image=image, gpu="T4", cpu=4, memory=114688, volumes={"/data": volume},
              timeout=4200, max_containers=1, single_use_containers=True)
def run(resume: bool = False):
    import hashlib
    import json
    import sys
    import time
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path
    import numpy as np
    from odvla.data import _moments, file_hash, load_episode, read_json, validate_splits, write_json
    from odvla.vla.dataset import CachedEpisodeDataset
    from odvla.vla.transfer import summarize_contract, audit_transfer, validate_artifacts
    sys.path.insert(0, "/root")
    from train_muse_vla import train
    started = time.monotonic()
    tasks = [(Path("/data/libero/milk_sam_development_v1"), Path("/data/libero/muse_middle_tokens_development_v1"),
              Path("/data/libero/source/milk_demo.hdf5")),
             (Path("/data/libero/object_raw"), Path("/data/libero/object_muse_middle_tokens_v1"),
              Path("/data/libero/source/alphabet_soup_demo.hdf5"))]
    out = Path("/data/runs/muse_multitask_milk_soup_motor_seed7_v1")
    if out.exists() != resume: raise ValueError("Use a fresh run or explicitly resume an existing run")
    controller = read_json("/data/eval/multitask_controller_audit_v1.json")
    if not controller["matching_simulator_configuration"] or not all(x["final_success"] for x in controller["tasks"].values()):
        raise ValueError("Controller/replay compatibility audit did not pass")
    manifests, splits_list, summaries, contracts, episodes = [], [], [], [], []
    semantics = {"embodiment": "LIBERO SingleArm Panda", "controller": "robosuite 1.4 OperationalSpaceController OSC_POSE",
        "action_channels": ["translation_x", "translation_y", "translation_z", "rotation_x", "rotation_y", "rotation_z", "gripper"],
        "action_frame": "robosuite 1.4 OSC_POSE default controller configuration",
        "action_units": "normalized [-1,1]; OSC scales translation 0.05 and rotation 0.5",
        "gripper_convention": "PandaGripper sign: minus open, plus close", "robot_channels": ["joint_states:7", "ee_states:6", "gripper_states:2"],
        "camera": "agentview native 512 JPEG quality 98", "rgb_orientation": "unflipped MuJoCo render; matches source",
        "control_hz": 20, "metrics": ["centroid_u", "centroid_v", "bbox_width", "bbox_height", "visible_area"],
        "metric_units": ["image_fraction"] * 5, "slot_roles": ["target", "basket"],
        "muse_revision": "cec94fcfd6512f2cfe708138c32533b346ea4f69", "muse_layer": 26,
        "muse_processor_sha256": "97e2a486dd9866b81f40cf4b8bc0c9ced9a7cd8a5bc65aa4cc2f4de0712dae77"}
    imports = None
    for root, cache, source in tasks:
        splits = read_json(root / "splits.json"); manifest = validate_splits(root, splits)
        provenance = read_json(cache / "manifest.json")
        if not provenance["complete"] or provenance["dataset_fingerprint"] != splits["fingerprint"]:
            raise ValueError("Unfinished or mismatched token cache")
        current = {k: manifest["import"][k] for k in ("fps_user_supplied", "robot_keys", "image_key", "flip_vertical", "alignment_assumption")}
        if imports is not None and current != imports: raise ValueError("Dataset import conventions differ")
        imports = current
        source_hash = file_hash(source)
        contract = {"schema_version": 1, "dataset_fingerprint": splits["fingerprint"], "semantics": semantics,
                    "audit_evidence": "docs/EXECUTION_REPORT.md; /data/eval/multitask_controller_audit_v1.json; recorded-state RGB render audit",
                    "episode_sources": {e["id"]: source_hash + ":" + e["source_demo"] for e in manifest["episodes"]}}
        summary = summarize_contract(contract, splits); validate_artifacts(contract, manifest, provenance)
        for e in manifest["episodes"]:
            if e["id"] in splits["train"]:
                ep = load_episode(root, e, manifest, include_images=False)
                if ep["robot"].shape[1] != 15 or ep["actions"].shape[1] != 7: raise ValueError("State/action dimensions differ")
                episodes.append(ep)
        manifests.append(manifest); splits_list.append(splits); summaries.append(summary); contracts.append(contract)
    audit_transfer(summaries[0], summaries[1]); audit_transfer(summaries[1], summaries[0])
    # Include the eight older used milk test sources in the protected inventory.
    old_root = Path("/data/libero/milk_raw"); old_manifest = read_json(old_root / "manifest.json")
    old_splits = read_json("/data/libero/milk_oracle/splits.json")
    milk_source_hash = file_hash(tasks[0][2]); old = {e["id"]: e for e in old_manifest["episodes"]}
    old_test = {milk_source_hash + ":" + old[e]["source_demo"] for e in old_splits["test"]}
    if old_test & set(summaries[0]["training_sources"]): raise ValueError("Used milk test source in training")
    combined = dict(summaries[0], training_sources=sorted(set(summaries[0]["training_sources"] + summaries[1]["training_sources"])),
                    protected_sources=sorted(set(summaries[0]["protected_sources"] + summaries[1]["protected_sources"]) | old_test))
    f = 5
    stats = {}
    stats["feature_mean"], stats["feature_std"] = _moments([e["features"].reshape(-1, f) for e in episodes],
                                                          [e["valid"].reshape(-1, f) for e in episodes])
    for k in ("robot", "actions", "context"):
        stats[k + "_mean"], stats[k + "_std"] = _moments([e[k] for e in episodes])
    spec = {"datasets": [{"fingerprint": s["fingerprint"], "cache_manifest_sha256": file_hash(c / "manifest.json"),
                           "training_sources": a["training_sources"], "protected_sources": a["protected_sources"]}
                          for (_, c, _), s, a in zip(tasks, splits_list, summaries)], "sampling": "uniform over pooled frames"}
    fingerprint = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    provenance = {"fingerprint": fingerprint, "dataset_audit": combined, "spec": spec}
    del episodes
    prepared = []
    for split in ("train", "val"):
        samples = []
        for root, cache, _ in tasks:
            dataset = CachedEpisodeDataset(root, cache, split, stats)
            with ThreadPoolExecutor(max_workers=8) as pool:
                # map retains index order, so paired batches and RNG schedules stay unchanged.
                for sample in pool.map(dataset.__getitem__, range(len(dataset))):
                    if time.monotonic() - started > 3000: raise TimeoutError("Preload budget exhausted")
                    samples.append(sample)
                    if len(samples) % 1000 == 0:
                        print(json.dumps({"split": split, "items": len(samples), "seconds": time.monotonic() - started}), flush=True)
        prepared.append(samples)
    print(json.dumps({"train_windows": len(prepared[0]), "val_windows": len(prepared[1]), "preload_seconds": time.monotonic() - started}), flush=True)
    audit_path = Path("/data/eval/multitask_source_contracts_v1.json")
    audit_contents = {"contracts": contracts, "combined": combined, "spec": spec}
    if audit_path.exists():
        if not resume or read_json(audit_path) != audit_contents: raise ValueError("Existing audit differs or resume not requested")
    else:
        if resume: raise FileNotFoundError("Missing existing audit")
        write_json(audit_path, audit_contents)
    def checkpoint():
        volume.commit()
        if time.monotonic() - started > 3500: raise TimeoutError("Runtime budget reached after saving checkpoint")
    try:
        result = train(tasks[0][0], tasks[0][1], out, "motor", 7, "cuda", steps=1000, batch_size=64, accumulation=1,
                       prepared=(stats, prepared[0], prepared[1]), prepared_provenance=provenance, checkpoint_hook=checkpoint,
                       resume=resume)
    finally:
        volume.commit()
    return {**result, "total_seconds": time.monotonic() - started}


@app.local_entrypoint()
def main(resume: bool = False): print(run.remote(resume))
