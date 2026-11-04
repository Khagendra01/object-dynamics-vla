"""Resumable, training-only multi-demonstration preparation for LIBERO-Object."""
import modal
from scripts.modal_muse_motor_rollout import runtime_base
from scripts.modal_muse_smolvla_live import vision_image, volume

app = modal.App("odvla-libero-object-multidemo")
cpu_image = runtime_base.add_local_python_source("odvla")
sam_image = vision_image.add_local_file("configs/libero_object_10_integration_points.json", "/root/ten_task_points.json")


@app.function(image=cpu_image, volumes={"/data": volume}, cpu=2, memory=8192, timeout=600)
def plan(demos_per_task: int = 3):
    import h5py
    from pathlib import Path
    from odvla.data import read_json
    if not 2 <= demos_per_task <= 10:
        raise ValueError("Use 2..10 demonstrations per new task")
    root = Path("/data")
    registry = read_json(root / "libero/ten_task_sources/registry.json")
    tasks = []
    for task in registry["tasks"]:
        count = 1 if task["task_id"] == "libero_object_milk" else demos_per_task
        demos = task["episodes"]["train"][:count]
        with h5py.File(root / task["source_path"], "r") as source:
            episodes = [{"episode": demo, "frames": int(source["data"][demo]["actions"].shape[0])} for demo in demos]
        tasks.append({"task_id": task["task_id"], "episodes": episodes})
    return {"tasks": tasks, "episodes": sum(len(t["episodes"]) for t in tasks),
            "frames": sum(e["frames"] for t in tasks for e in t["episodes"]),
            "episodes_over_200": [f'{t["task_id"]}/{e["episode"]}' for t in tasks for e in t["episodes"] if e["frames"] > 200],
            "scope": "training episode metadata only; no held-out episodes opened"}


@app.function(image=cpu_image, volumes={"/data": volume}, cpu=2, memory=8192, timeout=1200)
def audit_images():
    """Verify every prepared training JPEG against its recorded sequence hash."""
    import hashlib
    from pathlib import Path
    from odvla.data import file_hash, read_json, safe_path, write_json
    root = Path("/data")
    base = root / "libero/ten_task_multidemo_v1"
    manifest_path = base / "renders.json"
    manifest = read_json(manifest_path)
    if manifest.get("status") != "complete":
        raise ValueError("Complete training renders are required")
    checked = []
    for item in manifest["entries"]:
        folder = safe_path(root, item["folder"]) / "rgb"
        paths = sorted(folder.glob("*.jpg"))
        if len(paths) != item["frames"] or any(p.name != f"{i:06}.jpg" for i, p in enumerate(paths)):
            raise ValueError(f"Noncontiguous RGB sequence: {item['task_id']}/{item['episode']}")
        digest = hashlib.sha256()
        for path in paths:
            digest.update(path.read_bytes())
        if digest.hexdigest() != item["rgb_sequence_sha256"]:
            raise ValueError(f"RGB sequence changed: {item['task_id']}/{item['episode']}")
        checked.append({"task_id": item["task_id"], "episode": item["episode"], "frames": len(paths)})
    result = {"status": "passed", "renders_manifest_sha256": file_hash(manifest_path),
              "episodes": checked, "frames": sum(x["frames"] for x in checked),
              "scope": "training RGB checksums only; no held-out episodes opened"}
    output = base / "image_audit.json"
    write_json(output, result)
    volume.commit()
    return {"status": "passed", "episodes": len(checked), "frames": result["frames"], "output": str(output)}


@app.function(image=cpu_image, volumes={"/data": volume}, cpu=4, memory=16384,
              timeout=5400, max_containers=1, single_use_containers=True)
def render(demos_per_task: int = 3, max_seconds: int = 1800):
    import hashlib
    import json
    import time
    from pathlib import Path
    import h5py
    import numpy as np
    from PIL import Image
    from robosuite.utils.transform_utils import quat2axisangle
    from odvla.data import file_hash, read_json, write_json

    if not 2 <= demos_per_task <= 10 or not 60 <= max_seconds <= 4800:
        raise ValueError("Use 2..10 demonstrations and 60..4800 seconds")
    root = Path("/data")
    source = read_json(root / "libero/ten_task_sources/registry.json")
    original = read_json(root / "libero/ten_task_integration_v1/renders.json")
    if source.get("status") != "sources_prepared" or original.get("status") != "complete":
        raise ValueError("Verified sources and original integration renders are required")
    originals = {r["task_id"]: r for r in original["entries"]}
    out = root / "libero/ten_task_multidemo_v1"
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "renders.json"
    if manifest_path.exists():
        prior = read_json(manifest_path)
        if prior["demos_per_new_task"] != demos_per_task:
            raise ValueError("Existing preparation has a different demonstration count")
    project = Path("/opt/LIBERO")
    config = Path.home() / ".libero/config.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(f"benchmark_root: {project}/libero/libero\nbddl_files: {project}/libero/libero/bddl_files\n"
                      f"init_states: {project}/libero/libero/init_files\ndatasets: {project}/libero/datasets\n"
                      f"assets: {project}/libero/libero/assets\n")
    from libero.libero.envs import OffScreenRenderEnv
    started, records = time.monotonic(), []

    def save(status):
        write_json(manifest_path, {"status": status, "demos_per_new_task": demos_per_task,
            "entries": records, "evaluation_episodes_opened": False,
            "alignment": "robot and RGB regenerated from states[t], action[t] retained"})
        volume.commit()

    for task in source["tasks"]:
        task_id = task["task_id"]
        selected = task["episodes"]["train"][:1 if task_id == "libero_object_milk" else demos_per_task]
        original_record = originals[task_id]
        if selected[0] != original_record["episode"]:
            raise ValueError(f"Training episode order changed: {task_id}")
        first = dict(original_record)
        first.update(folder=f"libero/ten_task_integration_v1/{task_id}/{selected[0]}", reused=True)
        records.append(first)
        env = None
        try:
            for demo in selected[1:]:
                folder = out / task_id / demo
                metadata = folder / "render_record.json"
                if metadata.exists():
                    prior = read_json(metadata)
                    if (prior["source_sha256"] != task["source_sha256"] or
                            file_hash(folder / "state_actions.npz") != prior["state_actions_sha256"]):
                        raise ValueError(f"Existing rendered episode changed: {task_id}/{demo}")
                    records.append(prior)
                    continue
                if time.monotonic() - started >= max_seconds:
                    save("incomplete")
                    return {"status": "incomplete", "completed_episodes": len(records)}
                if env is None:
                    env = OffScreenRenderEnv(bddl_file_name=str(project / "libero/libero/bddl_files" / task["bddl"]),
                                             camera_heights=512, camera_widths=512, camera_names=["agentview"])
                    env.reset()
                rgb_dir = folder / "rgb"
                rgb_dir.mkdir(parents=True, exist_ok=True)
                with h5py.File(root / task["source_path"], "r") as hdf5:
                    item = hdf5["data"][demo]
                    actions = np.asarray(item["actions"], dtype=np.float32)
                    if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all() or np.max(np.abs(actions)) > 1.00001:
                        raise ValueError(f"Invalid OSC actions: {task_id}/{demo}")
                    robots, digest = [], hashlib.sha256()
                    for t in range(len(actions)):
                        obs = env.set_init_state(np.asarray(item["states"][t]))
                        robot = np.concatenate([obs["robot0_joint_pos"], obs["robot0_eef_pos"],
                            quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]]).astype(np.float32)
                        if robot.shape != (15,) or not np.isfinite(robot).all():
                            raise ValueError(f"Invalid robot state: {task_id}/{demo}")
                        robots.append(robot)
                        env.sim.forward()
                        frame = rgb_dir / f"{t:06}.jpg"
                        Image.fromarray(np.asarray(env.sim.render(camera_name="agentview", width=512, height=512))).save(frame, quality=98)
                        digest.update(frame.read_bytes())
                np.savez_compressed(folder / "state_actions.npz", robot=np.stack(robots), actions=actions,
                                    timestamps=np.arange(len(actions), dtype=np.float64) / 20)
                record = {"task_id": task_id, "instruction": task["instruction"], "episode": demo, "split": "train",
                          "frames": len(actions), "source_sha256": task["source_sha256"],
                          "rgb_sequence_sha256": digest.hexdigest(), "first_rgb_sha256": file_hash(rgb_dir / "000000.jpg"),
                          "state_actions_sha256": file_hash(folder / "state_actions.npz"),
                          "folder": str(folder.relative_to(root)), "reused": False}
                write_json(metadata, record)
                records.append(record)
                save("incomplete")
                print(json.dumps({"task": task_id, "episode": demo, "frames": len(actions)}), flush=True)
        finally:
            if env is not None:
                env.close()
    save("complete")
    return {"status": "complete", "tasks": 10, "episodes": len(records), "output": str(out)}


@app.function(image=sam_image, gpu="A100-80GB", cpu=4, memory=131072, volumes={"/data": volume},
              timeout=7200, max_containers=1, single_use_containers=True)
def sam(stage: str = "preview", max_seconds: int = 1800):
    import hashlib
    import json
    import os
    import sys
    import tempfile
    import time
    from pathlib import Path
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    from odvla.data import file_hash, read_json, write_json
    from odvla.measurements import mask_metrics
    from odvla.sam_incremental import IncrementalPointTracker
    from odvla.sam_prefix import replay_current
    from sam3.model_builder import build_sam3_multiplex_video_predictor
    sys.path.insert(0, "/root")
    from extract_sam31 import _patch_multiplex_init_state

    if stage not in ("preview", "extract") or not 60 <= max_seconds <= 6000:
        raise ValueError("Use preview/extract and 60..6000 seconds")
    root = Path("/data")
    out = root / "libero/ten_task_multidemo_v1"
    manifest = read_json(out / "renders.json")
    if manifest.get("status") != "complete":
        raise ValueError("Complete training-only renders required")
    prompts = read_json("/root/ten_task_points.json")
    points_sha = file_hash("/root/ten_task_points.json")
    if points_sha != "952ae932888a3e447cab1c5994179e00a38dd1ccfbecd87db40183e72e8b4f5d":
        raise ValueError("Task point configuration changed; do not reuse the original SAM measurements")
    checkpoint = root / "checkpoints/sam3.1/sam3.1_multiplex.pt"
    if file_hash(checkpoint) != "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6":
        raise ValueError("SAM checkpoint checksum mismatch")
    torch.manual_seed(2468)
    np.random.seed(2468)
    predictor = build_sam3_multiplex_video_predictor(checkpoint_path=str(checkpoint), use_fa3=False,
        compile=False, warm_up=False, async_loading_frames=False)
    _patch_multiplex_init_state(predictor)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    entries, started = [], time.monotonic()
    sheet = Image.new("RGB", (3 * 384, 10 * 414), "white")
    draw = ImageDraw.Draw(sheet)
    task_order = {task_id: i for i, task_id in enumerate(prompts["tasks"])}
    column = {task_id: 0 for task_id in task_order}
    for record in manifest["entries"]:
        task_id, episode = record["task_id"], record["episode"]
        folder = root / record["folder"]
        first = folder / "rgb/000000.jpg"
        points = prompts["tasks"][task_id]["points"]
        first_sha = file_hash(first)
        if record.get("reused") and prompts["tasks"][task_id]["rgb_sha256"] != first_sha:
            raise ValueError(f"Original annotated frame changed: {task_id}/{episode}")
        if record.get("first_rgb_sha256", first_sha) != first_sha:
            raise ValueError(f"Initial RGB changed: {task_id}/{episode}")
        dest = folder / "sam_multidemo.npz"
        if stage == "extract" and record.get("reused"):
            # Original integration SAM is retained without modifying it. The
            # new trainer masks the unverified chocolate target slot on load.
            entries.append({"task_id": task_id, "episode": episode,
                            "sam_path": str((folder / "sam_measurements.npz").relative_to(root)),
                            "reused": True})
            continue
        if stage == "extract" and dest.exists():
            with np.load(dest, allow_pickle=False) as z:
                if (z["first_rgb_sha256"].item() != first_sha or z["points_sha256"].item() != points_sha
                        or len(z["features"]) != record["frames"]):
                    raise ValueError(f"Existing SAM measurements differ: {task_id}/{episode}")
            entries.append({"task_id": task_id, "episode": episode, "sam_path": str(dest.relative_to(root)), "reused": True})
            continue
        if time.monotonic() - started >= max_seconds:
            write_json(out / f"sam_{stage}.json", {"status": "incomplete", "entries": entries})
            volume.commit()
            return {"status": "incomplete", "episodes": len(entries)}
        paths = [first] if stage == "preview" else sorted((folder / "rgb").glob("*.jpg"))
        if not 1 <= len(paths) <= 200:
            raise ValueError(f"Expected 1..200 observed frames: {task_id}/{episode}")
        tracker = IncrementalPointTracker(predictor, first, points, prompts["object_ids"], capacity=len(paths))
        masks = np.zeros((len(paths), 2, 128, 128), dtype=bool)
        ids = np.zeros((len(paths), 2), dtype=np.uint8)
        parity_frames = []
        try:
            for t, frame in enumerate(paths):
                masks[t], ids[t] = tracker.step(frame)
                if stage == "extract" and t in (0, 9):
                    with tempfile.TemporaryDirectory(prefix="odvla-multidemo-prefix-") as temp:
                        for item in paths[:t + 1]:
                            os.symlink(item, Path(temp) / item.name)
                        expected_masks, expected_ids = replay_current(predictor, temp, t + 1, points, prompts["object_ids"])
                    if not np.array_equal(masks[t], expected_masks) or not np.array_equal(ids[t], expected_ids):
                        raise ValueError(f"SAM prefix mismatch: {task_id}/{episode}/{t}")
                    parity_frames.append(t)
                if t == 0 and stage == "preview":
                    original = np.asarray(Image.open(first).convert("RGB")).copy()
                    for slot, color in enumerate(((255, 70, 70), (70, 230, 120))):
                        overlay = np.asarray(Image.fromarray(masks[t, slot].astype(np.uint8)).resize((512, 512), Image.Resampling.NEAREST), dtype=bool)
                        original[overlay] = (original[overlay] * .55 + np.asarray(color) * .45).astype(np.uint8)
                    x, y = column[task_id] * 384, task_order[task_id] * 414
                    sheet.paste(Image.fromarray(original).resize((384, 384)), (x, y + 30))
                    draw.text((x + 3, y + 5), f"{task_id.removeprefix('libero_object_')} / {episode}", fill="black")
                    column[task_id] += 1
        finally:
            tracker.close()
        features, valid = mask_metrics(masks)
        valid &= ids[:, :, None] == 1
        # The chocolate target was visually inspected but is occluded behind
        # the ketchup bottle. Do not turn an uncertain point into a positive
        # target-measurement signal. The basket slot remains available.
        target_masked = task_id == "libero_object_chocolate_pudding"
        if target_masked:
            valid[:, 0, :] = False
        item = {"task_id": task_id, "episode": episode, "frames": len(paths), "first_rgb_sha256": first_sha,
                "visible_per_slot": valid.any(axis=2).sum(axis=0).tolist(), "prefix_parity_frames": parity_frames,
                "target_masked_unverified": target_masked, "initialization": "assistant-proposed fixed RGB points"}
        if stage == "extract":
            np.savez_compressed(dest, features=features, valid=valid, id_status=ids,
                                first_rgb_sha256=np.asarray(first_sha), points_sha256=np.asarray(points_sha),
                                target_masked_unverified=np.asarray(target_masked))
            item["sam_path"] = str(dest.relative_to(root))
            volume.commit()
        entries.append(item)
        print(json.dumps(item), flush=True)
    if stage == "preview":
        sheet.save(out / "sam_preview.png")
    write_json(out / f"sam_{stage}.json", {"status": "complete", "entries": entries,
        "initialization": "assistant-proposed fixed task-specific RGB points; not autonomous target selection",
        "chocolate_target": "inspected and heavily occluded; target measurements masked as unverified"})
    volume.commit()
    return {"status": "complete", "stage": stage, "episodes": len(entries)}


@app.local_entrypoint()
def main(stage: str = "render", demos_per_task: int = 3, max_seconds: int = 1800, background: bool = False):
    if stage == "plan":
        print(plan.remote(demos_per_task))
        return
    if stage == "audit-images":
        if background:
            print(f"Spawned image audit: {audit_images.spawn().object_id}")
        else:
            print(audit_images.remote())
        return
    if stage not in ("render", "preview", "extract"):
        raise ValueError("Use plan, render, preview or extract")
    function = render if stage == "render" else sam
    args = (demos_per_task, max_seconds) if stage == "render" else (stage, max_seconds)
    if background:
        print(f"Spawned {stage}: {function.spawn(*args).object_id}")
    else:
        print(function.remote(*args))
