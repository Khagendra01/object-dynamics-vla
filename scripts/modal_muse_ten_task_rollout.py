"""Bounded, parity-gated closed-loop integration of the shared ten-task policy.

The provided points cover one training demonstration per task. These are not
held-out task-success evaluations and do not automate target grounding.
"""
import modal
from scripts.modal_muse_smolvla_live import app, Policy, vision_image, volume

image = vision_image.add_local_file("configs/libero_object_10_integration_points.json", "/root/ten_task_points.json")


@app.function(image=image, volumes={"/data": volume}, cpu=2, memory=16384, timeout=900)
def freeze(name: str = "muse_ten_task_frozen_v1"):
    """Copy the completed shared checkpoint and record exact first-feature hashes."""
    import hashlib
    import json
    import shutil
    from pathlib import Path
    import h5py
    import numpy as np
    from odvla.data import file_hash, read_json
    from odvla.vla.bounded_features import BoundedFeatureCache, frozen_feature_key

    if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
        raise ValueError("Invalid frozen output name")
    fit = Path("/data/eval/muse_ten_task_integration_v1")
    report = read_json(fit / "report.json")
    if report.get("status") not in ("completed", "budget_stopped") or not report.get("completed_updates"):
        raise ValueError("Shared training has no finished checkpoint")
    step = int(report["completed_updates"])
    digest = file_hash(fit / "last.pt")
    if digest != report["checkpoint_sha256"]:
        raise ValueError("Shared training checkpoint hash changed")
    root = Path("/data/libero/ten_task_integration_v1")
    records = read_json(root / "renders.json")["entries"]
    registry = {t["task_id"]: t for t in read_json("/data/libero/ten_task_sources/registry.json")["tasks"]}
    annotations = read_json("/root/ten_task_points.json")["tasks"]
    cache = BoundedFeatureCache("/data/libero/ten_task_bounded_nf4_v1", int(report["contract"]["cache_gib"] * 1024 ** 3))
    expected = {}
    for item in records:
        task, episode = item["task_id"], item["episode"]
        first = root / task / episode / "rgb/000000.jpg"
        image_hash = file_hash(first)
        if image_hash != annotations[task]["rgb_sha256"]:
            raise ValueError(f"Annotated initial RGB mismatch: {task}")
        key = frozen_feature_key(task_id=task, episode_id=episode, frame=0, instruction=item["instruction"],
            image_sha256=image_hash, muse_revision=report["contract"]["muse_revision"],
            processor_sha256=report["processor_sha256"], backend=report["backend"], chat_date="2026-10-30")
        tokens = cache.get(key)
        if tokens is None:
            raise ValueError(f"Initial feature evicted; regenerate evaluation first: {task}")
        feature_hash = hashlib.sha256(tokens["features_bfloat16_bits"].tobytes()).hexdigest()
        with h5py.File(Path("/data") / registry[task]["source_path"], "r") as source:
            state_hash = hashlib.sha256(np.asarray(source["data"][episode]["states"][0]).tobytes()).hexdigest()
        expected[task] = {"episode": episode, "rgb_sha256": image_hash,
                          "feature_sha256": feature_hash, "initial_state_sha256": state_hash}
    out = Path("/data/eval") / name
    out.mkdir(parents=True, exist_ok=False)
    target = out / f"checkpoint_{step}.pt"
    shutil.copyfile(fit / "last.pt", target)
    if file_hash(target) != digest:
        raise ValueError("Frozen checkpoint copy mismatch")
    shutil.copyfile(fit / "stats.npz", out / "stats.npz")
    metadata = {"status": "frozen", "completed_updates": step, "normalize_queries": True,
        "dynamics_objective": False, "muse_revision": report["contract"]["muse_revision"],
        "committed_checkpoints": [{"step": step, "sha256": digest}],
        "source_report_sha256": file_hash(fit / "report.json"), "expected_inputs": expected,
        "scope": "one manually prompted training initialization per task; integration only"}
    (out / "report.json").write_text(json.dumps(metadata, indent=2) + "\n")
    volume.commit()
    return {"fit": name, "step": step, "checkpoint_sha256": digest, "tasks": len(expected)}


@app.function(image=image, gpu="H200", cpu=4, memory=131072, volumes={"/data": volume},
              timeout=900, max_containers=1, single_use_containers=True)
def rollout(task_id: str, name: str, fit_name: str = "muse_ten_task_frozen_v1", checkpoint_step: int = 20,
            max_steps: int = 30, seed_base: int = 7):
    import hashlib
    import json
    import subprocess
    import sys
    import time
    from pathlib import Path
    import h5py
    import numpy as np
    import torch
    from PIL import Image
    from robosuite.utils.transform_utils import quat2axisangle
    from odvla.data import file_hash, read_json
    from odvla.measurements import mask_metrics
    from odvla.sam_incremental import IncrementalPointTracker
    from odvla.sam_prefix import replay_current
    from odvla.vla.dataset import observation_window
    from odvla.vla.muse_context import FrozenMuseTokens
    from sam3.model_builder import build_sam3_multiplex_video_predictor
    sys.path.insert(0, "/root")
    from extract_sam31 import _patch_multiplex_init_state

    if not 1 <= max_steps <= 200 or not 0 <= seed_base <= 100000:
        raise ValueError("Invalid rollout bound")
    for value in (task_id, name, fit_name):
        if not value or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in value):
            raise ValueError("Invalid task or output name")
    root = Path("/data")
    registry = {t["task_id"]: t for t in read_json(root / "libero/ten_task_sources/registry.json")["tasks"]}
    annotations = read_json("/root/ten_task_points.json")
    if task_id not in registry or task_id not in annotations["tasks"]:
        raise ValueError("Unknown ten-task integration task")
    task, annotation = registry[task_id], annotations["tasks"][task_id]
    episode = annotation["episode"]
    if episode not in task["episodes"]["train"]:
        raise ValueError("Integration rollout must use its named training episode")
    fit = root / "eval" / fit_name
    metadata = read_json(fit / "report.json")
    expected = metadata["expected_inputs"][task_id]
    if expected["episode"] != episode:
        raise ValueError("Frozen evaluation metadata mismatch")
    checkpoint = fit / f"checkpoint_{checkpoint_step}.pt"
    checkpoint_hash = file_hash(checkpoint)
    milestone = next(x for x in metadata["committed_checkpoints"] if x["step"] == checkpoint_step)
    if checkpoint_hash != milestone["sha256"] or not metadata["normalize_queries"] or metadata["dynamics_objective"]:
        raise ValueError("Frozen shared checkpoint mismatch")
    policy = Policy(mode="quantized", fit_name=fit_name, checkpoint_step=checkpoint_step)
    ready = policy.ready.remote()
    if ready["checkpoint_sha256"] != checkpoint_hash or ready["checkpoint_step"] != checkpoint_step:
        raise ValueError("Action worker loaded a different checkpoint")
    folder = root / "libero/ten_task_integration_v1" / task_id / episode
    with np.load(folder / "state_actions.npz", allow_pickle=False) as z:
        expected_robot = z["robot"][0].copy()
    with np.load(folder / "sam_measurements.npz", allow_pickle=False) as z:
        expected_sam, expected_valid = z["features"][0].copy(), z["valid"][0].copy()
    with np.load(fit / "stats.npz", allow_pickle=False) as z:
        stats = {k: z[k].copy() for k in z.files}
    with h5py.File(root / task["source_path"], "r") as source:
        initial = np.asarray(source["data"][episode]["states"][0]).copy()
    state_hash = hashlib.sha256(initial.tobytes()).hexdigest()
    if state_hash != expected["initial_state_sha256"]:
        raise ValueError("Initial simulator state mismatch")
    dest = root / "eval" / name
    dest.mkdir(parents=True, exist_ok=False)
    frames = dest / "frames"
    frames.mkdir()
    observations = dest / "observations"
    observations.mkdir()
    started = time.monotonic()
    report = {"status": "started", "task_id": task_id, "instruction": task["instruction"],
              "episode": episode, "scope": "training initialization; bounded integration smoke, not held-out evaluation",
              "sam_initialization": annotations["initialization_method"], "checkpoint_sha256": checkpoint_hash,
              "checkpoint_step": checkpoint_step, "seed_base": seed_base, "max_steps": max_steps,
              "initial_state_sha256": state_hash, "success": False}
    env = muse = incremental = None
    trace, noises = [], []
    try:
        project = Path("/opt/LIBERO")
        config = Path.home() / ".libero/config.yaml"
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(f"benchmark_root: {project}/libero/libero\nbddl_files: {project}/libero/libero/bddl_files\n"
                          f"init_states: {project}/libero/libero/init_files\ndatasets: {project}/libero/datasets\n"
                          f"assets: {project}/libero/libero/assets\n")
        from libero.libero.envs import OffScreenRenderEnv
        torch.set_num_threads(2)
        torch.manual_seed(2468)
        np.random.seed(2468)
        sam_checkpoint = root / "checkpoints/sam3.1/sam3.1_multiplex.pt"
        if file_hash(sam_checkpoint) != "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6":
            raise ValueError("SAM checkpoint mismatch")
        tracker = build_sam3_multiplex_video_predictor(checkpoint_path=str(sam_checkpoint), use_fa3=False,
            compile=False, warm_up=False, async_loading_frames=False)
        _patch_multiplex_init_state(tracker)
        muse = FrozenMuseTokens(root / "checkpoints/muse-glimmer-30b" / metadata["muse_revision"], 26, quantization="nf4")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        env = OffScreenRenderEnv(bddl_file_name=str(project / "libero/libero/bddl_files" / task["bddl"]),
                                camera_heights=512, camera_widths=512, camera_names=["agentview"])
        env.reset()
        obs = env.set_init_state(initial.copy())
        robots, features, valids = [], [], []
        for step in range(max_steps):
            env.sim.forward()
            rgb = np.asarray(env.sim.render(camera_name="agentview", width=512, height=512))
            frame = frames / f"{step:06}.jpg"
            Image.fromarray(rgb).save(frame, quality=98)
            robots.append(np.concatenate([obs["robot0_joint_pos"], obs["robot0_eef_pos"],
                quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]]).astype(np.float32))
            if incremental is None:
                incremental = IncrementalPointTracker(tracker, frame, annotation["points"], annotations["object_ids"], capacity=max_steps)
            masks, ids = incremental.step(frame)
            if step in (0, 9, 29, 69, 99, 149, 199):
                replay_masks, replay_ids = replay_current(tracker, frames, step + 1, annotation["points"], annotations["object_ids"])
                if not np.array_equal(masks, replay_masks) or not np.array_equal(ids, replay_ids):
                    raise ValueError("Incremental SAM differs from observed-prefix replay")
            measured, valid = mask_metrics(masks[None])
            valid &= ids[None, :, None] == 1
            features.append(measured[0])
            valids.append(valid[0])
            observed = {"robot": np.stack(robots), "features": np.stack(features), "valid": np.stack(valids),
                        "timestamps": np.arange(len(robots), dtype=np.float64) / 20,
                        "actions": np.zeros((len(robots), 7), np.float32)}
            window = observation_window(observed, len(robots) - 1, stats)
            with Image.open(frame) as image_rgb, torch.autocast("cuda", enabled=False):
                token = muse.encode(image_rgb, task["instruction"], current_date="2026-10-30")
            if step == 0:
                rgb_hash = hashlib.sha256(frame.read_bytes()).hexdigest()
                feature_hash = hashlib.sha256(token["features_bfloat16_bits"].tobytes()).hexdigest()
                parity = {"rgb": rgb_hash == expected["rgb_sha256"], "muse": feature_hash == expected["feature_sha256"],
                          "robot": bool(np.allclose(robots[0], expected_robot, rtol=0, atol=1e-6)),
                          "sam": bool(np.allclose(measured[0], expected_sam, rtol=0, atol=1e-6)),
                          "sam_valid": bool(np.array_equal(valid[0], expected_valid))}
                report["initial_parity"] = parity
                (dest / "initial_input_parity.json").write_text(json.dumps(parity, indent=2) + "\n")
                volume.commit()
                if not all(parity.values()):
                    raise ValueError("Initial training input parity failed; no action executed")
            generator = torch.Generator(device="cuda").manual_seed(seed_base + step)
            noise = torch.randn((1, 50, 32), generator=generator, device="cuda").cpu().numpy()
            result = policy.predict.remote(token, window, step, frame.read_bytes(), robots[-1], noise, True)
            if result["checkpoint_sha256"] != checkpoint_hash or result["checkpoint_step"] != checkpoint_step:
                raise ValueError("Action worker changed checkpoint")
            if not np.array_equal(noise, result["noise"]):
                raise ValueError("Sampling noise round-trip mismatch")
            action = result["action"]
            clipped = np.clip(action, -1, 1)
            np.savez(observations / f"{step:06}.npz", raw_robot=robots[-1], sam_masks=masks,
                     sampled_noise=noise, sampled_action=action, executed_action=clipped)
            noises.append(noise)
            obs, _, done, _ = env.step(clipped)
            success = bool(env.check_success())
            trace.append({"step": step, "success": success, "done": bool(done),
                          "noise_sha256": hashlib.sha256(noise.tobytes()).hexdigest(),
                          "sampled_action": action.tolist(), "executed_action": clipped.tolist()})
            if (step + 1) % 10 == 0:
                (dest / "trace.json").write_text(json.dumps(trace, indent=2) + "\n")
                volume.commit()
                print(json.dumps({"task": task_id, "steps": step + 1, "success": success}), flush=True)
            if success or done:
                break
        env.sim.forward()
        Image.fromarray(np.asarray(env.sim.render(camera_name="agentview", width=512, height=512))).save(
            frames / f"{len(trace):06}.jpg", quality=98)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", "20", "-i", str(frames / "%06d.jpg"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(dest / "rollout.mp4")], check=True)
        report.update(status="completed", steps=len(trace), success=bool(trace and trace[-1]["success"]),
                      truncated_at_limit=bool(len(trace) == max_steps and not trace[-1]["success"] and not trace[-1]["done"]))
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc), steps=len(trace))
        raise
    finally:
        if noises:
            np.save(dest / "noise.npy", np.concatenate(noises, axis=0), allow_pickle=False)
        (dest / "trace.json").write_text(json.dumps(trace, indent=2) + "\n")
        report["elapsed_seconds"] = time.monotonic() - started
        (dest / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        if incremental is not None:
            incremental.close()
        if env is not None:
            env.close()
        if muse is not None:
            muse.close()
        volume.commit()
    return report


@app.local_entrypoint()
def ten_task_main(stage: str = "freeze", task_id: str = "libero_object_milk", name: str = "ten_task_milk_smoke_v1",
         fit_name: str = "muse_ten_task_frozen_v1", checkpoint_step: int = 20,
         max_steps: int = 30, seed_base: int = 7, background: bool = False):
    if stage == "freeze":
        print(freeze.remote(fit_name))
    elif stage == "rollout":
        args = (task_id, name, fit_name, checkpoint_step, max_steps, seed_base)
        if background:
            print(f"Spawned ten-task rollout: {rollout.spawn(*args).object_id}")
        else:
            print(rollout.remote(*args))
    else:
        raise ValueError("Use freeze or rollout")
