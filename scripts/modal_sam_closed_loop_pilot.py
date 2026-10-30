"""Small paired LIBERO pilot with fresh prefix-only SAM sessions each step."""
import modal
from scripts.modal_libero_oracle import volume

image = (modal.Image.debian_slim(python_version="3.10")
         .apt_install("git", "libosmesa6-dev", "libgl1", "libglew-dev", "patchelf", "ffmpeg")
         .run_commands("git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git /opt/LIBERO",
                       "cd /opt/LIBERO && git checkout 8f1084e3132a39270c3a13ebe37270a43ece2a01")
         .pip_install("mujoco==2.3.7", "robosuite==1.4.0", "bddl==1.0.1", "gym==0.25.2", "cloudpickle==2.1.0",
                      "future", "easydict", "hydra-core==1.2.0", "matplotlib", "termcolor", "numba", "scipy", "h5py", "pyyaml")
         .run_commands("pip install -e /opt/LIBERO")
         .env({"MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa"})
         .pip_install("torch==2.10.0", "torchvision", index_url="https://download.pytorch.org/whl/cu128")
         .run_commands("git clone https://github.com/facebookresearch/sam3.git /opt/sam3",
                       "cd /opt/sam3 && git checkout f1647e8d0c1d280991f4d1b7d974c2704537fe66")
         .pip_install("/opt/sam3")
         .pip_install("numpy<2", "Pillow", "opencv-python-headless", "setuptools", "einops", "pycocotools", "psutil")
         .add_local_python_source("odvla")
         .add_local_file("scripts/__init__.py", "/root/scripts/__init__.py")
         .add_local_file("scripts/modal_libero_oracle.py", "/root/scripts/modal_libero_oracle.py")
         .add_local_file("scripts/extract_libero_oracle_masks.py", "/root/extract_libero_oracle_masks.py")
         .add_local_file("scripts/extract_sam31.py", "/root/extract_sam31.py")
         .add_local_file("sam31_milk_development_config.json", "/root/sam31_milk_development_config.json"))
app = modal.App("odvla-sam-closed-loop-pilot")


@app.function(image=image, gpu="A100-80GB", cpu=4, memory=32768, volumes={"/data": volume},
              timeout=3600, single_use_containers=True)
def pilot(max_steps: int = 30):
    import hashlib
    import json
    import sys
    import tempfile
    import time
    from pathlib import Path
    import h5py
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    from robosuite.utils.transform_utils import quat2axisangle
    from odvla.experiment import load_run
    from odvla.inference import predict
    from odvla.measurements import mask_metrics
    from odvla.sam_prefix import replay_current
    from sam3.model_builder import build_sam3_multiplex_video_predictor
    sys.path.insert(0, "/root")
    from extract_sam31 import _patch_multiplex_init_state

    if not 1 <= max_steps <= 30:
        raise ValueError("This bounded pilot permits 1–30 steps")
    if not json.loads(Path("/data/eval/milk_sam_prefix_benchmark_v1.json").read_text())["all_matched"]:
        raise ValueError("Prefix/cache benchmark did not pass")
    dest = Path("/data/eval/milk_sam_closed_loop_pilot_v2")
    if dest.exists():
        raise FileExistsError("Refusing to overwrite pilot")
    cfg = json.loads(Path("/root/sam31_milk_development_config.json").read_text())
    checkpoint = Path("/data/checkpoints/sam3.1/sam3.1_multiplex.pt")
    def file_hash(path):
        sha = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                sha.update(chunk)
        return sha.hexdigest()
    if file_hash(checkpoint) != cfg["checkpoint_sha256"]:
        raise ValueError("SAM checkpoint hash mismatch")
    splits = json.loads(Path("/data/libero/milk_sam_development_v1/splits.json").read_text())
    episode = "episode_0014"  # Already-used validation initialization.
    if episode not in splits["val"]:
        raise ValueError("Pilot initialization is not in the development validation split")
    root = Path("/opt/LIBERO")
    config = Path.home() / ".libero/config.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(f"benchmark_root: {root}/libero/libero\nbddl_files: {root}/libero/libero/bddl_files\n"
                      f"init_states: {root}/libero/libero/init_files\ndatasets: {root}/libero/datasets\n"
                      f"assets: {root}/libero/libero/assets\n")
    from libero.libero.envs import OffScreenRenderEnv
    bddl = root / "libero/libero/bddl_files/libero_object/pick_up_the_milk_and_place_it_in_the_basket.bddl"
    torch.manual_seed(2468); np.random.seed(2468); torch.set_num_threads(2)
    tracker = build_sam3_multiplex_video_predictor(checkpoint_path=str(checkpoint), use_fa3=False,
        compile=False, warm_up=False, async_loading_frames=False)
    _patch_multiplex_init_state(tracker)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    with h5py.File("/data/libero/source/milk_demo.hdf5", "r") as source:
        initial_state = np.asarray(source["data/demo_21/states"][0]).copy()
    state_sha = hashlib.sha256(initial_state.tobytes()).hexdigest()
    dest.mkdir(parents=True)
    results = []
    for variant in ("action_only", "joint"):
        run = Path("/data/runs/milk_sam_matched_policy_v1") / f"{variant}_seed7"
        model, policy, stats = load_run(run, "cuda")
        if policy["dataset_fingerprint"] != splits["fingerprint"] or policy["model"]["context_dim"]:
            raise ValueError("Unexpected policy fingerprint or context")
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=512, camera_widths=512,
                                 camera_names=["agentview"])
        env.reset()
        obs = env.set_init_state(initial_state.copy())
        def robot_vector(observation):
            return np.concatenate([observation["robot0_joint_pos"], observation["robot0_eef_pos"],
                quat2axisangle(observation["robot0_eef_quat"]), observation["robot0_gripper_qpos"]]).astype(np.float32)
        traces, all_masks, frames = [], [], []
        robots, features, valids = [], [], []
        success = False
        failure = None
        initial_rgb_sha = None
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            try:
                for step in range(max_steps):
                    started = time.monotonic()
                    env.sim.forward()
                    rgb = np.asarray(env.sim.render(camera_name="agentview", width=512, height=512))
                    file = folder / f"{step:06}.jpg"
                    Image.fromarray(rgb).save(file, quality=98)
                    if step == 0:
                        initial_rgb_sha = file_hash(file)
                    masks, ids = replay_current(tracker, folder, step + 1,
                        cfg["first_frame_points_xy_pixels_512"], cfg["object_ids"])
                    values, valid = mask_metrics(masks[None])
                    present = masks.any(axis=(1, 2))
                    frame = {"step": step, "id_status": ids.tolist(), "nonempty": present.tolist(),
                             "tracking_seconds": time.monotonic() - started}
                    frames.append(rgb.copy()); all_masks.append(masks.copy())
                    if not np.all(ids == 1) or not present.all():
                        frame["tracking_failure"] = True
                        traces.append(frame)
                        failure = "missing, empty or duplicated target ID; no action executed"
                        break
                    if step == 0:
                        robots = [robot_vector(obs)] * policy["history"]
                        features = [values[0]] * policy["history"]
                        valids = [valid[0]] * policy["history"]
                    else:
                        robots.append(robot_vector(obs)); features.append(values[0]); valids.append(valid[0])
                    n = len(robots)
                    prefix = {"robot": np.stack(robots), "features": np.stack(features), "valid": np.stack(valids),
                        "context": np.zeros((n, 0), np.float32), "timestamps": np.arange(n, dtype=np.float64) / 20.0}
                    action = np.asarray(predict(prefix, model, policy, stats, device="cuda", seed=7 + step)["actions_source_units"][0])
                    if not np.isfinite(action).all():
                        frame["nonfinite_action"] = True; traces.append(frame)
                        failure = "nonfinite action; no action executed"
                        break
                    accepted = np.clip(action, -1, 1)
                    obs, _, done, _ = env.step(accepted)
                    success = bool(env.check_success())
                    frame.update({"sample_seed": 7 + step, "sampled_action": action.tolist(),
                        "executed_action": accepted.tolist(), "clipped_channels": (action != accepted).tolist(),
                        "success": success, "done": bool(done)})
                    traces.append(frame)
                    if success or done:
                        break
            finally:
                env.close()
        np.savez_compressed(dest / f"{variant}_masks.npz", masks=np.stack(all_masks))
        selected = sorted(set((0, len(frames) // 2, len(frames) - 1)))
        montage = Image.new("RGB", (512 * len(selected), 540), "white")
        draw = ImageDraw.Draw(montage)
        for col, idx in enumerate(selected):
            rgb = frames[idx].copy()
            for slot, color in enumerate(([255, 0, 0], [0, 255, 255])):
                mask = np.asarray(Image.fromarray(all_masks[idx][slot].astype(np.uint8)).resize((512, 512), Image.Resampling.NEAREST), bool)
                rgb[mask] = (.6 * rgb[mask] + .4 * np.asarray(color)).astype(np.uint8)
            montage.paste(Image.fromarray(rgb), (512 * col, 28))
            draw.text((512 * col + 4, 5), f"{variant} step {idx}: red milk, cyan basket", fill="black")
        montage.save(dest / f"{variant}_review.png")
        results.append({"variant": variant, "train_seed": 7, "policy_checkpoint_sha256": file_hash(run / "weights.pt"),
            "initial_rgb_sha256": initial_rgb_sha, "initial_state_sha256": state_sha,
            "executed_steps": sum("executed_action" in t for t in traces), "success": success,
            "tracking_failure": failure, "trace": traces})
    result = {"episode": episode, "source_demo": "demo_21", "split": "reused validation development",
        "max_steps": max_steps, "simulated_control_hz": 20, "initialization": "frozen RGB points",
        "perception": "fresh SAM session with observed prefix only; no oracle masks",
        "sampling": "paired seeds 7+step; first action of each chunk, clipped [-1,1]",
        "history_bootstrap": "repeat initial observation four times", "monitor": "disabled/passive",
        "not_realtime": True, "initial_rgb_identical": results[0]["initial_rgb_sha256"] == results[1]["initial_rgb_sha256"],
        "results": results}
    (dest / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    volume.commit()
    return {"output": str(dest), "initial_rgb_identical": result["initial_rgb_identical"],
            "results": [{k: v for k, v in r.items() if k != "trace"} for r in results]}


@app.local_entrypoint()
def main(max_steps: int = 30):
    print(pilot.remote(max_steps))
