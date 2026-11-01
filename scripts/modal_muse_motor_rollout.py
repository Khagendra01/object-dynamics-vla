"""Prefix-observation, first-action replanning pilot for the two-task motor expert."""
import modal

volume = modal.Volume.from_name("object-dynamics-vla-data", create_if_missing=False)
runtime_base = (modal.Image.debian_slim(python_version="3.11")
         .apt_install("git", "libosmesa6-dev", "libgl1", "libglew-dev", "patchelf", "ffmpeg")
         .run_commands("git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git /opt/LIBERO",
                       "git -C /opt/LIBERO checkout 8f1084e3132a39270c3a13ebe37270a43ece2a01")
         .pip_install("mujoco==2.3.7", "robosuite==1.4.0", "bddl==1.0.1", "gym==0.25.2", "cloudpickle==2.1.0",
                      "future", "easydict", "hydra-core==1.3.2", "matplotlib", "termcolor", "numba", "scipy", "h5py", "pyyaml")
         .run_commands("pip install -e /opt/LIBERO")
         .env({"MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa"})
         .pip_install("torch==2.10.0", "torchvision==0.25.0", index_url="https://download.pytorch.org/whl/cu128")
         .pip_install("numpy<2", "Pillow", "transformers==5.21.0", "accelerate", "safetensors"))
image = runtime_base.add_local_python_source("odvla")
app = modal.App("odvla-muse-motor-rollout")


@app.function(image=image, gpu="A100-80GB", cpu=4, memory=131072,
              volumes={"/data": volume}, timeout=1800, max_containers=1, single_use_containers=True)
def run(max_steps: int = 10, suffix: str = "v2"):
    import hashlib
    import json
    import subprocess
    import time
    from pathlib import Path
    import h5py
    import numpy as np
    import torch
    import transformers
    from PIL import Image
    from robosuite.utils.transform_utils import quat2axisangle
    from odvla.data import file_hash, read_json
    from odvla.vla.action_expert import MuseActionExpert
    from odvla.vla.dataset import observation_window
    from odvla.vla.muse_context import FrozenMuseTokens
    if not 1 <= max_steps <= 30: raise ValueError("Pilot cap must be 1..30")
    if not suffix.isalnum(): raise ValueError("Suffix must be alphanumeric")
    dest = Path(f"/data/eval/muse_multitask_motor_pilot_{suffix}")
    if dest.exists(): raise FileExistsError(dest)
    root = Path("/opt/LIBERO"); cfg = Path.home() / ".libero/config.yaml"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(f"benchmark_root: {root}/libero/libero\nbddl_files: {root}/libero/libero/bddl_files\n"
                   f"init_states: {root}/libero/libero/init_files\ndatasets: {root}/libero/datasets\nassets: {root}/libero/libero/assets\n")
    from libero.libero.envs import OffScreenRenderEnv
    run_path = Path("/data/runs/muse_multitask_milk_soup_motor_seed7_v1")
    metadata = read_json(run_path / "configuration.json")
    if metadata["variant"] != "motor": raise ValueError("Expected motor checkpoint")
    splits = read_json("/data/libero/milk_sam_development_v1/splits.json")
    if "episode_0014" not in splits["val"]: raise ValueError("Expected reused development state")
    with np.load(run_path / "stats.npz", allow_pickle=False) as z: stats = {k: z[k] for k in z.files}
    with h5py.File("/data/libero/source/milk_demo.hdf5", "r") as source:
        initial = np.asarray(source["data/demo_21/states"][0]).copy()
    env = OffScreenRenderEnv(bddl_file_name=str(root / "libero/libero/bddl_files/libero_object/pick_up_the_milk_and_place_it_in_the_basket.bddl"),
                             camera_heights=512, camera_widths=512, camera_names=["agentview"])
    env.reset(); obs = env.set_init_state(initial.copy())
    torch.set_num_threads(2); torch.manual_seed(2468)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    muse = FrozenMuseTokens("/data/checkpoints/muse-glimmer-30b/cec94fcfd6512f2cfe708138c32533b346ea4f69", 26)
    reference = read_json("/data/libero/muse_middle_tokens_development_v1/manifest.json")
    if muse.processor_hash != reference["processor_config_sha256"] or transformers.__version__ != reference["transformers"]:
        raise ValueError("Runtime processor differs from cache")
    model = MuseActionExpert().cuda().eval()
    model.load_state_dict(torch.load(run_path / "weights_best.pt", map_location="cuda", weights_only=True))
    dest.mkdir(parents=True); frames = dest / "frames"; frames.mkdir()
    robots = []; trace = []; success = False; parity = None
    try:
        for step in range(max_steps):
            start = time.monotonic(); env.sim.forward()
            rgb = np.asarray(env.sim.render(camera_name="agentview", width=512, height=512))
            frame = frames / f"{step:06}.jpg"; Image.fromarray(rgb).save(frame, quality=98)
            render_s = time.monotonic() - start
            robots.append(np.concatenate([obs["robot0_joint_pos"], obs["robot0_eef_pos"],
                          quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]]).astype(np.float32))
            started = time.monotonic()
            token = muse.encode(Image.open(frame), "pick up the milk and place it in the basket", current_date="2026-10-30")
            if step == 0:
                with np.load("/data/libero/muse_middle_tokens_development_v1/episode_0014/000000.npz", allow_pickle=False) as saved:
                    parity = {k: bool(np.array_equal(saved[k], token[k])) for k in token}
                if not all(parity.values()): raise ValueError(f"Runtime Muse token parity failed: {parity}")
            torch.cuda.synchronize(); muse_s = time.monotonic() - started
            n = len(robots)
            ep = {"robot": np.stack(robots), "timestamps": np.arange(n, dtype=np.float64) / 20,
                  "features": np.zeros((n, 2, 5), np.float32), "valid": np.zeros((n, 2, 5), bool),
                  "actions": np.zeros((n, 7), np.float32)}
            window = observation_window(ep, n - 1, stats)
            def tensor(key): return torch.from_numpy(window[key][None]).cuda()
            bits = torch.from_numpy(token["features_bfloat16_bits"].view(np.int16)).view(torch.bfloat16)[None].cuda()
            mask = torch.from_numpy(token["attention_mask"])[None].cuda()
            started = time.monotonic()
            with torch.no_grad(), torch.autocast("cuda", enabled=False):
                context, valid = model.context(bits, mask, tensor("robot"), tensor("history_valid"),
                    tensor("history_timestamps"), tensor("sam_values"), tensor("sam_deltas"),
                    tensor("sam_valid"), tensor("sam_age"), use_sam=False)
                noise = torch.from_numpy(np.random.default_rng(7 + step).normal(size=(1, 50, 7)).astype(np.float32)).cuda()
                sampled = model.sample(context, valid, noise, steps=10)[0, 0].float().cpu().numpy()
            action = sampled * stats["actions_std"] + stats["actions_mean"]
            if not np.isfinite(action).all(): raise ValueError("Nonfinite action")
            clipped = np.clip(action, -1, 1); torch.cuda.synchronize(); policy_s = time.monotonic() - started
            started = time.monotonic(); obs, _, done, _ = env.step(clipped); success = bool(env.check_success())
            trace.append({"step": step, "sample_seed": 7 + step, "sampled_action": action.tolist(),
                          "executed_action": clipped.tolist(), "clipped": (clipped != action).tolist(),
                          "render_seconds": render_s, "muse_seconds": muse_s, "policy_seconds": policy_s,
                          "sim_seconds": time.monotonic() - started, "success": success, "done": bool(done)})
            print(json.dumps(trace[-1], allow_nan=False), flush=True)
            if success or done: break
        env.sim.forward(); rgb = np.asarray(env.sim.render(camera_name="agentview", width=512, height=512))
        Image.fromarray(rgb).save(frames / f"{len(trace):06}.jpg", quality=98)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", "20", "-i", str(frames / "%06d.jpg"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(dest / "rollout.mp4")], check=True)
        result = {"episode": "episode_0014", "demo": "demo_21", "split": "reused validation development",
                  "checkpoint_sha256": file_hash(run_path / "weights_best.pt"), "initial_state_sha256": hashlib.sha256(initial.tobytes()).hexdigest(),
                  "initial_token_cache_parity": parity, "history": "repeat first robot observation, mask padded history",
                  "sam": "not used: motor checkpoint", "monitor": "disabled", "oracle_masks": False, "muse_chat_date": "2026-10-30",
                  "executed_steps": len(trace), "success": success, "trace": trace,
                  "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                  "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3}
        (dest / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        volume.commit(); return {k: v for k, v in result.items() if k != "trace"}
    finally:
        muse.close(); env.close()


@app.local_entrypoint()
def main(max_steps: int = 10, suffix: str = "v2"): print(run.remote(max_steps, suffix))
