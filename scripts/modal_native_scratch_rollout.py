"""Native scratch-expert rollout on the fitted training initialization."""
import modal
from scripts.modal_smolvla_reference import reference_base, MODEL_REVISION
from scripts.modal_muse_motor_rollout import runtime_base

volume = modal.Volume.from_name("object-dynamics-vla-data", create_if_missing=False)
policy_image = reference_base.add_local_python_source("odvla")
sim_image = runtime_base.add_local_python_source("odvla")
app = modal.App("odvla-native-scratch-rollout")


@app.cls(image=sim_image, cpu=4, memory=8192, volumes={"/data": volume},
         timeout=900, max_containers=1, scaledown_window=120)
class Simulator:
    @modal.method()
    def initialize(self, name: str):
        import hashlib
        from pathlib import Path
        import h5py
        import numpy as np
        from odvla.data import read_json
        if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
            raise ValueError("Invalid output name")
        root = Path("/opt/LIBERO")
        config = Path.home() / ".libero/config.yaml"
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(f"benchmark_root: {root}/libero/libero\nbddl_files: {root}/libero/libero/bddl_files\n"
            f"init_states: {root}/libero/libero/init_files\ndatasets: {root}/libero/datasets\nassets: {root}/libero/libero/assets\n")
        from libero.libero.envs import OffScreenRenderEnv
        dataset = Path("/data/libero/milk_sam_development_v1")
        if "episode_0000" not in read_json(dataset / "splits.json")["train"]: raise ValueError("Not a training initialization")
        self.entry = next(e for e in read_json(dataset / "manifest.json")["episodes"] if e["id"] == "episode_0000")
        with h5py.File("/data/libero/source/milk_demo.hdf5", "r") as f:
            initial = np.asarray(f[f'data/{self.entry["source_demo"]}/states'][0]).copy()
        np.random.seed(2468)
        self.dest = Path("/data/eval") / name
        self.dest.mkdir(parents=True, exist_ok=False)
        self.frames = self.dest / "frames"
        self.frames.mkdir()
        self.env = OffScreenRenderEnv(bddl_file_name=str(root / "libero/libero/bddl_files/libero_object/pick_up_the_milk_and_place_it_in_the_basket.bddl"),
            camera_heights=512, camera_widths=512, camera_names=["agentview"])
        self.env.reset()
        self.obs = self.env.set_init_state(initial.copy())
        self.index = 0
        return {"initial_state_sha256": hashlib.sha256(initial.tobytes()).hexdigest(),
            "instruction": self.entry["instruction"], "demo": self.entry["source_demo"]}

    @modal.method()
    def observe(self):
        import hashlib
        import numpy as np
        from PIL import Image
        from robosuite.utils.transform_utils import quat2axisangle
        self.env.sim.forward()
        rgb = np.asarray(self.env.sim.render(camera_name="agentview", width=512, height=512))
        path = self.frames / f"{self.index:06}.jpg"
        Image.fromarray(rgb).save(path, quality=98)
        data = path.read_bytes()
        robot = np.concatenate([self.obs["robot0_joint_pos"], self.obs["robot0_eef_pos"],
            quat2axisangle(self.obs["robot0_eef_quat"]), self.obs["robot0_gripper_qpos"]]).astype(np.float32)
        return {"jpeg": data, "rgb_sha256": hashlib.sha256(data).hexdigest(), "robot": robot.tolist()}

    @modal.method()
    def step(self, action):
        import numpy as np
        self.obs, _, done, _ = self.env.step(np.asarray(action, dtype=np.float32))
        self.index += 1
        return {"done": bool(done), "success": bool(self.env.check_success())}

    @modal.method()
    def finish(self):
        import subprocess
        import numpy as np
        from PIL import Image
        self.env.sim.forward()
        Image.fromarray(np.asarray(self.env.sim.render(camera_name="agentview", width=512, height=512))).save(
            self.frames / f"{self.index:06}.jpg", quality=98)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", "20", "-i", str(self.frames / "%06d.jpg"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(self.dest / "rollout.mp4")], check=True)
        self.env.close()
        volume.commit()


@app.function(image=policy_image, gpu="H200", cpu=4, memory=16384, volumes={"/data": volume},
              timeout=900, max_containers=1, single_use_containers=True)
def run(name: str = "native_scratch_episode0000_rollout_v1", max_steps: int = 200):
    import hashlib
    import io
    import json
    import time
    from pathlib import Path
    import numpy as np
    import torch
    from PIL import Image
    from odvla.data import file_hash, read_json
    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
    from lerobot.utils.constants import ACTION, OBS_STATE
    if not 1 <= max_steps <= 200 or not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
        raise ValueError("Invalid bounded rollout request")
    fit = Path("/data/eval/native_scratch_episode0000_v1")
    metadata = read_json(fit / "report.json")
    if metadata["status"] != "completed" or metadata["requested_updates"] != 3000 or metadata["robot_policy_checkpoint_loaded"]:
        raise ValueError("Native scratch fit provenance mismatch")
    with np.load(fit / "stats.npz", allow_pickle=False) as z: stats = {k: z[k].copy() for k in z.files}
    smol = Path("/data/checkpoints/smolvlm2-reference") / MODEL_REVISION
    if file_hash(smol / "model.safetensors") != metadata["pretrained_smolvlm_sha256"]:
        raise ValueError("SmolVLM checkpoint mismatch")
    key = "observation.images.agentview"
    cfg = SmolVLAConfig(device="cuda", vlm_model_name=str(smol), load_vlm_weights=True,
        input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(15,)),
            key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 512, 512))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))})
    torch.manual_seed(7)
    policy = SmolVLAPolicy(cfg).cuda().eval()
    path = fit / "weights.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    params = dict(policy.named_parameters())
    if set(checkpoint["trainable_policy"]) != {n for n, p in params.items() if p.requires_grad}:
        raise ValueError("Scratch trainable parameter mismatch")
    with torch.no_grad():
        for k, v in checkpoint["trainable_policy"].items(): params[k].copy_(v)
    pre, _ = make_smolvla_pre_post_processors(cfg, {
        OBS_STATE: {"mean": torch.from_numpy(stats["robot_mean"]), "std": torch.from_numpy(stats["robot_std"])},
        ACTION: {"mean": torch.from_numpy(stats["actions_mean"]), "std": torch.from_numpy(stats["actions_std"])}})
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sim = Simulator()
    started = time.monotonic()
    initial = sim.initialize.remote(name)
    baseline = read_json("/data/eval/muse_smolvla_live_episode_0000_prenormmatchedv1/report.json")
    if initial["initial_state_sha256"] != baseline["initial_state_sha256"]:
        raise ValueError("Initial simulator state differs from Muse rollout")
    rows, success = [], False
    first_rgb = None
    try:
        for step in range(max_steps):
            obs = sim.observe.remote()
            if first_rgb is None:
                first_rgb = obs["rgb_sha256"]
                if first_rgb != metadata["frame_sha256"][0]: raise ValueError("Initial RGB differs from training frame")
            with Image.open(io.BytesIO(obs["jpeg"])) as img: rgb = np.asarray(img.convert("RGB")).copy()
            b = pre({OBS_STATE: torch.tensor(obs["robot"], dtype=torch.float32),
                key: torch.from_numpy(rgb).permute(2, 0, 1), "task": initial["instruction"]})
            generator = torch.Generator(device="cuda").manual_seed(7 + step)
            noise = torch.randn((1, 50, cfg.max_action_dim), generator=generator, device="cuda")
            noise_hash = hashlib.sha256(noise.cpu().numpy().tobytes()).hexdigest()
            if baseline["trace"][step]["noise_sha256"] != noise_hash:
                raise ValueError(f"Noise bytes differ from Muse rollout at step {step}")
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                normalized = policy.predict_action_chunk(b, noise=noise)[0, 0].float().cpu().numpy()
            action = normalized * stats["actions_std"] + stats["actions_mean"]
            if not np.isfinite(action).all(): raise ValueError("Nonfinite sampled action")
            clipped = np.clip(action, -1, 1)
            transition = sim.step.remote(clipped.tolist())
            success = transition["success"]
            rows.append({"step": step, "sample_seed": 7 + step, "noise_sha256": noise_hash,
                "sampled_action": action.tolist(), "executed_action": clipped.tolist(),
                "clipped_channels": (clipped != action).tolist(), **transition})
            if (step+1) % 25 == 0:
                print(json.dumps({"steps": step+1, "success": success, "elapsed_seconds": time.monotonic()-started}), flush=True)
            if success or transition["done"]: break
    finally:
        sim.finish.remote()
    volume.reload()
    result = {"episode": "episode_0000", "demo": initial["demo"], "split": "training initialization; in-sample",
        "policy": "official native SmolVLA, frozen pretrained SmolVLM, fresh action expert trained on one episode",
        "robot_pretrained_checkpoint_loaded": False, "sam_used": False, "dynamics_objective": False,
        "checkpoint_sha256": file_hash(path), "checkpoint_step": checkpoint["completed_updates"],
        "initial_state_sha256": initial["initial_state_sha256"], "initial_rgb_sha256": first_rgb,
        "sampling": "CUDA torch seed 7+step, exact noise hashes checked against Muse normalized rollout",
        "steps": len(rows), "success": success, "elapsed_seconds": time.monotonic()-started,
        "peak_gpu_gib": torch.cuda.max_memory_allocated()/1024**3, "trace": rows}
    (Path("/data/eval") / name / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    volume.commit()
    return {k: v for k, v in result.items() if k != "trace"}


@app.local_entrypoint()
def main(name: str = "native_scratch_episode0000_rollout_v1", max_steps: int = 200):
    print(run.remote(name, max_steps))
