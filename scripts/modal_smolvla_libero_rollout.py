"""SmolVLA development rollout with separate pinned policy and simulator environments."""
import modal
from scripts.modal_smolvla_reference import reference_base
from scripts.modal_muse_motor_rollout import runtime_base

volume = modal.Volume.from_name("object-dynamics-vla-data", create_if_missing=False)
def with_sources(image):
    return (image.add_local_python_source("odvla")
            .add_local_file("scripts/__init__.py", "/root/scripts/__init__.py")
            .add_local_file("scripts/modal_smolvla_reference.py", "/root/scripts/modal_smolvla_reference.py")
            .add_local_file("scripts/modal_muse_motor_rollout.py", "/root/scripts/modal_muse_motor_rollout.py"))

policy_image = with_sources(reference_base)
sim_image = with_sources(runtime_base)
app = modal.App("odvla-smolvla-libero-rollout")


@app.cls(image=sim_image, cpu=4, memory=8192, volumes={"/data": volume}, timeout=1800, max_containers=1)
class Simulator:
    @modal.method()
    def initialize(self, episode: str, suffix: str, run_name: str):
        import hashlib
        from pathlib import Path
        import h5py
        import numpy as np
        from odvla.data import read_json
        if episode not in ("episode_0014", "episode_0020") or not suffix.isalnum() or not run_name.isalnum(): raise ValueError("Only alphanumeric runs and audited validation states")
        root = Path("/opt/LIBERO"); config = Path.home() / ".libero/config.yaml"
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(f"benchmark_root: {root}/libero/libero\nbddl_files: {root}/libero/libero/bddl_files\n"
                          f"init_states: {root}/libero/libero/init_files\ndatasets: {root}/libero/datasets\nassets: {root}/libero/libero/assets\n")
        from libero.libero.envs import OffScreenRenderEnv
        dataset = Path("/data/libero/milk_sam_development_v1")
        if episode not in read_json(dataset / "splits.json")["val"]: raise ValueError("Not a validation episode")
        self.entry = next(e for e in read_json(dataset / "manifest.json")["episodes"] if e["id"] == episode)
        with h5py.File("/data/libero/source/milk_demo.hdf5", "r") as f:
            initial = np.asarray(f[f'data/{self.entry["source_demo"]}/states'][0]).copy()
        self.dest = Path(f"/data/eval/smolvla_libero_{run_name}_{episode}_{suffix}")
        if self.dest.exists(): raise FileExistsError(self.dest)
        self.dest.mkdir(parents=True); self.frames = self.dest / "frames"; self.frames.mkdir()
        self.env = OffScreenRenderEnv(bddl_file_name=str(root / "libero/libero/bddl_files/libero_object/pick_up_the_milk_and_place_it_in_the_basket.bddl"),
                                     camera_heights=512, camera_widths=512, camera_names=["agentview"])
        self.env.reset(); self.obs = self.env.set_init_state(initial); self.index = 0
        return {"episode": episode, "demo": self.entry["source_demo"], "instruction": self.entry["instruction"],
                "initial_state_sha256": hashlib.sha256(initial.tobytes()).hexdigest(), "destination": str(self.dest)}

    @modal.method()
    def observe(self):
        import hashlib
        import time
        import numpy as np
        from PIL import Image
        from robosuite.utils.transform_utils import quat2axisangle
        started = time.monotonic(); self.env.sim.forward()
        rgb = np.asarray(self.env.sim.render(camera_name="agentview", width=512, height=512))
        path = self.frames / f"{self.index:06}.jpg"; Image.fromarray(rgb).save(path, quality=98)
        robot = np.concatenate([self.obs["robot0_joint_pos"], self.obs["robot0_eef_pos"],
                                quat2axisangle(self.obs["robot0_eef_quat"]), self.obs["robot0_gripper_qpos"]]).tolist()
        data = path.read_bytes()
        return {"jpeg": data, "rgb_sha256": hashlib.sha256(data).hexdigest(), "robot": robot,
                "render_seconds": time.monotonic() - started}

    @modal.method()
    def step(self, action):
        import time
        import numpy as np
        clock = time.monotonic(); self.obs, _, done, _ = self.env.step(np.asarray(action, dtype=np.float32))
        self.index += 1
        return {"done": bool(done), "success": bool(self.env.check_success()), "sim_seconds": time.monotonic() - clock}

    @modal.method()
    def finish(self):
        import subprocess
        import numpy as np
        from PIL import Image
        self.env.sim.forward()
        Image.fromarray(np.asarray(self.env.sim.render(camera_name="agentview", width=512, height=512))).save(self.frames / f"{self.index:06}.jpg", quality=98)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", "20", "-i", str(self.frames / "%06d.jpg"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(self.dest / "rollout.mp4")], check=True)
        self.env.close(); volume.commit()


@app.function(image=policy_image, gpu="T4", cpu=4, memory=16384, volumes={"/data": volume},
              timeout=1800, max_containers=1, single_use_containers=True)
def run(episode: str = "episode_0014", max_steps: int = 200, suffix: str = "v2", run_name: str = "milk500v1"):
    import io
    import json
    import time
    from pathlib import Path
    import numpy as np
    import torch
    from PIL import Image
    from odvla.data import file_hash, read_json
    from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
    from lerobot.utils.constants import ACTION, OBS_STATE
    if not 1 <= max_steps <= 200 or not run_name.isalnum(): raise ValueError("Invalid run or development step cap")
    checkpoint = Path(f"/data/runs/smolvla_libero_{run_name}")
    metadata = read_json(checkpoint / "configuration.json")
    if metadata["dataset_fingerprint"] != read_json("/data/libero/milk_sam_development_v1/splits.json")["fingerprint"] or metadata["checkpoint_revision"] != "5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6":
        raise ValueError("Rollout checkpoint provenance mismatch")
    with np.load(checkpoint / "stats.npz", allow_pickle=False) as z: stats = {k: z[k].copy() for k in z.files}
    source = Path("/data/checkpoints/smolvla-base/5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6")
    cfg = PreTrainedConfig.from_pretrained(str(source)); cfg.device = "cuda"
    cfg.vlm_model_name = "/data/checkpoints/smolvlm2-reference/7b375e1b73b11138ff12fe22c8f2822d8fe03467"
    cfg.load_vlm_weights = False; cfg.train_expert_only = True; cfg.freeze_vision_encoder = True
    key = "observation.images.agentview"
    cfg.input_features = {OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(15,)), key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 512, 512))}
    cfg.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))}
    model = SmolVLAPolicy(cfg).cuda().eval()
    model.load_state_dict(torch.load(checkpoint / "weights_best.pt", map_location="cuda", weights_only=True), strict=True)
    stat = {OBS_STATE: {"mean": torch.from_numpy(stats["robot_mean"]), "std": torch.from_numpy(stats["robot_std"])},
            ACTION: {"mean": torch.from_numpy(stats["actions_mean"]), "std": torch.from_numpy(stats["actions_std"])}}
    pre, post = make_smolvla_pre_post_processors(cfg, stat)
    sim = Simulator(); initial = sim.initialize.remote(episode, suffix, run_name); rows = []; success = False; first_rgb = None
    try:
        for step in range(max_steps):
            transfer_clock = time.monotonic(); obs = sim.observe.remote(); observation_seconds = time.monotonic() - transfer_clock
            if first_rgb is None: first_rgb = obs["rgb_sha256"]
            rgb = np.asarray(Image.open(io.BytesIO(obs["jpeg"])).convert("RGB")).copy()
            b = pre({OBS_STATE: torch.tensor(obs["robot"], dtype=torch.float32), key: torch.from_numpy(rgb).permute(2, 0, 1), "task": initial["instruction"]})
            noise = torch.from_numpy(np.random.default_rng(7 + step).normal(size=(1, 50, 32)).astype(np.float32)).cuda()
            clock = time.monotonic()
            with torch.inference_mode(): action = post(model.predict_action_chunk(b, noise=noise))[0, 0].numpy()
            if not np.isfinite(action).all(): raise ValueError("Nonfinite action")
            clipped = np.clip(action, -1, 1); torch.cuda.synchronize(); infer_seconds = time.monotonic() - clock
            transition = sim.step.remote(clipped.tolist()); success = transition["success"]
            rows.append({"step": step, "sample_seed": 7 + step, "sampled_action": action.tolist(),
                         "executed_action": clipped.tolist(), "clipped_channels": (action != clipped).tolist(),
                         "observation_roundtrip_seconds": observation_seconds, "render_seconds": obs["render_seconds"],
                         "policy_seconds": infer_seconds, **transition})
            if (step + 1) % 25 == 0: print(json.dumps({"step": step + 1, "success": success}), flush=True)
            if success or transition["done"]: break
    finally:
        sim.finish.remote()
    volume.reload()  # The separate simulator container committed the video directory.
    result = {"episode": episode, "demo": initial["demo"], "split": "reused validation",
              "policy": "official SmolVLA pretrained robot checkpoint, fine-tuned on LIBERO OSC_POSE",
              "checkpoint_sha256": file_hash(checkpoint / "weights_best.pt"),
              "initial_state_sha256": initial["initial_state_sha256"], "initial_rgb_sha256": first_rgb,
              "sampling": "seed 7+step; replan every action", "sam": "not used", "oracle_masks": False,
              "steps": len(rows), "success": success, "peak_gpu_gib": torch.cuda.max_memory_allocated() / 1024**3, "trace": rows}
    dest = Path(initial["destination"]); (dest / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    volume.commit(); return {k: v for k, v in result.items() if k != "trace"}


@app.local_entrypoint()
def main(episode: str = "episode_0014", max_steps: int = 200, suffix: str = "v2", run_name: str = "milk500v1"): print(run.remote(episode, max_steps, suffix, run_name))
