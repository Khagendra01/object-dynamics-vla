"""Live frozen Muse/SAM + native SmolVLA expert LIBERO development rollout."""
import modal
from scripts.modal_smolvla_reference import image as policy_base
from scripts.modal_muse_sam_paired_rollout import image as vision_base

volume = modal.Volume.from_name("object-dynamics-vla-data", create_if_missing=False)


def mounts(image):
    for name in ("__init__.py", "modal_smolvla_reference.py", "modal_muse_motor_rollout.py",
                 "modal_muse_sam_paired_rollout.py", "extract_sam31.py"):
        image = image.add_local_file(f"scripts/{name}", f"/root/scripts/{name}")
    return image.add_local_file("sam31_milk_development_config.json", "/root/sam31_milk_development_config.json")


policy_image = mounts(policy_base)
vision_image = mounts(vision_base)
app = modal.App("odvla-muse-sam-smolvla-live")


@app.cls(image=policy_image, gpu="T4", cpu=2, memory=16384, timeout=300,
         scaledown_window=120, volumes={"/data": volume}, max_containers=1)
class Policy:
    mode: str = modal.parameter(default="updated")

    @modal.enter()
    def load(self):
        from pathlib import Path
        import torch
        from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.utils.constants import ACTION, OBS_STATE
        from odvla.data import file_hash, read_json
        from odvla.vla.smolvla_bridge import MuseSamPrefix
        from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
        import numpy as np

        run = Path("/data/runs/muse_smolvla_bridge_action250v1")
        adapted = Path("/data/runs/smolvla_libero_milk1000v1")
        source = Path("/data/checkpoints/smolvla-base/5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6")
        metadata = read_json(run / "configuration.json")
        if file_hash(adapted / "weights_best.pt") != metadata["adapted_checkpoint_sha256"]:
            raise ValueError("Adapted checkpoint hash mismatch")
        if file_hash(source / "model.safetensors") != metadata["source_checkpoint_sha256"]:
            raise ValueError("Original checkpoint hash mismatch")
        cfg = PreTrainedConfig.from_pretrained(str(source)); cfg.device = "cuda"; cfg.load_vlm_weights = False
        cfg.vlm_model_name = "/data/checkpoints/smolvlm2-reference/7b375e1b73b11138ff12fe22c8f2822d8fe03467"
        cfg.input_features = {OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(15,)),
            "observation.images.agentview": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 512, 512))}
        cfg.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))}
        policy = SmolVLAPolicy.from_pretrained(str(source), config=cfg, strict=True).cuda()
        policy.load_state_dict(torch.load(adapted / "weights_best.pt", map_location="cuda", weights_only=True), strict=True)
        if self.mode not in ("updated", "native", "repaired"): raise ValueError("Invalid policy mode")
        self.policy = policy.eval(); self.model = policy.model.eval()
        checkpoint = torch.load(run / "weights_best.pt", map_location="cuda", weights_only=False)
        params = dict(self.model.named_parameters())
        prefixes = ("vlm_with_expert.lm_expert.", "state_proj.", "action_in_proj.", "action_out_proj.",
                    "action_time_mlp_in.", "action_time_mlp_out.")
        if set(checkpoint["policy_trainable"]) != {name for name in params if name.startswith(prefixes)}:
            raise ValueError("Native expert tensor set mismatch")
        if self.mode == "updated":
            with torch.no_grad():
                for name, value in checkpoint["policy_trainable"].items(): params[name].copy_(value)
        self.bridge = None
        if self.mode != "native":
            self.bridge = MuseSamPrefix().cuda().eval()
            if self.mode == "repaired":
                repair_dir = Path("/data/runs/muse_smolvla_bridge_repair1250v1")
                repair_config = read_json(repair_dir / "configuration.json")
                if repair_config["native_sha256"] != file_hash(adapted / "weights_best.pt") or repair_config["dataset_fingerprint"] != metadata["dataset_fingerprint"]:
                    raise ValueError("Repaired bridge provenance mismatch")
                repaired = repair_dir / "weights_best.pt"
                checkpoint = torch.load(repaired, map_location="cuda", weights_only=True)
                self.bridge.load_state_dict(checkpoint["bridge"], strict=True)
                checkpoint_hash = file_hash(repaired)
            else:
                self.bridge.load_state_dict(checkpoint["bridge"], strict=True)
                checkpoint_hash = file_hash(run / "weights_best.pt")
        with np.load(run / "stats.npz", allow_pickle=False) as z: self.stats = {k: z[k].copy() for k in z.files}
        self.pre, self.post = make_smolvla_pre_post_processors(cfg, {
            OBS_STATE: {"mean": torch.from_numpy(self.stats["robot_mean"]), "std": torch.from_numpy(self.stats["robot_std"])},
            ACTION: {"mean": torch.from_numpy(self.stats["actions_mean"]), "std": torch.from_numpy(self.stats["actions_std"])}})
        if self.mode == "native":
            self.info = {"checkpoint_step": 1000, "checkpoint_sha256": file_hash(adapted / "weights_best.pt"), "mode": self.mode}
        else:
            self.info = {"checkpoint_step": checkpoint["step"], "checkpoint_sha256": checkpoint_hash, "mode": self.mode}

    @modal.method()
    def predict(self, token, window, step, jpeg=None, raw_robot=None, saved_noise=None):
        import io
        import hashlib
        import numpy as np
        import torch
        import torch.nn.functional as F
        from PIL import Image
        from lerobot.utils.constants import OBS_STATE
        from odvla.vla.smolvla_bridge import expert_velocity
        if saved_noise is not None:
            if saved_noise.shape != (1, 50, 32) or saved_noise.dtype != np.float32: raise ValueError("Invalid saved noise")
            noise_hash = hashlib.sha256(saved_noise.tobytes()).hexdigest()
            noise = torch.from_numpy(saved_noise).cuda()
        else:
            generator = torch.Generator(device="cuda").manual_seed(7 + step)
            noise = torch.randn((1, 50, 32), generator=generator, device="cuda")
            noise_hash = None
        if self.mode == "native":
            if jpeg is None or raw_robot is None or saved_noise is None: raise ValueError("Native policy requires RGB, state and paired noise")
            rgb = np.asarray(Image.open(io.BytesIO(jpeg)).convert("RGB")).copy()
            batch = self.pre({OBS_STATE: torch.from_numpy(raw_robot), "observation.images.agentview": torch.from_numpy(rgb).permute(2, 0, 1),
                              "task": "pick up the milk and place it in the basket"})
            with torch.no_grad(): action = self.post(self.policy.predict_action_chunk(batch, noise=noise))[0, 0].numpy()
            if not np.isfinite(action).all(): raise ValueError("Non-finite native action")
            return {"action": action, "noise_sha256": noise_hash, **self.info}
        bits = torch.from_numpy(token["features_bfloat16_bits"].view(np.int16)).view(torch.bfloat16)[None].cuda()
        mask = torch.from_numpy(token["attention_mask"])[None].cuda()
        def tensor(key): return torch.from_numpy(window[key][None]).cuda()
        with torch.no_grad():
            if self.mode == "repaired":
                if jpeg is None or raw_robot is None: raise ValueError("Repaired policy requires raw state and RGB")
                rgb = np.asarray(Image.open(io.BytesIO(jpeg)).convert("RGB")).copy()
                state_batch = self.pre({OBS_STATE: torch.from_numpy(raw_robot), "observation.images.agentview": torch.from_numpy(rgb).permute(2, 0, 1),
                                        "task": "pick up the milk and place it in the basket"})
                state = self.model.state_proj(self.policy.prepare_state(state_batch))
            else:
                state = self.model.state_proj(F.pad(tensor("robot")[:, -1], (0, 17)))
            keys, values, valid = self.bridge(bits, mask, state, tensor("sam_values"), tensor("sam_deltas"),
                tensor("sam_valid"), tensor("sam_age"), tensor("history_valid"), use_sam=True)
            x = noise
            for k in range(10):
                x = x - expert_velocity(self.model, keys, values, valid, x,
                    torch.full((1,), 1 - k / 10, device="cuda")) / 10
            normalized = x[0, 0, :7].float().cpu().numpy()
        action = normalized * self.stats["actions_std"] + self.stats["actions_mean"]
        if not np.isfinite(action).all(): raise ValueError("Non-finite action")
        return {"action": action, "noise_sha256": noise_hash, **self.info}


@app.function(image=policy_image, cpu=2, memory=4096, timeout=120, volumes={"/data": volume})
def make_noise_bank(name: str = "matched7v1"):
    import json
    from pathlib import Path
    import numpy as np
    from odvla.data import file_hash
    if not name.isalnum(): raise ValueError("Noise bank must be alphanumeric")
    path = Path(f"/data/eval/muse_smolvla_noise_{name}.npy")
    if path.exists(): raise FileExistsError(path)
    noise = np.stack([np.random.default_rng(7 + step).normal(size=(1, 50, 32)).astype(np.float32) for step in range(200)])
    np.save(path, noise); volume.commit()
    return {"path": str(path), "sha256": file_hash(path), "shape": list(noise.shape), "seed": "7+step; NumPy"}


@app.function(image=policy_image, cpu=2, memory=4096, timeout=600, volumes={"/data": volume})
def paired_smoke(noise_bank: str = "matched7v1"):
    from pathlib import Path
    import numpy as np
    from odvla.data import read_json
    if not noise_bank.isalnum(): raise ValueError("Invalid noise bank")
    bank = np.load(f"/data/eval/muse_smolvla_noise_{noise_bank}.npy", allow_pickle=False)
    sample_path = Path("/data/eval/muse_smolvla_bridge_diagnostic_v1/frame_000000.npz")
    with np.load(sample_path, allow_pickle=False) as z: sample = {k: z[k].copy() for k in z.files}
    token = {k: sample[k] for k in ("features_bfloat16_bits", "attention_mask", "input_ids", "positions")}
    jpeg = Path("/data/eval/muse_smolvla_live_episode_0014_full200v2/frames/000000.jpg").read_bytes()
    expected = read_json("/data/eval/muse_smolvla_bridge_diagnostic_v1/results_v3.json")["frames"][0]["configurations"]
    native = Policy(mode="native").predict.remote(token, sample, 0, jpeg, sample["raw_robot"], bank[0])
    repaired = Policy(mode="repaired").predict.remote(token, sample, 0, jpeg, sample["raw_robot"], bank[0])
    np.testing.assert_allclose(native["action"], expected["native_original"]["first_action"], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(repaired["action"], expected["bridge_repaired_frozen"]["first_action"], rtol=1e-5, atol=1e-5)
    if native["noise_sha256"] != repaired["noise_sha256"]: raise ValueError("Noise hash mismatch")
    return {"native_first_action": native["action"].tolist(), "repaired_first_action": repaired["action"].tolist(),
            "noise_sha256": native["noise_sha256"], "offline_action_parity": True}


@app.function(image=vision_image, gpu="A100-80GB", cpu=4, memory=131072, volumes={"/data": volume},
              timeout=2100, max_containers=1, single_use_containers=True)
def run(max_steps: int = 1, suffix: str = "smoke1", episode: str = "episode_0014", mode: str = "updated", noise_bank: str = ""):
    import hashlib
    import json
    import subprocess
    import sys
    import time
    from pathlib import Path
    import h5py
    import numpy as np
    import torch
    import transformers
    from PIL import Image
    from robosuite.utils.transform_utils import quat2axisangle
    from odvla.data import file_hash, read_json
    from odvla.measurements import mask_metrics
    from odvla.sam_prefix import replay_current
    from odvla.sam_incremental import IncrementalPointTracker
    from odvla.vla.dataset import observation_window
    from odvla.vla.muse_context import FrozenMuseTokens, backend_snapshot
    from sam3.model_builder import build_sam3_multiplex_video_predictor
    sys.path.insert(0, "/root")
    from extract_sam31 import _patch_multiplex_init_state

    if not 1 <= max_steps <= 200 or not suffix.isalnum() or episode not in ("episode_0014", "episode_0020") or mode not in ("updated", "native", "repaired"):
        raise ValueError("Invalid bounded rollout arguments")
    if mode in ("native", "repaired") and not noise_bank.isalnum(): raise ValueError("Paired mode requires a saved noise bank")
    if noise_bank and not noise_bank.isalnum(): raise ValueError("Invalid noise bank")
    bank_path = Path(f"/data/eval/muse_smolvla_noise_{noise_bank}.npy") if noise_bank else None
    bank = np.load(bank_path, allow_pickle=False) if bank_path else None
    if bank is not None and (bank.shape != (200, 1, 50, 32) or bank.dtype != np.float32): raise ValueError("Invalid noise bank")
    dest = Path(f"/data/eval/muse_smolvla_live_{episode}_{suffix}")
    if dest.exists(): raise FileExistsError(dest)
    config = read_json("/root/sam31_milk_development_config.json")
    sam_path = Path("/data/checkpoints/sam3.1/sam3.1_multiplex.pt")
    if file_hash(sam_path) != config["checkpoint_sha256"]: raise ValueError("SAM checkpoint mismatch")
    root = Path("/opt/LIBERO"); cfg = Path.home() / ".libero/config.yaml"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(f"benchmark_root: {root}/libero/libero\nbddl_files: {root}/libero/libero/bddl_files\n"
                   f"init_states: {root}/libero/libero/init_files\ndatasets: {root}/libero/datasets\nassets: {root}/libero/libero/assets\n")
    from libero.libero.envs import OffScreenRenderEnv
    dataset = Path("/data/libero/milk_sam_development_v1")
    if episode not in read_json(dataset / "splits.json")["val"]: raise ValueError("Expected reused validation state")
    entry = next(e for e in read_json(dataset / "manifest.json")["episodes"] if e["id"] == episode)
    with h5py.File("/data/libero/source/milk_demo.hdf5", "r") as source:
        initial = np.asarray(source[f'data/{entry["source_demo"]}/states'][0]).copy()
    run_path = Path("/data/runs/muse_smolvla_bridge_action250v1")
    with np.load(run_path / "stats.npz", allow_pickle=False) as z: stats = {k: z[k].copy() for k in z.files}
    torch.set_num_threads(2); torch.manual_seed(2468); np.random.seed(2468)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    dest.mkdir(parents=True); frames = dest / "frames"; frames.mkdir()
    env = None; muse = None; incremental = None; traces = []; success = False; parity = None
    started = time.monotonic(); policy = Policy(mode=mode); backend = {"before_construction": backend_snapshot(torch)}
    try:
        if mode != "native":
            tracker = build_sam3_multiplex_video_predictor(checkpoint_path=str(sam_path), use_fa3=False,
                compile=False, warm_up=False, async_loading_frames=False)
            _patch_multiplex_init_state(tracker)
            backend["after_sam_construction"] = backend_snapshot(torch)
            muse = FrozenMuseTokens("/data/checkpoints/muse-glimmer-30b/cec94fcfd6512f2cfe708138c32533b346ea4f69", 26)
            backend["after_construction"] = backend_snapshot(torch, muse.model)
            reference = read_json("/data/libero/muse_middle_tokens_development_v1/manifest.json")
            if muse.processor_hash != reference["processor_config_sha256"] or transformers.__version__ != reference["transformers"]:
                raise ValueError("Muse processor differs from training cache")
        env = OffScreenRenderEnv(bddl_file_name=str(root / "libero/libero/bddl_files/libero_object/pick_up_the_milk_and_place_it_in_the_basket.bddl"),
            camera_heights=512, camera_widths=512, camera_names=["agentview"])
        env.reset(); obs = env.set_init_state(initial.copy()); robots = []; features = []; valids = []
        for step in range(max_steps):
            env.sim.forward(); rgb = np.asarray(env.sim.render(camera_name="agentview", width=512, height=512))
            frame = frames / f"{step:06}.jpg"; Image.fromarray(rgb).save(frame, quality=98)
            robots.append(np.concatenate([obs["robot0_joint_pos"], obs["robot0_eef_pos"],
                quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]]).astype(np.float32))
            token = {}; window = {}
            if mode != "native":
                if incremental is None:
                    incremental = IncrementalPointTracker(tracker, frame, config["first_frame_points_xy_pixels_512"], config["object_ids"], capacity=200)
                masks, ids = incremental.step(frame)
                if step in (0, 9, 29, 69, 99, 149, 199):
                    reference_masks, reference_ids = replay_current(tracker, frames, step + 1,
                        config["first_frame_points_xy_pixels_512"], config["object_ids"])
                    if not np.array_equal(ids, reference_ids) or not np.array_equal(masks, reference_masks):
                        raise ValueError("Incremental SAM differs from prefix replay")
                if not np.all(ids == 1) or not masks.any(axis=(1, 2)).all(): raise ValueError("Missing or duplicate SAM object")
                measure, valid = mask_metrics(masks[None]); features.append(measure[0]); valids.append(valid[0]); n = len(robots)
                observed = {"robot": np.stack(robots), "features": np.stack(features), "valid": np.stack(valids),
                            "timestamps": np.arange(n, dtype=np.float64) / 20, "actions": np.zeros((n, 7), np.float32)}
                window = observation_window(observed, n - 1, stats)
                with torch.autocast("cuda", enabled=False):
                    if step == 0:
                        backend["immediately_before_muse_after_sam"] = backend_snapshot(torch, muse.model)
                        (dest / "backend.json").write_text(json.dumps(backend, indent=2) + "\n")
                    token = muse.encode(Image.open(frame), "pick up the milk and place it in the basket", current_date="2026-10-30")
                if step == 0:
                    with np.load(f"/data/libero/muse_middle_tokens_development_v1/{episode}/000000.npz", allow_pickle=False) as z:
                        parity = {k: bool(np.array_equal(z[k], token[k])) for k in token}
                        cached = torch.from_numpy(z["features_bfloat16_bits"].copy().view(np.int16)).view(torch.bfloat16).float()
                        live = torch.from_numpy(token["features_bfloat16_bits"].copy().view(np.int16)).view(torch.bfloat16).float()
                        difference = (cached - live).abs()
                        parity["feature_max_abs"] = float(difference.max())
                        parity["feature_mean_abs"] = float(difference.mean())
                        parity["feature_cosine"] = float(torch.nn.functional.cosine_similarity(cached.flatten()[None], live.flatten()[None]))
                    if not all(parity[k] for k in token): raise ValueError(f"Muse cache mismatch: {parity}")
            result = policy.predict.remote(token, window, step, frame.read_bytes(), robots[-1], bank[step] if bank is not None else None)
            action = result["action"]; clipped = np.clip(action, -1, 1)
            obs, _, done, _ = env.step(clipped); success = bool(env.check_success())
            traces.append({"step": step, "sample_seed": 7 + step, "sampled_action": action.tolist(),
                           "executed_action": clipped.tolist(), "clipped_channels": (action != clipped).tolist(),
                           "success": success, "checkpoint_step": result["checkpoint_step"],
                           "checkpoint_sha256": result["checkpoint_sha256"], "noise_sha256": result["noise_sha256"]})
            (dest / "trace.json").write_text(json.dumps(traces, indent=2) + "\n")
            if (step + 1) % 25 == 0:
                volume.commit()
                print(json.dumps({"completed_steps": step + 1, "success": success, "elapsed_seconds": time.monotonic() - started}), flush=True)
            if success or done: break
        env.sim.forward(); Image.fromarray(np.asarray(env.sim.render(camera_name="agentview", width=512, height=512))).save(frames / f"{len(traces):06}.jpg", quality=98)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", "20", "-i", str(frames / "%06d.jpg"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(dest / "rollout.mp4")], check=True)
        report = {"episode": episode, "demo": entry["source_demo"], "split": "reused validation development",
                  "initial_state_sha256": hashlib.sha256(initial.tobytes()).hexdigest(), "muse_cache_parity": parity,
                  "sam": "not used" if mode == "native" else "live frozen point-prompted tracker", "oracle_masks": False, "steps": len(traces),
                  "policy_mode": mode, "noise_bank_sha256": file_hash(bank_path) if bank_path else None,
                  "success": success, "seconds": time.monotonic() - started, "peak_gpu_gib": torch.cuda.max_memory_allocated() / 1024**3,
                  "trace": traces}
        (dest / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n"); volume.commit()
        return {k: v for k, v in report.items() if k != "trace"}
    except Exception as exc:
        (dest / "failure.json").write_text(json.dumps({"error": f"{type(exc).__name__}: {exc}", "completed_steps": len(traces)}, indent=2) + "\n")
        volume.commit(); raise
    finally:
        if incremental is not None: incremental.close()
        if env is not None: env.close()
        if muse is not None: muse.close()


@app.local_entrypoint()
def main(max_steps: int = 1, suffix: str = "smoke1", episode: str = "episode_0014", mode: str = "updated", noise_bank: str = "", stage: str = "run"):
    if stage == "noise": print(make_noise_bank.remote(noise_bank))
    elif stage == "smoke": print(paired_smoke.remote(noise_bank))
    elif stage == "run": print(run.remote(max_steps, suffix, episode, mode, noise_bank))
    else: raise ValueError("Invalid stage")
