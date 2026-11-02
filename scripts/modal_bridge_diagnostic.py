"""Paired-noise offline ablation on recorded combined-policy trajectory."""
import modal
from scripts.modal_muse_smolvla_live import policy_image, vision_image

volume = modal.Volume.from_name("object-dynamics-vla-data", create_if_missing=False)
policy_image = policy_image.add_local_file("scripts/modal_muse_smolvla_live.py", "/root/scripts/modal_muse_smolvla_live.py")
vision_image = vision_image.add_local_file("scripts/modal_muse_smolvla_live.py", "/root/scripts/modal_muse_smolvla_live.py")
app = modal.App("odvla-bridge-diagnostic")
ROOT = "/data/eval/muse_smolvla_bridge_diagnostic_v1"
FRAMES = (0, 30, 70, 120, 160, 199)


@app.function(image=vision_image, gpu="A100-80GB", cpu=4, memory=131072, timeout=900,
              volumes={"/data": volume}, max_containers=1, single_use_containers=True)
def prepare():
    import hashlib
    import io
    import json
    import shutil
    import sys
    import tempfile
    from pathlib import Path
    import h5py
    import numpy as np
    import torch
    import transformers
    from PIL import Image
    from robosuite.utils.transform_utils import quat2axisangle
    from odvla.data import file_hash, read_json
    from odvla.measurements import mask_metrics
    from odvla.sam_incremental import IncrementalPointTracker
    from odvla.sam_prefix import replay_current
    from odvla.vla.dataset import observation_window
    from odvla.vla.muse_context import FrozenMuseTokens, backend_snapshot
    from sam3.model_builder import build_sam3_multiplex_video_predictor
    sys.path.insert(0, "/root")
    from extract_sam31 import _patch_multiplex_init_state
    dest = Path(ROOT)
    if dest.exists(): raise FileExistsError(dest)
    source = Path("/data/eval/muse_smolvla_live_episode_0014_full200v2")
    trace = read_json(source / "trace.json")
    if len(trace) != 200 or any(row["step"] != i for i, row in enumerate(trace)):
        raise ValueError("Expected complete recorded rollout trace")
    config = read_json("/root/sam31_milk_development_config.json")
    sam_path = Path("/data/checkpoints/sam3.1/sam3.1_multiplex.pt")
    if file_hash(sam_path) != config["checkpoint_sha256"]: raise ValueError("SAM checkpoint mismatch")
    with np.load("/data/runs/muse_smolvla_bridge_action250v1/stats.npz", allow_pickle=False) as z:
        stats = {k: z[k].copy() for k in z.files}
    root = Path("/opt/LIBERO"); cfg = Path.home() / ".libero/config.yaml"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(f"benchmark_root: {root}/libero/libero\nbddl_files: {root}/libero/libero/bddl_files\n"
                   f"init_states: {root}/libero/libero/init_files\ndatasets: {root}/libero/datasets\nassets: {root}/libero/libero/assets\n")
    from libero.libero.envs import OffScreenRenderEnv
    with h5py.File("/data/libero/source/milk_demo.hdf5", "r") as f:
        initial = np.asarray(f["data/demo_21/states"][0]).copy()
    torch.set_num_threads(2); torch.manual_seed(2468); np.random.seed(2468)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    backend = {"before_construction": backend_snapshot(torch)}
    dest.mkdir(parents=True); records = []; env = tracker_state = muse = None
    try:
        tracker = build_sam3_multiplex_video_predictor(checkpoint_path=str(sam_path), use_fa3=False,
            compile=False, warm_up=False, async_loading_frames=False)
        _patch_multiplex_init_state(tracker)
        backend["after_sam_construction"] = backend_snapshot(torch)
        muse = FrozenMuseTokens("/data/checkpoints/muse-glimmer-30b/cec94fcfd6512f2cfe708138c32533b346ea4f69", 26)
        ref = read_json("/data/libero/muse_middle_tokens_development_v1/manifest.json")
        if muse.processor_hash != ref["processor_config_sha256"] or transformers.__version__ != ref["transformers"]:
            raise ValueError("Muse processor mismatch")
        backend["after_construction"] = backend_snapshot(torch, muse.model)
        env = OffScreenRenderEnv(bddl_file_name=str(root / "libero/libero/bddl_files/libero_object/pick_up_the_milk_and_place_it_in_the_basket.bddl"),
                                 camera_heights=512, camera_widths=512, camera_names=["agentview"])
        env.reset(); obs = env.set_init_state(initial)
        robots = []; features = []; valids = []
        for step in range(max(FRAMES) + 1):
            frame = source / "frames" / f"{step:06}.jpg"
            if tracker_state is None:
                tracker_state = IncrementalPointTracker(tracker, frame, config["first_frame_points_xy_pixels_512"], config["object_ids"], capacity=200)
            masks, ids = tracker_state.step(frame)
            if not np.all(ids == 1) or not masks.any(axis=(1, 2)).all(): raise ValueError(f"SAM failure at {step}")
            measure, valid = mask_metrics(masks[None])
            robots.append(np.concatenate([obs["robot0_joint_pos"], obs["robot0_eef_pos"],
                quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]]).astype(np.float32))
            features.append(measure[0]); valids.append(valid[0]); n = len(robots)
            if step in FRAMES:
                env.sim.forward(); rgb = np.asarray(env.sim.render(camera_name="agentview", width=512, height=512))
                rendered = io.BytesIO(); Image.fromarray(rgb).save(rendered, format="JPEG", quality=98)
                actual_hash = hashlib.sha256(rendered.getvalue()).hexdigest(); saved_hash = file_hash(frame)
                if actual_hash != saved_hash:
                    raise ValueError(f"Replay RGB mismatch at {step}: {actual_hash} != {saved_hash}")
                if step in (0, 70, 199):
                    with tempfile.TemporaryDirectory() as temp:
                        for i in range(step + 1):
                            shutil.copyfile(source / "frames" / f"{i:06}.jpg", Path(temp) / f"{i:06}.jpg")
                        ref_masks, ref_ids = replay_current(tracker, temp, step + 1,
                            config["first_frame_points_xy_pixels_512"], config["object_ids"])
                    if not np.array_equal(ids, ref_ids) or not np.array_equal(masks, ref_masks):
                        raise ValueError(f"SAM prefix mismatch at {step}")
                observed = {"robot": np.stack(robots), "features": np.stack(features), "valid": np.stack(valids),
                    "timestamps": np.arange(n, dtype=np.float64) / 20, "actions": np.zeros((n, 7), np.float32)}
                window = observation_window(observed, n - 1, stats)
                with torch.autocast("cuda", enabled=False):
                    backend[str(step)] = backend_snapshot(torch, muse.model)
                    token = muse.encode(Image.open(frame), "pick up the milk and place it in the basket", current_date="2026-10-30")
                noise = np.random.default_rng(7 + step).normal(size=(1, 50, 32)).astype(np.float32)
                np.savez_compressed(dest / f"frame_{step:06}.npz", **window, **token, noise=noise,
                                    raw_robot=robots[-1], rgb_sha256=saved_hash)
                records.append({"step": step, "rgb_sha256": saved_hash, "frame": str(frame),
                                "observation_sha256": file_hash(dest / f"frame_{step:06}.npz")})
                (dest / "backend.json").write_text(json.dumps(backend, indent=2) + "\n")
                volume.commit(); print(json.dumps({"prepared": step, "rgb_parity": True}), flush=True)
            if step < max(FRAMES): obs, _, _, _ = env.step(np.asarray(trace[step]["executed_action"], np.float32))
        manifest = {"source": str(source), "trace_sha256": file_hash(source / "trace.json"),
                    "source_kind": "replayed recorded executed actions; RGB exact at selected frames", "records": records,
                    "noise": "saved numpy.default_rng(7+step), identical across all configurations", "sam": "frozen, prefix-only"}
        (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n"); volume.commit(); return manifest
    except Exception as exc:
        (dest / "failure.json").write_text(json.dumps({"error": f"{type(exc).__name__}: {exc}", "records": records}, indent=2) + "\n")
        volume.commit(); raise
    finally:
        if tracker_state is not None: tracker_state.close()
        if env is not None: env.close()
        if muse is not None: muse.close()


@app.function(image=policy_image, gpu="T4", cpu=4, memory=24576, timeout=900,
              volumes={"/data": volume}, max_containers=1, single_use_containers=True)
def evaluate(suffix: str = "v3"):
    import hashlib
    import json
    from pathlib import Path
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from odvla.data import file_hash, read_json
    from odvla.vla.smolvla_bridge import MuseSamPrefix, expert_velocity
    from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, make_att_2d_masks
    from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
    from lerobot.utils.constants import ACTION, OBS_STATE, OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK

    dest = Path(ROOT); volume.reload()
    manifest = read_json(dest / "manifest.json")
    if not suffix.isalnum(): raise ValueError("Suffix must be alphanumeric")
    output = dest / f"results_{suffix}.json"
    if output.exists(): raise FileExistsError(output)
    source = Path("/data/checkpoints/smolvla-base/5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6")
    adapted = Path("/data/runs/smolvla_libero_milk1000v1")
    trained = Path("/data/runs/muse_smolvla_bridge_action250v1")
    aligned = Path("/data/runs/muse_smolvla_bridge_alignment250v1")
    repaired = Path("/data/runs/muse_smolvla_bridge_repair1250v1")
    meta = read_json(trained / "configuration.json")
    if file_hash(source / "model.safetensors") != meta["source_checkpoint_sha256"] or file_hash(adapted / "weights_best.pt") != meta["adapted_checkpoint_sha256"]:
        raise ValueError("Native checkpoint provenance mismatch")
    with np.load(adapted / "stats.npz", allow_pickle=False) as z: stats = {k: z[k].copy() for k in z.files}
    cfg = PreTrainedConfig.from_pretrained(str(source)); cfg.device = "cuda"; cfg.load_vlm_weights = False
    cfg.vlm_model_name = "/data/checkpoints/smolvlm2-reference/7b375e1b73b11138ff12fe22c8f2822d8fe03467"
    key = "observation.images.agentview"
    cfg.input_features = {OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(15,)), key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 512, 512))}
    cfg.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))}
    policy = SmolVLAPolicy.from_pretrained(str(source), config=cfg, strict=True).cuda()
    policy.load_state_dict(torch.load(adapted / "weights_best.pt", map_location="cuda", weights_only=True), strict=True)
    policy.eval(); model = policy.model
    updated = torch.load(trained / "weights_best.pt", map_location="cuda", weights_only=False)
    aligned_state = torch.load(aligned / "training_state.pt", map_location="cuda", weights_only=False)
    params = dict(model.named_parameters())
    prefixes = ("vlm_with_expert.lm_expert.", "state_proj.", "action_in_proj.", "action_out_proj.",
                "action_time_mlp_in.", "action_time_mlp_out.")
    if set(updated["policy_trainable"]) != {n for n in params if n.startswith(prefixes)}:
        raise ValueError("Incomplete updated expert tensor set")
    original = {n: p.detach().clone() for n, p in params.items() if n in updated["policy_trainable"]}
    bridges = {"aligned": MuseSamPrefix().cuda().eval(), "updated": MuseSamPrefix().cuda().eval(),
               "repaired": MuseSamPrefix().cuda().eval()}
    bridges["aligned"].load_state_dict(aligned_state["bridge"], strict=True)
    bridges["updated"].load_state_dict(updated["bridge"], strict=True)
    bridges["repaired"].load_state_dict(torch.load(repaired / "weights_best.pt", map_location="cuda", weights_only=True)["bridge"], strict=True)
    pre, _ = make_smolvla_pre_post_processors(cfg, {
        OBS_STATE: {"mean": torch.from_numpy(stats["robot_mean"]), "std": torch.from_numpy(stats["robot_std"])},
        ACTION: {"mean": torch.from_numpy(stats["actions_mean"]), "std": torch.from_numpy(stats["actions_std"])}})
    def apply(which):
        tensors = original if which == "original" else updated["policy_trainable"]
        for name, value in tensors.items(): params[name].copy_(value)
    results = []
    with torch.no_grad():
        for record in manifest["records"]:
            step = record["step"]; path = dest / f"frame_{step:06}.npz"
            if file_hash(path) != record["observation_sha256"] or file_hash(record["frame"]) != record["rgb_sha256"]:
                raise ValueError("Saved observation provenance mismatch")
            with np.load(path, allow_pickle=False) as z: sample = {k: z[k].copy() for k in z.files}
            rgb = np.asarray(Image.open(record["frame"]).convert("RGB")).copy()
            teacher = pre({OBS_STATE: torch.from_numpy(sample["raw_robot"]), key: torch.from_numpy(rgb).permute(2, 0, 1),
                           "task": "pick up the milk and place it in the basket"})
            state = policy.prepare_state(teacher)
            images, img_masks = policy.prepare_images(teacher)
            noise = torch.from_numpy(sample["noise"]).cuda()
            if tuple(noise.shape) != (1, 50, 32): raise ValueError("Bad stored noise")
            configs = (("native_original", "original", "native"), ("bridge_frozen", "original", "aligned"),
                       ("bridge_repaired_frozen", "original", "repaired"), ("native_updated", "updated", "native"),
                       ("bridge_updated", "updated", "updated"))
            outputs = {}; reference_x = []; reference_v = []; official_error = None
            for name, version, conditioning in configs:
                apply(version)
                if conditioning == "native":
                    prefix, pmask, par = model.embed_prefix(images, img_masks, teacher[OBS_LANGUAGE_TOKENS], teacher[OBS_LANGUAGE_ATTENTION_MASK], state)
                    _, cache = model.vlm_with_expert.forward(attention_mask=make_att_2d_masks(pmask, par),
                        position_ids=torch.cumsum(pmask, dim=1) - 1, inputs_embeds=[prefix, None], use_cache=True)
                    keys = [layer.keys for layer in cache.layers]; values = [layer.values for layer in cache.layers]; mask = pmask
                else:
                    bits = torch.from_numpy(sample["features_bfloat16_bits"].view(np.int16)).view(torch.bfloat16)[None].cuda()
                    valid = torch.from_numpy(sample["attention_mask"])[None].cuda()
                    def t(k): return torch.from_numpy(sample[k][None]).cuda()
                    state_embedding = model.state_proj(state)
                    keys, values, mask = bridges[conditioning](bits, valid, state_embedding, t("sam_values"), t("sam_deltas"),
                        t("sam_valid"), t("sam_age"), t("history_valid"), use_sam=True)
                x = noise.clone(); own_v = []; on_reference_v = []
                for k in range(10):
                    time = torch.full((1,), 1 - k / 10, device="cuda")
                    if name == "native_original": reference_x.append(x.clone())
                    v = expert_velocity(model, keys, values, mask, x, time)
                    own_v.append(v[:, :, :7].cpu().numpy())
                    if name == "native_original":
                        reference_v.append(v[:, :, :7].cpu().numpy()); on_reference_v.append(reference_v[-1])
                    else:
                        on_reference_v.append(expert_velocity(model, keys, values, mask, reference_x[k], time)[:, :, :7].cpu().numpy())
                    x = x - v / 10
                if name == "native_original":
                    official = policy.predict_action_chunk(teacher, noise=noise.clone())
                    torch.testing.assert_close(x[:, :, :7], official, atol=2e-4, rtol=2e-4)
                    official_error = float((x[:, :, :7] - official).abs().max())
                action = x[0, 0, :7].cpu().numpy() * stats["actions_std"] + stats["actions_mean"]
                outputs[name] = {"first_action": action.tolist(), "own_velocity": np.stack(own_v),
                                 "reference_path_velocity": np.stack(on_reference_v)}
            reference = np.stack(reference_v)
            arrays = {"noise": sample["noise"], "reference_x": torch.stack(reference_x).cpu().numpy()}
            row = {"step": step, "noise_sha256": hashlib.sha256(sample["noise"].tobytes()).hexdigest(),
                   "observation_sha256": file_hash(path), "rgb_sha256": record["rgb_sha256"],
                   "native_official_max_abs": official_error, "configurations": {}}
            for name, data in outputs.items():
                arrays[f"{name}_own_velocity"] = data["own_velocity"]
                arrays[f"{name}_reference_path_velocity"] = data["reference_path_velocity"]
                arrays[f"{name}_first_action"] = np.array(data["first_action"], np.float32)
                delta = data["reference_path_velocity"] - reference
                row["configurations"][name] = {"first_action": data["first_action"],
                    "first_action_delta_vs_native": (np.array(data["first_action"]) - outputs["native_original"]["first_action"]).tolist(),
                    "reference_path_velocity_mse_by_denoise_step": np.mean(delta ** 2, axis=(1, 2, 3)).tolist()}
            np.savez_compressed(dest / f"outputs_{step:06}_{suffix}.npz", **arrays)
            results.append(row); print(json.dumps(row), flush=True)
    result = {"source": manifest, "frames": results, "note": "reused failed-rollout observations; offline only; same stored noise per configuration",
              "updated": "expert plus state/action/time projections changed during joint adaptation",
              "native_checkpoint_sha256": file_hash(adapted / "weights_best.pt"),
              "aligned_bridge_sha256": file_hash(aligned / "training_state.pt"), "updated_checkpoint_sha256": file_hash(trained / "weights_best.pt"),
              "repaired_bridge_sha256": file_hash(repaired / "weights_best.pt")}
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n"); volume.commit(); return result


@app.function(image=policy_image, cpu=2, memory=8192, timeout=120, volumes={"/data": volume})
def inspect_first_frame():
    import numpy as np
    import torch
    from odvla.data import read_json
    with np.load(f"{ROOT}/frame_000000.npz", allow_pickle=False) as a, np.load(
            "/data/libero/muse_middle_tokens_development_v1/episode_0014/000000.npz", allow_pickle=False) as b:
        parity = {k: bool(np.array_equal(a[k], b[k])) for k in ("features_bfloat16_bits", "attention_mask", "input_ids", "positions")}
        x = torch.from_numpy(a["features_bfloat16_bits"].copy().view(np.int16)).view(torch.bfloat16).float()
        y = torch.from_numpy(b["features_bfloat16_bits"].copy().view(np.int16)).view(torch.bfloat16).float()
        parity["mean_absolute_difference"] = float((x-y).abs().mean())
        parity["maximum_absolute_difference"] = float((x-y).abs().max())
    return {"cache_parity": parity, "backend": read_json(f"{ROOT}/backend.json")}


@app.local_entrypoint()
def main(stage: str = "prepare"):
    if stage == "prepare": print(prepare.remote())
    elif stage == "evaluate": print(evaluate.remote())
    elif stage == "inspect": print(inspect_first_frame.remote())
    else: raise ValueError("stage must be prepare, evaluate or inspect")
