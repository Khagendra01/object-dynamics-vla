"""Paired Muse/SAM development rollout with strictly observed-prefix replay."""
import modal
from scripts.modal_muse_motor_rollout import runtime_base

volume = modal.Volume.from_name("object-dynamics-vla-data", create_if_missing=False)
image = (runtime_base.run_commands("git clone https://github.com/facebookresearch/sam3.git /opt/sam3",
                                   "git -C /opt/sam3 checkout f1647e8d0c1d280991f4d1b7d974c2704537fe66")
         .pip_install("/opt/sam3")
         .pip_install("numpy<2", "transformers==5.21.0", "torch==2.10.0", "torchvision==0.25.0",
                      "opencv-python-headless<4.12", "Pillow", "einops", "pycocotools", "psutil")
         .add_local_python_source("odvla")
         .add_local_file("scripts/__init__.py", "/root/scripts/__init__.py")
         .add_local_file("scripts/modal_muse_motor_rollout.py", "/root/scripts/modal_muse_motor_rollout.py")
         .add_local_file("scripts/extract_sam31.py", "/root/extract_sam31.py")
         .add_local_file("sam31_milk_development_config.json", "/root/sam31_milk_development_config.json"))
app = modal.App("odvla-muse-sam-paired-rollout")


@app.function(image=image, gpu="A100-80GB", cpu=4, memory=131072,
              volumes={"/data": volume}, timeout=2400, max_containers=1, single_use_containers=True)
def run(max_steps: int = 30, max_seconds: int = 1800, suffix: str = "pilot30", tracking_mode: str = "prefix", train_seed: int = 7, episode: str = "episode_0014"):
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
    from odvla.vla.action_expert import MuseActionExpert
    from odvla.vla.dataset import observation_window
    from odvla.vla.muse_context import FrozenMuseTokens
    from sam3.model_builder import build_sam3_multiplex_video_predictor
    sys.path.insert(0, "/root")
    from extract_sam31 import _patch_multiplex_init_state

    if not 1 <= max_steps <= 200 or not 120 <= max_seconds <= 2100 or not suffix.isalnum() or tracking_mode not in ("prefix", "incremental") or train_seed not in (7, 19, 31) or episode not in ("episode_0014", "episode_0020", "episode_0022"):
        raise ValueError("Invalid bounded rollout arguments")
    if tracking_mode == "incremental":
        for variant in ("M1", "M2"):
            evidence = read_json(f"/data/eval/sam_incremental_probe_{variant}_70_capacity200_v1.json")
            if not evidence["all_exact"] or evidence["frames_compared"] != 70:
                raise ValueError("Incremental tracker lacks required recorded parity evidence")
    dest = Path(f"/data/eval/multitask_muse_sam_paired_{suffix}")
    if dest.exists(): raise FileExistsError(dest)
    if not read_json("/data/eval/milk_sam_prefix_benchmark_v1.json")["all_matched"]:
        raise ValueError("SAM prefix benchmark missing")
    config = read_json("/root/sam31_milk_development_config.json")
    sam_path = Path("/data/checkpoints/sam3.1/sam3.1_multiplex.pt")
    if file_hash(sam_path) != config["checkpoint_sha256"]: raise ValueError("SAM checkpoint mismatch")
    root = Path("/opt/LIBERO"); local_config = Path.home() / ".libero/config.yaml"
    local_config.parent.mkdir(parents=True, exist_ok=True)
    local_config.write_text(f"benchmark_root: {root}/libero/libero\nbddl_files: {root}/libero/libero/bddl_files\n"
                            f"init_states: {root}/libero/libero/init_files\ndatasets: {root}/libero/datasets\nassets: {root}/libero/libero/assets\n")
    from libero.libero.envs import OffScreenRenderEnv
    manifest = read_json("/data/libero/milk_sam_development_v1/manifest.json")
    entry = next(e for e in manifest["episodes"] if e["id"] == episode)
    demo = entry["source_demo"]
    with h5py.File("/data/libero/source/milk_demo.hdf5", "r") as source:
        initial = np.asarray(source[f"data/{demo}/states"][0]).copy()
    if episode not in read_json("/data/libero/milk_sam_development_v1/splits.json")["val"]:
        raise ValueError("Not the reused validation initialization")
    torch.set_num_threads(2); torch.manual_seed(2468); np.random.seed(2468)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    started = time.monotonic(); dest.mkdir(parents=True)
    tracker = None; muse = None
    try:
        tracker = build_sam3_multiplex_video_predictor(checkpoint_path=str(sam_path), use_fa3=False,
            compile=False, warm_up=False, async_loading_frames=False)
        _patch_multiplex_init_state(tracker)
        muse = FrozenMuseTokens("/data/checkpoints/muse-glimmer-30b/cec94fcfd6512f2cfe708138c32533b346ea4f69", 26)
        reference = read_json("/data/libero/muse_middle_tokens_development_v1/manifest.json")
        if muse.processor_hash != reference["processor_config_sha256"] or transformers.__version__ != reference["transformers"]:
            raise ValueError("Muse processor differs from cache")
        runs = Path(f"/data/runs/multitask_milk_matched_seed{train_seed}_v1")
        results = []
        for variant in ("M1", "M2"):
            run_path = runs / variant; model = MuseActionExpert().cuda().eval()
            model.load_state_dict(torch.load(run_path / "weights_best.pt", map_location="cuda", weights_only=True))
            with np.load(run_path / "stats.npz", allow_pickle=False) as z: stats = {k: z[k] for k in z.files}
            env = OffScreenRenderEnv(bddl_file_name=str(root / "libero/libero/bddl_files/libero_object/pick_up_the_milk_and_place_it_in_the_basket.bddl"),
                                     camera_heights=512, camera_widths=512, camera_names=["agentview"])
            env.reset(); obs = env.set_init_state(initial.copy())
            folder = dest / variant; folder.mkdir(); frames = folder / "frames"; frames.mkdir()
            robots = []; features = []; valids = []; traces = []; masks_saved = []
            failure = None; success = False; parity = None; incremental = None
            try:
                for step in range(max_steps):
                    if time.monotonic() - started > max_seconds:
                        failure = "runtime budget reached before next action"; break
                    clock = time.monotonic(); env.sim.forward()
                    rgb = np.asarray(env.sim.render(camera_name="agentview", width=512, height=512))
                    frame = frames / f"{step:06}.jpg"; Image.fromarray(rgb).save(frame, quality=98)
                    render_time = time.monotonic() - clock; clock = time.monotonic()
                    try:
                        if tracking_mode == "incremental":
                            if incremental is None:
                                incremental = IncrementalPointTracker(tracker, frame, config["first_frame_points_xy_pixels_512"], config["object_ids"], capacity=200)
                            masks, ids = incremental.step(frame)
                        else:
                            masks, ids = replay_current(tracker, frames, step + 1, config["first_frame_points_xy_pixels_512"], config["object_ids"])
                    except Exception as exc:
                        failure = f"tracker exception: {type(exc).__name__}: {exc}"
                        traces.append({"step": step, "tracking_failure": failure}); break
                    tracking_time = time.monotonic() - clock; present = masks.any(axis=(1, 2))
                    masks_saved.append(masks)
                    record = {"step": step, "id_status": ids.tolist(), "nonempty": present.tolist(),
                              "render_seconds": render_time, "tracking_seconds": tracking_time}
                    if tracking_mode == "incremental" and step in (0, 9, 29, 69, 99, 149, 199):
                        ref_clock = time.monotonic()
                        reference_masks, reference_ids = replay_current(tracker, frames, step + 1,
                            config["first_frame_points_xy_pixels_512"], config["object_ids"])
                        record["reference_seconds"] = time.monotonic() - ref_clock
                        record["reference_different_pixels"] = (masks != reference_masks).sum((1, 2)).tolist()
                        if not np.array_equal(ids, reference_ids) or not np.array_equal(masks, reference_masks):
                            failure = "incremental SAM differs from prefix replay"; record["tracking_failure"] = failure
                            traces.append(record); break
                    if not np.all(ids == 1) or not present.all():
                        failure = "missing, empty or duplicated object ID"; record["tracking_failure"] = failure
                        traces.append(record); break
                    measure, valid = mask_metrics(masks[None])
                    robots.append(np.concatenate([obs["robot0_joint_pos"], obs["robot0_eef_pos"],
                        quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]]).astype(np.float32))
                    features.append(measure[0]); valids.append(valid[0]); n = len(robots)
                    ep = {"robot": np.stack(robots), "features": np.stack(features), "valid": np.stack(valids),
                          "timestamps": np.arange(n, dtype=np.float64) / 20, "actions": np.zeros((n, 7), np.float32)}
                    window = observation_window(ep, n - 1, stats)
                    clock = time.monotonic()
                    with torch.autocast("cuda", enabled=False):
                        token = muse.encode(Image.open(frame), "pick up the milk and place it in the basket", current_date="2026-10-30")
                    if step == 0:
                        with np.load(f"/data/libero/muse_middle_tokens_development_v1/{episode}/000000.npz", allow_pickle=False) as z:
                            parity = {k: bool(np.array_equal(z[k], token[k])) for k in token}
                        if not all(parity.values()): raise ValueError(f"Muse cache/runtime mismatch: {parity}")
                    torch.cuda.synchronize(); record["muse_seconds"] = time.monotonic() - clock
                    bits = torch.from_numpy(token["features_bfloat16_bits"].view(np.int16)).view(torch.bfloat16)[None].cuda()
                    mask = torch.from_numpy(token["attention_mask"])[None].cuda()
                    def tensor(key): return torch.from_numpy(window[key][None]).cuda()
                    clock = time.monotonic()
                    with torch.no_grad(), torch.autocast("cuda", enabled=False):
                        context, cmask = model.context(bits, mask, tensor("robot"), tensor("history_valid"),
                            tensor("history_timestamps"), tensor("sam_values"), tensor("sam_deltas"), tensor("sam_valid"), tensor("sam_age"), True)
                        noise = torch.from_numpy(np.random.default_rng(7 + step).normal(size=(1, 50, 7)).astype(np.float32)).cuda()
                        normalized = model.sample(context, cmask, noise, steps=10)[0, 0].float().cpu().numpy()
                    action = normalized * stats["actions_std"] + stats["actions_mean"]
                    if not np.isfinite(action).all(): raise ValueError("Nonfinite sampled action")
                    clipped = np.clip(action, -1, 1); torch.cuda.synchronize(); record["policy_seconds"] = time.monotonic() - clock
                    clock = time.monotonic(); obs, _, done, _ = env.step(clipped); success = bool(env.check_success())
                    record.update({"sample_seed": 7 + step, "sampled_action": action.tolist(), "executed_action": clipped.tolist(),
                                   "clipped_channels": (clipped != action).tolist(), "sim_seconds": time.monotonic() - clock,
                                   "success": success, "done": bool(done)})
                    traces.append(record)
                    if (step + 1) % 10 == 0: print(json.dumps({"variant": variant, "step": step + 1, "tracking_seconds": tracking_time}), flush=True)
                    if success or done: break
                if traces and "executed_action" in traces[-1]:
                    env.sim.forward(); Image.fromarray(np.asarray(env.sim.render(camera_name="agentview", width=512, height=512))).save(frames / f"{len(traces):06}.jpg", quality=98)
                if list(frames.glob("*.jpg")):
                    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", "20", "-i", str(frames / "%06d.jpg"),
                                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(folder / "rollout.mp4")], check=True)
            finally:
                if incremental is not None: incremental.close()
                env.close()
            result = {"variant": variant, "checkpoint_sha256": file_hash(run_path / "weights_best.pt"),
                      "initial_rgb_sha256": file_hash(frames / "000000.jpg") if (frames / "000000.jpg").exists() else None,
                      "muse_cache_parity": parity, "executed_steps": sum("executed_action" in row for row in traces),
                      "success": success, "tracking_failure": failure, "trace": traces}
            (folder / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            if masks_saved: np.savez_compressed(folder / "masks.npz", masks=np.stack(masks_saved))
            results.append(result); volume.commit(); del model; torch.cuda.empty_cache()
        report = {"episode": episode, "demo": demo, "split": "reused validation development", "train_seed": train_seed,
                  "initial_state_sha256": hashlib.sha256(initial.tobytes()).hexdigest(), "max_steps": max_steps,
                  "sampling": "paired seeds 7+step; first action of 50; clipped [-1,1]", "muse_chat_date": "2026-10-30",
                  "sam": "frozen RGB tracking; fixed initial points", "tracking_mode": tracking_mode,
                  "oracle_masks": False, "monitor": "disabled",
                  "initial_rgb_identical": results[0]["initial_rgb_sha256"] == results[1]["initial_rgb_sha256"],
                  "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                  "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
                  "elapsed_seconds": time.monotonic() - started, "results": results}
        (dest / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        volume.commit(); return {**{k: v for k, v in report.items() if k != "results"},
                                  "results": [{k: v for k, v in r.items() if k != "trace"} for r in results]}
    except Exception as exc:
        (dest / "failure.json").write_text(json.dumps({"error": f"{type(exc).__name__}: {exc}",
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3}, indent=2) + "\n")
        volume.commit(); raise
    finally:
        if muse is not None: muse.close()


@app.local_entrypoint()
def main(max_steps: int = 30, max_seconds: int = 1800, suffix: str = "pilot30", tracking_mode: str = "prefix", train_seed: int = 7, episode: str = "episode_0014"):
    print(run.remote(max_steps, max_seconds, suffix, tracking_mode, train_seed, episode))
