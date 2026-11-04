"""Bounded real-data ten-task integration of the frozen Muse/SAM policy.

This runs a small integration update over one new-task episode each while
retaining the original milk training split. It is not evidence of closed-loop
success or a full multi-task training campaign.
"""
import modal
from scripts.modal_quantized_muse_preflight import image, volume, MUSE_REVISION

app = modal.App("odvla-muse-ten-task-integration")
BASELINE_SHA = "fed8976c64c4b0842cb9158b675501c18d4ac8c8b2ae248e0aa1ff7947600bc2"
STATS_SHA = "e32acff474d12f798459df2ca80ac65f8fa2c7251f26d4f32d66de5e73926b54"


@app.function(image=image, gpu="H200", cpu=4, memory=131072, volumes={"/data": volume},
              timeout=3600, max_containers=1, single_use_containers=True)
def train(name: str = "muse_ten_task_integration_v1", updates: int = 2, batch_size: int = 8,
          accumulation: int = 8, max_seconds: int = 1200, cache_gib: float = 2.0, resume: bool = False):
    import hashlib
    import importlib.metadata
    import json
    import os
    import shutil
    import sys
    import time
    from pathlib import Path
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from odvla.data import file_hash, load_episode, read_json, validate_splits
    from odvla.vla.bounded_features import BoundedFeatureCache, frozen_feature_key
    from odvla.vla.dataset import observation_window
    from odvla.vla.multitask_sampling import TaskBalancedSampler
    from odvla.vla.muse_context import FrozenMuseTokens
    from odvla.vla.smolvla_bridge import expert_velocity
    from scripts.modal_muse_dataset_train import build_policy
    sys.path.insert(0, "/root")
    from train_muse_vla import make_batch

    if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
        raise ValueError("Use a simple output name")
    if not 1 <= updates <= 100 or not 1 <= batch_size <= 8 or not 1 <= accumulation <= 64:
        raise ValueError("Invalid bounded integration size")
    if batch_size * accumulation != 64 or not 60 <= max_seconds <= 2400 or not .1 <= cache_gib <= 20:
        raise ValueError("Use effective batch 64, 60..2400 seconds, and a 0.1..20 GiB cache")
    root = Path("/data/libero/ten_task_integration_v1")
    renders = read_json(root / "renders.json")
    sam = read_json(root / "sam_measurements_manifest.json")
    source_registry = read_json("/data/libero/ten_task_sources/registry.json")
    controller_audit = read_json("/data/libero/ten_task_sources/controller_audit.json")
    if renders.get("status") != "complete" or sam.get("status") != "complete" or source_registry.get("status") != "sources_prepared":
        raise ValueError("Ten-task rendered and SAM training inputs are incomplete")
    if controller_audit.get("status") != "passed" or controller_audit.get("task_count") != 10:
        raise ValueError("Ten-task declared controller/observation contract was not verified")
    sources = {item["task_id"]: item for item in source_registry["tasks"]}
    records = renders["entries"]
    if len(records) != 10 or len({r["task_id"] for r in records}) != 10 or set(sources) != {r["task_id"] for r in records}:
        raise ValueError("Expected exactly ten distinct source tasks")
    baseline = Path("/data/eval/muse_aligned_dataset_v1")
    if file_hash(baseline / "checkpoint_50.pt") != BASELINE_SHA or file_hash(baseline / "stats.npz") != STATS_SHA:
        raise ValueError("Frozen milk checkpoint or normalization changed")
    with np.load(baseline / "stats.npz", allow_pickle=False) as z:
        stats = {k: z[k].copy() for k in z.files}
    episodes, lengths, provenance = {}, {}, {}
    for record in records:
        task, episode = record["task_id"], record["episode"]
        if record["split"] != "train" or episode not in sources[task]["episodes"]["train"]:
            raise ValueError(f"Non-training episode in integration: {task}/{episode}")
        folder = root / task / episode
        with np.load(folder / "state_actions.npz", allow_pickle=False) as data:
            ep = {k: data[k].copy() for k in data.files}
        with np.load(folder / "sam_measurements.npz", allow_pickle=False) as data:
            ep.update(features=data["features"].copy(), valid=data["valid"].copy())
        n = len(ep["timestamps"])
        if (n != record["frames"] or ep["robot"].shape != (n, 15) or ep["actions"].shape != (n, 7)
                or ep["features"].shape != (n, 2, 5) or ep["valid"].shape != (n, 2, 5)
                or ep["valid"].dtype != np.bool_):
            raise ValueError(f"Aligned SAM/state/action dimensions mismatch: {task}")
        episodes[task] = {episode: {"episode": episode, "instruction": record["instruction"], "data": ep,
                                    "rgb_dir": folder / "rgb"}}
        lengths[task] = {episode: n}
        provenance[task] = {"episode": episode, "frames": n, "instruction": record["instruction"],
                            "source_sha256": record["source_sha256"], "rgb_sequence_sha256": record["rgb_sequence_sha256"],
                            "state_actions_sha256": file_hash(folder / "state_actions.npz"),
                            "sam_sha256": file_hash(folder / "sam_measurements.npz")}
    # Preserve the entire original milk training split rather than reducing the
    # previous task to just the single integration demonstration. Task-balanced
    # sampling still assigns milk the same expected task probability as each
    # new task, then chooses uniformly among its training episodes.
    milk_root = Path("/data/libero/muse_aligned_development_v1")
    milk_splits = read_json(milk_root / "splits.json")
    milk_manifest = validate_splits(milk_root, milk_splits)
    milk_entries = {item["id"]: item for item in milk_manifest["episodes"]}
    retained = []
    for episode_id in milk_splits["train"]:
        entry = milk_entries[episode_id]
        source_demo = entry["source_demo"]
        if source_demo in episodes["libero_object_milk"]:
            continue
        data = load_episode(milk_root, entry, milk_manifest, include_images=False)
        episodes["libero_object_milk"][source_demo] = {
            "episode": source_demo, "instruction": entry["instruction"], "data": data,
            "rgb_dir": Path("/data/libero/milk_sam_native_v1") / episode_id}
        lengths["libero_object_milk"][source_demo] = len(data["timestamps"])
        retained.append({"episode_id": episode_id, "source_demo": source_demo,
                         "episode_sha256": file_hash(milk_root / entry["file"])})
    if set(episodes["libero_object_milk"]) != set(sources["libero_object_milk"]["episodes"]["train"]):
        raise ValueError("Retained milk episodes do not match the audited training split")
    provenance["libero_object_milk"].update(retained_training_episodes=retained,
        aligned_dataset_fingerprint=milk_splits["fingerprint"],
        render_provenance_sha256=file_hash("/data/libero/milk_sam_native_v1/provenance.json"))
    contract = {"tasks": provenance, "controller_audit_sha256": file_hash("/data/libero/ten_task_sources/controller_audit.json"),
                "baseline_sha256": BASELINE_SHA, "stats_sha256": STATS_SHA,
                "normalize_queries": True, "dynamics_objective": False, "muse_revision": MUSE_REVISION,
                "batch_size": batch_size, "accumulation": accumulation, "cache_gib": cache_gib,
                "sam_initialization": "assistant-proposed first-frame RGB points; not autonomous or independently verified"}
    out = Path("/data/eval") / name
    if out.exists() != resume:
        raise ValueError("Use a fresh output name or explicitly resume")
    out.mkdir(parents=True, exist_ok=True)
    if not resume:
        shutil.copyfile(baseline / "stats.npz", out / "stats.npz")
    elif file_hash(out / "stats.npz") != STATS_SHA:
        raise ValueError("Saved normalization changed")
    torch.manual_seed(7)
    model, prefix, cfg = build_policy("cuda")
    model.vlm_with_expert.vlm.eval().requires_grad_(False)
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    trainable = list(prefix.parameters()) + list(params.values())
    optimizer = torch.optim.AdamW(trainable, lr=1e-4, betas=(.9, .95))
    sampler = TaskBalancedSampler(lengths, seed=7)
    checkpoint = torch.load(out / "last.pt" if resume else baseline / "checkpoint_50.pt", map_location="cpu", weights_only=True)
    if resume and checkpoint["contract"] != contract:
        raise ValueError("Resume contract changed")
    if not checkpoint.get("normalize_queries") or checkpoint.get("dynamics_objective") is not False:
        raise ValueError("Warm-start architecture/objective mismatch")
    if set(checkpoint["trainable_policy"]) != set(params):
        raise ValueError("Warm-start trainable parameter set mismatch")
    prefix.load_state_dict(checkpoint["prefix"], strict=True)
    with torch.no_grad():
        for key, value in checkpoint["trainable_policy"].items():
            params[key].copy_(value)
    step = 0
    resume_rng = None
    if resume:
        optimizer.load_state_dict(checkpoint["optimizer"])
        sampler.load_state_dict(checkpoint["sampler"])
        resume_rng = (checkpoint["torch_rng"], checkpoint["cuda_rng"])
        step = int(checkpoint["completed_updates"])
    del checkpoint
    muse = None
    started = time.monotonic()
    report = {"status": "started", "contract": contract, "target_updates": updates,
              "scope": "one new-task demo each plus the original milk training split; integration only",
              "frozen_muse": True, "frozen_sam": True, "effective_batch": batch_size * accumulation,
              "dynamics_objective": False, "milk_normalization_reused": True,
              "source_optimizer_restarted": True, "resumed_optimizer": resume, "evaluation_episodes_opened": False,
              "rollouts_executed": False, "updates": []}
    if resume and (out / "report.json").exists():
        report["updates"] = read_json(out / "report.json").get("updates", [])
    try:
        muse = FrozenMuseTokens(Path("/data/checkpoints/muse-glimmer-30b") / MUSE_REVISION, 26, quantization="nf4")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        quantized = [n for n, m in muse.model.named_modules() if type(m).__name__ == "Linear4bit"]
        if not quantized or any(not n.startswith(muse.module_name + ".") for n in quantized):
            raise ValueError("Unexpected NF4 module scope")
        backend = {"gpu": torch.cuda.get_device_name(0), "torch": str(torch.__version__),
                   "transformers": importlib.metadata.version("transformers"),
                   "bitsandbytes": importlib.metadata.version("bitsandbytes"),
                   "matmul_tf32": False, "cudnn_tf32": False, "autocast": False}
        milk = episodes["libero_object_milk"]["demo_0"]
        with Image.open(milk["rgb_dir"] / "000000.jpg") as rgb, torch.autocast("cuda", enabled=False):
            reference = muse.encode(rgb, milk["instruction"], current_date="2026-10-30")
        feature_sha = hashlib.sha256(reference["features_bfloat16_bits"].tobytes()).hexdigest()
        if feature_sha != "c0edda8a3151ff2721b8a882fd8c319d0ff2beffd40540e986c6f35e42477d24":
            raise ValueError("Live NF4 Muse features differ from the frozen milk baseline")
        alternate_instruction = next(iter(episodes["libero_object_tomato_sauce"].values()))["instruction"]
        with Image.open(milk["rgb_dir"] / "000000.jpg") as rgb, torch.autocast("cuda", enabled=False):
            alternate = muse.encode(rgb, alternate_instruction, current_date="2026-10-30")
        alternate_sha = hashlib.sha256(alternate["features_bfloat16_bits"].tobytes()).hexdigest()
        if alternate_sha == feature_sha:
            raise ValueError("Changing the instruction did not change same-scene Muse features")
        report["same_scene_instruction_probe"] = {
            "scope": "frozen Muse features only; not evidence of changed actions",
            "image_task": "libero_object_milk", "instructions": [milk["instruction"], alternate_instruction],
            "feature_sha256": [feature_sha, alternate_sha], "different": True}
        if resume_rng is not None:
            torch.set_rng_state(resume_rng[0])
            torch.cuda.set_rng_state_all(resume_rng[1])
        cache = BoundedFeatureCache("/data/libero/ten_task_bounded_nf4_v1", int(cache_gib * 1024 ** 3))
        cache_stats = {"hits": 0, "misses": 0}

        def sample_for(task, episode, t):
            item = episodes[task][episode]
            image_path = item["rgb_dir"] / f"{t:06}.jpg"
            identity = frozen_feature_key(task_id=task, episode_id=episode, frame=t,
                instruction=item["instruction"], image_sha256=file_hash(image_path),
                muse_revision=MUSE_REVISION, processor_sha256=muse.processor_hash,
                backend=backend, chat_date="2026-10-30")
            tokens = cache.get(identity)
            if tokens is None:
                cache_stats["misses"] += 1
                with Image.open(image_path) as rgb, torch.autocast("cuda", enabled=False):
                    tokens = muse.encode(rgb, item["instruction"], current_date="2026-10-30")
                cache.put(identity, tokens)
            else:
                cache_stats["hits"] += 1
            sample = observation_window(item["data"], t, stats, history=4, horizon=50)
            sample.update(muse_bits=tokens["features_bfloat16_bits"], muse_valid=tokens["attention_mask"])
            return sample

        prefix.train()
        model.vlm_with_expert.lm_expert.train()
        while step < updates and time.monotonic() - started < max_seconds:
            indices = sampler.take(batch_size * accumulation)
            denominator = sum(min(50, lengths[task][episode] - t) for task, episode, t in indices) * 7
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            for start in range(0, len(indices), batch_size):
                samples = [sample_for(task, episode, t) for task, episode, t in indices[start:start + batch_size]]
                b = make_batch(samples, "cuda")
                actions = F.pad(b["actions"], (0, cfg.max_action_dim - 7))
                noise = model.sample_noise(actions.shape, "cuda")
                time_value = model.sample_time(len(actions), "cuda")
                xt = time_value[:, None, None] * noise + (1 - time_value[:, None, None]) * actions
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    state = F.pad(b["robot"][:, -1], (0, cfg.max_state_dim - 15))
                    keys, values, mask = prefix(b["muse"], b["muse_valid"], model.state_proj(state),
                        b["sam_values"], b["sam_deltas"], b["sam_valid"], b["sam_age"], b["history_valid"], use_sam=True)
                    prediction = expert_velocity(model, keys, values, mask, xt, time_value)
                    loss = (((prediction - (noise - actions))[:, :, :7].square() * b["action_valid"][..., None]).sum()
                            / max(denominator, 1))
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite multi-task loss")
                loss.backward()
                total_loss += float(loss.detach())
            grad = torch.nn.utils.clip_grad_norm_(trainable, 10, error_if_nonfinite=True)
            optimizer.step()
            step += 1
            report["updates"].append({"step": step, "flow_loss": total_loss, "gradient_norm": float(grad),
                                      "examples_per_task": dict(sampler.examples_per_task),
                                      "cache_bytes": cache.size_bytes, "seconds": time.monotonic() - started})
            state = {"contract": contract, "completed_updates": step, "normalize_queries": True,
                "dynamics_objective": False, "prefix": {k: v.detach().cpu() for k, v in prefix.state_dict().items()},
                "trainable_policy": {k: v.detach().cpu() for k, v in params.items()}, "optimizer": optimizer.state_dict(),
                "sampler": sampler.state_dict(), "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}
            temporary = out / "last.pt.tmp"
            torch.save(state, temporary)
            os.replace(temporary, out / "last.pt")
            report.update(status="checkpointed", completed_updates=step, checkpoint_sha256=file_hash(out / "last.pt"),
                          examples_per_task=dict(sampler.examples_per_task), cache_bytes=cache.size_bytes,
                          backend=backend, processor_sha256=muse.processor_hash, milk_feature_parity=True)
            (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            volume.commit()
            print(json.dumps(report["updates"][-1]), flush=True)
        report["status"] = "completed" if step == updates else "budget_stopped"
        if step and max_seconds - (time.monotonic() - started) >= 120:
            # Diagnostic only: these are three windows from each *training*
            # episode, not held-out or closed-loop success measurements.
            prefix.eval()
            model.vlm_with_expert.lm_expert.eval()
            per_task = []
            for task, task_episodes in sorted(episodes.items()):
                item = next(iter(task_episodes.values()))
                n = len(item["data"]["timestamps"])
                indices = np.unique(np.linspace(0, n - 1, 3, dtype=int))
                samples = [sample_for(task, item["episode"], int(t)) for t in indices]
                b = make_batch(samples, "cuda")
                seed = int.from_bytes(hashlib.sha256(task.encode()).digest()[:4], "big")
                noise = np.random.default_rng(seed).standard_normal((len(indices), 50, 32), dtype=np.float32)
                action = torch.from_numpy(noise).cuda()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    state = F.pad(b["robot"][:, -1], (0, cfg.max_state_dim - 15))
                    keys, values, mask = prefix(b["muse"], b["muse_valid"], model.state_proj(state),
                        b["sam_values"], b["sam_deltas"], b["sam_valid"], b["sam_age"], b["history_valid"], use_sam=True)
                    for tick in range(10):
                        action = action - .1 * expert_velocity(model, keys, values, mask, action,
                            torch.full((len(indices),), 1 - tick / 10, device="cuda"))
                pred = action[:, 0, :7].float().cpu().numpy()
                target = b["actions"][:, 0, :7].float().cpu().numpy()
                if not np.isfinite(pred).all():
                    raise ValueError(f"Nonfinite offline action predictions: {task}")
                error = (pred - target) * stats["actions_std"][:7]
                raw_target = target * stats["actions_std"][:7] + stats["actions_mean"][:7]
                raw_pred = pred * stats["actions_std"][:7] + stats["actions_mean"][:7]
                per_task.append({"task_id": task, "episode": item["episode"], "frames": indices.tolist(),
                    "first_action_raw_mae_by_channel": np.abs(error).mean(0).tolist(),
                    "gripper_sign_correct": int(((raw_pred[:, 6] > 0) == (raw_target[:, 6] > 0)).sum()),
                    "windows": len(indices)})
            report["offline_training_observations"] = {"scope": "three windows per task from training episodes; not held out",
                                                       "noise": "NumPy task-stable seeded noise; 10 Euler steps",
                                                       "tasks": per_task}
            # Change only the Muse instruction on the same image and the same
            # robot/SAM inputs. This measures sensitivity, not target selection.
            window = observation_window(milk["data"], 0, stats, history=4, horizon=50)
            pair = []
            for tokens in (reference, alternate):
                item = dict(window)
                item.update(muse_bits=tokens["features_bfloat16_bits"], muse_valid=tokens["attention_mask"])
                pair.append(item)
            b = make_batch(pair, "cuda")
            fixed = np.random.default_rng(923).standard_normal((1, 50, 32), dtype=np.float32)
            action = torch.from_numpy(np.repeat(fixed, 2, axis=0)).cuda()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                state = F.pad(b["robot"][:, -1], (0, cfg.max_state_dim - 15))
                keys, values, mask = prefix(b["muse"], b["muse_valid"], model.state_proj(state),
                    b["sam_values"], b["sam_deltas"], b["sam_valid"], b["sam_age"], b["history_valid"], use_sam=True)
                for tick in range(10):
                    action = action - .1 * expert_velocity(model, keys, values, mask, action,
                        torch.full((2,), 1 - tick / 10, device="cuda"))
            raw = action[:, 0, :7].float().cpu().numpy() * stats["actions_std"][:7] + stats["actions_mean"][:7]
            report["same_scene_instruction_probe"].update(
                scope="same image, same robot/SAM inputs, same noise; action sensitivity only, not target selection",
                first_actions_raw=raw.tolist(), max_abs_action_difference=float(np.max(np.abs(raw[0] - raw[1]))))
        else:
            report["offline_training_observations"] = {"status": "skipped due to time budget"}
        report["cache_bytes"] = cache.size_bytes
        report["cache_stats_this_run"] = cache_stats
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        volume.commit()
        if muse is not None:
            muse.close()
    return {k: v for k, v in report.items() if k != "contract"}


@app.local_entrypoint()
def main(name: str = "muse_ten_task_integration_v1", updates: int = 2, batch_size: int = 8,
         accumulation: int = 8, max_seconds: int = 1200, cache_gib: float = 2.0,
         resume: bool = False, background: bool = False):
    args = (name, updates, batch_size, accumulation, max_seconds, cache_gib, resume)
    if background:
        print(f"Spawned ten-task integration: {train.spawn(*args).object_id}")
    else:
        print(train.remote(*args))
