"""Task-balanced multi-demonstration training with bounded RAM/disk features."""
import modal
from scripts.modal_quantized_muse_preflight import image, volume, MUSE_REVISION

app = modal.App("odvla-muse-multidemo-campaign")
SOURCE_SHA = "7c86adc0fa8f941b8deb21048aa36efa54bdc190ca7dba7e1843922a55c4d39e"
STATS_SHA = "e32acff474d12f798459df2ca80ac65f8fa2c7251f26d4f32d66de5e73926b54"


@app.function(image=image, cpu=2, memory=8192, volumes={"/data": volume}, timeout=600)
def audit():
    from pathlib import Path
    from odvla.data import write_json
    from odvla.vla.multidemo import load_multidemo
    episodes, lengths, provenance = load_multidemo("/data", demos_per_task=3, expected_tasks=10)
    report = {"status": "ready", "task_count": len(episodes),
              "episodes_per_task": {task: len(items) for task, items in episodes.items()},
              "frames_per_task": {task: sum(counts.values()) for task, counts in lengths.items()},
              "chocolate_target_masked": all(not item["data"]["valid"][:, 0, :].any()
                                              for item in episodes["libero_object_chocolate_pudding"].values()),
              "held_out_episodes_loaded_for_optimization": False,
              "split_audit_scope": "milk split validation hashes train/development files; only training arrays are loaded",
              "source_dataset_episodes": provenance}
    output = Path("/data/eval/muse_ten_task_multidemo_audit_v1.json")
    write_json(output, report)
    volume.commit()
    return {k: v for k, v in report.items() if k != "source_dataset_episodes"}


@app.function(image=image, gpu="H200", cpu=4, memory=131072, volumes={"/data": volume},
              timeout=3600, max_containers=1, single_use_containers=True)
def train(name: str = "muse_ten_task_multidemo_v1", updates: int = 10, batch_size: int = 8,
          accumulation: int = 8, max_seconds: int = 900, disk_gib: float = 20, ram_mib: int = 512,
          save_every: int = 5, resume: bool = False):
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
    from odvla.data import file_hash, read_json
    from odvla.vla.bounded_features import TieredFeatureCache, frozen_feature_key
    from odvla.vla.dataset import observation_window
    from odvla.vla.multidemo import load_multidemo
    from odvla.vla.multitask_sampling import TaskBalancedSampler
    from odvla.vla.muse_context import FrozenMuseTokens
    from odvla.vla.smolvla_bridge import expert_velocity
    from scripts.modal_muse_dataset_train import build_policy
    sys.path.insert(0, "/root")
    from train_muse_vla import make_batch

    if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
        raise ValueError("Use a simple output name")
    if not 1 <= updates <= 1000 or not 1 <= batch_size <= 8 or batch_size * accumulation != 64:
        raise ValueError("Expected a bounded run with effective batch 64")
    if not 60 <= max_seconds <= 2400 or not .1 <= disk_gib <= 20 or not 0 <= ram_mib <= 2048 or save_every < 1:
        raise ValueError("Invalid runtime, cache or checkpoint bound")
    root = Path("/data")
    episodes, lengths, provenance = load_multidemo(root, demos_per_task=3, expected_tasks=10)
    source = root / "eval/muse_ten_task_frozen_v1"
    if file_hash(source / "checkpoint_20.pt") != SOURCE_SHA or file_hash(source / "stats.npz") != STATS_SHA:
        raise ValueError("Frozen ten-task checkpoint or milk normalization changed")
    with np.load(source / "stats.npz", allow_pickle=False) as saved:
        stats = {k: saved[k].copy() for k in saved.files}
    contract = {"tasks": provenance, "source_checkpoint_sha256": SOURCE_SHA, "stats_sha256": STATS_SHA,
        "controller_audit_sha256": file_hash(root / "libero/ten_task_sources/controller_audit.json"),
        "batch_size": batch_size, "accumulation": accumulation, "disk_gib": disk_gib, "ram_mib": ram_mib,
        "muse_revision": MUSE_REVISION, "normalize_queries": True, "dynamics_objective": False,
        "sam_initialization": "assistant-proposed RGB points; chocolate target unverified and masked"}
    out = root / "eval" / name
    if out.exists() != resume:
        raise ValueError("Use a fresh name or explicitly resume")
    out.mkdir(parents=True, exist_ok=True)
    if not resume:
        shutil.copyfile(source / "stats.npz", out / "stats.npz")
    elif file_hash(out / "stats.npz") != STATS_SHA:
        raise ValueError("Saved normalization changed")
    if resume:
        previous_report = read_json(out / "report.json")
        if file_hash(out / "last.pt") != previous_report.get("checkpoint_sha256"):
            raise ValueError("Resume checkpoint does not match the committed report")
    torch.manual_seed(7)
    model, prefix, cfg = build_policy("cuda")
    model.vlm_with_expert.vlm.eval().requires_grad_(False)
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    trainable = list(prefix.parameters()) + list(params.values())
    optimizer = torch.optim.AdamW(trainable, lr=1e-4, betas=(.9, .95))
    sampler = TaskBalancedSampler(lengths, seed=17)
    checkpoint = torch.load(out / "last.pt" if resume else source / "checkpoint_20.pt", map_location="cpu", weights_only=True)
    if resume and checkpoint["contract"] != contract:
        raise ValueError("Resume dataset/cache contract changed")
    if not checkpoint.get("normalize_queries") or checkpoint.get("dynamics_objective") is not False:
        raise ValueError("Source architecture/objective mismatch")
    if set(checkpoint["trainable_policy"]) != set(params):
        raise ValueError("Source parameter set mismatch")
    prefix.load_state_dict(checkpoint["prefix"], strict=True)
    with torch.no_grad():
        for key, value in checkpoint["trainable_policy"].items():
            params[key].copy_(value)
    optimizer.load_state_dict(checkpoint["optimizer"])
    step = 0
    seen = {task: set() for task in lengths}
    per_episode = {task: {} for task in lengths}
    resume_rng = None
    if resume:
        sampler.load_state_dict(checkpoint["sampler"])
        step = int(checkpoint["completed_updates"])
        if updates <= step:
            raise ValueError("Resume target must exceed the completed update count")
        seen = {task: set(items) for task, items in checkpoint["seen_windows"].items()}
        per_episode = checkpoint["presentations_per_episode"]
        resume_rng = (checkpoint["torch_rng"], checkpoint["cuda_rng"])
    del checkpoint
    started = time.monotonic()
    report = {"status": "started", "contract": contract, "phase_updates_target": updates,
        "scope": "three training demos for each new task plus original 34 milk training demos",
        "source_optimizer_continued": True, "source_sampler_restarted_for_expanded_data": True,
        "exact_resume_of_old_dataset": False, "resumed_current_campaign": resume,
        "frozen_muse": True, "frozen_sam": True, "dynamics_objective": False,
        "evaluation_episodes_opened": False, "rollouts_executed": False, "updates": []}
    if resume and (out / "report.json").exists():
        report["updates"] = [row for row in previous_report.get("updates", []) if row["step"] <= step]

    def write_report():
        temporary = out / "report.json.tmp"
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        os.replace(temporary, out / "report.json")
    muse = None
    timings = {"feature_extraction_seconds": 0.0, "cache_lookup_seconds": 0.0}
    try:
        muse = FrozenMuseTokens(root / "checkpoints/muse-glimmer-30b" / MUSE_REVISION, 26, quantization="nf4")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        converted = [n for n, module in muse.model.named_modules() if type(module).__name__ == "Linear4bit"]
        if not converted or any(not n.startswith(muse.module_name + ".") for n in converted):
            raise ValueError("Unexpected NF4 conversion scope")
        backend = {"gpu": torch.cuda.get_device_name(0), "torch": str(torch.__version__),
            "transformers": importlib.metadata.version("transformers"),
            "bitsandbytes": importlib.metadata.version("bitsandbytes"), "matmul_tf32": False,
            "cudnn_tf32": False, "autocast": False}
        milk = episodes["libero_object_milk"]["demo_0"]
        with Image.open(milk["rgb_dir"] / "000000.jpg") as rgb, torch.autocast("cuda", enabled=False):
            reference = muse.encode(rgb, milk["instruction"], current_date="2026-10-30")
        if hashlib.sha256(reference["features_bfloat16_bits"].tobytes()).hexdigest() != "c0edda8a3151ff2721b8a882fd8c319d0ff2beffd40540e986c6f35e42477d24":
            raise ValueError("Frozen Muse no longer matches the milk reference")
        if resume_rng is not None:
            torch.set_rng_state(resume_rng[0])
            torch.cuda.set_rng_state_all(resume_rng[1])
        else:
            torch.manual_seed(17)
        cache = TieredFeatureCache(root / "libero/ten_task_bounded_nf4_v1", int(disk_gib * 1024 ** 3), ram_mib * 1024 ** 2)

        def sample_for(task, episode, t):
            item = episodes[task][episode]
            image_path = item["rgb_dir"] / f"{t:06}.jpg"
            key = frozen_feature_key(task_id=task, episode_id=episode, frame=t,
                instruction=item["instruction"], image_sha256=file_hash(image_path), muse_revision=MUSE_REVISION,
                processor_sha256=muse.processor_hash, backend=backend, chat_date="2026-10-30")
            stamp = time.monotonic()
            tokens = cache.get(key)
            timings["cache_lookup_seconds"] += time.monotonic() - stamp
            if tokens is None:
                stamp = time.monotonic()
                with Image.open(image_path) as rgb, torch.autocast("cuda", enabled=False):
                    tokens = muse.encode(rgb, item["instruction"], current_date="2026-10-30")
                timings["feature_extraction_seconds"] += time.monotonic() - stamp
                cache.put(key, tokens)
            sample = observation_window(item["data"], t, stats, history=4, horizon=50)
            sample.update(muse_bits=tokens["features_bfloat16_bits"], muse_valid=tokens["attention_mask"])
            return sample

        def save(status):
            state = {"contract": contract, "completed_updates": step, "normalize_queries": True,
                "dynamics_objective": False, "prefix": {k: v.detach().cpu() for k, v in prefix.state_dict().items()},
                "trainable_policy": {k: v.detach().cpu() for k, v in params.items()}, "optimizer": optimizer.state_dict(),
                "sampler": sampler.state_dict(), "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                "seen_windows": {k: sorted(v) for k, v in seen.items()}, "presentations_per_episode": per_episode}
            temp = out / "last.pt.tmp"
            torch.save(state, temp)
            os.replace(temp, out / "last.pt")
            report.update(status=status, completed_updates=step, checkpoint_sha256=file_hash(out / "last.pt"),
                examples_per_task=dict(sampler.examples_per_task), unique_windows_per_task={k: len(v) for k, v in seen.items()},
                presentations_per_episode=per_episode, cache_stats_this_run=dict(cache.stats),
                cache_disk_bytes=cache.size_bytes, cache_ram_bytes=cache.memory_bytes,
                timings_this_run=dict(timings), backend=backend, processor_sha256=muse.processor_hash,
                milk_feature_parity=True)
            write_report()
            volume.commit()

        prefix.train()
        model.vlm_with_expert.lm_expert.train()
        while step < updates and time.monotonic() - started < max_seconds:
            indices = sampler.take(batch_size * accumulation)
            denominator = sum(min(50, lengths[task][episode] - frame) for task, episode, frame in indices) * 7
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            for start in range(0, len(indices), batch_size):
                batch_indices = indices[start:start + batch_size]
                samples = [sample_for(task, episode, frame) for task, episode, frame in batch_indices]
                b = make_batch(samples, "cuda")
                actions = F.pad(b["actions"], (0, cfg.max_action_dim - 7))
                noise = model.sample_noise(actions.shape, "cuda")
                t = model.sample_time(len(actions), "cuda")
                xt = t[:, None, None] * noise + (1 - t[:, None, None]) * actions
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    robot = F.pad(b["robot"][:, -1], (0, cfg.max_state_dim - 15))
                    keys, values, mask = prefix(b["muse"], b["muse_valid"], model.state_proj(robot),
                        b["sam_values"], b["sam_deltas"], b["sam_valid"], b["sam_age"], b["history_valid"], use_sam=True)
                    prediction = expert_velocity(model, keys, values, mask, xt, t)
                    loss = (((prediction - (noise - actions))[:, :, :7].square() * b["action_valid"][..., None]).sum()
                            / max(denominator, 1))
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite multi-demonstration flow loss")
                loss.backward()
                total_loss += float(loss.detach())
            grad = torch.nn.utils.clip_grad_norm_(trainable, 10, error_if_nonfinite=True)
            optimizer.step()
            step += 1
            for task, episode, frame in indices:
                seen[task].add(f"{episode}:{frame}")
                per_episode[task][episode] = per_episode[task].get(episode, 0) + 1
            row = {"step": step, "flow_loss": total_loss, "gradient_norm": float(grad),
                   "examples_per_task": dict(sampler.examples_per_task), "cache": dict(cache.stats),
                   "disk_bytes": cache.size_bytes, "ram_bytes": cache.memory_bytes,
                   "seconds": time.monotonic() - started}
            report["updates"].append(row)
            print(json.dumps(row), flush=True)
            if step % save_every == 0 or step == updates or time.monotonic() - started >= max_seconds:
                save("checkpointed")
        if step and (not (out / "last.pt").exists() or report.get("completed_updates") != step):
            save("checkpointed")
        report["status"] = "completed" if step == updates else "budget_stopped"
        report["training_cache_stats_this_run"] = dict(cache.stats)
        if step and max_seconds - (time.monotonic() - started) >= 120:
            prefix.eval()
            model.vlm_with_expert.lm_expert.eval()
            per_task = []
            for task, task_episodes in sorted(episodes.items()):
                item = next(iter(task_episodes.values()))
                n = len(item["data"]["timestamps"])
                frames = np.unique(np.linspace(0, n - 1, 3, dtype=int))
                samples = [sample_for(task, item["episode"], int(frame)) for frame in frames]
                b = make_batch(samples, "cuda")
                fixed_seed = int.from_bytes(hashlib.sha256(task.encode()).digest()[:4], "big")
                noise = np.random.default_rng(fixed_seed).standard_normal((len(frames), 50, 32), dtype=np.float32)
                action = torch.from_numpy(noise).cuda()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    robot = F.pad(b["robot"][:, -1], (0, cfg.max_state_dim - 15))
                    keys, values, mask = prefix(b["muse"], b["muse_valid"], model.state_proj(robot),
                        b["sam_values"], b["sam_deltas"], b["sam_valid"], b["sam_age"], b["history_valid"], use_sam=True)
                    for tick in range(10):
                        action = action - .1 * expert_velocity(model, keys, values, mask, action,
                            torch.full((len(frames),), 1 - tick / 10, device="cuda"))
                prediction = action[:, 0, :7].float().cpu().numpy()
                target = b["actions"][:, 0, :7].float().cpu().numpy()
                if not np.isfinite(prediction).all():
                    raise ValueError(f"Nonfinite offline prediction: {task}")
                raw_pred = prediction * stats["actions_std"][:7] + stats["actions_mean"][:7]
                raw_target = target * stats["actions_std"][:7] + stats["actions_mean"][:7]
                classes = {}
                for label, selected in (("negative", raw_target[:, 6] < 0), ("positive", raw_target[:, 6] > 0)):
                    classes[label] = {"count": int(selected.sum()),
                        "correct": int(((raw_pred[selected, 6] > 0) == (raw_target[selected, 6] > 0)).sum())}
                per_task.append({"task_id": task, "episode": item["episode"], "frames": frames.tolist(),
                    "first_action_raw_mae_by_channel": np.abs(raw_pred - raw_target).mean(0).tolist(),
                    "gripper": classes})
            report["offline_training_observations"] = {"scope": "three windows per task from training episodes; not held out",
                "noise": "NumPy task-stable seeded noise; 10 Euler steps", "tasks": per_task}
        else:
            report["offline_training_observations"] = {"status": "skipped due to time budget"}
        report["elapsed_seconds_this_run"] = time.monotonic() - started
        report["cache_stats_this_run"] = dict(cache.stats)
        report["cache_disk_bytes"] = cache.size_bytes
        report["cache_ram_bytes"] = cache.memory_bytes
        report["timings_this_run"] = timings
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        write_report()
        volume.commit()
        if muse is not None:
            muse.close()
    return {k: v for k, v in report.items() if k != "contract"}


@app.local_entrypoint()
def main(name: str = "muse_ten_task_multidemo_v1", updates: int = 10, batch_size: int = 8,
         accumulation: int = 8, max_seconds: int = 900, disk_gib: float = 20, ram_mib: int = 512,
         save_every: int = 5, resume: bool = False, background: bool = False, stage: str = "train"):
    if stage == "audit":
        print(audit.remote())
        return
    if stage != "train":
        raise ValueError("Use audit or train")
    args = (name, updates, batch_size, accumulation, max_seconds, disk_gib, ram_mib, save_every, resume)
    if background:
        print(f"Spawned multi-demonstration training: {train.spawn(*args).object_id}")
    else:
        print(train.remote(*args))
