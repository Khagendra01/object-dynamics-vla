"""Persistent NF4 cache and batched training on the aligned development split.

The architecture is the existing query-normalized prefix + official expert.
There is no dynamics objective or automatic simulator rollout. Nothing runs on
Modal merely by importing this module or asking for CLI help.
"""
import modal
from scripts.modal_quantized_muse_preflight import image, volume, MUSE_REVISION, SMOL_CONFIG_REVISION

app = modal.App("odvla-muse-aligned-dataset")


def simple_name(value):
    if not value or len(value) > 100 or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in value):
        raise ValueError("Expected a simple asset name")
    return value


@app.function(image=image, gpu="H200", cpu=4, memory=131072,
              volumes={"/data": volume}, timeout=3600, max_containers=1,
              single_use_containers=True)
def cache_nf4(dataset: str = "muse_aligned_development_v1", cache_name: str = "muse_aligned_nf4_v1", max_seconds: int = 600):
    """Resume at a committed frame, validating identity and existing RGB hashes."""
    import hashlib
    import importlib.metadata
    import json
    import os
    import time
    from pathlib import Path
    import numpy as np
    import torch
    import transformers
    from PIL import Image
    from odvla.data import file_hash, load_episode, read_json, validate_splits
    from odvla.vla.muse_context import FrozenMuseTokens, backend_snapshot

    if not 30 <= max_seconds <= 3000:
        raise ValueError("Cache function budget must be 30..3000 seconds")
    root = Path("/data/libero") / simple_name(dataset)
    dest = Path("/data/libero") / simple_name(cache_name)
    splits = read_json(root / "splits.json")
    manifest = validate_splits(root, splits)
    if not (root / "alignment_report.json").exists() or splits.get("test"):
        raise ValueError("Expected the aligned development dataset without test episodes")
    wanted = set(splits["train"]) | set(splits["val"])
    entries = [e for e in manifest["episodes"] if e["id"] in wanted]
    reference = read_json(Path("/data/eval/quantized_muse_aligned7500_v1/report.json"))
    if reference.get("dataset_fingerprint") != splits["fingerprint"] or reference.get("episode") != "episode_0000" or reference.get("muse_revision") != MUSE_REVISION:
        raise ValueError("Aligned episode-0000 parity reference does not match")
    reference_frames = {int(item["frame"]): item for item in reference["frames"]}
    if not reference_frames:
        raise ValueError("Aligned parity reference has no frame hashes")

    def check_reference(episode_id, t, rgb_hash, feature_hash):
        if episode_id == "episode_0000":
            expected = reference_frames.get(t)
            if expected is None or expected["rgb_sha256"] != rgb_hash or expected["feature_sha256"] != feature_hash:
                raise ValueError(f"Aligned RGB/Muse parity failed at episode_0000 frame {t}")
    identity = {"dataset_fingerprint": splits["fingerprint"], "muse_revision": MUSE_REVISION,
        "quantization": "NF4 double quant, BF16 compute/storage; frozen text decoder",
        "layer_index": 26, "chat_current_date": "2026-10-30", "hardware": "H200",
        "torch": str(torch.__version__), "transformers": transformers.__version__,
        "bitsandbytes": importlib.metadata.version("bitsandbytes"),
        "matmul_tf32": False, "cudnn_tf32": False}
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "identity.json").exists() and read_json(dest / "identity.json") != identity:
        raise ValueError("Cache identity changed; choose a new cache name")
    if (dest / "manifest.json").exists() and read_json(dest / "manifest.json").get("complete"):
        if read_json(dest / "manifest.json")["dataset_fingerprint"] != splits["fingerprint"]:
            raise ValueError("Completed cache belongs to another dataset")
        return {"complete": True, "cache": str(dest), "existing": True}

    def write(path, data):
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
        os.replace(temporary, path)

    write(dest / "identity.json", identity)
    started = time.monotonic()
    model = FrozenMuseTokens(Path("/data/checkpoints/muse-glimmer-30b") / MUSE_REVISION,
                             26, quantization="nf4")
    converted = [n for n, m in model.model.named_modules() if type(m).__name__ == "Linear4bit"]
    if not converted or any(not n.startswith(model.module_name + ".") for n in converted):
        model.close()
        raise ValueError("Unexpected NF4 conversion; cache not produced")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    completed = []
    try:
        for entry in entries:
            episode = load_episode(root, entry, manifest, include_images=False)
            count = len(episode["timestamps"])
            folder = dest / entry["id"]
            folder.mkdir(exist_ok=True)
            path = folder / "manifest.json"
            record = {"episode": entry["id"], "frames": count, "records": [],
                "instruction_sha256": hashlib.sha256(entry["instruction"].encode()).hexdigest()}
            if path.exists():
                prior = read_json(path)
                if any(prior[k] != record[k] for k in ("episode", "frames", "instruction_sha256")):
                    raise ValueError("Existing cache episode differs")
                record = prior
            rgb_root = Path("/data/libero/milk_sam_native_v1") / entry["id"]
            # Check already committed frames without re-running Muse.
            for t, item in enumerate(record["records"]):
                if (item["frame"] != t or item["timestamp"] != float(episode["timestamps"][t]) or
                        file_hash(folder / f"{t:06}.npz") != item["sha256"] or
                        file_hash(rgb_root / f"{t:06}.jpg") != item["rgb_sha256"]):
                    raise ValueError("Committed cache frame or RGB changed")
                check_reference(entry["id"], t, item["rgb_sha256"], item["feature_sha256"])
            for t in range(len(record["records"]), count):
                if time.monotonic() - started >= max_seconds:
                    result = dict(identity, complete=False, episodes=completed,
                        partial_episode=entry["id"], partial_frames=t,
                        elapsed_seconds=time.monotonic() - started)
                    write(dest / "manifest.json", result)
                    volume.commit()
                    return result
                rgb_path = rgb_root / f"{t:06}.jpg"
                with Image.open(rgb_path) as rgb, torch.autocast("cuda", enabled=False):
                    tokens = model.encode(rgb, entry["instruction"], current_date="2026-10-30")
                target = folder / f"{t:06}.npz"
                temporary = target.with_suffix(".npz.tmp")
                with temporary.open("wb") as stream:
                    np.savez(stream, **tokens)
                os.replace(temporary, target)
                record["records"].append({"frame": t, "timestamp": float(episode["timestamps"][t]),
                    "rgb_sha256": file_hash(rgb_path), "sha256": file_hash(target),
                    "feature_sha256": hashlib.sha256(tokens["features_bfloat16_bits"].tobytes()).hexdigest(),
                    "tokens": len(tokens["attention_mask"])})
                check_reference(entry["id"], t, record["records"][-1]["rgb_sha256"], record["records"][-1]["feature_sha256"])
                write(path, record)
                if (t + 1) % 25 == 0:
                    volume.commit()
                    print(json.dumps({"episode": entry["id"], "cached_frames": t + 1}), flush=True)
            completed.append(entry["id"])
        result = dict(identity, complete=True, episodes=completed,
            backend=backend_snapshot(torch, model.model),
            processor_config_sha256=model.processor_hash, elapsed_seconds=time.monotonic() - started)
        write(dest / "manifest.json", result)
        volume.commit()
        return result
    finally:
        model.close()


def build_policy(device):
    """Use the same pinned official expert and compatibility shim as the tested runs."""
    from pathlib import Path
    import torch
    import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as qwen
    if not hasattr(qwen, "apply_multimodal_rotary_pos_emb"):
        def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
            cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section * 2, dim=-1))], dim=-1).unsqueeze(unsqueeze_dim)
            sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section * 2, dim=-1))], dim=-1).unsqueeze(unsqueeze_dim)
            return q * cos + qwen.rotate_half(q) * sin, k * cos + qwen.rotate_half(k) * sin
        qwen.apply_multimodal_rotary_pos_emb = apply_multimodal_rotary_pos_emb
    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.utils.constants import ACTION, OBS_STATE
    from odvla.vla.smolvla_bridge import MuseSamPrefix
    cfg = SmolVLAConfig(device=device,
        vlm_model_name=str(Path("/data/checkpoints/smolvlm2-reference") / SMOL_CONFIG_REVISION),
        load_vlm_weights=False,
        input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(15,)),
            "observation.images.agentview": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 512, 512))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))})
    model = SmolVLAPolicy(cfg).to(device).model
    model.vlm_with_expert.vlm.eval().requires_grad_(False)
    tc = model.vlm_with_expert.config.text_config
    prefix = MuseSamPrefix(layers=model.vlm_with_expert.num_vlm_layers,
        heads=tc.num_key_value_heads, head_dim=tc.head_dim, normalize_queries=True).to(device)
    return model, prefix, cfg


@app.function(image=image, gpu="A100-80GB", cpu=4, memory=32768,
              volumes={"/data": volume}, timeout=4500, max_containers=1,
              single_use_containers=True)
def train(dataset: str = "muse_aligned_development_v1", cache_name: str = "muse_aligned_nf4_v1",
          name: str = "muse_aligned_dataset_v1", updates: int = 1000, batch_size: int = 8, accumulation: int = 8,
          max_seconds: int = 1800, save_every: int = 250, resume: bool = False,
          init_fit: str = "quantized_muse_aligned7500_v1"):
    import hashlib
    import json
    import os
    import sys
    import time
    from functools import lru_cache
    from pathlib import Path
    import numpy as np
    import torch
    import torch.nn.functional as F
    from odvla.data import file_hash, fit_stats, read_json, validate_splits
    from odvla.vla.batches import ShuffledWindows, validate_resume_contract
    from odvla.vla.dataset import CachedEpisodeDataset
    from odvla.vla.smolvla_bridge import expert_velocity
    sys.path.insert(0, "/root")
    from train_muse_vla import make_batch

    for value in (dataset, cache_name, name):
        simple_name(value)
    if init_fit:
        simple_name(init_fit)
    if (not 1 <= updates <= 100000 or not 1 <= batch_size <= 32 or not 1 <= accumulation <= 64 or
            batch_size * accumulation > 256 or not 30 <= max_seconds <= 3000 or save_every < 1):
        raise ValueError("Invalid training size or runtime budget")
    root, cached = Path("/data/libero") / dataset, Path("/data/libero") / cache_name
    splits = read_json(root / "splits.json")
    validate_splits(root, splits)
    identity = read_json(cached / "identity.json")
    if (identity["dataset_fingerprint"] != splits["fingerprint"] or
            identity["muse_revision"] != MUSE_REVISION or identity["layer_index"] != 26 or
            identity["chat_current_date"] != "2026-10-30" or not identity["quantization"].startswith("NF4")):
        raise ValueError("Expected the aligned dataset's pinned NF4 cache")
    if splits.get("test") or not (root / "alignment_report.json").exists():
        raise ValueError("Expected aligned train/development data, no test access")
    stats = fit_stats(root, splits)
    trainset = CachedEpisodeDataset(root, cached, "train", stats)
    valset = CachedEpisodeDataset(root, cached, "val", stats)
    if not len(trainset) or not len(valset):
        raise ValueError("Both train and development splits must contain windows")
    out = Path("/data/eval") / name
    if out.exists() != resume:
        raise ValueError("Use a new output name, or explicitly resume it")
    out.mkdir(parents=True, exist_ok=True)
    contract = {"dataset_fingerprint": splits["fingerprint"], "cache_manifest_sha256": file_hash(cached / "manifest.json"),
        "batch_size": batch_size, "accumulation": accumulation, "updates": updates,
        "init_fit": init_fit, "normalize_queries": True, "dynamics_objective": False,
        "lr": 1e-4, "betas": [.9, .95], "weight_decay": .01,
        "muse_revision": MUSE_REVISION, "smol_config_revision": SMOL_CONFIG_REVISION}
    report_path = out / "report.json"
    if resume:
        validate_resume_contract(read_json(report_path)["contract"], contract)
    torch.manual_seed(7)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, prefix, cfg = build_policy("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    trainable = list(prefix.parameters()) + list(params.values())
    optimizer = torch.optim.AdamW(trainable, lr=1e-4, betas=(.9, .95))
    sampler = ShuffledWindows(len(trainset), seed=7)

    def load_weights(state):
        if set(state["trainable_policy"]) != set(params):
            raise ValueError("Trainable parameter set changed")
        prefix.load_state_dict(state["prefix"], strict=True)
        with torch.no_grad():
            for key, value in state["trainable_policy"].items():
                params[key].copy_(value)

    if resume:
        report = read_json(report_path)
        with np.load(out / "stats.npz", allow_pickle=False) as previous_stats:
            if set(previous_stats.files) != set(stats) or any(not np.array_equal(previous_stats[k], stats[k]) for k in stats):
                raise ValueError("Resume normalization differs")
        state_path = out / "last.pt"
        if file_hash(state_path) != report["last_checkpoint_sha256"]:
            raise ValueError("Last checkpoint changed")
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        validate_resume_contract(state["contract"], contract)
        load_weights(state)
        optimizer.load_state_dict(state["optimizer"])
        sampler.load_state_dict(state["sampler"])
        step = state["completed_updates"]
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        report["contract"] = contract
        del state
    else:
        source_hash = None
        if init_fit:
            source_root = Path("/data/eval") / init_fit
            with np.load(source_root / "stats.npz", allow_pickle=False) as source_stats:
                if set(source_stats.files) != set(stats) or any(not np.array_equal(source_stats[k], stats[k]) for k in stats):
                    raise ValueError("Warm-start normalization differs; do not silently change coordinates")
            source_path = source_root / "preflight_weights.pt"
            state = torch.load(source_path, map_location="cpu", weights_only=True)
            if state.get("dataset_fingerprint") != splits["fingerprint"] or not state.get("normalize_queries"):
                raise ValueError("Warm-start dataset/normalization metadata differs")
            load_weights(state)
            source_hash = file_hash(source_path)
            del state
        step = 0
        np.savez(out / "stats.npz", **stats)
        reference = read_json(cached / "episode_0000" / "manifest.json")
        report = {"status": "started", "contract": contract,
            "training_episodes": sorted(trainset.entries), "validation_episodes": sorted(valset.entries),
            "train_windows": len(trainset), "validation_windows": len(valset),
            "batch_size": batch_size, "effective_batch_size": batch_size * accumulation,
            "source_checkpoint_sha256": source_hash, "optimizer_restart": bool(init_fit),
            "update_numbering": "new dataset-training phase; excludes source single-episode updates",
            "dataset_fingerprint": splits["fingerprint"], "muse_revision": MUSE_REVISION,
            "normalize_queries": True, "dynamics_objective": False, "test_accessed": False,
            "frames": reference["records"], "committed_checkpoints": [], "evaluations": []}

    @lru_cache(maxsize=128)
    def training_item(index):
        return trainset[index]

    def conditioning(b):
        state = F.pad(b["robot"][:, -1], (0, cfg.max_state_dim - 15))
        return prefix(b["muse"], b["muse_valid"], model.state_proj(state), b["sam_values"],
            b["sam_deltas"], b["sam_valid"], b["sam_age"], b["history_valid"], use_sam=True)

    def evaluate():
        prefix.eval(); model.vlm_with_expert.lm_expert.eval()
        # Four fixed windows per development episode, with noise derived from
        # stable dataset indices. Evaluation does not consume training RNG.
        indices = []
        by_episode = {}
        for i, (eid, _) in enumerate(valset.index):
            by_episode.setdefault(eid, []).append(i)
        for group in by_episode.values():
            indices.extend(group[i] for i in np.unique(np.linspace(0, len(group)-1, min(4, len(group)), dtype=int)))
        normalized, raw, targets = [], [], []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for start in range(0, len(indices), batch_size):
                ids = indices[start:start + batch_size]
                b = make_batch([valset[i] for i in ids], "cuda")
                keys, values, mask = conditioning(b)
                noise = np.stack([np.random.default_rng(921 + i).standard_normal((50, 32)).astype(np.float32) for i in ids])
                action = torch.from_numpy(noise).cuda()
                for tick in range(10):
                    action = action - .1 * expert_velocity(model, keys, values, mask, action,
                        torch.full((len(ids),), 1 - tick / 10, device="cuda"))
                prediction = action[:, 0, :7].float().cpu().numpy()
                target = b["actions"][:, 0].cpu().numpy()
                if not np.isfinite(prediction).all():
                    raise ValueError("Nonfinite sampled evaluation action")
                normalized.append(prediction - target)
                raw.append((prediction - target) * stats["actions_std"])
                targets.append(target * stats["actions_std"] + stats["actions_mean"])
        errors, raw_errors, truth = np.concatenate(normalized), np.concatenate(raw), np.concatenate(targets)
        prediction = truth + raw_errors
        classes = {}
        for key, select in (("negative", truth[:, 6] < 0), ("positive", truth[:, 6] > 0)):
            classes[key] = {"correct": int((np.sign(prediction[select, 6]) == np.sign(truth[select, 6])).sum()),
                            "count": int(select.sum())}
        result = {"step": step, "scope": "reused development observations, four windows per episode",
            "windows": len(indices), "first_action_mae_normalized_by_channel": np.abs(errors).mean(0).tolist(),
            "first_action_mae_raw_by_channel": np.abs(raw_errors).mean(0).tolist(), "gripper": classes}
        prefix.train(); model.vlm_with_expert.lm_expert.train()
        return result

    def save():
        state = {"contract": contract, "completed_updates": step,
            "normalize_queries": True, "dynamics_objective": False,
            "dataset_fingerprint": splits["fingerprint"], "muse_revision": MUSE_REVISION,
            "prefix": {k: v.detach().cpu() for k, v in prefix.state_dict().items()},
            "trainable_policy": {k: v.detach().cpu() for k, v in params.items()},
            "optimizer": optimizer.state_dict(), "sampler": sampler.state_dict(),
            "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}
        path = out / f"checkpoint_{step}.pt"
        temporary = path.with_suffix(".pt.tmp")
        torch.save(state, temporary); os.replace(temporary, path)
        # A separate atomic last checkpoint permits resuming partial chunks.
        temporary = out / "last.pt.tmp"
        torch.save(state, temporary); os.replace(temporary, out / "last.pt")
        report["last_checkpoint_sha256"] = file_hash(out / "last.pt")
        report["completed_updates"] = step
        report["training_windows_presented"] = step * batch_size * accumulation
        report["committed_checkpoints"] = [x for x in report["committed_checkpoints"] if x["step"] != step]
        report["committed_checkpoints"].append({"step": step, "sha256": file_hash(path)})
        temporary = report_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        os.replace(temporary, report_path); volume.commit()

    prefix.train(); model.vlm_with_expert.lm_expert.train()
    started = time.monotonic()
    completed_on_entry = step
    try:
        while step < updates and time.monotonic() - started < max_seconds:
            batches = [make_batch([training_item(i) for i in sampler.take(batch_size)], "cuda") for _ in range(accumulation)]
            denominator = sum(int(b["action_valid"].sum()) for b in batches) * 7
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            for b in batches:
                actions = F.pad(b["actions"], (0, cfg.max_action_dim - 7))
                noise = model.sample_noise(actions.shape, "cuda")
                t = model.sample_time(len(actions), "cuda")
                xt = t[:, None, None] * noise + (1 - t[:, None, None]) * actions
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    keys, values, mask = conditioning(b)
                    prediction = expert_velocity(model, keys, values, mask, xt, t)
                    loss = (((prediction - (noise - actions))[:, :, :7].square() *
                             b["action_valid"][..., None]).sum() / max(denominator, 1))
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite flow loss")
                loss.backward(); total_loss += float(loss.detach())
            torch.nn.utils.clip_grad_norm_(trainable, 10, error_if_nonfinite=True)
            optimizer.step(); step += 1
            if step == 1 or step % 25 == 0:
                print(json.dumps({"step": step, "flow_loss": total_loss,
                    "effective_batch": batch_size * accumulation,
                    "training_windows_presented": step * batch_size * accumulation,
                    "seconds": time.monotonic() - started}), flush=True)
            if step % save_every == 0 or step == updates:
                report["status"] = "checkpointed"
                save()
                if max_seconds - (time.monotonic() - started) > 120:
                    result = evaluate(); report["evaluations"].append(result)
                    print(json.dumps(result), flush=True)
                    save()
        report["status"] = "completed" if step == updates else "budget_stopped"
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        report["training_chunk_seconds"] = time.monotonic() - started
        seconds = report["training_chunk_seconds"]
        report["chunk_updates"] = step - completed_on_entry
        report["requested_training_loop_cost_estimate_usd"] = seconds * (2.50 + 4 * .04716 + 32 * .007992) / 3600
        report["cost_excludes"] = "cache extraction, setup before training loop, image build/startup, checkpoint writeback, storage; not an invoice"
        save()
    return {k: v for k, v in report.items() if k != "frames"}


@app.local_entrypoint()
def main(stage: str = "cache", dataset: str = "muse_aligned_development_v1", cache_name: str = "muse_aligned_nf4_v1",
         name: str = "muse_aligned_dataset_v1", updates: int = 1000, batch_size: int = 8,
         accumulation: int = 8, max_seconds: int = 600, save_every: int = 250,
         resume: bool = False, init_fit: str = "quantized_muse_aligned7500_v1", background: bool = False):
    if stage == "cache":
        print(cache_nf4.remote(dataset, cache_name, max_seconds))
    elif stage == "train":
        args = (dataset, cache_name, name, updates, batch_size, accumulation,
                max_seconds, save_every, resume, init_fit)
        if background:
            print(f"Spawned training call: {train.spawn(*args).object_id}")
        else:
            print(train.remote(*args))
    else:
        raise ValueError("Use --stage cache or --stage train")
