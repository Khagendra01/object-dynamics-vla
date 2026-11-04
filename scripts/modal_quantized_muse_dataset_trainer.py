"""Dataset trainer for the aligned demonstration split with frozen NF4 Muse features.

This script changes data coverage and minibatch order only. The NF4 feature
source, MuseSamPrefix architecture and noise-at-one flow objective are fixed.
Checkpoints contain model, optimizer, RNG and exact sample-order state.
"""
import json

import modal
from scripts.modal_quantized_muse_preflight import image, volume, MUSE_REVISION, SMOL_CONFIG_REVISION

app = modal.App("odvla-quantized-muse-dataset-trainer")

ALIGNED_FINGERPRINT = "86d9fdf00329e8f18df23bab67d70dbfae914bc68c6884cdd8938f8518386bed"
CHAT_DATE = "2026-10-30"
H200_PER_SECOND = 4.54 / 3600


def _rotary_shim():
    import torch
    import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as qwen
    if not hasattr(qwen, "apply_multimodal_rotary_pos_emb"):
        def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
            sections = mrope_section * 2
            cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(sections, dim=-1))], dim=-1).unsqueeze(unsqueeze_dim)
            sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(sections, dim=-1))], dim=-1).unsqueeze(unsqueeze_dim)
            return (q * cos) + (qwen.rotate_half(q) * sin), (k * cos) + (qwen.rotate_half(k) * sin)
        qwen.apply_multimodal_rotary_pos_emb = apply_multimodal_rotary_pos_emb


def validate_request(name, dataset, cache, split, batch_size, updates, max_train_seconds,
                     checkpoint_every, lr, seed, resume_step):
    for value in (name, dataset, cache):
        if not value or len(value) > 80 or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in value):
            raise ValueError("Use simple dataset/cache/output names")
    if split not in ("train", "val"):
        raise ValueError("Training is limited to train or validation; no test bank")
    if not 1 <= batch_size <= 32 or not 1 <= updates <= 50000:
        raise ValueError("Use batch size 1..32 and updates 1..50000")
    if not 60 <= max_train_seconds <= 5400 or not 1 <= checkpoint_every <= updates:
        raise ValueError("Use a 60..5400 second cap and 1..updates checkpoint interval")
    if not 1e-5 <= lr <= 1e-3 or not 0 <= resume_step <= updates:
        raise ValueError("Use learning rate 1e-5..1e-3 and a resume step within updates")


def _dataset_and_cache(dataset, cache, split, stats=None):
    import json
    from pathlib import Path
    from odvla.data import file_hash, fit_stats, read_json, validate_splits
    from odvla.vla.dataset import CachedEpisodeDataset
    root = Path("/data/libero") / dataset
    cache_root = Path("/data/libero") / cache
    splits = read_json(root / "splits.json")
    manifest = validate_splits(root, splits)
    if splits["fingerprint"] != ALIGNED_FINGERPRINT:
        raise ValueError("Refusing to train on a dataset that is not the audited aligned development set")
    if splits["test"]:
        raise ValueError("The untouched test split must remain empty")
    if stats is None:
        stats = fit_stats(root, splits)
    cache_manifest = read_json(cache_root / "manifest.json")
    if not cache_manifest.get("complete"):
        raise ValueError("Feature cache is incomplete; finish the feature-cache build first")
    if cache_manifest.get("dataset_fingerprint") != splits["fingerprint"]:
        raise ValueError("Feature cache dataset fingerprint mismatch")
    if cache_manifest.get("muse_revision") != MUSE_REVISION or cache_manifest.get("layer_index") != 26:
        raise ValueError("Feature cache Muse revision/layer mismatch")
    if cache_manifest.get("chat_current_date") != CHAT_DATE or cache_manifest.get("test_used") is not False:
        raise ValueError("Feature cache date/test provenance mismatch")
    dataset_view = CachedEpisodeDataset(root, cache_root, split, stats)
    recorded = cache_manifest.get("episode_manifests")
    if not isinstance(recorded, dict):
        raise ValueError("Feature cache does not record per-episode manifest hashes")
    for episode in dataset_view.entries:
        path = cache_root / episode / "manifest.json"
        if recorded.get(episode) != file_hash(path):
            raise ValueError(f"Feature cache episode manifest changed: {episode}")
    return root, cache_root, splits, manifest, stats, dataset_view, file_hash(cache_root / "manifest.json")


@app.function(image=image, cpu=2, memory=8192, volumes={"/data": volume}, timeout=300)
def plan(dataset: str = "muse_aligned_development_v1", cache: str = "muse_nf4_aligned_train_v1"):
    from pathlib import Path
    from odvla.data import read_json, validate_splits
    root = Path("/data/libero") / dataset
    splits = read_json(root / "splits.json")
    manifest = validate_splits(root, splits)
    if splits["fingerprint"] != ALIGNED_FINGERPRINT:
        raise ValueError("Not the audited aligned development dataset")
    counts = {split: len(splits[split]) for split in ("train", "val", "test")}
    per_episode = {}
    for entry in manifest["episodes"]:
        if entry["id"] in set(splits["train"] + splits["val"]):
            import numpy as np
            with np.load(root / entry["file"], allow_pickle=False) as z:
                per_episode[entry["id"]] = len(z["timestamps"])
    cache_root = Path("/data/libero") / cache
    cache_state = {"path": str(cache_root), "exists": cache_root.exists()}
    if (cache_root / "manifest.json").exists():
        record = read_json(cache_root / "manifest.json")
        cache_state.update(complete=bool(record.get("complete")), split=record.get("split"),
                           dataset_fingerprint=record.get("dataset_fingerprint"),
                           frames=record.get("frames"), episodes=record.get("complete_episodes"),
                           reference_frames_verified=record.get("reference_frames_verified"))
    return {"dataset": dataset, "dataset_fingerprint": splits["fingerprint"], "split_counts": counts,
            "train_frames": sum(per_episode[e] for e in splits["train"]),
            "val_frames": sum(per_episode[e] for e in splits["val"]),
            "per_episode_frames": per_episode, "cache": cache_state,
            "expected_train_cache": "build with --split train",
            "expected_all_cache": "build with --split all (train+val)"}


@app.function(image=image, gpu="H200", cpu=8, memory=65536, volumes={"/data": volume},
              timeout=7200, max_containers=1, single_use_containers=True)
def train(name: str = "quantized_muse_aligned_dataset_v1", dataset: str = "muse_aligned_development_v1",
          cache: str = "muse_nf4_aligned_train_v1", split: str = "train", batch_size: int = 8,
          updates: int = 5000, max_train_seconds: int = 1200, checkpoint_every: int = 2500,
          lr: float = 1e-4, seed: int = 7, resume_step: int = 0):
    import importlib.metadata
    import json
    import os
    import sys
    import time
    from pathlib import Path
    import numpy as np
    import torch
    import torch.nn.functional as F
    _rotary_shim()
    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.utils.constants import ACTION, OBS_STATE
    from odvla.data import file_hash, read_json
    from odvla.vla.checkpointing import checkpoint_payload, validate_checkpoint
    from odvla.vla.sampler import ShuffledBatchSampler
    from odvla.vla.smolvla_bridge import MuseSamPrefix, expert_velocity
    sys.path.insert(0, "/root")
    from train_muse_vla import make_batch

    validate_request(name, dataset, cache, split, batch_size, updates, max_train_seconds,
                     checkpoint_every, lr, seed, resume_step)
    out = Path("/data/eval") / name
    if resume_step == 0:
        if out.exists():
            raise FileExistsError("Use a fresh output name or resume an existing complete checkpoint")
    else:
        if not out.exists() or not (out / f"checkpoint_{resume_step}.pt").exists():
            raise FileNotFoundError("Requested resume checkpoint does not exist")
    started = time.monotonic()
    root, cache_root, splits, manifest, stats, dataset_view, cache_sha = _dataset_and_cache(dataset, cache, split)
    prior = None
    if resume_step:
        if not (out / "stats.npz").exists():
            raise FileNotFoundError("Resume run has no saved normalization statistics")
        with np.load(out / "stats.npz", allow_pickle=False) as saved:
            saved_stats = {k: saved[k].copy() for k in saved.files}
        if set(saved_stats) != set(stats) or any(not np.array_equal(saved_stats[k], stats[k]) for k in stats):
            raise ValueError("Saved normalization statistics no longer match the aligned train split")
        stats = saved_stats
        stats_sha = file_hash(out / "stats.npz")
        if (out / "report.json").exists():
            prior = read_json(out / "report.json")
    else:
        out.mkdir(parents=True, exist_ok=False)
        np.savez(out / "stats.npz", **stats)
        stats_sha = file_hash(out / "stats.npz")
    episode_frames = {episode: len(view["timestamps"]) for episode, view in dataset_view.episodes.items()}
    report = {"status": "started", "scope": "aligned training split; shuffled minibatches; frozen NF4 features; "
              "unchanged MuseSamPrefix and flow objective",
              "purpose": "broader demonstration coverage and proper minibatches; hypothesis, not a proven control cure",
              "dataset": dataset, "dataset_fingerprint": splits["fingerprint"], "split": split,
              "cache": cache, "cache_manifest_sha256": cache_sha, "stats_sha256": stats_sha,
              "muse_revision": MUSE_REVISION, "smol_config_revision": SMOL_CONFIG_REVISION,
              "chat_current_date": CHAT_DATE, "normalize_queries": True, "dynamics_objective": False,
              "muse_model_loaded": False, "feature_source": "cached NF4 layer-26 hidden states",
              "robot_pretrained_weights_loaded": False, "teacher_used": False,
              "old_bf16_muse_cache_used": False, "test_accessed": False,
              "episodes": episode_frames, "episode_count": len(episode_frames), "frames": len(dataset_view),
              "batch_size": batch_size, "gradient_accumulation": 1, "requested_updates": updates,
              "max_train_seconds": max_train_seconds, "checkpoint_every": checkpoint_every,
              "lr": lr, "seed": seed, "optimizer": "AdamW betas=(0.9,0.95), default weight decay",
              "sampling": "one uniform permutation per epoch; shuffled minibatches; every frame is used",
              "architecture": "NF4 Muse layer 26 -> MuseSamPrefix(normalize_queries=True) -> SmolVLA expert; flow only",
              "resume_step": resume_step, "resume_history": [],
              "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
              "packages": {k: importlib.metadata.version(k)
                           for k in ("torch", "transformers", "bitsandbytes", "accelerate")}}
    if resume_step:
        report["resume_history"] = (prior or {}).get("resume_history", []) + [{
            "from_step": resume_step, "checkpoint_sha256": file_hash(out / f"checkpoint_{resume_step}.pt")}]
        report["committed_checkpoints"] = (prior or {}).get("committed_checkpoints", [])
        report["prior_updates"] = (prior or {}).get("updates", [])

    model_root = Path("/data/checkpoints/smolvlm2-reference") / SMOL_CONFIG_REVISION
    if not (model_root / "config.json").exists():
        raise FileNotFoundError("Missing existing SmolVLA config; no implicit download in GPU job")
    torch.manual_seed(seed)
    cfg = SmolVLAConfig(device="cuda", vlm_model_name=str(model_root), load_vlm_weights=False,
        input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(15,)),
                        "observation.images.agentview": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 512, 512))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))})
    policy = SmolVLAPolicy(cfg).cuda()
    model = policy.model
    model.vlm_with_expert.vlm.eval().requires_grad_(False)
    tc = model.vlm_with_expert.config.text_config
    if tc.hidden_size != 960:
        raise ValueError("Existing prefix adapter expects the pinned 960-wide expert interface")
    prefix = MuseSamPrefix(muse_width=6656, layers=model.vlm_with_expert.num_vlm_layers,
        heads=tc.num_key_value_heads, head_dim=tc.head_dim, normalize_queries=True).cuda()
    expert = model.vlm_with_expert.lm_expert
    trainable = list(prefix.parameters()) + [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, betas=(.9, .95))
    sampler = ShuffledBatchSampler(len(dataset_view), batch_size, seed=seed)
    step = 0
    if resume_step:
        checkpoint = torch.load(out / f"checkpoint_{resume_step}.pt", map_location="cpu", weights_only=True)
        validate_checkpoint(checkpoint, expected={"dataset": dataset, "dataset_fingerprint": splits["fingerprint"],
            "cache": cache, "cache_manifest_sha256": cache_sha, "stats_sha256": stats_sha,
            "batch_size": batch_size, "lr": lr, "seed": seed})
        if checkpoint["completed_updates"] != resume_step:
            raise ValueError("Resume step does not match checkpoint contents")
        prefix.load_state_dict(checkpoint["prefix"], strict=True)
        params = dict(model.named_parameters())
        expected_trainable = {n for n, p in params.items() if p.requires_grad}
        if set(checkpoint["trainable_policy"]) != expected_trainable:
            raise ValueError("Checkpoint trainable parameter set mismatch")
        with torch.no_grad():
            for key, value in checkpoint["trainable_policy"].items():
                params[key].copy_(value)
        optimizer.load_state_dict(checkpoint["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to("cuda")
        sampler = ShuffledBatchSampler.from_state(checkpoint["sampler_state"])
        if sampler.size != len(dataset_view) or sampler.batch_size != batch_size:
            raise ValueError("Checkpoint sampler shape mismatch")
        torch.set_rng_state(checkpoint["torch_rng"])
        torch.cuda.set_rng_state_all([state.to("cuda") for state in checkpoint["cuda_rng"]])
        step = resume_step
        report["resume_checkpoint_sha256"] = file_hash(out / f"checkpoint_{resume_step}.pt")
    else:
        report["resume_checkpoint_sha256"] = None

    def cpu_copy(value):
        if torch.is_tensor(value):
            return value.detach().cpu()
        if isinstance(value, dict):
            return {key: cpu_copy(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cpu_copy(item) for item in value]
        return value

    def save_checkpoint(completed, status):
        checkpoint = checkpoint_payload(
            completed_updates=completed,
            prefix={k: v.detach().cpu() for k, v in prefix.state_dict().items()},
            trainable_policy={n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
            optimizer=cpu_copy(optimizer.state_dict()),
            torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all(),
            sampler_state=sampler.state_dict(),
            provenance={"purpose": "aligned full-split dataset training",
                "normalize_queries": True, "dynamics_objective": False,
                "dataset": dataset, "dataset_fingerprint": splits["fingerprint"],
                "cache": cache, "cache_manifest_sha256": cache_sha, "stats_sha256": stats_sha,
                "batch_size": batch_size, "lr": lr, "seed": seed,
                "muse_revision": MUSE_REVISION, "smol_config_revision": SMOL_CONFIG_REVISION,
                "chat_current_date": CHAT_DATE, "layer_index": 26,
                "architecture": report["architecture"], "status": status,
                "resumed_from": report["resume_checkpoint_sha256"]})
        path = out / f"checkpoint_{completed}.pt"
        tmp = out / f"checkpoint_{completed}.tmp"
        torch.save(checkpoint, tmp)
        os.replace(tmp, path)
        item = {"step": completed, "sha256": file_hash(path),
                "recent_loss_mean": float(np.mean(losses[-100:])) if losses else None,
                "training_seconds": time.monotonic() - train_started}
        report["updates_reached"] = completed
        report["last_checkpoint"] = item
        report["committed_checkpoints"] = [x for x in report.get("committed_checkpoints", []) if x["step"] != completed] + [item]
        report["status"] = status
        report["updates"] = report.get("prior_updates", []) + rows
        (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        volume.commit()
        print(json.dumps({"committed_checkpoint": item}), flush=True)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    prefix.train(); expert.train()
    losses, rows = [], []
    train_started = time.monotonic()
    last_saved = step
    status = "started"
    while step < updates:
        if step > resume_step and time.monotonic() - train_started >= max_train_seconds:
            status = "budget_stopped"
            break
        batch_indices = sampler.next_batch()
        samples = [dataset_view[int(index)] for index in batch_indices]
        b = make_batch(samples, "cuda")
        update_started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        target_actions = F.pad(b["actions"], (0, cfg.max_action_dim - 7))
        noise = model.sample_noise(target_actions.shape, "cuda")
        time_value = model.sample_time(len(target_actions), "cuda")
        xt = time_value[:, None, None] * noise + (1 - time_value[:, None, None]) * target_actions
        with torch.autocast("cuda", dtype=torch.bfloat16):
            state = F.pad(b["robot"][:, -1], (0, cfg.max_state_dim - 15))
            keys, values, mask = prefix(b["muse"], b["muse_valid"], model.state_proj(state),
                b["sam_values"], b["sam_deltas"], b["sam_valid"], b["sam_age"],
                b["history_valid"], use_sam=True)
            prediction = expert_velocity(model, keys, values, mask, xt, time_value)
            valid = b["action_valid"][..., None]
            loss = ((prediction - (noise - target_actions)).square()[:, :, :7] * valid).sum() / (
                valid.sum() * 7).clamp_min(1)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite flow loss")
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(trainable, 10, error_if_nonfinite=True)
        if step == resume_step:
            report["first_expert_gradient_norm"] = float(sum(
                p.grad.float().square().sum() for p in expert.parameters() if p.grad is not None).sqrt())
            report["first_prefix_gradient_norm"] = float(prefix.kv[0].weight.grad.float().norm())
            report["first_sam_encoder_gradient_norm"] = float(prefix.sam.weight.grad.float().norm())
            if not all(report[key] for key in ("first_expert_gradient_norm", "first_prefix_gradient_norm",
                                               "first_sam_encoder_gradient_norm")):
                raise ValueError("Missing expert or conditioning gradient")
        optimizer.step()
        step += 1
        losses.append(float(loss.detach()))
        rows.append({"step": step, "flow_loss": float(loss.detach()), "gradient_norm": float(grad),
                     "update_seconds": time.monotonic() - update_started})
        if step <= 5 or step % 50 == 0:
            print(json.dumps({"step": step, "loss": rows[-1]["flow_loss"],
                              "recent_loss_mean": float(np.mean(losses[-100:])),
                              "seconds": time.monotonic() - train_started}), flush=True)
        if step % checkpoint_every == 0 or step == updates:
            save_checkpoint(step, "completed_dataset_fit" if step == updates else "checkpoint")
            last_saved = step
    if step and step != last_saved:
        save_checkpoint(step, status if status != "started" else "completed_dataset_fit")
    if step == 0:
        raise ValueError("No updates were completed")
    report["updates_reached"] = step
    status = "completed_dataset_fit" if step == updates else "budget_stopped"
    report["status"] = status
    report["updates"] = report.pop("prior_updates", []) + rows
    report["training_seconds_this_run"] = time.monotonic() - train_started
    report["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 1024 ** 3
    report["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 1024 ** 3
    report["requested_resource_cost_estimate_usd"] = {
        "gpu_at_4_54_per_hour": (time.monotonic() - started) * H200_PER_SECOND,
        "excludes": "image builds, startup, storage; not an invoice"}
    report["feature_extraction_seconds"] = 0.0
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    volume.commit()
    return {k: v for k, v in report.items() if k != "updates"}


@app.function(image=image, gpu="H200", cpu=8, memory=65536, volumes={"/data": volume},
              timeout=5400, max_containers=1, single_use_containers=True)
def evaluate(name: str = "quantized_muse_aligned_dataset_eval_v1",
             fit_name: str = "quantized_muse_aligned_dataset_v1", checkpoint_step: int = 0,
             dataset: str = "muse_aligned_development_v1", cache: str = "muse_nf4_aligned_train_v1",
             split: str = "train", draws: int = 1, max_frames: int = 0, batch_size: int = 8):
    import json
    import sys
    import time
    from pathlib import Path
    import numpy as np
    import torch
    import torch.nn.functional as F
    _rotary_shim()
    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.utils.constants import ACTION, OBS_STATE
    from odvla.data import file_hash, fit_stats, read_json
    from odvla.vla.checkpointing import validate_checkpoint
    from odvla.vla.smolvla_bridge import MuseSamPrefix, expert_velocity
    sys.path.insert(0, "/root")
    from train_muse_vla import make_batch
    for value in (name, fit_name, dataset, cache):
        if not value or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in value):
            raise ValueError("Invalid output/fit/dataset/cache name")
    if split not in ("train", "val", "all") or not 1 <= draws <= 3 or not 1 <= batch_size <= 32 or max_frames < 0:
        raise ValueError("Use train/val/all, 1..3 draws, 1..32 batch, nonnegative max_frames")
    fit = Path("/data/eval") / fit_name
    report_in = read_json(fit / "report.json")
    if report_in.get("status") not in ("completed_dataset_fit", "budget_stopped", "checkpoint"):
        raise ValueError("Fit has no committed checkpoint")
    if checkpoint_step == 0:
        checkpoint_step = report_in["committed_checkpoints"][-1]["step"]
    checkpoint_path = fit / f"checkpoint_{checkpoint_step}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError("Requested checkpoint does not exist")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    validate_checkpoint(checkpoint, expected={"dataset": dataset, "dataset_fingerprint": report_in["dataset_fingerprint"],
        "cache": cache, "cache_manifest_sha256": report_in["cache_manifest_sha256"], "stats_sha256": report_in["stats_sha256"]})
    if checkpoint["completed_updates"] != checkpoint_step:
        raise ValueError("Checkpoint step mismatch")
    with np.load(fit / "stats.npz", allow_pickle=False) as saved:
        stats = {k: saved[k].copy() for k in saved.files}
    if file_hash(fit / "stats.npz") != report_in["stats_sha256"]:
        raise ValueError("Fit stats changed")
    root, cache_root, splits, manifest, _stats, dataset_view, cache_sha = _dataset_and_cache(dataset, cache, split, stats)
    current_stats = fit_stats(root, splits)
    if set(stats) != set(current_stats) or any(not np.array_equal(stats[k], current_stats[k]) for k in stats):
        raise ValueError("Evaluation normalization disagrees with the aligned train split")
    if cache_sha != checkpoint["cache_manifest_sha256"]:
        raise ValueError("Cache changed since the checkpoint was written")
    out = Path("/data/eval") / name
    if out.exists():
        raise FileExistsError("Use a fresh evaluation name")
    out.mkdir(parents=True)
    started = time.monotonic()
    report = {"status": "started", "scope": "all frames of the selected split; offline teacher-forced first actions; no simulator",
              "fit": fit_name, "fit_status": report_in["status"],
              "checkpoint_step": checkpoint_step, "checkpoint_sha256": file_hash(checkpoint_path),
              "dataset": dataset, "dataset_fingerprint": splits["fingerprint"], "split": split,
              "cache": cache, "cache_manifest_sha256": cache_sha, "stats_sha256": report_in["stats_sha256"],
              "frames_total": len(dataset_view), "frames_evaluated": 0, "draws": draws,
              "batch_size": batch_size, "normalize_queries": True, "dynamics_objective": False,
              "integration": "10 Euler steps from noise; times 1.0..0.1; first action",
              "muse_model_loaded": False, "simulator_actions_executed": False, "test_accessed": False,
              "checkpoint_batch_size": checkpoint["batch_size"], "checkpoint_seed": checkpoint["seed"]}
    model_root = Path("/data/checkpoints/smolvlm2-reference") / SMOL_CONFIG_REVISION
    torch.manual_seed(checkpoint["seed"])
    cfg = SmolVLAConfig(device="cuda", vlm_model_name=str(model_root), load_vlm_weights=False,
        input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(15,)),
                        "observation.images.agentview": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 512, 512))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))})
    policy = SmolVLAPolicy(cfg).cuda().eval()
    model = policy.model
    model.vlm_with_expert.vlm.eval().requires_grad_(False)
    tc = model.vlm_with_expert.config.text_config
    prefix = MuseSamPrefix(muse_width=6656, layers=model.vlm_with_expert.num_vlm_layers,
        heads=tc.num_key_value_heads, head_dim=tc.head_dim, normalize_queries=True).cuda().eval()
    prefix.load_state_dict(checkpoint["prefix"], strict=True)
    params = dict(model.named_parameters())
    if set(checkpoint["trainable_policy"]) != {n for n, p in params.items() if p.requires_grad}:
        raise ValueError("Checkpoint trainable parameter set mismatch")
    with torch.no_grad():
        for key, value in checkpoint["trainable_policy"].items():
            params[key].copy_(value)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    n_eval = len(dataset_view) if max_frames == 0 else min(len(dataset_view), max_frames)
    if n_eval < 1:
        raise ValueError("No frames selected")
    noise = np.random.default_rng(923).standard_normal((draws, n_eval, 50, 32), dtype=np.float32)
    np.save(out / "noise.npy", noise, allow_pickle=False)
    noise_sha = file_hash(out / "noise.npy")
    predictions = np.zeros((draws, n_eval, 7), dtype=np.float32)
    targets = np.zeros((n_eval, 7), dtype=np.float32)
    for start in range(0, n_eval, batch_size):
        end = min(start + batch_size, n_eval)
        samples = [dataset_view[i] for i in range(start, end)]
        b = make_batch(samples, "cuda")
        targets[start:end] = b["actions"][:, 0, :7].float().cpu().numpy()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = F.pad(b["robot"][:, -1], (0, cfg.max_state_dim - 15))
            keys, values, mask = prefix(b["muse"], b["muse_valid"], model.state_proj(state),
                b["sam_values"], b["sam_deltas"], b["sam_valid"], b["sam_age"],
                b["history_valid"], use_sam=True)
            for draw in range(draws):
                action = torch.from_numpy(np.ascontiguousarray(noise[draw, start:end])).cuda()
                for k in range(10):
                    action = action - .1 * expert_velocity(model, keys, values, mask, action,
                        torch.full((end - start,), 1 - k / 10, device="cuda"))
                prediction = action[:, 0, :7].float().cpu().numpy()
                if not np.isfinite(prediction).all():
                    raise ValueError("Nonfinite evaluation action")
                predictions[draw, start:end] = prediction
        if end == n_eval or ((start // batch_size) % 20 == 0):
            print(json.dumps({"frames": end, "seconds": time.monotonic() - started}), flush=True)
    action_std, action_mean = stats["actions_std"][:7], stats["actions_mean"][:7]
    predictions_raw = predictions * action_std + action_mean
    targets_raw = targets * action_std + action_mean
    labels = dataset_view.index[:n_eval]

    def metrics(pred, target):
        classes = {}
        for sign, label in ((-1, "negative"), (1, "positive")):
            chosen = target[:, 6] * sign > 0
            classes[label] = {"count": int(chosen.sum()),
                "sign_accuracy": float(((pred[chosen, 6] > 0) == (target[chosen, 6] > 0)).mean()) if chosen.any() else None,
                "mae": float(np.abs(pred[chosen, 6] - target[chosen, 6]).mean()) if chosen.any() else None}
        return {"arm_mae_by_channel": np.abs(pred[:, :6] - target[:, :6]).mean(0).tolist(),
                "arm_rmse_by_channel": np.sqrt(np.square(pred[:, :6] - target[:, :6]).mean(0)).tolist(),
                "gripper_mae": float(np.abs(pred[:, 6] - target[:, 6]).mean()),
                "gripper_classes": classes}

    flat_pred = predictions_raw.transpose(1, 0, 2).reshape(-1, 7)
    flat_target = np.repeat(targets_raw, draws, axis=0)
    per_draw = [metrics(predictions_raw[draw], targets_raw) for draw in range(draws)]
    episode_names = [name for name, _ in labels]
    per_episode = []
    for episode in dict.fromkeys(episode_names):
        chosen = np.asarray([item == episode for item in episode_names])
        indices = np.flatnonzero(chosen)
        pred = predictions_raw[:, indices].transpose(1, 0, 2).reshape(-1, 7)
        target = np.repeat(targets_raw[indices], draws, axis=0)
        per_episode.append({"episode": episode, "frames": len(indices), "metrics": metrics(pred, target)})
    np.savez(out / "evaluation.npz", predictions_raw=predictions_raw, targets_raw=targets_raw,
             episodes=np.asarray(episode_names), frames=np.asarray([t for _, t in labels]),
             draws=np.asarray(draws))
    aggregate = metrics(flat_pred, flat_target)
    report.update(status="completed", frames_evaluated=n_eval, labels_scope=f"{split} split, dataset order",
        noise_sha256=noise_sha, noise_generator="NumPy default_rng(923), float32, index-stable",
        arm_mae_by_channel=aggregate["arm_mae_by_channel"],
        arm_rmse_by_channel=aggregate["arm_rmse_by_channel"],
        gripper_mae=aggregate["gripper_mae"],
        gripper_classes=aggregate["gripper_classes"],
        per_draw=per_draw, per_episode=per_episode, evaluation_sha256=file_hash(out / "evaluation.npz"),
        elapsed_seconds=time.monotonic() - started,
        requested_resource_cost_estimate_usd={"gpu_at_4_54_per_hour": (time.monotonic() - started) * H200_PER_SECOND,
            "excludes": "image builds, startup, storage; not an invoice"})
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    volume.commit()
    return report


@app.local_entrypoint()
def main(stage: str = "plan", name: str = "quantized_muse_aligned_dataset_v1",
         dataset: str = "muse_aligned_development_v1", cache: str = "muse_nf4_aligned_train_v1",
         split: str = "train", batch_size: int = 8, updates: int = 5000,
         max_train_seconds: int = 1200, checkpoint_every: int = 2500, lr: float = 1e-4,
         seed: int = 7, resume_step: int = 0, fit_name: str = "quantized_muse_aligned_dataset_v1",
         checkpoint_step: int = 0, draws: int = 1, max_frames: int = 0):
    if stage == "plan":
        result = plan.remote(dataset, cache)
    elif stage == "train":
        result = train.remote(name, dataset, cache, split, batch_size, updates,
                              max_train_seconds, checkpoint_every, lr, seed, resume_step)
    elif stage == "evaluate":
        result = evaluate.remote(name, fit_name, checkpoint_step, dataset, cache, split, draws,
                                 max_frames, batch_size)
    else:
        raise ValueError("stage must be plan, train or evaluate")
    print(json.dumps(result, indent=2) if not isinstance(result, str) else result)
