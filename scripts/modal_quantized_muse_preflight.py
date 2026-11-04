"""Bounded NF4 Muse + SAM-window + random official-expert training preflight.

No robot-trained checkpoint, teacher, old Muse cache, simulator masks, or test
episodes are used. This verifies a training interface, not successful control.
"""
import modal
from scripts.modal_smolvla_reference import reference_base, REVISION

MUSE_REVISION = "cec94fcfd6512f2cfe708138c32533b346ea4f69"
SMOL_CONFIG_REVISION = "7b375e1b73b11138ff12fe22c8f2822d8fe03467"
image = (reference_base
    .pip_install("transformers==5.21.0", "bitsandbytes==0.48.2", "accelerate", "numpy<2")
    .add_local_python_source("odvla")
    .add_local_file("scripts/train_muse_vla.py", "/root/train_muse_vla.py"))
volume = modal.Volume.from_name("object-dynamics-vla-data", create_if_missing=False)
app = modal.App("odvla-quantized-muse-preflight")


def validate_request(updates, frames, max_seconds, name):
    if not 2 <= updates <= 8000 or not (frames == 0 or 2 <= frames <= 64):
        raise ValueError("Use 2..8000 updates and 2..64 frames, or 0 for the full episode")
    if not 30 <= max_seconds <= 600:
        raise ValueError("Training wall-clock cap must be 30..600 seconds")
    if not name or len(name) > 80 or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
        raise ValueError("Use a simple, fresh output name")


@app.function(image=image, gpu="H200", cpu=4, memory=131072,
              volumes={"/data": volume}, timeout=900, max_containers=1,
              single_use_containers=True)
def run(updates: int = 20, frames: int = 4, max_seconds: int = 120,
        name: str = "quantized_muse_preflight_v1", normalize_queries: bool = False,
        dataset: str = "milk_sam_development_v1"):
    import hashlib
    import importlib.metadata
    import json
    import sys
    import time
    from pathlib import Path
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from odvla.data import file_hash, fit_stats, load_episode, read_json, validate_splits
    from odvla.vla.dataset import observation_window
    from odvla.vla.muse_context import FrozenMuseTokens, backend_snapshot
    from odvla.vla.smolvla_bridge import MuseSamPrefix, expert_velocity
    # LeRobot eagerly imports Wall-OSS, which references a rotary helper removed
    # in Transformers 5.21. Restore the 5.5 helper for that unrelated import;
    # neither Wall-OSS nor this helper is used by the SmolVLA expert.
    import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as qwen
    if not hasattr(qwen, "apply_multimodal_rotary_pos_emb"):
        def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
            sections = mrope_section * 2
            cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(sections, dim=-1))], dim=-1).unsqueeze(unsqueeze_dim)
            sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(sections, dim=-1))], dim=-1).unsqueeze(unsqueeze_dim)
            return (q * cos) + (qwen.rotate_half(q) * sin), (k * cos) + (qwen.rotate_half(k) * sin)
        qwen.apply_multimodal_rotary_pos_emb = apply_multimodal_rotary_pos_emb
    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.utils.constants import ACTION, OBS_STATE
    sys.path.insert(0, "/root")
    from train_muse_vla import make_batch

    validate_request(updates, frames, max_seconds, name)
    if not dataset or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in dataset):
        raise ValueError("Invalid dataset name")
    started = time.monotonic()
    out = Path("/data/eval") / name
    out.mkdir(parents=True, exist_ok=False)
    report = {"status": "started", "purpose": "full training demonstration fitting" if frames == 0 else ("bounded single-demonstration subset fitting" if frames > 8 else "training-interface preflight only"),
        "muse_revision": MUSE_REVISION, "lerobot_commit": REVISION,
        "smol_config_revision": SMOL_CONFIG_REVISION,
        "robot_pretrained_weights_loaded": False, "teacher_used": False,
        "old_bf16_muse_cache_used": False, "dynamics_objective": False,
        "sam": "existing causal SAM cache, RGB-initialized; not live extraction in this probe",
        "test_accessed": False, "seed": 7, "requested_updates": updates, "dataset": dataset,
        "requested_frames": frames, "training_seconds_cap": max_seconds,
        "batch_size": 1, "gradient_accumulation": 1,
        "normalize_queries": normalize_queries,
        "scope": "Muse hidden states -> existing learned prefix -> random official expert; architectural adaptation, not exact SmolVLA reproduction"}
    muse = None

    def save():
        report["remote_function_seconds"] = time.monotonic() - started
        seconds = report["remote_function_seconds"]
        report["requested_resource_cost_estimate_usd"] = {
            "gpu_at_4_54_per_hour": seconds * 4.54 / 3600,
            "cpu_4_physical_cores_at_0_0000131_per_second": seconds * 4 * .0000131,
            "host_memory_128_gib_at_0_00000222_per_second": seconds * 128 * .00000222,
            "excludes": "image builds, startup before function entry, storage and other functions; not an invoice"}
        (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        volume.commit()

    try:
        root = Path("/data/libero") / dataset
        splits = read_json(root / "splits.json")
        manifest = validate_splits(root, splits)
        stats = fit_stats(root, splits)
        entry = next(e for e in manifest["episodes"] if e["id"] in splits["train"])
        ep = load_episode(root, entry, manifest, include_images=False)
        if frames and len(ep["timestamps"]) < frames + 1:
            raise ValueError("Training episode is too short")
        rgb_root = Path("/data/libero/milk_sam_native_v1") / entry["id"]
        frame_indices = (list(range(len(ep["timestamps"]))) if frames == 0 else
                         np.linspace(0, len(ep["timestamps"]) - 2, frames, dtype=int).tolist()
                         if frames > 8 else list(range(frames)))
        if frames and len(set(frame_indices)) != frames:
            raise ValueError("Training episode has too few distinct frames")
        model_root = Path("/data/checkpoints/muse-glimmer-30b") / MUSE_REVISION
        smol_config = Path("/data/checkpoints/smolvlm2-reference") / SMOL_CONFIG_REVISION
        for path in (model_root / "config.json", smol_config / "config.json"):
            if not path.exists():
                raise FileNotFoundError(f"Missing existing asset: {path}; no implicit download in GPU job")
        for t in frame_indices:
            if not (rgb_root / f"{t:06}.jpg").exists():
                raise FileNotFoundError(f"Missing RGB frame {t}")
        report.update(dataset_fingerprint=splits["fingerprint"], episode=entry["id"],
            instruction=entry["instruction"], packages={k: importlib.metadata.version(k)
                for k in ("torch", "transformers", "bitsandbytes", "accelerate")},
            cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0))
        torch.manual_seed(7)
        cfg = SmolVLAConfig(device="cuda", vlm_model_name=str(smol_config),
            load_vlm_weights=False,
            input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(15,)),
                "observation.images.agentview": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 512, 512))},
            output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))})
        # Constructor only: no from_pretrained(), state_dict load, or native VLM forward.
        policy = SmolVLAPolicy(cfg).cuda()
        model = policy.model
        model.vlm_with_expert.vlm.eval().requires_grad_(False)
        tc = model.vlm_with_expert.config.text_config
        if tc.hidden_size != 960:
            raise ValueError("Existing prefix adapter expects the pinned 960-wide expert interface")
        muse = FrozenMuseTokens(model_root, layer_index=26, quantization="nf4")
        # Constructors can modify these flags. Set and record them after both loads.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        report["backend_before_muse"] = backend_snapshot(torch, muse.model)
        quantized = [n for n, m in muse.model.named_modules() if type(m).__name__ == "Linear4bit"]
        report["quantized_linear_modules"] = quantized
        if not quantized:
            raise ValueError("No actual Linear4bit modules found")
        if any("vision" in n.lower() for n in quantized):
            raise ValueError("Vision modules were quantized unexpectedly; fix exclusions before proceeding")
        outside_decoder = [n for n in quantized if not n.startswith(muse.module_name + ".")]
        if outside_decoder:
            raise ValueError(f"Unexpected NF4 modules outside Muse text decoder: {outside_decoder[:8]}")
        report["quantization"] = muse.model.config.quantization_config.to_dict()
        report["muse_model_memory_bytes"] = muse.model.get_memory_footprint()
        samples, records = [], []
        feature_start = time.monotonic()
        for t in frame_indices:
            path = rgb_root / f"{t:06}.jpg"
            torch.cuda.synchronize()
            stamp = time.monotonic()
            with Image.open(path) as rgb:
                tokens = muse.encode(rgb, entry["instruction"], current_date="2026-10-30")
            torch.cuda.synchronize()
            sample = observation_window(ep, t, stats, history=4, horizon=50)
            sample.update(muse_bits=tokens["features_bfloat16_bits"], muse_valid=tokens["attention_mask"])
            samples.append(make_batch([sample], "cuda"))
            records.append({"frame": t, "rgb_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "feature_sha256": hashlib.sha256(tokens["features_bfloat16_bits"].tobytes()).hexdigest(),
                "shape": list(tokens["features_bfloat16_bits"].shape),
                "muse_forward_seconds": time.monotonic() - stamp})
            if frames == 0 and len(records) % 25 == 0:
                print(json.dumps({"features_extracted": len(records), "seconds": time.monotonic() - feature_start}), flush=True)
        report["frames"] = records
        report["feature_extraction_seconds"] = time.monotonic() - feature_start
        torch.manual_seed(8)
        prefix = MuseSamPrefix(muse_width=samples[0]["muse"].shape[-1],
            layers=model.vlm_with_expert.num_vlm_layers,
            heads=tc.num_key_value_heads, head_dim=tc.head_dim, normalize_queries=normalize_queries).cuda()
        expert = model.vlm_with_expert.lm_expert
        trainable = list(prefix.parameters()) + [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=1e-4, betas=(.9, .95))

        def conditioning(b):
            state = F.pad(b["robot"][:, -1], (0, cfg.max_state_dim - 15))
            return prefix(b["muse"], b["muse_valid"], model.state_proj(state),
                b["sam_values"], b["sam_deltas"], b["sam_valid"], b["sam_age"],
                b["history_valid"], use_sam=True)

        def evaluate_actions():
            # The same fixed per-frame noise before and after training makes this
            # a comparable in-sample diagnostic, not a generalization metric.
            prefix.eval(); expert.eval()
            indices = np.linspace(0, len(samples) - 1, min(8, len(samples)), dtype=int)
            results = []
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for index in indices:
                    b = samples[int(index)]
                    keys, values, mask = conditioning(b)
                    generator = torch.Generator(device="cuda").manual_seed(1000 + int(index))
                    action = torch.randn((1, 50, cfg.max_action_dim), device="cuda", generator=generator)
                    for k in range(10):
                        action = action - .1 * expert_velocity(model, keys, values, mask, action,
                            torch.full((1,), 1 - k / 10, device="cuda"))
                    pred = action[0, 0, :7].float()
                    target = b["actions"][0, 0].float()
                    if not torch.isfinite(pred).all(): raise ValueError("Nonfinite evaluation action")
                    results.append({"frame": frame_indices[int(index)], "prediction_normalized": pred.cpu().tolist(),
                        "target_normalized": target.cpu().tolist(), "mae_normalized": float((pred - target).abs().mean())})
            prefix.train(); expert.train()
            return {"scope": "eight in-sample first actions; fixed sampling noise", "frames": results,
                "mean_first_action_mae_normalized": float(np.mean([r["mae_normalized"] for r in results]))}

        if frames == 0:
            report["sampled_action_fit_before"] = evaluate_actions()

        expert_parameter = next(p for p in expert.parameters() if p.requires_grad)
        prefix_parameter = prefix.kv[0].weight
        initial_expert = expert_parameter.detach().clone()
        initial_prefix = prefix_parameter.detach().clone()
        rows = []
        prefix.train(); expert.train()
        train_started = time.monotonic()
        for step in range(updates):
            if step >= 2 and time.monotonic() - train_started >= max_seconds:
                break
            b = samples[step % len(samples)]
            torch.cuda.synchronize(); stamp = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            target_actions = F.pad(b["actions"], (0, cfg.max_action_dim - 7))
            noise = model.sample_noise(target_actions.shape, "cuda")
            time_value = model.sample_time(len(target_actions), "cuda")
            xt = time_value[:, None, None] * noise + (1 - time_value[:, None, None]) * target_actions
            with torch.autocast("cuda", dtype=torch.bfloat16):
                keys, values, mask = conditioning(b)
                prediction = expert_velocity(model, keys, values, mask, xt, time_value)
                valid = b["action_valid"][..., None]
                losses = (prediction - (noise - target_actions)).square()
                # Match the official policy's real action dimensions and padding.
                loss = (losses[:, :, :7] * valid).sum() / (valid.sum() * 7).clamp_min(1)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite flow loss")
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(trainable, 10, error_if_nonfinite=True)
            if step == 0:
                report["first_expert_gradient_norm"] = float(sum(p.grad.float().square().sum()
                    for p in expert.parameters() if p.grad is not None).sqrt())
                report["first_prefix_gradient_norm"] = float(prefix_parameter.grad.float().norm())
                report["first_sam_encoder_gradient_norm"] = float(prefix.sam.weight.grad.float().norm())
                if not all(report[k] for k in ("first_expert_gradient_norm", "first_prefix_gradient_norm",
                                              "first_sam_encoder_gradient_norm")):
                    raise ValueError("Missing expert or conditioning gradient")
            optimizer.step()
            torch.cuda.synchronize()
            rows.append({"step": step + 1, "flow_loss": float(loss.detach()),
                "gradient_norm": float(grad), "update_seconds": time.monotonic() - stamp})
            if updates <= 20 or step == 0 or (step + 1) % 100 == 0:
                print(json.dumps(rows[-1]), flush=True)
        report["updates"] = rows
        if frames == 0:
            report["sampled_action_fit_after"] = evaluate_actions()
        report["expert_parameter_update_norm"] = float((expert_parameter - initial_expert).detach().float().norm())
        report["prefix_parameter_update_norm"] = float((prefix_parameter - initial_prefix).detach().float().norm())
        if not report["expert_parameter_update_norm"] or not report["prefix_parameter_update_norm"]:
            raise ValueError("Optimizer did not update both expert and conditioning")
        frozen = list(muse.model.parameters()) + list(model.vlm_with_expert.vlm.parameters())
        report["frozen_backbone_gradient_count"] = sum(p.grad is not None for p in frozen)
        if any(p.requires_grad or p.grad is not None for p in frozen):
            raise ValueError("Frozen-backbone gradient contract failed")
        prefix.eval(); expert.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            keys, values, mask = conditioning(samples[0])
            torch.manual_seed(921)
            action = model.sample_noise((1, 50, cfg.max_action_dim), "cuda")
            for i in range(10):
                t = torch.full((1,), 1 - i / 10, device="cuda")
                action = action - .1 * expert_velocity(model, keys, values, mask, action, t)
        if not torch.isfinite(action).all():
            raise ValueError("Nonfinite sampled actions")
        report.update(sampled_action_shape=list(action[:, :, :7].shape),
            sampled_first_action_normalized=action[0, 0, :7].float().cpu().tolist(),
            sampled_action_executed=False, peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
            peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024**3,
            backend_after_sampling=backend_snapshot(torch, muse.model))
        # First two updates include cold kernels and Adam-state allocation.
        steady = [r["update_seconds"] for r in rows[2:]]
        report["warm_update_seconds_median"] = float(np.median(steady)) if steady else None
        report["throughput_scope"] = "batch-1 expert+prefix updates on newly extracted NF4 windows; excludes Muse forwards"
        np.savez(out / "stats.npz", **stats)
        torch.save({"prefix": prefix.state_dict(), "trainable_policy": {
            n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
            "completed_updates": len(rows), "normalize_queries": normalize_queries,
            "dataset_fingerprint": splits["fingerprint"]}, out / "preflight_weights.pt")
        report["weights_sha256"] = file_hash(out / "preflight_weights.pt")
        report["stats_sha256"] = file_hash(out / "stats.npz")
        report["status"] = "completed_full_episode_fit" if frames == 0 else ("completed_subset_fit" if frames > 8 else "passed_training_interface_only")
        # Quantization metadata contains Transformers enum instances; return
        # plain JSON values so the local Modal client needs no Transformers.
        return json.loads(json.dumps(report, allow_nan=False))
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        if muse is not None:
            muse.close()
        save()


@app.local_entrypoint()
def main(updates: int = 20, frames: int = 4, max_seconds: int = 120,
         name: str = "quantized_muse_preflight_v1", normalize_queries: bool = False,
         dataset: str = "milk_sam_development_v1"):
    import json
    validate_request(updates, frames, max_seconds, name)
    print(json.dumps(run.remote(updates, frames, max_seconds, name, normalize_queries, dataset), indent=2))
