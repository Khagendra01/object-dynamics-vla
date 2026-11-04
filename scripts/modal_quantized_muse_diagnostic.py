"""Matched-noise in-sample diagnostics for the NF4 Muse/SAM action expert."""
import modal
from scripts.modal_quantized_muse_preflight import image, volume, MUSE_REVISION, SMOL_CONFIG_REVISION

app = modal.App("odvla-quantized-muse-diagnostic")


@app.function(image=image, gpu="H200", cpu=4, memory=131072, volumes={"/data": volume},
              timeout=600, max_containers=1, single_use_containers=True)
def run(name: str = "quantized_muse_full_diagnostic_v1", fit_name: str = "quantized_muse_full_episode_v1"):
    import json
    import sys
    import time
    from pathlib import Path
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
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
    from odvla.data import file_hash, fit_stats, load_episode, read_json, validate_splits
    from odvla.vla.dataset import observation_window
    from odvla.vla.muse_context import FrozenMuseTokens
    from odvla.vla.smolvla_bridge import MuseSamPrefix, expert_velocity
    sys.path.insert(0, "/root")
    from train_muse_vla import make_batch

    if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
        raise ValueError("Invalid output name")
    if not fit_name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in fit_name):
        raise ValueError("Invalid fit name")
    out = Path("/data/eval") / name
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"status": "started", "scope": "in-sample matched-noise action and conditioning diagnostic; no simulator actions"}
    muse = None
    try:
        root = Path("/data/libero/milk_sam_development_v1")
        splits = read_json(root / "splits.json")
        manifest = validate_splits(root, splits)
        entry = next(e for e in manifest["episodes"] if e["id"] == "episode_0000")
        if entry["id"] not in splits["train"]:
            raise ValueError("Expected the training episode")
        episode = load_episode(root, entry, manifest, include_images=False)
        fit = Path("/data/eval") / fit_name
        previous = read_json(fit / "report.json")
        if previous["status"] not in ("completed_full_episode_fit", "completed_subset_fit", "passed_training_interface_only") or previous["dataset_fingerprint"] != splits["fingerprint"]:
            raise ValueError("Fit provenance mismatch")
        if (fit / "stats.npz").exists():
            with np.load(fit / "stats.npz", allow_pickle=False) as z:
                stats = {k: z[k].copy() for k in z.files}
        else:
            # Early preflights predate saving stats; they used the same train split.
            stats = fit_stats(root, splits)
        cfg = SmolVLAConfig(device="cuda", vlm_model_name=str(Path("/data/checkpoints/smolvlm2-reference") / SMOL_CONFIG_REVISION),
            load_vlm_weights=False, input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(15,)),
                "observation.images.agentview": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 512, 512))},
            output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))})
        torch.manual_seed(7)
        policy = SmolVLAPolicy(cfg).cuda().eval()
        model = policy.model
        model.vlm_with_expert.vlm.eval().requires_grad_(False)
        tc = model.vlm_with_expert.config.text_config
        checkpoint_path = fit / "preflight_weights.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        prefix = MuseSamPrefix(muse_width=6656, layers=model.vlm_with_expert.num_vlm_layers,
            heads=tc.num_key_value_heads, head_dim=tc.head_dim,
            normalize_queries=checkpoint.get("normalize_queries", False)).cuda().eval()
        prefix.load_state_dict(checkpoint["prefix"], strict=True)
        params = dict(model.named_parameters())
        if set(checkpoint["trainable_policy"]) != {n for n, p in params.items() if p.requires_grad}:
            raise ValueError("Checkpoint parameter mismatch")
        with torch.no_grad():
            for key, value in checkpoint["trainable_policy"].items(): params[key].copy_(value)
        muse = FrozenMuseTokens(Path("/data/checkpoints/muse-glimmer-30b") / MUSE_REVISION, 26, quantization="nf4")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        quantized = [n for n, m in muse.model.named_modules() if type(m).__name__ == "Linear4bit"]
        if not quantized or any(not n.startswith(muse.module_name + ".") for n in quantized):
            raise ValueError("Unexpected NF4 conversion")
        indices = np.linspace(0, len(episode["timestamps"]) - 1, 8, dtype=int).tolist()
        rows, conditioning_differences, attention_diagnostics = [], [], []
        rgb_root = Path("/data/libero/milk_sam_native_v1") / entry["id"]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for t in indices:
                with Image.open(rgb_root / f"{t:06}.jpg") as rgb:
                    # Match training: the Muse forward itself runs with autocast disabled.
                    with torch.autocast("cuda", enabled=False):
                        tokens = muse.encode(rgb, entry["instruction"], current_date="2026-10-30")
                sample = observation_window(episode, t, stats, history=4, horizon=50)
                sample.update(muse_bits=tokens["features_bfloat16_bits"], muse_valid=tokens["attention_mask"])
                b = make_batch([sample], "cuda")
                target = episode["actions"][t].astype(float)
                full_keys = full_values = None
                for condition in ("full", "no_sam", "zero_muse", "zero_state"):
                    state = F.pad(b["robot"][:, -1], (0, cfg.max_state_dim - 15))
                    if condition == "zero_state": state = torch.zeros_like(state)
                    features = torch.zeros_like(b["muse"]) if condition == "zero_muse" else b["muse"]
                    keys, values, mask = prefix(features, b["muse_valid"], model.state_proj(state),
                        b["sam_values"], b["sam_deltas"], b["sam_valid"], b["sam_age"],
                        b["history_valid"], use_sam=condition != "no_sam")
                    if condition == "full":
                        full_keys, full_values = keys, values
                        vision = prefix.muse(torch.where(b["muse_valid"][..., None], features, 0).float())
                        q = prefix.queries[None] + prefix.resample(prefix.queries[None], vision, vision,
                            key_padding_mask=~b["muse_valid"], need_weights=False)[0]
                        if prefix.normalize_queries:
                            q = F.layer_norm(q, (q.shape[-1],))
                        valid = b["sam_valid"] & b["history_valid"][:, :, None, None]
                        st = prefix.sam(torch.cat((torch.where(valid, b["sam_values"], 0),
                            torch.where(valid, b["sam_deltas"], 0), valid.float(),
                            torch.where(valid, b["sam_age"], 0)), dim=-1))
                        st = st + prefix.slot(torch.arange(st.shape[2], device=st.device))[None, None]
                        st = st + prefix.history(torch.arange(st.shape[1], device=st.device))[None, :, None]
                        state_token = prefix.state(model.state_proj(state).float())[:, None]
                        context = torch.cat((state_token, st.reshape(1, -1, st.shape[-1])), dim=1)
                        context_mask = torch.cat((torch.ones(1, 1, dtype=torch.bool, device=st.device),
                            valid.any(dim=-1).reshape(1, -1)), dim=1)
                        fuse_output, weights = prefix.fuse(q, context, context, key_padding_mask=~context_mask,
                            need_weights=True, average_attn_weights=False)
                        attention_diagnostics.append({"frame": int(t), "state_attention_mean": float(weights[..., 0].float().mean()),
                            "state_attention_max": float(weights[..., 0].float().max()),
                            "valid_sam_tokens": int(context_mask[:, 1:].sum()),
                            "query_mean_norm": float(q.float().norm(dim=-1).mean()),
                            "fuse_output_mean_norm": float(fuse_output.float().norm(dim=-1).mean()),
                            "state_token_norm": float(state_token.float().norm())})
                    else:
                        diffs = [float((a.float() - c.float()).abs().max()) for a, c in zip(
                            [*keys, *values], [*full_keys, *full_values])]
                        conditioning_differences.append({"frame": int(t), "condition": condition,
                            "max_prefix_kv_abs_difference": max(diffs),
                            "normalized_robot_l2": float(b["robot"][:, -1].float().norm()),
                            "state_projection_l2": float(model.state_proj(F.pad(b["robot"][:, -1],
                                (0, cfg.max_state_dim - 15))).float().norm())})
                    for repeat in range(4):
                        seed = 1000 + int(t) + 10000 * repeat
                        generator = torch.Generator(device="cuda").manual_seed(seed)
                        action = torch.randn((1, 50, cfg.max_action_dim), device="cuda", generator=generator)
                        for k in range(10):
                            action = action - .1 * expert_velocity(model, keys, values, mask, action,
                                torch.full((1,), 1 - k / 10, device="cuda"))
                        prediction = action[0, 0, :7].float().cpu().numpy() * stats["actions_std"] + stats["actions_mean"]
                        if not np.isfinite(prediction).all(): raise ValueError("Nonfinite predicted action")
                        rows.append({"frame": int(t), "condition": condition, "seed": seed,
                            "target": target.tolist(), "prediction": prediction.astype(float).tolist(),
                            "predicted_gripper_chunk": (action[0, :, 6].float().cpu().numpy() * stats["actions_std"][6] + stats["actions_mean"][6]).astype(float).tolist(),
                            "target_gripper_chunk": episode["actions"][t:min(t + 50, len(episode["actions"])), 6].astype(float).tolist()})
                print(json.dumps({"completed_frame": int(t)}), flush=True)
        summary = {}
        for condition in ("full", "no_sam", "zero_muse", "zero_state"):
            selected = [r for r in rows if r["condition"] == condition]
            pred = np.asarray([r["prediction"] for r in selected])
            target = np.asarray([r["target"] for r in selected])
            summary[condition] = {"mean_absolute_error_by_channel": np.abs(pred - target).mean(axis=0).tolist(),
                "gripper_sign_accuracy": float(((pred[:, 6] > 0) == (target[:, 6] > 0)).mean()),
                "gripper_mean_prediction": float(pred[:, 6].mean()),
                "gripper_positive_predictions": int((pred[:, 6] > 0).sum())}
        original = previous.get("sampled_action_fit_after", {}).get("frames", [])
        parity = []
        for before in original:
            row = next(r for r in rows if r["condition"] == "full" and r["frame"] == before["frame"] and r["seed"] == 1000 + before["frame"])
            expected = np.asarray(before["prediction_normalized"]) * stats["actions_std"] + stats["actions_mean"]
            parity.append(float(np.max(np.abs(np.asarray(row["prediction"]) - expected))))
        report.update(status="completed", episode=entry["id"], dataset_fingerprint=splits["fingerprint"],
            checkpoint_sha256=file_hash(checkpoint_path), checkpoint_step=checkpoint["completed_updates"],
            sample_count=len(indices), seeds_per_frame=4, action_mean=stats["actions_mean"].tolist(),
            action_std=stats["actions_std"].tolist(), gripper_demo_counts={
                "negative": int((episode["actions"][:, 6] < 0).sum()),
                "positive": int((episode["actions"][:, 6] > 0).sum()),
                "zero": int((episode["actions"][:, 6] == 0).sum())},
            prior_saved_prediction_max_abs_diff=max(parity) if parity else None, summary=summary, rows=rows, conditioning_differences=conditioning_differences,
            attention_diagnostics=attention_diagnostics,
            limitations="Eight in-sample frames; ablations are out-of-distribution interventions, not task success tests.")
        return json.loads(json.dumps(report, allow_nan=False))
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        if muse is not None: muse.close()
        report["elapsed_seconds"] = time.monotonic() - started
        (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        volume.commit()


@app.function(image=image, gpu="H200", cpu=2, memory=65536, volumes={"/data": volume},
              timeout=360, max_containers=1, single_use_containers=True)
def full_evaluate(name: str = "quantized_muse_allframes_v1", fit_name: str = "quantized_muse_full_prenorm_v1", checkpoint_step: int = 0, extra_draws: int = 0):
    """Evaluate all training observations with the native learning-curve noise bank."""
    import json
    import sys
    import time
    from pathlib import Path
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as qwen
    if not hasattr(qwen, "apply_multimodal_rotary_pos_emb"):
        def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
            sections = mrope_section * 2
            cos = torch.cat([m[i % 3] for i,m in enumerate(cos.split(sections, dim=-1))], dim=-1).unsqueeze(unsqueeze_dim)
            sin = torch.cat([m[i % 3] for i,m in enumerate(sin.split(sections, dim=-1))], dim=-1).unsqueeze(unsqueeze_dim)
            return (q*cos)+(qwen.rotate_half(q)*sin), (k*cos)+(qwen.rotate_half(k)*sin)
        qwen.apply_multimodal_rotary_pos_emb = apply_multimodal_rotary_pos_emb
    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.utils.constants import ACTION, OBS_STATE
    from odvla.data import file_hash, load_episode, read_json, validate_splits
    from odvla.vla.dataset import observation_window
    from odvla.vla.muse_context import FrozenMuseTokens
    from odvla.vla.smolvla_bridge import MuseSamPrefix, expert_velocity
    sys.path.insert(0, "/root")
    from train_muse_vla import make_batch
    for value in (name, fit_name):
        if not value or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in value):
            raise ValueError("Invalid output or fit name")
    if extra_draws not in (0, 3): raise ValueError("Use the original bank or exactly three additional draws")
    out = Path("/data/eval") / name; out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); report = {"status": "started", "scope": "150 in-sample frames, one saved shared noise tensor each"}
    muse = None
    try:
        fit = Path("/data/eval") / fit_name
        prior = read_json(fit / "report.json")
        bank = Path("/data/eval/native_exposure_curve_v1")
        bank_report = read_json(bank / "report.json")
        if file_hash(bank / "noise.npy") != bank_report["noise_sha256"]: raise ValueError("Noise bank hash mismatch")
        original_noise = np.load(bank / "noise.npy", allow_pickle=False)
        if original_noise.shape != (150,50,32) or original_noise.dtype != np.float32: raise ValueError("Noise bank shape mismatch")
        if extra_draws:
            noise = np.random.default_rng(923).standard_normal((extra_draws,150,50,32), dtype=np.float32)
            np.save(out / "additional_noise.npy", noise, allow_pickle=False)
            noise_sha = file_hash(out / "additional_noise.npy")
            report.update(noise_sha256=noise_sha, noise_generator="NumPy default_rng(923), float32",
                noise_shape=list(noise.shape), scope="150 in-sample frames, three additional saved noise draws each")
            (out / "report.json").write_text(json.dumps(report,indent=2)+"\n"); volume.commit()
        else:
            noise = original_noise[None]
            noise_sha = bank_report["noise_sha256"]
        root = Path("/data/libero/milk_sam_development_v1")
        splits = read_json(root / "splits.json"); manifest = validate_splits(root, splits)
        if splits["fingerprint"] != prior["dataset_fingerprint"] or splits["fingerprint"] != bank_report["dataset_fingerprint"]:
            raise ValueError("Dataset mismatch")
        entry = next(e for e in manifest["episodes"] if e["id"] == "episode_0000")
        ep = load_episode(root, entry, manifest, include_images=False)
        with np.load(fit / "stats.npz", allow_pickle=False) as z: stats = {k: z[k].copy() for k in z.files}
        with np.load(bank / "stats.npz", allow_pickle=False) as z:
            if any(not np.array_equal(stats[k], z[k]) for k in stats): raise ValueError("Normalization mismatch")
        cfg = SmolVLAConfig(device="cuda", vlm_model_name=str(Path("/data/checkpoints/smolvlm2-reference")/SMOL_CONFIG_REVISION),
            load_vlm_weights=False, input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(15,)),
                "observation.images.agentview": PolicyFeature(type=FeatureType.VISUAL, shape=(3,512,512))},
            output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))})
        torch.manual_seed(7); policy = SmolVLAPolicy(cfg).cuda().eval(); model = policy.model
        model.vlm_with_expert.vlm.eval().requires_grad_(False); tc = model.vlm_with_expert.config.text_config
        checkpoint_path = fit / (f"checkpoint_{checkpoint_step}.pt" if checkpoint_step else "preflight_weights.pt")
        if checkpoint_step and checkpoint_step not in (5000, 7500, 10000): raise ValueError("Unknown milestone")
        if checkpoint_step and checkpoint_step not in [r["step"] for r in prior["committed_checkpoints"]]:
            raise ValueError("Checkpoint was not recorded as committed")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint_step and checkpoint["completed_updates"] != checkpoint_step: raise ValueError("Checkpoint step mismatch")
        prefix = MuseSamPrefix(muse_width=6656, layers=model.vlm_with_expert.num_vlm_layers, heads=tc.num_key_value_heads,
            head_dim=tc.head_dim, normalize_queries=checkpoint.get("normalize_queries", False)).cuda().eval()
        prefix.load_state_dict(checkpoint["prefix"], strict=True); params = dict(model.named_parameters())
        if set(checkpoint["trainable_policy"]) != {n for n,p in params.items() if p.requires_grad}: raise ValueError("Checkpoint mismatch")
        with torch.no_grad():
            for k,v in checkpoint["trainable_policy"].items(): params[k].copy_(v)
        muse = FrozenMuseTokens(Path("/data/checkpoints/muse-glimmer-30b") / MUSE_REVISION, 26, quantization="nf4")
        converted = [n for n,m in muse.model.named_modules() if type(m).__name__ == "Linear4bit"]
        if not converted or any(not n.startswith(muse.module_name+".") for n in converted): raise ValueError("NF4 conversion mismatch")
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        rows = []; rgb_root = Path("/data/libero/milk_sam_native_v1")/entry["id"]
        for t in range(150):
            with Image.open(rgb_root/f"{t:06}.jpg") as rgb: tokens = muse.encode(rgb, entry["instruction"], current_date="2026-10-30")
            sample = observation_window(ep, t, stats, history=4, horizon=50)
            sample.update(muse_bits=tokens["features_bfloat16_bits"], muse_valid=tokens["attention_mask"])
            b = make_batch([sample], "cuda")
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                state = F.pad(b["robot"][:, -1], (0, cfg.max_state_dim-15))
                keys,values,mask = prefix(b["muse"],b["muse_valid"],model.state_proj(state),b["sam_values"],b["sam_deltas"],
                    b["sam_valid"],b["sam_age"],b["history_valid"],use_sam=True)
                for draw in range(len(noise)):
                    action = torch.from_numpy(noise[draw,t:t+1]).cuda()
                    for k in range(10):
                        action = action - .1*expert_velocity(model,keys,values,mask,action,torch.full((1,),1-k/10,device="cuda"))
                    predicted = action[0,0,:7].float().cpu().numpy()
                    raw = predicted*stats["actions_std"]+stats["actions_mean"]
                    if not np.isfinite(raw).all(): raise ValueError("Nonfinite action")
                    rows.append({"frame": t, "draw":draw, "prediction": raw.astype(float).tolist(), "target": ep["actions"][t].astype(float).tolist()})
            if (t+1)%25 == 0: print(json.dumps({"frames":t+1,"seconds":time.monotonic()-started}),flush=True)
        pred=np.asarray([r["prediction"] for r in rows]); target=np.asarray([r["target"] for r in rows]); classes={}
        for sign,label in ((-1,"negative"),(1,"positive")):
            chosen=target[:,6]*sign>0
            classes[label]={"count":int(chosen.sum()),"sign_accuracy":float(((pred[chosen,6]>0)==(target[chosen,6]>0)).mean()),
                "mae":float(np.abs(pred[chosen,6]-target[chosen,6]).mean())}
        report.update(status="completed", frames=150, checkpoint_sha256=file_hash(checkpoint_path),
            checkpoint_step=checkpoint["completed_updates"], noise_sha256=noise_sha, dataset_fingerprint=splits["fingerprint"],
            arm_mae_by_channel=np.abs(pred[:,:6]-target[:,:6]).mean(0).tolist(), arm_rmse_by_channel=np.sqrt(np.square(pred[:,:6]-target[:,:6]).mean(0)).tolist(),
            gripper_classes=classes, gripper_mae=float(np.abs(pred[:,6]-target[:,6]).mean()), draws_per_frame=len(noise), rows=rows)
        if extra_draws:
            report["per_draw"] = []
            for draw in range(len(noise)):
                subset=[r for r in rows if r["draw"]==draw]
                p=np.asarray([r["prediction"] for r in subset]); a=np.asarray([r["target"] for r in subset]); cc={}
                for sign,label in ((-1,"negative"),(1,"positive")):
                    chosen=a[:,6]*sign>0
                    cc[label]={"count":int(chosen.sum()),"sign_accuracy":float(((p[chosen,6]>0)==(a[chosen,6]>0)).mean()),
                        "mae":float(np.abs(p[chosen,6]-a[chosen,6]).mean())}
                report["per_draw"].append({"draw":draw,"arm_mae_by_channel":np.abs(p[:,:6]-a[:,:6]).mean(0).tolist(),
                    "gripper_mae":float(np.abs(p[:,6]-a[:,6]).mean()),"gripper_classes":cc})
        return {k:v for k,v in report.items() if k!="rows"}
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc)); raise
    finally:
        if muse is not None: muse.close()
        report["elapsed_seconds"]=time.monotonic()-started
        report["requested_resource_cost_estimate_usd"]=report["elapsed_seconds"]*(4.54+2*.0473+64*.008)/3600
        (out/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n"); volume.commit()


@app.local_entrypoint()
def main(name: str = "quantized_muse_full_diagnostic_v1", fit_name: str = "quantized_muse_full_episode_v1", stage: str = "diagnose", checkpoint_step: int = 0, extra_draws: int = 0):
    if stage == "full": result = full_evaluate.remote(name, fit_name, checkpoint_step, extra_draws)
    elif stage == "diagnose": result = run.remote(name, fit_name)
    else: raise ValueError("Unknown stage")
    print({k: v for k, v in result.items() if k != "rows"})
