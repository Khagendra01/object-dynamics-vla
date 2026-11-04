"""Matched single-demonstration native SmolVLA scratch-expert reference."""
import modal
from scripts.modal_smolvla_reference import image, REVISION, MODEL_REVISION

volume = modal.Volume.from_name("object-dynamics-vla-data", create_if_missing=False)
app = modal.App("odvla-native-scratch-reference")


@app.function(image=image, gpu="H200", cpu=4, memory=131072, volumes={"/data": volume},
              timeout=2400, max_containers=1, single_use_containers=True)
def train(steps: int = 3000, name: str = "native_scratch_episode0000_v1"):
    import hashlib
    import importlib.metadata
    import json
    import time
    from pathlib import Path
    import numpy as np
    import torch
    from PIL import Image
    from odvla.data import file_hash, load_episode, read_json, validate_splits
    from odvla.vla.dataset import observation_window
    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
    from lerobot.utils.constants import ACTION, OBS_STATE

    if not 2 <= steps <= 3000 or not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
        raise ValueError("Invalid bounded training request")
    out = Path("/data/eval") / name
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"status": "started", "requested_updates": steps, "batch_size": 1,
        "seed_model": 7, "seed_training": 8, "optimizer": "AdamW lr=1e-4 betas=(0.9,0.95) default weight_decay=0.01",
        "training_precision": "CUDA BF16 autocast", "frame_schedule": "episode order repeated, one frame per update",
        "lerobot_commit": REVISION, "smolvlm_revision": MODEL_REVISION, "robot_policy_checkpoint_loaded": False,
        "expert_initialization": "fresh random SmolVLA expert and projections", "backbone": "frozen pretrained SmolVLM2-500M-Video-Instruct",
        "sam_used": False, "dynamics_objective": False, "test_accessed": False}
    try:
        root = Path("/data/libero/milk_sam_development_v1")
        splits = read_json(root / "splits.json")
        manifest = validate_splits(root, splits)
        entry = next(e for e in manifest["episodes"] if e["id"] == "episode_0000")
        if entry["id"] not in splits["train"]: raise ValueError("Episode is not in training split")
        ep = load_episode(root, entry, manifest, include_images=False)
        reference = Path("/data/eval/quantized_muse_full_episode_v1")
        prior = read_json(reference / "report.json")
        if prior["dataset_fingerprint"] != splits["fingerprint"] or prior["episode"] != entry["id"]:
            raise ValueError("Muse comparison provenance mismatch")
        with np.load(reference / "stats.npz", allow_pickle=False) as z:
            stats = {k: z[k].copy() for k in z.files}
        smol = Path("/data/checkpoints/smolvlm2-reference") / MODEL_REVISION
        if not (smol / "model.safetensors").is_file(): raise FileNotFoundError("Missing pinned pretrained SmolVLM weights")
        key = "observation.images.agentview"
        cfg = SmolVLAConfig(device="cuda", vlm_model_name=str(smol), load_vlm_weights=True,
            input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(15,)),
                key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 512, 512))},
            output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))})
        torch.manual_seed(7)
        policy = SmolVLAPolicy(cfg).cuda()
        vlm = policy.model.vlm_with_expert.vlm
        if any(p.requires_grad for p in vlm.parameters()): raise ValueError("Native VLM must be frozen")
        expert = policy.model.vlm_with_expert.lm_expert
        pre, _ = make_smolvla_pre_post_processors(cfg, {
            OBS_STATE: {"mean": torch.from_numpy(stats["robot_mean"]), "std": torch.from_numpy(stats["robot_std"])},
            ACTION: {"mean": torch.from_numpy(stats["actions_mean"]), "std": torch.from_numpy(stats["actions_std"])}})
        batches, hashes = [], []
        rgb_root = Path("/data/libero/milk_sam_native_v1") / entry["id"]
        prep_start = time.monotonic()
        for t in range(len(ep["timestamps"])):
            frame = rgb_root / f"{t:06}.jpg"
            with Image.open(frame) as im: rgb = np.asarray(im.convert("RGB")).copy()
            count = min(50, len(ep["timestamps"]) - t)
            actions = np.zeros((1, 50, 7), dtype=np.float32)
            actions[0, :count] = ep["actions"][t:t + count]
            b = pre({OBS_STATE: torch.from_numpy(ep["robot"][t].copy()),
                key: torch.from_numpy(rgb).permute(2, 0, 1), ACTION: torch.from_numpy(actions),
                "action_is_pad": torch.from_numpy((np.arange(50) >= count)[None]), "task": entry["instruction"]})
            w = observation_window(ep, t, stats, history=4, horizon=50)
            torch.testing.assert_close(b[OBS_STATE][0].cpu(), torch.from_numpy(w["robot"][-1]), rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(b[ACTION][0, :count].cpu(), torch.from_numpy(w["actions"][:count]), rtol=1e-5, atol=1e-5)
            batches.append(b)
            hashes.append(hashlib.sha256(frame.read_bytes()).hexdigest())
        if hashes != [x["rgb_sha256"] for x in prior["frames"]]: raise ValueError("RGB frames differ from Muse fit")
        report.update(episode=entry["id"], instruction=entry["instruction"], dataset_fingerprint=splits["fingerprint"],
            frames=len(batches), frame_sha256=hashes, preprocess_seconds=time.monotonic()-prep_start,
            pretrained_smolvlm_sha256=file_hash(smol / "model.safetensors"),
            packages={n: importlib.metadata.version(n) for n in ("torch", "transformers", "lerobot")},
            gpu=torch.cuda.get_device_name(0), action_stats_source=str(reference / "stats.npz"))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.manual_seed(8)

        def evaluate():
            policy.eval()
            results = []
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for t in np.linspace(0, len(batches)-1, 8, dtype=int):
                    generator = torch.Generator(device="cuda").manual_seed(1000 + int(t))
                    noise = torch.randn((1, 50, cfg.max_action_dim), generator=generator, device="cuda")
                    pred = policy.predict_action_chunk(batches[int(t)], noise=noise)[0, 0].float()
                    target = batches[int(t)][ACTION][0, 0].float()
                    if not torch.isfinite(pred).all(): raise ValueError("Nonfinite sampled action")
                    results.append({"frame": int(t), "prediction_normalized": pred.cpu().tolist(),
                        "target_normalized": target.cpu().tolist(), "mae_normalized": float((pred-target).abs().mean())})
            return {"scope": "eight in-sample first actions; same fixed sampling noise as Muse",
                "frames": results, "mean_first_action_mae_normalized": float(np.mean([r["mae_normalized"] for r in results]))}

        report["sampled_action_fit_before"] = evaluate()
        trainable = [p for p in policy.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=1e-4, betas=(.9, .95))
        losses = []
        policy.train()
        vlm.eval()
        train_started = time.monotonic()
        for step in range(steps):
            b = batches[step % len(batches)]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = policy.forward(b)
            if not torch.isfinite(loss): raise ValueError("Nonfinite flow loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(trainable, 10, error_if_nonfinite=True)
            if step == 0:
                report["first_expert_gradient_norm"] = float(sum(p.grad.float().square().sum() for p in expert.parameters() if p.grad is not None).sqrt())
                if not report["first_expert_gradient_norm"] or any(p.grad is not None for p in vlm.parameters()):
                    raise ValueError("Expert/frozen-backbone gradient contract failed")
            optimizer.step()
            losses.append(float(loss.detach()))
            if step == 0 or (step + 1) % 100 == 0:
                print(json.dumps({"step": step+1, "loss": losses[-1], "gradient_norm": float(norm),
                    "training_seconds": time.monotonic()-train_started}), flush=True)
        report["training_seconds"] = time.monotonic()-train_started
        report["losses"] = losses
        report["sampled_action_fit_after"] = evaluate()
        report["frozen_backbone_gradient_count"] = sum(p.grad is not None for p in vlm.parameters())
        report["peak_allocated_gib"] = torch.cuda.max_memory_allocated()/1024**3
        np.savez(out / "stats.npz", **stats)
        torch.save({"trainable_policy": {n: p.detach().cpu() for n, p in policy.named_parameters() if p.requires_grad},
            "completed_updates": steps}, out / "weights.pt")
        report["status"] = "completed" if steps == 3000 else "completed_smoke"
        return json.loads(json.dumps({k: v for k, v in report.items() if k != "losses"}, allow_nan=False))
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        report["remote_function_seconds"] = time.monotonic() - started
        (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        volume.commit()


@app.function(image=image, gpu="H200", cpu=4, memory=16384, volumes={"/data": volume},
              timeout=600, max_containers=1, single_use_containers=True)
def diagnose(name: str = "native_scratch_episode0000_diagnostic_v1"):
    import json
    import time
    from pathlib import Path
    import numpy as np
    import torch
    from PIL import Image
    from odvla.data import file_hash, load_episode, read_json, validate_splits
    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
    from lerobot.utils.constants import ACTION, OBS_STATE
    if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
        raise ValueError("Invalid output name")
    out = Path("/data/eval") / name
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"status": "started", "scope": "eight in-sample frames, four fixed noise seeds per frame"}
    try:
        fit = Path("/data/eval/native_scratch_episode0000_v1")
        prior = read_json(fit / "report.json")
        if prior["status"] != "completed": raise ValueError("Native full fit is missing")
        with np.load(fit / "stats.npz", allow_pickle=False) as z: stats = {k: z[k].copy() for k in z.files}
        root = Path("/data/libero/milk_sam_development_v1")
        splits = read_json(root / "splits.json")
        manifest = validate_splits(root, splits)
        if prior["dataset_fingerprint"] != splits["fingerprint"]: raise ValueError("Dataset mismatch")
        entry = next(e for e in manifest["episodes"] if e["id"] == "episode_0000")
        ep = load_episode(root, entry, manifest, include_images=False)
        smol = Path("/data/checkpoints/smolvlm2-reference") / MODEL_REVISION
        if file_hash(smol / "model.safetensors") != prior["pretrained_smolvlm_sha256"]: raise ValueError("Backbone mismatch")
        key = "observation.images.agentview"
        cfg = SmolVLAConfig(device="cuda", vlm_model_name=str(smol), load_vlm_weights=True,
            input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(15,)),
                key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 512, 512))},
            output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))})
        torch.manual_seed(7)
        policy = SmolVLAPolicy(cfg).cuda().eval()
        checkpoint = torch.load(fit / "weights.pt", map_location="cpu", weights_only=True)
        params = dict(policy.named_parameters())
        if set(checkpoint["trainable_policy"]) != {n for n, p in params.items() if p.requires_grad}:
            raise ValueError("Native trainable parameter mismatch")
        with torch.no_grad():
            for k, v in checkpoint["trainable_policy"].items(): params[k].copy_(v)
        pre, _ = make_smolvla_pre_post_processors(cfg, {
            OBS_STATE: {"mean": torch.from_numpy(stats["robot_mean"]), "std": torch.from_numpy(stats["robot_std"])},
            ACTION: {"mean": torch.from_numpy(stats["actions_mean"]), "std": torch.from_numpy(stats["actions_std"])}})
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        rows = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for t in np.linspace(0, len(ep["timestamps"])-1, 8, dtype=int):
                with Image.open(Path("/data/libero/milk_sam_native_v1") / entry["id"] / f"{t:06}.jpg") as im:
                    rgb = np.asarray(im.convert("RGB")).copy()
                b = pre({OBS_STATE: torch.from_numpy(ep["robot"][t].copy()),
                    key: torch.from_numpy(rgb).permute(2, 0, 1), "task": entry["instruction"]})
                for repeat in range(4):
                    seed = 1000 + int(t) + 10000 * repeat
                    generator = torch.Generator(device="cuda").manual_seed(seed)
                    noise = torch.randn((1, 50, cfg.max_action_dim), generator=generator, device="cuda")
                    predicted = policy.predict_action_chunk(b, noise=noise)[0, 0].float().cpu().numpy()
                    raw = predicted * stats["actions_std"] + stats["actions_mean"]
                    if not np.isfinite(raw).all(): raise ValueError("Nonfinite action")
                    rows.append({"frame": int(t), "seed": seed, "prediction": raw.astype(float).tolist(),
                        "prediction_normalized": predicted.astype(float).tolist(), "target": ep["actions"][t].astype(float).tolist()})
        pred = np.asarray([r["prediction"] for r in rows])
        target = np.asarray([r["target"] for r in rows])
        parity = []
        for old in prior["sampled_action_fit_after"]["frames"]:
            row = next(r for r in rows if r["frame"] == old["frame"] and r["seed"] == 1000 + old["frame"])
            parity.append(float(np.max(np.abs(np.asarray(row["prediction_normalized"]) - old["prediction_normalized"]))))
        report.update(status="completed", checkpoint_sha256=file_hash(fit / "weights.pt"),
            dataset_fingerprint=splits["fingerprint"], sample_count=len(rows), rows=rows,
            mean_absolute_error_by_channel=np.abs(pred-target).mean(axis=0).tolist(),
            gripper_sign_accuracy=float(((pred[:, 6] > 0) == (target[:, 6] > 0)).mean()),
            gripper_positive_predictions=int((pred[:, 6] > 0).sum()),
            prior_evaluation_max_abs_difference=max(parity))
        return {k: v for k, v in report.items() if k != "rows"}
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        volume.commit()


@app.function(image=image, gpu="H200", cpu=2, memory=16384, volumes={"/data": volume},
              timeout=690, max_containers=1, single_use_containers=True)
def continue_exposure(name: str = "native_exposure_curve_v1"):
    """Restart AdamW at 3000 and record fixed-noise learning curve under a time budget."""
    import hashlib
    import json
    import time
    from pathlib import Path
    import numpy as np
    import torch
    from PIL import Image
    from odvla.data import file_hash, load_episode, read_json, validate_splits
    from odvla.vla.dataset import observation_window
    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
    from lerobot.utils.constants import ACTION, OBS_STATE

    if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
        raise ValueError("Invalid output name")
    out = Path("/data/eval") / name
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"status": "started", "budget": "690s H200, 2 CPU, 16GiB hard function timeout; stop updates at 600s to save/evaluate; startup/build/storage excluded",
        "requested_resource_hourly_usd": {"h200": 4.54, "cpu_2": 2 * .0473, "memory_16_gib": 16 * .008},
        "original_optimizer_state_available": False, "continuation_optimizer": "fresh AdamW; moments reset at update 3000",
        "architecture_changed": False, "data_changed": False, "batch_size": 1, "seed_after_restart": 8008,
        "dynamics_objective": False, "evaluation_scope": "all training observations; fixed shared noise; no rollout",
        "milestones": [], "updates_reached": 3000}
    try:
        fit = Path("/data/eval/native_scratch_episode0000_v1")
        prior = read_json(fit / "report.json")
        if prior["status"] != "completed" or prior["requested_updates"] != 3000:
            raise ValueError("Expected completed 3000-update native fit")
        root = Path("/data/libero/milk_sam_development_v1")
        splits = read_json(root / "splits.json")
        manifest = validate_splits(root, splits)
        if prior["dataset_fingerprint"] != splits["fingerprint"]: raise ValueError("Dataset mismatch")
        entry = next(e for e in manifest["episodes"] if e["id"] == "episode_0000")
        ep = load_episode(root, entry, manifest, include_images=False)
        if len(ep["timestamps"]) != 150: raise ValueError("Expected 150 frames")
        with np.load(fit / "stats.npz", allow_pickle=False) as z: stats = {k: z[k].copy() for k in z.files}
        smol = Path("/data/checkpoints/smolvlm2-reference") / MODEL_REVISION
        if file_hash(smol / "model.safetensors") != prior["pretrained_smolvlm_sha256"]: raise ValueError("Backbone mismatch")
        key = "observation.images.agentview"
        cfg = SmolVLAConfig(device="cuda", vlm_model_name=str(smol), load_vlm_weights=True,
            input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(15,)),
                key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 512, 512))},
            output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))})
        torch.manual_seed(7)
        policy = SmolVLAPolicy(cfg).cuda().eval()
        vlm = policy.model.vlm_with_expert.vlm
        ck = torch.load(fit / "weights.pt", map_location="cpu", weights_only=True)
        params = dict(policy.named_parameters())
        if ck["completed_updates"] != 3000 or set(ck["trainable_policy"]) != {n for n, p in params.items() if p.requires_grad}:
            raise ValueError("Native checkpoint parameter mismatch")
        with torch.no_grad():
            for k, v in ck["trainable_policy"].items(): params[k].copy_(v)
        pre, _ = make_smolvla_pre_post_processors(cfg, {
            OBS_STATE: {"mean": torch.from_numpy(stats["robot_mean"]), "std": torch.from_numpy(stats["robot_std"])},
            ACTION: {"mean": torch.from_numpy(stats["actions_mean"]), "std": torch.from_numpy(stats["actions_std"])}})
        batches = []
        for t in range(150):
            frame = Path("/data/libero/milk_sam_native_v1") / entry["id"] / f"{t:06}.jpg"
            if file_hash(frame) != prior["frame_sha256"][t]: raise ValueError(f"RGB mismatch at {t}")
            with Image.open(frame) as im: rgb = np.asarray(im.convert("RGB")).copy()
            count = min(50, 150-t)
            actions = np.zeros((1, 50, 7), np.float32); actions[0, :count] = ep["actions"][t:t+count]
            b = pre({OBS_STATE: torch.from_numpy(ep["robot"][t].copy()), key: torch.from_numpy(rgb).permute(2, 0, 1),
                ACTION: torch.from_numpy(actions), "action_is_pad": torch.from_numpy((np.arange(50) >= count)[None]), "task": entry["instruction"]})
            w = observation_window(ep, t, stats, history=4, horizon=50)
            torch.testing.assert_close(b[ACTION][0, :count].cpu(), torch.from_numpy(w["actions"][:count]), rtol=1e-5, atol=1e-5)
            batches.append(b)
        noise = np.random.default_rng(921).standard_normal((150, 50, cfg.max_action_dim), dtype=np.float32)
        np.save(out / "noise.npy", noise, allow_pickle=False)
        np.savez(out / "stats.npz", **stats)
        report.update(dataset_fingerprint=splits["fingerprint"], source_checkpoint_sha256=file_hash(fit / "weights.pt"),
            noise_sha256=file_hash(out / "noise.npy"), noise_shape=list(noise.shape), noise_generator="NumPy default_rng(921), float32")
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        volume.commit()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.manual_seed(8008)
        optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=1e-4, betas=(.9, .95))

        def save_checkpoint(step):
            torch.save({"completed_updates": step, "trainable_policy": {n: p.detach().cpu() for n, p in policy.named_parameters() if p.requires_grad},
                "optimizer": optimizer.state_dict(), "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                "frame_schedule_next_index": step % 150, "noise_sha256": report["noise_sha256"],
                "source_checkpoint_sha256": report["source_checkpoint_sha256"], "seed_after_restart": 8008}, out / f"checkpoint_{step}.pt")

        def evaluate(step):
            policy.eval(); rows = []
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for t, b in enumerate(batches):
                    n = torch.from_numpy(noise[t:t+1]).cuda()
                    pred_norm = policy.predict_action_chunk(b, noise=n)[0, 0].float().cpu().numpy()
                    raw = pred_norm * stats["actions_std"] + stats["actions_mean"]
                    if not np.isfinite(raw).all(): raise ValueError("Nonfinite evaluation action")
                    rows.append({"frame": t, "prediction": raw.astype(float).tolist(), "target": ep["actions"][t].astype(float).tolist(),
                        "prediction_normalized": pred_norm.astype(float).tolist()})
            pred = np.asarray([x["prediction"] for x in rows]); target = np.asarray([x["target"] for x in rows])
            classes = {}
            for sign, label in ((-1, "negative"), (1, "positive")):
                chosen = target[:, 6] * sign > 0
                classes[label] = {"count": int(chosen.sum()), "sign_accuracy": float(((pred[chosen, 6] > 0) == (target[chosen, 6] > 0)).mean()),
                    "mae": float(np.abs(pred[chosen, 6] - target[chosen, 6]).mean())}
            result = {"step": step, "frames": len(rows), "arm_mae_by_channel": np.abs(pred[:, :6] - target[:, :6]).mean(0).tolist(),
                "arm_rmse_by_channel": np.square(pred[:, :6] - target[:, :6]).mean(0).__pow__(.5).tolist(),
                "gripper_classes": classes, "gripper_mae": float(np.abs(pred[:, 6] - target[:, 6]).mean()), "rows": rows}
            (out / f"evaluation_{step}.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            return {k: v for k, v in result.items() if k != "rows"}

        report["milestones"].append(evaluate(3000))
        policy.train(); vlm.eval()
        losses = []
        step = 3000
        while step < 10000:
            if time.monotonic() - started >= 600: break
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16): loss, _ = policy.forward(batches[step % 150])
            if not torch.isfinite(loss): raise ValueError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad], 10, error_if_nonfinite=True)
            optimizer.step(); step += 1
            losses.append(float(loss.detach()))
            if step % 500 == 0:
                print(json.dumps({"step": step, "recent_loss_mean": float(np.mean(losses[-100:])), "seconds": time.monotonic()-started}), flush=True)
            if step in (5000, 10000):
                save_checkpoint(step)
                report["updates_reached"] = step
                report["milestones"].append(evaluate(step))
                report["elapsed_seconds"] = time.monotonic()-started
                (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
                volume.commit()
                policy.train(); vlm.eval()
        report["updates_reached"] = step
        if step not in (5000, 10000):
            save_checkpoint(step)
            report["milestones"].append(evaluate(step))
        report["status"] = "completed" if step == 10000 else "budget_stopped"
        report["training_loss_last_100"] = float(np.mean(losses[-100:])) if losses else None
        return json.loads(json.dumps(report, allow_nan=False))
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        seconds = time.monotonic() - started
        report["elapsed_seconds"] = seconds
        report["requested_resource_cost_estimate_usd"] = seconds * (4.54 + 2 * .0473 + 16 * .008) / 3600
        (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        volume.commit()


@app.function(image=image, gpu="H100", cpu=2, memory=16384, volumes={"/data": volume},
              timeout=125, max_containers=1, single_use_containers=True)
def finish_exposure(name: str = "native_exposure_finish_v1"):
    """Resume the saved optimizer at 9018; hard bound the remaining spend."""
    import json
    import time
    from pathlib import Path
    import numpy as np
    import torch
    from PIL import Image
    from odvla.data import file_hash, load_episode, read_json, validate_splits
    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
    from lerobot.utils.constants import ACTION, OBS_STATE
    if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
        raise ValueError("Invalid output name")
    out = Path("/data/eval") / name
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"status": "started", "gpu": "H100", "optimizer": "restored from 9018 checkpoint",
        "hard_timeout_seconds": 125, "original_optimizer_restart_at": 3000,
        "budget_note": "125s H100 + 2 CPU + 16 GiB costs about $0.145 at listed rates; startup and storage excluded"}
    try:
        source = Path("/data/eval/native_exposure_curve_v1")
        old = read_json(source / "report.json")
        if old["updates_reached"] != 9018 or old["status"] != "budget_stopped": raise ValueError("Unexpected source continuation")
        with np.load(source / "stats.npz", allow_pickle=False) as z: stats = {k: z[k].copy() for k in z.files}
        if file_hash(source / "noise.npy") != old["noise_sha256"]: raise ValueError("Noise bank mismatch")
        noise = np.load(source / "noise.npy", allow_pickle=False)
        root = Path("/data/libero/milk_sam_development_v1")
        splits = read_json(root / "splits.json"); manifest = validate_splits(root, splits)
        if splits["fingerprint"] != old["dataset_fingerprint"]: raise ValueError("Dataset mismatch")
        entry = next(e for e in manifest["episodes"] if e["id"] == "episode_0000")
        ep = load_episode(root, entry, manifest, include_images=False)
        key = "observation.images.agentview"
        cfg = SmolVLAConfig(device="cuda", vlm_model_name=str(Path("/data/checkpoints/smolvlm2-reference") / MODEL_REVISION),
            load_vlm_weights=True, input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(15,)),
                key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 512, 512))},
            output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))})
        torch.manual_seed(7)
        policy = SmolVLAPolicy(cfg).cuda()
        ck = torch.load(source / "checkpoint_9018.pt", map_location="cpu", weights_only=True)
        params = dict(policy.named_parameters())
        if ck["completed_updates"] != 9018 or ck["noise_sha256"] != old["noise_sha256"] or set(ck["trainable_policy"]) != {n for n, p in params.items() if p.requires_grad}:
            raise ValueError("Continuation checkpoint mismatch")
        with torch.no_grad():
            for k, v in ck["trainable_policy"].items(): params[k].copy_(v)
        optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=1e-4, betas=(.9, .95))
        optimizer.load_state_dict(ck["optimizer"])
        pre, _ = make_smolvla_pre_post_processors(cfg, {
            OBS_STATE: {"mean": torch.from_numpy(stats["robot_mean"]), "std": torch.from_numpy(stats["robot_std"])},
            ACTION: {"mean": torch.from_numpy(stats["actions_mean"]), "std": torch.from_numpy(stats["actions_std"])}})
        batches = []
        for t in range(150):
            with Image.open(Path("/data/libero/milk_sam_native_v1") / entry["id"] / f"{t:06}.jpg") as im: rgb = np.asarray(im.convert("RGB")).copy()
            count = min(50, 150-t); actions = np.zeros((1, 50, 7), np.float32); actions[0, :count] = ep["actions"][t:t+count]
            batches.append(pre({OBS_STATE: torch.from_numpy(ep["robot"][t].copy()), key: torch.from_numpy(rgb).permute(2, 0, 1),
                ACTION: torch.from_numpy(actions), "action_is_pad": torch.from_numpy((np.arange(50)>=count)[None]), "task": entry["instruction"]}))
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.set_rng_state(ck["torch_rng"]); torch.cuda.set_rng_state_all(ck["cuda_rng"])
        policy.train(); policy.model.vlm_with_expert.vlm.eval()
        step = 9018
        while step < 10000 and time.monotonic() - started < 92:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16): loss, _ = policy.forward(batches[step % 150])
            if not torch.isfinite(loss): raise ValueError("Nonfinite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad], 10, error_if_nonfinite=True)
            optimizer.step(); step += 1
        report["updates_reached"] = step
        torch.save({"completed_updates": step, "trainable_policy": {n: p.detach().cpu() for n,p in policy.named_parameters() if p.requires_grad},
            "optimizer": optimizer.state_dict(), "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
            "frame_schedule_next_index": step % 150, "noise_sha256": old["noise_sha256"],
            "source_checkpoint_sha256": file_hash(source / "checkpoint_9018.pt")}, out / f"checkpoint_{step}.pt")
        report["status"] = "checkpoint_saved"
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        volume.commit()
        policy.eval(); rows = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for t, b in enumerate(batches):
                pred = policy.predict_action_chunk(b, noise=torch.from_numpy(noise[t:t+1]).cuda())[0,0].float().cpu().numpy()
                raw = pred * stats["actions_std"] + stats["actions_mean"]
                rows.append({"frame": t, "prediction": raw.astype(float).tolist(), "target": ep["actions"][t].astype(float).tolist()})
        pred = np.asarray([r["prediction"] for r in rows]); target = np.asarray([r["target"] for r in rows])
        classes = {}
        for sign, label in ((-1, "negative"), (1, "positive")):
            chosen = target[:,6] * sign > 0
            classes[label] = {"count": int(chosen.sum()), "sign_accuracy": float(((pred[chosen,6]>0)==(target[chosen,6]>0)).mean()),
                "mae": float(np.abs(pred[chosen,6]-target[chosen,6]).mean())}
        evaluation = {"step": step, "frames": 150, "arm_mae_by_channel": np.abs(pred[:,:6]-target[:,:6]).mean(0).tolist(),
            "arm_rmse_by_channel": np.sqrt(np.square(pred[:,:6]-target[:,:6]).mean(0)).tolist(), "gripper_classes": classes,
            "gripper_mae": float(np.abs(pred[:,6]-target[:,6]).mean()), "rows": rows}
        (out / f"evaluation_{step}.json").write_text(json.dumps(evaluation, indent=2, allow_nan=False)+"\n")
        report.update(status="completed" if step == 10000 else "budget_stopped", evaluation={k:v for k,v in evaluation.items() if k != "rows"}, noise_sha256=old["noise_sha256"])
        return report
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report["requested_resource_cost_estimate_usd"] = report["elapsed_seconds"] * (3.95 + 2*.0473 + 16*.008) / 3600
        (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        volume.commit()


@app.local_entrypoint()
def main(steps: int = 3000, name: str = "native_scratch_episode0000_v1", stage: str = "train"):
    if stage == "train": print(train.remote(steps, name))
    elif stage == "diagnose": print(diagnose.remote(name))
    elif stage == "continue": print(continue_exposure.remote(name))
    elif stage == "finish": print(finish_exposure.remote(name))
    else: raise ValueError("Unknown stage")
