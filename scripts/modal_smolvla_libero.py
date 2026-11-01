"""Fine-tune the official robot-trained SmolVLA policy on LIBERO OSC_POSE."""
import modal
from scripts.modal_smolvla_reference import image as reference_image

image = (reference_image.add_local_file("scripts/__init__.py", "/root/scripts/__init__.py")
         .add_local_file("scripts/modal_smolvla_reference.py", "/root/scripts/modal_smolvla_reference.py"))
volume = modal.Volume.from_name("object-dynamics-vla-data", create_if_missing=False)
app = modal.App("odvla-smolvla-libero-finetune")
POLICY_REVISION = "5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6"


@app.function(image=image, gpu="T4", volumes={"/data": volume}, cpu=4, memory=16384,
              timeout=1800, max_containers=1, single_use_containers=True)
def run(steps: int = 10, suffix: str = "smoke10"):
    import hashlib
    import json
    import time
    from pathlib import Path
    import numpy as np
    import torch
    from PIL import Image
    from odvla.data import file_hash, fit_stats, load_episode, read_json, validate_splits
    from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
    from lerobot.utils.constants import ACTION, OBS_STATE
    if not 1 <= steps <= 1000 or not suffix.isalnum(): raise ValueError("Invalid run budget or suffix")
    root = Path("/data/libero/milk_sam_development_v1"); cache = Path("/data/libero/milk_sam_native_v1")
    source = Path("/data/checkpoints/smolvla-base") / POLICY_REVISION
    out = Path(f"/data/runs/smolvla_libero_{suffix}")
    if out.exists(): raise FileExistsError(out)
    if file_hash(source / "model.safetensors") != "7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb":
        raise ValueError("Unexpected checkpoint hash")
    splits = read_json(root / "splits.json"); manifest = validate_splits(root, splits); stats = fit_stats(root, splits)
    entries = {e["id"]: e for e in manifest["episodes"] if e["id"] in splits["train"] + splits["val"]}
    episodes = {eid: load_episode(root, entry, manifest, include_images=False) for eid, entry in entries.items()}
    indices = {split: [(eid, t) for eid in splits[split] for t in range(len(episodes[eid]["timestamps"]))]
               for split in ("train", "val")}
    image_key = "observation.images.agentview"
    cfg = PreTrainedConfig.from_pretrained(str(source))
    cfg.device = "cuda"; cfg.vlm_model_name = "/data/checkpoints/smolvlm2-reference/7b375e1b73b11138ff12fe22c8f2822d8fe03467"
    cfg.load_vlm_weights = False  # Strictly load all VLM and expert tensors from robot checkpoint below.
    cfg.input_features = {OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(15,)),
                          image_key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 512, 512))}
    cfg.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))}
    cfg.train_expert_only = True; cfg.freeze_vision_encoder = True
    torch.manual_seed(7)
    policy = SmolVLAPolicy.from_pretrained(str(source), config=cfg, strict=True).cuda()
    vlm = policy.model.vlm_with_expert.vlm
    if any(p.requires_grad for p in vlm.parameters()): raise ValueError("Expected frozen backbone")
    lerobot_stats = {OBS_STATE: {"mean": torch.from_numpy(stats["robot_mean"]), "std": torch.from_numpy(stats["robot_std"])},
                     ACTION: {"mean": torch.from_numpy(stats["actions_mean"]), "std": torch.from_numpy(stats["actions_std"])}}
    pre, post = make_smolvla_pre_post_processors(cfg, lerobot_stats)
    def batch(refs):
        images = []; states = []; actions = []; pads = []; tasks = []
        for eid, t in refs:
            ep = episodes[eid]; n = min(50, len(ep["timestamps"]) - t)
            a = np.zeros((50, 7), np.float32); a[:n] = ep["actions"][t:t+n]
            image = np.asarray(Image.open(cache / eid / f"{t:06}.jpg").convert("RGB")).copy()
            images.append(torch.from_numpy(image).permute(2, 0, 1)); states.append(ep["robot"][t]); actions.append(a)
            pads.append(np.arange(50) >= n); tasks.append(entries[eid]["instruction"])
        return pre({image_key: torch.stack(images), OBS_STATE: torch.from_numpy(np.stack(states)),
                    ACTION: torch.from_numpy(np.stack(actions)), "action_is_pad": torch.from_numpy(np.stack(pads)), "task": tasks})
    # Fixed paired validation diagnostic: same references, Gaussian noise, flow time and sampler.
    val_refs = []
    for eid in splits["val"]:
        n = len(episodes[eid]["timestamps"])
        val_refs.extend(((eid, 0), (eid, n // 2)))
    val_batches = [batch(val_refs[i:i+4]) for i in range(0, len(val_refs), 4)]
    def evaluate():
        policy.eval(); losses = []; errors = []; signs = []
        with torch.no_grad():
            for i, b in enumerate(val_batches):
                generator = torch.Generator(device="cuda").manual_seed(921 + i)
                noise = torch.randn((len(b[OBS_STATE]), 50, 32), generator=generator, device="cuda")
                t = torch.full((len(noise),), .5, device="cuda")
                loss, _ = policy.forward(b, noise=noise, time=t); losses.append(float(loss))
                sampled = policy.predict_action_chunk(b, noise=noise)
                pred = post(sampled).numpy()[:, 0]; target = b[ACTION][:, 0].cpu().numpy() * stats["actions_std"] + stats["actions_mean"]
                errors.extend(np.abs(pred - target)); signs.extend((np.sign(pred[:, -1]) == np.sign(target[:, -1])).tolist())
        return {"flow_mse": float(np.mean(losses)), "native_first_action_mae_by_channel": np.mean(errors, axis=0).tolist(),
                "gripper_sign_agreement": float(np.mean(signs)), "windows": len(val_refs)}
    out.mkdir(parents=True); np.savez(out / "stats.npz", **stats)
    metadata = {"policy": "lerobot/smolvla_base", "checkpoint_revision": POLICY_REVISION,
                "checkpoint_sha256": file_hash(source / "model.safetensors"), "dataset_fingerprint": splits["fingerprint"],
                "source_semantics": "6-channel SO100-style joint actions", "target_semantics": "7-channel LIBERO Panda OSC_POSE normalized commands",
                "adaptation": "strict full checkpoint load; train expert, state/action projections, frozen VLM; target-only train statistics",
                "camera": "one agentview RGB; original checkpoint has three camera slots", "dataset_overlap": "not audited; development only",
                "steps": steps, "batch_size": 4, "seed": 7,
                "selection": "fixed validation flow MSE on first and middle frames per validation episode; reused for selection and scoring"}
    (out / "configuration.json").write_text(json.dumps(metadata, indent=2) + "\n")
    before = evaluate(); print(json.dumps({"step": 0, **before}), flush=True)
    optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=1e-4, betas=(.9, .95), weight_decay=1e-10)
    rng = np.random.default_rng(7); started = time.monotonic(); losses = []; evaluations = []; best = float("inf")
    for step in range(1, steps + 1):
        policy.train(); refs = [indices["train"][int(i)] for i in rng.integers(len(indices["train"]), size=4)]
        b = batch(refs); optimizer.zero_grad(set_to_none=True); loss, _ = policy.forward(b); loss.backward()
        if step == 1 and any(p.grad is not None for p in vlm.parameters()): raise ValueError("Frozen VLM received gradients")
        grad = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), 10)); optimizer.step()
        losses.append(float(loss.detach()))
        with (out / "train.jsonl").open("a") as log:
            log.write(json.dumps({"step": step, "loss": losses[-1], "gradient_norm": grad, "seconds": time.monotonic()-started}) + "\n")
        if step == 1 or step % 10 == 0: print(json.dumps({"step": step, "train_loss": losses[-1], "grad": grad, "seconds": time.monotonic()-started}), flush=True)
        if step % 250 == 0 or step == steps:
            metrics = evaluate(); row = {"step": step, **metrics}; evaluations.append(row)
            torch.save(policy.state_dict(), out / f"weights_step{step:05}.pt")
            if metrics["flow_mse"] < best:
                best = metrics["flow_mse"]; torch.save(policy.state_dict(), out / "weights_best.pt")
                (out / "best_validation.json").write_text(json.dumps(row, indent=2) + "\n")
            torch.save({"step": step, "optimizer": optimizer.state_dict(), "numpy_rng": rng.bit_generator.state,
                        "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}, out / "training_state.pt")
            (out / "evaluations.json").write_text(json.dumps(evaluations, indent=2) + "\n")
            volume.commit(); print(json.dumps(row), flush=True)
    after = evaluations[-1]; volume.commit()
    result = {"before": before, "after": after, "first_training_loss": losses[0], "last_training_loss": losses[-1],
              "train_seconds": time.monotonic() - started, "peak_gpu_gib": torch.cuda.max_memory_allocated() / 1024**3,
              "output": str(out), "frozen_vlm_gradient_count": sum(p.grad is not None for p in vlm.parameters())}
    (out / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n"); volume.commit(); return result


@app.local_entrypoint()
def main(steps: int = 10, suffix: str = "smoke10"): print(run.remote(steps, suffix))
