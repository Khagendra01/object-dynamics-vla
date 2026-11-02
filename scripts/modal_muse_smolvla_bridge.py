"""Bounded Muse/SAM to pretrained SmolVLA connector alignment experiment."""
import modal
from scripts.modal_smolvla_reference import image as base_image

image = (base_image.add_local_file("scripts/__init__.py", "/root/scripts/__init__.py")
         .add_local_file("scripts/modal_smolvla_reference.py", "/root/scripts/modal_smolvla_reference.py")
         .add_local_file("scripts/train_muse_vla.py", "/root/train_muse_vla.py"))
volume = modal.Volume.from_name("object-dynamics-vla-data", create_if_missing=False)
app = modal.App("odvla-muse-smolvla-bridge")


@app.function(image=image, gpu="T4", volumes={"/data": volume}, cpu=4, memory=24576,
              timeout=900, max_containers=1, single_use_containers=True)
def run(steps: int = 5, suffix: str = "smoke", phase: str = "align", init_bridge: str = ""):
    import json
    import sys
    import time
    from pathlib import Path
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from odvla.data import file_hash, read_json
    from odvla.vla.dataset import CachedEpisodeDataset
    from odvla.vla.smolvla_bridge import MuseSamPrefix, expert_velocity
    from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, make_att_2d_masks
    from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
    from lerobot.utils.constants import ACTION, OBS_STATE, OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
    sys.path.insert(0, "/root")
    from train_muse_vla import make_batch

    if not 1 <= steps <= 1000 or not suffix.isalnum() or phase not in ("align", "action"):
        raise ValueError("Invalid budget, suffix or phase")
    if phase == "action" and not init_bridge.isalnum(): raise ValueError("Action phase requires an alignment run")
    out = Path(f"/data/runs/muse_smolvla_bridge_{suffix}")
    if out.exists(): raise FileExistsError(out)
    root = Path("/data/libero/milk_sam_development_v1")
    cache = Path("/data/libero/muse_middle_tokens_development_v1")
    frames = Path("/data/libero/milk_sam_native_v1")
    adapted = Path("/data/runs/smolvla_libero_milk1000v1")
    source = Path("/data/checkpoints/smolvla-base/5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6")
    if file_hash(source / "model.safetensors") != "7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb":
        raise ValueError("Checkpoint hash mismatch")
    splits = read_json(root / "splits.json")
    if read_json(adapted / "configuration.json")["dataset_fingerprint"] != splits["fingerprint"]:
        raise ValueError("Adapted checkpoint dataset mismatch")
    with np.load(adapted / "stats.npz", allow_pickle=False) as z: stats = {k: z[k].copy() for k in z.files}
    trainset = CachedEpisodeDataset(root, cache, "train", stats)
    valset = CachedEpisodeDataset(root, cache, "val", stats)
    image_key = "observation.images.agentview"
    cfg = PreTrainedConfig.from_pretrained(str(source))
    cfg.device = "cuda"; cfg.vlm_model_name = "/data/checkpoints/smolvlm2-reference/7b375e1b73b11138ff12fe22c8f2822d8fe03467"
    cfg.load_vlm_weights = False
    cfg.input_features = {OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(15,)),
                          image_key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 512, 512))}
    cfg.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))}
    policy = SmolVLAPolicy.from_pretrained(str(source), config=cfg, strict=True).cuda()
    policy.load_state_dict(torch.load(adapted / "weights_best.pt", map_location="cuda", weights_only=True), strict=True)
    policy.eval().requires_grad_(False)
    model = policy.model
    torch.manual_seed(7); bridge = MuseSamPrefix().cuda()
    if phase == "action":
        init = Path(f"/data/runs/muse_smolvla_bridge_{init_bridge}")
        if read_json(init / "configuration.json")["dataset_fingerprint"] != splits["fingerprint"]:
            raise ValueError("Bridge dataset mismatch")
        bridge.load_state_dict(torch.load(init / "training_state.pt", map_location="cuda", weights_only=False)["bridge"], strict=True)
        for component in (model.vlm_with_expert.lm_expert, model.state_proj, model.action_in_proj,
                          model.action_out_proj, model.action_time_mlp_in, model.action_time_mlp_out):
            component.requires_grad_(True)
    optim = torch.optim.AdamW(list(bridge.parameters()) + [p for p in model.parameters() if p.requires_grad],
                              lr=1e-4 if phase == "align" else 1e-5, betas=(.9, .95))
    pre, _ = make_smolvla_pre_post_processors(cfg, {
        OBS_STATE: {"mean": torch.from_numpy(stats["robot_mean"]), "std": torch.from_numpy(stats["robot_std"])},
        ACTION: {"mean": torch.from_numpy(stats["actions_mean"]), "std": torch.from_numpy(stats["actions_std"])}})

    def minibatch(samples):
        b = make_batch(samples, "cuda")
        rgbs = [torch.from_numpy(np.asarray(Image.open(frames / s["episode"] / f'{s["t"]:06}.jpg').convert("RGB")).copy()).permute(2, 0, 1) for s in samples]
        raw_state = np.stack([trainset.episodes[s["episode"]]["robot"][s["t"]] if s["episode"] in trainset.episodes else valset.episodes[s["episode"]]["robot"][s["t"]] for s in samples])
        teacher = pre({image_key: torch.stack(rgbs), OBS_STATE: torch.from_numpy(raw_state), "task": [s["instruction"] for s in samples]})
        return b, teacher

    def objective(samples, seed, grad):
        b, teacher = minibatch(samples)
        generator = torch.Generator(device="cuda").manual_seed(seed)
        noise = torch.randn((len(samples), 50, 32), generator=generator, device="cuda")
        if phase == "align":
            t = torch.rand((len(samples),), generator=generator, device="cuda") * .998 + .001
        else:
            t = torch.tensor(np.random.default_rng(seed).beta(1.5, 1, len(samples)) * .999 + .001,
                             device="cuda", dtype=torch.float32)
        actions = F.pad(b["actions"], (0, 25))
        xt = t[:, None, None] * noise + (1 - t[:, None, None]) * actions
        with torch.no_grad():
            images, masks = policy.prepare_images(teacher)
            state = policy.prepare_state(teacher)
            if phase == "align":
                prefix, pmask, par = model.embed_prefix(images, masks, teacher[OBS_LANGUAGE_TOKENS], teacher[OBS_LANGUAGE_ATTENTION_MASK], state)
                _, kv = model.vlm_with_expert.forward(attention_mask=make_att_2d_masks(pmask, par),
                    position_ids=torch.cumsum(pmask, dim=1) - 1, inputs_embeds=[prefix, None], use_cache=True)
                target = expert_velocity(model, [x.keys for x in kv.layers], [x.values for x in kv.layers], pmask, xt, t)
            else:
                target = noise - actions
        with torch.set_grad_enabled(grad):
            state_embedding = model.state_proj(state)
            keys, values, mask = bridge(b["muse"], b["muse_valid"], state_embedding, b["sam_values"], b["sam_deltas"],
                b["sam_valid"], b["sam_age"], b["history_valid"], use_sam=True)
            prediction = expert_velocity(model, keys, values, mask, xt, t)
            valid = b["action_valid"][..., None].float()
            loss = ((prediction[:, :, :7] - target[:, :, :7]).square() * valid).sum() / (valid.sum() * 7).clamp_min(1)
            return loss

    out.mkdir(parents=True); np.savez(out / "stats.npz", **stats)
    config = {"kind": "connector-only teacher velocity alignment" if phase == "align" else "joint expert and connector action flow", "steps": steps, "seed": 7,
              "dataset_fingerprint": splits["fingerprint"], "muse_cache_sha256": file_hash(cache / "manifest.json"),
              "source_checkpoint_sha256": file_hash(source / "model.safetensors"),
              "adapted_checkpoint_sha256": file_hash(adapted / "weights_best.pt"),
              "teacher": "SmolVLA 1000-update LIBERO adapted; training scaffold only", "sam": "cached measurements",
              "phase": phase, "init_bridge": init_bridge or None,
              "time_sampling": "uniform (.001,.999), paired with teacher" if phase == "align" else "Beta(1.5,1)*.999+.001",
              "validation": "first and middle frame of each reused validation episode; checkpoint selection and reporting", "test_accessed": False}
    (out / "configuration.json").write_text(json.dumps(config, indent=2) + "\n")
    rng = np.random.default_rng(7); fixed = []
    for eid in valset.entries:
        n = len(valset.episodes[eid]["timestamps"])
        fixed.extend([valset.index.index((eid, 0)), valset.index.index((eid, n // 2))])
    def evaluate():
        bridge.eval()
        return float(np.mean([float(objective([valset[i]], 921 + j, False)) for j, i in enumerate(fixed)]))
    start = time.monotonic(); before = evaluate(); rows = []; evaluations = [{"step": 0, "objective_mse": before}]
    best = before
    for step in range(1, steps + 1):
        bridge.train(); sample = trainset[int(rng.integers(len(trainset)))]; optim.zero_grad(set_to_none=True)
        loss = objective([sample], 7000 + step, True); loss.backward()
        grad = float(torch.nn.utils.clip_grad_norm_([p for group in optim.param_groups for p in group["params"]], 10)); optim.step()
        rows.append({"step": step, "objective_mse": float(loss.detach()), "gradient_norm": grad})
        if step == 1 or step % 10 == 0 or step == steps: print(json.dumps(rows[-1]), flush=True)
        if step % 50 == 0:
            score = evaluate(); evaluations.append({"step": step, "objective_mse": score})
            checkpoint = {"bridge": bridge.state_dict(), "policy_trainable": {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}, "optimizer": optim.state_dict(), "step": step,
                        "numpy_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state_all()}
            torch.save(checkpoint, out / "training_state.pt")
            if score < best:
                best = score
                torch.save({"bridge": bridge.state_dict(), "policy_trainable": checkpoint["policy_trainable"], "step": step}, out / "weights_best.pt")
            (out / "evaluations.json").write_text(json.dumps(evaluations, indent=2) + "\n")
            (out / "train.json").write_text(json.dumps(rows, indent=2) + "\n")
            volume.commit()
    after = evaluate()
    torch.save({"bridge": bridge.state_dict(), "policy_trainable": {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}, "optimizer": optim.state_dict(), "step": steps,
                "numpy_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all()}, out / "training_state.pt")
    result = {"before_objective_mse": before, "after_objective_mse": after, "steps": steps, "phase": phase,
              "train": rows, "seconds": time.monotonic() - start, "peak_gpu_gib": torch.cuda.max_memory_allocated() / 1024**3,
              "output": str(out), "evaluations": evaluations, "note": "offline development only; no closed-loop or task-success claim"}
    (out / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n"); volume.commit()
    return result


@app.local_entrypoint()
def main(steps: int = 5, suffix: str = "smoke", phase: str = "align", init_bridge: str = ""):
    print(run.remote(steps, suffix, phase, init_bridge))
