"""Frozen-motor bridge distillation on demonstration and student denoising states."""
import modal
from scripts.modal_muse_smolvla_bridge import image as base_image

image = base_image.add_local_file("scripts/modal_muse_smolvla_bridge.py", "/root/scripts/modal_muse_smolvla_bridge.py")
volume = modal.Volume.from_name("object-dynamics-vla-data", create_if_missing=False)
app = modal.App("odvla-bridge-repair")


def setup():
    import sys
    from pathlib import Path
    import numpy as np
    import torch
    from odvla.data import file_hash, read_json
    from odvla.vla.dataset import CachedEpisodeDataset
    from odvla.vla.smolvla_bridge import MuseSamPrefix
    from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
    from lerobot.utils.constants import ACTION, OBS_STATE
    sys.path.insert(0, "/root")
    from train_muse_vla import make_batch
    dataset = Path("/data/libero/milk_sam_development_v1")
    cache = Path("/data/libero/muse_middle_tokens_development_v1")
    source = Path("/data/checkpoints/smolvla-base/5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6")
    adapted = Path("/data/runs/smolvla_libero_milk1000v1")
    initial = Path("/data/runs/muse_smolvla_bridge_alignment250v1")
    splits = read_json(dataset / "splits.json")
    if read_json(initial / "configuration.json")["dataset_fingerprint"] != splits["fingerprint"]:
        raise ValueError("Initial bridge dataset mismatch")
    if file_hash(source / "model.safetensors") != "7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb":
        raise ValueError("Source hash mismatch")
    with np.load(adapted / "stats.npz", allow_pickle=False) as z: stats = {k: z[k].copy() for k in z.files}
    train = CachedEpisodeDataset(dataset, cache, "train", stats)
    val = CachedEpisodeDataset(dataset, cache, "val", stats)
    cfg = PreTrainedConfig.from_pretrained(str(source)); cfg.device = "cuda"; cfg.load_vlm_weights = False
    cfg.vlm_model_name = "/data/checkpoints/smolvlm2-reference/7b375e1b73b11138ff12fe22c8f2822d8fe03467"
    key = "observation.images.agentview"
    cfg.input_features = {OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(15,)),
                          key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 512, 512))}
    cfg.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))}
    policy = SmolVLAPolicy.from_pretrained(str(source), config=cfg, strict=True).cuda()
    policy.load_state_dict(torch.load(adapted / "weights_best.pt", map_location="cuda", weights_only=True), strict=True)
    policy.eval().requires_grad_(False)
    bridge = MuseSamPrefix().cuda()
    bridge.load_state_dict(torch.load(initial / "training_state.pt", map_location="cuda", weights_only=False)["bridge"], strict=True)
    pre, _ = make_smolvla_pre_post_processors(cfg, {
        OBS_STATE: {"mean": torch.from_numpy(stats["robot_mean"]), "std": torch.from_numpy(stats["robot_std"])},
        ACTION: {"mean": torch.from_numpy(stats["actions_mean"]), "std": torch.from_numpy(stats["actions_std"])}})
    provenance = {"dataset_fingerprint": splits["fingerprint"], "cache_manifest_sha256": file_hash(cache / "manifest.json"),
                  "native_sha256": file_hash(adapted / "weights_best.pt"), "initial_bridge_sha256": file_hash(initial / "training_state.pt")}
    return policy, bridge, pre, stats, train, val, make_batch, key, provenance


def inputs(policy, bridge, pre, samples, dataset, make_batch, key, stats):
    from pathlib import Path
    import numpy as np
    import torch
    from PIL import Image
    from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
    from lerobot.utils.constants import OBS_STATE, OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
    batch = make_batch(samples, "cuda")
    images = [torch.from_numpy(np.asarray(Image.open(Path("/data/libero/milk_sam_native_v1") / s["episode"] / f'{s["t"]:06}.jpg').convert("RGB")).copy()).permute(2, 0, 1) for s in samples]
    raw = np.stack([dataset.episodes[s["episode"]]["robot"][s["t"]] for s in samples])
    teacher = pre({key: torch.stack(images), OBS_STATE: torch.from_numpy(raw), "task": [s["instruction"] for s in samples]})
    model = policy.model
    with torch.no_grad():
        rgb, masks = policy.prepare_images(teacher); state = policy.prepare_state(teacher)
        prefix, pmask, par = model.embed_prefix(rgb, masks, teacher[OBS_LANGUAGE_TOKENS], teacher[OBS_LANGUAGE_ATTENTION_MASK], state)
        _, cache = model.vlm_with_expert.forward(attention_mask=make_att_2d_masks(pmask, par),
            position_ids=torch.cumsum(pmask, dim=1) - 1, inputs_embeds=[prefix, None], use_cache=True)
        teacher_kv = ([layer.keys for layer in cache.layers], [layer.values for layer in cache.layers], pmask)
        state_embedding = model.state_proj(state)
    student_kv = bridge(batch["muse"], batch["muse_valid"], state_embedding, batch["sam_values"],
        batch["sam_deltas"], batch["sam_valid"], batch["sam_age"], batch["history_valid"], use_sam=True)
    return batch, teacher_kv, student_kv, state_embedding


@app.function(image=image, gpu="T4", cpu=4, memory=24576, volumes={"/data": volume},
              timeout=3600, max_containers=1, single_use_containers=True)
def train(steps: int = 1000, batch_size: int = 4, suffix: str = "repair1000v1", resume: bool = False):
    import json
    import time
    from pathlib import Path
    import numpy as np
    import torch
    from odvla.data import file_hash
    from odvla.vla.bridge_repair import distillation
    if not 1 <= steps <= 5000 or not 1 <= batch_size <= 8 or not suffix.isalnum(): raise ValueError("Invalid training budget")
    out = Path(f"/data/runs/muse_smolvla_bridge_{suffix}")
    if out.exists() != resume: raise ValueError("Fresh output required unless resuming")
    policy, bridge, pre, stats, trainset, valset, make_batch, key, provenance = setup()
    if any(p.requires_grad for p in policy.parameters()): raise ValueError("Motor policy must remain frozen")
    optimizer = torch.optim.AdamW(bridge.parameters(), lr=3e-5, betas=(.9, .95))
    rng = np.random.default_rng(7); order = rng.permutation(len(trainset)); offset = 0; start = 0; visited = set(); best = float("inf")
    fixed = []
    for eid in valset.entries:
        n = len(valset.episodes[eid]["timestamps"]); fixed.extend([valset.index.index((eid, 0)), valset.index.index((eid, n // 2))])
    config = {**provenance, "steps": steps, "batch_size": batch_size, "seed": 7, "train_windows": len(trainset),
              "objective": "equal weighted teacher velocity MSE on ordinary Beta interpolant and detached student denoising state, gripper channel weight 3",
              "student_depth": "uniform integer 1..9", "learning_rate": 3e-5, "expert": "frozen, including state/action/time projections",
              "selection": "16 reused validation windows, fixed interpolant and depth-5 student state, teacher velocity MSE",
              "dynamics_objective": False, "training_split": "train only"}
    if resume:
        if json.loads((out / "configuration.json").read_text()) != config: raise ValueError("Resume configuration mismatch")
        checkpoint = torch.load(out / "training_state.pt", map_location="cpu", weights_only=False)
        bridge.load_state_dict(checkpoint["bridge"], strict=True); optimizer.load_state_dict(checkpoint["optimizer"])
        rng.bit_generator.state = checkpoint["numpy_rng"]; torch.set_rng_state(checkpoint["torch_rng"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"]); order = checkpoint["order"]; offset = checkpoint["offset"]
        start = checkpoint["step"]; visited = set(checkpoint["visited"]); best = checkpoint["best"]
    else:
        out.mkdir(parents=True); (out / "configuration.json").write_text(json.dumps(config, indent=2) + "\n")
        np.savez(out / "stats.npz", **stats)
    def checkpoint(step):
        torch.save({"step": step, "bridge": bridge.state_dict(), "optimizer": optimizer.state_dict(),
                    "numpy_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all(), "order": order, "offset": offset,
                    "visited": sorted(visited), "best": best}, out / "training_state.pt")
        volume.commit()
    def evaluate():
        bridge.eval(); rows = []
        with torch.no_grad():
            for start_idx in range(0, len(fixed), batch_size):
                samples = [valset[i] for i in fixed[start_idx:start_idx+batch_size]]
                b, teacher, student, _ = inputs(policy, bridge, pre, samples, valset, make_batch, key, stats)
                gen = torch.Generator(device="cuda").manual_seed(921 + start_idx)
                noise = torch.randn((len(samples), 50, 32), generator=gen, device="cuda")
                t = torch.full((len(samples),), .5, device="cuda")
                interp, on_policy = distillation(policy, b, teacher, student, noise, t, 5)
                rows.append((float(interp), float(on_policy)))
        return np.mean(rows, axis=0).tolist()
    started = time.monotonic()
    if start == 0:
        value = evaluate(); (out / "initial_validation.json").write_text(json.dumps({"interpolant": value[0], "student_state": value[1]}, indent=2) + "\n")
        print(json.dumps({"step": 0, "validation": value}), flush=True)
        checkpoint(0)
    for step in range(start + 1, steps + 1):
        indices = []
        while len(indices) < batch_size:
            if offset == len(order): order = rng.permutation(len(trainset)); offset = 0
            indices.append(int(order[offset])); offset += 1
        visited.update(indices); samples = [trainset[i] for i in indices]
        bridge.train(); optimizer.zero_grad(set_to_none=True)
        b, teacher, student, _ = inputs(policy, bridge, pre, samples, trainset, make_batch, key, stats)
        noise = torch.from_numpy(rng.standard_normal((batch_size, 50, 32), dtype=np.float32)).cuda()
        t = torch.from_numpy((rng.beta(1.5, 1, batch_size) * .999 + .001).astype(np.float32)).cuda()
        depth = int(rng.integers(1, 10))
        li, ls = distillation(policy, b, teacher, student, noise, t, depth); loss = .5 * (li + ls)
        loss.backward(); grad = float(torch.nn.utils.clip_grad_norm_(bridge.parameters(), 10)); optimizer.step()
        if step == 1 and (not grad or any(p.grad is not None for p in policy.parameters())):
            raise ValueError("Frozen-expert gradient contract failed")
        if step % 25 == 0 or step == 1:
            print(json.dumps({"step": step, "interpolant": float(li.detach()), "student_state": float(ls.detach()),
                              "gradient_norm": grad, "unique_train_windows": len(visited), "seconds": time.monotonic()-started}), flush=True)
        if step % 250 == 0 or step == steps:
            value = evaluate(); score = sum(value) / 2
            if score < best:
                best = score; torch.save({"bridge": bridge.state_dict(), "step": step, "score": score}, out / "weights_best.pt")
            row = {"step": step, "interpolant": value[0], "student_state": value[1], "score": score,
                   "unique_train_windows": len(visited), "presentations": step * batch_size}
            with (out / "validation.jsonl").open("a") as f: f.write(json.dumps(row) + "\n")
            checkpoint(step); print(json.dumps(row), flush=True)
    return {"output": str(out), "step": steps, "unique_train_windows": len(visited), "best": best,
            "seconds": time.monotonic()-started, "peak_gpu_gib": torch.cuda.max_memory_allocated()/1024**3,
            "frozen_policy_gradients": sum(p.grad is not None for p in policy.parameters())}


@app.function(image=image, gpu="T4", cpu=4, memory=24576, volumes={"/data": volume},
              timeout=3600, max_containers=1, single_use_containers=True)
def evaluate(suffix: str = "repair1250v1", batch_size: int = 4):
    import json
    import time
    from pathlib import Path
    import numpy as np
    import torch
    import torch.nn.functional as F
    from odvla.data import file_hash, read_json
    from odvla.vla.smolvla_bridge import MuseSamPrefix, expert_velocity
    if not suffix.isalnum() or not 1 <= batch_size <= 8: raise ValueError("Invalid evaluation settings")
    out = Path(f"/data/eval/muse_smolvla_bridge_{suffix}_fullval_v2.json")
    if out.exists(): raise FileExistsError(out)
    run = Path(f"/data/runs/muse_smolvla_bridge_{suffix}")
    policy, baseline, pre, stats, _, valset, make_batch, key, provenance = setup()
    config = read_json(run / "configuration.json")
    if any(config[k] != v for k, v in provenance.items()): raise ValueError("Evaluation provenance mismatch")
    repair = MuseSamPrefix().cuda().eval(); saved = torch.load(run / "weights_best.pt", map_location="cuda", weights_only=True)
    repair.load_state_dict(saved["bridge"], strict=True); baseline.eval(); model = policy.model
    fixed = {(eid, t) for eid in valset.entries for t in (0, len(valset.episodes[eid]["timestamps"]) // 2)}
    accum = {name: {"absolute": np.zeros(7), "squared": np.zeros(7), "velocity": np.zeros((10, 7)),
                    "grip_agree": 0, "teacher_negative": 0, "teacher_positive": 0,
                    "negative_correct": 0, "positive_correct": 0, "selected_abs": np.zeros(7),
                    "selected_grip_agree": 0, "selected_count": 0} for name in ("aligned_baseline", "repaired")}
    per_episode = {eid: {name: {"count": 0, "abs": np.zeros(7), "grip_agree": 0} for name in accum} for eid in valset.entries}
    started = time.monotonic(); count = 0
    with torch.no_grad():
        for start in range(0, len(valset), batch_size):
            samples = [valset[i] for i in range(start, min(start + batch_size, len(valset)))]
            b, teacher, old, state = inputs(policy, baseline, pre, samples, valset, make_batch, key, stats)
            new = repair(b["muse"], b["muse_valid"], state, b["sam_values"], b["sam_deltas"],
                         b["sam_valid"], b["sam_age"], b["history_valid"], use_sam=True)
            # Index-stable noise independent of batch size; shared between all configurations.
            noise = torch.from_numpy(np.stack([np.random.default_rng(921 + i).normal(size=(50, 32)).astype(np.float32)
                                               for i in range(start, start + len(samples))])).cuda()
            reference = noise.clone(); xs = {name: noise.clone() for name in accum}; kv = {"aligned_baseline": old, "repaired": new}
            batch_velocity = {name: np.zeros((10, 7)) for name in accum}
            for k in range(10):
                t = torch.full((len(samples),), 1 - k / 10, device="cuda")
                vref = expert_velocity(model, *teacher, reference, t)
                for name in accum:
                    v_on_reference = expert_velocity(model, *kv[name], reference, t)
                    batch_velocity[name][k] = ((v_on_reference[:, :, :7] - vref[:, :, :7]) ** 2).mean(dim=(0, 1)).cpu().numpy() * len(samples)
                    own = expert_velocity(model, *kv[name], xs[name], t)
                    xs[name] = xs[name] - own / 10
                reference = reference - vref / 10
            teacher_action = reference[:, 0, :7].cpu().numpy() * stats["actions_std"] + stats["actions_mean"]
            selected = np.array([(s["episode"], s["t"]) in fixed for s in samples])
            for name in accum:
                predicted = xs[name][:, 0, :7].cpu().numpy() * stats["actions_std"] + stats["actions_mean"]
                error = predicted - teacher_action; same = np.sign(predicted[:, 6]) == np.sign(teacher_action[:, 6])
                a = accum[name]; a["absolute"] += np.abs(error).sum(0); a["squared"] += (error ** 2).sum(0)
                a["velocity"] += batch_velocity[name]; a["grip_agree"] += int(same.sum())
                negative = teacher_action[:, 6] < 0; positive = teacher_action[:, 6] > 0
                a["teacher_negative"] += int(negative.sum()); a["teacher_positive"] += int(positive.sum())
                a["negative_correct"] += int((same & negative).sum()); a["positive_correct"] += int((same & positive).sum())
                a["selected_abs"] += np.abs(error[selected]).sum(0); a["selected_grip_agree"] += int(same[selected].sum()); a["selected_count"] += int(selected.sum())
                for j, sample in enumerate(samples):
                    e = per_episode[sample["episode"]][name]; e["count"] += 1; e["abs"] += np.abs(error[j]); e["grip_agree"] += int(same[j])
            count += len(samples)
            if count % 100 == 0 or count == len(valset): print(json.dumps({"evaluated": count, "seconds": time.monotonic()-started}), flush=True)
    result = {"windows": count, "episodes": len(valset.entries), "split": "reused validation; 16 used for selection",
              "noise": "one deterministic NumPy array per index, shared between native and both bridges",
              "checkpoint_step": saved["step"], "checkpoint_sha256": file_hash(run / "weights_best.pt"),
              "metrics": {}, "per_episode": {}, "seconds": time.monotonic()-started}
    for name, a in accum.items():
        result["metrics"][name] = {"first_action_mae_vs_teacher": (a["absolute"] / count).tolist(),
            "first_action_rmse_vs_teacher": np.sqrt(a["squared"] / count).tolist(),
            "teacher_gripper_sign_agreement": a["grip_agree"] / count,
            "teacher_negative_count": a["teacher_negative"], "teacher_positive_count": a["teacher_positive"],
            "teacher_negative_recall": a["negative_correct"] / max(1, a["teacher_negative"]),
            "teacher_positive_recall": a["positive_correct"] / max(1, a["teacher_positive"]),
            "velocity_mse_by_denoising_step_and_channel_on_teacher_path": (a["velocity"] / count).tolist(),
            "selection_windows": a["selected_count"], "selection_mae": (a["selected_abs"] / a["selected_count"]).tolist(),
            "selection_teacher_gripper_agreement": a["selected_grip_agree"] / a["selected_count"]}
    for eid, group in per_episode.items():
        result["per_episode"][eid] = {name: {"windows": a["count"], "mae": (a["abs"] / a["count"]).tolist(),
                                                 "teacher_gripper_agreement": a["grip_agree"] / a["count"]} for name, a in group.items()}
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n"); volume.commit(); return result


@app.local_entrypoint()
def main(steps: int = 1000, batch_size: int = 4, suffix: str = "repair1000v1", resume: bool = False, stage: str = "train"):
    if stage == "train": print(train.remote(steps, batch_size, suffix, resume))
    elif stage == "evaluate": print(evaluate.remote(suffix, batch_size))
    else: raise ValueError("Invalid stage")
