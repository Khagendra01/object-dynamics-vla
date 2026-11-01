"""Official pinned SmolVLA adapter smoke on one real LIBERO training window."""
import modal

REVISION = "552777b7c384e8e71a02303da39583f9a2653760"
MODEL = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
MODEL_REVISION = "7b375e1b73b11138ff12fe22c8f2822d8fe03467"
app = modal.App("odvla-official-smolvla-reference")
volume = modal.Volume.from_name("object-dynamics-vla-data", create_if_missing=False)
image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("git", "libgl1", "ffmpeg")
         .run_commands("git clone https://github.com/huggingface/lerobot.git /opt/lerobot",
                       f"git -C /opt/lerobot checkout {REVISION}",
                       "pip install 'torch==2.10.0' 'torchvision==0.25.0'",
                       "pip install '/opt/lerobot[smolvla]'")
         .add_local_python_source("odvla"))


@app.function(image=image, volumes={"/data": volume}, cpu=2, memory=8192, timeout=1200, max_containers=1)
def download():
    from pathlib import Path
    from huggingface_hub import snapshot_download
    revision = MODEL_REVISION
    path = Path("/data/checkpoints/smolvlm2-reference") / revision
    snapshot_download(MODEL, revision=revision, local_dir=str(path))
    volume.commit()
    return {"path": str(path), "revision": revision}


@app.function(image=image, gpu="T4", volumes={"/data": volume}, cpu=4, memory=16384,
              timeout=1200, max_containers=1, single_use_containers=True)
def run(model_path: str, model_revision: str, output_name: str):
    import hashlib
    import json
    import time
    from pathlib import Path
    import numpy as np
    import torch
    import transformers
    from PIL import Image
    from odvla.data import fit_stats, load_episode, read_json, validate_splits
    from odvla.vla.dataset import observation_window
    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
    from lerobot.utils.constants import ACTION, OBS_STATE, OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK

    if Path(output_name).name != output_name or not output_name.endswith(".json"):
        raise ValueError("Output name must be a simple .json filename")
    dest = Path("/data/eval") / output_name
    if dest.exists(): raise FileExistsError(dest)
    started = time.monotonic()
    root = Path("/data/libero/milk_sam_development_v1")
    splits = read_json(root / "splits.json"); manifest = validate_splits(root, splits)
    stats = fit_stats(root, splits)
    entry = next(e for e in manifest["episodes"] if e["id"] in splits["train"])
    episode = load_episode(root, entry, manifest, include_images=False)
    window = observation_window(episode, 0, stats, history=4, horizon=50)
    frame = Path("/data/libero/milk_sam_native_v1") / entry["id"] / "000000.jpg"
    rgb = np.asarray(Image.open(frame).convert("RGB")).copy()
    image_key = "observation.images.agentview"
    cfg = SmolVLAConfig(device="cuda", vlm_model_name=model_path, load_vlm_weights=True,
        input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(15,)),
                        image_key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 512, 512))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))})
    torch.manual_seed(7)
    policy = SmolVLAPolicy(cfg).cuda()
    stats_lerobot = {OBS_STATE: {"mean": torch.from_numpy(stats["robot_mean"]), "std": torch.from_numpy(stats["robot_std"])},
                     ACTION: {"mean": torch.from_numpy(stats["actions_mean"]), "std": torch.from_numpy(stats["actions_std"])}}
    pre, post = make_smolvla_pre_post_processors(cfg, stats_lerobot)
    actions = np.zeros((50, 7), dtype=np.float32)
    count = int(window["action_valid"].sum())
    actions[:count] = episode["actions"][:count]
    raw = {OBS_STATE: torch.from_numpy(episode["robot"][0].copy()),
           image_key: torch.from_numpy(rgb).permute(2, 0, 1), ACTION: torch.from_numpy(actions)[None],
           "action_is_pad": torch.from_numpy(~window["action_valid"])[None], "task": entry["instruction"]}
    batch = pre(raw)
    torch.testing.assert_close(batch[OBS_STATE][0].cpu(), torch.from_numpy(window["robot"][-1]), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(batch[ACTION][0, :count].cpu(), torch.from_numpy(window["actions"][:count]), rtol=1e-5, atol=1e-5)
    if tuple(batch[ACTION].shape) != (1, 50, 7): raise ValueError("Unexpected action shape")
    end = observation_window(episode, len(episode["timestamps"]) - 2, stats, history=4, horizon=50)
    end_raw = dict(raw)
    end_raw[ACTION] = torch.zeros((1, 50, 7))
    end_raw[ACTION][0, :2] = torch.from_numpy(episode["actions"][-2:].copy())
    end_raw["action_is_pad"] = torch.from_numpy(~end["action_valid"])[None]
    end_batch = pre(end_raw)
    if tuple(end_batch["action_is_pad"].shape) != (1, 50) or int((~end_batch["action_is_pad"]).sum()) != 2:
        raise ValueError("Episode-end action padding mismatch")
    torch.testing.assert_close(end_batch[ACTION][0, :2].cpu(), torch.from_numpy(end["actions"][:2]), rtol=1e-5, atol=1e-5)
    vlm = policy.model.vlm_with_expert.vlm
    if any(p.requires_grad for p in vlm.parameters()): raise ValueError("VLM not frozen")
    optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=1e-4)
    torch.manual_seed(7); policy.train(); optimizer.zero_grad(set_to_none=True)
    loss, diagnostics = policy.forward(batch)
    loss.backward()
    expert = policy.model.vlm_with_expert.lm_expert
    gradient_norm = sum(float(p.grad.detach().float().square().sum()) for p in expert.parameters() if p.grad is not None) ** .5
    if not gradient_norm or any(p.grad is not None for p in vlm.parameters()): raise ValueError("Gradient contract failed")
    parameter = next(p for p in expert.parameters() if p.grad is not None)
    before = parameter.detach().clone()
    optimizer.step()
    update_norm = float((parameter.detach() - before).float().norm())
    if update_norm == 0: raise ValueError("Expert optimizer did not update")
    policy.eval()
    torch.manual_seed(921)
    sampled_normalized = policy.predict_action_chunk(batch)
    sampled = post(sampled_normalized)
    if not torch.isfinite(sampled).all(): raise ValueError("Nonfinite sampled action")
    result = {"lerobot_commit": REVISION, "model_id": MODEL, "model_revision": model_revision,
              "reference": "official SmolVLA implementation, frozen pretrained SmolVLM2, randomly initialized expert; no robot-policy checkpoint",
              "dataset_fingerprint": splits["fingerprint"], "episode": entry["id"], "frame": 0,
              "instruction": entry["instruction"], "rgb_sha256": hashlib.sha256(frame.read_bytes()).hexdigest(),
              "rgb_shape": list(rgb.shape), "state_shape": list(batch[OBS_STATE].shape),
              "action_shape": list(batch[ACTION].shape), "valid_action_count": count,
              "episode_end_valid_count": int((~end_batch["action_is_pad"]).sum()),
              "token_ids": batch[OBS_LANGUAGE_TOKENS][0].cpu().tolist(),
              "token_attention_mask": batch[OBS_LANGUAGE_ATTENTION_MASK][0].cpu().tolist(),
              "raw_first_action": actions[0].tolist(), "normalized_first_action": batch[ACTION][0, 0].cpu().tolist(),
              "sampled_first_action": sampled[0, 0].tolist(), "loss": float(loss.detach()),
              "loss_diagnostics": diagnostics, "expert_gradient_norm": gradient_norm, "expert_parameter_update_norm": update_norm,
              "frozen_vlm_gradient_count": sum(p.grad is not None for p in vlm.parameters()),
              "peak_gpu_gib": torch.cuda.max_memory_allocated() / 1024**3,
              "elapsed_seconds": time.monotonic() - started, "torch": str(torch.__version__),
              "transformers": transformers.__version__, "test_accessed": False}
    dest.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    volume.commit()
    return result


@app.local_entrypoint()
def main(output_name: str = "official_smolvla_reference_v2.json"):
    model = download.remote()
    print("Pinned reference backbone:", model)
    print(run.remote(model["path"], model["revision"], output_name))
