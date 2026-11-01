"""Full development first-action metrics for pretrained and LIBERO-adapted SmolVLA."""
import modal
from scripts.modal_smolvla_libero import image

image = image.add_local_file("scripts/modal_smolvla_libero.py", "/root/scripts/modal_smolvla_libero.py")
volume = modal.Volume.from_name("object-dynamics-vla-data", create_if_missing=False)
app = modal.App("odvla-smolvla-libero-offline")


@app.function(image=image, gpu="T4", volumes={"/data": volume}, cpu=4, memory=16384, timeout=1800, max_containers=1)
def run(run_name: str = "milk1000v1", include_unadapted: bool = False):
    import json
    import time
    from pathlib import Path
    import numpy as np
    import torch
    from PIL import Image
    from odvla.data import load_episode, read_json, validate_splits
    from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
    from lerobot.utils.constants import ACTION, OBS_STATE
    if not run_name.isalnum(): raise ValueError("Run name must be alphanumeric")
    out = Path(f"/data/eval/smolvla_libero_{run_name}_offline_{'withbase' if include_unadapted else 'v1'}.json")
    if out.exists(): raise FileExistsError(out)
    root = Path("/data/libero/milk_sam_development_v1"); frames = Path("/data/libero/milk_sam_native_v1")
    splits = read_json(root / "splits.json"); manifest = validate_splits(root, splits)
    entries = {e["id"]: e for e in manifest["episodes"] if e["id"] in splits["val"]}
    episodes = {eid: load_episode(root, entry, manifest, include_images=False) for eid, entry in entries.items()}
    refs = [(eid, t) for eid in splits["val"] for t in range(len(episodes[eid]["timestamps"]))]
    checkpoint = Path(f"/data/runs/smolvla_libero_{run_name}")
    metadata = read_json(checkpoint / "configuration.json")
    if metadata["dataset_fingerprint"] != splits["fingerprint"] or metadata["checkpoint_revision"] != "5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6":
        raise ValueError("Evaluation checkpoint provenance mismatch")
    with np.load(checkpoint / "stats.npz", allow_pickle=False) as z: stats = {k: z[k].copy() for k in z.files}
    source = Path("/data/checkpoints/smolvla-base/5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6")
    key = "observation.images.agentview"; rows = []; started = time.monotonic()
    for variant in (("pretrained_unadapted", "adapted_selected") if include_unadapted else ("adapted_selected",)):
        cfg = PreTrainedConfig.from_pretrained(str(source)); cfg.device = "cuda"
        cfg.vlm_model_name = "/data/checkpoints/smolvlm2-reference/7b375e1b73b11138ff12fe22c8f2822d8fe03467"
        cfg.load_vlm_weights = False; cfg.train_expert_only = True; cfg.freeze_vision_encoder = True
        cfg.input_features = {OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(15,)), key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 512, 512))}
        cfg.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))}
        if variant == "pretrained_unadapted": policy = SmolVLAPolicy.from_pretrained(str(source), config=cfg, strict=True).cuda().eval()
        else:
            policy = SmolVLAPolicy(cfg).cuda().eval()
            policy.load_state_dict(torch.load(checkpoint / "weights_best.pt", map_location="cuda", weights_only=True), strict=True)
        info = {OBS_STATE: {"mean": torch.from_numpy(stats["robot_mean"]), "std": torch.from_numpy(stats["robot_std"])},
                ACTION: {"mean": torch.from_numpy(stats["actions_mean"]), "std": torch.from_numpy(stats["actions_std"])}}
        pre, post = make_smolvla_pre_post_processors(cfg, info)
        rng = np.random.default_rng(921); error2 = np.zeros(7); absolute = np.zeros(7); confusion = np.zeros((2, 3), int)
        with torch.inference_mode():
            for start in range(0, len(refs), 4):
                chunk = refs[start:start+4]
                imgs = [torch.from_numpy(np.asarray(Image.open(frames / eid / f"{t:06}.jpg").convert("RGB")).copy()).permute(2, 0, 1) for eid, t in chunk]
                state = np.stack([episodes[eid]["robot"][t] for eid, t in chunk])
                target = np.stack([episodes[eid]["actions"][t] for eid, t in chunk])
                b = pre({OBS_STATE: torch.from_numpy(state), key: torch.stack(imgs), "task": [entries[eid]["instruction"] for eid, _ in chunk]})
                noise = torch.from_numpy(rng.normal(size=(len(chunk), 50, 32)).astype(np.float32)).cuda()
                pred = post(policy.predict_action_chunk(b, noise=noise)).numpy()[:, 0]
                delta = pred - target; error2 += ((delta / stats["actions_std"])**2).sum(0); absolute += np.abs(delta).sum(0)
                for truth, estimate in zip(np.sign(target[:, -1]), np.sign(pred[:, -1])):
                    if truth not in (-1, 1): raise ValueError("Unexpected zero gripper target")
                    confusion[int(truth > 0), {-1: 0, 1: 1, 0: 2}[int(estimate)]] += 1
                if (start // 4) % 100 == 0: print(json.dumps({"variant": variant, "windows": start, "seconds": time.monotonic() - started}), flush=True)
        n = len(refs); counts = confusion.sum(1)
        row = {"variant": variant, "windows": n, "first_action_normalized_mse_by_channel": (error2 / n).tolist(),
               "first_action_native_mae_by_channel": (absolute / n).tolist(), "gripper_confusion_minus_plus_zero": confusion.tolist(),
               "gripper_sign_agreement": float((confusion[0, 0] + confusion[1, 1]) / n),
               "always_minus_accuracy": float(counts[0] / n), "always_plus_accuracy": float(counts[1] / n),
               "gripper_class_recall": [float(confusion[i, i] / counts[i]) if counts[i] else None for i in range(2)]}
        rows.append(row); print(json.dumps(row), flush=True); del policy; torch.cuda.empty_cache()
    result = {"split": "reused validation, also used for checkpoint selection", "sampling": "seed 921, 10 steps", "run_name": run_name,
              "normalization": "LIBERO training-only statistics",
              "dataset_overlap": "checkpoint robot pretraining overlap not independently audited", "rows": rows}
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n"); volume.commit(); return result


@app.local_entrypoint()
def main(run_name: str = "milk1000v1", include_unadapted: bool = False): print(run.remote(run_name, include_unadapted))
