"""Offline sampled-action and input-sensitivity evaluation of the combined policy."""
import modal
from scripts.modal_muse_smolvla_bridge import image, volume

app = modal.App("odvla-muse-smolvla-eval")
image = image.add_local_file("scripts/modal_muse_smolvla_bridge.py", "/root/scripts/modal_muse_smolvla_bridge.py")


@app.function(image=image, gpu="T4", volumes={"/data": volume}, cpu=4, memory=24576,
              timeout=600, max_containers=1, single_use_containers=True)
def run(suffix: str = "v1"):
    import json
    import sys
    from pathlib import Path
    import numpy as np
    import torch
    import torch.nn.functional as F
    from odvla.data import read_json
    from odvla.vla.dataset import CachedEpisodeDataset
    from odvla.vla.smolvla_bridge import MuseSamPrefix, expert_velocity
    from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.utils.constants import ACTION, OBS_STATE
    sys.path.insert(0, "/root")
    from train_muse_vla import make_batch
    root = Path("/data/libero/milk_sam_development_v1")
    cache = Path("/data/libero/muse_middle_tokens_development_v1")
    run = Path("/data/runs/muse_smolvla_bridge_action250v1")
    adapted = Path("/data/runs/smolvla_libero_milk1000v1")
    source = Path("/data/checkpoints/smolvla-base/5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6")
    if not suffix.isalnum(): raise ValueError("Suffix must be alphanumeric")
    out = Path(f"/data/eval/muse_smolvla_bridge_action250v1_offline16_{suffix}.json")
    if out.exists(): raise FileExistsError(out)
    if read_json(run / "configuration.json")["dataset_fingerprint"] != read_json(root / "splits.json")["fingerprint"]:
        raise ValueError("Dataset provenance mismatch")
    with np.load(run / "stats.npz", allow_pickle=False) as z: stats = {k: z[k].copy() for k in z.files}
    dataset = CachedEpisodeDataset(root, cache, "val", stats)
    cfg = PreTrainedConfig.from_pretrained(str(source)); cfg.device = "cuda"; cfg.load_vlm_weights = False
    cfg.vlm_model_name = "/data/checkpoints/smolvlm2-reference/7b375e1b73b11138ff12fe22c8f2822d8fe03467"
    cfg.input_features = {OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(15,)),
        "observation.images.agentview": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 512, 512))}
    cfg.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))}
    policy = SmolVLAPolicy.from_pretrained(str(source), config=cfg, strict=True).cuda()
    policy.load_state_dict(torch.load(adapted / "weights_best.pt", map_location="cuda", weights_only=True), strict=True)
    model = policy.model.eval(); checkpoint = torch.load(run / "weights_best.pt", map_location="cuda", weights_only=False)
    params = dict(model.named_parameters())
    prefixes = ("vlm_with_expert.lm_expert.", "state_proj.", "action_in_proj.", "action_out_proj.",
                "action_time_mlp_in.", "action_time_mlp_out.")
    expected = {name for name in params if name.startswith(prefixes)}
    if set(checkpoint["policy_trainable"]) != expected:
        raise ValueError("Invalid trained expert keys")
    with torch.no_grad():
        for name, value in checkpoint["policy_trainable"].items(): params[name].copy_(value)
    bridge = MuseSamPrefix().cuda().eval(); bridge.load_state_dict(checkpoint["bridge"], strict=True)
    refs = []
    for eid in dataset.entries:
        n = len(dataset.episodes[eid]["timestamps"])
        refs.extend((dataset.index.index((eid, 0)), dataset.index.index((eid, n // 2))))
    rows = []
    with torch.no_grad():
        for j, i in enumerate(refs):
            sample = dataset[i]; b = make_batch([sample], "cuda")
            state = model.state_proj(F.pad(b["robot"][:, -1], (0, 17)))
            def sample_action(use_sam, perturb_muse=False):
                muse = torch.zeros_like(b["muse"]) if perturb_muse else b["muse"]
                keys, values, mask = bridge(muse, b["muse_valid"], state, b["sam_values"], b["sam_deltas"],
                    b["sam_valid"], b["sam_age"], b["history_valid"], use_sam=use_sam)
                generator = torch.Generator(device="cuda").manual_seed(921 + j)
                x = torch.randn((1, 50, 32), generator=generator, device="cuda")
                for k in range(10):
                    x = x - expert_velocity(model, keys, values, mask, x, torch.full((1,), 1 - k / 10, device="cuda")) / 10
                return x[0, 0, :7].cpu().numpy() * stats["actions_std"] + stats["actions_mean"]
            actual = sample_action(True); without_sam = sample_action(False); without_muse = sample_action(True, True)
            truth = sample["actions"][0] * stats["actions_std"] + stats["actions_mean"]
            rows.append({"episode": sample["episode"], "frame": sample["t"], "prediction": actual.tolist(),
                         "target": truth.tolist(), "sam_mask_delta": np.abs(actual - without_sam).tolist(),
                         "muse_zero_delta": np.abs(actual - without_muse).tolist()})
    pred = np.array([r["prediction"] for r in rows]); target = np.array([r["target"] for r in rows])
    result = {"checkpoint_step": checkpoint["step"], "split": "reused validation; same 16 windows used for selection",
              "windows": len(rows), "mae_by_channel": np.abs(pred - target).mean(axis=0).tolist(),
              "gripper_sign_agreement": float(np.mean(np.sign(pred[:, 6]) == np.sign(target[:, 6]))),
              "mean_abs_action_change_without_sam": np.mean([r["sam_mask_delta"] for r in rows], axis=0).tolist(),
              "mean_abs_action_change_zero_muse": np.mean([r["muse_zero_delta"] for r in rows], axis=0).tolist(),
              "interpretation": "input sensitivity only, not accuracy or causal benefit; no rollout", "rows": rows}
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n"); volume.commit()
    return {k: v for k, v in result.items() if k != "rows"}


@app.local_entrypoint()
def main(suffix: str = "v1"):
    print(run.remote(suffix))
