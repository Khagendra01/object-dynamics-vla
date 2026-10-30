"""Bounded eight-episode, fixed-first-frame training diagnostic; not validation."""
import modal
from scripts.modal_real_ablations import image as base_image, volume

image = (base_image.add_local_file("scripts/__init__.py", "/root/scripts/__init__.py")
         .add_local_file("scripts/modal_real_ablations.py", "/root/scripts/modal_real_ablations.py"))
app = modal.App("odvla-vla-overfit-diagnostic")


@app.function(image=image, gpu="T4", cpu=4, memory=16384, volumes={"/data": volume}, timeout=1200)
def run(updates: int = 100):
    import json
    import time
    from pathlib import Path
    import numpy as np
    import torch
    from odvla.vla.action_expert import MuseActionExpert
    from odvla.vla.dataset import observation_window
    if not 1 <= updates <= 100:
        raise ValueError("Diagnostic limited to 100 updates")
    ds = Path("/data/libero/milk_sam_development_v1")
    cache = Path("/data/libero/muse_middle_tokens_development_v1")
    out = Path("/data/runs/muse_vla_overfit8_v1")
    if out.exists(): raise FileExistsError("Refusing to overwrite")
    split = json.loads((ds / "splits.json").read_text())
    cache_manifest = json.loads((cache / "manifest.json").read_text())
    if not cache_manifest["complete"] or cache_manifest["dataset_fingerprint"] != split["fingerprint"]:
        raise ValueError("Incomplete or mismatched cache")
    entries = split["train"][:8]
    with np.load("/data/runs/milk_sam_matched_policy_v1/action_only_seed7/stats.npz", allow_pickle=False) as z:
        stats = {key: z[key] for key in z.files}
    device = "cuda"
    data = []
    for episode in entries:
        with np.load(ds / "episodes" / f"{episode}.npz", allow_pickle=False) as z:
            ep = {k: z[k] for k in z.files}
        w = observation_window(ep, 0, stats)
        with np.load(cache / episode / "000000.npz", allow_pickle=False) as z:
            token = torch.from_numpy(z["features_bfloat16_bits"].view(np.int16).copy()).view(torch.bfloat16)[None].to(device)
            mask = torch.from_numpy(z["attention_mask"].copy())[None].to(device)
        data.append((token, mask, {k: torch.from_numpy(np.asarray(v))[None].to(device)
                                  for k, v in w.items() if k != "timestamp"}))
    torch.manual_seed(7); torch.set_num_threads(2)
    model = MuseActionExpert().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(.9, .95), weight_decay=1e-5)
    noise_rng = torch.Generator(device=device).manual_seed(7001)
    time_rng = torch.Generator(device=device).manual_seed(7002)
    out.mkdir(parents=True)
    logs, start = [], time.monotonic()
    def evaluate():
        model.eval(); scores = []
        with torch.no_grad():
            for i, (muse, mm, w) in enumerate(data):
                ctx, cm = model.context(muse, mm, w["robot"], w["history_valid"], w["history_timestamps"],
                                        w["sam_values"], w["sam_deltas"], w["sam_valid"], w["sam_age"])
                generator = torch.Generator(device=device).manual_seed(987 + i)
                a = w["actions"]; noise = torch.randn(a.shape, generator=generator, device=device)
                _, flow, _ = model.losses(ctx, cm, a, w["action_valid"], noise, torch.tensor([.5], device=device),
                                          w["change"], w["change_valid"])
                scores.append(float(flow))
        return float(np.mean(scores))
    before = evaluate()
    for step in range(1, updates + 1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        for muse, mm, w in data:
            ctx, cm = model.context(muse, mm, w["robot"], w["history_valid"], w["history_timestamps"],
                                    w["sam_values"], w["sam_deltas"], w["sam_valid"], w["sam_age"])
            actions = w["actions"]
            noise = torch.randn(actions.shape, generator=noise_rng, device=device)
            # Beta(1.5, 1) inverse CDF, independent RNG stream, as in pinned SmolVLA.
            time_value = torch.rand((1,), generator=time_rng, device=device).pow(2 / 3) * .999 + .001
            loss, flow, dynamics = model.losses(ctx, cm, actions, w["action_valid"], noise, time_value,
                                                w["change"], w["change_valid"], weight=.1)
            (loss / len(data)).backward()
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 10))
        optimizer.step()
        if step % 25 == 0:
            score = evaluate(); logs.append({"step": step, "fixed_training_flow_mse": score, "last_grad_norm": norm})
            torch.save(model.state_dict(), out / "weights.pt")
            (out / "progress.json").write_text(json.dumps(logs, indent=2) + "\n"); volume.commit()
    result = {"purpose": "8 training episode fixed-first-frame overfit diagnostic, not held-out validation",
        "episodes": entries, "updates": updates, "microbatch": 1, "effective_batch": 8,
        "initial_fixed_training_flow_mse": before, "checkpoints": logs,
        "seconds": time.monotonic() - start, "peak_gpu_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
        "dataset_fingerprint": split["fingerprint"], "action_flow_rng_seed": 7001, "flow_time_rng_seed": 7002,
        "validation_or_test_used": False, "reference_masks_used": False}
    (out / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n"); volume.commit()
    return result


@app.local_entrypoint()
def main(updates: int = 100): print(run.remote(updates))
