"""Measure real-cache expert forward/backward throughput before training."""
import modal
from scripts.modal_real_ablations import image as base_image, volume

image = (base_image.add_local_file("scripts/__init__.py", "/root/scripts/__init__.py")
         .add_local_file("scripts/modal_real_ablations.py", "/root/scripts/modal_real_ablations.py"))
app = modal.App("odvla-vla-train-benchmark")


@app.function(image=image, gpu="T4", cpu=4, memory=16384, volumes={"/data": volume}, timeout=900)
def benchmark():
    import json
    import time
    from pathlib import Path
    import numpy as np
    import torch
    from odvla.vla.dataset import observation_window
    from odvla.vla.action_expert import MuseActionExpert
    root = Path("/data/libero/milk_sam_development_v1")
    token_root = Path("/data/libero/muse_middle_tokens_development_v1")
    out = Path("/data/eval/muse_vla_train_benchmark_v1.json")
    if out.exists():
        raise FileExistsError("Refusing to overwrite")
    with np.load(root / "episodes/episode_0000.npz", allow_pickle=False) as z:
        ep = {k: z[k] for k in z.files}
    with np.load("/data/runs/milk_sam_matched_policy_v1/action_only_seed7/stats.npz", allow_pickle=False) as z:
        stats = {k: z[k] for k in z.files}
    torch.manual_seed(7); torch.set_num_threads(2)
    model = MuseActionExpert().cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(.9, .95), weight_decay=1e-5)
    losses = []
    start = time.monotonic()
    for t in range(8):
        w = observation_window(ep, t, stats)
        with np.load(token_root / "episode_0000" / f"{t:06}.npz", allow_pickle=False) as z:
            muse = torch.from_numpy(z["features_bfloat16_bits"].view(np.int16).copy()).view(torch.bfloat16)[None].cuda()
            mm = torch.from_numpy(z["attention_mask"].copy())[None].cuda()
        def get(key): return torch.from_numpy(np.asarray(w[key]))[None].cuda()
        ctx, mask = model.context(muse, mm, get("robot"), get("history_valid"), get("history_timestamps"),
                                  get("sam_values"), get("sam_deltas"), get("sam_valid"), get("sam_age"))
        action = get("actions")
        loss, _, _ = model.losses(ctx, mask, action, get("action_valid"), torch.randn_like(action),
                                  torch.tensor([.5], device="cuda"), get("change"), get("change_valid"))
        (loss / 8).backward(); losses.append(float(loss))
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
    opt.step(); torch.cuda.synchronize()
    elapsed = time.monotonic() - start
    result = {"device": torch.cuda.get_device_name(), "microbatch": 1, "accumulation": 8,
        "real_frames": list(range(8)), "seconds": elapsed, "seconds_per_example": elapsed / 8,
        "projected_64_example_step_seconds": elapsed * 8, "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
        "losses": losses, "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "note": "one update only; no quality claim or full 5000-step run"}
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n"); volume.commit()
    return result


@app.local_entrypoint()
def main(): print(benchmark.remote())
