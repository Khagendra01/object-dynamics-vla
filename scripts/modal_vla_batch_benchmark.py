"""Measure batched expert-only training on real cached 429-token inputs."""
import modal
from scripts.modal_real_ablations import image as base_image, volume

image = (base_image.add_local_file("scripts/__init__.py", "/root/scripts/__init__.py")
         .add_local_file("scripts/modal_real_ablations.py", "/root/scripts/modal_real_ablations.py"))
app = modal.App("odvla-vla-batch-benchmark")


@app.function(image=image, gpu="T4", cpu=4, memory=16384, volumes={"/data": volume}, timeout=600)
def run(batch_size: int = 8):
    import json
    import time
    from pathlib import Path
    import numpy as np
    import torch
    from odvla.vla.action_expert import MuseActionExpert
    from odvla.vla.dataset import observation_window
    if batch_size not in (8, 64): raise ValueError("batch_size must be 8 or 64")
    out = Path(f"/data/eval/muse_vla_batch_benchmark_b{batch_size}_v1.json")
    if out.exists(): raise FileExistsError("Refusing to overwrite")
    with np.load("/data/libero/milk_sam_development_v1/episodes/episode_0000.npz", allow_pickle=False) as z:
        ep = {k: z[k] for k in z.files}
    with np.load("/data/runs/milk_sam_matched_policy_v1/action_only_seed7/stats.npz", allow_pickle=False) as z:
        stats = {k: z[k] for k in z.files}
    windows = [observation_window(ep, t, stats) for t in range(8)]
    cache = Path("/data/libero/muse_middle_tokens_development_v1/episode_0000")
    tokens, masks = [], []
    for t in range(8):
        with np.load(cache / f"{t:06}.npz", allow_pickle=False) as z:
            tokens.append(z["features_bfloat16_bits"].view(np.int16).copy())
            masks.append(z["attention_mask"].copy())
    repeat = batch_size // 8
    muse = torch.from_numpy(np.stack(tokens * repeat)).view(torch.bfloat16).cuda()
    mm = torch.from_numpy(np.stack(masks * repeat)).cuda()
    def get(k): return torch.from_numpy(np.stack([w[k] for w in windows] * repeat)).cuda()
    torch.manual_seed(7); model = MuseActionExpert().cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(.9, .95), weight_decay=1e-5)
    start = time.monotonic(); values = []
    for _ in range(10):
        ctx, mask = model.context(muse, mm, get("robot"), get("history_valid"), get("history_timestamps"),
            get("sam_values"), get("sam_deltas"), get("sam_valid"), get("sam_age"))
        a = get("actions")
        loss, _, _ = model.losses(ctx, mask, a, get("action_valid"), torch.randn_like(a),
                                  torch.full((batch_size,), .5, device="cuda"), get("change"), get("change_valid"))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); values.append(float(loss.detach()))
    torch.cuda.synchronize(); elapsed = time.monotonic() - start
    result = {"gpu": torch.cuda.get_device_name(), "batch_size": batch_size, "updates": 10, "seconds": elapsed,
        "seconds_per_batch": elapsed / 10, "estimated_64_example_update_seconds": elapsed / 10 * 64 / batch_size,
        "peak_gpu_gib": torch.cuda.max_memory_allocated() / 1024 ** 3, "losses": values,
        "includes": "resident-token forward/backward and optimizer; excludes cache IO, validation and checkpointing"}
    out.write_text(json.dumps(result, indent=2) + "\n"); volume.commit(); return result


@app.local_entrypoint()
def main(batch_size: int = 8): print(run.remote(batch_size))
