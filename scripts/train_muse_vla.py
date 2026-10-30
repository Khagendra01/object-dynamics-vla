"""Train one declared Muse arm from a complete cache, with fixed validation noise."""
import argparse
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from odvla.data import file_hash, fit_stats, read_json, write_json
from odvla.vla.action_expert import MuseActionExpert
from odvla.vla.dataset import CachedEpisodeDataset


def make_batch(samples, device):
    lengths = [len(s["muse_valid"]) for s in samples]
    batch, maximum = len(samples), max(lengths)
    bits = np.zeros((batch, maximum, samples[0]["muse_bits"].shape[1]), dtype=np.int16)
    mask = np.zeros((batch, maximum), dtype=bool)
    for i, sample in enumerate(samples):
        bits[i, :lengths[i]] = sample["muse_bits"].view(np.int16)
        mask[i, :lengths[i]] = sample["muse_valid"]
    result = {"muse": torch.from_numpy(bits).view(torch.bfloat16).to(device),
              "muse_valid": torch.from_numpy(mask).to(device)}
    keys = ("robot", "history_valid", "history_timestamps", "sam_values", "sam_deltas", "sam_valid", "sam_age",
            "actions", "action_valid", "change", "change_valid")
    result.update({k: torch.from_numpy(np.stack([s[k] for s in samples])).to(device) for k in keys})
    return result


def context(model, batch, use_sam):
    return model.context(batch["muse"], batch["muse_valid"], batch["robot"], batch["history_valid"],
        batch["history_timestamps"], batch["sam_values"], batch["sam_deltas"], batch["sam_valid"], batch["sam_age"], use_sam)


def evaluate(model, dataset, device, use_sam, batch_size=8):
    """Fixed seed, index-stable flow-noise/time schedule shared by variants."""
    model.eval(); total = count = 0.0
    noise_rng = np.random.default_rng(921); time_rng = np.random.default_rng(922)
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            b = make_batch([dataset[i] for i in range(start, min(start + batch_size, len(dataset)))], device)
            a = b["actions"]; n = torch.from_numpy(noise_rng.normal(size=tuple(a.shape)).astype(np.float32)).to(device)
            t = torch.from_numpy((time_rng.beta(1.5, 1, size=len(a)) * .999 + .001).astype(np.float32)).to(device)
            ctx, mask = context(model, b, use_sam)
            _, flow, _ = model.losses(ctx, mask, a, b["action_valid"], n, t, b["change"], b["change_valid"], weight=0)
            valid = int(b["action_valid"].sum()) * a.shape[-1]
            total += float(flow) * valid; count += valid
    return total / count


def train(dataset, cache, out, variant, seed, device, steps=5000, batch_size=8, accumulation=8,
          prepared=None, checkpoint_hook=None, motor_checkpoint=None, allow_scratch=False):
    if variant not in ("motor", "M0", "M1", "M2") or batch_size * accumulation != 64:
        raise ValueError("Use motor/M0/M1/M2 and effective batch size 64")
    if variant != "motor" and motor_checkpoint is None and not allow_scratch:
        raise ValueError("Comparison variants require a common motor checkpoint")
    if variant == "motor" and motor_checkpoint is not None:
        raise ValueError("Motor pretraining must start from scratch")
    root, out = Path(dataset), Path(out)
    if out.exists(): raise FileExistsError("Use a fresh experiment output")
    splits = read_json(root / "splits.json")
    if prepared is None:
        stats = fit_stats(root, splits)
        trainset = CachedEpisodeDataset(root, cache, "train", stats)
        valset = CachedEpisodeDataset(root, cache, "val", stats)
    else:
        stats, trainset, valset = prepared
    torch.manual_seed(seed)
    model = MuseActionExpert().to(device)
    checkpoint_hash = None
    if motor_checkpoint is not None:
        checkpoint = Path(motor_checkpoint)
        metadata = read_json(checkpoint.parent / "configuration.json")
        if metadata["variant"] != "motor" or metadata["dataset_fingerprint"] != splits["fingerprint"]:
            raise ValueError("Motor checkpoint provenance or dataset fingerprint mismatch")
        with np.load(checkpoint.parent / "stats.npz", allow_pickle=False) as saved:
            if set(saved.files) != set(stats) or any(not np.array_equal(saved[k], stats[k]) for k in stats):
                raise ValueError("Motor checkpoint normalization statistics mismatch")
        checkpoint_hash = file_hash(checkpoint)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(.9, .95), weight_decay=1e-5)
    batch_rng = np.random.default_rng(seed); noise_rng = np.random.default_rng(seed + 10000)
    time_rng = np.random.default_rng(seed + 20000)
    out.mkdir(parents=True)
    np.savez(out / "stats.npz", **stats)
    write_json(out / "configuration.json", {"variant": variant, "seed": seed, "steps": steps,
        "batch_size": batch_size, "accumulation": accumulation, "dataset_fingerprint": splits["fingerprint"],
        "selection": "minimum fixed-schedule validation flow MSE", "training_data": "train only",
        "trainable_parameters": {n: p.numel() for n, p in model.named_parameters() if p.requires_grad},
        "flow_convention": "noise at one", "sam": variant in ("M1", "M2"), "dynamics_weight": .1 if variant == "M2" else 0,
        "motor_checkpoint_sha256": checkpoint_hash, "scratch_diagnostic": variant != "motor" and checkpoint_hash is None})
    best = float("inf"); started = time.monotonic(); gradient_diagnostic = None
    for step in range(1, steps + 1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        # Independent RNGs ensure all variants receive identical batches/noise/timesteps.
        for microbatch in range(accumulation):
            indices = batch_rng.integers(len(trainset), size=batch_size)
            b = make_batch([trainset[int(i)] for i in indices], device)
            a = b["actions"]
            noise = torch.from_numpy(noise_rng.normal(size=tuple(a.shape)).astype(np.float32)).to(device)
            t = torch.from_numpy((time_rng.beta(1.5, 1, size=len(a)) * .999 + .001).astype(np.float32)).to(device)
            ctx, mask = context(model, b, variant in ("M1", "M2"))
            loss, flow, dyn = model.losses(ctx, mask, a, b["action_valid"], noise, t, b["change"], b["change_valid"], .1 if variant == "M2" else 0)
            if step == 1 and microbatch == 0:
                shared = list(model.context_encoder.parameters()) + list(model.expert.parameters())
                def gradient_norm(objective):
                    grads = torch.autograd.grad(objective, shared, retain_graph=True, allow_unused=True)
                    return math.sqrt(sum(float(g.detach().float().square().sum()) for g in grads if g is not None))
                flow_norm, dynamics_norm = gradient_norm(flow), gradient_norm(dyn)
                gradient_diagnostic = {"step": step, "flow_norm": flow_norm,
                    "unweighted_dynamics_norm": dynamics_norm,
                    "unweighted_ratio": dynamics_norm / flow_norm if flow_norm else None}
                write_json(out / "shared_gradient_diagnostic.json", gradient_diagnostic)
            (loss / accumulation).backward()
        grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 10))
        warmup = min(step / 250, 1.0)
        decay = .5 * (1 + math.cos(math.pi * max(0, step - 250) / max(1, steps - 250)))
        for group in optimizer.param_groups: group["lr"] = 1e-4 * warmup * decay
        optimizer.step()
        if step % 250 == 0 or step == steps:
            value = evaluate(model, valset, device, variant in ("M1", "M2"), batch_size)
            torch.save(model.state_dict(), out / f"weights_step{step:05}.pt")
            if value < best:
                best = value
                torch.save(model.state_dict(), out / "weights_best.pt")
                write_json(out / "best_validation.json", {"step": step, "flow_mse": value})
            row = {"step": step, "val_flow_mse": value, "last_flow": float(flow.detach()),
                   "last_dynamics": float(dyn.detach()), "gradient_norm": grad,
                   "elapsed_seconds": time.monotonic() - started}
            with (out / "train.jsonl").open("a") as f: f.write(json.dumps(row, allow_nan=False) + "\n")
            print(json.dumps(row), flush=True)
            if checkpoint_hook is not None:
                checkpoint_hook()
    return {"best_validation_flow_mse": best, "output": str(out)}


def main():
    p = argparse.ArgumentParser(); p.add_argument("--dataset", required=True); p.add_argument("--cache", required=True)
    p.add_argument("--out", required=True); p.add_argument("--variant", choices=("motor", "M0", "M1", "M2"), required=True)
    p.add_argument("--seed", type=int, default=7); p.add_argument("--device", default="cuda")
    p.add_argument("--steps", type=int, default=5000); p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--accumulation", type=int, default=8)
    p.add_argument("--motor-checkpoint", help="Selected motor weights_best.pt; required for comparisons")
    p.add_argument("--allow-scratch", action="store_true", help="Permit scratch diagnostic, not a controlled comparison")
    a = p.parse_args(); print(train(a.dataset, a.cache, a.out, a.variant, a.seed, a.device, a.steps, a.batch_size, a.accumulation, motor_checkpoint=a.motor_checkpoint, allow_scratch=a.allow_scratch))


if __name__ == "__main__": main()
