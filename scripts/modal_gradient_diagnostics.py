"""Measure individual loss gradients on the shared encoder of frozen checkpoints."""
import modal

app = modal.App("odvla-gradient-diagnostics")
volume = modal.Volume.from_name("object-dynamics-vla-data")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch>=2.7", "numpy>=1.24,<3", "Pillow>=10")
         .add_local_python_source("odvla"))


@app.function(image=image, gpu="A100-80GB", cpu=4, memory=16384, timeout=3600,
              volumes={"/data": volume})
def measure():
    import json
    from pathlib import Path
    import numpy as np
    import torch
    from odvla.data import build_windows, read_json, validate_splits
    from odvla.experiment import load_run
    from odvla.model import masked_mse

    dataset = Path("/data/libero/milk_oracle")
    rows = []
    for seed in (7, 19, 31):
        run = Path(f"/data/runs/libero_milk_oracle_ablations/objects_joint_seed{seed}")
        model, config, stats = load_run(run, "cuda")
        model.eval()
        splits = read_json(run / "splits.json")
        validate_splits(dataset, splits)
        windows = build_windows(dataset, splits["train"], config["history"], config["horizon"], stats)
        rng = np.random.default_rng(1000 + seed)
        torch.manual_seed(2000 + seed)
        params = list(model.encoder.parameters())
        batch_rows = []
        for batch_index in range(32):
            indices = rng.choice(len(windows["x"]), size=64, replace=False)
            x, a, y, mask = [torch.from_numpy(windows[k][indices]).to("cuda") for k in ("x", "a", "y", "mask")]
            h, _ = model.encode(x)
            noise = torch.randn_like(a)
            tau = torch.rand((len(a), 1), device="cuda")
            flow = model.velocity(h, (1 - tau) * noise + tau * a, tau)
            action_loss = (flow - (a - noise)).square().mean()
            dynamics_loss = masked_mse(model.dynamics(h, a, False), y, mask)
            ga = torch.autograd.grad(action_loss, params, retain_graph=True)
            gd = torch.autograd.grad(dynamics_loss, params)
            ga = torch.cat([g.reshape(-1) for g in ga])
            gd = torch.cat([g.reshape(-1) for g in gd])
            na, nd = ga.norm(), gd.norm()
            cosine = torch.dot(ga, gd) / (na * nd).clamp_min(1e-20)
            batch_rows.append({"batch": batch_index, "action_loss": action_loss.item(),
                "dynamics_loss": dynamics_loss.item(), "action_grad_norm": na.item(),
                "dynamics_grad_norm": nd.item(), "cosine": cosine.item()})
        rows.append({"train_seed": seed, "split": "train", "batches": batch_rows,
                     "checkpoint_epoch": read_json(run / "best_validation.json")["epoch"]})
    report = {"scope": "32 random training batches at each frozen validation-selected joint checkpoint; encoder parameters only, before clipping",
              "rows": rows}
    output = Path("/data/eval/milk_shared_gradient_diagnostics.json")
    output.write_text(json.dumps(report, indent=2) + "\n")
    volume.commit()
    for r in rows:
        b = r["batches"]
        print({"seed": r["train_seed"], "epoch": r["checkpoint_epoch"],
               "action_grad_norm_mean": float(np.mean([x["action_grad_norm"] for x in b])),
               "dynamics_grad_norm_mean": float(np.mean([x["dynamics_grad_norm"] for x in b])),
               "cosine_mean": float(np.mean([x["cosine"] for x in b])),
               "cosine_negative_fraction": float(np.mean([x["cosine"] < 0 for x in b]))})
    return {"path": str(output), "seeds": len(rows)}


@app.local_entrypoint()
def main():
    print(measure.remote())
