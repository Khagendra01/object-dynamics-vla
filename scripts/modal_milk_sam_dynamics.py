"""One-step action-conditioned versus action-free SAM dynamics, validation only."""
import modal
from scripts.modal_real_ablations import image as base_image, volume

image = (base_image.add_local_file("scripts/__init__.py", "/root/scripts/__init__.py")
         .add_local_file("scripts/modal_real_ablations.py", "/root/scripts/modal_real_ablations.py"))
app = modal.App("odvla-milk-sam-dynamics")


@app.function(image=image, gpu="T4", cpu=4, memory=16384, volumes={"/data": volume}, timeout=7200)
def run(epochs: int = 25):
    from pathlib import Path
    import numpy as np
    import torch
    from torch import nn
    from odvla.data import build_windows, fit_stats, read_json, validate_splits, write_json
    from odvla.measurements import mask_metrics
    ds = Path("/data/libero/milk_sam_development_v1")
    out = Path("/data/runs/milk_sam_one_step_v1")
    if out.exists():
        raise FileExistsError("Use fresh output directory")
    splits = read_json(ds / "splits.json")
    manifest = validate_splits(ds, splits)
    stats = fit_stats(ds, splits)
    train = build_windows(ds, splits["train"], 4, 1, stats)
    val = build_windows(ds, splits["val"], 4, 1, stats)
    device = "cuda"
    torch.set_num_threads(2)
    out.mkdir(parents=True)
    xtr = torch.from_numpy(np.concatenate((train["x"], train["a"]), axis=1)).to(device)
    xva = torch.from_numpy(np.concatenate((val["x"], val["a"]), axis=1)).to(device)
    ytr, mtr = [torch.from_numpy(train[k]).to(device) for k in ("y", "mask")]
    yva, mva = [torch.from_numpy(val[k]).to(device) for k in ("y", "mask")]
    predictions = {"persistence": np.zeros_like(val["y"])}
    histories = []
    for seed in (7, 19, 31):
        for conditioned in (False, True):
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed)
            model = nn.Sequential(nn.Linear(xtr.shape[1], 128), nn.GELU(), nn.Linear(128, 128),
                                  nn.GELU(), nn.Linear(128, ytr.shape[1])).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            a = xtr.clone(); b = xva.clone()
            if not conditioned:
                a[:, -train["a"].shape[1]:] = 0
                b[:, -val["a"].shape[1]:] = 0
            best = float("inf")
            selected = None
            for epoch in range(1, epochs + 1):
                model.train()
                for indices in np.array_split(rng.permutation(len(a)), max(1, int(np.ceil(len(a) / 64)))):
                    pred = model(a[indices])
                    loss = ((pred - ytr[indices]).square() * mtr[indices]).sum() / mtr[indices].sum().clamp_min(1)
                    optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
                model.eval()
                with torch.no_grad():
                    prediction = model(b)
                    score = float((((prediction - yva).square() * mva).sum() / mva.sum()).item())
                if score < best:
                    best, selected = score, epoch
                    predictions[f"{'action' if conditioned else 'no_action'}_seed{seed}"] = prediction.cpu().numpy()
                    torch.save(model.state_dict(), out / f"{'action' if conditioned else 'no_action'}_seed{seed}.pt")
            histories.append({"seed": seed, "conditioned": conditioned, "best_epoch": selected,
                              "selection": "validation SAM one-step normalized MSE", "best_mse": best})
    # References loaded only after all models/checkpoints are trained.
    refs = {}
    for name in splits["val"]:
        with np.load(f"/data/libero/milk_masks_oracle/{name}.npz", allow_pickle=False) as z:
            refs[name] = mask_metrics(z["masks"].astype(bool))
    reference = np.stack([refs[r["episode"]][0][r["t"] + 1] - refs[r["episode"]][0][r["t"]]
                          for r in val["refs"]])
    reference_valid = np.stack([refs[r["episode"]][1][r["t"] + 1] & refs[r["episode"]][1][r["t"]]
                                for r in val["refs"]])
    sam = val["y"].reshape(-1, 2, 5) * stats["feature_std"]
    valid = val["mask"].reshape(-1, 2, 5).astype(bool) & reference_valid
    scores = {}
    for name, pred in predictions.items():
        native = pred.reshape(-1, 2, 5) * stats["feature_std"]
        scores[name] = {}
        for slot, obj in enumerate(manifest["object_slots"]):
            for j, metric in enumerate(manifest["metric_names"]):
                moving = np.abs(reference[:, slot, j]) > 0.002
                cell = {}
                for group, selection in (("all", valid[:, slot, j]), ("moving", valid[:, slot, j] & moving),
                                         ("stationary", valid[:, slot, j] & ~moving)):
                    cell[group] = {"count": int(selection.sum()),
                        "sam_delta_mae": float(np.abs(native[:, slot, j] - sam[:, slot, j])[selection].mean()) if selection.any() else None,
                        "reference_delta_mae": float(np.abs(native[:, slot, j] - reference[:, slot, j])[selection].mean()) if selection.any() else None}
                scores[name][f"{obj}/{metric}"] = cell
    result = {"dataset_fingerprint": splits["fingerprint"], "seeds": [7, 19, 31], "epochs": epochs,
        "input": "SAM observation history through t, plus a[t] in action-conditioned variant only",
        "target": "SAM one-step change", "selection": "minimum validation SAM normalized MSE for both learned variants",
        "moving_definition": "absolute simulator reference change > 0.002 per object and metric; scoring only",
        "reference_use": "loaded after training; evaluation only", "scores": scores, "runs": histories,
        "validation_windows": len(val["x"]), "test_accessed": False}
    write_json(out / "comparison.json", result)
    volume.commit()
    return {"output": str(out), "runs": histories, "validation_windows": len(val["x"])}


@app.local_entrypoint()
def main(epochs: int = 25):
    print(run.remote(epochs))
