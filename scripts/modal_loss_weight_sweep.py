"""Validation-only loss-weight sweep; never evaluates the held-out test split."""
import modal

app = modal.App("odvla-loss-weight-sweep")
volume = modal.Volume.from_name("object-dynamics-vla-data")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch>=2.7", "numpy>=1.24,<3", "Pillow>=10")
         .add_local_python_source("odvla"))


@app.function(image=image, gpu="A100-80GB", cpu=4, memory=16384, timeout=3600,
              volumes={"/data": volume})
def sweep():
    from pathlib import Path
    from odvla.data import make_splits, write_json
    from odvla.experiment import train, evaluate
    dataset = Path("/data/libero/milk_oracle")
    root = Path("/data/runs/milk_weight_sweep_val")
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Use a fresh sweep directory")
    root.mkdir(parents=True, exist_ok=True)
    split = root / "splits.json"
    write_json(split, make_splits(dataset, seed=17))
    rows = []
    for seed in (7, 19, 31):
        for weight in (1.0, 10.0, 100.0):
            run = root / f"weight{weight:g}_seed{seed}"
            train(dataset, split, run, epochs=30, seed=seed, device="cuda", dynamics_weight=weight)
            metrics = evaluate(dataset, run, "val", "cuda")
            rows.append({"seed": seed, "weight": weight, "split": "val",
                         "flow_mse": metrics["flow_mse"],
                         "sampled_action_mse": metrics["sampled_action_normalized_mse"],
                         "dynamics_mse": metrics["dynamics"]["normalized_mse"]})
            write_json(root / "comparison.json", rows)
    volume.commit()
    return {"rows": rows, "caveat": "each weight selects its checkpoint by flow + weight*dynamics on validation"}


@app.local_entrypoint()
def main():
    print(sweep.remote())
