"""Train the matched real-data ablation suite on Modal (validation only)."""
import os
import sys
from pathlib import Path

import modal

app = modal.App("object-dynamics-vla-real-ablations")
volume = modal.Volume.from_name("object-dynamics-vla-data", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch>=2.7", "numpy>=1.24,<3", "Pillow>=10")
         .add_local_python_source("odvla"))


@app.function(image=image, gpu=os.environ.get("ODVLA_MODAL_GPU", "A100-80GB"),
              cpu=4, memory=16384, timeout=7200, max_containers=1,
              volumes={"/data": volume})
def run_suite(dataset_name: str, out_name: str, epochs: int, seeds: str) -> dict:
    from odvla.data import make_splits, write_json
    from odvla.experiment import train, evaluate
    base = Path("/data").resolve()
    ds = (base / dataset_name).resolve()
    root = (base / out_name).resolve()
    if not ds.is_relative_to(base) or not root.is_relative_to(base / "runs"):
        raise ValueError("Use dataset relative to /data and output under /data/runs")
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Use a fresh suite directory")
    root.mkdir(parents=True)
    split_file = root / "splits.json"
    write_json(split_file, make_splits(ds))
    variants = {
        "no_objects": dict(no_objects=True, dynamics_weight=0),
        "objects_action_only": dict(dynamics_weight=0),
        "objects_joint": dict(dynamics_weight=1),
        "objects_joint_no_dynamics_actions": dict(dynamics_weight=1, no_dynamics_actions=True),
    }
    rows = []
    for seed in [int(x) for x in seeds.split(",")]:
        for name, options in variants.items():
            run = root / f"{name}_seed{seed}"
            train(ds, split_file, run, epochs=epochs, seed=seed, device="cuda", **options)
            metrics = evaluate(ds, run, "val", "cuda")
            rows.append({"variant": name, "seed": seed, "split": "val",
                         "flow_mse": metrics["flow_mse"],
                         "sampled_action_normalized_mse": metrics["sampled_action_normalized_mse"],
                         "dynamics_mse": metrics["dynamics"]["normalized_mse"],
                         "moving_dynamics_mse": metrics["dynamics"]["moving_normalized_mse"]})
            write_json(root / "comparison.json", rows)
    volume.commit()
    return {"root": str(root), "comparison": rows}


@app.local_entrypoint()
def main(dataset_name: str = "libero/object_oracle",
         out_name: str = "runs/libero_oracle_ablations",
         epochs: int = 30, seeds: str = "7,19,31"):
    result = run_suite.remote(dataset_name, out_name, epochs, seeds)
    print({"root": result["root"], "rows": len(result["comparison"])})
    for row in result["comparison"]:
        print(row)
