"""Evaluate a trained real-data run on Modal and persist metrics."""
import os

import modal

app = modal.App("object-dynamics-vla-real-eval")
volume = modal.Volume.from_name("object-dynamics-vla-data", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch>=2.7", "numpy>=1.24,<3", "Pillow>=10")
         .add_local_python_source("odvla"))


@app.function(image=image, gpu=os.environ.get("ODVLA_MODAL_GPU", "A100-80GB"),
              cpu=2, memory=16384, timeout=3600, max_containers=1,
              volumes={"/data": volume})
def evaluate_run(dataset_name: str, run_name: str, split: str, out_name: str) -> dict:
    from pathlib import Path
    from odvla.experiment import evaluate
    base = Path("/data").resolve()
    ds = (base / dataset_name).resolve()
    run = (base / run_name).resolve()
    out = (base / out_name).resolve()
    if not ds.is_relative_to(base) or not run.is_relative_to(base / "runs") or not out.is_relative_to(base):
        raise ValueError("Paths must be relative to /data; run under /data/runs")
    metrics = evaluate(ds, run, split, "cuda", output=out)
    volume.commit()
    return {"out": str(out), "metrics": metrics}


@app.local_entrypoint()
def main(dataset_name: str = "libero/object_oracle",
         run_name: str = "runs/libero_oracle_ablations/objects_joint_seed7",
         split: str = "test",
         out_name: str = "eval/libero_oracle_joint_seed7_test.json"):
    result = evaluate_run.remote(dataset_name, run_name, split, out_name)
    m = result["metrics"]
    print({"out": result["out"], "flow_mse": m["flow_mse"],
           "dynamics_mse": m["dynamics"]["normalized_mse"],
           "moving_dynamics_mse": m["dynamics"]["moving_normalized_mse"],
           "sampled_action_normalized_mse": m["sampled_action_normalized_mse"]})
