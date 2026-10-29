"""Optional cloud training on a user-populated Volume. Never called by local smoke.

Install Modal, authenticate, upload the canonical dataset and splits, then run
explicitly. This script spends cloud credit only when its entrypoint is invoked.
Override the default A100-80GB with ODVLA_MODAL_GPU (for example T4 or L4) and
the 1800s timeout with ODVLA_MODAL_TIMEOUT.
"""
import os

import modal

app = modal.App("object-dynamics-vla-experiments")
volume = modal.Volume.from_name("object-dynamics-vla-data", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch>=2.7", "numpy>=1.24,<3", "Pillow>=10")
         .add_local_python_source("odvla"))


@app.function(image=image, gpu=os.environ.get("ODVLA_MODAL_GPU", "A100-80GB"),
              cpu=2, memory=16384, timeout=int(os.environ.get("ODVLA_MODAL_TIMEOUT", "1800")),
              max_containers=1, volumes={"/data": volume})
def run_training(dataset_name: str, run_name: str, epochs: int = 30):
    from pathlib import Path
    from odvla.experiment import train
    base = Path("/data").resolve()
    ds = (base / dataset_name).resolve()
    run = (base / "runs" / run_name).resolve()
    if not ds.is_relative_to(base) or not run.is_relative_to(base / "runs"):
        raise ValueError("Use relative dataset/run paths within the mounted volume")
    config = train(ds, ds / "splits.json", run, epochs=epochs, device="cuda", threads=2)
    volume.commit()
    return {"run": str(run), "config": config}


@app.local_entrypoint()
def main(dataset_name: str, run_name: str, epochs: int = 30):
    print(run_training.remote(dataset_name, run_name, epochs))
