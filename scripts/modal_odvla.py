"""Run dataset audit/import/replay commands on a Modal Volume.

The volume is mounted at /data. This runner does not train; use modal_train.py
for GPU training. Example:
  modal run scripts/modal_odvla.py --args "audit /data/libero/source.hdf5 --out /data/libero/source_audit.json"
"""
import shlex
import subprocess
import sys

import modal

app = modal.App("object-dynamics-vla-data-prep")
volume = modal.Volume.from_name("object-dynamics-vla-data", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("numpy>=1.24,<3", "Pillow>=10", "h5py>=3.10")
         .add_local_python_source("odvla"))


@app.function(image=image, cpu=4, memory=8192, timeout=3600,
              max_containers=1, volumes={"/data": volume})
def run_odvla(args: list[str]) -> dict:
    result = subprocess.run([sys.executable, "-m", "odvla", *args],
                            capture_output=True, text=True)
    volume.commit()
    return {"returncode": result.returncode,
            "stdout": result.stdout[-8000:], "stderr": result.stderr[-8000:]}


@app.local_entrypoint()
def main(args: str):
    result = run_odvla.remote(shlex.split(args))
    print(result["stdout"])
    if result["stderr"]:
        print(result["stderr"], file=sys.stderr)
    if result["returncode"] != 0:
        raise SystemExit(result["returncode"])
