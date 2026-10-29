"""Run the single-frame SAM 3.1 adapter on a Modal GPU.

The checkpoint and dataset live on the object-dynamics-vla-data Volume at
/data. Example:
  modal run scripts/modal_sam31.py --episode episode_0000 \
    --out /data/libero/masks_sam31_ep0
"""
import os
import subprocess
import sys

import modal

app = modal.App("object-dynamics-vla-sam31")
volume = modal.Volume.from_name("object-dynamics-vla-data", create_if_missing=True)
SAM_COMMIT = "f1647e8d0c1d280991f4d1b7d974c2704537fe66"
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("torch==2.10.0", "torchvision",
                 index_url="https://download.pytorch.org/whl/cu128")
    .run_commands(
        "git clone https://github.com/facebookresearch/sam3.git /opt/sam3",
        f"cd /opt/sam3 && git checkout {SAM_COMMIT}",
    )
    .pip_install("/opt/sam3")
    .pip_install("numpy<2", "Pillow", "opencv-python-headless", "setuptools",
                 "einops", "pycocotools", "psutil")
    .add_local_python_source("odvla")
    .add_local_file("scripts/extract_sam31.py", "/root/extract_sam31.py")
)


@app.function(image=image, gpu=os.environ.get("ODVLA_SAM_GPU", "A100-80GB"),
              volumes={"/data": volume}, timeout=7200, max_containers=1)
def extract(dataset: str, out: str, checkpoint: str, episode: str) -> dict:
    cmd = [sys.executable, "/root/extract_sam31.py", "--dataset", dataset,
           "--out", out, "--checkpoint", checkpoint]
    if episode:
        cmd += ["--episode", episode]
    result = subprocess.run(cmd, capture_output=True, text=True)
    volume.commit()
    files = {}
    names = ["provenance.json"]
    if episode:
        names.append(episode + ".npz")
    for name in names:
        path = os.path.join(out, name)
        if name and os.path.isfile(path):
            files[name] = os.path.getsize(path)
    return {"returncode": result.returncode, "stdout": result.stdout[-12000:],
            "stderr": result.stderr[-12000:], "files": files}


@app.local_entrypoint()
def main(dataset: str = "/data/libero/object_raw",
         out: str = "/data/libero/masks_sam31",
         checkpoint: str = "/data/checkpoints/sam3.1/sam3.1_multiplex.pt",
         episode: str = "episode_0000"):
    result = extract.remote(dataset, out, checkpoint, episode)
    print(result["stdout"])
    if result["stderr"]:
        print(result["stderr"], file=sys.stderr)
    print({"returncode": result["returncode"], "files": result["files"]})
    if result["returncode"] != 0:
        raise SystemExit(result["returncode"])
