"""Render LIBERO simulator-oracle masks on Modal.

Example:
  modal run scripts/modal_libero_oracle.py --episode episode_0000 \
    --out /data/libero/masks_oracle_ep0
"""
import os
import subprocess
import sys

import modal

app = modal.App("object-dynamics-vla-libero-oracle")
volume = modal.Volume.from_name("object-dynamics-vla-data", create_if_missing=True)
LIBERO_COMMIT = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libosmesa6-dev", "libgl1", "libglew-dev", "patchelf", "ffmpeg")
    .pip_install("torch==2.2.0", index_url="https://download.pytorch.org/whl/cpu")
    .run_commands(
        "git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git /opt/LIBERO",
        f"cd /opt/LIBERO && git checkout {LIBERO_COMMIT}",
    )
    .pip_install("numpy==1.23.5", "mujoco==2.3.7", "robosuite==1.4.0",
                 "bddl==1.0.1", "gym==0.25.2", "cloudpickle==2.1.0",
                 "future", "easydict", "hydra-core==1.2.0", "matplotlib",
                 "termcolor", "numba", "scipy", "opencv-python-headless",
                 "h5py", "Pillow", "pyyaml")
    .run_commands("pip install -e /opt/LIBERO")
    .env({"MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa"})
    .add_local_file("scripts/extract_libero_oracle_masks.py",
                    "/root/extract_libero_oracle_masks.py")
)


@app.function(image=image, volumes={"/data": volume}, cpu=8, memory=16384,
              timeout=7200, max_containers=1)
def render(source, dataset, out, bddl, bodies, episode, max_rgb_diff) -> dict:
    cmd = [sys.executable, "/root/extract_libero_oracle_masks.py",
           "--source", source, "--dataset", dataset, "--out", out,
           "--bddl", bddl, "--bodies", bodies,
           "--libero-commit", LIBERO_COMMIT,
           "--max-rgb-diff", str(max_rgb_diff)]
    if episode:
        cmd += ["--episode", episode]
    result = subprocess.run(cmd, capture_output=True, text=True)
    volume.commit()
    files = {}
    if os.path.isdir(out):
        for name in sorted(os.listdir(out)):
            files[name] = os.path.getsize(os.path.join(out, name))
    return {"returncode": result.returncode, "stdout": result.stdout[-16000:],
            "stderr": result.stderr[-16000:], "files": files}


@app.local_entrypoint()
def main(source: str = "/data/libero/source/alphabet_soup_demo.hdf5",
         dataset: str = "/data/libero/object_raw",
         out: str = "/data/libero/masks_oracle_ep0",
         bddl: str = "/opt/LIBERO/libero/libero/bddl_files/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket.bddl",
         bodies: str = "alphabet_soup_1,basket_1",
         episode: str = "episode_0000",
         max_rgb_diff: float = 20.0):
    result = render.remote(source, dataset, out, bddl, bodies, episode, max_rgb_diff)
    print(result["stdout"])
    if result["stderr"]:
        print(result["stderr"], file=sys.stderr)
    print({"returncode": result["returncode"], "files": result["files"]})
    if result["returncode"] != 0:
        raise SystemExit(result["returncode"])
