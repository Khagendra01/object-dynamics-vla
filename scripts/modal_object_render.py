"""Render audited alphabet-soup train/validation observations at native resolution."""
import modal
from scripts.modal_libero_oracle import image as base_image

volume = modal.Volume.from_name("object-dynamics-vla-data", create_if_missing=False)
image = (base_image.add_local_file("scripts/__init__.py", "/root/scripts/__init__.py")
         .add_local_file("scripts/modal_libero_oracle.py", "/root/scripts/modal_libero_oracle.py"))
app = modal.App("odvla-soup-render")


@app.function(image=image, volumes={"/data": volume}, cpu=4, memory=16384, timeout=5400, max_containers=1)
def render(full: bool = False):
    import hashlib
    import io
    import json
    import time
    from pathlib import Path
    import h5py
    import numpy as np
    from PIL import Image
    root = Path("/opt/LIBERO")
    config = Path.home() / ".libero/config.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(f"benchmark_root: {root}/libero/libero\nbddl_files: {root}/libero/libero/bddl_files\n"
                      f"init_states: {root}/libero/libero/init_files\ndatasets: {root}/libero/datasets\n"
                      f"assets: {root}/libero/libero/assets\n")
    from libero.libero.envs import OffScreenRenderEnv
    data_root = Path("/data/libero/object_raw")
    manifest = json.loads((data_root / "manifest.json").read_text())
    splits = json.loads((data_root / "splits.json").read_text())
    chosen = set(splits["train"] + splits["val"])
    entries = [entry for entry in manifest["episodes"] if entry["id"] in chosen]
    if not full: entries = entries[:1]
    out = Path("/data/libero/object_native_512_v1" if full else "/data/eval/object_native_512_pilot_v1")
    if out.exists() and not full: raise FileExistsError("Use a fresh pilot output directory")
    if (out / "provenance.json").exists(): raise FileExistsError("Completed render already exists")
    out.mkdir(parents=True, exist_ok=True)
    bddl = root / "libero/libero/bddl_files/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket.bddl"
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=512, camera_widths=512,
                             camera_names=["agentview"])
    env.reset(); started = time.monotonic(); records = []
    try:
        with h5py.File("/data/libero/source/alphabet_soup_demo.hdf5", "r") as source:
            for entry in entries:
                episode, demo = entry["id"], entry["source_demo"]
                record = source["data"][demo]; folder = out / episode; folder.mkdir(exist_ok=True)
                expected = {f"{t:06}.jpg" for t in range(len(record["states"]))}
                if not {p.name for p in folder.iterdir()}.issubset(expected):
                    raise ValueError(f"Unexpected files in {folder}")
                digest = hashlib.sha256(); diffs = []
                for t in range(len(record["states"])):
                    env.set_state(np.asarray(record["states"][t])); env.sim.forward()
                    rgb128 = env.sim.render(camera_name="agentview", width=128, height=128)
                    diffs.append(float(np.abs(rgb128.astype(np.int16) - np.asarray(record["obs/agentview_rgb"][t]).astype(np.int16)).mean()))
                    path = folder / f"{t:06}.jpg"
                    if path.exists():
                        rgb = env.sim.render(camera_name="agentview", width=512, height=512)
                        check = io.BytesIO(); Image.fromarray(rgb).save(check, format="JPEG", quality=98)
                        if path.read_bytes() != check.getvalue():
                            raise ValueError(f"Existing frame differs from regenerated frame: {path}")
                    else:
                        rgb = env.sim.render(camera_name="agentview", width=512, height=512)
                        Image.fromarray(rgb).save(path, quality=98)
                    digest.update(path.read_bytes())
                result = {"episode": episode, "demo": demo, "split": "train" if episode in splits["train"] else "val",
                          "frames": len(diffs), "jpeg_sequence_sha256": digest.hexdigest(),
                          "rgb128_mean_abs_diff": float(np.mean(diffs)), "rgb128_max_abs_diff": max(diffs)}
                records.append(result)
                print(json.dumps({**result, "elapsed": time.monotonic() - started}), flush=True)
                volume.commit()
    finally:
        env.close()
    output = {"source": "MuJoCo native 512 agentview RGB from recorded states", "jpeg_quality": 98,
              "fps": 20, "dataset_fingerprint": splits["fingerprint"], "test_episodes_rendered": False,
              "entries": records, "elapsed_seconds": time.monotonic() - started}
    (out / "provenance.json").write_text(json.dumps(output, indent=2) + "\n")
    volume.commit()
    return output


@app.local_entrypoint()
def main(full: bool = False): print(render.remote(full))
