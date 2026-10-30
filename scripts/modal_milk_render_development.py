"""Render only the pre-existing milk train/validation split at native resolution."""
import modal
from scripts.modal_libero_oracle import image as base_image, volume

image = (base_image.add_local_file("scripts/__init__.py", "/root/scripts/__init__.py")
         .add_local_file("scripts/modal_libero_oracle.py", "/root/scripts/modal_libero_oracle.py"))
app = modal.App("odvla-milk-development-render")


@app.function(image=image, volumes={"/data": volume}, cpu=4, memory=16384, timeout=7200)
def render():
    import hashlib
    import json
    from pathlib import Path
    import h5py
    import numpy as np
    from PIL import Image, ImageDraw
    root = Path("/opt/LIBERO")
    config = Path.home() / ".libero/config.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(f"benchmark_root: {root}/libero/libero\nbddl_files: {root}/libero/libero/bddl_files\n"
                      f"init_states: {root}/libero/libero/init_files\ndatasets: {root}/libero/datasets\n"
                      f"assets: {root}/libero/libero/assets\n")
    from libero.libero.envs import OffScreenRenderEnv
    source = Path("/data/libero/source/milk_demo.hdf5")
    manifest = json.loads(Path("/data/libero/milk_raw/manifest.json").read_text())
    splits = json.loads(Path("/data/libero/milk_oracle/splits.json").read_text())
    chosen = set(splits["train"] + splits["val"])
    entries = [e for e in manifest["episodes"] if e["id"] in chosen]
    output = Path("/data/libero/milk_sam_native_v1")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a fresh render directory")
    output.mkdir(parents=True)
    env = OffScreenRenderEnv(bddl_file_name=str(root / "libero/libero/bddl_files/libero_object/pick_up_the_milk_and_place_it_in_the_basket.bddl"),
                             camera_heights=512, camera_widths=512, camera_names=["agentview"])
    env.reset()
    sheet = Image.new("RGB", (512 * 7, 540 * 6), "white")
    draw = ImageDraw.Draw(sheet)
    results = []
    with h5py.File(source, "r") as data:
        for index, entry in enumerate(entries):
            episode, demo = entry["id"], entry["source_demo"]
            record = data["data"][demo]
            folder = output / episode
            folder.mkdir()
            diffs = []
            digest = hashlib.sha256()
            for t in range(len(record["states"])):
                env.set_state(np.asarray(record["states"][t]))
                env.sim.forward()
                image128 = env.sim.render(camera_name="agentview", width=128, height=128)
                diffs.append(float(np.abs(image128.astype(np.int16) - np.asarray(record["obs/agentview_rgb"][t]).astype(np.int16)).mean()))
                rgb = env.sim.render(camera_name="agentview", width=512, height=512)
                path = folder / f"{t:06}.jpg"
                Image.fromarray(rgb).save(path, quality=98)
                digest.update(path.read_bytes())
                if t == 0:
                    cell = Image.open(path).convert("RGB")
                    mark = ImageDraw.Draw(cell)
                    for x, y in ((128, 277), (410, 210)):
                        mark.ellipse((x-4, y-4, x+4, y+4), fill="red")
                    x, y = (index % 7) * 512, (index // 7) * 540
                    sheet.paste(cell, (x, y + 28))
                    draw.text((x + 4, y + 5), f"{episode} / {demo}", fill="black")
            results.append({"episode": episode, "demo": demo, "split": "train" if episode in splits["train"] else "val",
                            "frames": len(record["states"]), "jpeg_sequence_sha256": digest.hexdigest(),
                            "rgb128_mean_abs_diff": float(np.mean(diffs)), "rgb128_max_abs_diff": max(diffs)})
    env.close()
    sheet.save(output / "first_frames.png")
    (output / "provenance.json").write_text(json.dumps({"source": "MuJoCo native 512 agentview RGB from recorded demonstration states",
        "jpeg_quality": 98, "fps": 20, "entries": results, "test_episodes_rendered": False}, indent=2) + "\n")
    volume.commit()
    return {"episodes": len(results), "frames": sum(x["frames"] for x in results), "output": str(output)}


@app.local_entrypoint()
def main():
    print(render.remote())
