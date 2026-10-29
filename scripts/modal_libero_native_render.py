"""Render native-resolution RGB from recorded validation/demo states for perception experiments."""
import modal
from scripts.modal_libero_oracle import image as libero_image, volume

image = (libero_image.add_local_file("scripts/__init__.py", "/root/scripts/__init__.py")
         .add_local_file("scripts/modal_libero_oracle.py", "/root/scripts/modal_libero_oracle.py"))
app = modal.App("odvla-libero-native-render")


@app.function(image=image, volumes={"/data": volume}, cpu=4, memory=16384, timeout=1800)
def render(full: bool = False):
    import json
    from pathlib import Path
    import h5py
    import numpy as np
    from PIL import Image

    root = Path("/opt/LIBERO")
    config = Path.home() / ".libero" / "config.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(f"benchmark_root: {root}/libero/libero\nbddl_files: {root}/libero/libero/bddl_files\n"
                      f"init_states: {root}/libero/libero/init_files\ndatasets: {root}/libero/datasets\n"
                      f"assets: {root}/libero/libero/assets\n")
    from libero.libero.envs import OffScreenRenderEnv
    bddl = root / "libero/libero/bddl_files/libero_object/pick_up_the_milk_and_place_it_in_the_basket.bddl"
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=512, camera_widths=512,
                             camera_names=["agentview"])
    env.reset()
    output = Path("/data/eval/milk_native_512_full" if full else "/data/eval/milk_native_512")
    output.mkdir(parents=True, exist_ok=True)
    results = []
    with h5py.File("/data/libero/source/milk_demo.hdf5", "r") as source:
        demo = source["data/demo_0"]
        indices = range(len(demo["states"])) if full else (0, 50, 100)
        for frame in indices:
            env.set_state(np.asarray(demo["states"][frame]))
            env.sim.forward()
            rgb128 = env.sim.render(camera_name="agentview", width=128, height=128)
            diff = float(np.abs(rgb128.astype(np.int16) - np.asarray(demo["obs/agentview_rgb"][frame]).astype(np.int16)).mean())
            rgb512 = env.sim.render(camera_name="agentview", width=512, height=512)
            name = f"{frame:06}.jpg" if full else f"frame_{frame:03}.png"
            Image.fromarray(rgb512).save(output / name, quality=98)
            results.append({"frame": frame, "native_size": [512, 512], "rgb128_mean_abs_diff": diff})
    env.close()
    (output / "provenance.json").write_text(json.dumps({"source": "MuJoCo native 512x512 agentview RGB from recorded demo states", "frames": results}, indent=2) + "\n")
    volume.commit()
    return {"frames": len(results), "rgb128_mean_abs_diff": float(np.mean([r["rgb128_mean_abs_diff"] for r in results])),
            "rgb128_max_abs_diff": max(r["rgb128_mean_abs_diff"] for r in results), "output": str(output)}


@app.local_entrypoint()
def main(full: bool = False):
    print(render.remote(full))
