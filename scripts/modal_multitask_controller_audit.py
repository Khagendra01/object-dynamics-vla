"""Inspect actual simulator controller configuration for both source tasks."""
import modal
from scripts.modal_libero_oracle import image as base_image

volume = modal.Volume.from_name("object-dynamics-vla-data", create_if_missing=False)
image = (base_image.add_local_file("scripts/__init__.py", "/root/scripts/__init__.py")
         .add_local_file("scripts/modal_libero_oracle.py", "/root/scripts/modal_libero_oracle.py"))
app = modal.App("odvla-multitask-controller-audit")


@app.function(image=image, volumes={"/data": volume}, cpu=2, memory=8192, timeout=600)
def run():
    import inspect
    import json
    from pathlib import Path
    import h5py
    import numpy as np
    root = Path("/opt/LIBERO")
    config = Path.home() / ".libero/config.yaml"; config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(f"benchmark_root: {root}/libero/libero\nbddl_files: {root}/libero/libero/bddl_files\n"
                      f"init_states: {root}/libero/libero/init_files\ndatasets: {root}/libero/datasets\nassets: {root}/libero/libero/assets\n")
    from libero.libero.envs import OffScreenRenderEnv
    rows = {}
    for name, file in (("milk", "pick_up_the_milk_and_place_it_in_the_basket"),
                       ("soup", "pick_up_the_alphabet_soup_and_place_it_in_the_basket")):
        env = OffScreenRenderEnv(bddl_file_name=str(root / f"libero/libero/bddl_files/libero_object/{file}.bddl"),
                                 camera_heights=128, camera_widths=128, camera_names=["agentview"])
        env.reset(); robot = env.robots[0]; ctrl = robot.controller; lo, hi = env.env.action_spec
        source = "/data/libero/source/milk_demo.hdf5" if name == "milk" else "/data/libero/source/alphabet_soup_demo.hdf5"
        with h5py.File(source, "r") as data:
            demo = data["data/demo_0"]
            shape = {key: list(demo[key].shape) for key in ("actions", "obs/joint_states", "obs/ee_states", "obs/gripper_states")}
            actions = np.asarray(demo["actions"]); states = np.asarray(demo["states"])
            env.set_init_state(states[0]); distance = []
            for i, a in enumerate(actions):
                env.step(a)
                if i + 1 < len(states): distance.append(float(np.abs(env.get_sim_state() - states[i + 1]).mean()))
            success = bool(env.check_success())
        rows[name] = {"robot": type(robot).__name__, "controller": type(ctrl).__name__,
                      "action_low": lo.tolist(), "action_high": hi.tolist(),
                      "input_min": ctrl.input_min.tolist(), "input_max": ctrl.input_max.tolist(),
                      "output_min": ctrl.output_min.tolist(), "output_max": ctrl.output_max.tolist(),
                      "gripper": type(robot.gripper).__name__, "gripper_mapping": inspect.getsource(type(robot.gripper).format_action),
                      "control_freq": env.env.control_freq, "shapes_demo_0": shape,
                      "replay_state_mae": float(np.mean(distance)), "final_success": success}
        env.close()
    keys = ("robot", "controller", "action_low", "action_high", "input_min", "input_max", "output_min", "output_max", "gripper", "gripper_mapping", "control_freq")
    result = {"tasks": rows, "matching_simulator_configuration": all(rows["milk"][k] == rows["soup"][k] for k in keys),
              "scope": "demo_0 replay only, observed simulator settings; not proof of all source controller metadata"}
    dest = Path("/data/eval/multitask_controller_audit_v1.json")
    if dest.exists(): raise FileExistsError(dest)
    dest.write_text(json.dumps(result, indent=2) + "\n"); volume.commit(); return result


@app.local_entrypoint()
def main(): print(run.remote())
