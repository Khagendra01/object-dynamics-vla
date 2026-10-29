"""Check the LIBERO controller by replaying recorded actions from a demo state.

This is open-loop demonstration replay, not policy evaluation or task success
for a learned model. It is a prerequisite for an eventual closed-loop adapter.
"""
import modal

from scripts.modal_libero_oracle import image as libero_image, volume

image = (libero_image.add_local_file("scripts/__init__.py", "/root/scripts/__init__.py")
         .add_local_file("scripts/modal_libero_oracle.py", "/root/scripts/modal_libero_oracle.py"))

app = modal.App("odvla-libero-action-replay")


@app.function(image=image, volumes={"/data": volume}, cpu=4, memory=16384, timeout=1800)
def replay():
    import json
    from pathlib import Path

    import h5py
    import numpy as np

    root = Path("/opt/LIBERO")
    config = Path.home() / ".libero" / "config.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(f"benchmark_root: {root}/libero/libero\nbddl_files: {root}/libero/libero/bddl_files\n"
                      f"init_states: {root}/libero/libero/init_files\ndatasets: {root}/libero/datasets\n"
                      f"assets: {root}/libero/libero/assets\n")
    from libero.libero.envs import OffScreenRenderEnv

    task = root / "libero/libero/bddl_files/libero_object/pick_up_the_milk_and_place_it_in_the_basket.bddl"
    env = OffScreenRenderEnv(bddl_file_name=str(task), camera_heights=128, camera_widths=128,
                             camera_names=["agentview"])
    with h5py.File("/data/libero/source/milk_demo.hdf5", "r") as source:
        demo = source["data/demo_0"]
        actions = np.asarray(demo["actions"])
        states = np.asarray(demo["states"])
        env.reset()
        env.set_init_state(states[0])
        distances, successes = [], []
        for i, action in enumerate(actions):
            _, reward, done, _ = env.step(action)
            if i + 1 < len(states):
                current = env.get_sim_state()
                distances.append(float(np.mean(np.abs(current - states[i + 1]))))
            if env.check_success():
                successes.append(i)
        result = {"task": "milk_in_basket", "demo": "demo_0", "steps": len(actions),
                  "success_observed": bool(successes),
                  "first_success_step": successes[0] if successes else None,
                  "final_reward": float(reward), "final_done": bool(done),
                  "state_mae_mean": float(np.mean(distances)),
                  "state_mae_max": float(np.max(distances)),
                  "interpretation": "Recorded demonstration actions replayed open loop; not learned policy success"}
    env.close()
    destination = Path("/data/eval/milk_recorded_action_replay.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    volume.commit()
    return result


@app.local_entrypoint()
def main(inspect_only: bool = False):
    print(inspect_controller.remote() if inspect_only else replay.remote())


@app.function(image=image, volumes={"/data": volume}, cpu=2, memory=8192, timeout=600)
def inspect_controller():
    import inspect
    from pathlib import Path
    root = Path("/opt/LIBERO")
    config = Path.home() / ".libero" / "config.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(f"benchmark_root: {root}/libero/libero\nbddl_files: {root}/libero/libero/bddl_files\n"
                      f"init_states: {root}/libero/libero/init_files\ndatasets: {root}/libero/datasets\n"
                      f"assets: {root}/libero/libero/assets\n")
    from libero.libero.envs import OffScreenRenderEnv
    task = root / "libero/libero/bddl_files/libero_object/pick_up_the_milk_and_place_it_in_the_basket.bddl"
    env = OffScreenRenderEnv(bddl_file_name=str(task), camera_heights=128,
                             camera_widths=128, camera_names=["agentview"])
    env.reset()
    robot = env.robots[0]
    low, high = env.env.action_spec
    result = {"action_low": low.tolist(), "action_high": high.tolist(),
              "controller_class": type(robot.controller).__name__,
              "controller_input_min": robot.controller.input_min.tolist(),
              "controller_input_max": robot.controller.input_max.tolist(),
              "gripper_class": type(robot.gripper).__name__,
              "gripper_format_action_source": inspect.getsource(type(robot.gripper).format_action)}
    env.close()
    return result
