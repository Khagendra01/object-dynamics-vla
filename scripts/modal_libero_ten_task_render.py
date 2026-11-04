"""Render aligned LIBERO-Object training observations for a ten-task integration."""
import modal
from scripts.modal_muse_motor_rollout import runtime_base

app = modal.App("odvla-libero-object-ten-task-render")
volume = modal.Volume.from_name("object-dynamics-vla-data", create_if_missing=False)
image = runtime_base.add_local_python_source("odvla")


@app.function(image=image, volumes={"/data": volume}, cpu=2, memory=8192, timeout=300)
def preview_train(task_id: str = "libero_object_chocolate_pudding", count: int = 8):
    import h5py
    import numpy as np
    from pathlib import Path
    from PIL import Image, ImageDraw
    from odvla.data import read_json
    root = Path("/data")
    data = read_json(root / "libero/ten_task_sources/registry.json")
    task = next(t for t in data["tasks"] if t["task_id"] == task_id)
    demos = task["episodes"]["train"][:count]
    sheet = Image.new("RGB", (512 * 4, 548 * ((len(demos) + 3) // 4)), "white")
    draw = ImageDraw.Draw(sheet)
    with h5py.File(root / task["source_path"], "r") as source:
        for i, demo in enumerate(demos):
            rgb = np.asarray(source["data"][demo]["obs/agentview_rgb"][0])
            image = Image.fromarray(rgb).resize((512, 512), Image.Resampling.NEAREST)
            x, y = (i % 4) * 512, (i // 4) * 548
            sheet.paste(image, (x, y + 36))
            draw.text((x + 5, y + 8), f"{task_id} / {demo}", fill="black")
    output = root / "libero/ten_task_integration_v1" / f"{task_id}_train_preview.png"
    sheet.save(output)
    volume.commit()
    return {"task_id": task_id, "episodes": demos, "path": str(output)}


@app.function(image=image, volumes={"/data": volume}, cpu=4, memory=16384,
              timeout=5400, max_containers=1, single_use_containers=True)
def render(contacts_only: bool = True):
    import hashlib
    import json
    from pathlib import Path
    import h5py
    import numpy as np
    from PIL import Image, ImageDraw
    from robosuite.utils.transform_utils import quat2axisangle
    from odvla.data import read_json, write_json

    root = Path("/data")
    registry = read_json(root / "libero/ten_task_sources/registry.json")
    if registry.get("status") != "sources_prepared" or len(registry["tasks"]) != 10:
        raise ValueError("Ten checksum-verified source tasks are required")
    output = root / "libero/ten_task_integration_v1"
    output.mkdir(parents=True, exist_ok=True)
    project = Path("/opt/LIBERO")
    config = Path.home() / ".libero/config.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(f"benchmark_root: {project}/libero/libero\nbddl_files: {project}/libero/libero/bddl_files\n"
                      f"init_states: {project}/libero/libero/init_files\ndatasets: {project}/libero/datasets\n"
                      f"assets: {project}/libero/libero/assets\n")
    from libero.libero.envs import OffScreenRenderEnv
    sheet = Image.new("RGB", (1024, 5 * 548), "white")
    draw = ImageDraw.Draw(sheet)
    records = []
    for index, task in enumerate(registry["tasks"]):
        # The bounded integration fixture uses one preassigned training demo per task.
        # It is not a full 40-demo-per-task training set and never opens eval demos.
        demo = task["episodes"]["train"][0]
        folder = output / task["task_id"] / demo
        rgb_dir = folder / "rgb"
        rgb_dir.mkdir(parents=True, exist_ok=True)
        env = OffScreenRenderEnv(bddl_file_name=str(project / "libero/libero/bddl_files" / task["bddl"]),
                                 camera_heights=512, camera_widths=512, camera_names=["agentview"])
        try:
            env.reset()
            with h5py.File(root / task["source_path"], "r") as source:
                item = source["data"][demo]
                count = 1 if contacts_only else len(item["actions"])
                robots, actions = [], np.asarray(item["actions"], dtype=np.float32)
                if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
                    raise ValueError(f"Invalid action representation in {task['task_id']}")
                if np.max(np.abs(actions)) > 1.00001:
                    raise ValueError(f"Out-of-range OSC action in {task['task_id']}")
                digest = hashlib.sha256()
                for t in range(count):
                    observation = env.set_init_state(np.asarray(item["states"][t]))
                    robot = np.concatenate([observation["robot0_joint_pos"], observation["robot0_eef_pos"],
                        quat2axisangle(observation["robot0_eef_quat"]), observation["robot0_gripper_qpos"]]).astype(np.float32)
                    if robot.shape != (15,) or not np.isfinite(robot).all():
                        raise ValueError(f"Invalid regenerated robot observation in {task['task_id']}")
                    robots.append(robot)
                    env.sim.forward()
                    rgb = np.asarray(env.sim.render(camera_name="agentview", width=512, height=512))
                    path = rgb_dir / f"{t:06}.jpg"
                    Image.fromarray(rgb).save(path, quality=98)
                    digest.update(path.read_bytes())
                    if t == 0:
                        sheet.paste(Image.fromarray(rgb), ((index % 2) * 512, (index // 2) * 548 + 36))
                        draw.text(((index % 2) * 512 + 5, (index // 2) * 548 + 8),
                                  f"{task['task_id']} / {demo}", fill="black")
                if not contacts_only:
                    np.savez_compressed(folder / "state_actions.npz", robot=np.stack(robots), actions=actions,
                                        timestamps=np.arange(len(actions), dtype=np.float64) / 20)
                records.append({"task_id": task["task_id"], "instruction": task["instruction"], "episode": demo,
                    "split": "train", "frames": count, "source_frames": len(actions),
                    "source_sha256": task["source_sha256"], "rgb_sequence_sha256": digest.hexdigest(),
                    "sam_initialization": task["sam_initialization"], "bddl": task["bddl"]})
                print(json.dumps({"task": task["task_id"], "demo": demo, "frames": count}), flush=True)
        finally:
            env.close()
        volume.commit()
    if contacts_only:
        sheet.save(output / "first_frames.png")
    write_json(output / ("contacts.json" if contacts_only else "renders.json"),
               {"status": "complete", "contacts_only": contacts_only, "entries": records,
                "evaluation_episodes_opened": False,
                "alignment": "robot observation and RGB regenerated from states[t], action[t] retained"})
    volume.commit()
    return {"tasks": len(records), "contacts_only": contacts_only, "output": str(output)}


@app.local_entrypoint()
def main(contacts_only: bool = True, background: bool = False, stage: str = "render"):
    if stage == "preview":
        print(preview_train.remote())
        return
    if stage != "render":
        raise ValueError("Use render or preview")
    if background:
        print(f"Spawned render: {render.spawn(contacts_only).object_id}")
    else:
        print(render.remote(contacts_only))
