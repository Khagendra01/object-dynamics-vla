"""Closed-loop rollout of frozen milk-task baselines in LIBERO.

Object-input variants use live simulator-oracle masks, not camera-only perception.
Initial simulator states come from held-out demonstrations.
"""
import modal
from scripts.modal_libero_oracle import image as libero_image, volume

image = (libero_image.add_local_python_source("odvla")
         .add_local_file("scripts/__init__.py", "/root/scripts/__init__.py")
         .add_local_file("scripts/modal_libero_oracle.py", "/root/scripts/modal_libero_oracle.py"))
app = modal.App("odvla-libero-policy-rollout")


@app.function(image=image, volumes={"/data": volume}, cpu=4, memory=16384, timeout=3600)
def rollout(max_steps: int = 200, variant: str = "no_objects", train_seed: int = 7,
            split: str = "test", sample_seed: int = 7, monitor_dir: str = ""):
    import json
    from pathlib import Path

    import h5py
    import mujoco
    import numpy as np
    import torch
    from robosuite.utils.transform_utils import quat2axisangle
    from odvla.data import load_manifest, read_json, encode_history
    from odvla.experiment import load_run
    from odvla.inference import predict
    from odvla.measurements import mask_metrics
    from odvla.diagnostics import one_step_forecast, summarize_actions
    from odvla.one_step import load_one_step

    root = Path("/opt/LIBERO")
    config = Path.home() / ".libero" / "config.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(f"benchmark_root: {root}/libero/libero\nbddl_files: {root}/libero/libero/bddl_files\n"
                      f"init_states: {root}/libero/libero/init_files\ndatasets: {root}/libero/datasets\n"
                      f"assets: {root}/libero/libero/assets\n")
    from libero.libero.envs import OffScreenRenderEnv

    task = root / "libero/libero/bddl_files/libero_object/pick_up_the_milk_and_place_it_in_the_basket.bddl"
    allowed = {"no_objects", "objects_action_only", "objects_joint", "objects_joint_no_dynamics_actions"}
    if variant not in allowed:
        raise ValueError(f"variant must be one of {sorted(allowed)}")
    run = Path("/data/runs/libero_milk_oracle_ablations") / f"{variant}_seed{train_seed}"
    model, policy_config, stats = load_run(run, "cpu")
    if policy_config["model"]["context_dim"]:
        raise ValueError("This adapter requires a numeric-only checkpoint")
    model.eval()
    monitor = load_one_step(monitor_dir, run, policy_config) if monitor_dir else None
    torch.set_num_threads(2)
    manifest = load_manifest("/data/libero/milk_oracle")
    if split not in ("val", "test"):
        raise ValueError("split must be val or test")
    split_ids = set(read_json(run / "splits.json")[split])
    entries = [entry for entry in manifest["episodes"] if entry["id"] in split_ids]
    results = []
    all_sampled, all_executed, all_demos = [], [], []
    with h5py.File("/data/libero/source/milk_demo.hdf5", "r") as source:
        for entry in entries:
            env = OffScreenRenderEnv(bddl_file_name=str(task), camera_heights=128,
                                     camera_widths=128, camera_names=["agentview"])
            env.reset()
            demo = source["data"][entry["source_demo"]]
            all_demos.append(np.asarray(demo["actions"], dtype=np.float64))
            obs = env.set_init_state(np.asarray(demo["states"][0]))
            geom_names = list(env.sim.model.geom_names)
            target_ids = [[i for i, name in enumerate(geom_names) if name.startswith(body + "_")]
                          for body in ("milk_1", "basket_1")]
            if any(not ids for ids in target_ids):
                raise ValueError("Missing simulator geoms for an object slot")
            def measurements():
                if policy_config["no_objects"]:
                    return np.zeros((2, 5), np.float32), np.zeros((2, 5), bool)
                seg = np.asarray(env.sim.render(camera_name="agentview", width=128, height=128,
                                                segmentation=True))
                masks = np.stack([np.isin(seg[..., 1], ids) &
                                  (seg[..., 0] == int(mujoco.mjtObj.mjOBJ_GEOM)) for ids in target_ids])
                features, valid = mask_metrics(masks[None])
                return features[0], valid[0]
            def robot_vector(observation):
                return np.concatenate([observation["robot0_joint_pos"], observation["robot0_eef_pos"],
                                       quat2axisangle(observation["robot0_eef_quat"]),
                                       observation["robot0_gripper_qpos"]]).astype(np.float32)
            history = policy_config["history"]
            robots = [robot_vector(obs)] * history
            recorded_robot = np.concatenate([np.asarray(demo["obs/joint_states"][:history]),
                                             np.asarray(demo["obs/ee_states"][:history]),
                                             np.asarray(demo["obs/gripper_states"][:history])], axis=1)
            bootstrap_gap = np.abs(recorded_robot - robots[0]).mean(axis=0)
            initial_features, initial_valid = measurements()
            features = [initial_features] * history
            valids = [initial_valid] * history
            clipped_steps = 0
            success = False
            traces = []
            for step in range(max_steps):
                n = len(robots)
                prefix = {"robot": np.stack(robots), "features": np.stack(features),
                          "valid": np.stack(valids), "context": np.zeros((n, 0), np.float32),
                          "timestamps": np.arange(n, dtype=np.float64) / 20.0}
                output = predict(prefix, model, policy_config, stats, device="cpu", seed=sample_seed + step)
                chunk = np.asarray(output["actions_source_units"], dtype=np.float64)
                action = chunk[0]
                integration_difference = None
                if step % 25 == 0:
                    x = encode_history(prefix, n - 1, history, stats, policy_config["no_objects"])
                    rng = np.random.default_rng(sample_seed + step)
                    noise = torch.from_numpy(rng.normal(size=(1, policy_config["model"]["action_dim"])).astype(np.float32))
                    with torch.no_grad():
                        h, _ = model.encode(torch.from_numpy(x[None]))
                        refined = model.sample_actions(h, noise, policy_config["flow_steps"] * 2)
                    refined = refined.numpy().reshape(policy_config["horizon"], -1)
                    refined = refined * stats["actions_std"] + stats["actions_mean"]
                    integration_difference = (refined[0] - action).tolist()
                if not np.isfinite(action).all():
                    raise ValueError(f"{entry['id']} step {step}: nonfinite action")
                bounded = np.clip(action, -1.0, 1.0)
                forecast = None
                if monitor is not None:
                    forecast = one_step_forecast(prefix, bounded, model, monitor, policy_config, stats)
                clipped_steps += int(not np.array_equal(action, bounded))
                before_features, before_valid = features[-1], valids[-1]
                obs, _, done, _ = env.step(bounded)
                robots.append(robot_vector(obs))
                new_features, new_valid = measurements()
                features.append(new_features)
                valids.append(new_valid)
                all_sampled.append(action.copy())
                all_executed.append(bounded.copy())
                traces.append({"sampled": action.tolist(), "executed": bounded.tolist(),
                    "integration_40_minus_20_first_action": integration_difference,
                    "predicted_delta": forecast.tolist() if forecast is not None else None,
                    "observed_delta": (new_features - before_features).tolist(),
                    "valid": (before_valid & new_valid).tolist()})
                success = bool(env.check_success())
                if success or done:
                    break
            results.append({"episode": entry["id"], "source_demo": entry["source_demo"],
                            "success": success, "steps": step + 1, "clipped_steps": clipped_steps,
                            "bootstrap_vs_recorded_first_history_mean_abs_difference": bootstrap_gap.tolist(),
                            "trace": traces})
            print(json.dumps({k: v for k, v in results[-1].items() if k != "trace"}), flush=True)
            env.close()
    sampled, executed = np.asarray(all_sampled), np.asarray(all_executed)
    clip_delta = np.abs(sampled - executed)
    changes = np.concatenate([np.diff(np.asarray([t["executed"] for t in r["trace"]]), axis=0)
                              for r in results if len(r["trace"]) > 1], axis=0)
    report = {"task": "milk_in_basket", "variant": f"{variant}_seed{train_seed}", "split": split,
              "sampling_seed": sample_seed,
              "channel_order": ["translation_x", "translation_y", "translation_z", "rotation_x", "rotation_y", "rotation_z", "gripper"],
              "perception": "none" if policy_config["no_objects"] else "privileged simulator-oracle masks",
              "initialization": "recorded held-out demonstration initial simulator states",
              "history_bootstrap": "repeat initial robot observation", "max_steps": max_steps,
              "action": "sample new chunk from current observations and execute first action, clipped to [-1,1]",
              "monitor": ("standalone one-step head receives only current history and accepted first action; no control intervention"
                          if monitor is not None else "disabled; no trained one-step head supplied"),
              "action_statistics": {"demonstrations": summarize_actions(np.concatenate(all_demos)),
                  "sampled": summarize_actions(sampled), "executed": summarize_actions(executed),
                  "consecutive_executed_delta": summarize_actions(changes),
                  "clip_frequency_per_channel": (clip_delta > 0).mean(0).tolist(),
                  "clip_magnitude_mean_per_channel": clip_delta.mean(0).tolist(),
                  "clip_magnitude_max_per_channel": clip_delta.max(0).tolist(),
                  "normalization_mean": stats["actions_mean"].tolist(),
                  "normalization_std": stats["actions_std"].tolist()},
              "successes": sum(row["success"] for row in results), "episodes": len(results),
              "results": results}
    destination = Path(f"/data/eval/milk_{variant}_seed{train_seed}_{split}_sample{sample_seed}_one_step_diagnostics.json")
    destination.write_text(json.dumps(report, indent=2) + "\n")
    volume.commit()
    return {k: v for k, v in report.items() if k != "results"}


@app.local_entrypoint()
def main(max_steps: int = 200, variant: str = "no_objects", train_seed: int = 7,
         split: str = "val", sample_seed: int = 7, monitor_dir: str = ""):
    print(rollout.remote(max_steps, variant, train_seed, split, sample_seed, monitor_dir))
