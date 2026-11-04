"""Inspect original recorded versus live initial robot observations without actions."""
import modal
from scripts.modal_muse_motor_rollout import runtime_base

volume=modal.Volume.from_name("object-dynamics-vla-data",create_if_missing=False)
image=runtime_base.add_local_python_source("odvla")
app=modal.App("odvla-candidate-input-alignment")


@app.function(image=image,cpu=4,memory=8192,volumes={"/data":volume},timeout=240,max_containers=1)
def inspect(name: str="muse_candidate_robot_alignment_v1"):
    import hashlib
    import json
    from pathlib import Path
    import h5py
    import numpy as np
    from robosuite.utils.transform_utils import quat2axisangle
    from odvla.data import load_episode,read_json,validate_splits
    if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name): raise ValueError("Invalid name")
    dest=Path("/data/eval")/name; dest.mkdir(parents=True,exist_ok=False)
    report={"status":"started","actions_executed":0}; env=None
    try:
        root=Path("/data/libero/milk_sam_development_v1"); splits=read_json(root/"splits.json"); manifest=validate_splits(root,splits)
        entry=next(e for e in manifest["episodes"] if e["id"]=="episode_0000")
        ep=load_episode(root,entry,manifest,include_images=False)
        libero=Path("/opt/LIBERO"); config=Path.home()/".libero/config.yaml"; config.parent.mkdir(parents=True,exist_ok=True)
        config.write_text(f"benchmark_root: {libero}/libero/libero\nbddl_files: {libero}/libero/libero/bddl_files\n"
            f"init_states: {libero}/libero/libero/init_files\ndatasets: {libero}/libero/datasets\nassets: {libero}/libero/libero/assets\n")
        from libero.libero.envs import OffScreenRenderEnv
        with h5py.File("/data/libero/source/milk_demo.hdf5","r") as f:
            demo=f[f'data/{entry["source_demo"]}']; initial=np.asarray(demo["states"][0]).copy()
            recorded=np.concatenate([np.asarray(demo["obs/joint_states"][0]),np.asarray(demo["obs/ee_states"][0]),
                np.asarray(demo["obs/gripper_states"][0])]).astype(np.float32)
            source_shapes={k:list(demo["obs/"+k].shape) for k in ("joint_states","ee_states","gripper_states")}
        np.random.seed(2468)
        env=OffScreenRenderEnv(bddl_file_name=str(libero/"libero/libero/bddl_files/libero_object/pick_up_the_milk_and_place_it_in_the_basket.bddl"),
            camera_heights=512,camera_widths=512,camera_names=["agentview"])
        env.reset(); obs=env.set_init_state(initial.copy())
        def robot(o): return np.concatenate([o["robot0_joint_pos"],o["robot0_eef_pos"],
            quat2axisangle(o["robot0_eef_quat"]),o["robot0_gripper_qpos"]]).astype(np.float32)
        live=robot(obs)
        env.sim.forward(); env._update_observables(force=True); refreshed=robot(env.env._get_observations())
        names=[f"joint_{i}" for i in range(7)]+["eef_x","eef_y","eef_z","eef_axis_x","eef_axis_y","eef_axis_z","gripper_0","gripper_1"]
        stats=np.load("/data/eval/quantized_muse_exposure_v1/stats.npz",allow_pickle=False)
        std=stats["robot_std"]
        dif=live-ep["robot"][0]
        report.update(status="completed",initial_state_sha256=hashlib.sha256(initial.tobytes()).hexdigest(),
            recorded_source_equals_dataset=bool(np.array_equal(recorded,ep["robot"][0])),
            live_equals_refreshed=bool(np.array_equal(live,refreshed)), source_shapes=source_shapes,
            channels=[{"name":n,"training":float(ep["robot"][0,i]),"source_hdf5":float(recorded[i]),"live":float(live[i]),
                "live_minus_training":float(dif[i]),"train_std":float(std[i]),"normalized_difference":float(dif[i]/std[i])}
                for i,n in enumerate(names)],
            closest_recorded_frames=[{"frame":i,"raw_mae":float(np.abs(live-ep["robot"][i]).mean()),
                "raw_max":float(np.abs(live-ep["robot"][i]).max())} for i in range(5)])
        return report
    except Exception as exc:
        report.update(status="failed",error_type=type(exc).__name__,error=str(exc)); raise
    finally:
        if env is not None: env.close()
        (dest/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n"); volume.commit()


@app.local_entrypoint()
def main(name: str="muse_candidate_robot_alignment_v1", stage: str="initial"):
    if stage == "initial": print(inspect.remote(name))
    elif stage == "sweep": print(sweep.remote(name))
    elif stage == "prepare": print(prepare_aligned.remote(name))
    elif stage == "build": print(build_aligned_dataset.remote(name))
    elif stage == "analyze": print(analyze_rollout.remote(name))
    else: raise ValueError("Unknown stage")


@app.function(image=image,cpu=4,memory=8192,volumes={"/data":volume},timeout=600,max_containers=1)
def sweep(name: str="muse_candidate_robot_sweep_v1"):
    """Compare saved simulator states to recorded observations, without stepping actions."""
    import json
    from pathlib import Path
    import h5py
    import numpy as np
    from robosuite.utils.transform_utils import quat2axisangle
    from odvla.data import load_episode,read_json,validate_splits
    if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name): raise ValueError("Invalid name")
    dest=Path("/data/eval")/name; dest.mkdir(parents=True,exist_ok=False); env=None
    report={"status":"started","actions_executed":0}
    try:
        root=Path("/data/libero/milk_sam_development_v1"); splits=read_json(root/"splits.json"); manifest=validate_splits(root,splits)
        entry=next(e for e in manifest["episodes"] if e["id"]=="episode_0000"); ep=load_episode(root,entry,manifest,include_images=False)
        libero=Path("/opt/LIBERO"); config=Path.home()/".libero/config.yaml"; config.parent.mkdir(parents=True,exist_ok=True)
        config.write_text(f"benchmark_root: {libero}/libero/libero\nbddl_files: {libero}/libero/libero/bddl_files\n"
            f"init_states: {libero}/libero/libero/init_files\ndatasets: {libero}/libero/datasets\nassets: {libero}/libero/libero/assets\n")
        from libero.libero.envs import OffScreenRenderEnv
        with h5py.File("/data/libero/source/milk_demo.hdf5","r") as f:
            demo=f[f'data/{entry["source_demo"]}']; states=np.asarray(demo["states"]).copy()
        env=OffScreenRenderEnv(bddl_file_name=str(libero/"libero/libero/bddl_files/libero_object/pick_up_the_milk_and_place_it_in_the_basket.bddl"),
            camera_heights=128,camera_widths=128,camera_names=["agentview"])
        env.reset(); live=[]
        for state in states:
            o=env.set_init_state(state)
            live.append(np.concatenate([o["robot0_joint_pos"],o["robot0_eef_pos"],quat2axisangle(o["robot0_eef_quat"]),
                o["robot0_gripper_qpos"]]).astype(np.float32))
        live=np.stack(live); recorded=ep["robot"]
        stats=np.load("/data/eval/quantized_muse_exposure_v1/stats.npz",allow_pickle=False); std=stats["robot_std"]
        np.savez(dest/"robot_vectors.npz",live=live,recorded=recorded,robot_std=std)
        offsets=[]
        for offset in range(-3,4):
            i=np.arange(max(0,-offset),min(len(live),len(recorded)-offset)); delta=live[i]-recorded[i+offset]
            offsets.append({"recorded_index_minus_state_index":offset,"pairs":len(i),
                "mean_absolute_error_by_channel":np.abs(delta).mean(0).tolist(),
                "normalized_mean_absolute_error_by_channel":(np.abs(delta)/std).mean(0).tolist(),
                "joint_mae":float(np.abs(delta[:,:7]).mean()),"eef_mae":float(np.abs(delta[:,7:13]).mean()),
                "gripper_mae":float(np.abs(delta[:,13:]).mean())})
        report.update(status="completed",frames=len(live),offsets=offsets,
            note="Each observation was regenerated from the saved simulator state without executing actions.")
        return report
    except Exception as exc:
        report.update(status="failed",error_type=type(exc).__name__,error=str(exc)); raise
    finally:
        if env is not None: env.close()
        (dest/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n"); volume.commit()


@app.function(image=image,cpu=2,memory=8192,volumes={"/data":volume},timeout=180,max_containers=1)
def prepare_aligned(name: str="muse_candidate_aligned_episode_v1"):
    """Create a separate episode with robot observations regenerated from states[t]."""
    import hashlib
    import json
    from pathlib import Path
    import numpy as np
    from odvla.data import file_hash,load_episode,read_json,validate_splits
    if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name): raise ValueError("Invalid name")
    out=Path("/data/libero")/name; out.mkdir(parents=True,exist_ok=False)
    root=Path("/data/libero/milk_sam_development_v1"); splits=read_json(root/"splits.json"); manifest=validate_splits(root,splits)
    entry=next(e for e in manifest["episodes"] if e["id"]=="episode_0000")
    ep=load_episode(root,entry,manifest,include_images=True)
    sweep=Path("/data/eval/muse_candidate_robot_sweep_v1")
    if read_json(sweep/"report.json")["status"]!="completed": raise ValueError("Robot sweep is incomplete")
    with np.load(sweep/"robot_vectors.npz",allow_pickle=False) as z:
        if not np.array_equal(z["recorded"],ep["robot"]): raise ValueError("Sweep inputs differ from current dataset")
        aligned=z["live"].copy()
    if aligned.shape!=ep["robot"].shape or not np.isfinite(aligned).all(): raise ValueError("Invalid regenerated robot observations")
    old=ep["robot"].copy(); ep["robot"]=aligned
    np.savez_compressed(out/"episode_0000.npz",**ep)
    report={"status":"prepared_not_trained","episode":"episode_0000","frames":len(aligned),
        "source_dataset_fingerprint":splits["fingerprint"],"source_episode_sha256":file_hash(root/entry["file"]),
        "robot_sweep_sha256":file_hash(sweep/"robot_vectors.npz"),"output_sha256":file_hash(out/"episode_0000.npz"),
        "initial_robot_sha256":hashlib.sha256(aligned[0].tobytes()).hexdigest(),
        "original_robot_sha256":hashlib.sha256(old[0].tobytes()).hexdigest(),
        "alignment":"robot observation regenerated from simulator states[t]; original action[t], rendered RGB[t], and cached SAM[t] retained",
        "scope":"single training episode diagnostic artifact, not a complete split or new fitted normalization",
        "note":"existing 7500 checkpoint was trained with original robot observations; this artifact does not repair its training history"}
    (out/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n"); volume.commit()
    return report


@app.function(image=image,cpu=8,memory=32768,volumes={"/data":volume},timeout=3600,max_containers=1,single_use_containers=True)
def build_aligned_dataset(name: str="muse_aligned_development_v1", source: str="milk_sam_development_v1"):
    """Rebuild every episode's robot observation from its source simulator state.

    RGB and actions already come from states[t]; the imported robot vector came
    from HDF5 obs[t], which the timing sweep showed equals obs(states[t+1]).
    """
    import hashlib
    import json
    import time
    from pathlib import Path
    import h5py
    import numpy as np
    from robosuite.utils.transform_utils import quat2axisangle
    from odvla.data import (file_hash,fingerprint,load_episode,load_manifest,
        read_json,validate_splits,write_json)
    for value in (name,source):
        if not value or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in value):
            raise ValueError("Invalid name")
    started=time.monotonic(); env=None
    src=Path("/data/libero")/source; dest=Path("/data/libero")/name
    if dest.exists():
        raise FileExistsError(dest)
    src_splits=read_json(src/"splits.json")
    if src_splits["fingerprint"]!="b7bae449415728262307c262677def270cd50380406183d2455540351a1c72c4":
        raise ValueError("Unexpected source dataset fingerprint")
    manifest=validate_splits(src,src_splits)
    dest.mkdir(parents=True)
    (dest/"manifest.json").write_text((src/"manifest.json").read_text(encoding="utf-8"))
    episodes=dest/"episodes"; episodes.mkdir()
    libero=Path("/opt/LIBERO"); config=Path.home()/".libero/config.yaml"; config.parent.mkdir(parents=True,exist_ok=True)
    config.write_text(f"benchmark_root: {libero}/libero/libero\nbddl_files: {libero}/libero/libero/bddl_files\n"
        f"init_states: {libero}/libero/libero/init_files\ndatasets: {libero}/libero/datasets\nassets: {libero}/libero/libero/assets\n")
    from libero.libero.envs import OffScreenRenderEnv
    env=OffScreenRenderEnv(bddl_file_name=str(libero/"libero/libero/bddl_files/libero_object/pick_up_the_milk_and_place_it_in_the_basket.bddl"),
        camera_heights=128,camera_widths=128,camera_names=["agentview"])
    env.reset()
    def robot(o):
        return np.concatenate([o["robot0_joint_pos"],o["robot0_eef_pos"],
            quat2axisangle(o["robot0_eef_quat"]),o["robot0_gripper_qpos"]]).astype(np.float32)
    summary=[]; expected_initial=None
    with h5py.File("/data/libero/source/milk_demo.hdf5","r") as f:
        for entry in manifest["episodes"]:
            ep=load_episode(src,entry,manifest,include_images=True)
            frames=len(ep["timestamps"])
            states=np.asarray(f[f'data/{entry["source_demo"]}/states']).copy()
            if len(states)<frames:
                raise ValueError(f'{entry["id"]}: {len(states)} source states < {frames} observations')
            aligned=np.stack([robot(env.set_init_state(state)) for state in states[:frames]])
            if not np.isfinite(aligned).all() or aligned.shape!=ep["robot"].shape:
                raise ValueError(f'{entry["id"]}: invalid regenerated robot observations')
            old=ep["robot"].copy(); ep["robot"]=aligned
            np.savez_compressed(episodes/f"{entry['id']}.npz",**ep)
            if entry["id"]=="episode_0000":
                expected_initial=hashlib.sha256(aligned[0].tobytes()).hexdigest()
                with np.load("/data/eval/muse_candidate_robot_sweep_v1/robot_vectors.npz",allow_pickle=False) as z:
                    if not np.array_equal(z["live"],aligned):
                        raise ValueError("Rebuilt episode_0000 robot differs from the audited sweep")
            delta=aligned-old
            summary.append({"episode":entry["id"],"frames":frames,"source_states":int(len(states)),
                "mean_abs_change":float(np.abs(delta).mean()),"max_abs_change":float(np.abs(delta).max())})
            if len(summary)%10==0:
                print(json.dumps({"episodes":len(summary),"seconds":time.monotonic()-started}),flush=True)
    env.close(); env=None
    new_splits=dict(src_splits)
    new_splits["fingerprint"]=fingerprint(dest)
    new_splits["aligned_from"]=src_splits["fingerprint"]
    new_splits["alignment"]="robot observation regenerated from source simulator states[t]; RGB, SAM features, actions and timestamps unchanged"
    write_json(dest/"splits.json",new_splits)
    validate_splits(dest,new_splits)
    report={"status":"completed","dataset":name,"episodes":len(summary),"source_fingerprint":src_splits["fingerprint"],
        "output_fingerprint":new_splits["fingerprint"],"initial_robot_sha256":expected_initial,
        "global_mean_abs_change":float(np.mean([s["mean_abs_change"] for s in summary])),
        "global_max_abs_change":float(np.max([s["max_abs_change"] for s in summary])),
        "actions_executed":0,"seconds":time.monotonic()-started,"per_episode":summary,
        "scope":"complete aligned copy of the development dataset; test episodes remain unaccessed",
        "note":"existing checkpoints were trained on the original robot observations and must not be reused as if aligned"}
    (dest/"alignment_report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    volume.commit()
    return {k:v for k,v in report.items() if k!="per_episode"}


@app.function(image=image,cpu=4,memory=8192,volumes={"/data":volume},timeout=600,max_containers=1)
def analyze_rollout(name: str="muse_aligned7500_episode0000_v1"):
    """Read saved masks/actions and locate the earliest pick-and-place break."""
    import json
    import time
    from pathlib import Path
    import numpy as np
    from odvla.data import load_episode, read_json, validate_splits
    from odvla.measurements import mask_metrics
    if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
        raise ValueError("Invalid name")
    run=Path("/data/eval")/name; report=read_json(run/"report.json"); obs_dir=run/"observations"
    root=Path("/data/libero/muse_aligned_development_v1"); splits=read_json(root/"splits.json")
    manifest=validate_splits(root,splits); entry=next(e for e in manifest["episodes"] if e["id"]=="episode_0000")
    ep=load_episode(root,entry,manifest,include_images=False)
    live=[]; grip=[]
    for step in range(report["steps"]):
        with np.load(obs_dir/f"{step:06}.npz",allow_pickle=False) as z:
            masks=z["sam_masks"]; metrics,valid=mask_metrics(masks[None])
            live.append({"step":step,"metrics":metrics[0].tolist(),"valid":valid[0].tolist(),
                "sampled_action":z["sampled_action"].tolist(),"executed_action":z["executed_action"].tolist(),
                "raw_robot":z["raw_robot"].tolist(),"gripper_qpos":z["raw_robot"][13:15].tolist(),
                "eef_z":float(z["raw_robot"][9])})
    def dist(a,b): return float(np.linalg.norm((np.asarray(a)-np.asarray(b))[:2]))
    recorded=ep["features"]; valid=ep["valid"]
    rows=[]
    for item in live:
        t=item["step"]; ref_t=min(t,len(recorded)-1)
        raw=np.asarray(item["raw_robot"]); ref_robot=ep["robot"][ref_t]
        action_ref=ep["actions"][t] if t<len(ep["actions"]) else None
        rows.append({"step":t,"recorded_reference_step":ref_t,"live_milk":item["metrics"][0],"recorded_milk":recorded[ref_t,0].tolist(),
            "milk_centroid_distance":dist(item["metrics"][0],recorded[ref_t,0]),
            "live_area":float(item["metrics"][0][4]),"recorded_area":float(recorded[ref_t,0][4]),
            "gripper_qpos_width":float(item["gripper_qpos"][0]-item["gripper_qpos"][1]),
            "eef_z":item["eef_z"],"sampled_gripper":float(item["sampled_action"][6]),
            "eef_position_error":float(np.linalg.norm(raw[7:10]-ref_robot[7:10])),
            "joint_mae":float(np.abs(raw[:7]-ref_robot[:7]).mean()),
            "gripper_command_agreement":(None if action_ref is None else bool(np.sign(item["sampled_action"][6])==np.sign(action_ref[6]))),
            "recorded_gripper":(None if action_ref is None else float(action_ref[6]))})
    def first(predicate):
        return next((r["step"] for r in rows if predicate(r)),None)
    lifted=first(lambda r:r["live_area"]<0.35*r["recorded_area"] and r["step"]>40)
    centroid_break=first(lambda r:r["step"]>40 and r["milk_centroid_distance"]>0.08)
    gripper_break=first(lambda r:r["step"]>40 and r["sampled_gripper"]<0)
    live_v0=rows[0]["live_milk"][1]; rec_v0=rows[0]["recorded_milk"][1]
    summary={"status":"completed","rollout":name,"steps":len(rows),
        "live_milk_centroid_v_upward_shift_after40":float(live_v0-min(r["live_milk"][1] for r in rows if r["step"]>40)),
        "recorded_milk_centroid_v_upward_shift_after40":float(rec_v0-min(r["recorded_milk"][1] for r in rows if r["step"]>40)),
        "milk_centroid_distance_max_after40":max((r["milk_centroid_distance"] for r in rows if r["step"]>40),default=None),
        "milk_visible_area_ratio_min_after40":min((r["live_area"]/r["recorded_area"] for r in rows if r["step"]>40 and r["recorded_area"]>0),default=None),
        "eef_z_min_after40":min((r["eef_z"] for r in rows if r["step"]>40),default=None),
        "eef_z_max_after40":max((r["eef_z"] for r in rows if r["step"]>40),default=None),
        "gripper_width_min":min(r["gripper_qpos_width"] for r in rows),"gripper_width_max":max(r["gripper_qpos_width"] for r in rows),
        "first_step_over_40_with_open_command":gripper_break,
        "first_step_over_40_with_milk_centroid_shift_gt_0.08":centroid_break,
        "first_step_over_40_with_milk_area_below_35pct":lifted,
        "first_step_over10_with_eef_error_gt_3cm":first(lambda r:r["step"]>10 and r["eef_position_error"]>0.03),
        "first_step_over10_with_joint_mae_gt_0.15rad":first(lambda r:r["step"]>10 and r["joint_mae"]>0.15),
        "first_step_with_three_consecutive_gripper_disagreements":next((rows[i]["step"] for i in range(len(rows)-2)
            if all(rows[j]["gripper_command_agreement"] is False for j in (i,i+1,i+2))),None),
        "gripper_command_agreement_0_149":float(np.mean([r["gripper_command_agreement"] for r in rows[:150] if r["gripper_command_agreement"] is not None])),
        "gripper_command_agreement_47_141":float(np.mean([r["gripper_command_agreement"] for r in rows[47:142] if r["gripper_command_agreement"] is not None])),
        "eef_position_error_at_47":rows[47]["eef_position_error"],"eef_position_error_at_90":rows[90]["eef_position_error"],
        "recorded_grasp_close_step":next((t for t in range(1,len(ep["actions"])) if ep["actions"][t,6]>0 and ep["actions"][t-1,6]<=0),None),
        "recorded_grasp_open_step":next((t for t in range(1,len(ep["actions"])) if ep["actions"][t,6]<0 and ep["actions"][t-1,6]>=0),None),
        "rows":rows,
        "note":"Live and recorded measurements are compared by step index, not by matched object state; masks are 128-pixel SAM outputs."}
    (run/"mask_trajectory.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n"); volume.commit()
    return {k:v for k,v in summary.items() if k!="rows"}
