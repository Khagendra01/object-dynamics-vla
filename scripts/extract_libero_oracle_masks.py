"""Render exact simulator instance masks for recorded LIBERO states.

This is an oracle perception baseline: it reads privileged simulator states from
the demonstration file to place the scene, then renders the current frame and its
instance segmentation. It is causal per frame (no future state is read), but the
policy using these masks is not camera-only. Always label the provenance.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import h5py
import mujoco
import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--bddl", required=True)
    p.add_argument("--bodies", required=True, help="Comma-separated simulator body prefixes in slot order")
    p.add_argument("--episode", help="Render one canonical episode only")
    p.add_argument("--libero-commit", default="")
    p.add_argument("--max-rgb-diff", type=float, default=20.0)
    a = p.parse_args()

    libero_root = Path(os.environ.get("LIBERO_ROOT", "/opt/LIBERO"))
    cfgdir = Path.home() / ".libero"
    cfgdir.mkdir(parents=True, exist_ok=True)
    (cfgdir / "config.yaml").write_text(
        f"benchmark_root: {libero_root}/libero/libero\n"
        f"bddl_files: {libero_root}/libero/libero/bddl_files\n"
        f"init_states: {libero_root}/libero/libero/init_files\n"
        f"datasets: {libero_root}/libero/datasets\n"
        f"assets: {libero_root}/libero/libero/assets\n")
    from libero.libero.envs import OffScreenRenderEnv

    dataset, out = Path(a.dataset), Path(a.out)
    manifest = json.load(open(dataset / "manifest.json"))
    bodies = [x.strip() for x in a.bodies.split(",") if x.strip()]
    if len(bodies) != len(manifest["object_slots"]):
        raise ValueError("--bodies must match manifest object_slots order and length")
    selected = [e for e in manifest["episodes"]
                if a.episode is None or e["id"] == a.episode]
    if not selected:
        raise ValueError("Unknown episode ID")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use an empty mask output directory")
    out.mkdir(parents=True, exist_ok=True)

    env = OffScreenRenderEnv(bddl_file_name=a.bddl, camera_heights=128,
                             camera_widths=128, camera_names=["agentview"])
    env.reset()
    geom_names = list(env.sim.model.geom_names)
    target_ids = {}
    for body in bodies:
        ids = [i for i, name in enumerate(geom_names) if name.startswith(body + "_")]
        if not ids:
            raise ValueError(f"No geoms found for body prefix {body!r}")
        target_ids[body] = ids

    counts = {"observed": 0, "missing": 0, "ambiguous": 0}
    diffs = []
    with h5py.File(a.source, "r") as f:
        for entry in selected:
            d = f["data"][entry["source_demo"]]
            states = np.asarray(d["states"])
            rgb = np.asarray(d["obs/agentview_rgb"])
            if states.shape[0] != d["actions"].shape[0]:
                raise ValueError(f"{entry['id']}: states/actions length mismatch")
            if rgb.shape[:3] != (states.shape[0], 128, 128):
                raise ValueError(f"{entry['id']}: unexpected RGB shape {rgb.shape}")
            t = states.shape[0]
            masks = np.zeros((t, len(bodies), 128, 128), bool)
            for ti in range(t):
                env.set_state(states[ti])
                env.sim.forward()
                img = np.asarray(env.sim.render(camera_name="agentview", width=128,
                                                height=128)).astype(np.int16)
                diff = float(np.abs(img - rgb[ti].astype(np.int16)).mean())
                diffs.append(diff)
                if diff > a.max_rgb_diff:
                    raise ValueError(f"{entry['id']} frame {ti}: RGB diff {diff:.2f} exceeds {a.max_rgb_diff}")
                seg = np.asarray(env.sim.render(camera_name="agentview", width=128,
                                                height=128, segmentation=True))
                for oi, body in enumerate(bodies):
                    geom_hit = np.isin(seg[..., 1], target_ids[body]) & (
                        seg[..., 0] == int(mujoco.mjtObj.mjOBJ_GEOM))
                    masks[ti, oi] = geom_hit
                    counts["observed" if geom_hit.any() else "missing"] += 1
            np.savez_compressed(out / (entry["id"] + ".npz"), masks=masks)
            print(json.dumps({"episode": entry["id"], "frames": t,
                              "observed": int(masks.reshape(t, -1).any(1).sum()),
                              "rgb_diff_mean": round(float(np.mean(diffs[-t:])), 3),
                              "rgb_diff_max": round(float(np.max(diffs[-t:])), 3)}),
                  flush=True)

    provenance = {
        "source": "LIBERO/MuJoCo instance segmentation from recorded simulator states",
        "causal": True,
        "identity_assumption": "Exact simulator geom names in slot order: " + ", ".join(bodies),
        "uses_video_tracking": False,
        "checkpoint_sha256": None,
        "package_version": "libero commit " + (a.libero_commit or "unknown"),
        "complete_dataset": a.episode is None,
        "counts": counts,
        "oracle": True,
        "oracle_inputs": "recorded privileged simulator states; not camera-only",
        "rgb_alignment_mean_abs_diff": float(np.mean(diffs)),
        "rgb_alignment_max_abs_diff": float(np.max(diffs)),
        "max_rgb_diff_threshold": a.max_rgb_diff,
        "source_sha256": sha256(a.source),
        "bddl_sha256": sha256(a.bddl),
        "model_execution_status": "rendered on Modal CPU with MuJoCo OSMesa",
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
