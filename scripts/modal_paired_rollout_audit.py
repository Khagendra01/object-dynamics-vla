"""Audit matching initial state and actual saved action-noise bytes in paired rollouts."""
import modal

app = modal.App("odvla-paired-rollout-audit")
volume = modal.Volume.from_name("object-dynamics-vla-data", create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.12").pip_install("numpy")


@app.function(image=image, cpu=2, memory=4096, timeout=120, volumes={"/data": volume})
def run():
    import hashlib
    import json
    from pathlib import Path
    import numpy as np
    base = Path("/data/eval")
    paths = {"native": base / "muse_smolvla_live_episode_0014_pairnativev1",
             "repaired": base / "muse_smolvla_live_episode_0014_pairrepairedv1"}
    output = base / "muse_smolvla_repair1250v1_paired_audit_v1.json"
    if output.exists(): raise FileExistsError(output)
    bank_path = base / "muse_smolvla_noise_matched7v1.npy"
    bank = np.load(bank_path, allow_pickle=False)
    if bank.shape != (200, 1, 50, 32) or bank.dtype != np.float32: raise ValueError("Invalid noise bank")
    def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
    bank_hash = digest(bank_path); reports = {k: json.loads((p / "report.json").read_text()) for k, p in paths.items()}
    if len({r["initial_state_sha256"] for r in reports.values()}) != 1: raise ValueError("Initial states differ")
    rgb_hashes = {k: digest(p / "frames/000000.jpg") for k, p in paths.items()}
    if len(set(rgb_hashes.values())) != 1: raise ValueError("Initial images differ")
    for name, report in reports.items():
        if report["policy_mode"] != name or report["noise_bank_sha256"] != bank_hash:
            raise ValueError("Policy mode or bank hash mismatch")
        if len(report["trace"]) != report["steps"]: raise ValueError("Trace length mismatch")
        for i, row in enumerate(report["trace"]):
            expected = hashlib.sha256(bank[i].tobytes()).hexdigest()
            if row["step"] != i or row["noise_sha256"] != expected: raise ValueError(f"Noise mismatch: {name} {i}")
    overlap = min(r["steps"] for r in reports.values())
    summaries = {}
    for name, report in reports.items():
        actions = np.array([r["executed_action"] for r in report["trace"]])
        flags = np.array([r["clipped_channels"] for r in report["trace"]])
        signs = np.sign(actions[:, 6])
        summaries[name] = {"steps": report["steps"], "success": report["success"],
            "clipped_steps": int(flags.any(axis=1).sum()), "clipped_by_channel": flags.sum(axis=0).tolist(),
            "gripper_sign_changes": int((signs[1:] != signs[:-1]).sum()),
            "initial_rgb_sha256": rgb_hashes[name], "seconds": report["seconds"]}
    native = np.array([r["executed_action"] for r in reports["native"]["trace"][:overlap]])
    repaired = np.array([r["executed_action"] for r in reports["repaired"]["trace"][:overlap]])
    result = {"bank_sha256": bank_hash, "matched_noise_steps": overlap, "matched_initial_state": True,
              "matched_initial_rgb": True, "results": summaries,
              "first_executed_action_delta": (repaired[0] - native[0]).tolist(),
              "mean_abs_executed_action_delta_different_observations": np.abs(repaired-native).mean(axis=0).tolist(),
              "interpretation": "same initial state and noise, but subsequent observations differ; one reused development state"}
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n"); volume.commit(); return result


@app.local_entrypoint()
def main():
    print(run.remote())
