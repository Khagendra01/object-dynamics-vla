"""Paired SAM-input policy training with common validation checkpoint rule."""
import modal
from scripts.modal_real_ablations import image as base_image, volume

image = (base_image.add_local_file("scripts/__init__.py", "/root/scripts/__init__.py")
         .add_local_file("scripts/modal_real_ablations.py", "/root/scripts/modal_real_ablations.py"))
app = modal.App("odvla-milk-sam-policy")


@app.function(image=image, gpu="T4", cpu=4, memory=16384, volumes={"/data": volume}, timeout=7200)
def run(epochs: int = 25):
    from pathlib import Path
    from odvla.experiment import train, evaluate
    from odvla.data import read_json, write_json
    ds = Path("/data/libero/milk_sam_development_v1")
    out = Path("/data/runs/milk_sam_matched_policy_v1")
    if out.exists():
        raise FileExistsError("Use fresh output directory")
    out.mkdir(parents=True)
    rows = []
    for seed in (7, 19, 31):
        for label, weight in (("action_only", 0.0), ("joint", 1.0)):
            path = out / f"{label}_seed{seed}"
            train(ds, ds / "splits.json", path, epochs=epochs, seed=seed, device="cuda",
                  dynamics_weight=weight, allow_experimental_perception=True, checkpoint_selection="flow")
            metrics = evaluate(ds, path, "val", "cuda")
            best = read_json(path / "best_validation.json")
            rows.append({"seed": seed, "variant": label, "best_epoch": best["epoch"],
                "val_flow_mse": metrics["flow_mse"],
                "sampled_action_normalized_mse": metrics["sampled_action_normalized_mse"],
                "sampled_action_mae_source_units": metrics["sampled_action_mae_source_units"],
                "val_dynamics_normalized_mse": metrics["dynamics"]["normalized_mse"]})
    result = {"dataset_fingerprint": read_json(ds / "splits.json")["fingerprint"], "rows": rows,
        "checkpoint_selection": "minimum validation flow MSE for every variant",
        "action_sampling": "same fixed seed 921 noise schedule, 20 flow steps",
        "claim": "offline validation only; no closed-loop SAM rollout performed",
        "test_accessed": False}
    write_json(out / "comparison.json", result)
    volume.commit()
    return result


@app.local_entrypoint()
def main(epochs: int = 25):
    print(run.remote(epochs))
