"""Download and inspect the released robot-trained SmolVLA checkpoint."""
import modal
from scripts.modal_smolvla_reference import image as reference_image

image = (reference_image.add_local_file("scripts/__init__.py", "/root/scripts/__init__.py")
         .add_local_file("scripts/modal_smolvla_reference.py", "/root/scripts/modal_smolvla_reference.py"))
volume = modal.Volume.from_name("object-dynamics-vla-data", create_if_missing=False)
app = modal.App("odvla-smolvla-checkpoint-audit")


@app.function(image=image, volumes={"/data": volume}, cpu=2, memory=8192, timeout=1200, max_containers=1)
def run():
    import json
    from pathlib import Path
    from huggingface_hub import HfApi, snapshot_download
    from safetensors import safe_open
    from odvla.data import file_hash
    model = "lerobot/smolvla_base"
    sha = HfApi().model_info(model).sha
    root = Path("/data/checkpoints/smolvla-base") / sha
    snapshot_download(model, revision=sha, local_dir=str(root), allow_patterns=["*.json", "*.safetensors", "README.md"])
    path = root / "model.safetensors"
    with safe_open(path, framework="pt", device="cpu") as f:
        keys = f.keys()
        relevant = {k: list(f.get_slice(k).get_shape()) for k in keys if
                    "lm_expert.layers.0.self_attn" in k or k.endswith(("action_in_proj.weight", "action_out_proj.weight", "state_proj.weight"))}
    config = json.loads((root / "config.json").read_text())
    result = {"model": model, "revision": sha, "checkpoint_sha256": file_hash(path), "config": config,
              "expert_shapes": relevant, "tensor_count": len(keys), "checkpoint_bytes": path.stat().st_size,
              "overlap": "Robot pretraining datasets and LIBERO source overlap require explicit audit; do not call this independent evaluation.",
              "adaptation": "Checkpoint robot action semantics must be remapped by supervised LIBERO fine-tuning; pretrained joint targets are not OSC_POSE commands."}
    out = Path("/data/eval/smolvla_pretrained_audit_v1.json")
    if out.exists():
        existing = json.loads(out.read_text())
        if existing["revision"] != sha or existing["checkpoint_sha256"] != result["checkpoint_sha256"]:
            raise ValueError("Existing audit does not match downloaded checkpoint")
        return existing
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    volume.commit(); return result


@app.local_entrypoint()
def main(): print(run.remote())
