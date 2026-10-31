"""One milk adaptation update from the audited two-task motor checkpoint."""
import modal
from scripts.modal_real_ablations import image as base_image

volume = modal.Volume.from_name("object-dynamics-vla-data", create_if_missing=False)
image = (base_image.add_local_file("scripts/__init__.py", "/root/scripts/__init__.py")
         .add_local_file("scripts/modal_real_ablations.py", "/root/scripts/modal_real_ablations.py")
         .add_local_file("scripts/train_muse_vla.py", "/root/train_muse_vla.py"))
app = modal.App("odvla-multitask-transfer-smoke")


@app.function(image=image, gpu="T4", cpu=4, memory=16384, volumes={"/data": volume}, timeout=1200)
def run():
    import sys
    from pathlib import Path
    from odvla.data import read_json, write_json
    sys.path.insert(0, "/root")
    from train_muse_vla import train
    audit = Path("/data/eval/milk_transfer_contract_v1.json")
    contract = read_json("/data/eval/multitask_source_contracts_v1.json")["contracts"][0]
    if audit.exists():
        if read_json(audit) != contract: raise ValueError("Existing contract differs")
    else:
        write_json(audit, contract)
    result = train("/data/libero/milk_sam_development_v1", "/data/libero/muse_middle_tokens_development_v1",
                   "/data/runs/multitask_to_milk_M1_smoke_v1", "M1", 7, "cuda", steps=1,
                   motor_checkpoint="/data/runs/muse_multitask_milk_soup_motor_seed7_v1/weights_best.pt",
                   dataset_audit=audit)
    volume.commit()
    return result


@app.local_entrypoint()
def main(): print(run.remote())
