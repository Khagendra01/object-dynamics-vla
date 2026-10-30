"""One real-cache motor update; validation and checkpointing included."""
import modal
from scripts.modal_real_ablations import image as base_image, volume

image = (base_image.add_local_file("scripts/__init__.py", "/root/scripts/__init__.py")
         .add_local_file("scripts/modal_real_ablations.py", "/root/scripts/modal_real_ablations.py")
         .add_local_file("scripts/train_muse_vla.py", "/root/train_muse_vla.py"))
app = modal.App("odvla-motor-smoke")


@app.function(image=image, gpu="T4", cpu=4, memory=16384,
              volumes={"/data": volume}, timeout=1800)
def run():
    import sys
    sys.path.insert(0, "/root")
    from train_muse_vla import train
    result = train("/data/libero/milk_sam_development_v1",
                   "/data/libero/muse_middle_tokens_development_v1",
                   "/data/runs/muse_motor_smoke_v1", "motor", 7, "cuda", steps=1)
    volume.commit()
    return result


@app.local_entrypoint()
def main():
    print(run.remote())
