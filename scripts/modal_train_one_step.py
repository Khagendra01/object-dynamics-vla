"""Train the standalone, causal one-step monitor on Modal; validation only."""
import modal

app = modal.App("odvla-one-step-monitor")
volume = modal.Volume.from_name("object-dynamics-vla-data")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch>=2.7", "numpy>=1.24,<3", "Pillow>=10")
         .add_local_python_source("odvla"))


@app.function(image=image, gpu="A100-80GB", cpu=4, memory=16384, timeout=3600,
              volumes={"/data": volume})
def train():
    from odvla.one_step import train_one_step
    result = train_one_step("/data/libero/milk_oracle",
                            "/data/runs/libero_milk_oracle_ablations/objects_joint_seed7",
                            "/data/runs/milk_one_step_joint_seed7", epochs=30, device="cuda")
    volume.commit()
    return result


@app.local_entrypoint()
def main():
    print(train.remote())
