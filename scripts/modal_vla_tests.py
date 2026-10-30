"""Run repository unit tests in a bounded CPU-only Modal container."""
import modal

app = modal.App("odvla-vla-cpu-tests")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.10.0", "numpy>=1.24,<3", "Pillow>=10", "h5py")
         .add_local_python_source("odvla")
         .add_local_python_source("scripts")
         .add_local_dir("tests", "/root/tests"))


@app.function(image=image, cpu=2, memory=4096, timeout=600)
def run():
    import subprocess
    result = subprocess.run(["python", "-m", "unittest", "discover", "-s", "/root/tests", "-v"],
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout, flush=True)
    if result.returncode:
        raise RuntimeError(f"Tests failed: exit {result.returncode}")


@app.local_entrypoint()
def main():
    run.remote()
