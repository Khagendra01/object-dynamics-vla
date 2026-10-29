"""Download and verify one public LIBERO task directly to the Modal volume."""
import modal

app = modal.App("odvla-libero-milk-download")
volume = modal.Volume.from_name("object-dynamics-vla-data")


@app.function(image=modal.Image.debian_slim(python_version="3.12"),
              volumes={"/data": volume}, timeout=1800)
def download():
    import hashlib
    import os
    import urllib.request
    from pathlib import Path

    url = ("https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/resolve/main/"
           "libero_object/pick_up_the_milk_and_place_it_in_the_basket_demo.hdf5")
    target = Path("/data/libero/source/milk_demo.hdf5")
    expected = "e56d954928e613f9db58085b7677ccf9c4824f1ab982265f71405d5bb14cfb4a"
    temp = target.with_suffix(".partial")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(str(target))
    digest = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=120) as response, temp.open("wb") as out:
        while chunk := response.read(1024 * 1024):
            digest.update(chunk)
            out.write(chunk)
    if digest.hexdigest() != expected:
        temp.unlink()
        raise ValueError("Downloaded file checksum does not match published LFS SHA256")
    os.replace(temp, target)
    volume.commit()
    return {"path": str(target), "bytes": target.stat().st_size, "sha256": expected}


@app.local_entrypoint()
def main():
    print(download.remote())
