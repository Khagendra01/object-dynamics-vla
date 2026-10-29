"""DAVIS human-mask-derived box initialization and prefix-consistency probe.

DAVIS is an independent non-robot video benchmark, not an annotation for LIBERO.
The source labels are used only to initialize frame 0 and score outputs.
"""
import modal

app = modal.App("odvla-davis-human-init")
volume = modal.Volume.from_name("object-dynamics-vla-data")
download_image = (modal.Image.debian_slim(python_version="3.12")
                  .pip_install("pyarrow", "Pillow", "numpy>=2"))
inference_image = (modal.Image.debian_slim(python_version="3.12")
                   .apt_install("git")
                   .pip_install("torch==2.10.0", "torchvision", index_url="https://download.pytorch.org/whl/cu128")
                   .run_commands("git clone https://github.com/facebookresearch/sam3.git /opt/sam3",
                       "cd /opt/sam3 && git checkout f1647e8d0c1d280991f4d1b7d974c2704537fe66")
                   .pip_install("/opt/sam3")
                   .pip_install("numpy<2", "Pillow", "opencv-python-headless", "setuptools", "einops", "pycocotools", "psutil")
                   .add_local_python_source("odvla")
                   .add_local_file("scripts/extract_sam31.py", "/root/extract_sam31.py"))


@app.function(image=download_image, volumes={"/data": volume}, cpu=4, memory=8192, timeout=1800)
def acquire():
    import hashlib
    import json
    import tempfile
    import urllib.request
    from pathlib import Path
    import pyarrow.parquet as pq

    revision = "ea8f94b3c5681836c11ec2de2d94492a819818a9"
    url = (f"https://huggingface.co/datasets/shijli/davis2017/resolve/{revision}/"
           "semi-supervised/validation-00000-of-00001.parquet")
    expected = "da2063a05e451f4997993d6fca35dfee64a029c6e476330a5bbe0b51286d93a4"
    out = Path("/data/davis2017/human_init_probe")
    if (out / "metadata.json").exists():
        return json.loads((out / "metadata.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp:
        parquet = Path(temp) / "validation.parquet"
        digest = hashlib.sha256()
        with urllib.request.urlopen(url, timeout=120) as src, parquet.open("wb") as dst:
            while chunk := src.read(1024 * 1024):
                digest.update(chunk)
                dst.write(chunk)
        if digest.hexdigest() != expected:
            raise ValueError("DAVIS parquet checksum mismatch")
        table = pq.read_table(parquet, columns=["video", "frame", "image", "mask"])
        video = table.column("video")[0].as_py()
        rows = sorted((r for r in table.to_pylist() if r["video"] == video), key=lambda r: r["frame"])
        if not rows or rows[0]["frame"] != 0:
            raise ValueError("Expected first-frame annotation")
        (out / "frames").mkdir(exist_ok=True)
        (out / "masks").mkdir(exist_ok=True)
        for i, row in enumerate(rows):
            if row["frame"] != i:
                raise ValueError("Expected contiguous ordered frames")
            (out / "frames" / f"{i:06}.jpg").write_bytes(row["image"]["bytes"])
            (out / "masks" / f"{i:06}.png").write_bytes(row["mask"]["bytes"])
    metadata = {"source": "shijli/davis2017 semi-supervised validation, repacked official DAVIS 2017",
                "revision": revision, "parquet_sha256": expected, "license": "CC BY-NC 4.0",
                "video": video, "frames": len(rows), "annotation": "first-frame human annotation; later annotations evaluation only"}
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    volume.commit()
    return metadata


@app.function(image=download_image, volumes={"/data": volume}, cpu=4, memory=8192, timeout=1800)
def acquire_more():
    """Select the next two validation videos in source order before inspecting results."""
    import hashlib
    import json
    import tempfile
    import urllib.request
    from pathlib import Path
    import pyarrow.parquet as pq

    revision = "ea8f94b3c5681836c11ec2de2d94492a819818a9"
    expected = "da2063a05e451f4997993d6fca35dfee64a029c6e476330a5bbe0b51286d93a4"
    url = (f"https://huggingface.co/datasets/shijli/davis2017/resolve/{revision}/"
           "semi-supervised/validation-00000-of-00001.parquet")
    with tempfile.TemporaryDirectory() as temp:
        parquet = Path(temp) / "validation.parquet"
        digest = hashlib.sha256()
        with urllib.request.urlopen(url, timeout=120) as source, parquet.open("wb") as out:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
                out.write(chunk)
        if digest.hexdigest() != expected:
            raise ValueError("DAVIS parquet checksum mismatch")
        table = pq.read_table(parquet, columns=["video", "frame", "image", "mask"])
        rows = table.to_pylist()
        names = list(dict.fromkeys(row["video"] for row in rows))[:3]
        if names[0] != "bike-packing" or len(names) != 3:
            raise ValueError("Unexpected validation source ordering")
        output = []
        for index, name in enumerate(names[1:], 1):
            selected = sorted((r for r in rows if r["video"] == name), key=lambda r: r["frame"])
            root = Path(f"/data/davis2017/human_init_extra_{index}")
            if root.exists() and any(root.iterdir()):
                raise FileExistsError(f"Refusing to overwrite {root}")
            (root / "frames").mkdir(parents=True)
            (root / "masks").mkdir()
            for i, row in enumerate(selected):
                if row["frame"] != i:
                    raise ValueError(f"Noncontiguous frames for {name}")
                (root / "frames" / f"{i:06}.jpg").write_bytes(row["image"]["bytes"])
                (root / "masks" / f"{i:06}.png").write_bytes(row["mask"]["bytes"])
            meta = {"source": "shijli/davis2017 semi-supervised validation, repacked official DAVIS 2017",
                    "revision": revision, "parquet_sha256": expected, "license": "CC BY-NC 4.0",
                    "video": name, "frames": len(selected), "selection": "source order, before tracking results",
                    "annotation": "first-frame human annotation; later annotations evaluation only"}
            (root / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
            output.append({"root": str(root), **meta})
    volume.commit()
    return output


@app.function(image=inference_image, gpu="A100-80GB", volumes={"/data": volume}, timeout=3600)
def track(point_workaround: bool = False):
    import json
    import shutil
    import sys
    import tempfile
    from pathlib import Path
    import numpy as np
    from PIL import Image
    from odvla.perception import merge_refined_masks
    from sam3.model_builder import build_sam3_multiplex_video_predictor
    sys.path.insert(0, "/root")
    from extract_sam31 import _patch_multiplex_init_state

    root = Path("/data/davis2017/human_init_probe")
    metadata = json.loads((root / "metadata.json").read_text())
    first = np.asarray(Image.open(root / "masks/000000.png"))
    ids = sorted(int(x) for x in np.unique(first) if x not in (0, 255))
    if not ids:
        raise ValueError("First DAVIS annotation contains no object")
    object_id = ids[0]
    ys, xs = np.where(first == object_id)
    height, width = first.shape
    box = [float(xs.min() / width), float(ys.min() / height),
           float((xs.max() - xs.min() + 1) / width), float((ys.max() - ys.min() + 1) / height)]
    nearest = np.argmin((xs - xs.mean()) ** 2 + (ys - ys.mean()) ** 2)
    point = [float((xs[nearest] + .5) / width), float((ys[nearest] + .5) / height)]
    model = build_sam3_multiplex_video_predictor(
        checkpoint_path="/data/checkpoints/sam3.1/sam3.1_multiplex.pt",
        use_fa3=False, compile=False, warm_up=False, async_loading_frames=False)
    _patch_multiplex_init_state(model)
    n = metadata["frames"]
    check_frames = [i for i in (10, 20) if i < n - 1]
    lengths = [x + 1 for x in check_frames] + [n]
    sequences = {}
    with tempfile.TemporaryDirectory() as temp:
        for length in lengths:
            folder = Path(temp) / str(length)
            folder.mkdir()
            for i in range(length):
                shutil.copyfile(root / "frames" / f"{i:06}.jpg", folder / f"{i:06}.jpg")
            sid = model.handle_request({"type": "start_session", "resource_path": str(folder),
                                        "offload_video_to_cpu": True})["session_id"]
            original_build = model.model._build_sam2_output
            try:
                prompt = {"type": "add_prompt", "session_id": sid, "frame_index": 0,
                          "output_prob_thresh": 0.0}
                if point_workaround:
                    prompt.update(points=[point], point_labels=[1], obj_id=1)
                else:
                    prompt.update(bounding_boxes=[box], bounding_box_labels=[1])
                initial = model.handle_request(prompt)["outputs"]
                if point_workaround:
                    target = 1
                    def merge(state, frame_idx, refined_obj_id_to_mask=None):
                        return merge_refined_masks(state["cached_frame_outputs"].get(frame_idx),
                                                   refined_obj_id_to_mask)
                    model.model._build_sam2_output = merge
                else:
                    initial_ids = np.asarray(initial["out_obj_ids"])
                    if len(initial_ids) != 1:
                        raise ValueError(f"Expected one initialized instance; received {initial_ids.tolist()}")
                    target = int(initial_ids[0])
                results = {}
                for item in model.handle_stream_request({"type": "propagate_in_video", "session_id": sid,
                        "propagation_direction": "forward", "start_frame_index": 0,
                        "output_prob_thresh": 0.0}):
                    output = item["outputs"]
                    matches = np.flatnonzero(np.asarray(output["out_obj_ids"]) == target)
                    masks = np.asarray(output["out_binary_masks"])
                    results[int(item["frame_index"])] = masks[matches[0]] if len(matches) == 1 else np.zeros((height, width), bool)
                sequences[length] = results
            finally:
                model.model._build_sam2_output = original_build
                model.handle_request({"type": "close_session", "session_id": sid})
    full = sequences[n]
    scores = []
    for i in range(n):
        truth = np.asarray(Image.open(root / "masks" / f"{i:06}.png"))
        valid = truth != 255
        reference = (truth == object_id) & valid
        predicted = full.get(i, np.zeros((height, width), bool)) & valid
        union = np.logical_or(reference, predicted).sum()
        scores.append(float(np.logical_and(reference, predicted).sum() / union) if union else None)
    prefix = {}
    for frame in check_frames:
        a, b = sequences[frame + 1].get(frame), full.get(frame)
        prefix[str(frame)] = {"available": a is not None and b is not None,
                              "different_pixels": int(np.count_nonzero(a != b)) if a is not None and b is not None else None,
                              "prefix_nonempty": bool(a.any()) if a is not None else False,
                              "full_nonempty": bool(b.any()) if b is not None else False}
    valid_scores = [s for s in scores if s is not None]
    report = {"metadata": metadata, "object_id": object_id, "box_xywh_normalized": box,
              "prompt_provenance": ("point" if point_workaround else "box") + " derived from first-frame human annotation; no future labels used to prompt",
              "point_xy_normalized": point if point_workaround else None,
              "missing_cache_workaround": point_workaround,
              "frames_returned": len(full), "nonempty_predictions": sum(bool(m.any()) for m in full.values()),
              "mean_region_iou": float(np.mean(valid_scores)), "iou_by_frame": scores,
              "prefix_checks": prefix, "evaluation_scope": "one DAVIS video; not LIBERO, not autonomous grounding"}
    path = Path("/data/eval/davis_human_init_sam31_point_workaround.json" if point_workaround else "/data/eval/davis_human_init_sam31_probe.json")
    path.write_text(json.dumps(report, indent=2) + "\n")
    volume.commit()
    return report


@app.local_entrypoint()
def main(point_workaround: bool = False, acquire_extra: bool = False):
    if acquire_extra:
        print("Additional clips:", acquire_more.remote())
        return
    print("Acquired:", acquire.remote())
    print("Tracking:", track.remote(point_workaround))
