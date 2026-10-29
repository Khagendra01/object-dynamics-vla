"""Fresh-container point-tracking controls on the existing DAVIS clip.

Only the first human annotation initializes the model. Raw frame-10 tracker
logits and masks are saved on the Modal volume, never in the public repository.
"""
import modal
from scripts.modal_davis_human_init import inference_image, volume

image = (inference_image.add_local_file("scripts/__init__.py", "/root/scripts/__init__.py")
         .add_local_file("scripts/modal_davis_human_init.py", "/root/scripts/modal_davis_human_init.py"))
app = modal.App("odvla-sam31-repeatability")
OUT = "/data/eval/sam31_repeatability_v1"


@app.function(image=image, gpu="A100-80GB", volumes={"/data": volume}, timeout=1800,
              single_use_containers=True)
def run_case(kind: str, repeat: int, clip_index: int = 0):
    import hashlib
    import json
    import shutil
    import sys
    import tempfile
    from pathlib import Path
    import numpy as np
    import torch
    from PIL import Image
    from odvla.perception import merge_refined_masks
    from sam3.model_builder import build_sam3_multiplex_video_predictor
    sys.path.insert(0, "/root")
    from extract_sam31 import _patch_multiplex_init_state

    if kind not in {"prefix", "full", "reverse_future", "repeat_future"} or repeat not in (1, 2):
        raise ValueError("Invalid case")
    if clip_index not in (0, 1, 2):
        raise ValueError("clip_index must be 0, 1, or 2")
    root = Path("/data/davis2017/human_init_probe" if clip_index == 0 else
                f"/data/davis2017/human_init_extra_{clip_index}")
    metadata = json.loads((root / "metadata.json").read_text())
    n = metadata["frames"]
    reference = np.asarray(Image.open(root / "masks/000000.png"))
    ids = sorted(int(x) for x in np.unique(reference) if x not in (0, 255))
    ys, xs = np.where(reference == ids[0])
    nearest = np.argmin((xs - xs.mean()) ** 2 + (ys - ys.mean()) ** 2)
    point = [float((xs[nearest] + .5) / reference.shape[1]),
             float((ys[nearest] + .5) / reference.shape[0])]
    torch.manual_seed(2468)
    torch.cuda.manual_seed_all(2468)
    np.random.seed(2468)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = build_sam3_multiplex_video_predictor(
        checkpoint_path="/data/checkpoints/sam3.1/sam3.1_multiplex.pt",
        use_fa3=False, compile=False, warm_up=False, async_loading_frames=False)
    _patch_multiplex_init_state(model)
    indices = list(range(11 if kind == "prefix" else n))
    if kind == "reverse_future":
        indices[11:] = list(reversed(indices[11:]))
    elif kind == "repeat_future":
        indices[11:] = [10] * (n - 11)
    file_hashes = []
    with tempfile.TemporaryDirectory() as work:
        for i, source_i in enumerate(indices):
            source = root / "frames" / f"{source_i:06}.jpg"
            file_hashes.append(hashlib.sha256(source.read_bytes()).hexdigest())
            shutil.copyfile(source, Path(work) / f"{i:06}.jpg")
        sid = model.handle_request({"type": "start_session", "resource_path": work,
                                    "offload_video_to_cpu": True})["session_id"]
        original_build = model.model._build_sam2_output
        original_tracker = model.model._propogate_tracker_one_frame_local_gpu
        progress = {"frames": [], "logits": None, "object_ids": None}
        try:
            model.handle_request({"type": "add_prompt", "session_id": sid, "frame_index": 0,
                                  "points": [point], "point_labels": [1], "obj_id": 1})
            def merge(state, frame_idx, refined_obj_id_to_mask=None):
                return merge_refined_masks(state["cached_frame_outputs"].get(frame_idx), refined_obj_id_to_mask)
            model.model._build_sam2_output = merge
            def tracker(*args, **kwargs):
                result = original_tracker(*args, **kwargs)
                frame = int(kwargs["frame_idx"])
                progress["frames"].append(frame)
                ids_here, logits, _ = result
                if frame == 10:
                    progress["object_ids"] = [int(x) for x in ids_here]
                    matches = [i for i, obj_id in enumerate(ids_here) if int(obj_id) == 1]
                    if len(matches) == 1:
                        progress["logits"] = logits[matches[0]].detach().float().cpu().numpy()
                return result
            model.model._propogate_tracker_one_frame_local_gpu = tracker
            found = None
            emitted = {}
            emission_processed = None
            stream = model.handle_stream_request({"type": "propagate_in_video", "session_id": sid,
                     "propagation_direction": "forward", "start_frame_index": 0,
                     "output_prob_thresh": 0.0})
            for item in stream:
                output = item["outputs"]
                matches = np.flatnonzero(np.asarray(output["out_obj_ids"]) == 1)
                frame = int(item["frame_index"])
                if kind == "full" and repeat == 1 and clip_index > 0:
                    emitted[frame] = (np.asarray(output["out_binary_masks"])[matches[0]]
                                      if len(matches) == 1 else np.zeros(reference.shape, bool))
                if frame == 10:
                    if len(matches) != 1 or progress["logits"] is None:
                        raise RuntimeError("Frame 10 did not produce the initialized object and raw logits")
                    found = {"mask": np.asarray(output["out_binary_masks"])[matches[0]],
                             "logits": progress["logits"].copy()}
                    emission_processed = (len(progress["frames"]), progress["frames"][-1])
                    if not (kind == "full" and repeat == 1 and clip_index > 0):
                        break
            stream.close()
            if found is None:
                raise RuntimeError("Frame 10 was not emitted")
        finally:
            model.model._build_sam2_output = original_build
            model.model._propogate_tracker_one_frame_local_gpu = original_tracker
            model.handle_request({"type": "close_session", "session_id": sid})
    out = Path(OUT if clip_index == 0 else f"{OUT}_extra_{clip_index}")
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{kind}_{repeat}"
    if (out / (stem + ".json")).exists():
        raise FileExistsError("Refusing to overwrite a previous measurement")
    np.savez_compressed(out / (stem + ".npz"), **found)
    quality = None
    if emitted:
        scores = []
        for i in range(n):
            truth = np.asarray(Image.open(root / "masks" / f"{i:06}.png"))
            valid = truth != 255
            ref = (truth == ids[0]) & valid
            prediction = emitted.get(i, np.zeros(reference.shape, bool)) & valid
            union = np.logical_or(ref, prediction).sum()
            if union:
                scores.append(float(np.logical_and(ref, prediction).sum() / union))
        quality = {"frames_returned": len(emitted), "nonempty_predictions": sum(bool(m.any()) for m in emitted.values()),
                   "mean_region_iou": float(np.mean(scores)), "scored_frames": len(scores),
                   "ground_truth_use": "post-hoc scoring only"}
    result = {"kind": kind, "repeat": repeat, "video": metadata["video"], "sequence_length": len(indices),
              "point_xy": point, "seed": 2468, "precision": "upstream CUDA bfloat16 autocast",
              "input_prefix_sha256": hashlib.sha256("".join(file_hashes[:11]).encode()).hexdigest(),
              "input_all_sha256": hashlib.sha256("".join(file_hashes).encode()).hexdigest(),
              "described_frame": 10, "tracker_frames_processed_at_emission": emission_processed[0],
              "last_processed_frame_at_emission": emission_processed[1],
              "is_final_input_frame": len(indices) == 11,
              "emission_path": "partial propagation loop yield; not a postprocessing flush",
              "logit_shape": list(found["logits"].shape),
              "logit_storage": "pre-threshold tracker logits cast to float32 for storage",
              "mask_area": int(found["mask"].sum()),
              "torch_version": str(torch.__version__), "cuda": str(torch.version.cuda),
              "full_video_quality": quality}
    (out / (stem + ".json")).write_text(json.dumps(result, indent=2) + "\n")
    volume.commit()
    return result


@app.function(image=image, volumes={"/data": volume}, timeout=600)
def summarize(clip_index: int = 0):
    import json
    from pathlib import Path
    import numpy as np
    if clip_index not in (0, 1, 2):
        raise ValueError("clip_index must be 0, 1, or 2")
    root = Path(OUT if clip_index == 0 else f"{OUT}_extra_{clip_index}")
    names = [f"{kind}_{repeat}" for kind in ("prefix", "full", "reverse_future", "repeat_future") for repeat in (1, 2)]
    cases = {name: json.loads((root / f"{name}.json").read_text()) for name in names}
    arrays = {name: dict(np.load(root / f"{name}.npz")) for name in names}
    pairs = [(f"{kind}_1", f"{kind}_2") for kind in ("prefix", "full", "reverse_future", "repeat_future")]
    pairs += [("full_1", "reverse_future_1"), ("full_1", "repeat_future_1"), ("prefix_1", "full_1")]
    comparisons = []
    for left, right in pairs:
        a, b = arrays[left], arrays[right]
        diff = np.abs(a["logits"] - b["logits"])
        comparisons.append({"left": left, "right": right,
            "logit_max_abs_diff": float(diff.max()), "logit_mean_abs_diff": float(diff.mean()),
            "logit_sign_flips": int(np.count_nonzero((a["logits"] > 0) != (b["logits"] > 0))),
            "binary_mask_different_pixels": int(np.count_nonzero(a["mask"] != b["mask"]))})
    report = {"cases": cases, "comparisons": comparisons,
              "design": "single-use containers, fixed seed and precision; order 1 then reverse order 2"}
    (root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    volume.commit()
    return report


@app.local_entrypoint()
def main(clip_index: int = 0):
    kinds = ("prefix", "full", "reverse_future", "repeat_future")
    for kind in kinds:
        print(run_case.remote(kind, 1, clip_index))
    for kind in reversed(kinds):
        print(run_case.remote(kind, 2, clip_index))
    print(summarize.remote(clip_index))
