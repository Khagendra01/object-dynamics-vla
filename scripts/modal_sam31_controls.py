"""Official-image positive control and image-inspection point prompts.

Points are agent-selected exploratory guesses, not human annotations. Oracle
masks are used only after inference to assess identity, never to set prompts.
"""
import modal
from scripts.modal_sam31 import image as sam_image, volume

image = (sam_image.add_local_file("scripts/__init__.py", "/root/scripts/__init__.py")
         .add_local_file("scripts/modal_sam31.py", "/root/scripts/modal_sam31.py"))
app = modal.App("odvla-sam31-controls")


@app.function(image=image, gpu="A100-80GB", volumes={"/data": volume}, timeout=1800)
def probe():
    import json
    import shutil
    import sys
    import tempfile
    from pathlib import Path
    import numpy as np
    import torch
    from PIL import Image
    from odvla.data import load_episode, load_manifest
    from sam3.model_builder import build_sam3_multiplex_video_predictor
    sys.path.insert(0, "/root")
    from extract_sam31 import _patch_multiplex_init_state

    model = build_sam3_multiplex_video_predictor(
        checkpoint_path="/data/checkpoints/sam3.1/sam3.1_multiplex.pt",
        use_fa3=False, compile=False, warm_up=False, async_loading_frames=False)
    _patch_multiplex_init_state(model)
    root = "/data/libero/milk_raw"
    manifest = load_manifest(root)
    episode = load_episode(root, manifest["episodes"][0], manifest)
    oracle = np.load("/data/libero/milk_masks_oracle/episode_0000.npz")["masks"]
    rows = []
    with tempfile.TemporaryDirectory() as folder:
        image_path = Path(folder) / "000000.jpg"
        for kind, point in (("official_shoe", None), ("milk_guess", [69 / 128, 47 / 128]),
                            ("basket_guess", [100 / 128, 58 / 128]),
                            ("basket_box_guess", [86 / 128, 45 / 128, 33 / 128, 31 / 128]),
                            ("native_milk_box", [110 / 512, 239 / 512, 40 / 512, 76 / 512]),
                            ("native_basket_box", [350 / 512, 183 / 512, 120 / 512, 117 / 512])):
            if kind == "official_shoe":
                shutil.copy("/opt/sam3/assets/images/test_image.jpg", image_path)
            elif kind.startswith("native_"):
                Image.open("/data/eval/milk_native_512/frame_000.png").save(image_path, quality=98)
            else:
                Image.fromarray(episode["rgb"][0]).save(image_path, quality=98)
            sid = model.handle_request({"type": "start_session", "resource_path": folder})["session_id"]
            try:
                request = {"type": "add_prompt", "session_id": sid, "frame_index": 0,
                           "output_prob_thresh": 0.15}
                if point is None:
                    request["text"] = "shoe"
                elif "box" in kind:
                    request.update(bounding_boxes=[point], bounding_box_labels=[1],
                                   output_prob_thresh=0.0)
                else:
                    request.update(points=[point], point_labels=[1], obj_id=1)
                with torch.inference_mode():
                    output = model.handle_request(request)["outputs"]
                masks = np.asarray(output["out_binary_masks"])
                if len(masks) and kind in ("official_shoe", "basket_box_guess"):
                    rgb = np.asarray(Image.open(image_path).convert("RGB")).copy()
                    union = masks.any(axis=0)
                    rgb[union] = (0.5 * rgb[union] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
                    Image.fromarray(rgb).save(f"/data/eval/sam31_{kind}_overlay.png")
                row = {"kind": kind, "point_normalized_xy": point, "instances": len(masks),
                       "areas": [int(mask.sum()) for mask in masks]}
                if point is not None:
                    masks128 = [np.asarray(Image.fromarray(mask.astype(np.uint8)).resize(
                        (128, 128), Image.Resampling.NEAREST), dtype=bool) for mask in masks]
                    row["ious_oracle_milk_basket"] = [
                        [float(np.logical_and(mask, reference).sum() /
                               max(np.logical_or(mask, reference).sum(), 1)) for reference in oracle[0]]
                        for mask in masks128]
                rows.append(row)
            finally:
                model.handle_request({"type": "close_session", "session_id": sid})
    result = {"scope": "official example plus agent-selected points from visual inspection; oracle evaluation only",
              "rows": rows}
    path = Path("/data/eval/sam31_positive_and_point_controls.json")
    path.write_text(json.dumps(result, indent=2) + "\n")
    volume.commit()
    return result


@app.local_entrypoint()
def main():
    print(probe.remote())
