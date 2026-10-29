"""Explore SAM text prompts on a milk training episode; oracle masks are evaluation only."""
import modal
from scripts.modal_sam31 import image as sam_image, volume

image = (sam_image.add_local_file("scripts/__init__.py", "/root/scripts/__init__.py")
         .add_local_file("scripts/modal_sam31.py", "/root/scripts/modal_sam31.py"))
app = modal.App("odvla-sam31-milk-probe")


@app.function(image=image, gpu="A100-80GB", volumes={"/data": volume}, timeout=1800)
def probe(native: bool = False):
    import json
    import tempfile
    from pathlib import Path

    import numpy as np
    import torch
    from PIL import Image
    from odvla.data import load_manifest, load_episode
    from sam3.model_builder import build_sam3_multiplex_video_predictor
    import sys
    sys.path.insert(0, "/root")
    from extract_sam31 import _patch_multiplex_init_state

    dataset = "/data/libero/milk_raw"
    manifest = load_manifest(dataset)
    episode = load_episode(dataset, manifest["episodes"][0], manifest)
    oracle = np.load("/data/libero/milk_masks_oracle/episode_0000.npz")["masks"]
    predictor = build_sam3_multiplex_video_predictor(
        checkpoint_path="/data/checkpoints/sam3.1/sam3.1_multiplex.pt",
        use_fa3=False, compile=False, warm_up=False, async_loading_frames=False)
    _patch_multiplex_init_state(predictor)
    rows = []
    with tempfile.TemporaryDirectory() as work:
        for frame in (0, 50, 100):
            image = (Image.open(f"/data/eval/milk_native_512/frame_{frame:03}.png").convert("RGB")
                     if native else Image.fromarray(episode["rgb"][frame]).resize((512, 512), Image.Resampling.BICUBIC))
            image.save(Path(work) / "000000.jpg", quality=98)
            for prompt in ("milk", "milk carton", "carton", "basket", "wire basket", "container"):
                sid = predictor.handle_request({"type": "start_session", "resource_path": work})["session_id"]
                try:
                    with torch.inference_mode():
                        output = predictor.handle_request({"type": "add_prompt", "session_id": sid,
                            "frame_index": 0, "text": prompt, "output_prob_thresh": 0.15})["outputs"]
                    predicted = np.asarray(output["out_binary_masks"])
                    ious = []
                    for mask in predicted:
                        resized = np.asarray(Image.fromarray(mask.astype(np.uint8)).resize(
                            (128, 128), Image.Resampling.NEAREST), dtype=bool)
                        ious.append([float(np.logical_and(resized, ref).sum() /
                                          max(np.logical_or(resized, ref).sum(), 1))
                                     for ref in oracle[frame]])
                    rows.append({"frame": frame, "prompt": prompt, "instances": len(predicted),
                                 "ious_against_oracle_milk_basket": ious})
                finally:
                    predictor.handle_request({"type": "close_session", "session_id": sid})
    result = {"purpose": "exploratory prompt probe on training episode; oracle masks used only to score predictions",
              "dataset": dataset, "episode": "episode_0000", "image_source": "native 512x512 simulator render" if native else "upsampled 128x128 RGB",
              "threshold": 0.15, "rows": rows}
    path = Path("/data/eval/milk_sam31_native_prompt_probe.json" if native else "/data/eval/milk_sam31_prompt_probe.json")
    path.write_text(json.dumps(result, indent=2) + "\n")
    volume.commit()
    return result


@app.local_entrypoint()
def main(native: bool = False):
    print(probe.remote(native))
