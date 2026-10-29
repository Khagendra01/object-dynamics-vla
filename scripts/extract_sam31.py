"""Causal correctness-first SAM 3.1 adapter: single-frame sessions, unique prompts.

This uses the SAM 3.1 multiplex model, but intentionally does NOT claim video
tracking: each frame is independent. It is slower than a validated streaming
adapter and rejects ambiguous multi-instance detections. See docs/INTEGRATIONS.md.
Real checkpoint execution was validated on a Modal A100-80GB; see
`docs/EXECUTION_REPORT.md` for mask-quality findings.
"""
import argparse
import importlib.metadata
import inspect
import json
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
from odvla.data import load_manifest, load_episode, write_json, file_hash
from odvla.perception import unique_mask


def _patch_multiplex_init_state(predictor):
    """Ignore offload_state_to_cpu, which the multiplex tracker does not accept.

    Sam3BasePredictor.start_session passes this keyword unconditionally; the
    SAM 3.1 multiplex model's init_state signature omits it. The offload flag is
    unnecessary for single-frame sessions, so dropping it is behavior-preserving.
    """
    cls = type(predictor.model)
    if "offload_state_to_cpu" in inspect.signature(cls.init_state).parameters:
        return
    original = cls.init_state

    def init_state(self, *args, offload_state_to_cpu=False, **kwargs):
        return original(self, *args, **kwargs)

    cls.init_state = init_state


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True); p.add_argument("--out", required=True)
    p.add_argument("--checkpoint", required=True, help="Explicit SAM 3.1 multiplex checkpoint")
    p.add_argument("--episode", help="Extract one episode to inspect before processing the dataset")
    a = p.parse_args()
    import torch
    from sam3.model_builder import build_sam3_multiplex_video_predictor
    if not torch.cuda.is_available(): raise RuntimeError("SAM extraction requires a configured CUDA GPU")
    checkpoint = Path(a.checkpoint)
    if not checkpoint.is_file(): raise FileNotFoundError(checkpoint)
    out = Path(a.out)
    if out.exists() and any(out.iterdir()): raise FileExistsError("Use an empty mask output directory")
    out.mkdir(parents=True, exist_ok=True)
    m = load_manifest(a.dataset)
    selected = [e for e in m["episodes"] if a.episode is None or e["id"] == a.episode]
    if not selected: raise ValueError("Unknown episode ID")
    predictor = build_sam3_multiplex_video_predictor(
        checkpoint_path=str(checkpoint), use_fa3=False, compile=False,
        warm_up=False, async_loading_frames=False)
    _patch_multiplex_init_state(predictor)
    counts = {"observed": 0, "missing": 0, "ambiguous": 0}
    for entry in selected:
        ep = load_episode(a.dataset, entry, m)
        if "rgb" not in ep: raise ValueError("RGB frames required")
        t, h, w, _ = ep["rgb"].shape
        masks = np.zeros((t, len(m["object_slots"]), h, w), bool)
        with tempfile.TemporaryDirectory(prefix="odvla-sam31-") as work:
            frame_dir = Path(work)
            for ti, rgb in enumerate(ep["rgb"]):
                # The session can see only this frame. No access to a later frame.
                Image.fromarray(rgb).save(frame_dir / "000000.jpg", quality=98)
                for oi, prompt in enumerate(m["object_slots"]):
                    response = predictor.handle_request({"type": "start_session", "resource_path": str(frame_dir)})
                    session_id = response["session_id"]
                    try:
                        with torch.inference_mode():
                            response = predictor.handle_request({"type": "add_prompt", "session_id": session_id,
                                                                 "frame_index": 0, "text": prompt})
                        masks[ti, oi], status = unique_mask(response["outputs"], (h, w))
                        counts[status] += 1
                    finally:
                        predictor.handle_request({"type": "close_session", "session_id": session_id})
                print(json.dumps({"episode": entry["id"], "frame": ti, "counts": counts}), flush=True)
        np.savez_compressed(out / (entry["id"] + ".npz"), masks=masks)
    try: version = importlib.metadata.version("sam3")
    except importlib.metadata.PackageNotFoundError: version = "editable installation; record git commit separately"
    write_json(out / "provenance.json", {"source": "SAM 3.1 multiplex; independent single-frame sessions",
               "causal": True, "identity_assumption": "One unique instance per ordered text prompt; audit manually",
               "uses_video_tracking": False, "checkpoint_sha256": file_hash(checkpoint),
               "package_version": version, "complete_dataset": a.episode is None,
               "counts": counts, "model_execution_status": "executed by scripts/extract_sam31.py"})


if __name__ == "__main__": main()
