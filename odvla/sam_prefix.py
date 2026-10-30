"""Slow initialized SAM replay on exactly the RGB frames supplied by the caller."""
from pathlib import Path
import numpy as np
from PIL import Image

from .experimental_sam31 import experimental_point_adapter


def replay_current(predictor, frame_directory, count, points, ids=(1, 2)):
    """Replay a fresh isolated session and return current 128-grid masks and ID status."""
    directory = Path(frame_directory)
    expected = [f"{i:06}.jpg" for i in range(count)]
    if count < 1 or sorted(p.name for p in directory.glob("*.jpg")) != expected:
        raise ValueError("Session directory must contain exactly the contiguous observed prefix")
    if len(points) != len(ids) or len(set(ids)) != len(ids):
        raise ValueError("Points and distinct IDs must match")
    sid = predictor.handle_request({"type": "start_session", "resource_path": str(directory),
                                    "offload_video_to_cpu": True})["session_id"]
    try:
        with experimental_point_adapter(predictor, disable_detector_prefetch=True):
            for obj_id, point in zip(ids, points):
                predictor.handle_request({"type": "add_prompt", "session_id": sid, "frame_index": 0,
                    "points": [[point[0] / 512, point[1] / 512]], "point_labels": [1], "obj_id": obj_id})
            stream = predictor.handle_stream_request({"type": "propagate_in_video", "session_id": sid,
                "propagation_direction": "forward", "start_frame_index": 0, "output_prob_thresh": 0.0})
            latest = None
            try:
                for item in stream:
                    if int(item["frame_index"]) == count - 1:
                        latest = item["outputs"]
            finally:
                stream.close()
            if latest is None:
                raise RuntimeError("SAM did not emit the current frame")
            output_ids = np.asarray(latest["out_obj_ids"])
            masks = np.asarray(latest["out_binary_masks"])
            result = np.zeros((len(ids), 128, 128), dtype=bool)
            status = np.zeros(len(ids), dtype=np.uint8)
            for slot, obj_id in enumerate(ids):
                matches = np.flatnonzero(output_ids == obj_id)
                status[slot] = min(len(matches), 2)
                if len(matches) == 1:
                    image = Image.fromarray(masks[int(matches[0])].astype(np.uint8))
                    result[slot] = np.asarray(image.resize((128, 128), Image.Resampling.NEAREST), bool)
            return result, status
    finally:
        predictor.handle_request({"type": "close_session", "session_id": sid})
