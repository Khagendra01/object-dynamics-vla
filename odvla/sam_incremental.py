"""Experimental SAM point tracker with bounded observed-frame updates.

This adapter depends on pinned SAM internals. Its parity evidence is scoped to
the development runs; callers must compare against prefix replay at runtime.
"""
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
from .experimental_sam31 import experimental_point_adapter


class IncrementalPointTracker:
    def __init__(self, predictor, first_frame, points, ids, capacity=200):
        if not 1 <= capacity <= 200 or len(points) != len(ids) or len(ids) == 0 or len(set(ids)) != len(ids):
            raise ValueError("Need distinct object IDs, matching points, and capacity 1..200")
        if any(len(point) != 2 for point in points): raise ValueError("Each point must have two coordinates")
        self.predictor, self.points, self.ids = predictor, points, ids
        self.capacity, self.index = capacity, 0
        self.temp = tempfile.TemporaryDirectory()
        self.session = None
        folder = Path(self.temp.name)
        data = Path(first_frame).read_bytes()
        self.first_frame_bytes = data
        try:
            # Reserve sequence slots with copies of the first observed image;
            # no future RGB is loaded or available in the tracker.
            for t in range(capacity): (folder / f"{t:06}.jpg").write_bytes(data)
            self.session = predictor.handle_request({"type": "start_session", "resource_path": str(folder),
                                                     "offload_video_to_cpu": True})["session_id"]
            self.state = predictor._all_inference_states[self.session]["state"]
            self.state["num_frames"] = 1
            with experimental_point_adapter(predictor, disable_detector_prefetch=True):
                for obj_id, point in zip(ids, points):
                    predictor.handle_request({"type": "add_prompt", "session_id": self.session, "frame_index": 0,
                        "points": [[point[0] / 512, point[1] / 512]], "point_labels": [1], "obj_id": obj_id})
        except Exception:
            self.close()
            raise

    def step(self, frame):
        import torch
        from sam3.model.io_utils import load_image_as_single_frame_video
        t = self.index
        if t >= self.capacity: raise ValueError("Incremental tracker capacity exceeded")
        if t == 0 and Path(frame).read_bytes() != self.first_frame_bytes:
            raise ValueError("First incremental frame differs from initialized image")
        if t:
            tensor, _, _ = load_image_as_single_frame_video(str(frame), image_size=self.predictor.model.image_size,
                offload_video_to_cpu=True, img_mean=self.predictor.model.image_mean, img_std=self.predictor.model.image_std)
            images = self.state["input_batch"].img_batch
            with torch.inference_mode(): images.tensors[t].copy_(tensor[0].to(images.device))
            self.state["num_frames"] = t + 1
            for tracker_state in self.state["sam2_inference_states"]: tracker_state["num_frames"] = t + 1
            # Internal dispatch marker selects tracker propagation for existing
            # objects; it does not add a new point or future observation.
            self.predictor.model.add_action_history(self.state, "refine", frame_idx=0, obj_ids=self.ids)
        with experimental_point_adapter(self.predictor, disable_detector_prefetch=True):
            stream = self.predictor.handle_stream_request({"type": "propagate_in_video", "session_id": self.session,
                "propagation_direction": "forward", "start_frame_index": t, "max_frame_num_to_track": 0,
                "output_prob_thresh": 0.0})
            try:
                outputs = list(stream)
            finally:
                stream.close()
        latest = next((v["outputs"] for v in outputs if v["frame_index"] == t), None)
        if latest is None: raise RuntimeError("Tracker did not emit current frame")
        output_ids, masks = np.asarray(latest["out_obj_ids"]), np.asarray(latest["out_binary_masks"])
        result = np.zeros((len(self.ids), 128, 128), dtype=bool)
        status = np.zeros(len(self.ids), dtype=np.uint8)
        for i, obj_id in enumerate(self.ids):
            matches = np.flatnonzero(output_ids == obj_id); status[i] = min(len(matches), 2)
            if len(matches) == 1:
                result[i] = np.asarray(Image.fromarray(masks[matches[0]].astype(np.uint8)).resize((128, 128), Image.Resampling.NEAREST), bool)
        self.index += 1
        return result, status

    def close(self):
        try:
            if self.session is not None:
                self.predictor.handle_request({"type": "close_session", "session_id": self.session})
                self.session = None
        finally:
            self.temp.cleanup()
