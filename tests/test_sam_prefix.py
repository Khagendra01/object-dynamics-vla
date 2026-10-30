import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image

from odvla.sam_prefix import replay_current


class _Adapter:
    def __init__(self):
        self.model = self
        self.tracker = self
        self.detector = self
        self.max_obj_ptrs_in_encoder = 16
        self.use_obj_ptrs_in_encoder = True
        self.add_tpos_enc_to_obj_ptrs = True
        self.only_obj_ptrs_in_the_past_for_eval = False
        self.closed = False

    def _build_sam2_output(self, *args, **kwargs):
        return {}

    def _get_tpos_enc(self, *args, **kwargs):
        return None

    def _prepare_backbone_feats(self, *args, **kwargs):
        return None

    def _build_multigpu_buffer_next_chunk(self, *args, **kwargs):
        return None

    def handle_request(self, request):
        if request["type"] == "start_session":
            return {"session_id": "s"}
        if request["type"] == "close_session":
            self.closed = True
        return {}

    def handle_stream_request(self, request):
        yield {"frame_index": 0, "outputs": {"out_obj_ids": np.array([1, 2]),
               "out_binary_masks": np.stack([np.ones((2, 2), bool), np.zeros((2, 2), bool)])}}


class PrefixReplayTests(unittest.TestCase):
    def test_exact_prefix_and_explicit_empty_mask(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            Image.new("RGB", (2, 2)).save(folder / "000000.jpg")
            fake = _Adapter()
            masks, status = replay_current(fake, folder, 1, [(0, 0), (1, 1)])
            self.assertEqual(status.tolist(), [1, 1])
            self.assertTrue(masks[0].all())
            self.assertFalse(masks[1].any())
            self.assertTrue(fake.closed)
            Image.new("RGB", (2, 2)).save(folder / "000001.jpg")
            with self.assertRaises(ValueError):
                replay_current(fake, folder, 1, [(0, 0), (1, 1)])


if __name__ == "__main__":
    unittest.main()
