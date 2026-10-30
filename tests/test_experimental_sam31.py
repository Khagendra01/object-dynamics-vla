import unittest
from types import SimpleNamespace

from odvla.experimental_sam31 import experimental_point_adapter


class ExperimentalAdapterTests(unittest.TestCase):
    def test_optional_prefetch_guard_skips_future_and_restores_methods(self):
        class Tracker:
            max_obj_ptrs_in_encoder = 16
            use_obj_ptrs_in_encoder = True
            add_tpos_enc_to_obj_ptrs = True
            only_obj_ptrs_in_the_past_for_eval = False
            def _get_tpos_enc(self, positions, device, max_abs_pos=None, dummy=False):
                return max_abs_pos
        class Detector:
            def __init__(self):
                self.calls = []
            def _build_multigpu_buffer_next_chunk(self, **kwargs):
                self.calls.append((kwargs["frame_idx_begin"], kwargs["frame_idx_end"]))
        class Model:
            def __init__(self):
                self.tracker = SimpleNamespace(model=Tracker())
                self.detector = Detector()
            def _build_sam2_output(self, state, frame, refined=None):
                return {}
            def _prepare_backbone_feats(self, state, frame_idx, reverse):
                self.detector._build_multigpu_buffer_next_chunk(frame_idx_begin=frame_idx, frame_idx_end=frame_idx + 1)
                self.detector._build_multigpu_buffer_next_chunk(frame_idx_begin=frame_idx + 1, frame_idx_end=frame_idx + 2)
        model = Model()
        original = model.detector._build_multigpu_buffer_next_chunk
        with experimental_point_adapter(SimpleNamespace(model=model), disable_detector_prefetch=True) as metadata:
            model._prepare_backbone_feats({}, 5, False)
            self.assertEqual(model.detector.calls, [(5, 6)])
            self.assertTrue(metadata["detector_prefetch_disabled"])
            with self.assertRaises(ValueError):
                model._prepare_backbone_feats({}, 5, True)
        self.assertEqual(model.detector._build_multigpu_buffer_next_chunk, original)

    def test_scoped_adapter_uses_configured_scale_and_restores_methods(self):
        class Tracker:
            max_obj_ptrs_in_encoder = 12
            use_obj_ptrs_in_encoder = True
            add_tpos_enc_to_obj_ptrs = True
            only_obj_ptrs_in_the_past_for_eval = False
            def _get_tpos_enc(self, positions, device, max_abs_pos=None, dummy=False):
                return positions, max_abs_pos, dummy
        tracker = Tracker()
        class Model:
            def __init__(self):
                self.tracker = SimpleNamespace(model=tracker)
            def _build_sam2_output(self, state, frame, refined=None):
                return {"original": True}
        predictor = SimpleNamespace(model=Model())
        original_merge = predictor.model._build_sam2_output
        original_tpos = tracker._get_tpos_enc
        with self.assertRaisesRegex(RuntimeError, "test"):
            with experimental_point_adapter(predictor) as metadata:
                self.assertEqual(metadata["pointer_scale_max"], 12)
                self.assertEqual(metadata["pointer_temporal_denominator"], 11)
                self.assertTrue(tracker.only_obj_ptrs_in_the_past_for_eval)
                self.assertEqual(tracker._get_tpos_enc([1], "cpu", max_abs_pos=3)[1], 12)
                self.assertEqual(tracker._get_tpos_enc([1], "cpu", max_abs_pos=3, dummy=True)[1], 3)
                self.assertEqual(predictor.model._build_sam2_output({"cached_frame_outputs": {}}, 0, {1: "mask"}), {1: "mask"})
                raise RuntimeError("test")
        self.assertEqual(predictor.model._build_sam2_output, original_merge)
        self.assertEqual(tracker._get_tpos_enc, original_tpos)
        self.assertFalse(tracker.only_obj_ptrs_in_the_past_for_eval)


if __name__ == "__main__":
    unittest.main()
