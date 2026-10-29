import unittest
from types import SimpleNamespace
import numpy as np
from odvla.perception import unique_mask


class IntegrationContractTests(unittest.TestCase):
    def test_ambiguous_instances_are_missing(self):
        masks = np.ones((2, 10, 20), bool)
        mask, status = unique_mask({"out_binary_masks": masks}, (10, 20))
        self.assertEqual(status, "ambiguous")
        self.assertFalse(mask.any())

    def test_single_instance_and_empty_output(self):
        one, status = unique_mask({"out_binary_masks": np.ones((1, 1, 10, 20), bool)}, (10, 20))
        self.assertEqual(status, "observed"); self.assertTrue(one.all())
        empty, status = unique_mask({"out_binary_masks": np.empty((0, 10, 20), bool)}, (10, 20))
        self.assertEqual(status, "missing"); self.assertFalse(empty.any())

    def test_wrong_mask_api_fails(self):
        with self.assertRaises(ValueError): unique_mask({"masks": np.ones((1, 4, 4), bool)}, (4, 4))
        with self.assertRaises(ValueError): unique_mask({"out_binary_masks": np.ones((1, 4, 4), bool)}, (8, 8))

    def test_sam_multiplex_init_state_compat(self):
        from scripts.extract_sam31 import _patch_multiplex_init_state
        class Model:
            def init_state(self, resource_path, offload_video_to_cpu=False,
                           async_loading_frames=False):
                return resource_path
        predictor = SimpleNamespace(model=Model())
        _patch_multiplex_init_state(predictor)
        self.assertEqual(predictor.model.init_state(
            resource_path="frame.jpg", offload_state_to_cpu=True), "frame.jpg")

    def test_muse_hidden_state_contract(self):
        from scripts.cache_muse import pick_last_hidden
        expected = np.zeros((1, 5, 12), np.float32)
        self.assertIs(pick_last_hidden(SimpleNamespace(hidden_states=[expected])), expected)
        with self.assertRaises(RuntimeError): pick_last_hidden(SimpleNamespace())


if __name__ == "__main__": unittest.main()
