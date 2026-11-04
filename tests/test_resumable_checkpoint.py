import tempfile
import unittest
from pathlib import Path

import torch

from odvla.vla.checkpointing import checkpoint_payload, validate_checkpoint
from odvla.vla.sampler import ShuffledBatchSampler


class ResumableCheckpointTest(unittest.TestCase):
    def payload(self):
        sampler = ShuffledBatchSampler(23, 4, seed=7)
        for _ in range(3):
            sampler.next_batch()
        return checkpoint_payload(
            completed_updates=12,
            prefix={"weight": torch.arange(4, dtype=torch.float32)},
            trainable_policy={"expert.weight": torch.ones(3)},
            optimizer={"state": {0: {"step": torch.tensor(12.0)}}, "param_groups": [{"lr": 1e-4}]},
            torch_rng=torch.arange(16, dtype=torch.uint8),
            cuda_rng=[torch.arange(8, dtype=torch.uint8)],
            sampler_state=sampler.state_dict(),
            provenance={"normalize_queries": True, "dynamics_objective": False,
                        "dataset": "aligned", "dataset_fingerprint": "abc",
                        "cache": "nf4", "cache_manifest_sha256": "def",
                        "stats_sha256": "ghi", "batch_size": 4, "lr": 1e-4, "seed": 7})

    def test_roundtrip_keeps_optimizer_rng_and_sample_order(self):
        payload = self.payload()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "checkpoint_12.pt"
            torch.save(payload, path)
            loaded = torch.load(path, map_location="cpu", weights_only=True)
        validate_checkpoint(loaded, expected={"dataset_fingerprint": "abc", "cache_manifest_sha256": "def"})
        original = ShuffledBatchSampler.from_state(payload["sampler_state"])
        restored = ShuffledBatchSampler.from_state(loaded["sampler_state"])
        for _ in range(20):
            self.assertEqual(original.next_batch().tolist(), restored.next_batch().tolist())
        self.assertTrue(torch.equal(loaded["torch_rng"], payload["torch_rng"]))
        self.assertEqual(loaded["optimizer"]["state"][0]["step"].item(), 12.0)

    def test_rejects_different_run_identity(self):
        payload = self.payload()
        with self.assertRaisesRegex(ValueError, "dataset_fingerprint mismatch"):
            validate_checkpoint(payload, expected={"dataset_fingerprint": "different"})

    def test_rejects_provenance_key_collision(self):
        sampler = ShuffledBatchSampler(9, 2, seed=1)
        with self.assertRaisesRegex(ValueError, "cannot replace"):
            checkpoint_payload(completed_updates=1, prefix={}, trainable_policy={}, optimizer={},
                torch_rng=torch.arange(1, dtype=torch.uint8), cuda_rng=[], sampler_state=sampler.state_dict(),
                provenance={"completed_updates": 99, "normalize_queries": True, "dynamics_objective": False,
                            "dataset": "d", "dataset_fingerprint": "f", "cache": "c",
                            "cache_manifest_sha256": "m", "stats_sha256": "s", "batch_size": 2,
                            "lr": 1e-4, "seed": 1})

    def test_requires_sampler_state_for_resume(self):
        payload = self.payload()
        payload["sampler_state"] = {}
        with self.assertRaisesRegex(ValueError, "sampler state"):
            validate_checkpoint(payload)


if __name__ == "__main__":
    unittest.main()
