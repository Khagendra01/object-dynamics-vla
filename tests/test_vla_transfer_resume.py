import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from odvla.data import write_json
from odvla.vla.action_expert import MuseActionExpert
from odvla.vla.transfer import SEMANTIC_FIELDS, summarize_contract, audit_transfer
from scripts.train_muse_vla import train


class TransferResumeTests(unittest.TestCase):
    def test_transfer_rejects_semantics_and_protected_overlap(self):
        splits = {"fingerprint": "a", "train": ["a"], "val": ["b"], "test": ["c"]}
        contract = {"schema_version": 1, "dataset_fingerprint": "a", "audit_evidence": "manual audit",
                    "semantics": {k: "same" for k in SEMANTIC_FIELDS},
                    "episode_sources": {"a": "source:1", "b": "source:2", "c": "source:3"}}
        source = summarize_contract(contract, splits)
        dest = dict(source, protected_sources=["other:1"])
        self.assertEqual(audit_transfer(source, dest)["overlap_count"], 0)
        with self.assertRaisesRegex(ValueError, "overlaps"):
            audit_transfer(source, dict(dest, protected_sources=["source:1"]))
        with self.assertRaisesRegex(ValueError, "controller"):
            audit_transfer(source, dict(dest, semantics=dict(dest["semantics"], controller="other")))
        contract["episode_sources"]["b"] = "source:1"
        with self.assertRaisesRegex(ValueError, "overlap"):
            summarize_contract(contract, splits)

    def test_interrupted_training_resumes_exact_schedule(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as td:
            root, cache = Path(td) / "data", Path(td) / "cache"
            root.mkdir(); cache.mkdir()
            write_json(root / "splits.json", {"fingerprint": "x"})
            write_json(cache / "manifest.json", {"test": "fixture"})
            stats = {"dummy": np.array([1.], np.float32)}
            sample = {"muse_bits": torch.zeros(3, 6, dtype=torch.bfloat16).view(torch.uint16).numpy(),
                "muse_valid": np.ones(3, bool), "robot": np.zeros((4, 3), np.float32),
                "history_valid": np.ones(4, bool), "history_timestamps": np.arange(4, dtype=np.float32),
                "sam_values": np.zeros((4, 2, 5), np.float32), "sam_deltas": np.zeros((4, 2, 5), np.float32),
                "sam_valid": np.ones((4, 2, 5), bool), "sam_age": np.zeros((4, 2, 5), np.float32),
                "actions": np.ones((3, 2), np.float32), "action_valid": np.ones(3, bool),
                "change": np.zeros((2, 5), np.float32), "change_valid": np.ones((2, 5), bool)}
            prepared = (stats, [sample], [sample])
            factory = lambda: MuseActionExpert(muse_width=6, width=8, heads=2, ff_width=16,
                context_layers=1, expert_layers=2, history=4, horizon=3, robot_dim=3, action_dim=2)
            def stop(): raise RuntimeError("interrupt")
            with patch("scripts.train_muse_vla.MuseActionExpert", factory), patch("scripts.train_muse_vla.validate_splits"):
                args = (root, cache, Path(td) / "full", "motor", 7, "cpu")
                train(*args, steps=2, batch_size=64, accumulation=1, prepared=prepared, save_every=1)
                args = (root, cache, Path(td) / "resume", "motor", 7, "cpu")
                with self.assertRaisesRegex(RuntimeError, "interrupt"):
                    train(*args, steps=2, batch_size=64, accumulation=1, prepared=prepared,
                          save_every=1, checkpoint_hook=stop)
                train(*args, steps=2, batch_size=64, accumulation=1, prepared=prepared, save_every=1, resume=True)
                a = torch.load(Path(td) / "full" / "weights_step00002.pt", weights_only=True)
                b = torch.load(Path(td) / "resume" / "weights_step00002.pt", weights_only=True)
                for name in a: torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)
                with self.assertRaisesRegex(ValueError, "steps"):
                    train(*args, steps=3, batch_size=64, accumulation=1, prepared=prepared, save_every=1, resume=True)


if __name__ == "__main__": unittest.main()
