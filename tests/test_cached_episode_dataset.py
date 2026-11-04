import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from odvla.data import file_hash, fit_stats, fingerprint, write_json
from odvla.vla.dataset import CachedEpisodeDataset


def episode(seed):
    rng = np.random.default_rng(seed)
    frames = 5
    features = rng.normal(size=(frames, 2, 5)).astype(np.float32)
    valid = np.ones_like(features, dtype=bool)
    valid[1, 1, 2] = False
    robot = rng.normal(size=(frames, 15)).astype(np.float32)
    actions = rng.normal(size=(frames, 7)).astype(np.float32)
    return {"features": features, "valid": valid, "robot": robot, "actions": actions,
            "timestamps": np.arange(frames, dtype=np.float32) / 20,
            "context": np.zeros((frames, 0), dtype=np.float32)}


class CachedEpisodeDatasetTest(unittest.TestCase):
    def test_reads_the_resumable_nf4_cache_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            root, cache = Path(temp) / "dataset", Path(temp) / "cache"
            root.mkdir()
            entries = [{"id": f"episode_{i:04d}", "file": f"episodes/episode_{i:04d}.npz",
                        "group": f"episode_{i:04d}", "instruction": "pick up the milk"}
                       for i in range(2)]
            for i, entry in enumerate(entries):
                path = root / entry["file"]
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(path, **episode(i))
            manifest = {"schema_version": 1, "metric_names": ["a", "b", "c", "d", "e"],
                        "metric_units": ["u"] * 5, "object_slots": ["milk", "basket"],
                        "action_convention": "fixture", "episodes": entries}
            write_json(root / "manifest.json", manifest)
            splits = {"train": ["episode_0000"], "val": ["episode_0001"], "test": [],
                      "allow_empty_test": True, "seed": 1, "schema_version": 1,
                      "fingerprint": fingerprint(root, manifest)}
            write_json(root / "splits.json", splits)
            stats = fit_stats(root, splits)
            cache.mkdir()
            write_json(cache / "manifest.json", {"complete": True, "dataset_fingerprint": splits["fingerprint"]})
            for entry in entries:
                folder = cache / entry["id"]
                folder.mkdir()
                records = []
                for frame in range(5):
                    rng = np.random.default_rng(100 + frame)
                    tokens = {"features_bfloat16_bits": rng.integers(0, 65535, (4, 6656), dtype=np.uint16),
                              "attention_mask": np.ones(4, dtype=bool), "input_ids": np.arange(4, dtype=np.int64),
                              "positions": np.arange(4, dtype=np.int32)}
                    path = folder / f"{frame:06}.npz"
                    np.savez(path, **tokens)
                    records.append({"frame": frame, "timestamp": frame / 20, "sha256": file_hash(path)})
                write_json(folder / "manifest.json", {"episode": entry["id"], "frames": 5,
                    "instruction_sha256": hashlib.sha256(entry["instruction"].encode()).hexdigest(),
                    "records": records})
            view = CachedEpisodeDataset(root, cache, "train", stats)
            self.assertEqual(len(view), 5)
            sample = view[2]
            self.assertEqual(sample["episode"], "episode_0000")
            self.assertEqual(sample["t"], 2)
            self.assertEqual(sample["muse_bits"].shape, (4, 6656))
            self.assertEqual(sample["robot"].shape, (4, 15))
            self.assertEqual(sample["actions"].shape, (50, 7))
            self.assertTrue(sample["action_valid"][:3].all())
            self.assertFalse(sample["action_valid"][3:].any())
            both = CachedEpisodeDataset(root, cache, "all", stats)
            self.assertEqual(len(both), 10)
            self.assertEqual({both[i]["episode"] for i in range(len(both))}, {"episode_0000", "episode_0001"})

    def test_rejects_incomplete_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            root, cache = Path(temp) / "dataset", Path(temp) / "cache"
            root.mkdir(); cache.mkdir()
            entries = [{"id": f"episode_{i:04d}", "file": f"episodes/episode_{i:04d}.npz",
                        "group": f"episode_{i:04d}", "instruction": "pick up the milk"} for i in range(2)]
            for i, entry in enumerate(entries):
                path = root / entry["file"]; path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(path, **episode(i))
            manifest = {"schema_version": 1, "metric_names": ["a", "b", "c", "d", "e"],
                        "metric_units": ["u"] * 5, "object_slots": ["milk", "basket"],
                        "action_convention": "fixture", "episodes": entries}
            write_json(root / "manifest.json", manifest)
            splits = {"train": ["episode_0000"], "val": ["episode_0001"], "test": [],
                      "allow_empty_test": True, "fingerprint": fingerprint(root, manifest)}
            write_json(root / "splits.json", splits)
            write_json(cache / "manifest.json", {"complete": False, "dataset_fingerprint": splits["fingerprint"]})
            with self.assertRaisesRegex(ValueError, "complete Muse cache"):
                CachedEpisodeDataset(root, cache, "train", fit_stats(root, splits))

if __name__ == "__main__":
    unittest.main()
