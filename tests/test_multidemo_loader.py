import tempfile
import unittest
from pathlib import Path

import numpy as np

from odvla.data import fingerprint, write_json
from odvla.vla.libero_object_tasks import registry
from odvla.vla.multidemo import load_multidemo, pad_object_slots


class MultiDemoLoaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base = self.root / "libero/ten_task_multidemo_v1"
        self.base.mkdir(parents=True)
        self.source = registry()
        records, sam = [], []
        for task in self.source["tasks"]:
            task_id = task["task_id"]
            names = ["demo_0"] if task_id == "libero_object_milk" else ["demo_0", "demo_1", "demo_2"]
            task["episodes"] = {"train": names, "development": ["dev"], "evaluation": ["heldout"]}
            for demo in names:
                folder = self.base / task_id / demo
                (folder / "rgb").mkdir(parents=True)
                n = 3
                np.savez(folder / "state_actions.npz", robot=np.zeros((n, 15), np.float32),
                         actions=np.zeros((n, 7), np.float32), timestamps=np.arange(n, dtype=np.float64))
                np.savez(folder / "sam.npz", features=np.ones((n, 2, 5), np.float32),
                         valid=np.ones((n, 2, 5), bool))
                records.append({"task_id": task_id, "episode": demo, "split": "train", "frames": n,
                    "folder": str(folder.relative_to(self.root)), "instruction": task["instruction"],
                    "source_sha256": task["source_sha256"], "rgb_sequence_sha256": "fixture"})
                sam.append({"task_id": task_id, "episode": demo,
                            "sam_path": str((folder / "sam.npz").relative_to(self.root))})
        self.records = records
        self.source["status"] = "sources_prepared"
        write_json(self.root / "libero/ten_task_sources/registry.json", self.source)
        write_json(self.root / "libero/ten_task_sources/controller_audit.json", {"status": "passed", "task_count": 10})
        write_json(self.base / "renders.json", {"status": "complete", "demos_per_new_task": 3, "entries": records})
        write_json(self.base / "sam_extract.json", {"status": "complete", "entries": sam})
        milk = self.root / "libero/muse_aligned_development_v1"
        (milk / "episodes").mkdir(parents=True)
        entries = []
        for i in range(2):
            ep_id = f"episode_{i:04d}"
            entries.append({"id": ep_id, "file": f"episodes/{ep_id}.npz", "group": ep_id,
                            "source_demo": f"demo_{i}", "instruction": "pick up the milk and place it in the basket"})
            np.savez(milk / entries[-1]["file"], features=np.ones((3, 2, 5), np.float32) * (i + 1),
                     valid=np.ones((3, 2, 5), bool), robot=np.ones((3, 15), np.float32) * i,
                     actions=np.ones((3, 7), np.float32) * i, timestamps=np.arange(3, dtype=np.float64))
        manifest = {"schema_version": 1, "metric_names": ["a", "b", "c", "d", "e"],
                    "metric_units": ["fraction"] * 5, "object_slots": ["target", "basket"],
                    "action_convention": "fixture", "episodes": entries}
        write_json(milk / "manifest.json", manifest)
        write_json(milk / "splits.json", {"train": ["episode_0000"], "val": ["episode_0001"], "test": [],
            "allow_empty_test": True, "fingerprint": fingerprint(milk, manifest)})

    def test_loads_training_only_and_masks_unverified_target(self):
        episodes, lengths, _ = load_multidemo(self.root)
        self.assertEqual(len(episodes), 10)
        self.assertEqual(len(lengths["libero_object_ketchup"]), 3)
        chocolate = episodes["libero_object_chocolate_pudding"]["demo_0"]["data"]["valid"]
        self.assertFalse(chocolate[:, 0].any())
        self.assertTrue(chocolate[:, 1].all())

    def test_variable_slots_are_padded_but_extra_objects_are_not_dropped(self):
        features = np.ones((3, 1, 5), np.float32)
        padded, valid = pad_object_slots(features, np.ones_like(features, dtype=bool))
        self.assertEqual(padded.shape, (3, 2, 5))
        self.assertTrue(valid[:, 0].all())
        self.assertFalse(valid[:, 1].any())
        self.assertFalse(padded[:, 1].any())
        with self.assertRaisesRegex(ValueError, "slot limit"):
            pad_object_slots(np.zeros((3, 3, 5), np.float32), np.ones((3, 3, 5), bool))

    def test_rejects_evaluation_episode_before_loading(self):
        self.records[0]["episode"] = "heldout"
        write_json(self.base / "renders.json", {"status": "complete", "demos_per_new_task": 3, "entries": self.records})
        with self.assertRaisesRegex(ValueError, "non-training"):
            load_multidemo(self.root)

    def test_loader_accepts_a_hundred_task_registry_without_milk_special_case(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base = root / "libero/hundred_task_prepared"
            base.mkdir(parents=True)
            tasks, records, sam = [], [], []
            for i in range(100):
                task = {"task_id": f"task_{i:03d}", "instruction": f"do task {i}", "source_sha256": "fixture",
                        "episodes": {"train": ["e0", "e1", "e2"], "development": ["dev"], "evaluation": ["test"]}}
                tasks.append(task)
                for demo in task["episodes"]["train"]:
                    folder = base / task["task_id"] / demo
                    (folder / "rgb").mkdir(parents=True)
                    np.savez(folder / "state_actions.npz", robot=np.zeros((3, 15), np.float32),
                        actions=np.zeros((3, 7), np.float32), timestamps=np.arange(3, dtype=np.float64))
                    np.savez(folder / "sam.npz", features=np.ones((3, 2, 5), np.float32), valid=np.ones((3, 2, 5), bool))
                    records.append({"task_id": task["task_id"], "episode": demo, "split": "train", "frames": 3,
                        "folder": str(folder.relative_to(root)), "instruction": task["instruction"],
                        "source_sha256": "fixture", "rgb_sequence_sha256": "fixture"})
                    sam.append({"task_id": task["task_id"], "episode": demo,
                                "sam_path": str((folder / "sam.npz").relative_to(root))})
            write_json(root / "libero/hundred_task_sources/registry.json", {"status": "sources_prepared", "tasks": tasks})
            write_json(root / "libero/hundred_task_sources/controller_audit.json", {"status": "passed", "task_count": 100})
            write_json(base / "renders.json", {"status": "complete", "demos_per_new_task": 3, "entries": records})
            write_json(base / "sam_extract.json", {"status": "complete", "entries": sam})
            episodes, _, _ = load_multidemo(root, expected_tasks=100, retain_milk=False,
                prepared_dir="libero/hundred_task_prepared", registry_dir="libero/hundred_task_sources")
            self.assertEqual(len(episodes), 100)
            self.assertTrue(all(len(task) == 3 for task in episodes.values()))


if __name__ == "__main__":
    unittest.main()
