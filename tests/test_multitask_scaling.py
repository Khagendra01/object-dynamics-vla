import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from odvla.vla.bounded_features import BoundedFeatureCache, frozen_feature_key
from odvla.vla.libero_object_tasks import registry
from odvla.vla.multitask_sampling import TaskBalancedSampler
from scripts.collect_ten_task_smokes import collect
from scripts.run_ten_task_smokes import CHECKPOINT


def key(task="milk", instruction="pick up the milk", frame=0):
    return frozen_feature_key(task_id=task, episode_id="demo_0", frame=frame,
        instruction=instruction, image_sha256="abc", muse_revision="rev", processor_sha256="processor",
        backend={"gpu": "H200", "tf32": False, "autocast": False}, chat_date="2026-10-30")


class MultitaskScalingTests(unittest.TestCase):
    def test_registry_has_ten_distinct_targets_and_same_controller(self):
        tasks = registry()["tasks"]
        self.assertEqual(len(tasks), 10)
        self.assertEqual(len({t["instruction"] for t in tasks}), 10)
        self.assertEqual(len({t["object_slots"][0]["identity"] for t in tasks}), 10)
        self.assertTrue(all(t["controller"] == tasks[0]["controller"] for t in tasks))
        self.assertTrue(all(len(t["source_sha256"]) == 64 for t in tasks))

    def test_task_balancing_ignores_episode_and_window_imbalance(self):
        sampler = TaskBalancedSampler({"small": {"one": 2},
                                       "large": {f"ep{i}": 10000 for i in range(20)}}, seed=2)
        rows = sampler.take(10000)
        small = sum(task == "small" for task, _, _ in rows)
        self.assertGreater(small, 4500)
        self.assertLess(small, 5500)
        self.assertEqual(sum(sampler.examples_per_task.values()), 10000)

    def test_sampler_resume_is_exact_and_rejects_dataset_change(self):
        tasks = {"a": {"e1": 3, "e2": 6}, "b": {"e3": 5}}
        first = TaskBalancedSampler(tasks, seed=9)
        first.take(23)
        restored = TaskBalancedSampler(tasks, seed=1)
        restored.load_state_dict(first.state_dict())
        self.assertEqual(first.take(100), restored.take(100))
        with self.assertRaisesRegex(ValueError, "does not match"):
            TaskBalancedSampler({"a": {"e1": 4}, "b": {"e3": 5}}).load_state_dict(first.state_dict())

    def test_cache_key_changes_with_task_instruction_and_frame(self):
        self.assertNotEqual(key(), key(task="soup"))
        self.assertNotEqual(key(), key(instruction="pick up the soup"))
        self.assertNotEqual(key(), key(frame=1))

    def test_byte_bound_lru_and_restart(self):
        with tempfile.TemporaryDirectory() as temp:
            cache = BoundedFeatureCache(temp, 1600)
            array = {"features_bfloat16_bits": np.arange(250, dtype=np.uint16)}
            a, b, c = key(frame=1), key(frame=2), key(frame=3)
            self.assertTrue(cache.put(a, array))
            time.sleep(.01)
            self.assertTrue(cache.put(b, array))
            time.sleep(.01)
            self.assertIsNotNone(cache.get(a))
            time.sleep(.01)
            self.assertTrue(cache.put(c, array))
            self.assertLessEqual(cache.size_bytes, 1600)
            self.assertIsNone(cache.get(b))
            self.assertTrue(np.array_equal(cache.get(a)["features_bfloat16_bits"], array["features_bfloat16_bits"]))
            smaller = BoundedFeatureCache(Path(temp), 800)
            self.assertLessEqual(smaller.size_bytes, 800)

    def test_oversized_entry_is_not_cached(self):
        with tempfile.TemporaryDirectory() as temp:
            cache = BoundedFeatureCache(temp, 100)
            self.assertFalse(cache.put(key(), {"feature": np.arange(500, dtype=np.uint8)}))
            self.assertEqual(cache.size_bytes, 0)

    def test_smoke_aggregation_keeps_failed_attempt_and_requires_parity(self):
        parity = {k: True for k in ("rgb", "muse", "robot", "sam", "sam_valid")}
        original = {"checkpoint_sha256": CHECKPOINT, "results": {
            t["task_id"]: {"task_id": t["task_id"], "status": "completed", "steps": 30,
                           "success": False, "initial_parity": dict(parity)} for t in registry()["tasks"]}}
        original["results"]["libero_object_bbq_sauce"].update(status="failed", steps=0,
                                                                 initial_parity={**parity, "sam": False})
        retry = {"task_id": "libero_object_bbq_sauce", "checkpoint_sha256": CHECKPOINT,
                 "status": "completed", "steps": 30, "success": False, "initial_parity": dict(parity)}
        result = collect(original, retry)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(result["tasks"]["libero_object_bbq_sauce"]["attempts"]), 2)
        original["results"]["libero_object_milk"]["initial_parity"] = {}
        self.assertEqual(collect(original, retry)["status"], "incomplete")


if __name__ == "__main__":
    unittest.main()
