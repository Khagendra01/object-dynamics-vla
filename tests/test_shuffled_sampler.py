import json
import unittest

from odvla.vla.sampler import ShuffledBatchSampler


class ShuffledBatchSamplerTest(unittest.TestCase):
    def test_same_seed_reproduces_shuffled_minibatches(self):
        first = ShuffledBatchSampler(37, 5, seed=11)
        second = ShuffledBatchSampler(37, 5, seed=11)
        for _ in range(30):
            self.assertEqual(first.next_batch().tolist(), second.next_batch().tolist())

    def test_each_epoch_covers_every_index_once(self):
        sampler = ShuffledBatchSampler(37, 5, seed=11)
        seen = []
        for _ in range(8):
            seen.extend(int(x) for x in sampler.next_batch())
        self.assertEqual(sorted(seen), list(range(37)))
        self.assertNotEqual(seen, list(range(37)))

    def test_resume_mid_epoch_is_exact(self):
        original = ShuffledBatchSampler(101, 7, seed=3)
        for _ in range(5):
            original.next_batch()
        resumed = ShuffledBatchSampler.from_state(json.loads(json.dumps(original.state_dict())))
        for _ in range(60):
            self.assertEqual(original.next_batch().tolist(), resumed.next_batch().tolist())

    def test_resume_at_epoch_boundary_uses_next_permutation(self):
        original = ShuffledBatchSampler(20, 5, seed=4)
        for _ in range(4):
            original.next_batch()
        self.assertEqual(original.cursor, original.size)
        resumed = ShuffledBatchSampler.from_state(original.state_dict())
        self.assertEqual(original.next_batch().tolist(), resumed.next_batch().tolist())
        self.assertEqual(original.epoch, resumed.epoch)

    def test_rejects_incomplete_state(self):
        state = ShuffledBatchSampler(9, 2, seed=1).state_dict()
        state["permutation"] = state["permutation"][:-1]
        with self.assertRaisesRegex(ValueError, "complete permutation"):
            ShuffledBatchSampler.from_state(state)


if __name__ == "__main__":
    unittest.main()
