import unittest
from odvla.vla.batches import ShuffledWindows, validate_resume_contract


class ShuffledWindowTests(unittest.TestCase):
    def test_each_epoch_covers_every_window(self):
        sampler = ShuffledWindows(17, seed=7)
        for _ in range(4):
            values = sampler.take(17)
            self.assertEqual(sorted(values), list(range(17)))
            self.assertNotEqual(values, list(range(17)))

    def test_resume_matches_across_epoch_and_batch_boundaries(self):
        sampler = ShuffledWindows(17, seed=7)
        sampler.take(13)
        saved = sampler.state_dict()
        expected = [sampler.take(n) for n in (8, 64, 1, 17, 3)]
        restored = ShuffledWindows(17, seed=999)
        restored.load_state_dict(saved)
        actual = [restored.take(n) for n in (8, 64, 1, 17, 3)]
        self.assertEqual(actual, expected)
        self.assertEqual(restored.state_dict(), sampler.state_dict())

    def test_invalid_or_wrong_dataset_state_is_rejected(self):
        sampler = ShuffledWindows(17)
        saved = sampler.state_dict()
        saved["order"][0] = saved["order"][1]
        with self.assertRaises(ValueError):
            sampler.load_state_dict(saved)
        with self.assertRaises(ValueError):
            ShuffledWindows(18).load_state_dict(ShuffledWindows(17).state_dict())
        with self.assertRaises(ValueError):
            sampler.take(0)

    def test_resume_only_allows_update_target_extension(self):
        saved = {"updates": 1000, "batch_size": 8, "cache": "original"}
        validate_resume_contract(saved, dict(saved, updates=2000))
        for request in (dict(saved, updates=999), dict(saved, batch_size=16),
                        dict(saved, cache="other")):
            with self.assertRaises(ValueError):
                validate_resume_contract(saved, request)


if __name__ == "__main__":
    unittest.main()
