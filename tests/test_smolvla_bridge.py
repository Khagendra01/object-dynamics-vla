import unittest

import torch

from odvla.vla.smolvla_bridge import MuseSamPrefix, PrefixCache


class BridgeTest(unittest.TestCase):
    def test_query_normalization_bounds_fusion_inputs_and_preserves_state_dict(self):
        torch.manual_seed(3)
        kwargs = dict(muse_width=12, width=16, latents=3, layers=2, heads=2, head_dim=4)
        legacy = MuseSamPrefix(**kwargs).eval()
        model = MuseSamPrefix(**kwargs, normalize_queries=True).eval()
        model.load_state_dict(legacy.state_dict(), strict=True)
        seen = []
        scale = model.resample.register_forward_hook(lambda module, args, output: (output[0] * 1000, output[1]))
        capture = model.fuse.register_forward_pre_hook(lambda module, args: seen.append(args[0].detach()))
        try:
            model(torch.randn(1, 4, 12), torch.ones(1, 4, dtype=torch.bool), torch.randn(1, 960),
                  torch.randn(1, 4, 2, 5), torch.randn(1, 4, 2, 5), torch.ones(1, 4, 2, 5, dtype=torch.bool),
                  torch.zeros(1, 4, 2, 5), torch.ones(1, 4, dtype=torch.bool))
        finally:
            scale.remove(); capture.remove()
        self.assertEqual(len(seen), 1)
        torch.testing.assert_close(seen[0].mean(dim=-1), torch.zeros(1, 3), atol=1e-5, rtol=0)
        torch.testing.assert_close(seen[0].square().mean(dim=-1), torch.ones(1, 3), atol=1e-4, rtol=0)

    def test_masked_features_do_not_affect_prefix(self):
        torch.manual_seed(1)
        model = MuseSamPrefix(muse_width=12, width=16, latents=3, layers=2, heads=2, head_dim=4).eval()
        muse = torch.randn(1, 4, 12)
        muse_valid = torch.tensor([[True, True, False, False]])
        state = torch.randn(1, 960)
        values = torch.randn(1, 4, 2, 5)
        delta = torch.randn_like(values)
        valid = torch.ones_like(values, dtype=torch.bool)
        valid[:, :, 1] = False
        age = torch.randn_like(values)
        history = torch.ones(1, 4, dtype=torch.bool)
        args = (muse, muse_valid, state, values, delta, valid, age, history)
        keys, _, mask = model(*args)
        muse2 = muse.clone(); muse2[:, 2:] = 1000
        values2 = values.clone(); values2[:, :, 1] = 1000
        delta2 = delta.clone(); delta2[:, :, 1] = -1000
        age2 = age.clone(); age2[:, :, 1] = 1000
        other, _, _ = model(muse2, muse_valid, state, values2, delta2, valid, age2, history)
        torch.testing.assert_close(keys[0], other[0])
        self.assertEqual(tuple(keys[0].shape), (1, 2, 3, 4))
        self.assertTrue(mask.all())
        sum(x.square().sum() for x in keys).backward()
        self.assertGreater(float(model.muse[1].weight.grad.norm()), 0)
        self.assertGreater(float(model.sam.weight.grad.norm()), 0)

    def test_cache_append_is_functional_and_differentiable(self):
        key = torch.randn(1, 2, 3, 4, requires_grad=True)
        value = torch.randn(1, 2, 3, 4, requires_grad=True)
        cache = PrefixCache([key], [value])
        suffix = torch.randn(1, 2, 2, 4, requires_grad=True)
        combined, _ = cache.update(suffix, suffix, 0)
        self.assertEqual(combined.shape[-2], 5)
        self.assertEqual(cache.layers[0].keys.shape[-2], 3)
        combined.sum().backward()
        self.assertTrue(torch.equal(key.grad, torch.ones_like(key)))


if __name__ == "__main__":
    unittest.main()
