import unittest
import numpy as np
import torch
from odvla.vla.action_expert import MuseActionExpert
from odvla.vla.dataset import observation_window
from odvla.vla.flow import flow_inputs


class VLAContractTests(unittest.TestCase):
    def test_muse_requires_fixed_chat_date(self):
        from PIL import Image
        from odvla.vla.muse_context import FrozenMuseTokens
        model = FrozenMuseTokens.__new__(FrozenMuseTokens)
        model.torch = torch
        with self.assertRaisesRegex(ValueError, "fixed chat-template"):
            model.encode(Image.new("RGB", (2, 2)), "task", current_date="")

    def test_incremental_tracker_rejects_invalid_configuration(self):
        from odvla.sam_incremental import IncrementalPointTracker
        with self.assertRaises(ValueError):
            IncrementalPointTracker(None, "unused", [(1, 2), (3, 4)], [1, 1])
        with self.assertRaises(ValueError):
            IncrementalPointTracker(None, "unused", [(1, 2)], [1], capacity=201)

    def test_comparison_requires_motor_checkpoint(self):
        from scripts.train_muse_vla import train
        with self.assertRaisesRegex(ValueError, "common motor checkpoint"):
            train("unused", "unused", "unused", "M1", 7, "cpu", steps=1)
        with self.assertRaisesRegex(ValueError, "start from scratch"):
            train("unused", "unused", "unused", "motor", 7, "cpu", steps=1,
                  motor_checkpoint="unused")

    def episode(self):
        t = 5
        return {"features": np.arange(t * 2 * 5, dtype=np.float32).reshape(t, 2, 5),
            "valid": np.ones((t, 2, 5), bool), "robot": np.arange(t * 3, dtype=np.float32).reshape(t, 3),
            "actions": np.arange(t * 2, dtype=np.float32).reshape(t, 2), "timestamps": np.arange(t) / 20}

    def stats(self):
        return {"feature_mean": np.zeros(5), "feature_std": np.ones(5), "robot_mean": np.zeros(3),
                "robot_std": np.ones(3), "actions_mean": np.zeros(2), "actions_std": np.ones(2)}

    def test_start_end_padding_and_future_isolation(self):
        ep = self.episode(); stats = self.stats()
        start = observation_window(ep, 0, stats, 4, 7)
        self.assertEqual(start["history_valid"].tolist(), [False, False, False, True])
        self.assertEqual(start["action_valid"].tolist(), [True] * 5 + [False] * 2)
        end = observation_window(ep, 4, stats, 4, 7)
        self.assertFalse(end["change_valid"].any())
        self.assertEqual(end["action_valid"].tolist(), [True] + [False] * 6)
        before = observation_window(ep, 2, stats, 4, 7)
        ep["features"][3:] = -999; ep["robot"][3:] = -999
        after = observation_window(ep, 2, stats, 4, 7)
        for key in ("robot", "sam_values", "sam_deltas", "sam_valid", "sam_age"):
            np.testing.assert_array_equal(before[key], after[key])

    def test_flow_endpoints(self):
        a = torch.tensor([[[2.0]]]); n = torch.tensor([[[5.0]]])
        x0, target = flow_inputs(a, n, torch.tensor([0.])); x1, _ = flow_inputs(a, n, torch.tensor([1.]))
        self.assertEqual(x0.item(), 2); self.assertEqual(x1.item(), 5); self.assertEqual(target.item(), 3)

    def test_shared_gradients_masking_and_one_action(self):
        torch.manual_seed(5)
        m = MuseActionExpert(muse_width=6, width=16, heads=4, ff_width=32, context_layers=1,
                             expert_layers=2, history=4, horizon=3, robot_dim=3, action_dim=2)
        muse = torch.randn(1, 3, 6, requires_grad=True); mask = torch.tensor([[True, True, False]])
        robot = torch.randn(1, 4, 3); hv = torch.tensor([[False, True, True, True]])
        values = torch.randn(1, 4, 2, 5); valid = torch.ones_like(values, dtype=torch.bool)
        context, cmask = m.context(muse, mask, robot, hv, torch.tensor([[0., 0., .05, .1]]), values, values, valid, values)
        padded_muse = muse.detach().clone(); padded_muse[:, 2] = 99999
        padded_robot = robot.clone(); padded_robot[:, 0] = -99999
        context2, _ = m.context(padded_muse, mask, padded_robot, hv, torch.tensor([[0., 0., .05, .1]]), values, values, valid, values)
        torch.testing.assert_close(context[cmask], context2[cmask])
        no_sam, _ = m.context(muse, mask, robot, hv, torch.tensor([[0., 0., .05, .1]]), values, values, valid, values, use_sam=False)
        no_sam2, _ = m.context(muse, mask, robot, hv, torch.tensor([[0., 0., .05, .1]]), values * 100, values * 100, valid, values * 100, use_sam=False)
        torch.testing.assert_close(no_sam, no_sam2)
        actions = torch.randn(1, 3, 2); av = torch.tensor([[True, True, False]])
        noise = torch.randn_like(actions); change = torch.randn(1, 2, 5); cv = torch.ones_like(change, dtype=torch.bool)
        total, flow, dyn = m.losses(context, cmask, actions, av, noise, torch.tensor([.3]), change, cv)
        flow.backward(retain_graph=True)
        self.assertGreater(m.expert[0].attn.in_proj_weight.grad.abs().sum().item(), 0)
        m.zero_grad(); dyn.backward(retain_graph=True)
        self.assertGreater(m.expert[0].attn.in_proj_weight.grad.abs().sum().item(), 0)
        self.assertIsNone(muse.grad)
        m.zero_grad()
        altered = actions.clone(); altered[:, 2] = 10000
        changed, _, _ = m.losses(context, cmask, altered, av, noise, torch.tensor([.3]), change, cv)
        torch.testing.assert_close(total, changed)
        cv_masked = cv.clone(); cv_masked[:, 0, 0] = False
        first = m.losses(context, cmask, actions, av, noise, torch.tensor([.3]), change, cv_masked)[2]
        alter_change = change.clone(); alter_change[:, 0, 0] = 123456
        second = m.losses(context, cmask, actions, av, noise, torch.tensor([.3]), alter_change, cv_masked)[2]
        torch.testing.assert_close(first, second)
        torch.testing.assert_close(m.dynamics(context, cmask, actions[:, 0]), m.dynamics(context, cmask, altered[:, 0]))
        m.zero_grad(); total.backward()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4); opt.step()
        sample = m.sample(context.detach(), cmask, torch.randn_like(actions), steps=2)
        self.assertEqual(sample.shape, (1, 3, 2)); self.assertTrue(torch.isfinite(sample).all())


if __name__ == "__main__": unittest.main()
