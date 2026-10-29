import unittest
import numpy as np
import torch

from odvla.diagnostics import accepted_chunk_forecast, one_step_forecast, summarize_actions
from odvla.model import OneStepDynamicsHead


class DiagnosticTests(unittest.TestCase):
    def test_forecast_uses_supplied_accepted_action_chunk(self):
        class Model:
            def encode(self, x):
                return x[:, :1], None

            def dynamics(self, h, actions, no_actions):
                self.seen = actions.detach().cpu().numpy()
                return torch.zeros((1, 20))

        model = Model()
        config = {"history": 2, "horizon": 2, "objects": 2, "features": 5,
                  "action_channels": 2, "no_objects": False, "no_dynamics_actions": False}
        prefix = {"timestamps": np.array([0., .05]), "features": np.zeros((2, 2, 5), np.float32),
                  "valid": np.ones((2, 2, 5), bool), "robot": np.zeros((2, 1), np.float32),
                  "context": np.zeros((2, 0), np.float32)}
        stats = {"feature_mean": np.zeros(5), "feature_std": np.ones(5),
                 "robot_mean": np.zeros(1), "robot_std": np.ones(1),
                 "context_mean": np.zeros(0), "context_std": np.ones(0),
                 "actions_mean": np.array([.5, 0]), "actions_std": np.array([.5, 2.])}
        actual = accepted_chunk_forecast(prefix, [[1., -1.], [0., 1.]], model, config, stats)
        np.testing.assert_allclose(model.seen, [[1., -.5, -1., .5]])
        self.assertEqual(actual.shape, (2, 5))

    def test_summaries_are_channelwise_and_reject_nonfinite(self):
        summary = summarize_actions([[0, -1], [1, 1]])
        self.assertEqual(summary["mean"], [.5, 0.])
        with self.assertRaises(ValueError):
            summarize_actions([[float("nan")]])

    def test_one_step_prediction_is_independent_of_later_actions(self):
        class Encoder:
            def encode(self, x):
                return x[:, :3], None

        config = {"history": 2, "objects": 1, "features": 5,
                  "action_channels": 2, "no_objects": False}
        prefix = {"timestamps": np.array([0., .05]), "features": np.zeros((2, 1, 5), np.float32),
                  "valid": np.ones((2, 1, 5), bool), "robot": np.zeros((2, 1), np.float32),
                  "context": np.zeros((2, 0), np.float32)}
        stats = {"feature_mean": np.zeros(5), "feature_std": np.ones(5),
                 "robot_mean": np.zeros(1), "robot_std": np.ones(1),
                 "context_mean": np.zeros(0), "context_std": np.ones(0),
                 "actions_mean": np.zeros(2), "actions_std": np.ones(2)}
        head = OneStepDynamicsHead(hidden=3, action_channels=2, objects=1, features=5)
        chunk_a = np.array([[.4, -.5], [1., 1.]], np.float32)
        chunk_b = np.array([[.4, -.5], [-100., 100.]], np.float32)
        a = one_step_forecast(prefix, chunk_a[0], Encoder(), head, config, stats)
        b = one_step_forecast(prefix, chunk_b[0], Encoder(), head, config, stats)
        np.testing.assert_array_equal(a, b)
        with self.assertRaises(ValueError):
            one_step_forecast(prefix, chunk_a, Encoder(), head, config, stats)


if __name__ == "__main__":
    unittest.main()
