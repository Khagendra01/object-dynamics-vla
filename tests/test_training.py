import tempfile
import unittest
from pathlib import Path
import numpy as np
try:
    import torch
except ImportError:
    torch = None
from odvla.synthetic import generate
from odvla.data import make_splits, write_json, read_json, load_manifest, load_episode


@unittest.skipIf(torch is None, "Install PyTorch for neural training tests")
class TrainingTests(unittest.TestCase):
    def test_both_losses_reach_shared_encoder(self):
        from odvla.model import ObjectDynamicsPolicy, masked_mse
        torch.manual_seed(0)
        model = ObjectDynamicsPolicy(43, 40, 5, 0, 6, 30, 16)
        x = torch.randn(3, 43); a = torch.randn(3, 6)
        h, _ = model.encode(x)
        loss = masked_mse(model.dynamics(h, a), torch.ones(3, 30), torch.ones(3, 30))
        loss.backward()
        self.assertGreater(model.encoder[0].weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.selector.weight.grad.abs().sum().item(), 0)
        model.zero_grad(); h, _ = model.encode(x)
        model.velocity(h, a, torch.ones(3, 1) * .5).square().mean().backward()
        self.assertGreater(model.encoder[0].weight.grad.abs().sum().item(), 0)

    def test_zero_valid_loss_is_finite(self):
        from odvla.model import masked_mse
        value = masked_mse(torch.ones(3), torch.zeros(3), torch.zeros(3))
        self.assertEqual(value.item(), 0)

    def test_train_evaluate_and_prefix_inference(self):
        from odvla.experiment import train, evaluate, load_run
        from odvla.inference import predict
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); ds = root / "ds"
            generate(ds, episodes=10, steps=24)
            split = root / "split.json"; write_json(split, make_splits(ds))
            run = root / "run"
            train(ds, split, run, epochs=3, hidden=32, batch_size=32)
            self.assertFalse(read_json(run / "data_summary.json")["test_accessed"])
            self.assertFalse((run / "test_metrics.json").exists())
            metrics = evaluate(ds, run)
            self.assertTrue(np.isfinite(metrics["dynamics"]["normalized_mse"]))
            model, cfg, stats = load_run(run)
            m = load_manifest(ds); ep = load_episode(ds, m["episodes"][0], m)
            prefix = {k: ep[k][:8] for k in ("features", "valid", "robot", "timestamps", "context")}
            result = predict(prefix, model, cfg, stats)
            self.assertEqual(np.asarray(result["actions_source_units"]).shape, (4, 2))
            self.assertEqual(result["execution_status"], "not executed; no robot driver included")


if __name__ == "__main__": unittest.main()
