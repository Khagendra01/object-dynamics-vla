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
        from odvla.one_step import train_one_step, load_one_step
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); ds = root / "ds"
            generate(ds, episodes=10, steps=24)
            split = root / "split.json"; write_json(split, make_splits(ds))
            run = root / "run"
            train(ds, split, run, epochs=3, hidden=32, batch_size=32)
            self.assertFalse(read_json(run / "data_summary.json")["test_accessed"])
            self.assertFalse((run / "test_metrics.json").exists())
            monitor_dir = root / "monitor"
            monitor_meta = train_one_step(ds, run, monitor_dir, epochs=2, batch_size=32)
            self.assertFalse(monitor_meta["test_accessed"])
            _, parent_config, _ = load_run(run)
            self.assertIsNotNone(load_one_step(monitor_dir, run, parent_config))
            bad_config = dict(parent_config, dataset_fingerprint="wrong")
            with self.assertRaises(ValueError):
                load_one_step(monitor_dir, run, bad_config)
            metrics = evaluate(ds, run)
            self.assertTrue(np.isfinite(metrics["dynamics"]["normalized_mse"]))
            model, cfg, stats = load_run(run)
            m = load_manifest(ds); ep = load_episode(ds, m["episodes"][0], m)
            prefix = {k: ep[k][:8] for k in ("features", "valid", "robot", "timestamps", "context")}
            result = predict(prefix, model, cfg, stats)
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                autocast_result = predict(prefix, model, cfg, stats)
            np.testing.assert_allclose(result["actions_source_units"], autocast_result["actions_source_units"])
            np.testing.assert_allclose(result["predicted_numeric_delta"], autocast_result["predicted_numeric_delta"])
            self.assertEqual(np.asarray(result["actions_source_units"]).shape, (4, 2))
            self.assertEqual(result["execution_status"], "not executed; no robot driver included")

    def test_development_only_experimental_perception_is_explicit(self):
        from odvla.data import validate_splits
        from odvla.experiment import train
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); ds = root / "ds"
            generate(ds, episodes=10, steps=24)
            manifest = read_json(ds / "manifest.json")
            manifest["perception"] = {"causal": False, "experimental_prefix_audited": True}
            write_json(ds / "manifest.json", manifest)
            splits = make_splits(ds)
            splits["train"].extend(splits["test"])
            splits["test"] = []
            with self.assertRaises(ValueError):
                validate_splits(ds, splits)
            splits["allow_empty_test"] = True
            validate_splits(ds, splits)
            path = root / "splits.json"; write_json(path, splits)
            with self.assertRaises(ValueError):
                train(ds, path, root / "blocked", epochs=1, hidden=16)
            run = root / "run"
            train(ds, path, run, epochs=1, hidden=16, allow_experimental_perception=True,
                  checkpoint_selection="flow")
            config = read_json(run / "config.json")
            best = read_json(run / "best_validation.json")
            self.assertEqual(config["checkpoint_selection"], "flow")
            self.assertAlmostEqual(best["selection_objective"], best["metrics"]["flow_mse"])


if __name__ == "__main__": unittest.main()
