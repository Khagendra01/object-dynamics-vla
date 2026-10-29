import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
from odvla.synthetic import generate
from odvla.data import (make_splits, validate_splits, load_manifest, load_episode, fit_stats,
                        sample_window, encode_history, write_json, safe_path)
from odvla.measurements import mask_metrics, backproject_visible_point
from odvla.adapters import import_libero, attach_masks


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.ds = self.root / "dataset"
        generate(self.ds, episodes=10, steps=20)
        self.m = load_manifest(self.ds)
        self.splits = make_splits(self.ds)
        self.stats = fit_stats(self.ds, self.splits)
        self.ep = load_episode(self.ds, self.m["episodes"][0], self.m)

    def tearDown(self):
        self.temp.cleanup()

    def test_group_disjoint_and_deterministic(self):
        self.assertEqual(self.splits, make_splits(self.ds))
        validate_splits(self.ds, self.splits)
        a = copy.deepcopy(self.splits)
        a["test"].append(a["train"][0])
        with self.assertRaises(ValueError): validate_splits(self.ds, a)

    def test_current_input_cannot_see_future(self):
        before = sample_window(self.ep, 6, 4, 3, self.stats)
        changed = copy.deepcopy(self.ep)
        changed["features"][7:] = 900
        changed["robot"][7:] = -100
        changed["valid"][7:] = True
        changed["actions"][6:] = 3
        after = sample_window(changed, 6, 4, 3, self.stats)
        np.testing.assert_array_equal(before["x"], after["x"])
        self.assertFalse(np.array_equal(before["y"], after["y"]))
        prefix = {k: v[:7] for k, v in self.ep.items() if k != "actions"}
        np.testing.assert_array_equal(before["x"], encode_history(prefix, 6, 4, self.stats))

    def test_action_and_next_state_alignment(self):
        win = sample_window(self.ep, 6, 4, 3, self.stats)
        np.testing.assert_allclose(win["a"].reshape(3, -1)[0],
                                   (self.ep["actions"][6] - self.stats["actions_mean"]) / self.stats["actions_std"])
        expected = (self.ep["features"][7] - self.ep["features"][6]) / self.stats["feature_std"]
        np.testing.assert_allclose(win["y"].reshape(3, 2, 5)[0], expected)

    def test_missing_measurements_not_supervised_as_zero_motion(self):
        changed = copy.deepcopy(self.ep)
        changed["valid"][7, 0] = False
        changed["features"][7, 0] = np.nan
        win = sample_window(changed, 6, 4, 3, self.stats)
        self.assertEqual(win["mask"].reshape(3, 2, 5)[0, 0].sum(), 0)
        self.assertTrue(np.isfinite(win["x"]).all())
        self.assertTrue(np.isfinite(win["y"]).all())

    def test_train_only_normalizer(self):
        before = fit_stats(self.ds, self.splits)
        for entry in self.m["episodes"]:
            if entry["id"] in self.splits["test"]:
                ep = load_episode(self.ds, entry, self.m)
                ep["features"] += 10000
                np.savez_compressed(self.ds / entry["file"], **ep)
        after = fit_stats(self.ds, self.splits)
        for key in before: np.testing.assert_array_equal(before[key], after[key])
        with self.assertRaises(ValueError): validate_splits(self.ds, self.splits)

    def test_mask_geometry_and_empty_masks(self):
        masks = np.zeros((2, 1, 10, 20), bool)
        masks[0, 0, 2:4, 4:8] = True
        f, v = mask_metrics(masks)
        np.testing.assert_allclose(f[0, 0], [0.3, 0.3, 0.2, 0.2, 0.04])
        self.assertTrue(v[0].all()); self.assertFalse(v[1].any())

    def test_backprojection_frames(self):
        mask = np.zeros((480, 640), bool); mask[260, 380] = True
        depth = np.full(mask.shape, 0.6)
        k = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]])
        transform = np.eye(4); transform[:3, :3] = np.diag([1, -1, -1]); transform[:3, 3] = [.4, .1, 1]
        point, valid = backproject_visible_point(mask, depth, k, transform)
        self.assertTrue(valid)
        np.testing.assert_allclose(point, [.46, .08, .4], atol=1e-6)
        _, valid = backproject_visible_point(np.zeros_like(mask), depth, k, transform)
        self.assertFalse(valid)

    def test_episode_validation_and_path_escape(self):
        with self.assertRaises(ValueError): safe_path(self.ds, "../../outside")
        ep = copy.deepcopy(self.ep); ep["timestamps"][2] = ep["timestamps"][1]
        entry = self.m["episodes"][0]
        np.savez_compressed(self.ds / entry["file"], **ep)
        with self.assertRaises(ValueError): load_episode(self.ds, entry, self.m)

    def test_libero_fixture_import(self):
        try: import h5py
        except ImportError: self.skipTest("h5py optional dependency unavailable")
        source = self.root / "fixture.hdf5"
        with h5py.File(source, "w") as f:
            demo = f.create_group("data/demo_0")
            demo.create_dataset("actions", data=np.ones((10, 7), np.float32))
            obs = demo.create_group("obs")
            for key, width in [("joint_states", 7), ("ee_states", 6), ("gripper_states", 2)]:
                obs.create_dataset(key, data=np.zeros((10, width), np.float32))
            obs.create_dataset("agentview_rgb", data=np.zeros((10, 16, 16, 3), np.uint8))
        out = self.root / "imported"
        m = import_libero(source, out, ["cup", "bowl"], 20, "put cup in bowl",
                          ["joint_states", "ee_states", "gripper_states"])
        ep = load_episode(out, m["episodes"][0], m)
        self.assertEqual(ep["robot"].shape, (10, 15))
        self.assertFalse(ep["valid"].any())
        self.assertAlmostEqual(ep["timestamps"][1], .05)

    def test_external_masks_require_provenance(self):
        masks = self.root / "masks"; masks.mkdir()
        write_json(masks / "provenance.json", {"causal": False})
        with self.assertRaises(ValueError): attach_masks(self.ds, masks, self.root / "attached")

    def test_external_mask_attachment_matches_measurements(self):
        masks = self.root / "masks"; masks.mkdir()
        write_json(masks / "provenance.json", {"causal": True, "source": "synthetic exact masks"})
        for entry in self.m["episodes"]:
            ep = load_episode(self.ds, entry, self.m)
            red = (ep["rgb"] == np.array([190, 65, 65])).all(axis=-1)
            blue = (ep["rgb"] == np.array([70, 110, 190])).all(axis=-1)
            np.savez_compressed(masks / (entry["id"] + ".npz"), masks=np.stack([red, blue], axis=1))
        out = self.root / "attached"
        manifest = attach_masks(self.ds, masks, out)
        attached = load_episode(out, manifest["episodes"][0], manifest)
        self.assertTrue(attached["valid"].any())
        self.assertIn("rgb", attached)
        self.assertTrue((attached["features"][attached["valid"]] >= 0).all())


if __name__ == "__main__": unittest.main()
