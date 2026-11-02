import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from odvla.demo import SKILLS, action_noise, overlay_frame, parse_muse_selection, route_token_options, run_episode, validate_run_name, write_viewer


class NativeDemoTests(unittest.TestCase):
    def test_router_constrains_tokens_and_terminates(self):
        sequences = [[1, 2, 3], [1, 4, 5]]
        self.assertEqual(route_token_options([], sequences, [9]), [1])
        self.assertEqual(route_token_options([1], sequences, [9]), [2, 4])
        self.assertEqual(route_token_options([1, 2, 3], sequences, [9]), [9])
        with self.assertRaises(ValueError):
            route_token_options([8], sequences, [9])

    def test_muse_only_routes_registered_skills(self):
        self.assertEqual(parse_muse_selection('{"skill_id":"milk_in_basket","reason":"Requested task"}')["skill_id"], "milk_in_basket")
        self.assertEqual(parse_muse_selection('```json\n{"skill_id":"unsupported","reason":"No such skill"}\n```')["skill_id"], "unsupported")
        for text in ('{"skill_id":"/tmp/execute","reason":"x"}',
                     '{"skill_id":"milk_in_basket","reason":"x","instruction":"run something"}',
                     'milk_in_basket', '[]', '{"skill_id":3,"reason":"x"}'):
            with self.assertRaises(ValueError):
                parse_muse_selection(text)
        for name in ("../overwrite", "", "/tmp/a", "bad name"):
            with self.assertRaises(ValueError):
                validate_run_name(name)

    def test_tracking_cannot_change_motor_inputs_or_actions(self):
        def execute(tracking):
            seen = []
            def observe():
                return {"jpeg": b"same-image", "robot": np.arange(15, dtype=np.float32), "tracking": tracking,
                        "muse_action": "must never reach the controller"}
            def predict(*, jpeg, robot, instruction, noise):
                seen.append((jpeg, robot.copy(), instruction, noise.copy()))
                return {"action": np.array([2, -.5, 0, 0, 0, 0, -2], np.float32),
                        "noise_sha256": hashlib.sha256(noise.tobytes()).hexdigest()}
            actions = []
            def advance(action):
                actions.append(action)
                return {"success": len(actions) == 2, "done": False}
            report = run_episode(SKILLS["milk_in_basket"], observe, predict, advance, max_steps=5)
            return report, actions, seen
        left, la, ls = execute({"status": "tracking", "arbitrary": [1, 2, 3]})
        right, ra, rs = execute({"status": "unavailable", "error": "SAM crashed"})
        self.assertEqual(la, ra)
        self.assertEqual(left["status"], "success")
        self.assertEqual(right["executed_steps"], 2)
        self.assertFalse(left["bridge_used"])
        self.assertEqual(la[0], [1., -.5, 0., 0., 0., 0., -1.])
        for a, b in zip(ls, rs):
            self.assertEqual(a[0], b[0]); self.assertEqual(a[2], b[2])
            np.testing.assert_array_equal(a[1], b[1]); np.testing.assert_array_equal(a[3], b[3])
        np.testing.assert_array_equal(ls[0][3], action_noise(0))

    def test_bad_action_or_noise_stops_before_execution(self):
        executed = []
        observe = lambda: {"jpeg": b"rgb", "robot": np.zeros(15, np.float32)}
        for prediction in ({"action": [float("nan")] * 7, "noise_sha256": "bad"},
                           {"action": [0.] * 7, "noise_sha256": "bad"}):
            result = run_episode(SKILLS["milk_in_basket"], observe, lambda **kwargs: prediction,
                                 lambda action: executed.append(action), max_steps=1)
            self.assertEqual(result["status"], "error")
            self.assertEqual(result["executed_steps"], 0)
        self.assertEqual(executed, [])

    def test_viewer_escapes_user_and_model_strings(self):
        with tempfile.TemporaryDirectory() as temp:
            report = {"status": "step_cap", "executed_steps": 0, "trace": [],
                      "request": '<script>alert("x")</script>', "selection": {"reason": "<img src=x>"}}
            page = write_viewer(temp, report).read_text()
            self.assertNotIn('<script>alert(', page)
            self.assertIn("&lt;script&gt;", page)
            self.assertIn("&lt;img src=x&gt;", page)
            rgb = np.zeros((32, 32, 3), np.uint8)
            overlay = overlay_frame(rgb, np.ones((2, 8, 8), bool), dry_run=True)
            self.assertEqual(overlay.size, (32, 144))
            np.testing.assert_array_equal(rgb, np.zeros_like(rgb))


if __name__ == "__main__":
    unittest.main()
