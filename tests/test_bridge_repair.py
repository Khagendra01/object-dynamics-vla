import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from odvla.vla.bridge_repair import distillation


class DistillationTest(unittest.TestCase):
    def test_teacher_and_student_receive_identical_states_and_time(self):
        calls = []
        bridge_weight = torch.tensor(0.5, requires_grad=True)
        teacher = ("teacher", None, None)
        student = ("student", None, None)

        def velocity(model, which, unused_value, unused_mask, x, t):
            calls.append((which, x.detach().clone(), t.detach().clone(), torch.is_grad_enabled()))
            return torch.ones_like(x) * (bridge_weight if which == "student" else 0.1)

        batch = {"actions": torch.randn(2, 50, 7), "action_valid": torch.ones(2, 50, dtype=torch.bool)}
        noise = torch.randn(2, 50, 32); times = torch.tensor([0.3, 0.7])
        with patch("odvla.vla.bridge_repair.expert_velocity", side_effect=velocity):
            a, b = distillation(SimpleNamespace(model=None), batch, teacher, student, noise, times, 3)
            (a + b).backward()
        # Three detached rollout calls, then two teacher and two student evaluations.
        self.assertEqual([c[0] for c in calls], ["student"] * 3 + ["teacher", "teacher", "student", "student"])
        for teacher_call, student_call in ((calls[3], calls[5]), (calls[4], calls[6])):
            torch.testing.assert_close(teacher_call[1], student_call[1], rtol=0, atol=0)
            torch.testing.assert_close(teacher_call[2], student_call[2], rtol=0, atol=0)
            self.assertFalse(teacher_call[3]); self.assertTrue(student_call[3])
        self.assertIsNotNone(bridge_weight.grad)
        self.assertNotEqual(float(bridge_weight.grad), 0)


if __name__ == "__main__":
    unittest.main()
