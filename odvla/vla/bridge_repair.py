"""Teacher-matched bridge objective on demonstration and student flow states."""
import torch
import torch.nn.functional as F

from .smolvla_bridge import expert_velocity


def distillation(policy, batch, teacher_kv, student_kv, noise, time, rollout_depth):
    model = policy.model
    actions = F.pad(batch["actions"], (0, 25))
    x_interp = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
    # Detached student rollout states keep memory bounded; both policies see the exact same x_t.
    with torch.no_grad():
        x_student = noise.clone()
        for k in range(rollout_depth):
            current = torch.full((len(noise),), 1 - k / 10, device=noise.device)
            x_student = x_student - expert_velocity(model, *student_kv, x_student, current) / 10
        t_student = torch.full((len(noise),), 1 - rollout_depth / 10, device=noise.device)
        teacher_interp = expert_velocity(model, *teacher_kv, x_interp, time)
        teacher_student = expert_velocity(model, *teacher_kv, x_student, t_student)
    student_interp = expert_velocity(model, *student_kv, x_interp, time)
    student_student = expert_velocity(model, *student_kv, x_student.detach(), t_student)
    weights = torch.tensor([1., 1., 1., 1., 1., 1., 3.], device=noise.device)[None, None]
    valid = batch["action_valid"][..., None].float()
    def loss(a, b):
        return (((a[:, :, :7] - b[:, :, :7]) ** 2) * valid * weights).sum() / (valid.sum() * weights.sum()).clamp_min(1)
    return loss(student_interp, teacher_interp), loss(student_student, teacher_student)
