"""Small reference architecture. Frozen VLM features enter through cached context.

No pretrained motor skills are bundled. Temporal history is flattened in v0.1;
replace the encoder with temporal attention only after the first ablations work.
"""
import torch
from torch import nn


class ObjectDynamicsPolicy(nn.Module):
    def __init__(self, input_dim, object_input_dim, metric_dim, context_dim,
                 action_dim, target_dim, hidden=128):
        super().__init__()
        self.object_input_dim = object_input_dim
        self.metric_dim = metric_dim
        self.context_dim = context_dim
        # One gate per measurement type, shared across objects and history.
        # With context, selection is instruction/scene-conditioned. Without it,
        # this is state-conditioned numeric selection, not language grounding.
        self.selector = nn.Linear(context_dim or input_dim, metric_dim)
        nn.init.zeros_(self.selector.weight)
        nn.init.zeros_(self.selector.bias)
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden), nn.SiLU(),
                                     nn.Linear(hidden, hidden), nn.SiLU())
        self.action_head = nn.Sequential(nn.Linear(hidden + action_dim + 1, hidden),
                                         nn.SiLU(), nn.Linear(hidden, action_dim))
        self.dynamics_head = nn.Sequential(nn.Linear(hidden + action_dim, hidden),
                                           nn.SiLU(), nn.Linear(hidden, target_dim))

    def encode(self, x):
        selector_input = x[:, -self.context_dim:] if self.context_dim else x
        gate = torch.sigmoid(self.selector(selector_input))
        repeats = self.object_input_dim // self.metric_dim
        gated = x[:, :self.object_input_dim] * gate.repeat(1, repeats)
        return self.encoder(torch.cat([gated, x[:, self.object_input_dim:]], dim=-1)), gate

    def velocity(self, h, noisy_actions, tau):
        return self.action_head(torch.cat([h, noisy_actions, tau], dim=-1))

    def dynamics(self, h, recorded_actions, no_actions=False):
        if no_actions:
            recorded_actions = torch.zeros_like(recorded_actions)
        return self.dynamics_head(torch.cat([h, recorded_actions], dim=-1))

    @torch.no_grad()
    def sample_actions(self, h, noise, steps=20):
        if steps < 1:
            raise ValueError("Flow integration needs steps >= 1")
        action = noise.clone()
        for i in range(steps):
            tau = torch.full((len(action), 1), i / steps, dtype=action.dtype, device=action.device)
            action = action + self.velocity(h, action, tau) / steps
        return action


class OneStepDynamicsHead(nn.Module):
    """Predict one observation delta from shared state and one executed action."""
    def __init__(self, hidden, action_channels, objects, features):
        super().__init__()
        self.action_channels = action_channels
        self.net = nn.Sequential(nn.Linear(hidden + action_channels, hidden), nn.SiLU(),
                                 nn.Linear(hidden, objects * features))

    def forward(self, encoded, executed_action):
        if executed_action.ndim != 2 or executed_action.shape[1] != self.action_channels:
            raise ValueError("One-step head accepts exactly one action per observation")
        return self.net(torch.cat([encoded, executed_action], dim=-1))


def masked_mse(predicted, target, mask):
    return ((predicted - target).square() * mask).sum() / mask.sum().clamp_min(1)
