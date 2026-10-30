"""Noise-at-one flow convention matching the pinned LeRobot implementation."""
import torch


def flow_inputs(actions, noise, time):
    t = time.reshape(-1, 1, 1)
    return (1 - t) * actions + t * noise, noise - actions


def masked_mean(values, valid):
    mask = torch.broadcast_to(valid, values.shape)
    return torch.where(mask, values, torch.zeros_like(values)).sum() / mask.sum().clamp_min(1)
