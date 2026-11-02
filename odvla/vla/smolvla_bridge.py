"""Observation-conditioned prefixes for the native LeRobot SmolVLA expert."""
from types import SimpleNamespace

import torch
from torch import nn


class PrefixCache:
    """Immutable prefix cache; suffix append is functional and keeps autograd."""

    def __init__(self, keys, values):
        if len(keys) != len(values) or not keys:
            raise ValueError("Expected one key/value pair per layer")
        self.layers = [SimpleNamespace(keys=k, values=v) for k, v in zip(keys, values)]

    def update(self, keys, values, layer_idx, *args, **kwargs):
        layer = self.layers[layer_idx]
        return torch.cat((layer.keys, keys), dim=-2), torch.cat((layer.values, values), dim=-2)


def expert_velocity(model, keys, values, prefix_valid, noisy_actions, timestep):
    """Execute the pinned expert with differentiable per-layer prefix K/V."""
    from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

    suffix, suffix_valid, suffix_attention = model.embed_suffix(noisy_actions, timestep)
    batch, length = suffix_valid.shape
    if prefix_valid.dtype != torch.bool or prefix_valid.shape[0] != batch:
        raise ValueError("Invalid prefix mask")
    n_layers = model.vlm_with_expert.num_vlm_layers
    expected = (batch, model.vlm_with_expert.num_key_value_heads, prefix_valid.shape[1],
                model.vlm_with_expert.config.text_config.head_dim)
    if len(keys) != n_layers or len(values) != n_layers:
        raise ValueError("Wrong number of prefix layers")
    if any(tuple(x.shape) != expected for x in [*keys, *values]):
        raise ValueError(f"Expected each prefix K/V to have shape {expected}")
    prefix_mask = prefix_valid[:, None, :].expand(batch, length, -1)
    suffix_mask = make_att_2d_masks(suffix_valid, suffix_attention)
    attention = torch.cat((prefix_mask, suffix_mask), dim=2)
    positions = prefix_valid.sum(dim=-1, keepdim=True) + torch.cumsum(suffix_valid, dim=1) - 1
    (_, out), _ = model.vlm_with_expert.forward(
        attention_mask=attention, position_ids=positions, past_key_values=PrefixCache(keys, values),
        inputs_embeds=[None, suffix], use_cache=True)
    return model.action_out_proj(out[:, -model.config.chunk_size:].float())


class MuseSamPrefix(nn.Module):
    """Resample Muse tokens and fuse causal state/SAM into native attention K/V."""

    def __init__(self, muse_width=6656, width=512, latents=64, layers=16, heads=5, head_dim=64,
                 objects=2, metrics=5, history=4):
        super().__init__()
        self.layers, self.heads, self.head_dim = layers, heads, head_dim
        self.muse = nn.Sequential(nn.LayerNorm(muse_width), nn.Linear(muse_width, width))
        self.queries = nn.Parameter(torch.randn(latents, width) * .02)
        self.resample = nn.MultiheadAttention(width, 8, batch_first=True)
        self.state = nn.Linear(960, width)
        self.sam = nn.Linear(metrics * 4, width)
        self.slot = nn.Embedding(objects, width)
        self.history = nn.Embedding(history, width)
        self.fuse = nn.MultiheadAttention(width, 8, batch_first=True)
        self.norm = nn.LayerNorm(width)
        self.kv = nn.ModuleList(nn.Linear(width, 2 * heads * head_dim) for _ in range(layers))

    def forward(self, muse, muse_valid, state_embedding, sam_values, sam_deltas, sam_valid, sam_age,
                history_valid, use_sam=True):
        if not muse_valid.any(dim=1).all():
            raise ValueError("Muse context cannot be empty")
        vision = self.muse(torch.where(muse_valid[..., None], muse, 0).float())
        q = self.queries[None].expand(len(muse), -1, -1)
        q = q + self.resample(q, vision, vision, key_padding_mask=~muse_valid, need_weights=False)[0]
        batch, hist, objects, _ = sam_values.shape
        valid = sam_valid & history_valid[:, :, None, None]
        parts = (torch.where(valid, sam_values, 0), torch.where(valid, sam_deltas, 0), valid.float(),
                 torch.where(valid, sam_age, 0))
        st = self.sam(torch.cat(parts, dim=-1))
        st = st + self.slot(torch.arange(objects, device=st.device))[None, None]
        st = st + self.history(torch.arange(hist, device=st.device))[None, :, None]
        state = self.state(state_embedding.float())[:, None]
        context = torch.cat((state, st.reshape(batch, hist * objects, -1)), dim=1)
        sam_mask = valid.any(dim=-1).reshape(batch, hist * objects) if use_sam else torch.zeros(
            batch, hist * objects, dtype=torch.bool, device=st.device)
        mask = torch.cat((torch.ones(batch, 1, dtype=torch.bool, device=st.device), sam_mask), dim=1)
        q = self.norm(q + self.fuse(q, context, context, key_padding_mask=~mask, need_weights=False)[0])
        pairs = [projection(q).reshape(batch, q.shape[1], 2, self.heads, self.head_dim) for projection in self.kv]
        keys = [p[:, :, 0].transpose(1, 2) for p in pairs]
        values = [p[:, :, 1].transpose(1, 2) for p in pairs]
        return keys, values, torch.ones(batch, q.shape[1], dtype=torch.bool, device=q.device)
