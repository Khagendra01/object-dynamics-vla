"""SmolVLA-inspired compact action expert; not the official architecture."""
import torch
from torch import nn
import torch.nn.functional as F

from .flow import flow_inputs, masked_mean


class ExpertBlock(nn.Module):
    def __init__(self, width, heads, ff_width, cross):
        super().__init__()
        self.cross = cross
        self.norm1 = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, ff_width), nn.GELU(), nn.Linear(ff_width, width))

    def forward(self, x, context, context_valid, token_valid):
        query = self.norm1(x)
        if self.cross:
            y = self.attn(query, context, context, key_padding_mask=~context_valid, need_weights=False)[0]
        else:
            n = x.shape[1]
            causal = torch.triu(torch.ones(n, n, dtype=torch.bool, device=x.device), diagonal=1)
            y = self.attn(query, query, query, attn_mask=causal, key_padding_mask=~token_valid, need_weights=False)[0]
        x = x + y
        return x + self.ff(self.norm2(x))


class MuseActionExpert(nn.Module):
    def __init__(self, muse_width=6656, width=512, heads=8, ff_width=2048,
                 context_layers=2, expert_layers=8, history=4, horizon=50,
                 robot_dim=15, action_dim=7, objects=2, metrics=5):
        super().__init__()
        self.horizon, self.objects, self.metrics = horizon, objects, metrics
        self.muse_proj = nn.Sequential(nn.LayerNorm(muse_width), nn.Linear(muse_width, width))
        self.robot_proj = nn.Linear(robot_dim, width)
        self.sam_proj = nn.Linear(metrics * 4, width)
        self.slot = nn.Embedding(objects, width)
        self.history_position = nn.Embedding(history, width)
        self.history_time = nn.Linear(1, width)
        self.missing_sam = nn.Parameter(torch.zeros(width))
        layer = nn.TransformerEncoderLayer(width, heads, ff_width, dropout=0, batch_first=True, norm_first=True)
        self.context_encoder = nn.TransformerEncoder(layer, context_layers, enable_nested_tensor=False)
        self.action_proj = nn.Linear(action_dim, width)
        self.flow_time = nn.Sequential(nn.Linear(1, width), nn.SiLU(), nn.Linear(width, width))
        self.action_position = nn.Embedding(horizon, width)
        self.dynamics_task = nn.Parameter(torch.zeros(width))
        self.expert = nn.ModuleList(ExpertBlock(width, heads, ff_width, cross=bool(i % 2)) for i in range(expert_layers))
        self.action_out = nn.Linear(width, action_dim)
        self.dynamics_out = nn.Linear(width, objects * metrics)

    def context(self, muse, muse_valid, robot, history_valid, history_timestamps, sam_values, sam_deltas, sam_valid, sam_age, use_sam=True):
        b, h, o, f = sam_values.shape
        safe_muse = torch.where(muse_valid[..., None], muse, torch.zeros_like(muse)).detach()
        vision = self.muse_proj(safe_muse.float())
        relative_time = (history_timestamps - history_timestamps[:, -1:]).float()
        pos = self.history_position(torch.arange(h, device=robot.device))[None] + self.history_time(relative_time[..., None])
        robot = torch.where(history_valid[..., None], robot, torch.zeros_like(robot))
        rt = self.robot_proj(robot) + pos
        safe_value = torch.where(sam_valid, sam_values, torch.zeros_like(sam_values))
        safe_delta = torch.where(sam_valid, sam_deltas, torch.zeros_like(sam_deltas))
        safe_age = torch.where(history_valid[:, :, None, None], sam_age, torch.zeros_like(sam_age))
        st = self.sam_proj(torch.cat((safe_value, safe_delta, sam_valid.float(), safe_age), -1))
        st = st + pos[:, :, None] + self.slot(torch.arange(o, device=robot.device))[None, None]
        if not use_sam:
            st = self.missing_sam[None, None, None] + pos[:, :, None] + self.slot(torch.arange(o, device=robot.device))[None, None]
            st = st.expand(b, -1, -1, -1)
        context = torch.cat((vision, rt, st.reshape(b, h * o, -1)), dim=1)
        cmask = torch.cat((muse_valid, history_valid, history_valid[:, :, None].expand(b, h, o).reshape(b, h * o)), dim=1)
        if not cmask.any(dim=1).all():
            raise ValueError("Each observation must contain at least one valid context token")
        return self.context_encoder(context, src_key_padding_mask=~cmask), cmask

    def _expert(self, x, context, context_valid, valid):
        for block in self.expert:
            x = block(x, context, context_valid, valid)
        return x

    def velocity(self, context, context_valid, noisy_actions, time, action_valid):
        b, h, _ = noisy_actions.shape
        safe = torch.where(action_valid[..., None], noisy_actions, torch.zeros_like(noisy_actions))
        tokens = self.action_proj(safe) + self.flow_time(time.reshape(b, 1))[:, None]
        tokens = tokens + self.action_position(torch.arange(h, device=tokens.device))[None]
        return self.action_out(self._expert(tokens, context, context_valid, action_valid))

    def dynamics(self, context, context_valid, current_action):
        query = self.action_proj(current_action)[:, None] + self.dynamics_task
        valid = torch.ones(query.shape[:2], dtype=torch.bool, device=query.device)
        return self.dynamics_out(self._expert(query, context, context_valid, valid)[:, 0]).reshape(-1, self.objects, self.metrics)

    def losses(self, context, context_valid, actions, action_valid, noise, time, change, change_valid, weight=0.1):
        actions = torch.where(action_valid[..., None], actions, torch.zeros_like(actions))
        noise = torch.where(action_valid[..., None], noise, torch.zeros_like(noise))
        x, target = flow_inputs(actions, noise, time)
        pred = self.velocity(context, context_valid, x, time, action_valid)
        flow = masked_mean((pred - target).square(), action_valid[..., None])
        dyn = self.dynamics(context, context_valid, actions[:, 0])
        safe_change = torch.where(change_valid, change, torch.zeros_like(change))
        dynamics = masked_mean(F.smooth_l1_loss(dyn, safe_change, reduction="none"), change_valid)
        return flow + weight * dynamics, flow, dynamics

    @torch.no_grad()
    def sample(self, context, context_valid, noise, steps=10):
        x = noise
        valid = torch.ones(noise.shape[:2], dtype=torch.bool, device=noise.device)
        for i in range(steps):
            t = torch.full((len(noise),), 1 - i / steps, device=noise.device)
            x = x - self.velocity(context, context_valid, x, t, valid) / steps
        return x
