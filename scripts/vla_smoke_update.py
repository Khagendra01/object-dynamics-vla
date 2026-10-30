"""Run a real episode window through the untrained expert with synthetic VLM tokens."""
import argparse
import json
import time
import numpy as np
import torch

from odvla.vla.action_expert import MuseActionExpert
from odvla.vla.dataset import observation_window


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", required=True)
    parser.add_argument("--stats", required=True)
    parser.add_argument("--t", type=int, default=0)
    parser.add_argument("--muse-tokens", help="Optional same-frame BF16 token cache")
    args = parser.parse_args()
    with np.load(args.episode, allow_pickle=False) as z:
        ep = {key: z[key] for key in z.files}
    with np.load(args.stats, allow_pickle=False) as z:
        stats = {key: z[key] for key in z.files}
    window = observation_window(ep, args.t, stats)
    torch.set_num_threads(2); torch.manual_seed(7)
    model = MuseActionExpert()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(.9, .95), weight_decay=1e-5)
    def tensor(key):
        return torch.from_numpy(np.asarray(window[key]))[None]
    if args.muse_tokens:
        with np.load(args.muse_tokens, allow_pickle=False) as z:
            muse = torch.from_numpy(z["features_bfloat16_bits"].view(np.int16).copy()).view(torch.bfloat16)[None].float()
            muse_mask = torch.from_numpy(z["attention_mask"].copy())[None].bool()
        token_source = "cached frozen Muse tokens; caller must provide the matching episode and frame"
    else:
        muse = torch.randn(1, 4, 6656)
        muse_mask = torch.ones(1, 4, dtype=torch.bool)
        token_source = "synthetic random token fixture (not Muse)"
    started = time.monotonic()
    context, mask = model.context(muse, muse_mask, tensor("robot"),
        tensor("history_valid"), tensor("history_timestamps"), tensor("sam_values"), tensor("sam_deltas"), tensor("sam_valid"), tensor("sam_age"))
    actions = tensor("actions")
    loss, flow, dynamics = model.losses(context, mask, actions, tensor("action_valid"), torch.randn_like(actions),
        torch.tensor([.5]), tensor("change"), tensor("change_valid"))
    optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    with torch.no_grad():
        context, mask = model.context(muse, muse_mask, tensor("robot"),
            tensor("history_valid"), tensor("history_timestamps"), tensor("sam_values"), tensor("sam_deltas"), tensor("sam_valid"), tensor("sam_age"))
        action = model.sample(context, mask, torch.randn_like(actions), 10)[0, 0].tolist()
    print(json.dumps({"input": "real episode window", "token_source": token_source, "token_shape": list(muse.shape),
        "t": args.t, "timestamp": window["timestamp"], "robot_shape": list(window["robot"].shape),
        "history_valid": window["history_valid"].tolist(), "action_valid_count": int(window["action_valid"].sum()),
        "loss": float(loss), "flow": float(flow), "dynamics": float(dynamics),
        "sampled_first_action_normalized": action, "elapsed_seconds": time.monotonic() - started,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "trainable_parameter_names": [n for n, p in model.named_parameters() if p.requires_grad]}, allow_nan=False))


if __name__ == "__main__": main()
