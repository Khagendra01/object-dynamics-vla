"""Train a separate one-action monitor head on a frozen policy encoder."""
from pathlib import Path
import numpy as np

from .data import build_windows, file_hash, read_json, validate_splits, write_json
from .experiment import load_run


def train_one_step(dataset, parent_run, output, epochs=20, batch_size=64, seed=7, device="cpu"):
    import torch
    from .model import OneStepDynamicsHead, masked_mse

    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    parent_run, output = Path(parent_run), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a fresh one-step monitor output directory")
    encoder, config, stats = load_run(parent_run, device)
    if config["no_objects"]:
        raise ValueError("One-step monitor requires an object-input encoder")
    splits = read_json(parent_run / "splits.json")
    validate_splits(dataset, splits)
    if config["dataset_fingerprint"] != splits["fingerprint"]:
        raise ValueError("Parent checkpoint/split fingerprint mismatch")
    train = build_windows(dataset, splits["train"], config["history"], config["horizon"], stats)
    val = build_windows(dataset, splits["val"], config["history"], config["horizon"], stats)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    encoder.eval().requires_grad_(False)
    head_config = {"hidden": config["model"]["hidden"], "action_channels": config["action_channels"],
                   "objects": config["objects"], "features": config["features"]}
    head = OneStepDynamicsHead(**head_config).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    output.mkdir(parents=True, exist_ok=True)
    feature_dim = config["objects"] * config["features"]
    def batch(data, indices):
        x = torch.from_numpy(data["x"][indices]).to(device)
        action = torch.from_numpy(data["a"][indices, :config["action_channels"]]).to(device)
        target = torch.from_numpy(data["y"][indices, :feature_dim]).to(device)
        mask = torch.from_numpy(data["mask"][indices, :feature_dim]).to(device)
        with torch.no_grad():
            h, _ = encoder.encode(x)
        return h, action, target, mask
    best, best_epoch = float("inf"), None
    log = []
    for epoch in range(1, epochs + 1):
        head.train()
        for indices in np.array_split(rng.permutation(len(train["x"])),
                                      max(1, int(np.ceil(len(train["x"]) / batch_size)))):
            h, action, target, mask = batch(train, indices)
            loss = masked_mse(head(h, action), target, mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        head.eval()
        numerator = denominator = 0.0
        with torch.no_grad():
            for indices in np.array_split(np.arange(len(val["x"])),
                                          max(1, int(np.ceil(len(val["x"]) / batch_size)))):
                h, action, target, mask = batch(val, indices)
                numerator += float(((head(h, action) - target).square() * mask).sum().item())
                denominator += float(mask.sum().item())
        if denominator == 0:
            raise ValueError("No valid one-step validation targets")
        score = numerator / denominator
        log.append({"epoch": epoch, "validation_one_step_normalized_mse": score})
        if score < best:
            best, best_epoch = score, epoch
            torch.save(head.state_dict(), output / "weights.pt")
    metadata = {"head": head_config, "parent_encoder_sha256": file_hash(parent_run / "weights.pt"),
                "dataset_fingerprint": splits["fingerprint"], "encoder_frozen": True,
                "input": "observation history and only action[t]", "target": "delta[t+1]",
                "best_validation_epoch": best_epoch, "best_validation_normalized_mse": best,
                "test_accessed": False, "epochs": epochs, "seed": seed}
    write_json(output / "metadata.json", metadata)
    write_json(output / "validation_log.json", log)
    return metadata


def load_one_step(path, parent_run, config, device="cpu"):
    import torch
    from .model import OneStepDynamicsHead
    path = Path(path)
    metadata = read_json(path / "metadata.json")
    if metadata["parent_encoder_sha256"] != file_hash(Path(parent_run) / "weights.pt"):
        raise ValueError("One-step monitor belongs to a different encoder checkpoint")
    if metadata["dataset_fingerprint"] != config["dataset_fingerprint"]:
        raise ValueError("One-step monitor dataset fingerprint mismatch")
    head = OneStepDynamicsHead(**metadata["head"]).to(device)
    head.load_state_dict(torch.load(path / "weights.pt", map_location=device, weights_only=True))
    head.eval()
    return head
