"""Validation-only suite. Freeze choices before evaluating the test split."""
import argparse
from pathlib import Path
from odvla.data import make_splits, write_json, read_json
from odvla.experiment import train, evaluate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True); p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=30); p.add_argument("--seeds", default="7,19,31")
    p.add_argument("--device", default="cuda"); p.add_argument("--max-per-episode", type=int, default=0)
    p.add_argument("--evaluate-test", action="store_true", help="Use only after choices are frozen")
    a = p.parse_args()
    root = Path(a.out)
    if root.exists() and any(root.iterdir()): raise FileExistsError("Use a fresh ablation output directory")
    root.mkdir(parents=True)
    split_file = root / "splits.json"
    write_json(split_file, make_splits(a.dataset))
    rows = []
    variants = {"no_objects": dict(no_objects=True, dynamics_weight=0),
                "objects_action_only": dict(dynamics_weight=0),
                "objects_joint": dict(dynamics_weight=1),
                "objects_joint_no_dynamics_actions": dict(dynamics_weight=1, no_dynamics_actions=True)}
    for seed in map(int, a.seeds.split(",")):
        for name, options in variants.items():
            run = root / f"{name}_seed{seed}"
            train(a.dataset, split_file, run, epochs=a.epochs, seed=seed, device=a.device,
                  max_per_episode=a.max_per_episode, **options)
            metrics = evaluate(a.dataset, run, "test" if a.evaluate_test else "val", a.device)
            rows.append({"variant": name, "seed": seed, "split": metrics["split"],
                         "flow_mse": metrics["flow_mse"],
                         "sampled_action_normalized_mse": metrics["sampled_action_normalized_mse"],
                         "dynamics_mse": metrics["dynamics"]["normalized_mse"],
                         "moving_dynamics_mse": metrics["dynamics"]["moving_normalized_mse"]})
            write_json(root / "comparison.json", rows)
    print(root / "comparison.json")


if __name__ == "__main__": main()
