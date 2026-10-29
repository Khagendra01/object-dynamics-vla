from __future__ import annotations
import argparse
import json
import platform
from pathlib import Path
from .data import write_json, read_json, load_manifest, load_episode, make_splits


def parser():
    p = argparse.ArgumentParser(description="Dataset-first object numerical dynamics experiments")
    s = p.add_subparsers(dest="command", required=True)
    s.add_parser("doctor", help="Print environment capabilities; no downloads")
    a = s.add_parser("audit"); a.add_argument("path"); a.add_argument("--out")
    a = s.add_parser("make-demo", help="Generate synthetic integration fixture")
    a.add_argument("--out", required=True); a.add_argument("--episodes", type=int, default=30)
    a.add_argument("--steps", type=int, default=64); a.add_argument("--seed", type=int, default=123)
    a = s.add_parser("split"); a.add_argument("--dataset", required=True)
    a.add_argument("--out", required=True); a.add_argument("--seed", type=int, default=17)
    a = s.add_parser("import-libero")
    for name in ("source", "out", "objects", "instruction"):
        a.add_argument("--" + name, required=True)
    a.add_argument("--fps", type=float, required=True)
    a.add_argument("--robot-keys", default="joint_states,ee_states,gripper_states")
    a.add_argument("--image-key", default="agentview_rgb"); a.add_argument("--flip-vertical", action="store_true")
    a = s.add_parser("attach-masks")
    for name in ("dataset", "masks", "out"):
        a.add_argument("--" + name, required=True)
    a = s.add_parser("train")
    for name in ("dataset", "splits", "out"):
        a.add_argument("--" + name, required=True)
    for name, default in [("epochs", 20), ("history", 4), ("horizon", 4), ("hidden", 128),
                          ("batch-size", 64), ("seed", 7), ("max-per-episode", 0), ("threads", 2)]:
        a.add_argument("--" + name, type=int, default=default)
    a.add_argument("--lr", type=float, default=1e-3)
    a.add_argument("--dynamics-weight", type=float, default=1.0)
    a.add_argument("--no-objects", action="store_true")
    a.add_argument("--no-dynamics-actions", action="store_true")
    a.add_argument("--device", default="cpu")
    a = s.add_parser("evaluate")
    a.add_argument("--dataset", required=True); a.add_argument("--run", required=True)
    a.add_argument("--split", choices=["val", "test"], default="test")
    a.add_argument("--device", default="cpu"); a.add_argument("--out")
    a = s.add_parser("report"); a.add_argument("--run", required=True); a.add_argument("--out", required=True)
    a = s.add_parser("replay")
    a.add_argument("--dataset", required=True); a.add_argument("--episode", required=True)
    a.add_argument("--out", required=True); a.add_argument("--stride", type=int, default=2)
    a = s.add_parser("predict")
    for name in ("dataset", "run", "episode", "out"):
        a.add_argument("--" + name, required=True)
    a.add_argument("--frame", type=int, required=True); a.add_argument("--seed", type=int, default=0)
    a.add_argument("--device", default="cpu")
    a = s.add_parser("smoke", help="Generate synthetic data, train, evaluate, and render a report")
    a.add_argument("--out", required=True); a.add_argument("--epochs", type=int, default=20)
    a.add_argument("--device", default="cpu")
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        if args.command == "doctor":
            result = {"python": platform.python_version(), "platform": platform.platform()}
            try:
                import torch
                result.update(torch=str(torch.__version__), cuda=torch.cuda.is_available(),
                              cuda_runtime=torch.version.cuda)
                if torch.cuda.is_available():
                    result.update(gpu=torch.cuda.get_device_name(),
                                  vram_gib=round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
                                  compute_capability=list(torch.cuda.get_device_capability()),
                                  compiled_arches=torch.cuda.get_arch_list())
                    # Actually launch a kernel, rather than treating CUDA discovery as sufficient.
                    result["cuda_kernel_check"] = float((torch.ones(8, device="cuda") * 2).sum().item()) == 16
            except ImportError:
                result["torch"] = "not installed"
        elif args.command == "audit":
            from .adapters import audit
            result = audit(args.path)
            if args.out: write_json(args.out, result)
        elif args.command == "make-demo":
            from .synthetic import generate
            result = generate(args.out, args.episodes, args.steps, args.seed)
        elif args.command == "split":
            result = make_splits(args.dataset, args.seed); write_json(args.out, result)
        elif args.command == "import-libero":
            from .adapters import import_libero
            result = import_libero(args.source, args.out, [x.strip() for x in args.objects.split(",")], args.fps,
                                   args.instruction, args.robot_keys.split(","), args.image_key, args.flip_vertical)
        elif args.command == "attach-masks":
            from .adapters import attach_masks
            result = attach_masks(args.dataset, args.masks, args.out)
        elif args.command == "train":
            from .experiment import train
            values = vars(args).copy()
            values.pop("command"); values["split_file"] = values.pop("splits"); values["output"] = values.pop("out")
            result = train(**values)
        elif args.command == "evaluate":
            from .experiment import evaluate
            result = evaluate(args.dataset, args.run, args.split, args.device, args.out)
        elif args.command == "report":
            from .presentation import report
            report(args.run, args.out); result = {"report": args.out}
        elif args.command == "replay":
            from .presentation import replay
            replay(args.dataset, args.episode, args.out, args.stride); result = {"replay": args.out}
        elif args.command == "predict":
            from .experiment import load_run
            from .inference import predict
            from .data import fingerprint
            model, cfg, stats = load_run(args.run, args.device)
            m = load_manifest(args.dataset)
            if fingerprint(args.dataset, m) != cfg["dataset_fingerprint"]:
                raise ValueError("Prediction dataset does not match run fingerprint")
            entry = next((e for e in m["episodes"] if e["id"] == args.episode), None)
            if entry is None: raise ValueError("Unknown episode ID")
            ep = load_episode(args.dataset, entry, m)
            if not 0 <= args.frame < len(ep["timestamps"]): raise ValueError("Frame out of bounds")
            prefix = {k: ep[k][:args.frame + 1] for k in ("features", "valid", "robot", "timestamps", "context")}
            result = predict(prefix, model, cfg, stats, args.device, args.seed)
            write_json(args.out, result)
        elif args.command == "smoke":
            from .synthetic import generate
            from .experiment import train, evaluate
            from .presentation import replay, report
            root = Path(args.out)
            if root.exists() and any(root.iterdir()): raise FileExistsError("Use a new --out directory")
            root.mkdir(parents=True, exist_ok=True)
            generate(root / "dataset")
            splits = make_splits(root / "dataset")
            write_json(root / "splits.json", splits)
            train(root / "dataset", root / "splits.json", root / "run", epochs=args.epochs, device=args.device)
            result = evaluate(root / "dataset", root / "run", device=args.device)
            replay(root / "dataset", "synthetic_0000", root / "replay.gif")
            report(root / "run", root / "report.html")
        else:
            raise AssertionError(args.command)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError, KeyError) as exc:
        p.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
