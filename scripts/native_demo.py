"""One-command launcher; --dry-run tests the UI locally without GPUs or Modal."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


def dry_run(output, steps=12):
    import hashlib
    import io
    import numpy as np
    from PIL import Image, ImageDraw
    from odvla.demo import SKILLS, overlay_frame, run_episode, write_viewer
    output = Path(output)
    if output.exists():
        raise FileExistsError("Choose a fresh output directory")
    output.mkdir(parents=True); frames = output / "frames"; frames.mkdir()
    raw_frames = output / "raw_frames"; raw_frames.mkdir()
    index = 0
    def observe():
        image = Image.new("RGB", (512, 512), (215, 221, 227)); draw = ImageDraw.Draw(image)
        draw.rectangle((320, 180, 455, 325), outline=(70, 85, 100), width=8)
        draw.rectangle((105 + index, 235, 151 + index, 330), fill=(245, 244, 233), outline=(60, 95, 150), width=3)
        draw.text((107 + index, 270), "MILK", fill=(25, 70, 120))
        image.save(raw_frames / f"{index:06}.jpg", quality=98)
        masks = np.zeros((2, 128, 128), bool)
        masks[0, 59:83, 26:38] = True; masks[1, 45:82, 80:114] = True
        overlay_frame(np.asarray(image), masks, step=index, tracking="synthetic masks", dry_run=True).save(frames / f"{index:06}.png")
        buffer = io.BytesIO(); image.save(buffer, format="JPEG", quality=98)
        return {"jpeg": buffer.getvalue(), "robot": np.zeros(15, np.float32),
                "tracking": {"status": "synthetic test data"}}
    def predict(*, jpeg, robot, instruction, noise):
        return {"action": noise[0, 0, :7] * .1,
                "noise_sha256": hashlib.sha256(noise.tobytes()).hexdigest()}
    def advance(action):
        nonlocal index
        index += 1
        return {"done": False, "success": False}
    report = run_episode(SKILLS["milk_in_basket"], observe, predict, advance, max_steps=steps)
    report.update(dry_run=True, motor_policy="synthetic test double; no model or robot executed",
                  request="Local synthetic launcher/recording check",
                  selection={"planner": "synthetic test double", "reason": "No Muse, SAM, simulator or robot was run."})
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "trace.json").write_text(json.dumps(report["trace"], indent=2) + "\n")
    final = overlay_frame(np.asarray(Image.open(raw_frames / f"{steps-1:06}.jpg")),
                          step=steps - 1, status=report["status"], tracking="synthetic test data", dry_run=True)
    final.save(frames / f"{steps-1:06}.png"); final.save(output / "final.png")
    for pattern, name in ((frames / "%06d.png", "overlay.mp4"), (raw_frames / "%06d.jpg", "raw.mp4")):
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", "20", "-i", str(pattern),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(output / name)], check=True)
    write_viewer(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", default="Pick up the milk and put it in the basket")
    parser.add_argument("--episode", choices=("episode_0014", "episode_0020"), default="episode_0014")
    parser.add_argument("--output", default="outputs/native_demo")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--skill", choices=("milk_in_basket",), help="Explicit skill; bypasses Muse task routing")
    parser.add_argument("--no-tracking", action="store_true", help="Explicitly disable SAM overlays")
    parser.add_argument("--check-only", action="store_true", help="Check existing remote artifacts without starting GPUs")
    parser.add_argument("--dry-run", action="store_true", help="Synthetic local UI check, no cloud or robot execution")
    args = parser.parse_args()
    if args.dry_run:
        if args.check_only:
            parser.error("Choose --dry-run or --check-only")
        result = dry_run(args.output)
        print(json.dumps({k: v for k, v in result.items() if k != "trace"}, indent=2))
        print(f"Open {Path(args.output).resolve() / 'index.html'}")
        return
    if importlib.util.find_spec("modal") is None:
        parser.exit(2, "Install the Modal extra in your existing authenticated environment: pip install -e '.[modal]'\n")
    from odvla.demo import validate_run_name
    if not 1 <= args.max_steps <= 200:
        parser.error("--max-steps must be between 1 and 200")
    if args.run_name:
        validate_run_name(args.run_name)
    root = Path(__file__).resolve().parents[1]
    command = [sys.executable, "-m", "modal", "run", str(root / "scripts/modal_native_demo.py"),
               "--request", args.request, "--episode", args.episode, "--output", str(Path(args.output).resolve()),
               "--max-steps", str(args.max_steps)]
    for flag, value in (("--run-name", args.run_name), ("--skill", args.skill)):
        if value:
            command.extend((flag, value))
    if args.no_tracking:
        command.append("--no-tracking")
    if args.check_only:
        command.append("--check-only")
    # Argument vectors preserve the user's request literally; no shell execution.
    result = subprocess.run(command, cwd=root)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
