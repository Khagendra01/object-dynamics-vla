"""Task routing and recording for the native-policy application.

Only RGB, proprioception, a registered instruction and sampling noise reach the
motor policy. Muse selects a skill once; tracking is an observation overlay.
"""
from dataclasses import dataclass
import hashlib
import html
import json
from pathlib import Path
import re
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class Skill:
    id: str
    label: str
    instruction: str
    bddl: str
    policy_run: str


SKILLS = {
    "milk_in_basket": Skill(
        "milk_in_basket", "Milk → basket", "pick up the milk and place it in the basket",
        "libero_object/pick_up_the_milk_and_place_it_in_the_basket.bddl",
        "smolvla_libero_milk1000v1",
    ),
}
VOLUME_NAME = "object-dynamics-vla-data"
ROUTE_CHOICES = (
    {"skill_id": "milk_in_basket", "reason": "The request matches the milk pick-and-place skill."},
    {"skill_id": "unsupported", "reason": "No available skill matches this request."},
)


def route_token_options(prefix, sequences, eos_ids):
    """Constrain generation to registered route JSON, including a valid EOS."""
    options = set()
    for sequence in sequences:
        if list(sequence[:len(prefix)]) == list(prefix):
            options.update(eos_ids if len(prefix) == len(sequence) else [sequence[len(prefix)]])
    if not options:
        raise ValueError("Router generated a prefix outside the registered choices")
    return sorted(options)


def validate_run_name(value):
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", value) is None:
        raise ValueError("Run name must contain 1–80 letters, digits, underscores or hyphens")
    return value


def parse_muse_selection(text):
    """Accept a bounded selection, never a model-generated path or instruction."""
    if not isinstance(text, str) or len(text) > 8000:
        raise ValueError("Invalid Muse response")
    value = text.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if len(lines) < 3 or lines[-1].strip() != "```":
            raise ValueError("Incomplete Muse JSON response")
        value = "\n".join(lines[1:-1])
    try:
        plan = json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Muse must return a JSON object with skill_id and reason") from exc
    if not isinstance(plan, dict) or set(plan) != {"skill_id", "reason"}:
        raise ValueError("Muse must return only skill_id and reason")
    if not isinstance(plan["skill_id"], str) or plan["skill_id"] not in (*SKILLS, "unsupported"):
        raise ValueError("Muse selected an unregistered skill")
    if not isinstance(plan["reason"], str) or not plan["reason"].strip() or len(plan["reason"]) > 1000:
        raise ValueError("Muse must give a short reason")
    return plan


def action_noise(step, seed=7):
    # Same NumPy rule as the successful native development policy.
    return np.random.default_rng(seed + step).normal(size=(1, 50, 32)).astype(np.float32)


def run_episode(skill, observe, predict, advance, record=None, *, max_steps=200,
                timeout_seconds=900, clock=time.monotonic):
    """Drive an episode through small ports, without coupling perception to motors.

    observe() returns jpeg/robot and optional tracking metadata. predict() is
    deliberately called with just four named arguments; no tracking or Muse
    output can leak into its conditioning. advance() owns simulator success.
    """
    if skill.id not in SKILLS or SKILLS[skill.id] != skill:
        raise ValueError("Use a registered skill")
    if not 1 <= max_steps <= 200 or not 1 <= timeout_seconds <= 1200:
        raise ValueError("Invalid episode budget")
    started = clock()
    report = {"skill_id": skill.id, "instruction": skill.instruction,
              "motor_policy": "native adapted SmolVLA including native SmolVLM",
              "bridge_used": False, "dynamics_used": False,
              "max_steps": max_steps, "success": False, "status": "step_cap", "trace": []}
    try:
        for step in range(max_steps):
            if clock() - started >= timeout_seconds:
                report["status"] = "runtime_limit"
                break
            observation = observe()
            jpeg, robot = observation["jpeg"], np.asarray(observation["robot"], np.float32)
            if not isinstance(jpeg, bytes) or not jpeg or robot.shape != (15,) or not np.isfinite(robot).all():
                raise ValueError("Expected JPEG bytes and 15 finite robot-state channels")
            noise = action_noise(step)
            expected_noise_hash = hashlib.sha256(noise.tobytes()).hexdigest()
            prediction = predict(jpeg=jpeg, robot=robot.copy(), instruction=skill.instruction, noise=noise.copy())
            action = np.asarray(prediction["action"], np.float32)
            if action.shape != (7,) or not np.isfinite(action).all():
                raise ValueError("Native policy returned an invalid action")
            if prediction["noise_sha256"] != expected_noise_hash:
                raise ValueError("Policy did not use the supplied sampling noise")
            executed = np.clip(action, -1, 1)
            transition = advance(executed.tolist())
            success = bool(transition["success"])
            row = {"step": step, "simulated_time": step / 20,
                   "robot": robot.tolist(), "rgb_sha256": hashlib.sha256(jpeg).hexdigest(),
                   "noise_sha256": expected_noise_hash,
                   "sampled_action": action.tolist(), "executed_action": executed.tolist(),
                   "clipped_channels": (action != executed).tolist(),
                   "success": success, "done": bool(transition["done"]),
                   "tracking": observation.get("tracking", {"status": "disabled"}),
                   "policy_seconds": prediction.get("policy_seconds"),
                   "checkpoint_sha256": prediction.get("checkpoint_sha256")}
            report["trace"].append(row)
            if record is not None:
                record(row)
            if success or transition["done"]:
                report["success"] = success
                report["status"] = "success" if success else "terminated"
                break
    except Exception as exc:
        report["status"] = "error"
        report["error"] = f"{type(exc).__name__}: {exc}"
    report["executed_steps"] = len(report["trace"])
    report["seconds"] = clock() - started
    report["clipped_steps"] = sum(any(r["clipped_channels"]) for r in report["trace"])
    return report


def overlay_frame(rgb, masks=None, *, step=0, status="running", tracking="disabled", dry_run=False):
    """Draw passive SAM masks and status; never returns motor observations."""
    rgb = np.asarray(rgb, np.uint8).copy()
    image = Image.fromarray(rgb).convert("RGB")
    colors = ((70, 220, 230), (255, 194, 75))
    if masks is not None:
        for mask, color in zip(masks, colors):
            mask = np.asarray(Image.fromarray(np.asarray(mask, np.uint8)).resize(image.size, Image.Resampling.NEAREST), bool)
            rgb[mask] = (0.7 * rgb[mask] + 0.3 * np.array(color)).astype(np.uint8)
        image = Image.fromarray(rgb)
    width, height = image.size
    result = Image.new("RGB", (width, height + 112), (15, 23, 42))
    result.paste(image, (0, 80))
    draw = ImageDraw.Draw(result)
    try:
        large = ImageFont.truetype("DejaVuSans.ttf", 21)
        small = ImageFont.truetype("DejaVuSans.ttf", 13)
    except OSError:
        large = small = ImageFont.load_default()
    label = "DRY RUN · no robot policy" if dry_run else "Milk → basket"
    draw.text((16, 10), label, font=large, fill=(241, 245, 249))
    draw.text((16, 43), f"Step {step}  ·  {step / 20:.2f}s simulated  ·  {status}", font=small, fill=(203, 213, 225))
    draw.text((16, height + 87), f"SAM: {tracking}   ·   cyan milk / gold basket", font=small, fill=(203, 213, 225))
    return result


def write_viewer(directory, report):
    """Write a local result viewer; model/user strings are always escaped."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    escape = lambda value: html.escape(str(value), quote=True)
    dry = report.get("dry_run", False)
    title = "Dry-run preview" if dry else "Milk → basket"
    statuses = {"success": "Task completed", "step_cap": "Step limit reached",
                "runtime_limit": "Time limit reached", "terminated": "Episode ended", "error": "Run interrupted"}
    status = "Synthetic local check — no robot execution" if dry else statuses.get(report["status"], report["status"])
    planner = report.get("selection", {})
    content = f'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{escape(title)}</title>
<style>body{{margin:0;background:#0b1220;color:#e5edf7;font:16px system-ui,sans-serif}}
main{{max-width:1000px;margin:48px auto;padding:0 24px}}h1{{font-size:38px;margin-bottom:8px}}
.muted{{color:#a5b4c8}}.status{{display:inline-block;background:#183347;border-radius:12px;padding:12px 18px;margin:18px 0}}
video{{width:100%;max-height:72vh;background:#000;border-radius:16px}}.grid{{display:flex;flex-wrap:wrap;gap:12px;margin:20px 0}}
.card{{flex:1;min-width:190px;background:#142137;border-radius:12px;padding:16px}}a{{color:#79d7ec}}
pre{{background:#142137;padding:20px;border-radius:12px;overflow:auto}}footer{{margin:28px 0;color:#a5b4c8}}</style>
<main><p class="muted">Muse · task selection / SmolVLA · actions / SAM · tracking</p>
<h1>{escape(title)}</h1><p>{escape(report.get("request", report.get("instruction", "")))}</p>
<div class="status">{escape(status)}</div>
<video controls preload="metadata" poster="final.png"><source src="overlay.mp4" type="video/mp4"></video>
<div class="grid"><div class="card"><b>{report["executed_steps"]}</b><br>actions executed</div>
<div class="card"><b>{report.get("clipped_steps", 0)}</b><br>steps clipped</div>
<div class="card"><b>{escape(planner.get("planner", "unknown"))}</b><br>task selection</div></div>
<p class="muted">{escape(planner.get("reason", ""))}</p>
<p><a href="raw.mp4">Original RGB video</a> · <a href="report.json">Full report</a> · <a href="trace.json">Action trace</a></p>
<details><summary>Run details</summary><pre>{escape(json.dumps({k:v for k,v in report.items() if k != "trace"}, indent=2))}</pre></details>
<footer>Initialized LIBERO development demo. Completion comes from the simulator task predicate.
Video is shown at 20 simulated frames/s; inference wall time can be longer.</footer></main></html>'''
    (directory / "index.html").write_text(content, encoding="utf-8")
    return directory / "index.html"
