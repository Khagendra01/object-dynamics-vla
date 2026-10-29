"""Portable inspection outputs; no external assets or dashboard service required."""
import html
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from .data import load_manifest, load_episode, read_json


def replay(dataset, episode_id, output, stride=2):
    m = load_manifest(dataset)
    matches = [e for e in m["episodes"] if e["id"] == episode_id]
    if not matches:
        raise ValueError("Unknown episode ID; run audit to list episodes")
    ep = load_episode(dataset, matches[0], m)
    if "rgb" not in ep:
        raise ValueError("No RGB frames in this episode")
    if m["metric_names"][:5] != ["centroid_u", "centroid_v", "bbox_width", "bbox_height", "visible_area"]:
        raise ValueError("Replay overlay currently supports the five image-space measurements")
    if stride < 1:
        raise ValueError("stride must be >=1")
    frames = []
    for t in range(0, len(ep["rgb"]), stride):
        camera = Image.fromarray(ep["rgb"][t]).resize((384, 384), Image.Resampling.NEAREST)
        frame = Image.new("RGB", (760, 424), "#f5f6f8")
        frame.paste(camera, (8, 32))
        draw = ImageDraw.Draw(frame)
        draw.text((10, 10), f"{episode_id} | t={ep['timestamps'][t]:.3f}s | observed measurements", fill="#182230")
        for j, name in enumerate(m["object_slots"]):
            color = ["#b32638", "#2d58a6", "#267446"][j % 3]
            top = 44 + j * 82
            draw.text((410, top), name, fill=color)
            if ep["valid"][t, j].all():
                u, v, w, h, area = ep["features"][t, j]
                # Centroid marker; width/height are not necessarily centered on centroid.
                cx, cy = 8 + 384 * float(u), 32 + 384 * float(v)
                draw.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), outline=color, width=2)
                draw.text((410, top + 18), f"centroid=({u:.3f}, {v:.3f})", fill="#182230")
                draw.text((410, top + 34), f"box_size=({w:.3f}, {h:.3f}) area={area:.4f}", fill="#182230")
            else:
                draw.text((410, top + 18), "MISSING / UNOBSERVED", fill="#182230")
        draw.text((410, 340), "Source action: " + np.array2string(ep["actions"][t], precision=3), fill="#182230")
        draw.text((410, 384), "Synthetic fixture" if m.get("synthetic") else "Recorded dataset", fill="#182230")
        frames.append(frame)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    dt = float(np.median(np.diff(ep["timestamps"])))
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=max(20, round(1000 * dt * stride)), loop=0)


def report(run, output):
    run = Path(run)
    config = read_json(run / "config.json")
    best = read_json(run / "best_validation.json")
    rows = [json.loads(line) for line in (run / "train.jsonl").read_text().splitlines() if line.strip()]
    test_file = run / "test_metrics.json"
    test = read_json(test_file) if test_file.exists() else None
    def esc(value):
        return html.escape(str(value))
    title = "SYNTHETIC SMOKE TEST" if config["synthetic"] else "OFFLINE DATASET EXPERIMENT"
    table = "".join("<tr>" + "".join(f"<td>{esc(row[k])}</td>" for k in
                    ("epoch", "train_flow_mse", "train_dynamics_mse", "val_flow_mse", "val_dynamics_mse")) + "</tr>" for row in rows)
    content = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Object dynamics experiment</title><style>body{{font:15px/1.55 system-ui;max-width:1000px;margin:32px auto;padding:0 20px;color:#192331}}table{{border-collapse:collapse;width:100%}}td,th{{text-align:left;border-bottom:1px solid #ddd;padding:8px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f5f7;padding:16px}}.scroll{{overflow:auto}}</style>
<h1>{title}</h1><p>Actual outputs from this run. Offline prediction and action error do not establish robot task success.</p>
<p>Best validation epoch: {best['epoch']}. Dynamics weight: {config['dynamics_weight']}. Seed: {config['seed']}.</p>
<h2>Training history</h2><div class="scroll"><table><tr><th>Epoch</th><th>Train flow MSE</th><th>Train dynamics MSE</th><th>Validation flow MSE</th><th>Validation dynamics MSE</th></tr>{table}</table></div>
<h2>Best validation metrics</h2><pre>{esc(json.dumps(best['metrics'], indent=2))}</pre>
<h2>Held-out test</h2><pre>{esc(json.dumps(test, indent=2) if test else 'Not evaluated. Select hyperparameters on validation first.')}</pre>
<h2>Run configuration</h2><pre>{esc(json.dumps(config, indent=2))}</pre></html>"""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")
