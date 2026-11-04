"""Launch bounded per-task integration smokes, at most two at a time.

Runs require an authorized Modal workspace with the frozen integration assets.
The credit check is conservative and based on a delayed billing summary; it is
not an invoice-level spending guarantee. No reserved episode is opened.
"""
import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from odvla.vla.libero_object_tasks import registry

CHECKPOINT = "7c86adc0fa8f941b8deb21048aa36efa54bdc190ca7dba7e1843922a55c4d39e"
APP_PATTERN = re.compile(r"https://modal\.com/apps/[^\s]+/(ap-[A-Za-z0-9]+)")


def command(*args):
    return subprocess.run([sys.executable, "-m", "modal", *args], capture_output=True, text=True, check=True)


def credits_used():
    bill = json.loads(command("billing", "summary", "--json").stdout)
    if float(bill["billed_cost"]) > 0:
        raise RuntimeError("Workspace reports a billed balance; stopping the free-credit campaign")
    return abs(float(bill["adjustments"].get("credits", "0")))


def fetch_report(name, directory):
    local = Path(directory) / f"{name}.json"
    command("volume", "get", "object-dynamics-vla-data", f"/eval/{name}/report.json", str(local), "--force")
    return json.loads(local.read_text())


def summary_row(report):
    if report.get("checkpoint_sha256") != CHECKPOINT:
        raise ValueError("Rollout report has a different frozen checkpoint")
    return {key: report.get(key) for key in ("task_id", "instruction", "episode", "scope", "status", "steps",
            "success", "truncated_at_limit", "initial_parity", "elapsed_seconds", "error_type", "error")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-credit-used", type=float, default=26.0)
    parser.add_argument("--max-parallel", type=int, default=2, choices=(1, 2))
    parser.add_argument("--output", default="outputs/ten_task_smoke_summary.json")
    args = parser.parse_args()
    tasks = [t["task_id"] for t in registry()["tasks"]]
    pending = [t for t in tasks if t != "libero_object_milk"]
    active, results = {}, {}
    with tempfile.TemporaryDirectory(prefix="odvla-ten-smokes-") as temp:
        results["libero_object_milk"] = summary_row(fetch_report("ten_task_milk_smoke_v1", temp))
        while pending or active:
            used = credits_used()
            if used >= args.max_credit_used:
                for app_id in active:
                    subprocess.run([sys.executable, "-m", "modal", "app", "stop", "--yes", app_id], check=False)
                raise RuntimeError(f"Reported credits used {used:.2f} reached conservative {args.max_credit_used:.2f} limit")
            while pending and len(active) < args.max_parallel:
                task = pending.pop(0)
                name = "ten_task_" + task.removeprefix("libero_object_") + "_smoke_v1"
                output = command("run", "--detach", "-m", "scripts.modal_muse_ten_task_rollout::ten_task_main",
                    "--stage", "rollout", "--task-id", task, "--name", name, "--max-steps", "30", "--background")
                match = APP_PATTERN.search(output.stdout + output.stderr)
                if not match or "Spawned ten-task rollout:" not in output.stdout:
                    raise RuntimeError(f"Rollout did not spawn: {task}: {output.stdout} {output.stderr}")
                active[match.group(1)] = (task, name)
                print(f"Launched {task}: {match.group(1)}; reported credits used ${used:.2f}", flush=True)
            time.sleep(20)
            apps = {item["app_id"]: item for item in json.loads(command("app", "list", "--json").stdout)}
            for app_id, (task, name) in list(active.items()):
                if app_id in apps and apps[app_id]["state"] != "stopped":
                    continue
                report = fetch_report(name, temp)
                results[task] = summary_row(report)
                print(f"Finished {task}: {results[task]}", flush=True)
                del active[app_id]
            finished = len(results) == len(tasks)
            failures = sum(row.get("status") != "completed" for row in results.values())
            summary = {"status": ("in_progress" if not finished else
                                  "complete_with_failures" if failures else "complete"),
                       "scope": "one 30-action smoke on a manually prompted training initialization per task; not held-out task-success evaluation",
                       "checkpoint_sha256": CHECKPOINT, "max_steps": 30, "results": results,
                       "failed_attempts": failures,
                       "credits_used_at_last_check": used}
            target = Path(args.output)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(".tmp")
            temporary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
            temporary.replace(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
