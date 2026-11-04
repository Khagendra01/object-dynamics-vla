"""Merge the ten short smoke results without hiding the failed BBQ attempt."""
import json
from pathlib import Path

from scripts.run_ten_task_smokes import CHECKPOINT, summary_row


def collect(original, bbq_retry):
    if original.get("checkpoint_sha256") != CHECKPOINT or len(original.get("results", {})) != 10:
        raise ValueError("Expected ten reports from the frozen integration checkpoint")
    retry = summary_row(bbq_retry)
    if retry["task_id"] != "libero_object_bbq_sauce":
        raise ValueError("Unexpected retry task")
    rows = {}
    for task, original_row in sorted(original["results"].items()):
        attempts = [original_row]
        if task == "libero_object_bbq_sauce":
            attempts.append(retry)
        required = {"rgb", "muse", "robot", "sam", "sam_valid"}
        completed = [row for row in attempts if row.get("status") == "completed"
                     and required <= set(row.get("initial_parity") or {})
                     and all(row["initial_parity"][key] is True for key in required)
                     and (row.get("steps") == 30 or row.get("success") is True)]
        rows[task] = {"attempts": attempts, "integration_smoke_completed": bool(completed)}
    return {"status": "completed" if all(x["integration_smoke_completed"] for x in rows.values()) else "incomplete",
            "scope": "one 30-action smoke per task on manually prompted training initializations; not a task-success rate",
            "checkpoint_sha256": CHECKPOINT, "tasks": rows}


def main():
    import subprocess
    import sys
    import tempfile
    with tempfile.TemporaryDirectory(prefix="odvla-collect-smokes-") as temp:
        report = Path(temp) / "bbq.json"
        subprocess.run([sys.executable, "-m", "modal", "volume", "get", "object-dynamics-vla-data",
            "/eval/ten_task_bbq_sauce_a100_smoke_v1/report.json", str(report), "--force"], check=True)
        original = json.loads(Path("outputs/ten_task_smoke_summary.json").read_text())
        result = collect(original, json.loads(report.read_text()))
    path = Path("outputs/ten_task_smoke_results.json")
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"{result['status']}: {sum(x['integration_smoke_completed'] for x in result['tasks'].values())}/10 tasks; {path}")


if __name__ == "__main__":
    main()
