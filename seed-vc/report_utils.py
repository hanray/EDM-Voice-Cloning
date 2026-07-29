import json
import os
from datetime import datetime


def write_generation_report(mode, inputs, settings, outputs, stages_executed, report_dir="reports"):
    """Write a single generation report JSON with the required schema."""
    os.makedirs(report_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_path = os.path.join(report_dir, f"report_{timestamp}.json")

    report_payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "mode": mode if mode else "unknown",
        "inputs": inputs,
        "settings": settings if settings is not None else {},
        "outputs": outputs,
        "stages_executed": stages_executed,
    }

    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report_payload, handle, ensure_ascii=True, indent=2)

    return report_path
