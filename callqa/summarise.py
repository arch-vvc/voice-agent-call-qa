"""`python -m callqa --summarise <folder>`: a Markdown table of the committed live runs. Offline runs are left out."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from callqa.report import needs_review_count

NONE_YET = "No live runs committed yet."
FILE_DATE = re.compile(r"(\d{4})(\d{2})(\d{2})-\d{6}")  # dashboard-20261117-093000-v1.json
HEADER = ["date", "agent version", "backend", "models", "callers", "turns", "passed", "failed", "needs review",
          "pass rate"]


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def run_date(path: Path) -> str:
    match = FILE_DATE.search(path.name)
    if match:
        return "-".join(match.groups())
    return time.strftime("%Y-%m-%d", time.localtime(path.stat().st_mtime))


def load_run(path: Path) -> dict | None:
    """One result file as a table row, or None when it is not a dashboard result (invalid JSON, no pass_rate)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        rate = data["pass_rate"]
        if isinstance(rate, bool) or not isinstance(rate, (int, float)):
            return None
        agent = data.get("agent") or {}
        backend = data.get("backend") or {}
        passed, total = int(data.get("passed", 0)), int(data.get("total", 0))
        review = needs_review_count(data)
        models = backend.get("models") or {}
        return {
            "date": run_date(path),
            "agent": str(agent.get("id") or agent.get("label") or "?"),
            "backend": str(backend.get("name") or "?"),
            "models": ", ".join(dict.fromkeys(str(m) for m in models.values())) or "-",
            "callers": data.get("callers", "-"),
            "turns": data.get("max_turns", "-"),
            "passed": passed,
            "failed": max(total - passed - review, 0),
            "needs_review": review,
            "rate": float(rate),
        }
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def summarise(folder: str | Path) -> str:
    folder = Path(folder)
    files = sorted(folder.glob("*.json")) if folder.is_dir() else []
    runs, skipped = [], []
    for path in files:
        run = load_run(path)
        if run is None:
            skipped.append(path.name)
        elif run["backend"] != "offline":
            runs.append(run)
    lines = []
    if not runs:
        lines.append(NONE_YET)
    else:
        runs.sort(key=lambda r: (r["date"], r["agent"]))
        lines.append("| " + " | ".join(HEADER) + " |")
        lines.append("|" + "---|" * len(HEADER))
        for r in runs:
            cells = [r["date"], r["agent"], r["backend"], r["models"], r["callers"], r["turns"], r["passed"],
                     r["failed"], r["needs_review"], pct(r["rate"])]
            lines.append("| " + " | ".join(str(c) for c in cells) + " |")
        lines.append("")
        for agent in sorted({r["agent"] for r in runs}):
            rates = [r["rate"] for r in runs if r["agent"] == agent]
            n = len(rates)
            lines.append(f"{agent}: {n} run{'s' if n != 1 else ''}, mean {pct(sum(rates) / n)}, "
                         f"lowest {pct(min(rates))}, highest {pct(max(rates))}")
    if skipped:
        lines.append(f"Skipped (not valid result files): {', '.join(skipped)}")
    return "\n".join(lines)
