#!/usr/bin/env python3
"""Rewrite the text between <!-- live:start --> and <!-- live:end --> in README.md with the live runs table.

    python scripts/update_readme_live.py [results/live] [--readme README.md]

Prints one plain sentence for the newest v1 and v2 pair run on the same day with the same caller count, or nothing
when there is no such pair. Exit 2 when the README has no markers. Running it twice changes nothing.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from callqa.summarise import load_run, summarise  # noqa: E402

START, END = "<!-- live:start -->", "<!-- live:end -->"
STAMP = re.compile(r"(\d{4})(\d{2})(\d{2})-(\d{6})")
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def replace_block(text: str, body: str) -> str | None:
    """The README text with the marked block replaced, or None when the markers are missing or out of order."""
    start, end = text.find(START), text.find(END)
    if start == -1 or end == -1 or end < start:
        return None
    return text[: start + len(START)] + "\n" + body.strip("\n") + "\n" + text[end:]


def portfolio_sentence(folder: Path) -> str:
    runs = []
    for path in sorted(folder.glob("*.json")):
        run = load_run(path)
        match = STAMP.search(path.name)
        if run and run["backend"] != "offline" and match and run["agent"] in ("v1", "v2"):
            run["stamp"] = match.group(4)
            runs.append(run)
    best = None
    for v2 in runs:
        if v2["agent"] != "v2":
            continue
        for v1 in runs:
            if v1["agent"] == "v1" and (v1["date"], v1["callers"], v1["backend"]) == (
                    v2["date"], v2["callers"], v2["backend"]):
                key = (v2["date"], max(v1["stamp"], v2["stamp"]))
                if best is None or key > best[0]:
                    best = (key, v1, v2)
    if best is None:
        return ""
    _, v1, v2 = best
    year, month, day = v2["date"].split("-")
    return (f"v1 {round(v1['rate'] * 100)}% to v2 {round(v2['rate'] * 100)}% on {v2['callers']} callers, "
            f"{v2['backend']}, {int(day)} {MONTHS[int(month) - 1]} {year}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("folder", nargs="?", default=str(ROOT / "results" / "live"))
    p.add_argument("--readme", default=str(ROOT / "README.md"))
    args = p.parse_args(argv)
    readme = Path(args.readme)
    updated = replace_block(readme.read_text(encoding="utf-8"), summarise(args.folder))
    if updated is None:
        print(f"error: {readme} needs the markers {START} and {END}, in that order", file=sys.stderr)
        return 2
    if updated != readme.read_text(encoding="utf-8"):
        readme.write_text(updated, encoding="utf-8")
    sentence = portfolio_sentence(Path(args.folder))
    if sentence:
        print(sentence)
    return 0


if __name__ == "__main__":
    sys.exit(main())
