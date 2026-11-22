"""`python -m callqa calibrate`: how often the judge agrees with a human, from the labels in evals/labels/.

The LLM judge and the rule judge are reported separately (a label's `judge_source`). A label with `"synthetic": true`
is invented test data: it is left out unless --include-synthetic, and then the report says so in a header line, on the
title line of each judge section and on each disagreement row it shows (the other lines carry no mark).
Only the latest label for a call (record, persona_id, repeat) counts. Only a label with synthetic false and
on_simulated_run not true counts as real; a missing or null flag fails closed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABELS = ROOT / "evals" / "labels" / "human_labels.jsonl"
BADGE_THRESHOLD = 20  # real LLM judge labels needed before a judge badge may show
SOURCES = (("llm", "LLM judge"), ("rules", "Rule judge"))


def load_labels(path) -> tuple[list[dict], list[str]]:
    """(labels, problems). A missing file is no labels. A bad line is reported and skipped, never a crash."""
    path = Path(path)
    if not path.exists():
        return [], []
    try:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError as exc:
        return [], [f"cannot read the labels file ({exc.strerror or type(exc).__name__})"]
    labels, problems = [], []
    for number, line in enumerate(text.split("\n"), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except (ValueError, RecursionError):
            problems.append(f"line {number}: not valid JSON")
            continue
        if not isinstance(obj, dict):
            problems.append(f"line {number}: not a JSON object")
        elif obj.get("kind") != "label":
            continue  # a note line, such as the header of the synthetic fixture
        elif not isinstance(obj.get("human_pass"), bool) or not isinstance(obj.get("judge_pass"), bool):
            problems.append(f"line {number}: human_pass and judge_pass must be true or false")
        elif obj.get("judge_source") not in ("llm", "rules"):
            problems.append(f"line {number}: judge_source must be llm or rules")
        else:
            labels.append(obj)
    return labels, problems


def _kappa(po: float, pe: float) -> str:
    if pe == 1:
        return "n/a"
    text = f"{(po - pe) / (1 - pe):.2f}"
    return "0.00" if text == "-0.00" else text


def is_real(label: dict) -> bool:
    """Fails closed: a missing or null synthetic flag, or a simulated run, is not real."""
    return label.get("synthetic") is False and label.get("on_simulated_run") is not True


def latest_per_call(labels: list[dict]) -> list[dict]:
    """The last entry in the file for each (record, persona_id, repeat)."""
    latest = {}
    for x in labels:
        latest[(x.get("record"), x.get("persona_id"), x.get("repeat"))] = x
    return list(latest.values())


def _stats(labels: list[dict]) -> dict:
    n = len(labels)
    cells = {(h, j): 0 for h in (True, False) for j in (True, False)}
    for x in labels:
        cells[(x["human_pass"], x["judge_pass"])] += 1
    agree = cells[(True, True)] + cells[(False, False)]
    human_pass = cells[(True, True)] + cells[(True, False)]
    judge_pass = cells[(True, True)] + cells[(False, True)]
    pe = (human_pass * judge_pass + (n - human_pass) * (n - judge_pass)) / (n * n) if n else 1
    both_failed = [x for x in labels if not x["human_pass"] and not x["judge_pass"]
                   and x.get("human_failing_turn") is not None and x.get("judge_failing_turn") is not None]
    return {
        "n": n,
        "agreement": agree / n if n else None, "agree": agree,
        "kappa": _kappa(agree / n, pe) if n else "n/a",
        "matrix": {"human_pass": {"judge_pass": cells[(True, True)], "judge_fail": cells[(True, False)]},
                   "human_fail": {"judge_pass": cells[(False, True)], "judge_fail": cells[(False, False)]}},
        "failing_turn": {"n": len(both_failed),
                         "agree": sum(x["human_failing_turn"] == x["judge_failing_turn"] for x in both_failed)},
        "disagreements": [
            {"persona_id": x.get("persona_id"), "record": x.get("record"), "human_pass": x["human_pass"],
             "human_failing_turn": x.get("human_failing_turn"), "judge_failing_turn": x.get("judge_failing_turn"),
             "human_reason": x.get("human_reason"), "judge_reason": x.get("judge_reason"),
             "synthetic": not is_real(x)}
            for x in labels if x["human_pass"] != x["judge_pass"]],
    }


def calibrate(labels: list[dict], include_synthetic: bool = False) -> dict:
    labels = latest_per_call(labels)
    used = [x for x in labels if include_synthetic or is_real(x)]
    return {"include_synthetic": include_synthetic,
            "synthetic_used": sum(not is_real(x) for x in used),
            "synthetic_skipped": len(labels) - len(used),
            "sources": {key: _stats([x for x in used if x["judge_source"] == key]) for key, _ in SOURCES}}


def summary(path=DEFAULT_LABELS) -> dict:
    """What the dashboard needs to decide on a judge badge: the count of real (not synthetic) LLM judge labels."""
    labels, problems = load_labels(path)
    real = sum(x["judge_source"] == "llm" and is_real(x) for x in latest_per_call(labels))
    message = f"A judge badge needs {BADGE_THRESHOLD} real LLM judge labels; there are {real}."
    if any(x.startswith("cannot read") for x in problems):
        message = "cannot read the labels file. " + message
    return {"real_llm_labels": real, "badge_threshold": BADGE_THRESHOLD, "badge_ready": real >= BADGE_THRESHOLD,
            "message": message}


def _pct(agree: int, n: int) -> str:
    """Whole percent, rounded down, so 100% only ever means every label agreed."""
    return "100%" if agree == n else f"{agree * 100 // n}%"


def format_report(result: dict) -> str:
    lines = []
    if result["include_synthetic"] and result["synthetic_used"]:
        lines.append(f"SYNTHETIC labels included ({result['synthetic_used']}): invented test data, not a real result.")
    for key, title in SOURCES:
        s = result["sources"][key]
        lines.append("")
        if not s["n"]:
            lines.append(f"{title}: 0 labels")
            continue
        mark = " (includes SYNTHETIC, not a real result)" if result["synthetic_used"] else ""
        m = s["matrix"]
        turn = s["failing_turn"]
        lines += [
            f"{title}: {s['n']} label{'s' if s['n'] != 1 else ''}{mark}",
            f"  agreement {_pct(s['agree'], s['n'])}, Cohen's kappa {s['kappa']}",
            "  confusion matrix (rows human, columns judge)",
            "                judge pass  judge fail",
            f"  human pass    {m['human_pass']['judge_pass']:>10}  {m['human_pass']['judge_fail']:>10}",
            f"  human fail    {m['human_fail']['judge_pass']:>10}  {m['human_fail']['judge_fail']:>10}",
            "  failing turn agreement: " + (f"{turn['agree']}/{turn['n']} calls both failed with a turn"
                                           if turn["n"] else "n/a (no call where both failed with a turn)"),
        ]
        if s["disagreements"]:
            lines.append("  disagreements")
        for d in s["disagreements"]:
            tag = " [SYNTHETIC]" if d["synthetic"] else ""
            verdict = "human pass, judge fail" if d["human_pass"] else "human fail, judge pass"
            lines += [f"    {d['persona_id']} ({d['record']}): {verdict}{tag}",
                      f"      human: {d['human_reason']}", f"      judge: {d['judge_reason']}"]
    return "\n".join(lines).lstrip("\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m callqa calibrate", description=__doc__.split("\n\n")[0])
    p.add_argument("--labels", default=str(DEFAULT_LABELS), metavar="FILE",
                   help="labels file (default evals/labels/human_labels.jsonl)")
    p.add_argument("--include-synthetic", action="store_true",
                   help="also count labels marked synthetic (invented test data, never a real result)")
    args = p.parse_args(argv)
    labels, problems = load_labels(args.labels)
    if problems:
        unreadable = [x for x in problems if x.startswith("cannot read")]
        if unreadable:
            print(f"{unreadable[0]}: {args.labels}")
        bad = [x for x in problems if not x.startswith("cannot read")]
        if bad:
            print(f"skipped {len(bad)} bad line{'s' if len(bad) != 1 else ''} in {args.labels}:")
            for problem in bad:
                print(f"  {problem}")
    result = calibrate(labels, args.include_synthetic)
    if not any(s["n"] for s in result["sources"].values()):
        skipped = result["synthetic_skipped"]
        if skipped:
            print(f"0 labels ({skipped} synthetic labels skipped; pass --include-synthetic to see them)")
        else:
            print("0 labels: nothing to calibrate yet. Label some judged calls first (POST /api/labels).")
        return 0
    print(format_report(result))
    return 0
