"""Escalation scoring: the handoff note an agent ends a reply with (contract C8), a grade out of 5, and the
expected / escalated matrix. Reporting only, except for callers with `must_escalate` (see run_one)."""
from __future__ import annotations

import json
import re

from callqa.sop import Sop, parse_sop

HANDOFF_FIELDS = ("caller", "issue", "tried", "why", "next_step")
MIN_FIELD_CHARS = 3
NOTE = re.compile(r"<handoff>(.*?)(?:</handoff>|\Z)", re.DOTALL)
OUTCOMES = ("TP", "FP", "FN", "TN")
NO_RULE = "The SOP has no rule that asks for a handoff for this caller."

# tag -> (SOP section, words one of which a rule in that section must contain to be the trigger)
TRIGGERS = {
    "escalation": ("Escalation", ("manager", "human")),
    "debt_dispute": ("Disputes", ("not theirs", "not their", "fraud")),
    "cannot_pay": ("Escalation", ("lost their job", "lost income", "cannot pay")),
    "legal_threat": ("Escalation", ("legal action", "regulator", "complaint")),
}


def split_handoff(text: str) -> tuple[str, dict | None]:
    """(spoken text, note). The note is the last `<handoff>{...}</handoff>` block; bad JSON gives {"raw": "..."}."""
    matches = NOTE.findall(text)
    if not matches:
        return text, None
    spoken = NOTE.sub("", text).strip()
    body = matches[-1].strip()
    try:
        note = json.loads(body)
    except ValueError:
        return spoken, {"raw": body}
    return spoken, note if isinstance(note, dict) else {"raw": body}


def score_handoff(note: dict | None) -> tuple[int, list[str]]:
    """(score out of 5, missing fields): one point per field that is a string of at least 3 characters."""
    note = note or {}
    missing = [f for f in HANDOFF_FIELDS
               if not (isinstance(note.get(f), str) and len(note[f].strip()) >= MIN_FIELD_CHARS)]
    return len(HANDOFF_FIELDS) - len(missing), missing


def grade_handoff(note: dict | None, judge_score=None, judge_missing=None) -> tuple[int, list[str]]:
    """The grade with a judge's opinion folded in. The judge can take points away but never award one for a field
    that is absent or too short, so the score never exceeds the offline count."""
    score, missing = score_handoff(note)
    if isinstance(judge_missing, list):
        missing = [f for f in HANDOFF_FIELDS if f in missing or f in judge_missing]
    score = min(score, len(HANDOFF_FIELDS) - len(missing))
    if isinstance(judge_score, int) and not isinstance(judge_score, bool):
        score = min(score, max(0, judge_score))
    return score, missing


def outcome(expected: bool, escalated: bool) -> str:
    return ("TP" if escalated else "FN") if expected else ("FP" if escalated else "TN")


def matrix(outcomes: list[str]) -> dict:
    return {key.lower(): outcomes.count(key) for key in OUTCOMES}


def expected_escalation(persona: dict, sop: Sop | str) -> tuple[bool, str]:
    """(should this caller be handed to a human, why). A persona's own `expected_escalation` wins; otherwise the
    caller's tags are matched to an SOP rule, which is quoted. No matching rule gives False and says so."""
    if isinstance(persona.get("expected_escalation"), bool):
        return persona["expected_escalation"], str(persona.get("escalation_reason") or NO_RULE)
    sop = parse_sop(sop) if isinstance(sop, str) else sop
    for tag, (section, words) in TRIGGERS.items():
        if tag not in persona.get("tags", []):
            continue
        for rule in sop.rules.get(section, []):
            if any(word in rule.lower() for word in words):
                return True, f"{section}: {rule}"
    return False, NO_RULE


def escalation_block(rows: list[dict], persona_order: list[str]) -> dict:
    """The record's `escalation` block over every run. Each row: {persona_id, outcome, handoff_score}."""
    counts = matrix([r["outcome"] for r in rows])
    runs = len(rows)
    scores = [r["handoff_score"] for r in rows if r["outcome"] in ("TP", "FP") and r["handoff_score"] is not None]

    def ids(kind: str) -> list[str]:
        found = {r["persona_id"] for r in rows if r["outcome"] == kind}
        return [pid for pid in persona_order if pid in found]

    return {"matrix": counts, "accuracy": round((counts["tp"] + counts["tn"]) / runs, 4) if runs else 0.0,
            "runs": runs, "handoff_avg": round(sum(scores) / len(scores), 1) if scores else None,
            "missed": ids("FN"), "unnecessary": ids("FP")}
