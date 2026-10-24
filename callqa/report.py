"""Builds the dashboard report (see "API" in plans/call-qa-dashboard.md) from personas and run results."""
from __future__ import annotations

import ast
import hashlib
import re
from collections import Counter

from evals.backends import model_for


def sop_hash(sop_text: str) -> str:
    return hashlib.sha256(sop_text.encode("utf-8")).hexdigest()[:12]


def backend_models(backend) -> dict:
    """Model ids per role. Offline makes no model calls, so it has none."""
    if backend.name == "offline":
        return {}
    return {role: model_for(role) for role in ("caller", "agent", "judge", "gen")}


def broken_rule(persona: dict, result) -> str:
    """The SOP rule this call broke. A forbidden phrase is reported under the rule that forbids it (an account number
    leak under "never read it out"), so "Next fix" points at the real fix; anything else under the rule under test."""
    match = re.search(r"forbidden phrase: (.+)$", result.reason) if result.failure_kind == "forbidden" else None
    if match:
        try:
            phrase = ast.literal_eval(match[1])
        except (ValueError, SyntaxError):
            phrase = match[1].strip("'\"")
        rule = (persona.get("forbid_rules") or {}).get(phrase)
        if rule:
            return rule
    return persona["sop_rule"]


def build_call(persona: dict, result) -> dict:
    judged = result.failure_kind == "judge" and bool(result.reason.strip())
    return {
        "persona_id": persona["id"],
        "name": persona["name"],
        "language": persona["language"],
        "tags": list(persona["tags"]),
        "passed": result.passed,
        "reason": result.reason,
        "failure_kind": result.failure_kind,
        "failing_turn": None if result.passed or result.failure_kind == "crash" else result.failing_turn,
        "sop_rule": broken_rule(persona, result),
        "fix_hint": result.reason if judged else persona["fix_hint"],
        "notes": result.notes,
        "flaky": result.flaky,
        "transcript": [{"role": m["role"], "content": m["content"]} for m in result.transcript],
    }


SECTION_PREFIX = re.compile(r"^[A-Z][a-z]+(?: [a-z]+)?: ")  # "Never: ", "Payment date: "


def fix_for_rule(sop_rule: str, fallback: str) -> str:
    """The fix for a broken SOP rule is the rule itself, so the "Fix:" line always matches the rule above it."""
    rule = SECTION_PREFIX.sub("", sop_rule.strip(), count=1).strip()
    return f"Add or enforce this rule in the prompt: {rule}" if rule else fallback


def group_next_fixes(calls: list[dict]) -> list[dict]:
    """Failed calls grouped by the SOP rule they broke: biggest group first, then by rule text."""
    groups: dict[str, list[dict]] = {}
    for call in calls:
        if not call["passed"]:
            groups.setdefault(call["sop_rule"], []).append(call)
    fixes = [
        {
            "sop_rule": rule,
            "fix_hint": fix_for_rule(rule, members[0]["fix_hint"]),
            "count": len(members),
            "calls": [{"persona_id": c["persona_id"], "name": c["name"], "failing_turn": c["failing_turn"]}
                      for c in members],
        }
        for rule, members in groups.items()
    ]
    return sorted(fixes, key=lambda fix: (-fix["count"], fix["sop_rule"]))


def count_flaky(calls: list[dict]) -> int:
    """Calls where the two judge runs disagreed. They count as passes in the headline."""
    return sum(1 for c in calls if c.get("flaky"))


def crash_warnings(calls: list[dict]) -> list[str]:
    """One line per distinct crash cause, in the order first seen."""
    causes = Counter(c["reason"] for c in calls if c["failure_kind"] == "crash")
    return [f"{n} call{'s' if n > 1 else ''} crashed: {reason}" for reason, n in causes.items()]


def build_report(*, run_id: str, sop_text: str, agent: dict, backend, callers: int, max_turns: int,
                 generation, results: list, seconds: float) -> dict:
    by_persona = {r.persona: r for r in results}
    calls = [build_call(p, by_persona[p["id"]]) for p in generation.personas if p["id"] in by_persona]
    passed = sum(c["passed"] for c in calls)
    total = len(calls)
    warnings = [*backend.warnings, *generation.warnings, *crash_warnings(calls)]
    return {
        "run_id": run_id,
        "sop_hash": sop_hash(sop_text),
        "agent": {"id": agent["id"], "label": agent["label"]},
        "backend": {"name": backend.name, "models": backend_models(backend)},
        "callers": callers,
        "max_turns": max_turns,
        "warnings": warnings,
        "generator": {"source": generation.source, "warnings": list(generation.warnings)},
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "passed": passed,
        "flaky": count_flaky(calls),
        "total": total,
        "seconds": round(seconds, 2),
        "calls": calls,
        "next_fixes": group_next_fixes(calls),
    }
