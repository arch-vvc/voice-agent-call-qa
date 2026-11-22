"""Builds the dashboard report (see "API" in plans/call-qa-dashboard.md) from personas and run results."""
from __future__ import annotations

import ast
import hashlib
import re
from collections import Counter
from datetime import datetime

from callqa.compliance import call_severity, compliance_block, merge_repeats
from callqa.consistency import consistency_block, first_failing, group_by_persona, incomplete
from callqa.escalation import escalation_block, expected_escalation, outcome
from callqa.metering import cost_block, cost_label, latency_block, load_prices
from callqa.personas import TRAIT_IDS
from callqa.runrecord import RECORD_KIND, RECORD_SCHEMA_VERSION
from callqa.sop import parse_sop
from evals.backends import ROLE_MODELS, model_for


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


def repeat_result(result) -> dict:
    return {"repeat": result.run, "seed": result.seed, "variant": result.variant, "passed": result.passed,
            "failure_kind": result.failure_kind, "failing_turn": result.failing_turn, "reason": result.reason,
            "transcript": [{"role": m["role"], "content": m["content"]} for m in result.transcript]}


def escalation_view(persona: dict, result, sop) -> dict:
    """One run's escalation facts: what was expected, what the agent did and how good the note was."""
    expected, reason = expected_escalation(persona, sop)
    return {"expected": expected, "reason": reason, "escalated": result.escalated,
            "outcome": outcome(expected, result.escalated), "handoff": result.handoff,
            "handoff_score": result.handoff_score if result.escalated else None,
            "handoff_missing": list(result.handoff_missing) if result.escalated else []}


def build_call(persona: dict, result, repeats: list | None = None, metering: bool = False, sop=None,
               compliance: list | None = None) -> dict:
    """One call per persona. `result` stands for the persona (its first failing repeat, else repeat 0) and `repeats`
    are all its results in repeat order (default: just `result`). `flaky` means some repeats passed and some failed;
    `failed_repeat` (0 based) is set then. `compliance` is the caller's rule results over every repeat (see
    callqa/compliance.py); without it the call has none and its severity is major or nothing."""
    repeats = repeats or [result]
    compliance = compliance or []
    passed_count = sum(r.passed for r in repeats)
    is_flaky = 0 < passed_count < len(repeats)
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
        "judge_split": result.judge_split,
        "transcript": [{"role": m["role"], "content": m["content"]} for m in result.transcript],
        # Neutral values until the feature that fills them lands (see C1 in plans/qa-upgrades.md).
        "traits": list(persona.get("traits", [])),
        "severity": call_severity(result.passed, compliance),
        "repeats_passed": passed_count,
        "repeats_total": len(repeats),
        "flaky": is_flaky,
        "failed_repeat": result.run if is_flaky else None,
        "repeat_results": [repeat_result(r) for r in repeats] if len(repeats) > 1 else [],
        "escalation": escalation_view(persona, result, sop) if sop is not None else None,
        "compliance": [dict(r) for r in compliance],
        "usage": (result.usage or None) if metering else None,
        "turn_latency_ms": list(result.turn_latency_ms) if metering else [],
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


def count_judge_split(calls: list[dict]) -> int:
    """Calls where the two judge runs disagreed. They count as passes in the headline."""
    return sum(1 for c in calls if c.get("judge_split"))


def crash_warnings(calls: list[dict]) -> list[str]:
    """One line per distinct crash cause, in the order first seen."""
    causes = Counter(c["reason"] for c in calls if c["failure_kind"] == "crash")
    return [f"{n} call{'s' if n > 1 else ''} crashed: {reason}" for reason, n in causes.items()]


def cost_view(backend, results: list, calls: int, prices: dict | None) -> dict:
    """Tokens summed over every result (all repeats), priced by role with the models the run used. Offline has no
    models, so it is priced at the default models a live run would use."""
    usage: dict = {}
    for result in results:
        for role, tokens in (result.usage or {}).items():
            entry = usage.setdefault(role, {"input_tokens": 0, "output_tokens": 0})
            entry["input_tokens"] += tokens["input_tokens"]
            entry["output_tokens"] += tokens["output_tokens"]
    models = backend_models(backend) or {role: model_for(role) for role in ROLE_MODELS}
    return cost_block(usage, models, prices or load_prices(), cost_label(backend.name), calls)


def latency_view(backend, results: list, budget_ms: float) -> dict:
    label = "simulated" if backend.name == "offline" else "measured"
    return latency_block([ms for r in results for ms in r.turn_latency_ms], label, budget_ms)


def traits_block(calls: list[dict]) -> dict:
    """Pass rate per trait over the calls that carry it, in the fixed trait order of C5. A trait no caller carries is
    left out. A call counts as passed when it passed every repeat."""
    block = {}
    for trait in TRAIT_IDS:
        carrying = [c for c in calls if trait in c["traits"]]
        if carrying:
            passed = sum(c["passed"] for c in carrying)
            block[trait] = {"passed": passed, "total": len(carrying), "pass_rate": round(passed / len(carrying), 4)}
    return block


OFFLINE_NOTE = "SYNTHETIC: offline rule based agents, simulated cost and latency"


def persona_set_view(generation, set_name: str | None = None) -> dict:
    ids = [p["id"] for p in generation.personas]
    name = set_name or ("story" if generation.source.startswith("templates") else "generated")
    return {"name": name, "count": len(ids), "ids_sha256": hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()}


def build_report(*, run_id: str, sop_text: str, agent: dict, backend, callers: int, max_turns: int,
                 generation, results: list, seconds: float, synthetic: bool = False, repeats: int = 1,
                 seed: int = 0, set_name: str | None = None, metering: bool = False, budget_ms: float = 1500,
                 prices: dict | None = None, rules=None) -> dict:
    """The run record (schema version 1). Blocks whose feature has not landed are None. With `repeats` above 1 each
    persona's repeats are grouped into one call, `passed` counts personas that passed every repeat (pass^k) and the
    `consistency` block is filled. With `metering` the `cost` and `latency` blocks and each call's `usage` and
    `turn_latency_ms` are filled too (labelled simulated, estimated or reported, see callqa/metering.py). With `rules`
    (a callqa.compliance.Rules, whose results the runs must have been given) each call carries its `compliance`
    results and `severity`, and the `compliance` block and `rules_file` are filled; without it they stay None."""
    groups = group_by_persona(results)
    sop = parse_sop(sop_text)
    merged = {pid: merge_repeats([r.compliance for r in group]) for pid, group in groups.items()} if rules else {}
    calls = [build_call(p, first_failing(groups[p["id"]]), groups[p["id"]], metering, sop, merged.get(p["id"]))
             for p in generation.personas if p["id"] in groups]
    rows = [{"persona_id": p["id"], **{k: view[k] for k in ("outcome", "handoff_score")}}
            for p in generation.personas for view in (escalation_view(p, r, sop) for r in groups.get(p["id"], []))]
    crashes, missing = crash_warnings(calls), incomplete(groups, repeats)
    for call in calls:  # fewer or more results than repeats: the persona cannot count as passing every repeat
        if call["repeats_total"] != repeats and call["passed"]:
            call.update(passed=False, failure_kind="crash", failing_turn=None,
                        reason=f"expected {repeats} repeats, got {call['repeats_total']}")
    passed = sum(c["passed"] for c in calls)
    total = len(calls)
    warnings = [*backend.warnings, *generation.warnings, *crashes, *missing]
    simulated = backend.name == "offline"
    return {
        "schema_version": RECORD_SCHEMA_VERSION,
        "kind": RECORD_KIND,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "synthetic": synthetic,
        "simulated": simulated,
        "note": OFFLINE_NOTE if simulated else "",
        "run_id": run_id,
        "sop_hash": sop_hash(sop_text),
        "agent": {"id": agent["id"], "label": agent["label"], "variant": agent.get("variant", agent["id"]),
                  "model": agent.get("model")},
        "backend": {"name": backend.name, "models": backend_models(backend)},
        "persona_set": persona_set_view(generation, set_name),
        "seed": seed,
        "repeats": repeats,
        "callers": callers,
        "max_turns": max_turns,
        "rules_file": rules.label if rules else None,
        "warnings": warnings,
        "generator": {"source": generation.source, "warnings": list(generation.warnings)},
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "passed": passed,
        "total": total,
        "judge_split": count_judge_split(calls),
        "seconds": round(seconds, 2),
        "consistency": consistency_block(groups, repeats) if repeats > 1 else None,
        "cost": cost_view(backend, results, len(results), prices) if metering else None,
        "latency": latency_view(backend, results, budget_ms) if metering else None,
        "escalation": escalation_block(rows, [p["id"] for p in generation.personas]),
        "compliance": compliance_block(rules, calls, backend.name) if rules else None,
        "traits": traits_block(calls),
        "calls": calls,
        "next_fixes": group_next_fixes(calls),
    }
