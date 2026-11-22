"""Compliance rules (contract C6): load a rules file, check a finished call against its deterministic rules, and hand
the judge the judge based ones. A demo aid, not legal advice and not a compliance certification: the rules file
carries that disclaimer and says which rules are verified against a primary source and which are not.

No network and no model calls here. A rules file is YAML; `load_rules` refuses anything it does not understand."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from callqa.sop import Sop

PRODUCT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_RULES_PATH = PRODUCT_DIR / "rules" / "collections_india.yaml"
SCHEMA_VERSION = 1
DEFAULT_CALL_TIME = "11:00"  # IST, when a caller has no `call_time`
OFFLINE_WHY = "judge based rule; offline runs deterministic rules only"

BASES = ("regulation", "draft", "best_practice")
SEVERITIES = ("critical", "major")
CHECKS = ("deterministic", "judge")
STATUSES = ("in_force", "draft", "superseded")
# kind -> how it is checked. Every kind in a rules file must be one of these.
KIND_CHECK = {
    "call_window": "deterministic",
    "first_agent_turn_names": "deterministic",
    "no_disclosure_to_non_holder": "deterministic",
    "forbidden_phrases": "deterministic",
    "judge_prompt": "judge",
}
KINDS = tuple(KIND_CHECK)
DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
CLOCK = re.compile(r"^(\d{1,2}):(\d{2})$")


class RulesError(ValueError):
    """The rules file cannot be read or is not valid."""


@dataclass(frozen=True)
class Rule:
    id: str
    title: str
    basis: str
    severity: str
    check: str
    kind: str
    params: dict
    applies_to: dict
    source: str
    quote: str
    verified: bool
    note: str = ""


@dataclass(frozen=True)
class Rules:
    id: str
    title: str
    disclaimer: str
    sources: list = field(default_factory=list)
    rules: list = field(default_factory=list)
    label: str = ""  # how the record names the file: relative to the demo folder when it lives there


# ---- loading and validation ----

def _text(data: dict, key: str, where: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RulesError(f"{where}: {key} must be a non empty string")
    return value.strip()


def _clock(value, where: str) -> tuple[int, int]:
    match = CLOCK.match(value) if isinstance(value, str) else None
    if not match or int(match[1]) > 23 or int(match[2]) > 59:
        raise RulesError(f"{where} must look like HH:MM, got {value!r}")
    return int(match[1]), int(match[2])


def _validate_params(kind: str, params: dict, where: str) -> None:
    if kind == "call_window":
        start, end = _clock(params.get("start"), f"{where} params.start"), _clock(params.get("end"),
                                                                                  f"{where} params.end")
        if start >= end:
            raise RulesError(f"{where}: params.start must be before params.end")
    elif kind == "first_agent_turn_names":
        _text(params, "fact", f"{where} params")
    elif kind == "forbidden_phrases":
        phrases = params.get("phrases")
        if not isinstance(phrases, list) or not phrases or not all(isinstance(p, str) and p.strip() for p in phrases):
            raise RulesError(f"{where}: params.phrases must be a non empty list of strings")
    elif kind == "judge_prompt":
        _text(params, "question", f"{where} params")


def _rule(raw, source_ids: set[str]) -> Rule:
    if not isinstance(raw, dict):
        raise RulesError("every rule must be a mapping")
    rid = _text(raw, "id", "rule")
    where = f"rule {rid}"
    kind = raw.get("kind")
    if kind not in KIND_CHECK:
        raise RulesError(f"{where}: unknown kind {kind!r}; choose one of {', '.join(KINDS)}")
    for key, allowed in (("basis", BASES), ("severity", SEVERITIES), ("check", CHECKS)):
        if raw.get(key) not in allowed:
            raise RulesError(f"{where}: {key} must be one of {', '.join(allowed)}, got {raw.get(key)!r}")
    if raw["check"] != KIND_CHECK[kind]:
        raise RulesError(f"{where}: kind {kind} is a {KIND_CHECK[kind]} check, not {raw['check']}")
    if raw.get("source") not in source_ids:
        raise RulesError(f"{where}: source {raw.get('source')!r} is not one of the file's sources")
    if not isinstance(raw.get("verified"), bool):
        raise RulesError(f"{where}: verified must be true or false")
    params = raw.get("params") or {}
    if not isinstance(params, dict):
        raise RulesError(f"{where}: params must be a mapping")
    _validate_params(kind, params, where)
    applies = raw.get("applies_to")
    tags = applies.get("tags_any", []) if isinstance(applies, dict) else None
    if not isinstance(applies, dict) or not isinstance(tags, list) or not all(isinstance(t, str) for t in tags) or (
            not applies.get("all") and not tags):
        raise RulesError(f"{where}: applies_to needs `all: true` or a non empty `tags_any` list")
    return Rule(rid, _text(raw, "title", where), raw["basis"], raw["severity"], raw["check"], kind, params,
                {"tags_any": list(tags), "all": bool(applies.get("all"))}, raw["source"],
                _text(raw, "quote", where), raw["verified"], str(raw.get("note") or "").strip())


def _source(raw) -> dict:
    if not isinstance(raw, dict):
        raise RulesError("every source must be a mapping")
    where = f"source {raw.get('id')}"
    out = {key: _text(raw, key, where) for key in ("id", "title", "url", "published", "read_on")}
    for key in ("published", "read_on"):
        if not DATE.match(out[key]):
            raise RulesError(f"{where}: {key} must be YYYY-MM-DD")
    if raw.get("status") not in STATUSES:
        raise RulesError(f"{where}: status must be one of {', '.join(STATUSES)}, got {raw.get('status')!r}")
    return {**out, "status": raw["status"], "note": str(raw.get("note") or "").strip()}


def load_rules(path) -> Rules:
    """Read and validate a rules file. Raises RulesError (a ValueError) for an unreadable or invalid file, an unknown
    rule kind, a rule that points at no source, or a bad severity, status, date or time."""
    path = Path(path)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise RulesError(f"cannot read the rules file {path}: {exc}") from None
    if not isinstance(data, dict):
        raise RulesError(f"{path}: the rules file must be a mapping")
    if data.get("schema_version") != SCHEMA_VERSION:
        raise RulesError(f"{path}: schema_version must be {SCHEMA_VERSION}")
    sources = [_source(s) for s in data.get("sources") or []]
    if not sources:
        raise RulesError(f"{path}: a rules file needs at least one source")
    source_ids = [s["id"] for s in sources]
    if len(set(source_ids)) != len(source_ids):
        raise RulesError(f"{path}: duplicate source id")
    rules = [_rule(r, set(source_ids)) for r in data.get("rules") or []]
    if not rules:
        raise RulesError(f"{path}: a rules file needs at least one rule")
    if len({r.id for r in rules}) != len(rules):
        raise RulesError(f"{path}: duplicate rule id")
    return Rules(_text(data, "id", str(path)), _text(data, "title", str(path)),
                 _text(data, "disclaimer", str(path)), sources, rules, _label(path))


def _label(path: Path) -> str:
    try:
        return path.resolve().relative_to(PRODUCT_DIR).as_posix()
    except ValueError:
        return str(path)


# ---- which rules a caller meets ----

def applies(rule: Rule, persona: dict) -> bool:
    return rule.applies_to["all"] or bool(set(rule.applies_to["tags_any"]) & set(persona.get("tags", [])))


def judge_rules(rules: Rules, persona: dict | None = None) -> list[Rule]:
    """The judge based rules for the judge prompt: all of them, or only those that apply to `persona`."""
    return [r for r in rules.rules if r.check == "judge" and (persona is None or applies(r, persona))]


# ---- the deterministic checks ----

def _result(rule: Rule, passed: bool, turn: int | None, reason: str) -> dict:
    return {"rule_id": rule.id, "title": rule.title, "severity": rule.severity, "pass": passed, "turn": turn,
            "reason": reason, "check": rule.check}


def _agent_turns(transcript: list[dict]) -> list[tuple[int, str]]:
    return [(i, m["content"]) for i, m in enumerate(transcript) if m["role"] == "assistant"]


def _first_hit(transcript: list[dict], needles: list[str]) -> tuple[int, str] | None:
    """(turn, needle) for the earliest agent turn that says any needle (case blind)."""
    for index, text in _agent_turns(transcript):
        lowered = text.lower()
        for needle in needles:
            if needle.lower() in lowered:
                return index, needle
    return None


def _check_window(rule: Rule, persona: dict, transcript: list[dict]) -> dict:
    stamp = str(persona.get("call_time") or DEFAULT_CALL_TIME)
    agent = _agent_turns(transcript)
    first = agent[0][0] if agent else None
    try:
        now = _clock(stamp, "call_time")
    except RulesError:
        return _result(rule, False, first, f"call_time {stamp!r} is not HH:MM (IST), so the window cannot be checked")
    start, end = _clock(rule.params["start"], "start"), _clock(rule.params["end"], "end")
    window = f"{rule.params['start']} to {rule.params['end']} IST"
    if start <= now <= end:
        return _result(rule, True, None, f"call placed at {stamp} IST, inside {window}")
    return _result(rule, False, first, f"call placed at {stamp} IST, outside {window}")


def _check_company(rule: Rule, transcript: list[dict], sop: Sop) -> dict | None:
    name = str(sop.facts.get(rule.params["fact"]) or "").strip()
    agent = _agent_turns(transcript)
    if not name or not agent:
        return None  # nothing to compare with: the rule is skipped, not passed
    index, text = agent[0]
    if name.lower() in text.lower():
        return _result(rule, True, None, f"the first agent turn names {name}")
    return _result(rule, False, index, f"the first agent turn never says the lender's name, {name}")


def _check_disclosure(rule: Rule, transcript: list[dict], sop: Sop) -> dict:
    secrets = list(dict.fromkeys([*sop.amount_strings(), *sop.due_date_strings(), *sop.account_strings()]))
    hit = _first_hit(transcript, secrets)
    if hit is None:
        return _result(rule, True, None, "no amount, due date or account detail was said")
    return _result(rule, False, hit[0], f"agent said {hit[1]!r} to a caller who is not the verified account holder")


def _check_phrases(rule: Rule, transcript: list[dict]) -> dict:
    hit = _first_hit(transcript, rule.params["phrases"])
    if hit is None:
        return _result(rule, True, None, "none of the listed phrases was said")
    return _result(rule, False, hit[0], f"agent said {hit[1]!r}")


def check_call(rules: Rules, persona: dict, transcript: list[dict], sop: Sop) -> list[dict]:
    """The deterministic rules that apply to this caller, each as {rule_id, title, severity, pass, turn, reason,
    check}. `turn` is a 0 based transcript index of an agent turn (None when the rule passed). Judge based rules
    are not here: see `judge_rules` and `combine_judge_compliance`. A rule with nothing to check is left out."""
    out = []
    for rule in rules.rules:
        if rule.check != "deterministic" or not applies(rule, persona):
            continue
        if rule.kind == "call_window":
            result = _check_window(rule, persona, transcript)
        elif rule.kind == "first_agent_turn_names":
            result = _check_company(rule, transcript, sop)
        elif rule.kind == "no_disclosure_to_non_holder":
            result = _check_disclosure(rule, transcript, sop)
        else:
            result = _check_phrases(rule, transcript)
        if result is not None:
            out.append(result)
    return out


# ---- judge based rules ----

def judge_section(rules: list[Rule]) -> str:
    """The part of the judge prompt that lists judge based rules; empty when there are none."""
    if not rules:
        return ""
    lines = "\n".join(f"- {r.id}: {r.params['question'].strip()}" for r in rules)
    return ("Compliance rules (answer each one under \"compliance\"; they never change \"pass\" above):\n"
            f"{lines}\n\n")


COMPLIANCE_ASK = (
    ' Also add "compliance": a list with one {"rule_id": "<id>", "pass": true or false, "turn": <the number of the '
    'AGENT turn that broke it, or null>, "reason": "<one sentence>"} for each compliance rule listed above.'
)


def combine_judge_compliance(rules: list[Rule], verdict_lists: list[list[dict]]) -> list[dict]:
    """Judge based results from the judge runs (one list of parsed entries per run). A rule fails only if every run
    answered it and every run failed it, as for the call itself; a rule no run answered is left out."""
    out = []
    for rule in rules:
        entries = [next((e for e in run if e["rule_id"] == rule.id), None) for run in verdict_lists]
        answered = [e for e in entries if e is not None]
        if not answered:
            continue
        failed = [e for e in answered if not e["pass"]]
        if len(answered) == len(entries) and len(failed) == len(answered):
            out.append(_result(rule, False, failed[0]["turn"], failed[0]["reason"] or "the judge says it was broken"))
        else:
            out.append(_result(rule, True, None, "the judge says it was met" if not failed
                               else "the two judge runs disagreed; counted as met"))
    return out


# ---- results across a call, a run and a record ----

def first_critical_failure(results: list[dict]) -> dict | None:
    return next((r for r in results if not r["pass"] and r["severity"] == "critical"), None)


def merge_repeats(per_repeat: list[list[dict]]) -> list[dict]:
    """One list of results for a caller that ran several repeats: per rule the first failing result, else the first."""
    chosen: dict[str, dict] = {}
    for results in per_repeat:
        for result in results:
            current = chosen.get(result["rule_id"])
            if current is None or (current["pass"] and not result["pass"]):
                chosen[result["rule_id"]] = result
    return list(chosen.values())


def call_severity(passed: bool, results: list[dict]) -> str | None:
    """critical when a critical rule failed, major for any other failure (a failed call or a failed major rule)."""
    if first_critical_failure(results) is not None:
        return "critical"
    if not passed or any(not r["pass"] for r in results):
        return "major"
    return None


def compliance_block(rules: Rules, calls: list[dict], backend_name: str) -> dict:
    """The record's `compliance` block. `calls` holds {persona_id, name, compliance: [results]} per caller."""
    offline = backend_name == "offline"
    evaluated = [r.id for r in rules.rules if r.check == "deterministic" or not offline]
    not_evaluated = [{"rule_id": r.id, "why": OFFLINE_WHY} for r in rules.rules if r.id not in evaluated]
    failures = []
    for rule in rules.rules:
        failing = sorted((c for c in calls for r in c["compliance"] if r["rule_id"] == rule.id and not r["pass"]),
                         key=lambda c: c["persona_id"])
        if failing:
            failures.append({"rule_id": rule.id, "title": rule.title, "severity": rule.severity,
                             "count": len(failing),
                             "calls": [{"persona_id": c["persona_id"], "name": c["name"], "failing_turn": next(
                                 r["turn"] for r in c["compliance"] if r["rule_id"] == rule.id)} for c in failing]})
    failures.sort(key=lambda f: SEVERITIES.index(f["severity"]))  # stable: critical first, then file order
    return {"rules_id": rules.id, "rules_file": rules.label, "disclaimer": rules.disclaimer,
            "evaluated": evaluated, "not_evaluated": not_evaluated, "failures": failures}
