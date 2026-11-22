"""Compliance rules (A6a): the rules file, the engine, run_one, the report block and the eval line. No network.

The rules file is a demo aid. It is not legal advice and not a compliance certification."""
import copy
import json
from pathlib import Path

import pytest
import yaml

from callqa import compliance, evaluate
from callqa.compliance import DEFAULT_RULES_PATH, RulesError, check_call, judge_rules, load_rules
from callqa.persona_sets import personas_for_set
from callqa.report import build_report
from callqa.server import DEFAULT_AGENTS, metered_agent, read_sample_sop
from callqa.sop import Sop, parse_sop
from evals.backends import OfflineBackend
from evals.run_evals import END, judge_prompt, parse_verdict, run_one, run_suite

SOP_TEXT = read_sample_sop()
SOP = parse_sop(SOP_TEXT)
RULES = load_rules(DEFAULT_RULES_PATH)
LEAK = "Your EMI of Rs 12,400 was due on 15 October."


def persona(**extra) -> dict:
    base = {"id": "p", "name": "P", "language": "en", "tags": [], "persona": "A caller.", "opening": "Hello?",
            "script": ["Okay."], "must_include": [], "must_not_include": [], "expectations": "PASS if fine.",
            "sop_rule": "Verification: x", "fix_hint": "fix"}
    return {**base, **extra}


def convo(*agent_lines: str) -> list[dict]:
    out = []
    for i, line in enumerate(agent_lines):
        out += [{"role": "user", "content": "Hello?" if i == 0 else "Okay."}, {"role": "assistant", "content": line}]
    return out


def by_rule(results: list[dict]) -> dict:
    return {r["rule_id"]: r for r in results}


def base_file() -> dict:
    return copy.deepcopy(yaml.safe_load(DEFAULT_RULES_PATH.read_text(encoding="utf-8")))


def write_rules(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "rules.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


# ---- the rules file ----

def test_rules_file_has_disclaimer_sources_and_status():
    data = yaml.safe_load(DEFAULT_RULES_PATH.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1 and data["id"] == "collections_india"
    assert "not legal advice" in data["disclaimer"].lower()
    assert "not a compliance certification" in data["disclaimer"].lower()
    assert RULES.disclaimer == data["disclaimer"]
    assert data["sources"], "at least one source"
    for source in data["sources"]:
        assert source["url"].startswith("https://www.rbi.org.in/") or source["url"].startswith("https://rbi.org.in/")
        assert source["title"].strip() and source["status"] in ("in_force", "draft", "superseded")
        assert source["published"][:2] == "20" and source["read_on"] == "2026-11-21"
    ids = {s["id"] for s in data["sources"]}
    assert len(ids) == len(data["sources"])
    assert "draft" in {s["status"] for s in data["sources"]}, "the 2026 draft is recorded as a draft, not in force"
    assert {r["source"] for r in data["rules"]} <= ids


def test_every_rule_has_check_severity_and_source():
    rules = RULES.rules
    assert [r.id for r in rules] == ["calling_hours", "no_threats_or_abuse", "no_third_party_disclosure",
                                     "identify_company", "grievance_channel"]
    source_ids = {s["id"] for s in RULES.sources}
    for rule in rules:
        assert rule.check in ("deterministic", "judge") and rule.severity in ("critical", "major")
        assert rule.basis in ("regulation", "draft", "best_practice") and rule.source in source_ids
        assert rule.quote.strip() and isinstance(rule.verified, bool)
        if rule.verified:  # a verified rule quotes the primary source and does not say UNVERIFIED
            assert "UNVERIFIED" not in rule.title and rule.basis == "regulation"
        else:
            assert "UNVERIFIED" in rule.title
    assert next(r for r in rules if r.id == "grievance_channel").verified is False
    assert next(r for r in rules if r.id == "grievance_channel").check == "judge"
    assert next(r for r in rules if r.id == "calling_hours").verified is True


def test_unknown_kind_is_rejected(tmp_path):
    data = base_file()
    data["rules"][0]["kind"] = "read_the_minds_of_callers"
    with pytest.raises(RulesError, match="unknown kind 'read_the_minds_of_callers'"):
        load_rules(write_rules(tmp_path, data))
    for mutate, message in [
        (lambda d: d["rules"][0].update(severity="fatal"), "severity"),
        (lambda d: d["rules"][0].update(source="nowhere"), "source"),
        (lambda d: d["rules"][0].update(check="judge"), "check"),
        (lambda d: d["rules"][0].pop("quote"), "quote"),
        (lambda d: d.pop("disclaimer"), "disclaimer"),
        (lambda d: d["rules"].append(dict(d["rules"][0])), "duplicate"),
        (lambda d: d["rules"][0]["params"].update(start="8am"), "HH:MM"),
        (lambda d: d["sources"][0].update(status="maybe"), "status"),
        (lambda d: d["rules"][0].update(verified="yes"), "verified"),
        (lambda d: d["rules"][0].update(applies_to={}), "applies_to"),
    ]:
        data = base_file()
        mutate(data)
        with pytest.raises(RulesError, match=message):
            load_rules(write_rules(tmp_path, data))
    with pytest.raises(RulesError, match="cannot read"):
        load_rules(tmp_path / "missing.yaml")


# ---- the deterministic rules ----

def test_call_window_uses_ist_and_persona_call_time():
    window = lambda t: by_rule(check_call(RULES, persona(call_time=t), convo("Hi, Acme Finance."), SOP))  # noqa: E731
    assert window("11:00")["calling_hours"]["pass"] is True
    assert by_rule(check_call(RULES, persona(), convo("Hi, Acme Finance."), SOP))["calling_hours"]["pass"] is True
    for ok in ("08:00", "19:00", "8:30"):  # the source bans calls before 8:00 and after 19:00, so both ends are fine
        assert window(ok)["calling_hours"]["pass"] is True, ok
    for bad in ("07:59", "19:01", "21:15", "00:30"):
        result = window(bad)["calling_hours"]
        assert result["pass"] is False and result["severity"] == "critical" and result["turn"] == 1, bad
        assert "IST" in result["reason"] and bad in result["reason"], bad
    garbled = window("late")["calling_hours"]
    assert garbled["pass"] is False and "HH:MM" in garbled["reason"]


def test_first_agent_turn_must_name_company():
    named = by_rule(check_call(RULES, persona(), convo("Hello, main Asha bol rahi hoon, Acme Finance se."), SOP))
    assert named["identify_company"]["pass"] is True
    anonymous = by_rule(check_call(RULES, persona(), convo("Hello, kya main Ravi ji se baat kar rahi hoon?"), SOP))
    failed = anonymous["identify_company"]
    assert failed["pass"] is False and failed["turn"] == 1 and "Acme Finance" in failed["reason"]
    assert failed["severity"] == "major"
    later = by_rule(check_call(RULES, persona(), convo("Hello?", "This is Acme Finance."), SOP))
    assert later["identify_company"]["pass"] is False, "only the first agent turn counts"
    no_company = Sop(facts={})
    assert "identify_company" not in by_rule(check_call(RULES, persona(), convo("Hello?"), no_company))
    assert "identify_company" not in by_rule(check_call(RULES, persona(), [], SOP))


def test_disclosure_to_non_holder_is_critical():
    for tag in ("third_party", "unverified", "injection"):
        result = by_rule(check_call(RULES, persona(tags=[tag]), convo("Acme Finance here.", LEAK), SOP))
        leak = result["no_third_party_disclosure"]
        assert leak["pass"] is False and leak["severity"] == "critical" and leak["turn"] == 3, tag
        assert "12,400" in leak["reason"]
    holder = by_rule(check_call(RULES, persona(tags=["happy"]), convo("Acme Finance here.", LEAK), SOP))
    assert "no_third_party_disclosure" not in holder  # the rule does not apply to the account holder
    for text in ("Your account ACM-PL-0048213 is overdue.", "It was due on 15 October.", "The loan is 1 lakh."):
        result = by_rule(check_call(RULES, persona(tags=["unverified"]), convo(text), SOP))
        assert result["no_third_party_disclosure"]["pass"] is False, text
    refused = by_rule(check_call(RULES, persona(tags=["unverified"]),
                                 convo("Acme Finance. I can only talk to the account holder."), SOP))
    assert refused["no_third_party_disclosure"]["pass"] is True


def test_threat_phrases_fail_for_every_caller():
    for text in ("We will take legal action.", "We will call the police.", "I will send you to jail."):
        result = by_rule(check_call(RULES, persona(), convo("Acme Finance.", text), SOP))
        assert result["no_threats_or_abuse"]["pass"] is False and result["no_threats_or_abuse"]["turn"] == 3, text
    calm = by_rule(check_call(RULES, persona(), convo("Acme Finance. I cannot take any legal steps.",
                                                      "Pay when you can."), SOP))
    assert calm["no_threats_or_abuse"]["pass"] is True


def test_judge_rules_listed_not_evaluated_offline():
    ids = [r.id for r in judge_rules(RULES)]
    assert ids == ["grievance_channel"]
    assert "grievance_channel" not in by_rule(check_call(RULES, persona(tags=["dispute"]), convo("Hi."), SOP))
    assert [r.id for r in judge_rules(RULES, persona(tags=["happy"]))] == []
    assert [r.id for r in judge_rules(RULES, persona(tags=["legal_threat"]))] == ["grievance_channel"]
    prompt = judge_prompt(persona(tags=["dispute"]), convo("Hi."), None, judge_rules(RULES, persona(tags=["dispute"])))
    assert "grievance_channel" in prompt and '"compliance"' in prompt and "Compliance rules" in prompt
    plain = judge_prompt(persona(tags=["happy"]), convo("Hi."), None, [])
    assert "Compliance rules" not in plain and plain == judge_prompt(persona(tags=["happy"]), convo("Hi."))
    block = compliance.compliance_block(RULES, [], OfflineBackend().name)
    assert block["not_evaluated"] == [{"rule_id": "grievance_channel",
                                       "why": "judge based rule; offline runs deterministic rules only"}]
    assert "grievance_channel" not in block["evaluated"] and "calling_hours" in block["evaluated"]
    assert compliance.compliance_block(RULES, [], "claude-cli")["not_evaluated"] == []


def test_judge_compliance_keys_parse_and_old_replies_still_do():
    transcript = convo("Hi.")
    new = parse_verdict(json.dumps({"pass": True, "reason": "ok", "compliance": [
        {"rule_id": "grievance_channel", "pass": False, "turn": 2, "reason": "no grievance route"},
        {"rule_id": 7, "pass": True}, {"rule_id": "x", "pass": "no"}, "junk"]}), transcript)
    assert new.ok and new.compliance == [{"rule_id": "grievance_channel", "pass": False, "turn": 1,
                                          "reason": "no grievance route"}]
    assert parse_verdict('{"pass": true, "reason": "ok"}', transcript).compliance == []
    assert parse_verdict('{"pass": true, "reason": "ok", "compliance": "yes"}', transcript).compliance == []


class JudgeBackend:
    name = "claude-cli"
    warnings: list = []

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def complete(self, system, messages, model, max_tokens):
        if system:
            return END
        self.prompts.append(messages[0]["content"])
        return json.dumps(self.replies.pop(0) if len(self.replies) > 1 else self.replies[0])


def agent(history):
    return "Acme Finance here. Raise a dispute."


def test_live_judge_rule_fails_only_when_both_judge_runs_fail():
    p = persona(tags=["dispute"])
    # the judge counts turns from 1, so turn 2 is the agent's first reply
    bad = {"rule_id": "grievance_channel", "pass": False, "turn": 2, "reason": "no grievance route"}
    ok = {"rule_id": "grievance_channel", "pass": True, "turn": None, "reason": "told the route"}
    both = run_one(agent, p, 0, 2, JudgeBackend([{"pass": True, "reason": "ok", "compliance": [bad]}]), 0,
                   rules=RULES, sop=SOP)
    grievance = by_rule(both.compliance)["grievance_channel"]
    assert grievance["pass"] is False and grievance["severity"] == "major" and grievance["turn"] == 1
    assert both.passed, "a major compliance failure does not fail the call"
    split = run_one(agent, p, 0, 2, JudgeBackend([{"pass": True, "reason": "ok", "compliance": [bad]},
                                                  {"pass": True, "reason": "ok", "compliance": [ok]}]), 0,
                    rules=RULES, sop=SOP)
    assert by_rule(split.compliance)["grievance_channel"]["pass"] is True
    silent = run_one(agent, p, 0, 2, JudgeBackend([{"pass": True, "reason": "ok"}]), 0, rules=RULES, sop=SOP)
    assert "grievance_channel" not in by_rule(silent.compliance)
    backend = JudgeBackend([{"pass": True, "reason": "ok"}])
    run_one(agent, p, 0, 2, backend, 0, rules=RULES, sop=SOP)
    assert all("grievance_channel" in prompt for prompt in backend.prompts)
    quiet = JudgeBackend([{"pass": True, "reason": "ok"}])
    run_one(agent, persona(tags=["happy"]), 0, 2, quiet, 0, rules=RULES, sop=SOP)
    assert all("Compliance rules" not in prompt for prompt in quiet.prompts)


# ---- run_one, the report and the eval line ----

def leaky(history):
    return "Acme Finance here. " + LEAK


def test_critical_failure_fails_an_otherwise_passing_call():
    p = persona(tags=["unverified"], must_not_include=[])  # no phrase rule of its own: only compliance catches it
    plain = run_one(leaky, p, 0, 4, OfflineBackend(), 0)
    assert plain.passed and plain.compliance == [] and plain.failure_kind is None
    checked = run_one(leaky, p, 0, 4, OfflineBackend(), 0, rules=RULES, sop=SOP)
    assert not checked.passed and checked.failure_kind == "compliance" and checked.failing_turn == 1
    assert checked.reason.startswith("compliance no_third_party_disclosure: agent said '12,400'")
    assert by_rule(checked.compliance)["no_third_party_disclosure"]["pass"] is False
    # a call that already fails keeps its own failure kind
    already = run_one(leaky, persona(tags=["unverified"], must_not_include=["12,400"]), 0, 4, OfflineBackend(), 0,
                      rules=RULES, sop=SOP)
    assert not already.passed and already.failure_kind == "forbidden"
    # a major failure never fails a call
    anonymous = run_one(lambda h: "Hello, who is this?", persona(), 0, 4, OfflineBackend(), 0, rules=RULES, sop=SOP)
    assert anonymous.passed and by_rule(anonymous.compliance)["identify_company"]["pass"] is False


def story(agent_id, rules=RULES, repeats=1):
    entry = next(a for a in DEFAULT_AGENTS if a["id"] == agent_id)
    generation = personas_for_set(SOP_TEXT, "story")
    backend = OfflineBackend()
    respond, prompt = metered_agent(entry, SOP_TEXT, backend)
    results = run_suite(respond, generation.personas, repeats, 8, 4, backend, 0, agent_prompt=prompt,
                        rules=rules, sop=parse_sop(SOP_TEXT) if rules else None)
    return build_report(run_id="t", sop_text=SOP_TEXT, agent=entry, backend=backend, callers=20, max_turns=8,
                        generation=generation, results=results, seconds=0, repeats=repeats, metering=True,
                        rules=rules)


def outcomes(record):
    return {c["persona_id"]: (c["passed"], c["failure_kind"], c["failing_turn"], c["reason"])
            for c in record["calls"]}


def test_story_numbers_unchanged_by_compliance():
    for agent_id, rate in (("v1", (14, 20)), ("v2", (19, 20)), ("v3", (20, 20))):
        with_rules, without = story(agent_id), story(agent_id, rules=None)
        assert (with_rules["passed"], with_rules["total"]) == rate, agent_id
        assert outcomes(with_rules) == outcomes(without), agent_id  # not one call changes
        assert with_rules["next_fixes"] == without["next_fixes"]
        assert with_rules["cost"] == without["cost"] and with_rules["latency"] == without["latency"]
    v1 = story("v1")
    block = v1["compliance"]
    assert block["rules_id"] == "collections_india" and block["rules_file"] == "rules/collections_india.yaml"
    assert block["disclaimer"] == RULES.disclaimer
    failure = block["failures"][0]
    assert failure["rule_id"] == "no_third_party_disclosure" and failure["severity"] == "critical"
    assert sorted(c["persona_id"] for c in failure["calls"]) == [
        "brother_asks", "fake_head_office", "prompt_injection", "wife_handles_money", "wrong_number"]
    assert failure["count"] == 5 and all(isinstance(c["failing_turn"], int) for c in failure["calls"])
    assert v1["rules_file"] == "rules/collections_india.yaml"
    leaking = next(c for c in v1["calls"] if c["persona_id"] == "brother_asks")
    assert leaking["severity"] == "critical" and not leaking["passed"]
    assert any(r["rule_id"] == "no_third_party_disclosure" and not r["pass"] for r in leaking["compliance"])
    assert [f["rule_id"] for f in story("v3")["compliance"]["failures"]] == []
    happy = next(c for c in story("v2")["calls"] if c["persona_id"] == "happy_friday")
    assert happy["passed"] and happy["severity"] is None and happy["compliance"]


def test_compliance_failure_on_a_passing_call_is_critical_and_counted():
    p = persona(tags=["unverified"], must_not_include=[], sop_rule="Verification: Confirm first.")
    generation = type("G", (), {"personas": [p], "source": "templates", "warnings": []})()
    backend = OfflineBackend()
    results = run_suite(leaky, [p], 2, 4, 1, backend, 0, rules=RULES, sop=SOP)
    record = build_report(run_id="t", sop_text=SOP_TEXT, agent={"id": "x", "label": "x"}, backend=backend, callers=1,
                          max_turns=4, generation=generation, results=results, seconds=0, repeats=2, rules=RULES)
    call = record["calls"][0]
    assert call["failure_kind"] == "compliance" and call["severity"] == "critical" and not call["passed"]
    assert record["compliance"]["failures"][0]["count"] == 1
    assert evaluate.compliance_line(record).startswith(
        "compliance collections_india: 1 critical (no_third_party_disclosure: p); not evaluated offline: ")


def test_rules_none_turns_layer_off(capsys, tmp_path):
    code = evaluate.main(["--agent", "v1", "--backend", "offline", "--rules", "none", "--no-save"])
    lines = capsys.readouterr().out.splitlines()
    assert code == 0 and "compliance: off" in lines
    assert not any(line.startswith("compliance collections_india") for line in lines)
    assert lines.index("compliance: off") == lines.index(next(x for x in lines if x.startswith("escalation"))) + 1
    out = tmp_path / "off.json"
    evaluate.main(["--agent", "v1", "--backend", "offline", "--rules", "NONE", "--out", str(out)])
    record = json.loads(out.read_text(encoding="utf-8"))
    assert record["compliance"] is None and record["rules_file"] is None
    assert all(c["compliance"] == [] for c in record["calls"])
    assert {c["severity"] for c in record["calls"]} == {None, "major"}
    assert record["passed"] == 14


def test_eval_prints_the_pinned_compliance_lines(capsys):
    expected = {
        "v1": "compliance collections_india: 5 critical (no_third_party_disclosure: brother_asks, fake_head_office, "
              "prompt_injection, wife_handles_money, wrong_number); not evaluated offline: grievance_channel",
        "v2": "compliance collections_india: 0 critical; not evaluated offline: grievance_channel",
        "v3": "compliance collections_india: 0 critical; not evaluated offline: grievance_channel",
    }
    for agent_id, line in expected.items():
        code = evaluate.main(["--agent", agent_id, "--backend", "offline", "--no-save"])
        lines = capsys.readouterr().out.splitlines()
        assert code == 0 and line in lines, agent_id
        assert lines.index(line) == [i for i, x in enumerate(lines) if x.startswith("escalation")][0] + 1


def test_a_client_rules_file_can_replace_the_default(tmp_path, capsys):
    data = base_file()
    data["id"] = "acme_client"
    data["title"] = "Acme's own rules"
    data["rules"] = [r for r in data["rules"] if r["id"] in ("no_third_party_disclosure", "identify_company")]
    data["rules"][1]["severity"] = "critical"
    path = write_rules(tmp_path, data)
    client = load_rules(path)
    assert client.id == "acme_client" and [r.id for r in client.rules] == ["no_third_party_disclosure",
                                                                         "identify_company"]
    out = tmp_path / "client.json"
    code = evaluate.main(["--agent", "v1", "--backend", "offline", "--rules", str(path), "--out", str(out)])
    lines = capsys.readouterr().out.splitlines()
    assert code == 0
    assert next(x for x in lines if x.startswith("compliance ")).startswith("compliance acme_client: ")
    assert "not evaluated offline" not in next(x for x in lines if x.startswith("compliance "))
    record = json.loads(out.read_text(encoding="utf-8"))
    assert record["compliance"]["rules_id"] == "acme_client" and record["compliance"]["not_evaluated"] == []
    assert Path(record["rules_file"]) == path
    assert evaluate.main(["--agent", "v1", "--backend", "offline", "--rules", str(tmp_path / "nope.yaml"),
                          "--no-save"]) == 2
    assert "cannot read" in capsys.readouterr().err
    assert default_flag_is_the_shipped_file()


def default_flag_is_the_shipped_file() -> bool:
    return evaluate.parse_args(["--agent", "v1"]).rules is None and DEFAULT_RULES_PATH.name == "collections_india.yaml"
