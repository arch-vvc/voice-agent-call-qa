"""Escalation scoring: handoff notes, the grade out of 5, the matrix, must_escalate and the eval line. No network."""
import json

from callqa import escalation, evaluate
from callqa.escalation import (
    HANDOFF_FIELDS,
    expected_escalation,
    grade_handoff,
    matrix,
    outcome,
    score_handoff,
    split_handoff,
)
from callqa.persona_sets import personas_for_set
from callqa.report import build_report
from callqa.server import DEFAULT_AGENTS, metered_agent, read_sample_sop
from callqa.sop import parse_sop
from evals import example_agent, example_agent_fixed
from evals.backends import OfflineBackend
from evals.run_evals import END, parse_verdict, run_one, run_suite

SOP = read_sample_sop()
FULL = {"caller": "Ravi Kumar", "issue": "says the loan is not his", "tried": "asked for the loan id",
        "why": "disputes team owns it", "next_step": "call back in 2 days"}


def note_text(note: dict) -> str:
    return f"<handoff>{json.dumps(note)}</handoff>"


def persona(**extra) -> dict:
    base = {"id": "p", "name": "P", "language": "en", "tags": [], "persona": "A caller.",
            "opening": "Hello?", "script": ["I want a human."], "must_include": [],
            "must_not_include": [], "expectations": "PASS if fine.", "sop_rule": "Escalation: x", "fix_hint": "fix"}
    return {**base, **extra}


def agent_with(reply_with_note: str):
    def respond(history):
        return reply_with_note if len(history) >= 3 else "Hello, this is Acme Finance."
    return respond


def test_split_handoff_strips_note_from_spoken_text():
    spoken, note = split_handoff(f"Main dispute team ko bhej deta hoon. {note_text(FULL)}")
    assert spoken == "Main dispute team ko bhej deta hoon." and note == FULL
    assert split_handoff("No note here.") == ("No note here.", None)
    spoken, note = split_handoff(f"A <handoff>{json.dumps({'caller': 'first'})}</handoff> B {note_text(FULL)}")
    assert note == FULL and "handoff" not in spoken


def test_bad_handoff_json_scores_0_with_all_fields_missing():
    spoken, note = split_handoff("Okay. <handoff>{not json</handoff>")
    assert spoken == "Okay." and note == {"raw": "{not json"}
    assert score_handoff(note) == (0, list(HANDOFF_FIELDS))
    assert split_handoff("x <handoff>[1, 2]</handoff>")[1] == {"raw": "[1, 2]"}
    assert score_handoff(None) == (0, list(HANDOFF_FIELDS))


def test_handoff_score_counts_fields_of_3_chars_or_more():
    assert score_handoff(FULL) == (5, [])
    assert score_handoff({**FULL, "tried": "no"}) == (4, ["tried"])
    assert score_handoff({**FULL, "tried": "abc"}) == (5, [])
    assert score_handoff({**FULL, "why": "  ab  "}) == (4, ["why"])
    assert score_handoff({**FULL, "issue": 12345}) == (4, ["issue"])
    partial = {"caller": "Ravi", "issue": "dispute"}
    assert score_handoff(partial) == (2, ["tried", "why", "next_step"])


def test_matrix_counts_every_run():
    assert [outcome(*pair) for pair in ((True, True), (False, True), (True, False), (False, False))] == [
        "TP", "FP", "FN", "TN"]
    assert matrix(["TP", "FN", "FN", "TN", "TN", "TN", "FP"]) == {"tp": 1, "fp": 1, "fn": 2, "tn": 3}
    rows = [{"persona_id": "a", "outcome": "FN", "handoff_score": None},
            {"persona_id": "a", "outcome": "FN", "handoff_score": None},
            {"persona_id": "b", "outcome": "TP", "handoff_score": 4},
            {"persona_id": "c", "outcome": "FP", "handoff_score": 5},
            {"persona_id": "d", "outcome": "TN", "handoff_score": None}]
    block = escalation.escalation_block(rows, ["d", "c", "b", "a"])
    assert block == {"matrix": {"tp": 1, "fp": 1, "fn": 2, "tn": 1}, "accuracy": 0.4, "runs": 5, "handoff_avg": 4.5,
                     "missed": ["a"], "unnecessary": ["c"]}
    assert escalation.escalation_block([], [])["handoff_avg"] is None


def test_must_escalate_without_note_fails_missing():
    p = persona(must_escalate=True)
    result = run_one(agent_with("I understand, sir."), p, 0, 4, OfflineBackend(), 0)
    assert not result.passed and result.failure_kind == "missing"
    assert result.reason == "agent never handed over (no handoff note)"
    assert result.escalated is False and result.handoff is None
    ok = run_one(agent_with(f"I will pass this on. {note_text(FULL)}"), p, 0, 4, OfflineBackend(), 0)
    assert ok.passed and ok.escalated and ok.handoff == FULL
    assert all("handoff" not in m["content"] for m in ok.transcript)
    other = run_one(agent_with("I understand, sir."), persona(), 0, 4, OfflineBackend(), 0)
    assert other.passed and not other.escalated  # no must_escalate: a missing note never fails the call


def test_must_escalate_failure_keeps_an_earlier_forbidden_failure():
    p = persona(must_escalate=True, must_not_include=["sue you"])
    result = run_one(agent_with("We will sue you."), p, 0, 4, OfflineBackend(), 0)
    assert result.failure_kind == "forbidden"


class JudgeBackend:
    """Live backend stand in: the caller hangs up at once and every judge call returns the same JSON."""
    name = "claude-cli"
    warnings: list = []

    def __init__(self, reply: dict):
        self.reply = json.dumps(reply)

    def complete(self, system, messages, model, max_tokens):
        return END if system else self.reply


def live_run(p, reply, agent_reply="I understand."):
    return run_one(agent_with(agent_reply), p, 0, 2, JudgeBackend(reply), 0)


def test_judge_cannot_score_absent_fields():
    assert grade_handoff({"caller": "Ravi"}, 5, []) == (1, ["issue", "tried", "why", "next_step"])
    assert grade_handoff(FULL, 5, []) == (5, [])
    assert grade_handoff(FULL, 3, []) == (3, [])
    assert grade_handoff(FULL, 5, ["tried", "bogus"]) == (4, ["tried"])
    assert grade_handoff(FULL, None, None) == (5, [])
    assert grade_handoff(FULL, True, []) == (5, [])
    assert grade_handoff(FULL, -2, []) == (0, [])
    assert grade_handoff({**FULL, "why": ""}, 5, ["tried"]) == (3, ["tried", "why"])


def test_old_judge_json_without_new_keys_still_parses():
    verdict = parse_verdict('{"pass": false, "reason": "leaked", "broken": "never", "turn": 2, "notes": "n"}',
                            [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}])
    assert (verdict.ok, verdict.reason, verdict.turn, verdict.notes) == (False, "leaked", 1, "n")
    assert verdict.escalated is None and verdict.handoff_score is None and verdict.handoff_missing == []
    new = parse_verdict('{"pass": true, "reason": "ok", "escalated": true, "handoff_score": 4, '
                        '"handoff_missing": ["tried"]}', [])
    assert new.ok and new.escalated is True and new.handoff_score == 4 and new.handoff_missing == ["tried"]
    odd = parse_verdict('{"pass": true, "reason": "ok", "escalated": "yes", "handoff_score": "x", '
                        '"handoff_missing": "tried"}', [])
    assert odd.ok and odd.escalated is None and odd.handoff_score is None and odd.handoff_missing == []


def test_live_escalated_needs_both_judge_runs_and_the_judge_cannot_overscore():
    p = persona(must_escalate=True)
    both = live_run(p, {"pass": True, "reason": "ok", "escalated": True, "handoff_score": 5, "handoff_missing": []})
    assert both.passed and both.escalated and both.handoff is None
    one = live_run(p, {"pass": True, "reason": "ok", "escalated": False})
    assert not one.passed and one.failure_kind == "missing" and not one.escalated
    plain = live_run(persona(), {"pass": True, "reason": "ok"})
    assert plain.passed and not plain.escalated


def test_live_split_judges_do_not_count_as_escalated():
    replies = iter([{"pass": True, "reason": "ok", "escalated": True}, {"pass": True, "reason": "ok",
                                                                        "escalated": False}])

    class Split(JudgeBackend):
        def complete(self, system, messages, model, max_tokens):
            return END if system else json.dumps(next(replies))

    result = run_one(agent_with("I understand."), persona(must_escalate=True), 0, 2, Split({}), 0)
    assert not result.escalated and result.failure_kind == "missing"


def test_the_last_handoff_note_wins():
    first, last = {**FULL, "issue": "first issue"}, {**FULL, "issue": "last issue"}

    def respond(history):
        return {1: "Hi.", 3: f"One moment. {note_text(first)}"}.get(len(history), f"Done. {note_text(last)}")

    result = run_one(respond, persona(script=["Wait.", "Okay."]), 0, 4, OfflineBackend(), 0)
    assert result.handoff == last


def test_judge_prompt_asks_for_escalation_only_when_it_matters():
    from evals.run_evals import judge_prompt
    transcript = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    assert "handoff_score" not in judge_prompt(persona(), transcript)
    asked = judge_prompt(persona(must_escalate=True), transcript)
    assert '"escalated"' in asked and '"handoff_score"' in asked and '"handoff_missing"' in asked
    with_note = judge_prompt(persona(must_escalate=True), transcript, FULL)
    assert "next_step" in with_note and "not his" in with_note


def test_expected_escalation_quotes_the_sop_rule_or_says_none():
    sop = parse_sop(SOP)
    wants, reason = expected_escalation(persona(tags=["escalation"]), sop)
    assert wants and reason == f"Escalation: {sop.rules['Escalation'][0]}"
    assert expected_escalation(persona(tags=["escalation"]), SOP)[0] is True  # SOP text works too
    off, why = expected_escalation(persona(tags=["hardship"]), sop)
    assert off is False and why == "The SOP has no rule that asks for a handoff for this caller."
    assert expected_escalation(persona(tags=["legal_threat"]), parse_sop("## Never\n- Be rude.\n"))[0] is False
    extended = parse_sop(SOP + "\n## Escalation\n- Hand the call to a human when the caller threatens legal action "
                         "or a complaint to a regulator.\n")
    assert expected_escalation(persona(tags=["legal_threat"]), extended)[0] is True
    assert expected_escalation(persona(tags=["escalation"]), parse_sop("## Never\n- Be rude.\n")) == (
        False, "The SOP has no rule that asks for a handoff for this caller.")
    given = persona(tags=[], expected_escalation=True, escalation_reason="SOP says so")
    assert expected_escalation(given, sop) == (True, "SOP says so")
    assert expected_escalation(persona(tags=["escalation"], expected_escalation=False), sop)[0] is False


def story_record(agent_id, repeats=1):
    sop = read_sample_sop()
    generation = personas_for_set(sop, "story")
    entry = next(a for a in DEFAULT_AGENTS if a["id"] == agent_id)
    backend = OfflineBackend()
    respond, prompt = metered_agent(entry, sop, backend)
    results = run_suite(respond, generation.personas, repeats, 8, 4, backend, 0, agent_prompt=prompt)
    return build_report(run_id="t", sop_text=sop, agent=entry, backend=backend, callers=20, max_turns=8,
                        generation=generation, results=results, seconds=0, repeats=repeats, metering=True)


def test_story_numbers_unchanged_by_escalation_scoring():
    v1, v2 = story_record("v1"), story_record("v2")
    assert (v1["passed"], v1["total"]) == (14, 20) and (v2["passed"], v2["total"]) == (19, 20)
    assert [c["persona_id"] for c in v2["calls"] if not c["passed"]] == ["wants_manager"]
    for record in (v1, v2):
        assert record["escalation"]["matrix"] == {"tp": 0, "fp": 0, "fn": 1, "tn": 19}
        assert record["escalation"]["missed"] == ["wants_manager"] and record["escalation"]["handoff_avg"] is None
        assert record["escalation"]["accuracy"] == 0.95 and record["escalation"]["runs"] == 20
    call = next(c for c in v2["calls"] if c["persona_id"] == "wants_manager")
    assert call["escalation"]["expected"] is True and call["escalation"]["outcome"] == "FN"
    assert call["escalation"]["escalated"] is False and call["escalation"]["handoff"] is None
    assert call["escalation"]["reason"].startswith("Escalation: ")
    quiet = next(c for c in v2["calls"] if c["persona_id"] == "lost_job")
    assert quiet["escalation"]["outcome"] == "TN" and quiet["escalation"]["handoff_score"] is None


def test_report_counts_notes_from_an_escalating_agent():
    p = persona(tags=["escalation"], must_escalate=True)
    backend, sop = OfflineBackend(), read_sample_sop()
    generation = type("G", (), {"personas": [p], "source": "templates", "warnings": []})()
    results = run_suite(agent_with(f"Done. {note_text({**FULL, 'tried': ''})}"), [p], 2, 4, 1, backend, 0)
    record = build_report(run_id="t", sop_text=sop, agent={"id": "x", "label": "x"}, backend=backend, callers=1,
                          max_turns=4, generation=generation, results=results, seconds=0, repeats=2)
    assert record["escalation"]["matrix"] == {"tp": 2, "fp": 0, "fn": 0, "tn": 0}
    assert record["escalation"]["handoff_avg"] == 4.0
    call = record["calls"][0]["escalation"]
    assert call["handoff_score"] == 4 and call["handoff_missing"] == ["tried"] and call["escalated"] is True
    assert call["handoff"]["caller"] == "Ravi Kumar"


def test_handoff_note_counts_toward_agent_output_and_latency():
    p = persona()
    plain = run_one(agent_with("Sure, passing you on."), p, 0, 4, OfflineBackend(), 0, agent_prompt="PROMPT")
    noted = run_one(agent_with(f"Sure, passing you on. {note_text(FULL)}"), p, 0, 4, OfflineBackend(), 0,
                    agent_prompt="PROMPT")
    extra = len(note_text(FULL))
    assert noted.usage["agent"]["output_tokens"] > plain.usage["agent"]["output_tokens"] + extra // 4 - 3
    assert noted.turn_latency_ms[-1] - plain.turn_latency_ms[-1] >= 10 * extra
    assert noted.turn_latency_ms[0] == plain.turn_latency_ms[0]
    assert noted.transcript == plain.transcript  # the spoken transcript carries no note


def test_eval_prints_the_escalation_line(capsys):
    for agent, line in (("v1", "escalation accuracy 95% (19/20 runs)  TP 0 FP 0 FN 1 TN 19  handoff avg n/a"),
                        ("v2", "escalation accuracy 95% (19/20 runs)  TP 0 FP 0 FN 1 TN 19  handoff avg n/a")):
        code = evaluate.main(["--agent", agent, "--set", "story", "--backend", "offline", "--no-save"])
        lines = capsys.readouterr().out.splitlines()
        assert code == 0 and lines[6] == line and lines[5].startswith("latency (simulated)")


def test_escalation_line_format():
    record = {"escalation": {"matrix": {"tp": 3, "fp": 0, "fn": 9, "tn": 84}, "accuracy": 0.9062, "runs": 96,
                             "handoff_avg": 4.0, "missed": [], "unnecessary": []}}
    assert evaluate.escalation_line(record) == (
        "escalation accuracy 91% (87/96 runs)  TP 3 FP 0 FN 9 TN 84  handoff avg 4.0/5")
    record["escalation"]["handoff_avg"] = None
    assert evaluate.escalation_line(record).endswith("handoff avg n/a")


def test_harness_default_agent_still_has_no_escalation_note():
    for respond in (example_agent.respond, example_agent_fixed.respond):
        reply = respond([{"role": "user", "content": "Manager se baat karao."}])
        assert split_handoff(reply)[1] is None
