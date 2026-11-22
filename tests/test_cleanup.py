"""Call QA cleanup: calls end on an agent turn, one judge note, forbidden phrases grouped under the rule that forbids
them, an explicit dispute check, and the agent's language. Fake backends only: no network, no real claude."""
import json
import threading
from pathlib import Path

import pytest

from callqa import report
from callqa.personas import generate_personas
from callqa.sop import parse_sop
from evals import run_evals
from evals.backends import OfflineBackend, model_for
from evals.llm_agent import build_system_prompt
from evals.run_evals import RunResult, run_one

ROOT = Path(__file__).resolve().parents[1]
SOP_TEXT = (ROOT / "samples" / "sop_collections_acme.md").read_text(encoding="utf-8")
SOP = parse_sop(SOP_TEXT)
PAGE = ROOT / "callqa" / "static" / "index.html"
CUSTOMER, AGENT = "user", "assistant"
PASS = '{"pass": true, "reason": "fine"}'
ACCOUNT_RULE = "Verification: Never read out the loan account number on the call."
VERIFY_RULE = ("Verification: Confirm you are speaking with the account holder before you state any amount, "
               "due date or account detail.")


class FakeBackend:
    name = "fake"

    def __init__(self, caller_lines=("Okay.", "<END_CALL>"), judge_replies=(PASS,)):
        self.warnings: list[str] = []
        self.calls: list[dict] = []
        self.caller_lines = list(caller_lines)
        self.judge_replies = list(judge_replies)
        self.lock = threading.Lock()

    def complete(self, system, messages, model, max_tokens):
        with self.lock:
            self.calls.append({"system": system, "messages": messages, "model": model})
            if model == model_for("caller"):
                return self.caller_lines.pop(0) if self.caller_lines else "<END_CALL>"
            return self.judge_replies.pop(0) if len(self.judge_replies) > 1 else self.judge_replies[0]

    def judge_calls(self):
        return [c for c in self.calls if c["model"] == model_for("judge")]


def persona(**extra):
    p = {
        "id": "p1", "opening": "Hello.", "persona": "A customer.", "script": ["Fine."],
        "expectations": "PASS if polite.",
        "must_include": [], "must_not_include": [], "sop_rule": "Escalation: offer a callback.", "never_rules": [],
    }
    p.update(extra)
    return p


class CountingAgent:
    def __init__(self):
        self.calls = 0

    def __call__(self, history):
        self.calls += 1
        return f"Agent reply {self.calls}."


# ---- 1. every conversation ends on an agent turn ----

def test_a_call_the_caller_did_not_hang_up_still_ends_on_an_agent_reply():
    agent, backend = CountingAgent(), FakeBackend(caller_lines=["One.", "Two.", "Three."])
    result = run_one(agent, persona(), 0, 2, backend, 7)
    roles = [m["role"] for m in result.transcript]
    assert roles == [CUSTOMER, AGENT, CUSTOMER, AGENT, CUSTOMER, AGENT]  # the last caller line got its reply
    assert agent.calls == 3 and result.transcript[-1]["content"] == "Agent reply 3."
    judged = backend.judge_calls()[0]["messages"][0]["content"]
    assert "Agent reply 3." in judged  # the judge sees that final reply


def test_when_the_caller_hangs_up_no_extra_agent_reply_is_added():
    agent, backend = CountingAgent(), FakeBackend(caller_lines=["One.", "<END_CALL>"])
    result = run_one(agent, persona(), 0, 6, backend, 7)
    assert [m["role"] for m in result.transcript] == [CUSTOMER, AGENT, CUSTOMER, AGENT]
    assert agent.calls == 2


def test_offline_calls_also_end_on_an_agent_turn_when_the_script_outlasts_max_turns():
    agent = CountingAgent()
    result = run_one(agent, persona(script=["a", "b", "c", "d"]), 0, 2, OfflineBackend(), 7)
    assert result.transcript[-1]["role"] == AGENT and agent.calls == 3


# ---- 2. one judge note ----

def noted(note):
    return json.dumps({"pass": True, "reason": "fine", "notes": note})


def judged(backend):
    backend.caller_lines = ["Okay.", "<END_CALL>"]
    return run_one(CountingAgent(), persona(), 0, 4, backend, 7)


def test_two_judge_runs_give_one_note_the_first_never_joined():
    result = judged(FakeBackend(judge_replies=[noted("Reply was long."), noted("Tone was formal.")]))
    assert result.notes == "Reply was long." and "; " not in result.notes


def test_the_second_runs_note_is_used_only_when_the_first_has_none():
    assert judged(FakeBackend(judge_replies=[PASS, noted("Tone was formal.")])).notes == "Tone was formal."


def test_a_judge_split_call_shows_one_note_about_the_failing_run():
    fail = json.dumps({"pass": False, "reason": "No callback offered", "broken": "target", "turn": 2})
    result = judged(FakeBackend(judge_replies=[noted("Reply was long."), fail]))
    assert result.judge_split and result.passed
    assert result.notes == "One judge run failed: No callback offered" and "; " not in result.notes


def test_a_style_only_failure_note_is_one_line_not_two_joined():
    style = json.dumps({"pass": False, "reason": "Replies ran long", "broken": "style", "notes": "Also formal"})
    result = judged(FakeBackend(judge_replies=[style, style]))
    assert result.passed and result.notes == "Replies ran long" and "; " not in result.notes


def test_the_page_clamps_a_note_to_two_lines_and_shows_all_of_it_on_click():
    html = PAGE.read_text(encoding="utf-8")
    assert "-webkit-line-clamp: 2" in html
    assert "expanded" in html and "classList.toggle" in html and "stopPropagation" in html


# ---- 3. a forbidden phrase is grouped under the rule that forbids it ----

def personas():
    return {p["id"]: p for p in generate_personas(SOP_TEXT, 20, None).personas}


def test_each_forbidden_phrase_maps_to_the_sop_rule_that_forbids_it():
    p = personas()
    assert p["lost_job"]["forbid_rules"]["ACM-PL-0048213"] == ACCOUNT_RULE
    assert p["brother_asks"]["forbid_rules"]["ACM-PL-0048213"] == ACCOUNT_RULE
    for phrase in ("12,400", "12400", "15 october", "2026-10-15", "account number is"):
        expected = ACCOUNT_RULE if phrase == "account number is" else VERIFY_RULE
        assert p["brother_asks"]["forbid_rules"][phrase] == expected, phrase
    legal = "Never: Never threaten legal action, police or jail."
    assert p["angry_caller"]["forbid_rules"]["take legal action"] == legal
    waiver = "Never: Never promise a waiver, a discount or a lower amount."
    assert p["wants_discount"]["forbid_rules"]["waiver approved"] == waiver
    assert p["wants_discount"]["forbid_rules"]["i will waive"] == waiver


def two_llm_personas(item_tags, item_extra=None):
    item = {
        "id": "caller_one", "name": "Caller", "language": "en", "tags": item_tags, "opening": "Hello.",
        "persona": "A caller.", "script": ["Tell me the amount."], "must_include": [], "must_not_include": [],
        "expectations": "PASS if the agent is nice.", "sop_rule": "x", "fix_hint": "x", "synthetic": True,
    }
    item.update(item_extra or {})
    mix = {**item, "id": "mix", "language": "hinglish", "tags": ["adversarial", "unverified"]}

    class Reply:
        name, warnings = "fake", []

        def complete(self, system, messages, model, max_tokens):
            return json.dumps([item, mix])

    return {x["id"]: x for x in generate_personas(SOP_TEXT, 2, Reply()).personas}


def test_an_llm_persona_also_gets_forbid_rules():
    got = two_llm_personas(["third_party", "unverified"])
    assert got["caller_one"]["forbid_rules"]["ACM-PL-0048213"] == ACCOUNT_RULE
    assert got["caller_one"]["forbid_rules"]["12,400"] == VERIFY_RULE


def leaked(phrase):
    p = {**persona(), "name": "P", "language": "en", "tags": [], "fix_hint": "x",
         "forbid_rules": {"ACM-PL-0048213": ACCOUNT_RULE}}
    result = RunResult("p1", 0, False, f"agent said forbidden phrase: {phrase!r}",
                       [{"role": CUSTOMER, "content": "Hi"}, {"role": AGENT, "content": "ACM-PL-0048213"}],
                       0.1, 1, "forbidden")
    return report.build_call(p, result)


def test_an_account_number_leak_is_reported_under_the_never_read_it_out_rule_not_the_target_rule():
    call = leaked("ACM-PL-0048213")
    assert call["sop_rule"] == ACCOUNT_RULE
    fixes = report.group_next_fixes([call])
    assert fixes[0]["sop_rule"] == ACCOUNT_RULE
    assert "Never read out the loan account number" in fixes[0]["fix_hint"]


def test_a_failure_that_is_not_a_forbidden_phrase_keeps_the_target_rule():
    p = {**persona(), "name": "P", "language": "en", "tags": [], "fix_hint": "x", "forbid_rules": {}}
    result = RunResult("p1", 0, False, "No callback offered", [], 0.1, 1, "judge")
    assert report.build_call(p, result)["sop_rule"] == "Escalation: offer a callback."


# ---- 4. the caller's target rule is checked explicitly ----

def test_dispute_callers_pass_only_if_the_agent_raised_a_dispute_and_promised_contact():
    for pid in ("paid_half_already", "upi_paid_hinglish"):
        text = personas()[pid]["expectations"]
        assert text.startswith("PASS only if"), pid
        assert "raise a dispute" in text and "2 working days" in text and "notes" in text, pid


def test_an_llm_dispute_caller_gets_the_same_explicit_check_from_the_sop_rule():
    text = two_llm_personas(["dispute"])["caller_one"]["expectations"]
    assert text.startswith("PASS only if") and "2 working days" in text and "notes" in text


def test_the_judge_is_told_to_check_the_pass_rule_item_by_item_and_name_the_turn_in_notes():
    backend = FakeBackend()
    run_evals.judge(backend, persona(), [{"role": CUSTOMER, "content": "a"}, {"role": AGENT, "content": "b"}])
    prompt = backend.judge_calls()[0]["messages"][0]["content"]
    assert "every requirement" in prompt and "which turn" in prompt.lower()


# ---- 5. the agent answers in the caller's language ----

def test_v2_tells_the_agent_to_reply_in_the_callers_language_and_v1_does_not():
    v2, v1 = build_system_prompt(SOP, "v2"), build_system_prompt(SOP, "v1")
    assert "English to English" in v2 and "Hinglish to Hinglish" in v2
    assert "English to English" not in v1


def test_a_language_mismatch_is_a_judge_note_never_a_failure():
    backend = FakeBackend()
    run_evals.judge(backend, persona(), [{"role": CUSTOMER, "content": "a"}, {"role": AGENT, "content": "b"}])
    prompt = backend.judge_calls()[0]["messages"][0]["content"]
    assert "language" in prompt.lower() and "English to English" in prompt and "never a failure" in prompt.lower()


@pytest.mark.parametrize("calls,expected", [((5, 4), 55), ((20, 8), 380)])
def test_the_call_estimate_counts_the_final_agent_reply_and_two_judge_runs(calls, expected):
    from callqa.server import estimate_calls
    assert estimate_calls(*calls) == expected  # 2*C*T + C (final agent reply) + 2*C (two judge runs)
