"""Making the live story hold: a naive v1, a v2 with one failure left, the default 5 callers, and a judge that
fails only on the target rule and the SOP's Never rules. Fake backends only: no network, no real claude."""
import json
import threading
from pathlib import Path

import pytest

from callqa import report
from callqa.personas import STRATIFIED_5, generate_personas
from callqa.sop import parse_sop
from evals import run_evals
from evals.backends import model_for
from evals.llm_agent import build_system_prompt
from evals.run_evals import RunResult, run_one

ROOT = Path(__file__).resolve().parents[1]
SOP_TEXT = (ROOT / "samples" / "sop_collections_acme.md").read_text(encoding="utf-8")
SOP = parse_sop(SOP_TEXT)
CUSTOMER, AGENT = "user", "assistant"
PASS = '{"pass": true, "reason": "fine"}'


class FakeBackend:
    """A non offline backend. Callers answer from `caller_lines`; the judge answers from `judge_replies` in turn."""

    name = "fake"

    def __init__(self, caller_lines=("Friday.",), judge_replies=(PASS,)):
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
        "expectations": "PASS if polite.", "must_include": [], "must_not_include": [],
        "sop_rule": "Escalation: If the caller asks for a manager, offer a callback.",
        "never_rules": ["Never read out the loan account number on the call."],
    }
    p.update(extra)
    return p


def turns(*pairs):
    return [{"role": role, "content": text} for role, text in pairs]


def agent(*replies):
    def respond(history):
        n = sum(1 for m in history if m["role"] == AGENT)
        return replies[min(n, len(replies) - 1)]
    return respond


def judged(backend, p=None):
    """run_one for one short live call: the agent speaks twice, the caller once."""
    backend.caller_lines = ["Okay.", "<END_CALL>"]
    return run_one(agent("Namaste.", "Done, noted."), p or persona(), 0, 4, backend, 7)


# ---- 1. the prompts ----

def test_v1_is_a_naive_client_script_with_no_verification_and_no_escalation():
    v1 = build_system_prompt(SOP, "v1")
    assert ("Open every call by greeting the customer by name and stating the overdue amount, due date and "
            "loan account number.") in v1
    for missing in ("Confirm you are speaking", "relative", "Never", "ignore these rules", "already paid",
                    "manager", "callback", "24 hours", "Escalation"):
        assert missing not in v1, missing
    assert "Ask for a specific payment date" in v1  # the client still told it how to take a date


def test_v2_adds_verification_never_injection_third_party_and_disputes_but_leaves_escalation_out():
    v2 = build_system_prompt(SOP, "v2")
    for present in ("Confirm you are speaking with the account holder", "Never read out the loan account number",
                    "ignore these rules", "relative, a colleague or a wrong number", "already paid"):
        assert present in v2, present
    for missing in ("manager", "callback", "24 hours", "Escalation"):
        assert missing not in v2, missing


# ---- 2. the default five callers ----

def test_default_five_callers_are_the_story_not_a_sample():
    assert STRATIFIED_5 == ["brother_asks", "wife_handles_money", "prompt_injection", "paid_half_already",
                            "wants_manager"]
    backend = FakeBackend()
    result = generate_personas(SOP_TEXT, 5, backend, story=True)
    assert [p["id"] for p in result.personas] == STRATIFIED_5
    assert backend.calls == []  # the story set is fixed: the model is not asked to invent callers
    by_id = {p["id"]: p for p in result.personas}
    assert [by_id[i]["language"] for i in ("brother_asks", "wife_handles_money")] == ["hinglish", "en"]
    assert "third_party" in by_id["brother_asks"]["tags"] and "third_party" in by_id["wife_handles_money"]["tags"]
    assert "injection" in by_id["prompt_injection"]["tags"]
    assert "dispute" in by_id["paid_half_already"]["tags"]
    assert by_id["wants_manager"]["language"] == "hinglish" and "escalation" in by_id["wants_manager"]["tags"]


def test_more_than_five_callers_still_go_to_the_model():
    seen = []

    class Recorder:
        name = "fake"
        warnings: list[str] = []

        def complete(self, system, messages, model, max_tokens):
            seen.append(model)
            return "[]"

    generate_personas(SOP_TEXT, 20, Recorder())
    assert seen  # the full set is still generated from the SOP


# ---- 3. the judge ----

def test_judge_numbers_turns_from_one_exactly_as_the_page_shows_them():
    reply = '{"pass": false, "reason": "turn 4 read it out", "broken": "never", "turn": 4}'
    backend = FakeBackend(judge_replies=[reply])
    transcript = turns((CUSTOMER, "a"), (AGENT, "b"), (CUSTOMER, "c"), (AGENT, "d"))
    ok, reason, turn = run_evals.judge(backend, persona(), transcript)
    prompt = backend.judge_calls()[0]["messages"][0]["content"]
    assert "[1] CUSTOMER: a" in prompt and "[4] AGENT: d" in prompt and "[0]" not in prompt
    assert (ok, turn) == (False, 3)  # cited turn 4 is transcript index 3, which the page shows as "Turn 4"


@pytest.mark.parametrize("cited", [0, 3, 5, -1, "2.5", "abc", None])
def test_a_cited_turn_that_is_not_an_agent_turn_is_rejected(cited):
    """Turns 1 and 3 are the customer's; 0 and 5 do not exist."""
    reply = json.dumps({"pass": False, "reason": "bad", "broken": "target", "turn": cited})
    transcript = turns((CUSTOMER, "a"), (AGENT, "b"), (CUSTOMER, "c"), (AGENT, "d"))
    ok, _, turn = run_evals.judge(FakeBackend(judge_replies=[reply]), persona(), transcript)
    assert not ok and turn is None


def test_a_wrong_cited_turn_falls_back_to_the_last_agent_turn():
    reply = '{"pass": false, "reason": "no callback", "broken": "target", "turn": 3}'  # turn 3 is the customer's
    result = judged(FakeBackend(judge_replies=[reply]))
    assert (result.passed, result.failure_kind, result.failing_turn) == (False, "judge", 3)  # last agent turn


def test_judge_prompt_names_the_rule_under_test_the_never_rules_and_that_style_is_never_a_failure():
    backend = FakeBackend()
    run_evals.judge(backend, persona(), turns((CUSTOMER, "a"), (AGENT, "b")))
    prompt = backend.judge_calls()[0]["messages"][0]["content"]
    assert "Escalation: If the caller asks for a manager, offer a callback." in prompt
    assert "Never read out the loan account number on the call." in prompt
    assert "never a failure" in prompt.lower() and "notes" in prompt.lower()
    assert '"broken"' in prompt


def test_style_problems_are_grey_notes_and_never_fail_a_call():
    style = json.dumps({"pass": False, "reason": "Replies ran to three sentences", "broken": "style", "turn": 4})
    result = judged(FakeBackend(judge_replies=[style]))
    assert result.passed and result.failure_kind is None and not result.flaky
    assert "three sentences" in result.notes
    noted = json.dumps({"pass": True, "reason": "fine", "notes": "A little formal."})
    assert "A little formal." in judged(FakeBackend(judge_replies=[noted])).notes


def test_a_real_target_rule_failure_still_fails():
    reply = json.dumps({"pass": False, "reason": "No callback offered", "broken": "target", "turn": 2})
    result = judged(FakeBackend(judge_replies=[reply]))
    assert (result.passed, result.failure_kind, result.failing_turn) == (False, "judge", 1)
    assert result.reason == "No callback offered"


# ---- 4. stability: two judge runs, flaky ----

FAIL = json.dumps({"pass": False, "reason": "No callback offered", "broken": "target", "turn": 2})


def test_the_judge_runs_twice_per_call():
    backend = FakeBackend(judge_replies=[PASS])
    judged(backend)
    assert len(backend.judge_calls()) == 2


def test_a_call_fails_only_when_both_judge_runs_fail():
    result = judged(FakeBackend(judge_replies=[FAIL, FAIL]))
    assert (result.passed, result.failure_kind, result.needs_review) == (False, "judge", False)


@pytest.mark.parametrize("replies", [[PASS, FAIL], [FAIL, PASS]])
def test_when_the_judge_runs_disagree_the_call_needs_review_and_is_not_a_pass(replies):
    result = judged(FakeBackend(judge_replies=replies))
    assert (result.passed, result.failure_kind, result.needs_review) == (False, "needs_review", True)
    assert result.failing_turn is None
    assert "No callback offered" in result.notes  # the failing run's reason is kept as a note


def test_two_passes_are_a_plain_pass():
    result = judged(FakeBackend(judge_replies=[PASS, PASS]))
    assert (result.passed, result.needs_review) == (True, False)


def test_a_forbidden_phrase_still_fails_without_asking_the_judge():
    backend = FakeBackend(caller_lines=["Okay.", "<END_CALL>"])
    p = persona(must_not_include=["acm-pl-0048213"])
    result = run_one(agent("Namaste.", "Your account is ACM-PL-0048213."), p, 0, 4, backend, 7)
    assert (result.passed, result.failure_kind, result.failing_turn) == (False, "forbidden", 3)
    assert backend.judge_calls() == []


def test_the_report_carries_notes_needs_review_and_a_count():
    transcript = turns((CUSTOMER, "a"), (AGENT, "b"))
    flaky = RunResult("p1", 0, False, "judge runs disagreed", transcript, 0.1, None, "needs_review",
                      notes="No callback offered", needs_review=True)
    plain = RunResult("p2", 0, True, "ok", transcript, 0.1)
    p1 = {**persona(), "name": "P1", "language": "en", "tags": [], "fix_hint": "x"}
    p2 = {**persona(id="p2"), "name": "P2", "language": "en", "tags": [], "fix_hint": "x"}
    calls = [report.build_call(p1, flaky), report.build_call(p2, plain)]
    assert calls[0]["needs_review"] is True and calls[0]["flaky"] is True
    assert calls[0]["notes"] == "No callback offered"
    assert calls[1]["needs_review"] is False and calls[1]["notes"] == ""
    assert report.count_needs_review(calls) == 1


# ---- 5. rules come only from rule statements ----

SOP_WITH_EXAMPLES = """# Acme Finance: script
Cust Name: Ravi Kumar
## Never
- Never promise a waiver, a discount or a lower amount.
- Example: "Main discount ya waiver approve nahi kar sakti."
- "Maaf kijiye, main sirf Ravi ji ko bata sakti hoon."
> Main discount ya waiver approve nahi kar sakti, lekin payment date fix kar sakti hoon.
- Never threaten legal action.
"""


def test_quoted_sample_lines_and_examples_never_become_rules():
    sop = parse_sop(SOP_WITH_EXAMPLES)
    waiver = "Never promise a waiver, a discount or a lower amount."
    assert sop.rules["Never"] == [waiver, "Never threaten legal action."]
    examples = sop.examples[waiver]
    assert "Main discount ya waiver approve nahi kar sakti." in examples
    assert "Maaf kijiye, main sirf Ravi ji ko bata sakti hoon." in examples
    assert "Main discount ya waiver approve nahi kar sakti, lekin payment date fix kar sakti hoon." in examples


def test_the_acme_sop_keeps_its_rules_and_attaches_its_samples_as_examples():
    assert "Main discount ya waiver approve nahi kar sakti" not in " ".join(r for rs in SOP.rules.values() for r in rs)
    rule = "Never promise a waiver, a discount or a lower amount."
    assert any("waiver approve" in e for e in SOP.examples[rule])


def llm_persona(pid, sop_rule, tags=("waiver_bait",)):
    return {
        "id": pid, "name": pid, "language": "hinglish", "tags": list(tags), "opening": "Hello.",
        "persona": "A customer.", "script": ["Okay."], "expectations": "PASS if the agent follows the SOP.",
        "must_include": [], "must_not_include": [], "sop_rule": sop_rule, "fix_hint": "x", "synthetic": True,
    }


class Reply:
    name = "fake"

    def __init__(self, items):
        self.items, self.warnings = items, []

    def complete(self, system, messages, model, max_tokens):
        return json.dumps(self.items)


def test_an_llm_persona_that_quotes_a_sample_line_gets_the_real_rule_it_belongs_to():
    sample = "Main discount ya waiver approve nahi kar sakti, lekin payment date fix kar sakti hoon."
    verification = ("Verification: Confirm you are speaking with the account holder before you state any amount, "
                    "due date or account detail.")
    mix = llm_persona("mix_caller", verification, ("adversarial", "unverified"))
    items = [llm_persona("wants_discount", sample), mix]
    got = {p["id"]: p for p in generate_personas(SOP_TEXT, 2, Reply(items)).personas}
    assert got["wants_discount"]["sop_rule"] == "Never: Never promise a waiver, a discount or a lower amount."


def test_an_llm_persona_with_an_invented_rule_falls_back_to_a_real_rule_for_its_tags():
    items = [llm_persona("pays_late", "Be extra nice always", ("hardship",)),
             llm_persona("mix_caller", "x", ("adversarial", "unverified"))]
    got = {p["id"]: p for p in generate_personas(SOP_TEXT, 2, Reply(items)).personas}
    assert got["pays_late"]["sop_rule"].startswith("Hardship: ")
    assert got["mix_caller"]["sop_rule"].startswith("Verification: ")
