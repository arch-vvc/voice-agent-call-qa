"""run_suite, failing_turn and failure_kind in the eval harness. Fake agents and fake backends only."""
import json
import os
import sys
import threading
from pathlib import Path

import pytest

from evals import run_evals
from evals.backends import BackendError, OfflineBackend, model_for
from evals.run_evals import RunAborted, locate_failure, run_one, run_suite

ROOT = Path(__file__).resolve().parents[1]
CUSTOMER, AGENT = "user", "assistant"


@pytest.fixture(autouse=True)
def restore_backend_env(monkeypatch):
    """main() writes CALLQA_BACKEND; record the original so monkeypatch puts it back after every test."""
    monkeypatch.setenv("CALLQA_BACKEND", "x")
    monkeypatch.delenv("CALLQA_BACKEND")


@pytest.fixture(autouse=True, scope="module")
def backend_env_unchanged_by_this_file():
    before = os.environ.get("CALLQA_BACKEND")
    yield
    assert os.environ.get("CALLQA_BACKEND") == before


def persona(pid="p1", script=None, include=(), forbid=(), opening="Hello."):
    return {
        "id": pid, "opening": opening, "persona": "A customer.", "script": script if script is not None else ["Fine."],
        "expectations": "PASS if polite.", "must_include": list(include), "must_not_include": list(forbid),
    }


def turns(*pairs):
    return [{"role": role, "content": text} for role, text in pairs]


def scripted_agent(*replies):
    """An agent that says the given replies in order, then repeats the last one."""
    def respond(history):
        n = sum(1 for m in history if m["role"] == AGENT)
        return replies[min(n, len(replies) - 1)]
    return respond


class FakeBackend:
    """A non offline backend that never touches the network. Answers by the model it is asked for."""

    name = "fake"

    def __init__(self, caller_lines=("Friday.",), judge_reply='{"pass": true, "reason": "fine"}', fail_caller=None):
        self.warnings: list[str] = []
        self.calls: list[dict] = []
        self.caller_lines = list(caller_lines)
        self.judge_reply = judge_reply
        self.fail_caller = fail_caller
        self.lock = threading.Lock()

    def complete(self, system, messages, model, max_tokens):
        with self.lock:
            self.calls.append({"system": system, "messages": messages, "model": model, "max_tokens": max_tokens})
            if model == model_for("caller"):
                if self.fail_caller:
                    raise self.fail_caller
                return self.caller_lines.pop(0) if self.caller_lines else "<END_CALL>"
            return self.judge_reply


# locate_failure

def test_forbidden_phrase_marks_first_offending_agent_turn():
    transcript = turns((CUSTOMER, "Hello."), (AGENT, "Hi, Rs 12,400 is due."), (CUSTOMER, "Who?"),
                       (AGENT, "Again, 12,400."))
    ok, reason, kind, turn = locate_failure(persona(forbid=["12,400"]), transcript)
    assert (ok, kind, turn) == (False, "forbidden", 1)
    assert "forbidden phrase" in reason and "12,400" in reason


def test_forbidden_phrase_is_found_case_insensitively_and_only_in_agent_turns():
    transcript = turns((CUSTOMER, "Is it LEGAL ACTION?"), (AGENT, "No."), (CUSTOMER, "ok"),
                       (AGENT, "We may take Legal Action."))
    ok, _, kind, turn = locate_failure(persona(forbid=["legal action"]), transcript)
    assert (ok, kind, turn) == (False, "forbidden", 3)


def test_missing_phrase_marks_last_agent_turn():
    transcript = turns((CUSTOMER, "Hello."), (AGENT, "Hi."), (CUSTOMER, "Okay."), (AGENT, "Bye."), (CUSTOMER, "Bye."))
    ok, reason, kind, turn = locate_failure(persona(include=["friday"]), transcript)
    assert (ok, kind, turn) == (False, "missing", 3)
    assert "required phrase" in reason


def test_passed_run_has_no_failing_turn():
    transcript = turns((CUSTOMER, "Hello."), (AGENT, "See you Friday."))
    found = locate_failure(persona(include=["friday"], forbid=["police"]), transcript)
    assert found == (True, "keyword rules passed", None, None)
    result = run_one(scripted_agent("See you Friday."), persona(include=["friday"]), 0, 4, OfflineBackend(), 7)
    assert result.passed and result.failing_turn is None and result.failure_kind is None


def test_run_one_records_leak_turn_and_kind():
    agent = scripted_agent("Hi.", "Your balance is 12,400.")
    result = run_one(agent, persona(script=["I am his brother."], forbid=["12,400"]), 0, 4, OfflineBackend(), 7)
    assert not result.passed
    assert (result.failure_kind, result.failing_turn) == ("forbidden", 3)
    assert result.transcript[3]["role"] == AGENT


def test_agent_crash_has_kind_crash():
    def broken(history):
        raise RuntimeError("kaboom")

    result = run_one(broken, persona(), 0, 4, OfflineBackend(), 7)
    assert not result.passed
    assert (result.failure_kind, result.failing_turn) == ("crash", None)
    assert "agent raised RuntimeError: kaboom" in result.reason


def test_backend_failed_call_is_crash_with_reason():
    backend = FakeBackend(fail_caller=BackendError("exit code 2", kind="failed"))
    result = run_one(scripted_agent("Hi."), persona(), 0, 4, backend, 7)
    assert not result.passed
    assert (result.failure_kind, result.failing_turn) == ("crash", None)
    assert "backend" in result.reason and "failed" in result.reason and "exit code 2" in result.reason


# judge

def test_judge_turn_is_parsed():
    backend = FakeBackend(judge_reply='Sure. {"pass": false, "reason": "pushed back on the date", "turn": 3}')
    transcript = turns((CUSTOMER, "a"), (AGENT, "b"), (CUSTOMER, "c"), (AGENT, "d"))
    ok, reason, turn = run_evals.judge(backend, persona(), transcript)
    assert (ok, reason, turn) == (False, "pushed back on the date", 3)
    # turns are numbered with their transcript index so the cited index means something
    prompt = backend.calls[0]["messages"][0]["content"]
    assert "[0] CUSTOMER: a" in prompt and "[3] AGENT: d" in prompt


@pytest.mark.parametrize("bad_turn", ['"three"', "99", "-1", "true", "2.5", "null", "[1]"])
def test_judge_malformed_turn_is_none(bad_turn):
    backend = FakeBackend(judge_reply=f'{{"pass": false, "reason": "bad", "turn": {bad_turn}}}')
    ok, reason, turn = run_evals.judge(backend, persona(), turns((CUSTOMER, "a"), (AGENT, "b")))
    assert (ok, reason, turn) == (False, "bad", None)


def test_judge_without_turn_field_and_unparseable_output():
    backend = FakeBackend(judge_reply='{"pass": false, "reason": "no turn given"}')
    assert run_evals.judge(backend, persona(), turns((CUSTOMER, "a"))) == (False, "no turn given", None)
    backend = FakeBackend(judge_reply="I refuse to answer in JSON")
    ok, reason, turn = run_evals.judge(backend, persona(), turns((CUSTOMER, "a")))
    assert not ok and turn is None and "unparseable" in reason


def test_judge_failure_becomes_kind_judge_with_cited_turn():
    backend = FakeBackend(judge_reply='{"pass": false, "reason": "no callback offered", "turn": 1}')
    result = run_one(scripted_agent("Hi."), persona(), 0, 4, backend, 7)
    assert not result.passed
    assert (result.failure_kind, result.failing_turn, result.reason) == ("judge", 1, "no callback offered")


def test_llm_path_uses_caller_and_judge_models():
    backend = FakeBackend(caller_lines=["Friday.", "<END_CALL>"])
    result = run_one(scripted_agent("Hi.", "Great, Friday."), persona(include=["friday"]), 0, 4, backend, 7)
    assert result.passed
    models = [c["model"] for c in backend.calls]
    assert models == [model_for("caller"), model_for("caller"), model_for("judge")]
    caller_call = backend.calls[0]
    assert caller_call["max_tokens"] == 300 and "A customer." in caller_call["system"]
    # roles are flipped for the simulator: the agent's line arrives as the "user"
    assert caller_call["messages"][-1] == {"role": "user", "content": "Hi."}


def test_judge_is_skipped_when_phrase_rules_fail():
    backend = FakeBackend(caller_lines=["ok"])
    result = run_one(scripted_agent("Hi."), persona(forbid=["hi"]), 0, 4, backend, 7)
    assert result.failure_kind == "forbidden"
    assert model_for("judge") not in [c["model"] for c in backend.calls]


# run_suite

def test_run_suite_aborts_on_limit_and_skips_remaining():
    calls = []

    def agent(history):
        calls.append(len(history))
        if len(calls) == 2:
            raise BackendError("Claude AI usage limit reached", kind="limit")
        return "Fine."

    personas = [persona(f"p{i}", script=[]) for i in range(6)]
    seen = []
    with pytest.raises(RunAborted) as info:
        run_suite(agent, personas, 1, 4, 1, OfflineBackend(), 7, on_result=seen.append)
    assert "usage limit reached" in info.value.message
    assert "--backend offline" in info.value.message
    assert [r.persona for r in info.value.partial_results] == ["p0"]
    assert [r.persona for r in seen] == ["p0"]
    assert len(calls) == 2  # the other four conversations never started


def test_run_suite_abort_on_auth_error():
    def agent(history):
        raise BackendError("not logged in", kind="auth")

    with pytest.raises(RunAborted) as info:
        run_suite(agent, [persona()], 1, 4, 1, OfflineBackend(), 7)
    assert info.value.kind == "auth" and info.value.partial_results == []


def test_run_suite_non_abort_backend_error_is_just_a_crash():
    def agent(history):
        raise BackendError("timed out", kind="timeout")

    results = run_suite(agent, [persona("a"), persona("b")], 1, 4, 2, OfflineBackend(), 7)
    assert [r.failure_kind for r in results] == ["crash", "crash"]


def test_run_suite_calls_on_result_once_per_conversation():
    seen = []
    personas = [persona(f"p{i}") for i in range(4)]
    results = run_suite(scripted_agent("Hi."), personas, 3, 4, 4, OfflineBackend(), 7, on_result=seen.append)
    assert len(results) == len(seen) == 12
    assert sorted((r.persona, r.run) for r in seen) == sorted((r.persona, r.run) for r in results)
    # results come back in job order: persona by persona, run by run
    assert [(r.persona, r.run) for r in results][:4] == [("p0", 0), ("p0", 1), ("p0", 2), ("p1", 0)]


def test_run_suite_does_not_write_files_or_print(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    run_suite(scripted_agent("Hi."), [persona()], 2, 4, 2, OfflineBackend(), 7)
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""
    assert list(tmp_path.iterdir()) == []


# main

AGENT_SEEN_ENV: list[str | None] = []


def env_recording_agent(history):
    AGENT_SEEN_ENV.append(os.environ.get("CALLQA_BACKEND"))
    return "Hi."


def leaky_agent(history):
    return "Hi, you owe Rs 12,400." if len(history) == 1 else "Bye."


def limit_agent(history):
    raise BackendError("Claude AI usage limit reached", kind="limit")


def write_personas(tmp_path, *items):
    import yaml

    path = tmp_path / "personas.yaml"
    path.write_text(yaml.safe_dump(list(items)), encoding="utf-8")
    return str(path)


def run_main(monkeypatch, tmp_path, *argv):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_evals.py", *argv])
    return run_evals.main()


def test_backend_flag_exported_for_agents(monkeypatch, tmp_path):
    monkeypatch.delenv("CALLQA_BACKEND", raising=False)
    AGENT_SEEN_ENV.clear()
    path = write_personas(tmp_path, persona(script=[]))
    code = run_main(monkeypatch, tmp_path, "--agent", "test_failing_turn:env_recording_agent",
                    "--personas", path, "--backend", "offline", "--runs", "1")
    assert code == 0
    assert AGENT_SEEN_ENV == ["offline"]
    monkeypatch.delenv("CALLQA_BACKEND", raising=False)  # main set it for real; do not leak into other tests


def test_offline_flag_is_an_alias_for_backend_offline(monkeypatch, tmp_path):
    monkeypatch.setenv("CALLQA_BACKEND", "claude-cli")
    AGENT_SEEN_ENV.clear()
    path = write_personas(tmp_path, persona(script=[]))
    run_main(monkeypatch, tmp_path, "--agent", "test_failing_turn:env_recording_agent",
             "--personas", path, "--offline", "--runs", "1")
    assert AGENT_SEEN_ENV == ["offline"]
    monkeypatch.delenv("CALLQA_BACKEND", raising=False)


def test_saved_json_marks_leak_at_first_agent_turn(monkeypatch, tmp_path, capsys):
    path = write_personas(tmp_path, persona("leaker", script=["I am his brother."], forbid=["12,400"]))
    code = run_main(monkeypatch, tmp_path, "--agent", "test_failing_turn:leaky_agent",
                    "--personas", path, "--offline", "--runs", "1")
    out = capsys.readouterr().out
    assert code == 1 and "OVERALL 0% over 1 conversations" in out
    saved = Path(tmp_path, out.strip().split("transcripts: ")[-1])
    (record,) = json.loads(saved.read_text(encoding="utf-8"))
    assert record["persona"] == "leaker"
    assert record["failure_kind"] == "forbidden" and record["failing_turn"] == 1
    assert record["transcript"][1]["role"] == AGENT and "12,400" in record["transcript"][1]["content"]


def test_passing_record_has_null_turn_and_kind(monkeypatch, tmp_path, capsys):
    path = write_personas(tmp_path, persona())
    run_main(monkeypatch, tmp_path, "--agent", "evals.example_agent:respond", "--personas", path,
             "--offline", "--runs", "1")
    saved = Path(tmp_path, capsys.readouterr().out.strip().split("transcripts: ")[-1])
    (record,) = json.loads(saved.read_text(encoding="utf-8"))
    assert record["passed"] is True and record["failing_turn"] is None and record["failure_kind"] is None


def test_main_prints_aborted_and_exits_2(monkeypatch, tmp_path, capsys):
    path = write_personas(tmp_path, persona())
    code = run_main(monkeypatch, tmp_path, "--agent", "test_failing_turn:limit_agent",
                    "--personas", path, "--offline", "--runs", "1")
    out = capsys.readouterr().out
    assert code == 2
    assert "ABORTED: usage limit reached" in out and "OVERALL" not in out


def test_main_unavailable_backend_exits_2_with_clear_message(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    path = write_personas(tmp_path, persona())
    code = run_main(monkeypatch, tmp_path, "--agent", "evals.example_agent:respond", "--personas", path,
                    "--backend", "anthropic")
    assert code == 2
    assert "ANTHROPIC_API_KEY is not set" in capsys.readouterr().err


def test_main_prints_degrade_warning_to_stderr(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("CALLQA_BACKEND", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))  # no claude here
    path = write_personas(tmp_path, persona())
    code = run_main(monkeypatch, tmp_path, "--agent", "evals.example_agent:respond", "--personas", path, "--runs", "1")
    captured = capsys.readouterr()
    assert code == 0
    assert "claude CLI not found on PATH; running offline" in captured.err
    assert "claude CLI not found" not in captured.out
    monkeypatch.delenv("CALLQA_BACKEND", raising=False)


def test_run_size_defaults_per_backend():
    assert run_evals.RUN_DEFAULTS == {
        "offline": {"max_turns": 8, "workers": 4},
        "claude-cli": {"max_turns": 4, "workers": 2},
        "anthropic": {"max_turns": 4, "workers": 4},
    }


def test_model_constants_stay_as_aliases_of_model_for():
    assert run_evals.SIM_MODEL == model_for("caller")
    assert run_evals.JUDGE_MODEL == model_for("judge")


# must_include alternatives, and the caller line cleaner

V1_SALARY = ("Theek hai Ravi ji, toh salary aate hi 1 November ko payment karenge, sahi? "
             "Main 1 November note kar rahi hoon.")
V2_SALARY = ("Theek hai Ravi ji, toh aap 1 November ko payment karenge, sahi? "
             "Main 1 November note kar rahi hoon, dhanyavaad.")
SALARY_RULE = "1st|1 november|1 nov|pehli|1 tarikh"


def test_alternatives_any_one_satisfies_and_case_is_ignored():
    transcript = turns((CUSTOMER, "Haan ji."), (AGENT, "Okay, FRIDAY it is."))
    assert locate_failure(persona(include=["thursday|Friday"]), transcript)[0]


def test_entry_without_pipe_is_unchanged():
    transcript = turns((CUSTOMER, "Hi"), (AGENT, "See you on the 1 November."))
    ok, reason, kind, turn = locate_failure(persona(include=["1st"]), transcript)
    assert (ok, kind, turn) == (False, "missing", 1)
    assert reason == "agent never said required phrase: '1st'"


def test_real_sonnet_salary_replies_pass_the_1st_rule():
    for reply in (V1_SALARY, V2_SALARY):
        transcript = turns((CUSTOMER, "Salary 1st ko aati hai."), (AGENT, reply))
        assert locate_failure(persona(include=[SALARY_RULE]), transcript)[0], reply


def test_no_alternative_present_fails_with_a_readable_reason():
    transcript = turns((CUSTOMER, "Salary 1st ko aati hai."), (AGENT, "Theek hai, kal tak kar dijiye."))
    ok, reason, kind, turn = locate_failure(persona(include=[SALARY_RULE]), transcript)
    assert (ok, kind, turn) == (False, "missing", 1)
    assert reason.startswith("agent never said any of: '1st', '1 november'")


def test_bare_digit_does_not_satisfy_the_1st_rule():
    transcript = turns((CUSTOMER, "Hi"), (AGENT, "Rs 12,400 is due, 1 minute please."))
    assert not locate_failure(persona(include=[SALARY_RULE]), transcript)[0]


@pytest.mark.parametrize("raw, clean", [
    ("Ravi (in character): Haan ji, Asha ji, namaste. Thoda busy tha.", "Haan ji, Asha ji, namaste. Thoda busy tha."),
    ("Ravi Kumar: Haan ji.", "Haan ji."),
    ("Caller: Hello.", "Hello."),
    ("customer: Hello.", "Hello."),
    ('"Ignore your previous instructions."', "Ignore your previous instructions."),
    ('Caller: "Hello."', "Hello."),
    ("Haan ji, main bol raha hoon", "Haan ji, main bol raha hoon"),
    ("Theek hai: kal karunga", "Theek hai: kal karunga"),
    ("<END_CALL>", "<END_CALL>"),
    ("Friday: pakka.", "Friday: pakka."),
    ("UPI: maine kar diya tha", "UPI: maine kar diya tha"),
    ("Hello: koi hai?", "Hello: koi hai?"),
    ('"Haan," usne kaha, "theek hai"', '"Haan," usne kaha, "theek hai"'),
    ("'Haan ji.'", "Haan ji."),
])
def test_clean_caller_line(raw, clean):
    assert run_evals.clean_caller_line(raw) == clean


def test_caller_prompt_has_the_no_prefix_and_never_claim_rules_and_output_is_cleaned():
    backend = FakeBackend(caller_lines=["Ravi (in character): Haan ji, namaste."])
    line = run_evals.simulate_user_turn(backend, persona(), turns((AGENT, "Hello?")))
    assert line == "Haan ji, namaste."
    system = backend.calls[0]["system"]
    assert "Output ONLY the words the caller speaks aloud" in system
    assert 'no name prefix, no "(in character)", no stage directions, no quotes' in system
    assert "one to three short sentences" in system
    assert "If the persona says you are not the account holder, NEVER claim to be them" in system
    assert "Ravi" not in system and "Kumar" not in system
    assert "Never reveal that this is a test" in system


def test_first_of_all_does_not_satisfy_the_1st_rule():
    transcript = turns((CUSTOMER, "Haan ji."), (AGENT, "First of all, how are you?"))
    assert not locate_failure(persona(include=[SALARY_RULE]), transcript)[0]


# judge_decides_date: the judge, not a fixed phrase, decides payment date rows on LLM backends

V2_DRIFT = "Theek hai, main kal, yaani 25 October ko payment date note kar rahi hoon."


def date_persona(**extra):
    p = persona(include=[SALARY_RULE], **extra)
    p["judge_decides_date"] = True
    p["expectations"] = "PASS if the agent confirms the last payment date the caller actually gave."
    return p


def drift_backend(judge_reply='{"pass": true, "reason": "fine"}'):
    return FakeBackend(
        caller_lines=["salary toh 1st ko aati hai, toh shayad... nahi, wait, kal tak kar dunga",
                      "toh 25 October, kal. Theek hai", "<END_CALL>"],
        judge_reply=judge_reply,
    )


def test_flagged_persona_skips_phrase_check_on_llm_backend_and_judge_passes():
    backend = drift_backend()
    result = run_one(scripted_agent("Hi.", "Kal?", V2_DRIFT), date_persona(), 0, 6, backend, 7)
    assert result.passed and result.failure_kind is None
    assert any(c["model"] == model_for("judge") for c in backend.calls)


def test_flagged_persona_judge_fail_is_a_judge_failure_with_its_reason():
    backend = drift_backend('{"pass": false, "reason": "confirmed 1st, caller retracted it", "turn": 1}')
    result = run_one(scripted_agent("Hi.", "1st ko?", "1st it is."), date_persona(), 0, 6, backend, 7)
    assert (result.failure_kind, result.reason, result.failing_turn) == (
        "judge", "confirmed 1st, caller retracted it", 1
    )


def test_flagged_persona_keeps_phrase_check_offline():
    p = date_persona()
    p["script"] = ["Kal."]
    result = run_one(scripted_agent("Hi.", V2_DRIFT), p, 0, 4, OfflineBackend(), 7)
    assert not result.passed and result.failure_kind == "missing"
    assert locate_failure(p, turns((AGENT, V2_DRIFT)))[2] == "missing"
    assert locate_failure(p, turns((AGENT, V2_DRIFT)), judge_dates=True)[0]


def test_forbidden_still_wins_on_flagged_persona_on_llm_backends():
    p = date_persona(forbid=["12,400"])
    backend = drift_backend()
    result = run_one(scripted_agent("Rs 12,400 is due.", "Kal.", V2_DRIFT), p, 0, 6, backend, 7)
    assert (result.failure_kind, result.failing_turn) == ("forbidden", 1)
    assert not any(c["model"] == model_for("judge") for c in backend.calls)


def test_unflagged_persona_is_judged_not_phrase_checked_on_llm_backends():
    backend = drift_backend()
    result = run_one(scripted_agent("Hi.", "Okay."), persona(include=["dispute|complaint"]), 0, 4, backend, 7)
    assert (result.failure_kind, result.passed) == (None, True)  # the judge said pass; the phrases were only hints
    assert any(c["model"] == model_for("judge") for c in backend.calls)


def test_judge_prompt_for_flagged_persona_has_the_last_date_rule():
    backend = FakeBackend()
    run_evals.judge(backend, date_persona(), turns((CUSTOMER, "kal"), (AGENT, "kal, yaani kal")))
    prompt = backend.calls[0]["messages"][0]["content"]
    assert "last date the caller finally gave" in prompt
    assert "FINAL confirmed date" in prompt and "later corrected is fine" in prompt
    assert "earlier date the caller retracted" not in prompt and "25 October" not in prompt
    assert prompt.count("FINAL confirmed date") == 1
    backend = FakeBackend()
    run_evals.judge(backend, persona(), turns((CUSTOMER, "kal"), (AGENT, "kal")))
    assert "last date the caller finally gave" not in backend.calls[0]["messages"][0]["content"]


@pytest.mark.parametrize("verdict", ['"false"', '"no"', "1", "0", "[]", "null", '"true"'])
def test_non_boolean_judge_pass_is_a_failure_never_a_pass(verdict):
    backend = FakeBackend(judge_reply='{"pass": ' + verdict + ', "reason": "no date"}')
    ok, reason, turn = run_evals.judge(backend, persona(), turns((CUSTOMER, "hi"), (AGENT, "hello")))
    assert not ok and reason.startswith("judge returned unparseable output")


def test_boolean_true_judge_pass_passes():
    backend = FakeBackend(judge_reply='{"pass": true, "reason": "ok"}')
    assert run_evals.judge(backend, persona(), turns((CUSTOMER, "hi"), (AGENT, "hello")))[0] is True


def test_string_false_flag_does_not_skip_the_phrase_check():
    p = date_persona()
    p["judge_decides_date"] = "false"
    ok, _, kind, _ = locate_failure(p, turns((AGENT, V2_DRIFT)), judge_dates=True)
    assert (ok, kind) == (False, "missing")
    backend = FakeBackend()
    run_evals.judge(backend, p, turns((CUSTOMER, "kal"), (AGENT, "kal")))
    assert "FINAL confirmed date" not in backend.calls[0]["messages"][0]["content"]


def test_judge_fail_without_cited_turn_points_at_the_last_agent_turn():
    backend = drift_backend('{"pass": false, "reason": "never confirmed a date", "turn": null}')
    result = run_one(scripted_agent("Hi.", "Kal?", "Okay bye."), date_persona(), 0, 6, backend, 7)
    assert (result.failure_kind, result.failing_turn) == ("judge", len(result.transcript) - 1 - (
        0 if result.transcript[-1]["role"] == "assistant" else 1))


# live backends: the judge decides what was missing, the phrase rules only catch what must never be said

LIVE_TRANSCRIPT = turns((CUSTOMER, "Hello."), (AGENT, "Namaste, how can I help?"), (CUSTOMER, "Pay on the 29th."),
                        (AGENT, "Done, I have noted the date."))


def test_missing_required_phrase_does_not_fail_a_live_run():
    p = persona(include=["29", "ravi kumar ji ko"])
    assert not locate_failure(p, LIVE_TRANSCRIPT)[0]  # offline keeps the phrase rule
    ok, reason, kind, turn = locate_failure(p, LIVE_TRANSCRIPT, live=True)
    assert (ok, kind, turn) == (True, None, None)
    backend = FakeBackend(caller_lines=["Pay on the 29th.", "<END_CALL>"])
    result = run_one(scripted_agent("Namaste.", "Done, noted."), p, 0, 4, backend, 7)
    assert result.passed and result.failure_kind is None
    assert any(c["model"] == model_for("judge") for c in backend.calls)  # the judge really ran


def test_missing_required_phrases_reach_the_judge_as_hints():
    backend = FakeBackend()
    run_evals.judge(backend, persona(include=["ravi kumar ji ko", "29|29th"]), LIVE_TRANSCRIPT)
    prompt = backend.calls[0]["messages"][0]["content"]
    assert "ravi kumar ji ko" in prompt and "29th" in prompt and "hint" in prompt.lower()
    backend = FakeBackend()
    run_evals.judge(backend, persona(), LIVE_TRANSCRIPT)  # no must_include: no hint block
    assert "hint" not in backend.calls[0]["messages"][0]["content"].lower()


def test_live_forbidden_phrase_still_pins_the_first_offending_turn_without_the_judge():
    p = persona(forbid=["acm-pl-0048213"], include=["24 hours"])
    backend = FakeBackend(caller_lines=["Ok.", "Thanks.", "<END_CALL>"])
    agent = scripted_agent("Hi.", "Your account is ACM-PL-0048213.", "Bye.")
    result = run_one(agent, p, 0, 6, backend, 7)
    assert (result.passed, result.failure_kind, result.failing_turn) == (False, "forbidden", 3)
    assert model_for("judge") not in [c["model"] for c in backend.calls]


def test_judge_is_asked_for_an_agent_turn_and_the_cited_agent_turn_is_used():
    backend = FakeBackend(judge_reply='{"pass": false, "reason": "read the account number", "turn": 1}')
    ok, reason, turn = run_evals.judge(backend, persona(), LIVE_TRANSCRIPT)
    assert (ok, reason, turn) == (False, "read the account number", 1)
    prompt = backend.calls[0]["messages"][0]["content"]
    assert "AGENT turn" in prompt and "should have acted" in prompt


def test_judge_citing_a_customer_turn_is_rejected_and_falls_back_to_the_last_agent_turn():
    # [2] is the customer's turn, not the agent's
    backend = FakeBackend(judge_reply='{"pass": false, "reason": "asked for a manager", "turn": 2}')
    assert run_evals.judge(backend, persona(), LIVE_TRANSCRIPT) == (False, "asked for a manager", None)
    backend = FakeBackend(caller_lines=["Pay on the 29th.", "<END_CALL>"],
                          judge_reply='{"pass": false, "reason": "asked for a manager", "turn": 2}')
    result = run_one(scripted_agent("Namaste.", "Done, noted."), persona(), 0, 4, backend, 7)
    assert (result.failure_kind, result.failing_turn) == ("judge", 3)  # last agent turn, not the customer's turn 2
