"""Repeats: script variants, pass@1, pass^k, flaky personas and the report grouping. Offline, no network."""
from pathlib import Path

import pytest

from callqa.consistency import first_failing, flaky, group_by_persona, pass_at_1, pass_hat_k
from callqa.persona_sets import personas_for_set
from callqa.report import build_report
from evals import example_agent, example_agent_fixed
from evals.backends import OfflineBackend
from evals.run_evals import RunResult, run_one, run_suite, variant_index

SOP = (Path(__file__).resolve().parents[1] / "samples" / "sop_collections_acme.md").read_text(encoding="utf-8")
AGENT = {"id": "v1", "label": "v1 naive prompt"}


def result(persona: str, run: int, passed: bool) -> RunResult:
    return RunResult(persona, run, passed, "ok" if passed else "bad", failure_kind=None if passed else "missing")


def full_persona(pid: str) -> dict:
    return next(p for p in personas_for_set(SOP, "full").personas if p["id"] == pid)


def flaky_runs(agent, seed: int) -> list[RunResult]:
    return run_suite(agent, [full_persona("flaky_line")], 3, 8, 1, OfflineBackend(), seed)


def report_for(set_name: str, agent, repeats: int, agent_entry: dict = AGENT) -> dict:
    generation = personas_for_set(SOP, set_name)
    results = run_suite(agent, generation.personas, repeats, 8, 4, OfflineBackend(), 0)
    return build_report(run_id="r", sop_text=SOP, agent=agent_entry, backend=OfflineBackend(), callers=len(results),
                        max_turns=8, generation=generation, results=results, seconds=0.1, repeats=repeats,
                        set_name=set_name)


def test_variant_index_is_seed_plus_repeat_mod_len():
    assert [variant_index(s, r, 3) for s, r in [(0, 0), (0, 2), (1, 2), (7, 0), (7, 1)]] == [0, 2, 0, 1, 2]
    persona = full_persona("flaky_line")
    for seed in (0, 1, 7):
        for repeat in range(3):
            got = run_one(example_agent.respond, persona, repeat, 8, OfflineBackend(), seed)
            assert (got.seed, got.variant) == (seed, (seed + repeat) % 3)
            replayed = [m["content"] for m in got.transcript if m["role"] == "user"][1]
            assert replayed == persona["script_variants"][got.variant][0]


def test_a_persona_without_variants_has_no_variant_and_replays_its_script():
    persona = personas_for_set(SOP, "story").personas[0]
    got = run_one(example_agent.respond, persona, 0, 8, OfflineBackend(), 0)
    assert got.variant is None and got.seed == 0 and got.passed


@pytest.mark.parametrize("agent", [example_agent.respond, example_agent_fixed.respond])
@pytest.mark.parametrize("seed", [0, 1, 7])
def test_flaky_line_fails_exactly_one_of_three_repeats_for_any_seed(agent, seed):
    runs = flaky_runs(agent, seed)
    assert sorted(r.run for r in runs) == [0, 1, 2]
    assert [r.passed for r in runs].count(False) == 1
    assert next(r for r in runs if not r.passed).failure_kind == "missing"


def test_same_seed_gives_identical_results():
    def key(runs):
        return [(r.persona, r.run, r.passed, r.reason, r.variant, r.transcript) for r in runs]
    assert key(flaky_runs(example_agent.respond, 7)) == key(flaky_runs(example_agent.respond, 7))


def test_pass_at_1_is_runs_passed_over_runs():
    groups = group_by_persona([result("a", 0, True), result("a", 1, False), result("b", 0, True), result("b", 1, True)])
    assert pass_at_1(groups) == 0.75


def test_pass_hat_k_counts_personas_passing_every_repeat():
    groups = group_by_persona([result("a", 0, True), result("a", 1, False), result("b", 0, True), result("b", 1, True),
                               result("c", 0, False), result("c", 1, False)])
    assert pass_hat_k(groups) == pytest.approx(1 / 3)
    assert flaky(groups) == [{"persona_id": "a", "passed": 1, "total": 2, "failed_repeat": 1}]


def test_results_are_grouped_in_repeat_order_whatever_order_they_finish():
    groups = group_by_persona([result("a", 2, True), result("a", 0, True), result("a", 1, False)])
    assert [r.run for r in groups["a"]] == [0, 1, 2]


def test_representative_result_is_first_failing_repeat():
    report = report_for("full", example_agent.respond, 3)
    call = next(c for c in report["calls"] if c["persona_id"] == "flaky_line")
    assert call["passed"] is False and call["failure_kind"] == "missing"
    assert (call["repeats_passed"], call["repeats_total"], call["flaky"], call["failed_repeat"]) == (2, 3, True, 2)
    assert call["transcript"] == call["repeat_results"][2]["transcript"]
    assert [r["passed"] for r in call["repeat_results"]] == [True, True, False]
    assert [r["variant"] for r in call["repeat_results"]] == [0, 1, 2]
    steady = next(c for c in report["calls"] if c["persona_id"] == "happy_friday")
    assert steady["passed"] and steady["flaky"] is False
    assert steady["transcript"] == steady["repeat_results"][0]["transcript"]


def test_three_repeat_report_has_the_consistency_block():
    report = report_for("full", example_agent_fixed.respond, 3, {"id": "v2", "label": "v2"})
    assert report["repeats"] == 3
    assert report["consistency"] == {
        "repeats": 3, "pass_at_1": round(68 / 93, 4), "runs_passed": 68, "runs_total": 93,
        "pass_hat_k": round(22 / 31, 4), "flaky": [{"persona_id": "flaky_line", "passed": 2, "total": 3,
                                                   "failed_repeat": 2}],
    }
    assert (report["passed"], report["total"]) == (22, 31)
    assert report["pass_rate"] == round(22 / 31, 4)
    assert report["persona_set"]["name"] == "full" and report["persona_set"]["count"] == 31


def test_one_repeat_report_equals_todays_numbers():
    v1 = report_for("story", example_agent.respond, 1)
    v2 = report_for("story", example_agent_fixed.respond, 1, {"id": "v2", "label": "v2"})
    assert (v1["passed"], v1["total"]) == (14, 20) and (v2["passed"], v2["total"]) == (19, 20)
    assert v1["consistency"] is None and v1["repeats"] == 1 and v1["persona_set"]["name"] == "story"
    assert all(c["repeat_results"] == [] and c["flaky"] is False for c in v1["calls"])


def synthetic_report(results: list, repeats: int, **kwargs) -> dict:
    generation = personas_for_set(SOP, "full")
    generation.personas = [full_persona("flaky_line")]
    return build_report(run_id="r", sop_text=SOP, agent=AGENT, backend=OfflineBackend(), callers=1, max_turns=8,
                        generation=generation, results=results, seconds=0.1, repeats=repeats, **kwargs)


def failing(run: int, reason: str, transcript: list) -> RunResult:
    return RunResult("flaky_line", run, False, reason, transcript, failure_kind="missing", failing_turn=1)


def test_with_two_failing_repeats_the_call_shows_the_first_and_agrees_with_flaky():
    a = [{"role": "user", "content": "A"}, {"role": "assistant", "content": "a"}]
    b = [{"role": "user", "content": "B"}, {"role": "assistant", "content": "b"}]
    runs = [result("flaky_line", 0, True), failing(1, "first", a), failing(2, "second", b)]
    call = synthetic_report(runs, 3)["calls"][0]
    assert (call["reason"], call["transcript"], call["failed_repeat"]) == ("first", a, 1)
    assert first_failing(runs) is runs[1]
    assert flaky(group_by_persona(runs)) == [{"persona_id": "flaky_line", "passed": 1, "total": 3, "failed_repeat": 1}]


def test_a_persona_failing_every_repeat_shows_repeat_0_and_has_no_failed_repeat():
    runs = [failing(0, "zero", []), failing(1, "one", []), failing(2, "two", [])]
    call = synthetic_report(runs, 3)["calls"][0]
    assert (call["reason"], call["failed_repeat"], call["flaky"]) == ("zero", None, False)
    assert flaky(group_by_persona(runs)) == []


def test_variants_are_never_chosen_on_a_live_backend():
    class Live(OfflineBackend):
        name = "claude-cli"

        def complete(self, system, messages, model, max_tokens):
            return "<END_CALL>"

    got = run_one(example_agent.respond, full_persona("flaky_line"), 0, 8, Live(), 0)
    assert got.variant is None


def test_report_carries_seed_repeats_and_set_name():
    report = synthetic_report([result("flaky_line", r, True) for r in range(3)], 3, seed=7, set_name="full")
    assert (report["seed"], report["repeats"], report["persona_set"]["name"]) == (7, 3, "full")


def test_a_persona_with_a_missing_repeat_does_not_pass_every_repeat():
    report = synthetic_report([result("flaky_line", r, True) for r in range(2)], 3)
    assert (report["passed"], report["calls"][0]["passed"]) == (0, False)
    assert "flaky_line: expected 3 repeats, got 2" in report["warnings"]
    assert report["consistency"]["pass_hat_k"] == 0.0 and report["consistency"]["runs_total"] == 2


def test_five_repeats_with_only_three_runs_is_a_warning_and_not_a_pass():
    report = synthetic_report([result("flaky_line", r, True) for r in range(3)], 5)
    assert report["passed"] == 0 and "flaky_line: expected 5 repeats, got 3" in report["warnings"]
