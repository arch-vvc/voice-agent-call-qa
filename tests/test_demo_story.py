"""The 3 minute story, offline: the same generated callers go through v1 and v2 and the numbers hold."""
from collections import Counter
from pathlib import Path

from callqa.personas import STRATIFIED_5, generate_personas
from evals import example_agent, example_agent_fixed
from evals.backends import OfflineBackend
from evals.run_evals import run_suite

SOP = (Path(__file__).resolve().parents[1] / "samples" / "sop_collections_acme.md").read_text(encoding="utf-8")
personas_by_id = {p["id"]: p for p in generate_personas(SOP, 20, None).personas}
UNVERIFIED = ["brother_asks", "wife_handles_money", "wrong_number", "prompt_injection", "fake_head_office"]


def run(agent, n):
    personas = generate_personas(SOP, n, None).personas
    results = run_suite(agent, personas, 1, 8, 4, OfflineBackend(), 0)
    return {r.persona: r for r in results}


def passed(results):
    return sum(r.passed for r in results.values())


def failures(results):
    return {pid: r for pid, r in results.items() if not r.passed}


def test_v1_fails_unverified_leaks_at_turn_1_and_escalation_at_turn_3():
    results = run(example_agent.respond, 20)
    assert len(results) == 20 and passed(results) == 14
    assert set(failures(results)) == {*UNVERIFIED, "wants_manager"}
    for pid in UNVERIFIED:
        assert (results[pid].failure_kind, results[pid].failing_turn) == ("forbidden", 1), pid
        assert "12,400" in results[pid].transcript[1]["content"]
    rules = [personas_by_id[pid]["sop_rule"] for pid in UNVERIFIED]
    assert len(set(rules)) == 1 and "Verification" in rules[0]
    groups = Counter(personas_by_id[r.persona]["sop_rule"] for r in failures(results).values())
    assert [count for _, count in groups.most_common()] == [5, 1]
    assert groups.most_common()[1][0].startswith("Escalation")
    manager = results["wants_manager"]
    assert (manager.failure_kind, manager.failing_turn) == ("missing", 3)


def test_v2_fails_only_the_escalation_caller():
    results = run(example_agent_fixed.respond, 20)
    assert passed(results) == 19
    assert set(failures(results)) == {"wants_manager"}
    assert results["wants_manager"].failure_kind == "missing"


def test_stratified_five_offline_v1_20_v2_80_meets_structural_criteria():
    v1, v2 = run(example_agent.respond, 5), run(example_agent_fixed.respond, 5)
    assert set(v1) == set(STRATIFIED_5)
    assert passed(v1) == 1 and passed(v2) == 4
    assert [r for r in v2.values() if r.failure_kind == "forbidden"] == []  # S1
    assert set(failures(v2)) <= {"wants_manager"}  # S2
    assert any(r.failure_kind == "forbidden" for pid, r in v1.items() if pid in UNVERIFIED)  # S3
