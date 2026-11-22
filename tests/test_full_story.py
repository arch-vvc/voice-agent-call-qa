"""Every persona of the full set, for every offline agent, against tests/fixtures/full_set_expected.yaml. No network."""
from pathlib import Path

import pytest
import yaml

from callqa.persona_sets import personas_for_set
from evals import example_agent, example_agent_fixed, example_agent_v3
from evals.backends import OfflineBackend
from evals.run_evals import run_suite

HERE = Path(__file__).resolve().parent
SOP = (HERE.parent / "samples" / "sop_collections_acme.md").read_text(encoding="utf-8")
ROWS = yaml.safe_load((HERE / "fixtures" / "full_set_expected.yaml").read_text(encoding="utf-8"))["rows"]
AGENTS = {"v1": example_agent.respond, "v2": example_agent_fixed.respond, "v3": example_agent_v3.respond}


def outcomes(agent_id: str, repeats: int = 1, seed: int = 0) -> list:
    personas = personas_for_set(SOP, "full").personas
    return run_suite(AGENTS[agent_id], personas, repeats, 8, 4, OfflineBackend(), seed)


def test_the_expected_rows_are_the_full_set_in_order():
    assert list(ROWS) == [p["id"] for p in personas_for_set(SOP, "full").personas]


@pytest.mark.parametrize("agent_id", AGENTS)
def test_every_row_matches_for_every_agent(agent_id):
    results = {r.persona: r for r in outcomes(agent_id)}
    got = {pid: "pass" if r.passed else r.failure_kind for pid, r in results.items()}
    assert got == {pid: row[agent_id] for pid, row in ROWS.items()}


@pytest.mark.parametrize("agent_id", AGENTS)
def test_totals_follow_from_the_rows(agent_id):
    results = outcomes(agent_id)
    assert sum(r.passed for r in results) == sum(row[agent_id] == "pass" for row in ROWS.values())
    assert len(results) == len(ROWS)
