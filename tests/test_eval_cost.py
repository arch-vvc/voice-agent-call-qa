"""Metering wiring (A3b): the eval cost lines, the latency budget flag, the dashboard report blocks and the agent's
own tokens. Offline only; the one LLM test uses a fake backend, so no model is called."""
import json
import threading
import time
import urllib.request
from pathlib import Path

import pytest

from callqa import evaluate
from callqa import report as report_module
from callqa.runrecord import load_record
from callqa.server import DEFAULT_AGENTS, agent_prompt_for, make_server, read_sample_sop
from evals.backends import OfflineBackend

GOLDEN = Path(__file__).resolve().parent / "golden"


def run_eval(capsys, *args):
    code = evaluate.main(list(args))
    out = capsys.readouterr()
    return code, out.out.splitlines(), out.err


def run_story(capsys, agent, *extra):
    return run_eval(capsys, "--agent", agent, "--set", "story", "--backend", "offline", "--no-save", *extra)


def golden(agent):
    return (GOLDEN / f"eval_story_{agent}.txt").read_text(encoding="utf-8").splitlines()


def test_eval_story_v1_matches_golden(capsys):
    code, lines, _ = run_story(capsys, "v1")
    assert code == 0 and lines == golden("v1")


def test_eval_story_v2_matches_golden(capsys):
    code, lines, _ = run_story(capsys, "v2")
    assert code == 0 and lines == golden("v2")


def test_golden_lines_are_in_the_pinned_order_and_labelled_simulated():
    for agent in ("v1", "v2"):
        lines = golden(agent)
        assert [line.split(" ")[0] for line in lines] == ["EVAL", "pass^1", "flaky:", "tokens", "cost", "latency",
                                                         "escalation", "compliance", "traits"]
        assert lines[3].startswith("tokens (simulated) caller ")
        assert lines[4].startswith("cost (simulated) $") and lines[4].endswith(
            "at config/pricing.yaml prices of 2026-11-06, verify before quoting")
        assert lines[5].startswith("latency (simulated) P50 ") and "budget 1500 ms:" in lines[5]


def test_budget_flag_counts_turns_over(capsys, tmp_path):
    seen = {}
    for budget in (None, 400, 100000):
        extra = [] if budget is None else ["--latency-budget-ms", str(budget)]
        folder = tmp_path / str(budget)
        code, lines, _ = run_eval(capsys, "--agent", "v1", "--backend", "offline", "--out", str(folder), *extra)
        assert code == 0
        record = load_record(next(folder.glob("*.json")))
        turns = [ms for call in record["calls"] for ms in call["turn_latency_ms"]]
        want = budget or 1500
        count = sum(ms > want for ms in turns)
        assert record["latency"]["budget_ms"] == want and record["latency"]["over_budget"] == count
        line = next(line for line in lines if line.startswith("latency ("))
        assert line.endswith(f"budget {want} ms: {count} of {len(turns)} agent turns over")
        seen[budget] = count
    assert seen[100000] == 0 < seen[400]
    assert seen[400] > seen[None]


@pytest.mark.parametrize("bad", ["0", "-5", "soon"])
def test_bad_budget_exits_2(capsys, bad):
    code, _, _ = run_eval(capsys, "--agent", "v1", "--backend", "offline", "--no-save", "--latency-budget-ms", bad)
    assert code == 2


def test_dashboard_report_has_cost_and_latency_blocks(tmp_path):
    srv = make_server(OfflineBackend(), 5, 8, 2, DEFAULT_AGENTS, results_dir=tmp_path / "runs")
    threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        payload = json.dumps({"sop": read_sample_sop(), "agent": "v2"}).encode()
        request = urllib.request.Request(base + "/api/run", payload,
                                         {"Content-Type": "application/json", "Origin": base})
        run_id = json.load(urllib.request.urlopen(request))["run_id"]
        for _ in range(3000):
            body = json.load(urllib.request.urlopen(f"{base}/api/run/{run_id}"))
            if body["status"] != "running":
                break
            time.sleep(0.01)
        report = body["report"]
    finally:
        srv.shutdown()
        srv.server_close()
    assert report["cost"]["label"] == "simulated" and report["cost"]["total"] > 0
    prompt_tokens = len(agent_prompt_for({"id": "v2"}, read_sample_sop())) // 4
    assert prompt_tokens > 100  # the v2 prompt goes in with every agent turn
    assert report["cost"]["by_role"]["agent"]["input_tokens"] >= report["latency"]["turns"] * prompt_tokens
    assert report["latency"]["label"] == "simulated" and report["latency"]["turns"] > 0
    assert report["latency"]["budget_ms"] == 1500
    assert all(call["usage"] and call["turn_latency_ms"] for call in report["calls"])


class FakeLive:
    """A claude-cli shaped fake: answers the judge, ends the caller's turn, and replies as the agent otherwise."""

    name = "claude-cli"
    warnings: list = []

    def complete(self, system, messages, model, max_tokens):
        if system == "":
            return '{"pass": true, "reason": "fine", "turn": null}'
        if system.startswith("You are role playing a customer"):
            return "<END_CALL>"
        return "Namaste ji, main Acme se bol raha hoon."


def test_llm_agent_tokens_metered_by_role(capsys, monkeypatch, tmp_path):
    monkeypatch.setattr(evaluate, "get_backend", lambda name=None, **_: FakeLive())
    code, lines, _ = run_eval(capsys, "--agent", "v2", "--callers", "2", "--backend", "claude-cli",
                              "--max-turns", "2", "--out", str(tmp_path))
    assert code == 0
    record = load_record(next(tmp_path.glob("*.json")))
    by_role = record["cost"]["by_role"]
    assert record["cost"]["label"] == "estimated" and record["latency"]["label"] == "measured"
    for role in ("caller", "agent", "judge"):
        assert by_role[role]["input_tokens"] > 0 and by_role[role]["output_tokens"] > 0
    assert by_role["agent"]["input_tokens"] > by_role["agent"]["output_tokens"]  # the SOP prompt goes in every turn
    assert any(line.startswith("tokens (estimated) caller ") for line in lines)
    assert any(line.startswith("latency (measured) P50 ") for line in lines)


def test_unknown_model_price_prints_unknown_cost(capsys, monkeypatch):
    prices = report_module.load_prices()
    prices["models"] = {k: {"input": None, "output": None} for k in prices["models"]}
    monkeypatch.setattr(report_module, "load_prices", lambda: prices)
    _, lines, _ = run_story(capsys, "v1")
    cost = next(line for line in lines if line.startswith("cost"))
    assert cost.startswith("cost: unknown, no price for ") and "claude-" in cost
