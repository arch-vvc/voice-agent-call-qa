"""The `eval` command: pinned output lines, pass^k threshold, saved records, the LLM call guard. No network."""
from pathlib import Path

import pytest
import yaml

from callqa import evaluate
from callqa.runrecord import load_record
from callqa.server import needs_confirmation
from evals.backends import BackendError

HERE = Path(__file__).resolve().parent
ROWS = yaml.safe_load((HERE / "fixtures" / "full_set_expected.yaml").read_text(encoding="utf-8"))["rows"]
FLAKY = {"flaky_line": 2}  # persona -> repeats (of 3) it passes: one of its three script variants misses "friday"
OFFLINE = ("--backend", "offline", "--no-save")


def pct(x):
    return format(x, ".0%")


def run_eval(capsys, *args):
    code = evaluate.main(list(args))
    out = capsys.readouterr()
    return code, out.out.splitlines(), out.err


def expected_full(agent):
    """(passing all 3 repeats, personas, runs passed, runs) for the offline full set, from the expected rows."""
    passing = [pid for pid, row in ROWS.items() if row[agent] == "pass"]
    every = [pid for pid in passing if pid not in FLAKY]
    runs_passed = sum(FLAKY.get(pid, 3) for pid in passing)
    return len(every), len(ROWS), runs_passed, len(ROWS) * 3


def test_eval_story_v1_prints_pinned_lines(capsys):
    code, lines, _ = run_eval(capsys, "--agent", "v1", "--set", "story", *OFFLINE)
    assert code == 0
    assert lines[:3] == [
        "EVAL agent=v1 set=story callers=20 repeats=1 seed=0 backend=offline",
        "pass^1 70% (14/20)  pass@1 70% (14/20 runs)",
        "flaky: n/a (repeats 1)",
    ]


@pytest.mark.parametrize("agent", ["v1", "v2"])
def test_eval_full_repeats_3_prints_pass_hat_k_and_flaky_line(capsys, agent):
    every, n, runs_passed, runs = expected_full(agent)
    code, lines, _ = run_eval(capsys, "--agent", agent, "--set", "full", "--repeats", "3", *OFFLINE)
    assert code == 0
    assert lines[0] == f"EVAL agent={agent} set=full callers={n} repeats=3 seed=0 backend=offline"
    assert lines[1] == (f"pass^3 {pct(every / n)} ({every}/{n})  "
                        f"pass@1 {pct(runs_passed / runs)} ({runs_passed}/{runs} runs)")
    # the third repeat replays the variant that never says friday (repeat numbers print from 1)
    assert lines[2] == "flaky (1): flaky_line 2/3 repeats passed, failed repeat 3"


def test_threshold_uses_pass_hat_k_when_repeats_over_1(capsys):
    every, n, runs_passed, runs = expected_full("v2")
    assert every / n < 0.72 < runs_passed / runs  # pass^3 is below the bar, pass@1 is above it
    args = ("--agent", "v2", "--set", "full", "--repeats", "3", *OFFLINE)
    code, lines, _ = run_eval(capsys, *args, "--threshold", "0.72")
    assert code == 1
    assert lines[-1] == "threshold 72% on pass^3: FAIL"
    code, lines, _ = run_eval(capsys, *args, "--threshold", "0.7")
    assert code == 0
    assert lines[-1] == "threshold 70% on pass^3: PASS"


def test_threshold_with_one_repeat_uses_the_pass_rate(capsys):
    args = ("--agent", "v1", "--set", "story", *OFFLINE)
    code, lines, _ = run_eval(capsys, *args, "--threshold", "0.8")
    assert (code, lines[-1]) == (1, "threshold 80% on pass^1: FAIL")
    code, lines, _ = run_eval(capsys, *args, "--threshold", "0.5")
    assert (code, lines[-1]) == (0, "threshold 50% on pass^1: PASS")


def test_eval_saves_record_unless_no_save(capsys, tmp_path):
    out = tmp_path / "records"
    code, lines, _ = run_eval(capsys, "--agent", "v2", "--backend", "offline", "--out", str(out))
    assert code == 0
    assert lines[-1].startswith("record: ")
    path = Path(lines[-1][len("record: "):])
    assert path.parent == out and path.exists()
    record = load_record(path)
    assert (record["agent"]["id"], record["persona_set"]["name"], record["passed"], record["total"]) == (
        "v2", "story", 19, 20)
    code, lines, _ = run_eval(capsys, "--agent", "v2", "--backend", "offline", "--out", str(out), "--no-save")
    assert not any(line.startswith("record:") for line in lines)
    assert len(list(out.glob("*.json"))) == 1


class LimitBackend:
    """Stands in for claude-cli. Every call is a limit error, so a run that gets past the guard aborts."""

    name = "claude-cli"
    requested = "claude-cli"
    warnings: list = []

    def __init__(self):
        self.calls = 0

    def complete(self, system, messages, model, max_tokens):
        self.calls += 1
        raise BackendError("fake limit", kind="limit")


def test_eval_llm_backend_over_200_calls_needs_yes(capsys, monkeypatch):
    backend = LimitBackend()
    monkeypatch.setattr(evaluate, "get_backend", lambda name=None, **_: backend)
    args = ("--agent", "v1", "--set", "full", "--repeats", "3", "--backend", "claude-cli", "--no-save")
    code, lines, err = run_eval(capsys, *args)  # (2*4 + 3) * 31 * 3 = 1023 calls
    assert code == 2 and lines == []
    assert "1023" in err and "--yes" in err
    assert backend.calls == 0
    code, lines, err = run_eval(capsys, *args, "--yes")  # past the guard: the fake limit aborts the run
    assert code == 2 and backend.calls > 0
    assert any(line.startswith("ABORTED") for line in lines)


def test_eval_llm_backend_at_or_under_200_calls_needs_no_yes(capsys, monkeypatch):
    backend = LimitBackend()
    monkeypatch.setattr(evaluate, "get_backend", lambda name=None, **_: backend)
    code, _, err = run_eval(capsys, "--agent", "v1", "--callers", "5", "--backend", "claude-cli", "--no-save")
    assert "--yes" not in err and backend.calls > 0  # 55 calls: it ran, and the fake limit aborted it
    assert code == 2


@pytest.mark.parametrize("bad", [
    ["--agent", "v9"],
    ["--agent", "v1", "--repeats", "0"],
    ["--agent", "v1", "--repeats", "11"],
    ["--agent", "v1", "--repeats", "two"],
    ["--agent", "v1", "--set", "huge"],
    ["--agent", "v1", "--callers", "0"],
    ["--agent", "v1", "--set", "full", "--callers", "5"],
    ["--agent", "v1", "--threshold", "1.5"],
    ["--agent", "v1", "--threshold", "-0.1"],
    ["--agent", "v1", "--workers", "0"],
    ["--agent", "v1", "--max-turns", "0"],
    ["--agent", "v1", "--no-such-flag"],
    [],
])
def test_bad_args_exit_2(capsys, bad):
    code, lines, err = run_eval(capsys, *OFFLINE, *bad)
    assert code == 2 and lines == []
    assert err.strip()


def test_eval_with_seed_replays_a_different_variant(capsys):
    code, lines, _ = run_eval(capsys, "--agent", "v2", "--set", "full", "--repeats", "3", "--seed", "7", *OFFLINE)
    assert code == 0
    assert lines[0].endswith("seed=7 backend=offline")
    assert lines[2] == "flaky (1): flaky_line 2/3 repeats passed, failed repeat 2"


def test_out_ending_in_json_is_the_record_file(capsys, tmp_path):
    target = tmp_path / "deep" / "er" / "full_v1.json"
    code, lines, _ = run_eval(capsys, "--agent", "v1", "--backend", "offline", "--out", str(target))
    assert code == 0 and lines[-1] == f"record: {target}"
    assert load_record(target)["agent"]["id"] == "v1"
    before = target.read_text(encoding="utf-8")
    code, lines, err = run_eval(capsys, "--agent", "v2", "--backend", "offline", "--out", str(target))
    assert code == 2 and "already exists" in err
    assert target.read_text(encoding="utf-8") == before


def test_out_without_json_is_a_folder(capsys, tmp_path):
    code, lines, _ = run_eval(capsys, "--agent", "v1", "--backend", "offline", "--out", str(tmp_path / "recs"))
    assert code == 0
    assert Path(lines[-1][len("record: "):]).parent == tmp_path / "recs"


class FakeBackend:
    def __init__(self, name):
        self.name = name


def test_needs_confirmation_boundary():
    assert not needs_confirmation(FakeBackend("claude-cli"), 200)
    assert needs_confirmation(FakeBackend("claude-cli"), 201)
    assert needs_confirmation(FakeBackend("anthropic"), 201)
    assert not needs_confirmation(FakeBackend("offline"), 10_000)


def test_eval_lines_separates_several_flaky_callers_with_semicolons():
    calls = [{"repeats_passed": 2, "repeats_total": 3}, {"repeats_passed": 1, "repeats_total": 3},
             {"repeats_passed": 3, "repeats_total": 3}]
    record = {"repeats": 3, "total": 3, "passed": 1, "seed": 0, "agent": {"id": "v1"},
              "persona_set": {"name": "full"}, "backend": {"name": "offline"}, "calls": calls,
              "consistency": {"flaky": [{"persona_id": "a", "passed": 2, "total": 3, "failed_repeat": 2},
                                        {"persona_id": "b", "passed": 1, "total": 3, "failed_repeat": 0}]}}
    assert evaluate.eval_lines(record)[2] == (
        "flaky (2): a 2/3 repeats passed, failed repeat 3; b 1/3 repeats passed, failed repeat 1")
