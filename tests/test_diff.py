"""The `diff` command and regression gate, on tmp records built by hand or by `eval --out`. No network."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from callqa import diff
from callqa.runrecord import RECORD_SCHEMA_VERSION

DEMO = Path(__file__).resolve().parents[1]
RUN_PY = DEMO / "run.py"


def make_record(agent, results, repeats=1, set_name="full", **extra):
    """A minimal valid run record. `results` maps persona id -> True/False, or -> repeats passed (int)."""
    calls = []
    for pid, value in results.items():
        passed = int(value) if isinstance(value, bool) else value
        total = repeats
        calls.append({"persona_id": pid, "repeats_passed": passed, "repeats_total": total})
    every = sum(c["repeats_passed"] == c["repeats_total"] for c in calls)
    record = {
        "schema_version": RECORD_SCHEMA_VERSION, "kind": "run", "created_at": "2026-11-05T17:07:24+05:30",
        "synthetic": True, "simulated": True, "note": "SYNTHETIC", "run_id": "abcd1234", "sop_hash": "x",
        "agent": {"id": agent, "label": agent, "variant": agent, "model": None},
        "backend": {"name": "offline", "models": {}},
        "persona_set": {"name": set_name, "count": len(calls), "ids_sha256": "same"},
        "seed": 0, "repeats": repeats, "callers": len(calls), "max_turns": 8, "rules_file": None, "warnings": [],
        "generator": {"source": "templates", "warnings": []},
        "pass_rate": every / len(calls), "passed": every, "total": len(calls), "judge_split": 0, "seconds": 0.1,
        "consistency": None, "cost": None, "latency": None, "escalation": None, "compliance": None, "traits": None,
        "calls": calls, "next_fixes": [],
    }
    record.update(extra)
    return record


def write(tmp_path, name, record):
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def run_diff(capsys, *argv):
    code = diff.main([str(a) for a in argv])
    out = capsys.readouterr()
    return code, out.out.splitlines(), out.err


@pytest.fixture
def pair(tmp_path):
    old = make_record("v1", {"zeta": True, "alpha": True, "fix_b": False, "fix_a": False, "stuck_b": False,
                             "stuck_a": False, "mid": True})
    new = make_record("v2", {"zeta": False, "alpha": False, "fix_b": True, "fix_a": True, "stuck_b": False,
                             "stuck_a": False, "mid": True})
    return write(tmp_path, "old", old), write(tmp_path, "new", new)


def test_newly_broken_fixed_and_still_failing_are_sorted_lists(capsys, pair):
    code, lines, _ = run_diff(capsys, *pair)
    assert code == 0
    assert lines == [
        "DIFF old (v1) -> new (v2)  set=full personas=7 repeats=1 backend=offline",
        "NEWLY BROKEN (2): alpha, zeta",
        "FIXED (2): fix_a, fix_b",
        "STILL FAILING (2): stuck_a, stuck_b",
        "pass^1 43% (3/7) -> 43% (3/7), +0 pts",
    ]


def test_empty_lists_print_none(capsys, tmp_path):
    rec = write(tmp_path, "same", make_record("v3", {"a": True}))
    _, lines, _ = run_diff(capsys, rec, rec)
    assert lines[1:4] == ["NEWLY BROKEN (0): none", "FIXED (0): none", "STILL FAILING (0): none"]


def test_gate_exits_1_on_any_regression_and_0_otherwise(capsys, tmp_path, pair):
    code, lines, _ = run_diff(capsys, *pair, "--gate")
    assert code == 1
    assert lines[-1] == "GATE FAILED: 2 previously passing personas now fail: alpha, zeta"
    one = write(tmp_path, "one", make_record("v2", {"a": True, "b": False}))
    two = write(tmp_path, "two", make_record("v3", {"a": False, "b": True}))
    code, lines, _ = run_diff(capsys, one, two, "--gate")
    assert (code, lines[-1]) == (1, "GATE FAILED: 1 previously passing persona now fails: a")
    code, lines, _ = run_diff(capsys, two, one, "--gate")  # b regresses
    assert code == 1
    good = write(tmp_path, "good", make_record("v3", {"a": True, "b": True}))
    code, lines, _ = run_diff(capsys, one, good, "--gate")
    assert (code, lines[-1]) == (0, "GATE OK: no previously passing persona fails")


def test_without_gate_exit_is_0(capsys, pair):
    code, lines, _ = run_diff(capsys, *pair)
    assert code == 0
    assert not any(line.startswith("GATE") for line in lines)


def test_pass_hat_k_regression_counts_flaky_as_failing(capsys, tmp_path):
    old = write(tmp_path, "old", make_record("v2", {"steady": 3, "wobbly": 3, "weak": 2}, repeats=3))
    new = write(tmp_path, "new", make_record("v2", {"steady": 3, "wobbly": 2, "weak": 3}, repeats=3))
    code, lines, _ = run_diff(capsys, old, new, "--gate")
    assert code == 1
    assert lines[1:4] == ["NEWLY BROKEN (1): wobbly", "FIXED (1): weak", "STILL FAILING (0): none"]
    assert lines[4] == "pass^3 67% (2/3) -> 67% (2/3), +0 pts"
    assert lines[-1] == "GATE FAILED: 1 previously passing persona now fails: wobbly"
    # a flaky persona that stays flaky is still failing, never newly broken
    flaky = write(tmp_path, "flaky", make_record("v2", {"steady": 3, "wobbly": 1, "weak": 3}, repeats=3))
    code, lines, _ = run_diff(capsys, new, flaky, "--gate")
    assert code == 0
    assert lines[1:4] == ["NEWLY BROKEN (0): none", "FIXED (0): none", "STILL FAILING (1): wobbly"]


def test_points_use_unrounded_rates(capsys, tmp_path):
    old = write(tmp_path, "old", make_record("v1", {"a": True, "b": False, "c": False}))  # 1/3 = 33%
    new = write(tmp_path, "new", make_record("v2", {"a": True, "b": True, "c": False}))  # 2/3 = 67%
    _, lines, _ = run_diff(capsys, old, new)
    # rounded rates would give 67 - 33 = +34; the unrounded difference is 33.33
    assert lines[-1] == "pass^1 33% (1/3) -> 67% (2/3), +33 pts"
    assert diff.diff_records(json.loads(old.read_text()), json.loads(new.read_text())).points == 33


def test_cost_and_latency_lines_only_when_both_present(capsys, tmp_path):
    cost = {"label": "simulated", "per_call": 0.0012, "total": 0.0024}
    latency = {"label": "simulated", "p95_ms": 1210}
    bare = write(tmp_path, "bare", make_record("v1", {"a": True}))
    old = write(tmp_path, "old", make_record("v1", {"a": True}, cost=cost, latency=latency))
    new = write(tmp_path, "new", make_record("v2", {"a": True}, cost={**cost, "per_call": 0.0009},
                                              latency={**latency, "p95_ms": 900}))
    _, lines, _ = run_diff(capsys, old, new)
    assert lines[-2:] == ["cost per call (simulated) $0.0012 -> $0.0009",
                          "P95 agent turn (simulated) 1210 ms -> 900 ms"]
    for a, b in ((bare, new), (old, bare), (bare, bare)):
        _, lines, _ = run_diff(capsys, a, b)
        assert len(lines) == 5 and lines[-1].startswith("pass^1")


def test_added_and_removed_personas_listed(capsys, tmp_path):
    old = write(tmp_path, "old", make_record("v1", {"keep": True, "gone_b": True, "gone_a": False}))
    new = write(tmp_path, "new", make_record("v2", {"keep": True, "new_b": False, "new_a": True}))
    code, lines, _ = run_diff(capsys, old, new, "--gate")
    assert code == 0  # a removed passing persona is not a regression, an added failing one is not either
    assert "ADDED (2): new_a, new_b" in lines
    assert "REMOVED (2): gone_a, gone_b" in lines
    assert lines.index("ADDED (2): new_a, new_b") < lines.index("REMOVED (2): gone_a, gone_b")
    assert lines[1] == "NEWLY BROKEN (0): none"


def test_mismatch_prints_warning_lines(capsys, tmp_path):
    old = write(tmp_path, "old", make_record("v1", {"a": True}, set_name="story"))
    new = write(tmp_path, "new", make_record("v1", {"a": True}, repeats=3, set_name="full"))
    new_record = json.loads(new.read_text())
    new_record["calls"][0]["repeats_passed"] = 3
    new_record["passed"] = 1
    _, lines, _ = run_diff(capsys, old, write(tmp_path, "new2", new_record))
    warnings = [line for line in lines if line.startswith("warning: ")]
    assert "warning: persona sets differ (story vs full), results are not comparable" in warnings
    assert "warning: repeats differ (1 vs 3)" in warnings
    assert lines[0].startswith("DIFF") and lines[1].startswith("warning: ")
    assert any(line.startswith("pass^1 100% (1/1) -> pass^3 100% (1/1)") for line in lines)


def test_bad_record_exits_2(capsys, tmp_path):
    good = write(tmp_path, "good", make_record("v1", {"a": True}))
    wrong_kind = write(tmp_path, "kind", make_record("v1", {"a": True}, kind="report"))
    newer = write(tmp_path, "newer", make_record("v1", {"a": True}, schema_version=99))
    half = tmp_path / "half.json"
    half.write_text('{"kind": "run", "schema_', encoding="utf-8")
    no_calls = write(tmp_path, "nocalls", make_record("v1", {"a": True}, calls=[{"name": "x"}]))
    for bad in (tmp_path / "missing.json", wrong_kind, newer, half, no_calls):
        for args in ((bad, good), (good, bad)):
            code, lines, err = run_diff(capsys, *args, "--gate")
            assert code == 2, bad
            assert lines == []
            assert err.startswith("error: ")


def cli(*argv):
    return subprocess.run([sys.executable, str(RUN_PY), *map(str, argv)], capture_output=True, text=True, timeout=300)


def test_cli_diff_through_the_dispatch_table(tmp_path):
    for agent in ("v1", "v2"):
        made = cli("eval", "--agent", agent, "--set", "full", "--repeats", "3", "--backend", "offline",
                   "--out", tmp_path / f"{agent}.json")
        assert made.returncode == 0, made.stderr
    v1, v2 = tmp_path / "v1.json", tmp_path / "v2.json"
    forward = cli("diff", v1, v2, "--gate")
    assert forward.stderr == ""
    lines = forward.stdout.splitlines()
    n = json.loads(v2.read_text())["persona_set"]["count"]  # 21 now, 22 once holder_says_bhai lands (A3d)
    assert lines[0] == f"DIFF v1 (v1) -> v2 (v2)  set=full personas={n} repeats=3 backend=offline"
    assert lines[1].startswith("NEWLY BROKEN (")
    assert any(line.startswith("FIXED (") for line in lines) and any(line.startswith("pass^3 ") for line in lines)
    regressed = diff.diff_records(json.loads(v1.read_text()), json.loads(v2.read_text())).regressed
    assert forward.returncode == (1 if regressed else 0)
    assert lines[-1].startswith("GATE FAILED: " if regressed else "GATE OK: ")
    backward = cli("diff", v2, v1, "--gate")  # v2 to v1 loses everything v2 fixed
    assert backward.returncode == 1
    assert backward.stdout.splitlines()[-1].startswith("GATE FAILED: ")
    plain = cli("diff", v2, v1)
    assert plain.returncode == 0 and "GATE" not in plain.stdout
    missing = cli("diff", v1, tmp_path / "nope.json")
    assert missing.returncode == 2 and "error: " in missing.stderr


def test_cli_smoke_and_eval_still_work(tmp_path):
    smoke = cli("--smoke")
    assert smoke.returncode == 0
    assert smoke.stdout.splitlines()[-1] == "SMOKE OK v1 70% (14/20) v2 95% (19/20)"
    ev = cli("eval", "--agent", "v1", "--set", "story", "--backend", "offline", "--no-save")
    assert ev.returncode == 0
    assert ev.stdout.splitlines()[1] == "pass^1 70% (14/20)  pass@1 70% (14/20 runs)"
