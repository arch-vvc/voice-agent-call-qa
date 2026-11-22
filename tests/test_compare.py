"""The `compare` command: a bake off over agent configs on the same callers and seeds. No network, no model."""
import json
import re
from pathlib import Path

import pytest

from callqa import __main__ as cli
from callqa import compare, evaluate
from evals.backends import BackendError

HERE = Path(__file__).resolve().parent
GOLDEN = HERE / "golden"
OFFLINE = ("--backend", "offline", "--no-save")


def run_compare(capsys, *args):
    code = compare.main(list(args))
    out = capsys.readouterr()
    return code, out.out.splitlines(), out.err


def eval_pass_cell(capsys, agent, *extra):
    """`pass^k 56% (18/32)` as the eval command prints it: the number the compare row must repeat."""
    code = evaluate.main(["--agent", agent, "--backend", "offline", "--no-save", *extra])
    lines = capsys.readouterr().out.splitlines()
    assert code == 0
    return re.match(r"pass\^\d (\d+% \(\d+/\d+\))", lines[1]).group(1)


def golden_cost_and_p95(agent):
    text = (GOLDEN / f"eval_story_{agent}.txt").read_text(encoding="utf-8")
    cost = re.search(r"cost \(simulated\) \$(\d\.\d{4}) per call", text).group(1)
    p95 = re.search(r"P95 (\d+) ms", text).group(1)
    return f"${cost}", p95


def row(config, passed, total, critical=0, cost=0.01, p95=1000, right=9, runs=10):
    return {"config": config, "passed": passed, "total": total, "critical": critical,
            "escalation_right": right, "escalation_runs": runs, "cost_per_call": cost, "p95_ms": p95}


# ---- the pinned offline output ----

def test_offline_compare_prints_pinned_lines(capsys):
    code, lines, _ = run_compare(capsys, "--set", "story", "--agents", "v1,v2", *OFFLINE)
    assert code == 0
    assert lines[0] == "COMPARE set=story callers=20 repeats=1 seed=0 backend=offline"
    assert lines[1] == ("note: offline agents are rule based, so models are not compared; "
                        "comparing prompt versions v1, v2")
    assert lines[2].split() == ["config", "pass^1", "critical", "escalation", "cost/call", "P95", "ms"]
    assert lines[2].startswith("config  pass^1         critical  escalation    cost/call  P95 ms")
    for line, agent in zip(lines[3:5], ("v1", "v2"), strict=True):
        cost, p95 = golden_cost_and_p95(agent)  # from the eval goldens, so the two commands cannot disagree
        cells = line.split()
        assert cells[0] == agent
        assert f"{cells[1]} {cells[2]}" == eval_pass_cell(capsys, agent, "--set", "story")
        assert cells[-2:] == [cost, p95]
        assert line.startswith(f"{agent}      ")  # the pinned column positions
    assert lines[5].startswith("best: pass^1 v2; critical ")
    assert "; cost " in lines[5] and "; P95 " in lines[5]
    assert any(line.startswith("disagree (") for line in lines)
    assert any(line.startswith("recommend: ") for line in lines)
    # offline numbers must say they are simulated, never real model quality
    assert any("simulated" in line and "real model quality" in line for line in lines)


def test_default_offline_compare_is_v1_v2_v3_on_the_full_set(capsys):
    code, lines, _ = run_compare(capsys, "--set", "full", *OFFLINE)
    assert code == 0
    assert lines[0] == "COMPARE set=full callers=31 repeats=1 seed=0 backend=offline"
    assert lines[1] == ("note: offline agents are rule based, so models are not compared; "
                        "comparing prompt versions v1, v2, v3")
    assert [ln.split()[0] for ln in lines[3:6]] == ["v1", "v2", "v3"]
    for line, agent in zip(lines[3:6], ("v1", "v2", "v3"), strict=True):
        cells = line.split()
        assert f"{cells[1]} {cells[2]}" == eval_pass_cell(capsys, agent, "--set", "full")
    assert any(ln.startswith("best: pass^1 v3;") for ln in lines)


def test_compare_saves_each_run_record_and_a_compare_record(capsys, tmp_path):
    out = tmp_path / "recs"
    code, lines, _ = run_compare(capsys, "--set", "story", "--agents", "v1,v2", "--backend", "offline",
                                 "--out", str(out))
    assert code == 0
    files = sorted(out.glob("*.json"))
    assert len(files) == 3  # two run records and one compare record
    kinds = sorted(json.loads(f.read_text(encoding="utf-8"))["kind"] for f in files)
    assert kinds == ["compare", "run", "run"]
    compare_file = next(f for f in files if json.loads(f.read_text(encoding="utf-8"))["kind"] == "compare")
    record = json.loads(compare_file.read_text(encoding="utf-8"))
    assert record["schema_version"] == 1 and record["simulated"] is True and record["synthetic"] is True
    assert [c["config"] for c in record["configs"]] == ["v1", "v2"]
    assert record["recommendation"]["config"] == "v2"
    assert sorted(record["run_files"]) == sorted(f.name for f in files if f != compare_file)
    assert any(line == f"record: {compare_file}" for line in lines)


def test_out_json_names_the_compare_record_and_never_overwrites(capsys, tmp_path):
    target = tmp_path / "cmp.json"
    code, _, _ = run_compare(capsys, "--agents", "v1,v2", "--backend", "offline", "--out", str(target))
    assert code == 0 and json.loads(target.read_text(encoding="utf-8"))["kind"] == "compare"
    assert len(list((tmp_path / "cmp_runs").glob("*.json"))) == 2
    code, _, err = run_compare(capsys, "--agents", "v1,v2", "--backend", "offline", "--out", str(target))
    assert code == 2 and "already exists" in err


def test_no_save_writes_nothing(capsys, tmp_path):
    out = tmp_path / "recs"
    code, lines, _ = run_compare(capsys, "--agents", "v1,v2", "--backend", "offline", "--no-save", "--out", str(out))
    assert code == 0 and not out.exists()
    assert not any(line.startswith("record:") for line in lines)


# ---- same personas, same seeds ----

def test_same_personas_and_seeds_for_every_config(capsys, monkeypatch):
    seen = []
    real = compare.run_suite

    def spy(respond, personas, runs, max_turns, workers, backend, seed, **kw):
        seen.append(([p["id"] for p in personas], runs, max_turns, seed))
        return real(respond, personas, runs, max_turns, workers, backend, seed, **kw)

    monkeypatch.setattr(compare, "run_suite", spy)
    code, _, _ = run_compare(capsys, "--set", "full", "--repeats", "2", "--seed", "7", *OFFLINE)
    assert code == 0
    assert len(seen) == 3
    assert seen[0][1:] == (2, 8, 7)
    assert seen[0] == seen[1] == seen[2]
    assert len(seen[0][0]) == 31


# ---- the table maths ----

def test_best_per_column_lists_ties():
    rows = [row("a", 18, 32, critical=0, cost=0.02, p95=900),
            row("b", 23, 32, critical=0, cost=0.01, p95=900),
            row("c", 23, 32, critical=2, cost=0.01, p95=1200, right=10)]
    best = compare.best_per_column(rows)
    assert best["pass"] == ["b", "c"]  # a tie lists every tied config, in the order given
    assert best["critical"] == ["a", "b"]
    assert best["escalation"] == ["c"]
    assert best["cost"] == ["b", "c"]
    assert best["p95"] == ["a", "b"]


def test_best_ignores_missing_cost_and_compares_exact_fractions():
    rows = [row("a", 1, 3, cost=None), row("b", 2, 6, cost=None), row("c", 1, 4, cost=None)]
    best = compare.best_per_column(rows)
    assert best["pass"] == ["a", "b"]  # 1/3 and 2/6 are equal; 1/4 is not
    assert best["cost"] == []  # nothing priced, nothing to win


def test_best_line_prints_n_a_for_a_column_nobody_won():
    best = {"pass": ["v3"], "critical": ["v3"], "escalation": ["v2", "v3"], "cost": [], "p95": ["v1"]}
    line = compare.best_line(best, "pass^1")
    assert line == "best: pass^1 v3; critical v3; escalation v2, v3; cost n/a; P95 v1"


def calls_with(**outcomes):
    return [{"persona_id": pid, "passed": ok} for pid, ok in outcomes.items()]


def test_disagreements_are_personas_with_mixed_outcomes():
    one = calls_with(a=True, b=True, c=False, d=False)
    two = calls_with(a=True, b=False, c=False, d=True)
    three = calls_with(a=True, b=True, c=False, d=True)
    assert compare.disagreements([one, two, three]) == ["b", "d"]  # sorted by id; a and c agree everywhere
    assert compare.disagreements([one, one]) == []
    assert compare.disagreements([one]) == []


# ---- the recommendation rule ----

def test_recommendation_rule():
    rule = compare.RECOMMEND_RULE
    assert rule == "highest pass^1 with zero critical compliance failures; ties broken by lower cost per call"
    # highest pass rate among configs with zero critical failures: a 100% config with a critical failure loses
    rows = [row("v1", 18, 32, critical=0), row("v2", 30, 32, critical=1), row("v3", 23, 32, critical=0)]
    assert compare.recommend(rows, 1)["config"] == "v3"
    # a tie on pass rate goes to the cheaper config
    tie = [row("a", 20, 32, cost=0.02), row("b", 20, 32, cost=0.01), row("c", 10, 32, cost=0.001)]
    assert compare.recommend(tie, 1)["config"] == "b"
    # an unpriced config loses a cost tie to a priced one
    unpriced = [row("a", 20, 32, cost=None), row("b", 20, 32, cost=0.5)]
    assert compare.recommend(unpriced, 1)["config"] == "b"
    # still tied after cost: no winner is invented
    same = [row("a", 20, 32, cost=0.01), row("b", 20, 32, cost=0.01)]
    result = compare.recommend(same, 1)
    assert result["config"] is None and "a, b" in result["text"] and result["text"].startswith("recommend: none")
    # every config has a critical failure
    bad = [row("a", 20, 32, critical=1), row("b", 21, 32, critical=3)]
    result = compare.recommend(bad, 1)
    assert result["config"] is None and "critical" in result["text"]
    # the pass^k name follows the repeats
    assert compare.recommend(tie, 3)["rule"].startswith("highest pass^3")
    assert compare.recommend(tie, 1)["text"] == f"recommend: b ({rule})"


def test_recommendation_uses_exact_fractions_not_rounded_rates():
    rows = [row("a", 1, 3, cost=0.9), row("b", 333, 1000, cost=0.1)]  # 33.3% vs 33.3% rounded, a is higher
    assert compare.recommend(rows, 1)["config"] == "a"


# ---- the markdown page ----

def sample_compare_record(label="simulated", simulated=True):
    rows = [row("v1", 17, 31, critical=5), row("v2", 22, 31, critical=0), row("v3", 31, 31, critical=0, cost=0.02)]
    return {
        "schema_version": 1, "kind": "compare", "created_at": "2026-11-21T10:00:00+05:30",
        "synthetic": True, "simulated": simulated,
        "persona_set": {"name": "full", "count": 31}, "repeats": 1, "seed": 0, "callers": 31,
        "backend": {"name": "offline" if simulated else "claude-cli"},
        "cost_label": label, "latency_label": "simulated" if simulated else "measured",
        "prices": {"source_url": "https://example.test/pricing", "retrieved_on": "2026-11-06"},
        "note": "n", "configs": rows, "best": compare.best_per_column(rows),
        "disagreements": ["a", "b"], "recommendation": compare.recommend(rows, 1),
    }


def test_export_md_is_one_page_and_labels_simulated_numbers(capsys, tmp_path):
    target = tmp_path / "cmp.md"
    code, lines, _ = run_compare(capsys, "--set", "full", "--backend", "offline", "--no-save",
                                 "--export-md", str(target))
    assert code == 0 and any(line.startswith("exported: ") for line in lines)
    text = target.read_text(encoding="utf-8")
    assert len(text.splitlines()) <= 45  # one page
    assert "simulated" in text and "verify prices before quoting" in text
    assert "SYNTHETIC" in text and "recommend" in text.lower()
    assert "v3" in text
    assert "real model quality" in text  # the offline caveat
    assert " - " not in text and "-based" not in text
    assert "—" not in text and "–" not in text


def test_markdown_for_a_live_run_says_estimated_not_simulated_quality():
    text = compare.render_markdown(sample_compare_record(label="estimated", simulated=False))
    assert "estimated" in text and "verify prices before quoting" in text
    assert "rule based" not in text
    assert text.count("\n") <= 45


def test_markdown_lists_ties_and_long_disagreement_lists_stay_short():
    record = sample_compare_record()
    record["disagreements"] = [f"persona_{i}" for i in range(40)]
    text = compare.render_markdown(record)
    assert "and 20 more" in text and "persona_39" not in text


# ---- the model suffix ----

def test_parse_agents_accepts_variant_and_variant_at_model():
    assert compare.parse_agents("v2@haiku, v2@sonnet") == [("v2", "haiku"), ("v2", "sonnet")]
    assert compare.parse_agents("v1,v3") == [("v1", None), ("v3", None)]
    for bad in ("", "v9", "v2@", "@haiku", "v2@haiku,v2@haiku", "v2@bad model"):
        with pytest.raises(ValueError):
            compare.parse_agents(bad)


def test_model_suffix_reaches_build_agent(capsys, monkeypatch):
    built = []

    def fake_build_agent(sop_text, variant, backend, model=None):
        built.append((variant, model))
        return lambda history: "Namaste"

    backend = LimitBackend()
    monkeypatch.setattr(compare, "build_agent", fake_build_agent)
    monkeypatch.setattr(compare, "get_backend", lambda name=None, **_: backend)
    code, lines, _ = run_compare(capsys, "--set", "story", "--callers", "5", "--backend", "claude-cli",
                                 "--no-save")  # 110 calls: no guard; the fake limit then aborts the run
    assert built == [("v2", "claude-haiku-5-5")]  # the alias is resolved through config/pricing.yaml
    assert code == 2 and any(line.startswith("ABORTED") for line in lines)
    assert backend.calls > 0
    cfg = compare.AgentConfig("v2", "sonnet")
    compare.build_config_agent(cfg, "sop", backend)
    assert built[-1] == ("v2", "claude-sonnet-5-5")


# ---- the guards ----

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


def test_llm_compare_over_200_calls_needs_yes(capsys, monkeypatch):
    backend = LimitBackend()
    monkeypatch.setattr(compare, "get_backend", lambda name=None, **_: backend)
    args = ("--set", "full", "--backend", "claude-cli", "--no-save")
    code, lines, err = run_compare(capsys, *args)  # 2 configs x (2*4 + 3) x 31 = 682 calls
    assert code == 2 and lines == []
    assert "682" in err and "--yes" in err
    assert backend.calls == 0
    code, lines, err = run_compare(capsys, *args, "--yes")  # past the guard: the fake limit aborts the run
    assert code == 2 and backend.calls > 0
    assert any(line.startswith("ABORTED") for line in lines)


def test_offline_never_needs_yes_however_large(capsys):
    code, _, err = run_compare(capsys, "--set", "full", "--repeats", "3", *OFFLINE)
    assert code == 0 and "--yes" not in err


@pytest.mark.parametrize("bad", [
    ["--agents", "v1"],
    ["--agents", "v1,v1"],
    ["--agents", "v1,v9"],
    ["--agents", "v1@haiku,v2"],  # offline agents have no model
    ["--repeats", "0"],
    ["--repeats", "11"],
    ["--set", "huge"],
    ["--set", "full", "--callers", "5"],
    ["--callers", "0"],
    ["--workers", "0"],
    ["--max-turns", "0"],
    ["--latency-budget-ms", "0"],
    ["--no-such-flag"],
])
def test_bad_args_exit_2(capsys, bad):
    code, lines, err = run_compare(capsys, *OFFLINE, *bad)
    assert code == 2 and lines == []


# ---- wiring ----

def test_compare_is_in_the_dispatch_table(capsys):
    assert cli.COMMANDS["compare"] is compare.main
    assert {"eval", "diff", "compare"} <= set(cli.COMMANDS)
    code = cli.main(["compare", "--agents", "v1,v2", *OFFLINE])
    out = capsys.readouterr().out
    assert code == 0 and out.startswith("COMPARE set=story")
