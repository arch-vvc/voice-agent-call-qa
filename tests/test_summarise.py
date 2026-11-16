"""`--summarise <folder>`: a Markdown table of committed live runs. Synthetic files only, no model calls."""
import json
import subprocess
import sys
from pathlib import Path

from callqa.__main__ import main
from callqa.summarise import NONE_YET, summarise

ROOT = Path(__file__).resolve().parents[1]
MODELS = {"caller": "model-a", "agent": "model-a", "judge": "model-b", "gen": "model-a"}


def write_run(folder, stamp, agent, rate, passed, total, backend="claude-cli", **extra):
    data = {"run_id": stamp, "agent": {"id": agent, "label": agent}, "pass_rate": rate, "passed": passed,
            "total": total, "callers": total, "max_turns": 4,
            "backend": {"name": backend, "models": {} if backend == "offline" else MODELS}, **extra}
    (folder / f"dashboard-{stamp}-{agent}.json").write_text(json.dumps(data), encoding="utf-8")


def test_empty_folder(tmp_path):
    assert summarise(tmp_path) == NONE_YET == "No live runs committed yet."


def test_missing_folder(tmp_path):
    assert summarise(tmp_path / "nope") == NONE_YET


def test_two_v1_and_two_v2_runs_give_mean_and_range(tmp_path):
    write_run(tmp_path, "20261117-090000", "v1", 0.6, 12, 20)
    write_run(tmp_path, "20261117-100000", "v1", 0.8, 16, 20)
    write_run(tmp_path, "20261118-090000", "v2", 0.9, 18, 20)
    write_run(tmp_path, "20261118-100000", "v2", 1.0, 20, 20)
    out = summarise(tmp_path)
    assert "| date | agent version | backend | models | callers | turns |" in out
    assert "| passed | failed | needs review | pass rate |" in out
    assert "| 2026-11-17 | v1 | claude-cli | model-a, model-b | 20 | 4 | 12 | 8 | 0 | 60.0% |" in out
    assert "v1: 2 runs, mean 70.0%, lowest 60.0%, highest 80.0%" in out
    assert "v2: 2 runs, mean 95.0%, lowest 90.0%, highest 100.0%" in out
    rows = [line for line in out.splitlines() if line.startswith("| 2026")]
    assert rows == sorted(rows)  # date then agent version


def test_offline_file_is_left_out(tmp_path):
    write_run(tmp_path, "20261117-090000", "v1", 0.7, 14, 20, backend="offline")
    assert summarise(tmp_path) == NONE_YET
    write_run(tmp_path, "20261117-100000", "v2", 0.9, 18, 20)
    out = summarise(tmp_path)
    assert "offline" not in out and "v1" not in out and "v2: 1 run," in out


def test_broken_files_are_skipped_and_named(tmp_path):
    write_run(tmp_path, "20261117-090000", "v1", 0.5, 10, 20)
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "no_rate.json").write_text(json.dumps({"passed": 1}), encoding="utf-8")
    out = summarise(tmp_path)
    assert "Skipped (not valid result files): broken.json, no_rate.json" in out
    assert "v1: 1 run," in out


def test_single_run_has_same_mean_lowest_and_highest(tmp_path):
    write_run(tmp_path, "20261117-090000", "v1", 0.85, 17, 20)
    assert "v1: 1 run, mean 85.0%, lowest 85.0%, highest 85.0%" in summarise(tmp_path)


def test_needs_review_is_counted_separately_and_old_flaky_files_work(tmp_path):
    write_run(tmp_path, "20261117-090000", "v1", 0.5, 10, 20, needs_review=3)
    write_run(tmp_path, "20261117-100000", "v2", 0.5, 10, 20, flaky=2)  # old file: only `flaky`
    out = summarise(tmp_path)
    assert "| 10 | 7 | 3 | 50.0% |" in out and "| 10 | 8 | 2 | 50.0% |" in out


def test_main_prints_the_table_and_exits_zero(tmp_path, capsys):
    assert main(["--summarise", str(tmp_path)]) == 0
    assert capsys.readouterr().out.strip() == NONE_YET


def test_the_committed_live_folder_is_empty_and_run_py_works():
    assert [p.name for p in (ROOT / "results" / "live").iterdir()] == [".gitkeep"]
    done = subprocess.run([sys.executable, str(ROOT / "run.py"), "--summarise", str(ROOT / "results" / "live")],
                          capture_output=True, text=True, check=True)
    assert done.stdout.strip() == NONE_YET
