"""Q3: scripts/update_readme_live.py rewrites only the live block of the README. Synthetic files in a temp folder."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("update_readme_live", ROOT / "scripts" / "update_readme_live.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

START, END = "<!-- live:start -->", "<!-- live:end -->"


def write_run(folder: Path, name: str, agent: str, passed: int, total: int, backend: str = "claude-cli") -> None:
    data = {"agent": {"id": agent}, "backend": {"name": backend, "models": {"agent": "m"}}, "callers": total,
            "max_turns": 4, "pass_rate": passed / total, "passed": passed, "total": total, "needs_review": 0,
            "calls": []}
    (folder / name).write_text(json.dumps(data))


@pytest.fixture
def setup(tmp_path):
    live = tmp_path / "live"
    live.mkdir()
    readme = tmp_path / "README.md"
    readme.write_text(f"# Title\n\nbefore text\n\n{START}\nold table\n{END}\n\nafter text\n")
    return live, readme


def test_first_run_replaces_block_and_prints_sentence(setup, capsys):
    live, readme = setup
    write_run(live, "dashboard-20261117-100000-v1.json", "v1", 4, 20)
    write_run(live, "dashboard-20261117-110000-v2.json", "v2", 14, 20)
    assert mod.main([str(live), "--readme", str(readme)]) == 0
    text = readme.read_text()
    assert text.startswith("# Title\n\nbefore text\n\n" + START)
    assert text.endswith(END + "\n\nafter text\n")
    assert "old table" not in text and "| 2026-11-17 | v2 |" in text
    assert capsys.readouterr().out.strip() == "v1 20% to v2 70% on 20 callers, claude-cli, 17 Nov 2026"


def test_second_run_changes_nothing(setup):
    live, readme = setup
    write_run(live, "dashboard-20261117-100000-v1.json", "v1", 4, 20)
    write_run(live, "dashboard-20261117-110000-v2.json", "v2", 14, 20)
    mod.main([str(live), "--readme", str(readme)])
    first = readme.read_bytes()
    mod.main([str(live), "--readme", str(readme)])
    assert readme.read_bytes() == first


def test_missing_markers_exit_2_and_leave_readme_alone(setup, capsys):
    live, readme = setup
    readme.write_text("# no markers here\n")
    assert mod.main([str(live), "--readme", str(readme)]) == 2
    assert readme.read_text() == "# no markers here\n"
    assert "live:start" in capsys.readouterr().err


def test_no_pair_prints_no_sentence(setup, capsys):
    live, readme = setup
    write_run(live, "dashboard-20261117-100000-v1.json", "v1", 4, 20)
    write_run(live, "dashboard-20261118-110000-v2.json", "v2", 14, 20)  # other day
    write_run(live, "dashboard-20261117-120000-v2.json", "v2", 4, 5)  # other caller count
    assert mod.main([str(live), "--readme", str(readme)]) == 0
    assert capsys.readouterr().out == ""


def test_newest_pair_wins_and_offline_runs_are_ignored(setup, capsys):
    live, readme = setup
    write_run(live, "dashboard-20261117-100000-v1.json", "v1", 0, 5)
    write_run(live, "dashboard-20261117-100500-v2.json", "v2", 4, 5)
    write_run(live, "dashboard-20261117-110000-v1.json", "v1", 4, 20)
    write_run(live, "dashboard-20261117-120000-v2.json", "v2", 14, 20)
    write_run(live, "dashboard-20261117-130000-v2.json", "v2", 19, 20, backend="offline")
    mod.main([str(live), "--readme", str(readme)])
    assert capsys.readouterr().out.strip() == "v1 20% to v2 70% on 20 callers, claude-cli, 17 Nov 2026"
