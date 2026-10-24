"""The harness is the repo's verification loop, so it gets tests of its own."""
import json
import subprocess
import sys
from pathlib import Path

PRODUCT = Path(__file__).resolve().parents[1]
REPO = PRODUCT.parents[1] if PRODUCT.parent.name == "demos" else PRODUCT  # monorepo root, or the repo itself once exported
SCRIPT = PRODUCT / "evals" / "run_evals.py"


def run(*args: str, cwd: Path = REPO) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--agent", "evals.example_agent:respond", "--offline", *args],
        cwd=cwd, capture_output=True, text=True, timeout=60,
    )


def test_passing_personas_exit_zero():
    result = run("--runs", "1", "--only", "happy_path,changes_mind,disputes_amount")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OVERALL 100%" in result.stdout


def test_known_leak_is_caught_and_fails_the_run():
    result = run("--runs", "1", "--only", "adversarial_data_leak")
    assert result.returncode == 1
    assert "forbidden phrase" in result.stdout


def test_transcripts_are_saved():
    result = run("--runs", "1", "--only", "happy_path")
    saved = Path(REPO, result.stdout.strip().split("transcripts: ")[-1])
    records = json.loads(saved.read_text(encoding="utf-8"))
    assert records[0]["persona"] == "happy_path"
    assert records[0]["transcript"][0]["role"] == "user"


def test_runs_the_same_from_inside_the_product_folder():
    result = run("--runs", "1", "--only", "happy_path", cwd=PRODUCT)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OVERALL 100%" in result.stdout
