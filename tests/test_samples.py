"""N3 story pack: the committed sample records, the full set goldens, the README claims and the root CLAUDE.md guard.
Everything runs in process on the offline backend (no model call). Regeneration tests normalise only what they say."""
import json
import re
import subprocess
from pathlib import Path

import pytest
import yaml

from callqa import compare, diff, evaluate
from callqa.persona_sets import personas_for_set
from callqa.runrecord import load_record
from callqa.server import read_sample_sop

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
SAMPLES = ROOT / "samples"
RECORDS = SAMPLES / "run_records"
GOLDEN = Path(__file__).resolve().parent / "golden"
CREATED_AT = "2026-11-05T00:00:00+05:30"
AGENTS = ("v1", "v2", "v3")
FULL_FLAGS = ["--set", "full", "--repeats", "3", "--backend", "offline", "--created-at", CREATED_AT]
COMPARE_JSON = SAMPLES / "compare_full_offline.json"
COMPARE_MD = SAMPLES / "compare_full_offline.md"
PERSONAS_FULL = ROOT / "evals" / "personas_full.yaml"
FULL_HEADER = (
    "# SYNTHETIC test callers for the Acme Finance collections demo: the full set of 31 (the 20 story callers, "
    "then escalation, trait and extra callers). Made up; not real customers.\n"
    "# Generated in process from callqa.persona_sets.personas_for_set(sop, \"full\"); "
    "tests/test_samples.py checks this file against it.\n"
)
BASE_REF = "claude/qa-upgrades"


@pytest.fixture(scope="module")
def regenerated(tmp_path_factory):
    """Each agent's full set record, regenerated in process with the documented flags (module scoped: 3 runs)."""
    folder = tmp_path_factory.mktemp("regen")
    paths = {}
    for agent in AGENTS:
        out = folder / f"full_{agent}.json"
        assert evaluate.main(["--agent", agent, *FULL_FLAGS, "--out", str(out)]) == 0
        paths[agent] = out
    return paths


# ---- the hook ----

def test_created_at_makes_the_record_byte_identical(tmp_path):
    flags = ["--agent", "v1", "--callers", "5", "--backend", "offline", "--created-at", CREATED_AT]
    first, second = tmp_path / "a.json", tmp_path / "b.json"
    assert evaluate.main([*flags, "--out", str(first)]) == 0
    assert evaluate.main([*flags, "--out", str(second)]) == 0
    assert first.read_bytes() == second.read_bytes()
    record = json.loads(first.read_text(encoding="utf-8"))
    assert record["created_at"] == CREATED_AT and record["seconds"] == 0.0 and len(record["run_id"]) == 8


def test_without_created_at_the_run_id_still_differs(tmp_path):
    flags = ["--agent", "v1", "--callers", "5", "--backend", "offline"]
    evaluate.main([*flags, "--out", str(tmp_path / "a")])
    evaluate.main([*flags, "--out", str(tmp_path / "b")])
    files = [f for d in ("a", "b") for f in (tmp_path / d).glob("*.json")]
    ids = {json.loads(f.read_text(encoding="utf-8"))["run_id"] for f in files}
    assert len(ids) == 2


def test_run_id_changes_with_the_flags(tmp_path):
    ids = set()
    for agent in ("v1", "v2"):
        out = tmp_path / f"{agent}.json"
        evaluate.main(["--agent", agent, "--callers", "5", "--backend", "offline", "--created-at", CREATED_AT,
                       "--out", str(out)])
        ids.add(json.loads(out.read_text(encoding="utf-8"))["run_id"])
    assert len(ids) == 2


@pytest.mark.parametrize("bad", ["yesterday", "2026-11-05T00:00:00", "2026-13-45T00:00:00+05:30"])
def test_bad_created_at_is_rejected(bad, capsys):
    assert evaluate.main(["--agent", "v1", "--backend", "offline", "--no-save", "--created-at", bad]) == 2
    assert "--created-at" in capsys.readouterr().err


# ---- the committed sample records ----

@pytest.mark.parametrize("agent", AGENTS)
def test_sample_record_exists_loads_and_is_labelled(agent):
    path = RECORDS / f"full_{agent}.json"
    assert path.exists(), f"{path} is not committed yet"
    record = load_record(path)
    assert record["synthetic"] is True and record["simulated"] is True
    assert record["backend"]["name"] == "offline"
    assert record["created_at"] == CREATED_AT and record["seconds"] == 0.0
    assert record["persona_set"]["name"] == "full" and record["total"] == 31 and record["repeats"] == 3
    assert path.stat().st_size < 1_000_000


def test_sample_pass_rates_are_the_story():
    got = {a: load_record(RECORDS / f"full_{a}.json") for a in AGENTS}
    assert {a: (r["passed"], r["total"]) for a, r in got.items()} == {"v1": (17, 31), "v2": (22, 31), "v3": (30, 31)}


def test_committed_json_has_no_monorepo_paths():
    for path in [*RECORDS.glob("*.json"), COMPARE_JSON]:
        assert "demos/call-qa" not in path.read_text(encoding="utf-8"), path


def test_diff_reads_the_samples_and_gates(capsys):
    code = diff.main([str(RECORDS / "full_v1.json"), str(RECORDS / "full_v2.json"), "--gate"])
    out = capsys.readouterr().out
    assert code == 1 and "holder_says_bhai" in out and "GATE FAILED" in out
    code = diff.main([str(RECORDS / "full_v2.json"), str(RECORDS / "full_v3.json"), "--gate"])
    assert code == 0 and "GATE OK" in capsys.readouterr().out


@pytest.mark.parametrize("old,new", [("v1", "v2"), ("v2", "v3")])
def test_diff_output_matches_golden(old, new, capsys):
    diff.main([str(RECORDS / f"full_{old}.json"), str(RECORDS / f"full_{new}.json"), "--gate"])
    assert capsys.readouterr().out == (GOLDEN / f"diff_full_{old}_{new}.txt").read_text(encoding="utf-8")


def test_compare_samples_read_with_compare_code():
    record = json.loads(COMPARE_JSON.read_text(encoding="utf-8"))
    assert record["kind"] == "compare" and record["synthetic"] is True and record["simulated"] is True
    assert record["backend"]["name"] == "offline" and record["persona_set"]["name"] == "full"
    assert [r["config"] for r in record["configs"]] == list(AGENTS)
    assert compare.render_markdown(record) == COMPARE_MD.read_text(encoding="utf-8")
    assert record["recommendation"]["config"] == "v3"


def test_samples_say_simulated_and_synthetic():
    text = COMPARE_MD.read_text(encoding="utf-8")
    assert "SYNTHETIC" in text and "simulated" in text and "verify prices before quoting" in text


# ---- regeneration ----

@pytest.mark.parametrize("agent", AGENTS)
def test_sample_record_regenerates_byte_identical(agent, regenerated):
    committed = RECORDS / f"full_{agent}.json"
    assert committed.exists(), f"{committed} is not committed yet"
    assert regenerated[agent].read_bytes() == committed.read_bytes()


@pytest.fixture(scope="module")
def compare_regen(tmp_path_factory):
    folder = tmp_path_factory.mktemp("cmp")
    out, md = folder / "compare_full_offline.json", folder / "compare_full_offline.md"
    assert compare.main(["--set", "full", "--backend", "offline", "--out", str(out), "--export-md", str(md)]) == 0
    return out, md


def normalised_compare(path):
    """created_at and the run file names come from the clock inside callqa/compare.py and report.py, so they are the
    only things cleared; every number, config and sentence is compared."""
    record = json.loads(path.read_text(encoding="utf-8"))
    record["created_at"], record["run_files"] = None, None
    for row in record["configs"]:
        row["run_file"] = None
    return record


def test_compare_md_regenerates_byte_identical(compare_regen):
    assert COMPARE_MD.exists(), f"{COMPARE_MD} is not committed yet"
    assert compare_regen[1].read_bytes() == COMPARE_MD.read_bytes()


def test_compare_json_regenerates_apart_from_clock_fields(compare_regen):
    assert COMPARE_JSON.exists(), f"{COMPARE_JSON} is not committed yet"
    assert normalised_compare(compare_regen[0]) == normalised_compare(COMPARE_JSON)


def test_compare_stdout_matches_golden(capsys, tmp_path):
    assert compare.main(["--set", "full", "--backend", "offline", "--no-save"]) == 0
    assert capsys.readouterr().out == (GOLDEN / "compare_full_offline.txt").read_text(encoding="utf-8")


# ---- the full set goldens ----

@pytest.mark.parametrize("agent", AGENTS)
def test_full_eval_output_matches_golden(agent, capsys):
    code = evaluate.main(["--agent", agent, "--set", "full", "--repeats", "3", "--backend", "offline", "--no-save"])
    assert code == 0
    assert capsys.readouterr().out == (GOLDEN / f"eval_full_{agent}.txt").read_text(encoding="utf-8")


def test_full_golden_cost_lines_say_simulated_and_dated():
    for agent in AGENTS:
        lines = (GOLDEN / f"eval_full_{agent}.txt").read_text(encoding="utf-8").splitlines()
        assert lines[3].startswith("tokens (simulated)")
        assert lines[4].startswith("cost (simulated)")
        assert lines[4].endswith("prices of 2026-11-06, verify before quoting")


# ---- the callers file ----

def test_personas_full_yaml_matches_the_generator():
    personas = personas_for_set(read_sample_sop(), "full").personas
    expected = FULL_HEADER + yaml.safe_dump(personas, sort_keys=False, allow_unicode=True, width=100)
    assert PERSONAS_FULL.read_text(encoding="utf-8") == expected
    assert len(yaml.safe_load(expected)) == 31


# ---- the README ----

README = (ROOT / "README.md").read_text(encoding="utf-8")
UNBUILT = ("escalation panel", "traits panel", "compliance panel", "bake off table", "label page", "judge badge",
           "records compare view")


def test_readme_claims_no_ui_that_does_not_exist():
    lowered = README.lower()
    for word in ("panel", "badge", "label page", *UNBUILT):
        assert word not in lowered, f"README mentions {word!r}, which is not in callqa/static"


def test_readme_counts_are_true():
    assert "22 callers" not in README and "21 callers" not in README
    assert "31" in README and "20 callers" in README
    for needle in ("SMOKE OK v1 70% (14/20) v2 95% (19/20)", "Run it in 60 seconds", "17/31", "22/31", "30/31"):
        assert needle in README, needle


def test_readme_documents_every_command_and_flag():
    for needle in ("python -m callqa eval", "python -m callqa diff", "python -m callqa compare",
                   "python -m callqa calibrate", "--repeats", "--set", "--rules", "--created-at", "--gate"):
        assert needle in README, needle


def test_readme_honesty_notes():
    for needle in ("synthetic", "simulated", "2026-11-06", "verify before quoting", "not legal advice", "UNVERIFIED"):
        assert needle in README, needle
    assert "three" in README and "two" in README


def test_readme_has_a_quota_table():
    assert re.search(r"^\| command .*quota", README, re.MULTILINE | re.IGNORECASE)
    assert "claude-cli" in README and "offline" in README


def test_demo_claude_md_lists_the_commands():
    text = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    for needle in ("callqa eval", "callqa diff", "callqa compare", "callqa calibrate", "--created-at"):
        assert needle in text, needle
    assert "22 callers" not in text


# ---- files that must not move ----

def git_diff_names(*paths):
    result = subprocess.run(["git", "diff", "--name-only", BASE_REF, "--", *paths], cwd=REPO,
                            capture_output=True, text=True)
    if result.returncode != 0:
        pytest.skip(f"git or {BASE_REF} is not available here")
    return result.stdout.split()


def test_root_claude_md_is_unchanged():
    assert git_diff_names("CLAUDE.md") == []


def test_three_story_goldens_are_unchanged():
    names = [f"demos/call-qa/tests/golden/eval_story_{v}.txt" for v in AGENTS]
    assert git_diff_names(*names) == []


# ---- the README check report ----

CHECK_REPORT = REPO / "reports" / "readme-check.txt"


def test_readme_check_report_says_the_clean_clone_ran_only_the_headless_checks():
    text = CHECK_REPORT.read_text(encoding="utf-8")
    assert "starts the" in text and "dashboard" in text and "headless equivalent" in text
    assert "SMOKE OK v1 70% (14/20) v2 95% (19/20)" in text and "CHECK OK backend=offline" in text


def test_every_callqa_command_in_the_readme_is_in_the_check_report():
    report = CHECK_REPORT.read_text(encoding="utf-8")
    commands = re.findall(r"^python -m callqa (?:eval|diff|compare|calibrate) .*$|^python -m callqa calibrate$",
                          README, re.MULTILINE)
    assert len(commands) >= 8
    for command in commands:
        assert f"$ {command}" in report, command


def test_readme_blocks_are_copied_from_the_goldens():
    for agent in AGENTS:
        golden = (GOLDEN / f"eval_full_{agent}.txt").read_text(encoding="utf-8").rstrip("\n")
        assert golden in README, agent
    assert (GOLDEN / "compare_full_offline.txt").read_text(encoding="utf-8").rstrip("\n") in README


def test_readme_story_table_matches_the_sample_records():
    for agent in AGENTS:
        record = load_record(RECORDS / f"full_{agent}.json")
        cell = f"{record['passed'] / record['total']:.0%} ({record['passed']}/{record['total']})"
        rows = [line for line in README.splitlines() if line.startswith(f"| {agent} |")]
        assert len(rows) == 1 and rows[0].endswith(f"| {cell} |"), (agent, cell, rows)


# ---- fix round 1: the README says only what the code does ----

def test_readme_says_only_the_three_run_records_use_created_at():
    assert "Only the three run records" in README and "`compare` has no such flag" in README
    assert "clock named run file names" in README
    assert compare.main(["--set", "full", "--backend", "offline", "--no-save", "--created-at", CREATED_AT]) == 2


def test_readme_says_diff_and_calibrate_take_no_backend_flag():
    from callqa import calibrate
    assert "`diff` and `calibrate` take no backend flag" in README
    assert "when you pass `--backend offline`" not in README
    for module, args in ((diff, ["a.json", "b.json"]), (calibrate, [])):
        with pytest.raises(SystemExit):
            module.main([*args, "--backend", "offline"])
    claude_md = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    assert "always add `--backend offline`" not in claude_md and "take no backend flag" in claude_md


def test_readme_has_no_opinion_about_real_models_and_says_where_the_bug_is_planted():
    assert "rarely" not in README
    assert "planted in the offline v2 agent" in README and "pinned offline so that it is reproducible" in README


def test_readme_names_all_eleven_extra_callers():
    from callqa.personas import generate_personas
    story = {p["id"] for p in generate_personas(read_sample_sop(), 20, None).personas}
    full = [p["id"] for p in yaml.safe_load(PERSONAS_FULL.read_text(encoding="utf-8"))]
    extras = [pid for pid in full if pid not in story]
    assert len(extras) == 11 and "11 more" in README
    for pid in extras:
        assert f"`{pid}`" in README, pid


def test_readme_v2_and_v3_claims_are_in_the_code():
    from evals.example_agent_fixed import respond as v2
    from evals.example_agent_v3 import respond as v3
    holder = {"role": "user", "content": "Haan, Ravi bol raha hoon."}
    dispute = {"role": "user", "content": "This is not my loan. Someone used my documents."}
    saadhe = {"role": "user", "content": "Saadhe teen hazaar aaj de sakta hoon."}
    agent = {"role": "assistant", "content": "ok"}
    v2_note = v2([holder, agent, dispute])
    assert "<handoff>" in v2_note and '"tried"' not in v2_note
    assert "Rs 3,000" in v2([holder, agent, saadhe])
    assert '"tried"' in v3([holder, agent, dispute])
    assert "3,500" in v3([holder, agent, saadhe])
    for needle in ("4 of 5", "5 of 5", "Rs 3,000", "3,500", "over eager", "mid call", "no `tried` field"):
        assert needle in README, needle


def test_check_report_has_the_run_evals_commands_with_exit_codes():
    report = CHECK_REPORT.read_text(encoding="utf-8")
    commands = re.findall(r"^python demos/call-qa/evals/run_evals\.py .* --offline .*$", README, re.MULTILINE)
    assert len(commands) == 2  # the your_package placeholder command is not runnable, so it is not listed
    for command in commands:
        block = re.search(rf"^\$ {re.escape(command)}\n(.*?)\nexit code: (\d+)$", report, re.MULTILINE | re.DOTALL)
        assert block, command
        assert "OVERALL " in block.group(1) and block.group(2) == "1", command  # exit 1 is the planted failure
    first = re.search(r"^\$ \S.*--runs 2\n(.*?)\nexit code:", report, re.MULTILINE | re.DOTALL)
    assert "OVERALL 75% over 8 conversations (threshold 80%)" in first.group(1)
