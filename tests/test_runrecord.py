"""Run record schema v1: every key present, saved by time and agent, loaded back unchanged. No network."""
import hashlib
import json
import re
import time
from pathlib import Path

import pytest

from callqa import runrecord
from callqa.personas import generate_personas
from callqa.report import build_report
from callqa.runrecord import RECORD_SCHEMA_VERSION, RecordError, load_record, record_name, save_record
from callqa.server import read_sample_sop
from evals import example_agent
from evals.backends import OfflineBackend
from evals.run_evals import RunResult, run_suite

FIXTURES = Path(__file__).resolve().parent / "fixtures"
NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.@-]+\.json$")  # C10

TOP_KEYS = {
    "schema_version", "kind", "created_at", "synthetic", "simulated", "note", "run_id", "sop_hash", "agent",
    "backend", "persona_set", "seed", "repeats", "callers", "max_turns", "rules_file", "warnings", "generator",
    "pass_rate", "passed", "total", "judge_split", "seconds", "consistency", "cost", "latency", "escalation",
    "compliance", "traits", "calls", "next_fixes",
}
FEATURE_BLOCKS = ("consistency", "compliance")  # cost and latency arrived with A3b, escalation with A4a
CALL_KEYS = {
    "persona_id", "name", "language", "tags", "passed", "reason", "failure_kind", "failing_turn", "sop_rule",
    "fix_hint", "notes", "judge_split", "transcript", "traits", "severity", "repeats_passed", "repeats_total",
    "flaky", "failed_repeat", "repeat_results", "escalation", "compliance", "usage", "turn_latency_ms",
}


@pytest.fixture(scope="module")
def record():
    sop = read_sample_sop()
    backend = OfflineBackend()
    generation = generate_personas(sop, 20, backend)
    results = run_suite(example_agent.respond, generation.personas, 1, 8, 4, backend, 0)
    return build_report(run_id="r1", sop_text=sop, agent={"id": "v1", "label": "v1 naive prompt"}, backend=backend,
                        callers=20, max_turns=8, generation=generation, results=results, seconds=0.5,
                        synthetic=True, metering=True)


def test_record_has_every_schema_v1_key(record):
    assert set(record) == TOP_KEYS
    assert record["schema_version"] == RECORD_SCHEMA_VERSION == 1 and record["kind"] == "run"
    assert record["created_at"][:4].isdigit() and record["created_at"][10] == "T"
    assert record["synthetic"] is True and record["simulated"] is True and record["note"].startswith("SYNTHETIC")
    assert record["agent"] == {"id": "v1", "label": "v1 naive prompt", "variant": "v1", "model": None}
    assert set(record["persona_set"]) == {"name", "count", "ids_sha256"} and record["persona_set"]["count"] == 20
    assert (record["seed"], record["repeats"], record["rules_file"]) == (0, 1, None)
    assert all(record[block] is None for block in FEATURE_BLOCKS)
    assert record["cost"]["label"] == record["latency"]["label"] == "simulated"
    assert (record["passed"], record["total"], record["pass_rate"]) == (14, 20, 0.7)
    for call in record["calls"]:
        assert set(call) == CALL_KEYS
        assert (call["repeat_results"], call["compliance"]) == ([], [])
        assert call["traits"] == (["hindi_numbers"] if call["persona_id"] == "lakh_question" else [])
        assert (call["repeats_total"], call["flaky"], call["failed_repeat"]) == (1, False, None)
        assert set(call["escalation"]) == {"expected", "reason", "escalated", "outcome", "handoff", "handoff_score",
                                           "handoff_missing"}
        assert set(call["usage"]) == {"caller", "agent", "judge"} and call["turn_latency_ms"]
        assert call["repeats_passed"] == int(call["passed"])
        assert call["severity"] == (None if call["passed"] else "major")


def test_save_names_record_by_time_and_agent_with_collision_suffix(record, tmp_path):
    record = {**record, "created_at": "2026-11-05T17:07:24+05:30", "agent": {**record["agent"], "id": "my@x/y"}}
    assert record_name(record) == "20261105-170724_my_x_y.json"
    first = save_record(record, tmp_path / "runs")
    second = save_record(record, tmp_path / "runs")
    third = save_record(record, tmp_path / "runs")
    assert [p.name for p in (first, second, third)] == [
        "20261105-170724_my_x_y.json", "20261105-170724_my_x_y-2.json", "20261105-170724_my_x_y-3.json"]


def test_load_rejects_newer_schema_and_wrong_kind(record, tmp_path):
    def write(name, data):
        path = tmp_path / name
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    with pytest.raises(RecordError, match="newer"):
        load_record(write("newer.json", {**record, "schema_version": 2}))
    with pytest.raises(RecordError, match="schema_version"):
        load_record(write("zero.json", {**record, "schema_version": 0}))
    with pytest.raises(RecordError, match="kind"):
        load_record(write("kind.json", {**record, "kind": "compare"}))
    with pytest.raises(RecordError, match="missing"):
        load_record(write("missing.json", {k: v for k, v in record.items() if k != "calls"}))
    with pytest.raises(RecordError):
        load_record(tmp_path / "absent.json")
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(RecordError):
        load_record(tmp_path / "bad.json")
    assert load_record(write("ok.json", record)) == record


def test_record_roundtrip_is_lossless_utf8(record, tmp_path):
    record = {**record, "note": "SYNTHETIC: अमित, ₹1,00,000 bhai 😀"}
    path = save_record(record, tmp_path)
    assert "₹1,00,000" in path.read_text(encoding="utf-8")  # written as UTF-8, not \u escapes
    assert load_record(path) == record
    assert time.strptime(path.name[:15], "%Y%m%d-%H%M%S")


@pytest.mark.parametrize("agent_id", [
    "evals.example_agent:respond", "v1 fixed", "a\\b", "अमित", "..", "x" * 300, "nul\x00byte", "../../etc/passwd"])
def test_any_agent_id_gives_a_safe_file_name_inside_the_folder(record, tmp_path, agent_id):
    record = {**record, "agent": {**record["agent"], "id": agent_id}}
    folder = tmp_path / "runs"
    path = save_record(record, folder)
    assert path.parent == folder and NAME_PATTERN.match(path.name), path.name
    assert [p.name for p in folder.iterdir()] == [path.name]  # no temp file left behind
    assert load_record(path) == record


@pytest.mark.parametrize("failing", ["fsync", "link"])
def test_failure_mid_write_leaves_no_file_under_the_final_name(record, tmp_path, monkeypatch, failing):
    def boom(*_args, **_kwargs):
        raise OSError("disk gone")

    monkeypatch.setattr(runrecord.os, failing, boom)
    with pytest.raises(OSError):
        save_record(record, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_half_written_file_is_a_record_error(record, tmp_path):
    path = tmp_path / "half.json"
    path.write_text(json.dumps(record, indent=2)[:400], encoding="utf-8")
    with pytest.raises(RecordError):
        load_record(path)


def test_judge_split_and_flaky_are_different_things(record):
    """`flaky` is about repeats (false at one repeat); a judge disagreement only sets `judge_split`."""
    sop = read_sample_sop()
    backend = OfflineBackend()
    generation = generate_personas(sop, 20, backend)
    persona = generation.personas[0]
    result = RunResult(persona["id"], 0, True, "ok", [{"role": "agent", "content": "hi"}], 0.1, judge_split=True)
    report = build_report(run_id="r", sop_text=sop, agent={"id": "v1", "label": "v1"}, backend=backend, callers=20,
                          max_turns=8, generation=type(generation)([persona], generation.source), results=[result],
                          seconds=0.1)
    call = report["calls"][0]
    assert call["judge_split"] is True and call["flaky"] is False
    assert report["judge_split"] == 1


def test_story_ids_hash_in_fixture_is_the_sha256_of_the_story_ids(record):
    ids = "\n".join(p["id"] for p in generate_personas(read_sample_sop(), 20, OfflineBackend()).personas)
    expected = hashlib.sha256(ids.encode("utf-8")).hexdigest()
    fixture = json.loads((FIXTURES / "report_example.json").read_text(encoding="utf-8"))
    assert record["persona_set"]["ids_sha256"] == fixture["persona_set"]["ids_sha256"] == expected
