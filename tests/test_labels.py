"""The label store: validate_label (schema C9) and append_jsonl. Temp paths only, never the real labels file."""
import copy
import hashlib
import json
import subprocess
import threading
from pathlib import Path

import pytest

from callqa import labels
from callqa.labels import LabelError, append_jsonl, validate_label

DEMO = Path(__file__).resolve().parents[1]
FIXTURE = DEMO / "tests" / "fixtures" / "report_full_v2_r3.json"
RECORD = json.loads(FIXTURE.read_text(encoding="utf-8"))
NAME = "20261106-100000_v2.json"
FAILING = next(c for c in RECORD["calls"] if c["failing_turn"] is not None)


def body(**changes):
    base = {"record": NAME, "persona_id": FAILING["persona_id"], "repeat": 0, "human_pass": False,
            "human_failing_turn": 1, "human_reason": "never offered a callback", "labeller": "bob"}
    base.update(changes)
    return base


def test_valid_label_copies_judge_fields_from_the_record():
    label = validate_label(body(), RECORD, NAME)
    repeat0 = FAILING["repeat_results"][0]
    assert label["schema_version"] == 1 and label["kind"] == "label" and len(label["label_id"]) == 8
    assert label["record"] == NAME and label["persona_id"] == FAILING["persona_id"] and label["repeat"] == 0
    assert label["human_pass"] is False and label["human_failing_turn"] == 1 and label["labeller"] == "bob"
    assert label["judge_pass"] is repeat0["passed"] and label["judge_failing_turn"] == repeat0["failing_turn"]
    assert label["judge_reason"] == repeat0["reason"]
    assert label["judge_source"] == "rules" and label["on_simulated_run"] is True and label["synthetic"] is True
    digest = hashlib.sha256(json.dumps(repeat0["transcript"], ensure_ascii=False).encode("utf-8")).hexdigest()
    assert label["transcript_sha256"] == digest and label["run_id"] == RECORD["run_id"]


def test_client_cannot_forge_judge_fields():
    for key in ("judge_pass", "judge_reason", "synthetic", "judge_source", "label_id", "run_id"):
        with pytest.raises(LabelError):
            validate_label(body(**{key: "x"}), RECORD, NAME)


def test_judge_verdict_is_the_records_even_when_the_human_disagrees():
    label = validate_label(body(human_pass=True, human_failing_turn=None), RECORD, NAME)
    assert label["human_pass"] is True and label["judge_pass"] is False and label["judge_failing_turn"] == 3


def test_llm_backend_gives_llm_judge_source():
    record = copy.deepcopy(RECORD)
    record["backend"] = {"name": "claude-cli", "models": {}}
    record["simulated"] = False
    label = validate_label(body(persona_id="happy_friday", human_failing_turn=None), record, NAME)
    assert label["judge_source"] == "llm" and label["on_simulated_run"] is False


@pytest.mark.parametrize("value", ["yes", 1, 0, None, "false"])
def test_bad_pass_value_is_rejected(value):
    with pytest.raises(LabelError, match="human_pass"):
        validate_label(body(human_pass=value), RECORD, NAME)


def test_unknown_keys_wrong_types_and_missing_keys_rejected():
    for bad in (body(extra=1), body(repeat="0"), body(repeat=True), body(human_reason=5), body(labeller=None),
                body(human_failing_turn="1"), body(human_failing_turn=True)):
        with pytest.raises(LabelError):
            validate_label(bad, RECORD, NAME)
    missing = body()
    del missing["labeller"]
    with pytest.raises(LabelError):
        validate_label(missing, RECORD, NAME)


def test_failing_turn_must_be_an_agent_turn():
    assert [t["role"] for t in FAILING["transcript"]][:2] == ["user", "assistant"]
    for turn in (0, 2, 99, -1):  # a caller turn, another caller turn, out of range, negative
        with pytest.raises(LabelError, match="failing turn"):
            validate_label(body(human_failing_turn=turn), RECORD, NAME)
    assert validate_label(body(human_failing_turn=3), RECORD, NAME)["human_failing_turn"] == 3


def test_a_pass_label_has_no_failing_turn():
    with pytest.raises(LabelError, match="failing turn"):
        validate_label(body(human_pass=True, human_failing_turn=1), RECORD, NAME)
    label = validate_label(body(human_pass=True, human_failing_turn=None), RECORD, NAME)
    assert label["human_pass"] is True and label["human_failing_turn"] is None


def test_reason_length_boundary():
    assert validate_label(body(human_reason="x" * 500), RECORD, NAME)
    with pytest.raises(LabelError, match="500"):
        validate_label(body(human_reason="x" * 501), RECORD, NAME)


def test_labeller_length_boundary_and_blank():
    assert validate_label(body(labeller="n" * 40), RECORD, NAME)
    with pytest.raises(LabelError, match="40"):
        validate_label(body(labeller="n" * 41), RECORD, NAME)
    with pytest.raises(LabelError):
        validate_label(body(labeller="   "), RECORD, NAME)


def test_unknown_persona_or_repeat_is_rejected():
    with pytest.raises(LabelError, match="persona"):
        validate_label(body(persona_id="nobody"), RECORD, NAME)
    for repeat in (3, -1):
        with pytest.raises(LabelError, match="repeat"):
            validate_label(body(repeat=repeat), RECORD, NAME)


def test_append_writes_one_line_and_creates_the_folder(tmp_path):
    path = tmp_path / "labels" / "human_labels.jsonl"
    append_jsonl(path, {"a": 1, "text": "नमस्ते"})
    append_jsonl(path, {"a": 2})
    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["a"] for line in lines] == [1, 2] and "नमस्ते" in lines[0]


def test_line_over_2kb_is_refused_and_nothing_written(tmp_path):
    path = tmp_path / "l.jsonl"
    with pytest.raises(LabelError, match="2 KB"):
        append_jsonl(path, {"x": "y" * 2048})
    assert not path.exists() or path.read_bytes() == b""
    append_jsonl(path, {"x": "y" * 2000})  # a line under the cap still goes in
    assert path.read_bytes().endswith(b"\n")


def test_file_at_5mb_refuses_the_append(tmp_path):
    path = tmp_path / "l.jsonl"
    path.write_bytes(b"x" * labels.MAX_FILE_BYTES)
    with pytest.raises(LabelError, match="5 MB") as info:
        append_jsonl(path, {"a": 1})
    assert info.value.status == 409
    assert path.stat().st_size == labels.MAX_FILE_BYTES


def test_eight_threads_leave_eight_intact_lines(tmp_path):
    path = tmp_path / "l.jsonl"
    barrier = threading.Barrier(8)

    def work(n):
        barrier.wait()
        append_jsonl(path, {"n": n, "pad": "p" * 1500})

    threads = [threading.Thread(target=work, args=(n,)) for n in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert sorted(r["n"] for r in rows) == list(range(8)) and all(len(r["pad"]) == 1500 for r in rows)


def record_with(persona="flaky_line", **call_changes):
    """A deep copy of the fixture with changes applied to one call."""
    record = copy.deepcopy(RECORD)
    call = next(c for c in record["calls"] if c["persona_id"] == persona)
    call.update(call_changes)
    return record, call


def turns(*roles):
    return [{"role": r, "content": f"turn {i}"} for i, r in enumerate(roles)]


def test_label_uses_the_chosen_repeat_not_the_calls_first_failing_one():
    # flaky_line passes repeat 0 and fails repeat 2; the call itself describes repeat 2
    label = validate_label(body(persona_id="flaky_line", human_pass=True, human_failing_turn=None), RECORD, NAME)
    assert label["judge_pass"] is True and label["judge_failing_turn"] is None
    label2 = validate_label(body(persona_id="flaky_line", repeat=2), RECORD, NAME)
    assert label2["judge_pass"] is False and label2["judge_failing_turn"] == 3


def test_failing_turn_is_checked_against_the_chosen_repeats_transcript():
    record, call = record_with()
    call["repeat_results"][0]["transcript"] = turns("user", "assistant", "user", "assistant", "user", "assistant")
    call["repeat_results"][2]["transcript"] = turns("user", "assistant", "user", "assistant")
    call["transcript"] = call["repeat_results"][2]["transcript"]
    assert validate_label(body(persona_id="flaky_line", repeat=0, human_failing_turn=5), record, NAME)
    with pytest.raises(LabelError, match="failing turn"):
        validate_label(body(persona_id="flaky_line", repeat=2, human_failing_turn=5), record, NAME)


def test_record_field_must_match_the_record_name():
    with pytest.raises(LabelError, match="record"):
        validate_label(body(record="other.json"), RECORD, NAME)


def test_a_padded_labeller_name_is_stored_trimmed():
    assert validate_label(body(labeller="  bob  "), RECORD, NAME)["labeller"] == "bob"


def test_record_without_repeat_results_only_has_repeat_zero():
    record, _ = record_with(repeat_results=[])
    assert validate_label(body(persona_id="flaky_line"), record, NAME)["repeat"] == 0
    with pytest.raises(LabelError, match="repeat"):
        validate_label(body(persona_id="flaky_line", repeat=1), record, NAME)


def test_lone_surrogate_is_a_label_error_not_a_crash(tmp_path):
    for changes in ({"human_reason": "\ud800"}, {"labeller": "\ud800"}):
        with pytest.raises(LabelError, match="cannot be saved"):
            validate_label(body(**changes), RECORD, NAME)
    with pytest.raises(LabelError, match="cannot be saved"):
        append_jsonl(tmp_path / "l.jsonl", {"a": "\ud800"})


WORST = "नमस्ते" * 100  # Devanagari, 3 bytes a character


def test_a_valid_human_label_always_fits_in_one_line_even_with_worst_case_judge_fields(tmp_path):
    record, call = record_with(reason=WORST, run_id="r" * 400)
    call["repeat_results"][0]["reason"] = WORST
    long_name = "n" * 105 + ".json"
    label = validate_label(body(record=long_name, persona_id="flaky_line", human_reason=WORST[:300],
                                labeller=WORST[:20], human_failing_turn=None, human_pass=True), record, long_name)
    assert len(labels.dumps_line(label).encode("utf-8")) + 1 <= labels.MAX_LINE_BYTES
    assert len(label["judge_reason"].encode("utf-8")) <= labels.MAX_JUDGE_REASON_BYTES
    append_jsonl(tmp_path / "l.jsonl", label)


def test_500_ascii_characters_fit_and_500_devanagari_say_which_field_to_shorten():
    assert validate_label(body(human_reason="x" * 500), RECORD, NAME)
    with pytest.raises(LabelError, match="shorten human_reason"):
        validate_label(body(human_reason=WORST[:500]), RECORD, NAME)
    quotes = '"' * 450  # each quote is two bytes once escaped
    assert validate_label(body(human_reason=quotes), RECORD, NAME)
    with pytest.raises(LabelError, match="shorten human_reason"):
        validate_label(body(human_reason='"' * 451), RECORD, NAME)


def test_labeller_byte_limit_names_the_field():
    with pytest.raises(LabelError, match="shorten labeller"):
        validate_label(body(labeller=WORST[:30]), RECORD, NAME)


def test_overlong_persona_id_or_record_name_is_refused():
    with pytest.raises(LabelError, match="too long"):
        validate_label(body(record="n" * 120 + ".json"), RECORD, "n" * 120 + ".json")


def test_a_crashed_call_has_no_verdict_to_label():
    record, _ = record_with(persona="happy_friday", passed=False, failure_kind="crash", repeat_results=[])
    with pytest.raises(LabelError, match="crashed"):
        validate_label(body(persona_id="happy_friday", human_failing_turn=None), record, NAME)
    record, call = record_with(persona="happy_friday")
    call["repeat_results"][1].update(passed=False, failure_kind="crash")
    with pytest.raises(LabelError, match="crashed"):
        validate_label(body(persona_id="happy_friday", repeat=1, human_failing_turn=None), record, NAME)
    assert validate_label(body(persona_id="happy_friday", repeat=0, human_failing_turn=None), record, NAME)


@pytest.mark.parametrize("passed,kind,backend,source", [
    (True, None, "claude-cli", "llm"), (False, "judge", "claude-cli", "llm"),
    (False, "forbidden", "claude-cli", "rules"), (False, "missing", "claude-cli", "rules"),
    (False, "misheard", "claude-cli", "rules"), (False, "compliance", "claude-cli", "rules"),
    (False, "judge", "offline", "rules"), (True, None, "offline", "rules"),
])
def test_judge_source_is_llm_only_for_a_model_verdict(passed, kind, backend, source):
    record, call = record_with(persona="happy_friday", repeat_results=[], passed=passed, failure_kind=kind)
    record["backend"] = {"name": backend, "models": {}}
    label = validate_label(body(persona_id="happy_friday", human_failing_turn=None), record, NAME)
    assert label["judge_source"] == source and label["judge_failure_kind"] == kind


def test_judge_reason_is_cut_to_a_fixed_length():
    record, _ = record_with(persona="happy_friday", repeat_results=[], reason="r" * 1000)
    label = validate_label(body(persona_id="happy_friday", human_failing_turn=None), record, NAME)
    assert label["judge_reason"] == "r" * labels.MAX_JUDGE_REASON_BYTES


@pytest.mark.parametrize("change", [{"calls": None}, {"calls": [None, "x", 3]}])
def test_damaged_calls_are_a_label_error(change):
    record = copy.deepcopy(RECORD)
    record.update(change)
    with pytest.raises(LabelError):
        validate_label(body(), record, NAME)


def test_damaged_repeat_or_transcript_entries_are_a_label_error():
    record, call = record_with()
    call["repeat_results"][0] = "junk"
    with pytest.raises(LabelError, match="damaged"):
        validate_label(body(persona_id="flaky_line"), record, NAME)
    record, call = record_with()
    call["repeat_results"][0]["transcript"] = [None, None]
    with pytest.raises(LabelError, match="failing turn"):
        validate_label(body(persona_id="flaky_line", human_failing_turn=1), record, NAME)


def test_a_line_of_exactly_2048_bytes_is_written_and_2049_is_not(tmp_path):
    path = tmp_path / "l.jsonl"
    append_jsonl(path, {"x": "y" * 2038})  # {"x": "..."} is 9 characters of frame, plus the newline
    assert path.stat().st_size == 2048
    with pytest.raises(LabelError, match="2 KB"):
        append_jsonl(path, {"x": "y" * 2039})
    assert path.stat().st_size == 2048


def test_a_file_one_line_below_the_cap_takes_exactly_one_more(tmp_path):
    path = tmp_path / "l.jsonl"
    line_size = len(labels.dumps_line({"a": 1}).encode("utf-8")) + 1
    path.write_bytes(b"x" * (labels.MAX_FILE_BYTES - line_size - 1) + b"\n")
    append_jsonl(path, {"a": 1})
    assert path.stat().st_size == labels.MAX_FILE_BYTES
    with pytest.raises(LabelError, match="5 MB"):
        append_jsonl(path, {"a": 1})
    path.write_bytes(b"x" * (labels.MAX_FILE_BYTES - line_size) + b"\n")  # one byte too full for another line
    with pytest.raises(LabelError, match="5 MB"):
        append_jsonl(path, {"a": 1})


def test_line_separator_characters_never_appear_raw(tmp_path):
    path = tmp_path / "l.jsonl"
    nasty = "a b c\u0085d\x0be\x0cf\x1cg"
    for n in range(3):
        append_jsonl(path, {"n": n, "text": nasty})
    text = path.read_text(encoding="utf-8")
    assert len(text.splitlines()) == len(text.split("\n")) - 1 == 3
    assert json.loads(text.splitlines()[0])["text"] == nasty


def test_a_torn_last_line_cannot_merge_with_the_next_one(tmp_path):
    path = tmp_path / "l.jsonl"
    path.write_bytes(b'{"torn": ')
    append_jsonl(path, {"ok": 1})
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines == ['{"torn": ', '{"ok": 1}']


def test_labels_folder_jsonl_is_git_ignored():
    result = subprocess.run(["git", "check-ignore", "-q", "demos/call-qa/evals/labels/human_labels.jsonl"],
                            cwd=DEMO.parents[1], capture_output=True)
    assert result.returncode == 0
