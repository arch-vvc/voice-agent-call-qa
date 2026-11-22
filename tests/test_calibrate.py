"""Calibration: how often the judge agrees with a human. Labels here are built in tmp_path; the fixture is synthetic."""
import http.client
import json
import threading
from pathlib import Path

import pytest

from callqa import calibrate
from callqa.__main__ import main as cli_main
from callqa.server import make_server
from evals.backends import OfflineBackend

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "labels_example.jsonl"
COUNTER = [0]


def label(human, judge, source="llm", hturn=None, jturn=None, synthetic=False):
    COUNTER[0] += 1
    n = COUNTER[0]
    return {"kind": "label", "label_id": f"t{n}", "persona_id": f"p{n}", "human_pass": human,
            "human_failing_turn": hturn, "human_reason": f"human {n}", "judge_pass": judge,
            "judge_failing_turn": jturn, "judge_reason": f"judge {n}", "judge_source": source,
            "synthetic": synthetic, "on_simulated_run": False}


def cells(tt, tf, ft, ff, source="llm"):
    """Labels for a 2x2 table: first letter human pass, second letter judge pass (T pass, F fail)."""
    return ([label(True, True, source) for _ in range(tt)] + [label(True, False, source) for _ in range(tf)]
            + [label(False, True, source) for _ in range(ft)] + [label(False, False, source) for _ in range(ff)])


def write(path, labels, extra=""):
    path.write_text("".join(json.dumps(x) + "\n" for x in labels) + extra, encoding="utf-8")
    return path


def llm(result):
    return result["sources"]["llm"]


def test_perfect_agreement():
    r = calibrate.calibrate([label(True, True), label(False, False), label(True, True), label(False, False)])
    s = llm(r)
    assert s["n"] == 4 and s["agreement"] == 1.0 and s["kappa"] == "1.00"
    assert s["matrix"] == {"human_pass": {"judge_pass": 2, "judge_fail": 0},
                           "human_fail": {"judge_pass": 0, "judge_fail": 2}}


def test_total_disagreement():
    r = calibrate.calibrate([label(True, False), label(False, True), label(True, False), label(False, True)])
    s = llm(r)
    assert s["agreement"] == 0.0 and s["kappa"] == "-1.00"
    assert s["matrix"]["human_pass"] == {"judge_pass": 0, "judge_fail": 2}
    assert s["matrix"]["human_fail"] == {"judge_pass": 2, "judge_fail": 0}
    assert len(s["disagreements"]) == 4


def test_matrix_cells_are_not_mixed_up():
    r = calibrate.calibrate(cells(4, 3, 2, 1))
    assert llm(r)["matrix"] == {"human_pass": {"judge_pass": 4, "judge_fail": 3},
                                "human_fail": {"judge_pass": 2, "judge_fail": 1}}


def test_expected_agreement_of_one_is_na_without_error():
    r = calibrate.calibrate([label(True, True), label(True, True), label(True, True)])
    assert llm(r)["kappa"] == "n/a" and llm(r)["agreement"] == 1.0
    assert "n/a" in calibrate.format_report(r)


def test_empty_and_missing_file_say_zero_labels(tmp_path, capsys):
    (tmp_path / "empty.jsonl").write_text("", encoding="utf-8")
    for name in ("empty.jsonl", "missing.jsonl"):
        assert calibrate.main(["--labels", str(tmp_path / name)]) == 0
        assert "0 labels" in (out := capsys.readouterr().out) and "cannot read" not in out


def test_synthetic_excluded_unless_included(tmp_path, capsys):
    path = write(tmp_path / "l.jsonl", [label(True, True), label(False, False, synthetic=True),
                                        label(True, False, synthetic=True)])
    labels = calibrate.load_labels(path)[0]
    assert llm(calibrate.calibrate(labels))["n"] == 1
    assert llm(calibrate.calibrate(labels, include_synthetic=True))["n"] == 3
    calibrate.main(["--labels", str(path)])
    assert "SYNTHETIC" not in capsys.readouterr().out
    calibrate.main(["--labels", str(path), "--include-synthetic"])
    assert "SYNTHETIC" in capsys.readouterr().out


def test_llm_and_rules_are_separate():
    r = calibrate.calibrate([label(True, True, "llm"), label(True, False, "rules"), label(False, False, "rules")])
    assert llm(r)["n"] == 1 and llm(r)["agreement"] == 1.0
    assert r["sources"]["rules"]["n"] == 2 and r["sources"]["rules"]["agreement"] == 0.5
    text = calibrate.format_report(r)
    assert "LLM judge" in text and "Rule judge" in text


def test_failing_turn_agreement_only_counts_calls_both_failed():
    r = calibrate.calibrate([
        label(False, False, hturn=2, jturn=2),     # counts, agrees
        label(False, False, hturn=1, jturn=3),     # counts, differs
        label(False, False, hturn=None, jturn=3),  # human gave no turn: not counted
        label(False, True, hturn=1, jturn=1),      # judge passed: not counted
        label(True, False, jturn=1),               # human passed: not counted
    ])
    assert llm(r)["failing_turn"] == {"n": 2, "agree": 1}


def test_disagreements_show_both_reasons():
    r = calibrate.calibrate([label(True, False)])
    d = llm(r)["disagreements"][0]
    assert d["human_reason"].startswith("human") and d["judge_reason"].startswith("judge")
    text = calibrate.format_report(r)
    assert d["human_reason"] in text and d["judge_reason"] in text


def test_malformed_lines_reported_and_skipped(tmp_path, capsys):
    bad_type = '{"kind":"label","human_pass":"yes","judge_pass":true,"judge_source":"llm"}\n'
    path = write(tmp_path / "l.jsonl", [label(True, True)], extra="{not json\n" + bad_type + "[1]\n")
    assert calibrate.main(["--labels", str(path)]) == 0
    out = capsys.readouterr().out
    assert "skipped 3" in out and "line 2" in out and "line 3" in out and "line 4" in out
    assert "LLM judge: 1 label" in out


def test_fixture_is_marked_synthetic_and_never_the_default(capsys):
    first = json.loads(FIXTURE.read_text(encoding="utf-8").splitlines()[0])
    assert first["synthetic"] is True and "SYNTHETIC" in first["note"]
    default = calibrate.DEFAULT_LABELS
    assert default.name == "human_labels.jsonl" and "fixtures" not in str(default)
    labels, problems = calibrate.load_labels(FIXTURE)
    assert not problems and labels and all(x["synthetic"] for x in labels)
    assert calibrate.main(["--labels", str(FIXTURE)]) == 0
    assert "0 labels" in capsys.readouterr().out
    assert calibrate.main(["--labels", str(FIXTURE), "--include-synthetic"]) == 0
    assert "SYNTHETIC" in capsys.readouterr().out


def test_cli_dispatch_and_help(capsys):
    with pytest.raises(SystemExit) as exc:
        cli_main(["calibrate", "--help"])
    assert exc.value.code == 0
    assert "--include-synthetic" in capsys.readouterr().out


def test_summary_counts_real_llm_labels_only(tmp_path):
    path = write(tmp_path / "l.jsonl", [label(True, True, "llm"), label(True, True, "rules"),
                                        label(True, True, "llm", synthetic=True)])
    s = calibrate.summary(path)
    assert s["real_llm_labels"] == 1 and s["badge_threshold"] == 20 and s["badge_ready"] is False
    assert "20" in s["message"]


def serve(tmp_path, labels_path):
    s = make_server(OfflineBackend(), 20, 8, 2, [{"id": "v1", "label": "v1"}], results_dir=tmp_path / "runs",
                    samples_dir=tmp_path / "none", labels_path=labels_path)
    threading.Thread(target=s.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
    return s


def get(s, host=None):
    port = s.server_address[1]
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.putrequest("GET", "/api/labels/summary", skip_host=True)
    conn.putheader("Host", host or f"127.0.0.1:{port}")
    conn.endheaders()
    resp = conn.getresponse()
    out = resp.status, json.loads(resp.read().decode("utf-8"))
    conn.close()
    return out


@pytest.fixture
def srv(tmp_path):
    s = serve(tmp_path, write(tmp_path / "labels.jsonl", [label(True, True, "llm"), label(False, True, "llm")]))
    yield s
    s.shutdown()
    s.server_close()


def test_summary_endpoint(srv):
    status, body = get(srv)
    assert status == 200 and body["real_llm_labels"] == 2 and body["badge_threshold"] == 20


def test_summary_endpoint_bad_host(srv):
    assert get(srv, host="evil.example")[0] == 403


def test_summary_endpoint_with_no_file(tmp_path):
    s = serve(tmp_path, tmp_path / "nope.jsonl")
    try:
        status, body = get(s)
        assert status == 200 and body["real_llm_labels"] == 0
    finally:
        s.shutdown()
        s.server_close()


def report(labels, include_synthetic=False):
    return calibrate.format_report(calibrate.calibrate(labels, include_synthetic))


def same_call(human, judge, record="r.json", persona="p", repeat=0, **extra):
    x = label(human, judge)
    x.update({"record": record, "persona_id": persona, "repeat": repeat}, **extra)
    return x


def test_latest_label_wins(tmp_path):
    path = write(tmp_path / "l.jsonl", [same_call(True, True) for _ in range(20)])
    assert calibrate.summary(path)["real_llm_labels"] == 1
    assert llm(calibrate.calibrate(calibrate.load_labels(path)[0]))["n"] == 1


def test_later_label_replaces_earlier_with_different_verdict():
    r = calibrate.calibrate([same_call(True, False), same_call(False, False)])
    assert llm(r)["n"] == 1 and llm(r)["agreement"] == 1.0 and llm(r)["disagreements"] == []
    r = calibrate.calibrate([same_call(False, False), same_call(True, False)])
    assert llm(r)["agreement"] == 0.0


def test_repeats_and_records_are_different_calls():
    r = calibrate.calibrate([same_call(True, True, repeat=0), same_call(True, True, repeat=1),
                             same_call(True, True, record="other.json"), same_call(True, True, persona="q")])
    assert llm(r)["n"] == 4


@pytest.mark.parametrize("make", ["dir", "unreadable"])
def test_unreadable_labels_path_does_not_crash(tmp_path, capsys, make):
    path = tmp_path / "labels"
    if make == "dir":
        path.mkdir()
    else:
        write(path, [label(True, True)])
        path.chmod(0)
    try:
        assert calibrate.main(["--labels", str(path)]) == 0
        out = capsys.readouterr().out
        assert "cannot read the labels file" in out and "0 labels" in out
        s = calibrate.summary(path)
        assert s["real_llm_labels"] == 0 and s["badge_threshold"] == 20 and "cannot read" in s["message"]
        srv = serve(tmp_path, path)
        try:
            status, body = get(srv)
            assert status == 200 and body["real_llm_labels"] == 0
        finally:
            srv.shutdown()
            srv.server_close()
    finally:
        path.chmod(0o644) if path.is_file() else None


def test_deeply_nested_line_is_skipped(tmp_path, capsys):
    path = write(tmp_path / "l.jsonl", [label(True, True)], extra="[" * 200000 + "\n")
    assert calibrate.main(["--labels", str(path)]) == 0
    assert "skipped 1" in capsys.readouterr().out
    srv = serve(tmp_path, path)
    try:
        status, body = get(srv)
        assert status == 200 and body["real_llm_labels"] == 1
    finally:
        srv.shutdown()
        srv.server_close()


def test_kappa_hand_computed():
    assert llm(calibrate.calibrate(cells(4, 3, 2, 1)))["kappa"] == "-0.09"
    assert llm(calibrate.calibrate(cells(20, 5, 10, 15)))["kappa"] == "0.40"


def test_printed_agreement_matrix_and_verdicts():
    text = report(cells(4, 3, 2, 1))
    assert "agreement 50%, Cohen's kappa -0.09" in text
    lines = [x.strip() for x in text.splitlines()]
    assert "judge pass  judge fail" in lines
    assert ["human", "pass", "4", "3"] in [x.split() for x in lines]
    assert ["human", "fail", "2", "1"] in [x.split() for x in lines]
    assert text.index("human pass ") < text.index("human fail ")
    assert "human pass, judge fail" in text and "human fail, judge pass" in text


def test_verdict_text_matches_each_direction():
    assert "human pass, judge fail" in report([label(True, False)])
    assert "human fail, judge pass" not in report([label(True, False)])
    assert "human fail, judge pass" in report([label(False, True)])
    assert "human pass, judge fail" not in report([label(False, True)])


def test_printed_turn_agreement_is_agree_over_n():
    text = report([label(False, False, hturn=1, jturn=1), label(False, False, hturn=1, jturn=1),
                   label(False, False, hturn=1, jturn=2), label(False, False, hturn=1, jturn=2),
                   label(False, False, hturn=1, jturn=2)])
    assert "failing turn agreement: 2/5 calls both failed with a turn" in text


def test_both_failed_without_judge_turn_is_excluded():
    r = calibrate.calibrate([label(False, False, hturn=1, jturn=None), label(False, False, hturn=2, jturn=2)])
    assert llm(r)["failing_turn"] == {"n": 1, "agree": 1}
    r = calibrate.calibrate([label(False, False, hturn=1, jturn=None)])
    assert llm(r)["failing_turn"] == {"n": 0, "agree": 0}
    assert "failing turn agreement: n/a" in calibrate.format_report(r)


def test_badge_boundary(tmp_path):
    assert calibrate.summary(write(tmp_path / "a.jsonl", cells(19, 0, 0, 0)))["badge_ready"] is False
    s = calibrate.summary(write(tmp_path / "b.jsonl", cells(20, 0, 0, 0)))
    assert s["badge_ready"] is True and s["real_llm_labels"] == 20


def test_unknown_judge_source_is_reported(tmp_path, capsys):
    bad = label(True, True, source="gpt")
    path = write(tmp_path / "l.jsonl", [label(True, True), bad])
    assert calibrate.main(["--labels", str(path)]) == 0
    out = capsys.readouterr().out
    assert "skipped 1" in out and "judge_source must be llm or rules" in out and "line 2" in out


def test_each_synthetic_marker_is_shown():
    labels = [label(True, True), label(True, False, synthetic=True)]
    text = report(labels, include_synthetic=True)
    assert text.splitlines()[0].startswith("SYNTHETIC labels included (1)")
    assert "LLM judge: 2 labels (includes SYNTHETIC, not a real result)" in text
    assert "[SYNTHETIC]" in text
    # a real disagreement carries no tag, and nothing is marked when synthetic is not included
    plain = report([label(True, False)], include_synthetic=True)
    assert "SYNTHETIC" not in plain
    assert "SYNTHETIC" not in report(labels)


def test_only_explicitly_real_labels_count(tmp_path):
    missing = label(True, True)
    del missing["synthetic"]
    null = label(True, True)
    null["synthetic"] = None
    simulated = label(True, True)
    simulated["on_simulated_run"] = True
    path = write(tmp_path / "l.jsonl", [missing, null, simulated])
    assert calibrate.summary(path)["real_llm_labels"] == 0
    assert llm(calibrate.calibrate(calibrate.load_labels(path)[0]))["n"] == 0
    assert llm(calibrate.calibrate(calibrate.load_labels(path)[0], include_synthetic=True))["n"] == 3
    assert calibrate.summary(write(tmp_path / "ok.jsonl", [label(True, True)]))["real_llm_labels"] == 1


def test_percent_is_floored_and_never_a_false_100():
    text = report(cells(199, 1, 0, 0))
    assert "agreement 99%" in text and "100%" not in text
    assert "agreement 100%" in report(cells(5, 0, 0, 0))


def test_kappa_is_never_negative_zero():
    r = calibrate.calibrate(cells(100, 101, 101, 100))  # kappa is about -0.005
    assert llm(r)["kappa"] == "0.00"
    assert "-0.00" not in calibrate.format_report(r)


def test_zero_labels_after_synthetic_skipped(tmp_path, capsys):
    path = write(tmp_path / "l.jsonl", [label(True, True, synthetic=True), label(True, False, synthetic=True)])
    assert calibrate.main(["--labels", str(path)]) == 0
    out = capsys.readouterr().out
    assert "0 labels (2 synthetic labels skipped; pass --include-synthetic to see them)" in out
    assert "label page" not in out.lower()
    (tmp_path / "e.jsonl").write_text("", encoding="utf-8")
    calibrate.main(["--labels", str(tmp_path / "e.jsonl")])
    out = capsys.readouterr().out
    assert "0 labels" in out and "skipped" not in out and "label page" not in out.lower()


def test_utf8_bom_does_not_lose_the_first_line(tmp_path):
    path = tmp_path / "l.jsonl"
    path.write_bytes(b"\xef\xbb\xbf" + (json.dumps(label(True, True)) + "\n").encode("utf-8"))
    labels, problems = calibrate.load_labels(path)
    assert len(labels) == 1 and not problems
