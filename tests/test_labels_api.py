"""POST /api/labels against a real server on port 0. The labels file is always under tmp_path."""
import http.client
import json
import os
import threading
from pathlib import Path

import pytest

from callqa import server as server_module
from callqa.server import make_server
from evals.backends import OfflineBackend

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "report_full_v2_r3.json"
NAME = "20261106-100000_v2.json"
RECORD = json.loads(FIXTURE.read_text(encoding="utf-8"))
PERSONA = next(c for c in RECORD["calls"] if c["failing_turn"] is not None)["persona_id"]
AGENTS = [{"id": "v1", "label": "v1"}, {"id": "v2", "label": "v2"}]


def good(**changes):
    base = {"record": NAME, "persona_id": PERSONA, "repeat": 0, "human_pass": False, "human_failing_turn": 1,
            "human_reason": "never offered a callback", "labeller": "bob"}
    base.update(changes)
    return base


@pytest.fixture
def env(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / NAME).write_text(json.dumps(RECORD), encoding="utf-8")
    path = tmp_path / "evals" / "labels" / "human_labels.jsonl"
    srv = make_server(OfflineBackend(), 20, 8, 2, AGENTS, results_dir=runs, samples_dir=tmp_path / "none",
                      labels_path=path)
    threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
    yield srv, path, tmp_path
    srv.shutdown()
    srv.server_close()


def post(srv, payload, path="/api/labels", host=None, origin=None, raw=None):
    port = srv.server_address[1]
    data = raw if raw is not None else json.dumps(payload).encode("utf-8")
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.putrequest("POST", path, skip_host=True, skip_accept_encoding=True)
    conn.putheader("Host", host or f"127.0.0.1:{port}")
    if origin:
        conn.putheader("Origin", origin)
    conn.putheader("Content-Type", "application/json")
    conn.putheader("Content-Length", str(len(data)))
    conn.endheaders()
    conn.send(data)
    resp = conn.getresponse()
    status, text = resp.status, resp.read().decode("utf-8")
    conn.close()
    return status, json.loads(text)


def test_valid_label_is_appended_as_one_line(env):
    srv, path, _ = env
    status, out = post(srv, good())
    assert status == 200 and out["ok"] is True and len(out["label_id"]) == 8
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["label_id"] == out["label_id"] and row["persona_id"] == PERSONA and row["human_pass"] is False


def test_same_origin_is_accepted(env):
    srv, path, _ = env
    port = srv.server_address[1]
    assert post(srv, good(), origin=f"http://127.0.0.1:{port}")[0] == 200


def test_bad_bodies_are_400_and_write_nothing(env):
    srv, path, _ = env
    cases = [good(human_pass="maybe"), good(extra=1), good(record="nope.json"), good(record="../../etc/passwd"),
             good(persona_id="nobody"), good(human_failing_turn=0), good(human_reason="x" * 501),
             good(labeller="n" * 41), [1, 2]]
    for payload in cases:
        status, out = post(srv, payload)
        assert status == 400 and "error" in out, payload
    assert post(srv, None, raw=b"{half")[0] == 400
    assert not path.exists()


def test_bad_host_or_origin_is_403(env):
    srv, path, _ = env
    port = srv.server_address[1]
    assert post(srv, good(), host=f"evil.example:{port}")[0] == 403
    assert post(srv, good(), origin=f"http://evil.example:{port}")[0] == 403
    assert not path.exists()


def test_oversized_body_is_413(env):
    srv, path, _ = env
    big = json.dumps(good(human_reason="x" * 300_000)).encode("utf-8")
    status, out = post(srv, None, raw=big)
    assert status == 413 and "too large" in out["error"]
    assert not path.exists()


def test_full_file_gives_409_and_other_routes_keep_their_400_for_big_bodies(env):
    srv, path, _ = env
    path.parent.mkdir(parents=True)
    path.write_bytes(b"x" * (5 * 1024 * 1024))
    assert post(srv, good())[0] == 409
    big = json.dumps({"sop": "x" * 300_000, "agent": "v1"}).encode("utf-8")
    assert post(srv, None, path="/api/run", raw=big)[0] == 400


def test_the_route_only_writes_inside_the_labels_folder(env):
    srv, path, tmp_path = env
    before = {p for p in tmp_path.rglob("*") if p.is_file()}
    post(srv, good())
    post(srv, good(record="../runs/" + NAME))
    created = {p for p in tmp_path.rglob("*") if p.is_file()} - before
    assert created == {path}


def test_deeply_nested_json_is_a_400_on_both_routes_and_the_server_lives(env):
    srv, path, _ = env
    nested = b"[" * 50_000 + b"]" * 50_000
    assert post(srv, None, raw=nested)[0] == 400
    assert post(srv, None, path="/api/run", raw=nested)[0] == 400
    assert post(srv, good())[0] == 200


def test_lone_surrogate_over_http_is_a_400(env):
    srv, path, _ = env
    status, out = post(srv, good(human_reason="\ud800"))
    assert status == 400 and "cannot be saved" in out["error"]
    assert not path.exists()


def test_oversized_body_is_drained_so_the_next_request_works(env):
    srv, path, _ = env
    assert post(srv, None, raw=json.dumps(good(human_reason="x" * 300_000)).encode("utf-8"))[0] == 413
    assert post(srv, good())[0] == 200  # a new connection after the refusal


def test_read_only_labels_folder_gives_500_without_the_absolute_path(env):
    srv, path, tmp_path = env
    if os.geteuid() == 0:
        pytest.skip("root ignores folder permissions")
    path.parent.mkdir(parents=True)
    path.parent.chmod(0o500)
    try:
        status, out = post(srv, good())
    finally:
        path.parent.chmod(0o700)
    assert status == 500 and "could not save" in out["error"]
    assert str(tmp_path) not in out["error"] and "/" not in out["error"]


def test_damaged_record_is_a_400_not_a_dropped_connection(env):
    srv, path, tmp_path = env
    damaged = {**RECORD, "calls": None}
    (tmp_path / "runs" / "damaged.json").write_text(json.dumps(damaged), encoding="utf-8")
    status, out = post(srv, good(record="damaged.json"))
    assert status == 400 and "damaged" in out["error"]


def test_default_labels_path_is_the_real_folder():
    assert server_module.LABELS_FILE == server_module.ROOT / "evals" / "labels" / "human_labels.jsonl"
