"""The records routes (C10): /records, /api/records and /api/records/<name>, against a real server on port 0."""
import http.client
import json
import threading
from pathlib import Path

import pytest

from callqa import server as server_module
from callqa.server import make_server
from evals.backends import OfflineBackend

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "report_full_v2_r3.json"
AGENTS = [{"id": "v1", "label": "v1 naive prompt"}, {"id": "v2", "label": "v2 fixed prompt"}]


def raw_get(srv, path, host=None):
    conn = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=10)
    conn.putrequest("GET", path, skip_host=True, skip_accept_encoding=True)
    conn.putheader("Host", host or f"127.0.0.1:{srv.server_address[1]}")
    conn.endheaders()
    resp = conn.getresponse()
    status, text = resp.status, resp.read().decode("utf-8")
    conn.close()
    return status, text


def write_record(directory, name, created_at, agent_id="v2", pass_rate=0.5, synthetic=True):
    record = json.loads(FIXTURE.read_text(encoding="utf-8"))
    record.update(created_at=created_at, pass_rate=pass_rate, synthetic=synthetic)
    record["agent"] = {**record["agent"], "id": agent_id, "label": f"{agent_id} label"}
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(json.dumps(record), encoding="utf-8")


def serve(runs, samples):
    srv = make_server(OfflineBackend(), 20, 8, 2, AGENTS, results_dir=runs, samples_dir=samples)
    threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
    return srv


@pytest.fixture
def server(tmp_path):
    runs, samples = tmp_path / "runs", tmp_path / "samples"
    write_record(runs, "20261106-100000_v1.json", "2026-11-06T10:00:00+05:30", "v1", 0.7, synthetic=False)
    write_record(samples, "sample_v2.json", "2026-11-05T09:00:00+05:30", "v2", 0.95)
    (runs / "notes.txt").write_text("not a record", encoding="utf-8")
    (runs / "broken.json").write_text("{half", encoding="utf-8")
    (tmp_path / "secret.json").write_text('{"secret": true}', encoding="utf-8")
    srv = serve(runs, samples)
    yield srv
    srv.shutdown()
    srv.server_close()


def test_records_page_is_served(server):
    status, text = raw_get(server, "/records")
    assert status == 200 and 'id="compare-button"' in text and 'id="run-button"' not in text


def test_records_listing_includes_samples_and_runs(server):
    status, text = raw_get(server, "/api/records")
    rows = json.loads(text)
    assert status == 200
    assert [r["name"] for r in rows] == ["20261106-100000_v1.json", "sample_v2.json"]  # newest first, junk skipped
    run, sample = rows
    assert set(run) == {"name", "kind", "agent", "set", "repeats", "backend", "pass_rate", "created_at", "synthetic"}
    assert (run["kind"], run["agent"], run["set"], run["repeats"], run["backend"]) == (
        "run", "v1 label", "full", 3, "offline")
    assert run["pass_rate"] == 0.7 and run["synthetic"] is False
    assert sample["synthetic"] is True and sample["pass_rate"] == 0.95


def test_listing_is_empty_when_no_folder_exists(tmp_path):
    srv = serve(tmp_path / "none", tmp_path / "also_none")
    try:
        assert raw_get(srv, "/api/records") == (200, "[]")
    finally:
        srv.shutdown()
        srv.server_close()


def test_one_record_is_returned_whole(server):
    status, text = raw_get(server, "/api/records/sample_v2.json")
    record = json.loads(text)
    assert status == 200 and record["pass_rate"] == 0.95 and len(record["calls"]) == 22
    assert raw_get(server, "/api/records/20261106-100000_v1.json")[0] == 200


@pytest.mark.parametrize("path", [
    "/api/records/..%2Fsecret.json",
    "/api/records/%2e%2e%2fsecret.json",
    "/api/records/../secret.json",
    "/api/records//etc/passwd",
    "/api/records/%2Fetc%2Fpasswd.json",
    "/api/records/secret.json",  # a real file, but not in the listing
    "/api/records/notes.txt",
    "/api/records/broken.json",  # matches the pattern, but it is not a record, so it is not listed
    "/api/records/missing.json",
    "/api/records/sample_v2.json%00.txt",
    "/api/records/",
])
def test_record_name_traversal_is_404(server, path):
    status, text = raw_get(server, path)
    assert status == 404, text
    assert "secret" not in text


def test_runs_win_over_samples_of_the_same_name(tmp_path):
    runs, samples = tmp_path / "runs", tmp_path / "samples"
    write_record(runs, "same.json", "2026-11-06T10:00:00+05:30", "v1", 0.1)
    write_record(samples, "same.json", "2026-11-05T10:00:00+05:30", "v2", 0.9)
    srv = serve(runs, samples)
    try:
        assert [r["name"] for r in json.loads(raw_get(srv, "/api/records")[1])] == ["same.json"]
        assert json.loads(raw_get(srv, "/api/records/same.json")[1])["pass_rate"] == 0.1
    finally:
        srv.shutdown()
        srv.server_close()


def test_records_routes_check_host(server):
    port = server.server_address[1]
    for path in ("/records", "/api/records", "/api/records/sample_v2.json"):
        assert raw_get(server, path, host=f"evil.example:{port}")[0] == 403
        assert raw_get(server, path, host=f"localhost:{port}")[0] == 200


def test_default_sample_folder_is_inside_the_demo():
    assert server_module.SAMPLE_RECORDS == server_module.ROOT / "samples" / "run_records"


# ---- fix round 1: listing edge cases and history rows ----

def listed(srv):
    return [r["name"] for r in json.loads(raw_get(srv, "/api/records")[1])]


def test_name_with_a_trailing_newline_is_neither_listed_nor_served(tmp_path):
    runs = tmp_path / "runs"
    write_record(runs, "b.json\n", "2026-11-06T10:00:00+05:30")
    srv = serve(runs, tmp_path / "none")
    try:
        assert listed(srv) == []
        assert raw_get(srv, "/api/records/b.json%0A")[0] == 404
    finally:
        srv.shutdown()
        srv.server_close()


def test_symlinks_dotfiles_and_oversized_files_are_skipped(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    write_record(runs, "good.json", "2026-11-06T10:00:00+05:30")
    write_record(runs, ".hidden.json", "2026-11-06T11:00:00+05:30")
    write_record(tmp_path / "elsewhere", "target.json", "2026-11-06T12:00:00+05:30")
    (runs / "link.json").symlink_to(tmp_path / "elsewhere" / "target.json")
    big = json.loads(FIXTURE.read_text(encoding="utf-8"))
    (runs / "big.json").write_text(json.dumps(big, indent=40), encoding="utf-8")
    cap = (runs / "good.json").stat().st_size + 1
    assert (runs / "big.json").stat().st_size > cap
    monkeypatch.setattr(server_module, "MAX_RECORD_BYTES", cap)
    srv = serve(runs, tmp_path / "none")
    try:
        assert listed(srv) == ["good.json"]
        for skipped in ("link.json", ".hidden.json", "big.json"):
            assert raw_get(srv, f"/api/records/{skipped}")[0] == 404
    finally:
        srv.shutdown()
        srv.server_close()


def test_name_with_an_at_sign_round_trips_encoded_or_not(tmp_path):
    runs = tmp_path / "runs"
    write_record(runs, "20261106-100000_v2@haiku.json", "2026-11-06T10:00:00+05:30")
    srv = serve(runs, tmp_path / "none")
    try:
        for part in ("v2@haiku", "v2%40haiku"):
            status, text = raw_get(srv, f"/api/records/20261106-100000_{part}.json")
            assert status == 200 and json.loads(text)["run_id"]
    finally:
        srv.shutdown()
        srv.server_close()


def test_history_rows_carry_set_and_repeats(tmp_path):
    import time
    import urllib.request
    srv = serve(tmp_path / "runs", tmp_path / "none")
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        body = json.dumps({"sop": server_module.read_sample_sop(), "agent": "v1", "set": "full", "repeats": 2})
        headers = {"Content-Type": "application/json"}
        req = urllib.request.Request(base + "/api/run", data=body.encode(), headers=headers)
        run_id = json.loads(urllib.request.urlopen(req, timeout=30).read())["run_id"]
        for _ in range(600):
            view = json.loads(urllib.request.urlopen(f"{base}/api/run/{run_id}", timeout=30).read())
            if view["status"] != "running":
                break
            time.sleep(0.05)
        assert view["status"] == "done"
        history = json.loads(urllib.request.urlopen(base + "/api/runs", timeout=30).read())
        assert (history[0]["set"], history[0]["repeats"]) == ("full", 2)
    finally:
        srv.shutdown()
        srv.server_close()
