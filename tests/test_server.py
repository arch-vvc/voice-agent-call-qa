"""The dashboard server and report, against a real server on port 0.

Offline tests use the rule based agents. LLM paths use a fake backend: no network, no real `claude`.
"""
import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from callqa import server
from callqa.personas import generate_personas
from callqa.server import make_server, run_check
from evals import example_agent
from evals.backends import BackendError, OfflineBackend

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
SOP = (ROOT / "samples" / "sop_collections_acme.md").read_text(encoding="utf-8")
DEFAULT_AGENTS = [{"id": "v1", "label": "v1 naive prompt"}, {"id": "v2", "label": "v2 fixed prompt"}]
END = "<END_CALL>"


class FakeBackend:
    """A claude-cli shaped backend. Handlers are keyed by role and take (system, messages)."""

    name = "claude-cli"
    requested = "claude-cli"

    def __init__(self, **handlers):
        self.warnings = []
        self.handlers = handlers
        self.calls = []
        self.lock = threading.Lock()

    def complete(self, system, messages, model, max_tokens):
        role = self._role(system, messages)
        with self.lock:
            self.calls.append((role, system, messages))
        handler = self.handlers.get(role)
        if handler is not None:
            return handler(system, messages)
        if role == "gen":
            raise BackendError("generator down", kind="failed")  # falls back to templates
        if role == "caller":
            return END
        if role == "judge":
            return '{"pass": true, "reason": "fine", "turn": null}'
        return "Is the 1st okay for you?"

    @staticmethod
    def _role(system, messages):
        if "You design test callers" in system:
            return "gen"
        if "role playing a customer" in system:
            return "caller"
        if "collections agent" in system:
            return "agent"
        return "judge"

    def count(self, role):
        return sum(1 for r, _, _ in self.calls if r == role)


def request(base, path, body=None, raw=None, content_type="application/json"):
    data = raw if raw is not None else (json.dumps(body).encode("utf-8") if body is not None else None)
    req = urllib.request.Request(base + path, data=data, headers={"Content-Type": content_type} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload, status = resp.read(), resp.status
    except urllib.error.HTTPError as err:
        payload, status = err.read(), err.code
    return status, payload.decode("utf-8")


def get_json(base, path):
    status, text = request(base, path)
    return status, json.loads(text)


def post_run(base, sop=SOP, agent="v1"):
    status, text = request(base, "/api/run", {"sop": sop, "agent": agent})
    return status, json.loads(text)


def wait_finished(base, run_id, timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        _, body = get_json(base, f"/api/run/{run_id}")
        if body["status"] != "running":
            return body
        time.sleep(0.01)
    raise AssertionError("run did not finish")


def run_to_end(base, agent="v1", sop=SOP):
    status, body = post_run(base, sop, agent)
    assert status == 202, body
    return wait_finished(base, body["run_id"])


@pytest.fixture
def start(tmp_path):
    servers = []

    def _start(backend=None, callers=20, max_turns=8, workers=4, agents=None):
        srv = make_server(backend or OfflineBackend(), callers, max_turns, workers,
                          agents or DEFAULT_AGENTS, results_dir=tmp_path / "results")
        threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
        servers.append(srv)
        return f"http://127.0.0.1:{srv.server_address[1]}", srv

    yield _start
    for srv in servers:
        srv.shutdown()
        srv.server_close()


def test_index_served(start):
    base, _ = start()
    status, text = request(base, "/")
    assert status == 200
    assert 'id="run-button"' in text
    assert text == (ROOT / "callqa" / "static" / "index.html").read_text(encoding="utf-8")


def test_binds_loopback_only(start):
    _, srv = start()
    assert srv.server_address[0] == "127.0.0.1"


def test_config_reports_backend_models_warnings_and_estimated_calls(start):
    backend = FakeBackend()
    backend.warnings = ["something to know"]
    base, _ = start(backend, callers=5, max_turns=4, workers=2)
    status, cfg = get_json(base, "/api/config")
    assert status == 200
    assert cfg["backend"]["name"] == "claude-cli" and cfg["backend"]["requested"] == "claude-cli"
    assert set(cfg["backend"]["models"]) == {"caller", "agent", "judge", "gen"}
    assert cfg["warnings"] == ["something to know"]
    assert (cfg["callers"], cfg["max_turns"], cfg["estimated_calls"]) == (5, 4, 55)
    assert cfg["agents"] == DEFAULT_AGENTS
    assert "₹12,400" in cfg["sample_sop"]
    assert cfg["agent_kind"] == "llm"


def test_estimated_calls_is_2ct_plus_3c(start):
    base, _ = start(callers=5, max_turns=4)
    assert get_json(base, "/api/config")[1]["estimated_calls"] == 55  # 2*5*4 + 5 final replies + 2*5 judge runs
    base2, _ = start(callers=20, max_turns=8)
    assert get_json(base2, "/api/config")[1]["estimated_calls"] == 380


def test_post_run_returns_run_id_then_poll_returns_report(start):
    base, _ = start()
    status, body = post_run(base)
    assert status == 202 and set(body) == {"run_id"}
    done = wait_finished(base, body["run_id"])
    assert done["status"] == "done"
    assert set(done) == {"status", "done", "total", "report"}
    assert (done["done"], done["total"]) == (20, 20)
    assert done["report"]["run_id"] == body["run_id"]


def test_poll_reports_progress(start):
    release = threading.Event()
    seen = set()

    def build(sop_text, backend):
        def agent(history):
            seen.add(history[0]["content"])
            if len(seen) > 1:  # the first conversation finishes, the rest wait
                release.wait(10)
            return "Is the 1st okay?"
        return agent

    base, _ = start(callers=3, workers=1, agents=[{"id": "slow", "label": "Slow", "build": build}])
    _, body = post_run(base, agent="slow")
    try:
        deadline = time.time() + 10
        while time.time() < deadline:
            _, state = get_json(base, f"/api/run/{body['run_id']}")
            if state["done"] >= 1:
                break
            time.sleep(0.01)
        assert state == {"status": "running", "done": 1, "total": 3}
    finally:
        release.set()
    assert wait_finished(base, body["run_id"])["status"] == "done"


def test_unknown_run_id_is_404(start):
    base, _ = start()
    status, body = get_json(base, "/api/run/nope")
    assert status == 404 and "error" in body
    assert get_json(base, "/api/nothing")[0] == 404


def test_second_run_while_running_is_409(start):
    release = threading.Event()

    def build(sop_text, backend):
        return lambda history: release.wait(10) and "Is the 1st okay?"

    base, _ = start(callers=2, workers=1, agents=[{"id": "slow", "label": "Slow", "build": build}])
    status, first = post_run(base, agent="slow")
    assert status == 202
    try:
        status, body = post_run(base, agent="slow")
        assert status == 409 and "error" in body
    finally:
        release.set()
    assert wait_finished(base, first["run_id"])["status"] == "done"
    assert post_run(base, agent="slow")[0] == 202


def test_run_v1_offline_is_70_percent_with_5_leaks_at_turn_1_and_escalation_at_turn_3(start):
    base, _ = start()
    report = run_to_end(base, "v1")["report"]
    assert (report["pass_rate"], report["passed"], report["total"]) == (0.7, 14, 20)
    by_id = {c["persona_id"]: c for c in report["calls"]}
    for pid in ("brother_asks", "wife_handles_money", "wrong_number", "prompt_injection", "fake_head_office"):
        call = by_id[pid]
        assert (call["failure_kind"], call["failing_turn"]) == ("forbidden", 1)
        assert "12,400" in call["transcript"][1]["content"]
    manager = by_id["wants_manager"]
    assert (manager["failure_kind"], manager["failing_turn"]) == ("missing", 3)
    assert report["warnings"] == [] and report["generator"] == {"source": "templates", "warnings": []}


def test_run_v2_offline_is_95_percent_with_one_next_fix(start):
    base, _ = start()
    report = run_to_end(base, "v2")["report"]
    assert (report["pass_rate"], report["passed"], report["total"]) == (0.95, 19, 20)
    assert len(report["next_fixes"]) == 1
    fix = report["next_fixes"][0]
    assert fix["sop_rule"].startswith("Escalation") and "callback" in fix["fix_hint"]
    assert fix["calls"] == [{"persona_id": "wants_manager", "name": fix["calls"][0]["name"], "failing_turn": 3}]


def test_next_fixes_grouped_by_sop_rule_largest_first(start):
    base, _ = start()
    fixes = run_to_end(base, "v1")["report"]["next_fixes"]
    assert [(f["count"], f["sop_rule"].split(":")[0]) for f in fixes] == [(5, "Verification"), (1, "Escalation")]
    assert len(fixes[0]["calls"]) == 5 and {c["failing_turn"] for c in fixes[0]["calls"]} == {1}
    assert "Confirm you are speaking with" in fixes[0]["fix_hint"]


def test_all_pass_has_no_next_fixes():
    from callqa.report import group_next_fixes
    assert group_next_fixes([]) == []


def test_llm_backend_builds_agents_from_pasted_sop(start):
    backend = FakeBackend()
    pasted = SOP.replace("Ravi Kumar", "Meera Rao").replace("Acme", "Zeta")
    base, _ = start(backend, callers=6, max_turns=2, workers=1)  # above the default five: generated, not the story set
    report = run_to_end(base, "v2", pasted)["report"]
    assert report["total"] == 6 and report["backend"]["name"] == "claude-cli"
    assert set(report["backend"]["models"]) == {"caller", "agent", "judge", "gen"}
    systems = [s for role, s, _ in backend.calls if role == "agent"]
    assert systems and all("Meera Rao" in s and "Ravi Kumar" not in s for s in systems)
    assert backend.count("gen") == 1
    assert any("Persona generation failed" in w for w in report["generator"]["warnings"])
    assert set(report["generator"]["warnings"]) <= set(report["warnings"])


def test_judge_reason_becomes_fix_hint(start):
    judge = '{"pass": false, "reason": "Agent promised a waiver in turn 1.", "turn": 1}'
    backend = FakeBackend(judge=lambda s, m: judge)
    base, _ = start(backend, callers=1, max_turns=2, workers=1)
    call = run_to_end(base)["report"]["calls"][0]
    assert (call["passed"], call["failure_kind"], call["failing_turn"]) == (False, "judge", 1)
    assert call["fix_hint"] == "Agent promised a waiver in turn 1."
    assert call["reason"] == "Agent promised a waiver in turn 1."


def test_usage_limit_becomes_run_error_not_500_and_not_in_history(start):
    def limit(system, messages):
        raise BackendError("usage limit reached", kind="limit")

    base, _ = start(FakeBackend(agent=limit), callers=3, max_turns=2, workers=1)
    body = run_to_end(base)
    assert body == {"status": "error", "error": "usage limit reached, try later or use --backend offline"}
    assert get_json(base, "/api/runs") == (200, [])
    assert post_run(base)[0] == 202  # the server is free for another run


def test_limit_while_generating_personas_is_a_run_error(start):
    def limit(system, messages):
        raise BackendError("rate limit", kind="limit")

    base, _ = start(FakeBackend(gen=limit), callers=6)
    body = run_to_end(base)
    assert body["status"] == "error" and "usage limit" in body["error"]


def test_backend_crash_on_one_call_adds_report_warning(start):
    state = {"n": 0}

    def flaky(system, messages):
        state["n"] += 1
        if state["n"] == 1:
            raise BackendError("claude -p exited 1", kind="failed")
        return "Is the 1st okay?"

    base, _ = start(FakeBackend(agent=flaky), callers=2, max_turns=1, workers=1)
    report = run_to_end(base)["report"]
    first, second = report["calls"]
    assert (first["passed"], first["failure_kind"], first["failing_turn"]) == (False, "crash", None)
    assert "claude -p exited 1" in first["reason"]
    assert second["failure_kind"] != "crash"
    crash_lines = [w for w in report["warnings"] if "crashed" in w]
    assert len(crash_lines) == 1 and "claude -p exited 1" in crash_lines[0]


def test_agent_exception_is_a_crash_not_a_500(start):
    def build(sop_text, backend):
        def agent(history):
            raise RuntimeError("boom")
        return agent

    base, _ = start(callers=2, agents=[{"id": "bad", "label": "Bad", "build": build}])
    report = run_to_end(base, "bad")["report"]
    assert report["passed"] == 0 and {c["failure_kind"] for c in report["calls"]} == {"crash"}
    assert len([w for w in report["warnings"] if "boom" in w]) == 1


def test_same_sop_reuses_personas_between_runs(start):
    backend = FakeBackend()
    base, _ = start(backend, callers=6, max_turns=1, workers=1)
    first = run_to_end(base, "v1")["report"]
    second = run_to_end(base, "v2")["report"]
    assert backend.count("gen") == 1
    assert [c["persona_id"] for c in first["calls"]] == [c["persona_id"] for c in second["calls"]]
    assert first["sop_hash"] == second["sop_hash"]
    run_to_end(base, "v1", SOP + "\n")  # an edited SOP is a new set of callers
    assert backend.count("gen") == 2


def test_runs_history_has_before_and_after(start):
    base, _ = start()
    assert get_json(base, "/api/runs") == (200, [])
    v1 = run_to_end(base, "v1")["report"]
    # The history must already hold the run by the time a poll says "done".
    status, post = post_run(base, agent="v2")
    body = wait_finished(base, post["run_id"])
    _, runs = get_json(base, "/api/runs")
    assert body["status"] == "done" and post["run_id"] in [r["run_id"] for r in runs]
    assert [r["agent_label"] for r in runs] == ["v1 naive prompt", "v2 fixed prompt"]
    assert runs[0] == {"run_id": v1["run_id"], "sop_hash": v1["sop_hash"], "agent_label": "v1 naive prompt",
                       "backend": "offline", "pass_rate": 0.7, "passed": 14, "total": 20}
    assert runs[1]["pass_rate"] == 0.95


def test_finished_run_is_saved_to_results_dir(start, tmp_path):
    base, _ = start()
    report = run_to_end(base, "v1")["report"]
    files = list((tmp_path / "results").glob("dashboard-*-v1.json"))
    assert len(files) == 1
    assert json.loads(files[0].read_text(encoding="utf-8"))["run_id"] == report["run_id"]


@pytest.mark.parametrize("body", [{"sop": "", "agent": "v1"}, {"sop": "   \n", "agent": "v1"}, {"agent": "v1"}])
def test_empty_sop_is_400(start, body):
    base, _ = start()
    status, text = request(base, "/api/run", body)
    assert status == 400 and json.loads(text)["error"]


def test_unknown_agent_is_400(start):
    base, _ = start()
    status, text = request(base, "/api/run", {"sop": SOP, "agent": "os:system"})
    assert status == 400 and "agent" in json.loads(text)["error"]


def test_oversized_body_is_400(start):
    base, _ = start()
    status, text = request(base, "/api/run", {"sop": "x" * 200_001, "agent": "v1"})
    assert status == 400 and "too large" in json.loads(text)["error"]


def test_non_json_body_is_400(start):
    base, _ = start()
    assert request(base, "/api/run", raw=b"sop=1", content_type="text/plain")[0] == 400
    assert request(base, "/api/run", raw=b"{not json")[0] == 400
    assert request(base, "/api/run", raw=b"[1]")[0] == 400


def test_report_keys_match_fixture(start):
    fixture = json.loads((FIXTURES / "report_example.json").read_text(encoding="utf-8"))
    base, _ = start()
    report = run_to_end(base, "v1")["report"]
    assert set(report) == set(fixture)
    for key in ("agent", "backend", "generator"):
        assert set(report[key]) == set(fixture[key])
    assert set(report["calls"][0]) == set(fixture["calls"][0])
    assert set(report["next_fixes"][0]) == set(fixture["next_fixes"][0])
    assert set(report["next_fixes"][0]["calls"][0]) == set(fixture["next_fixes"][0]["calls"][0])
    for key, value in fixture.items():
        if value is not None and key not in ("calls", "next_fixes"):
            assert isinstance(report[key], type(value)), key
    for key, value in fixture["calls"][1].items():
        if value is not None:
            assert isinstance(report["calls"][1][key], type(value)), key


def test_config_keys_match_fixture(start):
    fixture = json.loads((FIXTURES / "config_example.json").read_text(encoding="utf-8"))
    base, _ = start(FakeBackend(), callers=5, max_turns=4, workers=2)
    cfg = get_json(base, "/api/config")[1]
    assert set(cfg) == set(fixture)
    assert set(cfg["backend"]) == set(fixture["backend"])
    assert set(cfg["backend"]["models"]) == set(fixture["backend"]["models"])
    for key, value in fixture.items():
        assert isinstance(cfg[key], type(value)), key
    assert cfg["agent_kind"] == fixture["agent_kind"]


def test_check_offline_20_and_5_lines():
    ok, lines = run_check(OfflineBackend(), 20, 8, 4)
    assert ok
    assert lines[-1] == ("CHECK OK backend=offline callers=20 v1 70% (14/20) v2 95% (19/20) "
                         "v2 leaks: none; v2 failures: wants_manager")
    ok, lines = run_check(OfflineBackend(), 5, 8, 4)
    assert ok
    assert lines[-1] == ("CHECK OK backend=offline callers=5 v1 20% (1/5) v2 80% (4/5) "
                         "v2 leaks: none; v2 failures: wants_manager")


def test_check_includes_backend_warnings_before_the_result_line():
    backend = OfflineBackend(warnings=["claude CLI not found on PATH; running offline with rule based demo agents"])
    ok, lines = run_check(backend, 5, 8, 4)
    assert ok and "claude CLI not found on PATH; running offline" in lines[0]
    assert lines[-1].startswith("CHECK OK")


def test_check_fails_when_v2_leaks():
    leaky = {"id": "v2", "label": "v2", "build": lambda sop, backend: example_agent.respond}
    ok, lines = run_check(OfflineBackend(), 20, 8, 4, agents=[{"id": "v1", "label": "v1"}, leaky])
    assert not ok
    assert lines[-1].startswith("CHECK FAILED")
    assert "S1" in lines[-1] and "brother_asks" in lines[-1]


def test_check_fails_when_v1_never_leaks():
    clean = {"id": "v1", "label": "v1", "build": lambda sop, backend: lambda history: "Who am I speaking with?"}
    ok, lines = run_check(OfflineBackend(), 5, 8, 4, agents=[clean, {"id": "v2", "label": "v2"}])
    assert not ok and "S3" in lines[-1]


def test_forbidden_reason_verbatim_and_needle_in_failing_turn(start):
    base, _ = start()
    report = run_to_end(base, "v1")["report"]
    personas = {p["id"]: p for p in generate_personas(SOP, 20, None).personas}
    forbidden = [c for c in report["calls"] if c["failure_kind"] == "forbidden"]
    assert len(forbidden) == 5
    for call in forbidden:
        assert call["reason"].startswith("agent said forbidden phrase: ")
        needle = call["reason"].rsplit(": ", 1)[1].strip("'\"")
        assert needle in personas[call["persona_id"]]["must_not_include"]
        assert needle.lower() in call["transcript"][call["failing_turn"]]["content"].lower()


def test_utf8_survives_the_wire(start):
    base, _ = start()
    report = run_to_end(base, "v1", SOP)["report"]
    leak = next(c for c in report["calls"] if c["persona_id"] == "brother_asks")
    assert "12,400" in leak["transcript"][1]["content"]
    _, text = request(base, "/api/config")
    assert "₹" in json.loads(text)["sample_sop"]


def test_offline_run_under_10_seconds(start):
    base, _ = start()
    started = time.time()
    body = run_to_end(base, "v1")
    assert body["status"] == "done" and time.time() - started < 10


def raw_request(srv, method, path, headers, body=b""):
    import http.client
    conn = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=10)
    conn.putrequest(method, path, skip_host=True, skip_accept_encoding=True)
    for name, value in headers.items():
        conn.putheader(name, value)
    conn.endheaders(body)
    resp = conn.getresponse()
    status, text = resp.status, resp.read().decode("utf-8")
    conn.close()
    return status, text


def test_foreign_host_is_403_on_get_and_post(start):
    base, srv = start()
    port = srv.server_address[1]
    body = json.dumps({"sop": SOP, "agent": "v1"}).encode()
    json_headers = {"Content-Type": "application/json", "Content-Length": str(len(body))}
    for path in ("/", "/api/config", "/api/runs"):
        assert raw_request(srv, "GET", path, {"Host": f"evil.example:{port}"})[0] == 403
    post = {**json_headers, "Host": f"evil.example:{port}", "Origin": f"http://evil.example:{port}"}
    assert raw_request(srv, "POST", "/api/run", post, body)[0] == 403
    assert raw_request(srv, "POST", "/api/run", {**json_headers, "Host": f"evil.example:{port}"}, body)[0] == 403
    assert get_json(base, "/api/runs") == (200, [])  # nothing was started


def test_foreign_origin_is_403_on_post(start):
    _, srv = start()
    port = srv.server_address[1]
    body = json.dumps({"sop": SOP, "agent": "v1"}).encode()
    headers = {"Content-Type": "application/json", "Content-Length": str(len(body)), "Host": f"127.0.0.1:{port}",
               "Origin": "http://evil.example"}
    assert raw_request(srv, "POST", "/api/run", headers, body)[0] == 403


def test_loopback_hosts_and_origins_still_work(start):
    base, srv = start()
    port = srv.server_address[1]
    for host in (f"127.0.0.1:{port}", f"localhost:{port}"):
        assert raw_request(srv, "GET", "/api/config", {"Host": host})[0] == 200
    body = json.dumps({"sop": SOP, "agent": "v1"}).encode()
    headers = {"Content-Type": "application/json", "Content-Length": str(len(body)),
               "Host": f"localhost:{port}", "Origin": f"http://localhost:{port}"}
    status, text = raw_request(srv, "POST", "/api/run", headers, body)
    assert status == 202
    assert wait_finished(base, json.loads(text)["run_id"])["status"] == "done"


def test_negative_content_length_is_400(start):
    _, srv = start()
    headers = {"Host": f"127.0.0.1:{srv.server_address[1]}", "Content-Type": "application/json",
               "Content-Length": "-1"}
    status, text = raw_request(srv, "POST", "/api/run", headers)
    assert status == 400 and "Content-Length" in json.loads(text)["error"]


@pytest.mark.parametrize("agent", [{}, ["v1"], 5, None])
def test_non_string_agent_is_400(start, agent):
    base, _ = start()
    status, _ = request(base, "/api/run", {"sop": "abc", "agent": agent})
    assert status == 400


def test_server_module_never_imports_request_text():
    source = Path(server.__file__).read_text(encoding="utf-8")
    assert "import_module" not in source and "__import__" not in source and "eval(" not in source
