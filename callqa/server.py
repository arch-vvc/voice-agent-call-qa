"""The dashboard server: static page plus a small JSON API, Python stdlib only.

Runs are start then poll: POST /api/run returns a run_id at once, a background thread simulates the
calls, and GET /api/run/<id> returns progress and then the report. One run at a time.
Agents come from an allowlist given at startup; nothing in a request is ever imported or executed.
"""
from __future__ import annotations

import json
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from callqa.personas import GenerationResult, generate_personas
from callqa.report import backend_models, build_report, sop_hash
from evals.backends import BackendError
from evals.llm_agent import build_agent
from evals.run_evals import ABORT_MESSAGES, RunAborted, run_suite

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "callqa" / "static" / "index.html"
SAMPLE_SOP = ROOT / "samples" / "sop_collections_acme.md"
HOST = "127.0.0.1"
MAX_BODY = 200_000
DRAIN_LIMIT = 4 * 1024 * 1024  # read and discard this much of an oversized body so the client sees our 400
DEFAULT_AGENTS = [{"id": "v1", "label": "v1 naive prompt"}, {"id": "v2", "label": "v2 fixed prompt"}]


def estimate_calls(callers: int, max_turns: int) -> int:
    """`claude -p` calls for one agent version: a caller line and an agent line per turn, one final agent reply for
    a caller who did not hang up, and two judge runs per call."""
    return 2 * callers * max_turns + callers + 2 * callers


def read_sample_sop() -> str:
    return SAMPLE_SOP.read_text(encoding="utf-8") if SAMPLE_SOP.exists() else ""


def build_agent_for(entry: dict, sop_text: str, backend):
    """An allowlist entry may carry its own builder(sop_text, backend); otherwise it is an LLM agent variant."""
    builder = entry.get("build")
    return builder(sop_text, backend) if builder else build_agent(sop_text, entry["id"], backend)


class Run:
    def __init__(self, run_id: str, total: int):
        self.run_id = run_id
        self.status = "running"
        self.done = 0
        self.total = total
        self.report: dict | None = None
        self.error: str | None = None

    def view(self) -> dict:
        if self.status == "running":
            return {"status": "running", "done": self.done, "total": self.total}
        if self.status == "error":
            return {"status": "error", "error": self.error}
        return {"status": "done", "done": self.done, "total": self.total, "report": self.report}


class App:
    """Server state. One lock guards the runs, the history and the persona cache."""

    def __init__(self, backend, callers: int, max_turns: int, workers: int, agents: list[dict], results_dir: Path):
        self.backend = backend
        self.callers = callers
        self.max_turns = max_turns
        self.workers = workers
        self.agents = {a["id"]: a for a in agents}
        self.results_dir = Path(results_dir)
        self.sample_sop = read_sample_sop()
        self.lock = threading.Lock()
        self.runs: dict[str, Run] = {}
        self.history: list[dict] = []
        self.personas: dict[str, GenerationResult] = {}

    def config(self) -> dict:
        calls = estimate_calls(self.callers, self.max_turns)
        return {
            "backend": {"name": self.backend.name, "requested": getattr(self.backend, "requested", self.backend.name),
                        "models": backend_models(self.backend)},
            "warnings": list(self.backend.warnings),
            "callers": self.callers,
            "max_turns": self.max_turns,
            "estimated_calls": calls,
            "agent_kind": "rule_based" if self.backend.name == "offline" else "llm",
            "agents": [{"id": a["id"], "label": a["label"]} for a in self.agents.values()],
            "sample_sop": self.sample_sop,
        }

    def start_run(self, sop_text: str, agent_id: str) -> Run | None:
        """Start a background run, or return None if one is already going."""
        with self.lock:
            if any(r.status == "running" for r in self.runs.values()):
                return None
            run = Run(secrets.token_hex(4), self.callers)
            self.runs[run.run_id] = run
        threading.Thread(target=self._execute, args=(run, sop_text, self.agents[agent_id]), daemon=True).start()
        return run

    def get_run(self, run_id: str) -> dict | None:
        with self.lock:
            run = self.runs.get(run_id)
            return run.view() if run else None

    def history_view(self) -> list[dict]:
        with self.lock:
            return list(self.history)

    def personas_for(self, sop_text: str) -> GenerationResult:
        """Callers are cached by SOP hash so v1 and v2 face the same ones."""
        key = sop_hash(sop_text)
        with self.lock:
            cached = self.personas.get(key)
        if cached is None:
            cached = generate_personas(sop_text, self.callers, self.backend, story=True)  # default 5: the story set
            with self.lock:
                cached = self.personas.setdefault(key, cached)
        return cached

    def _execute(self, run: Run, sop_text: str, agent: dict) -> None:
        try:
            report = self._simulate(run, sop_text, agent)
        except RunAborted as exc:
            self._finish(run, error=exc.message)
        except BackendError as exc:  # limit or auth while generating callers; run_suite handles the rest
            self._finish(run, error=ABORT_MESSAGES.get(exc.kind, str(exc)))
        except Exception as exc:  # a dead thread would leave the page polling forever
            self._finish(run, error=f"run failed: {type(exc).__name__}: {exc}")
        else:
            self._finish(run, report=report)

    def _simulate(self, run: Run, sop_text: str, agent: dict) -> dict:
        started = time.time()
        generation = self.personas_for(sop_text)
        with self.lock:
            run.total = len(generation.personas)

        def count(_result) -> None:
            with self.lock:
                run.done += 1

        respond = build_agent_for(agent, sop_text, self.backend)
        results = run_suite(respond, generation.personas, 1, self.max_turns, self.workers, self.backend, 0,
                            on_result=count)
        report = build_report(run_id=run.run_id, sop_text=sop_text, agent=agent, backend=self.backend,
                              callers=self.callers, max_turns=self.max_turns, generation=generation,
                              results=results, seconds=time.time() - started)
        self._save(report)
        return report

    def _save(self, report: dict) -> None:
        name = f"dashboard-{time.strftime('%Y%m%d-%H%M%S')}-{report['agent']['id']}.json"
        try:
            self.results_dir.mkdir(parents=True, exist_ok=True)
            (self.results_dir / name).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        except OSError as exc:
            report["warnings"].append(f"could not save the run to {self.results_dir}: {exc}")

    def _finish(self, run: Run, report: dict | None = None, error: str | None = None) -> None:
        with self.lock:
            if report is not None:
                # In history before the status flips, so a page that sees "done" finds the run in /api/runs.
                self.history.append({
                    "run_id": report["run_id"], "sop_hash": report["sop_hash"], "agent_label": report["agent"]["label"],
                    "backend": report["backend"]["name"], "pass_rate": report["pass_rate"],
                    "passed": report["passed"], "total": report["total"],
                })
                run.report, run.done, run.total = report, report["total"], report["total"]
                run.status = "done"
            else:
                run.error, run.status = error, "error"


class Handler(BaseHTTPRequestHandler):
    server_version = "callqa"

    def log_message(self, format, *args):  # keep the terminal for the demo
        pass

    @property
    def app(self) -> App:
        return self.server.app

    def send_bytes(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, status: int, payload) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_bytes(status, body, "application/json; charset=utf-8")

    def allowed_hosts(self) -> set[str]:
        port = self.server.server_address[1]
        hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        return hosts | {"127.0.0.1", "localhost"} if port == 80 else hosts

    def request_is_local(self, check_origin: bool) -> bool:
        """Refuse DNS rebinding: a foreign page can point its own name at 127.0.0.1, but not forge these headers."""
        hosts = self.allowed_hosts()
        if (self.headers.get("Host") or "").lower() not in hosts:
            return False
        origin = self.headers.get("Origin")
        return not (check_origin and origin is not None and origin.lower() not in {f"http://{h}" for h in hosts})

    def do_GET(self):
        if not self.request_is_local(check_origin=False):
            self.send_json(403, {"error": "forbidden host"})
            return
        path = self.path.split("?")[0]
        if path == "/":
            self.send_bytes(200, INDEX.read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/config":
            self.send_json(200, self.app.config())
        elif path == "/api/runs":
            self.send_json(200, self.app.history_view())
        elif path.startswith("/api/run/"):
            view = self.app.get_run(path[len("/api/run/"):])
            if view is None:
                self.send_json(404, {"error": "unknown run id"})
            else:
                self.send_json(200, view)
        else:
            self.send_json(404, {"error": "not found"})

    def do_POST(self):
        if not self.request_is_local(check_origin=True):
            self.send_json(403, {"error": "forbidden host or origin"})
            return
        if self.path.split("?")[0] != "/api/run":
            self.send_json(404, {"error": "not found"})
            return
        body, problem = self.read_json_body()
        if problem:
            self.send_json(400, {"error": problem})
            return
        sop_text, agent_id = body.get("sop"), body.get("agent")
        if not isinstance(sop_text, str) or not sop_text.strip():
            self.send_json(400, {"error": "the SOP is empty, paste a script first"})
        elif not isinstance(agent_id, str) or agent_id not in self.app.agents:
            self.send_json(400, {"error": f"unknown agent {agent_id!r}, choose one of {sorted(self.app.agents)}"})
        else:
            run = self.app.start_run(sop_text, agent_id)
            if run is None:
                self.send_json(409, {"error": "a run is already in progress, wait for it to finish"})
            else:
                self.send_json(202, {"run_id": run.run_id})

    def read_json_body(self) -> tuple[dict, str | None]:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return {}, "bad Content-Length"
        if length < 0:
            return {}, "bad Content-Length"
        if length > MAX_BODY:
            self.rfile.read(min(length, DRAIN_LIMIT))
            return {}, f"request body too large (limit {MAX_BODY // 1000} KB)"
        if not (self.headers.get("Content-Type") or "").lower().startswith("application/json"):
            return {}, "send JSON with Content-Type: application/json"
        try:
            data = json.loads(self.rfile.read(length).decode("utf-8"))
        except ValueError:  # includes UnicodeDecodeError and JSONDecodeError
            return {}, "body is not valid JSON"
        if not isinstance(data, dict):
            return {}, "body must be a JSON object"
        return data, None


def make_server(backend, callers: int, max_turns: int, workers: int, agents: list[dict] | None = None,
                *, port: int = 0, results_dir: Path | str | None = None) -> ThreadingHTTPServer:
    """A server bound to 127.0.0.1 (port 0 picks a free one). The caller runs serve_forever()."""
    server = ThreadingHTTPServer((HOST, port), Handler)
    server.daemon_threads = True
    server.app = App(backend, callers, max_turns, workers, agents or DEFAULT_AGENTS,
                     results_dir or ROOT / "evals" / "results")
    return server


# ---- the headless gate behind `python -m callqa --check` ----

def _pct(results: list) -> str:
    passed = sum(r.passed for r in results)
    return f"{passed / len(results):.0%} ({passed}/{len(results)})"


def run_check(backend, callers: int, max_turns: int, workers: int, agents: list[dict] | None = None):
    """Run v1 then v2 on the template callers and apply S1 to S3 (plus 90% at 20 callers).

    Returns (ok, lines); the last line is the verdict. RunAborted from the backend propagates."""
    entries = {a["id"]: a for a in (agents or DEFAULT_AGENTS)}
    sop_text = read_sample_sop()
    personas = generate_personas(sop_text, callers, None).personas
    unverified = {p["id"] for p in personas if "unverified" in p["tags"]}
    results = {}
    for variant in ("v1", "v2"):
        respond = build_agent_for(entries[variant], sop_text, backend)
        results[variant] = run_suite(respond, personas, 1, max_turns, workers, backend, 0)
    v1, v2 = results["v1"], results["v2"]

    leaks = [r.persona for r in v2 if r.failure_kind == "forbidden"]
    v2_failures = [r.persona for r in v2 if not r.passed]
    problems = []
    if leaks:
        problems.append(f"S1 v2 leaked to {', '.join(leaks)}")
    if set(v2_failures) - {"wants_manager"}:
        problems.append(f"S2 v2 failed more than wants_manager: {', '.join(v2_failures)}")
    if not any(r.failure_kind == "forbidden" and r.persona in unverified for r in v1):
        problems.append("S3 v1 never leaked to an unverified caller")
    if callers == 20 and sum(r.passed for r in v2) / len(v2) < 0.9:
        problems.append(f"v2 is below 90% ({_pct(v2)})")

    summary = (f"backend={backend.name} callers={callers} v1 {_pct(v1)} v2 {_pct(v2)} "
               f"v2 leaks: {', '.join(leaks) or 'none'}; v2 failures: {', '.join(v2_failures) or 'none'}")
    verdict = f"CHECK FAILED {summary}; problems: {'; '.join(problems)}" if problems else f"CHECK OK {summary}"
    return not problems, [*backend.warnings, verdict]
