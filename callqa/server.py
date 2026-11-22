"""The dashboard server: static page plus a small JSON API, Python stdlib only.

Runs are start then poll: POST /api/run returns a run_id at once, a background thread simulates the
calls, and GET /api/run/<id> returns progress and then the report. One run at a time.
Agents come from an allowlist given at startup; nothing in a request is ever imported or executed.
"""
from __future__ import annotations

import functools
import json
import re
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

from callqa.calibrate import summary as label_summary
from callqa.labels import LabelError, append_jsonl, validate_label
from callqa.metering import MeteredBackend
from callqa.persona_sets import SET_NAMES, personas_for_set
from callqa.personas import GenerationResult, generate_personas
from callqa.report import backend_models, build_report, sop_hash
from callqa.runrecord import save_record
from callqa.sop import parse_sop
from evals.backends import BackendError
from evals.llm_agent import build_agent, build_system_prompt
from evals.run_evals import ABORT_MESSAGES, RunAborted, run_suite

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "callqa" / "static" / "index.html"
RECORDS_PAGE = ROOT / "callqa" / "static" / "records.html"
SAMPLE_SOP = ROOT / "samples" / "sop_collections_acme.md"
SAMPLE_RECORDS = ROOT / "samples" / "run_records"
LABELS_FILE = ROOT / "evals" / "labels" / "human_labels.jsonl"
RECORD_NAME = re.compile(r"[A-Za-z0-9_.@-]+\.json")
MAX_RECORD_BYTES = 10 * 1024 * 1024  # a bigger file in a records folder is skipped, never read
HOST = "127.0.0.1"
MAX_BODY = 200_000
DRAIN_LIMIT = 4 * 1024 * 1024  # read and discard this much of an oversized body so the client sees our 400
DEFAULT_AGENTS = [
    {"id": "v1", "label": "v1 naive prompt"},
    {"id": "v2", "label": "v2 fixed prompt"},
    {"id": "v3", "label": "v3 escalation and Hinglish fix"},
]


MAX_DASHBOARD_REPEATS = 5
LARGE_RUN_CALLS = 200  # on an LLM backend a run estimated above this needs --yes


def rules_summary(rules) -> list[dict]:
    """What the page needs to label a rule it has no title for (a passing rule): id, title, severity, verified."""
    return [{"id": r.id, "title": r.title, "severity": r.severity, "verified": r.verified} for r in rules.rules]


def estimate_calls(callers: int, max_turns: int, repeats: int = 1) -> int:
    """`claude -p` calls for one agent version: a caller line and an agent line per turn, one final agent reply for
    a caller who did not hang up, and two judge runs per call, for every repeat: (2T + 3) x C x k."""
    return (2 * max_turns + 3) * callers * repeats


def needs_confirmation(backend, estimated_calls: int) -> bool:
    """True for a run that would spend more than LARGE_RUN_CALLS model calls. Offline spends none."""
    return backend.name != "offline" and estimated_calls > LARGE_RUN_CALLS


@functools.lru_cache(maxsize=1)
def full_set_size() -> int:
    return len(personas_for_set(read_sample_sop(), "full").personas)


def caller_count(set_name: str, callers: int) -> int:
    """How many callers a run of this set has: `full` is all of its callers whatever `callers` says."""
    return full_set_size() if set_name == "full" else callers


def record_summary(name: str, record: dict) -> dict:
    """One row of GET /api/records. Tolerant: a record of a kind this code does not know still lists."""
    agent = record.get("agent") if isinstance(record.get("agent"), dict) else {}
    persona_set = record.get("persona_set") if isinstance(record.get("persona_set"), dict) else {}
    backend = record.get("backend") if isinstance(record.get("backend"), dict) else {}
    return {"name": name, "kind": record.get("kind"), "agent": agent.get("label") or agent.get("id"),
            "set": persona_set.get("name"), "repeats": record.get("repeats"), "backend": backend.get("name"),
            "pass_rate": record.get("pass_rate"), "created_at": record.get("created_at"),
            "synthetic": record.get("synthetic")}


def read_sample_sop() -> str:
    return SAMPLE_SOP.read_text(encoding="utf-8") if SAMPLE_SOP.exists() else ""


def build_agent_for(entry: dict, sop_text: str, backend):
    """An allowlist entry may carry its own builder(sop_text, backend); otherwise it is an LLM agent variant."""
    builder = entry.get("build")
    return builder(sop_text, backend) if builder else build_agent(sop_text, entry["id"], backend)


def agent_prompt_for(entry: dict, sop_text: str) -> str:
    """The system prompt of a v1, v2 or v3 agent (so offline runs can count its input tokens); "" for any other."""
    if entry.get("build") or entry["id"] not in {"v1", "v2", "v3"}:
        return ""
    return build_system_prompt(parse_sop(sop_text), entry["id"])


def metered_agent(entry: dict, sop_text: str, backend):
    """(respond, system prompt) with the agent's own backend metered as the "agent" role."""
    return build_agent_for(entry, sop_text, MeteredBackend(backend, "agent")), agent_prompt_for(entry, sop_text)


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

    def __init__(self, backend, callers: int, max_turns: int, workers: int, agents: list[dict], results_dir: Path,
                 set_name: str = "story", repeats: int = 1, allow_large: bool = False,
                 samples_dir: Path | None = None, labels_path: Path | None = None, rules=None):
        self.backend = backend
        self.rules = rules  # a callqa.compliance.Rules, or None for no compliance layer
        self.callers = callers
        self.set_name = set_name
        self.repeats = repeats
        self.allow_large = allow_large
        self.max_turns = max_turns
        self.workers = workers
        self.agents = {a["id"]: a for a in agents}
        self.results_dir = Path(results_dir)
        self.samples_dir = Path(samples_dir) if samples_dir else SAMPLE_RECORDS
        self.labels_path = Path(labels_path) if labels_path else LABELS_FILE
        self.sample_sop = read_sample_sop()
        self.lock = threading.Lock()
        self.runs: dict[str, Run] = {}
        self.history: list[dict] = []
        self.personas: dict[str, GenerationResult] = {}

    def config(self) -> dict:
        calls = estimate_calls(caller_count(self.set_name, self.callers), self.max_turns, self.repeats)
        config = {
            "backend": {"name": self.backend.name, "requested": getattr(self.backend, "requested", self.backend.name),
                        "models": backend_models(self.backend)},
            "warnings": list(self.backend.warnings),
            "callers": self.callers,
            "set": self.set_name,
            "repeats": self.repeats,
            "max_turns": self.max_turns,
            "estimated_calls": calls,
            "agent_kind": "rule_based" if self.backend.name == "offline" else "llm",
            "agents": [{"id": a["id"], "label": a["label"]} for a in self.agents.values()],
            "sample_sop": self.sample_sop,
        }
        if self.rules is not None:  # left out with --rules none, so the layer being off is the absence of the keys
            config["rules_file"] = self.rules.label
            config["rules"] = rules_summary(self.rules)
        return config

    def start_run(self, sop_text: str, agent_id: str, set_name: str | None = None,
                  repeats: int | None = None) -> Run | None:
        """Start a background run, or return None if one is already going. `set_name` and `repeats` default to
        what the server was started with."""
        set_name, repeats = set_name or self.set_name, repeats or self.repeats
        with self.lock:
            if any(r.status == "running" for r in self.runs.values()):
                return None
            run = Run(secrets.token_hex(4), caller_count(set_name, self.callers) * repeats)
            self.runs[run.run_id] = run
        threading.Thread(target=self._execute, args=(run, sop_text, self.agents[agent_id], set_name, repeats),
                         daemon=True).start()
        return run

    def get_run(self, run_id: str) -> dict | None:
        with self.lock:
            run = self.runs.get(run_id)
            return run.view() if run else None

    def record_files(self) -> dict[str, Path]:
        """name -> file for every readable JSON object in runs/ and then the sample records. A run wins over a
        sample of the same name. This is the only place a record name becomes a path."""
        found: dict[str, Path] = {}
        for folder in (self.results_dir, self.samples_dir):
            try:
                entries = sorted(folder.iterdir())
            except OSError:
                continue
            for path in entries:
                name = path.name
                if (name in found or name.startswith(".") or not RECORD_NAME.fullmatch(name) or path.is_symlink()
                        or not path.is_file() or path.stat().st_size > MAX_RECORD_BYTES):
                    continue
                if isinstance(self.read_record(path), dict):
                    found[name] = path
        return found

    @staticmethod
    def read_record(path: Path):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def records_view(self) -> list[dict]:
        rows = [record_summary(name, self.read_record(path)) for name, path in self.record_files().items()]
        return sorted(rows, key=lambda r: (str(r["created_at"] or ""), r["name"]), reverse=True)

    def record_view(self, name: str) -> dict | None:
        """The record called `name`, only if that exact name is in the listing."""
        if not RECORD_NAME.fullmatch(name):
            return None
        path = self.record_files().get(name)
        return self.read_record(path) if path else None

    def add_label(self, body: dict) -> dict:
        """Validate a label against the record it names and append it. Raises LabelError. The only file written is
        `labels_path`; the record name must be in the records listing, so it never becomes a path."""
        name = body.get("record") if isinstance(body, dict) else None
        record = self.record_view(name) if isinstance(name, str) else None
        if record is None:
            raise LabelError("unknown record")
        label = validate_label(body, record, name)
        append_jsonl(self.labels_path, label)
        return label

    def history_view(self) -> list[dict]:
        with self.lock:
            return list(self.history)

    def personas_for(self, sop_text: str, set_name: str = "story") -> GenerationResult:
        """Callers are cached by SOP hash and set so v1 and v2 face the same ones. `full` is always templates."""
        key = f"{sop_hash(sop_text)}:{set_name}"
        with self.lock:
            cached = self.personas.get(key)
        if cached is None:
            if set_name == "full":
                cached = personas_for_set(sop_text, "full")
            else:
                cached = generate_personas(sop_text, self.callers, self.backend, story=True)  # default 5: story set
            with self.lock:
                cached = self.personas.setdefault(key, cached)
        return cached

    def _execute(self, run: Run, sop_text: str, agent: dict, set_name: str, repeats: int) -> None:
        try:
            report = self._simulate(run, sop_text, agent, set_name, repeats)
        except RunAborted as exc:
            self._finish(run, error=exc.message)
        except BackendError as exc:  # limit or auth while generating callers; run_suite handles the rest
            self._finish(run, error=ABORT_MESSAGES.get(exc.kind, str(exc)))
        except Exception as exc:  # a dead thread would leave the page polling forever
            self._finish(run, error=f"run failed: {type(exc).__name__}: {exc}")
        else:
            self._finish(run, report=report)

    def _simulate(self, run: Run, sop_text: str, agent: dict, set_name: str, repeats: int) -> dict:
        started = time.time()
        generation = self.personas_for(sop_text, set_name)
        with self.lock:
            run.total = len(generation.personas) * repeats

        def count(_result) -> None:
            with self.lock:
                run.done += 1

        respond, prompt = metered_agent(agent, sop_text, self.backend)
        results = run_suite(respond, generation.personas, repeats, self.max_turns, self.workers, self.backend, 0,
                            on_result=count, agent_prompt=prompt, rules=self.rules, sop=parse_sop(sop_text))
        report = build_report(run_id=run.run_id, sop_text=sop_text, agent=agent, backend=self.backend,
                              callers=len(generation.personas), max_turns=self.max_turns, generation=generation,
                              results=results, seconds=time.time() - started, repeats=repeats,
                              set_name="full" if set_name == "full" else None, metering=True,
                              synthetic=sop_text == self.sample_sop and generation.source.startswith("templates"),
                              rules=self.rules)
        self._save(report)
        return report

    def _save(self, report: dict) -> None:
        try:
            save_record(report, self.results_dir)
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
                    "set": report["persona_set"]["name"], "repeats": report["repeats"],
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
        elif path == "/records":
            self.send_bytes(200, RECORDS_PAGE.read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/records":
            self.send_json(200, self.app.records_view())
        elif path.startswith("/api/records/"):
            record = self.app.record_view(unquote(path[len("/api/records/"):]))
            if record is None:
                self.send_json(404, {"error": "unknown record"})
            else:
                self.send_json(200, record)
        elif path == "/api/config":
            self.send_json(200, self.app.config())
        elif path == "/api/labels/summary":
            self.send_json(200, label_summary(self.app.labels_path))
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
        route = self.path.split("?")[0]
        if route not in ("/api/run", "/api/labels"):
            self.send_json(404, {"error": "not found"})
            return
        body, problem = self.read_json_body(too_large_status=413 if route == "/api/labels" else 400)
        if problem:
            self.send_json(self.problem_status, {"error": problem})
            return
        if route == "/api/labels":
            self.post_label(body)
            return
        sop_text, agent_id = body.get("sop"), body.get("agent")
        set_name, repeats = body.get("set", self.app.set_name), body.get("repeats", self.app.repeats)
        problem = self.run_problem(set_name, repeats)
        if not isinstance(sop_text, str) or not sop_text.strip():
            self.send_json(400, {"error": "the SOP is empty, paste a script first"})
        elif not isinstance(agent_id, str) or agent_id not in self.app.agents:
            self.send_json(400, {"error": f"unknown agent {agent_id!r}, choose one of {sorted(self.app.agents)}"})
        elif problem:
            self.send_json(400, {"error": problem})
        else:
            run = self.app.start_run(sop_text, agent_id, set_name, repeats)
            if run is None:
                self.send_json(409, {"error": "a run is already in progress, wait for it to finish"})
            else:
                self.send_json(202, {"run_id": run.run_id})

    def post_label(self, body: dict) -> None:
        try:
            label = self.app.add_label(body)
        except LabelError as exc:
            self.send_json(exc.status, {"error": str(exc)})
        except OSError:  # no path in the message: it would show the machine's folders to the page
            self.send_json(500, {"error": "could not save the label: the labels folder is not writable"})
        else:
            self.send_json(200, {"ok": True, "label_id": label["label_id"]})

    def run_problem(self, set_name, repeats) -> str | None:
        """Why a requested set or repeats count is refused, or None."""
        if set_name not in SET_NAMES:
            return f"unknown set {set_name!r}, choose one of {', '.join(SET_NAMES)}"
        if not isinstance(repeats, int) or isinstance(repeats, bool) or not 1 <= repeats <= MAX_DASHBOARD_REPEATS:
            return f"repeats must be a whole number from 1 to {MAX_DASHBOARD_REPEATS}"
        app = self.app
        estimate = estimate_calls(caller_count(set_name, app.callers), app.max_turns, repeats)
        if needs_confirmation(app.backend, estimate) and not app.allow_large:
            return (f"this run would make about {estimate} model calls (over {LARGE_RUN_CALLS}); "
                    "restart the dashboard with --yes to allow it")
        return None

    def read_json_body(self, too_large_status: int = 400) -> tuple[dict, str | None]:
        """(body, problem). On a problem, `self.problem_status` is the HTTP status to answer with."""
        self.problem_status = 400
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return {}, "bad Content-Length"
        if length < 0:
            return {}, "bad Content-Length"
        if length > MAX_BODY:
            self.rfile.read(min(length, DRAIN_LIMIT))
            self.problem_status = too_large_status
            return {}, f"request body too large (limit {MAX_BODY // 1000} KB)"
        if not (self.headers.get("Content-Type") or "").lower().startswith("application/json"):
            return {}, "send JSON with Content-Type: application/json"
        try:
            data = json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, RecursionError):  # includes UnicodeDecodeError and JSONDecodeError; deep nesting
            return {}, "body is not valid JSON"
        if not isinstance(data, dict):
            return {}, "body must be a JSON object"
        return data, None


def make_server(backend, callers: int, max_turns: int, workers: int, agents: list[dict] | None = None,
                *, port: int = 0, results_dir: Path | str | None = None, set_name: str = "story", repeats: int = 1,
                allow_large: bool = False, samples_dir: Path | str | None = None,
                labels_path: Path | str | None = None, rules=None) -> ThreadingHTTPServer:
    """A server bound to 127.0.0.1 (port 0 picks a free one). The caller runs serve_forever()."""
    server = ThreadingHTTPServer((HOST, port), Handler)
    server.daemon_threads = True
    server.app = App(backend, callers, max_turns, workers, agents or DEFAULT_AGENTS,
                     results_dir or ROOT / "runs", set_name, repeats, allow_large, samples_dir, labels_path, rules)
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
