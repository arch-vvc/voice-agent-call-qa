"""Vercel entry point for the Call QA demo. A stateless WSGI wrapper around the real dashboard code.

Vercel runs each request on a fresh, short lived instance, so the local server's "start a background run then
poll it" design cannot work as is. The offline backend finishes a 20 caller run in well under a second, so here
the run id carries the SOP and agent (compressed), and GET /api/run/<id> simply runs it again: the offline
backend is deterministic, so the report is identical. The before and after history lives in a small cookie.
Nothing is stored on the server and no model is ever called.
"""
import base64
import hashlib
import json
import os
import sys
import tempfile
import zlib
from http.cookies import SimpleCookie
from pathlib import Path
from urllib.parse import parse_qs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["CALLQA_BACKEND"] = "offline"

from callqa.server import DEFAULT_AGENTS, INDEX, MAX_BODY, App, Run  # noqa: E402
from evals.backends import get_backend  # noqa: E402
from evals.run_evals import RUN_DEFAULTS  # noqa: E402

MAX_TOKEN = 7000  # characters of run id, which keeps the URL short enough for any proxy
MAX_HISTORY = 8
COOKIE = "callqa_history"

_backend = get_backend("offline")
_defaults = RUN_DEFAULTS["offline"]


def make_app() -> App:
    return App(_backend, 20, _defaults["max_turns"], _defaults["workers"], DEFAULT_AGENTS,
               Path(tempfile.gettempdir()) / "callqa-results")


def encode_run(sop: str, agent: str) -> str:
    raw = zlib.compress(json.dumps({"s": sop, "a": agent}).encode("utf-8"), 9)
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_run(token: str) -> dict:
    padded = token + "=" * (-len(token) % 4)
    data = json.loads(zlib.decompress(base64.urlsafe_b64decode(padded.encode("ascii"))).decode("utf-8"))
    if not isinstance(data.get("s"), str) or not isinstance(data.get("a"), str):
        raise ValueError("bad run id")
    return data


def run_report(token: str) -> dict:
    spec = decode_run(token)
    app = make_app()
    if spec["a"] not in app.agents:
        raise ValueError("unknown agent")
    run_id = hashlib.sha1(token.encode("ascii")).hexdigest()[:8]
    return app._simulate(Run(run_id, app.callers), spec["s"], app.agents[spec["a"]])


def history_entry(report: dict) -> dict:
    return {"run_id": report["run_id"], "sop_hash": report["sop_hash"], "agent_label": report["agent"]["label"],
            "backend": report["backend"]["name"], "pass_rate": report["pass_rate"],
            "passed": report["passed"], "total": report["total"]}


def read_history(environ) -> list:
    jar = SimpleCookie(environ.get("HTTP_COOKIE", ""))
    if COOKIE not in jar:
        return []
    try:
        raw = base64.urlsafe_b64decode(jar[COOKIE].value + "=" * (-len(jar[COOKIE].value) % 4))
        items = json.loads(raw.decode("utf-8"))
        return items if isinstance(items, list) else []
    except ValueError:
        return []


def history_cookie(items: list) -> str:
    raw = base64.urlsafe_b64encode(json.dumps(items[-MAX_HISTORY:]).encode("utf-8")).decode("ascii").rstrip("=")
    return f"{COOKIE}={raw}; Path=/; Max-Age=3600; HttpOnly; Secure; SameSite=Lax"


def request_path(environ) -> str:
    """The path the visitor asked for. vercel.json rewrites every request to this file as ?p=<path>, because the
    function itself only sees its own address."""
    wanted = parse_qs(environ.get("QUERY_STRING", "")).get("p", [""])[0]
    return "/" + wanted.lstrip("/")


def respond(start_response, status: int, body: bytes, content_type: str, extra=()):
    reason = {200: "OK", 202: "Accepted", 400: "Bad Request", 404: "Not Found"}.get(status, "Error")
    headers = [("Content-Type", content_type), ("Content-Length", str(len(body))), ("Cache-Control", "no-store"),
               ("X-Content-Type-Options", "nosniff"), *extra]
    start_response(f"{status} {reason}", headers)
    return [body]


def reply_json(start_response, status: int, payload, extra=()):
    return respond(start_response, status, json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8", extra)


def app(environ, start_response):
    method = environ["REQUEST_METHOD"]
    path = request_path(environ)
    if method == "GET" and path in ("/", "/index.html"):
        return respond(start_response, 200, INDEX.read_bytes(), "text/html; charset=utf-8")
    if method == "GET" and path == "/api/config":
        return reply_json(start_response, 200, make_app().config())
    if method == "GET" and path == "/api/runs":
        return reply_json(start_response, 200, read_history(environ))
    if method == "GET" and path.startswith("/api/run/"):
        token = path[len("/api/run/"):]
        if len(token) > MAX_TOKEN:
            return reply_json(start_response, 400, {"error": "that run id is too long"})
        try:
            report = run_report(token)
        except Exception:
            return reply_json(start_response, 404, {"error": "unknown run id"})
        return reply_json(start_response, 200, {"status": "done", "done": report["total"],
                                                "total": report["total"], "report": report})
    if method == "POST" and path == "/api/run":
        try:
            length = int(environ.get("CONTENT_LENGTH") or 0)
        except ValueError:
            length = -1
        if length < 0 or length > MAX_BODY:
            return reply_json(start_response, 400, {"error": f"request body too large (limit {MAX_BODY // 1000} KB)"})
        if not (environ.get("CONTENT_TYPE") or "").lower().startswith("application/json"):
            return reply_json(start_response, 400, {"error": "send JSON with Content-Type: application/json"})
        try:
            body = json.loads(environ["wsgi.input"].read(length).decode("utf-8"))
        except ValueError:
            return reply_json(start_response, 400, {"error": "body is not valid JSON"})
        sop, agent = (body.get("sop"), body.get("agent")) if isinstance(body, dict) else (None, None)
        if not isinstance(sop, str) or not sop.strip():
            return reply_json(start_response, 400, {"error": "the SOP is empty, paste a script first"})
        if agent not in {a["id"] for a in DEFAULT_AGENTS}:
            return reply_json(start_response, 400, {"error": f"unknown agent {agent!r}, choose v1 or v2"})
        token = encode_run(sop, agent)
        if len(token) > MAX_TOKEN:
            return reply_json(start_response, 400, {"error": "this online demo takes a SOP of about 8,000 "
                                                             "characters or fewer. Run the repo locally for longer ones."})
        report = run_report(token)
        history = [h for h in read_history(environ) if h.get("run_id") != report["run_id"]]
        history.append(history_entry(report))
        return reply_json(start_response, 202, {"run_id": token},
                          [("Set-Cookie", history_cookie(history))])
    return reply_json(start_response, 404, {"error": "not found"})
