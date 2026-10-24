"""Pluggable LLM backends for the eval harness: one interface for the caller simulator, the agent
under test, the judge and the persona generator.

  offline     no model calls (scripted callers, phrase rule judge, rule based agents)
  claude-cli  `claude -p` on your Claude login, no API key
  anthropic   the SDK, only when ANTHROPIC_API_KEY is set

Pick one with get_backend(name), or leave name empty to use $CALLQA_BACKEND, then claude-cli
(which degrades to offline, with a warning, when `claude` is not on PATH).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Protocol

ABORT_KINDS = {"limit", "auth", "model"}  # these end a whole run; every other error only fails one call
ERROR_KINDS = {"limit", "auth", "model", "timeout", "failed", "unavailable"}
BACKEND_NAMES = ("offline", "claude-cli", "anthropic")
DEGRADE_WARNING = "claude CLI not found on PATH; running offline with rule based demo agents"

# role -> (env var, older env var that is still honoured, default model)
ROLE_MODELS = {
    "caller": ("CALLQA_CALLER_MODEL", "EVAL_SIM_MODEL", "claude-haiku-5-5"),
    "agent": ("CALLQA_AGENT_MODEL", None, "claude-sonnet-5-5"),
    "judge": ("JUDGE_MODEL", "EVAL_JUDGE_MODEL", "claude-sonnet-5-5"),
    "gen": ("CALLQA_GEN_MODEL", None, "claude-sonnet-5-5"),
}

# Matched case insensitively against the CLI's output, but only when the call failed.
LIMIT_MARKERS = ("usage limit", "rate limit", "limit reached", "hit your limit", "too many requests", r"\b429\b")
AUTH_MARKERS = (
    "not logged in", "/login", "invalid api key", "authentication", "failed to authenticate", "oauth",
    "session expired", "could not be refreshed",
)
MODEL_MARKERS = ("unrecognized_model",)

# The claude CLI takes aliases; it rejects full ids it does not know. Anthropic backend keeps full ids.
CLI_MODEL_ALIASES = (("claude-sonnet", "sonnet"), ("claude-haiku", "haiku"), ("claude-opus", "opus"))


def cli_model(model: str) -> str:
    """The name to give `claude --model`: a family alias for full ids, anything else unchanged."""
    for prefix, alias in CLI_MODEL_ALIASES:
        if model.startswith(prefix):
            return alias
    return model


class BackendError(Exception):
    """A backend call failed. `kind` is one of ERROR_KINDS; BackendError("limit") also works."""

    def __init__(self, message: str = "", kind: str | None = None):
        if kind is None:
            kind = message if message in ERROR_KINDS else "failed"
        super().__init__(message or kind)
        self.kind = kind


class LLMBackend(Protocol):
    name: str
    warnings: list[str]

    def complete(self, system: str, messages: list[dict], model: str, max_tokens: int) -> str: ...


def model_for(role: str) -> str:
    """Model id for a role ("caller", "agent", "judge", "gen"). Reads the environment at call time."""
    if role not in ROLE_MODELS:
        raise ValueError(f"unknown role {role!r}, expected one of {sorted(ROLE_MODELS)}")
    env, old_env, default = ROLE_MODELS[role]
    return os.environ.get(env) or (old_env and os.environ.get(old_env)) or default


class OfflineBackend:
    """Makes no model calls. Callers branch on `backend.name == "offline"` and use scripts and rules."""

    name = "offline"

    def __init__(self, warnings: list[str] | None = None, requested: str = "offline"):
        self.warnings = list(warnings or [])
        self.requested = requested

    def complete(self, system: str, messages: list[dict], model: str, max_tokens: int) -> str:
        raise BackendError("the offline backend makes no model calls", kind="unavailable")


class ClaudeCliBackend:
    """Runs `claude -p` once per call on the user's Claude login.

    Caller text, SOP text and model output are untrusted (the personas include prompt injection), so
    none of it goes in argv. The system prompt and the transcript go on stdin, and the child runs in
    a fresh empty temp dir so no CLAUDE.md or .claude/ is picked up. ANTHROPIC_API_KEY is removed from
    the child env so the call uses the login, not API billing. The CLI has no max_tokens flag, so
    max_tokens is ignored and the system prompt keeps replies short.
    """

    name = "claude-cli"

    def __init__(self, runner=subprocess.run, which=shutil.which, requested: str = "claude-cli"):
        self.runner = runner
        self.which = which
        self.requested = requested
        self.warnings: list[str] = []

    def complete(self, system: str, messages: list[dict], model: str, max_tokens: int) -> str:
        for attempt in range(2):  # at most one retry, and never for limit, auth or a missing binary
            try:
                return self._call(system, messages, model)
            except BackendError as exc:
                if attempt == 1 or exc.kind in ABORT_KINDS or exc.kind == "unavailable":
                    raise
        raise AssertionError("unreachable")  # pragma: no cover

    def _call(self, system: str, messages: list[dict], model: str) -> str:
        model = cli_model(model)
        base = [
            "claude", "-p", "--model", model, "--output-format", "json",
            "--tools", "", "--disallowedTools", "mcp__*", "--strict-mcp-config", "--disable-slash-commands",
            "--no-session-persistence", "--setting-sources", "local", "--safe-mode",
        ]
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        timeout = float(os.environ.get("CALLQA_CLI_TIMEOUT") or 90)
        transcript = render_transcript(messages)
        with tempfile.TemporaryDirectory(prefix="callqa-") as workdir:
            system_file = Path(workdir, "system.txt")
            system_file.write_text(system, encoding="utf-8")
            proc = self._run(base + ["--system-prompt-file", str(system_file)], transcript, workdir, env, timeout)
            if proc.returncode != 0 and "unknown option '--system-prompt-file'" in (proc.stderr or ""):
                # Older CLI without the flag: the system prompt goes first on stdin instead.
                stdin = f"<system>\n{_defang(system)}\n</system>\n\n{transcript}"
                proc = self._run(base, stdin, workdir, env, timeout)
        return parse_cli_output(proc, model)

    def _run(self, argv: list[str], stdin: str, workdir: str, env: dict, timeout: float):
        try:
            return self.runner(argv, input=stdin, capture_output=True, text=True, timeout=timeout, cwd=workdir, env=env)
        except subprocess.TimeoutExpired:
            raise BackendError(f"claude -p timed out after {timeout:g}s", kind="timeout") from None
        except OSError as exc:
            raise BackendError(f"could not run claude: {exc}", kind="unavailable") from None


# Every character a reader could treat as a line break, not just "\n".
_BREAKS = re.compile("\r\n?|[\v\f\x1c-\x1e\x85  ]")


def _defang(text: str) -> str:
    """Stop untrusted text from opening or closing a <system> block (any spacing or case, any attributes)."""
    return re.sub(r"<\s*(/?)\s*system\b[^>]*>", r"[\1system]", text, flags=re.I)


def render_transcript(messages: list[dict]) -> str:
    """A labelled transcript. Continuation lines are indented so a caller cannot fake an ASSISTANT turn."""
    turns = []
    for m in messages:
        body = _defang(_BREAKS.sub("\n", m["content"])).replace("\n", "\n    ")
        turns.append(f"{m['role'].upper()}: {body}")
    return "\n\n".join(turns) + "\n\nWrite only the next assistant message.\n"


# A limit message can come back as a normal looking reply; match only at the very start of the reply.
LIMIT_REPLY = re.compile(r"(claude ai usage limit reached|you'?ve hit your (usage )?limit)", re.I)


def parse_cli_output(proc, model: str = "") -> str:
    """Return the reply text of a finished `claude -p --output-format json` run, or raise BackendError."""
    stdout, stderr = proc.stdout or "", proc.stderr or ""
    try:
        data = json.loads(stdout)
    except ValueError:
        data = None
    result = data.get("result") if isinstance(data, dict) else None
    failed = proc.returncode != 0 or not isinstance(result, str) or bool(data.get("is_error"))
    if not failed:
        if LIMIT_REPLY.match(result.strip()):
            raise BackendError(_snippet("usage limit", result, "", ""), kind="limit")
        return result.strip()

    # Not the whole JSON: fields like duration_ms or session_id can contain "429" by chance.
    parts = [stderr, result if isinstance(result, str) else ""]
    if not isinstance(data, dict):
        parts.append(stdout)
    blob = " ".join(parts).lower()
    if any(re.search(marker, blob) for marker in LIMIT_MARKERS):
        raise BackendError(_snippet("usage limit", result, stdout, stderr), kind="limit")
    if any(marker in blob for marker in MODEL_MARKERS):
        raise BackendError(
            f"the claude CLI does not recognise model {model or 'unknown'!r}; set CALLQA_AGENT_MODEL, "
            "CALLQA_CALLER_MODEL or JUDGE_MODEL to an alias such as sonnet or haiku, or use --backend offline",
            kind="model")
    if any(marker in blob for marker in AUTH_MARKERS):
        raise BackendError(_snippet("not logged in", result, stdout, stderr), kind="auth")
    raise BackendError(_snippet(f"claude -p failed (exit {proc.returncode})", result, stdout, stderr), kind="failed")


def _snippet(prefix: str, result, stdout: str, stderr: str) -> str:
    """Prefer the CLI's own readable `result` text, then stderr, then a prefix of the raw output."""
    if isinstance(result, str) and result.strip():
        text = result.strip()[:300]
    else:
        text = (stderr.strip() or stdout.strip())[:200]
    return f"{prefix}: {text}" if text else prefix


class AnthropicBackend:
    """The Anthropic SDK. The client is built lazily so tests can pass a fake one."""

    name = "anthropic"

    def __init__(self, client=None, requested: str = "anthropic"):
        self._client = client
        self.requested = requested
        self.warnings: list[str] = []

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError:
                raise BackendError("pip install anthropic to use --backend anthropic", kind="unavailable") from None
            self._client = anthropic.Anthropic(max_retries=1)
        return self._client

    def complete(self, system: str, messages: list[dict], model: str, max_tokens: int) -> str:
        extra = {"system": system} if system else {}
        try:
            resp = self._get_client().messages.create(model=model, max_tokens=max_tokens, messages=messages, **extra)
        except BackendError:
            raise
        except Exception as exc:
            raise BackendError(f"{type(exc).__name__}: {exc}", kind=_sdk_error_kind(exc)) from exc
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()


def _sdk_error_kind(exc: Exception) -> str:
    names = {cls.__name__ for cls in type(exc).__mro__}
    if "RateLimitError" in names:
        return "limit"
    if names & {"AuthenticationError", "PermissionDeniedError"}:
        return "auth"
    if "APITimeoutError" in names:
        return "timeout"
    return "failed"


def get_backend(name: str | None = None, *, runner=subprocess.run, which=shutil.which, client=None) -> LLMBackend:
    """Build a backend. `name` overrides $CALLQA_BACKEND; with neither, use claude-cli and degrade to
    offline (with a warning) when `claude` is missing. An explicit choice that cannot work is an error."""
    requested = (name or os.environ.get("CALLQA_BACKEND") or "").strip().lower()
    if not requested:
        if which("claude"):
            return ClaudeCliBackend(runner=runner, which=which)
        return OfflineBackend(warnings=[DEGRADE_WARNING], requested="claude-cli")
    if requested == "offline":
        return OfflineBackend()
    if requested == "claude-cli":
        if not which("claude"):
            raise BackendError(
                "claude CLI not found on PATH; install Claude Code or use --backend offline", kind="unavailable")
        return ClaudeCliBackend(runner=runner, which=which)
    if requested == "anthropic":
        if client is None and not os.environ.get("ANTHROPIC_API_KEY"):
            raise BackendError(
                "ANTHROPIC_API_KEY is not set; export it or use --backend claude-cli or offline", kind="unavailable")
        return AnthropicBackend(client=client)
    raise BackendError(f"unknown backend {requested!r}, choose one of {', '.join(BACKEND_NAMES)}", kind="unavailable")
