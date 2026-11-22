"""The human label store (schema C9): validate a label against a run record, then append it as one JSON line.

The client sends only seven fields. The judge fields are copied from the record here, so a client cannot forge them.
"""
from __future__ import annotations

import hashlib
import json
import secrets
import threading
from datetime import datetime
from pathlib import Path

try:
    import fcntl
except ImportError:  # Windows: the thread lock still holds inside one process
    fcntl = None

SCHEMA_VERSION = 1
MAX_LINE_BYTES = 2048
MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_REASON = 500
MAX_LABELLER = 40
# Byte budgets (size once written as JSON, so a Devanagari reason counts 3 bytes a character). Chosen so that the
# worst case label fits one 2 KB line: the fields the labeller does not control are cut or refused first.
MAX_REASON_BYTES = 900
MAX_LABELLER_BYTES = 60
MAX_JUDGE_REASON_BYTES = 160
MAX_PERSONA_ID_BYTES = 48
MAX_RECORD_NAME_BYTES = 110
MAX_RUN_ID_CHARS = 16
CLIENT_KEYS = ("record", "persona_id", "repeat", "human_pass", "human_failing_turn", "human_reason", "labeller")
# The failure kind a model judge decides (a passed call is also a judge verdict on an LLM backend). Every other kind
# (forbidden, missing, misheard, compliance, anything unknown) is a deterministic rule, so it is never "llm".
JUDGE_KINDS = {"judge"}

_LOCK = threading.Lock()


class LabelError(ValueError):
    """A label that is refused. `status` is the HTTP status the route answers with."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def dumps_line(obj) -> str:
    """JSON text with no raw line separator: json escapes the control characters, and U+0085, U+2028 and U+2029
    are escaped here, so str.splitlines() and a split on newline always agree on the number of lines."""
    text = json.dumps(obj, ensure_ascii=False)
    return text.replace("\x85", "\\u0085").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


def wire_bytes(text: str) -> int:
    """Bytes `text` takes inside a JSON line (escapes counted, the two surrounding quotes not)."""
    return len(dumps_line(text).encode("utf-8")) - 2


def _cut(text: str, limit: int) -> str:
    """`text` shortened until it takes at most `limit` bytes in a line."""
    text = text.encode("utf-8", "replace").decode("utf-8")  # a lone surrogate in a record becomes U+FFFD
    while wire_bytes(text) > limit:
        text = text[:-1]
    return text


def _damaged() -> LabelError:
    return LabelError("this record is damaged, there is nothing to label")


def _call_view(record: dict, persona_id: str, repeat: int) -> dict:
    """The judge's verdict and the transcript for one persona and repeat, as the record stored them."""
    calls = record.get("calls")
    if not isinstance(calls, list):
        raise _damaged()
    call = next((c for c in calls if isinstance(c, dict) and c.get("persona_id") == persona_id), None)
    if call is None:
        raise LabelError(f"unknown persona {persona_id!r} in this record")
    repeats = call.get("repeat_results") or []
    if not isinstance(repeats, list):
        raise _damaged()
    if repeats:
        if not 0 <= repeat < len(repeats):
            raise LabelError(f"repeat must be from 0 to {len(repeats) - 1}")
        view = repeats[repeat]
    else:
        if repeat != 0:
            raise LabelError("repeat must be 0: this record has one repeat")
        view = call
    if not isinstance(view, dict) or not isinstance(view.get("transcript") or [], list):
        raise _damaged()
    return view


def validate_label(body: dict, record: dict, record_name: str) -> dict:
    """Return the full C9 label for a client body, or raise LabelError. `record` is the run record named
    `record_name`; the caller has already checked that the name is in the records listing."""
    try:
        return _validate(body, record, record_name)
    except UnicodeError:  # a lone surrogate such as "\ud800" cannot be written as UTF 8
        raise LabelError("the text has a character that cannot be saved (a lone surrogate); remove it") from None


def _validate(body: dict, record: dict, record_name: str) -> dict:
    if not isinstance(body, dict):
        raise LabelError("body must be a JSON object")
    unknown = sorted(set(body) - set(CLIENT_KEYS))
    missing = [k for k in CLIENT_KEYS if k not in body]
    if unknown:
        raise LabelError(f"unknown keys: {', '.join(map(str, unknown))}")
    if missing:
        raise LabelError(f"missing keys: {', '.join(missing)}")
    if body["record"] != record_name:
        raise LabelError("record does not match")
    persona_id, repeat, human_pass = body["persona_id"], body["repeat"], body["human_pass"]
    turn, reason, labeller = body["human_failing_turn"], body["human_reason"], body["labeller"]
    if not isinstance(persona_id, str):
        raise LabelError("persona_id must be a string")
    if not _is_int(repeat):
        raise LabelError("repeat must be a whole number")
    if not isinstance(human_pass, bool):
        raise LabelError("human_pass must be true or false")
    if not isinstance(reason, str) or len(reason) > MAX_REASON:
        raise LabelError(f"human_reason must be text of at most {MAX_REASON} characters")
    if not isinstance(labeller, str) or not labeller.strip() or len(labeller) > MAX_LABELLER:
        raise LabelError(f"labeller must be a name of 1 to {MAX_LABELLER} characters")
    labeller = labeller.strip()
    if wire_bytes(reason) > MAX_REASON_BYTES:
        raise LabelError(f"human_reason is {wire_bytes(reason)} bytes once saved (non English text takes up to 3 "
                         f"bytes a character); the limit is {MAX_REASON_BYTES}, shorten human_reason")
    if wire_bytes(labeller) > MAX_LABELLER_BYTES:
        raise LabelError(f"labeller is {wire_bytes(labeller)} bytes once saved; the limit is "
                         f"{MAX_LABELLER_BYTES}, shorten labeller")
    if wire_bytes(persona_id) > MAX_PERSONA_ID_BYTES or wire_bytes(record_name) > MAX_RECORD_NAME_BYTES:
        raise LabelError("the persona id or the record name is too long to label")
    view = _call_view(record, persona_id, repeat)
    kind = view.get("failure_kind")
    if kind == "crash":
        raise LabelError("this call crashed, there is no judge verdict to compare")
    transcript = view.get("transcript") or []
    if turn is not None:
        if human_pass:
            raise LabelError("a pass label has no failing turn")
        if (not _is_int(turn) or not 0 <= turn < len(transcript) or not isinstance(transcript[turn], dict)
                or transcript[turn].get("role") != "assistant"):
            raise LabelError("the failing turn must be an agent turn of this call")
    backend = record.get("backend") if isinstance(record.get("backend"), dict) else {}
    judge_pass = bool(view.get("passed"))
    llm_verdict = backend.get("name") != "offline" and (judge_pass or kind in JUDGE_KINDS)
    judge_reason, run_id = view.get("reason"), record.get("run_id")
    label = {
        "schema_version": SCHEMA_VERSION, "kind": "label", "label_id": secrets.token_hex(4),
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "record": record_name, "run_id": _cut(str(run_id), MAX_RUN_ID_CHARS) if run_id is not None else None,
        "persona_id": persona_id, "repeat": repeat,
        "transcript_sha256": hashlib.sha256(
            json.dumps(transcript, ensure_ascii=False).encode("utf-8", "replace")).hexdigest(),
        "human_pass": human_pass, "human_failing_turn": turn, "human_reason": reason, "labeller": labeller,
        "judge_pass": judge_pass, "judge_failing_turn": view.get("failing_turn"),
        "judge_reason": _cut(judge_reason, MAX_JUDGE_REASON_BYTES) if isinstance(judge_reason, str) else None,
        "judge_failure_kind": _cut(kind, 16) if isinstance(kind, str) else None,
        "judge_source": "llm" if llm_verdict else "rules",
        "on_simulated_run": bool(record.get("simulated")), "synthetic": bool(record.get("synthetic")),
    }
    size = len(dumps_line(label).encode("utf-8")) + 1
    if size > MAX_LINE_BYTES:  # not expected within the byte budgets above; say which field to shorten if it happens
        raise LabelError(f"this label is {size} bytes, over the 2 KB line cap; shorten human_reason")
    return label


def append_jsonl(path: Path | str, obj: dict) -> None:
    """Append one JSON object as one line. Refuses a line over 2 KB and a file already at 5 MB. A thread lock plus
    flock (where the platform has it) keeps simultaneous appends from interleaving. If the file does not end in a
    newline (a torn earlier write) a newline is written first, so the new line is never glued to the old one."""
    path = Path(path)
    try:
        line = (dumps_line(obj) + "\n").encode("utf-8")
    except UnicodeError:
        raise LabelError("the text has a character that cannot be saved (a lone surrogate); remove it") from None
    if len(line) > MAX_LINE_BYTES:
        raise LabelError(f"that line is {len(line)} bytes; the limit is 2 KB")
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK, open(path, "a+b") as handle:
        if fcntl:
            fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            size = handle.seek(0, 2)
            if size:
                handle.seek(-1, 2)
                if handle.read(1) != b"\n":
                    line = b"\n" + line
            if size + len(line) > MAX_FILE_BYTES:
                raise LabelError("the labels file is full (5 MB); archive it and start a new one", status=409)
            handle.write(line)
            handle.flush()
        finally:
            if fcntl:
                fcntl.flock(handle, fcntl.LOCK_UN)
