"""Run records: the report a finished run returns is the file it is saved as (schema version 1)."""
from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path

RECORD_SCHEMA_VERSION = 1
RECORD_KIND = "run"
REQUIRED_KEYS = (
    "schema_version", "kind", "created_at", "synthetic", "simulated", "note", "run_id", "sop_hash", "agent",
    "backend", "persona_set", "seed", "repeats", "callers", "max_turns", "rules_file", "warnings", "generator",
    "pass_rate", "passed", "total", "judge_split", "seconds", "consistency", "cost", "latency", "escalation",
    "compliance", "traits", "calls", "next_fixes",
)


class RecordError(Exception):
    """A run record that cannot be read: missing, not JSON, the wrong kind, a bad schema or a missing key."""


def record_name(record: dict) -> str:
    """`<YYYYmmdd-HHMMSS>_<agent id>.json`, the time taken from `created_at`. Any character outside
    letters, digits, `_`, `.` and `-` in the agent id becomes `_`, so the name is always a safe file name."""
    stamp = datetime.fromisoformat(record["created_at"]).strftime("%Y%m%d-%H%M%S")
    agent = re.sub(r"[^A-Za-z0-9_.-]", "_", record["agent"]["id"])[:80]
    return f"{stamp}_{agent}.json"


def save_record(record: dict, directory: Path | str) -> Path:
    """Write the record as UTF-8 JSON into `directory`, atomically: the text goes to a temp file in the same folder,
    which is then hard linked to the final name. A crash leaves no half file under the final name, and a name
    collision (FileExistsError from the link) gets a -2, -3 suffix, never an overwrite."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    base = Path(record_name(record))
    text = json.dumps(record, indent=2, ensure_ascii=False)
    fd, tmp_name = tempfile.mkstemp(dir=directory, prefix=".tmp-", suffix=".part")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        attempt = 1
        while True:
            name = base.name if attempt == 1 else f"{base.stem}-{attempt}{base.suffix}"
            try:
                os.link(tmp, directory / name)
            except FileExistsError:
                attempt += 1
            else:
                return directory / name
    finally:
        tmp.unlink(missing_ok=True)


def load_record(path: Path | str) -> dict:
    try:
        record = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:  # includes a half written file
        raise RecordError(f"cannot read run record {path}: {exc}") from exc
    if not isinstance(record, dict) or record.get("kind") != RECORD_KIND:
        raise RecordError(f"{path} is not a run record (kind must be {RECORD_KIND!r})")
    version = record.get("schema_version")
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise RecordError(f"{path} needs an integer schema_version of 1 or more")
    if version > RECORD_SCHEMA_VERSION:
        raise RecordError(f"{path} has a newer schema_version {version}; this code reads up to {RECORD_SCHEMA_VERSION}")
    missing = [key for key in REQUIRED_KEYS if key not in record]
    if missing:
        raise RecordError(f"{path} is missing keys: {', '.join(missing)}")
    return record
