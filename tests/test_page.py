"""Static checks for the dashboard page and the JSON fixtures (no browser, stdlib only)."""

import json
import re
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / "callqa" / "static" / "index.html"
FIXTURES = Path(__file__).resolve().parent / "fixtures"

REQUIRED_IDS = [
    "sop-input", "agent-select", "run-button", "mode-badge", "warnings", "progress",
    "pass-rate", "before-after", "next-fix", "calls", "loading", "error-banner",
]
REQUIRED_CLASSES = ["call", "failed", "turn", "failing"]


class _Scan(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.attr_values = []
        self.script = ""
        self._in_script = False

    def handle_starttag(self, tag, attrs):
        for name, value in attrs:
            if name == "id":
                self.ids.add(value)
            self.attr_values.append((tag, name, value or ""))
        if tag == "script":
            self._in_script = True

    def handle_endtag(self, tag):
        if tag == "script":
            self._in_script = False

    def handle_data(self, data):
        if self._in_script:
            self.script += data


def _scan():
    scan = _Scan()
    scan.feed(PAGE.read_text(encoding="utf-8"))
    return scan


def test_page_has_required_hooks():
    html = PAGE.read_text(encoding="utf-8")
    scan = _scan()
    assert '<meta charset="utf-8">' in html.lower()
    for hook in REQUIRED_IDS:
        assert hook in scan.ids, f"missing id {hook}"
    # classes are created by the script, so they must appear in the script or the markup
    for cls in REQUIRED_CLASSES:
        assert re.search(rf"['\"\s.]{cls}['\"\s{{,)]", html), f"class {cls} not referenced"
    assert re.search(r"""['"]mark['"]""", _scan().script), "forbidden phrase must be wrapped in a mark element"


def test_page_polls_run_status():
    script = _scan().script
    assert "/api/run" in script
    assert "/api/runs" in script
    assert "setTimeout" in script or "setInterval" in script
    for status in ("running", "done", "error"):
        assert status in script


def test_page_reads_backend_and_estimated_calls_from_config():
    script = _scan().script
    assert "/api/config" in script
    for key in ("estimated_calls", "sample_sop", "agents", "backend", "callers", "warnings"):
        assert key in script
    assert "about" in script


def test_page_has_no_external_urls():
    html = PAGE.read_text(encoding="utf-8")
    assert not re.search(r"(https?:)?//[a-z0-9.-]+\.[a-z]{2,}", html, re.IGNORECASE)
    for tag, name, value in _scan().attr_values:
        if name in ("src", "href", "action"):
            assert not value.startswith(("http:", "https:", "//")), f"{tag} {name}={value}"
    assert "@import" not in html


def test_page_never_uses_innerHTML():
    script = _scan().script
    assert script.strip()
    for banned in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function"):
        assert banned not in script, banned
    assert "textContent" in script


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _load(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_report_fixture_matches_contract():
    report = _load("report_example.json")
    assert set(report) == {
        "run_id", "sop_hash", "agent", "backend", "callers", "max_turns", "warnings", "generator",
        "pass_rate", "passed", "total", "seconds", "calls", "next_fixes",
    }
    assert isinstance(report["run_id"], str) and isinstance(report["sop_hash"], str)
    assert set(report["agent"]) == {"id", "label"}
    assert set(report["backend"]) == {"name", "models"}
    assert isinstance(report["backend"]["models"], dict)
    assert isinstance(report["callers"], int) and isinstance(report["max_turns"], int)
    assert isinstance(report["warnings"], list)
    assert set(report["generator"]) == {"source", "warnings"}
    assert _is_number(report["pass_rate"]) and _is_number(report["seconds"])
    assert isinstance(report["passed"], int) and isinstance(report["total"], int)

    call_keys = {
        "persona_id", "name", "language", "tags", "passed", "reason", "failure_kind", "failing_turn",
        "sop_rule", "fix_hint", "transcript",
    }
    calls = report["calls"]
    assert len(calls) == report["total"] == report["callers"] == 20
    assert sum(c["passed"] for c in calls) == report["passed"] == 14
    assert report["pass_rate"] == 0.7
    assert any(not c["passed"] and c["language"] == "hinglish" for c in calls)
    for call in calls:
        assert set(call) == call_keys
        assert call["language"] in ("en", "hinglish")
        assert isinstance(call["tags"], list) and isinstance(call["passed"], bool)
        assert isinstance(call["reason"], str) and isinstance(call["sop_rule"], str)
        assert isinstance(call["fix_hint"], str)
        assert call["transcript"]
        for turn in call["transcript"]:
            assert set(turn) == {"role", "content"}
            assert turn["role"] in ("user", "assistant")
            assert isinstance(turn["content"], str)
        if call["passed"]:
            assert call["failure_kind"] is None and call["failing_turn"] is None
        else:
            assert call["failure_kind"] in ("forbidden", "missing", "judge", "crash")
            turn = call["failing_turn"]
            assert isinstance(turn, int) and not isinstance(turn, bool)
            assert 0 <= turn < len(call["transcript"])
            assert call["transcript"][turn]["role"] == "assistant"

    failed = {c["persona_id"]: c for c in calls if not c["passed"]}
    assert set(failed) == {"brother_asks", "wife_handles_money", "wrong_number", "prompt_injection",
                           "fake_head_office", "wants_manager"}
    for pid, call in failed.items():
        if pid == "wants_manager":
            assert call["failure_kind"] == "missing" and call["failing_turn"] == 3
        else:
            assert call["failure_kind"] == "forbidden" and call["failing_turn"] == 1
            assert "12,400" in call["transcript"][1]["content"]

    fixes = report["next_fixes"]
    assert [f["count"] for f in fixes] == [5, 1]
    assert "Verification" in fixes[0]["sop_rule"] and "Escalation" in fixes[1]["sop_rule"]
    for fix in fixes:
        assert set(fix) == {"sop_rule", "fix_hint", "count", "calls"}
        assert fix["count"] == len(fix["calls"])
        for item in fix["calls"]:
            assert set(item) == {"persona_id", "name", "failing_turn"}
            assert failed[item["persona_id"]]["failing_turn"] == item["failing_turn"]


def test_config_fixture_matches_contract():
    config = _load("config_example.json")
    assert set(config) == {
        "backend", "warnings", "callers", "max_turns", "estimated_calls", "agent_kind", "agents", "sample_sop",
    }
    assert set(config["backend"]) == {"name", "requested", "models"}
    assert config["backend"]["name"] == "claude-cli"
    assert set(config["backend"]["models"]) == {"caller", "agent", "judge", "gen"}
    assert len(config["warnings"]) == 1 and isinstance(config["warnings"][0], str)
    assert config["callers"] == 5 and config["max_turns"] == 4
    assert config["estimated_calls"] == 2 * 5 * 4 + 5 == 45
    assert isinstance(config["agent_kind"], str)
    assert [a["id"] for a in config["agents"]] == ["v1", "v2"]
    for agent in config["agents"]:
        assert set(agent) == {"id", "label"}
    assert isinstance(config["sample_sop"], str) and config["sample_sop"].strip()
