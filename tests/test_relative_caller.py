"""O4: the dashboard's --rules flag and the story compliance lines. Offline only, no model is ever called."""
import json
import threading
import time
import urllib.request
from pathlib import Path

import pytest

from callqa import __main__ as cli
from callqa import evaluate
from callqa.compliance import DEFAULT_RULES_PATH
from callqa.server import read_sample_sop

HERE = Path(__file__).resolve().parent
GOLDEN = HERE / "golden"
STORY_V1_RULES = HERE / "fixtures" / "report_story_v1_rules.json"
LABEL = "rules/collections_india.yaml"
LINE_V1 = ("compliance collections_india: 5 critical (no_third_party_disclosure: brother_asks, fake_head_office, "
           "prompt_injection, wife_handles_money, wrong_number); not evaluated offline: grievance_channel")
LINE_CLEAN = "compliance collections_india: 0 critical; not evaluated offline: grievance_channel"


def _serve(tmp_path, *flags):
    """The dashboard as `python -m callqa --offline` builds it, on a free port, results under tmp_path."""
    args = cli.parse_args(["--offline", "--no-browser", *flags])
    cfg = cli.resolve_config(args)
    server = cli.bind_server(cfg, 0, results_dir=tmp_path / "runs")
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def _call(base, path, body=None):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    request = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read())


def _finished_run(base):
    run_id = _call(base, "/api/run", {"sop": read_sample_sop(), "agent": "v1"})["run_id"]
    for _ in range(3000):
        body = _call(base, f"/api/run/{run_id}")
        if body["status"] != "running":
            break
        time.sleep(0.01)
    assert body["status"] == "done", body
    return body["report"]


def _config_and_report(tmp_path, *flags):
    server, base = _serve(tmp_path, *flags)
    try:
        return _call(base, "/api/config"), _finished_run(base)
    finally:
        server.shutdown()
        server.server_close()


def test_dashboard_rules_flag_reaches_report(tmp_path, capsys):
    for flags in ((), ("--rules", str(DEFAULT_RULES_PATH))):
        config, report = _config_and_report(tmp_path, *flags)
        assert config["rules_file"] == LABEL
        assert [r["id"] for r in config["rules"]][:1] == ["calling_hours"]
        assert report["rules_file"] == LABEL
        block = report["compliance"]
        assert block["rules_id"] == "collections_india" and block["rules_file"] == LABEL
        assert sum(f["count"] for f in block["failures"] if f["severity"] == "critical") == 5
        by_id = {c["persona_id"]: c for c in report["calls"]}
        assert by_id["brother_asks"]["severity"] == "critical" and by_id["brother_asks"]["compliance"]
        assert by_id["happy_friday"]["severity"] is None
        assert [e["rule_id"] for e in block["not_evaluated"]] == ["grievance_channel"]
    # none turns the layer off: no block, no per call results, nothing in the config
    config, report = _config_and_report(tmp_path, "--rules", "none")
    assert config.get("rules_file") is None and config.get("rules") is None
    assert report["compliance"] is None and report["rules_file"] is None
    assert all(c["severity"] in (None, "major") and c["compliance"] == [] for c in report["calls"])
    # a bad path exits 2 with a message, before any server starts
    assert cli.main(["--offline", "--no-browser", "--rules", str(tmp_path / "missing.yaml")]) == 2
    assert "cannot read the rules file" in capsys.readouterr().err


def test_the_rules_flag_default_is_the_one_eval_uses():
    args = cli.parse_args(["--offline"])
    assert args.rules is None  # None means the shared default, DEFAULT_RULES_PATH
    assert cli.resolve_config(args).rules.label == LABEL
    assert cli.resolve_config(cli.parse_args(["--offline", "--rules", "NONE"])).rules is None


def test_story_compliance_lines_pinned():
    # the committed goldens (read only)
    v1 = (GOLDEN / "eval_story_v1.txt").read_text(encoding="utf-8").splitlines()
    assert LINE_V1 in v1
    for name in ("v2", "v3"):
        lines = (GOLDEN / f"eval_story_{name}.txt").read_text(encoding="utf-8").splitlines()
        assert LINE_CLEAN in lines
        assert not any("critical (" in line for line in lines)
    # and the formatting function, from the saved v1 story record
    record = json.loads(STORY_V1_RULES.read_text(encoding="utf-8"))
    assert evaluate.compliance_line(record) == LINE_V1
    record["compliance"]["failures"] = []
    assert evaluate.compliance_line(record) == LINE_CLEAN
    record["compliance"] = None
    assert evaluate.compliance_line(record) == "compliance: off"


def test_rules_flag_needs_a_value():
    with pytest.raises(SystemExit) as exc:
        cli.parse_args(["--rules"])
    assert exc.value.code == 2
