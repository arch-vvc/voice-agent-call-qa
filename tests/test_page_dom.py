"""DOM tests for the dashboard page: run index.html's script in a stub DOM under node (D13).

`tests/dom/stub_dom.js` is a small fake DOM and fetch. Each test hands it a scenario (canned API
answers plus clicks), node prints the DOM facts, and the asserts are here. No browser, no npm, no
network. Without node on PATH the tests skip. Visual review stays with design-agent.
"""

import copy
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
STUB = HERE / "dom" / "stub_dom.js"
PAGE = HERE.parent / "callqa" / "static" / "index.html"
RECORDS_PAGE = HERE.parent / "callqa" / "static" / "records.html"
FIXTURE = HERE / "fixtures" / "report_example.json"
FIXTURE_R3 = HERE / "fixtures" / "report_full_v2_r3.json"

needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node not on PATH")

CONFIG = {
    "backend": {"name": "offline"},
    "callers": 20,
    "estimated_calls": 0,
    "sample_sop": "Synthetic sample SOP",
    "agents": [{"id": "v1", "label": "v1 naive prompt"}, {"id": "v2", "label": "v2 fixed prompt"}],
    "warnings": [],
}


def _report():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _report_r3():
    return json.loads(FIXTURE_R3.read_text(encoding="utf-8"))


def _scenario(report, final_run=None, running_polls=0):
    """Canned API answers: config, a run that polls `running_polls` times then ends, the runs list."""
    final = final_run or {"status": "done", "done": 20, "total": 20, "report": report}
    running = {"status": "running", "done": 3, "total": 20}
    return {
        "routes": {
            "GET /api/config": {"body": CONFIG},
            "POST /api/run": {"body": {"run_id": "r1"}},
            "GET /api/runs": {"body": []},
        },
        "sequences": {"GET /api/run/r1": [{"body": running}] * running_polls + [{"body": final}]},
        "clicks": ["#run-button"],
    }


def run_page(tmp_path, scenario, page=PAGE):
    path = tmp_path / "scenario.json"
    path.write_text(json.dumps(scenario), encoding="utf-8")
    done = subprocess.run(
        ["node", str(STUB), str(page), str(path)], capture_output=True, text=True, timeout=60
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


@needs_node
def test_v1_fixture_renders_70_percent_six_failed_cards_five_marks(tmp_path):
    facts = run_page(tmp_path, _scenario(_report(), running_polls=2))
    assert facts["passRate"]["text"] == "70%"
    assert facts["passRate"]["className"] == "bad"
    assert facts["passSummary"] == "14 of 20 calls follow the SOP"
    assert len(facts["cards"]) == 20
    failed = [c for c in facts["cards"] if "failed" in c["className"].split()]
    assert len(failed) == 6
    assert all(c["verdict"] == "FAIL" for c in failed)
    # five forbidden phrase failures highlight "12,400"; the sixth (missing phrase) has no mark
    assert facts["marks"] == ["12,400"] * 5
    assert sum(1 for c in failed if c["marks"]) == 5
    assert facts["errorBanner"]["hidden"] is True
    assert facts["runButtonDisabled"] is False
    assert facts["fetchLog"].count("GET /api/run/r1") == 3  # two "running" polls, then done


@needs_node
def test_first_fix_opens_brother_asks_card(tmp_path):
    facts = run_page(tmp_path, _scenario(_report()))
    assert [c["id"] for c in facts["cards"] if c["open"]] == ["call-brother_asks"]
    assert facts["nextFix"]["hidden"] is False
    assert facts["nextFix"]["buttons"][0] == "Brother asks, Turn 2"
    brother = next(c for c in facts["cards"] if c["id"] == "call-brother_asks")
    assert brother["failingTurns"] == 1
    assert brother["marks"] == ["12,400"]
    # failed cards come first, biggest fix group first
    assert facts["cards"][0]["id"] == "call-brother_asks"
    assert facts["cards"][5]["id"] == "call-wants_manager"
    # clicking the other fix button opens that card as well
    scenario = _scenario(_report())
    scenario["clicks"].append("button.fixcall:Wants a manager")
    again = run_page(tmp_path, scenario)
    opened = sorted(c["id"] for c in again["cards"] if c["open"])
    assert opened == ["call-brother_asks", "call-wants_manager"]


@needs_node
def test_error_status_stops_polling_and_shows_banner(tmp_path):
    error = {"status": "error", "error": "judge backend exploded"}
    facts = run_page(tmp_path, _scenario(None, final_run=error, running_polls=1))
    assert facts["errorBanner"] == {"hidden": False, "text": "judge backend exploded"}
    assert facts["fetchLog"].count("GET /api/run/r1") == 2  # one "running", then the error; no more
    assert facts["pendingTimers"] == 0
    assert facts["runButtonDisabled"] is False
    assert facts["loadingHidden"] is True
    assert facts["progress"]["hidden"] is True
    assert facts["cards"] == []
    assert facts["passRate"]["text"] != "70%"  # no report was rendered


# ---- A3e: consistency, flaky chips, repeat tabs, cost strip, slow turns, picker, records link ----

def _r3_scenario(steps=(), queries=None, config=None, report=None):
    scenario = _scenario(report or _report_r3())
    scenario["clicks"] = ["#run-button"]
    scenario["steps"] = list(steps)
    scenario["queries"] = queries or {}
    if config:
        scenario["routes"]["GET /api/config"] = {"body": {**CONFIG, **config}}
    return scenario


@needs_node
def test_consistency_shows_pass_hat_k_and_pass_at_1(tmp_path):
    facts = run_page(tmp_path, _r3_scenario(queries={"consistency": "#consistency"}))
    box = facts["queries"]["consistency"][0]
    assert box["hidden"] is False
    assert "pass^3 86% (19 of 22 callers passed all 3 repeats)" in box["text"]
    assert "pass@1 89% (59 of 66 runs)" in box["text"]
    # the headline says callers, not calls, once there are repeats; its rate is pass^k
    assert facts["passRate"]["text"] == "86%"
    assert facts["passSummary"] == "19 of 22 callers passed every repeat"
    assert "calls follow the SOP" not in facts["passSummary"]


@needs_node
def test_one_repeat_hides_consistency_and_keeps_the_calls_wording(tmp_path):
    scenario = _scenario(_report())
    scenario["queries"] = {"consistency": "#consistency", "tabs": ".repeat-tab", "chips": ".chip.flaky"}
    facts = run_page(tmp_path, scenario)
    assert facts["queries"]["consistency"][0]["hidden"] is True
    assert facts["queries"]["tabs"] == [] and facts["queries"]["chips"] == []
    assert facts["passSummary"] == "14 of 20 calls follow the SOP"


@needs_node
def test_flaky_chip_opens_failing_repeat_tab(tmp_path):
    queries = {"chips": "#consistency .chip.flaky", "tabs": "#call-flaky_line .repeat-tab",
               "failing": "#call-flaky_line .turn.failing", "card": "#call-flaky_line"}
    # look at repeat 1 first (it passed), then use the chip: it must jump to repeat 3, the failing one
    steps = [{"click": "#call-flaky_line .repeat-tab", "text": "Repeat 1"},
             {"click": ".chip.flaky", "text": "flaky_line"}]
    facts = run_page(tmp_path, _r3_scenario(steps, queries))
    chips = facts["queries"]["chips"]
    assert len(chips) == 1 and chips[0]["tag"] == "button"
    assert "flaky_line" in chips[0]["text"] and "2/3" in chips[0]["text"] and "repeat 3" in chips[0]["text"]
    tabs = facts["queries"]["tabs"]
    assert [t["text"] for t in tabs] == ["Repeat 1", "Repeat 2", "Repeat 3 (failed)"]
    assert [("active" in t["className"].split()) for t in tabs] == [False, False, True]
    assert facts["queries"]["card"][0]["open"] is True
    assert len(facts["queries"]["failing"]) == 1  # the failing turn of repeat 3 is on screen


@needs_node
def test_repeat_tab_switches_the_transcript(tmp_path):
    flaky = next(c for c in _report_r3()["calls"] if c["persona_id"] == "flaky_line")
    queries = {"tabs": "#call-flaky_line .repeat-tab", "failing": "#call-flaky_line .turn.failing",
               "text": "#call-flaky_line .transcript"}
    steps = [{"click": "#call-flaky_line .repeat-tab", "text": "Repeat 2"}]
    facts = run_page(tmp_path, _r3_scenario(steps, queries))
    assert [("active" in t["className"].split()) for t in facts["queries"]["tabs"]] == [False, True, False]
    assert facts["queries"]["failing"] == []  # repeat 2 passed
    shown = facts["queries"]["text"][0]["text"]
    assert flaky["repeat_results"][1]["transcript"][2]["content"] in shown
    assert flaky["repeat_results"][2]["transcript"][2]["content"] not in shown


@needs_node
def test_cost_strip_says_simulated(tmp_path):
    report = _report_r3()
    facts = run_page(tmp_path, _r3_scenario(queries={"strip": "#cost-latency"}))
    strip = facts["queries"]["strip"][0]
    assert strip["hidden"] is False
    assert "Cost (simulated)" in strip["text"] and "Latency (simulated)" in strip["text"]
    assert f"${report['cost']['per_call']:.4f} per call" in strip["text"]
    assert f"${report['cost']['total']:.4f} total" in strip["text"]
    assert "P50 1130 ms" in strip["text"] and "P95 1670 ms" in strip["text"] and "budget 1500 ms" in strip["text"]
    assert "48 of 147 agent turns over" in strip["text"]
    assert "verify prices before quoting" in strip["text"]


@needs_node
@pytest.mark.parametrize("label", ["estimated", "reported"])
def test_cost_strip_carries_whatever_label_the_report_has(tmp_path, label):
    report = _report_r3()
    report["cost"]["label"] = label
    text = run_page(tmp_path, _r3_scenario(queries={"strip": "#cost-latency"}, report=report))
    text = text["queries"]["strip"][0]["text"]
    assert f"Cost ({label})" in text and "Cost (simulated)" not in text


@needs_node
def test_unknown_price_says_unknown_and_missing_blocks_hide_the_strip(tmp_path):
    report = _report_r3()
    report["cost"].update(per_call=None, total=None, unknown_models=["mystery-model"])
    facts = run_page(tmp_path, _r3_scenario(queries={"strip": "#cost-latency"}, report=report))
    text = facts["queries"]["strip"][0]["text"]
    assert "cost unknown" in text and "mystery-model" in text and "$" not in text
    report["cost"] = report["latency"] = None
    facts = run_page(tmp_path, _r3_scenario(queries={"strip": "#cost-latency"}, report=report))
    assert facts["queries"]["strip"][0]["hidden"] is True


@needs_node
def test_over_budget_turns_are_marked_slow(tmp_path):
    report = _report_r3()
    budget = report["latency"]["budget_ms"]
    expected = sum(1 for c in report["calls"] for ms in c["turn_latency_ms"] if ms > budget)
    assert 0 < expected < sum(len(c["turn_latency_ms"]) for c in report["calls"])
    facts = run_page(tmp_path, _r3_scenario(queries={"slow": "#calls .turn.slow", "agent": "#calls .turn.assistant"}))
    slow = facts["queries"]["slow"]
    assert len(slow) == expected
    assert all("assistant" in t["className"].split() for t in slow)
    assert all(re.search(r"\d+ ms", t["text"]) for t in facts["queries"]["agent"])


@needs_node
def test_picker_sends_chosen_set_and_repeats(tmp_path):
    queries = {"sets": "#set-select option", "repeats": "#repeats-select option", "set": "#set-select",
               "reps": "#repeats-select"}
    config = {"set": "full", "repeats": 3}
    facts = run_page(tmp_path, _r3_scenario(queries=queries, config=config))
    assert [o["value"] for o in facts["queries"]["sets"]] == ["story", "full"]
    assert [o["value"] for o in facts["queries"]["repeats"]] == ["1", "2", "3", "4", "5"]
    assert (facts["queries"]["set"][0]["value"], facts["queries"]["reps"][0]["value"]) == ("full", "3")
    posted = [b["body"] for b in facts["fetchBodies"] if b["key"] == "POST /api/run"]
    assert posted[0]["set"] == "full" and posted[0]["repeats"] == 3
    steps = [{"select": "#set-select", "value": "story"}, {"select": "#repeats-select", "value": "2"},
             {"click": "#run-button"}]
    scenario = _r3_scenario(steps, config=config)
    scenario["clicks"] = []
    posted = [b["body"] for b in run_page(tmp_path, scenario)["fetchBodies"] if b["key"] == "POST /api/run"]
    assert (posted[0]["set"], posted[0]["repeats"]) == ("story", 2)


@needs_node
def test_dashboard_links_to_records(tmp_path):
    scenario = _scenario(_report())
    scenario["queries"] = {"link": "#records-link"}
    link = run_page(tmp_path, scenario)["queries"]["link"][0]
    assert link["tag"] == "a" and link["href"] == "/records"


# ---- A3e: the records page ----

def _record_pair():
    """OLD and NEW records from the r3 fixture: one persona breaks, one is fixed, one fails in both, and the last
    persona exists only in NEW. Returns (old, new, id of the persona only in new)."""
    old, new = _report_r3(), _report_r3()
    old["agent"] = {**old["agent"], "id": "v1", "label": "v1 naive prompt"}

    def set_pass(record, persona_id, ok):
        call = next(c for c in record["calls"] if c["persona_id"] == persona_id)
        call["repeats_passed"] = call["repeats_total"] if ok else call["repeats_total"] - 1
        call["passed"] = ok

    for persona_id, before, after in (("happy_friday", True, False), ("happy_hinglish", False, True),
                                      ("salary_on_1st", False, False), ("flaky_line", True, True)):
        set_pass(old, persona_id, before)
        set_pass(new, persona_id, after)
    only_in_new = old["calls"].pop()["persona_id"]
    for record in (old, new):
        record["total"] = len(record["calls"])
        record["passed"] = sum(c["repeats_passed"] == c["repeats_total"] for c in record["calls"])
        record["pass_rate"] = round(record["passed"] / record["total"], 4)
    return old, new, only_in_new


def _records_scenario(steps=(), queries=None):
    old, new, _ = _record_pair()
    row = {"kind": "run", "set": "full", "repeats": 3, "backend": "offline", "synthetic": True}
    listing = [
        {**row, "name": "new_v2.json", "agent": "v2 fixed prompt", "pass_rate": new["pass_rate"],
         "created_at": "2026-11-06T10:00:00+05:30"},
        {**row, "name": "old_v1.json", "agent": "v1 naive prompt", "pass_rate": old["pass_rate"],
         "created_at": "2026-11-05T10:00:00+05:30"},
    ]
    return {
        "routes": {"GET /api/records": {"body": listing},
                   "GET /api/records/old_v1.json": {"body": old}, "GET /api/records/new_v2.json": {"body": new}},
        "steps": list(steps), "queries": queries or {},
    }


COMPARE = [{"select": "#old-select", "value": "old_v1.json"}, {"select": "#new-select", "value": "new_v2.json"},
           {"click": "#compare-button"}]


@needs_node
def test_records_page_lists_records_in_both_pickers(tmp_path):
    queries = {"old": "#old-select option", "new": "#new-select option", "link": "#dashboard-link"}
    facts = run_page(tmp_path, _records_scenario(queries=queries), page=RECORDS_PAGE)
    for key in ("old", "new"):
        assert [o["value"] for o in facts["queries"][key]] == ["new_v2.json", "old_v1.json"]
    assert "v2 fixed prompt" in facts["queries"]["new"][0]["text"]
    assert "pass^3" in facts["queries"]["new"][0]["text"]
    assert facts["queries"]["link"][0]["href"] == "/"


@needs_node
def test_diff_view_lists_newly_broken_first_in_red(tmp_path):
    queries = {"rows": "#diff-list .diff-row", "heads": "#diff-list .diff-heading", "summary": "#diff-summary",
               "gate": "#diff-gate"}
    facts = run_page(tmp_path, _records_scenario(COMPARE, queries), page=RECORDS_PAGE)
    old, new, _ = _record_pair()
    rows = facts["queries"]["rows"]
    kinds = [next(c for c in r["className"].split() if c != "diff-row") for r in rows]
    ok_before = {c["persona_id"]: c["repeats_passed"] == c["repeats_total"] for c in old["calls"]}
    still_failing = [c for c in new["calls"] if c["persona_id"] in ok_before and not ok_before[c["persona_id"]]
                     and c["repeats_passed"] != c["repeats_total"]]
    assert len(still_failing) >= 2  # salary_on_1st plus the fixture's own failing callers
    assert kinds == ["broken", "fixed"] + ["still"] * (len(still_failing))  # broken, then fixed, then still failing
    assert "happy_friday" in rows[0]["text"] and "NEWLY BROKEN" in facts["queries"]["heads"][0]["text"]
    assert "happy_hinglish" in rows[1]["text"] and "salary_on_1st" in " ".join(r["text"] for r in rows[2:])
    assert [h["text"].split(" (")[0] for h in facts["queries"]["heads"]] == ["NEWLY BROKEN", "FIXED", "STILL FAILING"]
    gate = facts["queries"]["gate"][0]
    assert gate["hidden"] is False and "1 previously passing caller now fails" in gate["text"]
    summary = facts["queries"]["summary"][0]["text"]
    assert f"pass^3 {round(old['passed'] / old['total'] * 100)}%" in summary
    assert f"pass^3 {round(new['passed'] / new['total'] * 100)}%" in summary
    assert facts["fetchLog"] == ["GET /api/records", "GET /api/records/old_v1.json", "GET /api/records/new_v2.json"]


@needs_node
def test_diff_names_callers_only_in_one_record_in_the_notes(tmp_path):
    _, _, only_in_new = _record_pair()
    facts = run_page(tmp_path, _records_scenario(COMPARE, {"notes": "#diff-notes"}), page=RECORDS_PAGE)
    assert f"ADDED (1): {only_in_new}" in facts["queries"]["notes"][0]["text"]


@needs_node
def test_diff_with_nothing_broken_says_so_and_hides_the_gate(tmp_path):
    scenario = _records_scenario(COMPARE, {"rows": "#diff-list .diff-row", "gate": "#diff-gate", "ok": "#diff-ok"})
    new = scenario["routes"]["GET /api/records/new_v2.json"]["body"]
    scenario["routes"]["GET /api/records/old_v1.json"]["body"] = copy.deepcopy(new)
    facts = run_page(tmp_path, scenario, page=RECORDS_PAGE)
    assert facts["queries"]["gate"][0]["hidden"] is True
    assert "No previously passing caller fails" in facts["queries"]["ok"][0]["text"]
    assert all("broken" not in r["className"].split() for r in facts["queries"]["rows"])


@needs_node
def test_records_page_shows_an_error_when_a_record_cannot_load(tmp_path):
    scenario = _records_scenario(COMPARE, {"err": "#records-error"})
    scenario["routes"]["GET /api/records/old_v1.json"] = {"status": 404, "body": {"error": "unknown record"}}
    facts = run_page(tmp_path, scenario, page=RECORDS_PAGE)
    assert facts["queries"]["err"][0]["hidden"] is False
    assert "unknown record" in facts["queries"]["err"][0]["text"]


# ---- A3e: static checks, no node needed ----

@pytest.mark.parametrize("page", [PAGE, RECORDS_PAGE], ids=["index", "records"])
def test_new_pages_never_use_innerHTML_or_external_urls(page):
    text = page.read_text(encoding="utf-8")
    for banned in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "new Function"):
        assert banned not in text, banned
    assert not re.search(r"https?:|//[a-z0-9.-]+\.[a-z]{2,}|<script[^>]*\ssrc=|<link[^>]|@import|url\(", text, re.I)
    assert not re.search(r"\s(?:src|href|action)=[\"'](?!/|#)", text)  # only same site paths and anchors


# ---- fix round 1 ----

def _css(page=PAGE):
    return re.search(r"<style>(.*?)</style>", page.read_text(encoding="utf-8"), re.S).group(1)


def test_no_slow_rule_can_beat_the_failing_turn_style():
    """A failing turn (the money shot) must stay red: any rule that styles a bare .turn.slow must exclude .failing."""
    rules = re.findall(r"([^{}]+)\{([^}]*)\}", _css())
    paints = [sel.strip() for sel, body in rules if re.search(r"background|border", body)
              for sel in sel.split(",") if re.search(r"\.turn\.slow\s*$|\.turn\.slow:", sel.strip())]
    assert paints, "expected a .turn.slow rule"
    assert all(":not(.failing)" in sel for sel in paints), paints


def test_the_slow_check_bites_on_a_rule_without_not_failing():
    broken = _css().replace(".turn.slow:not(.failing)", ".turn.slow")
    rules = re.findall(r"([^{}]+)\{([^}]*)\}", broken)
    paints = [sel.strip() for sel, body in rules if re.search(r"background|border", body)
              if re.search(r"\.turn\.slow\s*$", sel.strip())]
    assert paints and not all(":not(.failing)" in sel for sel in paints)


@needs_node
def test_failing_turn_with_slow_latency_keeps_both_classes_so_css_can_pick_failing(tmp_path):
    report = _report()
    call = next(c for c in report["calls"] if c["persona_id"] == "brother_asks")
    call["turn_latency_ms"] = [9999] * len(call["turn_latency_ms"])
    report["latency"]["budget_ms"] = 1500
    facts = run_page(tmp_path, {**_scenario(report), "queries": {"both": "#call-brother_asks .turn.failing.slow"}})
    assert len(facts["queries"]["both"]) == 1


@needs_node
def test_non_representative_repeat_tab_shows_no_ms_or_slow_marks(tmp_path):
    queries = {"slow": "#call-flaky_line .turn.slow", "ms": "#call-flaky_line .ms"}
    first = run_page(tmp_path, _r3_scenario(queries=queries))
    assert len(first["queries"]["slow"]) == 1 and len(first["queries"]["ms"]) == 2  # repeat 3 is the representative
    steps = [{"click": "#call-flaky_line .repeat-tab", "text": "Repeat 1"}]
    other = run_page(tmp_path, _r3_scenario(steps, queries))
    assert other["queries"]["slow"] == [] and other["queries"]["ms"] == []


@needs_node
def test_a_turn_exactly_at_the_budget_is_not_slow(tmp_path):
    report = _report_r3()
    budget = report["latency"]["budget_ms"]
    call = next(c for c in report["calls"] if c["persona_id"] == "happy_friday")
    n = len(call["turn_latency_ms"])
    call["turn_latency_ms"] = [budget] * (n - 1) + [budget + 1]
    queries = {"slow": "#call-happy_friday .turn.slow"}
    scenario = _r3_scenario(queries=queries, report=report)
    scenario["steps"] = []
    # open the card by its fix button is not possible (it passes), so read the DOM directly
    facts = run_page(tmp_path, scenario)
    assert len(facts["queries"]["slow"]) == 1


@needs_node
def test_picker_label_uses_the_configured_caller_count(tmp_path):
    scenario = _r3_scenario(queries={"sets": "#set-select option"}, config={"callers": 5})
    sets = run_page(tmp_path, scenario)["queries"]["sets"]
    assert "5 callers" in sets[0]["text"] and "20 callers" not in sets[0]["text"]


def test_per_call_cost_divides_by_conversations_not_callers(tmp_path):
    from callqa.evaluate import main
    out = tmp_path / "r3.json"
    args = ["--agent", "v2", "--set", "full", "--repeats", "3", "--backend", "offline", "--out", str(out)]
    assert main(args) in (0, 1)
    record = json.loads(out.read_text(encoding="utf-8"))
    conversations = record["callers"] * record["repeats"]
    assert conversations == 93
    assert record["cost"]["per_call"] == pytest.approx(record["cost"]["total"] / conversations, abs=1e-6)
    shipped = _report_r3()["cost"]
    assert shipped["per_call"] == pytest.approx(shipped["total"] / 66, abs=1e-6)


def _runs_scenario(runs, report):
    scenario = _scenario(report)
    scenario["routes"]["GET /api/runs"] = {"body": runs}
    scenario["queries"] = {"box": "#before-after"}
    return scenario


def _run_row(report, agent_label, set_name, repeats):
    return {"run_id": "old", "sop_hash": report["sop_hash"], "agent_label": agent_label, "pass_rate": 0.7,
            "passed": 14, "total": 20, "set": set_name, "repeats": repeats}


@needs_node
def test_before_after_does_not_compare_runs_of_different_set_or_repeats(tmp_path):
    report = _report_r3()
    for row in (_run_row(report, "v1 naive prompt", "story", 1), _run_row(report, "v1 naive prompt", "full", 1),
                _run_row(report, "v1 naive prompt", "story", 3)):
        box = run_page(tmp_path, _runs_scenario([row], report))["queries"]["box"][0]["text"]
        assert "same callers" not in box and "→" not in box, box
        assert "not comparable" in box


@needs_node
def test_before_after_compares_runs_of_the_same_set_and_repeats(tmp_path):
    report = _report_r3()
    row = _run_row(report, "v1 naive prompt", "full", 3)
    box = run_page(tmp_path, _runs_scenario([row], report))["queries"]["box"][0]["text"]
    assert "v1 70% → v2 86%" in box and "same callers" in box


@needs_node
def test_records_page_lists_only_run_records_and_encodes_names(tmp_path):
    scenario = _records_scenario(COMPARE_AT, {"old": "#old-select option"})
    scenario["routes"]["GET /api/records"]["body"].append(
        {"name": "cmp.json", "kind": "compare", "agent": "x", "set": "full", "repeats": 1, "backend": "offline",
         "pass_rate": 0.5, "created_at": "2026-11-07T10:00:00+05:30", "synthetic": True})
    listing = scenario["routes"]["GET /api/records"]["body"]
    listing[0]["name"] = "v2@haiku.json"
    scenario["routes"]["GET /api/records/v2%40haiku.json"] = scenario["routes"].pop("GET /api/records/new_v2.json")
    facts = run_page(tmp_path, scenario, page=RECORDS_PAGE)
    assert [o["value"] for o in facts["queries"]["old"]] == ["v2@haiku.json", "old_v1.json"]  # no compare record
    assert "GET /api/records/v2%40haiku.json" in facts["fetchLog"]


COMPARE_AT = [{"select": "#old-select", "value": "old_v1.json"}, {"select": "#new-select", "value": "v2@haiku.json"},
              {"click": "#compare-button"}]


# ---- A4d / O1: escalation panel and handoff notes ----

FIXTURE_CURRENT = HERE / "fixtures" / "report_full_v2_r3_current.json"


def _report_current():
    return json.loads(FIXTURE_CURRENT.read_text(encoding="utf-8"))


def _esc_scenario(report=None, steps=(), queries=None):
    return _r3_scenario(steps, queries or {}, report=report or _report_current())


def test_current_fixture_is_the_committed_v2_full_record():
    samples = HERE.parent / "samples" / "run_records" / "full_v2.json"
    assert FIXTURE_CURRENT.read_bytes() == samples.read_bytes()


@needs_node
def test_escalation_matrix_cells_match_fixture(tmp_path):
    block = _report_current()["escalation"]
    queries = {"panel": "#escalation", **{k: f"#escalation .esc-cell.{k} .esc-num" for k in ("tp", "fp", "fn", "tn")},
               "line": "#escalation .esc-summary"}
    facts = run_page(tmp_path, _esc_scenario(queries=queries))
    assert facts["queries"]["panel"][0]["hidden"] is False
    for key in ("tp", "fp", "fn", "tn"):
        assert [c["text"] for c in facts["queries"][key]] == [str(block["matrix"][key])], key
    line = facts["queries"]["line"][0]["text"]
    assert line == "Handoff note average 4.0 of 5"  # the summary line above already says accuracy and runs


def test_escalation_panel_sits_below_next_fix_and_above_the_cards():
    html = PAGE.read_text(encoding="utf-8")
    assert html.index('id="next-fix"') < html.index('id="escalation"') < html.index('id="calls"')


@needs_node
def test_missed_escalation_opens_card(tmp_path):
    queries = {"missed": "#escalation .esc-missed button.esc-call", "unnecessary": "#escalation .esc-unnecessary"}
    steps = [{"click": "#escalation .esc-missed button.esc-call", "text": "Threatens consumer court"}]
    facts = run_page(tmp_path, _esc_scenario(steps=steps, queries=queries))
    assert [b["text"] for b in facts["queries"]["missed"]] == [
        "Asks for a manager", "Lost his job, cannot pay or name a date",
        "Threatens consumer court and a regulator complaint"]
    assert "call-consumer_court" in [c["id"] for c in facts["cards"] if c["open"]]
    assert facts["queries"]["unnecessary"] == []  # nothing was unnecessary in this record, so no list


@needs_node
def test_unnecessary_escalation_lists_the_caller_and_opens_card(tmp_path):
    report = _report_current()
    report["escalation"]["unnecessary"] = ["lost_job_no_date"]
    report["escalation"]["missed"] = []
    queries = {"missed": "#escalation .esc-missed", "buttons": "#escalation .esc-unnecessary button.esc-call"}
    steps = [{"click": "#escalation .esc-unnecessary button.esc-call", "text": "Lost his job"}]
    facts = run_page(tmp_path, _esc_scenario(report, steps, queries))
    assert facts["queries"]["missed"] == []
    assert [b["text"] for b in facts["queries"]["buttons"]] == ["Lost his job, cannot pay or name a date"]
    assert "call-lost_job_no_date" in [c["id"] for c in facts["cards"] if c["open"]]


@needs_node
def test_handoff_card_lists_missing_fields(tmp_path):
    queries = {"note": "#call-not_my_loan .handoff", "score": "#call-not_my_loan .handoff-score",
               "missing": "#call-not_my_loan .handoff-missing", "fields": "#call-not_my_loan .handoff-field",
               "other": "#call-wants_manager .handoff"}
    facts = run_page(tmp_path, _esc_scenario(queries=queries))
    assert len(facts["queries"]["note"]) == 1
    assert "4 of 5" in facts["queries"]["score"][0]["text"]
    missing = facts["queries"]["missing"][0]["text"]
    assert "tried" in missing and "caller" not in missing
    fields = " ".join(f["text"] for f in facts["queries"]["fields"])
    assert "Disputes team to contact the caller within 2 working days" in fields
    assert "Caller on the Ravi Kumar account" in fields
    assert facts["queries"]["other"] == []  # a card that never handed over has no note


@needs_node
def test_handoff_with_nothing_missing_says_so_and_raw_note_is_shown(tmp_path):
    report = _report_current()
    call = next(c for c in report["calls"] if c["persona_id"] == "not_my_loan")
    call["escalation"]["handoff_score"] = 5
    call["escalation"]["handoff_missing"] = []
    wants = next(c for c in report["calls"] if c["persona_id"] == "wants_manager")
    wants["escalation"].update(escalated=True, outcome="TP", handoff={"raw": "garbled json"}, handoff_score=0,
                               handoff_missing=["caller", "issue", "tried", "why", "next_step"])
    queries = {"score": "#call-not_my_loan .handoff-score", "missing": "#call-not_my_loan .handoff-missing",
               "raw": "#call-wants_manager .handoff-raw", "gone": "#call-wants_manager .handoff-missing"}
    facts = run_page(tmp_path, _esc_scenario(report, queries=queries))
    assert "5 of 5" in facts["queries"]["score"][0]["text"]
    assert facts["queries"]["missing"] == []
    assert "garbled json" in facts["queries"]["raw"][0]["text"]
    assert "next_step" in facts["queries"]["gone"][0]["text"]


@needs_node
def test_escalation_panel_absent_when_report_has_no_escalation_key(tmp_path):
    report = _report_current()
    del report["escalation"]
    for call in report["calls"]:
        call.pop("escalation", None)
    queries = {"panel": "#escalation", "cells": "#escalation .esc-cell", "notes": ".handoff"}
    facts = run_page(tmp_path, _esc_scenario(report, queries=queries))
    assert facts["queries"]["panel"][0]["hidden"] is True
    assert facts["queries"]["cells"] == [] and facts["queries"]["notes"] == []
    # the older fixture never had the block, and null counts as absent
    old = run_page(tmp_path, _esc_scenario(_report_r3(), queries=queries))
    assert old["queries"]["panel"][0]["hidden"] is True and old["queries"]["notes"] == []
    report["escalation"] = None
    assert run_page(tmp_path, _esc_scenario(report, queries=queries))["queries"]["panel"][0]["hidden"] is True


_LIVE = {}


def _live_v1_report():
    """A real v1 story run, built in process through the dashboard's own run path (offline, no network)."""
    if "v1" not in _LIVE:
        import tempfile
        import threading
        import time
        import urllib.request

        from callqa.server import make_server
        from evals.backends import OfflineBackend

        sop = (HERE.parent / "samples" / "sop_collections_acme.md").read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as results:
            server = make_server(OfflineBackend(), 20, 8, 4, [{"id": "v1", "label": "v1 naive prompt"}],
                                 results_dir=results)
            threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
            base = f"http://127.0.0.1:{server.server_address[1]}"

            def call(path, body=None):
                data = json.dumps(body).encode("utf-8") if body is not None else None
                req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    return json.loads(resp.read())

            try:
                run_id = call("/api/run", {"sop": sop, "agent": "v1"})["run_id"]
                for _ in range(3000):
                    body = call(f"/api/run/{run_id}")
                    if body["status"] != "running":
                        break
                    time.sleep(0.01)
                assert body["status"] == "done", body
                _LIVE["v1"] = body["report"]
            finally:
                server.shutdown()
                server.server_close()
    return copy.deepcopy(_LIVE["v1"])


@needs_node
def test_live_v1_story_report_carries_the_block_and_the_panel_stays_one_closed_line(tmp_path):
    report = _live_v1_report()
    assert report["escalation"]["matrix"] == {"tp": 0, "fp": 0, "fn": 1, "tn": 19}  # a real v1 run has the block
    queries = {"panel": "#escalation", "summary": "#escalation summary", "details": "#escalation details"}
    scenario = _scenario(report)
    scenario["queries"] = queries
    facts = run_page(tmp_path, scenario)
    panel = facts["queries"]["panel"][0]
    assert panel["hidden"] is False and panel["childTags"] == ["details"]
    assert facts["queries"]["details"][0]["open"] is False
    summary = facts["queries"]["summary"][0]
    assert summary["text"] == "Escalation: accuracy 95% (19 of 20 runs), 1 missed" and "\n" not in summary["text"]
    # the money shot is untouched: first failing card open, five marks, six failed cards
    assert [c["id"] for c in facts["cards"] if c["open"]] == ["call-brother_asks"]
    assert facts["cards"][0]["id"] == "call-brother_asks"
    assert facts["marks"] == ["12,400"] * 5
    assert len([c for c in facts["cards"] if "failed" in c["className"].split()]) == 6
    assert facts["passRate"]["text"] == "70%"


def test_panel_comes_before_the_cards_and_holds_one_summary_line_of_markup():
    html = PAGE.read_text(encoding="utf-8")
    assert html.count('id="escalation"') == 1
    assert html.index('id="escalation"') < html.index('id="calls"')
    assert re.search(r'<div class="panel" id="escalation" hidden></div>', html)  # empty until the script fills it


@needs_node
def test_matrix_orientation_rows_expected_columns_actual_and_hot_only_on_fn(tmp_path):
    queries = {"grid": "#escalation .esc-grid"}
    facts = run_page(tmp_path, _esc_scenario(queries=queries))
    grid = facts["queries"]["grid"][0]
    texts, classes = grid["childTexts"], grid["childClasses"]
    assert texts[:3] == ["", "Handed over", "Not handed over"]
    assert texts[3] == "Should have" and texts[6] == "Should not"
    # row 1: tp then fn; row 2: fp then tn
    assert [c.split()[1] for c in (classes[4], classes[5], classes[7], classes[8])] == ["tp", "fn", "fp", "tn"]
    assert [("hot" in c.split()) for c in classes[4:6] + classes[7:9]] == [False, True, False, False]
    # no misses: nothing is red
    report = _report_current()
    report["escalation"]["matrix"].update(fn=0)
    clean = run_page(tmp_path, _esc_scenario(report, queries=queries))["queries"]["grid"][0]
    assert not any("hot" in c.split() for c in clean["childClasses"])
    # an unnecessary handoff is not a fail: fp is never red
    report["escalation"]["matrix"].update(fp=4)
    odd = run_page(tmp_path, _esc_scenario(report, queries=queries))["queries"]["grid"][0]
    assert not any("hot" in c.split() for c in odd["childClasses"])


@needs_node
def test_a_block_without_a_matrix_hides_the_panel_and_the_page_still_renders(tmp_path):
    report = _report_current()
    report["escalation"] = {"accuracy": 0.9}
    facts = run_page(tmp_path, _esc_scenario(report, queries={"panel": "#escalation", "cells": ".esc-cell"}))
    assert facts["queries"]["panel"][0]["hidden"] is True and facts["queries"]["cells"] == []
    assert len(facts["cards"]) == 31 and facts["passRate"]["text"] == "71%"


@needs_node
def test_escalated_with_no_note_says_so_and_a_null_score_prints_no_score(tmp_path):
    report = _report_current()
    wants = next(c for c in report["calls"] if c["persona_id"] == "wants_manager")
    wants["escalation"].update(escalated=True, outcome="TP", handoff=None, handoff_score=None, handoff_missing=[])
    mine = next(c for c in report["calls"] if c["persona_id"] == "not_my_loan")
    mine["escalation"]["handoff_score"] = None
    queries = {"none": "#call-wants_manager .handoff-none", "title": "#call-not_my_loan .handoff-title"}
    facts = run_page(tmp_path, _esc_scenario(report, queries=queries))
    assert facts["queries"]["none"][0]["text"] == "Handed over, no handoff note written (0 of 5)"
    title = facts["queries"]["title"][0]["text"]
    assert "null" not in title and "of 5" not in title and title.startswith("Handoff note")


@needs_node
def test_a_missed_callers_card_quotes_the_rule_and_only_missed_cards_do(tmp_path):
    report = _report_current()
    queries = {"missed": "#call-wants_manager .esc-reason", "court": "#call-consumer_court .esc-reason",
               "fine": "#call-not_my_loan .esc-reason", "all": ".esc-reason"}
    facts = run_page(tmp_path, _esc_scenario(report, queries=queries))
    assert facts["queries"]["missed"][0]["text"] == "Should have been handed over."
    assert len(facts["queries"]["court"]) == 1 and facts["queries"]["fine"] == []
    assert len(facts["queries"]["all"]) == 3  # the three missed callers, nobody else


# ---- O1 fix round 1: the panel is one closed line by default ----

@needs_node
def test_escalation_is_one_closed_summary_line_with_the_blocks_numbers(tmp_path):
    block = _report_current()["escalation"]
    queries = {"details": "#escalation details", "summary": "#escalation summary", "kids": "#escalation",
               "details_open": "#escalation .esc-details"}
    facts = run_page(tmp_path, _esc_scenario(queries=queries))
    assert len(facts["queries"]["details"]) == 1 and facts["queries"]["details"][0]["open"] is False
    assert len(facts["queries"]["summary"]) == 1
    runs, hit = block["runs"], block["matrix"]["tp"] + block["matrix"]["tn"]
    assert facts["queries"]["summary"][0]["text"] == (
        f"Escalation: accuracy {round(block['accuracy'] * 100)}% ({hit} of {runs} runs), "
        f"{block['matrix']['fn']} missed")
    # no matrix markup outside the details: the panel's only child is the details element
    assert facts["queries"]["kids"][0]["childTags"] == ["details"]
    assert facts["queries"]["summary"][0]["childTags"] == []


@needs_node
def test_escalation_summary_adds_unnecessary_only_when_there_are_some(tmp_path):
    report = _report_current()
    report["escalation"]["matrix"].update(fp=2)
    facts = run_page(tmp_path, _esc_scenario(report, queries={"summary": "#escalation summary"}))
    assert facts["queries"]["summary"][0]["text"].endswith("9 missed, 2 unnecessary")
    report["escalation"]["matrix"].update(fn=0, fp=0)
    facts = run_page(tmp_path, _esc_scenario(report, queries={"summary": "#escalation summary"}))
    text = facts["queries"]["summary"][0]["text"]
    assert "unnecessary" not in text and text.endswith("runs), 0 missed")


@needs_node
def test_matrix_and_caller_buttons_live_inside_the_closed_details(tmp_path):
    queries = {"cells": "#escalation details .esc-cell", "buttons": "#escalation details .esc-call",
               "details": "#escalation details"}
    steps = [{"click": "#escalation details .esc-call", "text": "Asks for a manager"}]
    facts = run_page(tmp_path, _esc_scenario(steps=steps, queries=queries))
    assert len(facts["queries"]["cells"]) == 4 and len(facts["queries"]["buttons"]) == 3
    # opening the card does not depend on the panel being open
    assert facts["queries"]["details"][0]["open"] is False
    assert "call-wants_manager" in [c["id"] for c in facts["cards"] if c["open"]]


def test_matrix_grid_keeps_labels_and_numbers_together():
    css = _css()
    rule = re.search(r"\.esc-grid\s*\{([^}]*)\}", css).group(1)
    assert "max-content" in rule and "grid-template-columns: auto" not in rule


# ---- O1 fix round 2 ----

@needs_node
def test_no_handoff_average_means_no_line_under_the_matrix(tmp_path):
    report = _report_current()
    report["escalation"]["handoff_avg"] = None
    queries = {"line": "#escalation .esc-summary", "summary": "#escalation summary"}
    facts = run_page(tmp_path, _esc_scenario(report, queries=queries))
    assert facts["queries"]["line"] == []
    assert facts["queries"]["summary"][0]["text"].startswith("Escalation: accuracy 90%")


@needs_node
def test_opened_details_never_repeats_the_summary_sentence(tmp_path):
    facts = run_page(tmp_path, _esc_scenario(queries={"all": "#escalation details"}))
    text = facts["queries"]["all"][0]["text"]
    assert text.count("accuracy") == 1 and text.count("runs)") == 1


@needs_node
def test_escalation_button_open_scrolls_card_top_and_sets_the_one_shot_flag(tmp_path):
    queries = {"card": "#call-consumer_court"}
    steps = [{"click": "#escalation .esc-call", "text": "Threatens consumer court"}]
    facts = run_page(tmp_path, _esc_scenario(steps=steps, queries=queries))
    card = facts["queries"]["card"][0]
    assert card["open"] is True and card["scrollBlocks"] == ["start"]
    assert "data-top" in card["attrs"]  # the toggle handler sees this and skips its failing turn scroll


@needs_node
def test_summary_click_and_fix_buttons_do_not_set_the_flag(tmp_path):
    queries = {"card": "#call-consumer_court", "fix": "#call-brother_asks"}
    steps = [{"click": "#call-consumer_court summary"}]
    facts = run_page(tmp_path, _esc_scenario(steps=steps, queries=queries))
    assert "data-top" not in facts["queries"]["card"][0]["attrs"]
    scenario = _scenario(_report())
    scenario["clicks"].append("button.fixcall:Wants a manager")
    scenario["queries"] = {"card": "#call-wants_manager"}
    card = run_page(tmp_path, scenario)["queries"]["card"][0]
    assert card["open"] is True and "data-top" not in card["attrs"]


def test_toggle_handler_honours_the_one_shot_flag_before_scrolling_the_turn():
    html = PAGE.read_text(encoding="utf-8")
    toggle = html[html.index("card.addEventListener('toggle'"):]
    toggle = toggle[:toggle.index("return card;")]
    guard = r"if \(card\.hasAttribute\('data-top'\)\) \{\s*card\.removeAttribute\('data-top'\);\s*return;"
    assert re.search(guard, toggle)
    assert toggle.index("data-top") < toggle.index(".turn.failing")
    assert "removeAttribute('data-top')" in toggle


def test_closed_panel_margins_are_pinned_to_the_eight_pixel_grid():
    css = _css()
    rules = re.findall(r"#escalation\s*\{([^}]*)\}", css)
    assert len(rules) == 1, "one rule sets the panel's spacing"
    assert "margin: 0 0 16px;" in rules[0] and "padding-top" not in rules[0]
    assert re.search(r"\.esc-line\s*\{[^}]*padding: 8px 0;[^}]*line-height: 24px", css)  # a 40px summary line


# ---- A5d / O2: trait filter and the misheard chip ----

TRAIT_LABELS_JS = re.compile(r"var TRAIT_LABELS = \{(.*?)\};", re.S)


def _trait_scenario(report=None, steps=(), queries=None):
    queries = {"select": "#trait-filter", "options": "#trait-filter option", "bar": "#trait-bar",
               "cards": "#calls details.call", **(queries or {})}
    return _r3_scenario(steps, queries, report=report or _report_current())


def _choose(value):
    return {"select": "#trait-filter", "value": value}


def _visible(facts):
    return [c["id"] for c in facts["queries"]["cards"] if not c["hidden"]]


@needs_node
def test_trait_filter_lists_pass_rate_per_trait(tmp_path):
    block = _report_current()["traits"]
    facts = run_page(tmp_path, _trait_scenario())
    assert facts["queries"]["bar"][0]["hidden"] is False
    options = facts["queries"]["options"]
    assert options[0]["text"] == "All traits" and options[0]["value"] == ""
    assert [o["value"] for o in options[1:]] == list(block)  # the report's own order
    assert facts["queries"]["select"][0]["value"] == ""
    by_value = {o["value"]: o["text"] for o in options}
    assert by_value["hindi_numbers"] == "Hindi number words: 50% (1 of 2)"
    assert by_value["en_to_hi_switch"] == "English to Hindi switch: 0% (0 of 1)"
    assert by_value["kannada_english"] == "Kannada English: 100% (1 of 1)"
    for value, stats in block.items():
        assert f"{round(stats['pass_rate'] * 100)}% ({stats['passed']} of {stats['total']})" in by_value[value]


@needs_node
def test_trait_filter_hides_other_cards(tmp_path):
    report = _report_current()
    facts = run_page(tmp_path, _trait_scenario(steps=[_choose("hindi_numbers")]))
    expect = {"call-" + c["persona_id"] for c in report["calls"] if "hindi_numbers" in c["traits"]}
    assert expect == {"call-lakh_question", "call-hindi_amount_partial"}
    assert set(_visible(facts)) == expect
    assert len(facts["queries"]["cards"]) == len(report["calls"])  # hidden, not removed
    other = run_page(tmp_path, _trait_scenario(steps=[_choose("indian_dates")]))
    assert _visible(other) == ["call-agle_mahine_date"]


@needs_node
def test_misheard_chip_on_v2_partial_amount(tmp_path):
    report = _report_current()
    assert [c["persona_id"] for c in report["calls"] if c["failure_kind"] == "misheard"] == ["hindi_amount_partial"]
    facts = run_page(tmp_path, _trait_scenario(queries={
        "all": "#calls .chip.misheard", "inside": "#call-hindi_amount_partial summary .chip.misheard"}))
    assert [c["text"] for c in facts["queries"]["all"]] == ["misheard"]  # one chip, on no other card
    assert [c["text"] for c in facts["queries"]["inside"]] == ["misheard"]


@needs_node
def test_select_hidden_without_a_traits_block_or_with_an_empty_one(tmp_path):
    for value in ("missing", None, {}):
        report = _report_current()
        if value == "missing":
            del report["traits"]
        else:
            report["traits"] = value
        facts = run_page(tmp_path, _trait_scenario(report=report))
        assert facts["queries"]["bar"][0]["hidden"] is True, value
        assert len(facts["queries"]["cards"]) == len(report["calls"])
        assert len(_visible(facts)) == len(report["calls"])


@needs_node
def test_a_single_trait_report_hides_the_select_so_the_story_view_gains_no_height(tmp_path):
    report = _live_v1_report()
    assert list(report["traits"]) == ["hindi_numbers"]  # the story set carries one trait
    facts = run_page(tmp_path, _trait_scenario(report=report, queries={"chips": "#calls .chip.misheard"}))
    assert facts["queries"]["bar"][0]["hidden"] is True
    assert facts["queries"]["options"] == []  # nothing is even built for one trait
    assert facts["queries"]["chips"] == []
    assert facts["passRate"]["text"] == "70%"
    assert facts["marks"] == ["12,400"] * 5
    failed = [c for c in facts["cards"] if "failed" in c["className"].split()]
    assert len(failed) == 6 and len(facts["cards"]) == 20
    assert facts["cards"][0]["id"] == "call-brother_asks"
    assert [c["id"] for c in facts["cards"] if c["open"]] == ["call-brother_asks"]
    assert len(_visible(facts)) == 20


def test_the_bar_sits_between_the_panels_and_the_cards_and_is_one_compact_line():
    html = PAGE.read_text(encoding="utf-8")
    assert re.search(r'<div id="trait-bar" class="trait-bar" hidden>', html)
    assert html.index('id="escalation"') < html.index('id="trait-bar"') < html.index('id="calls"')
    rule = re.search(r"\.trait-bar \{([^}]*)\}", _css())
    assert rule and "display: flex" in rule.group(1) and "margin: 0 0 8px" in rule.group(1)


@needs_node
def test_the_first_failing_card_is_open_and_visible_until_a_trait_is_chosen(tmp_path):
    report = _report_current()
    facts = run_page(tmp_path, _trait_scenario())
    first = report["next_fixes"][0]["calls"][0]["persona_id"]
    assert [c["id"] for c in facts["queries"]["cards"] if c["open"]] == ["call-" + first]
    assert len(_visible(facts)) == len(report["calls"])
    assert facts["queries"]["select"][0]["value"] == ""


@needs_node
def test_choosing_all_traits_restores_every_card(tmp_path):
    report = _report_current()
    facts = run_page(tmp_path, _trait_scenario(steps=[_choose("tamil_english"), _choose("")]))
    assert len(_visible(facts)) == len(report["calls"])
    only = run_page(tmp_path, _trait_scenario(steps=[_choose("tamil_english")]))
    assert _visible(only) == ["call-tamil_first"]


@needs_node
def test_a_card_with_two_traits_shows_under_either_filter(tmp_path):
    report = _report_current()
    for call in report["calls"]:
        if call["persona_id"] == "kannada_friday":
            call["traits"] = ["kannada_english", "indian_dates"]
    for trait, expect in (("kannada_english", ["call-kannada_friday"]),
                          ("indian_dates", ["call-kannada_friday", "call-agle_mahine_date"])):
        facts = run_page(tmp_path, _trait_scenario(report=report, steps=[_choose(trait)]))
        assert sorted(_visible(facts)) == sorted(expect), trait


@needs_node
def test_a_panel_button_for_a_hidden_card_clears_the_filter_first(tmp_path):
    steps = [_choose("tamil_english"), {"click": "button.fixcall", "text": "Asks for a manager"}]
    facts = run_page(tmp_path, _trait_scenario(steps=steps))
    assert "call-wants_manager" in _visible(facts)
    assert len(_visible(facts)) == len(_report_current()["calls"])
    assert facts["queries"]["select"][0]["value"] == ""


@needs_node
def test_trait_names_and_rates_reach_the_dom_as_text_not_markup(tmp_path):
    report = _report_current()
    report["traits"] = {"<b>x</b>": {"passed": 1, "total": 2, "pass_rate": 0.5},
                        "hindi_numbers": report["traits"]["hindi_numbers"]}
    facts = run_page(tmp_path, _trait_scenario(report=report))
    assert [o["text"] for o in facts["queries"]["options"]][1] == "<b>x</b>: 50% (1 of 2)"  # as plain text
    html = PAGE.read_text(encoding="utf-8")
    body = html[html.index("function renderTraitFilter"):]
    body = body[:body.index("\n  }\n")]
    assert "innerHTML" not in body and "el('option'" in body


def test_trait_labels_in_the_page_match_the_persona_sets_labels():
    from callqa.persona_sets import TRAITS

    block = TRAIT_LABELS_JS.search(PAGE.read_text(encoding="utf-8")).group(1)
    assert dict(re.findall(r"(\w+): '([^']*)'", block)) == {key: value["label"] for key, value in TRAITS.items()}


# ---- O2 fix round 1: the n key with a trait chosen, panel clicks, new runs, rounding, prototype ids ----

def _key(target="body", key="n"):
    return {"key": key, "target": target}


def _opened(facts):
    return [c["id"] for c in facts["queries"]["cards"] if c["open"]]


def _failed_ids(report):
    return ["call-" + c["persona_id"] for c in report["calls"] if not c["passed"]]


@needs_node
def test_n_opens_the_next_failing_card_in_page_order(tmp_path):
    facts = run_page(tmp_path, _trait_scenario(steps=[_key("#calls"), _key("#calls")],
                                               queries={"failed": "#calls details.call.failed"}))
    page_order = [c["id"] for c in facts["queries"]["failed"]]
    assert page_order[:2] == ["call-flaky_line", "call-hindi_amount_partial"]
    assert _opened(facts) == page_order[:2]  # the first was open already, the second n opens the next one
    once = run_page(tmp_path, _trait_scenario(steps=[_key("#calls")], queries={"failed": "#calls details.call.failed"}))
    assert _opened(once) == ["call-flaky_line"]
    other = run_page(tmp_path, _trait_scenario(steps=[_key("#calls", "x"), _key("#calls", "x")]))
    assert _opened(other) == ["call-flaky_line"]  # only n moves on


@needs_node
def test_n_skips_cards_the_trait_filter_hides(tmp_path):
    steps = [_choose("tamil_english"), _key("#calls")]
    facts = run_page(tmp_path, _trait_scenario(steps=steps))
    assert "call-tamil_first" in _opened(facts)
    assert len(_visible(facts)) == 1
    two = run_page(tmp_path, _trait_scenario(steps=[_choose("hindi_numbers"), _key("#calls"), _key("#calls")]))
    opened = set(_opened(two))
    assert "call-hindi_amount_partial" in opened
    assert opened <= {"call-flaky_line", "call-hindi_amount_partial"}  # no hidden card opened


@needs_node
def test_n_works_while_the_trait_select_has_focus(tmp_path):
    steps = [_choose("tamil_english"), _key("#trait-filter")]
    facts = run_page(tmp_path, _trait_scenario(steps=steps))
    assert "call-tamil_first" in _opened(facts)


@needs_node
def test_n_is_ignored_in_the_sop_textarea_and_in_other_selects(tmp_path):
    for target in ("#sop-input", "#agent-select", "#set-select"):
        facts = run_page(tmp_path, _trait_scenario(steps=[_key(target), _key(target)]))
        assert _opened(facts) == ["call-flaky_line"], target


@needs_node
def test_a_visible_cards_panel_button_keeps_the_trait_choice(tmp_path):
    steps = [_choose("tamil_english"), {"click": "button.fixcall", "text": "Tamil English"}]
    facts = run_page(tmp_path, _trait_scenario(steps=steps))
    assert _visible(facts) == ["call-tamil_first"]
    assert facts["queries"]["select"][0]["value"] == "tamil_english"
    assert "call-tamil_first" in _opened(facts)


@needs_node
def test_a_new_run_rebuilds_the_filter_and_shows_every_card(tmp_path):
    report = _report_current()
    steps = [_choose("tamil_english"), {"click": "#run-button"}]
    facts = run_page(tmp_path, _trait_scenario(report=report, steps=steps))
    assert len(_visible(facts)) == len(report["calls"])
    options = facts["queries"]["options"]
    assert [o["value"] for o in options] == [""] + list(report["traits"])  # rebuilt once, not appended twice
    assert facts["queries"]["bar"][0]["hidden"] is False


@needs_node
def test_a_rate_of_one_eighth_rounds_to_13_percent_like_the_headline(tmp_path):
    report = _report_current()
    report["traits"] = {"hindi_numbers": {"passed": 1, "total": 8, "pass_rate": 0.125},
                        "indian_dates": {"passed": 7, "total": 8, "pass_rate": 0.875}}
    facts = run_page(tmp_path, _trait_scenario(report=report))
    texts = [o["text"] for o in facts["queries"]["options"]]
    assert texts == ["All traits", "Hindi number words: 13% (1 of 8)", "Indian style dates: 88% (7 of 8)"]


@needs_node
def test_prototype_names_are_shown_as_ids_not_looked_up(tmp_path):
    report = _report_current()
    stats = {"passed": 1, "total": 2, "pass_rate": 0.5}
    report["traits"] = {"constructor": stats, "toString": stats, "hindi_numbers": stats}
    facts = run_page(tmp_path, _trait_scenario(report=report))
    texts = [o["text"] for o in facts["queries"]["options"]]
    assert texts[1:] == ["constructor: 50% (1 of 2)", "toString: 50% (1 of 2)", "Hindi number words: 50% (1 of 2)"]


# ---- O4: the compliance panel and critical cards ----

FIXTURE_STORY_RULES = HERE / "fixtures" / "report_story_v1_rules.json"
# in the order the fix list gives them
CRITICAL_IDS = ["brother_asks", "wife_handles_money", "wrong_number", "prompt_injection", "fake_head_office"]


def _story_rules():
    return json.loads(FIXTURE_STORY_RULES.read_text(encoding="utf-8"))


def _rules_config():
    """What GET /api/config says for the default rules file, built from the real file."""
    from callqa.compliance import DEFAULT_RULES_PATH, load_rules
    from callqa.server import rules_summary

    rules = load_rules(DEFAULT_RULES_PATH)
    return {"rules_file": rules.label, "rules": rules_summary(rules)}


def _cmp_scenario(report=None, steps=(), queries=None, config=True):
    queries = {"panel": "#compliance", "summary": "#compliance summary", "details": "#compliance details",
               "rows": "#compliance .cmp-rule", "cards": "#calls details.call", **(queries or {})}
    return _r3_scenario(steps, queries, report=report or _story_rules(), config=_rules_config() if config else None)


@needs_node
def test_compliance_panel_is_above_next_fix(tmp_path):
    html = PAGE.read_text(encoding="utf-8")
    assert html.count('id="compliance"') == 1
    assert html.index('id="cost-latency"') < html.index('id="compliance"') < html.index('id="next-fix"')
    assert re.search(r'<div class="panel" id="compliance" hidden></div>', html)  # empty until the script fills it
    facts = run_page(tmp_path, _cmp_scenario())
    assert facts["queries"]["panel"][0]["hidden"] is False
    # one closed details, one summary line: nothing else sits in the panel
    assert facts["queries"]["panel"][0]["childTags"] == ["details"]
    assert len(facts["queries"]["details"]) == 1 and facts["queries"]["details"][0]["open"] is False
    assert [s["text"] for s in facts["queries"]["summary"]] == ["Compliance: 5 critical, 1 rule not evaluated offline"]


@needs_node
def test_summary_wording_counts_callers_and_rules(tmp_path):
    report = _story_rules()
    report["compliance"]["not_evaluated"].append({"rule_id": "extra", "why": "judge based"})
    text = run_page(tmp_path, _cmp_scenario(report))["queries"]["summary"][0]["text"]
    assert text == "Compliance: 5 critical, 2 rules not evaluated offline"
    report["compliance"]["failures"][0]["severity"] = "major"  # a major failure is not counted as critical
    text = run_page(tmp_path, _cmp_scenario(report))["queries"]["summary"][0]["text"]
    assert text == "Compliance: no critical failures, 2 rules not evaluated offline"
    report["compliance"]["not_evaluated"] = []
    text = run_page(tmp_path, _cmp_scenario(report))["queries"]["summary"][0]["text"]
    assert text == "Compliance: no critical failures" and "\n" not in text


@needs_node
def test_critical_cards_sort_first(tmp_path):
    report = _story_rules()
    report["next_fixes"].reverse()  # the major failure now leads the fix list, the critical cards must still lead
    facts = run_page(tmp_path, _cmp_scenario(report, queries={"chips": "#calls summary .chip.critical"}))
    ids = [c["id"] for c in facts["queries"]["cards"]]
    assert ids[:5] == ["call-" + i for i in CRITICAL_IDS]  # critical first, in the order they had in the report
    assert ids[5] == "call-wants_manager"
    assert [c["id"] for c in facts["cards"] if c["open"]] == ["call-brother_asks"]  # the first failing card opens
    assert [c["text"] for c in facts["queries"]["chips"]] == ["critical"] * 5
    wants = next(c for c in facts["queries"]["cards"] if c["id"] == "call-wants_manager")
    assert "critical" not in wants["className"].split()
    assert ids[6:] == ["call-" + c["persona_id"] for c in report["calls"] if c["passed"]]  # passes keep their order


@needs_node
def test_the_v1_money_shot_survives_the_new_order(tmp_path):
    facts = run_page(tmp_path, _cmp_scenario())
    assert facts["passRate"]["text"] == "70%"
    assert [c["id"] for c in facts["cards"] if c["open"]] == ["call-brother_asks"]
    assert facts["cards"][0]["id"] == "call-brother_asks" and facts["cards"][5]["id"] == "call-wants_manager"
    assert facts["marks"] == ["12,400"] * 5
    assert len([c for c in facts["cards"] if "failed" in c["className"].split()]) == 6
    assert facts["nextFix"]["buttons"][0] == "Brother asks for the amount, Turn 2"


def test_critical_chip_is_a_chip_and_red_pen_means_fails():
    css = _css()
    rule = re.search(r"\.chip\.critical\s*\{([^}]*)\}", css).group(1)
    assert "color: var(--fail)" in rule
    assert "font-size" not in rule and "padding" not in rule  # same size as the other chips


@needs_node
def test_unverified_label_on_the_rules_that_say_so(tmp_path):
    facts = run_page(tmp_path, _cmp_scenario(queries={
        "unverified": "#compliance .cmp-tag.unverified", "verified": "#compliance .cmp-tag.verified"}))
    assert len(facts["queries"]["rows"]) == 5
    assert [t["text"] for t in facts["queries"]["unverified"]] == ["UNVERIFIED", "UNVERIFIED"]
    assert [t["text"] for t in facts["queries"]["verified"]] == ["verified"] * 3
    labelled = {}
    for row in facts["queries"]["rows"]:
        tag = next(c for c in row["childClasses"] if c.startswith("cmp-tag"))
        labelled.setdefault(tag, []).append(row["childTexts"][0])
    assert len(labelled["cmp-tag unverified"]) == 2
    assert any("names the lender" in n for n in labelled["cmp-tag unverified"])  # identify_company
    assert any("grievance" in n for n in labelled["cmp-tag unverified"])  # grievance_channel


@needs_node
def test_rows_say_passed_failed_or_not_evaluated_offline(tmp_path):
    facts = run_page(tmp_path, _cmp_scenario())
    texts = {r["childTexts"][0]: " | ".join(r["childTexts"][1:]) for r in facts["queries"]["rows"]}
    third = next(v for k, v in texts.items() if k.startswith("Give no amount"))
    assert "critical" in third and "5 callers" in third
    hours = next(v for k, v in texts.items() if k.startswith("Call borrowers only"))
    assert "passed" in hours and "critical" in hours
    grievance = next(v for k, v in texts.items() if "grievance" in k)
    assert "not evaluated offline" in grievance
    assert "judge based rule; offline runs deterministic rules only" in grievance
    assert "Not legal advice" not in " ".join(texts.values())  # the disclaimer is its own block, not a row


@needs_node
def test_disclaimer_comes_first_inside_the_panel(tmp_path):
    queries = {"note": "#compliance .cmp-disclaimer", "body": "#compliance .cmp-body"}
    facts = run_page(tmp_path, _cmp_scenario(queries=queries))
    assert facts["queries"]["note"][0]["text"] == _story_rules()["compliance"]["disclaimer"]
    assert facts["queries"]["body"][0]["childClasses"][0] == "cmp-disclaimer"


@needs_node
def test_without_rule_titles_the_rows_fall_back_to_ids_and_claim_nothing(tmp_path):
    facts = run_page(tmp_path, _cmp_scenario(config=False, queries={"tags": "#compliance .cmp-tag"}))
    names = [r["childTexts"][0] for r in facts["queries"]["rows"]]
    assert "calling_hours" in names and "identify_company" in names and "grievance_channel" in names
    assert any(n.startswith("Give no amount") for n in names)  # a failing rule carries its own title
    assert [t["text"] for t in facts["queries"]["tags"]] == ["verified"]  # only the one title that is known


@needs_node
def test_compliance_caller_buttons_open_the_right_card(tmp_path):
    steps = [{"click": "#compliance .cmp-call", "text": "wrong number"}]
    facts = run_page(tmp_path, _cmp_scenario(steps=steps, queries={"buttons": "#compliance .cmp-call"}))
    texts = [b["text"] for b in facts["queries"]["buttons"]]
    assert texts == ["Brother asks for the amount", "Claims to be head office", "Tries a prompt injection",
                     "Wife handles the money", "Says it is a wrong number"]
    assert sorted(c["id"] for c in facts["cards"] if c["open"]) == ["call-brother_asks", "call-wrong_number"]
    assert facts["queries"]["details"][0]["open"] is False  # opening a card does not depend on the panel


@needs_node
def test_panel_absent_when_compliance_is_null_or_missing(tmp_path):
    report = _story_rules()
    report["compliance"] = None
    for call in report["calls"]:
        call["compliance"], call["severity"] = [], (None if call["passed"] else "major")
    facts = run_page(tmp_path, _cmp_scenario(report, queries={"chips": "#calls .chip.critical"}))
    assert facts["queries"]["panel"][0]["hidden"] is True and facts["queries"]["panel"][0]["childTags"] == []
    assert facts["queries"]["chips"] == []
    del report["compliance"]  # older records have no key at all
    assert run_page(tmp_path, _cmp_scenario(report))["queries"]["panel"][0]["hidden"] is True
    old = run_page(tmp_path, _esc_scenario(_report_r3(), queries={"panel": "#compliance"}))
    assert old["queries"]["panel"][0]["hidden"] is True and len(old["cards"]) > 0


@needs_node
def test_a_second_run_rebuilds_the_panel_not_appends(tmp_path):
    scenario = _cmp_scenario()
    scenario["clicks"] = ["#run-button", "#run-button"]
    facts = run_page(tmp_path, scenario)
    assert len(facts["queries"]["details"]) == 1 and len(facts["queries"]["rows"]) == 5


@needs_node
def test_n_key_follows_the_new_card_order(tmp_path):
    facts = run_page(tmp_path, _cmp_scenario(steps=[_key("#calls"), _key("#calls")]))
    # the first n opens failed[0] (already open), the second the next failing card in page order
    assert sorted(c["id"] for c in facts["cards"] if c["open"]) == ["call-brother_asks", "call-wife_handles_money"]


@needs_node
def test_trait_filter_still_hides_and_critical_cards_still_lead(tmp_path):
    report = _report_current()
    failed = [c for c in report["calls"] if not c["passed"]]
    last = failed[-1]
    assert failed[0] is not last
    last["severity"] = "critical"
    facts = run_page(tmp_path, _trait_scenario(report=report, queries={"chips": "#calls .chip.critical"}))
    assert facts["queries"]["cards"][0]["id"] == "call-" + last["persona_id"]
    assert len(facts["queries"]["chips"]) == 1
    assert facts["queries"]["bar"][0]["hidden"] is False
    trait = "hindi_numbers"
    chosen = run_page(tmp_path, _trait_scenario(report=report, steps=[_choose(trait)]))
    assert set(_visible(chosen)) == {"call-" + c["persona_id"] for c in report["calls"] if trait in c["traits"]}


def test_compliance_panel_spacing_is_on_the_eight_pixel_grid():
    css = _css()
    rules = re.findall(r"#compliance\s*\{([^}]*)\}", css)
    assert len(rules) == 1 and "margin: 0 0 8px;" in rules[0] and "padding-top" not in rules[0]
    assert re.search(r"\.cmp-line\s*\{[^}]*padding: 4px 0;[^}]*line-height: 24px", css)  # one 32px summary line
    assert ".cmp-call:hover" in css and ".cmp-line:hover" in css  # hover states on the new controls
    assert ".cmp-call:focus-visible" in css and ".cmp-line:focus-visible" in css  # and focus states
    block = "".join(re.findall(r"(?:#compliance|\.cmp-)[^{]*\{[^}]*\}", css))
    assert not re.search(r"#[0-9a-fA-F]{3,6}\b", block)  # tokens only, no new colours


# ---- O5: the bake off table on the records page ----

FIXTURE_COMPARE = HERE / "fixtures" / "compare_example.json"
OPEN_BAKEOFF = [{"select": "#bakeoff-select", "value": "compare_full_offline.json"}, {"click": "#bakeoff-open"}]


def _compare():
    return json.loads(FIXTURE_COMPARE.read_text(encoding="utf-8"))


def _run_file_record(config, persona_ids):
    """A tiny run record: every persona has a two turn transcript that names the config."""
    calls = [{"persona_id": pid, "name": pid, "passed": config == "v3", "reason": f"{config} reason for {pid}",
              "transcript": [{"role": "user", "content": f"caller says {pid}"},
                             {"role": "assistant", "content": f"{config} agent answers <b>{pid}</b>"}]}
             for pid in persona_ids]
    return {"kind": "run", "calls": calls}


def _bakeoff_scenario(record=None, steps=OPEN_BAKEOFF, queries=None):
    record = record or _compare()
    scenario = _records_scenario(steps, queries)
    listing = scenario["routes"]["GET /api/records"]["body"]
    listing.insert(0, {"name": "compare_full_offline.json", "kind": "compare", "agent": "bake off v1, v2, v3",
                       "set": "full", "repeats": record.get("repeats"), "backend": "offline", "pass_rate": None,
                       "created_at": "2026-11-22T02:12:06+05:30", "synthetic": True})
    scenario["routes"]["GET /api/records/compare_full_offline.json"] = {"body": record}
    for entry in record["configs"]:
        scenario["routes"]["GET /api/records/" + entry["run_file"]] = {
            "body": _run_file_record(entry["config"], record["disagreements"])}
    return scenario


def _bakeoff(tmp_path, record=None, steps=OPEN_BAKEOFF, queries=None):
    return run_page(tmp_path, _bakeoff_scenario(record, steps, queries), page=RECORDS_PAGE)


BAKEOFF_QUERIES = {"heads": "#bakeoff-table th.config", "rows": "#bakeoff-table tr.metric"}


def _best_by_row(facts):
    """metric row class -> the config names (column headers) whose cell is marked best."""
    heads = [h["text"] for h in facts["queries"]["heads"]]
    out = {}
    for row in facts["queries"]["rows"]:
        metric = next(c for c in row["className"].split() if c.startswith("m-"))
        out[metric] = [heads[i] for i, cls in enumerate(row["childClasses"][1:]) if "best" in cls.split()]
    return out


@needs_node
def test_compare_table_highlights_winner_per_column(tmp_path):
    facts = _bakeoff(tmp_path, queries=BAKEOFF_QUERIES)
    assert [x["text"] for x in facts["queries"]["heads"]] == ["v1", "v2", "v3"]
    best = _compare()["best"]
    expected = {"m-pass": best["pass"], "m-critical": best["critical"], "m-escalation": best["escalation"],
                "m-cost": best["cost"], "m-p95": best["p95"]}
    assert _best_by_row(facts) == expected
    assert expected["m-p95"] == ["v1", "v2"]  # the fixture has a real tie and both are highlighted
    assert "GET /api/records/compare_full_offline.json" in facts["fetchLog"]


@needs_node
def test_a_tie_highlights_every_tied_config_and_no_other(tmp_path):
    record = _compare()
    record["best"]["pass"] = ["v2", "v3"]
    record["best"]["cost"] = ["v1", "v2", "v3"]
    best = _best_by_row(_bakeoff(tmp_path, record, queries=BAKEOFF_QUERIES))
    assert best["m-pass"] == ["v2", "v3"] and best["m-cost"] == ["v1", "v2", "v3"]
    assert best["m-escalation"] == ["v3"]


@needs_node
def test_winner_is_marked_by_form_not_colour_alone(tmp_path):
    facts = _bakeoff(tmp_path, queries={**BAKEOFF_QUERIES, "marks": "#bakeoff-table td.best mark"})
    best_cells = sum(len(v) for v in _best_by_row(facts).values())
    assert len(facts["queries"]["marks"]) == best_cells == 7  # one mark element in every winning cell
    for row in facts["queries"]["rows"]:
        assert "best" in " ".join(row["childTexts"][1:])  # and the visible word "best" in every row
    css = _css(RECORDS_PAGE)
    assert re.search(r"\.cell\.best\s*\{[^}]*font-weight:\s*700", css)  # and a bold weight
    assert re.search(r"mark\s*\{[^}]*background:\s*var\(--mark\)", css)  # highlighter on mark only
    assert not re.search(r"\.cell[^{]*\{[^}]*var\(--mark\)", css)


@needs_node
def test_zero_critical_is_not_red_and_a_nonzero_count_may_be(tmp_path):
    facts = _bakeoff(tmp_path, queries={"crit": "#bakeoff-table tr.m-critical"})
    classes = [c.split() for c in facts["queries"]["crit"][0]["childClasses"][1:]]
    assert ["bad" in c for c in classes] == [True, False, False]  # v1 has 5, v2 and v3 have 0


@needs_node
def test_recommendation_and_labels_are_shown(tmp_path):
    record = _compare()
    facts = _bakeoff(tmp_path, queries={"rec": "#bakeoff-recommendation", "labels": "#bakeoff-labels",
                                        "note": "#bakeoff-note"})
    assert facts["queries"]["rec"][0]["text"] == record["recommendation"]["text"]
    assert "Cost per call: simulated" in facts["queries"]["labels"][0]["text"]
    assert "Latency: simulated" in facts["queries"]["labels"][0]["text"]
    assert record["note"] in facts["queries"]["note"][0]["text"]


@needs_node
def test_simulated_label_shown_offline(tmp_path):
    queries = {"sim": "#bakeoff-simulated"}
    shown = _bakeoff(tmp_path, queries=queries)["queries"]["sim"][0]
    assert shown["hidden"] is False and "simulated" in shown["text"].lower()
    record = _compare()
    record["simulated"] = False
    record["cost_label"] = record["latency_label"] = "measured"
    live = _bakeoff(tmp_path, record, queries={**queries, "labels": "#bakeoff-labels"})["queries"]
    assert live["sim"][0]["hidden"] is True
    assert "measured" in live["labels"][0]["text"] and "simulated" not in live["labels"][0]["text"]


@needs_node
def test_disagreement_links_open_both_transcripts(tmp_path):
    record = _compare()
    first = record["disagreements"][0]
    steps = OPEN_BAKEOFF + [{"click": "button.dis-link", "text": "v1"}, {"click": "button.dis-link", "text": "v3"}]
    facts = _bakeoff(tmp_path, steps=steps, queries={"rows": "#bakeoff-disagreements .dis-row",
                                                     "links": "#bakeoff-disagreements button.dis-link",
                                                     "ts": "#bakeoff-disagreements .dis-transcript",
                                                     "turns": "#bakeoff-disagreements .dis-transcript .turn"})
    assert len(facts["queries"]["rows"]) == len(record["disagreements"])
    assert len(facts["queries"]["links"]) == 3 * len(record["disagreements"])  # one per config per persona
    v1, v3 = record["configs"][0]["run_file"], record["configs"][2]["run_file"]
    assert f"GET /api/records/{v1}" in facts["fetchLog"] and f"GET /api/records/{v3}" in facts["fetchLog"]
    texts = [t["text"] for t in facts["queries"]["ts"]]
    assert len(texts) == 2 and "v1 reason for " + first in texts[0] and "v3 reason for " + first in texts[1]
    assert f"v1 agent answers <b>{first}</b>" in texts[0]  # markup in a transcript stays text
    assert len(facts["queries"]["turns"]) == 4


@needs_node
def test_a_failed_transcript_fetch_shows_a_message(tmp_path):
    record = _compare()
    scenario = _bakeoff_scenario(record, OPEN_BAKEOFF + [{"click": "button.dis-link", "text": "v2"}],
                                 {"err": "#bakeoff-disagreements .dis-error", "ts": ".dis-transcript"})
    scenario["routes"]["GET /api/records/" + record["configs"][1]["run_file"]] = {
        "status": 404, "body": {"error": "unknown record"}}
    facts = run_page(tmp_path, scenario, page=RECORDS_PAGE)
    assert len(facts["queries"]["err"]) == 1 and "v2" in facts["queries"]["err"][0]["text"]
    assert "unknown record" in facts["queries"]["err"][0]["text"] and facts["queries"]["ts"] == []


@needs_node
def test_a_persona_missing_from_a_run_file_says_so(tmp_path):
    record = _compare()
    scenario = _bakeoff_scenario(record, OPEN_BAKEOFF + [{"click": "button.dis-link", "text": "v1"}],
                                 {"err": "#bakeoff-disagreements .dis-error"})
    scenario["routes"]["GET /api/records/" + record["configs"][0]["run_file"]] = {"body": {"calls": []}}
    facts = run_page(tmp_path, scenario, page=RECORDS_PAGE)
    assert "not in" in facts["queries"]["err"][0]["text"]


@needs_node
def test_only_compare_records_fill_the_bakeoff_picker_and_run_pickers_stay_run_only(tmp_path):
    facts = _bakeoff(tmp_path, steps=(), queries={"b": "#bakeoff-select option", "old": "#old-select option",
                                                  "res": "#bakeoff-result"})
    assert [o["value"] for o in facts["queries"]["b"]] == ["compare_full_offline.json"]
    assert [o["value"] for o in facts["queries"]["old"]] == ["new_v2.json", "old_v1.json"]
    assert facts["queries"]["res"][0]["hidden"] is True  # nothing shows until a bake off is opened
    assert facts["fetchLog"] == ["GET /api/records"]


@needs_node
def test_bakeoff_panel_is_hidden_without_compare_records_and_the_diff_still_works(tmp_path):
    scenario = _records_scenario(COMPARE, {"panel": "#bakeoff-panel", "rows": "#diff-list .diff-row"})
    facts = run_page(tmp_path, scenario, page=RECORDS_PAGE)
    assert facts["queries"]["panel"][0]["hidden"] is True and len(facts["queries"]["rows"]) >= 3


@needs_node
def test_a_bakeoff_record_that_cannot_load_shows_the_page_error(tmp_path):
    scenario = _bakeoff_scenario(steps=OPEN_BAKEOFF, queries={"err": "#records-error", "res": "#bakeoff-result"})
    scenario["routes"]["GET /api/records/compare_full_offline.json"] = {
        "status": 404, "body": {"error": "unknown record"}}
    facts = run_page(tmp_path, scenario, page=RECORDS_PAGE)
    assert "unknown record" in facts["queries"]["err"][0]["text"] and facts["queries"]["res"][0]["hidden"] is True


def test_bakeoff_fixture_is_a_byte_copy_of_the_committed_sample():
    sample = HERE.parent / "samples" / "compare_full_offline.json"
    assert FIXTURE_COMPARE.read_bytes() == sample.read_bytes()


def test_bakeoff_controls_have_hover_and_focus_and_the_table_scrolls_in_its_own_box():
    css = _css(RECORDS_PAGE)
    for selector in ("#bakeoff-open:hover", ".dis-link:hover", ".dis-link:focus-visible"):
        assert selector in css, selector
    assert re.search(r"#bakeoff-scroll\s*\{[^}]*overflow-x:\s*auto", css)
