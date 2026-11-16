"""Judge disagreement is "needs review": not a pass, not a failure, no failing turn."""
from pathlib import Path

from callqa import report
from tests.test_story_hold import FAIL, PASS, FakeBackend, judged, persona, turns

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "callqa" / "static" / "index.html"


def call_for(result, pid="p1"):
    p = {**persona(id=pid), "name": pid, "language": "en", "tags": [], "fix_hint": "x"}
    return report.build_call(p, result)


def test_both_judge_runs_pass_is_passed():
    result = judged(FakeBackend(judge_replies=[PASS, PASS]))
    assert result.passed and not result.needs_review
    assert call_for(result)["needs_review"] is False


def test_both_judge_runs_fail_is_failed_and_in_failures():
    result = judged(FakeBackend(judge_replies=[FAIL, FAIL]))
    assert not result.passed and not result.needs_review
    fixes = report.group_next_fixes([call_for(result)])
    assert [c["persona_id"] for fix in fixes for c in fix["calls"]] == ["p1"]


def test_judge_disagreement_needs_review_not_passed_not_in_failures():
    result = judged(FakeBackend(judge_replies=[PASS, FAIL]))
    call = call_for(result)
    assert result.needs_review and not result.passed
    assert call["passed"] is False and call["needs_review"] is True
    assert call["failing_turn"] is None
    assert report.group_next_fixes([call]) == []


def test_report_with_one_of_each_counts_correctly():
    transcript = turns(("user", "a"), ("assistant", "b"))
    ok = persona(id="a")
    ok_result = judged(FakeBackend(judge_replies=[PASS, PASS]), ok)
    bad = judged(FakeBackend(judge_replies=[FAIL, FAIL]), persona(id="b"))
    review = judged(FakeBackend(judge_replies=[PASS, FAIL]), persona(id="c"))
    for r, pid in ((ok_result, "a"), (bad, "b"), (review, "c")):
        r.persona, r.transcript = pid, transcript

    class Gen:
        warnings: list = []
        source = "test"
        personas = [{**persona(id=i), "name": i, "language": "en", "tags": [], "fix_hint": "x"} for i in "abc"]

    class Backend:
        name = "fake"
        warnings: list = []

    built = report.build_report(run_id="r", sop_text="s", agent={"id": "v1", "label": "v1"}, backend=Backend(),
                                callers=3, max_turns=2, generation=Gen(), results=[ok_result, bad, review], seconds=1)
    assert (built["passed"], built["needs_review"], built["total"]) == (1, 1, 3)
    assert built["flaky"] == built["needs_review"]
    assert built["pass_rate"] == 0.3333
    assert [c["persona_id"] for fix in built["next_fixes"] for c in fix["calls"]] == ["b"]


def test_old_result_file_with_only_flaky_loads_with_the_same_count():
    old = {"passed": 17, "total": 20, "flaky": 2, "calls": [{"passed": True, "flaky": True}]}  # pre needs_review file
    assert report.needs_review_count(old) == 2
    assert report.needs_review_count({**old, "needs_review": 1, "flaky": 1}) == 1
    assert report.is_needs_review({"flaky": True}) and not report.is_needs_review({"flaky": False})


def test_page_says_need_review_and_never_counted_as_a_pass():
    html = PAGE.read_text(encoding="utf-8")
    assert "need review" in html
    assert "counted as a pass" not in html
    assert "needs review" in html
