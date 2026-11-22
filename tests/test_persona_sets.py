"""Named persona sets: story is today's 20 (and STRATIFIED_5 below that), full is story then the extras."""
from pathlib import Path

import pytest

from callqa.persona_sets import EXTRAS, personas_for_set
from callqa.personas import STRATIFIED_5, generate_personas

SOP = (Path(__file__).resolve().parents[1] / "samples" / "sop_collections_acme.md").read_text(encoding="utf-8")


def ids(generation) -> list[str]:
    return [p["id"] for p in generation.personas]


def test_story_is_todays_twenty_unchanged():
    assert personas_for_set(SOP, "story").personas == generate_personas(SOP, 20, None).personas


def test_story_with_five_callers_is_stratified_five():
    assert ids(personas_for_set(SOP, "story", 5)) == STRATIFIED_5
    assert personas_for_set(SOP, "story", 5).personas == generate_personas(SOP, 5, None, story=True).personas


def test_full_set_is_story_then_extras_in_order():
    story, full = personas_for_set(SOP, "story").personas, personas_for_set(SOP, "full").personas
    assert full[:20] == story
    assert [p["id"] for p in full[20:]] == [spec["id"] for spec in EXTRAS] == [
        "flaky_line", "holder_says_bhai", "not_my_loan", "lost_job_no_date", "consumer_court",
        "hindi_amount_partial", "switch_to_hindi", "tamil_first", "kannada_friday", "clipped_paid",
        "agle_mahine_date"]


def test_full_set_rejects_callers():
    with pytest.raises(ValueError, match="--set full"):
        personas_for_set(SOP, "full", 5)


def test_unknown_set_is_an_error():
    with pytest.raises(ValueError, match="unknown persona set"):
        personas_for_set(SOP, "huge")


def test_extras_are_synthetic_template_callers_with_variants():
    extra = personas_for_set(SOP, "full").personas[20]
    assert extra["synthetic"] is True and len(extra["script_variants"]) == 3
    assert extra["must_include"] == ["friday|shukravar|shukrawar"]
    assert "Ravi" in extra["opening"]
