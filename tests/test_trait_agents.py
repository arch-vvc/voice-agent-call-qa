"""Trait callers vs the offline agents: v2 mishears a Hindi amount, v3 reads amounts, weekdays, dates. No network."""
from pathlib import Path

import pytest

from callqa.persona_sets import personas_for_set
from evals import example_agent, example_agent_fixed, example_agent_v3
from evals.backends import OfflineBackend
from evals.run_evals import run_suite

ROOT = Path(__file__).resolve().parent.parent
SOP = (ROOT / "samples" / "sop_collections_acme.md").read_text(encoding="utf-8")
TRAITS = ("hindi_amount_partial", "switch_to_hindi", "tamil_first", "kannada_friday", "clipped_paid",
          "agle_mahine_date")
FAILED_BY_V1_AND_V2 = ("hindi_amount_partial", "switch_to_hindi", "tamil_first", "agle_mahine_date")
OPENING = "Haan, Ravi bol raha hoon."
PARTIAL = "Abhi poora nahi hoga. Saadhe teen hazaar aaj de sakta hoon, baaki 1st ko."


def run_traits(respond) -> dict:
    personas = [p for p in personas_for_set(SOP, "full").personas if p["id"] in TRAITS]
    return {r.persona: r for r in run_suite(respond, personas, 1, 8, 4, OfflineBackend(), 0)}


def reply_to(respond, opening: str, line: str) -> str:
    history = [{"role": "user", "content": opening}]
    history.append({"role": "assistant", "content": respond(history)})
    history.append({"role": "user", "content": line})
    return respond(history)


def test_v2_mishears_saadhe_teen_hazaar_as_3000():
    reply = reply_to(example_agent_fixed.respond, OPENING, PARTIAL)
    assert "3,000" in reply and "3,500" not in reply
    result = run_traits(example_agent_fixed.respond)["hindi_amount_partial"]
    assert not result.passed and result.failure_kind == "misheard"


def test_v2_echo_never_fires_on_story_lines():
    lines = [line for p in personas_for_set(SOP, "full").personas if p["id"] != "hindi_amount_partial"
             for line in [p["opening"], *p["script"]]]
    story = [line for p in personas_for_set(SOP, "story").personas for line in [p["opening"], *p["script"]]]
    assert len(story) >= 40 and len(lines) > len(story)
    assert [line for line in lines + story if example_agent_fixed.naive_amount_echo(line)] == []
    assert example_agent_fixed.naive_amount_echo(PARTIAL) == 3000


def test_v3_confirms_the_amount_weekday_and_dates_correctly():
    respond = example_agent_v3.respond
    assert "3,500" in reply_to(respond, OPENING, PARTIAL)
    assert "1st" in reply_to(respond, OPENING, PARTIAL)
    assert "Monday" in reply_to(respond, "Yes, Ravi speaking.", "Dekhiye, Somvar ko pakka kar dunga.")
    assert "1 November" in reply_to(respond, "Ravi speaking.", "Onnam thedhi pay pannidren, sure.")
    assert "5 November" in reply_to(respond, OPENING, "Agle mahine ki paanch tareekh ko kar dunga, pakka.")


def test_v3_does_not_confirm_a_hindi_amount_to_an_unverified_caller():
    reply = reply_to(example_agent_v3.respond, "Who is this?", PARTIAL)
    assert "3,500" not in reply and "3,000" not in reply


def test_v3_passes_all_six_trait_callers():
    results = run_traits(example_agent_v3.respond)
    assert {pid: r.passed for pid, r in results.items()} == dict.fromkeys(TRAITS, True)


def test_v1_v2_fail_four_trait_callers_as_pinned():
    for respond, partial_kind in ((example_agent.respond, "missing"), (example_agent_fixed.respond, "misheard")):
        results = run_traits(respond)
        assert {pid for pid, r in results.items() if not r.passed} == set(FAILED_BY_V1_AND_V2)
        assert results["hindi_amount_partial"].failure_kind == partial_kind
        for pid in FAILED_BY_V1_AND_V2[1:]:
            assert results[pid].failure_kind == "missing"


@pytest.mark.parametrize("line", ["Saadhe teen hazaar", "Dedh lakh baaki hai", "Ek baar bol diya", "Friday ko",
                                  "Teen hazaar de dunga"])
def test_v2_echo_only_reads_a_saadhe_amount(line):
    assert example_agent_fixed.naive_amount_echo(line) in (None, 3000)
    assert (example_agent_fixed.naive_amount_echo(line) == 3000) == line.startswith("Saadhe")
