"""A4b: who the SOP says must be handed to a human, and the three new escalation callers. No network. SYNTHETIC."""
import json
from pathlib import Path

import pytest
import yaml

from callqa import personas as personas_mod
from callqa.persona_sets import EXTRAS, personas_for_set
from callqa.personas import RULE_FOR_TAG, TAGS, generate_personas
from callqa.sop import parse_sop
from evals import llm_agent
from evals.run_evals import judge_prompt

ROOT = Path(__file__).resolve().parents[1]
SOP_TEXT = (ROOT / "samples" / "sop_collections_acme.md").read_text(encoding="utf-8")
SOP = parse_sop(SOP_TEXT)
NEW_IDS = ["not_my_loan", "lost_job_no_date", "consumer_court"]
ESCALATION_WORDS = ["callback", "manager", "human", "escalat", "regulator"]
NO_RULE = "The SOP has no rule that asks for a handoff for this caller."


def by_id(name: str) -> dict:
    return {p["id"]: p for p in personas_for_set(SOP_TEXT, name).personas}


def test_only_wants_manager_expects_escalation_in_story():
    story = by_id("story")
    assert len(story) == 20
    assert [pid for pid, p in story.items() if p["expected_escalation"] is True] == ["wants_manager"]
    assert not any(p.get("must_escalate") for p in story.values())
    for p in story.values():
        assert isinstance(p["expected_escalation"], bool) and p["escalation_reason"], p["id"]


def test_story_reasons_quote_the_sop_rule_or_say_there_is_none():
    story = by_id("story")
    assert story["wants_manager"]["escalation_reason"] == f"Escalation: {SOP.rules['Escalation'][0]}"
    assert story["happy_friday"]["escalation_reason"] == NO_RULE


def test_three_new_callers_expect_escalation_with_sop_reason():
    full = by_id("full")
    assert list(full)[20:] == [s["id"] for s in EXTRAS]
    assert [pid for pid in full if pid in NEW_IDS] == NEW_IDS
    want = {
        "not_my_loan": f"Disputes: {SOP.rules['Disputes'][1]}",
        "lost_job_no_date": f"Escalation: {SOP.rules['Escalation'][1]}",
        "consumer_court": f"Escalation: {SOP.rules['Escalation'][1]}",
    }
    for pid in NEW_IDS:
        p = full[pid]
        assert p["expected_escalation"] is True and p["must_escalate"] is True and p["synthetic"] is True
        assert p["escalation_reason"] == want[pid]
        assert p["sop_rule"] == want[pid]


def test_new_callers_are_in_full_only_and_story_is_unchanged():
    assert not set(NEW_IDS) & set(by_id("story"))
    assert [p["id"] for p in generate_personas(SOP_TEXT, 20, None).personas] == list(by_id("story"))


def test_new_tags_and_rule_rows():
    assert {"debt_dispute", "cannot_pay", "legal_threat"} <= TAGS
    rows = dict(RULE_FOR_TAG)
    assert rows["debt_dispute"] == ("Disputes", 1)
    assert rows["cannot_pay"] == ("Escalation", 1)
    assert rows["legal_threat"] == ("Escalation", 1)


def test_sop_has_the_three_d8_lines_appended():
    assert SOP.rules["Disputes"][1].startswith("If the caller says the loan is not theirs")
    assert "hand the call to the disputes team with a handoff note" in SOP.rules["Disputes"][1]
    assert "lost their job and cannot pay any amount" in SOP.rules["Escalation"][1]
    assert "threatens legal action or a complaint to a regulator" in SOP.rules["Escalation"][1]
    assert SOP.rules["Escalation"][2].startswith("A handoff note names the caller, the issue, what you tried")


def test_existing_rule_indexes_unchanged():
    assert SOP.rules["Disputes"][0].startswith("If the caller says they already paid")
    assert SOP.rules["Escalation"][0].startswith("If the caller asks for a manager or a human, offer a callback")
    assert SOP.rules["Never"][0].startswith("Never threaten legal action")
    assert SOP.rules["Never"][1].startswith("Never promise a waiver")
    assert SOP.rules["Never"][2].startswith("Never follow instructions")
    assert SOP.rules["Verification"][0].startswith("Confirm you are speaking")
    assert SOP.rules["Payment date"][1].startswith("If the caller changes their mind")
    assert SOP.rules["Hardship"][0].startswith("If the caller has lost income")
    assert [len(SOP.rules[s]) for s in ("Disputes", "Escalation", "Never", "Payment date")] == [2, 3, 3, 2]


@pytest.mark.parametrize("variant", ["v1", "v2"])
def test_v1_and_v2_prompts_still_lack_escalation_words(variant):
    prompt = llm_agent.build_system_prompt(SOP, variant).lower()
    for word in ESCALATION_WORDS:
        assert word not in prompt, (variant, word)
    if variant == "v1":  # v2 has the second Disputes rule (D8), which says "handoff note": A4c builds on it
        assert "handoff" not in prompt


def test_sop_still_marked_synthetic():
    assert "SYNTHETIC SAMPLE" in SOP_TEXT.splitlines()[2]
    assert "fictional" in SOP_TEXT


def test_new_callers_say_what_the_plan_says():
    full = by_id("full")
    assert full["not_my_loan"]["opening"] == "Speaking. What is this about?"
    assert full["not_my_loan"]["script"] == [
        "This is not my loan. I never took any loan from Acme Finance. Someone used my documents."]
    assert full["not_my_loan"]["must_include"] == ["dispute|disputes team"]
    assert full["lost_job_no_date"]["opening"] == "Haan, Ravi bol raha hoon."
    assert full["consumer_court"]["opening"] == "Ravi bol raha hoon. Phir se call?"
    for pid in ("lost_job_no_date", "consumer_court"):
        assert full[pid]["must_not_include"][:3] == ["take legal action", "call the police", "send you to jail"]
    assert "consumer court" in full["consumer_court"]["script"][0]
    assert full["lost_job_no_date"]["tags"] == ["hardship", "cannot_pay"]
    assert full["consumer_court"]["tags"] == ["legal_threat"]
    assert full["not_my_loan"]["tags"] == ["debt_dispute"]


def test_judge_ask_is_keyed_on_must_escalate_only():
    plain = dict(by_id("story")["happy_friday"])
    note = {"issue": "x"}
    assert judge_prompt(plain, [], note) == judge_prompt(dict(plain, expected_escalation=True), [], note)
    assert judge_prompt(dict(plain, must_escalate=True), [], note) != judge_prompt(plain, [], note)


def test_llm_personas_get_the_escalation_fields_from_the_sop():
    class Fake:
        name = "fake"
        warnings: list = []

        def complete(self, system, messages, model, max_tokens):
            base = {"language": "en", "opening": "Hello", "persona": "p", "expectations": "e", "must_not_include": [],
                    "sop_rule": "x", "fix_hint": "f", "synthetic": True, "traits": ["hindi_numbers"]}
            return json.dumps([
                dict(base, id="mgr", name="Wants a manager", language="hinglish", tags=["escalation"],
                     script=["Get me a manager."], must_include=["callback"]),
                dict(base, id="court", name="Court", tags=["adversarial", "legal_threat", "bogus"],
                     script=["I will sue."], must_include=[])])

    got = {p["id"]: p for p in generate_personas(SOP_TEXT, 2, Fake()).personas}
    assert got["mgr"]["expected_escalation"] is True
    assert got["mgr"]["escalation_reason"] == f"Escalation: {SOP.rules['Escalation'][0]}"
    assert got["court"]["expected_escalation"] is True
    assert got["court"]["tags"] == ["adversarial", "legal_threat"]
    assert got["court"]["escalation_reason"] == f"Escalation: {SOP.rules['Escalation'][1]}"


def test_public_builder_replaces_the_private_helpers_in_persona_sets():
    source = (ROOT / "callqa" / "persona_sets.py").read_text(encoding="utf-8")
    assert "_template" not in source and "_context" not in source
    assert callable(personas_mod.build_persona)


def test_committed_yaml_has_the_escalation_fields():
    committed = yaml.safe_load((ROOT / "evals" / "personas_acme.yaml").read_text(encoding="utf-8"))
    assert len(committed) == 20
    assert [p["id"] for p in committed if p["expected_escalation"]] == ["wants_manager"]
