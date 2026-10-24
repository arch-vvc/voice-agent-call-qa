"""The "Fix:" text of a group comes from the SOP rule the group broke, never from a judge's free text."""
from callqa.report import fix_for_rule, group_next_fixes

RULE = "Never: Never read out the loan account number on the call."


def call(pid, rule, fix_hint, turn=1):
    return {"persona_id": pid, "name": pid, "passed": False, "sop_rule": rule, "fix_hint": fix_hint,
            "failing_turn": turn}


def test_group_fix_text_derives_from_the_broken_rule_not_the_judge_reason():
    judge_reason = "The agent offered to take payment on someone else's behalf."
    calls = [call("lost_job", RULE, judge_reason), call("brother", RULE, "Confirm you are speaking...")]
    fixes = group_next_fixes(calls)
    assert len(fixes) == 1
    assert "Never read out the loan account number on the call." in fixes[0]["fix_hint"]
    assert "behalf" not in fixes[0]["fix_hint"] and "speaking" not in fixes[0]["fix_hint"]


def test_fix_text_drops_the_section_prefix_and_names_the_rule():
    expected = "Add or enforce this rule in the prompt: Never read out the loan account number on the call."
    assert fix_for_rule(RULE, "x") == expected
    # only a leading section name is cut, not any text before a colon
    assert fix_for_rule("If the caller says: stop.", "x").endswith("If the caller says: stop.")


def test_without_a_rule_the_persona_hint_is_the_fallback():
    assert fix_for_rule("", "Follow the SOP.") == "Follow the SOP."
