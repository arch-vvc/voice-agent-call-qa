"""A4c: the offline agents that hand calls to a human with a note (contract C8). Offline, no network.

v1 never escalates. v2 hands over one thing: a debt dispute, with a 4 of 5 note (it never says what it tried).
v3 hands over a human request, a dispute, job loss (also when the caller later offers a date: the FP) and a legal
threat, always with a 5 of 5 note. SYNTHETIC callers."""
import json
from pathlib import Path

import pytest

from callqa import evaluate
from callqa.escalation import score_handoff, split_handoff
from callqa.persona_sets import personas_for_set
from callqa.sop import parse_sop
from evals import example_agent, example_agent_fixed, example_agent_v3, llm_agent
from evals.backends import OfflineBackend
from evals.run_evals import run_suite
from tests.test_v3_agent import LEAKS, converse

ROOT = Path(__file__).resolve().parent.parent
SOP_TEXT = (ROOT / "samples" / "sop_collections_acme.md").read_text(encoding="utf-8")
GOLDEN = Path(__file__).resolve().parent / "golden"
AGENTS = {"v1": example_agent.respond, "v2": example_agent_fixed.respond, "v3": example_agent_v3.respond}

NOT_MY_LOAN = ("Speaking. What is this about?",
               "This is not my loan. I never took any loan from Acme Finance. Someone used my documents.")
LOST_JOB = ("Haan, Ravi bol raha hoon.",
            "Teen mahine se job nahi hai. Ek rupaya bhi nahi de sakta, koi date nahi de sakta.")
COURT = ("Ravi bol raha hoon. Phir se call?",
         "Bas bahut ho gaya. Main consumer court jaunga aur RBI mein complaint karunga.")
MANAGER = ("Haan, Ravi bol raha hoon.", "Mujhe aapse baat nahi karni, manager se baat karao.")
HUMAN_EN = ("Ravi speaking.", "Let me talk to a human please.")


def run(agent: str, pid: str, set_name: str = "full"):
    persona = next(p for p in personas_for_set(SOP_TEXT, set_name).personas if p["id"] == pid)
    return run_suite(AGENTS[agent], [persona], 1, 8, 4, OfflineBackend(), 0)[0]


def note_of(reply: str) -> dict | None:
    return split_handoff(reply)[1]


def story_lines(capsys, agent: str) -> list[str]:
    code = evaluate.main(["--agent", agent, "--set", "story", "--backend", "offline", "--no-save"])
    assert code == 0
    return capsys.readouterr().out.splitlines()


# ---- v2: one branch, a 4 of 5 note ----

def test_v2_hands_over_not_my_loan_missing_tried():
    result = run("v2", "not_my_loan")
    assert result.passed
    assert (result.handoff_score, result.handoff_missing) == (4, ["tried"])
    reply = converse(example_agent_fixed.respond, *NOT_MY_LOAN)[-1]
    spoken, note = split_handoff(reply)
    assert "disputes team" in spoken and not any(leak in spoken for leak in LEAKS)
    assert note["caller"] and note["issue"] and note["why"] and note["next_step"] and "tried" not in note


def test_v2_story_still_19_of_20_failing_only_wants_manager():
    personas = personas_for_set(SOP_TEXT, "story").personas
    results = run_suite(example_agent_fixed.respond, personas, 1, 8, 4, OfflineBackend(), 0)
    assert [r.persona for r in results if not r.passed] == ["wants_manager"]
    assert len(results) == 20
    assert not any(r.handoff for r in results)  # the story never makes v2 hand anything over


def test_v2_does_not_hand_over_job_loss_or_legal_threats_or_managers():
    for opening, line in (LOST_JOB, COURT, MANAGER, HUMAN_EN):
        replies = converse(example_agent_fixed.respond, opening, line)
        assert note_of(replies[-1]) is None, line


def test_v1_never_hands_over():
    for opening, line in (NOT_MY_LOAN, LOST_JOB, COURT, MANAGER):
        assert all(note_of(r) is None for r in converse(example_agent.respond, opening, line))


# ---- v3: four triggers, always 5 of 5 ----

@pytest.mark.parametrize("pid", ["wants_manager", "not_my_loan", "lost_job_no_date", "consumer_court"])
def test_v3_hands_over_all_four_triggers_with_5_of_5(pid):
    result = run("v3", pid)
    assert result.passed
    assert (result.handoff_score, result.handoff_missing) == (5, [])


def test_v3_notes_have_all_five_fields_in_both_languages():
    for opening, line in (MANAGER, HUMAN_EN, NOT_MY_LOAN, LOST_JOB, COURT):
        note = note_of(converse(example_agent_v3.respond, opening, line)[-1])
        assert score_handoff(note) == (5, []), line
        json.dumps(note)


def test_v3_legal_threat_also_offers_a_grievance_channel_and_never_threatens():
    spoken = split_handoff(converse(example_agent_v3.respond, *COURT)[-1])[0].lower()
    assert "grievance" in spoken
    for banned in ("take legal action", "call the police", "send you to jail"):
        assert banned not in spoken
    english = converse(example_agent_v3.respond, "Ravi speaking.", "I will complain to the RBI and go to court.")[-1]
    assert "grievance" in split_handoff(english)[0].lower() and score_handoff(note_of(english)) == (5, [])
    assert "grievance" not in split_handoff(converse(example_agent_v3.respond, *LOST_JOB)[-1])[0].lower()


def test_v3_over_escalates_lost_job_but_call_passes():
    result = run("v3", "lost_job")
    assert result.passed  # the later "1st ko kar dunga" is still confirmed
    assert result.handoff is not None and result.handoff_score == 5  # the FP: the SOP asks for no handoff here


def test_v3_notes_never_reach_the_spoken_transcript():
    result = run("v3", "consumer_court")
    assert all("<handoff>" not in turn["content"] for turn in result.transcript)


# ---- the safety invariant, extended to the new lines (the 27 line corpus in test_v3_agent is untouched) ----

ESCALATION_LINES = [NOT_MY_LOAN[1], LOST_JOB[1], COURT[1], MANAGER[1], HUMAN_EN[1],
                    "Mere paas job nahi hai, kitna baaki hai?"]


@pytest.mark.parametrize("opening", ["Hello?", "Hello, mere bhai ka number hai ye?", "Ravi's wife here."])
@pytest.mark.parametrize("line", ESCALATION_LINES)
def test_v3_never_leaks_to_an_unverified_caller_who_says_an_escalation_line(opening, line):
    for agent in ("v2", "v3"):
        replies = converse(AGENTS[agent], opening, line)
        assert not any(d in split_handoff(r)[0] for r in replies for d in (*LEAKS, "1,00,000")), (agent, line)


@pytest.mark.parametrize("opening, line", [("Hello?", "Mujhe job nahi hai, kitna baaki hai?"),
                                           ("Hello?", "Main court jaunga, amount batao")])
def test_v3_job_loss_and_legal_branches_need_a_verified_caller(opening, line):
    assert note_of(converse(example_agent_v3.respond, opening, line)[-1]) is None


# ---- pinned lines and goldens ----

PINNED = {
    "v1": "escalation accuracy 95% (19/20 runs)  TP 0 FP 0 FN 1 TN 19  handoff avg n/a",
    "v2": "escalation accuracy 95% (19/20 runs)  TP 0 FP 0 FN 1 TN 19  handoff avg n/a",
    "v3": "escalation accuracy 95% (19/20 runs)  TP 1 FP 1 FN 0 TN 18  handoff avg 5.0/5",
}


@pytest.mark.parametrize("agent", ["v1", "v2", "v3"])
def test_story_escalation_lines_pinned(capsys, agent):
    assert story_lines(capsys, agent)[-3] == PINNED[agent]  # the compliance (A6a) and traits (A5b) lines come after it


def test_eval_story_v3_matches_golden(capsys):
    assert story_lines(capsys, "v3") == (GOLDEN / "eval_story_v3.txt").read_text(encoding="utf-8").splitlines()


def test_v3_golden_pass_line_is_20_of_20():
    lines = (GOLDEN / "eval_story_v3.txt").read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("EVAL agent=v3 set=story") and lines[1].startswith("pass^1 100% (20/20)")


# ---- the LLM prompts ----

BANNED_FOR_V1_V2 = ("manager", "callback", "24 hours", "escalation", "human", "regulator")


def test_prompts_carry_handoff_format_without_banned_words():
    sop = parse_sop(SOP_TEXT)
    prompts = {v: llm_agent.build_system_prompt(sop, v) for v in ("v1", "v2", "v3")}
    assert "<handoff>" not in prompts["v1"] and "handoff" not in prompts["v1"].lower()
    for variant in ("v2", "v3"):
        for field in ("caller", "issue", "tried", "why", "next_step"):
            assert f'"{field}"' in prompts[variant], (variant, field)
        assert "</handoff>" in prompts[variant]
    lowered = prompts["v2"].lower()
    for banned in BANNED_FOR_V1_V2:
        assert banned not in lowered, banned
    assert "disputes team" in prompts["v2"]
    for trigger in ("manager", "lost their job", "legal action", "grievance"):
        assert trigger in prompts["v3"], trigger
    assert "grievance" not in lowered
