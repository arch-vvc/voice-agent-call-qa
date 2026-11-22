"""Trait callers (C5): one offline template per trait, apply_trait, the LLM generator mix, the traits block and line.
SYNTHETIC callers only, no network."""
import json
from pathlib import Path

import pytest
import yaml

from callqa import evaluate
from callqa.persona_sets import EXTRAS, TRAITS, apply_trait, personas_for_set
from callqa.personas import STRATIFIED_5, TRAIT_IDS, generate_personas
from callqa.report import build_report, traits_block
from callqa.server import DEFAULT_AGENTS
from evals import example_agent, example_agent_fixed, example_agent_v3
from evals.backends import OfflineBackend
from evals.run_evals import run_suite

HERE = Path(__file__).resolve().parent
SOP = (HERE.parent / "samples" / "sop_collections_acme.md").read_text(encoding="utf-8")
ROWS = yaml.safe_load((HERE / "fixtures" / "full_set_expected.yaml").read_text(encoding="utf-8"))["rows"]
AGENTS = {"v1": example_agent.respond, "v2": example_agent_fixed.respond, "v3": example_agent_v3.respond}
ORDER = ["hindi_numbers", "en_to_hi_switch", "tamil_english", "kannada_english", "clipped_regional_english",
         "indian_dates"]
NEW_IDS = ["hindi_amount_partial", "switch_to_hindi", "tamil_first", "kannada_friday", "clipped_paid",
           "agle_mahine_date"]


class FakeBackend:
    name = "fake"

    def __init__(self, reply):
        self.reply, self.warnings, self.calls = reply, [], []

    def complete(self, system, messages, model, max_tokens):
        self.calls.append({"system": system, "messages": messages})
        return self.reply


def llm_persona(pid, language="en", tags=("happy",), **extra):
    item = {
        "id": pid, "name": pid.replace("_", " ").title(), "language": language, "tags": list(tags),
        "opening": "Hello.", "persona": "A customer.", "script": ["Okay, Friday."],
        "expectations": "PASS if the agent confirms Friday.", "must_include": ["friday"], "must_not_include": [],
        "sop_rule": "Ask for a specific payment date and repeat the final date back to the caller.",
        "fix_hint": "Repeat the date back.", "synthetic": True,
    }
    item.update(extra)
    return item


def full() -> dict:
    return {p["id"]: p for p in personas_for_set(SOP, "full").personas}


def test_trait_ids_are_the_fixed_c5_order():
    assert list(TRAIT_IDS) == ORDER and list(TRAITS) == ORDER
    for trait in TRAITS.values():
        assert trait["label"] and trait["instruction"]


def test_every_trait_has_an_offline_template():
    personas = personas_for_set(SOP, "full").personas
    for trait in ORDER:
        assert any(trait in p["traits"] for p in personas), trait
    assert [spec["id"] for spec in EXTRAS][-6:] == NEW_IDS
    assert full()["lakh_question"]["traits"] == ["hindi_numbers"]


def test_story_set_has_a_trait_and_stratified_5_is_exempt():
    story = personas_for_set(SOP, "story")
    assert [p["id"] for p in story.personas if p["traits"]] == ["lakh_question"]
    five = personas_for_set(SOP, "story", 5)
    assert [p["id"] for p in five.personas] == STRATIFIED_5
    assert not any(p["traits"] for p in five.personas) and five.warnings == []


def test_llm_set_without_a_trait_is_repaired_with_warning():
    reply = json.dumps([llm_persona(f"mix_{i}", "hinglish", ["adversarial"]) for i in range(6)])
    result = generate_personas(SOP, 6, FakeBackend(reply))
    assert [p["id"] for p in result.personas][-1] == "hindi_amount_partial"
    assert result.personas[-1]["traits"] == ["hindi_numbers"]
    assert [p["id"] for p in result.personas][:5] == [f"mix_{i}" for i in range(5)]
    assert any("trait" in w and "hindi_amount_partial" in w for w in result.warnings)


def test_llm_set_with_a_trait_is_not_repaired_and_gets_the_instruction():
    items = [llm_persona(f"mix_{i}", "hinglish", ["adversarial"]) for i in range(3)]
    items[1]["traits"] = ["tamil_english", "not_a_trait"]
    result = generate_personas(SOP, 3, FakeBackend(json.dumps(items)))
    assert [p["id"] for p in result.personas] == ["mix_0", "mix_1", "mix_2"]
    assert not any("trait" in w for w in result.warnings)
    assert result.personas[1]["traits"] == ["tamil_english"]
    assert TRAITS["tamil_english"]["instruction"] in result.personas[1]["persona"]
    assert result.personas[0]["traits"] == [] and result.personas[0]["persona"] == "A customer."


def test_generator_prompt_asks_for_traits():
    backend = FakeBackend(json.dumps([llm_persona("a", "hinglish", ["adversarial"], traits=["indian_dates"])]))
    generate_personas(SOP, 1, backend)
    system = backend.calls[0]["system"]
    assert f"traits (a subset of: {', '.join(ORDER)}" in system and "at least one" in system


def test_apply_trait_appends_instruction():
    base = full()["happy_friday"]
    out = apply_trait(base, "hindi_numbers")
    assert out["persona"] == f"{base['persona']} {TRAITS['hindi_numbers']['instruction']}"
    assert out["traits"] == ["hindi_numbers"] and base["traits"] == []  # a copy, the input is untouched
    assert apply_trait(out, "hindi_numbers") == out  # idempotent
    both = apply_trait(out, "indian_dates")
    assert both["traits"] == ["hindi_numbers", "indian_dates"]
    with pytest.raises(ValueError, match="unknown trait"):
        apply_trait(base, "klingon")


def test_said_values_personas_carry_sop_values_for_the_misheard_check():
    personas = full()
    said = [p for p in personas.values() if p.get("said_values")]
    assert {p["id"] for p in said} == {"hindi_amount_partial", "agle_mahine_date"}
    assert personas["hindi_amount_partial"]["said_values"] == {"amount_inr": 3500}
    assert personas["agle_mahine_date"]["said_values"] == {"date": "2026-11-05"}
    for p in said:
        assert p["sop_values"] == {"amounts_inr": [12400, 100000], "dates": ["2026-10-15"],
                                   "call_date": "2026-10-23"}
    assert not any("said_values" in p for p in personas.values() if p["id"] not in {x["id"] for x in said})


def test_trait_personas_are_labelled_synthetic_and_name_their_story_base():
    personas = full()
    story = [p["id"] for p in personas_for_set(SOP, "story").personas]
    for pid in NEW_IDS:
        assert personas[pid]["synthetic"] is True and personas[pid]["base"] in story
        assert len(personas[pid]["traits"]) == 1


def run(agent_id: str):
    generation = personas_for_set(SOP, "full")
    results = run_suite(AGENTS[agent_id], generation.personas, 1, 8, 4, OfflineBackend(), 0)
    agent = next(a for a in DEFAULT_AGENTS if a["id"] == agent_id)
    return generation, build_report(run_id="t", sop_text=SOP, agent=agent, backend=OfflineBackend(), callers=26,
                                    max_turns=8, generation=generation, results=results, seconds=0.0,
                                    set_name="full")


def expected_traits(agent_id: str, generation) -> dict:
    """Per trait (passed, total) from the expected rows and the personas' own traits."""
    want: dict = {}
    for p in generation.personas:
        for trait in p["traits"]:
            got = want.setdefault(trait, [0, 0])
            got[0] += ROWS[p["id"]][agent_id] == "pass"
            got[1] += 1
    return {t: want[t] for t in ORDER if t in want}


def test_trait_rows_are_pinned_as_today():
    # A5c: v3 passes all six; v1 and v2 fail the same four (v2 mishears the Hindi amount instead of missing it).
    for agent in ("v1", "v2", "v3"):
        for pid in ("hindi_amount_partial", "switch_to_hindi", "tamil_first", "agle_mahine_date"):
            mishears = (agent, pid) == ("v2", "hindi_amount_partial")
            want = "pass" if agent == "v3" else "misheard" if mishears else "missing"
            assert ROWS[pid][agent] == want, (pid, agent)
        assert ROWS["kannada_friday"][agent] == ROWS["clipped_paid"][agent] == "pass"


@pytest.mark.parametrize("agent_id", AGENTS)
def test_traits_block_and_line_per_agent(agent_id, capsys):
    generation, record = run(agent_id)
    want = expected_traits(agent_id, generation)
    assert list(record["traits"]) == list(want) and len(want) == 6
    for trait, (passed, total) in want.items():
        assert record["traits"][trait] == {"passed": passed, "total": total,
                                           "pass_rate": round(passed / total, 4)}
        assert [c["persona_id"] for c in record["calls"] if trait in c["traits"]] == [
            p["id"] for p in generation.personas if trait in p["traits"]]
    line = "traits " + ", ".join(f"{t} {format(p / n, '.0%')} ({p}/{n})" for t, (p, n) in want.items())
    assert evaluate.traits_line(record) == line
    code = evaluate.main(["--agent", agent_id, "--set", "full", "--backend", "offline", "--no-save"])
    out = capsys.readouterr().out.splitlines()
    assert code == 0 and line in out and out.index(line) == len(out) - 1


def test_v1_full_set_traits_line_is_the_plans_pinned_text():
    _, record = run("v1")
    assert evaluate.traits_line(record) == (
        "traits hindi_numbers 50% (1/2), en_to_hi_switch 0% (0/1), tamil_english 0% (0/1), "
        "kannada_english 100% (1/1), clipped_regional_english 100% (1/1), indian_dates 0% (0/1)")


def test_story_traits_line_lists_only_traits_present_and_stratified_5_has_none(capsys):
    assert evaluate.main(["--agent", "v1", "--set", "story", "--backend", "offline", "--no-save"]) == 0
    assert capsys.readouterr().out.splitlines()[-1] == "traits hindi_numbers 100% (1/1)"
    assert evaluate.main(["--agent", "v1", "--set", "story", "--callers", "5", "--backend", "offline",
                          "--no-save"]) == 0
    assert not any(line.startswith("traits") for line in capsys.readouterr().out.splitlines())
    assert traits_block([]) == {}


def test_traits_line_comes_before_threshold_and_record(capsys):
    code = evaluate.main(["--agent", "v1", "--set", "story", "--backend", "offline", "--no-save",
                          "--threshold", "0.5"])
    out = capsys.readouterr().out.splitlines()
    assert code == 0 and out[-2] == "traits hindi_numbers 100% (1/1)" and out[-1].startswith("threshold")
