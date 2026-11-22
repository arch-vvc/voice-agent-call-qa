"""T7: the LLM agent under test. Fake backends only: no network, no real `claude`."""
from pathlib import Path

import pytest

from callqa.sop import parse_sop
from evals import example_agent, example_agent_fixed, llm_agent
from evals.backends import BackendError, OfflineBackend, model_for

SOP_TEXT = Path(__file__).resolve().parent.parent.joinpath("samples", "sop_collections_acme.md").read_text()
SOP = parse_sop(SOP_TEXT)
HISTORY = [{"role": "user", "content": "Hello?"}]


class FakeBackend:
    name = "fake"
    warnings: list = []

    def __init__(self, reply="Namaste, main Asha bol rahi hoon."):
        self.reply = reply
        self.calls = []

    def complete(self, system, messages, model, max_tokens):
        self.calls.append((system, messages, model, max_tokens))
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def test_prompts_are_deterministic():
    for variant in ("v1", "v2"):
        again = llm_agent.build_system_prompt(parse_sop(SOP_TEXT), variant)
        assert llm_agent.build_system_prompt(SOP, variant) == again


def test_unknown_variant_raises():
    with pytest.raises(ValueError):
        llm_agent.build_system_prompt(SOP, "v4")


def test_v1_prompt_opens_with_amount_date_and_account_number_and_has_no_verification_rule():
    prompt = llm_agent.build_system_prompt(SOP, "v1")
    assert "Open every call by greeting the customer by name and stating the overdue amount, due date and " in prompt
    assert "loan account number" in prompt
    assert "good time" not in prompt
    assert "Confirm you are speaking with the account holder" not in prompt
    assert "Never threaten" not in prompt and "raise a dispute" not in prompt  # no dispute rule in the client script
    for kept in ("Ask for a specific payment date", "do not pressure them"):
        assert kept in prompt
    assert "Ravi Kumar" in prompt and "Acme Finance" in prompt


def test_v2_prompt_puts_verification_and_never_rules_before_facts():
    prompt = llm_agent.build_system_prompt(SOP, "v2")
    verification = prompt.index("Confirm you are speaking with the account holder")
    never = prompt.index("Never threaten")
    facts = prompt.index("12,400")
    assert verification < facts and never < facts
    assert "confirm you are speaking with Ravi Kumar" in prompt
    assert "state the overdue amount and due date" not in prompt


def test_neither_prompt_includes_escalation():
    for variant in ("v1", "v2"):
        prompt = llm_agent.build_system_prompt(SOP, variant)
        assert "manager" not in prompt.lower()
        assert "24 hours" not in prompt
        assert "callback" not in prompt.lower()
        assert "24 ghante" not in prompt
        assert "Escalation" not in prompt


def test_prompt_uses_indian_number_format():
    for variant in ("v1", "v2"):
        prompt = llm_agent.build_system_prompt(SOP, variant)
        assert "12,400" in prompt and "1,00,000" in prompt
        assert "12400" not in prompt and "100000" not in prompt


def test_offline_backend_returns_rule_based_agents():
    assert llm_agent.build_agent(SOP_TEXT, "v1", OfflineBackend()) is example_agent.respond
    assert llm_agent.build_agent(SOP_TEXT, "v2", OfflineBackend()) is example_agent_fixed.respond


def test_build_agent_sends_system_prompt_history_and_agent_model_to_backend(monkeypatch):
    monkeypatch.delenv("CALLQA_AGENT_MODEL", raising=False)
    backend = FakeBackend()
    reply = llm_agent.build_agent(SOP_TEXT, "v2", backend)(HISTORY)
    assert reply == "Namaste, main Asha bol rahi hoon."
    assert backend.calls == [(llm_agent.build_system_prompt(SOP, "v2"), HISTORY, model_for("agent"), 300)]


def test_agent_model_env_override(monkeypatch):
    monkeypatch.setenv("CALLQA_AGENT_MODEL", "my-agent-model")
    backend = FakeBackend()
    llm_agent.build_agent(SOP_TEXT, "v1", backend)(HISTORY)
    assert backend.calls[0][2] == "my-agent-model"
    llm_agent.build_agent(SOP_TEXT, "v1", backend, model="explicit")(HISTORY)
    assert backend.calls[1][2] == "explicit"


@pytest.mark.parametrize("empty", ["", "   \n"])
def test_empty_reply_returns_safe_line(empty):
    assert llm_agent.build_agent(SOP_TEXT, "v1", FakeBackend(empty))(HISTORY) == "Sorry, could you say that again?"


def test_backend_errors_propagate():
    error = BackendError("limit")
    with pytest.raises(BackendError) as caught:
        llm_agent.build_agent(SOP_TEXT, "v1", FakeBackend(error))(HISTORY)
    assert caught.value is error


def test_responders_use_callqa_backend_env(monkeypatch):
    monkeypatch.setenv("CALLQA_BACKEND", "offline")
    monkeypatch.delenv("CALLQA_SOP", raising=False)
    assert llm_agent.respond_v1(HISTORY) == example_agent.respond(HISTORY)
    assert llm_agent.respond_v2(HISTORY) == example_agent_fixed.respond(HISTORY)


def test_responders_cache_and_read_sop_env(monkeypatch, tmp_path):
    sop_file = tmp_path / "sop.md"
    sop_file.write_text(SOP_TEXT)
    fake = FakeBackend()
    monkeypatch.setenv("CALLQA_SOP", str(sop_file))
    monkeypatch.setattr(llm_agent, "get_backend", lambda: fake)
    monkeypatch.setattr(llm_agent, "_AGENTS", {})
    parses = []
    real_parse = llm_agent.parse_sop
    monkeypatch.setattr(llm_agent, "parse_sop", lambda text: parses.append(1) or real_parse(text))
    llm_agent.respond_v1(HISTORY)
    llm_agent.respond_v1(HISTORY)
    assert len(fake.calls) == 2 and len(parses) == 1


OTHER_SOP = """# Zeta Bank: card reminders

Company: Zeta Bank
Customer Name: Meena Iyer
Amount due: Rs. 5400/-
Due date: 3rd Nov 2026

## Opening
- Greet the customer.

## Payment date
- Ask for a date.
"""


def test_other_company_sop_without_escalation_builds_both_prompts():
    other = parse_sop(OTHER_SOP)
    for variant in ("v1", "v2"):
        prompt = llm_agent.build_system_prompt(other, variant)
        assert "Zeta Bank" in prompt and "5,400" in prompt
    assert llm_agent.build_system_prompt(parse_sop(""), "v2")
