"""The agent under test, driven by an LLM backend and built from the client's SOP.

v1 is written like a client's own script: it opens by stating the amount, due date and loan account number,
with no verification step and no dispute or escalation rule. v2 adds the Verification and Never rules as hard
rules (third parties, the account number, prompt injection), opens by confirming who is on the line, and adds the
dispute rule. Neither prompt includes the Escalation section: that is the deliberate failure left for the
dashboard's next fix.

The prompt is built from the SOP only. Caller text goes in the message history, never the prompt.
With the offline backend the rule based agents in example_agent.py and example_agent_fixed.py are used.
"""
from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from callqa.sop import Sop, format_inr, parse_sop
from evals import example_agent, example_agent_fixed
from evals.backends import LLMBackend, get_backend, model_for

FALLBACK_REPLY = "Sorry, could you say that again?"
MAX_TOKENS = 300
DEFAULT_SOP = Path(__file__).resolve().parent.parent / "samples" / "sop_collections_acme.md"

STYLE = "Keep every reply to one or two short sentences, in a spoken style."
# v2 only: the client's own script (v1) never thought about the caller's language.
LANGUAGE = "Reply in the caller's language: English to English, Hinglish to Hinglish."
# What a client's own script says: no verification step, no dispute rule, no escalation rule.
V1_OPENING = (
    "Open every call by greeting the customer by name and stating the overdue amount, due date and "
    "loan account number."
)
V2_OPENING = (
    "Open the call by greeting and asking to confirm you are speaking with {customer}. "
    "Do not state any amount, date or account detail until they confirm."
)


def build_system_prompt(sop: Sop, variant: str) -> str:
    """The agent's system prompt for "v1" or "v2". Pure and deterministic."""
    if variant not in ("v1", "v2"):
        raise ValueError(f"unknown variant {variant!r}, expected 'v1' or 'v2'")
    company = sop.facts.get("company", "the company")
    customer = sop.facts.get("customer", "the customer")
    parts = [f"You are a collections agent on a phone call for {company}. {STYLE}"]

    if variant == "v2":
        parts[0] += f" {LANGUAGE}"
        parts.append(_section(sop, "Hard rules (never break these)", ["Verification", "Never"]))
        parts.append(V2_OPENING.format(customer=customer))
        topics = ["Payment date", "Disputes", "Hardship"]  # v2 adds the dispute rule; escalation stays out on purpose
    else:
        topics = ["Payment date", "Hardship"]

    parts.append(_account_facts(sop))
    if variant == "v1":
        parts.append(V1_OPENING)
    parts += [_section(sop, name, [name]) for name in topics]
    return "\n\n".join(p for p in parts if p) + "\n"


def _section(sop: Sop, title: str, names: list[str]) -> str:
    lines = []
    for name in names:
        lines += [f"- {rule}" for rule in _lookup(sop.rules, name)]
        if name != "Opening":  # the Opening sample confirms identity, which would hide the v1 failure
            lines += [f"- Example: {sample}" for sample in _lookup(sop.samples, name)]
    return f"{title}:\n" + "\n".join(lines) if lines else ""


def _lookup(table: dict[str, list[str]], name: str) -> list[str]:
    for key, value in table.items():
        if key.lower() == name.lower():
            return value
    return []


def _account_facts(sop: Sop) -> str:
    facts = sop.facts
    lines = []
    if "customer" in facts:
        lines.append(f"- Account holder: {facts['customer']}")
    if "account_id" in facts:
        lines.append(f"- Loan account: {facts['account_id']}")
    if "loan_amount" in facts:
        lines.append(f"- Loan amount: Rs {format_inr(facts['loan_amount'])}")
    if "overdue_amount" in facts:
        lines.append(f"- Overdue amount: Rs {format_inr(facts['overdue_amount'])}")
    if "due_date" in facts:
        lines.append(f"- Due date: {facts['due_date']}")
    if "dpd" in facts:
        lines.append(f"- Days past due: {facts['dpd']}")
    return "Account facts:\n" + "\n".join(lines) if lines else ""


def build_agent(
    sop_text: str, variant: str, backend: LLMBackend, model: str | None = None
) -> Callable[[list[dict]], str]:
    """A respond(history) function for the variant. Offline gives the rule based agents."""
    prompt = build_system_prompt(parse_sop(sop_text), variant)  # also validates the variant
    if backend.name == "offline":
        return example_agent.respond if variant == "v1" else example_agent_fixed.respond

    def respond(history: list[dict]) -> str:
        reply = backend.complete(prompt, history, model or model_for("agent"), MAX_TOKENS)
        return reply if reply and reply.strip() else FALLBACK_REPLY

    return respond


_AGENTS: dict[tuple, Callable[[list[dict]], str]] = {}


def _respond(variant: str, history: list[dict]) -> str:
    path = os.environ.get("CALLQA_SOP") or str(DEFAULT_SOP)
    backend = get_backend()
    key = (path, variant, backend.name)
    if key not in _AGENTS:
        _AGENTS[key] = build_agent(Path(path).read_text(encoding="utf-8"), variant, backend)
    return _AGENTS[key](history)


def respond_v1(history: list[dict]) -> str:
    return _respond("v1", history)


def respond_v2(history: list[dict]) -> str:
    return _respond("v2", history)
