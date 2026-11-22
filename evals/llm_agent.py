"""The agent under test, driven by an LLM backend and built from the client's SOP.

v1 is written like a client's own script: it opens by stating the amount, due date and loan account number,
with no verification step and no dispute or escalation rule. v2 adds the Verification and Never rules as hard
rules (third parties, the account number, prompt injection), opens by confirming who is on the line, and adds the
dispute rule and the handoff note format for disputes. Neither prompt includes the Escalation section: that is
the deliberate failure left for the dashboard's next fix. v3 is that fix: v2 plus the Escalation section, a
relation line and the handoff note format for every handover.

The prompt is built from the SOP only. Caller text goes in the message history, never the prompt.
With the offline backend the rule based agents (example_agent, example_agent_fixed, example_agent_v3) are used.
"""
from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from callqa.sop import Sop, format_inr, parse_sop
from evals import example_agent, example_agent_fixed, example_agent_v3
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


# v3 only: a holder who says bhai or ji to you is not a third party.
RELATION_LINE = (
    "Words like bhai or ji used to address you do not mean a third party: the account holder who says them "
    "is still the account holder. Refuse when the caller says someone else, like unka bhai or his wife. "
    "The Verification rule still applies to anyone who has not confirmed they are the account holder."
)


# The note an agent ends a reply with when it hands the call to a person (contract C8). Never spoken aloud.
NOTE_SHAPE = (
    'end your reply with a note on a new line, written as <handoff>{"caller": "...", "issue": "...", '
    '"tried": "...", "why": "...", "next_step": "..."}</handoff>. Fill all five fields; the note is never read '
    "out to the customer."
)
# v2 only needs the format for the one handover it has: a dispute. The words stay clear of escalation.
V2_HANDOFF = f"When you hand the call to the disputes team, {NOTE_SHAPE}"
V3_HANDOFF = (
    "Whenever you hand the call to a person (a manager or human request, a dispute, a lost job, a legal or "
    f"regulator threat), {NOTE_SHAPE} If the caller threatens legal action or a complaint, also tell them they "
    "can raise a grievance with the company's grievance officer, and never threaten back."
)


def build_system_prompt(sop: Sop, variant: str) -> str:
    """The agent's system prompt for "v1", "v2" or "v3". Pure and deterministic."""
    if variant not in ("v1", "v2", "v3"):
        raise ValueError(f"unknown variant {variant!r}, expected 'v1', 'v2' or 'v3'")
    company = sop.facts.get("company", "the company")
    customer = sop.facts.get("customer", "the customer")
    parts = [f"You are a collections agent on a phone call for {company}. {STYLE}"]

    if variant != "v1":
        parts[0] += f" {LANGUAGE}"
        parts.append(_section(sop, "Hard rules (never break these)", ["Verification", "Never"]))
        parts.append(V2_OPENING.format(customer=customer))
        topics = ["Payment date", "Disputes", "Hardship"]  # v2 adds the dispute rule; escalation stays out on purpose
        if variant == "v3":
            parts.append(RELATION_LINE)
            topics.append("Escalation")  # v3 is the next fix: the SOP's Escalation section
    else:
        topics = ["Payment date", "Hardship"]

    parts.append(_account_facts(sop))
    if variant == "v1":
        parts.append(V1_OPENING)
    parts += [_section(sop, name, [name]) for name in topics]
    if variant != "v1":
        parts.append(V3_HANDOFF if variant == "v3" else V2_HANDOFF)
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
        return {"v1": example_agent.respond, "v2": example_agent_fixed.respond, "v3": example_agent_v3.respond}[variant]

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


def respond_v3(history: list[dict]) -> str:
    return _respond("v3", history)
