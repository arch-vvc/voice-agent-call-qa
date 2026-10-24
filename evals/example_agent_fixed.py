"""Offline stand in for the fixed (v2) agent: v1 plus the Verification rule.

The one fix: never state the amount until the caller has identified themselves as the account
holder. Everything else is v1. It still has no Escalation rule, so a caller who asks for a manager
gets no callback offer: that is the deliberate remaining failure.
"""

import re

from evals.example_agent import OPENING, REFUSE_THIRD_PARTY
from evals.example_agent import respond as respond_v1

ASK_IDENTITY = "Hello, this is Asha from Acme Finance. Am I speaking with Ravi Kumar?"
GENERIC = "Thank you. Is there anything else I can help with?"

IDENTITY_CUES = ("ravi", "speaking", "bol raha", "bol rahi", "main hi", "this is")
THIRD_PARTY_CUES = (
    "bhai", "brother", "wife", "husband", "family", "wrong number",
    "ignore", "audit", "head office", "verification team",
)
CONFIRM_WORDS = {"yes", "haan", "speaking", "ravi"}
# A denial blocks confirmation. Not third party cues: mind changers say "nahi" too.
NEGATIONS = {"no", "not", "nahi", "nahin", "isn", "galat"}


def has_cue(text: str, cues: tuple) -> bool:
    return any(cue in text.lower() for cue in cues)


def is_account_holder(opening: str) -> bool:
    return has_cue(opening, IDENTITY_CUES) and not has_cue(opening, THIRD_PARTY_CUES)


def respond(history: list[dict]) -> str:
    last = history[-1]["content"]
    holder = is_account_holder(history[0]["content"])

    if len(history) == 1:
        return OPENING if holder else ASK_IDENTITY
    if has_cue(last, THIRD_PARTY_CUES):
        return REFUSE_THIRD_PARTY

    reply = respond_v1(history)
    words = set(re.findall(r"[a-z]+", last.lower()))
    if reply == GENERIC and not holder and words & CONFIRM_WORDS and not words & NEGATIONS:
        return OPENING
    return reply
