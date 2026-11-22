"""Offline stand in for the fixed (v2) agent: v1 plus the Verification rule.

The one fix: never state the amount until the caller has identified themselves as the account
holder. Everything else is v1, plus one small branch: a caller who says the loan is not theirs gets the
disputes team and a handoff note scoring 4 of 5 (it never says what it tried). It still has no Escalation rule, so a
caller who asks for a manager gets no callback offer: that is the deliberate remaining failure.

The other small branch is a naive Hindi amount echo (a real speech to text slip): "saadhe teen hazaar" is 3,500, but
v2 reads the digits it recognises (teen hazaar is 3,000) and drops the "saadhe" fraction word. It fires only on a
"saadhe N hazaar" phrase, which no story caller says, so the story numbers do not move.
"""

import json
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


NOT_MY_LOAN = re.compile(
    r"not\s+my\s+loan|never\s+took\s+(?:any|a|this)\s+loan|used\s+my\s+documents|mera\s+loan\s+nahi|loan\s+maine\s+nahi",
    re.IGNORECASE)
DISPUTE_REPLY = (
    "I'm sorry about this. I'll stop collection on this call and pass it to the disputes team, "
    "who will contact you within 2 working days."
)


# v2's naive reading: the number word it knows, then hazaar. "Saadhe" (plus a half) is the word it drops.
SAADHE_HAZAAR = re.compile(r"\bsaadhe\s+(ek|do|teen|chaar|paanch|chhe|saat|aath|nau|das)\s+hazaar\b", re.IGNORECASE)
DIGITS = {"ek": 1, "do": 2, "teen": 3, "chaar": 4, "paanch": 5, "chhe": 6, "saat": 7, "aath": 8, "nau": 9, "das": 10}


def naive_amount_echo(text: str) -> int | None:
    """The amount v2 hears in "saadhe teen hazaar": 3000, not 3500. None for any other line."""
    match = SAADHE_HAZAAR.search(text)
    return DIGITS[match.group(1).lower()] * 1000 if match else None


def handoff_note(**fields: str) -> str:
    """A reply suffix, `<handoff>{...}</handoff>` (contract C8). The harness strips it from the spoken text."""
    return "<handoff>" + json.dumps(fields, ensure_ascii=False) + "</handoff>"


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
    if NOT_MY_LOAN.search(last):  # no `tried` field: v2 forgets to say what it did first (4 of 5)
        return DISPUTE_REPLY + " " + handoff_note(
            caller="Caller on the Ravi Kumar account",
            issue="Says the loan is not theirs and that someone used their documents",
            why="Disputed ownership, so collection must stop",
            next_step="Disputes team to contact the caller within 2 working days")

    heard = naive_amount_echo(last) if holder else None
    if heard:
        later = " And I've noted the 1st for the rest." if "1st" in last.lower() else ""
        return f"Okay, Rs {heard:,} today, I've noted that.{later}"

    reply = respond_v1(history)
    words = set(re.findall(r"[a-z]+", last.lower()))
    if reply == GENERIC and not holder and words & CONFIRM_WORDS and not words & NEGATIONS:
        return OPENING
    return reply
