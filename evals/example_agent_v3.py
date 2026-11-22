"""Offline stand in for the v3 agent: v2 plus the next fixes. Everything not listed here is v2.

Branches, in order:
1. A third party named on the opening stays flagged for the whole call (never confirmed, never given the amount).
2. A third party named mid call is refused. Third party means a relation word after a possessive ("unka bhai",
   "Ravi's father", "ki behen", "bada bhai"), before "hoon / here / speaking / bol raha" ("wife bol rahi hoon"), or
   after "the" or "my" ("my brother"). v2's own cues (wife, brother, family, wrong number, audit ...) still fire.
3. Only when the opening is the account holder's: a vocative "bhai" ("Arre bhai, ...", "Bhai sun") is dropped before
   v2 looks at the line, because v2 refused the real holder as a brother. "bhai ka loan", "mere bhai" and "Bhai hoon"
   are relations and stay. A stranger's vocative never verifies him, so his lines go to v2 untouched.
Design rule: for an unverified caller v3 is never less safe than v2.
4. A verified caller who asks to speak to a manager or human gets the SOP's callback offer within 24 hours.
5. Handoff notes (contract C8), always all five fields, for a verified caller: a manager or human request, a debt
   dispute ("not my loan"), job loss, and a legal or regulator threat (which also gets a grievance channel).
   Job loss is deliberately over eager: it also fires for a caller who goes on to offer a date (`lost_job`), which
   is the FP the dashboard shows. The date is still confirmed on the next turn, so that call passes.
6. Indian amounts, weekdays and dates (callqa/indic.py), for a verified caller: "saadhe teen hazaar" is Rs 3,500,
   "Somvar" is Monday, "onnam thedhi" is 1 November, "agle mahine ki paanch tareekh" is 5 November. All are read back.
"""

import calendar
import re
from datetime import date

from callqa.indic import find_dates, parse_hindi_amount, weekday_from_word
from callqa.sop import format_inr
from evals.example_agent import REFUSE_THIRD_PARTY
from evals.example_agent_fixed import (
    ASK_IDENTITY,
    NOT_MY_LOAN,
    THIRD_PARTY_CUES,
    handoff_note,
    has_cue,
    is_account_holder,
)
from evals.example_agent_fixed import respond as respond_v2

RELATION = (r"bhai|brother|wife|husband|mummy|maa|papa|beta|behen|sister|mother|father|biwi|patni|saali|mama|chacha"
            r"|uncle|aunty|dost|friend|colleague|coworker|neighbour")
POSSESSIVE = r"(?:unka|uska|inka|unke|uske|inke|unki|uski|inki|ka|ki|ke|his|her|their|\w+'s)"
ADJECTIVE = r"(?:(?:bada|bade|badi|chhota|chhote|chhoti|elder|younger)\s+)?"
SELF = r"(?:hoon|hu|here|speaking|bol\s+rah[ai])"
THIRD_PARTY = re.compile(
    rf"\b{POSSESSIVE}\s+{ADJECTIVE}(?:{RELATION})\b"  # "unka bhai", "Ravi's father", "ki bada behen"
    rf"|\b(?:{RELATION})\s+{SELF}\b"  # "wife bol rahi hoon Ravi ki"
    rf"|\b(?:the|my)\s+(?:{RELATION})\b"  # "I am the wife", "my brother Ravi"
    rf"|\b(?:from|at)\s+\w+'s\s+office\b",  # "calling from Ravi's office"
    re.IGNORECASE,
)
# "bhai" alone is v2's cue that wrongly fires on the filler; every other v2 cue stays unconditional.
KEPT_V2_CUES = tuple(cue for cue in THIRD_PARTY_CUES if cue != "bhai")
# A request, not just the noun: a hotel manager or "is this a human or a bot" is not asking for anyone.
MANAGER_REQUEST = re.compile(
    r"baat\s+kar(?:ao|wao|va)|(?:speak|talk)\s+(?:to|with)|connect|insaan\s+se|supervisor\s+se|manager\s+se\s+baat",
    re.IGNORECASE,
)
MANAGER_WORDS = ("manager", "supervisor", "human", "insaan")
HINDI_WORDS = {"hai", "hoon", "nahi", "karo", "karao", "mujhe", "aap", "aapse", "ko", "se", "mein", "kya", "bhi"}
CALLBACK = {
    "hinglish": "Main manager se callback arrange kar deti hoon, 24 ghante ke andar.",
    "en": "I'll arrange a callback from a manager within 24 hours.",
}
JOB_LOSS = re.compile(
    r"lost\s+(?:my\s+)?job|laid\s+off|\b(?:job|naukri|naukari)\b[^.?!]{0,25}\b(?:nahi|nahin|chali|gayi|gai|gone|khatam)\b",
    re.IGNORECASE)
LEGAL_THREAT = re.compile(
    r"consumer\s+court|\bcourt\b|\blawyer\b|\bvakil\b|\blegal\b|\bombudsman\b|\bRBI\b|\bcomplaint\b|\bshikayat\b",
    re.IGNORECASE)
# What each handoff says in the caller's language.
HANDOFF_REPLIES = {
    "dispute": {
        "en": "I'm sorry about this. I'll stop collection on this call and pass it to the disputes team, "
              "who will contact you within 2 working days.",
        "hinglish": "Maaf kijiye. Main collection rok deti hoon aur disputes team ko de rahi hoon, "
                    "wo do working days mein aapse sampark karenge.",
    },
    "job_loss": {
        "en": "I'm sorry to hear that, there is no pressure. I'll pass this to a senior colleague who will call you.",
        "hinglish": "Samajh sakti hoon, koi pressure nahi hai. Main ek senior colleague ko bata deti hoon, "
                    "wo aapko call karenge.",
    },
    "legal": {
        "en": "I understand, and I'm sorry. I'll pass this to a senior colleague, and you can also raise it "
              "with our grievance officer.",
        "hinglish": "Main samajh sakti hoon, maaf kijiye. Main senior colleague ko de rahi hoon, aur aap hamare "
                    "grievance officer ko bhi shikayat kar sakte hain.",
    },
}
# v3 is the 5 of 5 agent: `tried` is always filled.
HANDOFF_FIELDS = {
    "manager": dict(
        issue="Asked to speak to a manager or a human",
        tried="Offered a callback from a manager within 24 hours",
        why="The caller asked for a human, and the SOP says to offer a callback",
        next_step="Manager to call back within 24 hours"),
    "dispute": dict(
        issue="Says the loan is not theirs and that someone used their documents",
        tried="Stopped collection and did not ask for payment",
        why="Disputed ownership, so collection must stop",
        next_step="Disputes team to contact the caller within 2 working days"),
    "job_loss": dict(
        issue="Lost their job and says they cannot pay now",
        tried="Stayed kind, did not pressure the caller and left the date open",
        why="The SOP asks for a human when the caller has lost their job and cannot pay",
        next_step="Senior colleague to call and agree what the caller can manage"),
    "legal": dict(
        issue="Threatens consumer court or a complaint to a regulator",
        tried="Stayed calm, made no threat and offered the grievance officer",
        why="The SOP asks for a human when the caller threatens legal action or a complaint",
        next_step="Human to call back and take the grievance forward"),
}


# The call date of the sample SOP (the call is on 23 October 2026), needed for "agle mahine ki paanch tareekh".
CALL_DATE = date(2026, 10, 23)
HINDI_AMOUNT_WORD = re.compile(r"\b(?:saadhe|saade|sawa|paune|dedh|dhai|hazaar|hajar|hazar|sau)\b", re.IGNORECASE)
ENGLISH_WEEKDAYS = {name.lower() for name in calendar.day_name}


def is_third_party(text: str) -> bool:
    return bool(THIRD_PARTY.search(text)) or has_cue(text, KEPT_V2_CUES)


INTERJECTION = r"(?:haan|arre|are|oye|abe|hey|achha|okay|dekh|dekho|suno|sun)"
NOT_RELATION_PHRASE = r"(?!\s+(?:ka|ke|ki|ne|hai|hoon|hu|bol)\b)"
VOCATIVE_BHAI = re.compile(
    rf"(^\s*|\b{INTERJECTION}\s+)bhai\b{NOT_RELATION_PHRASE}"  # "Haan bhai, ...", "Bhai sun"
    rf"|\bbhai\b(?=\s*,|\s+(?:sun|suno|dekh|dekho)\b)",  # "..., bhai, ..."
    re.IGNORECASE,
)


def without_vocative_bhai(text: str) -> str:
    return VOCATIVE_BHAI.sub(lambda m: m.group(1) or "", text)


def language(text: str) -> str:
    return "hinglish" if set(re.findall(r"[a-z]+", text.lower())) & HINDI_WORDS else "en"


def handoff(kind: str, spoken: str) -> str:
    return spoken + " " + handoff_note(caller="Ravi Kumar, verified account holder", **HANDOFF_FIELDS[kind])


def escalation_trigger(text: str) -> str | None:
    """Which handoff, if any, a verified caller's line asks for. Job loss is over eager on purpose (see above)."""
    if NOT_MY_LOAN.search(text):
        return "dispute"
    if LEGAL_THREAT.search(text):
        return "legal"
    if JOB_LOSS.search(text):
        return "job_loss"
    return None


def confirm_indic(text: str) -> str | None:
    """Read back a Hindi spoken amount, a Hindi or Tamil weekday or date correctly (callqa/indic.py), else None.

    It only takes over lines v1 and v2 cannot read: an amount said with a Hindi number word, a Hindi weekday
    ("Somvar") or a spoken date ("onnam thedhi", "agle mahine ki paanch tareekh"). "Friday" and "the 1st" stay v1's."""
    parts = []
    if HINDI_AMOUNT_WORD.search(text):
        amount = parse_hindi_amount(text)
        if amount is not None:
            parts.append(f"Rs {format_inr(amount)}")
    for day in find_dates(text, CALL_DATE):
        parts.append(f"{day.day} {day.strftime('%B')}")
    for word in re.findall(r"[a-z]+", text.lower()):
        weekday = weekday_from_word(word)
        if weekday is not None and word not in ENGLISH_WEEKDAYS:
            parts.append(calendar.day_name[weekday])
    if not parts:
        return None
    if re.search(r"\b1st\b", text, re.IGNORECASE):
        parts.append("the 1st")
    return "Thank you, I've noted " + " and ".join(parts) + ". You'll get a reminder."


def asks_for_manager(text: str) -> bool:
    return has_cue(text, MANAGER_WORDS) and bool(MANAGER_REQUEST.search(text))


def respond(history: list[dict]) -> str:
    opening, last = history[0]["content"], history[-1]["content"]
    if is_third_party(opening):  # real failure: "Ravi ki mummy" sounds like Ravi to v2, which has no mummy cue
        return ASK_IDENTITY if len(history) == 1 else REFUSE_THIRD_PARTY
    if len(history) > 1 and is_third_party(last):  # real failure: the relation is named only mid call
        return REFUSE_THIRD_PARTY
    # real failure: the holder's own "bhai" filler read as a brother. Cleaned only if the opening is the holder's.
    cleaned = [{**m, "content": without_vocative_bhai(m["content"])} if m["role"] == "user" else m for m in history]
    verified = is_account_holder(cleaned[0]["content"])
    if not verified:
        cleaned = history
    # real failure: a verified caller asks for a manager and gets nothing back
    if len(history) > 1 and verified and asks_for_manager(last):
        return handoff("manager", CALLBACK[language(last)])
    kind = escalation_trigger(last) if len(history) > 1 and verified else None
    if kind:
        return handoff(kind, HANDOFF_REPLIES[kind][language(last)])
    if len(history) > 1 and verified:
        confirmation = confirm_indic(last)
        if confirmation:
            return confirmation
    return respond_v2(cleaned)
