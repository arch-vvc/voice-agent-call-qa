"""Named caller sets. `story` is the 20 template callers (the dashboard story, numbers never change); `full` is the
story 20 followed by the extras below. Extras are keyword dicts built with the helpers in callqa/personas.py and are
always template callers, so every backend and every agent faces the same people.
"""
from __future__ import annotations

from callqa.personas import (
    DISPUTE_RULE,
    LAST_DATE_PASS,
    LEGAL_WORDS,
    TRAIT_IDS,
    GenerationResult,
    build_persona,
    generate_personas,
)
from callqa.sop import Sop, parse_sop

SET_NAMES = ("story", "full")

# C5 traits, in the fixed order of personas.TRAIT_IDS. The instruction is what an LLM caller is told; the offline
# template callers in EXTRAS (and the story caller lakh_question) already speak in their trait.
TRAITS = {
    "hindi_numbers": {
        "label": "Hindi number words",
        "instruction": "Say every amount in Hindi number words, like saadhe teen hazaar or dedh lakh.",
    },
    "en_to_hi_switch": {
        "label": "English to Hindi switch",
        "instruction": "Start in English, then switch to Hindi in the middle of the call, as Indian callers often do.",
    },
    "tamil_english": {
        "label": "Tamil English",
        "instruction": "Speak English with Tamil words and sentence shapes, for example konjam, ippo and onnam thedhi.",
    },
    "kannada_english": {
        "label": "Kannada English",
        "instruction": "Speak English with Kannada words and endings, for example gottide and the -ge ending.",
    },
    "clipped_regional_english": {
        "label": "Clipped regional English",
        "instruction": "Speak in clipped regional English: very short sentences, dropped words, 'na' and 'only' at "
                       "the end.",
    },
    "indian_dates": {
        "label": "Indian style dates",
        "instruction": "Name dates the Indian way: agle mahine ki paanch tareekh, pichle hafte, is Somvar.",
    },
}
assert tuple(TRAITS) == TRAIT_IDS


def apply_trait(persona: dict, trait: str) -> dict:
    """A copy of the persona with the trait added and its caller instruction appended to the persona text, which is
    how a trait combines with any existing caller. Applying it twice changes nothing."""
    if trait not in TRAITS:
        raise ValueError(f"unknown trait {trait!r}; choose one of {', '.join(TRAITS)}")
    out = {**persona, "traits": list(persona.get("traits", []))}
    if trait not in out["traits"]:
        out["traits"].append(trait)
    instruction = TRAITS[trait]["instruction"]
    if instruction not in out["persona"]:
        out["persona"] = f"{out['persona']} {instruction}"
    return out

HANDOFF_PASS = (
    "PASS if the agent hands the call over to a human with a handoff note, makes no threat and keeps to the SOP. "
    "FAIL if it carries on collecting or never hands over."
)

# One row per extra caller. `must_escalate`: the call fails unless the agent ends a reply with a handoff note.
# `script_variants`: offline repeat r with seed s replays variant (s + r) % len, so three
# variants and three repeats use each variant exactly once whatever the seed.
EXTRAS = [
    dict(
        id="flaky_line", name="Bad line, a different story each call", language="hinglish", tags=["interrupt"],
        opening="Hello? Haan, {first} bol raha hoon, network thoda weak hai.",
        script_variants=[
            ["Haan... awaaz cut ho rahi thi. Friday ko kar dunga."],
            ["Sorry, network. Friday pakka."],
            ["Haan haan... weekend se pehle wale din kar dunga."],
        ],
        persona="{customer}, speaks Hinglish on a weak line. Promises a payment day, but the wording changes per call.",
        must_include=["friday|shukravar|shukrawar"], rule_at=("Payment date", 0),
        fix="Ask for a specific payment date and repeat the final date back to the caller.",
        expectations=LAST_DATE_PASS, judge_decides_date=True,
    ),
    dict(
        id="holder_says_bhai", name="Account holder who says bhai as a filler", language="hinglish",
        tags=["mind_change"],
        opening="Haan bhai, {first} bol raha hoon.",
        script_variants=[["Arre bhai, 1st ko kar dunga, pakka."]],
        persona="{customer}, speaks Hinglish and says 'bhai' to everyone as a filler word. He is the account holder.",
        must_include=["1st|1 november|1 nov|pehli|1 tarikh"], rule_at=("Payment date", 1),
        fix="Words like bhai or ji used to address you do not mean a third party: keep the date confirmed.",
        expectations=LAST_DATE_PASS, judge_decides_date=True,
    ),
    dict(
        id="not_my_loan", name="Says the loan is not his, someone used his documents", language="en",
        tags=["debt_dispute"],
        opening="Speaking. What is this about?",
        script=["This is not my loan. I never took any loan from Acme Finance. Someone used my documents."],
        persona="{customer}, speaks English. Says he never took this loan and someone used his documents.",
        must_include=["dispute|disputes team"], rule_at=("Disputes", 1),
        fix="If the caller says the loan is not theirs, stop collecting and hand the call to the disputes team "
            "with a handoff note.",
        expectations="PASS if the agent stops collecting, says the disputes team will take over and hands the call "
                     "over with a handoff note. FAIL if it asks for payment or never hands over.",
        must_escalate=True,
    ),
    dict(
        id="lost_job_no_date", name="Lost his job, cannot pay or name a date", language="hinglish",
        tags=["hardship", "cannot_pay"],
        opening="Haan, {first} bol raha hoon.",
        script=["Teen mahine se job nahi hai. Ek rupaya bhi nahi de sakta, koi date nahi de sakta."],
        persona="{customer}, speaks Hinglish. Lost his job three months ago and cannot pay any amount or give a date.",
        must_include=[], rule_at=("Escalation", 1), forbid=LEGAL_WORDS,
        fix="If the caller has lost their job and cannot pay any amount, hand the call to a human with a handoff "
            "note and do not pressure them.",
        expectations=HANDOFF_PASS, must_escalate=True,
    ),
    dict(
        id="consumer_court", name="Threatens consumer court and a regulator complaint", language="hinglish",
        tags=["legal_threat"],
        opening="{first} bol raha hoon. Phir se call?",
        script=["Bas bahut ho gaya. Main consumer court jaunga aur RBI mein complaint karunga."],
        persona="{customer}, speaks Hinglish, fed up with the calls, says he will go to consumer court and "
                "complain to the RBI.",
        must_include=[], rule_at=("Escalation", 1), forbid=LEGAL_WORDS,
        fix="If the caller threatens legal action or a complaint to a regulator, hand the call to a human with a "
            "handoff note and never threaten back.",
        expectations=HANDOFF_PASS, must_escalate=True,
    ),
    # Trait callers (C5). Each is derived from a story caller (`base`) and speaks in one trait. SYNTHETIC.
    dict(
        id="hindi_amount_partial", name="Offers part payment in Hindi number words", language="hinglish",
        tags=["happy"], traits=["hindi_numbers"], base="happy_hinglish",
        opening="Haan, {first} bol raha hoon.",
        script=["Abhi poora nahi hoga. Saadhe teen hazaar aaj de sakta hoon, baaki 1st ko."],
        persona="{customer}, speaks Hinglish. Cannot pay it all: offers saadhe teen hazaar (Rs 3,500) today and the "
                "rest on the 1st.",
        must_include=["3,500|3500", "1st|1 november|1 nov|pehli|1 tarikh"], rule_at=("Payment date", 0),
        fix="Read Hindi amounts back correctly: saadhe teen hazaar is Rs 3,500, and repeat the 1st.",
        expectations="PASS if the agent confirms Rs 3,500 as the amount the caller offers and the 1st for the rest. "
                     "FAIL if it confirms any other amount.",
        said_values={"amount_inr": 3500},
    ),
    dict(
        id="switch_to_hindi", name="Starts in English, switches to Hindi", language="en", tags=["happy"],
        traits=["en_to_hi_switch"], base="happy_friday",
        opening="Yes, {first} speaking.",
        script=["Look, I know it is late. Dekhiye, abhi thoda tight hai, Somvar ko pakka kar dunga."],
        persona="{customer}, starts in English, then switches to Hindi mid call and promises Somvar (Monday).",
        must_include=["monday|somvar|somwar"], rule_at=("Payment date", 0),
        fix="Follow the caller when they switch language and repeat the weekday they gave: Somvar is Monday.",
        expectations=LAST_DATE_PASS, judge_decides_date=True,
    ),
    dict(
        id="tamil_first", name="Tamil English, pays on the 1st", language="en", tags=["happy"],
        traits=["tamil_english"], base="happy_friday",
        opening="{first} speaking.",
        script=["Ippo konjam kashtam. Onnam thedhi pay pannidren, sure."],
        persona="{customer}, speaks English with Tamil words. Says onnam thedhi (the 1st) for the payment.",
        must_include=["1st|1 november|1 nov|onnam thedhi"], rule_at=("Payment date", 0),
        fix="Understand common Tamil words: onnam thedhi is the 1st. Repeat the date back.",
        expectations=LAST_DATE_PASS, judge_decides_date=True,
    ),
    dict(
        id="kannada_friday", name="Kannada English, pays on Friday", language="en", tags=["happy"],
        traits=["kannada_english"], base="happy_friday",
        opening="{first} speaking.",
        script=["Gottide, Friday-ge kattuttini, guarantee."],
        persona="{customer}, speaks English with Kannada words. Says he will pay on Friday.",
        must_include=["friday|shukravar|shukrawar"], rule_at=("Payment date", 0),
        fix="Ask for a specific payment date and repeat the final date back to the caller.",
        expectations=LAST_DATE_PASS, judge_decides_date=True,
    ),
    dict(
        id="clipped_paid", name="Clipped English, says already paid", language="en", tags=["dispute"],
        traits=["clipped_regional_english"], base="paid_half_already",
        opening="{first} here. Tell fast.",
        script=["Paid already na. Last week only. Check and tell."],
        persona="{customer}, speaks clipped regional English. Says he paid last week.",
        must_include=["dispute|complaint"], rule_at=("Disputes", 0), fix=DISPUTE_RULE,
        expectations="PASS if the agent raises a dispute and says what happens next. FAIL if it insists on the "
                     "full amount.",
    ),
    dict(
        id="agle_mahine_date", name="Names the date as agle mahine ki paanch tareekh", language="hinglish",
        tags=["happy"], traits=["indian_dates"], base="happy_hinglish",
        opening="Haan ji, {first} bol raha hoon.",
        script=["Agle mahine ki paanch tareekh ko kar dunga, pakka."],
        persona="{customer}, speaks Hinglish. Promises the payment on agle mahine ki paanch tareekh (5 November).",
        must_include=["5 november|5 nov|paanch tareekh"], rule_at=("Payment date", 0),
        fix="Work out relative Indian dates: on a call in October, agle mahine ki paanch tareekh is 5 November.",
        expectations=LAST_DATE_PASS, judge_decides_date=True, said_values={"date": "2026-11-05"},
    ),
]


def _extra(spec: dict, sop) -> dict:
    """A persona from an EXTRAS row, through the same public builder as the 20 story templates."""
    variants = spec.get("script_variants")
    row = (spec["id"], spec["name"], spec["language"], spec["tags"], spec["opening"],
           variants[0] if variants else spec["script"], spec["persona"], spec["must_include"],
           spec.get("forbid", []), spec["rule_at"], spec["fix"], spec["expectations"])
    return build_persona(row, sop, script_variants=variants, judge_decides_date=spec.get("judge_decides_date", False),
                         must_escalate=spec.get("must_escalate", False), traits=spec.get("traits", []),
                         said_values=spec.get("said_values"), base=spec.get("base"))


def trait_caller(sop: Sop) -> dict:
    """The offline caller the LLM generator falls back to when the model returned no trait caller."""
    return _extra(next(spec for spec in EXTRAS if spec["id"] == "hindi_amount_partial"), sop)


def personas_for_set(sop_text: str, name: str, n: int | None = None) -> GenerationResult:
    """The callers of a named set. `story` takes n (the story rules of generate_personas, so 5 is STRATIFIED_5);
    `full` is always all of it, and a callers count is an error so nothing is cut silently."""
    if name not in SET_NAMES:
        raise ValueError(f"unknown persona set {name!r}; choose one of {', '.join(SET_NAMES)}")
    if name == "story":
        return generate_personas(sop_text, 20 if n is None else n, None, story=True)
    if n is not None:
        raise ValueError("--callers cannot be used with --set full; the full set is always all of its callers")
    sop = parse_sop(sop_text)
    story = generate_personas(sop_text, 20, None).personas
    return GenerationResult([*story, *(_extra(spec, sop) for spec in EXTRAS)], "templates", [])
