"""Indian amounts and dates, and the misheard failure. The lexicon is small on purpose. No network."""
import json
import threading
from datetime import date

from callqa.indic import find_amounts, find_dates, parse_hindi_amount, parse_spoken_date, weekday_from_word
from evals.backends import OfflineBackend, model_for
from evals.run_evals import locate_failure, run_one

CALL_DATE = date(2026, 10, 23)  # SOP due date 15 October 2026 plus 8 days past due


def test_saadhe_teen_hazaar_is_3500():
    assert parse_hindi_amount("Saadhe teen hazaar aaj de sakta hoon") == 3500
    assert parse_hindi_amount("saade teen hajar") == 3500
    assert parse_hindi_amount("saadhe 3 hazaar") == 3500


def test_dedh_lakh_is_150000():
    assert parse_hindi_amount("dedh lakh ka loan hai") == 150000
    assert parse_hindi_amount("dedh lac") == 150000
    assert parse_hindi_amount("ढाई") is None  # Devanagari is not in the lexicon: no guess


def test_dhai_sawa_paune_cases():
    assert parse_hindi_amount("dhai hazaar") == 2500
    assert parse_hindi_amount("dhai sau") == 250
    assert parse_hindi_amount("sawa lakh") == 125000
    assert parse_hindi_amount("sawa sau") == 125
    assert parse_hindi_amount("paune do lakh") == 175000
    assert parse_hindi_amount("paune teen hazaar") == 2750
    assert parse_hindi_amount("saadhe teen sau") == 350
    assert parse_hindi_amount("saadhe teen") is None  # a fraction word with no unit is not an amount


def test_barah_hazaar_chaar_sau_is_12400():
    assert parse_hindi_amount("barah hazaar chaar sau") == 12400
    assert parse_hindi_amount("do hazaar paanch sau pachas") == 2550
    assert parse_hindi_amount("ek lakh bees hazaar") == 120000
    assert parse_hindi_amount("do crore") == 20_000_000
    assert parse_hindi_amount("Rs 1,00,000") == 100000
    assert parse_hindi_amount("₹12,400 overdue hai") == 12400
    assert parse_hindi_amount("3.5 hazaar") == 3500
    assert parse_hindi_amount("pachas rupaye") == 50


def test_ek_baar_is_not_an_amount():
    for text in ("ek baar", "do baar bola", "Kal baat karte hain", "hazaar", "15 october", "5 november ko", "1st ko",
                 "Friday", "teen baje", "", "kuch nahi"):
        assert parse_hindi_amount(text) is None, text
    assert find_amounts("Your account ACM-PL-0048213 is due 15-Oct-2026, 15/10/26") == []


def test_find_amounts_returns_every_amount_in_order():
    assert find_amounts("Rs 12,400 overdue, you can pay saadhe teen hazaar now and 8,900 later") == [12400, 3500, 8900]


def test_agle_and_pichle_mahine_ki_paanch_tareekh():
    assert parse_spoken_date("Agle mahine ki paanch tareekh ko kar dunga", CALL_DATE) == date(2026, 11, 5)
    assert parse_spoken_date("pichle mahine ki paanch tareekh", CALL_DATE) == date(2026, 9, 5)
    assert parse_spoken_date("is mahine ki 28 tareekh", CALL_DATE) == date(2026, 10, 28)
    assert parse_spoken_date("is mahine ki pachees tareekh ko", CALL_DATE) == date(2026, 10, 25)
    assert parse_spoken_date("agle mahine ki 5 tareekh", date(2026, 12, 20)) == date(2027, 1, 5)
    assert parse_spoken_date("pichle mahine ki paanch tareekh", date(2026, 1, 9)) == date(2025, 12, 5)


def test_pehli_tareekh_is_the_next_1st():
    assert parse_spoken_date("pehli tareekh ko pakka", CALL_DATE) == date(2026, 11, 1)
    assert parse_spoken_date("dus tareekh", CALL_DATE) is None  # "dus" is not in the lexicon: no guess


def test_onnam_thedhi_is_the_1st():
    assert parse_spoken_date("Onnam thedhi kodukkiren", CALL_DATE) == date(2026, 11, 1)


def test_explicit_dates():
    assert parse_spoken_date("I will pay on 5 November", CALL_DATE) == date(2026, 11, 5)
    assert parse_spoken_date("5th nov please", CALL_DATE) == date(2026, 11, 5)
    assert parse_spoken_date("by november 5th", CALL_DATE) == date(2026, 11, 5)
    assert parse_spoken_date("on 2026-11-05", None) == date(2026, 11, 5)
    assert parse_spoken_date("5/11", CALL_DATE) == date(2026, 11, 5)
    assert parse_spoken_date("5 january", CALL_DATE) == date(2027, 1, 5)
    assert find_dates("first 15 October then the 5th of November", CALL_DATE) == [date(2026, 10, 15), date(2026, 11, 5)]


def test_unparseable_dates_are_none_never_a_guess():
    for text in ("kal", "Friday", "jaldi", "31 november", "agle mahine ki 31 tareekh", "", "tareekh"):
        assert parse_spoken_date(text, CALL_DATE) is None, text
    assert parse_spoken_date("agle mahine ki paanch tareekh", None) is None  # no call date: relative is unknown
    assert parse_spoken_date("5 november", None) is None


def test_somvar_is_monday():
    assert weekday_from_word("somvar") == 0
    assert weekday_from_word("Mangalvar") == 1
    assert weekday_from_word("budhvar") == 2
    assert weekday_from_word("guruvar") == 3
    assert weekday_from_word("Shukravar,") == 4
    assert weekday_from_word("shanivar") == 5
    assert weekday_from_word("ravivar") == 6
    assert weekday_from_word("friday") == 4
    assert weekday_from_word("kal") is None and weekday_from_word("") is None


# --- the misheard check -----------------------------------------------------------------------------------------

SOP_VALUES = {"amounts_inr": [12400, 100000], "dates": ["2026-10-15"], "call_date": "2026-10-23"}
SAID = "Abhi poora nahi hoga. Saadhe teen hazaar aaj de sakta hoon."


def persona(said, **extra) -> dict:
    base = {"id": "hindi_amount_partial", "name": "P", "language": "hinglish", "tags": [], "persona": "A caller.",
            "opening": "Haan, Ravi bol raha hoon.", "script": [], "must_include": ["3,500|3500"],
            "must_not_include": [], "expectations": "PASS if fine.", "sop_rule": "", "fix_hint": "",
            "said_values": said, "sop_values": SOP_VALUES}
    return {**base, **extra}


def convo(*lines):
    """Alternate caller and agent lines, the caller first."""
    return [{"role": "user" if i % 2 == 0 else "assistant", "content": text} for i, text in enumerate(lines)]


def test_misheard_amount_is_its_own_failure_kind():
    history = convo("Haan, Ravi bol raha hoon.", "Namaste Ravi ji.", SAID, "Theek hai, Rs 3,000 aaj le lete hain.")
    ok, reason, kind, turn = locate_failure(persona({"amount_inr": 3500}), history)
    assert (ok, kind, turn) == (False, "misheard", 3)
    assert "3,000" in reason and "3,500" in reason
    # it comes before the required phrase rule: the persona also requires 3,500 and the agent never said it


def test_right_amount_passes_and_a_later_correction_passes():
    ok_history = convo("Haan.", "Namaste.", SAID, "Theek hai, Rs 3,500 aaj.")
    assert locate_failure(persona({"amount_inr": 3500}), ok_history)[0] is True
    fixed = convo("Haan.", "Namaste.", SAID, "Rs 3,000 aaj.", "Nahi, saadhe teen hazaar.", "Sorry, Rs 3,500 aaj.")
    assert locate_failure(persona({"amount_inr": 3500}), fixed)[0] is True
    unfixed = convo("Haan.", "Namaste.", SAID, "Rs 3,500 aaj.", "Aur kuch?", "Phir Rs 3,000 note kiya.")
    assert locate_failure(persona({"amount_inr": 3500}), unfixed)[2] == "misheard"


def test_sop_amounts_are_not_misheard():
    history = convo("Haan.", "Namaste.", SAID, "Aapka overdue Rs 12,400 hai, to Rs 3,500 aaj theek hai.")
    assert locate_failure(persona({"amount_inr": 3500}), history)[0] is True
    only_sop = convo("Haan.", "Namaste.", SAID, "Aapka overdue Rs 12,400 hai.")  # no corrected value to hide behind
    assert locate_failure(persona({"amount_inr": 3500}, must_include=[]), only_sop)[0] is True
    # without sop_values the amounts in must_not_include still count as the SOP's own
    bare = persona({"amount_inr": 3500}, must_not_include=["12,400"])
    bare.pop("sop_values")
    history = convo("Haan.", "Namaste.", SAID, "Overdue 12,400 note, Rs 3,500 aaj.")
    assert locate_failure(bare, history)[2] == "forbidden"  # the leak rule decides, not misheard


def test_misheard_only_after_the_callers_value():
    history = convo("Haan.", "Namaste, last month you paid Rs 9,000 na?", SAID, "Rs 3,500 aaj, theek hai.")
    assert locate_failure(persona({"amount_inr": 3500}), history)[0] is True
    early = convo("Haan.", "Namaste, last month you paid Rs 9,000 na?", SAID, "Theek hai, note kar liya.")
    assert locate_failure(persona({"amount_inr": 3500}, must_include=[]), early)[0] is True
    # the caller's value never parses: the check is skipped, never guessed
    history = convo("Haan.", "Namaste.", "Thoda sa de dunga.", "Rs 3,000 aaj.")
    assert locate_failure(persona({"amount_inr": 3500}, must_include=[]), history)[0] is True


def test_misheard_date_and_sop_date_ignored():
    said = {"date": "2026-11-05"}
    caller = "Agle mahine ki paanch tareekh ko kar dunga, pakka."
    wrong = convo("Haan.", "Namaste.", caller, "Theek hai, 15 November ko note kiya.")
    ok, reason, kind, turn = locate_failure(persona(said, must_include=[]), wrong)
    assert (ok, kind, turn) == (False, "misheard", 3) and "2026-11-15" in reason
    right = convo("Haan.", "Namaste.", caller, "Due date 15 October thi, ab 5 November note kiya.")
    assert locate_failure(persona(said, must_include=[]), right)[0] is True
    # a relative date with no call date cannot be checked
    no_date = persona(said, must_include=[], sop_values={"amounts_inr": [], "dates": []})
    assert locate_failure(no_date, wrong)[0] is True


def test_story_personas_without_said_values_are_untouched():
    plain = persona({"amount_inr": 3500})
    plain.pop("said_values")
    history = convo("Haan.", "Namaste.", SAID, "Rs 3,000 aaj.")
    ok, _, kind, _ = locate_failure(plain, history)
    assert (ok, kind) == (False, "missing")  # the old required phrase rule, as before


def test_offline_run_one_reports_misheard():
    def mishearing_agent(history):
        return "Namaste." if len(history) == 1 else "Theek hai, Rs 3,000 aaj."

    pers = persona({"amount_inr": 3500}, script=[SAID])
    result = run_one(mishearing_agent, pers, 0, 6, OfflineBackend(), 0)
    assert (result.passed, result.failure_kind) == (False, "misheard")
    assert result.transcript[result.failing_turn]["role"] == "assistant"


class JudgeBackend:
    name = "fake"

    def __init__(self, reply):
        self.reply, self.prompts, self.warnings, self.lock = reply, [], [], threading.Lock()

    def complete(self, system, messages, model, max_tokens):
        if model == model_for("caller"):
            return "<END_CALL>"
        self.prompts.append(messages[0]["content"])
        return self.reply


def test_judge_may_return_broken_misheard():
    reply = json.dumps({"pass": False, "reason": "Agent confirmed Rs 3,000.", "broken": "misheard", "turn": 2})
    backend = JudgeBackend(reply)
    pers = persona({"amount_inr": 3500}, must_include=[])
    result = run_one(lambda h: "Namaste. Theek hai.", pers, 0, 2, backend, 0)
    assert (result.passed, result.failure_kind) == (False, "misheard")
    assert '"misheard"' in backend.prompts[0]
    # a plain judge failure stays "judge", and callers without said_values never see the misheard option
    plain = persona({"amount_inr": 3500}, must_include=[])
    plain.pop("said_values")
    other = JudgeBackend(json.dumps({"pass": False, "reason": "bad", "broken": "target", "turn": 1}))
    result = run_one(lambda h: "Namaste.", plain, 0, 2, other, 0)
    assert result.failure_kind == "judge" and "misheard" not in other.prompts[0]
