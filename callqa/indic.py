"""Indian spoken amounts and dates, and the misheard check. Reads romanised Hindi and Hinglish.

THE LEXICON IS SMALL AND INCOMPLETE ON PURPOSE. It covers only what the demo callers and agents say:
  - number words used in amounts: ek to das, gyarah to unnis, bees, pachees, tees, chalis, pachas, saath, sattar,
    assi, nabbe (a few spellings each; not every number between 21 and 99 has a word here)
  - scales: sau, hazaar / hajar, lakh / lac, crore
  - fractions: sawa (plus a quarter), saadhe (plus a half), paune (minus a quarter), dedh (1.5), dhai (2.5)
  - dates: "agle / pichle / is mahine ki N tareekh", "pehli tareekh", Tamil "onnam thedhi", plus plain
    "5 November", "November 5th", "5/11", "2026-11-05"
  - weekdays: Hindi names (somvar to ravivar, a few spellings) and English names
Anything outside it returns None (or is left out of a list). Nothing here ever guesses: an unparseable phrase is
"unknown", never a best effort number. Devanagari script is not read at all.

Choices a reader should know about:
  - A bare number word is not an amount ("ek baar" is "once"). An amount needs a scale word, a currency marker
    (Rs, rupaye, the rupee sign) or digits of 100 or more.
  - A bare "N tareekh" means the next such day after the call date. A year-less "5 January" is the year that puts
    it nearest the call date.
  - Relative dates need the call date; without it they are None.
"""
from __future__ import annotations

import re
from datetime import date
from fractions import Fraction

from callqa.sop import format_inr

NUMBER_WORDS = {
    "ek": 1, "do": 2, "teen": 3, "chaar": 4, "char": 4, "paanch": 5, "panch": 5, "chhe": 6, "chhah": 6, "che": 6,
    "saat": 7, "aath": 8, "nau": 9, "das": 10, "gyarah": 11, "barah": 12, "terah": 13, "chaudah": 14, "pandrah": 15,
    "solah": 16, "satrah": 17, "atharah": 18, "unnis": 19, "bees": 20, "pachees": 25, "tees": 30, "chalis": 40,
    "pachas": 50, "saath": 60, "sattar": 70, "assi": 80, "nabbe": 90,
}
SCALES = {"sau": 100, "hazaar": 1000, "hajar": 1000, "hazar": 1000, "lakh": 100_000, "lac": 100_000,
          "lakhs": 100_000, "crore": 10_000_000}
FRACTION_WORDS = {"sawa": Fraction(1, 4), "saadhe": Fraction(1, 2), "saade": Fraction(1, 2), "paune": Fraction(-1, 4)}
FIXED_FRACTIONS = {"dedh": Fraction(3, 2), "dhai": Fraction(5, 2)}
CURRENCY_BEFORE = {"rs", "inr", "₹"}
CURRENCY_AFTER = {"rs", "rupaye", "rupaiye", "rupees", "rupee", "rupay", "rupaya"}
ORDINALS = {"pehli": 1, "pehla": 1, "pahli": 1, "doosri": 2, "dusri": 2, "teesri": 3, "chauthi": 4, "onnam": 1}
TAREEKH = {"tareekh", "tarikh", "taarikh", "tareek", "thedhi", "theti"}
MONTHS = {"january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6, "july": 7, "august": 8,
          "september": 9, "october": 10, "november": 11, "december": 12,
          "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10,
          "nov": 11, "dec": 12}
MONTH_RE = "|".join(sorted(MONTHS, key=len, reverse=True))
RELATIVE = {"agle": 1, "agli": 1, "next": 1, "pichle": -1, "pichli": -1, "last": -1, "previous": -1,
            "is": 0, "iss": 0, "isi": 0, "this": 0}
MAHINA = {"mahine", "mahina", "month"}
FILLER = {"ki", "ka", "ke", "ko"}
WEEKDAYS = {
    "somvar": 0, "somwar": 0, "sombar": 0, "monday": 0,
    "mangalvar": 1, "mangalwar": 1, "mangal": 1, "tuesday": 1,
    "budhvar": 2, "budhwar": 2, "budh": 2, "wednesday": 2,
    "guruvar": 3, "guruwar": 3, "veervar": 3, "brihaspativar": 3, "thursday": 3,
    "shukravar": 4, "shukrawar": 4, "sukravar": 4, "friday": 4,
    "shanivar": 5, "shaniwar": 5, "shanivaar": 5, "saturday": 5,
    "ravivar": 6, "raviwar": 6, "itvar": 6, "itwar": 6, "sunday": 6,
}
# a number followed by one of these is a date, a time or a count, not money
NOT_MONEY_AFTER = set(MONTHS) | TAREEKH | {"st", "nd", "rd", "th", "baje", "baar", "din", "day", "days", "am", "pm",
                                          "minute", "minutes", "ghante", "hours", "mahine", "months"}
TOKEN = re.compile(r"₹|\d+(?:,\d+)*(?:\.\d+)?|[^\W\d_]+")


def _tokens(text: str) -> list[tuple[str, str, int, int]]:
    """(kind, text, start, end) with kind num, word or cur. Digits glued to a date or an id (15/10, ACM-PL-004) are
    marked kind "skip"."""
    low = text.lower()
    out = []
    for m in TOKEN.finditer(low):
        word = m[0]
        kind = "cur" if word == "₹" else "num" if word[0].isdigit() else "word"
        out.append([kind, word, m.start(), m.end()])
    for i, tok in enumerate(out):
        if tok[0] != "num":
            continue
        before = low[tok[2] - 1] if tok[2] else " "
        after = low[tok[3]] if tok[3] < len(low) else " "
        nxt = out[i + 1] if i + 1 < len(out) else None
        prev = out[i - 1] if i else None
        near_next = nxt is not None and nxt[2] - tok[3] <= 1
        if before in "/-" or after in "/-" or (before == "." and tok[2] > 1 and low[tok[2] - 2].isdigit()):
            tok[0] = "skip"
        elif near_next and nxt[1] in NOT_MONEY_AFTER:
            tok[0] = "skip"
        elif prev is not None and prev[1] in MONTHS and tok[2] - prev[3] <= 1:
            tok[0] = "skip"
    return [tuple(t) for t in out]


def _number(token: tuple[str, str, int, int]) -> Fraction | None:
    kind, word = token[0], token[1]
    if kind == "num":
        return Fraction(word.replace(",", ""))
    return Fraction(NUMBER_WORDS[word]) if word in NUMBER_WORDS else None


def _is_amount_word(token: tuple[str, str, int, int]) -> bool:
    kind, word = token[0], token[1]
    return kind == "num" or word in NUMBER_WORDS or word in SCALES or word in FRACTION_WORDS or word in FIXED_FRACTIONS


def _value(run: list[tuple[str, str, int, int]]) -> tuple[int | None, bool]:
    """(value, has_scale) for a run of number tokens, or (None, False) when the run does not add up."""
    total, current = Fraction(0), Fraction(0)
    frac_kind: str | None = None
    frac_base: Fraction | None = None
    fixed: Fraction | None = None
    has_scale = False
    for tok in run:
        word = tok[1]
        if word == "aur":
            continue
        if word in FRACTION_WORDS:
            frac_kind, frac_base = word, None
        elif word in FIXED_FRACTIONS:
            fixed = FIXED_FRACTIONS[word]
        elif word in SCALES:
            has_scale = True
            if fixed is not None:
                mult = fixed
            elif frac_kind is not None:
                step = FRACTION_WORDS[frac_kind]
                if frac_base is None:
                    if frac_kind == "saadhe" or frac_kind == "saade":
                        return None, False  # "saadhe lakh" is not an idiom: no guess
                    mult = (Fraction(1) if step > 0 else Fraction(0)) + step
                else:
                    mult = frac_base + step
            elif current > 0:
                mult, current = current, Fraction(0)
            else:
                return None, False  # a scale word with nothing to multiply
            frac_kind, frac_base, fixed = None, None, None
            if SCALES[word] == 100:
                current = mult * 100
            else:
                total += mult * SCALES[word]
        else:
            number = _number(tok)
            if number is None:
                return None, False
            if frac_kind is not None and frac_base is None:
                frac_base = number
            else:
                current += number
    if frac_kind is not None or fixed is not None:
        return None, False  # a fraction word with no scale after it
    result = total + current
    if result <= 0 or result.denominator != 1:
        return None, False
    return int(result), has_scale


def _amount_runs(text: str) -> list[tuple[int, int]]:
    """[(position, amount)] for each amount in the text, in order."""
    tokens = _tokens(text)
    found, i = [], 0
    while i < len(tokens):
        if not _is_amount_word(tokens[i]) or tokens[i][0] == "skip":
            i += 1
            continue
        j = i
        while j < len(tokens):
            tok = tokens[j]
            joins_aur = tok[1] == "aur" and j > i and j + 1 < len(tokens) and _is_amount_word(tokens[j + 1])
            if joins_aur or (_is_amount_word(tok) and tok[0] != "skip"):
                j += 1
            else:
                break
        run = tokens[i:j]
        value, has_scale = _value(run)
        before = tokens[i - 1][1] if i else ""
        after = tokens[j][1] if j < len(tokens) else ""
        currency = before in CURRENCY_BEFORE or after in CURRENCY_AFTER
        single_digits = len(run) == 1 and run[0][0] == "num"
        if value is not None and (has_scale or currency or (single_digits and value >= 100)):
            found.append((tokens[i][2], value))
        i = j
    return found


def find_amounts(text: str) -> list[int]:
    """Every rupee amount in the text, in order of appearance: Rs 12,400, 1,00,000, saadhe teen hazaar, dedh lakh."""
    return [value for _, value in _amount_runs(text)]


def parse_hindi_amount(text: str) -> int | None:
    """The first rupee amount in the text, or None. Not an amount: "ek baar", a date, a bare "hazaar"."""
    amounts = find_amounts(text)
    return amounts[0] if amounts else None


def weekday_from_word(word: str) -> int | None:
    """Monday is 0 and Sunday is 6 (the same as date.weekday()). Hindi and English names; anything else is None."""
    return WEEKDAYS.get(word.strip().lower().strip(".,!?;:'\""))


def _nearest_year(day: int, month: int, call_date: date) -> date | None:
    """The date with this day and month in the year that puts it nearest the call date, or None if it never exists."""
    options = []
    for year in (call_date.year - 1, call_date.year, call_date.year + 1):
        try:
            options.append(date(year, month, day))
        except ValueError:
            continue
    return min(options, key=lambda d: abs((d - call_date).days)) if options else None


def _shift_month(call_date: date, months: int, day: int) -> date | None:
    index = call_date.year * 12 + call_date.month - 1 + months
    try:
        return date(index // 12, index % 12 + 1, day)
    except ValueError:
        return None


def _next_day_number(call_date: date, day: int) -> date | None:
    """The next date after the call date that has this day of the month."""
    for step in range(0, 13):
        candidate = _shift_month(call_date, step, day)
        if candidate is not None and candidate > call_date:
            return candidate
    return None


def _year(raw: str | None, day: int, month: int, call_date: date | None) -> date | None:
    try:
        if raw:
            return date(int(raw), month, day)
    except ValueError:
        return None
    return _nearest_year(day, month, call_date) if call_date is not None else None


def _tareekh_dates(text: str, call_date: date | None) -> list[tuple[int, int, date]]:
    tokens = _tokens(text)
    # "5 tareekh" marks the 5 as a date part for amounts; here every digit counts
    tokens = [("num", t[1], t[2], t[3]) if t[0] == "skip" and t[1][0].isdigit() else t for t in tokens]
    out = []
    for t, tok in enumerate(tokens):
        if tok[1] not in TAREEKH:
            continue
        j = t
        while j > 0 and (tokens[j - 1][1] in ORDINALS or _number(tokens[j - 1]) is not None
                         or tokens[j - 1][1] in ("aur",)):
            j -= 1
        phrase = tokens[j:t]
        if not phrase:
            continue
        if len(phrase) == 1 and phrase[0][1] in ORDINALS:
            day = ORDINALS[phrase[0][1]]
        else:
            if any(p[1] in ORDINALS for p in phrase):
                continue
            value, _ = _value(phrase)
            day = value
        if day is None or not 1 <= day <= 31 or call_date is None:
            continue
        k = j - 1
        if k >= 0 and tokens[k][1] in FILLER:
            k -= 1
        if k >= 0 and tokens[k][1] in MAHINA:
            qualifier = tokens[k - 1][1] if k >= 1 else ""
            if qualifier not in RELATIVE:
                continue  # "mahine ki 5 tareekh" with no is, agle or pichle: which month is unknown
            result = _shift_month(call_date, RELATIVE[qualifier], day)
            start = tokens[k - 1][2]
        else:
            result = _next_day_number(call_date, day)
            start = tokens[j][2]
        if result is not None:
            out.append((start, tok[3], result))
    return out


def _explicit_dates(text: str, call_date: date | None) -> list[tuple[int, int, date]]:
    low = text.lower()
    out = []

    def add(m: re.Match, result: date | None) -> None:
        if result is not None:
            out.append((m.start(), m.end(), result))

    for m in re.finditer(r"\b(20\d\d)-(\d{2})-(\d{2})\b", low):
        try:
            add(m, date(int(m[1]), int(m[2]), int(m[3])))
        except ValueError:
            continue
    pattern = rf"\b(\d{{1,2}})(?:st|nd|rd|th)?[\s\-]+(?:of\s+)?({MONTH_RE})\b(?:[\s\-,]+(20\d\d)\b)?"
    for m in re.finditer(pattern, low):
        add(m, _year(m[3], int(m[1]), MONTHS[m[2]], call_date))
    for m in re.finditer(rf"\b({MONTH_RE})\b[\s\-]+(\d{{1,2}})(?:st|nd|rd|th)?\b(?!\d)(?:[\s\-,]+(20\d\d)\b)?", low):
        add(m, _year(m[3], int(m[2]), MONTHS[m[1]], call_date))
    for m in re.finditer(r"(?<![\d/])(\d{1,2})/(\d{1,2})(?:/(\d{4}|\d{2}))?(?![\d/])", low):
        year = m[3] if m[3] is None or len(m[3]) == 4 else str(2000 + int(m[3]))
        add(m, _year(year, int(m[1]), int(m[2]), call_date) if 1 <= int(m[2]) <= 12 else None)
    return out


def find_dates(text: str, call_date: date | None) -> list[date]:
    """Every date the text names, in order of appearance. Weekdays and words like "kal" are not dates here."""
    spans = _explicit_dates(text, call_date) + _tareekh_dates(text, call_date)
    spans.sort(key=lambda s: (s[0], -s[1]))
    kept: list[tuple[int, int, date]] = []
    for span in spans:
        if not kept or span[0] >= kept[-1][1]:
            kept.append(span)
    return [s[2] for s in kept]


def parse_spoken_date(text: str, call_date: date | None) -> date | None:
    """The first date in the text, or None. `call_date` is needed for relative dates and for a year-less date."""
    dates = find_dates(text, call_date)
    return dates[0] if dates else None


# --- the misheard check ------------------------------------------------------------------------------------------

def _iso(value) -> date | None:
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return None


def misheard_failure(persona: dict, transcript: list[dict]) -> tuple[str, int] | None:
    """(reason, agent turn) when the agent says a different amount or date than the caller said, else None.

    Only for personas with `said_values` ({"amount_inr": int} or {"date": "YYYY-MM-DD"}). Only agent turns after the
    caller's own turn that states the value count; if no caller turn parses to the value, nothing is checked.
    The SOP's own amounts and dates (`sop_values`: amounts_inr, dates, call_date; and anything in must_not_include)
    are ignored. An agent that gets it wrong and then says the right value later has corrected itself and passes."""
    said = persona.get("said_values")
    if not isinstance(said, dict):
        return None
    sop = persona.get("sop_values") or {}
    call_date = _iso(sop.get("call_date"))
    banned = " ".join(str(x) for x in persona.get("must_not_include", []))
    if "amount_inr" in said:
        target = said["amount_inr"]
        ignore = {*sop.get("amounts_inr", []), *find_amounts(banned)}

        def find(text: str) -> list:
            return find_amounts(text)

        def show(value) -> str:
            return f"Rs {format_inr(value)}"
    elif "date" in said:
        target = _iso(said["date"])
        if target is None:
            return None
        ignore = {d for d in map(_iso, sop.get("dates", [])) if d} | set(find_dates(banned, call_date))

        def find(text: str) -> list:
            return find_dates(text, call_date)

        def show(value) -> str:
            return value.isoformat()
    else:
        return None
    start = next((i for i, m in enumerate(transcript) if m["role"] == "user" and target in find(m["content"])), None)
    if start is None:
        return None
    agent_turns = [(i, find(m["content"])) for i, m in enumerate(transcript) if i > start and m["role"] == "assistant"]
    for turn, values in agent_turns:
        wrong = [v for v in values if v != target and v not in ignore]
        if not wrong or target in values:
            continue
        if any(target in later for later_turn, later in agent_turns if later_turn > turn):
            continue
        return f"agent heard {show(wrong[0])}, the caller said {show(target)}", turn
    return None

