"""Parse a client's collections SOP (markdown, messy CRM style fields) into facts and rules.

No network and no model calls. Unrecognised input never raises: it adds to `Sop.warnings`.
"""

import re
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

# Field name (lowercased, punctuation turned into spaces) -> fact key.
FIELD_ALIASES = {
    "customer": ["cust name", "customer name", "borrower", "borrower name"],
    "company": ["company", "lender", "organisation", "organization", "nbfc"],
    "account_id": ["loan a c", "loan account", "loan account no", "account id", "account no", "account number", "acct"],
    "loan_amount": ["loan amt", "loan amount", "principal", "sanctioned amount"],
    "overdue_amount": [
        "amt due",
        "emi overdue",
        "amount outstanding",
        "outstanding amt",
        "outstanding amount",
        "overdue amount",
        "overdue amt",
        "emi due",
        "amount due",
    ],
    "due_date": ["due dt", "due date", "emi due date"],
    "dpd": ["dpd", "days past due"],
    "languages": ["lang pref", "language", "language pref", "preferred language"],
}
ALIAS_TO_FACT = {alias: fact for fact, aliases in FIELD_ALIASES.items() for alias in aliases}

FIELD_LINE = re.compile(r"^([A-Za-z][A-Za-z0-9_ ./]{0,30}?)\s*:\s*(.*)$")
NUMBER = re.compile(r"(\d[\d,]*(?:\.\d+)?)(?!\d|\.\d)\s*(?:(lakhs?|lacs?|l|crores?|cr|k)\b)?", re.IGNORECASE)
UNIT_MULTIPLIER = {"k": 1_000, "l": 100_000, "lakh": 100_000, "lac": 100_000, "crore": 10_000_000, "cr": 10_000_000}
MONTH_NAMES = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
MONTHS = {name: number for number, name in enumerate(MONTH_NAMES, 1)}


@dataclass
class Sop:
    facts: dict = field(default_factory=dict)
    rules: dict[str, list[str]] = field(default_factory=dict)  # section name -> rule lines
    samples: dict[str, list[str]] = field(default_factory=dict)  # section name -> sample lines ("> ...")
    mandatory: set[str] = field(default_factory=set)  # sections marked MANDATORY
    warnings: list[str] = field(default_factory=list)

    def amount_strings(self) -> list[str]:
        """What an unverified caller must never hear: amounts (Indian format, plain digits) and the account id."""
        out = []
        for key in ("overdue_amount", "loan_amount"):
            n = self.facts.get(key)
            if n is None:
                continue
            out += [format_inr(n), str(n)]
            if key == "loan_amount" and n >= 100_000 and n % 100_000 == 0:
                out.append(f"{n // 100_000} lakh")
        if self.facts.get("account_id"):
            out.append(self.facts["account_id"])
        return out

    def account_strings(self) -> list[str]:
        """The loan account number. The SOP says it is never read out, to anyone."""
        return [self.facts["account_id"]] if self.facts.get("account_id") else []

    def due_date_strings(self) -> list[str]:
        """The due date as an agent might say it (15 October, 15th October, 15/10, 2026-10-15), lowercased."""
        iso = self.facts.get("due_date")
        if not iso:
            return []
        due = date.fromisoformat(iso)
        month, short = due.strftime("%B").lower(), due.strftime("%b").lower()
        suffix = "th" if 10 <= due.day % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(due.day % 10, "th")
        forms = [
            f"{due.day} {month}", f"{due.day}{suffix} {month}", f"{month} {due.day}", f"{due.day} {short}",
            f"{due.day:02d}/{due.month:02d}", f"{due.day}/{due.month}", iso,
        ]
        return list(dict.fromkeys(forms))


def format_inr(n: int) -> str:
    """100000 -> '1,00,000' (last three digits, then groups of two)."""
    digits = str(abs(int(n)))
    if len(digits) > 3:
        head, tail = digits[:-3], digits[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        digits = ",".join([head, *groups, tail]) if head else ",".join([*groups, tail])
    return ("-" if n < 0 else "") + digits


def parse_inr(text: str) -> int | None:
    """'Rs 1,00,000', '₹12,400', 'Rs. 12400/-', '1.2 lakh' -> whole rupees. None if there is no number."""
    match = NUMBER.search(text or "")
    if not match:
        return None
    value = Decimal(match.group(1).replace(",", "").rstrip("."))
    unit = (match.group(2) or "").lower().rstrip("s")
    return int(round(value * UNIT_MULTIPLIER.get(unit, 1)))


def parse_date(text: str) -> date | None:
    """Day first dates: 15/10/2026, 15/10/26, 15-Oct-26, 15th October 2026, 2026-10-15. 05/10/2026 is 5 October."""
    text = re.sub(r"\(.*?\)", " ", text or "")  # drop "(some sheets say ...)"
    iso = re.search(r"(?<!\d)(\d{4})-(\d{1,2})-(\d{1,2})(?!\d)", text)
    if iso:
        return _make_date(int(iso[1]), int(iso[2]), int(iso[3]))
    numeric = re.search(r"(?<!\d)(\d{1,2})[/.-](\d{1,2})[/.-](\d{4}|\d{2})(?!\d)", text)
    if numeric:
        return _make_date(_full_year(numeric[3]), int(numeric[2]), int(numeric[1]))
    named = re.search(
        r"(?<!\d)(\d{1,2})(?:st|nd|rd|th)?[\s-]+(?:of\s+)?([A-Za-z]{3,9})[\s,.-]+(\d{4}|\d{2})(?!\d)", text
    )
    if named and named[2][:3].lower() in MONTHS:
        return _make_date(_full_year(named[3]), MONTHS[named[2][:3].lower()], int(named[1]))
    return None


def _full_year(text: str) -> int:
    return int(text) + 2000 if len(text) == 2 else int(text)


def _make_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def parse_sop(text: str) -> Sop:
    sop = Sop()
    if not (text or "").strip():
        sop.warnings.append("SOP is empty")
        return sop

    raw: dict[str, str] = {}  # fact key -> raw value
    title = ""
    section = "General"
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            heading = line.lstrip("#").strip()
            if line.startswith("##"):
                section = re.sub(r"\(.*?\)", "", heading).strip() or "General"
                if "mandatory" in heading.lower():
                    sop.mandatory.add(section)
            elif not title:
                title = heading
            continue
        if line.startswith(("- ", "* ")):
            sop.rules.setdefault(section, []).append(line[2:].strip())
            continue
        if line.startswith(">"):
            sample = line.lstrip("> ").strip()
            if not sample.upper().startswith("SYNTHETIC"):
                sop.samples.setdefault(section, []).append(sample)
            continue
        match = FIELD_LINE.match(line)
        if not match:
            continue
        name, value = match[1].strip(), match[2].strip()
        if not value:
            sop.warnings.append(f"Field '{name}' is blank")
            continue
        fact = ALIAS_TO_FACT.get(re.sub(r"[^a-z0-9]+", " ", name.lower()).strip())
        if fact and fact not in raw:
            raw[fact] = value

    _fill_facts(sop, raw, title)
    if not sop.rules:
        sop.warnings.append("No rules found: put each rule on a line starting with '- ' under a '## Section' heading")
    return sop


def _fill_facts(sop: Sop, raw: dict[str, str], title: str) -> None:
    facts = sop.facts
    if "customer" in raw:
        facts["customer"] = raw["customer"]
        facts["first_name"] = raw["customer"].split()[0]
    if "company" in raw:
        facts["company"] = raw["company"]
    elif ":" in title:
        facts["company"] = title.split(":")[0].strip()
    if "account_id" in raw:
        facts["account_id"] = raw["account_id"]
    for key in ("loan_amount", "overdue_amount"):
        if key in raw:
            amount = parse_inr(raw[key])
            if amount is None:
                sop.warnings.append(f"Could not read an amount from '{raw[key]}' ({key})")
            else:
                facts[key] = amount
    if "due_date" in raw:
        due = parse_date(raw["due_date"])
        if due is None:
            sop.warnings.append(f"Could not read a date from '{raw['due_date']}' (due_date)")
        else:
            facts["due_date"] = due.isoformat()
    if "dpd" in raw:
        digits = re.search(r"\d+", raw["dpd"])
        if digits:
            facts["dpd"] = int(digits[0])
    if "languages" in raw:
        facts["languages"] = [x.strip() for x in re.split(r"[/,&]|\band\b", raw["languages"]) if x.strip()]
    for key, label in (("customer", "customer name"), ("overdue_amount", "overdue amount")):
        if key not in facts:
            sop.warnings.append(f"No {label} found in the SOP fields")
