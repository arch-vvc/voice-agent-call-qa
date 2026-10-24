from datetime import date
from pathlib import Path

import pytest

from callqa.sop import format_inr, parse_date, parse_inr, parse_sop

SAMPLE = Path(__file__).resolve().parent.parent / "samples" / "sop_collections_acme.md"


@pytest.fixture(scope="module")
def sample():
    return parse_sop(SAMPLE.read_text(encoding="utf-8"))


def test_sample_sop_facts(sample):
    f = sample.facts
    assert f["customer"] == "Ravi Kumar"
    assert f["first_name"] == "Ravi"
    assert f["company"] == "Acme Finance"
    assert f["overdue_amount"] == 12400
    assert f["loan_amount"] == 100000
    assert f["account_id"] == "ACM-PL-0048213"
    assert f["due_date"] == "2026-10-15"
    assert f["dpd"] == 8
    assert sample.amount_strings() == ["12,400", "12400", "1,00,000", "100000", "1 lakh", "ACM-PL-0048213"]
    # the blank Alt phone field is reported, not fatal
    assert any("Alt phone" in w for w in sample.warnings)


@pytest.mark.parametrize("alias", ["Amt Due", "EMI overdue", "amount_outstanding", "Outstanding Amt"])
def test_messy_field_aliases(alias):
    sop = parse_sop(f"Cust Name: Priya Shah\n{alias}: Rs 7,500\n")
    assert sop.facts["overdue_amount"] == 7500
    assert sop.facts["first_name"] == "Priya"


def test_indian_number_formats():
    assert parse_inr("1,00,000") == 100000
    assert parse_inr("Rs 1,00,000") == 100000
    assert parse_inr("₹12,400") == 12400
    assert parse_inr("Rs. 12400/-") == 12400
    assert parse_inr("1.2 lakh") == 120000
    assert parse_inr("2 crore") == 20000000
    assert parse_inr("5k") == 5000
    assert parse_inr("Rs 12,400.") == 12400
    assert parse_inr("12400Rs") == 12400
    assert parse_inr("₹12,400only") == 12400
    assert parse_inr("1,00,000INR") == 100000
    assert parse_inr("1.2L") == 120000
    assert parse_inr("1L") == 100000
    assert parse_inr("2 lakhs") == 200000
    assert parse_inr("no money here") is None
    assert format_inr(100000) == "1,00,000"
    assert format_inr(12400) == "12,400"
    assert format_inr(999) == "999"
    assert format_inr(12345678) == "1,23,45,678"


@pytest.mark.parametrize(
    "text",
    [
        "15/10/2026",
        "15/10/26",
        "15-Oct-26",
        "15-Oct-2026",
        "15th October 2026",
        "2026-10-15",
        "15-Oct-2026 (some sheets say 15/10/26)",
    ],
)
def test_mixed_date_formats(text):
    assert parse_date(text) == date(2026, 10, 15)


def test_dates_are_day_first():
    assert parse_date("05/10/2026") == date(2026, 10, 5)
    assert parse_date("31/02/2026") is None
    assert parse_date("sometime soon") is None


def test_rules_by_section(sample):
    assert list(sample.rules) == [
        "Opening",
        "Verification",
        "Never",
        "Payment date",
        "Disputes",
        "Hardship",
        "Escalation",
    ]
    assert "Verification" in sample.mandatory
    assert any("before you state any amount" in r for r in sample.rules["Verification"])
    escalation = " ".join(sample.rules["Escalation"]).lower()
    assert "callback" in escalation and "24 hours" in escalation
    assert any("Hello, main Asha" in s for s in sample.samples["Opening"])


def test_unrelated_sop_gives_warnings_not_crash():
    sop = parse_sop("Grandma's dal recipe: soak the lentils, boil, add jeera.\nServes 4.")
    assert sop.facts == {}
    assert sop.rules == {}
    assert len(sop.warnings) >= 2
    assert parse_sop("").warnings


def test_speaker_labels_in_sample_dialogue_are_not_fields():
    text = (
        "## Sample call\nAgent: Hello, am I speaking with Ravi?\nCustomer: Haan bolo, kaun?\n"
        "## Rules\n- Verify first\n"
    )
    sop = parse_sop(text)
    assert "customer" not in sop.facts
    assert any("customer name" in w for w in sop.warnings)
