"""Q2 guards: v1 never drifts, and no v2 prompt leaks a persona id or a committed caller line."""
import json
import re
from pathlib import Path

import yaml

from callqa.sop import parse_sop
from evals import llm_agent

ROOT = Path(__file__).resolve().parent.parent
SOP = parse_sop((ROOT / "samples" / "sop_collections_acme.md").read_text())

V1_PROMPT = (
    "You are a collections agent on a phone call for Acme Finance. Keep every reply to one or two short sentences, "
    "in a spoken style.\n\n"
    "Account facts:\n- Account holder: Ravi Kumar\n- Loan account: ACM-PL-0048213\n- Loan amount: Rs 1,00,000\n"
    "- Overdue amount: Rs 12,400\n- Due date: 2026-10-15\n- Days past due: 8\n\n"
    "Open every call by greeting the customer by name and stating the overdue amount, due date and loan account "
    "number.\n\n"
    "Payment date:\n- Ask for a specific payment date and repeat the final date back to the caller.\n"
    "- If the caller changes their mind, confirm the last date they gave.\n"
    "- Example: Toh aap 1st ko payment karenge, sahi? Main 1st note kar rahi hoon.\n\n"
    "Hardship:\n- If the caller has lost income, be kind, do not pressure them, and still ask for a date they can "
    "manage.\n- Example: Samajh sakti hoon. Aap koi ek date bata dijiye jab aap thoda payment kar sakein.\n"
)


def _persona_ids() -> set[str]:
    ids = {p["id"] for p in yaml.safe_load((ROOT / "evals" / "personas_acme.yaml").read_text())}
    for path in (ROOT / "results" / "live").glob("*.json"):
        ids |= {c["persona_id"] for c in json.loads(path.read_text())["calls"]}
    return ids


def _caller_sentences() -> set[str]:
    out = set()
    for path in (ROOT / "results" / "live").glob("*.json"):
        for call in json.loads(path.read_text())["calls"]:
            for turn in call["transcript"]:
                if turn["role"] == "user":
                    out |= {s.strip() for s in re.split(r"(?<=[.!?])\s+", turn["content"]) if len(s.strip()) >= 15}
    return out


def test_v1_prompt_is_byte_for_byte_unchanged():
    assert llm_agent.build_system_prompt(SOP, "v1") == V1_PROMPT


def test_v2_prompt_has_no_persona_id_and_no_committed_caller_line():
    prompt = llm_agent.build_system_prompt(SOP, "v2")
    ids = _persona_ids()
    sentences = _caller_sentences()
    assert ids and sentences  # the guard must be looking at something
    assert not [i for i in ids if i in prompt]
    assert not [s for s in sentences if s in prompt]
