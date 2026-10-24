"""A deliberately simple rule based collections agent so the eval harness runs out of the box.

It has one planted bug: its opening line reads out the amount owed before checking who is on
the line, so the adversarial_data_leak persona fails. Swap it for your real agent: any function
respond(history) -> str works.
"""

OPENING = (
    "Hi, this is Asha calling from Acme Finance about an outstanding amount of Rs 12,400 "
    "that was due on 15 October. When would you be able to pay?"
)
REFUSE_THIRD_PARTY = (
    "I'm sorry, I can only discuss account details with the account holder. "
    "Please ask them to call us back."
)
DISPUTE = (
    "Thanks for telling me. I've raised a dispute so the team can check that payment. "
    "We'll call you back within 2 working days."
)


def respond(history: list[dict]) -> str:
    last = history[-1]["content"].lower()
    turn = sum(1 for m in history if m["role"] == "assistant")

    if any(w in last for w in ("brother", "family", "account number")):
        return REFUSE_THIRD_PARTY
    if "paid" in last and ("wrong" in last or "already" in last):
        return DISPUTE
    if "1st" in last:
        return "No problem, I've noted the 1st as your payment date. You'll get a reminder that morning."
    if "friday" in last:
        return "Great, Friday works. I've noted Friday as your payment date. Thank you!"
    if turn == 0:
        return OPENING
    return "Thank you. Is there anything else I can help with?"
