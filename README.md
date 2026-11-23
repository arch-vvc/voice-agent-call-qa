# Call QA for voice agents

Prove a voice agent follows a client's script before it goes live.

Paste a client's SOP. Simulated callers (confused, mind changers, Hinglish, adversarial) talk to the agent, a judge scores every call against the SOP, and you see the exact turn where each call went wrong. Fix the prompt, rerun, watch the pass rate move.

Live demo: https://voice-agent-call-qa.vercel.app

![The Call QA dashboard on the v1 run: a failing call card with the unverified caller turn marked, where the agent reads out Rs 12,400 before checking who answered](docs/v1-failing-turn.png)

## Run it in 60 seconds
```bash
git clone https://github.com/arch-vvc/voice-agent-call-qa.git
cd voice-agent-call-qa
pip install -r requirements.txt
python run.py --backend offline
```
Your browser opens with a collections SOP for Acme Finance already pasted. No API key needed. Python 3.10 or newer.

What you see (20 callers):
1. Run "v1 naive prompt": **70%**. Five calls fail on turn 2, where the agent reads out Rs 12,400 before checking who picked up (a brother, a wife, a wrong number, a prompt injection, a fake head office caller).
2. Switch to "v2 fixed prompt" and run again: **95%**. The one remaining failure is shown as the next fix: a Hinglish caller asked for a manager and never got a callback offer.

Headless checks:
```bash
python run.py --smoke                      # prints: SMOKE OK v1 70% (14/20) v2 95% (19/20)
python run.py --check --backend offline    # exit 0 when the before and after story holds
```

## The v1, v2, v3 story
Three agent versions on a harder set: 31 callers, each run 3 times and scored as pass^3 (a caller passes only if all 3 runs pass).

| agent | what changed | pass^3 on 31 callers |
|---|---|---|
| v1 | reads Rs 12,400 out in its opening line before checking who answered | 55% (17/31) |
| v2 | adds a rule: no amount until the caller says they are Ravi | 71% (22/31) |
| v3 | drops "bhai" as a third party cue, offers the manager callback, hands over with a 5 of 5 note, reads Hindi amounts and dates correctly | 97% (30/31) |

`diff --gate` compares two runs and exits 1 if the newer one regressed. It catches v2's regression and passes v2 to v3.

## Plug in your own agent
Any function `respond(history) -> str` works, where `history` is a list of `{"role": "user" | "assistant", "content": str}`.
```bash
python evals/run_evals.py --agent your_package.agent:respond --runs 3 --threshold 0.85
```
Personas live in `evals/personas.yaml`, each with a pass rule. For the dashboard, `--agent id=label=module:function` adds your agent to the picker.

## Commands
- `eval`: one agent against one caller set, with repeats per caller. Prints the pass rate, cost and latency, escalation accuracy and compliance, and saves a run record.
- `diff`: compares two run records and says which callers broke, which got fixed and which still fail.
- `compare`: a bake off of several agent configs on the same callers, with a recommendation by one fixed rule.
- `calibrate`: how often the judge agrees with a human, from labels you saved.

## Tests
```bash
python -m pytest -q
```

## Folder layout
```text
callqa/      dashboard server, SOP reader, reports, eval, diff, compare
evals/       harness, backends, example agents, personas
samples/     the Acme Finance SOP and committed run records
rules/       compliance rules
tests/       tests, fixtures and golden outputs
run.py       entry point
```
