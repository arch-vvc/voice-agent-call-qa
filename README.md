# Call QA for voice agents

Prove a voice agent follows a client's script before it goes live.

Paste a client's SOP. Simulated callers (happy path, confused, mind changers, Hinglish, adversarial) talk to the agent for several turns, a judge scores every call against the SOP, and you see the exact turn where each call went wrong. Fix the prompt, rerun, watch the pass rate move.

> Status: dashboard and eval harness run with no API key. All sample data (the Acme Finance SOP, accounts, callers) is synthetic.

## Run it in 60 seconds
```bash
git clone https://github.com/arch-vvc/lets-fucking-get-a-fde-job-and-make-money.git
cd lets-fucking-get-a-fde-job-and-make-money
pip install -r requirements.txt
python demos/call-qa/run.py
```
Or from inside the folder: `cd demos/call-qa && python -m callqa`. Both do the same thing and work from any directory.

Your browser opens with a synthetic collections SOP for Acme Finance already pasted. There is no API key needed.

What you see offline (20 callers, rule based demo agents):
1. Run "v1 naive prompt": **70%**. Five calls fail on turn 2, where the agent reads out Rs 12,400 before checking who picked up (a brother, a wife, a wrong number, a prompt injection, a fake head office caller).
2. Switch to "v2 fixed prompt" and run again: **95%**. One failure is left, shown as the next fix: a Hinglish caller asked for a manager and never got a callback offer.

Needs Python 3.10 or newer. `pip install -r requirements.txt` installs pyyaml (needed to run), the anthropic SDK (only used by `--backend anthropic`) and the test and lint tools.

Backends (`--backend`, or `CALLQA_BACKEND`):
- `claude-cli` (default): agents, callers and judge run through `claude -p` on your own Claude login. A default run is 5 callers x 4 turns, about 55 `claude -p` calls per agent version (about 110 for v1 then v2; the judge runs twice per call). The default 5 callers are a fixed story set, not a sample: two people other than the account holder asking a leading question (one Hinglish), a prompt injection, an "already paid" dispute and a Hinglish manager request. Add `--callers 20` for the full set, which is generated from the SOP.
- `anthropic`: the SDK, only when `ANTHROPIC_API_KEY` is set. Without the key it stops with a clear error.
- `offline` (or `--offline`): no model calls, rule based agents, 20 callers. Used by the tests and for demos without a model. If you leave the backend unset and `claude` is not on your PATH, it falls back to offline and says so.

Headless checks:
```bash
python demos/call-qa/run.py --smoke                      # always offline, prints: SMOKE OK v1 70% (14/20) v2 95% (19/20)
python demos/call-qa/run.py --check --backend offline    # free; exit 0 when the before/after story holds
```
The same two commands work as `python -m callqa --smoke` and `python -m callqa --check --backend offline` from inside the folder. Bare `--check` uses the selected backend, so on a machine with `claude` it makes about 110 `claude -p` calls on your plan.

### Same check from the terminal
```bash
python demos/call-qa/evals/run_evals.py --agent evals.example_agent:respond --offline --runs 2
python demos/call-qa/evals/run_evals.py --agent evals.example_agent:respond --offline --runs 2 --personas demos/call-qa/evals/personas_acme.yaml
```
The first command runs four personas and shows one failure: the example collections agent reads out the amount owed to an unverified caller. That is the kind of bug this tool exists to catch.

Exit code 1 is expected here: the planted failure pulls the pass rate (75%) below the 80% threshold, so the run fails the gate on purpose.

Leave out `--offline` and pick `--backend claude-cli` or `--backend anthropic` to swap the scripted callers and phrase rules for an LLM caller and an LLM judge.

The command works from any directory. Personas and results live next to the script in `demos/call-qa/evals/` (results are git ignored), and the sample SOP is in `demos/call-qa/samples/`.

Dashboard commands:
```bash
python -m callqa
python -m callqa --check --backend offline
```
Offline demo story is 70% for v1 and 95% for v2, with no API key needed. On `claude-cli`,
`python -m callqa --check` can make about 110 calls (about 55 per agent version).

## Plug in your own agent
Any function `respond(history) -> str` works, where `history` is a list of `{"role": "user" | "assistant", "content": str}`.
```bash
python demos/call-qa/evals/run_evals.py --agent your_package.agent:respond --runs 3 --threshold 0.85
```
Personas live in `demos/call-qa/evals/personas.yaml`. Each one has a pass rule a stranger could apply, plus phrases the agent must or must never say. For the dashboard, `--agent id=label=module:function` adds your own agent to the picker.

## Tests
```bash
python -m pytest -q demos/call-qa/tests      # from the repo root
```
`python -m pytest -q` at the repo root runs every product's tests. No test starts the real `claude` CLI or touches the network.

## Folder layout
```text
callqa/      dashboard server, SOP reader, persona generator, report builder, static page
evals/       harness (run_evals.py), backends, example agents, personas
samples/     the synthetic Acme Finance collections SOP
tests/       the tests, with fixtures
run.py       thin entry point: python demos/call-qa/run.py
```
