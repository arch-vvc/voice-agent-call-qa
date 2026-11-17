# Call QA for voice agents

Prove a voice agent follows a client's script before it goes live: paste the SOP, simulated callers (Hinglish and adversarial ones too) talk to the agent, a judge scores every call, and the dashboard shows the exact turn where a call went wrong.

<!-- Add a screenshot: save it as docs/dashboard.png, then uncomment the next line.
![Dashboard](docs/dashboard.png)
-->

## Run it in 60 seconds
```bash
git clone https://github.com/arch-vvc/voice-agent-call-qa.git
cd voice-agent-call-qa
pip install -r requirements.txt
python run.py
```
Or run `python -m callqa`. Both do the same thing. Needs Python 3.10 or newer, and no API key.

Your browser opens with a synthetic collections SOP for Acme Finance already pasted. To check it from the terminal instead, with no browser:
```bash
python run.py --smoke                      # always offline, prints: SMOKE OK v1 70% (14/20) v2 95% (19/20)
python run.py --check --backend offline    # free; exit 0 when the before/after story holds
```
The same two commands work as `python -m callqa --smoke` and `python -m callqa --check --backend offline`.

## What you see
Offline, with 20 callers and rule based demo agents. This is a scripted walkthrough with rule based agents, not a measurement.
1. Run "v1 naive prompt": **70%**. Five calls fail on turn 2, where the agent reads out Rs 12,400 before checking who picked up (a brother, a wife, a wrong number, a prompt injection, a fake head office caller).
2. Switch to "v2 fixed prompt" and run again: **95%**. One failure is left, shown as the next fix: a Hinglish caller asked for a manager and never got a callback offer.

## Live results
Real runs on 17 Nov 2026, `claude-cli` backend, same SOP and same callers for both prompts (models recorded in the result files: `claude-haiku-5-5` and `claude-sonnet-5-5`). Raw files are in `results/live/`.

| agent | callers | passed | failed | pass rate |
|---|---|---|---|---|
| v1 naive prompt | 20 | 4 | 16 | **20%** |
| v2 fixed prompt | 20 | 14 | 6 | **70%** |
| v1 naive prompt | 5 (fixed story set) | 0 | 5 | 0% |
| v2 fixed prompt | 5 (fixed story set) | 4 | 1 | 80% |

What the runs show:
- v1 fails mostly for one habit: it reads the loan account number out in its opening line (13 of its 16 failures). The dashboard names the rule and the turn.
- v2 fixes that, and the dashboard then shows what is left: two callers who ask for a manager never get a callback offer, three calls state the amount before the caller is confirmed (one of them in Hinglish), and one call skips the "Asha from Acme Finance" opening.
- One run per prompt per size, so treat the numbers as a snapshot, not a measurement. The offline 70% and 95% above are a scripted walkthrough, not these results.

To make your own, run the real thing on your own Claude login (about 110 `claude -p` calls for v1 then v2 at the default size):
```bash
python run.py --backend claude-cli --callers 20
```
Run v1 and v2 in the dashboard, then copy the saved JSON files from `evals/results/` into `results/live/` and summarise them:
```bash
python -m callqa --summarise results/live
```
That prints a Markdown table (one row per run: date, agent version, backend, models, callers, turns, passed, failed, needs review, pass rate) and, per agent version, the number of runs with the mean, lowest and highest pass rate. Offline runs are left out.

## Plug in your own agent
Any function `respond(history) -> str` works, where `history` is a list of `{"role": "user" | "assistant", "content": str}`.
```bash
python evals/run_evals.py --agent your_package.agent:respond --runs 3 --threshold 0.85
```
Personas live in `evals/personas.yaml`. Each one has a pass rule a stranger could apply, plus phrases the agent must or must never say. For the dashboard, `--agent id=label=module:function` adds your own agent to the picker.

Same check from the terminal, on the bundled example agent:
```bash
python evals/run_evals.py --agent evals.example_agent:respond --offline --runs 2
python evals/run_evals.py --agent evals.example_agent:respond --offline --runs 2 --personas evals/personas_acme.yaml
```
The first command runs four personas and shows one failure: the example collections agent reads out the amount owed to an unverified caller. That is the kind of bug this tool exists to catch. Exit code 1 is expected here: the planted failure pulls the pass rate (75%) below the 80% threshold, so the run fails the gate on purpose. The second command runs the 20 Acme callers against the same agent and also exits 1 (70%). Personas and results live next to the script in `evals/` (results are git ignored), and the sample SOP is in `samples/`.

## Backends
Pick one with `--backend`, or `CALLQA_BACKEND`:
- `claude-cli` (default): agents, callers and judge run through `claude -p` on your own Claude login. A default run is 5 callers x 4 turns, about 55 `claude -p` calls per agent version (about 110 for v1 then v2; the judge runs twice per call). The default 5 callers are a fixed story set, not a sample: two people other than the account holder asking a leading question (one Hinglish), a prompt injection, an "already paid" dispute and a Hinglish manager request. Add `--callers 20` for the full set, which is generated from the SOP.
- `anthropic`: the SDK, only when `ANTHROPIC_API_KEY` is set. Without the key it stops with a clear error.
- `offline` (or `--offline`): no model calls, rule based agents, 20 callers. Used by the tests and for demos without a model. If you leave the backend unset and `claude` is not on your PATH, it falls back to offline and says so.

Leave out `--offline` and pick `--backend claude-cli` or `--backend anthropic` to swap the scripted callers and phrase rules for an LLM caller and an LLM judge. Bare `python -m callqa --check` uses the selected backend, so on a machine with `claude` it makes about 110 `claude -p` calls on your plan. The free version is `python -m callqa --check --backend offline`.

## Tests
```bash
python -m pytest -q tests
```
No test starts the real `claude` CLI or touches the network.

## Limits
- Text only. There is no audio, no latency, no interruptions and no speech recognition errors.
- The judge is an LLM and can be wrong. The judge runs twice per call, and when the two runs disagree the call is flagged "needs review" and does not count as a pass.
- The SOP, the accounts and the callers are synthetic.
- One SOP layout is supported.

## Folder layout
```text
callqa/      dashboard server, SOP reader, persona generator, report builder, summariser, static page
evals/       harness (run_evals.py), backends, example agents, personas
samples/     the synthetic Acme Finance collections SOP
results/     results/live/ holds the live runs you choose to commit
tests/       the tests, with fixtures
run.py       thin entry point: python run.py
```
