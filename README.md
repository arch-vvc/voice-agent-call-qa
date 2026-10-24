# Call QA for voice agents

Prove a voice agent follows a client's script before it goes live.

Paste a client's SOP. Simulated callers (happy path, confused, mind changers, Hinglish, adversarial) talk to the agent for several turns, a judge scores every call against the SOP, and you see the exact turn where each call went wrong. Fix the prompt, rerun, watch the pass rate move.

> Status: early build. The eval harness works today; the dashboard is next.

## Run it in 60 seconds
```bash
git clone https://github.com/arch-vvc/lets-fucking-get-a-fde-job-and-make-money.git
cd lets-fucking-get-a-fde-job-and-make-money
pip install -r requirements.txt
python demos/call-qa/evals/run_evals.py --agent evals.example_agent:respond --offline --runs 2
```
You should see four personas and one failure: the example collections agent reads out the amount owed to an unverified caller. That is the kind of bug this tool exists to catch.

Exit code 1 is expected here: the planted failure pulls the pass rate (75%) below the 80% threshold, so the run fails the gate on purpose.

With `ANTHROPIC_API_KEY` set, run without offline mode to use an LLM caller and an LLM judge instead of scripted lines and phrase rules.

The command works from any directory. Personas and results live next to the script in `demos/call-qa/evals/`.

Dashboard commands:
```bash
python -m callqa
python -m callqa --check --backend offline
```
Offline demo story is 70% for v1 and 95% for v2, with no API key needed. On `claude-cli`,
`python -m callqa --check` can make about 90 calls (about 45 per agent version).

## Plug in your own agent
Any function `respond(history) -> str` works, where `history` is a list of `{"role": "user" | "assistant", "content": str}`.
```bash
python demos/call-qa/evals/run_evals.py --agent your_package.agent:respond --runs 3 --threshold 0.85
```
Personas live in `demos/call-qa/evals/personas.yaml`. Each one has a pass rule a stranger could apply, plus phrases the agent must or must never say.

## Tests
```bash
python -m pytest -q demos/call-qa/tests
```
