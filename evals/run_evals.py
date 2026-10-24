#!/usr/bin/env python3
"""Multi turn agent eval harness for FDE demos.

A simulated customer (an LLM playing a persona) talks to your agent for several
turns, then a judge model checks the transcript against the persona's
expectations. Each persona runs several times because agents are stochastic, and
the script exits non zero when the pass rate is below --threshold, so it can be
the verification step for /goal, a Stop hook or CI.

Your agent is any function `respond(history) -> str`, where history is a list of
{"role": "user" | "assistant", "content": str} messages, newest last.

  python demos/call-qa/evals/run_evals.py --agent evals.example_agent:respond --runs 3
  python demos/call-qa/evals/run_evals.py --agent app.agent:respond --offline   # no API key needed

Works from any directory. Personas and results live next to this file, and the call QA folder
(demos/call-qa) is put on sys.path so `evals.example_agent` resolves.

Online mode needs ANTHROPIC_API_KEY and `pip install anthropic pyyaml`.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    sys.exit("pip install pyyaml")

HERE = Path(__file__).resolve().parent
PRODUCT_DIR = HERE.parent

SIM_MODEL = os.environ.get("EVAL_SIM_MODEL", "claude-haiku-5-5")
JUDGE_MODEL = os.environ.get("EVAL_JUDGE_MODEL", "claude-sonnet-5-5")
END = "<END_CALL>"


@dataclass
class RunResult:
    persona: str
    run: int
    passed: bool
    reason: str
    transcript: list[dict] = field(default_factory=list)
    seconds: float = 0.0


def load_agent(spec: str):
    module_name, _, func_name = spec.partition(":")
    if not func_name:
        sys.exit("--agent must look like package.module:function")
    sys.path.insert(0, os.getcwd())
    sys.path.insert(0, str(PRODUCT_DIR))
    return getattr(importlib.import_module(module_name), func_name)


def _client():
    try:
        import anthropic
    except ImportError:
        sys.exit("pip install anthropic (or use --offline)")
    return anthropic.Anthropic()


def _text(resp) -> str:
    return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()


def simulate_user_turn(client, persona: dict, history: list[dict]) -> str:
    """Ask the simulator model for the customer's next line. Roles are flipped
    because, to the simulator, the agent is the other speaker."""
    flipped = [{"role": "assistant" if m["role"] == "user" else "user", "content": m["content"]} for m in history]
    system = (
        "You are role playing a customer calling a company's AI agent. Stay in character.\n"
        f"PERSONA:\n{persona['persona']}\n\n"
        "Speak like a real person on the phone: short turns, one thing at a time, natural fillers. "
        f"When your goal is met or you would hang up, reply with only {END}."
    )
    resp = client.messages.create(model=SIM_MODEL, max_tokens=300, system=system, messages=flipped)
    return _text(resp) or END


def judge(client, persona: dict, transcript: list[dict]) -> tuple[bool, str]:
    convo = "\n".join(f"{'CUSTOMER' if m['role'] == 'user' else 'AGENT'}: {m['content']}" for m in transcript)
    prompt = (
        "You grade a conversation between a customer and an AI agent.\n"
        f"Pass rule:\n{persona['expectations']}\n\n"
        f"Transcript:\n{convo}\n\n"
        'Reply with JSON only: {"pass": true or false, "reason": "<one sentence citing the turn>"}'
    )
    resp = client.messages.create(model=JUDGE_MODEL, max_tokens=300, messages=[{"role": "user", "content": prompt}])
    raw = _text(resp)
    try:
        data = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
        return bool(data["pass"]), str(data.get("reason", ""))
    except (ValueError, KeyError):
        return False, f"judge returned unparseable output: {raw[:200]}"


def keyword_checks(persona: dict, transcript: list[dict]) -> tuple[bool, str]:
    """Deterministic rules that run in both modes. They catch hard failures
    (leaking data, missing a required disclosure) without trusting a judge."""
    agent_text = " ".join(m["content"].lower() for m in transcript if m["role"] == "assistant")
    for needle in persona.get("must_include", []):
        if needle.lower() not in agent_text:
            return False, f"agent never said required phrase: {needle!r}"
    for needle in persona.get("must_not_include", []):
        if needle.lower() in agent_text:
            return False, f"agent said forbidden phrase: {needle!r}"
    return True, "keyword rules passed"


def run_one(agent, persona: dict, run: int, max_turns: int, offline: bool, seed: int) -> RunResult:
    random.seed(seed + run)
    start = time.time()
    client = None if offline else _client()
    history: list[dict] = [{"role": "user", "content": persona["opening"]}]
    scripted = list(persona.get("script", []))
    try:
        for _ in range(max_turns):
            history.append({"role": "assistant", "content": str(agent(list(history)))})
            if offline:
                nxt = scripted.pop(0) if scripted else END
            else:
                nxt = simulate_user_turn(client, persona, history)
            if END in nxt:
                break
            history.append({"role": "user", "content": nxt})
    except Exception as exc:  # the agent crashing is a failed run, not a crashed harness
        reason = f"agent raised {type(exc).__name__}: {exc}"
        return RunResult(persona["id"], run, False, reason, history, round(time.time() - start, 2))

    ok, reason = keyword_checks(persona, history)
    if ok and not offline:
        ok, reason = judge(client, persona, history)
    return RunResult(persona["id"], run, ok, reason, history, round(time.time() - start, 2))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agent", required=True, help="module:function, e.g. app.agent:respond")
    p.add_argument("--personas", default=str(HERE / "personas.yaml"))
    p.add_argument("--runs", type=int, default=3, help="runs per persona")
    p.add_argument("--threshold", type=float, default=0.8, help="minimum overall pass rate")
    p.add_argument("--max-turns", type=int, default=8)
    p.add_argument("--workers", type=int, default=4, help="conversations run in parallel")
    p.add_argument("--only", help="comma separated persona ids")
    p.add_argument("--offline", action="store_true", help="replay persona scripts and use keyword rules only")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    agent = load_agent(args.agent)
    personas = yaml.safe_load(Path(args.personas).read_text(encoding="utf-8"))
    if args.only:
        wanted = set(args.only.split(","))
        personas = [x for x in personas if x["id"] in wanted]
    jobs = [(pp, r) for pp in personas for r in range(args.runs)]

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda j: run_one(agent, j[0], j[1], args.max_turns, args.offline, args.seed), jobs))

    print(f"\n{'persona':<24}{'pass rate':>10}   failures")
    for pp in personas:
        rs = [r for r in results if r.persona == pp["id"]]
        rate = sum(r.passed for r in rs) / len(rs)
        fails = "; ".join(sorted({r.reason for r in rs if not r.passed}))[:110]
        print(f"{pp['id']:<24}{rate:>9.0%}   {fails}")
    overall = sum(r.passed for r in results) / len(results)
    print(f"\nOVERALL {overall:.0%} over {len(results)} conversations (threshold {args.threshold:.0%})")

    out_dir = HERE / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"run-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps([asdict(r) for r in results], indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"transcripts: {out}")
    return 0 if overall >= args.threshold else 1


if __name__ == "__main__":
    sys.exit(main())
