#!/usr/bin/env python3
"""Multi turn agent eval harness for FDE demos.

A simulated customer (an LLM playing a persona) talks to your agent for several
turns, then a judge model checks the transcript against the persona's
expectations. Each persona runs several times because agents are stochastic, and
the script exits non zero when the pass rate is below --threshold, so it can be
the verification step for /goal, a Stop hook or CI.

Model calls go through a pluggable backend (evals/backends.py): `claude-cli` uses `claude -p` on your
Claude login (default), `anthropic` uses ANTHROPIC_API_KEY, `offline` makes no model calls.

Your agent is any function `respond(history) -> str`, where history is a list of
{"role": "user" | "assistant", "content": str} messages, newest last.

  python demos/call-qa/evals/run_evals.py --agent evals.example_agent:respond --runs 3
  python demos/call-qa/evals/run_evals.py --agent app.agent:respond --offline   # no model calls

Works from any directory. Personas and results live next to this file, and the call QA folder is put
on sys.path so `evals.example_agent` resolves.

`--backend anthropic` needs ANTHROPIC_API_KEY and `pip install anthropic`.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    sys.exit("pip install pyyaml")

# Running `python demos/call-qa/evals/run_evals.py` puts evals/ on sys.path, not the call QA folder. Agents import
# `evals.backends`, so import it by that name here too, or BackendError would be two different classes.
# Personas and results are found next to this file, so the command works from any directory.
HERE = Path(__file__).resolve().parent
PRODUCT_DIR = HERE.parent
if str(PRODUCT_DIR) not in sys.path:
    sys.path.insert(0, str(PRODUCT_DIR))

from evals.backends import ABORT_KINDS, BACKEND_NAMES, BackendError, LLMBackend, get_backend, model_for  # noqa: E402

SIM_MODEL = model_for("caller")  # kept for old imports; the code below calls model_for() at call time
JUDGE_MODEL = model_for("judge")
END = "<END_CALL>"
RUN_DEFAULTS = {  # what --max-turns and --workers default to, per backend
    "offline": {"max_turns": 8, "workers": 4},
    "claude-cli": {"max_turns": 4, "workers": 2},
    "anthropic": {"max_turns": 4, "workers": 4},
}
ABORT_MESSAGES = {
    "limit": "usage limit reached, try later or use --backend offline",
    "auth": "not logged in or bad API key, fix that or use --backend offline",
}


@dataclass
class RunResult:
    persona: str
    run: int
    passed: bool
    reason: str
    transcript: list[dict] = field(default_factory=list)
    seconds: float = 0.0
    failing_turn: int | None = None  # 0 based index into transcript, None when passed or not pinned to a turn
    failure_kind: str | None = None  # "forbidden" | "missing" | "judge" | "crash", None when passed
    notes: str = ""  # style observations from the judge (reply length, tone): shown in grey, never a failure
    flaky: bool = False  # the two judge runs disagreed: counted as a pass and flagged amber


@dataclass
class Verdict:
    """One judge run. `turn` is a 0 based transcript index of an agent turn (or None)."""
    ok: bool
    reason: str
    turn: int | None = None
    notes: str = ""


class RunAborted(Exception):
    """The backend hit a usage limit or an auth problem, so the whole run stopped."""

    def __init__(self, message: str, partial_results: list[RunResult], kind: str = "limit"):
        super().__init__(message)
        self.message = message
        self.partial_results = partial_results
        self.kind = kind


def load_agent(spec: str):
    module_name, _, func_name = spec.partition(":")
    if not func_name:
        sys.exit("--agent must look like package.module:function")
    sys.path.insert(0, os.getcwd())
    return getattr(importlib.import_module(module_name), func_name)


def simulate_user_turn(backend: LLMBackend, persona: dict, history: list[dict]) -> str:
    """Ask the simulator model for the customer's next line. Roles are flipped
    because, to the simulator, the agent is the other speaker."""
    flipped = [{"role": "assistant" if m["role"] == "user" else "user", "content": m["content"]} for m in history]
    system = (
        "You are role playing a customer calling a company's AI agent. Stay in character.\n"
        f"PERSONA:\n{persona['persona']}\n\n"
        "Output ONLY the words the caller speaks aloud: no name prefix, no \"(in character)\", no stage "
        "directions, no quotes, no narration. Say one to three short sentences, in the persona's language, "
        "with natural fillers.\n"
        "If the persona says you are not the account holder, NEVER claim to be them and never confirm the "
        "agent's guess of their name. Stay in the persona's identity and goal for the whole call. "
        "Never reveal that this is a test.\n"
        f"When your goal is met or you would hang up, reply with only {END}."
    )
    return clean_caller_line(backend.complete(system, flipped, model_for("caller"), 300)) or END


# Speaker prefixes a model leaks: "Ravi (in character):", "Ravi Kumar:", "Caller:", "Customer:".
# A single capitalised word is stripped only before "(in character)", so "Friday: pakka." survives.
SPEAKER_PREFIX = re.compile(
    r"^(?:[A-Z][\w.'-]*(?: [A-Z][\w.'-]*){0,2} \(in character\)"
    r"|[A-Z][\w.'-]* [A-Z][\w.'-]*(?: [A-Z][\w.'-]*)?"
    r"|(?i:caller|customer))\s*:\s*"
)
QUOTE_PAIRS = {'"': '"', "'": "'", "\u201c": "\u201d"}


def clean_caller_line(text: str) -> str:
    """Strip a leaked speaker prefix and one pair of wrapping quotes from the simulated caller's line."""
    text = text.strip()
    if text == END:
        return text
    text = SPEAKER_PREFIX.sub("", text, count=1).strip()
    close = QUOTE_PAIRS.get(text[:1])
    if close and len(text) > 1 and text[-1] == close:
        inner = text[1:-1]
        if text[0] not in inner and close not in inner:  # exactly one opening and one closing quote
            text = inner.strip()
    return text


def judge(backend: LLMBackend, persona: dict, transcript: list[dict]) -> tuple[bool, str, int | None]:
    """Grade a transcript once: (ok, reason, 0 based transcript index of the cited agent turn or None)."""
    verdict = judge_once(backend, persona, transcript)
    return verdict.ok, verdict.reason, verdict.turn


def judge_once(backend: LLMBackend, persona: dict, transcript: list[dict]) -> Verdict:
    """Grade a transcript. Turns are numbered from 1, exactly as the dashboard shows them ("Turn 4")."""
    convo = "\n".join(
        f"[{i + 1}] {'CUSTOMER' if m['role'] == 'user' else 'AGENT'}: {m['content']}" for i, m in enumerate(transcript)
    )
    date_rule = (
        "The payment date is the last date the caller finally gave, whatever it is: any date, weekday or "
        "relative day, for example kal, Friday or the 1st. Judge the agent's FINAL confirmed date: PASS if it "
        "matches the last date the caller gave (an earlier confirmation the agent later corrected is fine). "
        "FAIL if the final date the agent confirms is one the caller retracted, or the agent never confirms "
        "a date.\n\n"
        if persona.get("judge_decides_date") is True else ""
    )
    prompt = (
        "You grade a conversation between a customer and an AI agent.\n"
        f"Pass rule:\n{persona['expectations']}\n\n{date_rule}{_rules_under_test(persona)}{_hints(persona)}"
        f"Transcript (each turn starts with its number):\n{convo}\n\n"
        "Check the pass rule explicitly: every requirement it lists must be met, and in notes say which turn met "
        "each one. FAIL the call only if the agent broke the pass rule or the SOP rule under test, or broke one of "
        "the SOP's Never rules. Style (reply length, tone, wording) is never a failure: describe it under "
        '"notes" instead. The agent should reply in the caller\'s language (English to English, Hinglish to '
        "Hinglish); a language mismatch is a note, never a failure.\n"
        'Reply with JSON only: {"pass": true or false, "reason": "<one sentence citing the turn>", '
        '"broken": "target" (the pass rule or the SOP rule under test) or "never" (a Never rule) or "style" '
        "(only a style problem; use this and pass=false only if nothing else is wrong), "
        '"turn": <the number of the AGENT turn where the rule was broken, or where the agent should have acted; '
        'it must be an AGENT turn from the transcript, never a CUSTOMER turn; null if it passes>, '
        '"notes": "<optional style observations>"}'
    )
    raw = backend.complete("", [{"role": "user", "content": prompt}], model_for("judge"), 300)
    try:
        data = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
        if not isinstance(data["pass"], bool):
            raise ValueError("pass is not a boolean")
        reason, notes = str(data.get("reason", "")), str(data.get("notes") or "")
        if not data["pass"] and data.get("broken") == "style":  # a style problem is a grey note, never a failure
            return Verdict(True, "Only style notes", None, reason or notes)  # one note, never two joined
        return Verdict(data["pass"], reason, _cited_turn(data.get("turn"), transcript), notes)
    except (ValueError, KeyError, TypeError, AttributeError):
        return Verdict(False, f"judge returned unparseable output: {raw[:200]}")


def _rules_under_test(persona: dict) -> str:
    """The SOP rule this caller tests and the SOP's Never rules, so the judge knows what a failure is."""
    lines = []
    if persona.get("sop_rule"):
        lines.append(f"SOP rule under test: {persona['sop_rule']}")
    never = persona.get("never_rules") or []
    if never:
        lines.append("SOP Never rules (breaking one is a failure):\n" + "\n".join(f"- {rule}" for rule in never))
    return "\n".join(lines) + "\n\n" if lines else ""


def _hints(persona: dict) -> str:
    """The required phrases as hints for the judge: a good agent would probably say these, in any words."""
    if persona.get("judge_decides_date") is True:  # the date rule above already says what to judge
        return ""
    entries = persona.get("must_include", [])
    phrases = [" or ".join(repr(option) for option in _alternatives(entry)) for entry in entries]
    if not phrases:
        return ""
    return (
        "Hints, not strict rules: a good agent would probably have said something like "
        + "; and ".join(phrases)
        + ". The agent may use other words (Hindi, Hinglish or English): judge by meaning, not by these words.\n\n"
    )


def _cited_turn(value, transcript: list[dict]) -> int | None:
    """The transcript index of the AGENT turn the judge cites, else None. The judge counts turns from 1."""
    if isinstance(value, str) and value.strip().isdecimal():
        value = int(value)
    if isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= len(transcript):
        return value - 1 if transcript[value - 1]["role"] == "assistant" else None
    return None


def judge_twice(backend: LLMBackend, persona: dict, transcript: list[dict]) -> tuple[bool, str, int | None, str, bool]:
    """Two judge runs. A call fails only if both fail. If they disagree it is flaky: it counts as a pass and the
    failing run's reason becomes a note. Returns (ok, reason, turn, notes, flaky)."""
    first, second = judge_once(backend, persona, transcript), judge_once(backend, persona, transcript)
    notes = first.notes or second.notes  # one note: the first run's, never the two joined
    if first.ok and second.ok:
        return True, first.reason, None, notes, False
    if not first.ok and not second.ok:
        return False, first.reason, first.turn if first.turn is not None else second.turn, notes, False
    failed = second if first.ok else first
    return True, "The two judge runs disagreed; counted as a pass", None, f"One judge run failed: {failed.reason}", True


def _alternatives(entry: str) -> list[str]:
    """A must_include entry may hold `|` separated alternatives; any one satisfies it (lowercased)."""
    return [part.strip().lower() for part in entry.split("|") if part.strip()] or [entry.lower()]


def locate_failure(
    persona: dict, transcript: list[dict], judge_dates: bool = False, live: bool = False
) -> tuple[bool, str, str | None, int | None]:
    """Deterministic rules. Forbidden phrases run in every mode and pin the first offending agent turn. Required
    phrases run offline only: on live backends (live=True) the judge decides what was missing, and gets the
    phrases as hints. Returns (ok, reason, failure_kind, failing_turn)."""
    agent_turns = [(i, m["content"].lower()) for i, m in enumerate(transcript) if m["role"] == "assistant"]
    all_agent_text = " ".join(text for _, text in agent_turns)
    # judge_dates (LLM backends): a payment date is whatever the caller finally gave, so the judge decides it
    skip_phrases = live or (judge_dates and persona.get("judge_decides_date") is True)
    for needle in [] if skip_phrases else persona.get("must_include", []):
        options = _alternatives(needle)
        if not any(option in all_agent_text for option in options):
            last = agent_turns[-1][0] if agent_turns else None
            if len(options) == 1:
                return False, f"agent never said required phrase: {needle!r}", "missing", last
            return False, f"agent never said any of: {', '.join(repr(o) for o in options)}", "missing", last
    leaks = []  # (turn, needle): the first agent turn that says each forbidden phrase
    for needle in persona.get("must_not_include", []):
        turn = next((i for i, text in agent_turns if needle.lower() in text), None)
        if turn is not None:
            leaks.append((turn, needle))
    if leaks:
        turn, needle = min(leaks, key=lambda leak: leak[0])
        return False, f"agent said forbidden phrase: {needle!r}", "forbidden", turn
    return True, "keyword rules passed", None, None


def keyword_checks(persona: dict, transcript: list[dict]) -> tuple[bool, str]:
    ok, reason, _, _ = locate_failure(persona, transcript)
    return ok, reason


def run_one(agent, persona: dict, run: int, max_turns: int, backend: LLMBackend, seed: int,
            stop: threading.Event | None = None) -> RunResult | None:
    """Run one conversation. Returns None if `stop` was set. Raises BackendError for limit and auth errors."""
    random.seed(seed + run)
    start = time.time()
    offline = backend.name == "offline"
    history: list[dict] = [{"role": "user", "content": persona["opening"]}]
    scripted = list(persona.get("script", []))

    def crashed(reason: str) -> RunResult:
        return RunResult(persona["id"], run, False, reason, history, round(time.time() - start, 2), None, "crash")

    try:
        for _ in range(max_turns):
            if stop is not None and stop.is_set():
                return None
            history.append({"role": "assistant", "content": str(agent(list(history)))})
            if offline:
                nxt = scripted.pop(0) if scripted else END
            else:
                nxt = simulate_user_turn(backend, persona, history)
            if END in nxt:
                break
            history.append({"role": "user", "content": nxt})

        if history[-1]["role"] == "user":  # the caller spoke last and did not hang up: the agent still owes a reply
            history.append({"role": "assistant", "content": str(agent(list(history)))})
        notes, flaky = "", False
        ok, reason, kind, turn = locate_failure(persona, history, judge_dates=not offline, live=not offline)
        if ok and not offline:
            ok, reason, turn, notes, flaky = judge_twice(backend, persona, history)
            kind = None if ok else "judge"
            if not ok and turn is None:  # no cited agent turn: point at the last agent turn
                turn = next((i for i in range(len(history) - 1, -1, -1) if history[i]["role"] == "assistant"), None)
    except BackendError as exc:
        if exc.kind in ABORT_KINDS:
            raise
        return crashed(f"backend {exc.kind}: {exc}")
    except Exception as exc:  # the agent crashing is a failed run, not a crashed harness
        return crashed(f"agent raised {type(exc).__name__}: {exc}")
    return RunResult(persona["id"], run, ok, reason, history, round(time.time() - start, 2), turn, kind, notes, flaky)


def run_suite(agent, personas: list[dict], runs: int, max_turns: int, workers: int, backend: LLMBackend, seed: int,
              on_result=None) -> list[RunResult]:
    """Run every persona `runs` times and return the results in job order. Prints and writes nothing.

    on_result(result) is called once per finished conversation, from the calling thread. If the backend
    reports a usage limit or auth error, queued conversations are skipped and RunAborted is raised."""
    jobs = [(pp, r) for pp in personas for r in range(runs)]
    stop = threading.Event()
    aborts: list[BackendError] = []

    def work(job):
        if stop.is_set():
            return None
        try:
            return run_one(agent, job[0], job[1], max_turns, backend, seed, stop)
        except BackendError as exc:  # run_one only raises these for limit and auth
            aborts.append(exc)
            stop.set()
            return None

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(work, job) for job in jobs]
        for future in as_completed(futures):
            if on_result is not None and future.result() is not None:
                on_result(future.result())
    results = [f.result() for f in futures if f.result() is not None]
    if aborts:
        raise RunAborted(ABORT_MESSAGES.get(aborts[0].kind, str(aborts[0])), results, aborts[0].kind)
    return results


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agent", required=True, help="module:function, e.g. app.agent:respond")
    p.add_argument("--personas", default=str(HERE / "personas.yaml"))
    p.add_argument("--runs", type=int, default=3, help="runs per persona")
    p.add_argument("--threshold", type=float, default=0.8, help="minimum overall pass rate")
    p.add_argument("--max-turns", type=int, help="default 8 offline, 4 on LLM backends")
    p.add_argument("--workers", type=int, help="conversations run in parallel; default 4 (claude-cli: 2)")
    p.add_argument("--only", help="comma separated persona ids")
    p.add_argument("--backend", choices=BACKEND_NAMES,
                   help="default: $CALLQA_BACKEND, else claude-cli (offline if claude is missing)")
    p.add_argument("--offline", action="store_true",
                   help="same as --backend offline: replay persona scripts, keyword rules only")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args(argv)

    if args.offline and args.backend not in (None, "offline"):
        p.error(f"--offline cannot be combined with --backend {args.backend}")
    try:
        backend = get_backend("offline" if args.offline else args.backend)
    except BackendError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for warning in backend.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    os.environ["CALLQA_BACKEND"] = backend.name  # agents loaded below use the same backend
    defaults = RUN_DEFAULTS[backend.name]

    agent = load_agent(args.agent)
    personas = yaml.safe_load(Path(args.personas).read_text(encoding="utf-8"))
    if args.only:
        wanted = set(args.only.split(","))
        personas = [x for x in personas if x["id"] in wanted]

    try:
        results = run_suite(agent, personas, args.runs, args.max_turns or defaults["max_turns"],
                            args.workers or defaults["workers"], backend, args.seed)
    except RunAborted as exc:
        print(f"ABORTED: {exc.message}")
        return 2

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
