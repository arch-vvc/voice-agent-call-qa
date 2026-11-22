"""`python -m callqa eval`: one agent, one caller set, k repeats. Prints the EVAL block, saves a run record and
exits 0, 1 (below --threshold on pass^k) or 2 (bad arguments, backend trouble, a call count that needs --yes)."""
from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import sys
import time
from datetime import datetime
from pathlib import Path

from callqa.compliance import DEFAULT_RULES_PATH, RulesError, load_rules
from callqa.persona_sets import SET_NAMES, personas_for_set
from callqa.report import build_report
from callqa.runrecord import save_record
from callqa.server import (
    DEFAULT_AGENTS,
    LARGE_RUN_CALLS,
    ROOT,
    estimate_calls,
    metered_agent,
    needs_confirmation,
    read_sample_sop,
)
from callqa.sop import parse_sop
from evals.backends import BACKEND_NAMES, BackendError, get_backend
from evals.run_evals import RUN_DEFAULTS, RunAborted, run_suite

MAX_REPEATS = 10
DEFAULT_BUDGET_MS = 1500


def pct(x: float) -> str:
    return format(x, ".0%")


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m callqa eval", description=__doc__)
    p.add_argument("--agent", required=True, help=f"one of {', '.join(a['id'] for a in DEFAULT_AGENTS)}")
    p.add_argument("--set", dest="set_name", default="story", choices=SET_NAMES, help="caller set (default story)")
    p.add_argument("--callers", type=int, help="story set only: how many of the 20 callers (default all 20)")
    p.add_argument("--repeats", type=int, default=1, help=f"runs per caller, 1 to {MAX_REPEATS} (default 1)")
    p.add_argument("--seed", type=int, default=0, help="default 0")
    p.add_argument("--backend", choices=BACKEND_NAMES, help="default: $CALLQA_BACKEND, else claude-cli")
    p.add_argument("--max-turns", type=int, help="turns per call (default 4 on LLM backends, 8 offline)")
    p.add_argument("--workers", type=int, help="calls in parallel (default 2 for claude-cli, 4 otherwise)")
    p.add_argument("--latency-budget-ms", type=float, default=DEFAULT_BUDGET_MS,
                   help=f"an agent turn slower than this is over budget (default {DEFAULT_BUDGET_MS})")
    p.add_argument("--threshold", type=float, help="0 to 1; exit 1 if pass^k (the pass rate at 1 repeat) is below it")
    p.add_argument("--out", type=Path, help="a folder for the run record (default runs/ under the demo), or a path "
                   "ending in .json to name the record file (never overwritten)")
    p.add_argument("--rules", metavar="PATH", help="compliance rules file, or none to turn the layer off "
                   f"(default {DEFAULT_RULES_PATH.relative_to(DEFAULT_RULES_PATH.parent.parent)})")
    p.add_argument("--created-at", metavar="ISO", help="stamp the record with this time (for example "
                   "2026-11-05T00:00:00+05:30) and make run_id and seconds fixed, so the same flags give a byte "
                   "identical record; used for the committed sample records")
    p.add_argument("--no-save", action="store_true", help="do not write a run record")
    p.add_argument("--yes", action="store_true", help=f"allow an LLM backend run of more than {LARGE_RUN_CALLS} calls")
    args = p.parse_args(argv)
    if args.agent not in {a["id"] for a in DEFAULT_AGENTS}:
        p.error(f"unknown agent {args.agent!r}, choose one of {', '.join(a['id'] for a in DEFAULT_AGENTS)}")
    if not 1 <= args.repeats <= MAX_REPEATS:
        p.error(f"--repeats must be from 1 to {MAX_REPEATS}")
    for flag in ("callers", "max_turns", "workers"):
        value = getattr(args, flag)
        if value is not None and value < 1:
            p.error(f"--{flag.replace('_', '-')} must be at least 1")
    if args.callers is not None and args.set_name != "story":
        p.error("--callers only works with --set story; the full set is always all of its callers")
    if not args.latency_budget_ms > 0:
        p.error("--latency-budget-ms must be more than 0")
    if args.created_at is not None:
        try:
            when = datetime.fromisoformat(args.created_at)
        except ValueError:
            p.error(f"--created-at must be an ISO time such as 2026-11-05T00:00:00+05:30, got {args.created_at!r}")
        if when.tzinfo is None:
            p.error("--created-at needs a time zone offset, for example +05:30")
    if args.threshold is not None and not 0 <= args.threshold <= 1:
        p.error("--threshold must be between 0 and 1, for example 0.9")
    return args


def eval_lines(record: dict) -> list[str]:
    """Lines 1 to 3 of the EVAL block."""
    k, total = record["repeats"], record["total"]
    runs_passed = sum(c["repeats_passed"] for c in record["calls"])
    runs = sum(c["repeats_total"] for c in record["calls"])
    flaky = (record["consistency"] or {}).get("flaky", [])
    if k == 1:
        flaky_line = "flaky: n/a (repeats 1)"
    elif flaky:  # repeats are 0 based in the record and 1 based on screen
        found = "; ".join(f"{f['persona_id']} {f['passed']}/{f['total']} repeats passed, "
                          f"failed repeat {f['failed_repeat'] + 1}" for f in flaky)
        flaky_line = f"flaky ({len(flaky)}): {found}"
    else:
        flaky_line = "flaky: none"
    return [
        f"EVAL agent={record['agent']['id']} set={record['persona_set']['name']} callers={total} repeats={k} "
        f"seed={record['seed']} backend={record['backend']['name']}",
        f"pass^{k} {pct(record['passed'] / total)} ({record['passed']}/{total})  "
        f"pass@1 {pct(runs_passed / runs)} ({runs_passed}/{runs} runs)",
        flaky_line,
    ]


def metering_lines(record: dict) -> list[str]:
    """The tokens, cost and latency lines (frozen in tests/golden/ for the story set)."""
    cost, latency = record["cost"], record["latency"]
    roles = cost["by_role"]
    tokens = " ".join(f"{role} {roles[role]['input_tokens']}/{roles[role]['output_tokens']}"
                      for role in ("caller", "agent", "judge"))
    if cost["total"] is None:
        spend = f"cost: unknown, no price for {', '.join(cost['unknown_models'])}"
    else:
        spend = (f"cost ({cost['label']}) ${cost['per_call']:.4f} per call, ${cost['total']:.4f} total at "
                 f"config/pricing.yaml prices of {cost['prices']['retrieved_on']}, verify before quoting")
    return [
        f"tokens ({cost['label']}) {tokens}",
        spend,
        f"latency ({latency['label']}) P50 {latency['p50_ms']:.0f} ms  P95 {latency['p95_ms']:.0f} ms  "
        f"budget {latency['budget_ms']:g} ms: {latency['over_budget']} of {latency['turns']} agent turns over",
    ]


def escalation_line(record: dict) -> str:
    """Accuracy is shown from the counts, so rounding in the record cannot change the line."""
    block = record["escalation"]
    m = block["matrix"]
    right, runs = m["tp"] + m["tn"], block["runs"]
    avg = "n/a" if block["handoff_avg"] is None else f"{block['handoff_avg']:.1f}/5"
    return (f"escalation accuracy {pct(right / runs if runs else 0.0)} ({right}/{runs} runs)  "
            f"TP {m['tp']} FP {m['fp']} FN {m['fn']} TN {m['tn']}  handoff avg {avg}")


def compliance_line(record: dict) -> str:
    """`compliance <rules id>: <n> critical (<rule id>: <ids>); not evaluated offline: <ids>`, or `compliance: off`.
    n counts critical rule failures; the parentheses are left out at 0 and the last part when every rule ran."""
    block = record.get("compliance")
    if not block:
        return "compliance: off"
    critical = [f for f in block["failures"] if f["severity"] == "critical"]
    line = f"compliance {block['rules_id']}: {sum(f['count'] for f in critical)} critical"
    if critical:
        line += " (" + "; ".join(f"{f['rule_id']}: {', '.join(sorted(c['persona_id'] for c in f['calls']))}"
                                 for f in critical) + ")"
    if block["not_evaluated"]:
        line += "; not evaluated offline: " + ", ".join(n["rule_id"] for n in block["not_evaluated"])
    return line


def traits_line(record: dict) -> str | None:
    """`traits <trait> <pct> (<p>/<n>), ...` for the traits the run's callers carry; None when none do (the five
    stratified story callers have no trait)."""
    block = record.get("traits") or {}
    if not block:
        return None
    return "traits " + ", ".join(f"{trait} {pct(t['passed'] / t['total'])} ({t['passed']}/{t['total']})"
                                 for trait, t in block.items())


def score(record: dict) -> float:
    """What --threshold is compared against: pass^k with repeats, else the pass rate."""
    return record["pass_rate"]


def write_record(record: dict, out: Path | None) -> Path:
    """Save to the default folder, to the folder `out`, or (a path ending in .json) to exactly that new file."""
    if out is None or out.suffix != ".json":
        return save_record(record, out or ROOT / "runs")
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(record, indent=2, ensure_ascii=False)
    try:
        with out.open("x", encoding="utf-8") as handle:  # "x" never overwrites
            handle.write(text)
    except FileExistsError:
        raise OSError(f"{out} already exists, not overwriting it") from None
    return out


def pin_record(record: dict, args: argparse.Namespace) -> None:
    """--created-at: the only three things that differ between two offline runs (time, random run id, wall clock
    seconds) are set from the flags, so the same flags write the same bytes."""
    record["created_at"] = args.created_at
    key = f"{record['agent']['id']}|{args.set_name}|{args.seed}|{args.repeats}|{args.created_at}"
    record["run_id"] = hashlib.sha256(key.encode("utf-8")).hexdigest()[:8]
    record["seconds"] = 0.0


def run_eval(args: argparse.Namespace) -> int:
    try:
        backend = get_backend(args.backend)
    except BackendError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for warning in backend.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    rules = None
    if (args.rules or "").lower() != "none":
        try:
            rules = load_rules(args.rules or DEFAULT_RULES_PATH)
        except RulesError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    defaults = RUN_DEFAULTS[backend.name]
    max_turns, workers = args.max_turns or defaults["max_turns"], args.workers or defaults["workers"]
    sop_text = read_sample_sop()
    generation = personas_for_set(sop_text, args.set_name, args.callers)
    callers = len(generation.personas)
    estimate = estimate_calls(callers, max_turns, args.repeats)
    if needs_confirmation(backend, estimate) and not args.yes:
        print(f"error: this run would make about {estimate} model calls on {backend.name} (limit {LARGE_RUN_CALLS}); "
              "add --yes to run it, or use --backend offline", file=sys.stderr)
        return 2
    agent = next(a for a in DEFAULT_AGENTS if a["id"] == args.agent)
    started = time.time()
    try:
        respond, prompt = metered_agent(agent, sop_text, backend)
        results = run_suite(respond, generation.personas, args.repeats, max_turns, workers, backend, args.seed,
                            agent_prompt=prompt, rules=rules, sop=parse_sop(sop_text))
    except RunAborted as exc:
        print(f"ABORTED: {exc.message}")
        return 2
    except BackendError as exc:
        print(f"ABORTED: {exc}")
        return 2
    record = build_report(run_id=secrets.token_hex(4), sop_text=sop_text, agent=agent, backend=backend,
                          callers=callers, max_turns=max_turns, generation=generation, results=results,
                          seconds=time.time() - started, repeats=args.repeats, seed=args.seed, set_name=args.set_name,
                          synthetic=generation.source.startswith("templates"), metering=True,
                          budget_ms=args.latency_budget_ms, rules=rules)
    if args.created_at:
        pin_record(record, args)
    lines = eval_lines(record) + metering_lines(record) + [escalation_line(record), compliance_line(record)]
    if traits_line(record):
        lines.append(traits_line(record))
    code = 0
    if args.threshold is not None:
        ok = score(record) >= args.threshold
        lines.append(f"threshold {pct(args.threshold)} on pass^{args.repeats}: {'PASS' if ok else 'FAIL'}")
        code = 0 if ok else 1
    if not args.no_save:
        try:
            path = write_record(record, args.out)
        except OSError as exc:
            print("\n".join(lines))
            print(f"error: could not save the run record: {exc}", file=sys.stderr)
            return 2
        lines.append(f"record: {path}")
    print("\n".join(lines))
    return code


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(sys.argv[1:] if argv is None else argv)
    except SystemExit as exc:  # argparse: 2 for bad arguments, 0 for --help
        return exc.code if isinstance(exc.code, int) else 2
    return run_eval(args)
