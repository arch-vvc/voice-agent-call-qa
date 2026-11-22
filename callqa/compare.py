"""`python -m callqa compare`: a bake off over agent configs on the same callers and seeds.

A config is a prompt version, optionally on a model: `v2` or `v2@haiku`. Every config runs the same callers with the
same seed and repeats, so the rows differ only in the agent. The command saves one run record per config plus a
`kind: "compare"` record, prints the table, the best config per column, the callers the configs disagree on and a
recommendation by one fixed rule. Exits 0, or 2 for bad arguments, backend trouble or a call count that needs --yes.
Offline, agents are rule based, so the numbers are simulated and say nothing about real model quality."""
from __future__ import annotations

import argparse
import re
import secrets
import sys
import time
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import NamedTuple

from callqa.compliance import DEFAULT_RULES_PATH, RulesError, load_rules
from callqa.evaluate import DEFAULT_BUDGET_MS, MAX_REPEATS, pct, write_record
from callqa.metering import MeteredBackend, cost_block, load_prices
from callqa.persona_sets import SET_NAMES, personas_for_set
from callqa.report import build_report
from callqa.runrecord import RECORD_SCHEMA_VERSION, save_record
from callqa.server import (
    DEFAULT_AGENTS,
    LARGE_RUN_CALLS,
    ROOT,
    agent_prompt_for,
    estimate_calls,
    needs_confirmation,
    read_sample_sop,
)
from callqa.sop import parse_sop
from evals.backends import BACKEND_NAMES, BackendError, get_backend
from evals.llm_agent import build_agent
from evals.run_evals import RUN_DEFAULTS, RunAborted, run_suite

COMPARE_KIND = "compare"
DEFAULT_LLM_AGENTS = "v2@haiku,v2@sonnet"
DEFAULT_OFFLINE_AGENTS = "v1,v2,v3"
RECOMMEND_RULE_TEMPLATE = "highest pass^{k} with zero critical compliance failures; ties broken by lower cost per call"
RECOMMEND_RULE = RECOMMEND_RULE_TEMPLATE.format(k=1)
OFFLINE_NOTE = "offline agents are rule based, so models are not compared; comparing prompt versions"
SIMULATED_LINE = ("simulated: offline numbers come from rule based agents with simulated cost and latency; "
                  "they say nothing about real model quality")
VERIFY_PRICES = "verify prices before quoting"
MIN_WIDTHS = (8, 15, 10, 14, 11)  # config, pass^k, critical, escalation, cost/call; the last column is not padded
MAX_LISTED = 20  # disagreeing callers named in the markdown page
VARIANTS = tuple(a["id"] for a in DEFAULT_AGENTS)
LABELS = {a["id"]: a["label"] for a in DEFAULT_AGENTS}
MODEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


class AgentConfig(NamedTuple):
    variant: str
    model: str | None = None

    @property
    def id(self) -> str:
        return f"{self.variant}@{self.model}" if self.model else self.variant


def parse_agents(text: str) -> list[AgentConfig]:
    """`v2@haiku,v2@sonnet` to [("v2", "haiku"), ("v2", "sonnet")]. Raises ValueError on anything else."""
    configs: list[AgentConfig] = []
    for part in (p.strip() for p in text.split(",")):
        variant, _, model = part.partition("@")
        if variant not in VARIANTS:
            raise ValueError(f"unknown agent {part!r}, use one of {', '.join(VARIANTS)}, optionally with @model")
        if "@" in part and not MODEL_RE.fullmatch(model):
            raise ValueError(f"bad model in {part!r}: write it like v2@haiku")
        config = AgentConfig(variant, model or None)
        if config in configs:
            raise ValueError(f"{part!r} is listed twice")
        configs.append(config)
    return configs


# ---- the table maths: pure functions of the rows ----

def best_per_column(rows: list[dict]) -> dict[str, list[str]]:
    """The winning config ids per column, every tied config listed in the order given. A column nobody has a value
    for (no price, no escalation runs) is an empty list."""
    def pick(values: dict, best):
        known = {config: v for config, v in values.items() if v is not None}
        top = best(known.values()) if known else None
        return [config for config, v in known.items() if v == top]

    return {
        "pass": pick({r["config"]: Fraction(r["passed"], r["total"] or 1) for r in rows}, max),
        "critical": pick({r["config"]: r["critical"] for r in rows}, min),
        "escalation": pick({r["config"]: Fraction(r["escalation_right"], r["escalation_runs"])
                            if r["escalation_runs"] else None for r in rows}, max),
        "cost": pick({r["config"]: r["cost_per_call"] for r in rows}, min),
        "p95": pick({r["config"]: r["p95_ms"] for r in rows}, min),
    }


def best_line(best: dict[str, list[str]], pass_name: str) -> str:
    def names(key):
        return ", ".join(best[key]) or "n/a"
    return (f"best: {pass_name} {names('pass')}; critical {names('critical')}; escalation {names('escalation')}; "
            f"cost {names('cost')}; P95 {names('p95')}")


def disagreements(call_lists: list[list[dict]]) -> list[str]:
    """Callers some configs pass and others fail, sorted by id. Each list is one config's `calls`."""
    outcomes: dict[str, set] = {}
    for calls in call_lists:
        for call in calls:
            outcomes.setdefault(call["persona_id"], set()).add(bool(call["passed"]))
    return sorted(pid for pid, seen in outcomes.items() if len(seen) > 1)


def recommend(rows: list[dict], repeats: int) -> dict:
    """The fixed rule: among configs with zero critical compliance failures, the highest pass^k; a tie goes to the lower
    cost per call (an unpriced config loses). A tie that is still open, or no eligible config, recommends nobody."""
    rule = RECOMMEND_RULE_TEMPLATE.format(k=repeats)
    name = f"pass^{repeats}"
    eligible = [r for r in rows if r["critical"] == 0]
    if not eligible:
        return {"config": None, "rule": rule, "text": "recommend: none (every config has critical compliance failures)"}

    def rate(r):
        return Fraction(r["passed"], r["total"]) if r["total"] else Fraction(0)

    top = max(rate(r) for r in eligible)
    tied = [r for r in eligible if rate(r) == top]
    cheapest = min((r["cost_per_call"] for r in tied if r["cost_per_call"] is not None), default=None)
    tied = [r for r in tied if r["cost_per_call"] == cheapest] if cheapest is not None else tied
    if len(tied) > 1:
        ids = ", ".join(r["config"] for r in tied)
        return {"config": None, "rule": rule,
                "text": f"recommend: none ({ids} tie on {name} and cost per call; rule: {rule})"}
    return {"config": tied[0]["config"], "rule": rule, "text": f"recommend: {tied[0]['config']} ({rule})"}


# ---- building and running configs ----

def resolve_model(name: str, prices: dict) -> str:
    """`haiku` to `claude-haiku-5-5` through the aliases in config/pricing.yaml; anything else is passed on as is."""
    return prices["aliases"].get(name, name)


def build_config_agent(config: AgentConfig, sop_text: str, backend, prices: dict | None = None):
    """The respond(history) function for a config, its calls metered as the "agent" role on the model it names."""
    model = resolve_model(config.model, prices or load_prices()) if config.model else None
    return build_agent(sop_text, config.variant, MeteredBackend(backend, "agent"), model)


def table_row(record: dict, config: str, run_file: str | None = None) -> dict:
    compliance = record.get("compliance") or {}
    critical = sum(f["count"] for f in compliance.get("failures", []) if f["severity"] == "critical")
    escalation = record["escalation"]["matrix"]
    cost, latency = record["cost"], record["latency"]
    return {
        "config": config, "variant": record["agent"]["variant"], "model": record["agent"]["model"],
        "passed": record["passed"], "total": record["total"], "critical": critical,
        "escalation_right": escalation["tp"] + escalation["tn"], "escalation_runs": record["escalation"]["runs"],
        "cost_per_call": cost["per_call"], "p95_ms": latency["p95_ms"], "run_file": run_file,
    }


def cells(row: dict) -> list[str]:
    runs = row["escalation_runs"]
    return [
        row["config"],
        f"{pct(row['passed'] / row['total'] if row['total'] else 0.0)} ({row['passed']}/{row['total']})",
        str(row["critical"]),
        f"{pct(row['escalation_right'] / runs)} ({row['escalation_right']}/{runs})" if runs else "n/a",
        "n/a" if row["cost_per_call"] is None else f"${row['cost_per_call']:.4f}",
        "n/a" if row["p95_ms"] is None else f"{row['p95_ms']:.0f}",
    ]


def table_lines(rows: list[dict], pass_name: str) -> list[str]:
    body = [cells(r) for r in rows]
    widths = [max(minimum, max(len(line[i]) for line in body) + 2) for i, minimum in enumerate(MIN_WIDTHS)]

    header = ["config", pass_name, "critical", "escalation", "cost/call"]
    return ["".join(h.ljust(w) for h, w in zip(header, widths, strict=True)) + "P95 ms",
            *("".join(c.ljust(w) for c, w in zip(line[:5], widths, strict=True)) + line[5] for line in body)]


def disagree_line(ids: list[str]) -> str:
    return f"disagree ({len(ids)}): {', '.join(ids) if ids else 'none'}"


def render_markdown(record: dict) -> str:
    """One page: the recommendation, the table and what each number is. Simulated and estimated numbers are labelled."""
    rows, k = record["configs"], record["repeats"]
    pass_name = f"pass^{k}"
    count = record["persona_set"]["count"]
    simulated = record["simulated"]
    ids = ", ".join(r["config"] for r in rows)
    ties = disagree_line(record["disagreements"][:MAX_LISTED])
    more = len(record["disagreements"]) - MAX_LISTED
    if more > 0:
        ties += f", and {more} more"
    out = [f"# Agent bake off: {record['persona_set']['name']} set, {count} callers", ""]
    out.append("SYNTHETIC callers and SOP (made up for this demo). "
               + ("Every number below is simulated: offline agents are rule based, so this compares prompt "
                  "versions, not models, and says nothing about real model quality." if simulated else
                  f"Run on {record['backend']['name']}; cost is {record['cost_label']} and latency is "
                  f"{record['latency_label']}."))
    out += ["", f"**{record['recommendation']['text']}**", "",
            f"| config | {pass_name} | critical | escalation | cost per call | P95 ms |", "|---|---|---|---|---|---|"]
    out += ["| " + " | ".join(cells(r)) + " |" for r in rows]
    out += ["", best_line(record["best"], pass_name) + ".", "", ties + ".", "", "## How to read this", ""]
    out += [
        f"- Configs compared: {ids}. Every config ran the same callers, seed {record['seed']} and {k} "
        f"repeat{'s' if k != 1 else ''}.",
        f"- {pass_name} counts callers who passed every repeat. Critical counts critical compliance failures "
        "(not legal advice).",
        f"- Cost is {record['cost_label']}: tokens times the prices in config/pricing.yaml dated "
        f"{record['prices']['retrieved_on']} ({record['prices']['source_url']}); {VERIFY_PRICES}.",
        f"- Latency is {record['latency_label']}, per agent turn.",
        f"- Recommendation rule: {record['recommendation']['rule']}.",
    ]
    return "\n".join(out) + "\n"


# ---- the command ----

def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m callqa compare", description=__doc__)
    p.add_argument("--agents", help="configs to compare: variant or variant@model, comma separated (default "
                   f"{DEFAULT_LLM_AGENTS} on LLM backends, {DEFAULT_OFFLINE_AGENTS} offline)")
    p.add_argument("--set", dest="set_name", default="story", choices=SET_NAMES, help="caller set (default story)")
    p.add_argument("--callers", type=int, help="story set only: how many of the 20 callers (default all 20)")
    p.add_argument("--repeats", type=int, default=1, help=f"runs per caller, 1 to {MAX_REPEATS} (default 1)")
    p.add_argument("--seed", type=int, default=0, help="default 0")
    p.add_argument("--backend", choices=BACKEND_NAMES, help="default: $CALLQA_BACKEND, else claude-cli")
    p.add_argument("--max-turns", type=int, help="turns per call (default 4 on LLM backends, 8 offline)")
    p.add_argument("--workers", type=int, help="calls in parallel (default 2 for claude-cli, 4 otherwise)")
    p.add_argument("--latency-budget-ms", type=float, default=DEFAULT_BUDGET_MS,
                   help=f"an agent turn slower than this is over budget (default {DEFAULT_BUDGET_MS})")
    p.add_argument("--out", type=Path, help="a folder for the records (default runs/ under the demo), or a path "
                   "ending in .json to name the compare record (run records go in a folder beside it)")
    p.add_argument("--export-md", type=Path, metavar="PATH", help="write a one page markdown recommendation")
    p.add_argument("--no-save", action="store_true", help="do not write run or compare records")
    p.add_argument("--yes", action="store_true", help=f"allow an LLM backend run of more than {LARGE_RUN_CALLS} calls")
    args = p.parse_args(argv)
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
    return args


def fail(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 2


def run_config(config, sop_text, backend, generation, args, max_turns, workers, rules, prices, run_id):
    """One config's run record. Raises RunAborted or BackendError."""
    agent = {"id": config.id, "label": LABELS[config.variant] + (f" on {config.model}" if config.model else ""),
             "variant": config.variant, "model": resolve_model(config.model, prices) if config.model else None}
    started = time.time()
    respond = build_config_agent(config, sop_text, backend, prices)
    results = run_suite(respond, generation.personas, args.repeats, max_turns, workers, backend, args.seed,
                        agent_prompt=agent_prompt_for({"id": config.variant}, sop_text), rules=rules,
                        sop=parse_sop(sop_text))
    record = build_report(run_id=run_id, sop_text=sop_text, agent=agent, backend=backend,
                          callers=len(generation.personas), max_turns=max_turns, generation=generation,
                          results=results, seconds=time.time() - started, repeats=args.repeats, seed=args.seed,
                          set_name=args.set_name, synthetic=generation.source.startswith("templates"),
                          metering=True, budget_ms=args.latency_budget_ms, rules=rules)
    if agent["model"]:  # the agent role is priced at this config's model, the other roles keep their defaults
        models = {**record["backend"]["models"], "agent": agent["model"]}
        usage = {role: {key: entry[key] for key in ("input_tokens", "output_tokens")}
                 for role, entry in record["cost"]["by_role"].items()}
        record["backend"]["models"] = models
        record["cost"] = cost_block(usage, models, prices, record["cost"]["label"], len(results))
    return record


def compare_record(configs, records, rows, backend, generation, args, max_turns, rules) -> dict:
    first = records[0]
    ids = [c.id for c in configs]
    note = first["note"] or f"models compared on {backend.name}; cost and latency are {first['cost']['label']}"
    return {
        "schema_version": RECORD_SCHEMA_VERSION, "kind": COMPARE_KIND,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "synthetic": first["synthetic"], "simulated": first["simulated"], "note": note,
        "agent": {"id": "compare", "label": f"bake off {', '.join(ids)}", "variant": None, "model": None},
        "backend": {"name": backend.name}, "persona_set": first["persona_set"],
        "seed": args.seed, "repeats": args.repeats, "callers": first["callers"], "max_turns": max_turns,
        "sop_hash": first["sop_hash"], "rules_file": rules.label if rules else None,
        "cost_label": first["cost"]["label"], "latency_label": first["latency"]["label"],
        "prices": first["cost"]["prices"], "pass_rate": None,
        "configs": rows, "best": best_per_column(rows),
        "disagreements": disagreements([r["calls"] for r in records]),
        "recommendation": recommend(rows, args.repeats), "run_files": [],
    }


def save_all(record: dict, records: list[dict], rows: list[dict], out: Path | None) -> Path:
    """Run records into a folder, then the compare record: `out` is that folder, or (a path ending in .json) the new
    compare record file, with the run records in `<name>_runs` beside it."""
    named = out is not None and out.suffix == ".json"
    folder = (out.parent / f"{out.stem}_runs") if named else (out or ROOT / "runs")
    for run, row in zip(records, rows, strict=True):
        row["run_file"] = save_record(run, folder).name
    record["run_files"] = [row["run_file"] for row in rows]
    return write_record(record, out) if named else save_record(record, folder)


def run_compare(args: argparse.Namespace) -> int:
    try:
        backend = get_backend(args.backend)
    except BackendError as exc:
        return fail(str(exc))
    for warning in backend.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    offline = backend.name == "offline"
    try:
        configs = parse_agents(args.agents or (DEFAULT_OFFLINE_AGENTS if offline else DEFAULT_LLM_AGENTS))
    except ValueError as exc:
        return fail(str(exc))
    if len(configs) < 2:
        return fail("compare needs at least two agents, for example --agents v1,v2")
    if offline and any(c.model for c in configs):
        return fail("offline agents are rule based and have no model, so @model needs a live backend "
                    "(claude-cli or anthropic)")
    out = args.out
    if not args.no_save and out is not None and out.suffix == ".json" and out.exists():
        return fail(f"{out} already exists, not overwriting it")
    try:
        rules = load_rules(DEFAULT_RULES_PATH)
        prices = load_prices()
    except (RulesError, ValueError) as exc:
        return fail(str(exc))
    defaults = RUN_DEFAULTS[backend.name]
    max_turns, workers = args.max_turns or defaults["max_turns"], args.workers or defaults["workers"]
    sop_text = read_sample_sop()
    generation = personas_for_set(sop_text, args.set_name, args.callers)
    callers = len(generation.personas)
    estimate = len(configs) * estimate_calls(callers, max_turns, args.repeats)
    if needs_confirmation(backend, estimate) and not args.yes:
        return fail(f"this run would make about {estimate} model calls on {backend.name} (limit {LARGE_RUN_CALLS}); "
                    "add --yes to run it, or use --backend offline")
    records = []
    try:
        for config in configs:
            records.append(run_config(config, sop_text, backend, generation, args, max_turns, workers, rules,
                                      prices, secrets.token_hex(4)))
    except RunAborted as exc:
        print(f"ABORTED: {exc.message}")
        return 2
    except BackendError as exc:
        print(f"ABORTED: {exc}")
        return 2
    rows = [table_row(r, c.id) for r, c in zip(records, configs, strict=True)]
    record = compare_record(configs, records, rows, backend, generation, args, max_turns, rules)
    pass_name = f"pass^{args.repeats}"
    lines = [f"COMPARE set={args.set_name} callers={callers} repeats={args.repeats} seed={args.seed} "
             f"backend={backend.name}"]
    ids = ", ".join(c.id for c in configs)
    lines.append(f"note: {OFFLINE_NOTE} {ids}" if offline else
                 f"note: comparing {ids}; cost is {record['cost_label']} and latency is {record['latency_label']}; "
                 f"{VERIFY_PRICES}")
    lines += table_lines(rows, pass_name)
    lines += [best_line(record["best"], pass_name), disagree_line(record["disagreements"]),
              record["recommendation"]["text"]]
    if offline:
        lines.append(SIMULATED_LINE)
    if not args.no_save:
        try:
            lines.append(f"record: {save_all(record, records, rows, out)}")
        except OSError as exc:
            print("\n".join(lines))
            return fail(f"could not save the records: {exc}")
    if args.export_md:
        try:
            args.export_md.parent.mkdir(parents=True, exist_ok=True)
            args.export_md.write_text(render_markdown(record), encoding="utf-8")
        except OSError as exc:
            print("\n".join(lines))
            return fail(f"could not write {args.export_md}: {exc}")
        lines.append(f"exported: {args.export_md}")
    print("\n".join(lines))
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(sys.argv[1:] if argv is None else argv)
    except SystemExit as exc:  # argparse: 2 for bad arguments, 0 for --help
        return exc.code if isinstance(exc.code, int) else 2
    return run_compare(args)
