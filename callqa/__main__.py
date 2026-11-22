"""`python -m callqa`: the dashboard, the headless checks (--smoke, --check), or eval, diff, compare, calibrate."""
from __future__ import annotations

import argparse
import importlib
import os
import sys
import threading
import webbrowser
from dataclasses import dataclass

from callqa import calibrate, compare, diff, evaluate
from callqa.compliance import DEFAULT_RULES_PATH, RulesError, load_rules
from callqa.persona_sets import SET_NAMES
from callqa.personas import generate_personas
from callqa.server import (
    DEFAULT_AGENTS,
    LARGE_RUN_CALLS,
    MAX_DASHBOARD_REPEATS,
    build_agent_for,
    caller_count,
    estimate_calls,
    make_server,
    needs_confirmation,
    read_sample_sop,
    run_check,
)
from evals.backends import BACKEND_NAMES, BackendError, get_backend
from evals.run_evals import RUN_DEFAULTS, RunAborted, run_suite

SMOKE_EXPECTED = {"v1": 14, "v2": 19}
SMOKE_CALLERS = 20
PORT_TRIES = 11  # the asked for port, then the next 10
# first argument -> that module's main(argv); later tasks add more
COMMANDS = {"eval": evaluate.main, "diff": diff.main, "compare": compare.main, "calibrate": calibrate.main}


@dataclass
class Config:
    backend: object
    callers: int
    max_turns: int
    workers: int
    agents: list[dict]
    set_name: str = "story"
    repeats: int = 1
    rules: object = None  # a callqa.compliance.Rules, or None when the layer is off


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m callqa", description=__doc__)
    p.add_argument("--backend", choices=BACKEND_NAMES,
                   help="default: $CALLQA_BACKEND, else claude-cli (offline, with a warning, if claude is missing)")
    p.add_argument("--offline", action="store_true", help="same as --backend offline")
    p.add_argument("--callers", type=int, help="simulated callers per run (default 5 on LLM backends, 20 offline)")
    p.add_argument("--set", choices=SET_NAMES, default="story",
                   help="dashboard caller set (default story: 5 callers on LLM backends, 20 offline; full: all 31)")
    p.add_argument("--repeats", type=int, default=1, metavar=f"1-{MAX_DASHBOARD_REPEATS}",
                   help="dashboard runs per caller (default 1)")
    p.add_argument("--yes", action="store_true",
                   help=f"allow a dashboard run of more than {LARGE_RUN_CALLS} model calls on an LLM backend")
    p.add_argument("--max-turns", type=int, help="turns per call (default 4 on LLM backends, 8 offline)")
    p.add_argument("--workers", type=int, help="calls in parallel (default 2 for claude-cli, 4 otherwise)")
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT") or 8000),
                   help="default 8000; if busy the next 10 ports are tried")
    p.add_argument("--no-browser", action="store_true", help="do not open the browser")
    p.add_argument("--smoke", action="store_true",
                   help="offline, 20 callers, v1 then v2, print SMOKE OK or what differs (ignores env)")
    p.add_argument("--check", action="store_true",
                   help="run v1 then v2 on the selected backend and gate the before/after (exit 0 or 1)")
    p.add_argument("--rules", metavar="PATH", help="compliance rules file for the dashboard, or none to turn the "
                   "layer off (default: rules/collections_india.yaml, the file eval uses)")
    p.add_argument("--agent", action="append", default=[], metavar="ID=LABEL=MODULE:FUNCTION",
                   help="add or replace a dashboard agent with your own respond(history) function; repeatable")
    args = p.parse_args(argv)
    for flag in ("callers", "max_turns", "workers"):
        value = getattr(args, flag)
        if value is not None and value < 1:
            p.error(f"--{flag.replace('_', '-')} must be at least 1")
    if not 1 <= args.repeats <= MAX_DASHBOARD_REPEATS:
        p.error(f"--repeats must be from 1 to {MAX_DASHBOARD_REPEATS}")
    if args.set == "full" and args.callers is not None:
        p.error("--callers cannot be used with --set full; the full set is always all of its callers")
    if args.offline and args.backend not in (None, "offline"):
        p.error(f"--offline cannot be combined with --backend {args.backend}")
    return args


def load_agent_entry(spec: str) -> dict:
    """`id=label=module:function` becomes an allowlist entry. Only ever called with command line text."""
    parts = spec.split("=", 2)
    if len(parts) != 3 or not all(parts) or ":" not in parts[2]:
        raise SystemExit(f"--agent must look like id=label=module:function, got {spec!r}")
    agent_id, label, target = parts
    module_name, _, func_name = target.partition(":")
    sys.path.insert(0, os.getcwd())
    try:
        respond = getattr(importlib.import_module(module_name), func_name)
    except (ImportError, AttributeError) as exc:
        raise SystemExit(f"--agent {agent_id}: cannot load {target}: {exc}") from None
    return {"id": agent_id, "label": label, "build": lambda sop_text, backend, respond=respond: respond}


def resolve_config(args: argparse.Namespace, which=None) -> Config:
    """Pick the backend and fill in per backend defaults. Raises BackendError if it cannot be used."""
    kwargs = {"which": which} if which else {}
    backend = get_backend("offline" if args.offline else args.backend, **kwargs)
    defaults = RUN_DEFAULTS[backend.name]
    agents = {a["id"]: a for a in DEFAULT_AGENTS}
    for spec in args.agent:
        entry = load_agent_entry(spec)
        agents[entry["id"]] = entry
    return Config(
        backend=backend,
        callers=args.callers or (20 if backend.name == "offline" else 5),
        max_turns=args.max_turns or defaults["max_turns"],
        workers=args.workers or defaults["workers"],
        agents=list(agents.values()),
        set_name=args.set,
        repeats=args.repeats,
        rules=None if (args.rules or "").lower() == "none" else load_rules(args.rules or DEFAULT_RULES_PATH),
    )


def bind_server(cfg: Config, port: int, allow_large: bool = False, results_dir=None):
    """Bind the asked for port, or the next free one of the following 10."""
    last_error = None
    for candidate in range(port, port + PORT_TRIES):
        try:
            return make_server(cfg.backend, cfg.callers, cfg.max_turns, cfg.workers, cfg.agents, port=candidate,
                               set_name=cfg.set_name, repeats=cfg.repeats, allow_large=allow_large,
                               results_dir=results_dir, rules=cfg.rules)
        except OSError as exc:
            last_error = exc
    raise SystemExit(f"error: no free port from {port} to {port + PORT_TRIES - 1}: {last_error}")


def smoke() -> int:
    """v1 then v2 offline through the real run machinery. No network, no server, no browser."""
    os.environ["CALLQA_BACKEND"] = "offline"
    backend = get_backend("offline")
    sop_text = read_sample_sop()
    personas = generate_personas(sop_text, SMOKE_CALLERS, None).personas
    entries = {a["id"]: a for a in DEFAULT_AGENTS}
    got = {}
    for variant in ("v1", "v2"):
        respond = build_agent_for(entries[variant], sop_text, backend)
        results = run_suite(respond, personas, 1, RUN_DEFAULTS["offline"]["max_turns"],
                            RUN_DEFAULTS["offline"]["workers"], backend, 0)
        got[variant] = (sum(r.passed for r in results), len(results))
    expected = {v: (n, SMOKE_CALLERS) for v, n in SMOKE_EXPECTED.items()}
    if got != expected:
        for v in ("v1", "v2"):
            if got[v] != expected[v]:
                print(f"SMOKE FAILED {v}: expected {expected[v][0]}/{expected[v][1]}, got {got[v][0]}/{got[v][1]}")
        return 1
    print(f"SMOKE OK v1 {got['v1'][0] / got['v1'][1]:.0%} ({got['v1'][0]}/{got['v1'][1]}) "
          f"v2 {got['v2'][0] / got['v2'][1]:.0%} ({got['v2'][0]}/{got['v2'][1]})")
    return 0


def check(cfg: Config) -> int:
    try:
        ok, lines = run_check(cfg.backend, cfg.callers, cfg.max_turns, cfg.workers, cfg.agents)
    except RunAborted as exc:
        print(f"ABORTED: {exc.message}")
        return 2
    for line in lines[len(cfg.backend.warnings):]:  # the warnings already went to stderr
        print(line)
    return 0 if ok else 1


def serve(cfg: Config, args: argparse.Namespace) -> int:
    server = bind_server(cfg, args.port, args.yes)
    url = f"http://127.0.0.1:{server.server_address[1]}"  # the server binds IPv4 only
    repeats = f", {cfg.repeats} repeats" if cfg.repeats > 1 else ""
    print(f"Call QA dashboard: {url}  (backend {cfg.backend.name}, {cfg.callers} callers{repeats}; Ctrl-C to stop)",
          flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        while thread.is_alive():
            thread.join(0.5)
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        server.shutdown()
        server.server_close()
        if any(r.status == "running" for r in list(server.app.runs.values())):
            # Pool workers would otherwise keep working (and spending calls) through the queued conversations.
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)
    return 0


def guard_large_run(cfg: Config, args: argparse.Namespace, agent_versions: int = 1) -> int:
    """2 (with the reason on stderr) for a run over LARGE_RUN_CALLS model calls without --yes, else 0.
    The dashboard runs one agent version at a time; --check runs two."""
    estimate = agent_versions * estimate_calls(caller_count(cfg.set_name, cfg.callers), cfg.max_turns, cfg.repeats)
    if needs_confirmation(cfg.backend, estimate) and not args.yes:
        print(f"error: this would make about {estimate} model calls on {cfg.backend.name} "
              f"(limit {LARGE_RUN_CALLS}); add --yes to allow it, or use --backend offline", file=sys.stderr)
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in COMMANDS:
        return COMMANDS[argv[0]](argv[1:])
    args = parse_args(argv)
    if args.smoke:
        return smoke()
    try:
        cfg = resolve_config(args)
    except (BackendError, RulesError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for warning in cfg.backend.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    os.environ["CALLQA_BACKEND"] = cfg.backend.name  # --agent functions use the same backend
    if args.check:
        return guard_large_run(cfg, args, agent_versions=2) or check(cfg)
    if guard_large_run(cfg, args):
        return 2
    return serve(cfg, args)


if __name__ == "__main__":
    sys.exit(main())
