"""Consistency over repeats: does the agent pass every time, or only sometimes?

A persona is run k times. `pass_at_1` is the share of single runs that passed. `pass_hat_k` is the share of personas
that passed all k repeats: the n = k case of the pass^k estimator in tau bench (Yao et al. 2024, arXiv 2406.12045).
A flaky persona passed some repeats but not all. Results are grouped per persona with `group_by_persona`.
"""
from __future__ import annotations


def group_by_persona(results: list) -> dict[str, list]:
    """Results per persona id, each list in repeat order (`result.run`). Persona order is the order first seen."""
    groups: dict[str, list] = {}
    for result in results:
        groups.setdefault(result.persona, []).append(result)
    return {pid: sorted(rs, key=lambda r: r.run) for pid, rs in groups.items()}


def runs_passed(groups: dict[str, list]) -> tuple[int, int]:
    """(runs passed, runs total) over every repeat of every persona."""
    runs = [r for rs in groups.values() for r in rs]
    return sum(r.passed for r in runs), len(runs)


def pass_at_1(groups: dict[str, list]) -> float:
    """Runs passed over runs."""
    passed, total = runs_passed(groups)
    return passed / total if total else 0.0


def passed_every_repeat(results: list, repeats: int | None = None) -> bool:
    """True when the persona has exactly `repeats` results (default: however many it has) and every one passed."""
    return (repeats is None or len(results) == repeats) and all(r.passed for r in results)


def incomplete(groups: dict[str, list], repeats: int) -> list[str]:
    """Warning lines for personas whose number of results is not `repeats`."""
    return [f"{pid}: expected {repeats} repeats, got {len(rs)}" for pid, rs in groups.items() if len(rs) != repeats]


def pass_hat_k(groups: dict[str, list], repeats: int | None = None) -> float:
    """Personas that passed every repeat over personas. With `repeats`, a persona with a different number of results
    does not count as passing."""
    if not groups:
        return 0.0
    return sum(passed_every_repeat(rs, repeats) for rs in groups.values()) / len(groups)


def first_failing(results: list):
    """The first failing repeat, else repeat 0: the result that stands for a persona in the report."""
    return next((r for r in results if not r.passed), results[0])


def flaky(groups: dict[str, list]) -> list[dict]:
    """Personas that passed some repeats but not all, with the first failing repeat (0 based)."""
    found = []
    for pid, rs in groups.items():
        passed = sum(r.passed for r in rs)
        if 0 < passed < len(rs):
            found.append({"persona_id": pid, "passed": passed, "total": len(rs),
                          "failed_repeat": next(r.run for r in rs if not r.passed)})
    return found


def consistency_block(groups: dict[str, list], repeats: int) -> dict:
    """The report's `consistency` block (C1 in plans/qa-upgrades.md)."""
    passed, total = runs_passed(groups)
    return {"repeats": repeats, "pass_at_1": round(pass_at_1(groups), 4), "runs_passed": passed, "runs_total": total,
            "pass_hat_k": round(pass_hat_k(groups, repeats), 4), "flaky": flaky(groups)}
