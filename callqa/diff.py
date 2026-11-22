"""`python -m callqa diff OLD.json NEW.json [--gate]`: what changed between two run records.

A persona passes in a record when it passed every repeat (pass^k, the n = k case of the tau bench estimator,
Yao et al. 2024, arXiv 2406.12045), so a persona that is flaky across repeats counts as failing.
Exit 0, or with --gate 1 when a persona that passed in OLD fails in NEW, or 2 for an unreadable record."""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

from callqa.runrecord import RecordError, load_record


def pct(x: float) -> str:
    return format(x, ".0%")


@dataclass
class Diff:
    newly_broken: list[str]
    fixed: list[str]
    still_failing: list[str]
    added: list[str]  # only in NEW
    removed: list[str]  # only in OLD
    old_rate: float  # unrounded pass^k
    new_rate: float
    points: int  # round((new - old) * 100) of the unrounded rates
    warnings: list[str] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)  # cost and latency deltas, only when both records have them

    @property
    def regressed(self) -> bool:
        return bool(self.newly_broken)


def persona_passes(record: dict) -> dict[str, bool]:
    """persona id -> passed every repeat."""
    try:
        return {c["persona_id"]: c["repeats_passed"] == c["repeats_total"] for c in record["calls"]}
    except (KeyError, TypeError):
        raise RecordError("a run record call lacks persona_id, repeats_passed or repeats_total") from None


def rate(record: dict) -> float:
    return record["passed"] / record["total"] if record["total"] else 0.0


def mismatch_warnings(old: dict, new: dict) -> list[str]:
    out = []
    if old["schema_version"] != new["schema_version"]:
        out.append(f"schema_version differs ({old['schema_version']} vs {new['schema_version']})")
    old_set, new_set = old["persona_set"], new["persona_set"]
    if old_set["name"] != new_set["name"]:
        out.append(f"persona sets differ ({old_set['name']} vs {new_set['name']}), results are not comparable")
    elif old_set["ids_sha256"] != new_set["ids_sha256"]:
        out.append(f"the {new_set['name']} persona set has different callers in the two records")
    if old["repeats"] != new["repeats"]:
        out.append(f"repeats differ ({old['repeats']} vs {new['repeats']})")
    if old["backend"]["name"] != new["backend"]["name"]:
        out.append(f"backends differ ({old['backend']['name']} vs {new['backend']['name']})")
    return out


def delta_lines(old: dict, new: dict) -> list[str]:
    """Cost and latency lines, printed only when BOTH records carry the block with a value."""
    out = []
    old_cost, new_cost = old.get("cost"), new.get("cost")
    if old_cost and new_cost and old_cost.get("per_call") is not None and new_cost.get("per_call") is not None:
        out.append(f"cost per call ({new_cost['label']}) ${old_cost['per_call']:.4f} -> ${new_cost['per_call']:.4f}")
    old_lat, new_lat = old.get("latency"), new.get("latency")
    if old_lat and new_lat and old_lat.get("p95_ms") is not None and new_lat.get("p95_ms") is not None:
        out.append(f"P95 agent turn ({new_lat['label']}) {old_lat['p95_ms']} ms -> {new_lat['p95_ms']} ms")
    return out


def diff_records(old: dict, new: dict) -> Diff:
    before, after = persona_passes(old), persona_passes(new)
    shared = sorted(set(before) & set(after))
    old_rate, new_rate = rate(old), rate(new)
    return Diff(
        newly_broken=[p for p in shared if before[p] and not after[p]],
        fixed=[p for p in shared if not before[p] and after[p]],
        still_failing=[p for p in shared if not before[p] and not after[p]],
        added=sorted(set(after) - set(before)),
        removed=sorted(set(before) - set(after)),
        old_rate=old_rate,
        new_rate=new_rate,
        points=round((new_rate - old_rate) * 100),
        warnings=mismatch_warnings(old, new),
        lines=delta_lines(old, new),
    )


def id_list(label: str, ids: list[str]) -> str:
    return f"{label} ({len(ids)}): {', '.join(ids) if ids else 'none'}"


def pass_line(old: dict, new: dict, d: Diff) -> str:
    old_k, new_k = old["repeats"], new["repeats"]
    first = f"pass^{old_k} {pct(d.old_rate)} ({old['passed']}/{old['total']})"
    second = f"{pct(d.new_rate)} ({new['passed']}/{new['total']})"
    if old_k != new_k:
        second = f"pass^{new_k} {second}"
    return f"{first} -> {second}, {d.points:+d} pts"


def render(old: dict, new: dict, old_name: str, new_name: str, d: Diff, gate: bool) -> list[str]:
    persona_set = new["persona_set"]
    out = [f"DIFF {old_name} ({old['agent']['id']}) -> {new_name} ({new['agent']['id']})  set={persona_set['name']} "
           f"personas={persona_set['count']} repeats={new['repeats']} backend={new['backend']['name']}"]
    out += [f"warning: {w}" for w in d.warnings]
    out += [id_list("NEWLY BROKEN", d.newly_broken), id_list("FIXED", d.fixed),
            id_list("STILL FAILING", d.still_failing)]
    if d.added:
        out.append(id_list("ADDED", d.added))
    if d.removed:
        out.append(id_list("REMOVED", d.removed))
    out.append(pass_line(old, new, d))
    out += d.lines
    if gate:
        n = len(d.newly_broken)
        if n:
            who = "persona now fails" if n == 1 else "personas now fail"
            out.append(f"GATE FAILED: {n} previously passing {who}: {', '.join(d.newly_broken)}")
        else:
            out.append("GATE OK: no previously passing persona fails")
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m callqa diff", description=__doc__)
    p.add_argument("old", type=Path, help="the earlier run record (.json)")
    p.add_argument("new", type=Path, help="the later run record (.json)")
    p.add_argument("--gate", action="store_true", help="exit 1 if any persona that passed in OLD fails in NEW")
    args = p.parse_args(argv)
    try:
        old, new = load_record(args.old), load_record(args.new)
        d = diff_records(old, new)
    except RecordError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print("\n".join(render(old, new, args.old.stem, args.new.stem, d, args.gate)))
    return 1 if args.gate and d.regressed else 0


if __name__ == "__main__":
    sys.exit(main())
