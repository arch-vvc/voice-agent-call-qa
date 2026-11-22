"""Tokens, cost and latency for a run.

Three honesty labels, never mixed up:
  simulated  offline runs: tokens are ceil(chars / 4) of the prompts and replies a live run would send
  estimated  claude-cli runs: the CLI reports no usage here, so the same chars / 4 formula
  reported   anthropic runs: the numbers the SDK returned

Prices come only from config/pricing.yaml. A model without a price gives a null cost and is named in
`unknown_models`; nothing is ever guessed.
"""
from __future__ import annotations

import contextlib
import contextvars
import datetime
import math
import threading
from pathlib import Path

import yaml

ROLES = ("caller", "agent", "judge")
JUDGE_RUNS = 2  # every call is judged twice
JUDGE_OUTPUT_TOKENS = 60  # fixed per judge run in simulation
PRICING_PATH = Path(__file__).resolve().parents[1] / "config" / "pricing.yaml"


class PricingError(ValueError):
    """config/pricing.yaml is missing something or has a bad value."""


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / 4)


def prompt_chars(system: str, messages: list[dict]) -> int:
    return len(system or "") + sum(len(m["content"]) for m in messages)


class Meter:
    """Token totals per role. Safe to share between threads."""

    def __init__(self):
        self._lock = threading.Lock()
        self._usage: dict[str, dict[str, int]] = {}

    def add(self, role: str, input_tokens: int, output_tokens: int) -> None:
        with self._lock:
            entry = self._usage.setdefault(role, {"input_tokens": 0, "output_tokens": 0})
            entry["input_tokens"] += input_tokens
            entry["output_tokens"] += output_tokens

    def snapshot(self) -> dict[str, dict[str, int]]:
        with self._lock:
            return {role: dict(entry) for role, entry in self._usage.items()}


_current: contextvars.ContextVar[Meter | None] = contextvars.ContextVar("callqa_meter", default=None)


@contextlib.contextmanager
def use_meter(meter: Meter):
    """Make `meter` the default for MeteredBackend(..., meter=None) created inside the block, so code that builds its
    own backend (the agent under test) is metered for the call being run."""
    token = _current.set(meter)
    try:
        yield meter
    finally:
        _current.reset(token)


class MeteredBackend:
    """Wraps a backend and counts tokens for one role. Name, warnings and requested still read through to the real
    backend. A backend with `complete_with_usage` gives reported numbers; one with only `complete` is estimated."""

    def __init__(self, backend, role: str, meter: Meter | None = None):
        self._backend = backend
        self.role = role
        self.meter = meter

    def __getattr__(self, attr):  # name, warnings, requested, anything else the real backend has
        return getattr(self._backend, attr)

    def complete(self, system: str, messages: list[dict], model: str, max_tokens: int) -> str:
        meter = self.meter or _current.get()
        with_usage = getattr(self._backend, "complete_with_usage", None)
        usage = None
        if with_usage is not None:
            text, usage = with_usage(system, messages, model, max_tokens)
        else:
            text = self._backend.complete(system, messages, model, max_tokens)
        if meter is not None:
            if usage is None:
                usage = {"input_tokens": math.ceil(prompt_chars(system, messages) / 4),
                         "output_tokens": estimate_tokens(text)}
            meter.add(self.role, usage["input_tokens"], usage["output_tokens"])
        return text


def simulated_latency_ms(reply: str) -> int:
    return 250 + 10 * len(reply)


def simulate_usage(persona: dict, transcript: list[dict], prompts: dict) -> tuple[dict, list[int]]:
    """What a live run of this finished offline call would have used: (usage by role, latency per agent turn in ms).
    `prompts` holds the system text each role would send ("agent", "caller") and the judge prompt ("judge"). The
    opening line is scripted, so only later caller lines count as caller output. `persona` is part of the contract
    but the prompts already carry everything the persona adds."""
    usage = {role: {"input_tokens": 0, "output_tokens": 0} for role in ROLES}
    latency: list[int] = []
    for i, message in enumerate(transcript):
        if message["role"] == "assistant":
            role, system = "agent", prompts.get("agent", "")
            latency.append(simulated_latency_ms(message["content"]))
        elif i > 0:
            role, system = "caller", prompts.get("caller", "")
        else:
            continue
        usage[role]["input_tokens"] += math.ceil((len(system) + prompt_chars("", transcript[:i])) / 4)
        usage[role]["output_tokens"] += estimate_tokens(message["content"])
    usage["judge"]["input_tokens"] = JUDGE_RUNS * estimate_tokens(prompts.get("judge", ""))
    usage["judge"]["output_tokens"] = JUDGE_RUNS * JUDGE_OUTPUT_TOKENS
    return usage, latency


def percentile(values: list[float], p: float):
    """Nearest rank percentile: sorted values, index ceil(p / 100 * n) - 1. None for no values."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered), max(1, math.ceil(p / 100 * len(ordered)))) - 1]


def latency_block(latencies: list[float], label: str, budget_ms: float) -> dict:
    return {"label": label, "p50_ms": percentile(latencies, 50), "p95_ms": percentile(latencies, 95),
            "budget_ms": budget_ms, "turns": len(latencies), "over_budget": sum(ms > budget_ms for ms in latencies)}


def load_prices(path: Path | str | None = None) -> dict:
    """Read and validate the pricing file. Returns source_url, retrieved_on, note, currency, models and aliases."""
    path = Path(path or PRICING_PATH)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise PricingError(f"cannot read {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise PricingError(f"{path.name} must be a mapping")
    if data.get("schema_version") != 1:
        raise PricingError("pricing schema_version must be 1")
    for key in ("source_url", "currency"):
        if not isinstance(data.get(key), str) or not data[key].strip():
            raise PricingError(f"pricing needs a non empty {key}")
    retrieved = data.get("retrieved_on")
    try:
        if not isinstance(retrieved, str):
            raise ValueError
        datetime.date.fromisoformat(retrieved)
    except ValueError:
        raise PricingError("pricing retrieved_on must be a YYYY-MM-DD date") from None
    if "verify" not in str(data.get("note", "")).lower():
        raise PricingError("pricing note must tell the reader to verify the prices")
    models = data.get("per_million_tokens")
    if not isinstance(models, dict) or not models:
        raise PricingError("pricing needs per_million_tokens")
    for model, entry in models.items():
        if not isinstance(entry, dict) or set(entry) != {"input", "output"}:
            raise PricingError(f"{model}: needs exactly input and output")
        for side, value in entry.items():
            number = isinstance(value, (int, float)) and not isinstance(value, bool)
            if value is not None and not (number and value >= 0):
                raise PricingError(f"{model}.{side} must be a number of zero or more, or null")
    aliases = data.get("aliases") or {}
    for alias, target in aliases.items():
        if target not in models:
            raise PricingError(f"alias {alias!r} points at unknown model {target!r}")
    return {"source_url": data["source_url"], "retrieved_on": retrieved, "note": data["note"],
            "currency": data["currency"], "models": models, "aliases": dict(aliases)}


def price_for(prices: dict, model: str) -> tuple[float, float] | None:
    """(input, output) USD per million tokens, or None when the model is unknown or its price is null."""
    entry = prices["models"].get(prices["aliases"].get(model, model))
    if not entry or entry["input"] is None or entry["output"] is None:
        return None
    return entry["input"], entry["output"]


def cost_block(usage: dict, models: dict, prices: dict, label: str, calls: int) -> dict:
    """The run record `cost` block. `usage` is tokens by role, `models` the model id by role."""
    by_role, unknown = {}, set()
    for role in ROLES:
        tokens = usage.get(role) or {}
        tin, tout = tokens.get("input_tokens", 0), tokens.get("output_tokens", 0)
        price = price_for(prices, models.get(role, ""))
        if tin == 0 and tout == 0:
            cost = 0.0
        elif price is None:
            cost = None
            unknown.add(models.get(role) or role)
        else:
            cost = round((tin * price[0] + tout * price[1]) / 1_000_000, 6)
        by_role[role] = {"input_tokens": tin, "output_tokens": tout, "cost": cost}
    known = all(entry["cost"] is not None for entry in by_role.values())
    total = round(sum(entry["cost"] for entry in by_role.values()), 6) if known else None
    return {
        "label": label, "currency": prices["currency"],
        "per_call": round(total / calls, 6) if total is not None and calls else None,
        "total": total, "by_role": by_role,
        "prices": {"source_url": prices["source_url"], "retrieved_on": prices["retrieved_on"]},
        "unknown_models": sorted(unknown),
    }


def cost_label(backend_name: str) -> str:
    return {"offline": "simulated", "anthropic": "reported"}.get(backend_name, "estimated")
