"""Extension point: a tool client the agent under test could call. Nothing in the run path imports this yet.

Where later tooldesk tasks plug in:
- B2 adds a real client to TOOL_CLIENTS (tooldesk over MCP) and logs every call under a `tool_calls` key on
  each call in the run record, which needs schema version 2. Until then the record stays at version 1.
- Only the fake client below exists today. It replays a script and never touches the network.
"""
from dataclasses import dataclass, field
from typing import Protocol

STATUSES = ("OK", "ERROR", "NEEDS_SUPERVISOR", "BLOCKED_BY_POLICY")


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ToolResult:
    status: str
    data: dict = field(default_factory=dict)
    error: str = ""

    def __post_init__(self):
        if self.status not in STATUSES:
            raise ValueError(f"unknown tool status {self.status!r}, expected one of {STATUSES}")


class ToolClient(Protocol):
    def list_tools(self) -> list[str]: ...

    def call(self, name: str, args: dict) -> ToolResult: ...


class FakeToolClient:
    """Replays a fixed script of results, one per call, and records the calls it saw."""

    def __init__(self, script: list[ToolResult], tools: list[str] | None = None):
        self._script = list(script)
        self._tools = list(tools or [])
        self._next = 0
        self.calls: list[ToolCall] = []

    def list_tools(self) -> list[str]:
        return list(self._tools)

    def call(self, name: str, args: dict) -> ToolResult:
        if name not in self._tools:
            return ToolResult("ERROR", error=f"unknown tool {name!r}")
        if self._next >= len(self._script):
            raise RuntimeError(f"FakeToolClient script ran out after {len(self._script)} calls")
        result = self._script[self._next]
        self._next += 1
        self.calls.append(ToolCall(name, dict(args)))
        return result


TOOL_CLIENTS = {"fake": FakeToolClient}
