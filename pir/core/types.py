"""Shared data types for closed-loop examples."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Failure:
    """A recoverable or terminal failure reported by an environment."""

    kind: str
    message: str
    recoverable: bool = True


@dataclass(frozen=True)
class StepResult:
    """Result returned by an environment step."""

    obs: Any
    reward: float
    done: bool
    info: dict[str, Any] = field(default_factory=dict)

    def as_tuple(self) -> tuple[Any, float, bool, dict[str, Any]]:
        return self.obs, self.reward, self.done, self.info


@dataclass
class Trace:
    """A lightweight record of a closed-loop run."""

    observations: list[Any] = field(default_factory=list)
    actions: list[Any] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    infos: list[dict[str, Any]] = field(default_factory=list)

    def append(
        self,
        obs: Any,
        action: Any,
        reward: float,
        info: dict[str, Any] | None = None,
    ) -> None:
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(float(reward))
        self.infos.append({} if info is None else dict(info))

    def failures(self) -> list[Failure]:
        return [
            info["failure"]
            for info in self.infos
            if isinstance(info.get("failure"), Failure)
        ]

    def summary(self) -> "TraceSummary":
        """Return compact run statistics for headless inspection."""

        return summarize_trace(self)


@dataclass(frozen=True)
class TraceSummary:
    """Compact statistics extracted from a closed-loop `Trace`."""

    steps: int
    total_reward: float
    success: bool
    failure_counts: dict[str, int]
    recoverable_failure_count: int
    terminal_failure_count: int
    counters: dict[str, int | float]
    final_info: dict[str, Any]

    @property
    def failure_kinds(self) -> list[str]:
        return sorted(self.failure_counts)

    @property
    def retry_count(self) -> int | float:
        return self.counters.get("retry_count", 0)


def summarize_trace(trace: Trace) -> TraceSummary:
    """Summarize rewards, failures, success, and loop counters in a trace."""

    failures = trace.failures()
    failure_counts: dict[str, int] = {}
    recoverable_failure_count = 0
    terminal_failure_count = 0
    for failure in failures:
        failure_counts[failure.kind] = failure_counts.get(failure.kind, 0) + 1
        if failure.recoverable:
            recoverable_failure_count += 1
        else:
            terminal_failure_count += 1

    final_info = dict(trace.infos[-1]) if trace.infos else {}
    return TraceSummary(
        steps=len(trace.actions),
        total_reward=sum(trace.rewards),
        success=any(bool(info.get("success")) for info in trace.infos),
        failure_counts=failure_counts,
        recoverable_failure_count=recoverable_failure_count,
        terminal_failure_count=terminal_failure_count,
        counters=_max_numeric_counters(trace.infos),
        final_info=final_info,
    )


def _max_numeric_counters(infos: list[dict[str, Any]]) -> dict[str, int | float]:
    counters: dict[str, int | float] = {}
    for info in infos:
        for key, value in info.items():
            if not key.endswith("_count"):
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            previous = counters.get(key)
            if previous is None or value > previous:
                counters[key] = value
    return counters
