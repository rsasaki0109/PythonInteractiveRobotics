"""Lightweight trace recorder."""

from __future__ import annotations

from typing import Any

from pir.core.types import Trace


class TraceRecorder:
    """Collect observations, actions, rewards, and info dictionaries."""

    def __init__(self) -> None:
        self.trace = Trace()

    def record(
        self,
        obs: Any,
        action: Any,
        reward: float,
        info: dict[str, Any] | None = None,
    ) -> None:
        self.trace.append(obs, action, reward, info)

    def reset(self) -> None:
        self.trace = Trace()
