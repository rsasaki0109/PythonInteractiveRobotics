"""Serialize a clarifying-question `Trace` into the playground's render shape.

The browser playground (docs/playground.js) draws scenes from a plain config
object: an ``initial`` snapshot plus a list of per-step events. Historically that
config was hand-written in JavaScript, a reimplementation of the Python loop that
could silently drift from the tested example. Pyodide lets the browser run the
*real* ``examples/embodied_ai/35_clarifying_question.py`` loop; this module turns
the resulting `Trace` into the exact JSON the JS renderer already consumes, so
the browser draws real Python output instead of a JS mock.

Everything here is pure Python + JSON-friendly scalars (no numpy, no matplotlib),
so it runs unchanged in CPython, in tests, and in Pyodide. The shape it produces
is pinned by tests/test_playground_trace.py — that test is the drift guard the
design memo (docs/pyodide_playground_strategy.md) calls "the contract".
"""

from __future__ import annotations

from math import log2
from typing import Any

from pir.core.types import Failure

# The clarifying tabletop is fixed: red block at (0.32, 0.56), blue at
# (0.68, 0.56). The JS renderer draws on a 0-100 SVG canvas, so positions are
# scaled by 100. This mirrors ClarifyingQuestionWorld.reset().
_SVG_SCALE = 100.0


def _entropy(distribution: dict[str, float]) -> float:
    total = sum(p * log2(p) for p in distribution.values() if p > 0.0)
    return -total if total else 0.0  # avoid -0.0 for a point-mass belief


def _belief(resolved_color: str | None) -> dict[str, Any]:
    """Two-block belief: uniform until clarified, then a point mass."""

    if resolved_color is None:
        distribution = {"red": 0.5, "blue": 0.5}
        entropy = _entropy(distribution)
        return {
            "red": distribution["red"],
            "blue": distribution["blue"],
            "entropy": entropy,
            "askGain": entropy,
            "policy": "ask",
        }
    distribution = {
        "red": 1.0 if resolved_color == "red" else 0.0,
        "blue": 1.0 if resolved_color == "blue" else 0.0,
    }
    return {
        "red": distribution["red"],
        "blue": distribution["blue"],
        "entropy": _entropy(distribution),
        "askGain": 0.0,
        "policy": "act",
    }


def _failure_kind(info: dict[str, Any]) -> str:
    failure = info.get("failure")
    return failure.kind if isinstance(failure, Failure) else ""


def _action_label(action: dict[str, Any]) -> str:
    action_type = action.get("type", "noop")
    if action_type == "ask":
        return "ask(which_block)"
    color = action.get("color")
    if color:
        return f"{action_type}({color})"
    return action_type


def _pick_at(info: dict[str, Any]) -> list[float] | None:
    position = info.get("pick_position")
    if position is None:
        return None
    return [
        round(float(position[0]) * _SVG_SCALE, 4),
        round(float(position[1]) * _SVG_SCALE, 4),
    ]


def _initial_snapshot(command: str) -> dict[str, Any]:
    return {
        "type": "tabletop",
        "command": command,
        "target": "unresolved",
        "agentState": "parse_command",
        "failure": "none",
        "belief": _belief(None),
        "picked": None,
        "pickAt": None,
        "focus": None,
        "question": None,
        "answer": None,
    }


def _step_snapshot(command: str, info: dict[str, Any], obs: dict[str, Any]) -> dict[str, Any]:
    resolved = info.get("resolved_goal") or {}
    color = resolved.get("color")
    return {
        "type": "tabletop",
        "command": command,
        "target": color or "unresolved",
        "agentState": info.get("agent_state", ""),
        "failure": _failure_kind(info) or "none",
        "belief": _belief(color),
        "picked": obs.get("picked_color"),
        "pickAt": _pick_at(info),
        "focus": obs.get("focus_color"),
        "question": obs.get("last_question") or info.get("question"),
        "answer": obs.get("last_answer"),
    }


def clarifying_trace_to_playground(
    trace: Any,
    *,
    command: str = "pick the block",
    answer: str = "red",
) -> dict[str, Any]:
    """Convert a clarifying-question `Trace` into the playground config object.

    The returned dict matches what docs/playground.js builds in
    ``buildClarifyingScenario`` — ``{command, totalSteps, initial, steps}`` where
    each step is ``{action, reward, failure, agentState, snapshot}`` — so the JS
    renderer can draw it with no changes. Unlike the JS mock, every field here is
    derived from the real loop's observations and info dicts.
    """

    _ = answer  # answer is encoded in the trace; kept for a self-describing call site
    steps: list[dict[str, Any]] = []
    for action, reward, info, obs in zip(
        trace.actions, trace.rewards, trace.infos, trace.observations
    ):
        steps.append(
            {
                "action": _action_label(action),
                "reward": round(float(reward), 4),
                "failure": _failure_kind(info),
                "agentState": info.get("agent_state", ""),
                "snapshot": _step_snapshot(command, info, obs),
            }
        )
    return {
        "command": command,
        "totalSteps": len(steps),
        "initial": _initial_snapshot(command),
        "steps": steps,
    }
