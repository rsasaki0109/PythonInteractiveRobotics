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


# --- pick_and_retry (continuous tabletop) -----------------------------------
#
# This loop has no red/blue distribution; its "belief" is a 2D position estimate
# (mean + shrinking radius) that the JS renderer draws spatially. Scene geometry
# (true object, occluder, initial camera) is ground truth the agent never sees,
# so the caller passes it in from the real Tabletop2D rather than this module
# hard-coding world constants. Everything emitted here is plain JSON.


def _xy(point: Any) -> list[float]:
    return [round(float(point[0]) * _SVG_SCALE, 4), round(float(point[1]) * _SVG_SCALE, 4)]


def _pick_and_retry_state(info: dict[str, Any], has_belief: bool) -> str:
    if info.get("success"):
        return "done"
    if _failure_kind(info) == "grasp_miss":
        return "update_belief_and_retry"
    if (info.get("action_type") or "") == "look":
        return "scan_for_object"
    return info.get("action_type") or "noop"


def _pick_and_retry_policy(info: dict[str, Any], has_belief: bool) -> str:
    if info.get("success"):
        return "done"
    if not has_belief:
        return "scan"
    if _failure_kind(info) == "grasp_miss":
        return "retry"
    return "act"


def pick_and_retry_trace_to_playground(
    trace: Any,
    *,
    object_xy: Any,
    occluder: Any,
    camera: Any,
    command: str = "pick the block",
) -> dict[str, Any]:
    """Convert a pick_and_retry `Trace` into the playground's tabletop2d config.

    ``object_xy`` / ``occluder`` / ``camera`` are the real Tabletop2D geometry in
    table coordinates (0..1); they are scaled to the renderer's 0..100 canvas.
    """

    obj = _xy(object_xy)
    occ = [round(float(v) * _SVG_SCALE, 4) for v in occluder]
    cam0 = _xy(camera)

    initial = {
        "type": "tabletop2d",
        "command": command,
        "target": "block",
        "agentState": "scan_for_object",
        "failure": "none",
        "object": obj,
        "occluder": occ,
        "camera": cam0,
        "detection": None,
        "pickAt": None,
        "holding": False,
        "belief": {"meanXY": None, "radius": None, "attempts": 0, "retries": 0, "policy": "scan"},
    }

    steps: list[dict[str, Any]] = []
    last_detection: list[float] | None = None
    for action, reward, info, obs in zip(
        trace.actions, trace.rewards, trace.infos, trace.observations
    ):
        detections = obs.get("detections") or []
        if detections:
            last_detection = _xy(detections[0]["position"])

        belief_mean = info.get("belief_mean")
        mean_xy = None if belief_mean is None else _xy(belief_mean)
        belief_radius = info.get("belief_radius")
        radius_svg = (
            None if belief_radius is None else round(float(belief_radius) * _SVG_SCALE, 3)
        )
        pick_position = info.get("pick_position")
        pick_at = None if pick_position is None else _xy(pick_position)
        holding = bool((obs.get("gripper") or {}).get("holding")) or bool(info.get("success"))
        attempts = int(info.get("attempts", 0))
        retries = int(info.get("retry_count", 0))
        failure = _failure_kind(info)
        action_type = info.get("action_type") or action.get("type", "noop")

        if action_type == "look":
            label = "look(scan)"
        elif action_type == "pick":
            label = f"pick(attempt {attempts})"
        else:
            label = action_type

        snapshot = {
            "type": "tabletop2d",
            "command": command,
            "target": "held" if holding else "block",
            "agentState": _pick_and_retry_state(info, mean_xy is not None),
            "failure": failure or "none",
            "object": obj,
            "occluder": occ,
            "camera": _xy(obs["camera"]) if obs.get("camera") is not None else cam0,
            "detection": None if holding else last_detection,
            "pickAt": pick_at,
            "holding": holding,
            "belief": {
                "meanXY": mean_xy,
                "radius": radius_svg,
                "attempts": attempts,
                "retries": retries,
                "policy": _pick_and_retry_policy(info, mean_xy is not None),
            },
        }
        steps.append(
            {
                "action": label,
                "reward": round(float(reward), 4),
                "failure": failure,
                "agentState": snapshot["agentState"],
                "snapshot": snapshot,
            }
        )

    return {
        "command": command,
        "totalSteps": len(steps),
        "initial": initial,
        "steps": steps,
    }


# --- "Beat the robot" challenge scoring -------------------------------------
#
# An agent is scored across many seeds, not one, so a policy that overfits a
# single lucky seed (e.g. dropping the retry schedule to grab the belief mean
# immediately) is exposed by a low success rate. This is the whole lesson: the
# retry/belief logic exists for robustness, not for one episode.


def score_pick_and_retry(run_agent: Any, agent_factory: Any, *, seeds: Any) -> dict[str, Any]:
    """Run ``agent_factory()`` over ``seeds`` and aggregate the trace summaries.

    ``run_agent`` is the example's own loop (passed in to keep this module
    decoupled from examples/); ``agent_factory`` is called once per seed for a
    fresh agent. Returns plain-JSON aggregate stats.
    """

    seeds = list(seeds)
    successes = 0
    steps_total = 0.0
    retries_total = 0.0
    reward_total = 0.0
    miss_total = 0.0
    for seed in seeds:
        summary = run_agent(agent_factory(), seed=seed, render=False).summary()
        if summary.success:
            successes += 1
        steps_total += summary.steps
        retries_total += int(summary.counters.get("retry_count", 0) or 0)
        reward_total += summary.total_reward
        miss_total += summary.failure_counts.get("grasp_miss", 0)

    n = len(seeds) or 1
    return {
        "episodes": len(seeds),
        "successes": successes,
        "success_rate": round(successes / n, 4),
        "mean_steps": round(steps_total / n, 3),
        "mean_retries": round(retries_total / n, 3),
        "mean_reward": round(reward_total / n, 3),
        "mean_grasp_miss": round(miss_total / n, 3),
    }
