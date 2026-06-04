"""Pin the trace-to-playground JSON contract shared by Python and JS.

docs/playground.js reads a fixed set of fields off each snapshot. If the Python
serializer (pir/viz/playground_trace.py) stops emitting one of them, the browser
would silently render a blank/garbled scene. These tests run the real clarifying
loop, serialize it, and assert the renderer's contract holds — including that the
output is plain JSON (no numpy leaks that would break json.dumps in Pyodide).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from pir.viz.playground_trace import (
    clarifying_trace_to_playground,
    pick_and_retry_trace_to_playground,
)
from pir.worlds.tabletop_2d import Tabletop2D

ROOT = Path(__file__).resolve().parents[1]

# Fields docs/playground.js reads off a tabletop snapshot (renderTabletop +
# renderBelief + the status strip). Keep in sync with the renderer.
SNAPSHOT_FIELDS = {
    "type",
    "command",
    "target",
    "agentState",
    "failure",
    "belief",
    "picked",
    "pickAt",
    "focus",
    "question",
    "answer",
}
BELIEF_FIELDS = {"red", "blue", "entropy", "askGain", "policy"}
STEP_FIELDS = {"action", "reward", "failure", "agentState", "snapshot"}


def _run(answer: str):
    path = ROOT / "examples" / "embodied_ai" / "35_clarifying_question.py"
    spec = importlib.util.spec_from_file_location("clarifying_question_contract", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run(command="pick the block", answer=answer, render=False)


def test_config_shape_matches_renderer_contract() -> None:
    config = clarifying_trace_to_playground(_run("red"), answer="red")

    assert config["command"] == "pick the block"
    assert config["totalSteps"] == len(config["steps"]) == 3

    initial = config["initial"]
    assert set(initial) == SNAPSHOT_FIELDS
    assert initial["type"] == "tabletop"
    assert initial["target"] == "unresolved"
    assert initial["belief"]["policy"] == "ask"
    assert abs(initial["belief"]["entropy"] - 1.0) < 1e-9

    for step in config["steps"]:
        assert set(step) == STEP_FIELDS
        snapshot = step["snapshot"]
        assert set(snapshot) == SNAPSHOT_FIELDS
        assert set(snapshot["belief"]) == BELIEF_FIELDS


def test_real_loop_resolves_and_picks_red() -> None:
    config = clarifying_trace_to_playground(_run("red"), answer="red")
    steps = config["steps"]

    # ask -> look -> pick, the real loop's three steps.
    assert [step["action"] for step in steps] == [
        "ask(which_block)",
        "look(red)",
        "pick(red)",
    ]
    assert steps[0]["failure"] == "ambiguous_goal"
    assert steps[0]["snapshot"]["target"] == "red"
    assert steps[0]["snapshot"]["belief"]["policy"] == "act"

    final = steps[-1]
    assert final["agentState"] == "done"
    assert final["snapshot"]["picked"] == "red"
    assert final["snapshot"]["pickAt"] == [32.0, 56.0]


def test_answer_blue_resolves_blue() -> None:
    config = clarifying_trace_to_playground(_run("blue"), answer="blue")
    final = config["steps"][-1]
    assert final["action"] == "pick(blue)"
    assert final["snapshot"]["picked"] == "blue"
    assert final["snapshot"]["pickAt"] == [68.0, 56.0]


def test_config_is_plain_json() -> None:
    # Pyodide returns this via json.dumps; a stray numpy array would raise here.
    config = clarifying_trace_to_playground(_run("red"), answer="red")
    reparsed = json.loads(json.dumps(config))
    assert reparsed == config


# --- pick_and_retry (continuous tabletop) -----------------------------------

TABLETOP2D_FIELDS = {
    "type",
    "command",
    "target",
    "agentState",
    "failure",
    "object",
    "occluder",
    "camera",
    "detection",
    "pickAt",
    "holding",
    "belief",
}
SPATIAL_BELIEF_FIELDS = {"meanXY", "radius", "attempts", "retries", "policy"}


def _load_pick_and_retry():
    path = ROOT / "examples" / "manipulation" / "01_pick_and_retry.py"
    spec = importlib.util.spec_from_file_location("pick_and_retry_contract", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_pick_and_retry(seed: int = 3):
    return _load_pick_and_retry().run(seed=seed, render=False)


def _geom(seed: int = 3) -> dict:
    world = Tabletop2D(seed=seed)
    return {
        "object_xy": list(map(float, world.obj.position)),
        "occluder": [float(v) for v in world.occluder],
        "camera": list(map(float, world.camera_pos)),
    }


def _pick_and_retry_config(seed: int = 3):
    return pick_and_retry_trace_to_playground(_run_pick_and_retry(seed), **_geom(seed))


def test_pick_and_retry_shape_matches_tabletop2d_contract() -> None:
    config = _pick_and_retry_config()

    assert config["command"] == "pick the block"
    assert config["totalSteps"] == len(config["steps"]) >= 1
    assert set(config["initial"]) == TABLETOP2D_FIELDS
    assert config["initial"]["type"] == "tabletop2d"
    # occluder is the real Tabletop2D rectangle scaled to the 0..100 canvas.
    assert config["initial"]["occluder"] == [43.0, 42.0, 57.0, 68.0]

    for step in config["steps"]:
        assert set(step) == STEP_FIELDS
        snapshot = step["snapshot"]
        assert set(snapshot) == TABLETOP2D_FIELDS
        assert set(snapshot["belief"]) == SPATIAL_BELIEF_FIELDS


def test_pick_and_retry_misses_then_picks_and_belief_appears() -> None:
    config = _pick_and_retry_config(seed=3)
    steps = config["steps"]

    # seed=3 is the hero seed: at least one grasp_miss before the pick succeeds.
    assert any(step["failure"] == "grasp_miss" for step in steps)

    final = steps[-1]
    assert final["agentState"] == "done"
    assert final["snapshot"]["holding"] is True

    # Belief becomes a concrete spatial estimate once the object is detected.
    assert any(step["snapshot"]["belief"]["meanXY"] is not None for step in steps)
    # Retry count is non-decreasing and ends at >=1 (it missed at least once).
    assert final["snapshot"]["belief"]["retries"] >= 1


def test_pick_and_retry_config_is_plain_json() -> None:
    config = _pick_and_retry_config()
    assert json.loads(json.dumps(config)) == config


def test_run_agent_tolerates_a_custom_agent_for_the_edit_cell() -> None:
    """The Phase-3 edit cell execs a user-defined agent and runs it via
    run_agent(). A custom agent that drops the belief attributes must still run
    and serialize (belief simply renders as unknown)."""
    import numpy as np

    module = _load_pick_and_retry()

    class GreedyAgent:
        """Always grab the latest detection — no belief, no retry schedule."""

        def reset(self):
            self._last = None

        def act(self, obs):
            detections = obs.get("detections") or []
            if detections:
                self._last = np.asarray(detections[0]["position"], dtype=float)
            if self._last is None:
                return {"type": "look", "target": np.array([0.84, 0.52])}
            return {"type": "pick", "position": self._last}

        def update(self, obs, reward, info):
            detections = obs.get("detections") or []
            if detections:
                self._last = np.asarray(detections[0]["position"], dtype=float)

    trace = module.run_agent(GreedyAgent(), seed=3, render=False)
    config = pick_and_retry_trace_to_playground(trace, **_geom())

    assert config["totalSteps"] >= 1
    assert json.loads(json.dumps(config)) == config
    # No belief attributes -> belief radius is unknown for every step.
    assert all(step["snapshot"]["belief"]["radius"] is None for step in config["steps"])
