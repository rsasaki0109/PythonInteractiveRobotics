"""Sort objects with probabilistic suction, retry, and belief updates.

The robot sees several objects and bins.  A suction pick can fail even when the
pose is correct, so the agent keeps a success-probability estimate per object,
updates it after failures, prepares the suction seal, retries, and sorts each
object into its color bin.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pir.core.types import Failure, StepResult, Trace


DRAW_COLORS = {
    "red": "tab:red",
    "blue": "tab:blue",
    "yellow": "goldenrod",
}

BIN_POSITIONS = {
    "red": np.array([0.22, 0.18], dtype=float),
    "yellow": np.array([0.50, 0.18], dtype=float),
    "blue": np.array([0.78, 0.18], dtype=float),
}


@dataclass
class SortObject:
    key: str
    color: str
    position: np.ndarray
    base_success: float
    radius: float = 0.040
    sorted: bool = False
    held: bool = False
    failures: int = 0
    prepared: bool = False


class ProbabilisticSuctionSortingWorld:
    """A tiny sorting table with stochastic suction outcomes."""

    def __init__(self, seed: int = 11, max_steps: int = 40) -> None:
        self.seed = seed
        self.max_steps = max_steps
        self.table_size = np.array([1.0, 1.0], dtype=float)
        self.suction_radius = 0.055
        self.outcome_rng = np.random.default_rng(seed)
        self._figure: Any | None = None
        self._axis: Any | None = None
        self.reset(seed=seed)

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            self.seed = seed
            self.outcome_rng = np.random.default_rng(seed)

        self.time = 0
        self.holding: str | None = None
        self.suction_attempts = 0
        self.suction_failures = 0
        self.prepare_count = 0
        self.place_count = 0
        self.last_pick_position: np.ndarray | None = None
        self.last_place_position: np.ndarray | None = None
        self.last_outcome_sample: float | None = None
        self.last_prepared_key: str | None = None
        self.objects = [
            SortObject("red:block", "red", np.array([0.26, 0.68], dtype=float), 0.86),
            SortObject("yellow:block", "yellow", np.array([0.50, 0.70], dtype=float), 0.82),
            SortObject("blue:block", "blue", np.array([0.74, 0.68], dtype=float), 0.42),
        ]
        return self.observe()

    def observe(self) -> dict[str, Any]:
        detections: list[dict[str, Any]] = []
        for obj in self.objects:
            if obj.sorted or obj.held:
                continue
            detections.append(
                {
                    "key": obj.key,
                    "color": obj.color,
                    "position": obj.position.copy(),
                    "confidence": 0.94,
                    "prepared": obj.prepared,
                }
            )

        return {
            "time": self.time,
            "detections": detections,
            "holding": self.holding,
            "sorted_keys": [obj.key for obj in self.objects if obj.sorted],
            "bins": {color: position.copy() for color, position in BIN_POSITIONS.items()},
            "suction_attempts": self.suction_attempts,
            "suction_failures": self.suction_failures,
            "prepare_count": self.prepare_count,
            "place_count": self.place_count,
        }

    def step(self, action: dict[str, Any]) -> StepResult:
        self.time += 1
        action_type = action.get("type", "observe")
        info: dict[str, Any] = {
            "time": self.time,
            "action_type": action_type,
            "success": False,
            "holding": self.holding,
            "sorted_count": self.sorted_count(),
            "suction_attempts": self.suction_attempts,
            "suction_failures": self.suction_failures,
            "prepare_count": self.prepare_count,
        }

        if action_type == "observe":
            return StepResult(self.observe(), -0.01, False, info)
        if action_type == "prepare_suction":
            return self._step_prepare(action, info)
        if action_type == "suction_pick":
            return self._step_pick(action, info)
        if action_type == "place":
            return self._step_place(action, info)

        info["failure"] = Failure(
            "invalid_action",
            f"unknown action type: {action_type}",
            recoverable=True,
        )
        return StepResult(self.observe(), -0.05, False, info)

    def _step_prepare(self, action: dict[str, Any], info: dict[str, Any]) -> StepResult:
        obj = self._object_by_key(str(action.get("key", "")))
        if obj is None or obj.sorted:
            info["failure"] = Failure("invalid_target", "cannot prepare missing object", True)
            return StepResult(self.observe(), -0.05, False, info)

        obj.prepared = True
        self.prepare_count += 1
        self.last_prepared_key = obj.key
        info.update(
            {
                "prepared_key": obj.key,
                "prepare_count": self.prepare_count,
                "estimated_success_boost": 0.30,
            }
        )
        return StepResult(self.observe(), 0.04, False, info)

    def _step_pick(self, action: dict[str, Any], info: dict[str, Any]) -> StepResult:
        if self.holding is not None:
            info["failure"] = Failure("already_holding", "place held object before picking", True)
            return StepResult(self.observe(), -0.05, False, info)

        obj = self._object_by_key(str(action.get("key", "")))
        if obj is None or obj.sorted:
            info["failure"] = Failure("invalid_target", "cannot pick missing object", True)
            return StepResult(self.observe(), -0.05, False, info)

        pick_position = np.clip(
            np.asarray(action.get("position", obj.position), dtype=float),
            [0.0, 0.0],
            self.table_size,
        )
        self.last_pick_position = pick_position.copy()
        self.suction_attempts += 1
        error = float(np.linalg.norm(pick_position - obj.position))
        probability = self._success_probability(obj, error)
        sample = float(self.outcome_rng.random())
        self.last_outcome_sample = sample
        picked = sample < probability

        info.update(
            {
                "target_key": obj.key,
                "pick_position": pick_position.copy(),
                "suction_attempts": self.suction_attempts,
                "suction_probability": probability,
                "outcome_sample": sample,
                "alignment_error": error,
            }
        )

        if picked:
            obj.held = True
            self.holding = obj.key
            info["pick_success"] = True
            return StepResult(self.observe(), 0.35, False, info)

        obj.failures += 1
        self.suction_failures += 1
        info["suction_failures"] = self.suction_failures
        info["failure"] = Failure(
            "suction_miss",
            "suction seal failed even though the target was observed",
            recoverable=True,
        )
        return StepResult(self.observe(), -0.14, False, info)

    def _step_place(self, action: dict[str, Any], info: dict[str, Any]) -> StepResult:
        if self.holding is None:
            info["failure"] = Failure("empty_gripper", "nothing is held for placing", True)
            return StepResult(self.observe(), -0.05, False, info)

        obj = self._object_by_key(self.holding)
        if obj is None:
            raise RuntimeError("held object missing")

        bin_position = BIN_POSITIONS[obj.color]
        requested_position = np.clip(
            np.asarray(action.get("position", bin_position), dtype=float),
            [0.0, 0.0],
            self.table_size,
        )
        place_error = float(np.linalg.norm(requested_position - bin_position))
        self.last_place_position = requested_position.copy()
        self.place_count += 1
        obj.position = bin_position.copy()
        obj.sorted = True
        obj.held = False
        self.holding = None
        done = self.sorted_count() == len(self.objects)
        info.update(
            {
                "placed_key": obj.key,
                "place_position": requested_position.copy(),
                "place_error": place_error,
                "place_count": self.place_count,
                "sorted_count": self.sorted_count(),
                "success": done,
            }
        )
        return StepResult(self.observe(), 1.0 if done else 0.25, done, info)

    def _success_probability(self, obj: SortObject, alignment_error: float) -> float:
        alignment = max(0.0, 1.0 - alignment_error / (self.suction_radius * 1.6))
        prepared_bonus = 0.30 if obj.prepared else 0.0
        failure_bonus = min(0.16, 0.08 * obj.failures)
        raw = obj.base_success + prepared_bonus + failure_bonus
        return float(np.clip(raw * (0.30 + 0.70 * alignment), 0.0, 0.98))

    def _object_by_key(self, key: str) -> SortObject | None:
        for obj in self.objects:
            if obj.key == key:
                return obj
        return None

    def sorted_count(self) -> int:
        return sum(obj.sorted for obj in self.objects)

    def render(self, agent: "ProbabilisticSuctionSortingAgent", info: dict[str, Any] | None = None) -> None:
        import matplotlib.pyplot as plt

        if self._figure is None or self._axis is None:
            plt.ion()
            self._figure, self._axis = plt.subplots(figsize=(5.4, 5.2))

        draw_suction_sorting_scene(self._axis, self, agent, info)
        self._figure.canvas.draw_idle()
        plt.pause(0.05)


class ProbabilisticSuctionSortingAgent:
    """Sort by color while updating per-object suction success estimates."""

    def __init__(self) -> None:
        self.priors = {
            "red:block": 0.86,
            "yellow:block": 0.82,
            "blue:block": 0.42,
        }
        self.reset()

    def reset(self) -> None:
        self.memory: dict[str, dict[str, Any]] = {}
        self.success_estimates = dict(self.priors)
        self.sorted_keys: set[str] = set()
        self.holding: str | None = None
        self.failed_key: str | None = None
        self.needs_prepare = False
        self.retry_count = 0
        self.failure_count = 0
        self.prepare_count = 0
        self.state = "observe_objects"
        self._last_integrated_time: int | None = None

    def act(self, obs: dict[str, Any]) -> dict[str, Any]:
        self._integrate_observation(obs)
        self.holding = obs.get("holding")

        if self.holding is not None:
            color = self.holding.split(":", 1)[0]
            self.state = "place_in_color_bin"
            return {"type": "place", "key": self.holding, "position": BIN_POSITIONS[color]}

        if self.needs_prepare and self.failed_key is not None:
            self.state = "prepare_suction_retry"
            return {"type": "prepare_suction", "key": self.failed_key}

        target_key = self._choose_target()
        if target_key is None:
            self.state = "observe_objects"
            return {"type": "observe"}

        target = self.memory[target_key]
        self.state = "suction_pick"
        return {
            "type": "suction_pick",
            "key": target_key,
            "position": np.asarray(target["position"], dtype=float).copy(),
        }

    def update(self, obs: dict[str, Any], reward: float, info: dict[str, Any]) -> None:
        self._integrate_observation(obs)

        failure = info.get("failure")
        if isinstance(failure, Failure) and failure.kind == "suction_miss":
            key = str(info["target_key"])
            self.failed_key = key
            self.needs_prepare = True
            self.retry_count += 1
            self.failure_count += 1
            self.success_estimates[key] = min(0.92, self.success_estimates[key] + 0.30)
            self.state = "suction_failed_update_estimate"
        elif info.get("action_type") == "prepare_suction":
            key = str(info["prepared_key"])
            self.failed_key = key
            self.needs_prepare = False
            self.prepare_count += 1
            self.success_estimates[key] = min(0.96, self.success_estimates[key] + 0.18)
            self.state = "retry_ready"
        elif info.get("pick_success"):
            self.holding = str(info["target_key"])
            self.failed_key = None
            self.needs_prepare = False
            self.state = "holding_object"
        elif "placed_key" in info:
            key = str(info["placed_key"])
            self.sorted_keys.add(key)
            self.memory.pop(key, None)
            self.state = "sorting_done" if info.get("success") else "choose_next_object"

        info["agent_state"] = self.state
        info["retry_count"] = self.retry_count
        info["failure_count"] = self.failure_count
        info["prepare_count"] = self.prepare_count
        info["memory_count"] = len(self.memory)
        info["sorted_count"] = len(self.sorted_keys)
        info["success_estimates"] = dict(self.success_estimates)

    def _integrate_observation(self, obs: dict[str, Any]) -> None:
        obs_time = int(obs.get("time", -1))
        if obs_time == self._last_integrated_time:
            return
        self._last_integrated_time = obs_time

        for key in obs.get("sorted_keys", []):
            self.sorted_keys.add(str(key))
            self.memory.pop(str(key), None)

        for detection in obs.get("detections", []):
            key = str(detection["key"])
            if key in self.sorted_keys:
                continue
            self.memory[key] = {
                "position": np.asarray(detection["position"], dtype=float).copy(),
                "color": detection["color"],
                "confidence": float(detection["confidence"]),
                "prepared": bool(detection.get("prepared", False)),
                "last_seen": obs_time,
            }
            self.success_estimates.setdefault(key, 0.65)

    def _choose_target(self) -> str | None:
        candidates = [key for key in self.memory if key not in self.sorted_keys]
        if not candidates:
            return None
        return max(candidates, key=lambda key: self.success_estimates.get(key, 0.0))


def draw_suction_sorting_scene(
    ax: Any,
    env: ProbabilisticSuctionSortingWorld,
    agent: ProbabilisticSuctionSortingAgent,
    info: dict[str, Any] | None = None,
) -> None:
    """Draw objects, bins, suction attempts, and success estimates."""

    from matplotlib.patches import Circle, Rectangle

    info = {} if info is None else info
    ax.clear()
    ax.set_title("probabilistic suction sorting")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)

    for color, position in BIN_POSITIONS.items():
        ax.add_patch(
            Rectangle(
                (position[0] - 0.09, position[1] - 0.06),
                0.18,
                0.12,
                color=DRAW_COLORS[color],
                alpha=0.16,
            )
        )
        ax.text(position[0], position[1] - 0.095, f"{color} bin", ha="center", fontsize=8)

    for obj in env.objects:
        if obj.held:
            draw_position = np.array([0.50, 0.44], dtype=float)
        else:
            draw_position = obj.position
        alpha = 0.45 if obj.sorted else 0.85
        ax.add_patch(
            Circle(
                draw_position,
                obj.radius,
                color=DRAW_COLORS[obj.color],
                alpha=alpha,
            )
        )
        estimate = agent.success_estimates.get(obj.key, obj.base_success)
        label = f"{obj.key}\np={estimate:.2f}"
        if obj.prepared and not obj.sorted:
            label += "\nprepared"
        ax.text(draw_position[0], draw_position[1] + 0.060, label, ha="center", fontsize=8)

    if env.last_pick_position is not None:
        ax.plot(*env.last_pick_position, marker="+", markersize=14, color="black", label="suction pick")
    if env.last_place_position is not None:
        ax.plot(*env.last_place_position, marker="v", markersize=9, color="black", label="place")

    if env.holding is not None:
        ax.plot(0.50, 0.50, marker="s", color="0.2", markersize=8)
        ax.text(0.50, 0.52, f"holding {env.holding}", ha="center", fontsize=8)

    status = (
        f"step={env.time}  state={info.get('agent_state', agent.state)}\n"
        f"sorted={env.sorted_count()}/{len(env.objects)}  failures={agent.failure_count}  "
        f"retries={agent.retry_count}  prepared={agent.prepare_count}\n"
        f"attempts={env.suction_attempts}"
    )
    if env.last_outcome_sample is not None:
        status += f"  sample={env.last_outcome_sample:.2f}"
    if "suction_probability" in info:
        status += f"  p={info['suction_probability']:.2f}"
    if "failure" in info:
        status += f"  failure={info['failure'].kind}"
    if info.get("success"):
        status += "  success"
    ax.text(0.02, 0.98, status, transform=ax.transAxes, va="top", fontsize=9)
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="lower left", fontsize=8)


def run(seed: int = 11, render: bool = True, max_steps: int = 40) -> Trace:
    env = ProbabilisticSuctionSortingWorld(seed=seed, max_steps=max_steps)
    agent = ProbabilisticSuctionSortingAgent()
    obs = env.reset(seed=seed)
    agent.reset()
    trace = Trace()

    for _ in range(max_steps):
        action = agent.act(obs)
        result = env.step(action)
        obs, reward, done, info = result.as_tuple()
        agent.update(obs, reward, info)
        trace.append(obs, action, reward, info)

        if render:
            env.render(agent, info)

        if done:
            break

    return trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    trace = run(seed=args.seed, render=not args.no_render, max_steps=args.max_steps)
    final_info = trace.infos[-1] if trace.infos else {}
    failures = [failure.kind for failure in trace.failures()]
    print(
        f"success={bool(final_info.get('success'))} steps={len(trace.actions)} "
        f"sorted={final_info.get('sorted_count', 0)} "
        f"failures={final_info.get('failure_count', 0)} "
        f"retries={final_info.get('retry_count', 0)} "
        f"prepared={final_info.get('prepare_count', 0)} "
        f"failures={failures}"
    )

    if not args.no_render:
        import matplotlib.pyplot as plt

        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
