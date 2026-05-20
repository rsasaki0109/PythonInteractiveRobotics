"""Search for a target object, remember it, pick it, and recover from a miss.

The target is not visible from the first viewpoints.  The agent must move the
camera, remember detections, choose the requested object instead of distractors,
attempt a grasp, detect that a low-confidence pose caused a miss, then move to a
better viewpoint and retry.
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


VIEWPOINTS: dict[str, np.ndarray] = {
    "home": np.array([0.18, 0.20], dtype=float),
    "top": np.array([0.22, 0.84], dtype=float),
    "right": np.array([0.86, 0.48], dtype=float),
    "close": np.array([0.82, 0.76], dtype=float),
}

DRAW_COLORS = {
    "red": "tab:red",
    "blue": "tab:blue",
    "yellow": "goldenrod",
}


@dataclass
class SearchObject:
    name: str
    color: str
    position: np.ndarray
    visible_from: tuple[str, ...]
    radius: float = 0.042
    picked: bool = False

    @property
    def key(self) -> str:
        return f"{self.color}:{self.name}"


class ObjectSearchPickWorld:
    """A small tabletop with distractors, occlusion, and viewpoint-dependent sensing."""

    def __init__(self, seed: int = 7, max_steps: int = 30) -> None:
        self.seed = seed
        self.max_steps = max_steps
        self.table_size = np.array([1.0, 1.0], dtype=float)
        self.occluder = np.array([0.48, 0.42, 0.66, 0.72], dtype=float)
        self.grasp_radius = 0.050
        self.rng = np.random.default_rng(seed)
        self._figure: Any | None = None
        self._axis: Any | None = None
        self.reset(seed=seed)

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            self.seed = seed
            self.rng = np.random.default_rng(seed)

        self.time = 0
        self.camera_name = "home"
        self.camera_pos = VIEWPOINTS[self.camera_name].copy()
        self.pick_attempts = 0
        self.search_failures = 0
        self.holding: str | None = None
        self.last_detections: list[dict[str, Any]] = []
        self.last_pick_position: np.ndarray | None = None
        self.objects = [
            SearchObject(
                "block",
                "red",
                np.array([0.73, 0.57], dtype=float),
                ("right", "close"),
            ),
            SearchObject(
                "block",
                "blue",
                np.array([0.33, 0.70], dtype=float),
                ("home", "top"),
            ),
            SearchObject(
                "cylinder",
                "yellow",
                np.array([0.54, 0.29], dtype=float),
                ("home", "right"),
            ),
        ]
        return self.observe()

    def observe(self) -> dict[str, Any]:
        detections: list[dict[str, Any]] = []
        for obj in self.objects:
            if obj.picked or self.camera_name not in obj.visible_from:
                continue

            confidence = self._confidence_for(obj)
            noise_scale = self._noise_for(obj)
            observed_position = np.clip(
                obj.position + self.rng.normal(0.0, noise_scale, size=2),
                [0.0, 0.0],
                self.table_size,
            )
            detections.append(
                {
                    "name": obj.name,
                    "color": obj.color,
                    "key": obj.key,
                    "position": observed_position,
                    "confidence": confidence,
                    "viewpoint": self.camera_name,
                }
            )

        self.last_detections = detections
        return {
            "time": self.time,
            "camera": self.camera_pos.copy(),
            "camera_name": self.camera_name,
            "detections": detections,
            "holding": self.holding,
            "pick_attempts": self.pick_attempts,
            "search_failures": self.search_failures,
        }

    def step(self, action: dict[str, Any]) -> StepResult:
        self.time += 1
        action_type = action.get("type", "look")
        info: dict[str, Any] = {
            "time": self.time,
            "action_type": action_type,
            "success": False,
            "pick_attempts": self.pick_attempts,
            "search_failures": self.search_failures,
        }

        if action_type == "look":
            return self._step_look(action, info)
        if action_type == "pick":
            return self._step_pick(action, info)

        info["failure"] = Failure(
            "invalid_action",
            f"unknown action type: {action_type}",
            recoverable=True,
        )
        return StepResult(self.observe(), -0.05, False, info)

    def _step_look(self, action: dict[str, Any], info: dict[str, Any]) -> StepResult:
        viewpoint = str(action.get("viewpoint", self.camera_name))
        if viewpoint not in VIEWPOINTS:
            info["failure"] = Failure(
                "invalid_viewpoint",
                f"unknown viewpoint: {viewpoint}",
                recoverable=True,
            )
            return StepResult(self.observe(), -0.05, False, info)

        self.camera_name = viewpoint
        self.camera_pos = VIEWPOINTS[viewpoint].copy()
        obs = self.observe()
        found_target = any(det["key"] == "red:block" for det in obs["detections"])
        info.update(
            {
                "viewpoint": viewpoint,
                "detections": len(obs["detections"]),
                "target_visible": found_target,
            }
        )
        if not found_target:
            self.search_failures += 1
            info["search_failures"] = self.search_failures
            info["failure"] = Failure(
                "target_not_visible",
                "the current viewpoint did not reveal the requested object",
                recoverable=True,
            )
        return StepResult(obs, 0.08 if found_target else -0.04, False, info)

    def _step_pick(self, action: dict[str, Any], info: dict[str, Any]) -> StepResult:
        raw_position = action.get("position")
        if raw_position is None:
            info["failure"] = Failure("invalid_action", "pick requires a position", True)
            return StepResult(self.observe(), -0.05, False, info)

        self.pick_attempts += 1
        pick_position = np.clip(np.asarray(raw_position, dtype=float), [0.0, 0.0], self.table_size)
        self.last_pick_position = pick_position.copy()
        target = self._target_object()
        error = float(np.linalg.norm(pick_position - target.position))
        target_confidence = self._last_target_confidence()
        high_confidence = target_confidence >= 0.85
        close_enough = error <= self.grasp_radius

        info.update(
            {
                "pick_attempts": self.pick_attempts,
                "pick_position": pick_position.copy(),
                "target_confidence": target_confidence,
                "grasp_error": error,
                "target_visible": target_confidence > 0.0,
            }
        )

        if close_enough and high_confidence:
            target.picked = True
            self.holding = target.key
            info["success"] = True
            return StepResult(self.observe(), 1.0, True, info)

        info["failure"] = Failure(
            "grasp_miss",
            "the target pose was too uncertain, so the gripper closed beside it",
            recoverable=True,
        )
        return StepResult(self.observe(), -0.16, False, info)

    def _target_object(self) -> SearchObject:
        for obj in self.objects:
            if obj.key == "red:block":
                return obj
        raise RuntimeError("target object missing")

    def _confidence_for(self, obj: SearchObject) -> float:
        if obj.key == "red:block" and self.camera_name == "right":
            return 0.62
        if obj.key == "red:block" and self.camera_name == "close":
            return 0.96
        return 0.84

    def _noise_for(self, obj: SearchObject) -> float:
        if obj.key == "red:block" and self.camera_name == "right":
            return 0.044
        if obj.key == "red:block" and self.camera_name == "close":
            return 0.010
        return 0.020

    def _last_target_confidence(self) -> float:
        for detection in self.last_detections:
            if detection["key"] == "red:block":
                return float(detection["confidence"])
        return 0.0

    def render(self, agent: "ObjectSearchPickAgent", info: dict[str, Any] | None = None) -> None:
        import matplotlib.pyplot as plt

        if self._figure is None or self._axis is None:
            plt.ion()
            self._figure, self._axis = plt.subplots(figsize=(5.2, 5.2))

        draw_object_search_pick_scene(self._axis, self, agent, info)
        self._figure.canvas.draw_idle()
        plt.pause(0.05)


class ObjectSearchPickAgent:
    """Search viewpoints, keep object memory, and retry after an uncertain pick."""

    def __init__(self, target_key: str = "red:block") -> None:
        self.target_key = target_key
        self.search_viewpoints = ["top", "right"]
        self.close_viewpoint = "close"
        self.reset()

    def reset(self) -> None:
        self.memory: dict[str, dict[str, Any]] = {}
        self.state = "search"
        self.next_view_index = 0
        self.retry_count = 0
        self.search_failure_count = 0
        self.needs_close_view = False
        self.last_pick_position: np.ndarray | None = None
        self._last_integrated_time: int | None = None

    def act(self, obs: dict[str, Any]) -> dict[str, Any]:
        self._integrate_observation(obs)

        if self.needs_close_view and obs.get("camera_name") != self.close_viewpoint:
            self.state = "move_to_confirming_view"
            return {"type": "look", "viewpoint": self.close_viewpoint}

        target = self.memory.get(self.target_key)
        if target is None:
            viewpoint = self.search_viewpoints[min(self.next_view_index, len(self.search_viewpoints) - 1)]
            self.next_view_index += int(self.next_view_index < len(self.search_viewpoints) - 1)
            self.state = "search_viewpoints"
            return {"type": "look", "viewpoint": viewpoint}

        confidence = float(target["confidence"])
        self.state = "pick_confirmed_target" if confidence >= 0.85 else "pick_uncertain_target"
        self.last_pick_position = np.asarray(target["position"], dtype=float).copy()
        return {
            "type": "pick",
            "key": self.target_key,
            "position": self.last_pick_position.copy(),
        }

    def update(self, obs: dict[str, Any], reward: float, info: dict[str, Any]) -> None:
        self._integrate_observation(obs)
        failure = info.get("failure")
        if isinstance(failure, Failure) and failure.kind == "target_not_visible":
            self.search_failure_count += 1
            self.state = "continue_search"
        elif isinstance(failure, Failure) and failure.kind == "grasp_miss":
            self.retry_count += 1
            self.needs_close_view = True
            self.state = "recover_with_close_view"
        elif info.get("success"):
            self.state = "holding_target"

        target = self.memory.get(self.target_key)
        info["agent_state"] = self.state
        info["memory_count"] = len(self.memory)
        info["retry_count"] = self.retry_count
        info["search_failure_count"] = self.search_failure_count
        info["target_in_memory"] = target is not None
        if target is not None:
            info["target_belief"] = np.asarray(target["position"], dtype=float).copy()
            info["target_confidence"] = float(target["confidence"])

    def _integrate_observation(self, obs: dict[str, Any]) -> None:
        obs_time = int(obs.get("time", -1))
        if obs_time == self._last_integrated_time:
            return
        self._last_integrated_time = obs_time

        for detection in obs.get("detections", []):
            key = str(detection["key"])
            position = np.asarray(detection["position"], dtype=float)
            confidence = float(detection["confidence"])
            previous = self.memory.get(key)
            if previous is None:
                self.memory[key] = {
                    "position": position.copy(),
                    "confidence": confidence,
                    "last_seen": obs_time,
                    "viewpoint": detection["viewpoint"],
                }
                continue

            previous_confidence = float(previous["confidence"])
            alpha = confidence / (confidence + previous_confidence)
            previous["position"] = (1.0 - alpha) * previous["position"] + alpha * position
            previous["confidence"] = max(previous_confidence * 0.92, confidence)
            previous["last_seen"] = obs_time
            previous["viewpoint"] = detection["viewpoint"]
            if key == self.target_key and confidence >= 0.85:
                self.needs_close_view = False


def draw_object_search_pick_scene(
    ax: Any,
    env: ObjectSearchPickWorld,
    agent: ObjectSearchPickAgent,
    info: dict[str, Any] | None = None,
) -> None:
    """Draw tabletop state, camera, detections, memory, and pick attempt."""

    from matplotlib.patches import Circle, Rectangle

    info = {} if info is None else info
    ax.clear()
    ax.set_title("object search and pick")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)

    xmin, ymin, xmax, ymax = env.occluder
    ax.add_patch(
        Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            color="0.25",
            alpha=0.18,
            label="occluder",
        )
    )

    for obj in env.objects:
        if obj.picked:
            continue
        ax.add_patch(
            Circle(
                obj.position,
                obj.radius,
                color=DRAW_COLORS.get(obj.color, obj.color),
                alpha=0.78,
            )
        )
        ax.text(obj.position[0], obj.position[1] - 0.075, obj.key, ha="center", fontsize=8)

    for name, position in VIEWPOINTS.items():
        marker = "s" if name == env.camera_name else "."
        alpha = 1.0 if name == env.camera_name else 0.35
        ax.plot(*position, marker=marker, color="tab:blue", alpha=alpha, markersize=8)
        ax.text(position[0], position[1] + 0.035, name, ha="center", fontsize=8)

    for detection in env.last_detections:
        position = detection["position"]
        ax.plot(*position, marker="x", markersize=10, color="tab:orange")
        ax.plot(
            [env.camera_pos[0], position[0]],
            [env.camera_pos[1], position[1]],
            color="tab:orange",
            alpha=0.25,
        )

    target_memory = agent.memory.get(agent.target_key)
    if target_memory is not None:
        center = np.asarray(target_memory["position"], dtype=float)
        radius = max(0.025, 0.12 * (1.0 - float(target_memory["confidence"])))
        ax.add_patch(
            Circle(
                center,
                radius,
                fill=False,
                linestyle="--",
                linewidth=2,
                color="tab:green",
                label="target memory",
            )
        )

    pick_position = info.get("pick_position", env.last_pick_position)
    if pick_position is not None:
        ax.plot(*pick_position, marker="+", markersize=14, color="black", label="pick")

    status = (
        f"step={env.time}  state={info.get('agent_state', agent.state)}\n"
        f"view={env.camera_name}  memory={len(agent.memory)}  "
        f"search_failures={agent.search_failure_count}\n"
        f"retries={agent.retry_count}  holding={env.holding}"
    )
    if "failure" in info:
        status += f"  failure={info['failure'].kind}"
    if info.get("success"):
        status += "  success"
    ax.text(0.02, 0.98, status, transform=ax.transAxes, va="top", fontsize=9)
    ax.legend(loc="lower left", fontsize=8)


def run(seed: int = 7, render: bool = True, max_steps: int = 30) -> Trace:
    env = ObjectSearchPickWorld(seed=seed, max_steps=max_steps)
    agent = ObjectSearchPickAgent()
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
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    trace = run(seed=args.seed, render=not args.no_render, max_steps=args.max_steps)
    final_info = trace.infos[-1] if trace.infos else {}
    failures = [failure.kind for failure in trace.failures()]
    print(
        f"success={bool(final_info.get('success'))} steps={len(trace.actions)} "
        f"memory={final_info.get('memory_count', 0)} "
        f"search_failures={final_info.get('search_failure_count', 0)} "
        f"retries={final_info.get('retry_count', 0)} "
        f"target_confidence={final_info.get('target_confidence', 0.0):.2f} "
        f"failures={failures}"
    )

    if not args.no_render:
        import matplotlib.pyplot as plt

        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
