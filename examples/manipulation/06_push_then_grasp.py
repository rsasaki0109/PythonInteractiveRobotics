"""Push an object out of a blocked pose, then grasp it.

This example treats pushing as a way to change the world before grasping.  The
target is visible, but it starts under a shelf where the gripper cannot close.
The robot first tries to pick, detects a blocked grasp, pushes the target into
open space, observes the changed state, and picks again.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pir.core.types import Failure, StepResult, Trace


class PushThenGraspWorld:
    """A tiny tabletop where grasping requires a prior push."""

    def __init__(self, seed: int = 9, max_steps: int = 25) -> None:
        self.seed = seed
        self.max_steps = max_steps
        self.table_size = np.array([1.0, 1.0], dtype=float)
        self.target_radius = 0.042
        self.grasp_radius = 0.055
        self.shelf = np.array([0.42, 0.55, 0.66, 0.70], dtype=float)
        self.detector_noise = 0.012
        self.rng = np.random.default_rng(seed)
        self._figure: Any | None = None
        self._axis: Any | None = None
        self.reset(seed=seed)

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            self.seed = seed
            self.rng = np.random.default_rng(seed)

        self.time = 0
        self.target_position = np.array([0.54, 0.62], dtype=float)
        self.target_picked = False
        self.holding: str | None = None
        self.pick_attempts = 0
        self.push_attempts = 0
        self.environment_changes = 0
        self.last_detection: np.ndarray | None = None
        self.last_pick_position: np.ndarray | None = None
        self.last_push_start: np.ndarray | None = None
        self.last_push_end: np.ndarray | None = None
        self.target_path = [self.target_position.copy()]
        return self.observe()

    def observe(self) -> dict[str, Any]:
        detections: list[dict[str, Any]] = []
        if not self.target_picked:
            detection = np.clip(
                self.target_position + self.rng.normal(0.0, self.detector_noise, size=2),
                [0.0, 0.0],
                self.table_size,
            )
            self.last_detection = detection.copy()
            detections.append(
                {
                    "name": "block",
                    "color": "red",
                    "key": "red:block",
                    "position": detection,
                    "confidence": 0.90,
                    "blocked": self.grasp_blocked(),
                }
            )

        return {
            "time": self.time,
            "detections": detections,
            "holding": self.holding,
            "grasp_blocked": self.grasp_blocked(),
            "pick_attempts": self.pick_attempts,
            "push_attempts": self.push_attempts,
            "environment_changes": self.environment_changes,
        }

    def step(self, action: dict[str, Any]) -> StepResult:
        self.time += 1
        action_type = action.get("type", "look")
        info: dict[str, Any] = {
            "time": self.time,
            "action_type": action_type,
            "success": False,
            "grasp_blocked": self.grasp_blocked(),
            "push_attempts": self.push_attempts,
            "pick_attempts": self.pick_attempts,
            "environment_changes": self.environment_changes,
        }

        if action_type == "look":
            return StepResult(self.observe(), -0.01, False, info)
        if action_type == "pick":
            return self._step_pick(action, info)
        if action_type == "push":
            return self._step_push(action, info)

        info["failure"] = Failure(
            "invalid_action",
            f"unknown action type: {action_type}",
            recoverable=True,
        )
        return StepResult(self.observe(), -0.05, False, info)

    def _step_pick(self, action: dict[str, Any], info: dict[str, Any]) -> StepResult:
        raw_position = action.get("position")
        if raw_position is None:
            info["failure"] = Failure("invalid_action", "pick requires a position", True)
            return StepResult(self.observe(), -0.05, False, info)

        pick_position = np.clip(np.asarray(raw_position, dtype=float), [0.0, 0.0], self.table_size)
        self.last_pick_position = pick_position.copy()
        self.pick_attempts += 1
        error = float(np.linalg.norm(pick_position - self.target_position))
        blocked = self.grasp_blocked()
        info.update(
            {
                "pick_position": pick_position.copy(),
                "pick_attempts": self.pick_attempts,
                "grasp_error": error,
                "grasp_blocked": blocked,
            }
        )

        if blocked:
            info["failure"] = Failure(
                "blocked_grasp",
                "target is visible but the shelf blocks gripper closure",
                recoverable=True,
            )
            return StepResult(self.observe(), -0.14, False, info)

        if error <= self.grasp_radius:
            self.target_picked = True
            self.holding = "red:block"
            info["success"] = True
            return StepResult(self.observe(), 1.0, True, info)

        info["failure"] = Failure(
            "grasp_miss",
            "gripper closed away from the target",
            recoverable=True,
        )
        return StepResult(self.observe(), -0.16, False, info)

    def _step_push(self, action: dict[str, Any], info: dict[str, Any]) -> StepResult:
        self.push_attempts += 1
        start = np.clip(np.asarray(action.get("start", self.target_position), dtype=float), [0, 0], self.table_size)
        direction = np.asarray(action.get("direction", [1.0, -0.7]), dtype=float)
        norm = float(np.linalg.norm(direction))
        if norm == 0.0:
            info["failure"] = Failure("invalid_push", "push direction must be nonzero", True)
            return StepResult(self.observe(), -0.05, False, info)

        direction = direction / norm
        distance = float(action.get("distance", 0.23))
        displacement = direction * np.clip(distance, 0.0, 0.28)
        previous = self.target_position.copy()
        self.target_position = np.clip(self.target_position + displacement, [0.08, 0.08], [0.92, 0.92])
        self.target_path.append(self.target_position.copy())
        self.environment_changes += 1
        self.last_push_start = start.copy()
        self.last_push_end = self.target_position.copy()

        info.update(
            {
                "push_attempts": self.push_attempts,
                "environment_changed": True,
                "environment_changes": self.environment_changes,
                "object_displacement": float(np.linalg.norm(self.target_position - previous)),
                "push_start": start.copy(),
                "push_end": self.target_position.copy(),
                "grasp_blocked": self.grasp_blocked(),
            }
        )
        return StepResult(self.observe(), 0.18, False, info)

    def grasp_blocked(self) -> bool:
        if self.target_picked:
            return False
        xmin, ymin, xmax, ymax = self.shelf
        x, y = self.target_position
        return bool(xmin <= x <= xmax and ymin <= y <= ymax)

    def render(self, agent: "PushThenGraspAgent", info: dict[str, Any] | None = None) -> None:
        import matplotlib.pyplot as plt

        if self._figure is None or self._axis is None:
            plt.ion()
            self._figure, self._axis = plt.subplots(figsize=(5.2, 5.2))

        draw_push_then_grasp_scene(self._axis, self, agent, info)
        self._figure.canvas.draw_idle()
        plt.pause(0.05)


class PushThenGraspAgent:
    """Try grasping, then push only after a blocked grasp is observed."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.target_belief: np.ndarray | None = None
        self.belief_radius = 0.08
        self.state = "observe_target"
        self.needs_push = False
        self.push_count = 0
        self.retry_count = 0
        self.blocked_count = 0
        self._last_integrated_time: int | None = None

    def act(self, obs: dict[str, Any]) -> dict[str, Any]:
        self._integrate_observation(obs)

        if self.target_belief is None:
            self.state = "look_for_target"
            return {"type": "look"}

        if self.needs_push:
            self.state = "push_to_change_world"
            return {
                "type": "push",
                "start": self.target_belief + np.array([-0.07, 0.05]),
                "direction": np.array([1.0, -0.72]),
                "distance": 0.24,
            }

        self.state = "try_grasp" if self.retry_count == 0 else "retry_grasp_after_push"
        return {"type": "pick", "position": self.target_belief.copy()}

    def update(self, obs: dict[str, Any], reward: float, info: dict[str, Any]) -> None:
        self._integrate_observation(obs)

        failure = info.get("failure")
        if isinstance(failure, Failure) and failure.kind == "blocked_grasp":
            self.blocked_count += 1
            self.retry_count += 1
            self.needs_push = True
            self.state = "blocked_grasp_detected"
        elif info.get("action_type") == "push":
            self.push_count += 1
            self.needs_push = False
            self.state = "reobserve_after_push"
            self.belief_radius = min(0.12, self.belief_radius + 0.04)
        elif info.get("success"):
            self.state = "holding_target"

        info["agent_state"] = self.state
        info["push_count"] = self.push_count
        info["retry_count"] = self.retry_count
        info["blocked_count"] = self.blocked_count
        info["belief_radius"] = self.belief_radius
        if self.target_belief is not None:
            info["target_belief"] = self.target_belief.copy()

    def _integrate_observation(self, obs: dict[str, Any]) -> None:
        obs_time = int(obs.get("time", -1))
        if obs_time == self._last_integrated_time:
            return
        self._last_integrated_time = obs_time

        detections = obs.get("detections", [])
        if not detections:
            return

        detection = detections[0]
        position = np.asarray(detection["position"], dtype=float)
        confidence = float(detection.get("confidence", 0.5))
        if self.target_belief is None:
            self.target_belief = position.copy()
        else:
            alpha = np.clip(0.35 + 0.45 * confidence, 0.35, 0.82)
            self.target_belief = (1.0 - alpha) * self.target_belief + alpha * position
        self.belief_radius = max(0.028, self.belief_radius * 0.72)


def draw_push_then_grasp_scene(
    ax: Any,
    env: PushThenGraspWorld,
    agent: PushThenGraspAgent,
    info: dict[str, Any] | None = None,
) -> None:
    """Draw shelf blockage, target displacement, belief, push, and pick."""

    from matplotlib.patches import Circle, Rectangle

    info = {} if info is None else info
    ax.clear()
    ax.set_title("push then grasp")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)

    xmin, ymin, xmax, ymax = env.shelf
    ax.add_patch(
        Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            color="0.20",
            alpha=0.20,
            label="shelf blocks grasp",
        )
    )

    target_path = np.asarray(env.target_path)
    if len(target_path) > 1:
        ax.plot(target_path[:, 0], target_path[:, 1], color="tab:red", linewidth=2, alpha=0.45)

    if not env.target_picked:
        ax.add_patch(
            Circle(
                env.target_position,
                env.target_radius,
                color="tab:red",
                alpha=0.82,
                label="target",
            )
        )

    if env.last_detection is not None and not env.target_picked:
        ax.plot(*env.last_detection, marker="x", markersize=10, color="tab:orange")

    if agent.target_belief is not None:
        ax.add_patch(
            Circle(
                agent.target_belief,
                agent.belief_radius,
                fill=False,
                linestyle="--",
                linewidth=2,
                color="tab:green",
                label="target belief",
            )
        )

    if env.last_push_start is not None and env.last_push_end is not None:
        delta = env.last_push_end - env.last_push_start
        ax.arrow(
            env.last_push_start[0],
            env.last_push_start[1],
            delta[0],
            delta[1],
            color="black",
            width=0.004,
            head_width=0.020,
            length_includes_head=True,
            alpha=0.70,
            label="push",
        )

    if env.last_pick_position is not None:
        ax.plot(*env.last_pick_position, marker="+", markersize=14, color="black", label="pick")

    status = (
        f"step={env.time}  state={info.get('agent_state', agent.state)}\n"
        f"blocked={env.grasp_blocked()}  pushes={agent.push_count}  "
        f"retries={agent.retry_count}\n"
        f"env_changes={env.environment_changes}  holding={env.holding}"
    )
    if "failure" in info:
        status += f"  failure={info['failure'].kind}"
    if info.get("success"):
        status += "  success"
    ax.text(0.02, 0.98, status, transform=ax.transAxes, va="top", fontsize=9)
    ax.legend(loc="lower left", fontsize=8)


def run(seed: int = 9, render: bool = True, max_steps: int = 25) -> Trace:
    env = PushThenGraspWorld(seed=seed, max_steps=max_steps)
    agent = PushThenGraspAgent()
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
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    trace = run(seed=args.seed, render=not args.no_render, max_steps=args.max_steps)
    final_info = trace.infos[-1] if trace.infos else {}
    failures = [failure.kind for failure in trace.failures()]
    print(
        f"success={bool(final_info.get('success'))} steps={len(trace.actions)} "
        f"pushes={final_info.get('push_count', 0)} "
        f"retries={final_info.get('retry_count', 0)} "
        f"blocked={final_info.get('blocked_count', 0)} "
        f"env_changes={final_info.get('environment_changes', 0)} "
        f"failures={failures}"
    )

    if not args.no_render:
        import matplotlib.pyplot as plt

        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
