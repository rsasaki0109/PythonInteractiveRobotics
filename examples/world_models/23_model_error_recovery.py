"""Detect a model regime shift, identify the new dynamics, then resume.

The robot starts with a correct identity model: every action moves exactly as
commanded. At a fixed step the world's dynamics shift - a constant offset is
added to every executed action. Prediction error spikes above a threshold and
the agent transitions from goal navigation to a short system-identification
phase, runs a few probe actions, averages the observed offsets, updates its
learned offset, and then resumes goal navigation with the corrected model.

This differs from `20_tiny_world_model_planning.py` which fits a continuous
residual across a static drift region. Here the model is structurally wrong
for a moment, the agent must detect that, pay the cost of probing, and then
move on.
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

from pir.core.random import make_rng
from pir.core.types import Failure, StepResult, Trace


PROBE_ACTIONS: tuple[np.ndarray, ...] = (
    np.array([0.060, 0.000], dtype=float),
    np.array([0.000, 0.060], dtype=float),
    np.array([-0.060, 0.000], dtype=float),
)


class ModelErrorRecoveryWorld:
    """2D continuous world that changes dynamics partway through."""

    def __init__(
        self,
        *,
        seed: int | None = 0,
        max_steps: int = 50,
        start: tuple[float, float] = (0.10, 0.30),
        goal: tuple[float, float] = (0.85, 0.75),
        goal_radius: float = 0.06,
        move_max: float = 0.08,
        noise_sigma: float = 0.003,
        regime_shift_at: int = 4,
        shift_offset: tuple[float, float] = (0.030, 0.020),
        error_threshold: float = 0.020,
    ) -> None:
        self.seed = seed
        self.size = 1.0
        self.start = np.asarray(start, dtype=float)
        self.goal = np.asarray(goal, dtype=float)
        self.goal_radius = goal_radius
        self.move_max = move_max
        self.noise_sigma = noise_sigma
        self.regime_shift_at = regime_shift_at
        self.shift_offset = np.asarray(shift_offset, dtype=float)
        self.error_threshold = error_threshold
        self.max_steps = max_steps
        self.rng = make_rng(seed)
        self._fig = None
        self._ax = None
        self.reset(seed=seed)

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            self.seed = seed
        if self.seed is not None:
            self.rng = make_rng(self.seed)
        self.robot = self.start.copy()
        self.time = 0
        self.regime_shift_active = False
        self.trajectory: list[np.ndarray] = [self.robot.copy()]
        self.last_action_delta: np.ndarray = np.zeros(2)
        self.last_actual_delta: np.ndarray = np.zeros(2)
        self.last_predicted_pos: np.ndarray | None = None
        self.last_model_error: float = 0.0
        return self.observe()

    def observe(self) -> dict[str, Any]:
        return {
            "time": self.time,
            "robot": self.robot.copy(),
            "goal": self.goal.copy(),
            "goal_radius": self.goal_radius,
            "move_max": self.move_max,
            "error_threshold": self.error_threshold,
            "regime_shift_active": self.regime_shift_active,
            "max_steps": self.max_steps,
        }

    def step(self, action: dict[str, Any]) -> StepResult:
        self.time += 1
        if self.time >= self.regime_shift_at:
            self.regime_shift_active = True

        action_delta = np.asarray(action.get("delta", [0.0, 0.0]), dtype=float)
        norm = float(np.linalg.norm(action_delta))
        if norm > self.move_max:
            action_delta = action_delta / max(norm, 1e-9) * self.move_max
        self.last_action_delta = action_delta.copy()

        offset = self.shift_offset.copy() if self.regime_shift_active else np.zeros(2)
        noise = self.rng.normal(0.0, self.noise_sigma, size=2)
        actual_delta = action_delta + offset + noise
        self.last_actual_delta = actual_delta.copy()
        self.robot = np.clip(self.robot + actual_delta, 0.0, self.size)
        self.trajectory.append(self.robot.copy())

        action_type = action.get("action_type", "move")
        info: dict[str, Any] = {
            "time": self.time,
            "action_type": action_type,
            "action_delta": action_delta.tolist(),
            "actual_delta": actual_delta.tolist(),
            "regime_shift_active": self.regime_shift_active,
            "success": False,
        }

        predicted = action.get("predicted_pos")
        if predicted is not None:
            predicted_arr = np.asarray(predicted, dtype=float)
            self.last_predicted_pos = predicted_arr.copy()
            model_error = float(np.linalg.norm(self.robot - predicted_arr))
            self.last_model_error = model_error
            info["predicted_pos"] = predicted_arr.tolist()
            info["model_error"] = model_error
            if model_error > self.error_threshold:
                info["failure"] = Failure(
                    "model_drift",
                    f"prediction error exceeded threshold ({model_error:.3f})",
                    True,
                )

        if "probe_index" in action:
            info["probe_index"] = action["probe_index"]

        d_goal = float(np.linalg.norm(self.robot - self.goal))
        info["distance_to_goal"] = d_goal
        if d_goal <= self.goal_radius:
            info["success"] = True
            return StepResult(self.observe(), 1.0, True, info)
        return StepResult(self.observe(), -0.02, False, info)

    def render(self, agent: Any | None = None, info: dict[str, Any] | None = None) -> None:
        import matplotlib.pyplot as plt

        if self._fig is None or self._ax is None:
            plt.ion()
            self._fig, self._ax = plt.subplots(figsize=(5.6, 5.0))
        ax = self._ax
        ax.clear()
        draw_model_error_recovery_scene(ax, self, agent, info)
        self._fig.canvas.draw_idle()
        plt.pause(0.1)


class ModelErrorRecoveryAgent:
    """Detect model_drift, run a 3-probe system identification, then resume."""

    def __init__(self, probes: tuple[np.ndarray, ...] = PROBE_ACTIONS) -> None:
        self.probes = tuple(np.asarray(p, dtype=float) for p in probes)
        self.reset()

    def reset(self) -> None:
        self.learned_offset = np.zeros(2, dtype=float)
        self.state: str = "go_to_goal"
        self.last_action_delta: np.ndarray | None = None
        self.last_predicted_pos: np.ndarray | None = None
        self.probe_index: int = 0
        self.probe_observed_offsets: list[np.ndarray] = []
        self.model_error_count: int = 0
        self.model_update_count: int = 0
        self.recovery_count: int = 0
        self.last_model_error: float = 0.0
        self.cumulative_error: float = 0.0

    def predict(self, robot: np.ndarray, action_delta: np.ndarray) -> np.ndarray:
        return robot + action_delta + self.learned_offset

    def act(self, obs: dict[str, Any]) -> dict[str, Any]:
        robot = np.asarray(obs["robot"], dtype=float)
        if self.state == "go_to_goal":
            return self._goal_action(obs, robot)
        if self.state == "system_id":
            return self._probe_action(obs, robot)
        return {
            "action_type": "wait",
            "delta": [0.0, 0.0],
            "predicted_pos": robot.tolist(),
        }

    def _goal_action(self, obs: dict[str, Any], robot: np.ndarray) -> dict[str, Any]:
        goal = np.asarray(obs["goal"], dtype=float)
        delta = goal - robot
        d = float(np.linalg.norm(delta))
        move_max = float(obs.get("move_max", 0.08))
        if d > move_max:
            action_delta = delta / max(d, 1e-9) * move_max
        else:
            action_delta = delta
        self.last_action_delta = action_delta.copy()
        self.last_predicted_pos = self.predict(robot, action_delta)
        return {
            "action_type": "move",
            "delta": action_delta.tolist(),
            "predicted_pos": self.last_predicted_pos.tolist(),
        }

    def _probe_action(self, obs: dict[str, Any], robot: np.ndarray) -> dict[str, Any]:
        probe = self.probes[self.probe_index].copy()
        self.last_action_delta = probe.copy()
        self.last_predicted_pos = self.predict(robot, probe)
        return {
            "action_type": "probe",
            "delta": probe.tolist(),
            "predicted_pos": self.last_predicted_pos.tolist(),
            "probe_index": int(self.probe_index),
        }

    def update(self, obs: dict[str, Any], reward: float, info: dict[str, Any]) -> None:
        if "model_error" in info:
            self.last_model_error = float(info["model_error"])
            self.cumulative_error += self.last_model_error

        failure = info.get("failure")
        if self.state == "go_to_goal":
            if isinstance(failure, Failure) and failure.kind == "model_drift":
                self.model_error_count += 1
                self.state = "system_id"
                self.probe_index = 0
                self.probe_observed_offsets = []
                self.recovery_count += 1
            if info.get("success"):
                self.state = "succeeded"
            return

        if self.state == "system_id":
            if self.last_action_delta is not None:
                actual_delta = np.asarray(info.get("actual_delta", [0.0, 0.0]), dtype=float)
                observed_offset = actual_delta - self.last_action_delta
                self.probe_observed_offsets.append(observed_offset)
            self.probe_index += 1
            if self.probe_index >= len(self.probes):
                mean_offset = np.mean(np.stack(self.probe_observed_offsets, axis=0), axis=0)
                self.learned_offset = self.learned_offset + mean_offset
                self.model_update_count += 1
                self.state = "go_to_goal"


def draw_model_error_recovery_scene(
    ax: Any,
    env: ModelErrorRecoveryWorld,
    agent: ModelErrorRecoveryAgent | None,
    info: dict[str, Any] | None,
) -> None:
    import matplotlib.patches as mpatches

    ax.set_xlim(0.0, env.size)
    ax.set_ylim(0.0, env.size)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("model error recovery: regime shift")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    bg_color = "0.97"
    if env.regime_shift_active:
        bg_color = (1.0, 0.93, 0.93)
    ax.add_patch(mpatches.Rectangle((0.0, 0.0), env.size, env.size, color=bg_color, ec="0.7"))

    ax.add_patch(
        mpatches.Circle(env.goal, env.goal_radius, color="tab:green", alpha=0.25)
    )
    ax.plot(*env.goal, marker="*", color="tab:green", markersize=16)

    goal_vec = env.goal - env.robot
    goal_dist = float(np.linalg.norm(goal_vec))
    if goal_dist > 1e-3:
        ax.annotate(
            "",
            xy=tuple(env.robot + 0.15 * goal_vec / goal_dist),
            xytext=tuple(env.robot),
            arrowprops=dict(arrowstyle="->", color="0.55", lw=1.0, alpha=0.7),
        )

    if len(env.trajectory) > 1:
        traj = np.asarray(env.trajectory)
        ax.plot(traj[:, 0], traj[:, 1], color="tab:blue", linewidth=1.5, alpha=0.7)
    ax.plot(*env.robot, marker="o", color="tab:blue", markersize=10)

    if env.last_predicted_pos is not None:
        ax.plot(
            env.last_predicted_pos[0],
            env.last_predicted_pos[1],
            marker="x",
            color="tab:purple",
            markersize=10,
            mew=2,
        )
        ax.annotate(
            "",
            xy=tuple(env.robot),
            xytext=tuple(env.last_predicted_pos),
            arrowprops=dict(
                arrowstyle="->",
                color="tab:red",
                lw=1.6,
                linestyle="--",
                alpha=0.85,
            ),
        )

    if env.regime_shift_active:
        ax.annotate(
            "regime shift active",
            xy=(0.5, 0.92),
            xycoords="axes fraction",
            ha="center",
            color="tab:red",
            fontsize=9,
            fontweight="bold",
        )

    if agent is not None:
        offset = np.asarray(agent.learned_offset, dtype=float)
        offset_norm = float(np.linalg.norm(offset))
        inset_origin = np.array([0.86, 0.10])
        ax.add_patch(
            mpatches.Rectangle(
                (inset_origin[0] - 0.08, inset_origin[1] - 0.05),
                0.16,
                0.13,
                facecolor="white",
                edgecolor="0.6",
                alpha=0.85,
                lw=0.8,
            )
        )
        ax.text(
            inset_origin[0],
            inset_origin[1] + 0.06,
            "learned offset",
            ha="center",
            fontsize=7,
            color="0.3",
        )
        if offset_norm > 1e-6:
            scale = 0.05 / max(offset_norm, 1e-6)
            ax.annotate(
                "",
                xy=tuple(inset_origin + offset * scale),
                xytext=tuple(inset_origin),
                arrowprops=dict(arrowstyle="->", color="tab:purple", lw=1.4),
            )
        else:
            ax.plot(*inset_origin, marker=".", color="0.5", markersize=6)

    status_parts: list[str] = [
        f"step={env.time}",
        f"err={env.last_model_error:.3f}",
    ]
    if agent is not None:
        status_parts.append(f"state={agent.state}")
        status_parts.append(f"offset=({agent.learned_offset[0]:+.3f},{agent.learned_offset[1]:+.3f})")
        status_parts.append(f"recoveries={agent.recovery_count}")
        status_parts.append(f"updates={agent.model_update_count}")
    if info is not None and "failure" in info:
        status_parts.append(f"failure={info['failure'].kind}")
    if info is not None and info.get("success"):
        status_parts.append("success")
    ax.text(0.02, 0.98, "  ".join(status_parts), transform=ax.transAxes, va="top", fontsize=9)


def run(seed: int = 0, render: bool = True, max_steps: int = 50) -> Trace:
    env = ModelErrorRecoveryWorld(seed=seed, max_steps=max_steps)
    agent = ModelErrorRecoveryAgent()
    obs = env.reset(seed=seed)
    agent.reset()
    trace = Trace()

    for _ in range(max_steps):
        action = agent.act(obs)
        result = env.step(action)
        obs, reward, done, info = result.as_tuple()
        agent.update(obs, reward, info)
        info["agent_state"] = agent.state
        info["learned_offset"] = agent.learned_offset.tolist()
        info["model_error_count"] = agent.model_error_count
        info["model_update_count"] = agent.model_update_count
        info["recovery_count"] = agent.recovery_count
        info["cumulative_error"] = agent.cumulative_error
        trace.append(obs, action, reward, info)

        if render:
            env.render(agent=agent, info=info)
        if done:
            break

    return trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    trace = run(seed=args.seed, render=not args.no_render, max_steps=args.max_steps)
    success = bool(trace.infos and trace.infos[-1].get("success"))
    failures = [failure.kind for failure in trace.failures()]
    final = trace.infos[-1] if trace.infos else {}
    print(
        f"success={success} steps={len(trace.actions)} "
        f"errors={final.get('model_error_count', 0)} "
        f"updates={final.get('model_update_count', 0)} "
        f"recoveries={final.get('recovery_count', 0)} "
        f"offset={final.get('learned_offset', [0.0, 0.0])} "
        f"failures={len(failures)}"
    )

    if not args.no_render:
        import matplotlib.pyplot as plt

        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
