"""Plan with a tiny action-conditioned world model and update it online.

The robot starts with a wrong internal dynamics model: it assumes every action
moves exactly as commanded.  The world has an invisible drift field.  After each
action, the robot compares predicted and actual next state, records model error,
updates an action-conditioned residual model, and replans.
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


ACTION_DELTAS: dict[str, np.ndarray] = {
    "E": np.array([0.075, 0.000], dtype=float),
    "W": np.array([-0.075, 0.000], dtype=float),
    "N": np.array([0.000, 0.075], dtype=float),
    "S": np.array([0.000, -0.075], dtype=float),
    "NE": np.array([0.055, 0.055], dtype=float),
    "SE": np.array([0.055, -0.055], dtype=float),
    "NW": np.array([-0.055, 0.055], dtype=float),
    "SW": np.array([-0.055, -0.055], dtype=float),
}


class TinyWorldModelWorld:
    """A continuous 2D world with a hidden drift region."""

    def __init__(self, seed: int = 0, max_steps: int = 80) -> None:
        self.seed = seed
        self.max_steps = max_steps
        self.goal_radius = 0.045
        self.error_threshold = 0.025
        self.drift_region = np.array([0.36, 0.16, 0.66, 0.86], dtype=float)
        self.drift = np.array([0.000, -0.043], dtype=float)
        self._figure: Any | None = None
        self._axis: Any | None = None
        self.reset(seed=seed)

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        self.time = 0
        self.robot = np.array([0.12, 0.23], dtype=float)
        self.goal = np.array([0.86, 0.78], dtype=float)
        self.trajectory = [self.robot.copy()]
        self.last_predicted_next: np.ndarray | None = None
        self.last_nominal_next: np.ndarray | None = None
        self.last_action_name: str | None = None
        return self.observe()

    def observe(self) -> dict[str, Any]:
        return {
            "time": self.time,
            "robot": self.robot.copy(),
            "goal": self.goal.copy(),
            "in_drift_region": self.in_drift_region(self.robot),
        }

    def step(self, action: dict[str, Any]) -> StepResult:
        self.time += 1
        action_name = str(action.get("name", "E"))
        commanded = ACTION_DELTAS.get(action_name)
        info: dict[str, Any] = {
            "time": self.time,
            "action_type": "world_model_step",
            "action_name": action_name,
            "success": False,
        }
        if commanded is None:
            info["failure"] = Failure("invalid_action", f"unknown action: {action_name}", True)
            return StepResult(self.observe(), -0.05, False, info)

        previous = self.robot.copy()
        region_key = self.region_key(previous)
        predicted_next = np.asarray(action.get("predicted_next", previous + commanded), dtype=float)
        predicted_next = np.clip(predicted_next, [0.0, 0.0], [1.0, 1.0])
        nominal_next = np.clip(previous + commanded, [0.0, 0.0], [1.0, 1.0])
        actual_delta = commanded + self._hidden_drift(previous)
        self.robot = np.clip(previous + actual_delta, [0.0, 0.0], [1.0, 1.0])
        self.trajectory.append(self.robot.copy())
        self.last_predicted_next = predicted_next.copy()
        self.last_nominal_next = nominal_next.copy()
        self.last_action_name = action_name

        model_error = float(np.linalg.norm(self.robot - predicted_next))
        goal_distance = float(np.linalg.norm(self.goal - self.robot))
        success = goal_distance <= self.goal_radius
        timeout = self.time >= self.max_steps

        info.update(
            {
                "previous_state": previous.copy(),
                "actual_next": self.robot.copy(),
                "predicted_next": predicted_next.copy(),
                "nominal_next": nominal_next.copy(),
                "commanded_delta": commanded.copy(),
                "actual_delta": (self.robot - previous).copy(),
                "region_key": region_key,
                "model_error": model_error,
                "goal_distance": goal_distance,
            }
        )

        if model_error > self.error_threshold:
            info["failure"] = Failure(
                "model_error",
                "predicted next state did not match the observed transition",
                recoverable=True,
            )

        if success:
            info["success"] = True
        elif timeout:
            info["failure"] = Failure(
                "timeout",
                "world-model planner did not reach the goal before max_steps",
                recoverable=False,
            )

        reward = 1.0 if success else -goal_distance - model_error
        return StepResult(self.observe(), reward, success or timeout, info)

    def in_drift_region(self, point: np.ndarray) -> bool:
        xmin, ymin, xmax, ymax = self.drift_region
        x, y = point
        return bool(xmin <= x <= xmax and ymin <= y <= ymax)

    def region_key(self, point: np.ndarray) -> str:
        return "drift" if self.in_drift_region(point) else "free"

    def _hidden_drift(self, point: np.ndarray) -> np.ndarray:
        return self.drift.copy() if self.in_drift_region(point) else np.zeros(2, dtype=float)

    def render(self, agent: "TinyWorldModelAgent", info: dict[str, Any] | None = None) -> None:
        import matplotlib.pyplot as plt

        if self._figure is None or self._axis is None:
            plt.ion()
            self._figure, self._axis = plt.subplots(figsize=(5.2, 5.2))

        draw_world_model_scene(self._axis, self, agent, info)
        self._figure.canvas.draw_idle()
        plt.pause(0.04)


class TinyWorldModelAgent:
    """Beam-search planner over a learned residual dynamics model."""

    def __init__(self) -> None:
        self.horizon = 7
        self.beam_width = 42
        self.learning_rate = 0.75
        self.reset()

    def reset(self) -> None:
        self.residuals: dict[tuple[str, str], np.ndarray] = {}
        self.counts: dict[tuple[str, str], int] = {}
        self.state = "plan_with_model"
        self.transition_count = 0
        self.model_update_count = 0
        self.model_error_count = 0
        self.last_prediction: list[np.ndarray] = []
        self.last_plan: list[str] = []

    def act(self, obs: dict[str, Any], env: TinyWorldModelWorld) -> dict[str, Any]:
        state = np.asarray(obs["robot"], dtype=float)
        self.last_plan, self.last_prediction = self._plan(state, env.goal, env)
        action_name = self.last_plan[0] if self.last_plan else "E"
        predicted_next = self.predict_next(state, action_name, env)
        self.state = "execute_model_plan"
        return {"name": action_name, "predicted_next": predicted_next}

    def update(self, obs: dict[str, Any], reward: float, info: dict[str, Any]) -> None:
        if "actual_next" not in info:
            return

        previous = np.asarray(info["previous_state"], dtype=float)
        actual_next = np.asarray(info["actual_next"], dtype=float)
        commanded = np.asarray(info["commanded_delta"], dtype=float)
        region_key = str(info["region_key"])
        action_name = str(info["action_name"])
        residual_sample = actual_next - previous - commanded
        key = (region_key, action_name)
        old = self.residuals.get(key, np.zeros(2, dtype=float))
        count = self.counts.get(key, 0)
        gain = self.learning_rate if count == 0 else 0.42
        self.residuals[key] = (1.0 - gain) * old + gain * residual_sample
        self.counts[key] = count + 1
        self.transition_count += 1
        self.model_update_count += 1

        failure = info.get("failure")
        if isinstance(failure, Failure) and failure.kind == "model_error":
            self.model_error_count += 1
            self.state = "update_world_model"
        elif info.get("success"):
            self.state = "goal_reached"
        else:
            self.state = "replan_with_updated_model"

        info["agent_state"] = self.state
        info["transition_count"] = self.transition_count
        info["model_update_count"] = self.model_update_count
        info["model_error_count"] = self.model_error_count
        info["learned_residual_norm"] = float(
            max((np.linalg.norm(value) for value in self.residuals.values()), default=0.0)
        )
        info["plan"] = list(self.last_plan)

    def predict_next(
        self,
        state: np.ndarray,
        action_name: str,
        env: TinyWorldModelWorld,
    ) -> np.ndarray:
        commanded = ACTION_DELTAS[action_name]
        residual = self.residuals.get((env.region_key(state), action_name), np.zeros(2, dtype=float))
        return np.clip(state + commanded + residual, [0.0, 0.0], [1.0, 1.0])

    def _plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        env: TinyWorldModelWorld,
    ) -> tuple[list[str], list[np.ndarray]]:
        beams: list[tuple[float, np.ndarray, list[str], list[np.ndarray]]] = [
            (float(np.linalg.norm(goal - start)), start.copy(), [], [start.copy()])
        ]
        for _ in range(self.horizon):
            expanded: list[tuple[float, np.ndarray, list[str], list[np.ndarray]]] = []
            for _, state, actions, path in beams:
                for action_name in ACTION_DELTAS:
                    next_state = self.predict_next(state, action_name, env)
                    distance = float(np.linalg.norm(goal - next_state))
                    step_cost = 0.010 * (len(actions) + 1)
                    edge_penalty = 0.20 if np.any(next_state <= 0.02) or np.any(next_state >= 0.98) else 0.0
                    cost = distance + step_cost + edge_penalty
                    expanded.append(
                        (
                            cost,
                            next_state,
                            actions + [action_name],
                            path + [next_state.copy()],
                        )
                    )
            expanded.sort(key=lambda item: item[0])
            beams = expanded[: self.beam_width]
        best = min(beams, key=lambda item: item[0])
        return best[2], best[3]


def draw_world_model_scene(
    ax: Any,
    env: TinyWorldModelWorld,
    agent: TinyWorldModelAgent,
    info: dict[str, Any] | None = None,
) -> None:
    """Draw true trajectory, predicted rollout, hidden drift, and model error."""

    from matplotlib.patches import Circle, Rectangle

    info = {} if info is None else info
    ax.clear()
    ax.set_title("tiny world-model planning")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)

    xmin, ymin, xmax, ymax = env.drift_region
    ax.add_patch(
        Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            facecolor="tab:cyan",
            alpha=0.10,
            edgecolor="tab:cyan",
            linestyle="--",
            label="hidden drift region",
        )
    )
    ax.arrow(
        (xmin + xmax) / 2,
        ymax - 0.08,
        env.drift[0],
        env.drift[1] * 2.0,
        color="tab:cyan",
        width=0.003,
        head_width=0.018,
        length_includes_head=True,
        alpha=0.9,
    )

    trajectory = np.asarray(env.trajectory)
    ax.plot(trajectory[:, 0], trajectory[:, 1], color="tab:blue", linewidth=2, label="actual")

    if agent.last_prediction:
        prediction = np.asarray(agent.last_prediction)
        ax.plot(
            prediction[:, 0],
            prediction[:, 1],
            linestyle="--",
            color="tab:purple",
            linewidth=2,
            label="model rollout",
        )

    if env.last_predicted_next is not None:
        ax.plot(*env.last_predicted_next, marker="x", color="tab:purple", markersize=10)
        ax.plot(
            [env.robot[0], env.last_predicted_next[0]],
            [env.robot[1], env.last_predicted_next[1]],
            color="tab:purple",
            alpha=0.25,
        )

    ax.add_patch(Circle(env.goal, env.goal_radius, color="tab:green", alpha=0.22))
    ax.plot(*env.goal, marker="*", color="tab:green", markersize=15, label="goal")
    ax.plot(*env.robot, marker="o", color="tab:blue", markersize=8, label="robot")

    learned = agent.residuals.get(("drift", info.get("action_name", "NE")))
    if learned is not None:
        anchor = np.array([xmax + 0.06, ymax - 0.05], dtype=float)
        ax.arrow(
            anchor[0],
            anchor[1],
            learned[0] * 2.0,
            learned[1] * 2.0,
            color="black",
            width=0.003,
            head_width=0.018,
            length_includes_head=True,
        )
        ax.text(anchor[0], anchor[1] + 0.035, "learned residual", fontsize=8)

    status = (
        f"step={env.time}  state={info.get('agent_state', agent.state)}\n"
        f"model_error={info.get('model_error', 0.0):.3f}  "
        f"errors={agent.model_error_count}  updates={agent.model_update_count}\n"
        f"plan={' '.join(agent.last_plan[:5])}"
    )
    if "failure" in info:
        status += f"  failure={info['failure'].kind}"
    if info.get("success"):
        status += "  success"
    ax.text(0.02, 0.98, status, transform=ax.transAxes, va="top", fontsize=9)
    ax.legend(loc="lower left", fontsize=8)


def run(seed: int = 0, render: bool = True, max_steps: int = 80) -> Trace:
    env = TinyWorldModelWorld(seed=seed, max_steps=max_steps)
    agent = TinyWorldModelAgent()
    obs = env.reset(seed=seed)
    agent.reset()
    trace = Trace()

    for _ in range(max_steps):
        action = agent.act(obs, env)
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    trace = run(seed=args.seed, render=not args.no_render, max_steps=args.max_steps)
    final_info = trace.infos[-1] if trace.infos else {}
    failures = [failure.kind for failure in trace.failures()]
    print(
        f"success={bool(final_info.get('success'))} steps={len(trace.actions)} "
        f"model_errors={final_info.get('model_error_count', 0)} "
        f"updates={final_info.get('model_update_count', 0)} "
        f"residual={final_info.get('learned_residual_norm', 0.0):.3f} "
        f"failures={failures}"
    )

    if not args.no_render:
        import matplotlib.pyplot as plt

        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
