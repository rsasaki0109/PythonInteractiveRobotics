"""Reach a moving target with prediction, occlusion, and online correction.

The arm should not chase the last visible target pose.  It observes a moving
target, estimates a tiny velocity belief, predicts through short occlusions,
servos toward the predicted intercept point, and corrects again when the target
reappears.
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


@dataclass(frozen=True)
class MovingReachConfig:
    base: np.ndarray
    link_lengths: np.ndarray
    joint_limits: np.ndarray
    max_joint_delta: float = 0.115
    touch_radius: float = 0.035
    stable_touches_required: int = 4


class MovingTargetReachWorld:
    """A 2-link arm reaches for a target that can disappear behind an occluder."""

    def __init__(self, seed: int = 5, max_steps: int = 90) -> None:
        self.seed = seed
        self.max_steps = max_steps
        self.config = MovingReachConfig(
            base=np.array([0.25, 0.24], dtype=float),
            link_lengths=np.array([0.42, 0.30], dtype=float),
            joint_limits=np.array([[-0.65, 2.70], [-2.70, 2.70]], dtype=float),
        )
        self.occluder = np.array([0.56, 0.48, 0.72, 0.66], dtype=float)
        self.observation_noise = 0.007
        self.rng = np.random.default_rng(seed)
        self._figure: Any | None = None
        self._axis: Any | None = None
        self.reset(seed=seed)

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            self.seed = seed
            self.rng = np.random.default_rng(seed)

        self.time = 0
        self.q = np.array([0.18, -1.48], dtype=float)
        self.touch_count = 0
        self.occluded_frames = 0
        self.last_observation: np.ndarray | None = None
        self.last_joint_delta = np.zeros(2, dtype=float)
        self.ee_path = [self.forward_kinematics(self.q)[-1]]
        self.target_path = [self.target_position(self.time)]
        return self.observe()

    def observe(self) -> dict[str, Any]:
        target = self.target_position(self.time)
        visible = not self._is_occluded(target)
        detections: list[dict[str, Any]] = []
        if visible:
            observation = np.clip(
                target + self.rng.normal(0.0, self.observation_noise, size=2),
                [0.0, 0.0],
                [1.0, 1.0],
            )
            self.last_observation = observation
            detections.append(
                {
                    "name": "moving_target",
                    "position": observation,
                    "confidence": 0.90,
                }
            )

        points = self.forward_kinematics(self.q)
        return {
            "time": self.time,
            "q": self.q.copy(),
            "end_effector": points[-1].copy(),
            "target_visible": visible,
            "detections": detections,
            "touch_count": self.touch_count,
            "occluded_frames": self.occluded_frames,
        }

    def step(self, action: dict[str, Any]) -> StepResult:
        self.time += 1
        action_type = action.get("type", "servo")
        info: dict[str, Any] = {
            "time": self.time,
            "action_type": action_type,
            "success": False,
        }

        if action_type != "servo":
            info["failure"] = Failure(
                "invalid_action",
                f"unknown action type: {action_type}",
                recoverable=True,
            )
            return StepResult(self.observe(), -0.05, False, info)

        raw_delta = np.asarray(action.get("joint_delta", np.zeros(2)), dtype=float)
        joint_delta = np.clip(
            raw_delta,
            -self.config.max_joint_delta,
            self.config.max_joint_delta,
        )
        self.q = np.clip(
            self.q + joint_delta,
            self.config.joint_limits[:, 0],
            self.config.joint_limits[:, 1],
        )
        self.last_joint_delta = joint_delta.copy()

        points = self.forward_kinematics(self.q)
        target = self.target_position(self.time)
        visible = not self._is_occluded(target)
        if visible:
            self.occluded_frames = 0
        else:
            self.occluded_frames += 1
            info["failure"] = Failure(
                "target_occluded",
                "target is hidden, so the agent must predict instead of observe",
                recoverable=True,
            )

        reach_error = float(np.linalg.norm(points[-1] - target))
        if reach_error <= self.config.touch_radius:
            self.touch_count += 1
        else:
            self.touch_count = 0

        done = self.touch_count >= self.config.stable_touches_required
        timeout = self.time >= self.max_steps
        info.update(
            {
                "reach_error": reach_error,
                "touch_count": self.touch_count,
                "target": target.copy(),
                "target_visible": visible,
                "occluded_frames": self.occluded_frames,
                "joint_delta": joint_delta.copy(),
            }
        )

        if done:
            info["success"] = True
        elif timeout:
            info["failure"] = Failure(
                "timeout",
                "moving target was not reached before max_steps",
                recoverable=False,
            )

        self.ee_path.append(points[-1].copy())
        self.target_path.append(target.copy())
        reward = 1.0 if done else -reach_error - (0.02 if not visible else 0.0)
        return StepResult(self.observe(), reward, done or timeout, info)

    def target_position(self, time: int) -> np.ndarray:
        t = float(time)
        return np.array(
            [
                0.69 + 0.135 * np.cos(0.13 * t),
                0.57 + 0.070 * np.sin(0.17 * t),
            ],
            dtype=float,
        )

    def forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        l1, l2 = self.config.link_lengths
        q1, q2 = q
        elbow = self.config.base + l1 * np.array([np.cos(q1), np.sin(q1)])
        ee = elbow + l2 * np.array([np.cos(q1 + q2), np.sin(q1 + q2)])
        return np.vstack([self.config.base, elbow, ee])

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        l1, l2 = self.config.link_lengths
        q1, q2 = q
        q12 = q1 + q2
        return np.array(
            [
                [-l1 * np.sin(q1) - l2 * np.sin(q12), -l2 * np.sin(q12)],
                [l1 * np.cos(q1) + l2 * np.cos(q12), l2 * np.cos(q12)],
            ],
            dtype=float,
        )

    def _is_occluded(self, point: np.ndarray) -> bool:
        xmin, ymin, xmax, ymax = self.occluder
        return bool(xmin <= point[0] <= xmax and ymin <= point[1] <= ymax)

    def render(self, agent: "MovingTargetReachAgent", info: dict[str, Any] | None = None) -> None:
        import matplotlib.pyplot as plt

        if self._figure is None or self._axis is None:
            plt.ion()
            self._figure, self._axis = plt.subplots(figsize=(5.2, 5.2))

        draw_moving_target_reaching_scene(self._axis, self, agent, info)
        self._figure.canvas.draw_idle()
        plt.pause(0.04)


class MovingTargetReachAgent:
    """Estimate target velocity and servo toward a predicted intercept point."""

    def __init__(self) -> None:
        self.damping = 0.075
        self.task_gain = 0.86
        self.lookahead_steps = 1.9
        self.max_delta_norm = 0.13
        self.reset()

    def reset(self) -> None:
        self.target_belief: np.ndarray | None = None
        self.velocity_belief = np.zeros(2, dtype=float)
        self.predicted_target: np.ndarray | None = None
        self.belief_radius = 0.075
        self.state = "observe"
        self.observed_updates = 0
        self.prediction_updates = 0
        self.servo_updates = 0
        self.occlusion_count = 0
        self.last_joint_delta = np.zeros(2, dtype=float)
        self.last_task_error = np.zeros(2, dtype=float)
        self._last_detection_time: int | None = None
        self._last_detection_position: np.ndarray | None = None
        self._last_integrated_time: int | None = None

    def act(self, obs: dict[str, Any], env: MovingTargetReachWorld) -> dict[str, Any]:
        self._integrate_observation(obs)
        if self.target_belief is None:
            self.state = "wait_for_target"
            return {"type": "servo", "joint_delta": np.zeros(2, dtype=float)}

        q = np.asarray(obs["q"], dtype=float)
        ee = np.asarray(obs["end_effector"], dtype=float)
        self.predicted_target = np.clip(
            self.target_belief + self.lookahead_steps * self.velocity_belief,
            [0.05, 0.05],
            [0.95, 0.95],
        )
        task_error = self.predicted_target - ee
        jacobian = env.jacobian(q)
        damped = jacobian @ jacobian.T + (self.damping**2) * np.eye(2)
        joint_delta = self.task_gain * jacobian.T @ np.linalg.solve(damped, task_error)

        delta_norm = float(np.linalg.norm(joint_delta))
        if delta_norm > self.max_delta_norm:
            joint_delta = joint_delta / delta_norm * self.max_delta_norm

        self.last_task_error = task_error.copy()
        self.last_joint_delta = joint_delta.copy()
        self.servo_updates += 1
        self.state = "predictive_servo"
        return {"type": "servo", "joint_delta": joint_delta}

    def update(self, obs: dict[str, Any], reward: float, info: dict[str, Any]) -> None:
        self._integrate_observation(obs)
        if isinstance(info.get("failure"), Failure) and info["failure"].kind == "target_occluded":
            self.occlusion_count += 1
            self.state = "predict_through_occlusion"
        elif info.get("success"):
            self.state = "reached_moving_target"

        info["agent_state"] = self.state
        info["observed_updates"] = self.observed_updates
        info["prediction_updates"] = self.prediction_updates
        info["servo_updates"] = self.servo_updates
        info["occlusion_count"] = self.occlusion_count
        info["velocity_norm"] = float(np.linalg.norm(self.velocity_belief))
        info["belief_radius"] = self.belief_radius
        info["task_error_norm"] = float(np.linalg.norm(self.last_task_error))
        if self.target_belief is not None:
            info["target_belief"] = self.target_belief.copy()
        if self.predicted_target is not None:
            info["predicted_target"] = self.predicted_target.copy()

    def _integrate_observation(self, obs: dict[str, Any]) -> None:
        obs_time = int(obs.get("time", -1))
        if obs_time == self._last_integrated_time:
            return
        self._last_integrated_time = obs_time

        detections = obs.get("detections", [])
        if detections:
            position = np.asarray(detections[0]["position"], dtype=float)
            if self._last_detection_time is not None and self._last_detection_position is not None:
                dt = max(1, obs_time - self._last_detection_time)
                measured_velocity = (position - self._last_detection_position) / float(dt)
                self.velocity_belief = 0.45 * self.velocity_belief + 0.55 * measured_velocity

            if self.target_belief is None:
                self.target_belief = position.copy()
            else:
                predicted_now = self.target_belief + self.velocity_belief
                self.target_belief = 0.25 * predicted_now + 0.75 * position

            self._last_detection_time = obs_time
            self._last_detection_position = position.copy()
            self.belief_radius = max(0.020, self.belief_radius * 0.72)
            self.observed_updates += 1
            return

        if self.target_belief is not None:
            self.target_belief = np.clip(
                self.target_belief + self.velocity_belief,
                [0.05, 0.05],
                [0.95, 0.95],
            )
            self.belief_radius = min(0.16, self.belief_radius + 0.018)
            self.prediction_updates += 1


def draw_moving_target_reaching_scene(
    ax: Any,
    env: MovingTargetReachWorld,
    agent: MovingTargetReachAgent,
    info: dict[str, Any] | None = None,
) -> None:
    """Draw the arm, true target, occluder, belief, prediction, and traces."""

    from matplotlib.patches import Circle, Rectangle

    info = {} if info is None else info
    ax.clear()
    ax.set_title("moving target reaching")
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

    ee_path = np.asarray(env.ee_path)
    target_path = np.asarray(env.target_path)
    if len(target_path) > 1:
        ax.plot(target_path[:, 0], target_path[:, 1], color="tab:red", alpha=0.35)
    if len(ee_path) > 1:
        ax.plot(ee_path[:, 0], ee_path[:, 1], color="tab:blue", linewidth=2)

    points = env.forward_kinematics(env.q)
    ax.plot(points[:, 0], points[:, 1], "-o", color="tab:blue", linewidth=4, markersize=7)

    target = env.target_position(env.time)
    visible = not env._is_occluded(target)
    target_alpha = 0.82 if visible else 0.22
    ax.add_patch(Circle(target, 0.020, color="tab:red", alpha=target_alpha, label="true target"))

    if env.last_observation is not None and visible:
        ax.plot(
            *env.last_observation,
            marker="x",
            markersize=10,
            color="tab:orange",
            label="visual observation",
        )

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

    if agent.predicted_target is not None:
        ax.plot(
            *agent.predicted_target,
            marker="+",
            markersize=13,
            color="black",
            label="predicted intercept",
        )

    ee = points[-1]
    if agent.predicted_target is not None:
        delta = agent.predicted_target - ee
        ax.arrow(
            ee[0],
            ee[1],
            delta[0],
            delta[1],
            color="black",
            width=0.003,
            head_width=0.018,
            length_includes_head=True,
            alpha=0.55,
        )

    status = (
        f"step={env.time}  state={info.get('agent_state', agent.state)}\n"
        f"reach_error={info.get('reach_error', np.linalg.norm(ee - target)):.3f}  "
        f"touches={info.get('touch_count', env.touch_count)}\n"
        f"observed={agent.observed_updates}  predicted={agent.prediction_updates}  "
        f"occlusions={agent.occlusion_count}"
    )
    if "failure" in info:
        status += f"  failure={info['failure'].kind}"
    if info.get("success"):
        status += "  success"
    ax.text(0.02, 0.98, status, transform=ax.transAxes, va="top", fontsize=9)
    ax.legend(loc="lower left", fontsize=8)


def run(seed: int = 5, render: bool = True, max_steps: int = 90) -> Trace:
    env = MovingTargetReachWorld(seed=seed, max_steps=max_steps)
    agent = MovingTargetReachAgent()
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
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=90)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    trace = run(seed=args.seed, render=not args.no_render, max_steps=args.max_steps)
    final_info = trace.infos[-1] if trace.infos else {}
    failures = [failure.kind for failure in trace.failures()]
    print(
        f"success={bool(final_info.get('success'))} steps={len(trace.actions)} "
        f"reach_error={final_info.get('reach_error', 0.0):.3f} "
        f"touches={final_info.get('touch_count', 0)} "
        f"occlusions={final_info.get('occlusion_count', 0)} "
        f"predictions={final_info.get('prediction_updates', 0)} "
        f"failures={failures}"
    )

    if not args.no_render:
        import matplotlib.pyplot as plt

        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
