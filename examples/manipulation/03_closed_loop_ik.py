"""Closed-loop IK servoing for a noisy moving target.

This example deliberately avoids a one-shot "solve IK once, then execute"
demo.  A 2-link planar arm observes a target through noise, updates a target
belief, computes a damped Jacobian step, moves a little, observes again, and
keeps correcting until the end-effector tracks the moving target for several
steps.
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
class ArmConfig:
    base: np.ndarray
    link_lengths: np.ndarray
    joint_limits: np.ndarray
    max_joint_delta: float = 0.10
    success_radius: float = 0.028
    stable_steps_required: int = 7


class ClosedLoopIKWorld:
    """A 2-link planar arm with noisy target observations."""

    def __init__(self, seed: int = 2, max_steps: int = 80) -> None:
        self.seed = seed
        self.max_steps = max_steps
        self.config = ArmConfig(
            base=np.array([0.28, 0.26], dtype=float),
            link_lengths=np.array([0.40, 0.28], dtype=float),
            joint_limits=np.array([[-0.55, 2.65], [-2.55, 2.55]], dtype=float),
        )
        self.observation_noise = 0.008
        self.rng = np.random.default_rng(seed)
        self._figure: Any | None = None
        self._axis: Any | None = None
        self.reset(seed=seed)

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            self.seed = seed
            self.rng = np.random.default_rng(seed)

        self.time = 0
        self.q = np.array([0.30, -1.75], dtype=float)
        self.stable_count = 0
        self.last_observed_target: np.ndarray | None = None
        self.last_joint_delta = np.zeros(2, dtype=float)
        self.ee_path = [self.forward_kinematics(self.q)[-1]]
        self.target_path = [self.target_position(self.time)]
        return self.observe()

    def observe(self) -> dict[str, Any]:
        target = self.target_position(self.time)
        observed_target = np.clip(
            target + self.rng.normal(0.0, self.observation_noise, size=2),
            [0.0, 0.0],
            [1.0, 1.0],
        )
        self.last_observed_target = observed_target
        points = self.forward_kinematics(self.q)
        return {
            "time": self.time,
            "q": self.q.copy(),
            "end_effector": points[-1].copy(),
            "target_observation": observed_target.copy(),
            "stable_count": self.stable_count,
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
        tracking_error = float(np.linalg.norm(points[-1] - target))
        if tracking_error <= self.config.success_radius:
            self.stable_count += 1
        else:
            self.stable_count = 0

        done = self.stable_count >= self.config.stable_steps_required
        timeout = self.time >= self.max_steps
        info.update(
            {
                "tracking_error": tracking_error,
                "stable_count": self.stable_count,
                "joint_delta": joint_delta.copy(),
                "target": target.copy(),
            }
        )

        if done:
            info["success"] = True
        elif timeout:
            info["failure"] = Failure(
                "timeout",
                "closed-loop IK did not stabilize before max_steps",
                recoverable=False,
            )

        self.ee_path.append(points[-1].copy())
        self.target_path.append(target.copy())
        reward = 1.0 if done else -tracking_error
        return StepResult(self.observe(), reward, done or timeout, info)

    def target_position(self, time: int) -> np.ndarray:
        t = float(time)
        return np.array(
            [
                0.70 + 0.035 * np.cos(0.20 * t),
                0.54 + 0.030 * np.sin(0.15 * t),
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

    def render(self, agent: "ClosedLoopIKAgent", info: dict[str, Any] | None = None) -> None:
        import matplotlib.pyplot as plt

        if self._figure is None or self._axis is None:
            plt.ion()
            self._figure, self._axis = plt.subplots(figsize=(5.2, 5.2))

        draw_closed_loop_ik_scene(self._axis, self, agent, info)
        self._figure.canvas.draw_idle()
        plt.pause(0.04)


class ClosedLoopIKAgent:
    """Damped least-squares Jacobian servo controller."""

    def __init__(self) -> None:
        self.damping = 0.08
        self.task_gain = 0.82
        self.max_delta_norm = 0.12
        self.reset()

    def reset(self) -> None:
        self.target_belief: np.ndarray | None = None
        self.belief_radius = 0.060
        self.state = "observe_target"
        self.servo_updates = 0
        self.belief_updates = 0
        self.last_task_error = np.zeros(2, dtype=float)
        self.last_joint_delta = np.zeros(2, dtype=float)
        self._last_integrated_time: int | None = None

    def act(self, obs: dict[str, Any], env: ClosedLoopIKWorld) -> dict[str, Any]:
        self._integrate_observation(obs)
        if self.target_belief is None:
            self.state = "observe_target"
            return {"type": "servo", "joint_delta": np.zeros(2, dtype=float)}

        q = np.asarray(obs["q"], dtype=float)
        ee = np.asarray(obs["end_effector"], dtype=float)
        task_error = self.target_belief - ee
        jacobian = env.jacobian(q)
        damped = jacobian @ jacobian.T + (self.damping**2) * np.eye(2)
        joint_delta = self.task_gain * jacobian.T @ np.linalg.solve(damped, task_error)

        delta_norm = float(np.linalg.norm(joint_delta))
        if delta_norm > self.max_delta_norm:
            joint_delta = joint_delta / delta_norm * self.max_delta_norm

        self.last_task_error = task_error.copy()
        self.last_joint_delta = joint_delta.copy()
        self.servo_updates += 1
        self.state = "jacobian_servo"
        return {"type": "servo", "joint_delta": joint_delta}

    def update(self, obs: dict[str, Any], reward: float, info: dict[str, Any]) -> None:
        self._integrate_observation(obs)
        if info.get("success"):
            self.state = "stable_tracking"
        elif "failure" in info:
            self.state = "failed"

        info["agent_state"] = self.state
        info["servo_updates"] = self.servo_updates
        info["belief_updates"] = self.belief_updates
        info["belief_radius"] = self.belief_radius
        info["task_error_norm"] = float(np.linalg.norm(self.last_task_error))

    def _integrate_observation(self, obs: dict[str, Any]) -> None:
        obs_time = int(obs.get("time", -1))
        if obs_time == self._last_integrated_time:
            return
        self._last_integrated_time = obs_time

        observed_target = np.asarray(obs["target_observation"], dtype=float)
        if self.target_belief is None:
            self.target_belief = observed_target.copy()
        else:
            previous = self.target_belief.copy()
            self.target_belief = 0.34 * self.target_belief + 0.66 * observed_target
            if float(np.linalg.norm(self.target_belief - previous)) > 0.003:
                self.belief_updates += 1

        self.belief_radius = max(0.018, self.belief_radius * 0.88)


def draw_closed_loop_ik_scene(
    ax: Any,
    env: ClosedLoopIKWorld,
    agent: ClosedLoopIKAgent,
    info: dict[str, Any] | None = None,
) -> None:
    """Draw true target, noisy observation, belief, arm, and trajectories."""

    from matplotlib.patches import Circle

    info = {} if info is None else info
    ax.clear()
    ax.set_title("closed-loop IK")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)

    reach = float(np.sum(env.config.link_lengths))
    ax.add_patch(
        Circle(
            env.config.base,
            reach,
            fill=False,
            linestyle=":",
            linewidth=1.5,
            color="0.55",
            label="workspace",
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
    ax.add_patch(Circle(target, 0.020, color="tab:red", alpha=0.80, label="true target"))
    if env.last_observed_target is not None:
        ax.plot(
            *env.last_observed_target,
            marker="x",
            markersize=10,
            color="tab:orange",
            label="noisy observation",
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

    ee = points[-1]
    if agent.target_belief is not None:
        delta = agent.target_belief - ee
        ax.arrow(
            ee[0],
            ee[1],
            delta[0],
            delta[1],
            color="black",
            width=0.003,
            head_width=0.018,
            length_includes_head=True,
            alpha=0.65,
        )

    status = (
        f"step={env.time}  state={info.get('agent_state', agent.state)}\n"
        f"tracking_error={info.get('tracking_error', np.linalg.norm(ee - target)):.3f}  "
        f"stable={info.get('stable_count', env.stable_count)}\n"
        f"servo_updates={agent.servo_updates}  belief_updates={agent.belief_updates}"
    )
    if "failure" in info:
        status += f"  failure={info['failure'].kind}"
    if info.get("success"):
        status += "  success"
    ax.text(0.02, 0.98, status, transform=ax.transAxes, va="top", fontsize=9)
    ax.legend(loc="lower left", fontsize=8)


def run(seed: int = 2, render: bool = True, max_steps: int = 80) -> Trace:
    env = ClosedLoopIKWorld(seed=seed, max_steps=max_steps)
    agent = ClosedLoopIKAgent()
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
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    trace = run(seed=args.seed, render=not args.no_render, max_steps=args.max_steps)
    final_info = trace.infos[-1] if trace.infos else {}
    failures = [failure.kind for failure in trace.failures()]
    print(
        f"success={bool(final_info.get('success'))} steps={len(trace.actions)} "
        f"tracking_error={final_info.get('tracking_error', 0.0):.3f} "
        f"stable={final_info.get('stable_count', 0)} "
        f"servo_updates={final_info.get('servo_updates', 0)} "
        f"belief_updates={final_info.get('belief_updates', 0)} "
        f"failures={failures}"
    )

    if not args.no_render:
        import matplotlib.pyplot as plt

        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
