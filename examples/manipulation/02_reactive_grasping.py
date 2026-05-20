"""Reactive grasping with visual drift, contact failure, and servo correction.

The robot does not compute one grasp pose and execute it blindly.  It keeps
observing a moving object, updates a small belief, servos the gripper toward
that belief, detects a miss through contact, then realigns with the corrected
visual observation.
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


class ReactiveGraspWorld:
    """A tiny tabletop where an object drifts while the gripper servos."""

    def __init__(self, seed: int = 4, max_steps: int = 60) -> None:
        self.seed = seed
        self.max_steps = max_steps
        self.table_size = np.array([1.0, 1.0], dtype=float)
        self.gripper_speed = 0.065
        self.grasp_radius = 0.048
        self.object_radius = 0.04
        self.detector_noise = 0.010
        self.initial_camera_bias = np.array([0.074, -0.046], dtype=float)
        self.rng = np.random.default_rng(seed)
        self._figure: Any | None = None
        self._axis: Any | None = None
        self.reset(seed=seed)

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            self.seed = seed
            self.rng = np.random.default_rng(seed)

        self.time = 0
        self.gripper = np.array([0.18, 0.20], dtype=float)
        self.object_position = np.array([0.70, 0.66], dtype=float)
        self.object_velocity = np.array([-0.0045, -0.0020], dtype=float)
        self.holding = False
        self.close_attempts = 0
        self.calibration_corrected = False
        self.last_detection: np.ndarray | None = None
        self.last_servo_target: np.ndarray | None = None
        self.gripper_path = [self.gripper.copy()]
        self.object_path = [self.object_position.copy()]
        return self.observe()

    def observe(self) -> dict[str, Any]:
        detections: list[dict[str, Any]] = []
        if not self.holding:
            bias = np.zeros(2) if self.calibration_corrected else self.initial_camera_bias
            noise = self.rng.normal(0.0, self.detector_noise, size=2)
            detection = self._clip(self.object_position + bias + noise)
            self.last_detection = detection
            detections.append(
                {
                    "name": "block",
                    "color": "tab:red",
                    "position": detection,
                    "confidence": 0.92 if self.calibration_corrected else 0.58,
                    "bias_corrected": self.calibration_corrected,
                }
            )

        return {
            "time": self.time,
            "gripper": {"position": self.gripper.copy(), "holding": self.holding},
            "detections": detections,
            "attempts": self.close_attempts,
            "calibration_corrected": self.calibration_corrected,
        }

    def step(self, action: dict[str, Any]) -> StepResult:
        self.time += 1
        action_type = action.get("type", "scan")
        info: dict[str, Any] = {
            "time": self.time,
            "action_type": action_type,
            "success": False,
            "attempts": self.close_attempts,
            "calibration_corrected": self.calibration_corrected,
        }

        if self.holding:
            return StepResult(self.observe(), 0.0, True, info)

        if action_type == "scan":
            self._move_object()
            info["alignment_error"] = self._alignment_error()
            return StepResult(self.observe(), -0.01, False, info)

        if action_type == "servo":
            return self._step_servo(action, info)

        if action_type == "close":
            return self._step_close(info)

        info["failure"] = Failure(
            "invalid_action",
            f"unknown action type: {action_type}",
            recoverable=True,
        )
        self._move_object()
        return StepResult(self.observe(), -0.05, False, info)

    def _step_servo(self, action: dict[str, Any], info: dict[str, Any]) -> StepResult:
        target = self._clip(np.asarray(action.get("target", self.gripper), dtype=float))
        delta = target - self.gripper
        distance = float(np.linalg.norm(delta))
        if distance > self.gripper_speed:
            delta = delta / distance * self.gripper_speed
        self.gripper = self._clip(self.gripper + delta)
        self.last_servo_target = target.copy()
        self._move_object()

        error = self._alignment_error()
        info.update(
            {
                "servo_target": target.copy(),
                "alignment_error": error,
                "attempts": self.close_attempts,
                "calibration_corrected": self.calibration_corrected,
            }
        )
        reward = -0.02 - error
        return StepResult(self.observe(), reward, False, info)

    def _step_close(self, info: dict[str, Any]) -> StepResult:
        self.close_attempts += 1
        error = self._alignment_error()
        info.update(
            {
                "attempts": self.close_attempts,
                "alignment_error": error,
                "grasp_radius": self.grasp_radius,
                "calibration_corrected": self.calibration_corrected,
            }
        )

        if error <= self.grasp_radius:
            self.holding = True
            self.object_position = self.gripper.copy()
            self.object_path.append(self.object_position.copy())
            info["success"] = True
            return StepResult(self.observe(), 1.0, True, info)

        self.calibration_corrected = True
        info["calibration_corrected"] = True
        info["failure"] = Failure(
            "grasp_miss",
            "contact failed, so the visual bias is corrected before retrying",
            recoverable=True,
        )
        return StepResult(self.observe(), -0.18, False, info)

    def _move_object(self) -> None:
        self.object_position = self._clip(self.object_position + self.object_velocity)
        for axis in range(2):
            if self.object_position[axis] < 0.16 or self.object_position[axis] > 0.86:
                self.object_velocity[axis] *= -1.0
        self.gripper_path.append(self.gripper.copy())
        self.object_path.append(self.object_position.copy())

    def _alignment_error(self) -> float:
        return float(np.linalg.norm(self.gripper - self.object_position))

    def _clip(self, position: np.ndarray) -> np.ndarray:
        return np.clip(position, [0.0, 0.0], self.table_size)

    def render(self, agent: "ReactiveGraspAgent", info: dict[str, Any] | None = None) -> None:
        import matplotlib.pyplot as plt

        if self._figure is None or self._axis is None:
            plt.ion()
            self._figure, self._axis = plt.subplots(figsize=(5.2, 5.2))

        draw_reactive_grasp_scene(self._axis, self, agent, info)
        self._figure.canvas.draw_idle()
        plt.pause(0.04)


class ReactiveGraspAgent:
    """Servo toward the newest object belief and retry after contact failure."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.belief_mean: np.ndarray | None = None
        self.belief_radius = 0.18
        self.state = "search"
        self.stable_count = 0
        self.miss_count = 0
        self.servo_steps = 0
        self.reactive_updates = 0
        self.total_belief_shift = 0.0
        self._last_integrated_time: int | None = None

    def act(self, obs: dict[str, Any]) -> dict[str, Any]:
        self._integrate_observation(obs)

        if self.belief_mean is None:
            self.state = "search"
            return {"type": "scan"}

        gripper = np.asarray(obs["gripper"]["position"], dtype=float)
        belief_error = float(np.linalg.norm(self.belief_mean - gripper))
        if belief_error < 0.022 and self.belief_radius < 0.050:
            self.stable_count += 1
        else:
            self.stable_count = 0

        if self.stable_count >= 2:
            self.state = "close_on_belief"
            return {"type": "close"}

        self.state = "servo_to_updated_belief"
        self.servo_steps += 1
        return {"type": "servo", "target": self.belief_mean.copy()}

    def update(self, obs: dict[str, Any], reward: float, info: dict[str, Any]) -> None:
        self._integrate_observation(obs)

        failure = info.get("failure")
        if isinstance(failure, Failure) and failure.kind == "grasp_miss":
            self.miss_count += 1
            self.stable_count = 0
            self.belief_radius = min(0.16, self.belief_radius + 0.07)
            self.state = "recover_realign"
        elif info.get("success"):
            self.state = "holding"

        info["agent_state"] = self.state
        info["miss_count"] = self.miss_count
        info["servo_steps"] = self.servo_steps
        info["reactive_updates"] = self.reactive_updates
        info["belief_shift"] = self.total_belief_shift

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
        new_gain = 0.78 if confidence > 0.8 else 0.64

        if self.belief_mean is None:
            self.belief_mean = position.copy()
        else:
            previous = self.belief_mean.copy()
            self.belief_mean = (1.0 - new_gain) * self.belief_mean + new_gain * position
            shift = float(np.linalg.norm(self.belief_mean - previous))
            self.total_belief_shift += shift
            if shift > 0.008:
                self.reactive_updates += 1

        shrink = 0.68 if confidence > 0.8 else 0.78
        self.belief_radius = max(0.028, self.belief_radius * shrink)


def draw_reactive_grasp_scene(
    ax: Any,
    env: ReactiveGraspWorld,
    agent: ReactiveGraspAgent,
    info: dict[str, Any] | None = None,
) -> None:
    """Draw true state, noisy observation, belief, and servo target."""

    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    info = {} if info is None else info
    ax.clear()
    ax.set_title("reactive grasping")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)

    object_path = np.asarray(env.object_path)
    gripper_path = np.asarray(env.gripper_path)
    if len(object_path) > 1:
        ax.plot(object_path[:, 0], object_path[:, 1], color="tab:red", alpha=0.35)
    if len(gripper_path) > 1:
        ax.plot(gripper_path[:, 0], gripper_path[:, 1], color="tab:blue", linewidth=2)

    if not env.holding:
        ax.add_patch(
            Circle(
                env.object_position,
                env.object_radius,
                color="tab:red",
                alpha=0.78,
                label="true object",
            )
        )

    if env.last_detection is not None and not env.holding:
        ax.plot(
            *env.last_detection,
            marker="x",
            markersize=10,
            color="tab:orange",
            label="visual detection",
        )

    if agent.belief_mean is not None:
        ax.add_patch(
            Circle(
                agent.belief_mean,
                agent.belief_radius,
                fill=False,
                linestyle="--",
                linewidth=2,
                color="tab:green",
                label="belief",
            )
        )

    if env.last_servo_target is not None:
        ax.plot(
            *env.last_servo_target,
            marker="+",
            markersize=13,
            color="black",
            label="servo target",
        )

    ax.add_patch(Circle(env.gripper, 0.030, color="tab:blue", label="gripper"))
    ax.plot(
        [env.gripper[0] - 0.045, env.gripper[0] + 0.045],
        [env.gripper[1] + 0.035, env.gripper[1] + 0.035],
        color="tab:blue",
        linewidth=3,
    )

    status = (
        f"step={env.time}  state={info.get('agent_state', agent.state)}\n"
        f"error={info.get('alignment_error', env._alignment_error()):.3f}  "
        f"misses={agent.miss_count}  updates={agent.reactive_updates}\n"
        f"bias_corrected={env.calibration_corrected}"
    )
    if "failure" in info:
        status += f"  failure={info['failure'].kind}"
    if info.get("success"):
        status += "  success"
    ax.text(0.02, 0.98, status, transform=ax.transAxes, va="top", fontsize=9)
    ax.legend(loc="lower left", fontsize=8)


def run(seed: int = 4, render: bool = True, max_steps: int = 60) -> Trace:
    env = ReactiveGraspWorld(seed=seed, max_steps=max_steps)
    agent = ReactiveGraspAgent()
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
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    trace = run(seed=args.seed, render=not args.no_render, max_steps=args.max_steps)
    final_info = trace.infos[-1] if trace.infos else {}
    failures = [failure.kind for failure in trace.failures()]
    print(
        f"success={bool(final_info.get('success'))} steps={len(trace.actions)} "
        f"misses={final_info.get('miss_count', 0)} "
        f"servo_steps={final_info.get('servo_steps', 0)} "
        f"belief_shift={final_info.get('belief_shift', 0.0):.3f} "
        f"failures={failures}"
    )

    if not args.no_render:
        import matplotlib.pyplot as plt

        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
