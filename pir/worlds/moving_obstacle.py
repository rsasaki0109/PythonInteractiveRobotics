"""Continuous 2D world with a single moving obstacle.

The world owns the robot pose, the goal, and the moving obstacle. The example
that uses this world keeps the MPC-style agent and the teaching loop, so the
package stays free of agent-specific knowledge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pir.core.types import Failure, StepResult


@dataclass(frozen=True)
class MPCConfig:
    """Small set of constants that define the toy world and controller."""

    dt: float = 1.0
    horizon: int = 8
    speed: float = 0.38
    goal_radius: float = 0.32
    robot_radius: float = 0.18
    obstacle_radius: float = 0.45
    safety_margin: float = 0.35
    world_min: float = 0.0
    world_max: float = 10.0


class MovingObstacleWorld:
    """Continuous 2D world with one robot, one goal, and one moving obstacle."""

    def __init__(self, seed: int = 0, max_steps: int = 120) -> None:
        self.config = MPCConfig()
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.step_count = 0
        self.robot = np.zeros(2, dtype=float)
        self.goal = np.zeros(2, dtype=float)
        self.obstacle = np.zeros(2, dtype=float)
        self.obstacle_velocity = np.zeros(2, dtype=float)
        self._figure: Any | None = None
        self._axis: Any | None = None

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.seed = seed
        self.step_count = 0
        self.robot = np.array([1.0, 1.0], dtype=float)
        self.goal = np.array([8.9, 8.6], dtype=float)

        jitter = self.rng.uniform(-0.15, 0.15, size=2)
        self.obstacle = np.array([5.1, 6.8], dtype=float) + jitter
        self.obstacle_velocity = np.array([0.0, -0.22], dtype=float)
        return self.observe()

    def observe(self) -> dict[str, Any]:
        return {
            "robot": self.robot.copy(),
            "goal": self.goal.copy(),
            "obstacle": self.obstacle.copy(),
            "obstacle_velocity": self.obstacle_velocity.copy(),
            "step": self.step_count,
        }

    def step(self, control: np.ndarray) -> StepResult:
        cfg = self.config
        self.step_count += 1
        control = np.asarray(control, dtype=float)
        speed = float(np.linalg.norm(control))
        if speed > cfg.speed:
            control = control / speed * cfg.speed

        self.robot = np.clip(
            self.robot + control * cfg.dt,
            cfg.world_min,
            cfg.world_max,
        )
        self._move_obstacle()

        goal_distance = float(np.linalg.norm(self.goal - self.robot))
        clearance = float(np.linalg.norm(self.robot - self.obstacle))
        collision_distance = cfg.robot_radius + cfg.obstacle_radius
        success = goal_distance <= cfg.goal_radius
        collision = clearance <= collision_distance
        timeout = self.step_count >= self.max_steps
        done = success or collision or timeout

        reward = -goal_distance - 0.1
        info: dict[str, Any] = {
            "success": success,
            "clearance": clearance,
            "goal_distance": goal_distance,
        }
        if collision:
            info["failure"] = Failure(
                kind="collision",
                message="Robot entered the moving obstacle radius.",
                recoverable=False,
            )
        elif timeout and not success:
            info["failure"] = Failure(
                kind="timeout",
                message="MPC did not reach the goal before max_steps.",
                recoverable=False,
            )
        return StepResult(self.observe(), reward, done, info)

    def predict_obstacle(self, start: np.ndarray, velocity: np.ndarray, steps: int) -> np.ndarray:
        """Predict future obstacle centers using the same bounce model as the world."""

        cfg = self.config
        centers = []
        position = np.asarray(start, dtype=float).copy()
        vel = np.asarray(velocity, dtype=float).copy()
        for _ in range(steps):
            position = position + vel * cfg.dt
            for axis in range(2):
                low = cfg.world_min + cfg.obstacle_radius
                high = cfg.world_max - cfg.obstacle_radius
                if position[axis] < low:
                    position[axis] = low + (low - position[axis])
                    vel[axis] *= -1.0
                elif position[axis] > high:
                    position[axis] = high - (position[axis] - high)
                    vel[axis] *= -1.0
            centers.append(position.copy())
        return np.asarray(centers)

    def render(self, agent: Any | None = None, info: dict[str, Any] | None = None) -> None:
        import matplotlib.pyplot as plt

        cfg = self.config
        if self._figure is None or self._axis is None:
            self._figure, self._axis = plt.subplots(figsize=(6, 6))
            plt.ion()
        ax = self._axis
        ax.clear()
        ax.set_title("Toy interactive MPC")
        ax.set_xlim(cfg.world_min, cfg.world_max)
        ax.set_ylim(cfg.world_min, cfg.world_max)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)

        last_prediction = getattr(agent, "last_prediction", None)
        last_obstacle_prediction = getattr(agent, "last_obstacle_prediction", None)
        if last_prediction is not None:
            ax.plot(
                last_prediction[:, 0],
                last_prediction[:, 1],
                color="tab:blue",
                linewidth=2,
                alpha=0.75,
                label="chosen rollout",
            )
        if last_obstacle_prediction is not None:
            ax.plot(
                last_obstacle_prediction[:, 0],
                last_obstacle_prediction[:, 1],
                color="tab:red",
                linestyle="--",
                alpha=0.6,
                label="obstacle prediction",
            )

        goal = plt.Circle(self.goal, cfg.goal_radius, color="tab:green", alpha=0.25)
        robot = plt.Circle(self.robot, cfg.robot_radius, color="tab:blue")
        obstacle = plt.Circle(self.obstacle, cfg.obstacle_radius, color="tab:red", alpha=0.65)
        safety = plt.Circle(
            self.obstacle,
            cfg.obstacle_radius + cfg.robot_radius + cfg.safety_margin,
            color="tab:red",
            fill=False,
            linestyle=":",
            alpha=0.6,
        )
        ax.add_patch(goal)
        ax.add_patch(safety)
        ax.add_patch(obstacle)
        ax.add_patch(robot)
        ax.scatter([self.goal[0]], [self.goal[1]], marker="*", s=160, color="tab:green")
        if info is not None:
            ax.text(0.02, 0.98, self._status_text(info), transform=ax.transAxes, va="top")
        ax.legend(loc="lower right")
        self._figure.canvas.draw_idle()
        plt.pause(0.001)

    def _move_obstacle(self) -> None:
        predicted = self.predict_obstacle(self.obstacle, self.obstacle_velocity, 1)
        next_position = predicted[0]
        next_velocity = self.obstacle_velocity.copy()
        cfg = self.config
        raw_position = self.obstacle + self.obstacle_velocity * cfg.dt
        for axis in range(2):
            low = cfg.world_min + cfg.obstacle_radius
            high = cfg.world_max - cfg.obstacle_radius
            if raw_position[axis] < low or raw_position[axis] > high:
                next_velocity[axis] *= -1.0
        self.obstacle = next_position
        self.obstacle_velocity = next_velocity

    def _status_text(self, info: dict[str, Any]) -> str:
        return (
            f"step={self.step_count}\n"
            f"state={info.get('agent_state', 'unknown')}\n"
            f"best_cost={info.get('best_cost', 0.0):.2f}\n"
            f"risk={info.get('predicted_collision_risk', 0.0):.2f}"
        )
