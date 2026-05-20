"""Interactive model-predictive control in a tiny continuous world.

This is an educational MPC-style example, not a full optimizer.  At every
step the robot samples a small set of candidate velocity commands, rolls each
one forward over a short horizon with a simple obstacle prediction model, and
executes only the first command from the lowest-cost rollout.  The obstacle is
then observed at its new position and the process repeats.
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

from pir.core.types import Failure, Trace


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

    def step(self, control: np.ndarray) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
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
        return self.observe(), reward, done, info

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

    def render(self, agent: "ToyMPCAgent", info: dict[str, Any]) -> None:
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

        if agent.last_prediction is not None:
            ax.plot(
                agent.last_prediction[:, 0],
                agent.last_prediction[:, 1],
                color="tab:blue",
                linewidth=2,
                alpha=0.75,
                label="chosen rollout",
            )
        if agent.last_obstacle_prediction is not None:
            ax.plot(
                agent.last_obstacle_prediction[:, 0],
                agent.last_obstacle_prediction[:, 1],
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


class ToyMPCAgent:
    """Sampled short-horizon controller for a point robot."""

    def __init__(self, config: MPCConfig) -> None:
        self.config = config
        self.replan_count = 0
        self.state = "plan"
        self.best_cost = float("inf")
        self.predicted_collision_risk = 0.0
        self.last_prediction: np.ndarray | None = None
        self.last_obstacle_prediction: np.ndarray | None = None

    def reset(self) -> None:
        self.replan_count = 0
        self.state = "plan"
        self.best_cost = float("inf")
        self.predicted_collision_risk = 0.0
        self.last_prediction = None
        self.last_obstacle_prediction = None

    def act(self, obs: dict[str, Any], world: MovingObstacleWorld) -> np.ndarray:
        cfg = self.config
        robot = np.asarray(obs["robot"], dtype=float)
        goal = np.asarray(obs["goal"], dtype=float)
        obstacle = np.asarray(obs["obstacle"], dtype=float)
        obstacle_velocity = np.asarray(obs["obstacle_velocity"], dtype=float)
        obstacle_prediction = world.predict_obstacle(
            obstacle,
            obstacle_velocity,
            cfg.horizon,
        )

        best_control = np.zeros(2, dtype=float)
        best_cost = float("inf")
        best_rollout: np.ndarray | None = None
        best_risk = 1.0

        for control in self._candidate_controls(robot, goal):
            rollout = self._rollout(robot, control, goal)
            cost, risk = self._rollout_cost(rollout, goal, obstacle_prediction)
            if cost < best_cost:
                best_control = control
                best_cost = cost
                best_rollout = rollout
                best_risk = risk

        self.replan_count += 1
        self.best_cost = best_cost
        self.predicted_collision_risk = best_risk
        self.last_prediction = best_rollout
        self.last_obstacle_prediction = obstacle_prediction
        self.state = "avoid_obstacle" if best_risk > 0.05 else "go_to_goal"
        if np.linalg.norm(goal - robot) <= cfg.goal_radius:
            self.state = "arrived"
        return best_control

    def info(self) -> dict[str, Any]:
        return {
            "agent_state": self.state,
            "best_cost": float(self.best_cost),
            "predicted_collision_risk": float(self.predicted_collision_risk),
            "replan_count": self.replan_count,
        }

    def _candidate_controls(self, robot: np.ndarray, goal: np.ndarray) -> list[np.ndarray]:
        cfg = self.config
        distance_to_goal = float(np.linalg.norm(goal - robot))
        goal_angle = float(np.arctan2(goal[1] - robot[1], goal[0] - robot[0]))
        offsets = np.deg2rad(np.array([0, -65, 65, -35, 35, -100, 100, 180], dtype=float))
        braking_speed = min(cfg.speed, distance_to_goal)
        speeds = (cfg.speed, cfg.speed * 0.65, braking_speed)

        controls = [np.zeros(2, dtype=float)]
        for speed in speeds:
            for angle in goal_angle + offsets:
                controls.append(speed * np.array([np.cos(angle), np.sin(angle)]))
        return controls

    def _rollout(self, robot: np.ndarray, control: np.ndarray, goal: np.ndarray) -> np.ndarray:
        cfg = self.config
        position = robot.copy()
        states = []
        for _ in range(cfg.horizon):
            position = np.clip(
                position + control * cfg.dt,
                cfg.world_min,
                cfg.world_max,
            )
            states.append(position.copy())
            if np.linalg.norm(position - goal) <= cfg.goal_radius:
                states.extend([position.copy()] * (cfg.horizon - len(states)))
                break
        return np.asarray(states)

    def _rollout_cost(
        self,
        rollout: np.ndarray,
        goal: np.ndarray,
        obstacle_prediction: np.ndarray,
    ) -> tuple[float, float]:
        cfg = self.config
        distances_to_goal = np.linalg.norm(rollout - goal, axis=1)
        obstacle_distances = np.linalg.norm(rollout - obstacle_prediction, axis=1)
        collision_distance = cfg.robot_radius + cfg.obstacle_radius
        warning_distance = collision_distance + cfg.safety_margin

        clearance_shortfall = np.maximum(0.0, warning_distance - obstacle_distances)
        collision_shortfall = np.maximum(0.0, collision_distance - obstacle_distances)
        risk = float(np.max(clearance_shortfall / warning_distance))

        final_distance = float(distances_to_goal[-1])
        path_distance = float(np.mean(distances_to_goal))
        obstacle_cost = float(65.0 * np.sum(clearance_shortfall**2))
        collision_cost = float(800.0 * np.sum(collision_shortfall**2))
        wall_cost = float(4.0 * np.sum(self._wall_penalty(rollout)))
        return final_distance * 5.0 + path_distance + obstacle_cost + collision_cost + wall_cost, risk

    def _wall_penalty(self, rollout: np.ndarray) -> np.ndarray:
        cfg = self.config
        low = np.maximum(0.0, 0.4 - (rollout - cfg.world_min))
        high = np.maximum(0.0, 0.4 - (cfg.world_max - rollout))
        return np.sum(low + high, axis=1)


def run(seed: int = 0, render: bool = True, max_steps: int = 120) -> Trace:
    env = MovingObstacleWorld(seed=seed, max_steps=max_steps)
    agent = ToyMPCAgent(env.config)
    obs = env.reset(seed=seed)
    agent.reset()
    trace = Trace()

    for _ in range(max_steps):
        action = agent.act(obs, env)
        obs, reward, done, info = env.step(action)
        info.update(agent.info())
        if done and info.get("success"):
            info["agent_state"] = "arrived"
        trace.append(obs, action, reward, info)

        if render:
            env.render(agent=agent, info=info)

        if done:
            break

    return trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    trace = run(seed=args.seed, render=not args.no_render, max_steps=args.max_steps)
    success = bool(trace.infos and trace.infos[-1].get("success"))
    failures = [failure.kind for failure in trace.failures()]
    final_info = trace.infos[-1] if trace.infos else {}
    print(
        f"success={success} steps={len(trace.actions)} "
        f"replan_count={final_info.get('replan_count', 0)} "
        f"best_cost={final_info.get('best_cost', 0.0):.3f} "
        f"predicted_collision_risk={final_info.get('predicted_collision_risk', 0.0):.3f} "
        f"failures={failures}"
    )

    if not args.no_render:
        import matplotlib.pyplot as plt

        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
