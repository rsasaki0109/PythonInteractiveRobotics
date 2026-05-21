"""Interactive model-predictive control in a tiny continuous world.

This is an educational MPC-style example, not a full optimizer.  At every
step the robot samples a small set of candidate velocity commands, rolls each
one forward over a short horizon with a simple obstacle prediction model, and
executes only the first command from the lowest-cost rollout.  The obstacle is
then observed at its new position and the process repeats.

Success: robot reaches the goal radius before max_steps.
Failure: collision (terminal) or timeout (terminal).
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

from pir.core.types import Trace
from pir.worlds.moving_obstacle import MPCConfig, MovingObstacleWorld


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
        obs, reward, done, info = env.step(action).as_tuple()
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
