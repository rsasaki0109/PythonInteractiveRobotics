"""Reactive obstacle avoidance with a fake grid lidar.

The agent has no global planner. It repeatedly observes free cells around the
robot, chooses a safe direction that still moves toward the goal, and observes
again after the environment changes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pir.core.types import Failure, Trace
from pir.sensors.fake_lidar import DIRECTIONS
from pir.worlds.grid_world import GridWorld2D


class ReactiveLidarAgent:
    """Choose goal-directed moves, but only through currently observed free cells."""

    def reset(self) -> None:
        self.state = "go_to_goal"
        self.avoidance_count = 0
        self.last_failure: Failure | None = None

    def act(self, obs: dict[str, Any]) -> str:
        robot = obs["robot"]
        goal = obs["goal"]
        lidar = obs["lidar"]

        goal_candidates = self._goal_directions(robot, goal)
        primary = goal_candidates[0]
        primary_blocked = lidar[primary]["free_cells"] == 0
        safe_goal_candidates = [
            direction
            for direction in goal_candidates
            if lidar[direction]["free_cells"] > 0
        ]
        if safe_goal_candidates:
            self.state = "avoid_obstacle" if primary_blocked else "go_to_goal"
            if primary_blocked:
                self.avoidance_count += 1
            return safe_goal_candidates[0]

        self.state = "avoid_obstacle"
        self.avoidance_count += 1
        safe_fallbacks = [
            direction
            for direction in ("east", "north", "south", "west")
            if lidar[direction]["free_cells"] > 0
        ]
        if safe_fallbacks:
            return min(
                safe_fallbacks,
                key=lambda direction: self._distance_after_move(robot, goal, direction),
            )

        return "stay"

    def update(self, obs: dict[str, Any], reward: float, info: dict[str, Any]) -> None:
        failure = info.get("failure")
        self.last_failure = failure if isinstance(failure, Failure) else None

    def _goal_directions(
        self,
        robot: tuple[int, int],
        goal: tuple[int, int],
    ) -> list[str]:
        row_delta = goal[0] - robot[0]
        col_delta = goal[1] - robot[1]
        vertical = "south" if row_delta > 0 else "north"
        horizontal = "east" if col_delta > 0 else "west"

        candidates: list[str] = []
        if abs(col_delta) >= abs(row_delta) and col_delta != 0:
            candidates.append(horizontal)
        if row_delta != 0:
            candidates.append(vertical)
        if abs(col_delta) < abs(row_delta) and col_delta != 0:
            candidates.append(horizontal)

        for direction in ("east", "north", "south", "west"):
            if direction not in candidates:
                candidates.append(direction)
        return candidates

    def _distance_after_move(
        self,
        robot: tuple[int, int],
        goal: tuple[int, int],
        direction: str,
    ) -> int:
        dr, dc = DIRECTIONS[direction]
        next_cell = (robot[0] + dr, robot[1] + dc)
        return abs(goal[0] - next_cell[0]) + abs(goal[1] - next_cell[1])


def run(seed: int = 0, render: bool = True, max_steps: int = 80) -> Trace:
    env = GridWorld2D(seed=seed, max_steps=max_steps)
    agent = ReactiveLidarAgent()
    obs = env.reset(seed=seed)
    agent.reset()
    trace = Trace()

    for _ in range(max_steps):
        action = agent.act(obs)
        result = env.step(action)
        obs, reward, done, info = result.as_tuple()
        agent.update(obs, reward, info)
        info["agent_state"] = agent.state
        info["avoidance_count"] = agent.avoidance_count
        trace.append(obs, action, reward, info)

        if render:
            env.render(agent=agent, info=info)

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
    success = bool(trace.infos and trace.infos[-1].get("success"))
    failures = [failure.kind for failure in trace.failures()]
    avoidance_count = trace.infos[-1].get("avoidance_count", 0) if trace.infos else 0
    print(
        f"success={success} steps={len(trace.actions)} "
        f"avoidance_count={avoidance_count} failures={failures}"
    )

    if not args.no_render:
        import matplotlib.pyplot as plt

        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
