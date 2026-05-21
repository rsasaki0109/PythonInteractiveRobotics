"""Avoid a moving obstacle by observing and predicting one step ahead."""

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
from pir.worlds.grid_world import DynamicObstacleGridWorld


class OneStepLookaheadAgent:
    """Greedy navigation with a one-step moving obstacle prediction."""

    def reset(self) -> None:
        self.state = "go_to_goal"
        self.wait_count = 0
        self.reactive_avoid_count = 0
        self.last_failure: Failure | None = None

    def act(self, obs: dict[str, Any]) -> str:
        robot = obs["robot"]
        goal = obs["goal"]
        lidar = obs["lidar"]
        dynamic = set(obs["dynamic_obstacles"])
        predicted = set(obs["predicted_dynamic_obstacles"])

        candidates: list[tuple[float, str]] = []
        for direction, (dr, dc) in DIRECTIONS.items():
            if lidar[direction]["free_cells"] <= 0:
                continue

            next_cell = (robot[0] + dr, robot[1] + dc)
            if next_cell in dynamic or next_cell in predicted:
                continue

            score = self._distance(next_cell, goal)
            if self._near_any(next_cell, dynamic | predicted):
                score += 3.0
            if direction not in self._goal_directions(robot, goal)[:2]:
                score += 1.5
            candidates.append((score, direction))

        if not candidates:
            self.state = "wait_for_gap"
            self.wait_count += 1
            return "stay"

        candidates.sort()
        direction = candidates[0][1]
        preferred = self._goal_directions(robot, goal)[0]
        if direction != preferred:
            self.state = "avoid_moving_obstacle"
            self.reactive_avoid_count += 1
        else:
            self.state = "go_to_goal"
        return direction

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

    def _distance(self, cell: tuple[int, int], goal: tuple[int, int]) -> int:
        return abs(goal[0] - cell[0]) + abs(goal[1] - cell[1])

    def _near_any(
        self,
        cell: tuple[int, int],
        obstacles: set[tuple[int, int]],
    ) -> bool:
        return any(abs(cell[0] - row) + abs(cell[1] - col) <= 1 for row, col in obstacles)


def run(seed: int = 0, render: bool = True, max_steps: int = 90) -> Trace:
    env = DynamicObstacleGridWorld(seed=seed, max_steps=max_steps)
    agent = OneStepLookaheadAgent()
    obs = env.reset(seed=seed)
    agent.reset()
    trace = Trace()

    for _ in range(max_steps):
        action = agent.act(obs)
        result = env.step(action)
        obs, reward, done, info = result.as_tuple()
        agent.update(obs, reward, info)
        info["agent_state"] = agent.state
        info["wait_count"] = agent.wait_count
        info["reactive_avoid_count"] = agent.reactive_avoid_count
        trace.append(obs, action, reward, info)

        if render:
            env.render(agent=agent, info=info)

        if done:
            break

    return trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=90)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    trace = run(seed=args.seed, render=not args.no_render, max_steps=args.max_steps)
    success = bool(trace.infos and trace.infos[-1].get("success"))
    failures = [failure.kind for failure in trace.failures()]
    avoid_count = trace.infos[-1].get("reactive_avoid_count", 0) if trace.infos else 0
    wait_count = trace.infos[-1].get("wait_count", 0) if trace.infos else 0
    print(
        f"success={success} steps={len(trace.actions)} "
        f"avoid_count={avoid_count} wait_count={wait_count} failures={failures}"
    )

    if not args.no_render:
        import matplotlib.pyplot as plt

        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
