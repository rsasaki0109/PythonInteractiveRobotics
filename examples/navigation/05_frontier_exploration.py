"""Frontier exploration: move to observe unknown space."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pir.core.types import Failure, Trace
from pir.planning import astar as grid_astar
from pir.sensors.fake_lidar import DIRECTIONS
from pir.worlds.grid_world import FREE, OCCUPIED, UNKNOWN, GridWorld2D


class FrontierExplorationAgent:
    """Select known-free cells next to unknown cells, then move there."""

    def reset(self) -> None:
        self.current_frontier: tuple[int, int] | None = None
        self.current_path: list[tuple[int, int]] = []
        self.frontier_switches = 0
        self.coverage = 0.0
        self.state = "find_frontier"
        self.last_failure: Failure | None = None

    def act(self, obs: dict[str, Any]) -> str:
        robot = obs["robot"]
        known_map = obs["known_map"]
        self.coverage = coverage_ratio(known_map)

        frontiers = find_frontiers(known_map)
        if not frontiers:
            self.state = "no_frontier"
            self.current_frontier = None
            self.current_path = []
            return "stay"

        if self._need_new_frontier(robot, known_map, frontiers):
            self.current_frontier = choose_frontier(known_map, robot, frontiers)
            self.current_path = astar_known_free(known_map, robot, self.current_frontier)
            self.frontier_switches += 1
            self.state = "choose_frontier"

        if len(self.current_path) < 2:
            self.current_frontier = None
            self.current_path = []
            self.state = "frontier_reached"
            return "stay"

        next_cell = self.current_path[1]
        return direction_to(robot, next_cell)

    def update(self, obs: dict[str, Any], reward: float, info: dict[str, Any]) -> None:
        failure = info.get("failure")
        self.last_failure = failure if isinstance(failure, Failure) else None
        self.coverage = coverage_ratio(obs["known_map"])

        robot = obs["robot"]
        if self.current_path and robot in self.current_path:
            while self.current_path and self.current_path[0] != robot:
                self.current_path.pop(0)

        if self.last_failure is not None and self.last_failure.recoverable:
            self.current_frontier = None
            self.current_path = []
            self.state = "recover_and_choose"
            return

        if self.current_frontier == robot:
            self.current_frontier = None
            self.current_path = []
            self.state = "observe_from_frontier"
        elif self.state == "choose_frontier":
            self.state = "move_to_frontier"

    def _need_new_frontier(
        self,
        robot: tuple[int, int],
        known_map: np.ndarray,
        frontiers: list[tuple[int, int]],
    ) -> bool:
        if self.current_frontier is None:
            return True
        if self.current_frontier == robot:
            return True
        if self.current_frontier not in frontiers:
            return True
        if not self.current_path:
            return True
        return any(known_map[cell] != FREE for cell in self.current_path)


def find_frontiers(known_map: np.ndarray) -> list[tuple[int, int]]:
    frontiers: list[tuple[int, int]] = []
    height, width = known_map.shape
    for row in range(height):
        for col in range(width):
            cell = (row, col)
            if known_map[cell] != FREE:
                continue
            if any(
                known_map[neighbor] == UNKNOWN
                for neighbor in grid_neighbors(known_map, cell)
            ):
                frontiers.append(cell)
    return frontiers


def choose_frontier(
    known_map: np.ndarray,
    robot: tuple[int, int],
    frontiers: list[tuple[int, int]],
) -> tuple[int, int]:
    def score(cell: tuple[int, int]) -> tuple[float, int, tuple[int, int]]:
        distance = heuristic(robot, cell)
        unknown_neighbors = sum(
            known_map[neighbor] == UNKNOWN
            for neighbor in grid_neighbors(known_map, cell)
        )
        return (distance - 2.5 * unknown_neighbors, distance, cell)

    return min(frontiers, key=score)


def astar_known_free(
    known_map: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]]:
    return grid_astar(known_map == FREE, start, goal)


def grid_neighbors(
    known_map: np.ndarray,
    cell: tuple[int, int],
) -> list[tuple[int, int]]:
    height, width = known_map.shape
    result: list[tuple[int, int]] = []
    for dr, dc in DIRECTIONS.values():
        neighbor = (cell[0] + dr, cell[1] + dc)
        row, col = neighbor
        if 0 <= row < height and 0 <= col < width:
            result.append(neighbor)
    return result


def heuristic(cell: tuple[int, int], goal: tuple[int, int]) -> int:
    return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])


def direction_to(start: tuple[int, int], end: tuple[int, int]) -> str:
    delta = (end[0] - start[0], end[1] - start[1])
    for direction, direction_delta in DIRECTIONS.items():
        if delta == direction_delta:
            return direction
    return "stay"


def coverage_ratio(known_map: np.ndarray) -> float:
    return float(np.count_nonzero(known_map != UNKNOWN) / known_map.size)


def run(
    seed: int = 0,
    render: bool = True,
    max_steps: int = 120,
    coverage_goal: float = 0.58,
) -> Trace:
    env = GridWorld2D(seed=seed, lidar_range=4, max_steps=max_steps)
    agent = FrontierExplorationAgent()
    obs = env.reset(seed=seed)
    agent.reset()
    trace = Trace()

    for _ in range(max_steps):
        action = agent.act(obs)
        result = env.step(action)
        obs, reward, env_done, info = result.as_tuple()
        agent.update(obs, reward, info)

        frontiers = find_frontiers(obs["known_map"])
        success = agent.coverage >= coverage_goal
        done = success or (env_done and info.get("failure") is not None)
        info["success"] = success
        info["agent_state"] = agent.state
        info["coverage"] = agent.coverage
        info["coverage_goal"] = coverage_goal
        info["frontier_count"] = len(frontiers)
        info["frontier_switches"] = agent.frontier_switches
        info["planned_path_length"] = len(agent.current_path)
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
    parser.add_argument("--coverage-goal", type=float, default=0.58)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    trace = run(
        seed=args.seed,
        render=not args.no_render,
        max_steps=args.max_steps,
        coverage_goal=args.coverage_goal,
    )
    success = bool(trace.infos and trace.infos[-1].get("success"))
    coverage = trace.infos[-1].get("coverage", 0.0) if trace.infos else 0.0
    switches = trace.infos[-1].get("frontier_switches", 0) if trace.infos else 0
    failures = [failure.kind for failure in trace.failures()]
    print(
        f"success={success} steps={len(trace.actions)} coverage={coverage:.2f} "
        f"frontier_switches={switches} failures={failures}"
    )

    if not args.no_render:
        import matplotlib.pyplot as plt

        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
