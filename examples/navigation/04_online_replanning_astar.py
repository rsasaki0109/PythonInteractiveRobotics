"""Online A* replanning when new obstacles are observed."""

from __future__ import annotations

import argparse
import heapq
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pir.core.types import Failure, Trace
from pir.sensors.fake_lidar import DIRECTIONS
from pir.worlds.grid_world import GridWorld2D, OCCUPIED


class AStarReplanningAgent:
    """Plan through unknown cells, then replan when observations invalidate the path."""

    def reset(self) -> None:
        self.current_path: list[tuple[int, int]] = []
        self.replan_count = 0
        self.path_invalidations = 0
        self.state = "need_plan"
        self.last_failure: Failure | None = None

    def act(self, obs: dict[str, Any]) -> str:
        robot = obs["robot"]
        goal = obs["goal"]
        known_map = obs["known_map"]

        if self._path_is_invalid(robot, known_map):
            self.path_invalidations += 1
            self.current_path = []

        if not self.current_path:
            self.current_path = astar(known_map, robot, goal)
            self.replan_count += 1
            self.state = "replan"
        else:
            self.state = "follow_path"

        if len(self.current_path) < 2:
            self.state = "no_path"
            return "stay"

        next_cell = self.current_path[1]
        if known_map[next_cell] == OCCUPIED:
            self.path_invalidations += 1
            self.current_path = astar(known_map, robot, goal)
            self.replan_count += 1
            if len(self.current_path) < 2:
                self.state = "no_path"
                return "stay"
            next_cell = self.current_path[1]

        return direction_to(robot, next_cell)

    def update(self, obs: dict[str, Any], reward: float, info: dict[str, Any]) -> None:
        failure = info.get("failure")
        self.last_failure = failure if isinstance(failure, Failure) else None

        robot = obs["robot"]
        if self.current_path and self.current_path[0] != robot:
            while self.current_path and self.current_path[0] != robot:
                self.current_path.pop(0)

        if self.last_failure is not None and self.last_failure.recoverable:
            self.current_path = []
            self.state = "recover_and_replan"

    def _path_is_invalid(self, robot: tuple[int, int], known_map: Any) -> bool:
        if not self.current_path:
            return True
        if robot not in self.current_path:
            return True
        return any(known_map[cell] == OCCUPIED for cell in self.current_path)


def astar(
    known_map: Any,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]]:
    """A* over the currently known map, treating unknown cells as traversable."""

    frontier: list[tuple[int, int, tuple[int, int]]] = []
    heapq.heappush(frontier, (heuristic(start, goal), 0, start))
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    cost_so_far: dict[tuple[int, int], int] = {start: 0}

    while frontier:
        _, cost, current = heapq.heappop(frontier)
        if current == goal:
            return reconstruct_path(came_from, current)

        for neighbor in neighbors(known_map, current):
            new_cost = cost + 1
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + heuristic(neighbor, goal)
                heapq.heappush(frontier, (priority, new_cost, neighbor))
                came_from[neighbor] = current

    return []


def neighbors(known_map: Any, cell: tuple[int, int]) -> list[tuple[int, int]]:
    height, width = known_map.shape
    result: list[tuple[int, int]] = []
    for dr, dc in DIRECTIONS.values():
        neighbor = (cell[0] + dr, cell[1] + dc)
        row, col = neighbor
        if row < 0 or row >= height or col < 0 or col >= width:
            continue
        if known_map[neighbor] == OCCUPIED:
            continue
        result.append(neighbor)
    return result


def reconstruct_path(
    came_from: dict[tuple[int, int], tuple[int, int] | None],
    current: tuple[int, int],
) -> list[tuple[int, int]]:
    path = [current]
    while came_from[current] is not None:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def heuristic(cell: tuple[int, int], goal: tuple[int, int]) -> int:
    return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])


def direction_to(start: tuple[int, int], end: tuple[int, int]) -> str:
    delta = (end[0] - start[0], end[1] - start[1])
    for direction, direction_delta in DIRECTIONS.items():
        if delta == direction_delta:
            return direction
    return "stay"


def run(seed: int = 0, render: bool = True, max_steps: int = 100) -> Trace:
    env = GridWorld2D(seed=seed, lidar_range=3, max_steps=max_steps)
    agent = AStarReplanningAgent()
    obs = env.reset(seed=seed)
    agent.reset()
    trace = Trace()

    for _ in range(max_steps):
        action = agent.act(obs)
        result = env.step(action)
        obs, reward, done, info = result.as_tuple()
        agent.update(obs, reward, info)
        info["agent_state"] = agent.state
        info["replan_count"] = agent.replan_count
        info["path_invalidations"] = agent.path_invalidations
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
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    trace = run(seed=args.seed, render=not args.no_render, max_steps=args.max_steps)
    success = bool(trace.infos and trace.infos[-1].get("success"))
    failures = [failure.kind for failure in trace.failures()]
    replan_count = trace.infos[-1].get("replan_count", 0) if trace.infos else 0
    invalidations = trace.infos[-1].get("path_invalidations", 0) if trace.infos else 0
    print(
        f"success={success} steps={len(trace.actions)} "
        f"replans={replan_count} invalidations={invalidations} failures={failures}"
    )

    if not args.no_render:
        import matplotlib.pyplot as plt

        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
