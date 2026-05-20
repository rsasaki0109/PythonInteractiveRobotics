"""Detect a blocked path, recover, and replan around it."""

from __future__ import annotations

import argparse
import heapq
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pir.core.types import Failure, Trace
from pir.sensors.fake_lidar import DIRECTIONS
from pir.worlds.blocked_path import BlockedPathWorld, move_cell
from pir.worlds.grid_world import OCCUPIED


Cell = tuple[int, int]


class BlockedPathRecoveryAgent:
    """Follow a path, remember blocked cells, step back, and replan."""

    def reset(self) -> None:
        self.known_blocked: set[Cell] = set()
        self.current_path: list[Cell] = []
        self.state = "plan"
        self.replan_count = 0
        self.recovery_count = 0
        self.detected_block_count = 0
        self.last_direction = "stay"
        self.pending_recovery: str | None = None
        self.last_failure: Failure | None = None

    def act(self, obs: dict[str, Any]) -> str:
        robot = obs["robot"]
        goal = obs["goal"]

        if self.pending_recovery is not None:
            action = self.pending_recovery
            self.pending_recovery = None
            self.recovery_count += 1
            self.state = "recover"
            self.last_direction = action
            return action

        if self._path_is_invalid(robot):
            self.current_path = []

        if not self.current_path:
            self.current_path = astar_with_blocked(
                obs["known_map"],
                robot,
                goal,
                self.known_blocked,
            )
            self.replan_count += 1
            self.state = "replan"
        else:
            self.state = "follow_path"

        if len(self.current_path) < 2:
            self.state = "wait"
            self.last_direction = "stay"
            return "stay"

        action = direction_to(robot, self.current_path[1])
        self.last_direction = action
        return action

    def update(self, obs: dict[str, Any], reward: float, info: dict[str, Any]) -> None:
        robot = obs["robot"]
        failure = info.get("failure")
        self.last_failure = failure if isinstance(failure, Failure) else None

        if isinstance(failure, Failure) and failure.kind == "blocked_path":
            blocked_cell = info.get("blocked_cell")
            if blocked_cell is not None:
                self.known_blocked.add(blocked_cell)
            self.current_path = []
            self.detected_block_count += 1
            self.pending_recovery = opposite_direction(self.last_direction)
            self.state = "detect_blocked"
            return

        if self.current_path and robot in self.current_path:
            while self.current_path and self.current_path[0] != robot:
                self.current_path.pop(0)
        elif self.current_path:
            self.current_path = []

    def _path_is_invalid(self, robot: Cell) -> bool:
        if not self.current_path:
            return True
        if robot not in self.current_path:
            return True
        return any(cell in self.known_blocked for cell in self.current_path)


def astar_with_blocked(
    known_map: np.ndarray,
    start: Cell,
    goal: Cell,
    blocked: set[Cell],
) -> list[Cell]:
    frontier: list[tuple[int, int, Cell]] = []
    heapq.heappush(frontier, (manhattan(start, goal), 0, start))
    came_from: dict[Cell, Cell | None] = {start: None}
    cost_so_far: dict[Cell, int] = {start: 0}

    while frontier:
        _, cost, current = heapq.heappop(frontier)
        if current == goal:
            return reconstruct_path(came_from, current)

        for neighbor in free_neighbors(known_map, current, blocked):
            new_cost = cost + 1
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + manhattan(neighbor, goal)
                heapq.heappush(frontier, (priority, new_cost, neighbor))
                came_from[neighbor] = current

    return []


def free_neighbors(known_map: np.ndarray, cell: Cell, blocked: set[Cell]) -> list[Cell]:
    result: list[Cell] = []
    for direction in ("north", "east", "south", "west"):
        neighbor = move_cell(cell, direction)
        row, col = neighbor
        if row < 0 or row >= known_map.shape[0] or col < 0 or col >= known_map.shape[1]:
            continue
        if known_map[neighbor] == OCCUPIED or neighbor in blocked:
            continue
        result.append(neighbor)
    return result


def reconstruct_path(came_from: dict[Cell, Cell | None], current: Cell) -> list[Cell]:
    path = [current]
    while came_from[current] is not None:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def direction_to(start: Cell, end: Cell) -> str:
    delta = (end[0] - start[0], end[1] - start[1])
    for direction, direction_delta in DIRECTIONS.items():
        if delta == direction_delta:
            return direction
    return "stay"


def opposite_direction(direction: str) -> str:
    return {
        "north": "south",
        "south": "north",
        "east": "west",
        "west": "east",
    }.get(direction, "stay")


def manhattan(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def run(seed: int = 0, render: bool = True, max_steps: int = 80) -> Trace:
    env = BlockedPathWorld(max_steps=max_steps)
    agent = BlockedPathRecoveryAgent()
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
        info["recovery_count"] = agent.recovery_count
        info["detected_block_count"] = agent.detected_block_count
        info["known_blocked"] = sorted(agent.known_blocked)
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
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    trace = run(seed=args.seed, render=not args.no_render, max_steps=args.max_steps)
    success = bool(trace.infos and trace.infos[-1].get("success"))
    failures = [failure.kind for failure in trace.failures()]
    final_info = trace.infos[-1] if trace.infos else {}
    print(
        f"success={success} steps={len(trace.actions)} "
        f"blocked={final_info.get('detected_block_count', 0)} "
        f"recoveries={final_info.get('recovery_count', 0)} "
        f"replans={final_info.get('replan_count', 0)} failures={failures}"
    )

    if not args.no_render:
        import matplotlib.pyplot as plt

        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
