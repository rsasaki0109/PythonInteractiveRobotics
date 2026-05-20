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

from pir.core.types import Failure, StepResult, Trace
from pir.sensors.fake_lidar import DIRECTIONS
from pir.worlds.grid_world import FREE, OCCUPIED


Cell = tuple[int, int]


class BlockedPathWorld:
    """Known grid where a temporary blocker appears on the planned path."""

    def __init__(
        self,
        *,
        height: int = 10,
        width: int = 14,
        start: Cell = (6, 1),
        goal: Cell = (6, 12),
        blocker_cell: Cell = (6, 6),
        blocker_spawn_time: int = 4,
        max_steps: int = 80,
    ) -> None:
        self.height = height
        self.width = width
        self.start = start
        self.goal = goal
        self.blocker_cell = blocker_cell
        self.blocker_spawn_time = blocker_spawn_time
        self.max_steps = max_steps
        self._fig = None
        self._ax = None
        self.reset()

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        self.static_map = self._make_map()
        self.robot = self.start
        self.time = 0
        self.trajectory = [self.robot]
        self.last_blocked_cell: Cell | None = None
        return self.observe()

    def observe(self) -> dict[str, Any]:
        return {
            "time": self.time,
            "robot": self.robot,
            "goal": self.goal,
            "known_map": self.static_map.copy(),
            "dynamic_blocker": self.blocker_cell if self.blocker_is_active() else None,
            "last_blocked_cell": self.last_blocked_cell,
            "trajectory": list(self.trajectory),
        }

    def step(self, action: str | dict[str, Any]) -> StepResult:
        self.time += 1
        direction = action.get("direction", "stay") if isinstance(action, dict) else action
        info: dict[str, Any] = {
            "time": self.time,
            "direction": direction,
            "success": False,
            "dynamic_blocker": self.blocker_cell if self.blocker_is_active() else None,
        }

        if direction == "stay":
            return StepResult(self.observe(), -0.02, False, info)

        if direction not in DIRECTIONS:
            info["failure"] = Failure("invalid_action", f"unknown direction: {direction}", True)
            return StepResult(self.observe(), -0.05, False, info)

        next_cell = move_cell(self.robot, direction)
        info["next_cell"] = next_cell

        if self._is_static_occupied(next_cell):
            info["failure"] = Failure("collision", "attempted to move into a wall", True)
            info["blocked_cell"] = next_cell
            return StepResult(self.observe(), -0.20, False, info)

        if self.blocker_is_active() and next_cell == self.blocker_cell:
            failure = Failure(
                "blocked_path",
                "planned path is blocked by a newly observed obstacle",
                True,
            )
            self.last_blocked_cell = next_cell
            info["failure"] = failure
            info["blocked_cell"] = next_cell
            return StepResult(self.observe(), -0.18, False, info)

        self.robot = next_cell
        self.trajectory.append(self.robot)

        if self.robot == self.goal:
            info["success"] = True
            return StepResult(self.observe(), 1.0, True, info)

        if self.time >= self.max_steps:
            info["failure"] = Failure("timeout", "maximum steps reached", False)
            return StepResult(self.observe(), -0.10, True, info)

        return StepResult(self.observe(), -0.01, False, info)

    def blocker_is_active(self) -> bool:
        return self.time >= self.blocker_spawn_time

    def _make_map(self) -> np.ndarray:
        grid = np.zeros((self.height, self.width), dtype=int)
        grid[0, :] = OCCUPIED
        grid[-1, :] = OCCUPIED
        grid[:, 0] = OCCUPIED
        grid[:, -1] = OCCUPIED
        grid[4, 2:11] = OCCUPIED
        grid[4, 5] = FREE
        grid[7, 3:10] = OCCUPIED
        grid[7, 8] = FREE
        grid[self.start] = FREE
        grid[self.goal] = FREE
        grid[self.blocker_cell] = FREE
        return grid

    def _is_static_occupied(self, cell: Cell) -> bool:
        row, col = cell
        if row < 0 or row >= self.height or col < 0 or col >= self.width:
            return True
        return bool(self.static_map[cell] == OCCUPIED)

    def render(self, agent: Any | None = None, info: dict[str, Any] | None = None) -> None:
        import matplotlib.pyplot as plt

        if self._fig is None or self._ax is None:
            plt.ion()
            self._fig, self._ax = plt.subplots(figsize=(7, 5))

        draw_blocked_path_scene(self._ax, self, agent, info)
        self._fig.canvas.draw_idle()
        plt.pause(0.05)


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


def move_cell(cell: Cell, direction: str) -> Cell:
    if direction == "stay":
        return cell
    dr, dc = DIRECTIONS[direction]
    return (cell[0] + dr, cell[1] + dc)


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


def draw_blocked_path_scene(
    ax: Any,
    env: BlockedPathWorld,
    agent: BlockedPathRecoveryAgent | None,
    info: dict[str, Any] | None,
) -> None:
    ax.clear()
    display = env.static_map.copy()
    ax.imshow(display, cmap="gray_r", origin="upper", vmin=FREE, vmax=OCCUPIED)

    rows = [cell[0] for cell in env.trajectory]
    cols = [cell[1] for cell in env.trajectory]
    ax.plot(cols, rows, color="tab:blue", linewidth=2, label="trajectory")

    if agent is not None and agent.current_path:
        path_rows = [cell[0] for cell in agent.current_path]
        path_cols = [cell[1] for cell in agent.current_path]
        ax.plot(path_cols, path_rows, "--", color="tab:purple", linewidth=2, label="planned path")

    if agent is not None:
        for row, col in agent.known_blocked:
            ax.plot(col, row, "x", color="tab:red", markersize=12, markeredgewidth=3)

    if env.blocker_is_active():
        ax.plot(
            env.blocker_cell[1],
            env.blocker_cell[0],
            "s",
            color="tab:red",
            markersize=12,
            label="new blocker",
        )

    ax.plot(env.robot[1], env.robot[0], "o", color="tab:blue", markersize=9, label="robot")
    ax.plot(env.goal[1], env.goal[0], "*", color="tab:green", markersize=14, label="goal")

    state = getattr(agent, "state", "none")
    replan_count = getattr(agent, "replan_count", 0)
    recovery_count = getattr(agent, "recovery_count", 0)
    status = f"step={env.time} state={state} replans={replan_count} recoveries={recovery_count}"
    if info is not None and "failure" in info:
        status += f"\nfailure={info['failure'].kind}"
    ax.text(0.02, 0.98, status, transform=ax.transAxes, va="top", fontsize=9)
    ax.set_title("blocked path recovery")
    ax.set_xticks(np.arange(-0.5, env.width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.height, 1), minor=True)
    ax.grid(which="minor", color="0.75", linewidth=0.6)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.legend(loc="lower left", fontsize=8)


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
