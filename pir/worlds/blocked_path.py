"""A small known-map grid world where a blocker appears mid-run.

The world owns the map, the robot pose, and the dynamic blocker. The example
that imports this world keeps the agent, the A* policy, and the teaching
loop, so the package stays free of agent-specific knowledge.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pir.core.types import Failure, StepResult
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


def move_cell(cell: Cell, direction: str) -> Cell:
    """Apply a cardinal direction step or stay-in-place."""

    if direction == "stay":
        return cell
    dr, dc = DIRECTIONS[direction]
    return (cell[0] + dr, cell[1] + dc)


def draw_blocked_path_scene(
    ax: Any,
    env: BlockedPathWorld,
    agent: Any | None,
    info: dict[str, Any] | None,
) -> None:
    ax.clear()
    display = env.static_map.copy()
    ax.imshow(display, cmap="gray_r", origin="upper", vmin=FREE, vmax=OCCUPIED)

    rows = [cell[0] for cell in env.trajectory]
    cols = [cell[1] for cell in env.trajectory]
    ax.plot(cols, rows, color="tab:blue", linewidth=2, label="trajectory")

    planned_path = getattr(agent, "current_path", None)
    if planned_path:
        path_rows = [cell[0] for cell in planned_path]
        path_cols = [cell[1] for cell in planned_path]
        ax.plot(path_cols, path_rows, "--", color="tab:purple", linewidth=2, label="planned path")

    known_blocked = getattr(agent, "known_blocked", None)
    if known_blocked:
        for row, col in known_blocked:
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
