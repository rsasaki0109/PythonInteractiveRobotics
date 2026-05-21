"""A small grid world for closed-loop navigation examples."""

from __future__ import annotations

from typing import Any

import numpy as np

from pir.core.random import make_rng
from pir.core.types import Failure, StepResult
from pir.sensors.fake_lidar import DIRECTIONS, cast_cardinal_lidar


UNKNOWN = -1
FREE = 0
OCCUPIED = 1


class GridWorld2D:
    """Discrete occupancy world with partial observation and collision checks."""

    def __init__(
        self,
        *,
        seed: int | None = 0,
        height: int = 11,
        width: int = 15,
        start: tuple[int, int] = (8, 2),
        goal: tuple[int, int] = (2, 12),
        lidar_range: int = 5,
        max_steps: int = 80,
    ) -> None:
        self.seed = seed
        self.height = height
        self.width = width
        self.start = start
        self.goal = goal
        self.lidar_range = lidar_range
        self.max_steps = max_steps
        self.rng = make_rng(seed)
        self._fig = None
        self._ax = None
        self.reset(seed=seed)

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            self.seed = seed
        if self.seed is not None:
            self.rng = make_rng(self.seed)

        self.true_map = self._make_default_map()
        self.known_map = np.full((self.height, self.width), UNKNOWN, dtype=int)
        self.robot = self.start
        self.time = 0
        self.trajectory = [self.robot]
        self.last_scan: dict[str, dict[str, Any]] | None = None
        return self.observe()

    def observe(self) -> dict[str, Any]:
        scan = cast_cardinal_lidar(
            self.true_map,
            self.robot,
            max_range=self.lidar_range,
        )
        self.last_scan = scan
        self._reveal_from_scan(scan)

        return {
            "time": self.time,
            "robot": self.robot,
            "goal": self.goal,
            "lidar": scan,
            "known_map": self.known_map.copy(),
            "trajectory": list(self.trajectory),
        }

    def step(self, action: str | dict[str, Any]) -> StepResult:
        self.time += 1
        direction = action.get("direction", "stay") if isinstance(action, dict) else action
        info: dict[str, Any] = {
            "time": self.time,
            "action_type": "move",
            "direction": direction,
            "success": False,
        }

        if direction == "stay":
            return self._finish_step(-0.02, False, info)

        if direction not in DIRECTIONS:
            info["failure"] = Failure("invalid_action", f"unknown direction: {direction}", True)
            return self._finish_step(-0.05, False, info)

        dr, dc = DIRECTIONS[direction]
        next_cell = (self.robot[0] + dr, self.robot[1] + dc)
        info["next_cell"] = next_cell

        if self._is_occupied(next_cell):
            info["failure"] = Failure("collision", "attempted to move into an obstacle", True)
            info["blocked_cell"] = next_cell
            return self._finish_step(-0.20, False, info)

        self.robot = next_cell
        self.trajectory.append(self.robot)

        if self.robot == self.goal:
            info["success"] = True
            return self._finish_step(1.0, True, info)

        if self.time >= self.max_steps:
            info["failure"] = Failure("timeout", "maximum steps reached", False)
            return self._finish_step(-0.10, True, info)

        return self._finish_step(-0.01, False, info)

    def _finish_step(self, reward: float, done: bool, info: dict[str, Any]) -> StepResult:
        return StepResult(self.observe(), reward, done, info)

    def _make_default_map(self) -> np.ndarray:
        grid = np.zeros((self.height, self.width), dtype=bool)
        grid[0, :] = True
        grid[-1, :] = True
        grid[:, 0] = True
        grid[:, -1] = True

        grid[2:9, 6] = True
        grid[5, 8:12] = True
        grid[7, 10] = True
        grid[self.start] = False
        grid[self.goal] = False
        return grid

    def _reveal_from_scan(self, scan: dict[str, dict[str, Any]]) -> None:
        self.known_map[self.robot] = FREE
        for ray in scan.values():
            for cell in ray["cells"]:
                if self._in_bounds(cell):
                    self.known_map[cell] = FREE
            hit = ray["hit"]
            if hit is not None and self._in_bounds(hit):
                self.known_map[hit] = OCCUPIED

    def _in_bounds(self, cell: tuple[int, int]) -> bool:
        row, col = cell
        return 0 <= row < self.height and 0 <= col < self.width

    def _is_occupied(self, cell: tuple[int, int]) -> bool:
        if not self._in_bounds(cell):
            return True
        return bool(self.true_map[cell])

    def render(self, agent: Any | None = None, info: dict[str, Any] | None = None) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap

        if self._fig is None or self._ax is None:
            plt.ion()
            self._fig, self._ax = plt.subplots(figsize=(7, 5))

        ax = self._ax
        ax.clear()
        ax.set_title("GridWorld2D: reactive obstacle avoidance")

        display = np.zeros_like(self.known_map, dtype=int)
        display[self.known_map == UNKNOWN] = 0
        display[self.known_map == FREE] = 1
        display[self.known_map == OCCUPIED] = 2
        cmap = ListedColormap(["0.72", "white", "0.1"])
        ax.imshow(display, cmap=cmap, origin="upper", vmin=0, vmax=2)

        if self.last_scan is not None:
            for ray in self.last_scan.values():
                for row, col in ray["cells"]:
                    ax.plot(col, row, ".", color="tab:cyan", markersize=5)

        rows = [cell[0] for cell in self.trajectory]
        cols = [cell[1] for cell in self.trajectory]
        ax.plot(cols, rows, color="tab:blue", linewidth=2, label="trajectory")

        planned_path = getattr(agent, "current_path", None)
        if planned_path:
            path_rows = [cell[0] for cell in planned_path]
            path_cols = [cell[1] for cell in planned_path]
            ax.plot(
                path_cols,
                path_rows,
                "--",
                color="tab:purple",
                linewidth=2,
                label="planned path",
            )

        current_frontier = getattr(agent, "current_frontier", None)
        if current_frontier is not None:
            ax.plot(
                current_frontier[1],
                current_frontier[0],
                "D",
                color="tab:orange",
                markersize=8,
                label="frontier goal",
            )

        ax.plot(self.robot[1], self.robot[0], "o", color="tab:blue", label="robot")
        ax.plot(self.goal[1], self.goal[0], "*", color="tab:green", markersize=14, label="goal")

        status = f"step={self.time}"
        if info is not None and "failure" in info:
            status += f" failure={info['failure'].kind}"
        ax.text(0.02, 0.98, status, transform=ax.transAxes, va="top")

        ax.set_xticks(np.arange(-0.5, self.width, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.height, 1), minor=True)
        ax.grid(which="minor", color="0.85", linewidth=0.7)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax.legend(loc="lower left", fontsize=8)
        self._fig.canvas.draw_idle()
        plt.pause(0.05)


class DynamicObstacleGridWorld(GridWorld2D):
    """Grid world where a moving obstacle can block the robot's next action."""

    def __init__(
        self,
        *,
        seed: int | None = 0,
        height: int = 10,
        width: int = 14,
        start: tuple[int, int] = (7, 1),
        goal: tuple[int, int] = (1, 12),
        lidar_range: int = 5,
        max_steps: int = 90,
    ) -> None:
        self.dynamic_path = [(7, col) for col in range(4, 10)]
        self.dynamic_index = 0
        self.dynamic_direction = 1
        self._previous_dynamic_cells: set[tuple[int, int]] = set()
        super().__init__(
            seed=seed,
            height=height,
            width=width,
            start=start,
            goal=goal,
            lidar_range=lidar_range,
            max_steps=max_steps,
        )

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        self.dynamic_index = 0
        self.dynamic_direction = 1
        self._previous_dynamic_cells = set()
        return super().reset(seed=seed)

    def observe(self) -> dict[str, Any]:
        for cell in self._previous_dynamic_cells:
            if self._in_bounds(cell) and not self.true_map[cell]:
                self.known_map[cell] = FREE

        dynamic_cells = self.dynamic_obstacles()
        scan = cast_cardinal_lidar(
            self._map_with_dynamic_obstacles(dynamic_cells),
            self.robot,
            max_range=self.lidar_range,
        )
        self.last_scan = scan
        self._reveal_from_scan(scan)
        self._previous_dynamic_cells = set(dynamic_cells)

        obs = {
            "time": self.time,
            "robot": self.robot,
            "goal": self.goal,
            "lidar": scan,
            "known_map": self.known_map.copy(),
            "trajectory": list(self.trajectory),
            "dynamic_obstacles": dynamic_cells,
            "predicted_dynamic_obstacles": self.predicted_dynamic_obstacles(),
        }
        return obs

    def step(self, action: str | dict[str, Any]) -> StepResult:
        self.time += 1
        direction = action.get("direction", "stay") if isinstance(action, dict) else action
        current_dynamic = set(self.dynamic_obstacles())
        predicted_dynamic = set(self.predicted_dynamic_obstacles())
        info: dict[str, Any] = {
            "time": self.time,
            "action_type": "move",
            "direction": direction,
            "dynamic_obstacles": sorted(current_dynamic),
            "predicted_dynamic_obstacles": sorted(predicted_dynamic),
            "near_miss": False,
            "success": False,
        }

        if direction == "stay":
            self._advance_dynamic_obstacles()
            info["near_miss"] = self._near_dynamic_obstacle()
            return self._finish_dynamic_step(-0.03, False, info)

        if direction not in DIRECTIONS:
            info["failure"] = Failure("invalid_action", f"unknown direction: {direction}", True)
            self._advance_dynamic_obstacles()
            return self._finish_dynamic_step(-0.05, False, info)

        dr, dc = DIRECTIONS[direction]
        next_cell = (self.robot[0] + dr, self.robot[1] + dc)
        info["next_cell"] = next_cell

        if self._is_static_occupied(next_cell):
            info["failure"] = Failure("collision", "attempted to move into a wall", True)
            info["blocked_cell"] = next_cell
            self._advance_dynamic_obstacles()
            return self._finish_dynamic_step(-0.20, False, info)

        if next_cell in current_dynamic:
            info["failure"] = Failure(
                "blocked_by_moving_obstacle",
                "moving obstacle currently occupies the target cell",
                True,
            )
            info["blocked_cell"] = next_cell
            self._advance_dynamic_obstacles()
            return self._finish_dynamic_step(-0.16, False, info)

        self.robot = next_cell
        self.trajectory.append(self.robot)
        self._advance_dynamic_obstacles()

        if self.robot in set(self.dynamic_obstacles()):
            info["failure"] = Failure(
                "dynamic_collision",
                "moving obstacle entered the robot cell",
                False,
            )
            return self._finish_dynamic_step(-1.0, True, info)

        info["near_miss"] = self._near_dynamic_obstacle()

        if self.robot == self.goal:
            info["success"] = True
            return self._finish_dynamic_step(1.0, True, info)

        if self.time >= self.max_steps:
            info["failure"] = Failure("timeout", "maximum steps reached", False)
            return self._finish_dynamic_step(-0.10, True, info)

        reward = -0.04 if info["near_miss"] else -0.01
        return self._finish_dynamic_step(reward, False, info)

    def dynamic_obstacles(self) -> list[tuple[int, int]]:
        return [self.dynamic_path[self.dynamic_index]]

    def predicted_dynamic_obstacles(self) -> list[tuple[int, int]]:
        index, direction = self._next_dynamic_state()
        return [self.dynamic_path[index]]

    def _advance_dynamic_obstacles(self) -> None:
        self.dynamic_index, self.dynamic_direction = self._next_dynamic_state()

    def _next_dynamic_state(self) -> tuple[int, int]:
        next_index = self.dynamic_index + self.dynamic_direction
        next_direction = self.dynamic_direction
        if next_index < 0 or next_index >= len(self.dynamic_path):
            next_direction *= -1
            next_index = self.dynamic_index + next_direction
        return next_index, next_direction

    def _make_default_map(self) -> np.ndarray:
        grid = np.zeros((self.height, self.width), dtype=bool)
        grid[0, :] = True
        grid[-1, :] = True
        grid[:, 0] = True
        grid[:, -1] = True
        grid[4, 3:11] = True
        grid[4, 7] = False
        grid[self.start] = False
        grid[self.goal] = False
        return grid

    def _map_with_dynamic_obstacles(
        self,
        dynamic_cells: list[tuple[int, int]],
    ) -> np.ndarray:
        occupancy = self.true_map.copy()
        for cell in dynamic_cells:
            if self._in_bounds(cell):
                occupancy[cell] = True
        return occupancy

    def _is_static_occupied(self, cell: tuple[int, int]) -> bool:
        if not self._in_bounds(cell):
            return True
        return bool(self.true_map[cell])

    def _near_dynamic_obstacle(self) -> bool:
        robot_row, robot_col = self.robot
        for row, col in self.dynamic_obstacles():
            if abs(row - robot_row) + abs(col - robot_col) <= 1:
                return True
        return False

    def _finish_dynamic_step(
        self,
        reward: float,
        done: bool,
        info: dict[str, Any],
    ) -> StepResult:
        info["dynamic_obstacles_after"] = self.dynamic_obstacles()
        return StepResult(self.observe(), reward, done, info)

    def render(self, agent: Any | None = None, info: dict[str, Any] | None = None) -> None:
        super().render(agent=agent, info=info)

        if self._ax is None or self._fig is None:
            return

        ax = self._ax
        for row, col in self.dynamic_obstacles():
            ax.plot(col, row, "s", color="tab:red", markersize=12, label="moving obstacle")
        for row, col in self.predicted_dynamic_obstacles():
            ax.plot(
                col,
                row,
                "s",
                color="tab:red",
                markersize=12,
                fillstyle="none",
                label="predicted next",
            )
        state = getattr(agent, "state", None)
        if state is not None:
            ax.text(0.02, 0.92, f"agent={state}", transform=ax.transAxes, va="top")
        ax.legend(loc="lower left", fontsize=8)
        self._fig.canvas.draw_idle()
