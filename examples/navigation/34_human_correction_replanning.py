"""Human correction that changes the planner's cost belief.

The robot first plans the shortest route through a center corridor. When it is
about to enter a zone a human dislikes, the world returns
`Failure(kind="human_correction", recoverable=True)` instead of moving the
robot. The agent treats the correction as new preference information, raises
the traversal cost of the corrected cells, replans, and reaches the same goal
through a longer route.

Success: robot reaches the goal cell after incorporating the correction.
Failure: human_correction (recoverable - a human rejects the next planned
cell and supplies cells to avoid), collision (recoverable), invalid_direction
(recoverable), timeout (terminal).
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

from pir.core.types import Failure, StepResult, Trace
from pir.planning import astar as grid_astar


FREE = 0
OCCUPIED = 1

Cell = tuple[int, int]

DIRECTIONS: dict[str, Cell] = {
    "north": (-1, 0),
    "south": (1, 0),
    "west": (0, -1),
    "east": (0, 1),
}

DEFAULT_MAP: tuple[tuple[int, ...], ...] = (
    (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
    (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
)

DEFAULT_CORRECTION_ZONE: tuple[Cell, ...] = tuple(
    (row, col) for row in (5, 6, 7) for col in (6, 7, 8)
)


class HumanCorrectionWorld:
    """Grid world where a human rejects traversal through one preference zone."""

    def __init__(
        self,
        *,
        static_map: tuple[tuple[int, ...], ...] = DEFAULT_MAP,
        correction_zone: tuple[Cell, ...] = DEFAULT_CORRECTION_ZONE,
        start: Cell = (6, 1),
        goal: Cell = (6, 13),
        max_steps: int = 60,
    ) -> None:
        self.static_map = np.asarray(static_map, dtype=int)
        self.walkable = self.static_map == FREE
        self.height, self.width = self.static_map.shape
        self.correction_zone: set[Cell] = set(correction_zone)
        self.start = start
        self.goal = goal
        self.max_steps = max_steps
        self.robot = start
        self.step_count = 0
        self.human_correction_count = 0
        self.last_correction_cell: Cell | None = None
        self.trajectory: list[Cell] = [start]
        self._fig: Any | None = None
        self._ax: Any | None = None

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        _ = seed
        self.robot = self.start
        self.step_count = 0
        self.human_correction_count = 0
        self.last_correction_cell = None
        self.trajectory = [self.start]
        return self.observe()

    def observe(self) -> dict[str, Any]:
        return {
            "step": self.step_count,
            "robot": self.robot,
            "goal": self.goal,
            "static_map": self.static_map.copy(),
            "walkable": self.walkable.copy(),
            "correction_zone": tuple(sorted(self.correction_zone)),
            "human_correction_count": self.human_correction_count,
            "last_correction_cell": self.last_correction_cell,
        }

    def step(self, action: str) -> StepResult:
        self.step_count += 1
        info: dict[str, Any] = {
            "success": False,
            "action": action,
            "human_correction_count": self.human_correction_count,
        }

        if action == "stay":
            return self._finish_step(-0.04, info)
        if action not in DIRECTIONS:
            info["failure"] = Failure(
                "invalid_direction",
                f"unknown direction: {action}",
                recoverable=True,
            )
            return self._finish_step(-0.10, info)

        dr, dc = DIRECTIONS[action]
        next_cell = (self.robot[0] + dr, self.robot[1] + dc)
        if not self._in_bounds(next_cell) or not self.walkable[next_cell]:
            info["failure"] = Failure(
                "collision",
                f"cell is not traversable: {next_cell}",
                recoverable=True,
            )
            return self._finish_step(-0.20, info)

        if next_cell in self.correction_zone:
            self.human_correction_count += 1
            self.last_correction_cell = next_cell
            info["human_correction_count"] = self.human_correction_count
            info["correction_cell"] = next_cell
            info["corrected_cells"] = tuple(sorted(self.correction_zone))
            info["failure"] = Failure(
                "human_correction",
                "Human rejected the next cell and marked a preference zone to avoid.",
                recoverable=True,
            )
            return self._finish_step(-0.35, info)

        self.robot = next_cell
        self.trajectory.append(self.robot)
        success = self.robot == self.goal
        info["success"] = success
        return self._finish_step(1.0 if success else -0.02, info)

    def _finish_step(self, reward: float, info: dict[str, Any]) -> StepResult:
        done = bool(info.get("success"))
        if not done and self.step_count >= self.max_steps:
            done = True
            info["failure"] = Failure(
                "timeout",
                "Human-corrected planner did not reach the goal before max_steps.",
                recoverable=False,
            )
        info["robot"] = self.robot
        info["entered_correction_zone_count"] = sum(
            1 for cell in self.trajectory if cell in self.correction_zone
        )
        return StepResult(self.observe(), reward, done, info)

    def _in_bounds(self, cell: Cell) -> bool:
        return 0 <= cell[0] < self.height and 0 <= cell[1] < self.width

    def render(self, agent: "HumanCorrectionAgent", info: dict[str, Any]) -> None:
        import matplotlib.pyplot as plt

        if self._fig is None or self._ax is None:
            plt.ion()
            self._fig, self._ax = plt.subplots(figsize=(6.4, 5.2))
        ax = self._ax
        ax.clear()
        draw_human_correction_scene(ax, self, agent, info)
        self._fig.canvas.draw_idle()
        plt.pause(0.001)


class HumanCorrectionAgent:
    """Plan first, then raise costs for cells named by human correction."""

    def __init__(self, correction_penalty: float = 12.0) -> None:
        self.correction_penalty = correction_penalty
        self.reset()

    def reset(self) -> None:
        self.corrected_cells: set[Cell] = set()
        self.current_path: list[Cell] = []
        self.baseline_path: list[Cell] = []
        self.state = "plan_shortest"
        self.replan_count = 0
        self.correction_count = 0
        self.last_failure: Failure | None = None

    def act(self, obs: dict[str, Any]) -> str:
        robot = tuple(obs["robot"])
        goal = tuple(obs["goal"])
        walkable = obs["walkable"]

        if self._path_invalid(robot):
            self.current_path = self._plan(walkable, robot, goal)
            self.replan_count += 1
            self.state = "replan_with_correction" if self.corrected_cells else "plan_shortest"
            if not self.baseline_path:
                self.baseline_path = list(self.current_path)
        else:
            self.state = "follow_path"

        if len(self.current_path) < 2:
            return "stay"
        return direction_to(robot, self.current_path[1])

    def update(self, obs: dict[str, Any], reward: float, info: dict[str, Any]) -> None:
        _ = reward
        robot = tuple(obs["robot"])
        failure = info.get("failure")
        self.last_failure = failure if isinstance(failure, Failure) else None

        if isinstance(failure, Failure) and failure.kind == "human_correction":
            self.corrected_cells.update(tuple(cell) for cell in info.get("corrected_cells", ()))
            self.current_path = []
            self.correction_count += 1
            self.state = "learn_from_correction"
            return

        if self.current_path and robot in self.current_path:
            while self.current_path and self.current_path[0] != robot:
                self.current_path.pop(0)
        elif self.current_path:
            self.current_path = []

    def _plan(self, walkable: np.ndarray, start: Cell, goal: Cell) -> list[Cell]:
        edge_cost = np.ones_like(walkable, dtype=float)
        for cell in self.corrected_cells:
            edge_cost[cell] = self.correction_penalty
        return grid_astar(walkable, start, goal, edge_cost=edge_cost)

    def _path_invalid(self, robot: Cell) -> bool:
        if not self.current_path:
            return True
        if robot not in self.current_path:
            return True
        return any(cell in self.corrected_cells for cell in self.current_path)


def direction_to(start: Cell, end: Cell) -> str:
    delta = (end[0] - start[0], end[1] - start[1])
    for direction, direction_delta in DIRECTIONS.items():
        if delta == direction_delta:
            return direction
    return "stay"


def draw_human_correction_scene(
    ax: Any,
    env: HumanCorrectionWorld,
    agent: HumanCorrectionAgent,
    info: dict[str, Any] | None = None,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Rectangle

    info = {} if info is None else info
    cmap = ListedColormap(["white", "0.12"])
    ax.imshow(env.static_map, cmap=cmap, origin="upper", vmin=0, vmax=1)

    for row, col in env.correction_zone:
        ax.add_patch(
            Rectangle(
                (col - 0.5, row - 0.5),
                1.0,
                1.0,
                facecolor="#f2b84b",
                edgecolor="none",
                alpha=0.32,
            )
        )
    for row, col in agent.corrected_cells:
        ax.add_patch(
            Rectangle(
                (col - 0.5, row - 0.5),
                1.0,
                1.0,
                facecolor="#d94b3d",
                edgecolor="none",
                alpha=0.34,
            )
        )

    if agent.baseline_path:
        rows = [r for r, _ in agent.baseline_path]
        cols = [c for _, c in agent.baseline_path]
        ax.plot(cols, rows, "--", color="0.35", linewidth=1.7, label="original shortest path")

    if agent.current_path:
        rows = [r for r, _ in agent.current_path]
        cols = [c for _, c in agent.current_path]
        ax.plot(cols, rows, color="tab:purple", linewidth=2.3, label="current plan")

    if len(env.trajectory) > 1:
        rows = [r for r, _ in env.trajectory]
        cols = [c for _, c in env.trajectory]
        ax.plot(cols, rows, color="tab:blue", linewidth=2.5, alpha=0.90, label="executed")

    correction_cell = info.get("correction_cell") or env.last_correction_cell
    if correction_cell is not None:
        ax.plot(
            correction_cell[1],
            correction_cell[0],
            marker="x",
            color="tab:red",
            markersize=12,
            markeredgewidth=3,
            label="human correction",
        )

    ax.plot(env.start[1], env.start[0], "o", color="tab:cyan", markersize=9)
    ax.plot(env.goal[1], env.goal[0], "*", color="tab:green", markersize=16)
    ax.plot(env.robot[1], env.robot[0], "o", color="tab:blue", markersize=10)

    status = (
        f"step={env.step_count} state={agent.state} "
        f"corrections={agent.correction_count} replans={agent.replan_count}"
    )
    failure = info.get("failure")
    if isinstance(failure, Failure):
        status += f" failure={failure.kind}"
    if info.get("success"):
        status += " success"
    ax.text(
        0.02,
        0.98,
        status,
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.7", alpha=0.86),
    )

    ax.set_title("human correction -> cost update -> replan")
    ax.set_xticks(np.arange(-0.5, env.width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.height, 1), minor=True)
    ax.grid(which="minor", color="0.86", linewidth=0.6)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="lower right", fontsize=7, framealpha=0.9)
    plt.tight_layout()


def run(seed: int = 0, render: bool = True, max_steps: int = 60) -> Trace:
    env = HumanCorrectionWorld(max_steps=max_steps)
    agent = HumanCorrectionAgent()
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
        info["correction_count"] = agent.correction_count
        info["corrected_cells"] = tuple(sorted(agent.corrected_cells))
        info["planned_path"] = tuple(agent.current_path)
        info["planned_path_length"] = len(agent.current_path)
        info["baseline_path_length"] = len(agent.baseline_path)
        info["baseline_crosses_correction_zone"] = any(
            cell in env.correction_zone for cell in agent.baseline_path
        )
        info["entered_correction_zone_count"] = sum(
            1 for cell in env.trajectory if cell in env.correction_zone
        )
        trace.append(obs, action, reward, info)

        if render:
            env.render(agent=agent, info=info)

        if done:
            break

    return trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    trace = run(seed=args.seed, render=not args.no_render, max_steps=args.max_steps)
    final = trace.infos[-1] if trace.infos else {}
    failures = [failure.kind for failure in trace.failures()]
    print(
        f"success={bool(final.get('success'))} steps={len(trace.actions)} "
        f"corrections={final.get('correction_count', 0)} "
        f"replans={final.get('replan_count', 0)} failures={failures}"
    )

    if not args.no_render:
        import matplotlib.pyplot as plt

        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
