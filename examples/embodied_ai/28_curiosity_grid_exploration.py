"""Curiosity-driven grid exploration with a visit-count map.

Each step the agent keeps a per-cell visit count and walks toward the least
visited reachable free cell.  The loop stops when the coverage of visited
free cells reaches a threshold or when no novel cell remains.

This differs from `05_frontier_exploration.py` in two ways:

* Frontier exploration picks an UNKNOWN-adjacent cell on the observed map.
  Curiosity uses a separate visit-count map and an *intrinsic* novelty
  signal that keeps decaying as the agent revisits cells.
* Frontier exploration succeeds when no UNKNOWN cells remain.  Curiosity
  succeeds when a fraction of free cells has been visited at least once.

Success: visited-cell coverage reaches `coverage_threshold` (default 0.70).
Failure: timeout (terminal); no reachable novel cell with positive
expected information gain (recoverable, ends the loop early).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
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
DIRECTIONS: dict[str, tuple[int, int]] = {
    "north": (-1, 0),
    "south": (1, 0),
    "east": (0, 1),
    "west": (0, -1),
}


@dataclass(frozen=True)
class CuriosityConfig:
    coverage_threshold: float = 0.70
    novelty_decay: float = 0.85
    max_target_age: int = 8


class CuriosityGridWorld:
    """Closed grid with a few interior walls and a robot that visits free cells."""

    def __init__(
        self,
        *,
        height: int = 9,
        width: int = 9,
        start: tuple[int, int] = (4, 4),
        max_steps: int = 120,
    ) -> None:
        self.height = height
        self.width = width
        self.start = start
        self.max_steps = max_steps
        self._fig = None
        self._ax = None
        self.reset()

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        del seed
        self.static_map = self._make_map()
        self.robot = self.start
        self.time = 0
        self.trajectory = [self.robot]
        return self.observe()

    def observe(self) -> dict[str, Any]:
        return {
            "time": self.time,
            "robot": self.robot,
            "map": self.static_map.copy(),
            "trajectory": list(self.trajectory),
        }

    def step(self, action: str) -> StepResult:
        self.time += 1
        info: dict[str, Any] = {"time": self.time, "direction": action, "success": False}

        if action == "stay":
            return StepResult(self.observe(), -0.02, False, info)
        if action not in DIRECTIONS:
            info["failure"] = Failure("invalid_action", f"unknown direction: {action}", True)
            return StepResult(self.observe(), -0.05, False, info)

        dr, dc = DIRECTIONS[action]
        next_cell = (self.robot[0] + dr, self.robot[1] + dc)
        if not self._is_free(next_cell):
            info["failure"] = Failure("collision", "attempted to move into a wall", True)
            return StepResult(self.observe(), -0.20, False, info)

        self.robot = next_cell
        self.trajectory.append(self.robot)
        if self.time >= self.max_steps:
            info["failure"] = Failure("timeout", "maximum steps reached", False)
            return StepResult(self.observe(), -0.10, True, info)
        return StepResult(self.observe(), -0.01, False, info)

    def _is_free(self, cell: tuple[int, int]) -> bool:
        row, col = cell
        if row < 0 or row >= self.height or col < 0 or col >= self.width:
            return False
        return bool(self.static_map[cell] == FREE)

    def _make_map(self) -> np.ndarray:
        grid = np.zeros((self.height, self.width), dtype=int)
        grid[0, :] = OCCUPIED
        grid[-1, :] = OCCUPIED
        grid[:, 0] = OCCUPIED
        grid[:, -1] = OCCUPIED
        grid[2, 2:5] = OCCUPIED
        grid[6, 4:7] = OCCUPIED
        grid[3:6, 6] = OCCUPIED
        grid[self.start] = FREE
        return grid

    def render(self, agent: "CuriosityExplorationAgent", info: dict[str, Any]) -> None:
        import matplotlib.pyplot as plt

        if self._fig is None or self._ax is None:
            plt.ion()
            self._fig, self._ax = plt.subplots(figsize=(6, 5))
        ax = self._ax
        ax.clear()
        ax.imshow(self.static_map, cmap="gray_r", origin="upper", vmin=FREE, vmax=OCCUPIED)
        visits = agent.visit_counts.astype(float)
        normalized = visits / (visits.max() + 1e-6)
        ax.imshow(normalized, cmap="viridis", origin="upper", alpha=0.45, vmin=0.0, vmax=1.0)
        rows = [cell[0] for cell in self.trajectory]
        cols = [cell[1] for cell in self.trajectory]
        ax.plot(cols, rows, color="tab:blue", linewidth=1.5, alpha=0.7)
        if agent.current_target is not None:
            ax.plot(
                agent.current_target[1],
                agent.current_target[0],
                "x",
                color="tab:red",
                markersize=12,
                markeredgewidth=3,
                label="curiosity target",
            )
        ax.plot(self.robot[1], self.robot[0], "o", color="tab:blue", markersize=10, label="robot")
        status = (
            f"step={self.time} coverage={agent.coverage:.2f}"
            f" novelty={agent.last_novelty:.2f} target_switches={agent.target_switches}"
        )
        if "failure" in info:
            status += f"\nfailure={info['failure'].kind}"
        ax.set_title("curiosity grid exploration")
        ax.text(0.02, 0.98, status, transform=ax.transAxes, va="top", fontsize=9)
        ax.set_xticks(np.arange(-0.5, self.width, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.height, 1), minor=True)
        ax.grid(which="minor", color="0.75", linewidth=0.5)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax.legend(loc="lower right", fontsize=8)
        self._fig.canvas.draw_idle()
        plt.pause(0.05)


class CuriosityExplorationAgent:
    """Pick the most novel reachable cell using a visit-count novelty score."""

    def __init__(self, world: CuriosityGridWorld, config: CuriosityConfig | None = None) -> None:
        self.world = world
        self.config = config or CuriosityConfig()
        self.reset()

    def reset(self) -> None:
        self.visit_counts = np.zeros((self.world.height, self.world.width), dtype=float)
        self.current_target: tuple[int, int] | None = None
        self.current_path: list[tuple[int, int]] = []
        self.target_age = 0
        self.target_switches = 0
        self.coverage = 0.0
        self.last_novelty = 0.0
        self.state = "explore"

    def act(self, obs: dict[str, Any]) -> str:
        robot = obs["robot"]
        static_map = obs["map"]
        self.visit_counts[robot] += 1.0
        self.coverage = self._coverage(static_map)

        if self.coverage >= self.config.coverage_threshold:
            self.state = "done"
            self.current_target = None
            self.current_path = []
            return "stay"

        if self._need_new_target(robot, static_map):
            target = self._pick_novel_target(static_map, robot)
            if target is None:
                self.state = "no_novelty"
                self.current_target = None
                self.current_path = []
                return "stay"
            self.current_target = target
            self.current_path = grid_astar(static_map == FREE, robot, target)
            self.target_age = 0
            self.target_switches += 1
            self.state = "explore"

        if len(self.current_path) < 2:
            self.current_target = None
            self.current_path = []
            return "stay"

        next_cell = self.current_path[1]
        self.current_path = self.current_path[1:]
        self.target_age += 1
        return direction_to(robot, next_cell)

    def update(self, obs: dict[str, Any], reward: float, info: dict[str, Any]) -> None:
        del reward, info
        self.visit_counts *= self.config.novelty_decay
        self.visit_counts[obs["robot"]] = max(self.visit_counts[obs["robot"]], 1.0)

    def info(self) -> dict[str, Any]:
        return {
            "coverage": float(self.coverage),
            "target_switches": int(self.target_switches),
            "last_novelty": float(self.last_novelty),
            "agent_state": self.state,
            "target": self.current_target,
        }

    def _coverage(self, static_map: np.ndarray) -> float:
        free_mask = static_map == FREE
        total_free = int(free_mask.sum())
        if total_free == 0:
            return 1.0
        visited = (self.visit_counts > 0.0) & free_mask
        return float(visited.sum()) / float(total_free)

    def _need_new_target(self, robot: tuple[int, int], static_map: np.ndarray) -> bool:
        if self.current_target is None:
            return True
        if self.current_target == robot:
            return True
        if self.target_age >= self.config.max_target_age:
            return True
        if static_map[self.current_target] == OCCUPIED:
            return True
        return False

    def _pick_novel_target(
        self,
        static_map: np.ndarray,
        robot: tuple[int, int],
    ) -> tuple[int, int] | None:
        best: tuple[int, int] | None = None
        best_score = -float("inf")
        for r in range(static_map.shape[0]):
            for c in range(static_map.shape[1]):
                if static_map[r, c] != FREE:
                    continue
                if (r, c) == robot:
                    continue
                novelty = 1.0 / (1.0 + self.visit_counts[r, c])
                distance = abs(r - robot[0]) + abs(c - robot[1])
                # Strongly favour novelty; among equally novel cells, prefer
                # farther targets so the agent commits to a real path.
                score = novelty * 10.0 + 0.1 * distance
                if score > best_score:
                    best_score = score
                    best = (r, c)
        if best is not None:
            self.last_novelty = 1.0 / (1.0 + self.visit_counts[best])
        return best




def direction_to(start: tuple[int, int], goal: tuple[int, int]) -> str:
    dr = goal[0] - start[0]
    dc = goal[1] - start[1]
    if dr == -1 and dc == 0:
        return "north"
    if dr == 1 and dc == 0:
        return "south"
    if dr == 0 and dc == 1:
        return "east"
    if dr == 0 and dc == -1:
        return "west"
    return "stay"


def run(
    seed: int = 0,
    render: bool = True,
    max_steps: int = 120,
    coverage_threshold: float = 0.70,
) -> Trace:
    world = CuriosityGridWorld(max_steps=max_steps)
    agent = CuriosityExplorationAgent(
        world,
        config=CuriosityConfig(coverage_threshold=coverage_threshold),
    )
    obs = world.reset(seed=seed)
    agent.reset()
    trace = Trace()

    for _ in range(max_steps):
        action = agent.act(obs)
        result = world.step(action)
        obs, reward, done, info = result.as_tuple()
        info.update(agent.info())
        if agent.coverage >= agent.config.coverage_threshold:
            info["success"] = True
            done = True
        trace.append(obs, action, reward, info)
        agent.update(obs, reward, info)

        if render:
            world.render(agent=agent, info=info)
        if done:
            break

    return trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--coverage-threshold", type=float, default=0.70)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    trace = run(
        seed=args.seed,
        render=not args.no_render,
        max_steps=args.max_steps,
        coverage_threshold=args.coverage_threshold,
    )
    final_info = trace.infos[-1] if trace.infos else {}
    print(
        f"success={final_info.get('success', False)} "
        f"steps={len(trace.actions)} "
        f"coverage={final_info.get('coverage', 0.0):.2f} "
        f"target_switches={final_info.get('target_switches', 0)} "
        f"failures={[failure.kind for failure in trace.failures()]}"
    )

    if not args.no_render:
        import matplotlib.pyplot as plt

        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
