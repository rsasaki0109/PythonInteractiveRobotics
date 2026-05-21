"""Toy active SLAM: choose moves that reduce pose and map uncertainty.

This is deliberately not a production SLAM system.  It keeps two small
distributions that are easy to inspect:

* ``pose_belief`` is a categorical belief over grid cells.
* ``map_prob`` stores each cell's probability of being occupied.

The active part is a one-step lookahead.  The agent scores each legal action by
the expected entropy drop from scanning the next cell, plus a simple
localization bonus for scans taken near distinctive map structure.
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
from pir.sensors.fake_lidar import DIRECTIONS, cast_cardinal_lidar
from pir.worlds.grid_world import FREE, OCCUPIED


Cell = tuple[int, int]


class ActiveSlamToyWorld:
    """Hidden occupancy grid with perfect cardinal lidar observations."""

    def __init__(
        self,
        *,
        seed: int | None = 0,
        height: int = 11,
        width: int = 15,
        start: Cell = (8, 2),
        goal: Cell = (2, 12),
        lidar_range: int = 4,
        max_steps: int = 100,
    ) -> None:
        self.seed = seed
        self.height = height
        self.width = width
        self.start = start
        self.goal = goal
        self.lidar_range = lidar_range
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self._fig = None
        self._axes = None
        self.reset(seed=seed)

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.true_map = make_demo_map(self.height, self.width, self.start, self.goal)
        self.robot = self.start
        self.time = 0
        self.trajectory = [self.robot]
        return self.observe()

    def observe(self) -> dict[str, Any]:
        scan = cast_cardinal_lidar(
            self.true_map,
            self.robot,
            max_range=self.lidar_range,
        )
        return {
            "time": self.time,
            "robot": self.robot,
            "goal": self.goal,
            "lidar": scan,
            "shape": self.true_map.shape,
            "trajectory": list(self.trajectory),
        }

    def step(self, action: str | dict[str, Any]) -> StepResult:
        self.time += 1
        direction = action.get("direction", "stay") if isinstance(action, dict) else action
        info: dict[str, Any] = {
            "time": self.time,
            "direction": direction,
            "true_robot": self.robot,
            "success": False,
        }

        if direction == "stay":
            return StepResult(self.observe(), -0.01, False, info)

        if direction not in DIRECTIONS:
            info["failure"] = Failure("invalid_action", f"unknown direction: {direction}", True)
            return StepResult(self.observe(), -0.05, False, info)

        next_cell = move_cell(self.robot, direction)
        info["next_cell"] = next_cell
        if not in_bounds(self.true_map, next_cell) or self.true_map[next_cell] == OCCUPIED:
            info["failure"] = Failure("collision", "true robot hit an obstacle", True)
            return StepResult(self.observe(), -0.20, False, info)

        self.robot = next_cell
        self.trajectory.append(self.robot)
        info["true_robot"] = self.robot

        if self.time >= self.max_steps:
            info["failure"] = Failure("timeout", "maximum steps reached", False)
            return StepResult(self.observe(), -0.10, True, info)

        return StepResult(self.observe(), -0.01, False, info)

    def render(
        self,
        agent: "ActiveSlamToyAgent | None" = None,
        info: dict[str, Any] | None = None,
    ) -> None:
        import matplotlib.pyplot as plt

        if self._fig is None or self._axes is None:
            plt.ion()
            self._fig, self._axes = plt.subplots(1, 2, figsize=(10, 4.8))

        draw_scene(self._axes, self, agent, info)
        self._fig.canvas.draw_idle()
        plt.pause(0.05)


class ActiveSlamToyAgent:
    """Maintain toy SLAM beliefs and choose actions by expected information gain."""

    def __init__(
        self,
        *,
        lidar_range: int = 4,
        pose_goal: float = 1.15,
        map_goal: float = 0.36,
    ) -> None:
        self.lidar_range = lidar_range
        self.pose_goal = pose_goal
        self.map_goal = map_goal
        self.reset()

    def reset(self) -> None:
        self.map_prob: np.ndarray | None = None
        self.pose_belief: np.ndarray | None = None
        self.estimated_cell: Cell | None = None
        self.current_target: Cell | None = None
        self.current_path: list[Cell] = []
        self.pose_entropy = 0.0
        self.map_entropy = 0.0
        self.information_gain = 0.0
        self.state = "initialize"
        self.last_failure: Failure | None = None

    def initialize(self, obs: dict[str, Any]) -> None:
        height, width = obs["shape"]
        self.map_prob = np.full((height, width), 0.5, dtype=float)
        self.map_prob[0, :] = 0.95
        self.map_prob[-1, :] = 0.95
        self.map_prob[:, 0] = 0.95
        self.map_prob[:, -1] = 0.95

        self.pose_belief = initial_pose_belief((height, width), obs["robot"])
        self._assimilate_scan(obs)
        self._refresh_metrics()

    def act(self, obs: dict[str, Any]) -> str:
        if self.pose_belief is None or self.map_prob is None:
            self.initialize(obs)

        assert self.pose_belief is not None
        assert self.map_prob is not None
        self.estimated_cell = max_belief_cell(self.pose_belief)

        action, gain = self._choose_active_action(obs["goal"])
        self.information_gain = gain
        return action

    def update(self, obs: dict[str, Any], reward: float, info: dict[str, Any]) -> None:
        assert self.pose_belief is not None
        assert self.map_prob is not None

        direction = info.get("direction", "stay")
        self.pose_belief = predict_pose_belief(self.pose_belief, self.map_prob, direction)
        self._assimilate_scan(obs)
        self.estimated_cell = max_belief_cell(self.pose_belief)
        self._refresh_metrics()

        failure = info.get("failure")
        self.last_failure = failure if isinstance(failure, Failure) else None
        if self.last_failure is not None and self.last_failure.recoverable:
            self.current_path = []
            self.current_target = None
            self.state = "recover"

    def _choose_active_action(self, goal: Cell) -> tuple[str, float]:
        assert self.pose_belief is not None
        assert self.map_prob is not None
        assert self.estimated_cell is not None

        candidates = ["north", "east", "south", "west", "stay"]
        scored: list[tuple[float, int, str]] = []
        current_distance = manhattan(self.estimated_cell, goal)

        for action in candidates:
            cell = move_cell(self.estimated_cell, action)
            if action != "stay" and not likely_free(self.map_prob, cell):
                continue

            map_gain = expected_map_gain(self.map_prob, cell, self.lidar_range)
            pose_gain = expected_pose_gain(self.map_prob, cell, self.lidar_range)
            goal_progress = current_distance - manhattan(cell, goal)
            revisit_penalty = float(self.pose_belief[cell]) if in_bounds(self.map_prob, cell) else 1.0

            gain = map_gain + 0.7 * pose_gain + 0.12 * goal_progress - 0.05 * revisit_penalty
            scored.append((gain, -manhattan(cell, goal), action))

        if not scored:
            self.state = "blocked"
            return "stay", 0.0

        best_gain, _, action = max(scored)
        self.state = "reduce_uncertainty" if best_gain > 0.08 else "move_to_goal"
        return action, float(best_gain)

    def _assimilate_scan(self, obs: dict[str, Any]) -> None:
        assert self.map_prob is not None
        assert self.pose_belief is not None

        robot = obs["robot"]
        scan = obs["lidar"]
        for direction, ray in scan.items():
            for cell in ray["cells"]:
                if in_bounds(self.map_prob, cell):
                    self.map_prob[cell] = 0.04
            hit = ray["hit"]
            if hit is not None and in_bounds(self.map_prob, hit):
                self.map_prob[hit] = 0.96

        self.map_prob[robot] = 0.01
        self.pose_belief *= scan_likelihood(self.map_prob, scan, self.lidar_range)
        total = self.pose_belief.sum()
        if total <= 0.0:
            self.pose_belief = initial_pose_belief(self.map_prob.shape, robot)
        else:
            self.pose_belief /= total

    def _refresh_metrics(self) -> None:
        assert self.map_prob is not None
        assert self.pose_belief is not None
        self.pose_entropy = entropy(self.pose_belief)
        self.map_entropy = mean_binary_entropy(self.map_prob)


def make_demo_map(height: int, width: int, start: Cell, goal: Cell) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    grid[0, :] = OCCUPIED
    grid[-1, :] = OCCUPIED
    grid[:, 0] = OCCUPIED
    grid[:, -1] = OCCUPIED
    grid[4, 3:12] = OCCUPIED
    grid[4, 7] = FREE
    grid[7, 2:10] = OCCUPIED
    grid[7, 5] = FREE
    grid[2:7, 11] = OCCUPIED
    grid[5, 11] = FREE
    for cell in (start, goal):
        grid[cell] = FREE
    return grid


def initial_pose_belief(shape: tuple[int, int], center: Cell) -> np.ndarray:
    rows, cols = np.indices(shape)
    distance = np.abs(rows - center[0]) + np.abs(cols - center[1])
    belief = np.exp(-0.65 * distance)
    belief[0, :] = 0.0
    belief[-1, :] = 0.0
    belief[:, 0] = 0.0
    belief[:, -1] = 0.0
    belief /= belief.sum()
    return belief


def predict_pose_belief(
    belief: np.ndarray,
    map_prob: np.ndarray,
    action: str,
) -> np.ndarray:
    predicted = np.zeros_like(belief)
    choices = motion_choices(action)

    for row in range(belief.shape[0]):
        for col in range(belief.shape[1]):
            mass = belief[row, col]
            if mass <= 0.0:
                continue
            for direction, probability in choices:
                next_cell = move_cell((row, col), direction)
                if likely_free(map_prob, next_cell):
                    predicted[next_cell] += mass * probability
                else:
                    predicted[row, col] += mass * probability

    total = predicted.sum()
    if total > 0.0:
        predicted /= total
    return predicted


def motion_choices(action: str) -> list[tuple[str, float]]:
    if action == "stay" or action not in DIRECTIONS:
        return [("stay", 1.0)]
    if action in ("north", "south"):
        return [(action, 0.76), ("stay", 0.12), ("east", 0.06), ("west", 0.06)]
    return [(action, 0.76), ("stay", 0.12), ("north", 0.06), ("south", 0.06)]


def scan_likelihood(
    map_prob: np.ndarray,
    observed_scan: dict[str, dict[str, Any]],
    lidar_range: int,
) -> np.ndarray:
    likelihood = np.ones_like(map_prob, dtype=float)
    for row in range(map_prob.shape[0]):
        for col in range(map_prob.shape[1]):
            cell = (row, col)
            if not likely_free(map_prob, cell):
                likelihood[cell] = 0.0
                continue
            expected = expected_scan_counts(map_prob, cell, lidar_range)
            error = sum(
                abs(expected[direction] - observed_scan[direction]["free_cells"])
                for direction in DIRECTIONS
            )
            likelihood[cell] = np.exp(-0.75 * error)
    return likelihood


def expected_scan_counts(
    map_prob: np.ndarray,
    cell: Cell,
    lidar_range: int,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for direction, (dr, dc) in DIRECTIONS.items():
        free_cells = 0
        for distance in range(1, lidar_range + 1):
            ray_cell = (cell[0] + dr * distance, cell[1] + dc * distance)
            if not in_bounds(map_prob, ray_cell) or map_prob[ray_cell] > 0.65:
                break
            free_cells += 1
        counts[direction] = free_cells
    return counts


def expected_map_gain(map_prob: np.ndarray, cell: Cell, lidar_range: int) -> float:
    if not in_bounds(map_prob, cell):
        return 0.0
    cells = visible_cells(cell, map_prob.shape, lidar_range)
    if not cells:
        return 0.0
    before = sum(binary_entropy(float(map_prob[seen])) for seen in cells)
    after = 0.18 * len(cells)
    return float(max(0.0, before - after))


def expected_pose_gain(map_prob: np.ndarray, cell: Cell, lidar_range: int) -> float:
    if not in_bounds(map_prob, cell):
        return 0.0
    counts = expected_scan_counts(map_prob, cell, lidar_range)
    asymmetry = abs(counts["north"] - counts["south"]) + abs(counts["east"] - counts["west"])
    nearby_known = sum(1.0 - binary_entropy(float(map_prob[seen])) for seen in visible_cells(cell, map_prob.shape, 2))
    return float(0.25 * asymmetry + 0.08 * nearby_known)


def visible_cells(cell: Cell, shape: tuple[int, int], lidar_range: int) -> list[Cell]:
    cells = [cell] if 0 <= cell[0] < shape[0] and 0 <= cell[1] < shape[1] else []
    for dr, dc in DIRECTIONS.values():
        for distance in range(1, lidar_range + 1):
            ray_cell = (cell[0] + dr * distance, cell[1] + dc * distance)
            if 0 <= ray_cell[0] < shape[0] and 0 <= ray_cell[1] < shape[1]:
                cells.append(ray_cell)
    return cells


def likely_free(map_prob: np.ndarray, cell: Cell) -> bool:
    return in_bounds(map_prob, cell) and map_prob[cell] < 0.70


def in_bounds(grid: np.ndarray, cell: Cell) -> bool:
    return 0 <= cell[0] < grid.shape[0] and 0 <= cell[1] < grid.shape[1]


def move_cell(cell: Cell, direction: str) -> Cell:
    if direction == "stay":
        return cell
    dr, dc = DIRECTIONS[direction]
    return (cell[0] + dr, cell[1] + dc)


def max_belief_cell(belief: np.ndarray) -> Cell:
    index = int(np.argmax(belief))
    row, col = np.unravel_index(index, belief.shape)
    return (int(row), int(col))


def entropy(distribution: np.ndarray) -> float:
    values = distribution[distribution > 0.0]
    return float(-np.sum(values * np.log(values)))


def binary_entropy(probability: float) -> float:
    p = float(np.clip(probability, 1.0e-9, 1.0 - 1.0e-9))
    return float(-(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)))


def mean_binary_entropy(map_prob: np.ndarray) -> float:
    interior = map_prob[1:-1, 1:-1]
    return float(np.mean([binary_entropy(float(value)) for value in interior.ravel()]))


def manhattan(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def draw_scene(
    axes: Any,
    env: ActiveSlamToyWorld,
    agent: ActiveSlamToyAgent | None,
    info: dict[str, Any] | None,
) -> None:
    map_ax, pose_ax = axes
    map_ax.clear()
    pose_ax.clear()

    map_prob = agent.map_prob if agent is not None and agent.map_prob is not None else np.full_like(env.true_map, 0.5, dtype=float)
    pose_belief = agent.pose_belief if agent is not None and agent.pose_belief is not None else np.zeros_like(env.true_map, dtype=float)

    map_ax.imshow(map_prob, cmap="gray_r", origin="upper", vmin=0.0, vmax=1.0)
    draw_grid_overlay(map_ax, env)
    map_ax.set_title("occupancy belief")

    pose_display = np.ma.masked_where(env.true_map == OCCUPIED, pose_belief)
    pose_ax.imshow(pose_display, cmap="viridis", origin="upper", vmin=0.0)
    obstacle_rows, obstacle_cols = np.where(env.true_map == OCCUPIED)
    pose_ax.plot(obstacle_cols, obstacle_rows, "s", color="0.12", markersize=7)
    draw_grid_overlay(pose_ax, env)
    pose_ax.set_title("pose belief")

    for ax in axes:
        rows = [cell[0] for cell in env.trajectory]
        cols = [cell[1] for cell in env.trajectory]
        ax.plot(cols, rows, color="tab:blue", linewidth=2)
        ax.plot(env.robot[1], env.robot[0], "o", color="tab:blue", markersize=8)
        ax.plot(env.goal[1], env.goal[0], "*", color="tab:green", markersize=14)
        if agent is not None and agent.estimated_cell is not None:
            row, col = agent.estimated_cell
            ax.plot(col, row, "x", color="white", markersize=10, markeredgewidth=2)

    if info is not None:
        status = (
            f"step={env.time} state={info.get('agent_state', 'n/a')}\n"
            f"pose H={info.get('pose_entropy', 0.0):.2f} "
            f"map H={info.get('map_entropy', 0.0):.2f} "
            f"gain={info.get('information_gain', 0.0):.2f}"
        )
        map_ax.text(0.02, 0.98, status, transform=map_ax.transAxes, va="top", fontsize=9)


def draw_grid_overlay(ax: Any, env: ActiveSlamToyWorld) -> None:
    ax.set_xticks(np.arange(-0.5, env.width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.height, 1), minor=True)
    ax.grid(which="minor", color="0.55", linewidth=0.5)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)


def run(seed: int = 0, render: bool = True, max_steps: int = 100) -> Trace:
    env = ActiveSlamToyWorld(seed=seed, max_steps=max_steps)
    agent = ActiveSlamToyAgent(lidar_range=env.lidar_range)
    obs = env.reset(seed=seed)
    agent.reset()
    agent.initialize(obs)
    trace = Trace()

    for _ in range(max_steps):
        action = agent.act(obs)
        result = env.step(action)
        obs, reward, env_done, info = result.as_tuple()
        agent.update(obs, reward, info)

        success = agent.pose_entropy <= agent.pose_goal and agent.map_entropy <= agent.map_goal
        info["pose_entropy"] = agent.pose_entropy
        info["map_entropy"] = agent.map_entropy
        info["information_gain"] = agent.information_gain
        info["agent_state"] = agent.state
        info["estimated_cell"] = agent.estimated_cell
        info["success"] = success
        trace.append(obs, action, reward, info)

        if render:
            env.render(agent=agent, info=info)

        if success or env_done:
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
    pose_entropy = trace.infos[-1].get("pose_entropy", 0.0) if trace.infos else 0.0
    map_entropy = trace.infos[-1].get("map_entropy", 0.0) if trace.infos else 0.0
    information_gain = trace.infos[-1].get("information_gain", 0.0) if trace.infos else 0.0
    failures = [failure.kind for failure in trace.failures()]
    print(
        f"success={success} steps={len(trace.actions)} "
        f"pose_entropy={pose_entropy:.2f} map_entropy={map_entropy:.2f} "
        f"information_gain={information_gain:.2f} failures={failures}"
    )

    if not args.no_render:
        import matplotlib.pyplot as plt

        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
