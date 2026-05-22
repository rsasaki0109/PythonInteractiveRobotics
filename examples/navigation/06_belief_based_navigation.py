"""Navigate with pose belief instead of direct access to true pose."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pir.core.random import make_rng
from pir.core.types import Failure, StepResult, Trace
from pir.planning import astar as grid_astar
from pir.sensors.fake_lidar import DIRECTIONS
from pir.worlds.grid_world import FREE, OCCUPIED


class BeliefGridWorld:
    """Known map, hidden true pose, noisy landmark observations."""

    def __init__(
        self,
        *,
        seed: int | None = 0,
        height: int = 11,
        width: int = 15,
        start: tuple[int, int] = (8, 2),
        goal: tuple[int, int] = (2, 12),
        sensor_range: int = 5,
        range_sigma: float = 0.45,
        max_steps: int = 90,
    ) -> None:
        self.seed = seed
        self.height = height
        self.width = width
        self.start = start
        self.goal = goal
        self.sensor_range = sensor_range
        self.range_sigma = range_sigma
        self.max_steps = max_steps
        self.landmarks = {
            "start": (8, 2),
            "middle": (5, 7),
            "goal": (2, 12),
        }
        self.rng = make_rng(seed)
        self._fig = None
        self._ax = None
        self.reset(seed=seed)

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            self.seed = seed
        if self.seed is not None:
            self.rng = make_rng(self.seed)

        self.map = self._make_map()
        self.true_robot = self.start
        self.time = 0
        self.trajectory = [self.true_robot]
        return self.observe()

    def observe(self) -> dict[str, Any]:
        measurements: dict[str, float] = {}
        for name, landmark in self.landmarks.items():
            distance = manhattan(self.true_robot, landmark)
            if distance <= self.sensor_range:
                noise = float(self.rng.normal(0.0, self.range_sigma))
                measurements[name] = max(0.0, distance + noise)

        return {
            "time": self.time,
            "map": self.map.copy(),
            "goal": self.goal,
            "landmarks": dict(self.landmarks),
            "landmark_ranges": measurements,
        }

    def step(self, action: str | dict[str, Any]) -> StepResult:
        self.time += 1
        direction = action.get("direction", "stay") if isinstance(action, dict) else action
        info: dict[str, Any] = {
            "time": self.time,
            "direction": direction,
            "success": False,
            "true_robot": self.true_robot,
        }

        if direction == "stay":
            return StepResult(self.observe(), -0.02, False, info)

        if direction not in DIRECTIONS:
            info["failure"] = Failure("invalid_action", f"unknown direction: {direction}", True)
            return StepResult(self.observe(), -0.05, False, info)

        dr, dc = DIRECTIONS[direction]
        next_cell = (self.true_robot[0] + dr, self.true_robot[1] + dc)
        info["next_cell"] = next_cell

        if self._is_occupied(next_cell):
            info["failure"] = Failure("collision", "true robot hit a known obstacle", True)
            return StepResult(self.observe(), -0.12, False, info)

        self.true_robot = next_cell
        self.trajectory.append(self.true_robot)
        info["true_robot"] = self.true_robot

        if self.true_robot == self.goal:
            info["success"] = True
            return StepResult(self.observe(), 1.0, True, info)

        if self.time >= self.max_steps:
            info["failure"] = Failure("timeout", "maximum steps reached", False)
            return StepResult(self.observe(), -0.10, True, info)

        return StepResult(self.observe(), -0.01, False, info)

    def _make_map(self) -> np.ndarray:
        grid = np.zeros((self.height, self.width), dtype=int)
        grid[0, :] = OCCUPIED
        grid[-1, :] = OCCUPIED
        grid[:, 0] = OCCUPIED
        grid[:, -1] = OCCUPIED
        grid[4, 3:11] = OCCUPIED
        grid[4, 7] = FREE
        grid[7, 6:13] = OCCUPIED
        grid[7, 9] = FREE
        for cell in [self.start, self.goal, *self.landmarks.values()]:
            grid[cell] = FREE
        return grid

    def _is_occupied(self, cell: tuple[int, int]) -> bool:
        row, col = cell
        if row < 0 or row >= self.height or col < 0 or col >= self.width:
            return True
        return bool(self.map[cell] == OCCUPIED)

    def render(self, agent: Any | None = None, info: dict[str, Any] | None = None) -> None:
        import matplotlib.pyplot as plt

        if self._fig is None or self._ax is None:
            plt.ion()
            self._fig, self._ax = plt.subplots(figsize=(7, 5))

        ax = self._ax
        ax.clear()
        draw_belief_scene(ax, self, agent, info)
        self._fig.canvas.draw_idle()
        plt.pause(0.05)


class BeliefNavigationAgent:
    """Bayes filter plus a policy that localizes before going to the goal."""

    def __init__(self, *, entropy_threshold: float = 2.35) -> None:
        self.entropy_threshold = entropy_threshold
        self.reset()

    def reset(self) -> None:
        self.belief: np.ndarray | None = None
        self.estimated_cell: tuple[int, int] | None = None
        self.current_target: tuple[int, int] | None = None
        self.current_path: list[tuple[int, int]] = []
        self.entropy = 0.0
        self.state = "initialize"
        self.localization_count = 0
        self.last_update_time: int | None = None
        self.last_failure: Failure | None = None

    def initialize(self, obs: dict[str, Any]) -> None:
        if self.belief is None:
            free = obs["map"] == FREE
            self.belief = free.astype(float)
            self.belief /= self.belief.sum()
        self._measurement_update(obs)

    def act(self, obs: dict[str, Any]) -> str:
        if self.belief is None:
            self.initialize(obs)

        assert self.belief is not None
        self.estimated_cell = max_belief_cell(self.belief)
        target = self._choose_target(obs)

        if target != self.current_target or not self.current_path:
            self.current_target = target
            self.current_path = astar_free(obs["map"], self.estimated_cell, target)

        if len(self.current_path) < 2:
            return "stay"

        next_cell = self.current_path[1]
        return direction_to(self.estimated_cell, next_cell)

    def update(self, obs: dict[str, Any], reward: float, info: dict[str, Any]) -> None:
        assert self.belief is not None
        direction = info.get("direction", "stay")
        self.belief = predict_belief(self.belief, obs["map"], direction)
        self._measurement_update(obs)
        self.estimated_cell = max_belief_cell(self.belief)

        if self.current_path and self.estimated_cell in self.current_path:
            while self.current_path and self.current_path[0] != self.estimated_cell:
                self.current_path.pop(0)
        else:
            self.current_path = []

        failure = info.get("failure")
        self.last_failure = failure if isinstance(failure, Failure) else None
        if self.last_failure is not None and self.last_failure.recoverable:
            self.current_path = []
            self.state = "recover_belief"

    def _measurement_update(self, obs: dict[str, Any]) -> None:
        assert self.belief is not None
        measurements = obs["landmark_ranges"]
        if not measurements:
            self.entropy = belief_entropy(self.belief)
            return

        likelihood = np.ones_like(self.belief)
        for name, measured_range in measurements.items():
            landmark = obs["landmarks"][name]
            for row in range(self.belief.shape[0]):
                for col in range(self.belief.shape[1]):
                    if obs["map"][row, col] == OCCUPIED:
                        likelihood[row, col] = 0.0
                        continue
                    expected = abs(row - landmark[0]) + abs(col - landmark[1])
                    error = expected - measured_range
                    likelihood[row, col] *= np.exp(-0.5 * (error / 0.7) ** 2)

        self.belief *= likelihood
        total = self.belief.sum()
        if total <= 0.0:
            free = obs["map"] == FREE
            self.belief = free.astype(float) / free.sum()
        else:
            self.belief /= total
        self.entropy = belief_entropy(self.belief)
        self.last_update_time = int(obs["time"])

    def _choose_target(self, obs: dict[str, Any]) -> tuple[int, int]:
        assert self.estimated_cell is not None
        landmarks = obs["landmarks"]
        near_landmark = any(manhattan(self.estimated_cell, cell) <= 1 for cell in landmarks.values())

        if self.entropy > self.entropy_threshold and not near_landmark:
            self.state = "localize"
            self.localization_count += 1
            return min(
                landmarks.values(),
                key=lambda cell: manhattan(self.estimated_cell, cell),
            )

        self.state = "go_to_goal"
        return obs["goal"]


def predict_belief(belief: np.ndarray, grid: np.ndarray, direction: str) -> np.ndarray:
    predicted = np.zeros_like(belief)
    candidates = [(direction, 0.72), ("stay", 0.16)]
    if direction in ("north", "south"):
        candidates.extend([("east", 0.06), ("west", 0.06)])
    elif direction in ("east", "west"):
        candidates.extend([("north", 0.06), ("south", 0.06)])
    else:
        candidates = [("stay", 1.0)]

    for row in range(belief.shape[0]):
        for col in range(belief.shape[1]):
            if belief[row, col] == 0.0:
                continue
            for candidate_direction, probability in candidates:
                next_cell = move_cell((row, col), candidate_direction)
                if is_free(grid, next_cell):
                    predicted[next_cell] += belief[row, col] * probability
                else:
                    predicted[row, col] += belief[row, col] * probability

    total = predicted.sum()
    if total > 0.0:
        predicted /= total
    return predicted


def astar_free(
    grid: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]]:
    return grid_astar(grid == 0, start, goal)


def move_cell(cell: tuple[int, int], direction: str) -> tuple[int, int]:
    if direction == "stay":
        return cell
    dr, dc = DIRECTIONS[direction]
    return (cell[0] + dr, cell[1] + dc)


def is_free(grid: np.ndarray, cell: tuple[int, int]) -> bool:
    row, col = cell
    if row < 0 or row >= grid.shape[0] or col < 0 or col >= grid.shape[1]:
        return False
    return bool(grid[cell] == FREE)


def direction_to(start: tuple[int, int], end: tuple[int, int]) -> str:
    delta = (end[0] - start[0], end[1] - start[1])
    for direction, direction_delta in DIRECTIONS.items():
        if delta == direction_delta:
            return direction
    return "stay"


def max_belief_cell(belief: np.ndarray) -> tuple[int, int]:
    index = int(np.argmax(belief))
    row, col = np.unravel_index(index, belief.shape)
    return (int(row), int(col))


def belief_entropy(belief: np.ndarray) -> float:
    values = belief[belief > 0.0]
    return float(-np.sum(values * np.log(values)))


def manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def draw_belief_scene(
    ax: Any,
    env: BeliefGridWorld,
    agent: BeliefNavigationAgent | None,
    info: dict[str, Any] | None,
) -> None:
    belief = agent.belief if agent is not None and agent.belief is not None else np.zeros_like(env.map, dtype=float)
    belief_display = np.ma.masked_where(env.map == OCCUPIED, belief)
    ax.imshow(belief_display, cmap="viridis", origin="upper", vmin=0.0)

    obstacle_rows, obstacle_cols = np.where(env.map == OCCUPIED)
    ax.plot(obstacle_cols, obstacle_rows, "s", color="0.1", markersize=8)

    rows = [cell[0] for cell in env.trajectory]
    cols = [cell[1] for cell in env.trajectory]
    ax.plot(cols, rows, color="tab:blue", linewidth=2, label="true trajectory")
    ax.plot(env.true_robot[1], env.true_robot[0], "o", color="tab:blue", markersize=8, label="true pose")
    ax.plot(env.goal[1], env.goal[0], "*", color="tab:green", markersize=14, label="goal")

    for name, landmark in env.landmarks.items():
        ax.plot(landmark[1], landmark[0], "s", color="tab:orange", markersize=7)
        ax.text(landmark[1] + 0.45, landmark[0] + 0.10, name, fontsize=10, color="tab:orange", fontweight="bold")

    if agent is not None and agent.estimated_cell is not None:
        row, col = agent.estimated_cell
        ax.plot(col, row, "x", color="white", markersize=10, markeredgewidth=2, label="estimated pose")

    if agent is not None and agent.current_path:
        path_rows = [cell[0] for cell in agent.current_path]
        path_cols = [cell[1] for cell in agent.current_path]
        ax.plot(path_cols, path_rows, "--", color="tab:purple", linewidth=2, label="belief path")

    state = getattr(agent, "state", "none")
    entropy = getattr(agent, "entropy", 0.0)
    status = f"step={env.time} state={state} entropy={entropy:.2f}"
    if info is not None and "failure" in info:
        status += f" failure={info['failure'].kind}"
    ax.text(0.02, 0.98, status, transform=ax.transAxes, va="top", color="white", fontsize=9)
    ax.set_title("belief-based navigation")
    ax.set_xticks(np.arange(-0.5, env.width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.height, 1), minor=True)
    ax.grid(which="minor", color="0.45", linewidth=0.5)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.legend(loc="lower left", fontsize=8)


def run(seed: int = 0, render: bool = True, max_steps: int = 90) -> Trace:
    env = BeliefGridWorld(seed=seed, max_steps=max_steps)
    agent = BeliefNavigationAgent()
    obs = env.reset(seed=seed)
    agent.reset()
    agent.initialize(obs)
    trace = Trace()

    for _ in range(max_steps):
        action = agent.act(obs)
        result = env.step(action)
        obs, reward, done, info = result.as_tuple()
        agent.update(obs, reward, info)
        info["agent_state"] = agent.state
        info["entropy"] = agent.entropy
        info["estimated_cell"] = agent.estimated_cell
        info["localization_count"] = agent.localization_count
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
    entropy = trace.infos[-1].get("entropy", 0.0) if trace.infos else 0.0
    localization_count = trace.infos[-1].get("localization_count", 0) if trace.infos else 0
    print(
        f"success={success} steps={len(trace.actions)} entropy={entropy:.2f} "
        f"localization_count={localization_count} failures={failures}"
    )

    if not args.no_render:
        import matplotlib.pyplot as plt

        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
