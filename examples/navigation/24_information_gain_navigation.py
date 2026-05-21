"""Navigation with an explicit information-gain detour before committing.

A vertical wall blocks the direct route from start to goal. The wall has one
candidate gate of unknown state and one known bottom opening. If the gate is
open, a short route exists. If closed, the agent must take a longer detour
via the bottom opening.

The agent's policy makes the value-of-information step explicit: it first
moves to an observation point that lidar-reveals the gate state, and only then
runs A* to the goal with the gate state known. This is in contrast to
`04_online_replanning_astar.py`, which discovers the same information passively
while running greedy A* on an optimistic map.
"""

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

from pir.core.random import make_rng
from pir.core.types import Failure, StepResult, Trace


UNKNOWN = -1
FREE = 0
OCCUPIED = 1

DIRECTIONS: dict[str, tuple[int, int]] = {
    "N": (-1, 0),
    "S": (1, 0),
    "E": (0, 1),
    "W": (0, -1),
}


class InformationGainNavigationWorld:
    """Grid with a vertical wall, one candidate gate, and a bottom opening."""

    def __init__(
        self,
        *,
        seed: int | None = 0,
        max_steps: int = 100,
        height: int = 20,
        width: int = 20,
        start: tuple[int, int] = (1, 1),
        goal: tuple[int, int] = (15, 18),
        barrier_col: int = 10,
        candidate_cell: tuple[int, int] = (7, 10),
        candidate_open: bool = True,
        bottom_opening: tuple[int, int] = (19, 10),
        lidar_range: int = 5,
    ) -> None:
        self.seed = seed
        self.height = height
        self.width = width
        self.start = tuple(start)
        self.goal = tuple(goal)
        self.barrier_col = barrier_col
        self.candidate_cell = tuple(candidate_cell)
        self.candidate_open_truth = bool(candidate_open)
        self.bottom_opening = tuple(bottom_opening)
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
        self.true_map = self._build_true_map()
        self.known_map = np.full((self.height, self.width), UNKNOWN, dtype=int)
        self.robot = self.start
        self.time = 0
        self.trajectory: list[tuple[int, int]] = [self.robot]
        self._lidar_scan()
        return self.observe()

    def _build_true_map(self) -> np.ndarray:
        m = np.full((self.height, self.width), FREE, dtype=int)
        for r in range(self.height):
            cell = (r, self.barrier_col)
            if cell == self.bottom_opening:
                continue
            if cell == self.candidate_cell:
                m[cell] = FREE if self.candidate_open_truth else OCCUPIED
                continue
            m[cell] = OCCUPIED
        return m

    def _lidar_scan(self) -> None:
        r, c = self.robot
        if 0 <= r < self.height and 0 <= c < self.width:
            self.known_map[r, c] = self.true_map[r, c]
        for dr, dc in DIRECTIONS.values():
            for step in range(1, self.lidar_range + 1):
                nr, nc = r + dr * step, c + dc * step
                if not (0 <= nr < self.height and 0 <= nc < self.width):
                    break
                value = int(self.true_map[nr, nc])
                self.known_map[nr, nc] = value
                if value == OCCUPIED:
                    break

    def observe(self) -> dict[str, Any]:
        return {
            "time": self.time,
            "robot": self.robot,
            "goal": self.goal,
            "known_map": self.known_map.copy(),
            "candidate_cell": self.candidate_cell,
            "bottom_opening": self.bottom_opening,
            "lidar_range": self.lidar_range,
            "height": self.height,
            "width": self.width,
        }

    def step(self, action: Any) -> StepResult:
        self.time += 1
        direction = action if isinstance(action, str) else action.get("direction") if isinstance(action, dict) else None
        info: dict[str, Any] = {"time": self.time, "direction": direction, "success": False}

        if direction is None or direction == "stay":
            info["agent_pos"] = list(self.robot)
            return StepResult(self.observe(), -0.02, False, info)

        if direction not in DIRECTIONS:
            info["failure"] = Failure("invalid_direction", f"unknown direction: {direction}", True)
            return StepResult(self.observe(), -0.05, False, info)

        dr, dc = DIRECTIONS[direction]
        next_cell = (self.robot[0] + dr, self.robot[1] + dc)
        if not (0 <= next_cell[0] < self.height and 0 <= next_cell[1] < self.width):
            info["failure"] = Failure("out_of_bounds", f"step would leave grid: {next_cell}", True)
            return StepResult(self.observe(), -0.1, False, info)
        if self.true_map[next_cell] == OCCUPIED:
            info["failure"] = Failure("collision", f"cell blocked: {next_cell}", True)
            return StepResult(self.observe(), -0.2, False, info)

        self.robot = next_cell
        self.trajectory.append(self.robot)
        self._lidar_scan()
        info["agent_pos"] = list(self.robot)

        if self.robot == self.goal:
            info["success"] = True
            return StepResult(self.observe(), 1.0, True, info)
        return StepResult(self.observe(), -0.02, False, info)

    def render(self, agent: Any | None = None, info: dict[str, Any] | None = None) -> None:
        import matplotlib.pyplot as plt

        if self._fig is None or self._ax is None:
            plt.ion()
            self._fig, self._ax = plt.subplots(figsize=(5.6, 5.2))
        ax = self._ax
        ax.clear()
        draw_information_gain_scene(ax, self, agent, info)
        self._fig.canvas.draw_idle()
        plt.pause(0.1)


def _heuristic(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(
    known_map: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    *,
    treat_unknown_as_free: bool = True,
) -> list[tuple[int, int]]:
    height, width = known_map.shape
    frontier: list[tuple[int, int, tuple[int, int]]] = []
    heapq.heappush(frontier, (_heuristic(start, goal), 0, start))
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    cost_so_far: dict[tuple[int, int], int] = {start: 0}

    while frontier:
        _, cost, current = heapq.heappop(frontier)
        if current == goal:
            break
        for dr, dc in DIRECTIONS.values():
            neighbor = (current[0] + dr, current[1] + dc)
            if not (0 <= neighbor[0] < height and 0 <= neighbor[1] < width):
                continue
            value = int(known_map[neighbor])
            if value == OCCUPIED:
                continue
            if value == UNKNOWN and not treat_unknown_as_free:
                continue
            new_cost = cost + 1
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + _heuristic(neighbor, goal)
                heapq.heappush(frontier, (priority, new_cost, neighbor))
                came_from[neighbor] = current

    if goal not in came_from:
        return []
    path: list[tuple[int, int]] = []
    cur: tuple[int, int] | None = goal
    while cur is not None:
        path.append(cur)
        cur = came_from[cur]
    path.reverse()
    return path


def _direction_to(start: tuple[int, int], end: tuple[int, int]) -> str:
    delta = (end[0] - start[0], end[1] - start[1])
    for direction, vec in DIRECTIONS.items():
        if vec == delta:
            return direction
    return "stay"


class InformationGainNavigationAgent:
    """Scout the gate state first, then run A* with full information."""

    def __init__(
        self,
        observation_point: tuple[int, int] = (7, 5),
        candidate_cell: tuple[int, int] = (7, 10),
        info_gain_weight: float = 1.5,
    ) -> None:
        self.observation_point = tuple(observation_point)
        self.candidate_cell = tuple(candidate_cell)
        self.info_gain_weight = info_gain_weight
        self.reset()

    def reset(self) -> None:
        self.state: str = "scout"
        self.belief: float = 0.5
        self.current_path: list[tuple[int, int]] = []
        self.scout_target: tuple[int, int] | None = None
        self.info_gain_step_count: int = 0
        self.navigation_step_count: int = 0
        self.replan_count: int = 0
        self.observed_candidate: bool = False
        self.commit_path_length: int | None = None

    @property
    def belief_entropy(self) -> float:
        p = float(np.clip(self.belief, 1e-9, 1.0 - 1e-9))
        return float(-p * np.log(p) - (1 - p) * np.log(1 - p))

    def act(self, obs: dict[str, Any]) -> str:
        known_map = obs["known_map"]
        robot = tuple(obs["robot"])
        goal = tuple(obs["goal"])
        self._update_belief_from_map(known_map)

        if self.state == "scout":
            return self._scout_step(known_map, robot, goal)
        return self._navigate_step(known_map, robot, goal)

    def _update_belief_from_map(self, known_map: np.ndarray) -> None:
        value = int(known_map[self.candidate_cell])
        if value == FREE:
            self.belief = 1.0
            self.observed_candidate = True
            if self.state == "scout":
                self.state = "navigate"
        elif value == OCCUPIED:
            self.belief = 0.0
            self.observed_candidate = True
            if self.state == "scout":
                self.state = "navigate"

    def _scout_step(self, known_map: np.ndarray, robot: tuple[int, int], goal: tuple[int, int]) -> str:
        target = self.observation_point
        self.scout_target = target
        if not self.current_path or self.current_path[-1] != target or robot not in self.current_path:
            self.current_path = astar(known_map, robot, target)
            self.replan_count += 1
        action = self._follow_path(robot)
        if action != "stay":
            self.info_gain_step_count += 1
        return action

    def _navigate_step(self, known_map: np.ndarray, robot: tuple[int, int], goal: tuple[int, int]) -> str:
        if not self.current_path or self.current_path[-1] != goal or robot not in self.current_path:
            self.current_path = astar(known_map, robot, goal)
            self.replan_count += 1
            if self.commit_path_length is None:
                self.commit_path_length = max(len(self.current_path) - 1, 0)
        # if path invalidated by new obstacle observation, replan
        if any(known_map[cell] == OCCUPIED for cell in self.current_path):
            self.current_path = astar(known_map, robot, goal)
            self.replan_count += 1
        action = self._follow_path(robot)
        if action != "stay":
            self.navigation_step_count += 1
        return action

    def _follow_path(self, robot: tuple[int, int]) -> str:
        while self.current_path and self.current_path[0] != robot:
            self.current_path.pop(0)
        if len(self.current_path) < 2:
            return "stay"
        return _direction_to(self.current_path[0], self.current_path[1])

    def update(self, obs: dict[str, Any], reward: float, info: dict[str, Any]) -> None:
        if info.get("success"):
            self.state = "succeeded"


def draw_information_gain_scene(
    ax: Any,
    env: InformationGainNavigationWorld,
    agent: InformationGainNavigationAgent | None,
    info: dict[str, Any] | None,
) -> None:
    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(["0.72", "white", "0.1"])
    display = np.zeros_like(env.known_map, dtype=int)
    display[env.known_map == UNKNOWN] = 0
    display[env.known_map == FREE] = 1
    display[env.known_map == OCCUPIED] = 2
    ax.imshow(display, cmap=cmap, origin="upper", vmin=0, vmax=2)

    if agent is not None and len(agent.current_path) > 1:
        rows = [c[0] for c in agent.current_path]
        cols = [c[1] for c in agent.current_path]
        ax.plot(cols, rows, "--", color="tab:purple", linewidth=1.6, alpha=0.7)

    rows = [c[0] for c in env.trajectory]
    cols = [c[1] for c in env.trajectory]
    ax.plot(cols, rows, color="tab:blue", linewidth=1.5, alpha=0.8)

    # candidate cell
    cr, cc = env.candidate_cell
    ax.plot(cc, cr, marker="o", color="tab:orange", markersize=10, mew=2, fillstyle="none")
    # bottom opening
    br, bc = env.bottom_opening
    ax.plot(bc, br, marker="s", color="tab:cyan", markersize=8, fillstyle="none")
    # observation point
    if agent is not None:
        opr, opc = agent.observation_point
        ax.plot(opc, opr, marker="x", color="tab:green", markersize=10, mew=2)

    # agent & goal
    ax.plot(env.robot[1], env.robot[0], "o", color="tab:blue", markersize=10)
    ax.plot(env.goal[1], env.goal[0], "*", color="tab:green", markersize=15)

    status_parts: list[str] = [f"step={env.time}"]
    if agent is not None:
        status_parts.append(f"state={agent.state}")
        status_parts.append(f"belief={agent.belief:.2f}")
        status_parts.append(f"info_steps={agent.info_gain_step_count}")
        status_parts.append(f"nav_steps={agent.navigation_step_count}")
        status_parts.append(f"replans={agent.replan_count}")
    if info is not None and "failure" in info:
        status_parts.append(f"failure={info['failure'].kind}")
    if info is not None and info.get("success"):
        status_parts.append("success")
    ax.set_title("information-gain navigation")
    ax.text(0.02, 0.98, "  ".join(status_parts), transform=ax.transAxes, va="top", fontsize=8)
    ax.set_xticks(np.arange(-0.5, env.width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.height, 1), minor=True)
    ax.grid(which="minor", color="0.85", linewidth=0.4)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)


def run(
    seed: int = 0,
    render: bool = True,
    max_steps: int = 100,
    candidate_open: bool = True,
) -> Trace:
    env = InformationGainNavigationWorld(seed=seed, max_steps=max_steps, candidate_open=candidate_open)
    agent = InformationGainNavigationAgent()
    obs = env.reset(seed=seed)
    agent.reset()
    trace = Trace()

    for _ in range(max_steps):
        action = agent.act(obs)
        result = env.step(action)
        obs, reward, done, info = result.as_tuple()
        agent.update(obs, reward, info)
        info["agent_state"] = agent.state
        info["belief"] = agent.belief
        info["belief_entropy"] = agent.belief_entropy
        info["info_gain_step_count"] = agent.info_gain_step_count
        info["navigation_step_count"] = agent.navigation_step_count
        info["replan_count"] = agent.replan_count
        info["observed_candidate"] = agent.observed_candidate
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
    parser.add_argument("--candidate-closed", action="store_true", help="set gate closed")
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    trace = run(
        seed=args.seed,
        render=not args.no_render,
        max_steps=args.max_steps,
        candidate_open=not args.candidate_closed,
    )
    success = bool(trace.infos and trace.infos[-1].get("success"))
    failures = [failure.kind for failure in trace.failures()]
    final = trace.infos[-1] if trace.infos else {}
    print(
        f"success={success} steps={len(trace.actions)} "
        f"info_steps={final.get('info_gain_step_count', 0)} "
        f"nav_steps={final.get('navigation_step_count', 0)} "
        f"replans={final.get('replan_count', 0)} "
        f"belief={final.get('belief', 0.5):.2f} "
        f"failures={failures}"
    )

    if not args.no_render:
        import matplotlib.pyplot as plt

        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
