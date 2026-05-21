"""Empowerment as an intrinsic motivation for grid navigation.

Empowerment measures how much the agent's action choices can influence
its future state. For a deterministic gridworld with cardinal actions
the k-step empowerment of a cell `s` is approximated by
`E_k(s) = log2(|reachable_in_k_steps(s)|)`. Cells in open space have
many reachable successors; cells in narrow corridors or near walls have
few. The agent uses `E_k` as a state-intrinsic shaping signal added to
its planning cost, so it prefers routes that keep options open even
when those routes are slightly longer.

This is structurally different from `28_curiosity_grid_exploration.py`,
where the intrinsic signal is *experience-dependent* (visit counts
decay over time). Empowerment is a property of the *world geometry
alone*, computed once before the agent moves.

This example computes two A* paths on the same grid: a baseline
Manhattan path and an empowerment-shaped path. The robot follows the
empowerment-shaped path and the trace records the mean empowerment of
each path so the student can compare them quantitatively.

Success: robot reaches the goal cell.
Failure: timeout (terminal), no_path (terminal - the goal is
unreachable from the start under the walkable map).
"""

from __future__ import annotations

import argparse
import heapq
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pir.core.types import Failure, StepResult, Trace


FREE = 0
OCCUPIED = 1
DIRECTIONS: tuple[tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))


@dataclass(frozen=True)
class EmpowermentConfig:
    empowerment_horizon: int = 3
    shaping_lambda: float = 0.45
    low_empowerment_threshold: float = 3.4


DEFAULT_STATIC_MAP: tuple[tuple[int, ...], ...] = (
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
)


def compute_empowerment(walkable: np.ndarray, k: int) -> np.ndarray:
    """Per-cell `E_k(s) = log2(|reachable_in_at_most_k_steps|)`."""

    height, width = walkable.shape
    empowerment = np.zeros((height, width), dtype=float)
    for r in range(height):
        for c in range(width):
            if not walkable[r, c]:
                continue
            empowerment[r, c] = np.log2(
                _reachable_in_k(walkable, (r, c), k)
            )
    return empowerment


def _reachable_in_k(
    walkable: np.ndarray, start: tuple[int, int], k: int
) -> int:
    height, width = walkable.shape
    seen = {start}
    frontier: deque[tuple[tuple[int, int], int]] = deque([(start, 0)])
    while frontier:
        (r, c), depth = frontier.popleft()
        if depth == k:
            continue
        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < height and 0 <= nc < width):
                continue
            if not walkable[nr, nc]:
                continue
            if (nr, nc) in seen:
                continue
            seen.add((nr, nc))
            frontier.append(((nr, nc), depth + 1))
    return len(seen)


def astar(
    walkable: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    edge_cost: np.ndarray | None = None,
) -> list[tuple[int, int]]:
    """A* with optional per-target-cell edge cost. Returns [] if no path."""

    height, width = walkable.shape
    if not walkable[start] or not walkable[goal]:
        return []
    cost: dict[tuple[int, int], float] = {start: 0.0}
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    heap: list[tuple[float, tuple[int, int]]] = [(0.0, start)]
    while heap:
        _, current = heapq.heappop(heap)
        if current == goal:
            break
        for dr, dc in DIRECTIONS:
            nr, nc = current[0] + dr, current[1] + dc
            if not (0 <= nr < height and 0 <= nc < width):
                continue
            if not walkable[nr, nc]:
                continue
            step = 1.0 if edge_cost is None else float(edge_cost[nr, nc])
            new_cost = cost[current] + step
            if (nr, nc) not in cost or new_cost < cost[(nr, nc)]:
                cost[(nr, nc)] = new_cost
                parent[(nr, nc)] = current
                h = abs(nr - goal[0]) + abs(nc - goal[1])
                heapq.heappush(heap, (new_cost + h, (nr, nc)))
    if goal not in parent:
        return []
    path: list[tuple[int, int]] = [goal]
    while parent[path[-1]] is not None:
        path.append(parent[path[-1]])  # type: ignore[arg-type]
    return list(reversed(path))


def empowerment_edge_cost(
    empowerment: np.ndarray,
    lam: float,
) -> np.ndarray:
    """Cost of stepping into a cell. Higher empowerment -> lower cost."""

    e_max = float(empowerment.max()) if empowerment.size else 0.0
    cost = 1.0 + lam * (e_max - empowerment)
    cost[empowerment <= 0.0] = 1e6
    return cost


class EmpowermentGridWorld:
    """Static gridworld for showing how empowerment shapes route choice."""

    def __init__(
        self,
        *,
        seed: int = 0,
        max_steps: int = 60,
        static_map: tuple[tuple[int, ...], ...] = DEFAULT_STATIC_MAP,
        start: tuple[int, int] = (0, 0),
        goal: tuple[int, int] = (9, 11),
    ) -> None:
        self.seed = seed
        self.max_steps = max_steps
        self.static_map = np.asarray(static_map, dtype=int)
        self.walkable = self.static_map == FREE
        self.height, self.width = self.static_map.shape
        self.start = start
        self.goal = goal
        self.robot: tuple[int, int] = start
        self.step_count = 0
        self.trajectory: list[tuple[int, int]] = [start]

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            self.seed = seed
        self.robot = self.start
        self.step_count = 0
        self.trajectory = [self.start]
        return self.observe()

    def observe(self) -> dict[str, Any]:
        return {
            "robot": self.robot,
            "goal": self.goal,
            "step": self.step_count,
            "walkable": self.walkable,
        }

    def step(self, action: tuple[int, int]) -> StepResult:
        self.step_count += 1
        nr, nc = self.robot[0] + action[0], self.robot[1] + action[1]
        info: dict[str, Any] = {}
        if not (0 <= nr < self.height and 0 <= nc < self.width):
            info["failure"] = Failure(
                kind="out_of_bounds",
                message="Action would move the robot off the grid.",
                recoverable=True,
            )
            return StepResult(self.observe(), -1.0, False, info)
        if not self.walkable[nr, nc]:
            info["failure"] = Failure(
                kind="collision",
                message=f"Cell ({nr},{nc}) is occupied.",
                recoverable=True,
            )
            return StepResult(self.observe(), -1.0, False, info)
        self.robot = (nr, nc)
        self.trajectory.append(self.robot)
        success = self.robot == self.goal
        timed_out = self.step_count >= self.max_steps and not success
        if success:
            info["success"] = True
        elif timed_out:
            info["failure"] = Failure(
                kind="timeout",
                message="Did not reach the goal before max_steps.",
                recoverable=False,
            )
        reward = -1.0 if not success else 0.0
        return StepResult(self.observe(), reward, success or timed_out, info)

    def render(self, agent: "EmpowermentNavigationAgent", info: dict[str, Any]) -> None:
        import matplotlib.pyplot as plt

        if not hasattr(self, "_fig") or self._fig is None:
            plt.ion()
            self._fig, self._ax = plt.subplots(figsize=(6.4, 5.4))
        ax = self._ax
        ax.clear()
        _draw_scene(ax, self, agent, info)
        self._fig.canvas.draw_idle()
        plt.pause(0.001)


def _draw_scene(
    ax: Any,
    env: EmpowermentGridWorld,
    agent: "EmpowermentNavigationAgent",
    info: dict[str, Any],
) -> None:
    ax.imshow(env.static_map, cmap="gray_r", origin="upper", vmin=0, vmax=1)
    masked = np.where(env.walkable, agent.empowerment, np.nan)
    ax.imshow(masked, cmap="viridis", origin="upper", alpha=0.55)

    if agent.baseline_path:
        rows = [r for r, _ in agent.baseline_path]
        cols = [c for _, c in agent.baseline_path]
        ax.plot(cols, rows, "--", color="tab:red", linewidth=1.6, alpha=0.85, label="baseline A*")
    if agent.shaped_path:
        rows = [r for r, _ in agent.shaped_path]
        cols = [c for _, c in agent.shaped_path]
        ax.plot(cols, rows, "-", color="tab:blue", linewidth=2.0, alpha=0.95, label="empowerment A*")

    traj_rows = [r for r, _ in env.trajectory]
    traj_cols = [c for _, c in env.trajectory]
    ax.plot(traj_cols, traj_rows, color="tab:cyan", linewidth=3.0, alpha=0.55)

    ax.plot(env.start[1], env.start[0], "o", color="white", markeredgecolor="black", markersize=10)
    ax.plot(env.goal[1], env.goal[0], "*", color="tab:green", markersize=16)
    ax.plot(env.robot[1], env.robot[0], "o", color="tab:blue", markersize=10)

    status = (
        f"step={env.step_count}  state={agent.state}\n"
        f"mean E baseline={agent.baseline_mean_empowerment:.2f} "
        f"shaped={agent.shaped_mean_empowerment:.2f}\n"
        f"low_emp_step_count={agent.low_empowerment_step_count} "
        f"detour_step_count={agent.detour_step_count}"
    )
    if "failure" in info:
        status += f"\nfailure={info['failure'].kind}"
    ax.text(0.02, 0.98, status, transform=ax.transAxes, va="top", fontsize=9, family="monospace")
    ax.set_title(
        f"Empowerment shaping (k={agent.config.empowerment_horizon}, "
        f"lambda={agent.config.shaping_lambda})"
    )
    ax.set_xticks(np.arange(-0.5, env.width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.height, 1), minor=True)
    ax.grid(which="minor", color="0.85", linewidth=0.5)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.legend(loc="lower right", fontsize=8)


class EmpowermentNavigationAgent:
    """Plan once with empowerment shaping; track mean empowerment along the path."""

    def __init__(
        self,
        env: EmpowermentGridWorld,
        config: EmpowermentConfig | None = None,
    ) -> None:
        self.env = env
        self.config = config or EmpowermentConfig()
        self.empowerment = compute_empowerment(env.walkable, self.config.empowerment_horizon)
        self.baseline_path = astar(env.walkable, env.start, env.goal)
        self.shaped_edge_cost = empowerment_edge_cost(
            self.empowerment, self.config.shaping_lambda
        )
        self.shaped_path = astar(
            env.walkable,
            env.start,
            env.goal,
            edge_cost=self.shaped_edge_cost,
        )
        self.baseline_set = {cell for cell in self.baseline_path}
        self.shaped_set = {cell for cell in self.shaped_path}
        self.baseline_mean_empowerment = self._mean_empowerment(self.baseline_path)
        self.shaped_mean_empowerment = self._mean_empowerment(self.shaped_path)
        self.reset()

    def reset(self) -> None:
        self.state = "follow_shaped"
        self.step_index = 0
        self.low_empowerment_step_count = 0
        self.detour_step_count = 0

    def _mean_empowerment(self, path: list[tuple[int, int]]) -> float:
        if not path:
            return 0.0
        return float(np.mean([self.empowerment[r, c] for r, c in path]))

    def act(self, obs: dict[str, Any]) -> tuple[int, int]:
        if not self.shaped_path:
            self.state = "no_path"
            return (0, 0)
        if self.step_index + 1 >= len(self.shaped_path):
            self.state = "arrived"
            return (0, 0)
        current = self.shaped_path[self.step_index]
        nxt = self.shaped_path[self.step_index + 1]
        return (nxt[0] - current[0], nxt[1] - current[1])

    def update(
        self, obs: dict[str, Any], reward: float, info: dict[str, Any]
    ) -> None:
        del reward
        current = obs["robot"]
        if current == self.shaped_path[self.step_index + 1] if self.step_index + 1 < len(
            self.shaped_path
        ) else False:
            self.step_index += 1
        elif current in self.shaped_set:
            for index, cell in enumerate(self.shaped_path):
                if cell == current:
                    self.step_index = index
                    break
        e_here = float(self.empowerment[current])
        if e_here < self.config.low_empowerment_threshold and current != self.env.start:
            self.low_empowerment_step_count += 1
        if current not in self.baseline_set:
            self.detour_step_count += 1
        info["agent_state"] = self.state
        info["empowerment_here"] = e_here

    def info(self) -> dict[str, Any]:
        return {
            "agent_state": self.state,
            "step_index": int(self.step_index),
            "low_empowerment_step_count": int(self.low_empowerment_step_count),
            "detour_step_count": int(self.detour_step_count),
            "baseline_path_length": len(self.baseline_path),
            "shaped_path_length": len(self.shaped_path),
            "baseline_mean_empowerment": float(self.baseline_mean_empowerment),
            "shaped_mean_empowerment": float(self.shaped_mean_empowerment),
            "empowerment_max": float(self.empowerment.max()),
        }


def run(seed: int = 0, render: bool = True, max_steps: int = 60) -> Trace:
    env = EmpowermentGridWorld(seed=seed, max_steps=max_steps)
    agent = EmpowermentNavigationAgent(env)
    obs = env.reset(seed=seed)
    agent.reset()
    trace = Trace()

    for _ in range(max_steps):
        action = agent.act(obs)
        if action == (0, 0):
            # arrived or no_path: terminate the run gracefully
            if agent.state == "no_path":
                info = {
                    "failure": Failure(
                        kind="no_path",
                        message="Empowerment-shaped A* could not reach the goal.",
                        recoverable=False,
                    )
                }
                trace.append(obs, action, 0.0, info)
            break
        obs, reward, done, info = env.step(action).as_tuple()
        agent.update(obs, reward, info)
        info.update(agent.info())
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
    final_info = trace.infos[-1] if trace.infos else {}
    failures = [failure.kind for failure in trace.failures()]
    print(
        f"success={final_info.get('success', False)} "
        f"steps={len(trace.actions)} "
        f"baseline_path={final_info.get('baseline_path_length', 0)} "
        f"shaped_path={final_info.get('shaped_path_length', 0)} "
        f"baseline_mean_E={final_info.get('baseline_mean_empowerment', 0.0):.2f} "
        f"shaped_mean_E={final_info.get('shaped_mean_empowerment', 0.0):.2f} "
        f"low_emp_steps={final_info.get('low_empowerment_step_count', 0)} "
        f"detour_steps={final_info.get('detour_step_count', 0)} "
        f"failures={failures}"
    )

    if not args.no_render:
        import matplotlib.pyplot as plt

        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
