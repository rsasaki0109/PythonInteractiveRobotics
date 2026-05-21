"""Inverse reward from a single demonstration on a grid.

A demonstrator walks from a start cell to a goal cell along a path
that detours through hidden "scenic" zones the engineer never told the
agent about. The agent observes the demo, extracts a per-cell feature
vector that includes scenic-zone membership, computes the feature
expectation of the demo path, and learns linear reward weights by
contrasting the demo feature expectation against the feature
expectation of a uniform random walk over walkable cells.

The agent then plans to a *different* goal in the same world with a
shaped A* whose edge cost is `1 - lambda * (w . phi(target_cell))`.
The learned reward should reproduce the demonstrator's preference for
scenic cells, so the planned path detours through scenic too, even
though the agent was never told what scenic means.

This is structurally different from `28_curiosity_grid_exploration.py`
and `32_empowerment_navigation.py`, which use intrinsic signals
defined by the engineer (visit count and reachable-set size). Here
the agent has no built-in preference for scenic zones - it has to
*infer* that preference from a single observed trajectory.

Success: learned A* path reaches the new goal and visits at least one
scenic cell.
Failure: timeout (terminal), no_demo_path (recoverable - the demo
trajectory is empty or does not end at the goal),
no_learned_path (terminal - shaped A* could not reach the new goal).
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
from pir.planning import CARDINAL_DIRECTIONS as DIRECTIONS, astar


FREE = 0
OCCUPIED = 1


@dataclass(frozen=True)
class IRLConfig:
    shaping_lambda: float = 0.65
    feature_clip: float = 4.0


DEFAULT_STATIC_MAP: tuple[tuple[int, ...], ...] = (
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
)


DEFAULT_SCENIC_ZONES: tuple[tuple[int, int], ...] = (
    (1, 8),
    (7, 2),
)


def build_features(
    walkable: np.ndarray,
    scenic_zones: tuple[tuple[int, int], ...],
) -> np.ndarray:
    """Return per-cell feature vector phi(s).

    Features:
      0 - 1 if scenic (a cell with `dist <= 1` from any scenic anchor), else 0.
      1 - 1 if next to a wall, else 0.
      2 - 1 if in the interior (more than 1 cell from any wall), else 0.
    """

    height, width = walkable.shape
    phi = np.zeros((height, width, 3), dtype=float)
    for r in range(height):
        for c in range(width):
            if not walkable[r, c]:
                continue
            scenic = any(abs(r - sr) + abs(c - sc) <= 1 for sr, sc in scenic_zones)
            wall_adjacent = False
            for dr, dc in DIRECTIONS:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < height and 0 <= nc < width) or not walkable[nr, nc]:
                    wall_adjacent = True
                    break
            phi[r, c, 0] = 1.0 if scenic else 0.0
            phi[r, c, 1] = 1.0 if wall_adjacent else 0.0
            phi[r, c, 2] = 1.0 if (not wall_adjacent and not scenic) else 0.0
    return phi


def feature_expectation(
    path: list[tuple[int, int]], features: np.ndarray
) -> np.ndarray:
    if not path:
        return np.zeros(features.shape[-1], dtype=float)
    return np.mean([features[r, c] for r, c in path], axis=0)


def uniform_feature_expectation(
    walkable: np.ndarray, features: np.ndarray
) -> np.ndarray:
    mask = walkable.astype(bool)
    if not mask.any():
        return np.zeros(features.shape[-1], dtype=float)
    return features[mask].mean(axis=0)


def learn_weights_from_demo(
    demo_path: list[tuple[int, int]],
    walkable: np.ndarray,
    features: np.ndarray,
    config: IRLConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (weights, mu_demo, mu_uniform).

    Weights are `clip(mu_demo - mu_uniform, -clip, +clip)`. Higher
    weight on a feature means the demo visited cells with that feature
    *more often* than chance.
    """

    cfg = config or IRLConfig()
    mu_demo = feature_expectation(demo_path, features)
    mu_uniform = uniform_feature_expectation(walkable, features)
    weights = np.clip(mu_demo - mu_uniform, -cfg.feature_clip, cfg.feature_clip)
    return weights, mu_demo, mu_uniform


def shaped_edge_cost(
    features: np.ndarray, weights: np.ndarray, walkable: np.ndarray, lam: float
) -> np.ndarray:
    """Cost of stepping into a cell. Higher `w . phi` -> lower cost.

    Floor the cost at a small positive number so Dijkstra/A* stays
    consistent.
    """

    cost = 1.0 - lam * (features @ weights)
    cost = np.maximum(cost, 0.01)
    cost[~walkable.astype(bool)] = 1e6
    return cost


def synthesize_demo(
    walkable: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    scenic_zones: tuple[tuple[int, int], ...],
) -> list[tuple[int, int]]:
    """Pretend a human walked: visit scenic anchors in order, then the goal."""

    if not scenic_zones:
        return astar(walkable, start, goal)
    waypoints = [start, *scenic_zones, goal]
    path: list[tuple[int, int]] = []
    for prev, nxt in zip(waypoints, waypoints[1:]):
        leg = astar(walkable, prev, nxt)
        if not leg:
            return []
        if path:
            leg = leg[1:]
        path.extend(leg)
    return path


class InverseRewardWorld:
    """Static grid with hidden scenic zones, a demo trajectory, and a fresh goal."""

    def __init__(
        self,
        *,
        seed: int = 0,
        max_steps: int = 80,
        static_map: tuple[tuple[int, ...], ...] = DEFAULT_STATIC_MAP,
        scenic_zones: tuple[tuple[int, int], ...] = DEFAULT_SCENIC_ZONES,
        demo_start: tuple[int, int] = (0, 0),
        demo_goal: tuple[int, int] = (9, 11),
        new_start: tuple[int, int] = (9, 0),
        new_goal: tuple[int, int] = (0, 11),
    ) -> None:
        self.seed = seed
        self.max_steps = max_steps
        self.static_map = np.asarray(static_map, dtype=int)
        self.walkable = self.static_map == FREE
        self.height, self.width = self.static_map.shape
        self.scenic_zones = scenic_zones
        self.demo_start = demo_start
        self.demo_goal = demo_goal
        self.new_start = new_start
        self.new_goal = new_goal
        self.robot: tuple[int, int] = new_start
        self.step_count = 0
        self.trajectory: list[tuple[int, int]] = [new_start]

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            self.seed = seed
        self.robot = self.new_start
        self.step_count = 0
        self.trajectory = [self.new_start]
        return self.observe()

    def observe(self) -> dict[str, Any]:
        return {
            "robot": self.robot,
            "goal": self.new_goal,
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
                message="Action moves the robot off the grid.",
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
        success = self.robot == self.new_goal
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

    def render(self, agent: "InverseRewardAgent", info: dict[str, Any]) -> None:
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
    env: InverseRewardWorld,
    agent: "InverseRewardAgent",
    info: dict[str, Any],
) -> None:
    ax.imshow(env.static_map, cmap="gray_r", origin="upper", vmin=0, vmax=1)

    scenic_mask = np.zeros_like(env.walkable, dtype=float)
    for r, c in agent.scenic_cells:
        scenic_mask[r, c] = 1.0
    masked = np.where(scenic_mask > 0, scenic_mask, np.nan)
    ax.imshow(masked, cmap="YlGn", origin="upper", alpha=0.4, vmin=0.0, vmax=1.0)

    if agent.demo_path:
        rows = [r for r, _ in agent.demo_path]
        cols = [c for _, c in agent.demo_path]
        ax.plot(cols, rows, "--", color="tab:purple", linewidth=1.6, alpha=0.85, label="demo")
    if agent.baseline_path:
        rows = [r for r, _ in agent.baseline_path]
        cols = [c for _, c in agent.baseline_path]
        ax.plot(cols, rows, "-.", color="tab:red", linewidth=1.4, alpha=0.65, label="baseline A*")
    if agent.learned_path:
        rows = [r for r, _ in agent.learned_path]
        cols = [c for _, c in agent.learned_path]
        ax.plot(cols, rows, "-", color="tab:blue", linewidth=2.0, alpha=0.95, label="learned-reward A*")

    traj_rows = [r for r, _ in env.trajectory]
    traj_cols = [c for _, c in env.trajectory]
    ax.plot(traj_cols, traj_rows, color="tab:cyan", linewidth=3.0, alpha=0.55)

    ax.plot(env.demo_start[1], env.demo_start[0], "P", color="tab:purple", markersize=10)
    ax.plot(env.demo_goal[1], env.demo_goal[0], "*", color="tab:purple", markersize=14)
    ax.plot(env.new_start[1], env.new_start[0], "o", color="white", markeredgecolor="black", markersize=10)
    ax.plot(env.new_goal[1], env.new_goal[0], "*", color="tab:green", markersize=16)
    for sr, sc in env.scenic_zones:
        ax.plot(sc, sr, "x", color="tab:olive", markersize=10, markeredgewidth=2)
    ax.plot(env.robot[1], env.robot[0], "o", color="tab:blue", markersize=10)

    weight_text = ", ".join(f"{w:+.2f}" for w in agent.weights)
    status = (
        f"step={env.step_count}  state={agent.state}\n"
        f"weights [scenic, wall_adj, interior] = [{weight_text}]\n"
        f"demo scenic={agent.demo_scenic_step_count}  "
        f"learned scenic={agent.learned_scenic_step_count}  "
        f"baseline scenic={agent.baseline_scenic_step_count}"
    )
    if "failure" in info:
        status += f"\nfailure={info['failure'].kind}"
    ax.text(0.02, 0.98, status, transform=ax.transAxes, va="top", fontsize=9, family="monospace")
    ax.set_title("Inverse reward from one demo: learn -> plan to new goal")
    ax.set_xticks(np.arange(-0.5, env.width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.height, 1), minor=True)
    ax.grid(which="minor", color="0.85", linewidth=0.5)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.legend(loc="lower right", fontsize=8)


class InverseRewardAgent:
    """Learn linear reward weights from one demo, then plan with shaped A*."""

    def __init__(
        self,
        env: InverseRewardWorld,
        config: IRLConfig | None = None,
    ) -> None:
        self.env = env
        self.config = config or IRLConfig()
        self.features = build_features(env.walkable, env.scenic_zones)
        self.demo_path = synthesize_demo(
            env.walkable, env.demo_start, env.demo_goal, env.scenic_zones
        )
        self.scenic_cells: set[tuple[int, int]] = set()
        scenic_dim = self.features[:, :, 0]
        for r in range(env.height):
            for c in range(env.width):
                if scenic_dim[r, c] > 0.5:
                    self.scenic_cells.add((r, c))
        self.weights, self.mu_demo, self.mu_uniform = learn_weights_from_demo(
            self.demo_path, env.walkable, self.features, self.config
        )
        self.shaped_cost = shaped_edge_cost(
            self.features, self.weights, env.walkable, self.config.shaping_lambda
        )
        self.learned_path = astar(
            env.walkable,
            env.new_start,
            env.new_goal,
            edge_cost=self.shaped_cost,
        )
        self.baseline_path = astar(env.walkable, env.new_start, env.new_goal)
        self.demo_scenic_step_count = self._count_scenic(self.demo_path)
        self.learned_scenic_step_count = self._count_scenic(self.learned_path)
        self.baseline_scenic_step_count = self._count_scenic(self.baseline_path)
        self.reset()

    def reset(self) -> None:
        self.state = "follow_learned"
        self.step_index = 0

    def _count_scenic(self, path: list[tuple[int, int]]) -> int:
        return sum(1 for cell in path if cell in self.scenic_cells)

    def act(self, obs: dict[str, Any]) -> tuple[int, int]:
        if not self.learned_path:
            self.state = "no_path"
            return (0, 0)
        if self.step_index + 1 >= len(self.learned_path):
            self.state = "arrived"
            return (0, 0)
        current = self.learned_path[self.step_index]
        nxt = self.learned_path[self.step_index + 1]
        return (nxt[0] - current[0], nxt[1] - current[1])

    def update(
        self, obs: dict[str, Any], reward: float, info: dict[str, Any]
    ) -> None:
        del reward
        current = obs["robot"]
        if self.step_index + 1 < len(self.learned_path) and current == self.learned_path[self.step_index + 1]:
            self.step_index += 1
        info["agent_state"] = self.state
        info["learned_weights"] = self.weights.tolist()
        info["scenic_here"] = current in self.scenic_cells

    def info(self) -> dict[str, Any]:
        return {
            "agent_state": self.state,
            "step_index": int(self.step_index),
            "demo_path_length": len(self.demo_path),
            "learned_path_length": len(self.learned_path),
            "baseline_path_length": len(self.baseline_path),
            "demo_scenic_step_count": int(self.demo_scenic_step_count),
            "learned_scenic_step_count": int(self.learned_scenic_step_count),
            "baseline_scenic_step_count": int(self.baseline_scenic_step_count),
            "learned_weights": self.weights.tolist(),
            "mu_demo": self.mu_demo.tolist(),
            "mu_uniform": self.mu_uniform.tolist(),
        }


def run(seed: int = 0, render: bool = True, max_steps: int = 80) -> Trace:
    env = InverseRewardWorld(seed=seed, max_steps=max_steps)
    agent = InverseRewardAgent(env)
    obs = env.reset(seed=seed)
    agent.reset()
    trace = Trace()

    if not agent.demo_path:
        info = {
            "failure": Failure(
                kind="no_demo_path",
                message="Could not synthesize a demo path through the scenic zones.",
                recoverable=True,
            )
        }
        trace.append(obs, (0, 0), 0.0, info)
        return trace

    if not agent.learned_path:
        info = {
            "failure": Failure(
                kind="no_learned_path",
                message="Shaped A* could not reach the new goal.",
                recoverable=False,
            )
        }
        trace.append(obs, (0, 0), 0.0, info)
        return trace

    for _ in range(max_steps):
        action = agent.act(obs)
        if action == (0, 0):
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
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    trace = run(seed=args.seed, render=not args.no_render, max_steps=args.max_steps)
    final_info = trace.infos[-1] if trace.infos else {}
    failures = [failure.kind for failure in trace.failures()]
    weights = final_info.get("learned_weights", [])
    weights_str = ", ".join(f"{w:+.2f}" for w in weights)
    print(
        f"success={final_info.get('success', False)} "
        f"steps={len(trace.actions)} "
        f"demo_path={final_info.get('demo_path_length', 0)} "
        f"learned_path={final_info.get('learned_path_length', 0)} "
        f"baseline_path={final_info.get('baseline_path_length', 0)} "
        f"demo_scenic={final_info.get('demo_scenic_step_count', 0)} "
        f"learned_scenic={final_info.get('learned_scenic_step_count', 0)} "
        f"baseline_scenic={final_info.get('baseline_scenic_step_count', 0)} "
        f"weights=[{weights_str}] "
        f"failures={failures}"
    )

    if not args.no_render:
        import matplotlib.pyplot as plt

        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
