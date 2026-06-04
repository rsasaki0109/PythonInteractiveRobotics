"""Plan a path with an RRT, discover a hidden obstacle on it, and replan around it.

A Rapidly-exploring Random Tree (LaValle, 1998) plans in continuous space by
growing a tree from the start: sample a random point, extend the nearest node a
short step toward it, keep the edge if it is collision-free, and stop when the
tree reaches the goal. It is the sampling-based counterpart to the grid A* in
`04_online_replanning_astar.py`.

This example is the online-replanning loop in continuous space. The robot plans
with the obstacles it currently knows about and starts following the path — but
one obstacle is hidden until the robot is close enough to sense it. When the
newly sensed obstacle is found to block the remaining path, the world reports a
`path_blocked` failure, and the agent replans an RRT from its current position
with the updated obstacle set and continues. Plan, discover, replan, arrive.

Success: the robot reaches the goal radius.
Failure: path_blocked (recoverable - a sensed obstacle invalidated the path and
triggered a replan), rrt_unreachable (terminal - sampling budget spent without a
path), collision (terminal), timeout (terminal).

References:
  * S. M. LaValle, "Rapidly-Exploring Random Trees: A New Tool for Path
    Planning," TR 98-11, Iowa State University, 1998.
  * S. M. LaValle and J. J. Kuffner, "Randomized Kinodynamic Planning," IJRR 2001.
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

from pir.core.random import make_rng
from pir.core.types import Failure, StepResult, Trace


@dataclass(frozen=True)
class Disc:
    center: tuple[float, float]
    radius: float


@dataclass
class RRTConfig:
    world_size: float = 10.0
    start: tuple[float, float] = (1.0, 1.0)
    goal: tuple[float, float] = (9.0, 9.0)
    goal_radius: float = 0.4
    robot_radius: float = 0.2
    speed: float = 0.3
    sensor_range: float = 2.2
    max_steps: int = 200
    # Obstacles the planner knows from the start, flanking the start-goal diagonal.
    known: tuple[Disc, ...] = (Disc((3.3, 4.0), 1.0), Disc((6.7, 6.0), 1.0))
    # An obstacle only sensed within sensor_range, sitting in the gap the first
    # plan usually threads through. Because RRT samples randomly, the first tree
    # occasionally routes around it and no replan is needed; usually it does not,
    # and the robot discovers the disc and replans. Either way it reaches the goal.
    hidden: Disc = Disc((5.0, 5.0), 1.4)


def _segment_hits_disc(a: np.ndarray, b: np.ndarray, disc: Disc, clearance: float) -> bool:
    center = np.asarray(disc.center, dtype=float)
    ab = b - a
    length_sq = float(ab @ ab)
    if length_sq < 1e-12:
        return float(np.linalg.norm(a - center)) <= disc.radius + clearance
    t = float(np.clip((center - a) @ ab / length_sq, 0.0, 1.0))
    closest = a + t * ab
    return float(np.linalg.norm(closest - center)) <= disc.radius + clearance


def plan_rrt(
    start: np.ndarray,
    goal: np.ndarray,
    obstacles: list[Disc],
    rng: np.random.Generator,
    *,
    world_size: float,
    clearance: float,
    step: float = 0.6,
    goal_sample_rate: float = 0.1,
    max_iter: int = 2000,
    goal_tol: float = 0.4,
) -> list[np.ndarray] | None:
    """Grow an RRT from start to goal; return a waypoint path or None."""

    nodes = [np.asarray(start, dtype=float)]
    parents = [-1]
    for _ in range(max_iter):
        if rng.random() < goal_sample_rate:
            sample = np.asarray(goal, dtype=float)
        else:
            sample = rng.uniform(0.0, world_size, size=2)
        # Nearest existing node, then a step toward the sample.
        dists = [float(np.linalg.norm(sample - n)) for n in nodes]
        i = int(np.argmin(dists))
        nearest = nodes[i]
        direction = sample - nearest
        norm = float(np.linalg.norm(direction))
        if norm < 1e-9:
            continue
        new_node = nearest + direction / norm * min(step, norm)
        if any(_segment_hits_disc(nearest, new_node, o, clearance) for o in obstacles):
            continue
        nodes.append(new_node)
        parents.append(i)
        if float(np.linalg.norm(new_node - goal)) <= goal_tol:
            # Reconstruct the path from this node back to the root.
            path = [np.asarray(goal, dtype=float)]
            j = len(nodes) - 1
            while j != -1:
                path.append(nodes[j])
                j = parents[j]
            path.reverse()
            return path
    return None


class RRTWorld:
    """Continuous 2D world with known and hidden discs; the robot senses locally."""

    def __init__(self, *, seed: int | None = 0, config: RRTConfig | None = None) -> None:
        self.config = config or RRTConfig()
        self.seed = seed
        self.reset(seed=seed)

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            self.seed = seed
        self.rng = make_rng(self.seed)
        self.pos = np.asarray(self.config.start, dtype=float).copy()
        self.goal = np.asarray(self.config.goal, dtype=float)
        self.time = 0
        self._hidden_found = False
        return self.observe()

    def observe(self) -> dict[str, Any]:
        sensed = list(self.config.known)
        hidden = self.config.hidden
        if (
            float(np.linalg.norm(self.pos - np.asarray(hidden.center))) - hidden.radius
            <= self.config.sensor_range
        ):
            sensed.append(hidden)
            self._hidden_found = True
        return {
            "time": self.time,
            "position": self.pos.copy(),
            "goal": self.goal.copy(),
            "sensed_obstacles": sensed,
            "hidden_found": self._hidden_found,
            "distance_to_goal": float(np.linalg.norm(self.goal - self.pos)),
        }

    def step(self, action: dict[str, Any]) -> StepResult:
        self.time += 1
        velocity = np.asarray(action.get("velocity", np.zeros(2)), dtype=float)
        self.pos = np.clip(self.pos + velocity, 0.0, self.config.world_size)
        info: dict[str, Any] = {
            "time": self.time,
            "position": self.pos.copy(),
            "distance_to_goal": float(np.linalg.norm(self.goal - self.pos)),
            "replans": int(action.get("replans", 0)),
            "success": False,
        }
        if action.get("path_blocked"):
            info["failure"] = Failure("path_blocked", "a sensed obstacle invalidated the path", True)
        if action.get("rrt_unreachable"):
            info["failure"] = Failure("rrt_unreachable", "RRT found no path", False)
            return StepResult(self.observe(), -1.0, True, info)

        all_obstacles = list(self.config.known) + [self.config.hidden]
        for disc in all_obstacles:
            if float(np.linalg.norm(self.pos - np.asarray(disc.center))) < disc.radius:
                info["failure"] = Failure("collision", "robot entered an obstacle", False)
                return StepResult(self.observe(), -1.0, True, info)

        if info["distance_to_goal"] <= self.config.goal_radius:
            info["success"] = True
            return StepResult(self.observe(), 1.0, True, info)
        if self.time >= self.config.max_steps:
            info["failure"] = Failure("timeout", "ran out of steps", False)
            return StepResult(self.observe(), -0.5, True, info)
        return StepResult(self.observe(), -0.01, False, info)


class RRTReplanAgent:
    """Follows an RRT path; replans when a sensed obstacle blocks the route."""

    def __init__(self, *, config: RRTConfig | None = None, seed: int | None = 0) -> None:
        self.config = config or RRTConfig()
        self.rng = make_rng(None if seed is None else seed + 104729)
        self.reset()

    def reset(self) -> None:
        self.path: list[np.ndarray] | None = None
        self.waypoint = 0
        self.replans = 0
        self.known_count = -1

    def _plan(self, pos: np.ndarray, obstacles: list[Disc]) -> bool:
        path = plan_rrt(
            pos, self.config.goal, obstacles, self.rng,
            world_size=self.config.world_size, clearance=self.config.robot_radius,
        )
        if path is None:
            self.path = None
            return False
        self.path = path
        self.waypoint = 1 if len(path) > 1 else 0
        return True

    def _path_blocked(self, pos: np.ndarray, obstacles: list[Disc]) -> bool:
        if not self.path:
            return True
        pts = [pos] + self.path[self.waypoint:]
        for a, b in zip(pts[:-1], pts[1:]):
            if any(_segment_hits_disc(a, b, o, self.config.robot_radius) for o in obstacles):
                return True
        return False

    def act(self, obs: dict[str, Any]) -> dict[str, Any]:
        pos = np.asarray(obs["position"], dtype=float)
        obstacles = list(obs["sensed_obstacles"])
        blocked = False

        if self.path is None or self._path_blocked(pos, obstacles):
            blocked = self.path is not None  # a real invalidation, not the first plan
            if blocked:
                self.replans += 1
            ok = self._plan(pos, obstacles)
            if not ok:
                return {"velocity": np.zeros(2), "rrt_unreachable": True, "replans": self.replans}

        target = self.path[self.waypoint]
        if float(np.linalg.norm(target - pos)) < self.config.speed and self.waypoint < len(self.path) - 1:
            self.waypoint += 1
            target = self.path[self.waypoint]
        direction = target - pos
        norm = float(np.linalg.norm(direction))
        velocity = np.zeros(2) if norm < 1e-9 else direction / norm * min(self.config.speed, norm)
        return {"velocity": velocity, "path_blocked": blocked, "replans": self.replans}

    def update(self, obs: dict[str, Any], reward: float, info: dict[str, Any]) -> None:
        info["replans"] = self.replans
        info["waypoint"] = self.waypoint
        info["path_len"] = 0 if self.path is None else len(self.path)


def run(seed: int = 0, render: bool = True, max_steps: int | None = None) -> Trace:
    config = RRTConfig()
    if max_steps is not None:
        config.max_steps = max_steps
    world = RRTWorld(seed=seed, config=config)
    agent = RRTReplanAgent(config=config, seed=seed)
    obs = world.reset(seed=seed)
    agent.reset()
    trace = Trace()

    for _ in range(config.max_steps):
        action = agent.act(obs)
        result = world.step(action)
        obs, reward, done, info = result.as_tuple()
        agent.update(obs, reward, info)
        trace.append(obs, action, reward, info)

        if render:
            _render(world, agent, info)

        if done:
            break

    return trace


def _render(world: RRTWorld, agent: RRTReplanAgent, info: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    if not hasattr(world, "_fig") or world._fig is None:
        plt.ion()
        world._fig, world._ax = plt.subplots(figsize=(5, 5))
    ax = world._ax
    ax.clear()
    ax.set_title("RRT replanning: plan, discover, replan, arrive")
    ax.set_xlim(0, world.config.world_size)
    ax.set_ylim(0, world.config.world_size)
    ax.set_aspect("equal", adjustable="box")
    for disc in world.config.known:
        ax.add_patch(Circle(disc.center, disc.radius, color="0.3", alpha=0.6))
    h = world.config.hidden
    ax.add_patch(Circle(h.center, h.radius, color="tab:orange" if world._hidden_found else "0.85",
                        alpha=0.6 if world._hidden_found else 0.3))
    if agent.path:
        xs = [p[0] for p in agent.path]
        ys = [p[1] for p in agent.path]
        ax.plot(xs, ys, "-", color="tab:green", linewidth=1, alpha=0.8)
    ax.plot(*world.goal, marker="*", color="tab:green", markersize=16)
    ax.plot(*world.pos, marker="o", color="tab:blue", markersize=9)
    ax.text(0.02, 0.97, f"d={info['distance_to_goal']:.2f} replans={info.get('replans', 0)}",
            transform=ax.transAxes, va="top")
    world._fig.canvas.draw_idle()
    plt.pause(0.03)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    trace = run(seed=args.seed, render=not args.no_render, max_steps=args.max_steps)
    final = trace.infos[-1]
    failures = sorted({f.kind for f in trace.failures()})
    print(
        f"reached={final.get('success', False)} steps={len(trace.actions)} "
        f"replans={final.get('replans', 0)} "
        f"final_dist={final.get('distance_to_goal', float('nan')):.2f} failures={failures}"
    )

    if not args.no_render:
        import matplotlib.pyplot as plt

        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
