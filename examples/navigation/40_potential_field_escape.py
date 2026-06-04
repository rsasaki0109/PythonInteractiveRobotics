"""Drive an artificial potential field to the goal, get trapped, and escape the minimum.

Artificial potential fields (Khatib, 1986) are the one-line local planner: pull
the robot down the gradient of an attractive potential toward the goal while
pushing it up a repulsive potential away from obstacles, and step along the net
force. It is elegant and it has one famous failure — a **local minimum**, a spot
where attraction and repulsion cancel but the goal is not reached, so the robot
stalls forever in front of an obstacle.

This example is that failure and its classic recovery in one loop. An obstacle
sits squarely on the straight line from start to goal, so plain gradient descent
walks the robot into the stagnation point in front of it and stops. The world
notices the lack of progress and reports a ``local_minimum`` failure; the agent
reacts the way real potential-field planners do — it switches to **boundary
following**, sliding tangentially around the obstacle (perpendicular to the
repulsive gradient) until the goal-ward path reopens, then resumes gradient
descent and reaches the goal.

Run with ``--no-escape`` to disable the recovery and watch plain potential-field
descent stall at the minimum until timeout.

Success: the robot reaches the goal radius.
Failure: local_minimum (recoverable - net force vanished before the goal; the
trigger for boundary following), collision (terminal), timeout (terminal).

References:
  * O. Khatib, "Real-Time Obstacle Avoidance for Manipulators and Mobile Robots,"
    ICRA 1985 / IJRR 1986 - the attractive/repulsive potential formulation.
  * Y. Koren and J. Borenstein, "Potential Field Methods and Their Inherent
    Limitations for Mobile Robot Navigation," ICRA 1991 - the local-minimum trap.
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
class Obstacle:
    center: tuple[float, float]
    radius: float


@dataclass
class APFConfig:
    world_size: float = 10.0
    start: tuple[float, float] = (1.0, 5.0)
    goal: tuple[float, float] = (9.0, 5.0)
    goal_radius: float = 0.35
    step_size: float = 0.25
    k_att: float = 0.12
    k_rep: float = 1.4
    influence: float = 2.6  # repulsion reaches this far past the obstacle surface
    improve_eps: float = 0.05  # net closing of the goal distance that counts as progress
    stuck_patience: int = 6  # steps without net progress before local_minimum fires
    max_steps: int = 120
    obstacles: tuple[Obstacle, ...] = (Obstacle(center=(5.0, 5.0), radius=1.1),)


class PotentialFieldWorld:
    """Continuous 2D world; reports a local_minimum failure when progress stalls."""

    def __init__(self, *, seed: int | None = 0, config: APFConfig | None = None) -> None:
        self.config = config or APFConfig()
        self.seed = seed
        self.reset(seed=seed)

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            self.seed = seed
        self.rng = make_rng(self.seed)
        self.pos = np.asarray(self.config.start, dtype=float).copy()
        self.goal = np.asarray(self.config.goal, dtype=float)
        self.time = 0
        self._stalled = 0
        self._best_dist = float(np.linalg.norm(self.goal - self.pos))
        return self.observe()

    def observe(self) -> dict[str, Any]:
        return {
            "time": self.time,
            "position": self.pos.copy(),
            "goal": self.goal.copy(),
            "obstacles": [(np.asarray(o.center, dtype=float), o.radius) for o in self.config.obstacles],
            "distance_to_goal": float(np.linalg.norm(self.goal - self.pos)),
        }

    def step(self, action: dict[str, Any]) -> StepResult:
        self.time += 1
        velocity = np.asarray(action.get("velocity", np.zeros(2)), dtype=float)
        before = self.pos.copy()
        self.pos = np.clip(self.pos + velocity, 0.0, self.config.world_size)
        progress = float(np.linalg.norm(self.pos - before))

        info: dict[str, Any] = {
            "time": self.time,
            "position": self.pos.copy(),
            "distance_to_goal": float(np.linalg.norm(self.goal - self.pos)),
            "progress": progress,
            "success": False,
        }

        for center, radius in [(np.asarray(o.center, dtype=float), o.radius) for o in self.config.obstacles]:
            if np.linalg.norm(self.pos - center) < radius:
                info["failure"] = Failure("collision", "robot entered an obstacle", False)
                return StepResult(self.observe(), -1.0, True, info)

        if info["distance_to_goal"] <= self.config.goal_radius:
            info["success"] = True
            return StepResult(self.observe(), 1.0, True, info)

        # Stalled away from the goal: the potential-field local minimum. Detected
        # by a lack of *net* progress (the robot oscillates in place around the
        # stagnation point), not by a single small step.
        if info["distance_to_goal"] < self._best_dist - self.config.improve_eps:
            self._best_dist = info["distance_to_goal"]
            self._stalled = 0
        else:
            self._stalled += 1
            if self._stalled >= self.config.stuck_patience:
                info["failure"] = Failure(
                    "local_minimum", "net force vanished before reaching the goal", True
                )
                self._stalled = 0  # give the recovery a fresh window before re-firing

        if self.time >= self.config.max_steps:
            info["failure"] = Failure("timeout", "ran out of steps", False)
            return StepResult(self.observe(), -0.5, True, info)

        return StepResult(self.observe(), -0.01, False, info)


class PotentialFieldAgent:
    """Gradient descent on the potential; boundary-follows out of a local minimum."""

    def __init__(self, *, config: APFConfig | None = None, escape: bool = True) -> None:
        self.config = config or APFConfig()
        self.escape_enabled = escape
        self.reset()

    def reset(self) -> None:
        self.escape_steps_left = 0
        self.escape_sign = 1.0

    def act(self, obs: dict[str, Any]) -> dict[str, Any]:
        pos = np.asarray(obs["position"], dtype=float)
        goal = np.asarray(obs["goal"], dtype=float)
        attractive = self.config.k_att * (goal - pos)
        repulsive, nearest_gradient = self._repulsive_force(pos, obs["obstacles"])
        net = attractive + repulsive

        if self.escape_steps_left > 0 and nearest_gradient is not None:
            # Boundary following: move tangent to the repulsive gradient (around the
            # obstacle) with a slight goal-ward bias, instead of down the net force.
            tangent = np.array([-nearest_gradient[1], nearest_gradient[0]]) * self.escape_sign
            direction = tangent + 0.35 * (goal - pos) / (np.linalg.norm(goal - pos) + 1e-9)
            self.escape_steps_left -= 1
            velocity = self._cap(direction)
        else:
            velocity = self._cap(net)

        self.last_net_norm = float(np.linalg.norm(net))
        return {"velocity": velocity}

    def update(self, obs: dict[str, Any], reward: float, info: dict[str, Any]) -> None:
        failure = info.get("failure")
        if (
            self.escape_enabled
            and isinstance(failure, Failure)
            and failure.kind == "local_minimum"
            and self.escape_steps_left == 0
        ):
            # Enter boundary-following for enough steps to clear the obstacle, and
            # commit to the side that points more toward the goal.
            pos = np.asarray(info["position"], dtype=float)
            goal = np.asarray(obs["goal"], dtype=float)
            _, gradient = self._repulsive_force(pos, obs["obstacles"])
            self.escape_steps_left = 14
            if gradient is not None:
                tangent = np.array([-gradient[1], gradient[0]])
                self.escape_sign = 1.0 if np.dot(tangent, goal - pos) >= 0 else -1.0
        info["escaping"] = self.escape_steps_left > 0
        info["net_force"] = getattr(self, "last_net_norm", 0.0)

    def _repulsive_force(self, pos: np.ndarray, obstacles: Any) -> tuple[np.ndarray, np.ndarray | None]:
        total = np.zeros(2)
        nearest_gradient: np.ndarray | None = None
        nearest_d = np.inf
        for center, radius in obstacles:
            center = np.asarray(center, dtype=float)
            offset = pos - center
            surface_d = float(np.linalg.norm(offset)) - radius
            if surface_d < self.config.influence and surface_d > 1e-6:
                direction = offset / (np.linalg.norm(offset) + 1e-9)
                magnitude = self.config.k_rep * (1.0 / surface_d - 1.0 / self.config.influence) / (surface_d**2)
                total += magnitude * direction
                if surface_d < nearest_d:
                    nearest_d = surface_d
                    nearest_gradient = direction
        return total, nearest_gradient

    def _cap(self, vector: np.ndarray) -> np.ndarray:
        """Move along the force, but no faster than step_size — so the robot truly
        slows to a halt as the net force vanishes at a local minimum."""
        norm = float(np.linalg.norm(vector))
        if norm < 1e-9:
            return np.zeros(2)
        return vector * (min(norm, self.config.step_size) / norm)


def run(
    seed: int = 0,
    render: bool = True,
    max_steps: int | None = None,
    escape: bool = True,
) -> Trace:
    config = APFConfig()
    if max_steps is not None:
        config.max_steps = max_steps
    world = PotentialFieldWorld(seed=seed, config=config)
    agent = PotentialFieldAgent(config=config, escape=escape)
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
            _render(world, info)

        if done:
            break

    return trace


def _render(world: PotentialFieldWorld, info: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    if not hasattr(world, "_fig") or world._fig is None:
        plt.ion()
        world._fig, world._ax = plt.subplots(figsize=(5, 5))
    ax = world._ax
    ax.clear()
    ax.set_title("Potential field: descend, trap, escape, reach goal")
    ax.set_xlim(0, world.config.world_size)
    ax.set_ylim(0, world.config.world_size)
    ax.set_aspect("equal", adjustable="box")
    for o in world.config.obstacles:
        ax.add_patch(Circle(o.center, o.radius, color="0.3", alpha=0.5))
        ax.add_patch(Circle(o.center, o.radius + world.config.influence, color="0.7", alpha=0.12))
    ax.plot(*world.goal, marker="*", color="tab:green", markersize=16)
    ax.plot(*world.pos, marker="o", color="tab:red" if info.get("escaping") else "tab:blue", markersize=9)
    ax.text(0.02, 0.97, f"d={info['distance_to_goal']:.2f} |F|={info.get('net_force', 0):.3f}"
            f"{' ESCAPING' if info.get('escaping') else ''}", transform=ax.transAxes, va="top")
    world._fig.canvas.draw_idle()
    plt.pause(0.03)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--no-escape", action="store_true", help="plain APF (stalls at the minimum)")
    args = parser.parse_args()

    trace = run(
        seed=args.seed,
        render=not args.no_render,
        max_steps=args.max_steps,
        escape=not args.no_escape,
    )
    final = trace.infos[-1]
    failures = sorted({f.kind for f in trace.failures()})
    print(
        f"reached={final.get('success', False)} steps={len(trace.actions)} "
        f"final_dist={final.get('distance_to_goal', float('nan')):.2f} "
        f"failures={failures} escape={not args.no_escape}"
    )

    if not args.no_render:
        import matplotlib.pyplot as plt

        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
