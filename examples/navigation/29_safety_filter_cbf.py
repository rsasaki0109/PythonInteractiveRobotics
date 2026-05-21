"""Safety filter that projects an unsafe nominal velocity onto a CBF constraint.

The nominal policy is a naive go-to-goal controller that does not know about
obstacles. A separate runtime safety filter checks a control-barrier-function
condition `dh/dt >= -alpha * h` for each obstacle. When the nominal velocity
would violate that condition, the filter projects it onto the closest safe
half-space; otherwise it passes the nominal velocity through.

This is intentionally a runtime-assurance pattern: the agent never replans
or learns. It is the safety filter, separate from the policy, that keeps
the robot away from the obstacles. Compare to
`02_reactive_obstacle_avoidance.py`, where the policy itself avoids
obstacles, and to `08_interactive_mpc.py`, where avoidance is folded into
the cost of an MPC controller.

Success: robot reaches the goal radius before max_steps.
Failure: safety_filter_stuck (recoverable - the filter could not find a
non-zero safe velocity, so the agent waited a step), collision (terminal
- the safety filter let the robot reach a barrier-violating state), or
timeout (terminal).
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


@dataclass(frozen=True)
class CBFConfig:
    dt: float = 1.0
    speed: float = 0.40
    goal_radius: float = 0.30
    robot_radius: float = 0.18
    safety_margin: float = 0.20
    alpha: float = 0.55
    world_min: float = 0.0
    world_max: float = 10.0
    stuck_speed: float = 0.04


@dataclass(frozen=True)
class StaticObstacle:
    position: tuple[float, float]
    radius: float


DEFAULT_OBSTACLES: tuple[StaticObstacle, ...] = (
    StaticObstacle(position=(3.6, 4.2), radius=0.85),
    StaticObstacle(position=(6.2, 5.6), radius=0.95),
)


class SafetyFilterWorld:
    """Continuous 2D world with one robot, one goal, and static circular obstacles."""

    def __init__(
        self,
        *,
        seed: int = 0,
        max_steps: int = 120,
        start: tuple[float, float] = (1.0, 1.0),
        goal: tuple[float, float] = (8.6, 8.4),
        obstacles: tuple[StaticObstacle, ...] = DEFAULT_OBSTACLES,
    ) -> None:
        self.config = CBFConfig()
        self.max_steps = max_steps
        self.seed = seed
        self._start = np.asarray(start, dtype=float)
        self._goal = np.asarray(goal, dtype=float)
        self.obstacles = obstacles
        self.robot = self._start.copy()
        self.step_count = 0
        self.trajectory: list[np.ndarray] = [self.robot.copy()]
        self._fig: Any | None = None
        self._ax: Any | None = None

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            self.seed = seed
        self.step_count = 0
        self.robot = self._start.copy()
        self.trajectory = [self.robot.copy()]
        return self.observe()

    def observe(self) -> dict[str, Any]:
        return {
            "robot": self.robot.copy(),
            "goal": self._goal.copy(),
            "obstacles": tuple(
                {"position": np.asarray(o.position, dtype=float), "radius": o.radius}
                for o in self.obstacles
            ),
            "step": self.step_count,
        }

    def step(self, control: np.ndarray) -> StepResult:
        cfg = self.config
        self.step_count += 1
        control = np.asarray(control, dtype=float)
        speed = float(np.linalg.norm(control))
        if speed > cfg.speed:
            control = control / speed * cfg.speed

        self.robot = np.clip(
            self.robot + control * cfg.dt,
            cfg.world_min,
            cfg.world_max,
        )
        self.trajectory.append(self.robot.copy())

        goal_distance = float(np.linalg.norm(self._goal - self.robot))
        min_clearance, _ = self._closest_obstacle()
        collision_distance = cfg.robot_radius
        success = goal_distance <= cfg.goal_radius
        collision = min_clearance <= collision_distance
        timeout = self.step_count >= self.max_steps
        done = success or collision or timeout

        reward = -goal_distance - 0.1
        info: dict[str, Any] = {
            "success": success,
            "goal_distance": goal_distance,
            "clearance": min_clearance,
        }
        if collision:
            info["failure"] = Failure(
                kind="collision",
                message="Robot penetrated an obstacle radius despite the safety filter.",
                recoverable=False,
            )
        elif timeout and not success:
            info["failure"] = Failure(
                kind="timeout",
                message="Safety-filtered controller did not reach the goal before max_steps.",
                recoverable=False,
            )
        return StepResult(self.observe(), reward, done, info)

    def _closest_obstacle(self) -> tuple[float, StaticObstacle | None]:
        cfg = self.config
        best: tuple[float, StaticObstacle | None] = (float("inf"), None)
        for obs in self.obstacles:
            center = np.asarray(obs.position, dtype=float)
            clearance = float(np.linalg.norm(self.robot - center)) - obs.radius
            if clearance < best[0]:
                best = (clearance, obs)
        return best

    def render(self, agent: "SafetyFilterAgent", info: dict[str, Any]) -> None:
        import matplotlib.pyplot as plt

        cfg = self.config
        if self._fig is None or self._ax is None:
            plt.ion()
            self._fig, self._ax = plt.subplots(figsize=(6, 6))
        ax = self._ax
        ax.clear()
        ax.set_xlim(cfg.world_min, cfg.world_max)
        ax.set_ylim(cfg.world_min, cfg.world_max)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        ax.set_title("Safety filter (CBF) on a nominal go-to-goal policy")

        for obs in self.obstacles:
            barrier = plt.Circle(
                obs.position,
                obs.radius + cfg.robot_radius + cfg.safety_margin,
                color="tab:red",
                fill=False,
                linestyle=":",
                alpha=0.55,
            )
            disc = plt.Circle(obs.position, obs.radius, color="tab:red", alpha=0.55)
            ax.add_patch(barrier)
            ax.add_patch(disc)

        goal = plt.Circle(self._goal, cfg.goal_radius, color="tab:green", alpha=0.25)
        ax.add_patch(goal)
        ax.scatter([self._goal[0]], [self._goal[1]], marker="*", s=160, color="tab:green")

        trajectory = np.asarray(self.trajectory)
        ax.plot(trajectory[:, 0], trajectory[:, 1], color="tab:blue", alpha=0.65, linewidth=1.5)

        robot = plt.Circle(self.robot, cfg.robot_radius, color="tab:blue")
        ax.add_patch(robot)

        u_nom = agent.last_u_nominal
        u_safe = agent.last_u_safe
        if u_nom is not None:
            ax.arrow(
                self.robot[0],
                self.robot[1],
                u_nom[0] * 2.5,
                u_nom[1] * 2.5,
                head_width=0.18,
                color="tab:orange",
                alpha=0.85,
                length_includes_head=True,
                label="u_nominal",
            )
        if u_safe is not None:
            ax.arrow(
                self.robot[0],
                self.robot[1],
                u_safe[0] * 2.5,
                u_safe[1] * 2.5,
                head_width=0.18,
                color="tab:blue",
                alpha=0.95,
                length_includes_head=True,
                label="u_safe",
            )

        status = (
            f"step={self.step_count} state={agent.state}\n"
            f"barrier_h_min={agent.last_barrier_h_min:.2f}"
            f" filter_active={agent.filter_active}\n"
            f"filter_active_count={agent.filter_active_count}"
            f" stuck_count={agent.stuck_count}"
        )
        if "failure" in info:
            status += f"\nfailure={info['failure'].kind}"
        ax.text(0.02, 0.98, status, transform=ax.transAxes, va="top", fontsize=9)
        ax.legend(loc="lower right", fontsize=8)
        self._fig.canvas.draw_idle()
        plt.pause(0.001)


class SafetyFilterAgent:
    """Naive go-to-goal nominal policy with a CBF-style runtime safety filter."""

    def __init__(self, config: CBFConfig) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.state = "go_to_goal"
        self.filter_active = False
        self.filter_active_count = 0
        self.stuck_count = 0
        self.last_u_nominal: np.ndarray | None = None
        self.last_u_safe: np.ndarray | None = None
        self.last_barrier_h_min = float("inf")
        self.closest_approach = float("inf")

    def act(self, obs: dict[str, Any]) -> np.ndarray:
        cfg = self.config
        robot = np.asarray(obs["robot"], dtype=float)
        goal = np.asarray(obs["goal"], dtype=float)
        obstacles = obs["obstacles"]

        u_nominal = self._nominal(robot, goal)
        u_safe, was_filtered, barrier_h_min = self._apply_filter(
            robot=robot,
            u_nominal=u_nominal,
            obstacles=obstacles,
        )

        speed = float(np.linalg.norm(u_safe))
        if speed > cfg.speed:
            u_safe = u_safe / speed * cfg.speed
            speed = cfg.speed

        if was_filtered:
            self.filter_active = True
            self.filter_active_count += 1
        else:
            self.filter_active = False

        if speed < cfg.stuck_speed:
            self.stuck_count += 1
            self.state = "stuck"
        elif was_filtered:
            self.state = "filter_active"
        else:
            self.state = "go_to_goal"

        self.last_u_nominal = u_nominal
        self.last_u_safe = u_safe
        self.last_barrier_h_min = barrier_h_min
        if barrier_h_min < self.closest_approach:
            self.closest_approach = barrier_h_min
        return u_safe

    def update(self, obs: dict[str, Any], reward: float, info: dict[str, Any]) -> None:
        del obs, reward
        if (
            self.state == "stuck"
            and info.get("failure") is None
            and not info.get("success", False)
        ):
            info["failure"] = Failure(
                kind="safety_filter_stuck",
                message="Filter clipped nominal velocity below the stuck threshold.",
                recoverable=True,
            )

    def info(self) -> dict[str, Any]:
        return {
            "u_nominal": (
                self.last_u_nominal.tolist()
                if self.last_u_nominal is not None
                else None
            ),
            "u_safe": (
                self.last_u_safe.tolist() if self.last_u_safe is not None else None
            ),
            "barrier_h_min": float(self.last_barrier_h_min),
            "closest_approach": float(self.closest_approach),
            "filter_active": bool(self.filter_active),
            "filter_active_count": int(self.filter_active_count),
            "stuck_count": int(self.stuck_count),
            "agent_state": self.state,
        }

    def _nominal(self, robot: np.ndarray, goal: np.ndarray) -> np.ndarray:
        delta = goal - robot
        distance = float(np.linalg.norm(delta))
        if distance < 1e-9:
            return np.zeros(2, dtype=float)
        direction = delta / distance
        target_speed = min(self.config.speed, distance)
        return direction * target_speed

    def _apply_filter(
        self,
        *,
        robot: np.ndarray,
        u_nominal: np.ndarray,
        obstacles: tuple[dict[str, Any], ...],
    ) -> tuple[np.ndarray, bool, float]:
        """Project u_nominal onto the half-spaces dh_i/dt >= -alpha * h_i.

        Each obstacle i contributes a linear constraint
            e_i^T u >= -alpha * h_i
        where h_i = |robot - center_i| - (radius_i + robot_radius + margin)
        and  e_i = (robot - center_i) / |robot - center_i| .
        Projection is iterative: pick the most-violated constraint, project
        onto its boundary, repeat until all are satisfied or the iteration
        budget is exhausted.
        """

        cfg = self.config
        u = u_nominal.copy()
        was_filtered = False
        barrier_h_min = float("inf")

        constraints: list[tuple[np.ndarray, float]] = []
        for obstacle in obstacles:
            center = np.asarray(obstacle["position"], dtype=float)
            radius = float(obstacle["radius"])
            offset = robot - center
            distance = float(np.linalg.norm(offset))
            if distance < 1e-9:
                offset = np.array([1.0, 0.0], dtype=float)
                distance = 1.0
            e_i = offset / distance
            h_i = distance - (radius + cfg.robot_radius + cfg.safety_margin)
            barrier_h_min = min(barrier_h_min, h_i)
            constraints.append((e_i, h_i))

        for _ in range(len(constraints) + 1):
            most_violated: tuple[float, np.ndarray, float] | None = None
            for e_i, h_i in constraints:
                rhs = -cfg.alpha * h_i
                violation = rhs - float(np.dot(e_i, u))
                if violation > 1e-6 and (most_violated is None or violation > most_violated[0]):
                    most_violated = (violation, e_i, rhs)
            if most_violated is None:
                break
            _, e_i, rhs = most_violated
            u = u + (rhs - float(np.dot(e_i, u))) * e_i
            was_filtered = True

        return u, was_filtered, barrier_h_min


def run(seed: int = 0, render: bool = True, max_steps: int = 120) -> Trace:
    env = SafetyFilterWorld(seed=seed, max_steps=max_steps)
    agent = SafetyFilterAgent(env.config)
    obs = env.reset(seed=seed)
    agent.reset()
    trace = Trace()

    for _ in range(max_steps):
        action = agent.act(obs)
        obs, reward, done, info = env.step(action).as_tuple()
        agent.update(obs, reward, info)
        info.update(agent.info())
        if done and info.get("success"):
            info["agent_state"] = "arrived"
        trace.append(obs, action, reward, info)

        if render:
            env.render(agent=agent, info=info)
        if done:
            break

    return trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    trace = run(seed=args.seed, render=not args.no_render, max_steps=args.max_steps)
    final_info = trace.infos[-1] if trace.infos else {}
    failures = [failure.kind for failure in trace.failures()]
    print(
        f"success={final_info.get('success', False)} "
        f"steps={len(trace.actions)} "
        f"filter_active_count={final_info.get('filter_active_count', 0)} "
        f"stuck_count={final_info.get('stuck_count', 0)} "
        f"closest_approach={final_info.get('closest_approach', 0.0):.3f} "
        f"failures={failures}"
    )

    if not args.no_render:
        import matplotlib.pyplot as plt

        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
