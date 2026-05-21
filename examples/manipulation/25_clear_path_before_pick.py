"""Clear an obstacle before picking the target: precondition failure recovery.

The target is on the tabletop in plain view, but another object sits on the
straight line from the gripper to the target. The agent's first pick attempt
returns a precondition failure. Recovery is to pick the obstacle, place it in
a known clear zone, and then retry the original pick.

This differs from `06_push_then_grasp.py`, which modifies the target's own
position so a previously blocked grasp opens up. Here the target stays put -
the recovery is to move a separate object out of the way.
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

from pir.core.random import make_rng
from pir.core.types import Failure, StepResult, Trace


class ClearPathBeforePickWorld:
    """Tabletop with a target, a blocking obstacle, and a known clear zone."""

    def __init__(
        self,
        *,
        seed: int | None = 0,
        max_steps: int = 15,
        target_position: tuple[float, float] = (0.78, 0.50),
        obstacle_position: tuple[float, float] = (0.50, 0.50),
        gripper_start: tuple[float, float] = (0.12, 0.50),
        clear_zone_center: tuple[float, float] = (0.50, 0.85),
        clear_zone_radius: float = 0.10,
        obstacle_radius: float = 0.06,
        target_radius: float = 0.05,
    ) -> None:
        self.seed = seed
        self.size = 1.0
        self.target_position_init = np.asarray(target_position, dtype=float)
        self.obstacle_position_init = np.asarray(obstacle_position, dtype=float)
        self.gripper_start = np.asarray(gripper_start, dtype=float)
        self.clear_zone_center = np.asarray(clear_zone_center, dtype=float)
        self.clear_zone_radius = clear_zone_radius
        self.obstacle_radius = obstacle_radius
        self.target_radius = target_radius
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
        self.gripper = self.gripper_start.copy()
        self.target_position = self.target_position_init.copy()
        self.obstacle_position = self.obstacle_position_init.copy()
        self.time = 0
        self.held: str | None = None
        self.target_picked = False
        self.obstacle_moved = False
        self.last_action_type: str | None = None
        self.last_place_position: np.ndarray | None = None
        self.trajectory: list[np.ndarray] = [self.gripper.copy()]
        return self.observe()

    def observe(self) -> dict[str, Any]:
        return {
            "time": self.time,
            "gripper": self.gripper.copy(),
            "target_position": None if self.target_picked else self.target_position.copy(),
            "obstacle_position": self.obstacle_position.copy(),
            "held": self.held,
            "target_picked": self.target_picked,
            "obstacle_moved": self.obstacle_moved,
            "clear_zone_center": self.clear_zone_center.copy(),
            "clear_zone_radius": self.clear_zone_radius,
            "target_radius": self.target_radius,
            "obstacle_radius": self.obstacle_radius,
        }

    def step(self, action: dict[str, Any]) -> StepResult:
        self.time += 1
        action_type = action.get("action_type") if isinstance(action, dict) else None
        info: dict[str, Any] = {"time": self.time, "action_type": action_type, "success": False}
        self.last_action_type = action_type

        if action_type == "pick":
            return self._handle_pick(action, info)
        if action_type == "place":
            return self._handle_place(action, info)
        if action_type == "wait":
            return StepResult(self.observe(), -0.02, False, info)

        info["failure"] = Failure("invalid_action", f"unknown action_type: {action_type}", True)
        return StepResult(self.observe(), -0.05, False, info)

    def _handle_pick(self, action: dict[str, Any], info: dict[str, Any]) -> StepResult:
        target_name = action.get("target")
        info["pick_target"] = target_name

        if self.held is not None:
            info["failure"] = Failure(
                "already_holding",
                f"already holding {self.held}",
                True,
            )
            return StepResult(self.observe(), -0.05, False, info)

        if target_name == "target":
            if not self.obstacle_moved:
                info["failure"] = Failure(
                    "precondition_blocked",
                    "obstacle on gripper path to target",
                    True,
                )
                return StepResult(self.observe(), -0.10, False, info)
            self.gripper = self.target_position.copy()
            self.trajectory.append(self.gripper.copy())
            self.held = "target"
            self.target_picked = True
            info["success"] = True
            info["pick_position"] = self.target_position.tolist()
            return StepResult(self.observe(), 1.0, True, info)

        if target_name == "obstacle":
            if self.obstacle_moved:
                info["failure"] = Failure(
                    "obstacle_already_moved",
                    "obstacle is already in the clear zone",
                    True,
                )
                return StepResult(self.observe(), -0.05, False, info)
            self.gripper = self.obstacle_position.copy()
            self.trajectory.append(self.gripper.copy())
            self.held = "obstacle"
            info["pick_position"] = self.obstacle_position.tolist()
            return StepResult(self.observe(), -0.02, False, info)

        info["failure"] = Failure(
            "invalid_target",
            f"unknown pick target: {target_name}",
            True,
        )
        return StepResult(self.observe(), -0.05, False, info)

    def _handle_place(self, action: dict[str, Any], info: dict[str, Any]) -> StepResult:
        if self.held is None:
            info["failure"] = Failure(
                "nothing_held",
                "place attempted with empty gripper",
                True,
            )
            return StepResult(self.observe(), -0.05, False, info)

        position = action.get("position")
        if position is None:
            info["failure"] = Failure("invalid_place", "missing place position", True)
            return StepResult(self.observe(), -0.05, False, info)
        pos = np.asarray(position, dtype=float)
        self.last_place_position = pos.copy()
        info["place_position"] = pos.tolist()

        d_clear = float(np.linalg.norm(pos - self.clear_zone_center))
        if d_clear > self.clear_zone_radius:
            info["failure"] = Failure(
                "place_out_of_clear_zone",
                f"place position outside the clear zone ({d_clear:.2f})",
                True,
            )
            return StepResult(self.observe(), -0.10, False, info)

        self.gripper = pos.copy()
        self.trajectory.append(self.gripper.copy())
        if self.held == "obstacle":
            self.obstacle_position = pos.copy()
            self.obstacle_moved = True
        elif self.held == "target":
            self.target_position = pos.copy()
        self.held = None
        return StepResult(self.observe(), -0.02, False, info)

    def render(self, agent: Any | None = None, info: dict[str, Any] | None = None) -> None:
        import matplotlib.pyplot as plt

        if self._fig is None or self._ax is None:
            plt.ion()
            self._fig, self._ax = plt.subplots(figsize=(5.2, 5.0))
        ax = self._ax
        ax.clear()
        draw_clear_path_before_pick_scene(ax, self, agent, info)
        self._fig.canvas.draw_idle()
        plt.pause(0.1)


class ClearPathBeforePickAgent:
    """Try the target, recover by clearing the obstacle, then retry."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.state: str = "try_target"
        self.precondition_failure_count: int = 0
        self.clear_step_count: int = 0
        self.retry_count: int = 0

    def act(self, obs: dict[str, Any]) -> dict[str, Any]:
        if self.state == "try_target":
            return {"action_type": "pick", "target": "target"}
        if self.state == "clear_obstacle":
            return {"action_type": "pick", "target": "obstacle"}
        if self.state == "place_obstacle":
            zone = np.asarray(obs["clear_zone_center"], dtype=float)
            return {"action_type": "place", "position": zone.tolist()}
        if self.state == "retry_target":
            return {"action_type": "pick", "target": "target"}
        return {"action_type": "wait"}

    def update(self, obs: dict[str, Any], reward: float, info: dict[str, Any]) -> None:
        if info.get("success"):
            self.state = "succeeded"
            return

        failure = info.get("failure")
        if isinstance(failure, Failure) and failure.kind == "precondition_blocked":
            self.precondition_failure_count += 1
            self.state = "clear_obstacle"
            return

        action_type = info.get("action_type")
        if action_type == "pick" and info.get("pick_target") == "obstacle" and obs.get("held") == "obstacle":
            self.clear_step_count += 1
            self.state = "place_obstacle"
            return

        if action_type == "place" and obs.get("held") is None and obs.get("obstacle_moved"):
            self.retry_count += 1
            self.state = "retry_target"
            return


def draw_clear_path_before_pick_scene(
    ax: Any,
    env: ClearPathBeforePickWorld,
    agent: ClearPathBeforePickAgent | None,
    info: dict[str, Any] | None,
) -> None:
    import matplotlib.patches as mpatches

    ax.set_xlim(0.0, env.size)
    ax.set_ylim(0.0, env.size)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("clear path before pick")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.add_patch(mpatches.Rectangle((0.0, 0.0), env.size, env.size, color="0.97", ec="0.7"))

    # clear zone
    ax.add_patch(
        mpatches.Circle(
            env.clear_zone_center,
            env.clear_zone_radius,
            color="tab:cyan",
            alpha=0.15,
            ec="tab:cyan",
            lw=1.2,
            linestyle="--",
        )
    )
    ax.text(
        env.clear_zone_center[0],
        env.clear_zone_center[1] + env.clear_zone_radius + 0.02,
        "clear zone",
        ha="center",
        fontsize=8,
        color="tab:cyan",
    )

    # gripper-to-target abstract path
    if not env.target_picked:
        ax.plot(
            [env.gripper[0], env.target_position[0]],
            [env.gripper[1], env.target_position[1]],
            color="0.6",
            linestyle=":",
            linewidth=1.0,
            alpha=0.7,
        )

    # target
    if not env.target_picked:
        ax.add_patch(
            mpatches.Circle(env.target_position, env.target_radius, color="tab:red", alpha=0.85)
        )
        ax.text(
            env.target_position[0],
            env.target_position[1] - env.target_radius - 0.025,
            "target",
            ha="center",
            fontsize=8,
            color="tab:red",
        )

    # obstacle
    ox, oy = env.obstacle_position
    ax.add_patch(
        mpatches.Rectangle(
            (ox - env.obstacle_radius, oy - env.obstacle_radius),
            env.obstacle_radius * 2,
            env.obstacle_radius * 2,
            color="tab:blue",
            alpha=0.85,
        )
    )
    ax.text(ox, oy - env.obstacle_radius - 0.025, "obstacle", ha="center", fontsize=8, color="tab:blue")

    # gripper
    if len(env.trajectory) > 1:
        traj = np.asarray(env.trajectory)
        ax.plot(traj[:, 0], traj[:, 1], color="tab:blue", linewidth=1.5, alpha=0.5)
    gripper_color = "tab:blue" if env.held is None else "tab:orange"
    ax.plot(env.gripper[0], env.gripper[1], marker="o", color=gripper_color, markersize=10)
    if env.held is not None:
        ax.text(
            env.gripper[0],
            env.gripper[1] + 0.04,
            f"holding {env.held}",
            ha="center",
            fontsize=8,
            color="tab:orange",
        )

    status_parts: list[str] = [f"step={env.time}"]
    if agent is not None:
        status_parts.append(f"state={agent.state}")
        status_parts.append(f"precon_fail={agent.precondition_failure_count}")
        status_parts.append(f"clears={agent.clear_step_count}")
        status_parts.append(f"retries={agent.retry_count}")
    if info is not None and "failure" in info:
        status_parts.append(f"failure={info['failure'].kind}")
    if info is not None and info.get("success"):
        status_parts.append("success")
    ax.text(0.02, 0.98, "  ".join(status_parts), transform=ax.transAxes, va="top", fontsize=9)


def run(seed: int = 0, render: bool = True, max_steps: int = 15) -> Trace:
    env = ClearPathBeforePickWorld(seed=seed, max_steps=max_steps)
    agent = ClearPathBeforePickAgent()
    obs = env.reset(seed=seed)
    agent.reset()
    trace = Trace()

    for _ in range(max_steps):
        action = agent.act(obs)
        result = env.step(action)
        obs, reward, done, info = result.as_tuple()
        agent.update(obs, reward, info)
        info["agent_state"] = agent.state
        info["precondition_failure_count"] = agent.precondition_failure_count
        info["clear_step_count"] = agent.clear_step_count
        info["retry_count"] = agent.retry_count
        trace.append(obs, action, reward, info)

        if render:
            env.render(agent=agent, info=info)
        if done:
            break

    return trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    trace = run(seed=args.seed, render=not args.no_render, max_steps=args.max_steps)
    success = bool(trace.infos and trace.infos[-1].get("success"))
    failures = [failure.kind for failure in trace.failures()]
    final = trace.infos[-1] if trace.infos else {}
    print(
        f"success={success} steps={len(trace.actions)} "
        f"precon_fail={final.get('precondition_failure_count', 0)} "
        f"clears={final.get('clear_step_count', 0)} "
        f"retries={final.get('retry_count', 0)} failures={failures}"
    )

    if not args.no_render:
        import matplotlib.pyplot as plt

        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
