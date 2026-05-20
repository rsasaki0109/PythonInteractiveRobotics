"""Object permanence: remember an object after it goes behind an occluder.

The agent observes a single object on a 2D table. After a few steps, an
occluder slides over the object and the observation channel reports the object
as no longer visible. Without memory, the agent would have no idea where to
look. With memory (object permanence), the agent persists the last known
position, walks to it, and peeks behind the occluder to recover the object.
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


class ObjectPermanenceWorld:
    """2D table with one object, an occluder, and an agent with limited FOV."""

    def __init__(
        self,
        *,
        seed: int | None = 0,
        max_steps: int = 30,
        agent_start: tuple[float, float] = (0.15, 0.5),
        object_position: tuple[float, float] = (0.42, 0.5),
        occluder_box: tuple[float, float, float, float] = (0.30, 0.35, 0.55, 0.65),
        occluder_appears_at: int = 1,
        observe_radius: float = 0.32,
        short_range_radius: float = 0.0,
        peek_radius: float = 0.12,
        move_speed: float = 0.10,
        object_color: str = "tab:red",
    ) -> None:
        self.seed = seed
        self.size = 1.0
        self.agent_start = np.asarray(agent_start, dtype=float)
        self.object_initial_pos = np.asarray(object_position, dtype=float)
        self.occluder_box = occluder_box
        self.occluder_appears_at = occluder_appears_at
        self.observe_radius = observe_radius
        self.short_range_radius = short_range_radius
        self.peek_radius = peek_radius
        self.move_speed = move_speed
        self.object_color = object_color
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
        self.agent = self.agent_start.copy()
        self.object_pos = self.object_initial_pos.copy()
        self.time = 0
        self.occluder_active = False
        self.holding = False
        self.last_action_type: str | None = None
        self.last_peek_target: np.ndarray | None = None
        self.trajectory: list[np.ndarray] = [self.agent.copy()]
        return self.observe()

    def _inside_occluder(self, pos: np.ndarray) -> bool:
        xmin, ymin, xmax, ymax = self.occluder_box
        return bool(xmin <= pos[0] <= xmax and ymin <= pos[1] <= ymax)

    def _object_visible(self) -> bool:
        distance = float(np.linalg.norm(self.agent - self.object_pos))
        if distance > self.observe_radius:
            return False
        if distance < self.short_range_radius:
            return True
        if self.occluder_active and self._inside_occluder(self.object_pos):
            return False
        return True

    def observe(self) -> dict[str, Any]:
        visible = self._object_visible()
        return {
            "time": self.time,
            "agent_pos": self.agent.copy(),
            "object_visible": visible,
            "object_pos_observed": self.object_pos.copy() if visible else None,
            "object_color_observed": self.object_color if visible else None,
            "occluder_active": self.occluder_active,
            "occluder_box": self.occluder_box,
            "holding": self.holding,
            "observe_radius": self.observe_radius,
            "peek_radius": self.peek_radius,
            "move_speed": self.move_speed,
        }

    def step(self, action: dict[str, Any]) -> StepResult:
        self.time += 1
        if self.time >= self.occluder_appears_at:
            self.occluder_active = True

        action_type = action.get("action_type") if isinstance(action, dict) else None
        info: dict[str, Any] = {"time": self.time, "action_type": action_type, "success": False}
        self.last_action_type = action_type

        if action_type == "move":
            delta = np.asarray(action.get("delta", [0.0, 0.0]), dtype=float)
            norm = float(np.linalg.norm(delta))
            if norm > self.move_speed:
                delta = delta / max(norm, 1e-9) * self.move_speed
            new_pos = np.clip(self.agent + delta, 0.0, self.size)
            self.agent = new_pos
            self.trajectory.append(self.agent.copy())
            info["agent_pos"] = self.agent.copy().tolist()
            return StepResult(self.observe(), -0.02, False, info)

        if action_type == "peek":
            target = np.asarray(action.get("target", self.agent), dtype=float)
            d_agent_target = float(np.linalg.norm(self.agent - target))
            d_target_object = float(np.linalg.norm(target - self.object_pos))
            self.last_peek_target = target.copy()
            info["peek_target"] = target.tolist()
            info["d_agent_target"] = d_agent_target
            info["d_target_object"] = d_target_object
            if d_agent_target > self.peek_radius:
                info["failure"] = Failure(
                    "peek_out_of_range",
                    f"agent too far from peek target ({d_agent_target:.2f})",
                    True,
                )
                return StepResult(self.observe(), -0.05, False, info)
            if d_target_object > self.peek_radius:
                info["failure"] = Failure(
                    "peek_miss",
                    f"no object near peek target ({d_target_object:.2f})",
                    True,
                )
                return StepResult(self.observe(), -0.10, False, info)
            self.holding = True
            info["success"] = True
            info["object_recovered_at"] = self.object_pos.copy().tolist()
            return StepResult(self.observe(), 1.0, True, info)

        info["failure"] = Failure("invalid_action", f"unknown action_type: {action_type}", True)
        return StepResult(self.observe(), -0.05, False, info)

    def render(self, agent: Any | None = None, info: dict[str, Any] | None = None) -> None:
        import matplotlib.pyplot as plt

        if self._fig is None or self._ax is None:
            plt.ion()
            self._fig, self._ax = plt.subplots(figsize=(5.6, 5.0))
        ax = self._ax
        ax.clear()
        draw_object_permanence_scene(ax, self, agent, info)
        self._fig.canvas.draw_idle()
        plt.pause(0.1)


class ObjectPermanenceAgent:
    """Walk to the remembered position when sight is lost; peek to recover."""

    def __init__(self, peek_radius: float = 0.12, move_speed: float = 0.10) -> None:
        self.peek_radius = peek_radius
        self.move_speed = move_speed
        self.reset()

    def reset(self) -> None:
        self.memory: np.ndarray | None = None
        self.memory_color: str | None = None
        self.memory_first_seen_time: int | None = None
        self.memory_last_seen_time: int | None = None
        self.observation_count: int = 0
        self.memory_persistence_count: int = 0
        self.state: str = "initial"

    @property
    def has_memory(self) -> bool:
        return self.memory is not None

    def act(self, obs: dict[str, Any]) -> dict[str, Any]:
        agent_pos = np.asarray(obs["agent_pos"], dtype=float)
        if obs.get("object_visible"):
            observed = np.asarray(obs["object_pos_observed"], dtype=float)
            if self.memory is None:
                self.memory_first_seen_time = int(obs.get("time", 0))
            self.memory = observed.copy()
            self.memory_color = obs.get("object_color_observed")
            self.memory_last_seen_time = int(obs.get("time", 0))
            self.observation_count += 1
            return self._approach_or_peek(agent_pos, label="see_and_approach")

        if self.memory is None:
            self.state = "scan"
            scan_delta = np.array([self.move_speed, 0.0], dtype=float)
            return {"action_type": "move", "delta": scan_delta.tolist()}

        # object permanence: keep belief alive across the disappearance
        self.memory_persistence_count += 1
        return self._approach_or_peek(agent_pos, label="recall_from_memory")

    def _approach_or_peek(self, agent_pos: np.ndarray, *, label: str) -> dict[str, Any]:
        assert self.memory is not None
        delta = self.memory - agent_pos
        distance = float(np.linalg.norm(delta))
        if distance <= self.peek_radius:
            self.state = "peek" if label != "see_and_approach" else "peek_visible"
            return {"action_type": "peek", "target": self.memory.tolist()}
        step = delta / max(distance, 1e-9) * self.move_speed
        self.state = label
        return {"action_type": "move", "delta": step.tolist()}

    def update(self, obs: dict[str, Any], reward: float, info: dict[str, Any]) -> None:
        if info.get("success"):
            self.state = "succeeded"


def draw_object_permanence_scene(
    ax: Any,
    env: ObjectPermanenceWorld,
    agent: ObjectPermanenceAgent | None,
    info: dict[str, Any] | None,
) -> None:
    import matplotlib.patches as mpatches

    ax.set_xlim(0.0, env.size)
    ax.set_ylim(0.0, env.size)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("object permanence toy")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    ax.add_patch(mpatches.Rectangle((0.0, 0.0), env.size, env.size, color="0.97", ec="0.7"))

    if env.occluder_active:
        xmin, ymin, xmax, ymax = env.occluder_box
        ax.add_patch(
            mpatches.Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                color="0.55",
                alpha=0.7,
                ec="0.3",
            )
        )

    # object: drawn with low alpha if occluded, full alpha if visible
    visible = not (env.occluder_active and env._inside_occluder(env.object_pos))
    object_alpha = 0.9 if visible else 0.35
    ax.add_patch(
        mpatches.Circle(env.object_pos, 0.03, color=env.object_color, alpha=object_alpha)
    )

    # agent FOV
    fov_circle = mpatches.Circle(
        env.agent,
        env.observe_radius,
        fill=False,
        linestyle="--",
        color="tab:blue",
        alpha=0.35,
    )
    ax.add_patch(fov_circle)

    # trajectory
    if len(env.trajectory) > 1:
        traj = np.asarray(env.trajectory)
        ax.plot(traj[:, 0], traj[:, 1], color="tab:blue", linewidth=1.5, alpha=0.6)
    ax.plot(*env.agent, marker="o", color="tab:blue", markersize=10)

    # memory marker
    if agent is not None and agent.memory is not None:
        ax.plot(
            agent.memory[0],
            agent.memory[1],
            marker="x",
            color="tab:green",
            markersize=12,
            mew=2,
        )
        ax.text(
            agent.memory[0],
            agent.memory[1] - 0.045,
            "remembered",
            ha="center",
            fontsize=7,
            color="tab:green",
        )

    if env.last_peek_target is not None:
        ax.plot(
            env.last_peek_target[0],
            env.last_peek_target[1],
            marker="*",
            color="tab:orange",
            markersize=14,
        )

    status_lines: list[str] = [
        f"step={env.time}",
        f"occluder={'on' if env.occluder_active else 'off'}",
    ]
    if agent is not None:
        status_lines.append(f"state={agent.state}")
        status_lines.append(f"obs={agent.observation_count}")
        status_lines.append(f"persist={agent.memory_persistence_count}")
        status_lines.append(f"memory={'yes' if agent.has_memory else 'no'}")
    if info is not None and "failure" in info:
        status_lines.append(f"failure={info['failure'].kind}")
    if info is not None and info.get("success"):
        status_lines.append("success")
    ax.text(0.02, 0.98, "  ".join(status_lines), transform=ax.transAxes, va="top", fontsize=9)


def run(
    seed: int = 0,
    render: bool = True,
    max_steps: int = 30,
) -> Trace:
    env = ObjectPermanenceWorld(seed=seed, max_steps=max_steps)
    agent = ObjectPermanenceAgent(peek_radius=env.peek_radius, move_speed=env.move_speed)
    obs = env.reset(seed=seed)
    agent.reset()
    trace = Trace()

    for _ in range(max_steps):
        action = agent.act(obs)
        result = env.step(action)
        obs, reward, done, info = result.as_tuple()
        agent.update(obs, reward, info)
        info["agent_state"] = agent.state
        info["observation_count"] = agent.observation_count
        info["memory_persistence_count"] = agent.memory_persistence_count
        info["has_memory"] = agent.has_memory
        if agent.memory is not None:
            info["memory_position"] = agent.memory.tolist()
        trace.append(obs, action, reward, info)

        if render:
            env.render(agent=agent, info=info)
        if done:
            break

    return trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    trace = run(seed=args.seed, render=not args.no_render, max_steps=args.max_steps)
    success = bool(trace.infos and trace.infos[-1].get("success"))
    failures = [failure.kind for failure in trace.failures()]
    final = trace.infos[-1] if trace.infos else {}
    print(
        f"success={success} steps={len(trace.actions)} "
        f"obs={final.get('observation_count', 0)} "
        f"persist={final.get('memory_persistence_count', 0)} failures={failures}"
    )

    if not args.no_render:
        import matplotlib.pyplot as plt

        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
