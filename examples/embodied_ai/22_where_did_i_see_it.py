"""Memory query: revisit the place where the target object was last seen.

The agent explores several waypoints, observing whatever objects fall inside
its FOV, and stores each sighting in a memory keyed by object name. When the
exploration finishes, the agent queries the memory for the target object,
walks back to its remembered position, and interacts there.

The point is that the action phase relies on a memory lookup, not on a live
observation - the agent has already moved past the target by the time the
query runs.
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


DEFAULT_OBJECTS: tuple[dict[str, Any], ...] = (
    {"name": "book", "color": "tab:blue", "position": (0.20, 0.30)},
    {"name": "mug", "color": "tab:red", "position": (0.50, 0.20)},
    {"name": "plant", "color": "tab:green", "position": (0.80, 0.50)},
)

DEFAULT_WAYPOINTS: tuple[tuple[float, float], ...] = (
    (0.20, 0.30),
    (0.50, 0.20),
    (0.80, 0.50),
)


class WhereDidISeeItWorld:
    """2D table with several scattered objects and one target the agent must find."""

    def __init__(
        self,
        *,
        seed: int | None = 0,
        max_steps: int = 30,
        agent_start: tuple[float, float] = (0.50, 0.70),
        target_name: str = "mug",
        objects: tuple[dict[str, Any], ...] | None = None,
        observe_radius: float = 0.18,
        interact_radius: float = 0.12,
        move_speed: float = 0.10,
    ) -> None:
        self.seed = seed
        self.size = 1.0
        self.agent_start = np.asarray(agent_start, dtype=float)
        self.target_name = target_name
        self.objects = tuple(objects or DEFAULT_OBJECTS)
        self.observe_radius = observe_radius
        self.interact_radius = interact_radius
        self.move_speed = move_speed
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
        self.time = 0
        self.holding = False
        self.last_action_type: str | None = None
        self.last_interact_target: np.ndarray | None = None
        self.trajectory: list[np.ndarray] = [self.agent.copy()]
        return self.observe()

    def _find_target(self) -> dict[str, Any] | None:
        for obj in self.objects:
            if obj["name"] == self.target_name:
                return obj
        return None

    def observe(self) -> dict[str, Any]:
        visible: list[dict[str, Any]] = []
        for obj in self.objects:
            distance = float(np.linalg.norm(self.agent - np.asarray(obj["position"])))
            if distance <= self.observe_radius:
                visible.append(
                    {
                        "name": obj["name"],
                        "color": obj["color"],
                        "position": tuple(obj["position"]),
                    }
                )
        target_obj = self._find_target()
        return {
            "time": self.time,
            "agent_pos": self.agent.copy(),
            "visible_objects": visible,
            "target": {
                "name": self.target_name,
                "color": target_obj["color"] if target_obj is not None else None,
            },
            "observe_radius": self.observe_radius,
            "interact_radius": self.interact_radius,
            "move_speed": self.move_speed,
            "holding": self.holding,
        }

    def step(self, action: dict[str, Any]) -> StepResult:
        self.time += 1
        action_type = action.get("action_type") if isinstance(action, dict) else None
        info: dict[str, Any] = {"time": self.time, "action_type": action_type, "success": False}
        self.last_action_type = action_type

        if action_type == "move":
            delta = np.asarray(action.get("delta", [0.0, 0.0]), dtype=float)
            norm = float(np.linalg.norm(delta))
            if norm > self.move_speed:
                delta = delta / max(norm, 1e-9) * self.move_speed
            self.agent = np.clip(self.agent + delta, 0.0, self.size)
            self.trajectory.append(self.agent.copy())
            info["agent_pos"] = self.agent.copy().tolist()
            return StepResult(self.observe(), -0.02, False, info)

        if action_type == "interact":
            target = np.asarray(action.get("target", self.agent), dtype=float)
            d_agent_target = float(np.linalg.norm(self.agent - target))
            info["interact_target"] = target.tolist()
            info["d_agent_target"] = d_agent_target
            target_obj = self._find_target()
            if target_obj is None:
                info["failure"] = Failure("target_missing", "target not in world", False)
                return StepResult(self.observe(), -0.5, True, info)
            true_pos = np.asarray(target_obj["position"], dtype=float)
            d_target_true = float(np.linalg.norm(target - true_pos))
            info["d_target_true"] = d_target_true
            if d_agent_target > self.interact_radius:
                info["failure"] = Failure(
                    "interact_out_of_range",
                    f"agent too far from interact target ({d_agent_target:.2f})",
                    True,
                )
                return StepResult(self.observe(), -0.1, False, info)
            if d_target_true > self.interact_radius:
                info["failure"] = Failure(
                    "memory_stale",
                    f"target moved since memory write ({d_target_true:.2f})",
                    True,
                )
                return StepResult(self.observe(), -0.1, False, info)
            self.last_interact_target = target.copy()
            self.holding = True
            info["success"] = True
            info["true_target_position"] = true_pos.tolist()
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
        draw_where_did_i_see_it_scene(ax, self, agent, info)
        self._fig.canvas.draw_idle()
        plt.pause(0.1)


class WhereDidISeeItAgent:
    """Explore waypoints, memorize sightings, then query and revisit on demand."""

    def __init__(
        self,
        waypoints: tuple[tuple[float, float], ...] | None = None,
        *,
        move_speed: float = 0.10,
        arrival_radius: float = 0.06,
        interact_radius: float = 0.12,
    ) -> None:
        self.waypoints = tuple(waypoints or DEFAULT_WAYPOINTS)
        self.move_speed = move_speed
        self.arrival_radius = arrival_radius
        self.interact_radius = interact_radius
        self.reset()

    def reset(self) -> None:
        self.memory: dict[str, dict[str, Any]] = {}
        self.state: str = "explore"
        self.waypoint_index: int = 0
        self.observation_count: int = 0
        self.memory_query_count: int = 0
        self.target_query: str | None = None
        self.query_position: np.ndarray | None = None
        self.target_in_memory: bool = False

    def act(self, obs: dict[str, Any]) -> dict[str, Any]:
        self._absorb_observations(obs)
        if self.state == "explore":
            return self._explore_step(obs)
        if self.state == "query":
            return self._query_memory(obs)
        if self.state == "revisit":
            return self._revisit_step(obs)
        if self.state == "interact":
            return self._interact_step(obs)
        # succeeded or unknown - stay put
        return {"action_type": "move", "delta": [0.0, 0.0]}

    def _absorb_observations(self, obs: dict[str, Any]) -> None:
        for vis in obs.get("visible_objects", []) or []:
            name = vis["name"]
            if name not in self.memory:
                self.observation_count += 1
            self.memory[name] = {
                "color": vis.get("color"),
                "position": np.asarray(vis["position"], dtype=float),
                "time": int(obs.get("time", 0)),
            }

    def _explore_step(self, obs: dict[str, Any]) -> dict[str, Any]:
        if self.waypoint_index >= len(self.waypoints):
            self.state = "query"
            return self._query_memory(obs)
        agent_pos = np.asarray(obs["agent_pos"], dtype=float)
        waypoint = np.asarray(self.waypoints[self.waypoint_index], dtype=float)
        delta = waypoint - agent_pos
        distance = float(np.linalg.norm(delta))
        if distance <= self.arrival_radius:
            self.waypoint_index += 1
            if self.waypoint_index >= len(self.waypoints):
                self.state = "query"
                return self._query_memory(obs)
            waypoint = np.asarray(self.waypoints[self.waypoint_index], dtype=float)
            delta = waypoint - agent_pos
            distance = float(np.linalg.norm(delta))
        step = delta / max(distance, 1e-9) * self.move_speed
        return {"action_type": "move", "delta": step.tolist()}

    def _query_memory(self, obs: dict[str, Any]) -> dict[str, Any]:
        self.memory_query_count += 1
        target = obs.get("target") or {}
        target_name = target.get("name")
        self.target_query = target_name
        if target_name is None or target_name not in self.memory:
            self.target_in_memory = False
            # fall back to re-exploring; here we just stop without producing failure
            self.state = "stuck"
            return {"action_type": "move", "delta": [0.0, 0.0]}
        self.target_in_memory = True
        self.query_position = self.memory[target_name]["position"].copy()
        self.state = "revisit"
        return self._revisit_step(obs)

    def _revisit_step(self, obs: dict[str, Any]) -> dict[str, Any]:
        assert self.query_position is not None
        agent_pos = np.asarray(obs["agent_pos"], dtype=float)
        delta = self.query_position - agent_pos
        distance = float(np.linalg.norm(delta))
        if distance <= self.interact_radius:
            self.state = "interact"
            return self._interact_step(obs)
        step = delta / max(distance, 1e-9) * self.move_speed
        return {"action_type": "move", "delta": step.tolist()}

    def _interact_step(self, obs: dict[str, Any]) -> dict[str, Any]:
        assert self.query_position is not None
        return {"action_type": "interact", "target": self.query_position.tolist()}

    def update(self, obs: dict[str, Any], reward: float, info: dict[str, Any]) -> None:
        if info.get("success"):
            self.state = "succeeded"


def draw_where_did_i_see_it_scene(
    ax: Any,
    env: WhereDidISeeItWorld,
    agent: WhereDidISeeItAgent | None,
    info: dict[str, Any] | None,
) -> None:
    import matplotlib.patches as mpatches

    ax.set_xlim(0.0, env.size)
    ax.set_ylim(0.0, env.size)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("where did I see it: memory query")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    ax.add_patch(mpatches.Rectangle((0.0, 0.0), env.size, env.size, color="0.97", ec="0.7"))

    for obj in env.objects:
        ax.add_patch(mpatches.Circle(obj["position"], 0.03, color=obj["color"], alpha=0.85))
        ax.text(
            obj["position"][0],
            obj["position"][1] - 0.05,
            obj["name"],
            ha="center",
            fontsize=7,
            color="0.3",
        )

    # FOV
    ax.add_patch(
        mpatches.Circle(
            env.agent,
            env.observe_radius,
            fill=False,
            linestyle="--",
            color="tab:blue",
            alpha=0.35,
        )
    )
    if len(env.trajectory) > 1:
        traj = np.asarray(env.trajectory)
        ax.plot(traj[:, 0], traj[:, 1], color="tab:blue", linewidth=1.5, alpha=0.6)
    ax.plot(*env.agent, marker="o", color="tab:blue", markersize=10)

    # remembered positions
    if agent is not None:
        for name, entry in agent.memory.items():
            pos = entry["position"]
            ax.plot(pos[0], pos[1], marker="x", color="tab:green", markersize=10, mew=2)
            ax.text(pos[0] + 0.02, pos[1] + 0.02, name, fontsize=7, color="tab:green")

    # query target
    if agent is not None and agent.query_position is not None:
        ax.plot(
            agent.query_position[0],
            agent.query_position[1],
            marker="*",
            color="tab:orange",
            markersize=16,
        )

    target_obj = env._find_target()
    if target_obj is not None:
        ax.text(
            0.02,
            0.04,
            f"goal: {target_obj['name']} ({target_obj['color']})",
            transform=ax.transAxes,
            fontsize=8,
            color="0.3",
        )

    status_parts: list[str] = [f"step={env.time}"]
    if agent is not None:
        status_parts.append(f"state={agent.state}")
        status_parts.append(f"wp={agent.waypoint_index}/{len(agent.waypoints)}")
        status_parts.append(f"memory={len(agent.memory)}")
        status_parts.append(f"queries={agent.memory_query_count}")
    if info is not None and "failure" in info:
        status_parts.append(f"failure={info['failure'].kind}")
    if info is not None and info.get("success"):
        status_parts.append("success")
    ax.text(0.02, 0.98, "  ".join(status_parts), transform=ax.transAxes, va="top", fontsize=9)


def run(
    seed: int = 0,
    render: bool = True,
    max_steps: int = 30,
    target_name: str = "mug",
) -> Trace:
    env = WhereDidISeeItWorld(seed=seed, max_steps=max_steps, target_name=target_name)
    agent = WhereDidISeeItAgent(
        move_speed=env.move_speed,
        interact_radius=env.interact_radius,
    )
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
        info["memory_query_count"] = agent.memory_query_count
        info["memory_size"] = len(agent.memory)
        info["target_in_memory"] = agent.target_in_memory
        info["waypoint_index"] = agent.waypoint_index
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
    parser.add_argument("--target", type=str, default="mug")
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    trace = run(
        seed=args.seed,
        render=not args.no_render,
        max_steps=args.max_steps,
        target_name=args.target,
    )
    success = bool(trace.infos and trace.infos[-1].get("success"))
    failures = [failure.kind for failure in trace.failures()]
    final = trace.infos[-1] if trace.infos else {}
    print(
        f"success={success} steps={len(trace.actions)} "
        f"obs={final.get('observation_count', 0)} "
        f"memory={final.get('memory_size', 0)} "
        f"queries={final.get('memory_query_count', 0)} failures={failures}"
    )

    if not args.no_render:
        import matplotlib.pyplot as plt

        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
