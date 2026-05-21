"""A tiny Vision-Language-Action loop without a neural model.

This is not a VLA model.  It is the smallest useful shape of one: parse a
language goal, read visual tokens, choose a discrete skill action, observe the
result, and change the next action after failure.
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


DRAW_COLORS = {
    "red": "tab:red",
    "blue": "tab:blue",
    "green": "tab:green",
}


def parse_vla_goal(command: str) -> dict[str, str]:
    words = command.lower().strip().split()
    if (
        len(words) == 6
        and words[0] == "place"
        and words[3] == "in"
        and words[5] == "bin"
    ):
        return {
            "intent": "place_in",
            "object_color": words[1],
            "object_name": words[2],
            "destination_color": words[4],
            "destination_name": words[5],
        }
    return {"intent": "unknown", "message": "use: place <color> <object> in <color> bin"}


@dataclass
class TinyVLAEntity:
    key: str
    kind: str
    color: str
    name: str
    position: np.ndarray
    radius: float
    held: bool = False
    placed: bool = False


class TinyVLAWorld:
    """A tabletop with visual tokens and a small skill API."""

    def __init__(self, command: str = "place red block in blue bin", max_steps: int = 25) -> None:
        self.command = command
        self.goal = parse_vla_goal(command)
        self.max_steps = max_steps
        self._figure: Any | None = None
        self._axis: Any | None = None
        self.reset()

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        self.time = 0
        self.camera_view = "wide"
        self.holding: str | None = None
        self.pick_attempts = 0
        self.place_attempts = 0
        self.visual_failures = 0
        self.last_visual_tokens: list[dict[str, Any]] = []
        self.last_pick_position: np.ndarray | None = None
        self.last_place_position: np.ndarray | None = None
        self.entities = {
            "red:block": TinyVLAEntity(
                key="red:block",
                kind="object",
                color="red",
                name="block",
                position=np.array([0.33, 0.66], dtype=float),
                radius=0.042,
            ),
            "green:block": TinyVLAEntity(
                key="green:block",
                kind="object",
                color="green",
                name="block",
                position=np.array([0.67, 0.62], dtype=float),
                radius=0.042,
            ),
            "blue:bin": TinyVLAEntity(
                key="blue:bin",
                kind="bin",
                color="blue",
                name="bin",
                position=np.array([0.72, 0.24], dtype=float),
                radius=0.070,
            ),
            "red:bin": TinyVLAEntity(
                key="red:bin",
                kind="bin",
                color="red",
                name="bin",
                position=np.array([0.24, 0.24], dtype=float),
                radius=0.070,
            ),
        }
        return self.observe()

    def observe(self) -> dict[str, Any]:
        tokens: list[dict[str, Any]] = []
        for entity in self.entities.values():
            if entity.held:
                continue
            token = self._visual_token(entity)
            if token is not None:
                tokens.append(token)
        self.last_visual_tokens = tokens
        return {
            "time": self.time,
            "goal_text": self.command,
            "camera_view": self.camera_view,
            "visual_tokens": tokens,
            "holding": self.holding,
            "pick_attempts": self.pick_attempts,
            "place_attempts": self.place_attempts,
        }

    def step(self, action: dict[str, Any]) -> StepResult:
        self.time += 1
        action_type = action.get("type", "look")
        info: dict[str, Any] = {
            "time": self.time,
            "action_type": action_type,
            "camera_view": self.camera_view,
            "success": False,
        }

        if action_type == "look":
            return self._step_look(action, info)
        if action_type == "skill":
            skill = action.get("skill")
            if skill == "pick":
                return self._step_pick(action, info)
            if skill == "place":
                return self._step_place(action, info)
            info["failure"] = Failure("unknown_skill", f"unknown skill: {skill}", True)
            return StepResult(self.observe(), -0.05, False, info)

        info["failure"] = Failure("invalid_action", f"unknown action type: {action_type}", True)
        return StepResult(self.observe(), -0.05, False, info)

    def _step_look(self, action: dict[str, Any], info: dict[str, Any]) -> StepResult:
        view = str(action.get("view", "wide"))
        if view not in {"wide", "close_red"}:
            info["failure"] = Failure("invalid_view", "camera view is not available", True)
            return StepResult(self.observe(), -0.04, False, info)
        self.camera_view = view
        info["camera_view"] = view
        return StepResult(self.observe(), -0.01, False, info)

    def _step_pick(self, action: dict[str, Any], info: dict[str, Any]) -> StepResult:
        key = str(action.get("target_key", ""))
        entity = self.entities.get(key)
        if entity is None or entity.kind != "object" or entity.placed:
            info["failure"] = Failure("invalid_target", "pick target is not an available object", True)
            return StepResult(self.observe(), -0.05, False, info)

        position = np.clip(
            np.asarray(action.get("position", entity.position), dtype=float),
            [0.0, 0.0],
            [1.0, 1.0],
        )
        confidence = float(action.get("confidence", 0.0))
        self.last_pick_position = position.copy()
        self.pick_attempts += 1
        error = float(np.linalg.norm(position - entity.position))
        info.update(
            {
                "skill": "pick",
                "target_key": key,
                "pick_position": position.copy(),
                "visual_confidence": confidence,
                "alignment_error": error,
                "pick_attempts": self.pick_attempts,
            }
        )

        if confidence < 0.80:
            self.visual_failures += 1
            info["visual_failures"] = self.visual_failures
            info["failure"] = Failure(
                "visual_pose_uncertain",
                "the visual token was too uncertain for a reliable pick",
                True,
            )
            return StepResult(self.observe(), -0.14, False, info)

        if error <= 0.055:
            entity.held = True
            self.holding = key
            info["pick_success"] = True
            return StepResult(self.observe(), 0.20, False, info)

        info["failure"] = Failure("skill_miss", "pick skill was executed at the wrong pose", True)
        return StepResult(self.observe(), -0.16, False, info)

    def _step_place(self, action: dict[str, Any], info: dict[str, Any]) -> StepResult:
        if self.holding is None:
            info["failure"] = Failure("empty_gripper", "place skill needs a held object", True)
            return StepResult(self.observe(), -0.05, False, info)

        destination_key = str(action.get("destination_key", ""))
        destination = self.entities.get(destination_key)
        held = self.entities[self.holding]
        if destination is None or destination.kind != "bin":
            info["failure"] = Failure("invalid_destination", "destination is not a bin", True)
            return StepResult(self.observe(), -0.05, False, info)

        self.place_attempts += 1
        self.last_place_position = destination.position.copy()
        held.held = False
        held.placed = True
        held.position = destination.position + np.array([0.0, 0.055])
        placed_key = self.holding
        self.holding = None

        success = (
            self.goal.get("intent") == "place_in"
            and placed_key == f"{self.goal['object_color']}:{self.goal['object_name']}"
            and destination_key == f"{self.goal['destination_color']}:{self.goal['destination_name']}"
        )
        info.update(
            {
                "skill": "place",
                "placed_key": placed_key,
                "destination_key": destination_key,
                "place_position": self.last_place_position.copy(),
                "place_attempts": self.place_attempts,
                "success": success,
            }
        )
        return StepResult(self.observe(), 1.0 if success else -0.05, success, info)

    def _visual_token(self, entity: TinyVLAEntity) -> dict[str, Any] | None:
        if self.camera_view == "close_red" and entity.key not in {"red:block", "blue:bin"}:
            return None

        confidence = 0.94
        position = entity.position.copy()
        if self.camera_view == "wide" and entity.key == "red:block":
            confidence = 0.58
            position = entity.position + np.array([0.082, -0.047], dtype=float)
        elif self.camera_view == "wide":
            confidence = 0.88 if entity.kind == "object" else 0.95

        return {
            "key": entity.key,
            "kind": entity.kind,
            "color": entity.color,
            "name": entity.name,
            "position": np.clip(position, [0.0, 0.0], [1.0, 1.0]),
            "confidence": confidence,
            "view": self.camera_view,
        }

    def render(self, agent: "TinyVLAAgent", info: dict[str, Any] | None = None) -> None:
        import matplotlib.pyplot as plt

        if self._figure is None or self._axis is None:
            plt.ion()
            self._figure, self._axis = plt.subplots(figsize=(5.2, 5.2))

        draw_tiny_vla_scene(self._axis, self, agent, info)
        self._figure.canvas.draw_idle()
        plt.pause(0.05)


class TinyVLAAgent:
    """A symbolic stand-in for the VLA loop structure."""

    def __init__(self, command: str = "place red block in blue bin") -> None:
        self.command = command
        self.goal = parse_vla_goal(command)
        self.reset()

    def reset(self) -> None:
        self.visual_memory: dict[str, dict[str, Any]] = {}
        self.state = "parse_language"
        self.needs_close_view = False
        self.recovery_count = 0
        self.skill_count = 0
        self.visual_updates = 0
        self._last_integrated_time: int | None = None

    def act(self, obs: dict[str, Any]) -> dict[str, Any]:
        self._integrate_visual_observation(obs)
        if self.goal.get("intent") != "place_in":
            self.state = "unsupported_language"
            return {"type": "look", "view": "wide"}

        target_key = f"{self.goal['object_color']}:{self.goal['object_name']}"
        destination_key = f"{self.goal['destination_color']}:{self.goal['destination_name']}"

        if obs.get("holding") == target_key:
            destination = self.visual_memory.get(destination_key)
            if destination is None:
                self.state = "look_for_destination"
                return {"type": "look", "view": "wide"}
            self.state = "action_place"
            self.skill_count += 1
            return {
                "type": "skill",
                "skill": "place",
                "target_key": target_key,
                "destination_key": destination_key,
            }

        if self.needs_close_view and obs.get("camera_view") != "close_red":
            self.state = "look_close_after_failure"
            return {"type": "look", "view": "close_red"}

        target = self.visual_memory.get(target_key)
        if target is None:
            self.state = "look_for_target"
            return {"type": "look", "view": "wide"}

        self.state = "action_pick_from_visual_token"
        self.skill_count += 1
        return {
            "type": "skill",
            "skill": "pick",
            "target_key": target_key,
            "position": np.asarray(target["position"], dtype=float).copy(),
            "confidence": float(target["confidence"]),
        }

    def update(self, obs: dict[str, Any], reward: float, info: dict[str, Any]) -> None:
        self._integrate_visual_observation(obs)

        failure = info.get("failure")
        if isinstance(failure, Failure) and failure.kind == "visual_pose_uncertain":
            self.needs_close_view = True
            self.recovery_count += 1
            self.state = "update_after_visual_failure"
        elif info.get("pick_success"):
            self.needs_close_view = False
            self.state = "object_in_gripper"
        elif info.get("success"):
            self.state = "goal_satisfied"

        info["agent_state"] = self.state
        info["parsed_goal"] = dict(self.goal)
        info["visual_memory_count"] = len(self.visual_memory)
        info["skill_count"] = self.skill_count
        info["recovery_count"] = self.recovery_count
        info["visual_updates"] = self.visual_updates
        info["visual_memory"] = {
            key: {
                "position": value["position"].copy(),
                "confidence": value["confidence"],
                "view": value["view"],
            }
            for key, value in self.visual_memory.items()
        }

    def _integrate_visual_observation(self, obs: dict[str, Any]) -> None:
        obs_time = int(obs.get("time", -1))
        if obs_time == self._last_integrated_time:
            return
        self._last_integrated_time = obs_time

        for token in obs.get("visual_tokens", []):
            key = str(token["key"])
            confidence = float(token["confidence"])
            position = np.asarray(token["position"], dtype=float)
            previous = self.visual_memory.get(key)
            if previous is None or confidence >= previous["confidence"]:
                self.visual_memory[key] = {
                    "kind": token["kind"],
                    "color": token["color"],
                    "name": token["name"],
                    "position": position.copy(),
                    "confidence": confidence,
                    "view": token["view"],
                    "last_seen": obs_time,
                }
                self.visual_updates += 1


def draw_tiny_vla_scene(
    ax: Any,
    env: TinyVLAWorld,
    agent: TinyVLAAgent,
    info: dict[str, Any] | None = None,
) -> None:
    """Draw language parse, visual tokens, memory, and chosen skill."""

    from matplotlib.patches import Circle, Rectangle

    info = {} if info is None else info
    ax.clear()
    ax.set_title("tiny VLA loop")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)

    if env.camera_view == "close_red":
        ax.add_patch(Rectangle((0.18, 0.48), 0.34, 0.34, facecolor="tab:blue", alpha=0.06))

    for entity in env.entities.values():
        color = DRAW_COLORS[entity.color]
        if entity.kind == "bin":
            ax.add_patch(
                Rectangle(
                    (entity.position[0] - 0.075, entity.position[1] - 0.050),
                    0.15,
                    0.10,
                    facecolor=color,
                    alpha=0.16,
                    edgecolor=color,
                )
            )
            ax.text(entity.position[0], entity.position[1] - 0.080, entity.key, ha="center", fontsize=8)
            continue

        alpha = 0.45 if entity.placed else 0.85
        if entity.held:
            draw_position = np.array([0.50, 0.44], dtype=float)
        else:
            draw_position = entity.position
        ax.add_patch(Circle(draw_position, entity.radius, color=color, alpha=alpha))
        ax.text(draw_position[0], draw_position[1] + 0.060, entity.key, ha="center", fontsize=8)

    for token in env.last_visual_tokens:
        position = token["position"]
        ax.plot(*position, marker="x", markersize=10, color="tab:orange")
        ax.text(
            position[0],
            position[1] - 0.055,
            f"{token['key']}\nconf={token['confidence']:.2f}",
            ha="center",
            fontsize=7,
            color="tab:orange",
        )

    for key, memory in agent.visual_memory.items():
        position = memory["position"]
        radius = max(0.020, 0.095 * (1.0 - float(memory["confidence"])))
        ax.add_patch(
            Circle(
                position,
                radius,
                fill=False,
                linestyle="--",
                linewidth=2,
                color="tab:green",
            )
        )

    if env.last_pick_position is not None:
        ax.plot(*env.last_pick_position, marker="+", markersize=14, color="black", label="pick skill")
    if env.last_place_position is not None:
        ax.plot(*env.last_place_position, marker="v", markersize=10, color="black", label="place skill")

    status = (
        f"language: {env.command}\n"
        f"step={env.time}  view={env.camera_view}  state={info.get('agent_state', agent.state)}\n"
        f"skills={agent.skill_count}  visual_updates={agent.visual_updates}  "
        f"recoveries={agent.recovery_count}\n"
        f"holding={env.holding}"
    )
    if "failure" in info:
        status += f"  failure={info['failure'].kind}"
    if info.get("success"):
        status += "  success"
    ax.text(0.02, 0.98, status, transform=ax.transAxes, va="top", fontsize=9)
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="lower left", fontsize=8)


def run(
    command: str = "place red block in blue bin",
    seed: int = 0,
    render: bool = True,
    max_steps: int = 25,
) -> Trace:
    env = TinyVLAWorld(command=command, max_steps=max_steps)
    agent = TinyVLAAgent(command)
    obs = env.reset(seed=seed)
    agent.reset()
    trace = Trace()

    for _ in range(max_steps):
        action = agent.act(obs)
        result = env.step(action)
        obs, reward, done, info = result.as_tuple()
        agent.update(obs, reward, info)
        trace.append(obs, action, reward, info)

        if render:
            env.render(agent, info)

        if done:
            break

    return trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs="?", default="place red block in blue bin")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    trace = run(
        command=args.command,
        seed=args.seed,
        render=not args.no_render,
        max_steps=args.max_steps,
    )
    final_info = trace.infos[-1] if trace.infos else {}
    failures = [failure.kind for failure in trace.failures()]
    print(
        f"success={bool(final_info.get('success'))} steps={len(trace.actions)} "
        f"skills={final_info.get('skill_count', 0)} "
        f"visual_updates={final_info.get('visual_updates', 0)} "
        f"recoveries={final_info.get('recovery_count', 0)} "
        f"failures={failures}"
    )

    if not args.no_render:
        import matplotlib.pyplot as plt

        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
