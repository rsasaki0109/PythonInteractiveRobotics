"""Door search POMDP: remember rooms, update belief, and find a hidden key."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pir.core.types import Failure, StepResult, Trace


class DoorSearchWorld:
    """A tiny partially observable house with doors and hidden containers."""

    def __init__(self, *, max_steps: int = 40) -> None:
        self.max_steps = max_steps
        self.room_positions = {
            "entry": (0.0, 0.0),
            "storage": (0.0, 1.0),
            "bedroom": (1.0, 0.0),
            "kitchen": (1.0, 1.0),
            "study": (2.0, 1.0),
        }
        self.doors = {
            frozenset(("entry", "storage")): "locked",
            frozenset(("entry", "bedroom")): "closed",
            frozenset(("entry", "kitchen")): "closed",
            frozenset(("kitchen", "study")): "closed",
        }
        self.containers = {
            "entry": [],
            "storage": ["box"],
            "bedroom": ["drawer"],
            "kitchen": ["cabinet"],
            "study": ["desk"],
        }
        self.hidden_key = ("kitchen", "cabinet")
        self._fig = None
        self._ax = None
        self.reset()

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        self.room = "entry"
        self.time = 0
        self.key_found = False
        self.open_doors: set[frozenset[str]] = set()
        self.inspected: set[tuple[str, str]] = set()
        self.trajectory = [self.room]
        return self.observe()

    def observe(self) -> dict[str, Any]:
        visible_doors = []
        for edge, state in self.doors.items():
            if self.room not in edge:
                continue
            other_room = next(room for room in edge if room != self.room)
            visible_doors.append(
                {
                    "to": other_room,
                    "state": "open" if edge in self.open_doors else state,
                }
            )

        return {
            "time": self.time,
            "room": self.room,
            "visible_doors": visible_doors,
            "containers": list(self.containers[self.room]),
            "inspected_here": [
                container
                for room, container in self.inspected
                if room == self.room
            ],
            "key_found": self.key_found,
        }

    def step(self, action: dict[str, str]) -> StepResult:
        self.time += 1
        action_type = action.get("type", "noop")
        info: dict[str, Any] = {
            "time": self.time,
            "action_type": action_type,
            "room": self.room,
            "success": False,
        }

        if action_type == "open_door":
            return self._open_door(action.get("to", ""), info)
        if action_type == "move":
            return self._move(action.get("to", ""), info)
        if action_type == "inspect":
            return self._inspect(action.get("container", ""), info)

        info["failure"] = Failure("invalid_action", f"unknown action type: {action_type}", True)
        return StepResult(self.observe(), -0.05, False, info)

    def _open_door(self, target: str, info: dict[str, Any]) -> StepResult:
        edge = frozenset((self.room, target))
        info["target_room"] = target
        if edge not in self.doors:
            info["failure"] = Failure("no_door", "there is no visible door to that room", True)
            return StepResult(self.observe(), -0.06, False, info)

        state = self.doors[edge]
        if state == "locked":
            info["failure"] = Failure("locked_door", "door is locked and cannot be opened", True)
            return StepResult(self.observe(), -0.10, False, info)

        self.open_doors.add(edge)
        return StepResult(self.observe(), -0.01, False, info)

    def _move(self, target: str, info: dict[str, Any]) -> StepResult:
        edge = frozenset((self.room, target))
        info["target_room"] = target
        if edge not in self.doors:
            info["failure"] = Failure("no_door", "there is no door to that room", True)
            return StepResult(self.observe(), -0.06, False, info)
        if edge not in self.open_doors:
            info["failure"] = Failure("closed_door", "door must be opened before moving", True)
            return StepResult(self.observe(), -0.06, False, info)

        self.room = target
        self.trajectory.append(self.room)
        info["room"] = self.room
        return StepResult(self.observe(), -0.01, False, info)

    def _inspect(self, container: str, info: dict[str, Any]) -> StepResult:
        info["container"] = container
        if container not in self.containers[self.room]:
            info["failure"] = Failure("invalid_container", "container is not in this room", True)
            return StepResult(self.observe(), -0.05, False, info)

        self.inspected.add((self.room, container))
        if (self.room, container) == self.hidden_key:
            self.key_found = True
            info["success"] = True
            info["found"] = "key"
            return StepResult(self.observe(), 1.0, True, info)

        info["failure"] = Failure("not_found", "the key was not in this container", True)
        return StepResult(self.observe(), -0.04, False, info)

    def render(self, agent: Any | None = None, info: dict[str, Any] | None = None) -> None:
        import matplotlib.pyplot as plt

        if self._fig is None or self._ax is None:
            plt.ion()
            self._fig, self._ax = plt.subplots(figsize=(6, 4.8))

        draw_door_search_scene(self._ax, self, agent, info)
        self._fig.canvas.draw_idle()
        plt.pause(0.05)


class DoorSearchAgent:
    """Search policy with visited-room memory and key-location belief."""

    def reset(self) -> None:
        self.key_belief = {
            "storage": 0.45,
            "bedroom": 0.25,
            "kitchen": 0.20,
            "study": 0.10,
        }
        self.visited_rooms = {"entry"}
        self.inspected_rooms: set[str] = set()
        self.blocked_rooms: set[str] = set()
        self.current_target: str | None = None
        self.pending: list[dict[str, str]] = []
        self.state = "choose_room"
        self.locked_failures = 0
        self.not_found_failures = 0

    def act(self, obs: dict[str, Any]) -> dict[str, str]:
        self.visited_rooms.add(obs["room"])

        if self.pending:
            return self.pending.pop(0)

        container = self._uninspected_container(obs)
        if self.current_target == obs["room"] and container is not None:
            self.state = "inspect"
            return {"type": "inspect", "container": container}

        self.current_target = self._choose_target_room()
        if self.current_target is None:
            self.state = "no_hypothesis"
            return {"type": "inspect", "container": obs["containers"][0]} if obs["containers"] else {"type": "noop"}

        self.pending = self._plan_to_target(obs["room"], self.current_target)
        if self.pending:
            self.state = "open_or_move"
            return self.pending.pop(0)

        self.state = "wait"
        return {"type": "noop"}

    def update(self, obs: dict[str, Any], reward: float, info: dict[str, Any]) -> None:
        failure = info.get("failure")
        if isinstance(failure, Failure) and failure.kind == "locked_door":
            target = info.get("target_room")
            if target is not None:
                self.blocked_rooms.add(target)
                self._remove_room_from_belief(target)
            self.pending = []
            self.current_target = None
            self.locked_failures += 1
            self.state = "update_after_locked_door"
            return

        if isinstance(failure, Failure) and failure.kind == "not_found":
            room = info.get("room")
            if room is not None:
                self.inspected_rooms.add(room)
                self._remove_room_from_belief(room)
            self.current_target = None
            self.not_found_failures += 1
            self.state = "update_after_not_found"
            return

        if info.get("success"):
            self.state = "found_key"

    def _choose_target_room(self) -> str | None:
        candidates = [
            room
            for room, probability in self.key_belief.items()
            if probability > 0.0
            and room not in self.blocked_rooms
            and room not in self.inspected_rooms
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda room: self.key_belief[room])

    def _plan_to_target(self, current: str, target: str) -> list[dict[str, str]]:
        path = room_path(current, target)
        actions: list[dict[str, str]] = []
        for next_room in path[1:]:
            actions.append({"type": "open_door", "to": next_room})
            actions.append({"type": "move", "to": next_room})
        return actions

    def _uninspected_container(self, obs: dict[str, Any]) -> str | None:
        inspected_here = set(obs.get("inspected_here", []))
        for container in obs.get("containers", []):
            if container not in inspected_here:
                return container
        return None

    def _remove_room_from_belief(self, room: str) -> None:
        if room not in self.key_belief:
            return
        self.key_belief[room] = 0.0
        total = sum(self.key_belief.values())
        if total <= 0.0:
            return
        for key in self.key_belief:
            self.key_belief[key] /= total


def room_path(start: str, goal: str) -> list[str]:
    graph = {
        "entry": ["storage", "bedroom", "kitchen"],
        "storage": ["entry"],
        "bedroom": ["entry"],
        "kitchen": ["entry", "study"],
        "study": ["kitchen"],
    }
    frontier: list[list[str]] = [[start]]
    visited = {start}
    while frontier:
        path = frontier.pop(0)
        if path[-1] == goal:
            return path
        for neighbor in graph[path[-1]]:
            if neighbor in visited:
                continue
            visited.add(neighbor)
            frontier.append(path + [neighbor])
    return [start]


def draw_door_search_scene(
    ax: Any,
    env: DoorSearchWorld,
    agent: DoorSearchAgent | None,
    info: dict[str, Any] | None,
) -> None:
    from matplotlib.patches import Rectangle

    ax.clear()
    ax.set_title("door search POMDP")
    ax.set_xlim(-0.55, 2.55)
    ax.set_ylim(-0.45, 1.55)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    for room, (x, y) in env.room_positions.items():
        probability = 0.0 if agent is None else agent.key_belief.get(room, 0.0)
        color = (1.0, 0.94 - 0.45 * probability, 0.72 - 0.45 * probability)
        ax.add_patch(Rectangle((x - 0.35, y - 0.25), 0.7, 0.5, facecolor=color, edgecolor="0.25"))
        label = f"{room}\np={probability:.2f}" if room != "entry" else room
        ax.text(x, y, label, ha="center", va="center", fontsize=8)

    for edge, state in env.doors.items():
        a, b = tuple(edge)
        xa, ya = env.room_positions[a]
        xb, yb = env.room_positions[b]
        is_open = edge in env.open_doors
        color = "tab:green" if is_open else ("tab:red" if state == "locked" else "0.35")
        linestyle = "-" if is_open else "--"
        ax.plot([xa, xb], [ya, yb], color=color, linestyle=linestyle, linewidth=2)

    trajectory_points = [env.room_positions[room] for room in env.trajectory]
    if len(trajectory_points) > 1:
        xs = [point[0] for point in trajectory_points]
        ys = [point[1] for point in trajectory_points]
        ax.plot(xs, ys, color="tab:blue", linewidth=2, alpha=0.7)

    robot_x, robot_y = env.room_positions[env.room]
    ax.plot(robot_x, robot_y + 0.28, "o", color="tab:blue", markersize=9)

    inspected = ", ".join(f"{room}:{container}" for room, container in sorted(env.inspected))
    state = getattr(agent, "state", "none")
    status = f"room={env.room} state={state}\ninspected={inspected or '-'}"
    if info is not None and "failure" in info:
        status += f"\nfailure={info['failure'].kind}"
    if env.key_found:
        status += "\nkey found"
    ax.text(-0.5, 1.48, status, ha="left", va="top", fontsize=9)


def run(seed: int = 0, render: bool = True, max_steps: int = 40) -> Trace:
    env = DoorSearchWorld(max_steps=max_steps)
    agent = DoorSearchAgent()
    obs = env.reset(seed=seed)
    agent.reset()
    trace = Trace()

    for _ in range(max_steps):
        action = agent.act(obs)
        result = env.step(action)
        obs, reward, done, info = result.as_tuple()
        agent.update(obs, reward, info)
        info["agent_state"] = agent.state
        info["key_belief"] = dict(agent.key_belief)
        info["visited_rooms"] = sorted(agent.visited_rooms)
        info["inspected_rooms"] = sorted(agent.inspected_rooms)
        info["blocked_rooms"] = sorted(agent.blocked_rooms)
        info["locked_failures"] = agent.locked_failures
        info["not_found_failures"] = agent.not_found_failures
        trace.append(obs, action, reward, info)

        if render:
            env.render(agent=agent, info=info)

        if done:
            break

    return trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    trace = run(seed=args.seed, render=not args.no_render, max_steps=args.max_steps)
    success = bool(trace.infos and trace.infos[-1].get("success"))
    failures = [failure.kind for failure in trace.failures()]
    final_info = trace.infos[-1] if trace.infos else {}
    print(
        f"success={success} steps={len(trace.actions)} "
        f"locked={final_info.get('locked_failures', 0)} "
        f"not_found={final_info.get('not_found_failures', 0)} failures={failures}"
    )

    if not args.no_render:
        import matplotlib.pyplot as plt

        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
