"""Goal-conditioned interaction in a tiny kitchen.

The agent receives a controlled-language goal, observes one station at a time,
remembers containers, handles failed inspections, opens the right container,
picks the target object, and places it at the goal location.
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

from pir.core.types import Failure, StepResult, Trace


STATION_POSITIONS = {
    "entry": np.array([0.10, 0.50], dtype=float),
    "counter": np.array([0.34, 0.78], dtype=float),
    "cabinet": np.array([0.62, 0.78], dtype=float),
    "fridge": np.array([0.86, 0.50], dtype=float),
    "table": np.array([0.50, 0.22], dtype=float),
}

CONTAINER_STATIONS = {
    "drawer": "counter",
    "cabinet": "cabinet",
    "fridge": "fridge",
}

CONTAINER_ORDER = ["drawer", "cabinet", "fridge"]


def parse_goal(command: str) -> dict[str, str]:
    words = command.lower().strip().split()
    if len(words) == 4 and words[0] == "bring" and words[2] == "to":
        return {"intent": "bring", "object": words[1], "destination": words[3]}
    return {"intent": "unknown", "message": "use: bring <object> to <station>"}


class MiniKitchenWorld:
    """A small kitchen where only the current station is observed."""

    def __init__(self, command: str = "bring mug to table", max_steps: int = 35) -> None:
        self.command = command
        self.goal = parse_goal(command)
        self.max_steps = max_steps
        self._figure: Any | None = None
        self._axis: Any | None = None
        self.reset()

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        self.time = 0
        self.location = "entry"
        self.holding: str | None = None
        self.open_containers = {"drawer"}
        self.inspected_containers: set[str] = set()
        self.revealed_objects: dict[str, str] = {}
        self.surface_objects = {
            "entry": [],
            "counter": ["plate"],
            "cabinet": [],
            "fridge": [],
            "table": [],
        }
        self.container_contents = {
            "drawer": ["spoon"],
            "cabinet": ["mug"],
            "fridge": ["apple"],
        }
        self.trajectory = [self.location]
        return self.observe()

    def observe(self) -> dict[str, Any]:
        containers_here = [
            {
                "name": name,
                "state": "open" if name in self.open_containers else "closed",
                "inspected": name in self.inspected_containers,
            }
            for name, station in CONTAINER_STATIONS.items()
            if station == self.location
        ]
        visible_objects = list(self.surface_objects[self.location])
        visible_objects.extend(
            obj
            for obj, station in self.revealed_objects.items()
            if station == self.location and obj != self.holding
        )
        return {
            "time": self.time,
            "location": self.location,
            "goal": dict(self.goal),
            "containers": containers_here,
            "visible_objects": visible_objects,
            "holding": self.holding,
            "open_containers": sorted(self.open_containers),
            "inspected_containers": sorted(self.inspected_containers),
        }

    def step(self, action: dict[str, str]) -> StepResult:
        self.time += 1
        action_type = action.get("type", "noop")
        info: dict[str, Any] = {
            "time": self.time,
            "action_type": action_type,
            "location": self.location,
            "success": False,
        }

        if action_type == "move":
            return self._move(action.get("to", ""), info)
        if action_type == "open_container":
            return self._open_container(action.get("container", ""), info)
        if action_type == "inspect":
            return self._inspect(action.get("container", ""), action.get("target", ""), info)
        if action_type == "pick":
            return self._pick(action.get("object", ""), info)
        if action_type == "place":
            return self._place(action.get("object", ""), action.get("to", ""), info)

        info["failure"] = Failure("invalid_action", f"unknown action type: {action_type}", True)
        return StepResult(self.observe(), -0.05, False, info)

    def _move(self, target: str, info: dict[str, Any]) -> StepResult:
        if target not in STATION_POSITIONS:
            info["failure"] = Failure("invalid_station", "unknown kitchen station", True)
            return StepResult(self.observe(), -0.06, False, info)
        self.location = target
        self.trajectory.append(target)
        info["location"] = self.location
        return StepResult(self.observe(), -0.01, False, info)

    def _open_container(self, container: str, info: dict[str, Any]) -> StepResult:
        if not self._container_here(container):
            info["failure"] = Failure("container_not_here", "container is not at this station", True)
            return StepResult(self.observe(), -0.05, False, info)
        self.open_containers.add(container)
        info["container"] = container
        info["open_containers"] = sorted(self.open_containers)
        return StepResult(self.observe(), -0.01, False, info)

    def _inspect(self, container: str, target: str, info: dict[str, Any]) -> StepResult:
        info["container"] = container
        if not self._container_here(container):
            info["failure"] = Failure("container_not_here", "container is not at this station", True)
            return StepResult(self.observe(), -0.05, False, info)
        if container not in self.open_containers:
            info["failure"] = Failure(
                "container_closed",
                "the object might be inside, but the container is closed",
                True,
            )
            return StepResult(self.observe(), -0.07, False, info)

        self.inspected_containers.add(container)
        contents = self.container_contents[container]
        if target in contents:
            self.revealed_objects[target] = self.location
            info["found_object"] = target
            return StepResult(self.observe(), 0.12, False, info)

        info["failure"] = Failure(
            "target_not_found",
            "the inspected container did not contain the goal object",
            True,
        )
        return StepResult(self.observe(), -0.04, False, info)

    def _pick(self, obj: str, info: dict[str, Any]) -> StepResult:
        info["object"] = obj
        if self.holding is not None:
            info["failure"] = Failure("hands_full", "place the held object before picking", True)
            return StepResult(self.observe(), -0.05, False, info)
        if obj not in self.observe()["visible_objects"]:
            info["failure"] = Failure("object_not_visible", "object must be observed before picking", True)
            return StepResult(self.observe(), -0.06, False, info)

        self.holding = obj
        if obj in self.revealed_objects:
            del self.revealed_objects[obj]
        for objects in self.surface_objects.values():
            if obj in objects:
                objects.remove(obj)
        info["holding"] = self.holding
        return StepResult(self.observe(), 0.10, False, info)

    def _place(self, obj: str, destination: str, info: dict[str, Any]) -> StepResult:
        info["object"] = obj
        info["destination"] = destination
        if self.holding != obj:
            info["failure"] = Failure("wrong_object", "the requested object is not being held", True)
            return StepResult(self.observe(), -0.05, False, info)
        if self.location != destination:
            info["failure"] = Failure("wrong_location", "move to the destination before placing", True)
            return StepResult(self.observe(), -0.05, False, info)

        self.holding = None
        self.surface_objects[destination].append(obj)
        success = (
            self.goal.get("intent") == "bring"
            and self.goal.get("object") == obj
            and self.goal.get("destination") == destination
        )
        info["success"] = success
        return StepResult(self.observe(), 1.0 if success else 0.05, success, info)

    def _container_here(self, container: str) -> bool:
        return CONTAINER_STATIONS.get(container) == self.location

    def render(self, agent: "MiniKitchenAgent", info: dict[str, Any] | None = None) -> None:
        import matplotlib.pyplot as plt

        if self._figure is None or self._axis is None:
            plt.ion()
            self._figure, self._axis = plt.subplots(figsize=(6.0, 4.8))

        draw_minikitchen_scene(self._axis, self, agent, info)
        self._figure.canvas.draw_idle()
        plt.pause(0.05)


class MiniKitchenAgent:
    """Goal-conditioned policy with container memory and simple recovery."""

    def __init__(self, command: str = "bring mug to table") -> None:
        self.command = command
        self.goal = parse_goal(command)
        self.reset()

    def reset(self) -> None:
        self.container_memory: dict[str, dict[str, Any]] = {}
        self.object_memory: dict[str, str] = {}
        self.inspected: set[str] = set()
        self.needs_open: set[str] = set()
        self.search_index = 0
        self.state = "parse_goal"
        self.closed_failures = 0
        self.not_found_failures = 0
        self.open_count = 0
        self._last_integrated_time: int | None = None

    def act(self, obs: dict[str, Any]) -> dict[str, str]:
        self._integrate_observation(obs)
        if self.goal.get("intent") != "bring":
            self.state = "unsupported_goal"
            return {"type": "noop"}

        target_object = self.goal["object"]
        destination = self.goal["destination"]

        if obs.get("holding") == target_object:
            if obs["location"] != destination:
                self.state = "move_to_goal"
                return {"type": "move", "to": destination}
            self.state = "place_goal_object"
            return {"type": "place", "object": target_object, "to": destination}

        if self.object_memory.get(target_object) == obs["location"]:
            self.state = "pick_goal_object"
            return {"type": "pick", "object": target_object}

        if target_object in self.object_memory:
            self.state = "move_to_remembered_object"
            return {"type": "move", "to": self.object_memory[target_object]}

        container = self._next_container()
        if container is None:
            self.state = "no_more_containers"
            return {"type": "noop"}

        station = CONTAINER_STATIONS[container]
        if obs["location"] != station:
            self.state = "move_to_search_container"
            return {"type": "move", "to": station}

        if container in self.needs_open:
            self.state = "open_container"
            return {"type": "open_container", "container": container}

        self.state = "inspect_container"
        return {"type": "inspect", "container": container, "target": target_object}

    def update(self, obs: dict[str, Any], reward: float, info: dict[str, Any]) -> None:
        self._integrate_observation(obs)

        failure = info.get("failure")
        if isinstance(failure, Failure) and failure.kind == "container_closed":
            container = str(info["container"])
            self.needs_open.add(container)
            self.closed_failures += 1
            self.state = "remember_closed_container"
        elif isinstance(failure, Failure) and failure.kind == "target_not_found":
            container = str(info["container"])
            self.inspected.add(container)
            self.not_found_failures += 1
            if self.search_index < len(CONTAINER_ORDER) - 1:
                self.search_index += 1
            self.state = "update_after_not_found"
        elif info.get("action_type") == "open_container":
            container = str(info["container"])
            self.needs_open.discard(container)
            self.open_count += 1
            self.state = "inspect_open_container"
        elif "found_object" in info:
            obj = str(info["found_object"])
            self.object_memory[obj] = obs["location"]
            self.inspected.add(str(info["container"]))
            self.state = "remember_object_location"
        elif info.get("action_type") == "pick":
            self.state = "holding_goal_object"
        elif info.get("success"):
            self.state = "goal_satisfied"

        info["agent_state"] = self.state
        info["parsed_goal"] = dict(self.goal)
        info["memory_count"] = len(self.object_memory) + len(self.container_memory)
        info["object_memory"] = dict(self.object_memory)
        info["closed_failures"] = self.closed_failures
        info["not_found_failures"] = self.not_found_failures
        info["open_count"] = self.open_count

    def _integrate_observation(self, obs: dict[str, Any]) -> None:
        obs_time = int(obs.get("time", -1))
        if obs_time == self._last_integrated_time:
            return
        self._last_integrated_time = obs_time

        for container in obs.get("containers", []):
            self.container_memory[container["name"]] = {
                "station": obs["location"],
                "state": container["state"],
                "inspected": container["inspected"],
                "last_seen": obs_time,
            }
        for obj in obs.get("visible_objects", []):
            self.object_memory[str(obj)] = obs["location"]

    def _next_container(self) -> str | None:
        for index in range(self.search_index, len(CONTAINER_ORDER)):
            container = CONTAINER_ORDER[index]
            if container not in self.inspected:
                self.search_index = index
                return container
        return None


def draw_minikitchen_scene(
    ax: Any,
    env: MiniKitchenWorld,
    agent: MiniKitchenAgent,
    info: dict[str, Any] | None = None,
) -> None:
    """Draw stations, current observation, memory, and goal progress."""

    from matplotlib.patches import Circle, Rectangle

    info = {} if info is None else info
    ax.clear()
    ax.set_title("goal-conditioned minikitchen")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    for station, position in STATION_POSITIONS.items():
        face = "0.94"
        if station == env.location:
            face = "#d9ecff"
        if station == env.goal.get("destination"):
            face = "#e7f7df" if station != env.location else "#cdecc3"
        ax.add_patch(
            Rectangle(
                (position[0] - 0.085, position[1] - 0.060),
                0.17,
                0.12,
                facecolor=face,
                edgecolor="0.30",
            )
        )
        ax.text(position[0], position[1], station, ha="center", va="center", fontsize=8)

    path = np.asarray([STATION_POSITIONS[station] for station in env.trajectory])
    if len(path) > 1:
        ax.plot(path[:, 0], path[:, 1], color="tab:blue", linewidth=2, alpha=0.6)

    ax.add_patch(Circle(STATION_POSITIONS[env.location], 0.028, color="tab:blue", label="agent"))

    for container, station in CONTAINER_STATIONS.items():
        pos = STATION_POSITIONS[station] + np.array([0.0, -0.095])
        is_open = container in env.open_containers
        edge = "tab:green" if is_open else "tab:red"
        ax.add_patch(
            Rectangle(
                (pos[0] - 0.055, pos[1] - 0.025),
                0.11,
                0.05,
                facecolor="white",
                edgecolor=edge,
                linestyle="-" if is_open else "--",
            )
        )
        ax.text(pos[0], pos[1], container, ha="center", va="center", fontsize=7)

    object_locations: dict[str, str] = {}
    for station, objects in env.surface_objects.items():
        for obj in objects:
            object_locations[obj] = station
    for obj, station in env.revealed_objects.items():
        object_locations[obj] = station
    if env.holding is not None:
        object_locations[env.holding] = env.location

    for obj, station in object_locations.items():
        pos = STATION_POSITIONS[station] + np.array([0.055, 0.080])
        color = "tab:red" if obj == env.goal.get("object") else "0.45"
        ax.add_patch(Circle(pos, 0.018, color=color, alpha=0.85))
        ax.text(pos[0], pos[1] + 0.030, obj, ha="center", fontsize=7)

    memory_text = ", ".join(f"{obj}@{station}" for obj, station in agent.object_memory.items())
    if not memory_text:
        memory_text = "empty"
    status = (
        f"goal: {env.command}\n"
        f"step={env.time}  state={info.get('agent_state', agent.state)}\n"
        f"holding={env.holding}  memory={memory_text}\n"
        f"closed={agent.closed_failures}  not_found={agent.not_found_failures}  opened={agent.open_count}"
    )
    if "failure" in info:
        status += f"  failure={info['failure'].kind}"
    if info.get("success"):
        status += "  success"
    ax.text(0.02, 0.98, status, transform=ax.transAxes, va="top", fontsize=9)


def run(
    command: str = "bring mug to table",
    seed: int = 0,
    render: bool = True,
    max_steps: int = 35,
) -> Trace:
    env = MiniKitchenWorld(command=command, max_steps=max_steps)
    agent = MiniKitchenAgent(command)
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
    parser.add_argument("command", nargs="?", default="bring mug to table")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=35)
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
        f"closed={final_info.get('closed_failures', 0)} "
        f"not_found={final_info.get('not_found_failures', 0)} "
        f"opened={final_info.get('open_count', 0)} "
        f"memory={final_info.get('object_memory', {})} "
        f"failures={failures}"
    )

    if not args.no_render:
        import matplotlib.pyplot as plt

        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
