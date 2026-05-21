"""Parse a small goal command and execute a failure-aware pick loop."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pir.core.types import Failure, Trace
from pir.worlds.tabletop_2d import Tabletop2D


SUPPORTED_COMMANDS = {
    "go to the red object",
    "find the key",
    "pick the red block",
    "bring the key to the door",
    "find the mug and bring it to the table",
    "find the red block and pick it",
}


def parse_goal_command(command: str) -> dict[str, str]:
    """Parse the initial controlled-language command set."""

    normalized = " ".join(command.lower().strip().split())
    if normalized not in SUPPORTED_COMMANDS:
        return {
            "intent": "unknown",
            "message": "unsupported command",
            "command": command,
        }

    if normalized == "find the red block and pick it":
        return {"intent": "find_and_pick", "object": "block", "color": "red"}
    if normalized == "pick the red block":
        return {"intent": "pick", "object": "block", "color": "red"}
    if normalized == "go to the red object":
        return {"intent": "go_to_object", "object": "object", "color": "red"}
    if normalized == "find the key":
        return {"intent": "find", "object": "key", "color": "any"}
    if normalized == "bring the key to the door":
        return {"intent": "bring_to", "object": "key", "target": "door", "color": "any"}
    if normalized == "find the mug and bring it to the table":
        return {"intent": "find_and_bring", "object": "mug", "target": "table", "color": "any"}

    return {
        "intent": "unknown",
        "message": "unsupported command",
        "command": command,
    }


class GoalCommandPickAgent:
    """Goal-conditioned tabletop agent with explicit memory and retry state."""

    def __init__(self, command: str) -> None:
        self.command = command
        self.goal = parse_goal_command(command)
        self.search_viewpoints = [
            np.array([0.84, 0.54]),
            np.array([0.20, 0.84]),
            np.array([0.78, 0.22]),
        ]
        self.offset_schedule = [
            np.array([-0.18, 0.00]),
            np.array([0.00, 0.00]),
            np.array([0.04, 0.00]),
            np.array([-0.04, 0.00]),
            np.array([0.00, 0.04]),
            np.array([0.00, -0.04]),
        ]
        self.reset()

    def reset(self) -> None:
        self.state = "parse_goal"
        self.memory: list[dict[str, Any]] = []
        self.belief_mean: np.ndarray | None = None
        self.belief_radius = 0.16
        self.search_count = 0
        self.retry_count = 0
        self._last_integrated_time: int | None = None

    def act(self, obs: dict[str, Any]) -> dict[str, Any]:
        if self.goal["intent"] != "find_and_pick":
            self.state = "unsupported_goal"
            return {"type": "noop"}

        self._integrate_observation(obs)

        if self.search_count == 0 or self.belief_mean is None:
            self.state = "search_object"
            target = self.search_viewpoints[self.search_count % len(self.search_viewpoints)]
            self.search_count += 1
            return {"type": "look", "target": target}

        self.state = "pick_object" if self.retry_count == 0 else "retry_pick"
        offset = self.offset_schedule[
            min(self.retry_count, len(self.offset_schedule) - 1)
        ]
        pick_position = np.clip(self.belief_mean + offset, 0.0, 1.0)
        return {"type": "pick", "position": pick_position}

    def update(self, obs: dict[str, Any], reward: float, info: dict[str, Any]) -> None:
        self._integrate_observation(obs)

        if self.goal["intent"] != "find_and_pick":
            return

        failure = info.get("failure")
        if isinstance(failure, Failure) and failure.kind == "grasp_miss":
            self.retry_count += 1
            self.belief_radius = min(0.22, self.belief_radius + 0.03)
            self.state = "update_belief_after_failure"
        elif info.get("success"):
            self.belief_radius = 0.025
            self.state = "done"

    def _integrate_observation(self, obs: dict[str, Any]) -> None:
        obs_time = int(obs.get("time", -1))
        if obs_time == self._last_integrated_time:
            return
        self._last_integrated_time = obs_time

        matching = [
            detection
            for detection in obs.get("detections", [])
            if self._matches_goal(detection)
        ]
        if not matching:
            return

        detection = matching[0]
        position = np.asarray(detection["position"], dtype=float)
        confidence = float(detection.get("confidence", 0.5))
        self.memory.append(
            {
                "time": obs_time,
                "position": position.copy(),
                "confidence": confidence,
            }
        )

        if self.belief_mean is None:
            self.belief_mean = position.copy()
        else:
            alpha = np.clip(0.40 + 0.40 * confidence, 0.40, 0.80)
            self.belief_mean = alpha * self.belief_mean + (1.0 - alpha) * position

        self.belief_radius = max(0.035, self.belief_radius * 0.72)
        if self.state in {"parse_goal", "search_object"}:
            self.state = "update_belief_from_detection"

    def _matches_goal(self, detection: dict[str, Any]) -> bool:
        object_ok = detection.get("name") == self.goal.get("object")
        goal_color = self.goal.get("color")
        detected_color = str(detection.get("color", "")).replace("tab:", "")
        color_ok = goal_color in {"any", detected_color}
        return object_ok and color_ok


def run(
    command: str = "find the red block and pick it",
    seed: int = 3,
    render: bool = True,
    max_steps: int = 40,
) -> Trace:
    env = Tabletop2D(seed=seed)
    agent = GoalCommandPickAgent(command)
    obs = env.reset(seed=seed)
    agent.reset()
    trace = Trace()

    if agent.goal["intent"] == "unknown":
        trace.append(
            obs,
            {"type": "parse_goal", "command": command},
            0.0,
            {
                "command": command,
                "parsed_goal": agent.goal,
                "agent_state": "unsupported_goal",
                "failure": Failure("unsupported_goal", "unsupported command", False),
            },
        )
        return trace

    for _ in range(max_steps):
        action = agent.act(obs)
        result = env.step(action)
        obs, reward, done, info = result.as_tuple()
        agent.update(obs, reward, info)
        info["command"] = command
        info["parsed_goal"] = dict(agent.goal)
        info["agent_state"] = agent.state
        info["retry_count"] = agent.retry_count
        info["search_count"] = agent.search_count
        info["memory_count"] = len(agent.memory)
        info["belief_radius"] = agent.belief_radius
        trace.append(obs, action, reward, info)

        if render:
            env.render(agent=agent, info=info)

        if done:
            break

    return trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        nargs="?",
        default="find the red block and pick it",
    )
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    trace = run(
        command=args.command,
        seed=args.seed,
        render=not args.no_render,
        max_steps=args.max_steps,
    )
    picked = bool(trace.infos and trace.infos[-1].get("success"))
    failures = [failure.kind for failure in trace.failures()]
    state = trace.infos[-1].get("agent_state", "none") if trace.infos else "none"
    print(
        f"picked={picked} steps={len(trace.actions)} state={state} "
        f"failures={failures}"
    )

    if not args.no_render:
        import matplotlib.pyplot as plt

        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
