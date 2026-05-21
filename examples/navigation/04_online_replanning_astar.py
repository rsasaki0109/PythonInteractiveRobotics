"""Online A* replanning when new obstacles are observed.

The agent plans through UNKNOWN cells under the optimistic assumption that
they are free. When a hidden wall enters the lidar field of view it is
revealed as OCCUPIED, the path becomes invalid, and A* runs again on the
updated map.

Success: robot reaches the goal cell.
Failure: timeout (terminal).

Compare to `09_blocked_path_recovery.py`. That example triggers replanning
through an *execution failure* (the agent tried to step into a blocked
cell and got back a `Failure`). This example triggers replanning through
*passive observation* (the lidar reveals a wall before the agent ever
touches it).
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

from pir.core.types import Failure, Trace
from pir.planning import astar as grid_astar
from pir.sensors.fake_lidar import DIRECTIONS
from pir.worlds.grid_world import GridWorld2D, OCCUPIED


class AStarReplanningAgent:
    """Plan through unknown cells, then replan when observations invalidate the path."""

    def reset(self) -> None:
        self.current_path: list[tuple[int, int]] = []
        self.replan_count = 0
        self.path_invalidations = 0
        self.state = "need_plan"
        self.last_failure: Failure | None = None

    def act(self, obs: dict[str, Any]) -> str:
        robot = obs["robot"]
        goal = obs["goal"]
        known_map = obs["known_map"]

        if self._path_is_invalid(robot, known_map):
            self.path_invalidations += 1
            self.current_path = []

        if not self.current_path:
            self.current_path = astar(known_map, robot, goal)
            self.replan_count += 1
            self.state = "replan"
        else:
            self.state = "follow_path"

        if len(self.current_path) < 2:
            self.state = "no_path"
            return "stay"

        next_cell = self.current_path[1]
        if known_map[next_cell] == OCCUPIED:
            self.path_invalidations += 1
            self.current_path = astar(known_map, robot, goal)
            self.replan_count += 1
            if len(self.current_path) < 2:
                self.state = "no_path"
                return "stay"
            next_cell = self.current_path[1]

        return direction_to(robot, next_cell)

    def update(self, obs: dict[str, Any], reward: float, info: dict[str, Any]) -> None:
        failure = info.get("failure")
        self.last_failure = failure if isinstance(failure, Failure) else None

        robot = obs["robot"]
        if self.current_path and self.current_path[0] != robot:
            while self.current_path and self.current_path[0] != robot:
                self.current_path.pop(0)

        if self.last_failure is not None and self.last_failure.recoverable:
            self.current_path = []
            self.state = "recover_and_replan"

    def _path_is_invalid(self, robot: tuple[int, int], known_map: Any) -> bool:
        if not self.current_path:
            return True
        if robot not in self.current_path:
            return True
        return any(known_map[cell] == OCCUPIED for cell in self.current_path)


def astar(
    known_map: Any,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]]:
    """A* over the currently known map, treating unknown cells as traversable."""

    return grid_astar(np.asarray(known_map) != OCCUPIED, start, goal)


def direction_to(start: tuple[int, int], end: tuple[int, int]) -> str:
    delta = (end[0] - start[0], end[1] - start[1])
    for direction, direction_delta in DIRECTIONS.items():
        if delta == direction_delta:
            return direction
    return "stay"


def run(seed: int = 0, render: bool = True, max_steps: int = 100) -> Trace:
    env = GridWorld2D(seed=seed, lidar_range=3, max_steps=max_steps)
    agent = AStarReplanningAgent()
    obs = env.reset(seed=seed)
    agent.reset()
    trace = Trace()

    for _ in range(max_steps):
        action = agent.act(obs)
        result = env.step(action)
        obs, reward, done, info = result.as_tuple()
        agent.update(obs, reward, info)
        info["agent_state"] = agent.state
        info["replan_count"] = agent.replan_count
        info["path_invalidations"] = agent.path_invalidations
        info["planned_path_length"] = len(agent.current_path)
        trace.append(obs, action, reward, info)

        if render:
            env.render(agent=agent, info=info)

        if done:
            break

    return trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    trace = run(seed=args.seed, render=not args.no_render, max_steps=args.max_steps)
    success = bool(trace.infos and trace.infos[-1].get("success"))
    failures = [failure.kind for failure in trace.failures()]
    replan_count = trace.infos[-1].get("replan_count", 0) if trace.infos else 0
    invalidations = trace.infos[-1].get("path_invalidations", 0) if trace.infos else 0
    print(
        f"success={success} steps={len(trace.actions)} "
        f"replans={replan_count} invalidations={invalidations} failures={failures}"
    )

    if not args.no_render:
        import matplotlib.pyplot as plt

        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
