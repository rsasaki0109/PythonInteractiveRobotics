"""Navigate a grid with several other agents, each chasing its own goal.

The robot shares the grid with two scripted agents that each move greedily
toward their own goal. The robot observes their positions, predicts where each
one will be on the next step, and runs A* over a map that treats both current
and predicted-next cells of every other agent as occupied.

This differs from `03_dynamic_obstacle_avoidance.py`, which avoids a single
random-walking obstacle, and from `08_interactive_mpc.py`, which replans a
short continuous trajectory around one moving body. Here the obstacles have
their own goal-directed policies, and the robot reasons about multiple
predicted next positions at once.
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
from pir.planning import astar as grid_astar


UNKNOWN = -1
FREE = 0
OCCUPIED = 1

DIRECTIONS: dict[str, tuple[int, int]] = {
    "N": (-1, 0),
    "S": (1, 0),
    "E": (0, 1),
    "W": (0, -1),
}

DEFAULT_OTHER_AGENTS: tuple[dict[str, Any], ...] = (
    {"id": "A", "start": (1, 4), "goal": (11, 4), "color": "tab:orange"},
    {"id": "B", "start": (11, 7), "goal": (1, 7), "color": "tab:purple"},
)


def _greedy_step(pos: tuple[int, int], goal: tuple[int, int]) -> tuple[int, int]:
    """Move one cardinal cell toward the goal. Tie-break prefers the row axis."""

    dr = goal[0] - pos[0]
    dc = goal[1] - pos[1]
    if dr == 0 and dc == 0:
        return pos
    if abs(dr) >= abs(dc) and dr != 0:
        return (pos[0] + (1 if dr > 0 else -1), pos[1])
    return (pos[0], pos[1] + (1 if dc > 0 else -1))


class MultiAgentNavigationWorld:
    """13x13 grid with the robot and two goal-seeking other agents."""

    def __init__(
        self,
        *,
        seed: int | None = 0,
        max_steps: int = 80,
        height: int = 13,
        width: int = 13,
        robot_start: tuple[int, int] = (5, 1),
        robot_goal: tuple[int, int] = (5, 11),
        other_agents: tuple[dict[str, Any], ...] = DEFAULT_OTHER_AGENTS,
    ) -> None:
        self.seed = seed
        self.height = height
        self.width = width
        self.robot_start = tuple(robot_start)
        self.robot_goal = tuple(robot_goal)
        self.other_agents_spec = tuple({**a, "start": tuple(a["start"]), "goal": tuple(a["goal"])} for a in other_agents)
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
        self.robot = self.robot_start
        self.others: list[dict[str, Any]] = []
        for spec in self.other_agents_spec:
            self.others.append(
                {
                    "id": spec["id"],
                    "position": spec["start"],
                    "goal": spec["goal"],
                    "color": spec.get("color", "tab:gray"),
                }
            )
        self.time = 0
        self.trajectory: list[tuple[int, int]] = [self.robot]
        return self.observe()

    def observe(self) -> dict[str, Any]:
        return {
            "time": self.time,
            "robot": self.robot,
            "goal": self.robot_goal,
            "others": [
                {
                    "id": o["id"],
                    "position": o["position"],
                    "goal": o["goal"],
                    "color": o["color"],
                }
                for o in self.others
            ],
            "height": self.height,
            "width": self.width,
        }

    def step(self, action: Any) -> StepResult:
        self.time += 1
        direction = action if isinstance(action, str) else action.get("direction") if isinstance(action, dict) else None
        info: dict[str, Any] = {"time": self.time, "direction": direction, "success": False}

        if direction in (None, "stay"):
            new_robot = self.robot
        elif direction not in DIRECTIONS:
            info["failure"] = Failure("invalid_direction", f"unknown direction: {direction}", True)
            return StepResult(self.observe(), -0.05, False, info)
        else:
            dr, dc = DIRECTIONS[direction]
            new_robot = (self.robot[0] + dr, self.robot[1] + dc)
            if not (0 <= new_robot[0] < self.height and 0 <= new_robot[1] < self.width):
                info["failure"] = Failure("out_of_bounds", f"step would leave grid: {new_robot}", True)
                return StepResult(self.observe(), -0.1, False, info)

        # Apply robot move, but reject if it lands on another agent's current cell.
        if new_robot in {o["position"] for o in self.others}:
            info["failure"] = Failure(
                "blocked_by_agent",
                f"cell occupied by another agent: {new_robot}",
                True,
            )
            self.trajectory.append(self.robot)
        else:
            self.robot = new_robot
            self.trajectory.append(self.robot)

        # Advance other agents greedily. An agent waits if its move would land
        # on the robot's new position or on another agent's current cell.
        occupied_after = {self.robot}
        new_others: list[dict[str, Any]] = []
        for o in self.others:
            next_pos = _greedy_step(o["position"], o["goal"])
            if next_pos == o["position"] or next_pos in occupied_after:
                new_others.append(o)
                occupied_after.add(o["position"])
            else:
                new_others.append({**o, "position": next_pos})
                occupied_after.add(next_pos)
        self.others = new_others

        info["agent_positions"] = [list(o["position"]) for o in self.others]
        info["robot_position"] = list(self.robot)

        if self.robot == self.robot_goal:
            info["success"] = True
            return StepResult(self.observe(), 1.0, True, info)
        return StepResult(self.observe(), -0.02, False, info)

    def render(self, agent: Any | None = None, info: dict[str, Any] | None = None) -> None:
        import matplotlib.pyplot as plt

        if self._fig is None or self._ax is None:
            plt.ion()
            self._fig, self._ax = plt.subplots(figsize=(5.6, 5.2))
        ax = self._ax
        ax.clear()
        draw_multi_agent_avoidance_scene(ax, self, agent, info)
        self._fig.canvas.draw_idle()
        plt.pause(0.1)


def astar(
    occupancy: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]]:
    return grid_astar(occupancy != OCCUPIED, start, goal)


def _direction_to(start: tuple[int, int], end: tuple[int, int]) -> str:
    delta = (end[0] - start[0], end[1] - start[1])
    for direction, vec in DIRECTIONS.items():
        if vec == delta:
            return direction
    return "stay"


class MultiAgentAvoidanceAgent:
    """Predict each other agent's next move and A* around the predictions."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.current_path: list[tuple[int, int]] = []
        self.replan_count: int = 0
        self.avoidance_count: int = 0
        self.wait_count: int = 0
        self.predicted_next: dict[str, tuple[int, int]] = {}
        self.state: str = "navigate"
        self.last_blocked_cells: list[tuple[int, int]] = []

    def act(self, obs: dict[str, Any]) -> str:
        robot = tuple(obs["robot"])
        goal = tuple(obs["goal"])
        others = obs["others"]
        height = int(obs["height"])
        width = int(obs["width"])

        # predict each other agent's next position
        self.predicted_next = {}
        for o in others:
            pos = tuple(o["position"])
            ogoal = tuple(o["goal"])
            self.predicted_next[o["id"]] = _greedy_step(pos, ogoal)

        # Build occupancy that blocks current AND predicted-next cells of others.
        occupancy = np.zeros((height, width), dtype=int)
        blocked: list[tuple[int, int]] = []
        for o in others:
            pos = tuple(o["position"])
            occupancy[pos] = OCCUPIED
            blocked.append(pos)
        for nxt in self.predicted_next.values():
            if 0 <= nxt[0] < height and 0 <= nxt[1] < width:
                occupancy[nxt] = OCCUPIED
                blocked.append(nxt)
        self.last_blocked_cells = blocked

        # If the robot's current cell got blocked (because robot equals a predicted cell)
        # leave it free so we can plan from it.
        occupancy[robot] = FREE

        # Plan / replan
        need_replan = (
            not self.current_path
            or self.current_path[-1] != goal
            or robot not in self.current_path
            or any(occupancy[cell] == OCCUPIED for cell in self.current_path[1:])
        )
        if need_replan:
            self.current_path = astar(occupancy, robot, goal)
            self.replan_count += 1

        # Follow path
        while self.current_path and self.current_path[0] != robot:
            self.current_path.pop(0)

        if len(self.current_path) < 2:
            # No move available; wait
            self.wait_count += 1
            self.state = "wait"
            return "stay"

        next_cell = self.current_path[1]
        # Count an avoidance only when the naive greedy step would have stepped
        # onto a current or predicted-next cell of another agent.
        naive_next = _greedy_step(robot, goal)
        blocked_set = set(self.last_blocked_cells)
        if next_cell != naive_next and naive_next in blocked_set:
            self.avoidance_count += 1
        self.state = "navigate"
        return _direction_to(robot, next_cell)

    def update(self, obs: dict[str, Any], reward: float, info: dict[str, Any]) -> None:
        if info.get("success"):
            self.state = "succeeded"


def draw_multi_agent_avoidance_scene(
    ax: Any,
    env: MultiAgentNavigationWorld,
    agent: MultiAgentAvoidanceAgent | None,
    info: dict[str, Any] | None,
) -> None:
    ax.set_xlim(-0.5, env.width - 0.5)
    ax.set_ylim(env.height - 0.5, -0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.set_xticks(np.arange(-0.5, env.width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.height, 1), minor=True)
    ax.grid(which="minor", color="0.85", linewidth=0.4)

    # planned path
    if agent is not None and len(agent.current_path) > 1:
        rows = [c[0] for c in agent.current_path]
        cols = [c[1] for c in agent.current_path]
        ax.plot(cols, rows, "--", color="tab:purple", linewidth=1.4, alpha=0.7)

    # robot trajectory
    rows = [c[0] for c in env.trajectory]
    cols = [c[1] for c in env.trajectory]
    ax.plot(cols, rows, color="tab:blue", linewidth=1.4, alpha=0.8)

    # other agents and their goals
    for o in env.others:
        r, c = o["position"]
        ax.plot(c, r, marker="o", color=o["color"], markersize=10)
        gr, gc = o["goal"]
        ax.plot(gc, gr, marker="*", color=o["color"], markersize=10, alpha=0.4)
        if agent is not None and o["id"] in agent.predicted_next:
            pr, pc = agent.predicted_next[o["id"]]
            ax.plot(pc, pr, marker="x", color=o["color"], markersize=10, mew=2, alpha=0.6)

    # robot and goal
    ax.plot(env.robot[1], env.robot[0], marker="o", color="tab:blue", markersize=12)
    ax.plot(env.robot_goal[1], env.robot_goal[0], marker="*", color="tab:green", markersize=18)

    status_parts: list[str] = [f"step={env.time}"]
    if agent is not None:
        status_parts.append(f"state={agent.state}")
        status_parts.append(f"replans={agent.replan_count}")
        status_parts.append(f"avoidances={agent.avoidance_count}")
        status_parts.append(f"waits={agent.wait_count}")
    if info is not None and "failure" in info:
        status_parts.append(f"failure={info['failure'].kind}")
    if info is not None and info.get("success"):
        status_parts.append("success")
    ax.set_title("multi-agent avoidance")
    ax.text(0.02, 0.98, "  ".join(status_parts), transform=ax.transAxes, va="top", fontsize=8)


def run(seed: int = 0, render: bool = True, max_steps: int = 80) -> Trace:
    env = MultiAgentNavigationWorld(seed=seed, max_steps=max_steps)
    agent = MultiAgentAvoidanceAgent()
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
        info["avoidance_count"] = agent.avoidance_count
        info["wait_count"] = agent.wait_count
        info["predicted_next"] = {
            agent_id: list(pos) for agent_id, pos in agent.predicted_next.items()
        }
        trace.append(obs, action, reward, info)

        if render:
            env.render(agent=agent, info=info)
        if done:
            break

    return trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    trace = run(seed=args.seed, render=not args.no_render, max_steps=args.max_steps)
    success = bool(trace.infos and trace.infos[-1].get("success"))
    failures = [failure.kind for failure in trace.failures()]
    final = trace.infos[-1] if trace.infos else {}
    print(
        f"success={success} steps={len(trace.actions)} "
        f"replans={final.get('replan_count', 0)} "
        f"avoidances={final.get('avoidance_count', 0)} "
        f"waits={final.get('wait_count', 0)} failures={failures}"
    )

    if not args.no_render:
        import matplotlib.pyplot as plt

        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
