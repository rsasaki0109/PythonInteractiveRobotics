"""Household task agent that clarifies, plans, stays safe, retries, and replans.

The command "put the block away" is intentionally underspecified. The robot
first asks which block, plans to the selected object, refuses an unsafe nominal
step, replans around the unsafe floor patch, retries after one grasp miss, then
accepts a human correction during delivery and replans before placing the block.

Success: the requested block is stored after clarification, safety filtering,
grasp retry, and human-corrected replanning.
Failure: ambiguous_goal (recoverable), unsafe_nominal_step (recoverable),
grasp_miss (recoverable), human_correction (recoverable), invalid_direction
(recoverable), invalid_target (recoverable), timeout (terminal).
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
from pir.planning import astar as grid_astar


FREE = 0
OCCUPIED = 1

Cell = tuple[int, int]

DIRECTIONS: dict[str, Cell] = {
    "north": (-1, 0),
    "south": (1, 0),
    "west": (0, -1),
    "east": (0, 1),
}

DRAW_COLORS = {
    "red": "tab:red",
    "blue": "tab:blue",
}

DEFAULT_MAP: tuple[tuple[int, ...], ...] = (
    (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
    (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
)

DEFAULT_SAFETY_ZONE: tuple[Cell, ...] = ((5, 1), (5, 2), (5, 3))
DEFAULT_HUMAN_ZONE: tuple[Cell, ...] = ((1, 6), (1, 7), (2, 6), (2, 7))


def parse_household_command(command: str) -> dict[str, Any]:
    """Parse a tiny household command grammar."""

    normalized = " ".join(command.lower().strip().split())
    if normalized == "put the block away":
        return {
            "intent": "put_away",
            "object": "block",
            "color": None,
            "destination": "storage",
            "ambiguous": True,
        }
    for color in ("red", "blue"):
        if normalized in {
            f"put the {color} block away",
            f"put {color} block away",
            f"store the {color} block",
            f"store {color} block",
        }:
            return {
                "intent": "put_away",
                "object": "block",
                "color": color,
                "destination": "storage",
                "ambiguous": False,
            }
    return {
        "intent": "unknown",
        "command": command,
        "message": "use: put the block away | put the red block away | put the blue block away",
    }


class HouseholdTaskWorld:
    """Grid household world with objects, storage, safety, and human correction."""

    def __init__(
        self,
        *,
        command: str = "put the block away",
        answer: str = "red",
        static_map: tuple[tuple[int, ...], ...] = DEFAULT_MAP,
        safety_zone: tuple[Cell, ...] = DEFAULT_SAFETY_ZONE,
        human_zone: tuple[Cell, ...] = DEFAULT_HUMAN_ZONE,
        start: Cell = (7, 1),
        storage_cell: Cell = (1, 10),
        max_steps: int = 80,
    ) -> None:
        self.command = command
        self.answer = answer.lower().strip()
        self.static_map = np.asarray(static_map, dtype=int)
        self.walkable = self.static_map == FREE
        self.height, self.width = self.static_map.shape
        self.safety_zone = set(safety_zone)
        self.human_zone = set(human_zone)
        self.start = start
        self.storage_cell = storage_cell
        self.max_steps = max_steps
        self._fig: Any | None = None
        self._ax: Any | None = None
        self.reset()

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        _ = seed
        self.robot = self.start
        self.step_count = 0
        self.held_color: str | None = None
        self.stored_color: str | None = None
        self.last_question: str | None = None
        self.last_answer: str | None = None
        self.last_failed_cell: Cell | None = None
        self.last_human_cell: Cell | None = None
        self.trajectory: list[Cell] = [self.start]
        self.failed_grasp_colors: set[str] = set()
        self.blocks: dict[str, dict[str, Any]] = {
            "red": {
                "name": "block",
                "color": "red",
                "cell": (3, 3),
                "picked": False,
                "stored": False,
            },
            "blue": {
                "name": "block",
                "color": "blue",
                "cell": (5, 9),
                "picked": False,
                "stored": False,
            },
        }
        return self.observe()

    def observe(self) -> dict[str, Any]:
        visual_tokens: list[dict[str, Any]] = []
        for color, block in self.blocks.items():
            if block["picked"] or block["stored"]:
                continue
            visual_tokens.append(
                {
                    "name": block["name"],
                    "color": color,
                    "cell": tuple(block["cell"]),
                    "confidence": 0.92,
                }
            )
        return {
            "time": self.step_count,
            "command": self.command,
            "robot": self.robot,
            "storage_cell": self.storage_cell,
            "walkable": self.walkable.copy(),
            "visual_tokens": tuple(visual_tokens),
            "held_color": self.held_color,
            "stored_color": self.stored_color,
            "safety_zone": tuple(sorted(self.safety_zone)),
            "human_zone": tuple(sorted(self.human_zone)),
            "last_question": self.last_question,
            "last_answer": self.last_answer,
        }

    def step(self, action: dict[str, Any]) -> StepResult:
        self.step_count += 1
        action_type = str(action.get("type", "noop"))
        info: dict[str, Any] = {
            "action_type": action_type,
            "success": False,
            "robot": self.robot,
            "held_color": self.held_color,
            "stored_color": self.stored_color,
        }

        if action_type == "ask":
            question = str(action.get("question", "Which block?"))
            choices = tuple(action.get("choices", ()))
            self.last_question = question
            self.last_answer = self.answer
            info.update(
                {
                    "question": question,
                    "choices": choices,
                    "answer": self.answer,
                    "failure": Failure(
                        "ambiguous_goal",
                        "The command matched multiple blocks, so the robot asked which color.",
                        recoverable=True,
                    ),
                }
            )
            return self._finish(-0.02, False, info)

        if action_type == "move":
            direction = str(action.get("direction", "stay"))
            if direction == "stay":
                return self._finish(-0.03, False, info)
            if direction not in DIRECTIONS:
                info["failure"] = Failure(
                    "invalid_direction",
                    f"unknown direction: {direction}",
                    recoverable=True,
                )
                return self._finish(-0.10, False, info)

            dr, dc = DIRECTIONS[direction]
            next_cell = (self.robot[0] + dr, self.robot[1] + dc)
            info["next_cell"] = next_cell
            if not self._in_bounds(next_cell) or not self.walkable[next_cell]:
                info["failure"] = Failure(
                    "collision",
                    f"blocked household cell: {next_cell}",
                    recoverable=True,
                )
                return self._finish(-0.20, False, info)

            if next_cell in self.safety_zone:
                self.last_failed_cell = next_cell
                info.update(
                    {
                        "blocked_cells": tuple(sorted(self.safety_zone)),
                        "filtered_cell": next_cell,
                        "failure": Failure(
                            "unsafe_nominal_step",
                            "Runtime safety check rejected a step through the wet floor zone.",
                            recoverable=True,
                        ),
                    }
                )
                return self._finish(-0.16, False, info)

            if self.held_color is not None and next_cell in self.human_zone:
                self.last_human_cell = next_cell
                info.update(
                    {
                        "corrected_cells": tuple(sorted(self.human_zone)),
                        "correction_cell": next_cell,
                        "failure": Failure(
                            "human_correction",
                            "Human redirected the robot away from the fragile display shelf.",
                            recoverable=True,
                        ),
                    }
                )
                return self._finish(-0.18, False, info)

            self.robot = next_cell
            self.trajectory.append(self.robot)
            info["robot"] = self.robot
            return self._finish(-0.02, False, info)

        if action_type == "pick":
            color = str(action.get("color", "")).lower()
            block = self.blocks.get(color)
            if block is None or block["stored"]:
                info["failure"] = Failure(
                    "invalid_target",
                    f"Cannot pick target color: {color}",
                    recoverable=True,
                )
                return self._finish(-0.10, False, info)
            if self.held_color is not None:
                info["failure"] = Failure(
                    "hands_full",
                    "The robot must place the held object before picking another one.",
                    recoverable=True,
                )
                return self._finish(-0.10, False, info)
            if self.robot != tuple(block["cell"]):
                info["failure"] = Failure(
                    "target_not_reached",
                    "The robot must navigate to the block before grasping.",
                    recoverable=True,
                )
                return self._finish(-0.08, False, info)
            if color not in self.failed_grasp_colors:
                self.failed_grasp_colors.add(color)
                info.update(
                    {
                        "target_color": color,
                        "failure": Failure(
                            "grasp_miss",
                            "The first grasp was too coarse, so the robot will retry.",
                            recoverable=True,
                        ),
                    }
                )
                return self._finish(-0.15, False, info)

            block["picked"] = True
            self.held_color = color
            info.update({"picked_color": color, "held_color": color, "pick_success": True})
            return self._finish(0.20, False, info)

        if action_type == "place":
            if self.held_color is None:
                info["failure"] = Failure(
                    "empty_gripper",
                    "The robot cannot place because it is not holding an object.",
                    recoverable=True,
                )
                return self._finish(-0.10, False, info)
            if self.robot != self.storage_cell:
                info["failure"] = Failure(
                    "wrong_location",
                    "The robot must reach storage before placing the block.",
                    recoverable=True,
                )
                return self._finish(-0.08, False, info)

            color = self.held_color
            self.blocks[color]["stored"] = True
            self.held_color = None
            self.stored_color = color
            info.update(
                {
                    "placed_color": color,
                    "stored_color": color,
                    "held_color": None,
                    "success": True,
                }
            )
            return self._finish(1.0, True, info)

        info["failure"] = Failure("invalid_action", f"unknown action: {action_type}", True)
        return self._finish(-0.06, False, info)

    def _finish(self, reward: float, done: bool, info: dict[str, Any]) -> StepResult:
        if not done and self.step_count >= self.max_steps:
            done = True
            info["failure"] = Failure(
                "timeout",
                "Household task did not finish before max_steps.",
                recoverable=False,
            )
        info["robot"] = self.robot
        info["held_color"] = self.held_color
        info["stored_color"] = self.stored_color
        info["trajectory_length"] = len(self.trajectory)
        return StepResult(self.observe(), reward, done, info)

    def _in_bounds(self, cell: Cell) -> bool:
        return 0 <= cell[0] < self.height and 0 <= cell[1] < self.width

    def render(self, agent: "HouseholdTaskAgent", info: dict[str, Any]) -> None:
        import matplotlib.pyplot as plt

        if self._fig is None or self._ax is None:
            plt.ion()
            self._fig, self._ax = plt.subplots(figsize=(6.3, 4.9))
        self._ax.clear()
        draw_household_task_scene(self._ax, self, agent, info)
        self._fig.canvas.draw_idle()
        plt.pause(0.001)


class HouseholdTaskAgent:
    """A tiny task-level policy that composes clarification and recovery loops."""

    def __init__(self, command: str) -> None:
        self.command = command
        self.initial_goal = parse_household_command(command)
        self.reset()

    def reset(self) -> None:
        self.goal = dict(self.initial_goal)
        self.state = "parse_command"
        self.phase = "resolve_goal"
        self.visual_memory: dict[str, dict[str, Any]] = {}
        self.current_path: list[Cell] = []
        self.path_goal: Cell | None = None
        self.blocked_cells: set[Cell] = set()
        self.corrected_cells: set[Cell] = set()
        self.question_count = 0
        self.clarification_count = 0
        self.safety_filter_count = 0
        self.retry_count = 0
        self.human_correction_count = 0
        self.replan_count = 0
        self.last_question: str | None = None
        self.last_answer: str | None = None
        self.last_failure: Failure | None = None

    def act(self, obs: dict[str, Any]) -> dict[str, Any]:
        self._integrate_tokens(obs)
        if self.goal["intent"] == "unknown":
            self.state = "unsupported_goal"
            return {"type": "noop"}

        if self.goal.get("color") is None:
            choices = tuple(sorted(self.visual_memory))
            self.phase = "clarify"
            self.state = "ask_clarification"
            self.question_count += 1
            self.last_question = f"Which block should I put away, {', '.join(choices)}?"
            return {"type": "ask", "question": self.last_question, "choices": choices}

        color = str(self.goal["color"])
        held_color = obs.get("held_color")
        robot = tuple(obs["robot"])

        if held_color is None:
            target = self._target_cell(color)
            if target is None:
                self.state = "target_not_visible"
                return {"type": "move", "direction": "stay"}
            if robot != target:
                self.phase = "navigate_to_pick"
                direction = self._next_direction(obs, target)
                self.state = "plan_to_pick" if self.replan_count == 0 else "follow_safe_plan"
                return {"type": "move", "direction": direction}
            self.phase = "pick"
            self.state = "retry_grasp" if self.retry_count else "coarse_grasp"
            return {"type": "pick", "color": color}

        storage = tuple(obs["storage_cell"])
        if robot != storage:
            self.phase = "deliver"
            direction = self._next_direction(obs, storage)
            self.state = (
                "replan_after_human_correction"
                if self.human_correction_count
                else "plan_to_storage"
            )
            return {"type": "move", "direction": direction}

        self.phase = "place"
        self.state = "place_in_storage"
        return {"type": "place"}

    def update(self, obs: dict[str, Any], reward: float, info: dict[str, Any]) -> None:
        _ = reward
        self._integrate_tokens(obs)
        failure = info.get("failure")
        self.last_failure = failure if isinstance(failure, Failure) else None
        action_type = info.get("action_type")

        if action_type == "ask":
            answer = str(info.get("answer", "")).lower().strip()
            self.last_answer = answer
            if answer in self.visual_memory:
                self.goal["color"] = answer
                self.goal["ambiguous"] = False
                self.clarification_count += 1
                self.current_path = []
                self.path_goal = None
                self.state = "update_goal_from_answer"
            else:
                self.state = "clarification_failed"
            return

        if isinstance(failure, Failure):
            if failure.kind == "unsafe_nominal_step":
                self.blocked_cells.update(tuple(cell) for cell in info.get("blocked_cells", ()))
                self.current_path = []
                self.path_goal = None
                self.safety_filter_count += 1
                self.state = "safety_filter_replan"
                return
            if failure.kind == "grasp_miss":
                self.retry_count += 1
                self.state = "recover_from_grasp_miss"
                return
            if failure.kind == "human_correction":
                self.corrected_cells.update(
                    tuple(cell) for cell in info.get("corrected_cells", ())
                )
                self.current_path = []
                self.path_goal = None
                self.human_correction_count += 1
                self.state = "learn_from_human_correction"
                return
            if failure.kind in {"collision", "target_not_reached"}:
                self.current_path = []
                self.path_goal = None
                self.state = f"recover_from_{failure.kind}"
                return

        if info.get("pick_success"):
            self.current_path = []
            self.path_goal = None
            self.state = "picked_up"
            return

        if info.get("success"):
            self.state = "done"
            self.phase = "done"
            return

        self._trim_path(tuple(obs["robot"]))

    def info(self) -> dict[str, Any]:
        return {
            "resolved_goal": dict(self.goal),
            "agent_state": self.state,
            "phase": self.phase,
            "question_count": self.question_count,
            "clarification_count": self.clarification_count,
            "safety_filter_count": self.safety_filter_count,
            "retry_count": self.retry_count,
            "human_correction_count": self.human_correction_count,
            "replan_count": self.replan_count,
            "memory_colors": tuple(sorted(self.visual_memory)),
            "blocked_cells": tuple(sorted(self.blocked_cells)),
            "corrected_cells": tuple(sorted(self.corrected_cells)),
            "planned_path": tuple(self.current_path),
            "planned_path_length": len(self.current_path),
        }

    def _integrate_tokens(self, obs: dict[str, Any]) -> None:
        for token in obs.get("visual_tokens", ()):
            if token.get("name") != self.goal.get("object"):
                continue
            color = str(token.get("color", ""))
            self.visual_memory[color] = dict(token)

    def _target_cell(self, color: str) -> Cell | None:
        token = self.visual_memory.get(color)
        if token is None:
            return None
        return tuple(token["cell"])

    def _next_direction(self, obs: dict[str, Any], goal: Cell) -> str:
        robot = tuple(obs["robot"])
        if self._path_invalid(robot, goal):
            blocked = self.blocked_cells | self.corrected_cells
            self.current_path = grid_astar(obs["walkable"], robot, goal, blocked=blocked)
            self.path_goal = goal
            self.replan_count += 1
        if len(self.current_path) < 2:
            return "stay"
        return direction_to(robot, self.current_path[1])

    def _path_invalid(self, robot: Cell, goal: Cell) -> bool:
        if self.path_goal != goal:
            return True
        if not self.current_path:
            return True
        if robot not in self.current_path:
            return True
        if any(cell in self.blocked_cells or cell in self.corrected_cells for cell in self.current_path):
            return True
        return False

    def _trim_path(self, robot: Cell) -> None:
        if robot not in self.current_path:
            return
        while self.current_path and self.current_path[0] != robot:
            self.current_path.pop(0)


def direction_to(start: Cell, end: Cell) -> str:
    delta = (end[0] - start[0], end[1] - start[1])
    for direction, direction_delta in DIRECTIONS.items():
        if delta == direction_delta:
            return direction
    return "stay"


def draw_household_task_scene(
    ax: Any,
    env: HouseholdTaskWorld,
    agent: HouseholdTaskAgent,
    info: dict[str, Any] | None = None,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Rectangle

    info = {} if info is None else info
    cmap = ListedColormap(["white", "0.12"])
    ax.imshow(env.static_map, cmap=cmap, origin="upper", vmin=0, vmax=1)

    for row, col in env.safety_zone:
        ax.add_patch(
            Rectangle(
                (col - 0.5, row - 0.5),
                1.0,
                1.0,
                facecolor="#f2b84b",
                edgecolor="none",
                alpha=0.42,
            )
        )
    for row, col in env.human_zone:
        ax.add_patch(
            Rectangle(
                (col - 0.5, row - 0.5),
                1.0,
                1.0,
                facecolor="#9b75d6",
                edgecolor="none",
                alpha=0.25,
            )
        )
    for row, col in agent.blocked_cells:
        ax.plot(col, row, "x", color="tab:orange", markersize=10, markeredgewidth=2)
    for row, col in agent.corrected_cells:
        ax.add_patch(
            Rectangle(
                (col - 0.5, row - 0.5),
                1.0,
                1.0,
                facecolor="#d94b3d",
                edgecolor="none",
                alpha=0.33,
            )
        )

    if agent.current_path:
        rows = [r for r, _ in agent.current_path]
        cols = [c for _, c in agent.current_path]
        ax.plot(cols, rows, "--", color="tab:purple", linewidth=2.0, label="plan")

    if len(env.trajectory) > 1:
        rows = [r for r, _ in env.trajectory]
        cols = [c for _, c in env.trajectory]
        ax.plot(cols, rows, color="tab:blue", linewidth=2.3, alpha=0.85, label="executed")

    for color, block in env.blocks.items():
        cell = tuple(block["cell"])
        if block["stored"]:
            continue
        if block["picked"]:
            ax.plot(env.robot[1], env.robot[0], "s", color=DRAW_COLORS[color], markersize=8)
            continue
        ax.plot(cell[1], cell[0], "s", color=DRAW_COLORS[color], markersize=11)
        ax.text(cell[1], cell[0] + 0.34, f"{color}", ha="center", fontsize=7)

    ax.plot(env.storage_cell[1], env.storage_cell[0], "*", color="tab:green", markersize=17)
    ax.text(
        env.storage_cell[1],
        env.storage_cell[0] + 0.38,
        "storage",
        ha="center",
        fontsize=7,
    )

    if env.last_failed_cell is not None:
        ax.plot(
            env.last_failed_cell[1],
            env.last_failed_cell[0],
            marker="P",
            color="tab:orange",
            markersize=13,
        )
    if env.last_human_cell is not None:
        ax.plot(
            env.last_human_cell[1],
            env.last_human_cell[0],
            marker="x",
            color="tab:red",
            markersize=12,
            markeredgewidth=3,
        )

    ax.plot(env.start[1], env.start[0], "o", color="tab:cyan", markersize=8)
    ax.plot(env.robot[1], env.robot[0], "o", color="tab:blue", markersize=10)

    failure = info.get("failure")
    status = (
        f'command="{env.command}"\n'
        f"state={agent.state} phase={agent.phase}\n"
        f"ask={agent.question_count} safety={agent.safety_filter_count} "
        f"retry={agent.retry_count} human={agent.human_correction_count} "
        f"replan={agent.replan_count}"
    )
    if env.last_answer:
        status += f"\nanswer={env.last_answer} target={agent.goal.get('color')}"
    if isinstance(failure, Failure):
        status += f"\nfailure={failure.kind}"
    if info.get("success"):
        status += "\nsuccess: stored"

    ax.text(
        0.02,
        0.98,
        status,
        transform=ax.transAxes,
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.65", alpha=0.9),
    )
    ax.set_title("household agent: clarify -> plan -> safe retry -> replan")
    ax.set_xticks(np.arange(-0.5, env.width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.height, 1), minor=True)
    ax.grid(which="minor", color="0.86", linewidth=0.6)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="lower right", fontsize=7, framealpha=0.9)
    plt.tight_layout()


def run(
    command: str = "put the block away",
    answer: str = "red",
    seed: int = 0,
    render: bool = True,
    max_steps: int = 80,
) -> Trace:
    _ = seed
    env = HouseholdTaskWorld(command=command, answer=answer, max_steps=max_steps)
    agent = HouseholdTaskAgent(command)
    obs = env.reset(seed=seed)
    agent.reset()
    trace = Trace()

    if agent.goal["intent"] == "unknown":
        trace.append(
            obs,
            {"type": "parse_command", "command": command},
            0.0,
            {
                "command": command,
                "parsed_goal": dict(agent.goal),
                "agent_state": "unsupported_goal",
                "success": False,
                "failure": Failure("unsupported_goal", "unsupported command", False),
            },
        )
        return trace

    for _ in range(max_steps):
        action = agent.act(obs)
        result = env.step(action)
        obs, reward, done, info = result.as_tuple()
        agent.update(obs, reward, info)
        info.update(
            {
                "command": command,
                "answer": answer,
                "parsed_goal": parse_household_command(command),
            }
        )
        info.update(agent.info())
        trace.append(obs, action, reward, info)

        if render:
            env.render(agent, info)
        if done:
            break

    return trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs="?", default="put the block away")
    parser.add_argument("--answer", default="red", choices=["red", "blue"])
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    trace = run(
        command=args.command,
        answer=args.answer,
        render=not args.no_render,
        max_steps=args.max_steps,
    )
    final = trace.infos[-1] if trace.infos else {}
    failures = [failure.kind for failure in trace.failures()]
    print(
        f"success={final.get('success', False)} steps={len(trace.actions)} "
        f"stored_color={final.get('stored_color')} "
        f"replans={final.get('replan_count', 0)} failures={failures}"
    )

    if not args.no_render:
        import matplotlib.pyplot as plt

        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
