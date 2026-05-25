"""Ask a clarifying question before acting on an ambiguous command.

The command "pick the block" is underspecified because the tabletop has both
a red block and a blue block. The agent does not guess. It asks a structured
question, receives a simulated human answer, updates the goal, confirms the
target visually, and then picks the requested block.

Success: the requested color block is picked after clarification.
Failure: ambiguous_goal (recoverable - the command omits the color and the
agent asks a question), unsupported_goal (terminal), invalid_target
(recoverable), grasp_miss (recoverable), timeout (terminal).
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


DRAW_COLORS = {
    "red": "tab:red",
    "blue": "tab:blue",
}


def parse_pick_command(command: str) -> dict[str, Any]:
    """Parse a tiny command grammar for block picking."""

    normalized = " ".join(command.lower().strip().split())
    if normalized == "pick the block":
        return {"intent": "pick", "object": "block", "color": None, "ambiguous": True}
    for color in ("red", "blue"):
        if normalized in {f"pick the {color} block", f"pick {color} block"}:
            return {
                "intent": "pick",
                "object": "block",
                "color": color,
                "ambiguous": False,
            }
    return {
        "intent": "unknown",
        "message": "use: pick the block | pick the red block | pick the blue block",
        "command": command,
    }


class ClarifyingQuestionWorld:
    """Two-block tabletop with a simulated answer to the robot's question."""

    def __init__(
        self,
        *,
        command: str = "pick the block",
        answer: str = "red",
        max_steps: int = 12,
    ) -> None:
        self.command = command
        self.answer = answer.lower().strip()
        self.max_steps = max_steps
        self._fig: Any | None = None
        self._ax: Any | None = None
        self.reset()

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        _ = seed
        self.time = 0
        self.focus_color: str | None = None
        self.picked_color: str | None = None
        self.last_question: str | None = None
        self.last_answer: str | None = None
        self.last_pick_position: np.ndarray | None = None
        self.blocks = {
            "red": {
                "name": "block",
                "color": "red",
                "position": np.array([0.32, 0.56], dtype=float),
                "radius": 0.055,
                "picked": False,
            },
            "blue": {
                "name": "block",
                "color": "blue",
                "position": np.array([0.68, 0.56], dtype=float),
                "radius": 0.055,
                "picked": False,
            },
        }
        return self.observe()

    def observe(self) -> dict[str, Any]:
        tokens: list[dict[str, Any]] = []
        for color, block in self.blocks.items():
            if block["picked"]:
                continue
            confidence = 0.98 if color == self.focus_color else 0.88
            tokens.append(
                {
                    "name": block["name"],
                    "color": color,
                    "position": np.asarray(block["position"], dtype=float).copy(),
                    "confidence": confidence,
                }
            )
        return {
            "time": self.time,
            "command": self.command,
            "visual_tokens": tokens,
            "focus_color": self.focus_color,
            "picked_color": self.picked_color,
            "last_question": self.last_question,
            "last_answer": self.last_answer,
        }

    def step(self, action: dict[str, Any]) -> StepResult:
        self.time += 1
        action_type = action.get("type", "noop")
        info: dict[str, Any] = {
            "time": self.time,
            "action_type": action_type,
            "success": False,
            "picked_color": self.picked_color,
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
                        "Command matched more than one block, so the robot asked for a color.",
                        recoverable=True,
                    ),
                }
            )
            return StepResult(self.observe(), -0.02, False, info)

        if action_type == "look":
            color = str(action.get("color", "")).lower()
            self.focus_color = color if color in self.blocks else None
            info["focus_color"] = self.focus_color
            return StepResult(self.observe(), -0.01, False, info)

        if action_type == "pick":
            color = str(action.get("color", "")).lower()
            block = self.blocks.get(color)
            if block is None or block["picked"]:
                info["failure"] = Failure(
                    "invalid_target",
                    f"Cannot pick target color: {color}",
                    recoverable=True,
                )
                return self._finish(-0.08, False, info)

            position = np.asarray(action.get("position", block["position"]), dtype=float)
            position = np.clip(position, 0.0, 1.0)
            self.last_pick_position = position.copy()
            error = float(np.linalg.norm(position - np.asarray(block["position"], dtype=float)))
            info.update({"target_color": color, "pick_position": position.copy(), "grasp_error": error})
            if error > 0.08:
                info["failure"] = Failure(
                    "grasp_miss",
                    "Pick command was aimed too far from the requested block.",
                    recoverable=True,
                )
                return self._finish(-0.15, False, info)

            block["picked"] = True
            self.picked_color = color
            info["picked_color"] = color
            info["success"] = True
            return self._finish(1.0, True, info)

        info["failure"] = Failure("invalid_action", f"unknown action: {action_type}", True)
        return self._finish(-0.06, False, info)

    def _finish(self, reward: float, done: bool, info: dict[str, Any]) -> StepResult:
        if not done and self.time >= self.max_steps:
            done = True
            info["failure"] = Failure(
                "timeout",
                "Clarifying-question loop did not finish before max_steps.",
                recoverable=False,
            )
        return StepResult(self.observe(), reward, done, info)

    def render(self, agent: "ClarifyingQuestionAgent", info: dict[str, Any]) -> None:
        import matplotlib.pyplot as plt

        if self._fig is None or self._ax is None:
            plt.ion()
            self._fig, self._ax = plt.subplots(figsize=(5.3, 4.8))
        self._ax.clear()
        draw_clarifying_question_scene(self._ax, self, agent, info)
        self._fig.canvas.draw_idle()
        plt.pause(0.001)


class ClarifyingQuestionAgent:
    """Resolve command ambiguity through one explicit question."""

    def __init__(self, command: str) -> None:
        self.command = command
        self.initial_goal = parse_pick_command(command)
        self.reset()

    def reset(self) -> None:
        self.goal = dict(self.initial_goal)
        self.state = "parse_command"
        self.visual_memory: dict[str, dict[str, Any]] = {}
        self.question_count = 0
        self.clarification_count = 0
        self.retry_count = 0
        self.confirmed_view = False
        self.last_question: str | None = None
        self.last_answer: str | None = None

    def act(self, obs: dict[str, Any]) -> dict[str, Any]:
        self._integrate_tokens(obs)
        if self.goal["intent"] == "unknown":
            self.state = "unsupported_goal"
            return {"type": "noop"}

        if self.goal.get("color") is None:
            choices = tuple(sorted(self.visual_memory))
            self.state = "ask_clarification"
            self.question_count += 1
            self.last_question = f"Which block, {', '.join(choices)}?"
            return {"type": "ask", "question": self.last_question, "choices": choices}

        color = str(self.goal["color"])
        if not self.confirmed_view:
            self.state = "confirm_target"
            return {"type": "look", "color": color}

        token = self.visual_memory.get(color)
        if token is None:
            self.state = "target_not_visible"
            return {"type": "look", "color": color}

        self.state = "pick_target" if self.retry_count == 0 else "retry_pick"
        position = np.asarray(token["position"], dtype=float)
        return {"type": "pick", "color": color, "position": position}

    def update(self, obs: dict[str, Any], reward: float, info: dict[str, Any]) -> None:
        _ = reward
        self._integrate_tokens(obs)
        action_type = info.get("action_type")

        if action_type == "ask":
            answer = str(info.get("answer", "")).lower().strip()
            self.last_answer = answer
            if answer in self.visual_memory:
                self.goal["color"] = answer
                self.goal["ambiguous"] = False
                self.clarification_count += 1
                self.confirmed_view = False
                self.state = "update_goal_from_answer"
            else:
                self.state = "clarification_failed"
            return

        if action_type == "look" and self.goal.get("color") is not None:
            self.confirmed_view = True
            self.state = "target_confirmed"
            return

        failure = info.get("failure")
        if isinstance(failure, Failure) and failure.kind == "grasp_miss":
            self.retry_count += 1
            self.confirmed_view = False
            self.state = "recover_from_miss"
        elif info.get("success"):
            self.state = "done"

    def _integrate_tokens(self, obs: dict[str, Any]) -> None:
        for token in obs.get("visual_tokens", []):
            if token.get("name") != self.goal.get("object"):
                continue
            color = str(token.get("color", ""))
            self.visual_memory[color] = dict(token)


def draw_clarifying_question_scene(
    ax: Any,
    env: ClarifyingQuestionWorld,
    agent: ClarifyingQuestionAgent,
    info: dict[str, Any] | None = None,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    info = {} if info is None else info
    ax.set_title("ambiguous command -> ask -> answer -> pick")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.22)

    for color, block in env.blocks.items():
        if block["picked"]:
            continue
        position = np.asarray(block["position"], dtype=float)
        radius = float(block["radius"])
        ax.add_patch(Circle(position, radius, color=DRAW_COLORS[color], alpha=0.86))
        ax.text(position[0], position[1] - 0.11, f"{color} block", ha="center", fontsize=8)

    target_color = agent.goal.get("color")
    if target_color in env.blocks and not env.blocks[target_color]["picked"]:
        pos = np.asarray(env.blocks[target_color]["position"], dtype=float)
        ax.add_patch(Circle(pos, 0.09, fill=False, color="tab:green", linewidth=2.0))

    if env.last_pick_position is not None:
        ax.plot(
            env.last_pick_position[0],
            env.last_pick_position[1],
            marker="+",
            markersize=15,
            markeredgewidth=2,
            color="black",
        )

    command = f'command: "{env.command}"'
    question = env.last_question or agent.last_question or ""
    answer = env.last_answer or agent.last_answer or ""
    status = (
        f"{command}\n"
        f"state={agent.state} questions={agent.question_count} "
        f"clarifications={agent.clarification_count}"
    )
    if question:
        status += f"\nQ: {question}"
    if answer:
        status += f"\nA: {answer}"
    failure = info.get("failure")
    if isinstance(failure, Failure):
        status += f"\nfailure={failure.kind}"
    if info.get("success"):
        status += "\nsuccess"

    ax.text(
        0.02,
        0.98,
        status,
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.65", alpha=0.9),
    )
    ax.tick_params(labelsize=8)
    plt.tight_layout()


def run(
    command: str = "pick the block",
    answer: str = "red",
    seed: int = 0,
    render: bool = True,
    max_steps: int = 12,
) -> Trace:
    _ = seed
    env = ClarifyingQuestionWorld(command=command, answer=answer, max_steps=max_steps)
    agent = ClarifyingQuestionAgent(command)
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
                "parsed_goal": parse_pick_command(command),
                "resolved_goal": dict(agent.goal),
                "agent_state": agent.state,
                "question_count": agent.question_count,
                "clarification_count": agent.clarification_count,
                "retry_count": agent.retry_count,
                "memory_colors": tuple(sorted(agent.visual_memory)),
            }
        )
        trace.append(obs, action, reward, info)

        if render:
            env.render(agent, info)
        if done:
            break

    return trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs="?", default="pick the block")
    parser.add_argument("--answer", default="red", choices=["red", "blue"])
    parser.add_argument("--max-steps", type=int, default=12)
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
        f"resolved_goal={final.get('resolved_goal')} failures={failures}"
    )

    if not args.no_render:
        import matplotlib.pyplot as plt

        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
