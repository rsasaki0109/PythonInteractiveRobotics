"""Pick under pose ambiguity: choose the grasp that maximizes expected success.

The object on the tabletop has one of three hidden orientations. Each grasp
type is best for a different orientation. The agent keeps a belief over the
three poses, picks the grasp with the highest expected success across that
belief, and runs a Bayes update on every miss until one grasp lands.
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


GRASP_LABELS: tuple[str, ...] = ("side_left", "top_down", "side_right")
POSE_LABELS: tuple[str, ...] = ("tilted_left", "upright", "tilted_right")

DEFAULT_SUCCESS_MATRIX: np.ndarray = np.array(
    [
        # columns: tilted_left, upright, tilted_right
        [0.88, 0.18, 0.10],  # side_left
        [0.20, 0.88, 0.20],  # top_down
        [0.10, 0.18, 0.88],  # side_right
    ]
)


class BeliefGraspWorld:
    """Tabletop with hidden object orientation and a known success matrix."""

    def __init__(
        self,
        *,
        seed: int | None = 0,
        true_pose: int | None = None,
        success_matrix: np.ndarray | None = None,
        max_attempts: int = 6,
    ) -> None:
        self.seed = seed
        self.true_pose_init = true_pose
        self.success_matrix = (
            DEFAULT_SUCCESS_MATRIX.copy() if success_matrix is None else np.asarray(success_matrix, dtype=float)
        )
        self.max_attempts = max_attempts
        self.rng = make_rng(seed)
        self._fig = None
        self._ax = None
        self.reset(seed=seed)

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            self.seed = seed
        if self.seed is not None:
            self.rng = make_rng(self.seed)
        if self.true_pose_init is None:
            self.true_pose = int(self.rng.integers(0, self.success_matrix.shape[1]))
        else:
            self.true_pose = int(self.true_pose_init)
        self.time = 0
        self.attempts = 0
        self.holding = False
        self.last_grasp: str | None = None
        self.last_result: str | None = None
        return self.observe()

    def observe(self) -> dict[str, Any]:
        return {
            "time": self.time,
            "attempts": self.attempts,
            "holding": self.holding,
            "last_grasp": self.last_grasp,
            "last_result": self.last_result,
            "max_attempts": self.max_attempts,
            "success_matrix": self.success_matrix.copy(),
            "grasp_labels": GRASP_LABELS,
            "pose_labels": POSE_LABELS,
        }

    def step(self, action: str | dict[str, Any]) -> StepResult:
        self.time += 1
        grasp = action.get("grasp") if isinstance(action, dict) else action
        info: dict[str, Any] = {"time": self.time, "grasp": grasp, "success": False}

        if grasp not in GRASP_LABELS:
            info["failure"] = Failure("invalid_grasp", f"unknown grasp: {grasp}", True)
            return StepResult(self.observe(), -0.05, False, info)

        if self.holding:
            info["failure"] = Failure("already_holding", "object already grasped", True)
            return StepResult(self.observe(), -0.02, False, info)

        self.attempts += 1
        grasp_index = GRASP_LABELS.index(grasp)
        success_prob = float(self.success_matrix[grasp_index, self.true_pose])
        roll = float(self.rng.random())
        succeeded = roll < success_prob
        info["grasp_success_prob"] = success_prob
        info["true_pose_index"] = self.true_pose
        info["attempts"] = self.attempts
        self.last_grasp = grasp

        if succeeded:
            self.holding = True
            info["success"] = True
            self.last_result = "success"
            return StepResult(self.observe(), 1.0, True, info)

        self.last_result = "miss"
        recoverable = self.attempts < self.max_attempts
        info["failure"] = Failure(
            "grasp_miss",
            f"{grasp} failed (success_prob={success_prob:.2f})",
            recoverable,
        )
        if not recoverable:
            return StepResult(self.observe(), -0.5, True, info)
        return StepResult(self.observe(), -0.15, False, info)

    def render(self, agent: Any | None = None, info: dict[str, Any] | None = None) -> None:
        import matplotlib.pyplot as plt

        if self._fig is None or self._ax is None:
            plt.ion()
            self._fig, self._ax = plt.subplots(figsize=(7, 4.5))
        ax = self._ax
        ax.clear()
        draw_belief_grasp_scene(ax, self, agent, info)
        self._fig.canvas.draw_idle()
        plt.pause(0.1)


class BeliefGraspAgent:
    """Maintain a belief over object pose; pick grasp argmax of expected success."""

    def __init__(self, success_matrix: np.ndarray | None = None) -> None:
        self.success_matrix = (
            DEFAULT_SUCCESS_MATRIX.copy()
            if success_matrix is None
            else np.asarray(success_matrix, dtype=float)
        )
        self.reset()

    def reset(self) -> None:
        n_poses = self.success_matrix.shape[1]
        self.belief = np.full(n_poses, 1.0 / n_poses)
        self.belief_update_count = 0
        self.failed_attempts = 0
        self.last_grasp: str | None = None
        self.last_expected_success: np.ndarray = self.success_matrix @ self.belief
        self.state = "initial"

    def act(self, obs: dict[str, Any]) -> dict[str, Any]:
        expected = self.success_matrix @ self.belief
        self.last_expected_success = expected
        grasp_index = int(np.argmax(expected))
        grasp = GRASP_LABELS[grasp_index]
        self.last_grasp = grasp
        self.state = "choose_grasp"
        return {"grasp": grasp}

    def update(self, obs: dict[str, Any], reward: float, info: dict[str, Any]) -> None:
        failure = info.get("failure")
        if (
            isinstance(failure, Failure)
            and failure.kind == "grasp_miss"
            and self.last_grasp is not None
        ):
            grasp_index = GRASP_LABELS.index(self.last_grasp)
            success_probs = self.success_matrix[grasp_index]
            likelihood = 1.0 - success_probs
            posterior = self.belief * likelihood
            total = posterior.sum()
            if total > 0:
                self.belief = posterior / total
                self.belief_update_count += 1
            self.failed_attempts += 1
            self.state = "update_belief"
        elif info.get("success"):
            self.state = "succeeded"


def draw_belief_grasp_scene(
    ax: Any,
    env: BeliefGraspWorld,
    agent: BeliefGraspAgent | None,
    info: dict[str, Any] | None,
) -> None:
    bars = np.arange(len(POSE_LABELS))
    belief = agent.belief if agent is not None else np.full(len(POSE_LABELS), 1.0 / len(POSE_LABELS))
    expected = (
        agent.last_expected_success
        if agent is not None
        else env.success_matrix @ np.full(len(POSE_LABELS), 1.0 / len(POSE_LABELS))
    )

    ax.bar(bars - 0.18, belief, width=0.32, color="tab:blue", label="pose belief")
    ax.bar(bars + 0.18, expected, width=0.32, color="tab:orange", label="expected grasp success")

    ax.set_xticks(bars)
    ax.set_xticklabels(
        [
            f"{POSE_LABELS[i]}\n(grasp {GRASP_LABELS[i]})"
            for i in range(len(POSE_LABELS))
        ],
        fontsize=8,
    )
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("probability")
    ax.set_title("belief-guided grasp selection")

    true_index = env.true_pose
    ax.axvline(true_index, color="tab:green", linestyle="--", alpha=0.5, label="true pose")

    status_parts = [
        f"attempts={env.attempts}",
        f"holding={env.holding}",
    ]
    if agent is not None:
        status_parts.append(f"misses={agent.failed_attempts}")
        status_parts.append(f"updates={agent.belief_update_count}")
        status_parts.append(f"state={agent.state}")
    if info is not None and info.get("grasp") is not None:
        status_parts.append(f"last_grasp={info['grasp']}")
    if info is not None and "failure" in info:
        status_parts.append(f"failure={info['failure'].kind}")
    status = "  |  ".join(status_parts)
    ax.text(0.02, 0.98, status, transform=ax.transAxes, va="top", fontsize=9)
    ax.legend(loc="upper right", fontsize=8)


def run(seed: int = 0, render: bool = True, max_steps: int = 10, true_pose: int | None = 0) -> Trace:
    env = BeliefGraspWorld(seed=seed, true_pose=true_pose)
    agent = BeliefGraspAgent()
    obs = env.reset(seed=seed)
    agent.reset()
    trace = Trace()

    for _ in range(max_steps):
        action = agent.act(obs)
        result = env.step(action)
        obs, reward, done, info = result.as_tuple()
        agent.update(obs, reward, info)
        info["agent_state"] = agent.state
        info["belief"] = agent.belief.tolist()
        info["belief_update_count"] = agent.belief_update_count
        info["failed_attempts"] = agent.failed_attempts
        info["expected_success"] = agent.last_expected_success.tolist()
        trace.append(obs, action, reward, info)

        if render:
            env.render(agent=agent, info=info)
        if done:
            break

    return trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--true-pose", type=int, default=None, help="0=tilted_left, 1=upright, 2=tilted_right")
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    trace = run(
        seed=args.seed,
        render=not args.no_render,
        max_steps=args.max_steps,
        true_pose=args.true_pose,
    )
    success = bool(trace.infos and trace.infos[-1].get("success"))
    failures = [failure.kind for failure in trace.failures()]
    final = trace.infos[-1] if trace.infos else {}
    print(
        f"success={success} steps={len(trace.actions)} "
        f"misses={final.get('failed_attempts', 0)} "
        f"updates={final.get('belief_update_count', 0)} failures={failures}"
    )

    if not args.no_render:
        import matplotlib.pyplot as plt

        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
