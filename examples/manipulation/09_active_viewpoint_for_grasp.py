"""Active viewpoint selection before grasping under pose ambiguity.

The object's pose is hidden behind a viewpoint-dependent occluder. The agent
keeps a belief over three possible poses, picks the viewpoint that maximally
reduces expected occlusion given the current belief, runs a Bayes update from
the observation, and finally grasps with the type that maximizes expected
success across the refined belief.
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


POSE_LABELS: tuple[str, ...] = ("tilted_left", "upright", "tilted_right")
GRASP_LABELS: tuple[str, ...] = ("side_left", "top_down", "side_right")
VIEW_LABELS: tuple[str, ...] = ("front", "side_left_cam", "side_right_cam")

DEFAULT_OCCLUSION_MATRIX: np.ndarray = np.array(
    [
        # rows: viewpoint, columns: pose (tilted_left, upright, tilted_right)
        [0.85, 0.10, 0.85],  # "front": only resolves upright cleanly
        [0.10, 0.80, 0.80],  # "side_left_cam": resolves tilted_left
        [0.80, 0.80, 0.10],  # "side_right_cam": resolves tilted_right
    ]
)

DEFAULT_SUCCESS_MATRIX: np.ndarray = np.array(
    [
        # rows: grasp, columns: pose
        [0.90, 0.18, 0.10],  # side_left
        [0.20, 0.90, 0.20],  # top_down
        [0.10, 0.18, 0.90],  # side_right
    ]
)

VIEW_POSITIONS: dict[str, tuple[float, float]] = {
    "front": (0.5, 0.92),
    "side_left_cam": (0.08, 0.5),
    "side_right_cam": (0.92, 0.5),
}


class ActiveViewpointGraspWorld:
    """Tabletop where each viewpoint partially occludes the object pose."""

    def __init__(
        self,
        *,
        seed: int | None = 0,
        true_pose: int | None = None,
        occlusion_matrix: np.ndarray | None = None,
        success_matrix: np.ndarray | None = None,
        max_attempts: int = 4,
        max_observations: int = 6,
    ) -> None:
        self.seed = seed
        self.true_pose_init = true_pose
        self.occlusion_matrix = (
            DEFAULT_OCCLUSION_MATRIX.copy()
            if occlusion_matrix is None
            else np.asarray(occlusion_matrix, dtype=float)
        )
        self.success_matrix = (
            DEFAULT_SUCCESS_MATRIX.copy()
            if success_matrix is None
            else np.asarray(success_matrix, dtype=float)
        )
        self.max_attempts = max_attempts
        self.max_observations = max_observations
        self.rng = make_rng(seed)
        self._fig = None
        self._axes = None
        self.reset(seed=seed)

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            self.seed = seed
        if self.seed is not None:
            self.rng = make_rng(self.seed)
        n_poses = self.occlusion_matrix.shape[1]
        if self.true_pose_init is None:
            self.true_pose = int(self.rng.integers(0, n_poses))
        else:
            self.true_pose = int(self.true_pose_init)
        self.time = 0
        self.attempts = 0
        self.observations = 0
        self.holding = False
        self.last_view: str | None = None
        self.last_observation: int | None = None
        self.last_grasp: str | None = None
        self.last_result: str | None = None
        return self.observe()

    def observe(self) -> dict[str, Any]:
        return {
            "time": self.time,
            "attempts": self.attempts,
            "observations": self.observations,
            "holding": self.holding,
            "last_view": self.last_view,
            "last_observation": self.last_observation,
            "last_grasp": self.last_grasp,
            "last_result": self.last_result,
            "max_attempts": self.max_attempts,
            "max_observations": self.max_observations,
            "occlusion_matrix": self.occlusion_matrix.copy(),
            "success_matrix": self.success_matrix.copy(),
            "view_labels": VIEW_LABELS,
            "pose_labels": POSE_LABELS,
            "grasp_labels": GRASP_LABELS,
        }

    def step(self, action: dict[str, Any]) -> StepResult:
        self.time += 1
        action_type = action.get("action_type") if isinstance(action, dict) else None
        info: dict[str, Any] = {"time": self.time, "action_type": action_type, "success": False}

        if action_type == "look":
            view = action.get("view")
            if view not in VIEW_LABELS:
                info["failure"] = Failure("invalid_view", f"unknown view: {view}", True)
                return StepResult(self.observe(), -0.05, False, info)
            view_index = VIEW_LABELS.index(view)
            occlusion = float(self.occlusion_matrix[view_index, self.true_pose])
            reliability = 1.0 - occlusion
            roll = float(self.rng.random())
            if roll < reliability:
                observed = self.true_pose
            else:
                n_poses = self.occlusion_matrix.shape[1]
                alternatives = [p for p in range(n_poses) if p != self.true_pose]
                observed = int(self.rng.choice(alternatives))
            self.observations += 1
            self.last_view = view
            self.last_observation = observed
            self.last_result = "look"
            info["view"] = view
            info["view_index"] = view_index
            info["observed_pose_index"] = observed
            info["reliability"] = reliability
            return StepResult(self.observe(), -0.02, False, info)

        if action_type == "grasp":
            grasp = action.get("grasp")
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
            self.last_grasp = grasp
            info["grasp"] = grasp
            info["grasp_success_prob"] = success_prob
            info["true_pose_index"] = self.true_pose
            info["attempts"] = self.attempts
            if roll < success_prob:
                self.holding = True
                self.last_result = "success"
                info["success"] = True
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

        info["failure"] = Failure("invalid_action", f"unknown action_type: {action_type}", True)
        return StepResult(self.observe(), -0.05, False, info)

    def render(self, agent: Any | None = None, info: dict[str, Any] | None = None) -> None:
        import matplotlib.pyplot as plt

        if self._fig is None or self._axes is None:
            plt.ion()
            self._fig, self._axes = plt.subplots(1, 2, figsize=(9.0, 4.4))
        for ax in self._axes:
            ax.clear()
        draw_active_viewpoint_scene(self._axes, self, agent, info)
        self._fig.canvas.draw_idle()
        plt.pause(0.1)


def _entropy(belief: np.ndarray) -> float:
    safe = np.clip(belief, 1e-12, 1.0)
    return float(-np.sum(safe * np.log(safe)))


class ActiveViewpointGraspAgent:
    """Pick the viewpoint that minimizes expected occlusion, then grasp."""

    def __init__(
        self,
        occlusion_matrix: np.ndarray | None = None,
        success_matrix: np.ndarray | None = None,
        *,
        belief_threshold: float = 0.7,
        max_scout_steps: int = 5,
    ) -> None:
        self.occlusion_matrix = (
            DEFAULT_OCCLUSION_MATRIX.copy()
            if occlusion_matrix is None
            else np.asarray(occlusion_matrix, dtype=float)
        )
        self.success_matrix = (
            DEFAULT_SUCCESS_MATRIX.copy()
            if success_matrix is None
            else np.asarray(success_matrix, dtype=float)
        )
        self.belief_threshold = belief_threshold
        self.max_scout_steps = max_scout_steps
        self.reset()

    def reset(self) -> None:
        n_poses = self.occlusion_matrix.shape[1]
        self.belief = np.full(n_poses, 1.0 / n_poses)
        self.scout_steps = 0
        self.scout_resets = 0
        self.view_count = 0
        self.belief_update_count = 0
        self.failed_attempts = 0
        self.last_view: str | None = None
        self.last_grasp: str | None = None
        self.last_expected_success: np.ndarray = self.success_matrix @ self.belief
        self.last_view_scores: np.ndarray = np.zeros(self.occlusion_matrix.shape[0])
        self.state = "initial"

    @property
    def entropy(self) -> float:
        return _entropy(self.belief)

    def act(self, obs: dict[str, Any]) -> dict[str, Any]:
        if self._should_grasp():
            return self._grasp_action()
        return self._look_action()

    def _should_grasp(self) -> bool:
        confident = float(self.belief.max()) >= self.belief_threshold
        budget_exhausted = self.scout_steps >= self.max_scout_steps
        return confident or budget_exhausted

    def _look_action(self) -> dict[str, Any]:
        # expected reliability per viewpoint under current belief
        reliability_matrix = 1.0 - self.occlusion_matrix
        scores = reliability_matrix @ self.belief
        self.last_view_scores = scores
        view_index = int(np.argmax(scores))
        self.last_view = VIEW_LABELS[view_index]
        self.state = "scout"
        return {"action_type": "look", "view": self.last_view}

    def _grasp_action(self) -> dict[str, Any]:
        expected = self.success_matrix @ self.belief
        self.last_expected_success = expected
        grasp_index = int(np.argmax(expected))
        self.last_grasp = GRASP_LABELS[grasp_index]
        self.state = "grasp"
        return {"action_type": "grasp", "grasp": self.last_grasp}

    def update(self, obs: dict[str, Any], reward: float, info: dict[str, Any]) -> None:
        if info.get("action_type") == "look" and "observed_pose_index" in info:
            view_index = info["view_index"]
            observed = info["observed_pose_index"]
            self._bayes_update_from_observation(view_index, observed)
            self.scout_steps += 1
            self.view_count += 1
            return
        if info.get("action_type") == "grasp":
            if info.get("success"):
                self.state = "succeeded"
                return
            failure = info.get("failure")
            if isinstance(failure, Failure) and failure.kind == "grasp_miss" and self.last_grasp is not None:
                grasp_index = GRASP_LABELS.index(self.last_grasp)
                likelihood = 1.0 - self.success_matrix[grasp_index]
                self._bayes_update(likelihood)
                self.failed_attempts += 1
                # re-open the scouting window for one more look
                self.scout_steps = max(self.max_scout_steps - 1, 0)
                self.scout_resets += 1
                self.state = "scout"

    def _bayes_update_from_observation(self, view_index: int, observed_pose: int) -> None:
        occlusion = self.occlusion_matrix[view_index]
        n_poses = self.belief.shape[0]
        likelihood = np.empty(n_poses)
        for pose in range(n_poses):
            reliability = 1.0 - occlusion[pose]
            if pose == observed_pose:
                likelihood[pose] = reliability
            else:
                likelihood[pose] = occlusion[pose] / max(n_poses - 1, 1)
        self._bayes_update(likelihood)

    def _bayes_update(self, likelihood: np.ndarray) -> None:
        posterior = self.belief * likelihood
        total = posterior.sum()
        if total > 0:
            self.belief = posterior / total
            self.belief_update_count += 1


def draw_active_viewpoint_scene(
    axes: Any,
    env: ActiveViewpointGraspWorld,
    agent: ActiveViewpointGraspAgent | None,
    info: dict[str, Any] | None,
) -> None:
    import matplotlib.patches as mpatches

    ax_scene, ax_bars = axes

    ax_scene.set_xlim(0.0, 1.0)
    ax_scene.set_ylim(0.0, 1.0)
    ax_scene.set_aspect("equal", adjustable="box")
    ax_scene.set_title("active viewpoint for grasp")
    ax_scene.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax_scene.add_patch(
        mpatches.Rectangle((0.18, 0.18), 0.64, 0.64, color="0.92", ec="0.7")
    )
    # draw object at center with orientation cue
    object_x, object_y = 0.5, 0.5
    radius = 0.07
    ax_scene.add_patch(mpatches.Circle((object_x, object_y), radius, color="tab:orange", alpha=0.85))
    tilt_angles = {0: 150.0, 1: 90.0, 2: 30.0}
    angle_deg = tilt_angles.get(env.true_pose, 90.0)
    angle_rad = np.deg2rad(angle_deg)
    dx = np.cos(angle_rad) * 0.11
    dy = np.sin(angle_rad) * 0.11
    ax_scene.plot(
        [object_x, object_x + dx],
        [object_y, object_y + dy],
        color="black",
        linewidth=2,
    )
    ax_scene.text(
        object_x,
        object_y - 0.12,
        f"true: {POSE_LABELS[env.true_pose]}",
        ha="center",
        fontsize=8,
        color="0.3",
    )

    chosen_view = info.get("view") if info else None
    if chosen_view is None and agent is not None:
        chosen_view = agent.last_view
    for label, (cam_x, cam_y) in VIEW_POSITIONS.items():
        color = "tab:blue" if label == chosen_view else "0.55"
        ax_scene.add_patch(mpatches.Rectangle((cam_x - 0.04, cam_y - 0.04), 0.08, 0.08, color=color))
        ax_scene.plot(
            [cam_x, object_x],
            [cam_y, object_y],
            color=color,
            linestyle="--" if label != chosen_view else "-",
            alpha=0.55 if label != chosen_view else 0.9,
        )
        ax_scene.text(cam_x, cam_y + 0.07, label, ha="center", fontsize=7, color=color)

    status_parts: list[str] = [f"obs={env.observations}", f"attempts={env.attempts}"]
    if agent is not None:
        status_parts.append(f"state={agent.state}")
        status_parts.append(f"entropy={agent.entropy:.2f}")
        status_parts.append(f"updates={agent.belief_update_count}")
    if info is not None and "view" in info:
        status_parts.append(f"view={info['view']}")
    if info is not None and "observed_pose_index" in info:
        status_parts.append(f"obs={POSE_LABELS[info['observed_pose_index']]}")
    if info is not None and "grasp" in info:
        status_parts.append(f"grasp={info['grasp']}")
    if info is not None and "failure" in info:
        status_parts.append(f"failure={info['failure'].kind}")
    if info is not None and info.get("success"):
        status_parts.append("success")
    ax_scene.text(0.02, 0.98, "  ".join(status_parts), transform=ax_scene.transAxes, va="top", fontsize=8)

    n_poses = len(POSE_LABELS)
    bars = np.arange(n_poses)
    belief = agent.belief if agent is not None else np.full(n_poses, 1.0 / n_poses)
    expected = (
        agent.last_expected_success
        if agent is not None
        else env.success_matrix @ np.full(n_poses, 1.0 / n_poses)
    )

    ax_bars.bar(bars - 0.2, belief, width=0.32, color="tab:blue", label="pose belief")
    ax_bars.bar(bars + 0.2, expected, width=0.32, color="tab:orange", label="expected grasp success")
    ax_bars.set_xticks(bars)
    ax_bars.set_xticklabels(
        [f"{POSE_LABELS[i]}\n(grasp {GRASP_LABELS[i]})" for i in range(n_poses)],
        fontsize=7,
    )
    ax_bars.set_ylim(0.0, 1.0)
    ax_bars.set_ylabel("probability")
    ax_bars.set_title("belief / grasp scores")
    ax_bars.axvline(env.true_pose, color="tab:green", linestyle="--", alpha=0.5, label="true pose")
    ax_bars.legend(loc="upper right", fontsize=7)


def run(
    seed: int = 0,
    render: bool = True,
    max_steps: int = 14,
    true_pose: int | None = 2,
) -> Trace:
    env = ActiveViewpointGraspWorld(seed=seed, true_pose=true_pose)
    agent = ActiveViewpointGraspAgent()
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
        info["entropy"] = agent.entropy
        info["belief_update_count"] = agent.belief_update_count
        info["view_count"] = agent.view_count
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
    parser.add_argument(
        "--true-pose",
        type=int,
        default=2,
        help="0=tilted_left, 1=upright, 2=tilted_right",
    )
    parser.add_argument("--max-steps", type=int, default=14)
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
        f"views={final.get('view_count', 0)} "
        f"misses={final.get('failed_attempts', 0)} "
        f"updates={final.get('belief_update_count', 0)} "
        f"entropy={final.get('entropy', 0.0):.2f} failures={failures}"
    )

    if not args.no_render:
        import matplotlib.pyplot as plt

        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
