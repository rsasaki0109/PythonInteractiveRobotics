"""A tiny tabletop world for failure-aware manipulation loops."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pir.core.random import make_rng
from pir.core.types import Failure, StepResult


@dataclass
class TabletopObject:
    name: str
    color: str
    position: np.ndarray
    radius: float = 0.045
    picked: bool = False


class Tabletop2D:
    """A simplified 2D tabletop with noisy detection and grasp failures."""

    def __init__(
        self,
        *,
        seed: int | None = 0,
        table_size: tuple[float, float] = (1.0, 1.0),
        detector_noise: float = 0.055,
        base_false_negative_rate: float = 0.08,
        grasp_radius: float = 0.045,
        base_grasp_success: float = 0.96,
        max_attempts: int = 8,
    ) -> None:
        self.seed = seed
        self.table_size = np.asarray(table_size, dtype=float)
        self.detector_noise = detector_noise
        self.base_false_negative_rate = base_false_negative_rate
        self.grasp_radius = grasp_radius
        self.base_grasp_success = base_grasp_success
        self.max_attempts = max_attempts
        self.occluder = np.array([0.43, 0.42, 0.57, 0.68], dtype=float)
        self.rng = make_rng(seed)
        self._fig = None
        self._ax = None
        self.reset(seed=seed)

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            self.seed = seed
        if self.seed is not None:
            self.rng = make_rng(self.seed)

        self.time = 0
        self.attempts = 0
        self.holding: str | None = None
        self.camera_pos = np.array([0.16, 0.50], dtype=float)
        self.obj = TabletopObject(
            name="block",
            color="tab:red",
            position=np.array([0.64, 0.54], dtype=float),
        )
        self.last_detection: np.ndarray | None = None
        return self.observe()

    def observe(self) -> dict[str, Any]:
        detections: list[dict[str, Any]] = []

        if not self.obj.picked:
            miss_rate = self._current_false_negative_rate()
            if self.rng.random() >= miss_rate:
                noise = self.rng.normal(0.0, self.detector_noise, size=2)
                position = self._clip(self.obj.position + noise)
                confidence = float(max(0.05, 1.0 - miss_rate))
                detections.append(
                    {
                        "name": self.obj.name,
                        "color": self.obj.color,
                        "position": position,
                        "confidence": confidence,
                    }
                )
                self.last_detection = position

        return {
            "time": self.time,
            "camera": self.camera_pos.copy(),
            "detections": detections,
            "gripper": {"holding": self.holding},
            "attempts": self.attempts,
        }

    def step(self, action: dict[str, Any]) -> StepResult:
        self.time += 1
        action_type = action.get("type", "noop")
        info: dict[str, Any] = {
            "time": self.time,
            "action_type": action_type,
            "attempts": self.attempts,
            "success": False,
        }

        if self.obj.picked:
            return StepResult(self.observe(), 0.0, True, info)

        if action_type == "look":
            target = np.asarray(action.get("target", self.camera_pos), dtype=float)
            self.camera_pos = self._clip(target)
            info["camera"] = self.camera_pos.copy()
            return StepResult(self.observe(), -0.01, False, info)

        if action_type == "pick":
            return self._step_pick(action, info)

        failure = Failure("invalid_action", f"unknown action type: {action_type}", True)
        info["failure"] = failure
        return StepResult(self.observe(), -0.05, False, info)

    def _step_pick(self, action: dict[str, Any], info: dict[str, Any]) -> StepResult:
        raw_position = action.get("position")
        if raw_position is None:
            failure = Failure("invalid_action", "pick action requires a position", True)
            info["failure"] = failure
            return StepResult(self.observe(), -0.05, False, info)

        pick_position = self._clip(np.asarray(raw_position, dtype=float))
        self.attempts += 1
        error = float(np.linalg.norm(pick_position - self.obj.position))
        success_probability = self._grasp_success_probability(error)
        success = self.rng.random() < success_probability

        info.update(
            {
                "attempts": self.attempts,
                "pick_position": pick_position.copy(),
                "grasp_error": error,
                "success_probability": success_probability,
            }
        )

        if success:
            self.obj.picked = True
            self.holding = self.obj.name
            info["success"] = True
            return StepResult(self.observe(), 1.0, True, info)

        recoverable = self.attempts < self.max_attempts
        failure = Failure(
            "grasp_miss",
            "gripper closed without lifting the object",
            recoverable,
        )
        info["failure"] = failure
        return StepResult(self.observe(), -0.15, not recoverable, info)

    def _grasp_success_probability(self, error: float) -> float:
        if error > self.grasp_radius * 1.8:
            return 0.0
        alignment = max(0.0, 1.0 - error / (self.grasp_radius * 1.8))
        return float(min(1.0, self.base_grasp_success * (0.15 + 0.85 * alignment)))

    def _current_false_negative_rate(self) -> float:
        miss_rate = self.base_false_negative_rate
        xmin, ymin, xmax, ymax = self.occluder
        camera_left = self.camera_pos[0] < xmin
        object_right = self.obj.position[0] > xmax
        y_blocked = ymin <= self.obj.position[1] <= ymax
        if camera_left and object_right and y_blocked:
            miss_rate += 0.42
        return float(min(0.85, miss_rate))

    def _clip(self, position: np.ndarray) -> np.ndarray:
        return np.clip(position, [0.0, 0.0], self.table_size)

    def render(self, agent: Any | None = None, info: dict[str, Any] | None = None) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, Rectangle

        if self._fig is None or self._ax is None:
            plt.ion()
            self._fig, self._ax = plt.subplots(figsize=(5, 5))

        ax = self._ax
        ax.clear()
        ax.set_title("Tabletop2D: pick, fail, update belief, retry")
        ax.set_xlim(0.0, self.table_size[0])
        ax.set_ylim(0.0, self.table_size[1])
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True, alpha=0.25)

        xmin, ymin, xmax, ymax = self.occluder
        ax.add_patch(
            Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                color="0.2",
                alpha=0.18,
                label="occluder",
            )
        )

        ax.plot(*self.camera_pos, marker="s", color="tab:blue", label="camera")
        if not self.obj.picked:
            ax.add_patch(
                Circle(
                    self.obj.position,
                    self.obj.radius,
                    color=self.obj.color,
                    alpha=0.85,
                    label="true object",
                )
            )

        if self.last_detection is not None and not self.obj.picked:
            ax.plot(
                *self.last_detection,
                marker="x",
                markersize=10,
                color="tab:orange",
                label="last detection",
            )

        belief_mean = getattr(agent, "belief_mean", None)
        if belief_mean is not None:
            ax.add_patch(
                Circle(
                    belief_mean,
                    getattr(agent, "belief_radius", 0.08),
                    fill=False,
                    linestyle="--",
                    color="tab:green",
                    linewidth=2,
                    label="agent belief",
                )
            )

        if info is not None and "pick_position" in info:
            pick_position = info["pick_position"]
            ax.plot(*pick_position, marker="+", markersize=14, color="black", label="pick")

        status = f"attempts={self.attempts}"
        if self.holding is not None:
            status += f" holding={self.holding}"
        elif info is not None and "failure" in info:
            status += f" failure={info['failure'].kind}"
        ax.text(0.02, 0.97, status, transform=ax.transAxes, va="top")

        ax.legend(loc="lower left", fontsize=8)
        self._fig.canvas.draw_idle()
        plt.pause(0.05)
