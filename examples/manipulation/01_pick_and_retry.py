"""Pick an object, detect failure, update belief, and retry differently."""

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


class PickAndRetryAgent:
    """A tiny belief-based agent for one-object tabletop picking."""

    def __init__(self) -> None:
        self.offset_schedule = [
            np.array([-0.18, 0.00]),
            np.array([0.00, 0.00]),
            np.array([0.04, 0.00]),
            np.array([-0.04, 0.00]),
            np.array([0.00, 0.04]),
            np.array([0.00, -0.04]),
        ]
        self.viewpoints = [
            np.array([0.84, 0.52]),
            np.array([0.20, 0.84]),
            np.array([0.78, 0.22]),
        ]
        self.reset()

    def reset(self) -> None:
        self.belief_mean: np.ndarray | None = None
        self.belief_radius = 0.14
        self.retry_count = 0
        self.look_count = 0
        self._last_integrated_time: int | None = None

    def act(self, obs: dict[str, Any]) -> dict[str, Any]:
        self._integrate_observation(obs)

        if self.belief_mean is None:
            target = self.viewpoints[self.look_count % len(self.viewpoints)]
            self.look_count += 1
            return {"type": "look", "target": target}

        offset = self.offset_schedule[
            min(self.retry_count, len(self.offset_schedule) - 1)
        ]
        pick_position = np.clip(self.belief_mean + offset, 0.0, 1.0)
        return {"type": "pick", "position": pick_position}

    def update(self, obs: dict[str, Any], reward: float, info: dict[str, Any]) -> None:
        self._integrate_observation(obs)

        failure = info.get("failure")
        if isinstance(failure, Failure) and failure.kind == "grasp_miss":
            self.retry_count += 1
            self.belief_radius = min(0.20, self.belief_radius + 0.025)
            info["retry_count"] = self.retry_count
            info["belief_radius"] = self.belief_radius
        elif info.get("success"):
            self.belief_radius = 0.025
            info["belief_radius"] = self.belief_radius

    def _integrate_observation(self, obs: dict[str, Any]) -> None:
        obs_time = int(obs.get("time", -1))
        if obs_time == self._last_integrated_time:
            return
        self._last_integrated_time = obs_time

        detections = obs.get("detections", [])
        if not detections:
            return

        detection = detections[0]
        position = np.asarray(detection["position"], dtype=float)
        confidence = float(detection.get("confidence", 0.5))

        if self.belief_mean is None:
            self.belief_mean = position.copy()
        else:
            alpha = np.clip(0.35 + 0.45 * confidence, 0.35, 0.80)
            self.belief_mean = alpha * self.belief_mean + (1.0 - alpha) * position

        self.belief_radius = max(0.035, self.belief_radius * 0.72)


def run(seed: int = 3, render: bool = True, max_steps: int = 40) -> Trace:
    env = Tabletop2D(seed=seed)
    agent = PickAndRetryAgent()
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
            env.render(agent=agent, info=info)

        if done:
            break

    return trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    trace = run(seed=args.seed, render=not args.no_render, max_steps=args.max_steps)
    failures = [failure.kind for failure in trace.failures()]
    picked = bool(trace.infos and trace.infos[-1].get("success"))
    print(
        f"picked={picked} steps={len(trace.actions)} "
        f"failures={failures} total_reward={sum(trace.rewards):.2f}"
    )

    if not args.no_render:
        import matplotlib.pyplot as plt

        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
