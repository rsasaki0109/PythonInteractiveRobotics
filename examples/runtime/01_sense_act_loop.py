"""The smallest closed-loop robotics example.

A point robot observes a noisy position, chooses a velocity, acts, and observes
again. The policy is intentionally simple: move toward the goal while repelling
away from an obstacle.
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


def observe(state: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    noisy_position = state["position"] + rng.normal(0.0, 0.018, size=2)
    return {
        "position": noisy_position,
        "goal": state["goal"].copy(),
        "obstacle_center": state["obstacle_center"].copy(),
        "obstacle_radius": state["obstacle_radius"],
    }


def policy(obs: dict[str, Any]) -> np.ndarray:
    position = obs["position"]
    goal = obs["goal"]
    obstacle_center = obs["obstacle_center"]
    obstacle_radius = obs["obstacle_radius"]

    to_goal = goal - position
    goal_distance = np.linalg.norm(to_goal)
    desired = to_goal / max(goal_distance, 1e-6)

    away = position - obstacle_center
    obstacle_distance = np.linalg.norm(away)
    clearance = obstacle_distance - obstacle_radius
    if clearance < 0.24:
        away_unit = away / max(obstacle_distance, 1e-6)
        tangent = np.array([-away_unit[1], away_unit[0]])
        if np.dot(tangent, to_goal) < 0.0:
            tangent = -tangent
        desired += 3.0 * away_unit * (0.24 - clearance)
        desired += 0.55 * tangent

    speed = 0.045
    norm = np.linalg.norm(desired)
    return speed * desired / max(norm, 1e-6)


def step(state: dict[str, Any], action: np.ndarray) -> tuple[float, bool, dict[str, Any]]:
    state["position"] = np.clip(state["position"] + action, 0.0, 1.0)
    state["trajectory"].append(state["position"].copy())

    obstacle_error = np.linalg.norm(state["position"] - state["obstacle_center"])
    goal_error = np.linalg.norm(state["position"] - state["goal"])
    info: dict[str, Any] = {
        "goal_error": float(goal_error),
        "obstacle_clearance": float(obstacle_error - state["obstacle_radius"]),
    }

    if obstacle_error <= state["obstacle_radius"]:
        info["failure"] = Failure("collision", "point robot entered the obstacle", False)
        return -1.0, True, info

    if goal_error < 0.04:
        info["success"] = True
        return 1.0, True, info

    return -0.01, False, info


def render(state: dict[str, Any], obs: dict[str, Any], action: np.ndarray) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    if state.get("fig") is None:
        plt.ion()
        state["fig"], state["ax"] = plt.subplots(figsize=(5, 5))

    ax = state["ax"]
    ax.clear()
    ax.set_title("sense -> act -> observe")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)

    trajectory = np.asarray(state["trajectory"])
    ax.plot(trajectory[:, 0], trajectory[:, 1], color="tab:blue", linewidth=2)
    ax.plot(*state["position"], marker="o", color="tab:blue", label="true pose")
    ax.plot(*obs["position"], marker="x", color="tab:orange", label="noisy observation")
    ax.plot(*state["goal"], marker="*", markersize=14, color="tab:green", label="goal")
    ax.add_patch(
        Circle(
            state["obstacle_center"],
            state["obstacle_radius"],
            color="tab:red",
            alpha=0.28,
            label="obstacle",
        )
    )
    ax.arrow(
        state["position"][0],
        state["position"][1],
        action[0],
        action[1],
        head_width=0.018,
        color="black",
        length_includes_head=True,
    )
    ax.legend(loc="lower right", fontsize=8)
    state["fig"].canvas.draw_idle()
    plt.pause(0.04)


def run(seed: int = 0, render_enabled: bool = True, max_steps: int = 120) -> Trace:
    rng = np.random.default_rng(seed)
    state: dict[str, Any] = {
        "position": np.array([0.12, 0.28], dtype=float),
        "goal": np.array([0.88, 0.76], dtype=float),
        "obstacle_center": np.array([0.50, 0.52], dtype=float),
        "obstacle_radius": 0.13,
        "trajectory": [np.array([0.12, 0.28], dtype=float)],
        "fig": None,
        "ax": None,
    }
    trace = Trace()
    obs = observe(state, rng)

    for _ in range(max_steps):
        action = policy(obs)
        reward, done, info = step(state, action)
        next_obs = observe(state, rng)
        trace.append(next_obs, action, reward, info)

        if render_enabled:
            render(state, next_obs, action)

        obs = next_obs
        if done:
            break

    return trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    trace = run(
        seed=args.seed,
        render_enabled=not args.no_render,
        max_steps=args.max_steps,
    )
    failures = trace.failures()
    status = "failed" if failures else "finished"
    print(f"{status}: steps={len(trace.actions)} total_reward={sum(trace.rewards):.2f}")

    if not args.no_render:
        import matplotlib.pyplot as plt

        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
