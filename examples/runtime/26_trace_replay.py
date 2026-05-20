"""Record a headless trace, then replay observations, actions, and infos.

This example uses the smallest runtime loop as the source run. The replay is
intentionally lightweight: it does not rerun physics or policy code. It walks
the recorded `Trace`, tracks cumulative reward, preserves failures, and can
render the observation/action history for inspection.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pir.core.types import Trace


SOURCE_EXAMPLE = ROOT / "examples" / "runtime" / "01_sense_act_loop.py"


def load_source_example() -> ModuleType:
    spec = importlib.util.spec_from_file_location("runtime_01_sense_act_loop", SOURCE_EXAMPLE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {SOURCE_EXAMPLE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def record_trace(seed: int = 0, max_steps: int = 40) -> Trace:
    source = load_source_example()
    return source.run(seed=seed, render_enabled=False, max_steps=max_steps)


def replay_trace(source_trace: Trace, render: bool = True, stride: int = 1) -> Trace:
    replayed = Trace()
    cumulative_reward = 0.0
    observed_path: list[np.ndarray] = []

    for index, (obs, action, reward, info) in enumerate(
        zip(
            source_trace.observations,
            source_trace.actions,
            source_trace.rewards,
            source_trace.infos,
            strict=True,
        )
    ):
        cumulative_reward += reward
        replay_info = dict(info)
        replay_info["replay_index"] = index
        replay_info["cumulative_reward"] = cumulative_reward
        replay_info["source_steps"] = len(source_trace.actions)
        replayed.append(obs, action, reward, replay_info)

        observed_path.append(np.asarray(obs["position"], dtype=float))
        if render and index % max(1, stride) == 0:
            render_replay(source_trace, index, observed_path, cumulative_reward)

    return replayed


def render_replay(
    trace: Trace,
    index: int,
    observed_path: list[np.ndarray],
    cumulative_reward: float,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    obs = trace.observations[index]
    action = np.asarray(trace.actions[index], dtype=float)
    position = np.asarray(obs["position"], dtype=float)

    if not hasattr(render_replay, "_fig"):
        plt.ion()
        render_replay._fig, render_replay._ax = plt.subplots(figsize=(5, 5))

    ax = render_replay._ax
    ax.clear()
    ax.set_title("trace replay: observation -> action -> info")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)

    path = np.asarray(observed_path)
    ax.plot(path[:, 0], path[:, 1], color="tab:orange", linewidth=2, label="observed path")
    ax.plot(*position, marker="x", color="tab:orange", markersize=9, label="observation")
    ax.plot(*obs["goal"], marker="*", markersize=14, color="tab:green", label="goal")
    ax.add_patch(
        Circle(
            obs["obstacle_center"],
            obs["obstacle_radius"],
            color="tab:red",
            alpha=0.28,
            label="obstacle",
        )
    )
    ax.arrow(
        position[0],
        position[1],
        action[0],
        action[1],
        head_width=0.018,
        color="black",
        length_includes_head=True,
    )

    info = trace.infos[index]
    status = f"step={index + 1}/{len(trace.actions)} reward_sum={cumulative_reward:.2f}"
    if info.get("success"):
        status += " success"
    if "failure" in info:
        status += f" failure={info['failure'].kind}"
    ax.text(0.02, 0.97, status, transform=ax.transAxes, va="top", fontsize=9)
    ax.legend(loc="lower right", fontsize=8)
    render_replay._fig.canvas.draw_idle()
    plt.pause(0.04)


def run(seed: int = 0, render: bool = True, max_steps: int = 40, stride: int = 1) -> Trace:
    source_trace = record_trace(seed=seed, max_steps=max_steps)
    return replay_trace(source_trace, render=render, stride=stride)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    trace = run(
        seed=args.seed,
        render=not args.no_render,
        max_steps=args.max_steps,
        stride=args.stride,
    )
    summary = trace.summary()
    print(
        f"replayed: steps={summary.steps} success={summary.success} "
        f"total_reward={summary.total_reward:.2f} failures={summary.failure_counts}"
    )

    if not args.no_render:
        import matplotlib.pyplot as plt

        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
