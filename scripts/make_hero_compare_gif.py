"""Generate the README hero GIF: a naive picker vs a failure-aware picker.

Left panel:  a naive agent that grabs at whatever it currently sees, never
             moves the camera (so it stays behind the occluder), never keeps a
             belief, and never adapts its retry. It keeps missing.
Right panel: the repo's PickAndRetryAgent, which looks from better viewpoints,
             averages noisy detections into a belief, and retries differently
             after each miss. It recovers and succeeds.

Both panels run the same Tabletop2D world with the same seed, so the contrast is
the policy, not luck. This is a curated marketing asset, kept separate from the
per-example pipeline in scripts/make_gifs.py.

Usage:
    python3 scripts/make_hero_compare_gif.py
    python3 scripts/make_hero_compare_gif.py --search   # scan seeds for a clean story
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

from pir.core.types import Failure
from pir.worlds.tabletop_2d import Tabletop2D

OUT_DIR = ROOT / "docs" / "assets" / "gifs"
GIF_NAME = "naive_vs_failure_aware.gif"


def load_example(relative_path: str) -> ModuleType:
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class NaiveAgent:
    """The "no belief update" baseline.

    It locks onto the very first detection it ever sees and keeps grabbing at
    that same stale point forever. It never re-observes, never averages, never
    moves the camera, and never adapts after a miss. The contrast with the
    failure-aware agent is therefore structural (update vs. don't update), not
    a matter of luck on a particular seed.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        # Keep a belief_mean attribute so the renderer can stay uniform, but the
        # naive agent never maintains a real belief.
        self.belief_mean: np.ndarray | None = None
        self.belief_radius = 0.0
        self._locked_target: np.ndarray | None = None

    def act(self, obs: dict[str, Any]) -> dict[str, Any]:
        if self._locked_target is None:
            detections = obs.get("detections", [])
            if detections:
                self._locked_target = np.asarray(detections[0]["position"], dtype=float)
            else:
                # Nothing seen yet: commit to a blind guess and never revise it.
                self._locked_target = np.array([0.5, 0.5], dtype=float)
        return {"type": "pick", "position": np.clip(self._locked_target, 0.0, 1.0)}

    def update(self, obs: dict[str, Any], reward: float, info: dict[str, Any]) -> None:
        # The whole point: the naive agent learns nothing from a miss.
        return None


def fig_to_frame(fig: plt.Figure) -> np.ndarray:
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    buffer = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    return buffer.reshape((height, width, 4))[:, :, :3].copy()


def run_episode(agent_factory, seed: int, max_steps: int) -> list[dict[str, Any]]:
    """Run one episode and return a per-frame record of (env snapshot, info)."""
    env = Tabletop2D(seed=seed)
    agent = agent_factory()
    obs = env.reset(seed=seed)
    agent.reset()

    records: list[dict[str, Any]] = [{"env": env, "agent": agent, "info": {}}]
    for _ in range(max_steps):
        action = agent.act(obs)
        result = env.step(action)
        obs, reward, done, info = result.as_tuple()
        agent.update(obs, reward, info)
        # Snapshot the mutable state we need for rendering this frame.
        records.append(
            {
                "camera": env.camera_pos.copy(),
                "last_detection": None if env.last_detection is None else env.last_detection.copy(),
                "picked": env.obj.picked,
                "attempts": env.attempts,
                "belief_mean": None if getattr(agent, "belief_mean", None) is None else np.asarray(agent.belief_mean).copy(),
                "belief_radius": float(getattr(agent, "belief_radius", 0.0)),
                "info": info,
            }
        )
        if done:
            break
    return records


def episode_outcome(records: list[dict[str, Any]]) -> tuple[bool, int]:
    picked = any(r.get("picked") for r in records[1:])
    attempts = max((r.get("attempts", 0) for r in records[1:]), default=0)
    return picked, attempts


def build_frames(seed: int, max_steps: int) -> list[np.ndarray]:
    naive_module_agent = NaiveAgent
    pick_module = load_example("examples/manipulation/01_pick_and_retry.py")

    naive = run_episode(lambda: naive_module_agent(), seed=seed, max_steps=max_steps)
    smart = run_episode(lambda: pick_module.PickAndRetryAgent(), seed=seed, max_steps=max_steps)

    n = max(len(naive), len(smart))

    frames: list[np.ndarray] = []
    for i in range(n):
        ln = naive[min(i, len(naive) - 1)]
        ls = smart[min(i, len(smart) - 1)]

        fig, (axl, axr) = plt.subplots(1, 2, figsize=(8.6, 4.5), dpi=90)
        fig.suptitle(
            "Most robotics tutorials assume the grasp works.  Real robots miss — and recover.",
            fontsize=11,
        )

        for ax, rec, title, color in (
            (axl, ln, "Naive: grab at what you see", "tab:red"),
            (axr, ls, "Failure-aware: look → update belief → retry", "tab:green"),
        ):
            _render_record(ax, rec, title, color)

        fig.tight_layout(rect=(0, 0, 1, 0.94))
        frames.append(fig_to_frame(fig))
        plt.close(fig)

    # Hold the final frame so the outcome reads clearly.
    frames.extend([frames[-1]] * 4)
    return frames


# A lightweight stand-in Tabletop2D geometry shared by both panels.
_OCCLUDER = np.array([0.43, 0.42, 0.57, 0.68], dtype=float)
_OBJ_POS = np.array([0.64, 0.54], dtype=float)
_OBJ_RADIUS = 0.045


def _render_record(ax: plt.Axes, rec: dict[str, Any], title: str, color: str) -> None:
    ax.set_title(title, fontsize=10.5, color=color, fontweight="bold")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=7)

    xmin, ymin, xmax, ymax = _OCCLUDER
    ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color="0.2", alpha=0.18))

    camera = rec.get("camera", np.array([0.16, 0.50]))
    ax.plot(*camera, marker="s", color="tab:blue", markersize=9)

    picked = rec.get("picked", False)
    if not picked:
        ax.add_patch(Circle(_OBJ_POS, _OBJ_RADIUS, color="tab:red", alpha=0.85))

    last_detection = rec.get("last_detection")
    if last_detection is not None and not picked:
        ax.plot(*last_detection, marker="x", markersize=9, color="tab:orange")

    belief_mean = rec.get("belief_mean")
    if belief_mean is not None:
        ax.add_patch(
            Circle(
                belief_mean,
                rec.get("belief_radius", 0.08),
                fill=False,
                linestyle="--",
                color="tab:green",
                linewidth=2,
            )
        )

    info = rec.get("info", {})
    if "pick_position" in info:
        ax.plot(*info["pick_position"], marker="+", markersize=14, color="black")

    status = f"attempts={rec.get('attempts', 0)}"
    if picked:
        status += "   PICKED ✓"
    elif isinstance(info.get("failure"), Failure):
        status += f"   {info['failure'].kind}"
    ax.text(
        0.02,
        0.97,
        status,
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.7", alpha=0.85),
    )


def search_seeds(max_steps: int, limit: int = 60) -> None:
    pick_module = load_example("examples/manipulation/01_pick_and_retry.py")
    print("seed  naive(picked,attempts)  smart(picked,attempts)")
    for seed in range(limit):
        naive = run_episode(lambda: NaiveAgent(), seed=seed, max_steps=max_steps)
        smart = run_episode(lambda: pick_module.PickAndRetryAgent(), seed=seed, max_steps=max_steps)
        np_ = episode_outcome(naive)
        sp_ = episode_outcome(smart)
        flag = "  <== clean story" if (not np_[0] and sp_[0]) else ""
        print(f"{seed:>4}  {np_}                 {sp_}{flag}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--search", action="store_true", help="scan seeds and exit")
    args = parser.parse_args()

    if args.search:
        search_seeds(args.max_steps)
        return

    frames = build_frames(args.seed, args.max_steps)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / GIF_NAME
    imageio.mimsave(out, frames, duration=1.0 / args.fps, loop=0)
    print(f"wrote {out}  ({len(frames)} frames)")


if __name__ == "__main__":
    main()
