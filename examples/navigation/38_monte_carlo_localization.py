"""Localize with a particle filter, get kidnapped, and recover by injecting particles.

Monte Carlo Localization (MCL) tracks a robot's pose on a known map with a
particle filter: each particle is a pose hypothesis, particles are pushed
through the motion model, weighted by how well their predicted range readings
match the real sensor, and resampled toward the likely poses. Plain MCL
converges beautifully from a uniform prior (global localization) — and then
fails catastrophically at the *kidnapped robot* problem: once the cloud has
collapsed onto the true pose, there are no particles left anywhere else, so if
the robot is suddenly teleported the filter has nothing to recover with.

Augmented MCL (Thrun et al., *Probabilistic Robotics*, Table 8.3) is the fix and
the lesson of this repo in one trick: track a fast and a slow running average of
the measurement likelihood, and when the fast average drops below the slow one —
i.e. the sensor suddenly disagrees with the belief, the signature of a failure —
inject random particles in proportion to that drop. Those random particles are
the recovery: a few of them land near the new true pose, get high weight, and
the cloud re-collapses there.

This example runs all three acts on one map: global localization from scratch,
a scheduled kidnap, and augmented recovery — with plain MCL available via
``--no-augment`` to watch it fail to recover.

Success: the estimate is within `localized_tol` of the true pose at the end,
*after* having been kidnapped and recovered.
Failure: kidnapped (recoverable - the scheduled teleport) and localization_lost
(recoverable - emitted each step the filter is diverged after having localized).

References:
  * S. Thrun, W. Burgard, D. Fox, "Probabilistic Robotics," MIT Press, 2005 -
    Augmented_MCL, Table 8.3 (w_slow / w_fast and random particle injection).
  * D. Fox, W. Burgard, F. Dellaert, S. Thrun, "Monte Carlo Localization for
    Mobile Robots," ICRA 1999.
  * The ROS Navigation `amcl` package implements this adaptive recovery.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pir.core.random import make_rng
from pir.core.types import Failure, StepResult, Trace

# Irregular landmarks with no reflective/rotational symmetry, so range-only
# readings pin down a single pose: a teleported robot lands on a genuinely
# different range signature (no symmetric "ghost" pose that the old, collapsed
# cloud could still explain) and only injection can recover.
LANDMARKS = np.array(
    [[1.0, 2.0], [2.5, 8.5], [5.0, 1.0], [8.5, 3.0], [9.0, 8.0], [6.0, 6.0], [3.5, 5.0]],
    dtype=float,
)


@dataclass
class MCLConfig:
    world_size: float = 10.0
    sensor_range: float = 6.0
    range_noise: float = 0.25
    bearing_noise: float = 0.10
    motion_noise: float = 0.08
    kidnap_step: int = 30
    localized_tol: float = 0.8
    max_steps: int = 60
    start: tuple[float, float] = (3.2, 3.2)
    kidnap_to: tuple[float, float] = (8.5, 8.2)


def loop_control(t: int) -> np.ndarray:
    """Position-independent velocity command tracing a small circle (~radius 1.3).

    Because it depends only on the step index, not on where the robot is, the
    true robot and every particle move *congruently*: after a kidnap the cloud
    keeps tracing the old loop while the true robot traces a congruent loop
    elsewhere, so plain MCL stays stranded the full kidnap distance away. This is
    what isolates the recovery to particle injection (no boundary funnel that
    would let a stuck filter coincidentally drift back).
    """
    return 0.5 * np.array([np.cos(0.4 * t), np.sin(0.4 * t)])


class MCLWorld:
    """Known map with landmarks; the true robot drives a loop and is kidnapped once."""

    def __init__(self, *, seed: int | None = 0, config: MCLConfig | None = None) -> None:
        self.config = config or MCLConfig()
        self.seed = seed
        self.reset(seed=seed)

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            self.seed = seed
        self.rng = make_rng(self.seed)
        self.time = 0
        self.kidnapped_at: int | None = None
        self.pose = np.asarray(self.config.start, dtype=float).copy()
        return self.observe(np.zeros(2))

    def observe(self, control: np.ndarray) -> dict[str, Any]:
        deltas = LANDMARKS - self.pose
        distances = np.linalg.norm(deltas, axis=1)
        bearings = np.arctan2(deltas[:, 1], deltas[:, 0])
        visible = distances <= self.config.sensor_range
        noisy_r = distances + self.rng.normal(0.0, self.config.range_noise, size=distances.shape)
        noisy_b = bearings + self.rng.normal(0.0, self.config.bearing_noise, size=bearings.shape)
        # Range + bearing per landmark (a known compass / world-frame heading).
        # The joint likelihood is sharply peaked, so the cloud has no smooth
        # cross-map gradient to creep along — after a kidnap a plain filter is
        # stranded in a likelihood valley, and only injection escapes it.
        observations = {
            int(i): (float(noisy_r[i]), float(noisy_b[i])) for i in np.where(visible)[0]
        }
        return {
            "time": self.time,
            "control": np.asarray(control, dtype=float).copy(),
            "landmarks": observations,
            "kidnapped": self.kidnapped_at == self.time,
        }

    def step(self, action: dict[str, Any]) -> StepResult:
        self.time += 1
        control = np.asarray(action.get("control", np.zeros(2)), dtype=float)
        noise = self.rng.normal(0.0, self.config.motion_noise, size=2)
        self.pose = np.clip(self.pose + control + noise, 0.0, self.config.world_size)

        info: dict[str, Any] = {"time": self.time, "true_pose": self.pose.copy()}
        if self.time == self.config.kidnap_step:
            # Teleport to a fixed far pose with a different landmark signature, so
            # a collapsed cloud cannot creep back along a likelihood gradient and
            # has no symmetric ghost to hide in. Only injecting fresh particles
            # recovers — the kidnapped robot problem.
            self.kidnapped_at = self.time
            self.pose = np.asarray(self.config.kidnap_to, dtype=float).copy()
            info["failure"] = Failure("kidnapped", "robot teleported without odometry", True)

        obs = self.observe(control)
        done = self.time >= self.config.max_steps
        return StepResult(obs, 0.0, done, info)


class AugmentedMCL:
    """A particle filter with Thrun's w_slow / w_fast random-injection recovery."""

    def __init__(
        self,
        *,
        num_particles: int = 800,
        augment: bool = True,
        config: MCLConfig | None = None,
        seed: int | None = 0,
    ) -> None:
        self.num_particles = num_particles
        self.augment = augment
        self.config = config or MCLConfig()
        self.alpha_slow = 0.05
        self.alpha_fast = 0.5
        self.rng = make_rng(None if seed is None else seed + 7919)
        self.reset()

    def reset(self) -> None:
        size = self.config.world_size
        self.particles = self.rng.uniform(0.0, size, size=(self.num_particles, 2))
        self.weights = np.full(self.num_particles, 1.0 / self.num_particles)
        self.w_slow = 0.0
        self.w_fast = 0.0
        self.localized_once = False
        self._estimate: np.ndarray | None = None
        self._t = 0

    def act(self, obs: dict[str, Any]) -> dict[str, Any]:
        # Open-loop control comes from the trajectory; localization does not steer.
        self._t = int(obs.get("time", self._t))
        return {"control": loop_control(self._t)}

    def update(self, obs: dict[str, Any], reward: float, info: dict[str, Any]) -> None:
        self._predict(obs.get("control", np.zeros(2)))
        avg_likelihood = self._correct(obs.get("landmarks", {}))
        self._update_running_averages(avg_likelihood)
        # Estimate from the *weighted* particles before resampling: a handful of
        # freshly injected particles sitting on the sharp likelihood peak carry
        # almost all the weight, so the estimate snaps to the recovered pose even
        # while most of the cloud is still scattered. A plain particle mean would
        # be dragged toward the map center by those scattered particles.
        self._estimate = self.weights @ self.particles
        inject_ratio = self._resample()

        estimate = self.estimate()
        true_pose = info.get("true_pose")
        error = float(np.linalg.norm(estimate - true_pose)) if true_pose is not None else float("nan")
        localized = error <= self.config.localized_tol
        if localized:
            self.localized_once = True
        elif self.localized_once and not info.get("kidnapped"):
            info.setdefault(
                "failure", Failure("localization_lost", "estimate diverged from true pose", True)
            )

        info["estimate"] = estimate
        info["error"] = error
        info["spread"] = self.spread()
        info["w_slow"] = float(self.w_slow)
        info["w_fast"] = float(self.w_fast)
        info["inject_ratio"] = float(inject_ratio)
        info["localized"] = bool(localized)

    # --- particle filter internals ---
    def _predict(self, control: np.ndarray) -> None:
        control = np.asarray(control, dtype=float)
        noise = self.rng.normal(0.0, self.config.motion_noise, size=self.particles.shape)
        self.particles = np.clip(self.particles + control + noise, 0.0, self.config.world_size)

    def _correct(self, landmarks: dict[int, tuple[float, float]]) -> float:
        if not landmarks:
            return float(np.mean(self.weights))
        # Absolute (un-normalized) measurement likelihood per particle. It must be
        # absolute — NOT rescaled by its own max — because Augmented MCL reads the
        # *level* of this average to detect a kidnap: after a teleport every
        # particle is wrong, the average collapses toward zero, and that collapse
        # is the recovery trigger. Rescaling by the max would hide it.
        log_w = np.zeros(self.num_particles)
        sigma_r = self.config.range_noise
        sigma_b = self.config.bearing_noise
        for idx, (measured_r, measured_b) in landmarks.items():
            delta = LANDMARKS[idx] - self.particles
            predicted_r = np.linalg.norm(delta, axis=1)
            predicted_b = np.arctan2(delta[:, 1], delta[:, 0])
            bearing_err = np.arctan2(
                np.sin(predicted_b - measured_b), np.cos(predicted_b - measured_b)
            )
            log_w += -0.5 * ((predicted_r - measured_r) / sigma_r) ** 2
            log_w += -0.5 * (bearing_err / sigma_b) ** 2
        likelihood = np.exp(log_w)
        avg_likelihood = float(np.mean(likelihood))
        total = float(likelihood.sum())
        if total > 0.0:
            self.weights = likelihood / total
        else:
            # Every particle is hopeless (a fresh kidnap): keep weights uniform so
            # resampling does not amplify noise; injection does the recovery.
            self.weights = np.full(self.num_particles, 1.0 / self.num_particles)
        return avg_likelihood

    def _update_running_averages(self, avg_likelihood: float) -> None:
        self.w_slow += self.alpha_slow * (avg_likelihood - self.w_slow)
        self.w_fast += self.alpha_fast * (avg_likelihood - self.w_fast)

    def _resample(self) -> float:
        inject_ratio = 0.0
        if self.augment and self.w_slow > 0.0:
            inject_ratio = max(0.0, 1.0 - self.w_fast / self.w_slow)

        size = self.config.world_size
        # Low-variance (systematic) resampling, with random injection per draw.
        positions = (self.rng.random() + np.arange(self.num_particles)) / self.num_particles
        cumulative = np.cumsum(self.weights)
        new_particles = np.empty_like(self.particles)
        inject_mask = self.rng.random(self.num_particles) < inject_ratio
        j = 0
        for i in range(self.num_particles):
            if inject_mask[i]:
                new_particles[i] = self.rng.uniform(0.0, size, size=2)
                continue
            while j < self.num_particles - 1 and positions[i] > cumulative[j]:
                j += 1
            new_particles[i] = self.particles[j]
        self.particles = new_particles
        self.weights = np.full(self.num_particles, 1.0 / self.num_particles)
        return inject_ratio

    def estimate(self) -> np.ndarray:
        est = getattr(self, "_estimate", None)
        return self.particles.mean(axis=0) if est is None else est

    def spread(self) -> float:
        return float(np.linalg.norm(self.particles - self.estimate(), axis=1).mean())


def run(
    seed: int = 0,
    render: bool = True,
    max_steps: int | None = None,
    augment: bool = True,
) -> Trace:
    config = MCLConfig()
    if max_steps is not None:
        config.max_steps = max_steps
    world = MCLWorld(seed=seed, config=config)
    obs = world.reset(seed=seed)
    agent = AugmentedMCL(augment=augment, config=config, seed=seed)
    agent.reset()
    trace = Trace()

    for _ in range(config.max_steps):
        action = agent.act(obs)
        result = world.step(action)
        obs, reward, done, info = result.as_tuple()
        agent.update(obs, reward, info)
        trace.append(obs, action, reward, info)

        if render:
            _render(world, agent, info)

        if done:
            break

    final = trace.infos[-1]
    kidnapped = any(i.get("failure") and i["failure"].kind == "kidnapped" for i in trace.infos)
    final["success"] = bool(kidnapped and final.get("localized"))
    final["recovered"] = final["success"]
    return trace


def _render(world: MCLWorld, agent: AugmentedMCL, info: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    if not hasattr(world, "_fig") or world._fig is None:
        plt.ion()
        world._fig, world._ax = plt.subplots(figsize=(5, 5))
    ax = world._ax
    ax.clear()
    ax.set_title("Augmented MCL: localize, get kidnapped, recover")
    ax.set_xlim(0, world.config.world_size)
    ax.set_ylim(0, world.config.world_size)
    ax.set_aspect("equal", adjustable="box")
    ax.scatter(agent.particles[:, 0], agent.particles[:, 1], s=4, c="tab:blue", alpha=0.25, label="particles")
    ax.scatter(*LANDMARKS.T, marker="^", c="black", s=60, label="landmarks")
    ax.plot(*world.pose, marker="o", c="tab:green", markersize=10, label="true pose")
    est = info.get("estimate")
    if est is not None:
        ax.plot(*est, marker="x", c="tab:red", markersize=12, label="estimate")
    ax.text(0.02, 0.97, f"error={info.get('error', float('nan')):.2f} inject={info.get('inject_ratio', 0):.2f}",
            transform=ax.transAxes, va="top")
    ax.legend(loc="lower left", fontsize=7)
    world._fig.canvas.draw_idle()
    plt.pause(0.05)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--no-augment", action="store_true", help="plain MCL (cannot recover)")
    args = parser.parse_args()

    trace = run(
        seed=args.seed,
        render=not args.no_render,
        max_steps=args.max_steps,
        augment=not args.no_augment,
    )
    final = trace.infos[-1]
    errors = [i.get("error", float("nan")) for i in trace.infos]
    kidnap = next((i for i, x in enumerate(trace.infos) if x.get("failure") and x["failure"].kind == "kidnapped"), None)
    print(
        f"recovered={final.get('success')} final_error={final.get('error'):.2f} "
        f"min_error_before_kidnap={min(errors[:kidnap]) if kidnap else float('nan'):.2f} "
        f"augment={not args.no_augment}"
    )

    if not args.no_render:
        import matplotlib.pyplot as plt

        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
