"""Options framework with interrupts on a battery-aware navigation task.

Each option is a triple `(initiation_set, intra_option_policy, beta)`
where beta is the termination condition. A separate meta-policy chooses
which option to run, and is allowed to *interrupt* a still-running
option when a better option becomes available.

This example uses two options:

* `GoToGoalOption` - velocity points at the goal; terminates when the
  robot is inside the goal radius.
* `DockAndChargeOption` - velocity points at the charger when far,
  velocity is zero while at the charger and the battery recharges;
  terminates when the battery is full enough.

The interrupt rule is: if the battery drops below `battery_low` while
any option other than `DockAndChargeOption` is running, the meta-policy
interrupts and switches to docking. After docking finishes, control
returns to the meta-policy which restarts `GoToGoalOption`.

This is structurally different from `08_interactive_mpc.py`, where a
single policy folds avoidance into the cost, and from
`25_clear_path_before_pick.py`, where recovery is triggered by a
*precondition failure* on a single skill. Here, the interruption is a
*runtime preference change* between two equally valid sub-policies,
triggered by a state-derived signal (battery level).

Success: robot reaches the goal radius with positive battery.
Failure: battery_drained (terminal - battery hit zero before docking),
option_not_initiated (recoverable - the meta-policy tried to start an
option whose initiation set did not include the current state),
timeout (terminal).
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

from pir.core.types import Failure, StepResult, Trace


@dataclass(frozen=True)
class OptionsConfig:
    dt: float = 1.0
    speed: float = 0.40
    goal_radius: float = 0.30
    charger_radius: float = 0.45
    battery_drain_per_move: float = 0.020
    battery_recharge_per_step: float = 0.18
    battery_low: float = 0.50
    battery_full: float = 0.95
    world_min: float = 0.0
    world_max: float = 10.0


class OptionsRobotWorld:
    """Continuous 2D world with one robot, one goal, one charger, one battery."""

    def __init__(
        self,
        *,
        seed: int = 0,
        max_steps: int = 160,
        start: tuple[float, float] = (1.0, 1.0),
        goal: tuple[float, float] = (8.6, 8.4),
        charger: tuple[float, float] = (5.0, 1.2),
        initial_battery: float = 1.0,
    ) -> None:
        self.config = OptionsConfig()
        self.max_steps = max_steps
        self.seed = seed
        self._start = np.asarray(start, dtype=float)
        self._goal = np.asarray(goal, dtype=float)
        self._charger = np.asarray(charger, dtype=float)
        self._initial_battery = float(initial_battery)
        self.robot = self._start.copy()
        self.battery = self._initial_battery
        self.step_count = 0
        self.trajectory: list[np.ndarray] = [self.robot.copy()]
        self._fig: Any | None = None
        self._ax: Any | None = None

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            self.seed = seed
        self.robot = self._start.copy()
        self.battery = self._initial_battery
        self.step_count = 0
        self.trajectory = [self.robot.copy()]
        return self.observe()

    def observe(self) -> dict[str, Any]:
        cfg = self.config
        return {
            "robot": self.robot.copy(),
            "goal": self._goal.copy(),
            "charger": self._charger.copy(),
            "battery": float(self.battery),
            "battery_low": cfg.battery_low,
            "battery_full": cfg.battery_full,
            "step": self.step_count,
            "at_charger": bool(
                np.linalg.norm(self.robot - self._charger) <= cfg.charger_radius
            ),
            "at_goal": bool(
                np.linalg.norm(self.robot - self._goal) <= cfg.goal_radius
            ),
        }

    def step(self, action: np.ndarray) -> StepResult:
        cfg = self.config
        self.step_count += 1
        velocity = np.asarray(action, dtype=float)
        speed = float(np.linalg.norm(velocity))
        if speed > cfg.speed:
            velocity = velocity / speed * cfg.speed
            speed = cfg.speed

        moving = speed > 1e-6
        self.robot = np.clip(
            self.robot + velocity * cfg.dt,
            cfg.world_min,
            cfg.world_max,
        )
        self.trajectory.append(self.robot.copy())

        at_charger = bool(np.linalg.norm(self.robot - self._charger) <= cfg.charger_radius)
        if moving:
            self.battery = max(0.0, self.battery - cfg.battery_drain_per_move)
        elif at_charger:
            self.battery = min(1.0, self.battery + cfg.battery_recharge_per_step)

        goal_distance = float(np.linalg.norm(self._goal - self.robot))
        at_goal = goal_distance <= cfg.goal_radius
        drained = self.battery <= 1e-9
        timed_out = self.step_count >= self.max_steps

        info: dict[str, Any] = {
            "success": at_goal,
            "goal_distance": goal_distance,
            "battery": float(self.battery),
            "at_charger": at_charger,
            "at_goal": at_goal,
            "moving": moving,
            "velocity": velocity.tolist(),
        }
        if drained and not at_goal:
            info["failure"] = Failure(
                kind="battery_drained",
                message="Battery hit zero before docking.",
                recoverable=False,
            )
        elif timed_out and not at_goal:
            info["failure"] = Failure(
                kind="timeout",
                message="Did not reach the goal before max_steps.",
                recoverable=False,
            )

        done = at_goal or drained or timed_out
        reward = -goal_distance - 0.01 - (1.0 if drained else 0.0)
        return StepResult(self.observe(), reward, done, info)

    def render(self, agent: "OptionsMetaPolicy", info: dict[str, Any]) -> None:
        import matplotlib.pyplot as plt

        if self._fig is None or self._ax is None:
            plt.ion()
            self._fig, self._ax = plt.subplots(figsize=(6.0, 6.0))
        ax = self._ax
        ax.clear()
        _draw_scene(ax, self, agent, info)
        self._fig.canvas.draw_idle()
        plt.pause(0.001)


def _draw_scene(
    ax: Any,
    env: OptionsRobotWorld,
    agent: "OptionsMetaPolicy",
    info: dict[str, Any],
) -> None:
    import matplotlib.patches as mpatches

    cfg = env.config
    ax.set_xlim(cfg.world_min, cfg.world_max)
    ax.set_ylim(cfg.world_min, cfg.world_max)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_title("Options with interrupts (go_to_goal vs dock_and_charge)")

    charger = mpatches.Circle(
        env._charger,
        cfg.charger_radius,
        color="tab:orange",
        alpha=0.3,
    )
    ax.add_patch(charger)
    ax.scatter([env._charger[0]], [env._charger[1]], marker="s", s=120, color="tab:orange")
    ax.text(env._charger[0], env._charger[1] - 0.7, "charger", ha="center", fontsize=8)

    goal = mpatches.Circle(env._goal, cfg.goal_radius, color="tab:green", alpha=0.3)
    ax.add_patch(goal)
    ax.scatter([env._goal[0]], [env._goal[1]], marker="*", s=160, color="tab:green")

    trajectory = np.asarray(env.trajectory)
    color_by_option = {
        "go_to_goal": "tab:blue",
        "dock_and_charge": "tab:orange",
        "none": "0.6",
    }
    for from_step, segment_option in enumerate(agent.trajectory_options):
        if from_step + 1 >= len(trajectory):
            break
        ax.plot(
            trajectory[from_step:from_step + 2, 0],
            trajectory[from_step:from_step + 2, 1],
            color=color_by_option.get(segment_option, "0.4"),
            alpha=0.8,
            linewidth=1.8,
        )

    robot_color = "tab:blue" if agent.current_option == "go_to_goal" else "tab:orange"
    if agent.current_option is None:
        robot_color = "0.4"
    ax.add_patch(mpatches.Circle(env.robot, 0.16, color=robot_color))

    bar_x = cfg.world_min + 0.25
    bar_y = cfg.world_max - 0.5
    bar_w = 2.8
    bar_h = 0.25
    ax.add_patch(
        mpatches.Rectangle((bar_x, bar_y), bar_w, bar_h, fill=False, color="black")
    )
    battery_fill = bar_w * float(env.battery)
    battery_color = "tab:green" if env.battery >= cfg.battery_low else "tab:red"
    ax.add_patch(
        mpatches.Rectangle(
            (bar_x, bar_y), battery_fill, bar_h, color=battery_color, alpha=0.7
        )
    )
    low_x = bar_x + bar_w * cfg.battery_low
    ax.plot([low_x, low_x], [bar_y, bar_y + bar_h], "--", color="black", alpha=0.5)
    ax.text(
        bar_x + bar_w + 0.1,
        bar_y + bar_h / 2,
        f"battery={env.battery:.2f}",
        va="center",
        fontsize=9,
    )

    status = (
        f"step={env.step_count}  option={agent.current_option or 'none'}\n"
        f"starts={agent.option_start_count} "
        f"interrupts={agent.option_interrupt_count} "
        f"battery_interrupts={agent.interrupts_due_to_battery_count}\n"
        f"dock_count={agent.dock_count} recharge_steps={agent.recharge_step_count}"
    )
    if "failure" in info:
        status += f"\nfailure={info['failure'].kind}"
    ax.text(0.02, 0.16, status, transform=ax.transAxes, va="top", fontsize=9, family="monospace")


class Option:
    """Skeleton (initiation_set, intra-policy, termination) triple."""

    name: str

    def can_initiate(self, obs: dict[str, Any]) -> bool:
        raise NotImplementedError

    def policy(self, obs: dict[str, Any], cfg: OptionsConfig) -> np.ndarray:
        raise NotImplementedError

    def beta(self, obs: dict[str, Any], cfg: OptionsConfig) -> bool:
        raise NotImplementedError


class GoToGoalOption(Option):
    name = "go_to_goal"

    def can_initiate(self, obs: dict[str, Any]) -> bool:
        return bool(obs["battery"] > 0.0) and not obs["at_goal"]

    def policy(self, obs: dict[str, Any], cfg: OptionsConfig) -> np.ndarray:
        delta = obs["goal"] - obs["robot"]
        distance = float(np.linalg.norm(delta))
        if distance < 1e-9:
            return np.zeros(2, dtype=float)
        speed = min(cfg.speed, distance)
        return delta / distance * speed

    def beta(self, obs: dict[str, Any], cfg: OptionsConfig) -> bool:
        return bool(obs["at_goal"])


class DockAndChargeOption(Option):
    name = "dock_and_charge"

    def can_initiate(self, obs: dict[str, Any]) -> bool:
        return bool(obs["battery"] < cfg_battery_full(obs))

    def policy(self, obs: dict[str, Any], cfg: OptionsConfig) -> np.ndarray:
        if obs["at_charger"]:
            return np.zeros(2, dtype=float)
        delta = obs["charger"] - obs["robot"]
        distance = float(np.linalg.norm(delta))
        if distance < 1e-9:
            return np.zeros(2, dtype=float)
        speed = min(cfg.speed, distance)
        return delta / distance * speed

    def beta(self, obs: dict[str, Any], cfg: OptionsConfig) -> bool:
        return bool(obs["battery"] >= cfg.battery_full)


def cfg_battery_full(obs: dict[str, Any]) -> float:
    return float(obs.get("battery_full", 0.95))


class OptionsMetaPolicy:
    """Meta-policy that picks among options and may interrupt them mid-run."""

    def __init__(self, config: OptionsConfig) -> None:
        self.config = config
        self.options: dict[str, Option] = {
            "go_to_goal": GoToGoalOption(),
            "dock_and_charge": DockAndChargeOption(),
        }
        self.reset()

    def reset(self) -> None:
        self.current_option: str | None = None
        self.option_start_count = 0
        self.option_interrupt_count = 0
        self.option_termination_count = 0
        self.interrupts_due_to_battery_count = 0
        self.dock_count = 0
        self.recharge_step_count = 0
        self.trajectory_options: list[str] = []
        self.last_velocity: np.ndarray | None = None

    def _should_interrupt(self, obs: dict[str, Any]) -> tuple[bool, str | None]:
        if (
            obs["battery"] < self.config.battery_low
            and self.current_option != "dock_and_charge"
        ):
            return True, "dock_and_charge"
        return False, None

    def _pick_default_option(self, obs: dict[str, Any]) -> str | None:
        if obs["at_goal"]:
            return None
        if obs["battery"] < self.config.battery_low:
            return "dock_and_charge"
        return "go_to_goal"

    def _start_option(self, name: str, obs: dict[str, Any]) -> bool:
        option = self.options[name]
        if not option.can_initiate(obs):
            return False
        self.current_option = name
        self.option_start_count += 1
        if name == "dock_and_charge":
            self.dock_count += 1
        return True

    def act(self, obs: dict[str, Any]) -> np.ndarray:
        cfg = self.config

        if self.current_option is not None:
            option = self.options[self.current_option]
            if option.beta(obs, cfg):
                self.option_termination_count += 1
                self.current_option = None

        if self.current_option is not None:
            should_interrupt, replacement = self._should_interrupt(obs)
            if should_interrupt and replacement is not None:
                option = self.options[replacement]
                if option.can_initiate(obs):
                    self.option_interrupt_count += 1
                    self.interrupts_due_to_battery_count += 1
                    self.current_option = None

        if self.current_option is None:
            default = self._pick_default_option(obs)
            if default is None:
                self.last_velocity = np.zeros(2, dtype=float)
                self.trajectory_options.append("none")
                return self.last_velocity
            if not self._start_option(default, obs):
                self.last_velocity = np.zeros(2, dtype=float)
                self.trajectory_options.append("none")
                return self.last_velocity

        option = self.options[self.current_option]
        velocity = option.policy(obs, cfg)
        if self.current_option == "dock_and_charge" and obs["at_charger"]:
            self.recharge_step_count += 1
        self.last_velocity = velocity
        self.trajectory_options.append(self.current_option)
        return velocity

    def update(
        self, obs: dict[str, Any], reward: float, info: dict[str, Any]
    ) -> None:
        del obs, reward, info

    def info(self) -> dict[str, Any]:
        return {
            "current_option": self.current_option,
            "option_start_count": int(self.option_start_count),
            "option_interrupt_count": int(self.option_interrupt_count),
            "option_termination_count": int(self.option_termination_count),
            "interrupts_due_to_battery_count": int(self.interrupts_due_to_battery_count),
            "dock_count": int(self.dock_count),
            "recharge_step_count": int(self.recharge_step_count),
        }


def run(seed: int = 0, render: bool = True, max_steps: int = 160) -> Trace:
    env = OptionsRobotWorld(seed=seed, max_steps=max_steps)
    agent = OptionsMetaPolicy(env.config)
    obs = env.reset(seed=seed)
    agent.reset()
    trace = Trace()

    for _ in range(max_steps):
        action = agent.act(obs)
        obs, reward, done, info = env.step(action).as_tuple()
        agent.update(obs, reward, info)
        info.update(agent.info())
        trace.append(obs, action, reward, info)
        if render:
            env.render(agent=agent, info=info)
        if done:
            break

    return trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=160)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    trace = run(seed=args.seed, render=not args.no_render, max_steps=args.max_steps)
    final_info = trace.infos[-1] if trace.infos else {}
    failures = [failure.kind for failure in trace.failures()]
    print(
        f"success={final_info.get('success', False)} "
        f"steps={len(trace.actions)} "
        f"battery_final={final_info.get('battery', 0.0):.2f} "
        f"starts={final_info.get('option_start_count', 0)} "
        f"interrupts={final_info.get('option_interrupt_count', 0)} "
        f"battery_interrupts={final_info.get('interrupts_due_to_battery_count', 0)} "
        f"dock_count={final_info.get('dock_count', 0)} "
        f"recharge_steps={final_info.get('recharge_step_count', 0)} "
        f"failures={failures}"
    )

    if not args.no_render:
        import matplotlib.pyplot as plt

        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
