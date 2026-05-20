"""Generate small README GIFs from the lightweight examples."""

from __future__ import annotations

import argparse
import importlib.util
import sys
import warnings
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import imageio.v2 as imageio

warnings.filterwarnings("ignore", message="Unable to import Axes3D.*")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle, Rectangle

from pir.worlds.grid_world import DynamicObstacleGridWorld, GridWorld2D, FREE, OCCUPIED, UNKNOWN
from pir.worlds.tabletop_2d import Tabletop2D


OUT_DIR = ROOT / "docs" / "assets" / "gifs"


def load_example(relative_path: str) -> ModuleType:
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def fig_to_frame(fig: plt.Figure) -> np.ndarray:
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    buffer = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    return buffer.reshape((height, width, 4))[:, :, :3].copy()


def save_gif(name: str, frames: list[np.ndarray], fps: int = 8) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    imageio.mimsave(path, frames, duration=1.0 / fps, loop=0)
    return path


def render_grid_frame(
    env: GridWorld2D,
    *,
    title: str,
    agent_state: str | None = None,
    planned_path: list[tuple[int, int]] | None = None,
    current_frontier: tuple[int, int] | None = None,
    coverage: float | None = None,
    dynamic: bool = False,
) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(5.6, 4.2), dpi=80)
    display = np.zeros_like(env.known_map, dtype=int)
    display[env.known_map == UNKNOWN] = 0
    display[env.known_map == FREE] = 1
    display[env.known_map == OCCUPIED] = 2
    cmap = ListedColormap(["0.72", "white", "0.1"])
    ax.imshow(display, cmap=cmap, origin="upper", vmin=0, vmax=2)

    if env.last_scan is not None:
        for ray in env.last_scan.values():
            for row, col in ray["cells"]:
                ax.plot(col, row, ".", color="tab:cyan", markersize=4)

    rows = [cell[0] for cell in env.trajectory]
    cols = [cell[1] for cell in env.trajectory]
    ax.plot(cols, rows, color="tab:blue", linewidth=2)

    if planned_path:
        path_rows = [cell[0] for cell in planned_path]
        path_cols = [cell[1] for cell in planned_path]
        ax.plot(path_cols, path_rows, "--", color="tab:purple", linewidth=2)

    if current_frontier is not None:
        ax.plot(
            current_frontier[1],
            current_frontier[0],
            "D",
            color="tab:orange",
            markersize=8,
        )

    ax.plot(env.robot[1], env.robot[0], "o", color="tab:blue", markersize=8)
    ax.plot(env.goal[1], env.goal[0], "*", color="tab:green", markersize=14)

    if dynamic and isinstance(env, DynamicObstacleGridWorld):
        for row, col in env.dynamic_obstacles():
            ax.plot(col, row, "s", color="tab:red", markersize=10)
        for row, col in env.predicted_dynamic_obstacles():
            ax.plot(col, row, "s", color="tab:red", markersize=10, fillstyle="none")

    ax.set_title(title)
    status = f"step={env.time}"
    if agent_state is not None:
        status += f"  state={agent_state}"
    if coverage is not None:
        status += f"  coverage={coverage:.2f}"
    ax.text(0.02, 0.98, status, transform=ax.transAxes, va="top", fontsize=9)
    ax.set_xticks(np.arange(-0.5, env.width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.height, 1), minor=True)
    ax.grid(which="minor", color="0.85", linewidth=0.6)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    fig.tight_layout()
    frame = fig_to_frame(fig)
    plt.close(fig)
    return frame


def render_tabletop_frame(
    env: Tabletop2D,
    agent: Any,
    info: dict[str, Any],
    *,
    title: str = "pick -> fail -> update belief -> retry",
) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(4.6, 4.6), dpi=80)
    ax.set_title(title)
    ax.set_xlim(0.0, env.table_size[0])
    ax.set_ylim(0.0, env.table_size[1])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)

    xmin, ymin, xmax, ymax = env.occluder
    ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color="0.2", alpha=0.18))
    ax.plot(*env.camera_pos, marker="s", color="tab:blue", markersize=8)

    if not env.obj.picked:
        ax.add_patch(Circle(env.obj.position, env.obj.radius, color=env.obj.color, alpha=0.85))
    if env.last_detection is not None and not env.obj.picked:
        ax.plot(*env.last_detection, marker="x", markersize=10, color="tab:orange")

    if agent.belief_mean is not None:
        ax.add_patch(
            Circle(
                agent.belief_mean,
                agent.belief_radius,
                fill=False,
                linestyle="--",
                color="tab:green",
                linewidth=2,
            )
        )

    if "pick_position" in info:
        ax.plot(*info["pick_position"], marker="+", markersize=14, color="black")

    status = f"attempts={env.attempts}"
    if "agent_state" in info:
        status += f"  {info['agent_state']}"
    if "failure" in info:
        status += f"  {info['failure'].kind}"
    if info.get("success"):
        status += "  success"
    ax.text(0.02, 0.97, status, transform=ax.transAxes, va="top", fontsize=9)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    frame = fig_to_frame(fig)
    plt.close(fig)
    return frame


def make_sense_act() -> Path:
    module = load_example("examples/runtime/01_sense_act_loop.py")
    rng = np.random.default_rng(0)
    state: dict[str, Any] = {
        "position": np.array([0.12, 0.28], dtype=float),
        "goal": np.array([0.88, 0.76], dtype=float),
        "obstacle_center": np.array([0.50, 0.52], dtype=float),
        "obstacle_radius": 0.13,
        "trajectory": [np.array([0.12, 0.28], dtype=float)],
    }
    obs = module.observe(state, rng)
    frames: list[np.ndarray] = []

    for step in range(28):
        action = module.policy(obs)
        reward, done, info = module.step(state, action)
        obs = module.observe(state, rng)
        if step % 2 == 0 or done:
            fig, ax = plt.subplots(figsize=(4.6, 4.6), dpi=80)
            trajectory = np.asarray(state["trajectory"])
            ax.plot(trajectory[:, 0], trajectory[:, 1], color="tab:blue", linewidth=2)
            ax.plot(*state["position"], marker="o", color="tab:blue")
            ax.plot(*obs["position"], marker="x", color="tab:orange")
            ax.plot(*state["goal"], marker="*", markersize=14, color="tab:green")
            ax.add_patch(
                Circle(
                    state["obstacle_center"],
                    state["obstacle_radius"],
                    color="tab:red",
                    alpha=0.28,
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
            ax.set_title("sense -> act -> observe")
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.25)
            ax.text(0.02, 0.97, f"step={step + 1}", transform=ax.transAxes, va="top", fontsize=9)
            fig.tight_layout()
            frames.append(fig_to_frame(fig))
            plt.close(fig)
        if done:
            break

    return save_gif("sense_act_loop.gif", frames)


def make_pick_and_retry() -> Path:
    module = load_example("examples/manipulation/01_pick_and_retry.py")
    env = Tabletop2D(seed=3)
    agent = module.PickAndRetryAgent()
    obs = env.reset(seed=3)
    agent.reset()
    frames = [render_tabletop_frame(env, agent, {})]

    for _ in range(10):
        action = agent.act(obs)
        result = env.step(action)
        obs, reward, done, info = result.as_tuple()
        agent.update(obs, reward, info)
        frames.append(render_tabletop_frame(env, agent, info))
        if done:
            break

    return save_gif("pick_and_retry.gif", frames)


def make_reactive_grasping() -> Path:
    module = load_example("examples/manipulation/02_reactive_grasping.py")
    env = module.ReactiveGraspWorld(seed=4, max_steps=60)
    agent = module.ReactiveGraspAgent()
    obs = env.reset(seed=4)
    agent.reset()
    frames: list[np.ndarray] = []

    def append_frame(info: dict[str, Any] | None = None) -> None:
        fig, ax = plt.subplots(figsize=(4.8, 4.8), dpi=80)
        module.draw_reactive_grasp_scene(ax, env, agent, info)
        fig.tight_layout()
        frames.append(fig_to_frame(fig))
        plt.close(fig)

    append_frame({})
    for step in range(60):
        action = agent.act(obs)
        result = env.step(action)
        obs, reward, done, info = result.as_tuple()
        agent.update(obs, reward, info)
        if step % 1 == 0 or done:
            append_frame(info)
        if done:
            break

    return save_gif("reactive_grasping.gif", frames)


def make_closed_loop_ik() -> Path:
    module = load_example("examples/manipulation/03_closed_loop_ik.py")
    env = module.ClosedLoopIKWorld(seed=2, max_steps=80)
    agent = module.ClosedLoopIKAgent()
    obs = env.reset(seed=2)
    agent.reset()
    frames: list[np.ndarray] = []

    def append_frame(info: dict[str, Any] | None = None) -> None:
        fig, ax = plt.subplots(figsize=(4.8, 4.8), dpi=80)
        module.draw_closed_loop_ik_scene(ax, env, agent, info)
        fig.tight_layout()
        frames.append(fig_to_frame(fig))
        plt.close(fig)

    append_frame({})
    for step in range(80):
        action = agent.act(obs, env)
        result = env.step(action)
        obs, reward, done, info = result.as_tuple()
        agent.update(obs, reward, info)
        if step % 1 == 0 or done:
            append_frame(info)
        if done:
            break

    return save_gif("closed_loop_ik.gif", frames)


def make_moving_target_reaching() -> Path:
    module = load_example("examples/manipulation/04_moving_target_reaching.py")
    env = module.MovingTargetReachWorld(seed=5, max_steps=90)
    agent = module.MovingTargetReachAgent()
    obs = env.reset(seed=5)
    agent.reset()
    frames: list[np.ndarray] = []

    def append_frame(info: dict[str, Any] | None = None) -> None:
        fig, ax = plt.subplots(figsize=(4.8, 4.8), dpi=80)
        module.draw_moving_target_reaching_scene(ax, env, agent, info)
        fig.tight_layout()
        frames.append(fig_to_frame(fig))
        plt.close(fig)

    append_frame({})
    for step in range(90):
        action = agent.act(obs, env)
        result = env.step(action)
        obs, reward, done, info = result.as_tuple()
        agent.update(obs, reward, info)
        should_show_occlusion = info.get("failure") is not None
        if step % 2 == 0 or done or should_show_occlusion:
            append_frame(info)
        if done:
            break

    return save_gif("moving_target_reaching.gif", frames)


def make_object_search_and_pick() -> Path:
    module = load_example("examples/manipulation/05_object_search_and_pick.py")
    env = module.ObjectSearchPickWorld(seed=7, max_steps=30)
    agent = module.ObjectSearchPickAgent()
    obs = env.reset(seed=7)
    agent.reset()
    frames: list[np.ndarray] = []

    def append_frame(info: dict[str, Any] | None = None) -> None:
        fig, ax = plt.subplots(figsize=(4.8, 4.8), dpi=80)
        module.draw_object_search_pick_scene(ax, env, agent, info)
        fig.tight_layout()
        frames.append(fig_to_frame(fig))
        plt.close(fig)

    append_frame({})
    for step in range(30):
        action = agent.act(obs)
        result = env.step(action)
        obs, reward, done, info = result.as_tuple()
        agent.update(obs, reward, info)
        if step % 1 == 0 or done:
            append_frame(info)
        if done:
            break

    return save_gif("object_search_and_pick.gif", frames)


def make_push_then_grasp() -> Path:
    module = load_example("examples/manipulation/06_push_then_grasp.py")
    env = module.PushThenGraspWorld(seed=9, max_steps=25)
    agent = module.PushThenGraspAgent()
    obs = env.reset(seed=9)
    agent.reset()
    frames: list[np.ndarray] = []

    def append_frame(info: dict[str, Any] | None = None) -> None:
        fig, ax = plt.subplots(figsize=(4.8, 4.8), dpi=80)
        module.draw_push_then_grasp_scene(ax, env, agent, info)
        fig.tight_layout()
        frames.append(fig_to_frame(fig))
        plt.close(fig)

    append_frame({})
    for step in range(25):
        action = agent.act(obs)
        result = env.step(action)
        obs, reward, done, info = result.as_tuple()
        agent.update(obs, reward, info)
        if step % 1 == 0 or done:
            append_frame(info)
        if done:
            break

    return save_gif("push_then_grasp.gif", frames)


def make_probabilistic_suction_sorting() -> Path:
    module = load_example("examples/manipulation/07_probabilistic_suction_sorting.py")
    env = module.ProbabilisticSuctionSortingWorld(seed=11, max_steps=40)
    agent = module.ProbabilisticSuctionSortingAgent()
    obs = env.reset(seed=11)
    agent.reset()
    frames: list[np.ndarray] = []

    def append_frame(info: dict[str, Any] | None = None) -> None:
        fig, ax = plt.subplots(figsize=(4.8, 4.8), dpi=80)
        module.draw_suction_sorting_scene(ax, env, agent, info)
        fig.tight_layout()
        frames.append(fig_to_frame(fig))
        plt.close(fig)

    append_frame({})
    for step in range(40):
        action = agent.act(obs)
        result = env.step(action)
        obs, reward, done, info = result.as_tuple()
        agent.update(obs, reward, info)
        if step % 1 == 0 or done:
            append_frame(info)
        if done:
            break

    return save_gif("probabilistic_suction_sorting.gif", frames)


def make_reactive_obstacle() -> Path:
    module = load_example("examples/navigation/02_reactive_obstacle_avoidance.py")
    env = GridWorld2D(seed=0)
    agent = module.ReactiveLidarAgent()
    obs = env.reset(seed=0)
    agent.reset()
    frames = [render_grid_frame(env, title="reactive obstacle avoidance", agent_state=agent.state)]

    for step in range(80):
        action = agent.act(obs)
        result = env.step(action)
        obs, reward, done, info = result.as_tuple()
        agent.update(obs, reward, info)
        if step % 2 == 0 or done:
            frames.append(
                render_grid_frame(
                    env,
                    title="reactive obstacle avoidance",
                    agent_state=agent.state,
                )
            )
        if done:
            break

    return save_gif("reactive_obstacle_avoidance.gif", frames)


def make_dynamic_obstacle() -> Path:
    module = load_example("examples/navigation/03_dynamic_obstacle_avoidance.py")
    env = DynamicObstacleGridWorld(seed=0)
    agent = module.OneStepLookaheadAgent()
    obs = env.reset(seed=0)
    agent.reset()
    frames = [
        render_grid_frame(
            env,
            title="dynamic obstacle avoidance",
            agent_state=agent.state,
            dynamic=True,
        )
    ]

    for step in range(90):
        action = agent.act(obs)
        result = env.step(action)
        obs, reward, done, info = result.as_tuple()
        agent.update(obs, reward, info)
        if step % 2 == 0 or done:
            frames.append(
                render_grid_frame(
                    env,
                    title="dynamic obstacle avoidance",
                    agent_state=agent.state,
                    dynamic=True,
                )
            )
        if done:
            break

    return save_gif("dynamic_obstacle_avoidance.gif", frames)


def make_online_replanning() -> Path:
    module = load_example("examples/navigation/04_online_replanning_astar.py")
    env = GridWorld2D(seed=0, lidar_range=3)
    agent = module.AStarReplanningAgent()
    obs = env.reset(seed=0)
    agent.reset()
    frames = [
        render_grid_frame(
            env,
            title="online A* replanning",
            agent_state=agent.state,
            planned_path=agent.current_path,
        )
    ]

    for step in range(100):
        action = agent.act(obs)
        result = env.step(action)
        obs, reward, done, info = result.as_tuple()
        agent.update(obs, reward, info)
        if step % 2 == 0 or done or agent.state == "replan":
            frames.append(
                render_grid_frame(
                    env,
                    title="online A* replanning",
                    agent_state=agent.state,
                    planned_path=agent.current_path,
                )
            )
        if done:
            break

    return save_gif("online_replanning_astar.gif", frames)


def make_frontier_exploration() -> Path:
    module = load_example("examples/navigation/05_frontier_exploration.py")
    env = GridWorld2D(seed=0, lidar_range=4)
    agent = module.FrontierExplorationAgent()
    obs = env.reset(seed=0)
    agent.reset()
    frames = [
        render_grid_frame(
            env,
            title="frontier exploration",
            agent_state=agent.state,
            planned_path=agent.current_path,
            current_frontier=agent.current_frontier,
            coverage=agent.coverage,
        )
    ]

    for step in range(120):
        action = agent.act(obs)
        result = env.step(action)
        obs, reward, env_done, info = result.as_tuple()
        agent.update(obs, reward, info)
        coverage = module.coverage_ratio(obs["known_map"])
        done = coverage >= 0.58 or (env_done and info.get("failure") is not None)
        if step % 2 == 0 or done or agent.state in {"choose_frontier", "observe_from_frontier"}:
            frames.append(
                render_grid_frame(
                    env,
                    title="frontier exploration",
                    agent_state=agent.state,
                    planned_path=agent.current_path,
                    current_frontier=agent.current_frontier,
                    coverage=coverage,
                )
            )
        if done:
            break

    return save_gif("frontier_exploration.gif", frames)


def make_belief_navigation() -> Path:
    module = load_example("examples/navigation/06_belief_based_navigation.py")
    env = module.BeliefGridWorld(seed=0)
    agent = module.BeliefNavigationAgent()
    obs = env.reset(seed=0)
    agent.reset()
    agent.initialize(obs)
    frames: list[np.ndarray] = []

    def append_frame(info: dict[str, Any] | None = None) -> None:
        fig, ax = plt.subplots(figsize=(5.6, 4.2), dpi=80)
        module.draw_belief_scene(ax, env, agent, info)
        fig.tight_layout()
        frames.append(fig_to_frame(fig))
        plt.close(fig)

    append_frame({})
    for step in range(90):
        action = agent.act(obs)
        result = env.step(action)
        obs, reward, done, info = result.as_tuple()
        agent.update(obs, reward, info)
        if step % 2 == 0 or done or agent.state == "localize":
            append_frame(info)
        if done:
            break

    return save_gif("belief_based_navigation.gif", frames)


def make_goal_command_pick() -> Path:
    module = load_example("examples/embodied_ai/01_goal_command_pick.py")
    command = "find the red block and pick it"
    env = Tabletop2D(seed=3)
    agent = module.GoalCommandPickAgent(command)
    obs = env.reset(seed=3)
    agent.reset()
    frames = [
        render_tabletop_frame(
            env,
            agent,
            {"agent_state": "parse_goal"},
            title="goal: find the red block and pick it",
        )
    ]

    for _ in range(40):
        action = agent.act(obs)
        result = env.step(action)
        obs, reward, done, info = result.as_tuple()
        agent.update(obs, reward, info)
        info["agent_state"] = agent.state
        info["retry_count"] = agent.retry_count
        info["memory_count"] = len(agent.memory)
        frames.append(
            render_tabletop_frame(
                env,
                agent,
                info,
                title="goal: find the red block and pick it",
            )
        )
        if done:
            break

    return save_gif("goal_command_pick.gif", frames)


def make_active_slam() -> Path:
    module = load_example("examples/navigation/07_active_slam_toy.py")
    env = module.ActiveSlamToyWorld(seed=0)
    agent = module.ActiveSlamToyAgent(lidar_range=env.lidar_range)
    obs = env.reset(seed=0)
    agent.reset()
    agent.initialize(obs)
    frames: list[np.ndarray] = []

    def append_frame(info: dict[str, Any] | None = None) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.8), dpi=80)
        module.draw_scene(axes, env, agent, info)
        fig.tight_layout()
        frames.append(fig_to_frame(fig))
        plt.close(fig)

    append_frame(
        {
            "agent_state": agent.state,
            "pose_entropy": agent.pose_entropy,
            "map_entropy": agent.map_entropy,
            "information_gain": agent.information_gain,
        }
    )
    for step in range(100):
        action = agent.act(obs)
        result = env.step(action)
        obs, reward, env_done, info = result.as_tuple()
        agent.update(obs, reward, info)
        success = agent.pose_entropy <= agent.pose_goal and agent.map_entropy <= agent.map_goal
        info["pose_entropy"] = agent.pose_entropy
        info["map_entropy"] = agent.map_entropy
        info["information_gain"] = agent.information_gain
        info["agent_state"] = agent.state
        info["success"] = success
        if step % 2 == 0 or success or env_done:
            append_frame(info)
        if success or env_done:
            break

    return save_gif("active_slam_toy.gif", frames)


def render_mpc_frame(env: Any, agent: Any, info: dict[str, Any]) -> np.ndarray:
    cfg = env.config
    fig, ax = plt.subplots(figsize=(4.8, 4.8), dpi=80)
    ax.set_title("interactive MPC")
    ax.set_xlim(cfg.world_min, cfg.world_max)
    ax.set_ylim(cfg.world_min, cfg.world_max)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)

    if agent.last_prediction is not None:
        ax.plot(
            agent.last_prediction[:, 0],
            agent.last_prediction[:, 1],
            color="tab:blue",
            linewidth=2,
        )
    if agent.last_obstacle_prediction is not None:
        ax.plot(
            agent.last_obstacle_prediction[:, 0],
            agent.last_obstacle_prediction[:, 1],
            color="tab:red",
            linestyle="--",
            alpha=0.7,
        )

    ax.add_patch(plt.Circle(env.goal, cfg.goal_radius, color="tab:green", alpha=0.25))
    ax.add_patch(
        plt.Circle(
            env.obstacle,
            cfg.obstacle_radius + cfg.robot_radius + cfg.safety_margin,
            color="tab:red",
            fill=False,
            linestyle=":",
            alpha=0.6,
        )
    )
    ax.add_patch(plt.Circle(env.obstacle, cfg.obstacle_radius, color="tab:red", alpha=0.65))
    ax.add_patch(plt.Circle(env.robot, cfg.robot_radius, color="tab:blue"))
    ax.scatter([env.goal[0]], [env.goal[1]], marker="*", s=160, color="tab:green")
    ax.text(
        0.02,
        0.98,
        (
            f"step={env.step_count}\n"
            f"state={info.get('agent_state', 'none')}\n"
            f"risk={info.get('predicted_collision_risk', 0.0):.2f}\n"
            f"replans={info.get('replan_count', 0)}"
        ),
        transform=ax.transAxes,
        va="top",
        fontsize=9,
    )
    fig.tight_layout()
    frame = fig_to_frame(fig)
    plt.close(fig)
    return frame


def make_interactive_mpc() -> Path:
    module = load_example("examples/navigation/08_interactive_mpc.py")
    env = module.MovingObstacleWorld(seed=0, max_steps=120)
    agent = module.ToyMPCAgent(env.config)
    obs = env.reset(seed=0)
    agent.reset()
    frames: list[np.ndarray] = []

    for step in range(120):
        action = agent.act(obs, env)
        obs, reward, done, info = env.step(action)
        info.update(agent.info())
        if done and info.get("success"):
            info["agent_state"] = "arrived"
        if step % 2 == 0 or done or info.get("predicted_collision_risk", 0.0) > 0.05:
            frames.append(render_mpc_frame(env, agent, info))
        if done:
            break

    return save_gif("interactive_mpc.gif", frames)


def make_blocked_path_recovery() -> Path:
    module = load_example("examples/navigation/09_blocked_path_recovery.py")
    env = module.BlockedPathWorld(max_steps=80)
    agent = module.BlockedPathRecoveryAgent()
    obs = env.reset(seed=0)
    agent.reset()
    frames: list[np.ndarray] = []

    def append_frame(info: dict[str, Any] | None = None) -> None:
        fig, ax = plt.subplots(figsize=(5.6, 4.2), dpi=80)
        module.draw_blocked_path_scene(ax, env, agent, info)
        fig.tight_layout()
        frames.append(fig_to_frame(fig))
        plt.close(fig)

    append_frame({})
    for step in range(80):
        action = agent.act(obs)
        result = env.step(action)
        obs, reward, done, info = result.as_tuple()
        agent.update(obs, reward, info)
        info["agent_state"] = agent.state
        info["replan_count"] = agent.replan_count
        info["recovery_count"] = agent.recovery_count
        if step % 2 == 0 or done or info.get("failure") is not None or agent.state == "recover":
            append_frame(info)
        if done:
            break

    return save_gif("blocked_path_recovery.gif", frames)


def make_localization_uncertainty_recovery() -> Path:
    module = load_example("examples/navigation/10_localization_uncertainty_recovery.py")
    env = module.LocalizationRecoveryWorld(seed=0, max_steps=60)
    agent = module.LocalizationRecoveryAgent()
    obs = env.reset(seed=0)
    agent.reset()
    agent.initialize(obs)
    frames: list[np.ndarray] = []

    def append_frame(info: dict[str, Any] | None = None) -> None:
        fig, ax = plt.subplots(figsize=(5.6, 4.2), dpi=80)
        module.draw_localization_scene(ax, env, agent, info)
        fig.tight_layout()
        frames.append(fig_to_frame(fig))
        plt.close(fig)

    append_frame({})
    for step in range(60):
        action = agent.act(obs)
        result = env.step(action)
        obs, reward, done, info = result.as_tuple()
        agent.update(obs, reward, info)
        info["agent_state"] = agent.state
        info["entropy"] = agent.entropy
        info["localization_recovery_count"] = agent.localization_recovery_count
        if step % 1 == 0 or done:
            append_frame(info)
        if done:
            break

    return save_gif("localization_uncertainty_recovery.gif", frames)


def make_goal_conditioned_minikitchen() -> Path:
    module = load_example("examples/embodied_ai/18_goal_conditioned_minikitchen.py")
    command = "bring mug to table"
    env = module.MiniKitchenWorld(command=command, max_steps=35)
    agent = module.MiniKitchenAgent(command)
    obs = env.reset(seed=0)
    agent.reset()
    frames: list[np.ndarray] = []

    def append_frame(info: dict[str, Any] | None = None) -> None:
        fig, ax = plt.subplots(figsize=(5.8, 4.6), dpi=80)
        module.draw_minikitchen_scene(ax, env, agent, info)
        fig.tight_layout()
        frames.append(fig_to_frame(fig))
        plt.close(fig)

    append_frame({})
    for step in range(35):
        action = agent.act(obs)
        result = env.step(action)
        obs, reward, done, info = result.as_tuple()
        agent.update(obs, reward, info)
        if step % 1 == 0 or done:
            append_frame(info)
        if done:
            break

    return save_gif("goal_conditioned_minikitchen.gif", frames)


def make_tiny_vla_loop() -> Path:
    module = load_example("examples/embodied_ai/19_tiny_vla_loop.py")
    command = "place red block in blue bin"
    env = module.TinyVLAWorld(command=command, max_steps=25)
    agent = module.TinyVLAAgent(command)
    obs = env.reset(seed=0)
    agent.reset()
    frames: list[np.ndarray] = []

    def append_frame(info: dict[str, Any] | None = None) -> None:
        fig, ax = plt.subplots(figsize=(4.8, 4.8), dpi=80)
        module.draw_tiny_vla_scene(ax, env, agent, info)
        fig.tight_layout()
        frames.append(fig_to_frame(fig))
        plt.close(fig)

    append_frame({})
    for step in range(25):
        action = agent.act(obs)
        result = env.step(action)
        obs, reward, done, info = result.as_tuple()
        agent.update(obs, reward, info)
        if step % 1 == 0 or done:
            append_frame(info)
        if done:
            break

    return save_gif("tiny_vla_loop.gif", frames)


def make_tiny_world_model_planning() -> Path:
    module = load_example("examples/world_models/20_tiny_world_model_planning.py")
    env = module.TinyWorldModelWorld(seed=0, max_steps=80)
    agent = module.TinyWorldModelAgent()
    obs = env.reset(seed=0)
    agent.reset()
    frames: list[np.ndarray] = []

    def append_frame(info: dict[str, Any] | None = None) -> None:
        fig, ax = plt.subplots(figsize=(4.8, 4.8), dpi=80)
        module.draw_world_model_scene(ax, env, agent, info)
        fig.tight_layout()
        frames.append(fig_to_frame(fig))
        plt.close(fig)

    append_frame({})
    for step in range(80):
        action = agent.act(obs, env)
        result = env.step(action)
        obs, reward, done, info = result.as_tuple()
        agent.update(obs, reward, info)
        if step % 1 == 0 or done:
            append_frame(info)
        if done:
            break

    return save_gif("tiny_world_model_planning.gif", frames)


def make_door_search_pomdp() -> Path:
    module = load_example("examples/embodied_ai/10_door_search_pomdp.py")
    env = module.DoorSearchWorld(max_steps=40)
    agent = module.DoorSearchAgent()
    obs = env.reset(seed=0)
    agent.reset()
    frames: list[np.ndarray] = []

    def append_frame(info: dict[str, Any] | None = None) -> None:
        fig, ax = plt.subplots(figsize=(5.6, 4.2), dpi=80)
        module.draw_door_search_scene(ax, env, agent, info)
        fig.tight_layout()
        frames.append(fig_to_frame(fig))
        plt.close(fig)

    append_frame({})
    for step in range(40):
        action = agent.act(obs)
        result = env.step(action)
        obs, reward, done, info = result.as_tuple()
        agent.update(obs, reward, info)
        info["agent_state"] = agent.state
        if step % 1 == 0 or done:
            append_frame(info)
        if done:
            break

    return save_gif("door_search_pomdp.gif", frames)


MAKERS: dict[str, Callable[[], Path]] = {
    "sense": make_sense_act,
    "pick": make_pick_and_retry,
    "grasp": make_reactive_grasping,
    "ik": make_closed_loop_ik,
    "moving": make_moving_target_reaching,
    "search_pick": make_object_search_and_pick,
    "push": make_push_then_grasp,
    "suction": make_probabilistic_suction_sorting,
    "reactive": make_reactive_obstacle,
    "dynamic": make_dynamic_obstacle,
    "replanning": make_online_replanning,
    "frontier": make_frontier_exploration,
    "belief": make_belief_navigation,
    "goal": make_goal_command_pick,
    "slam": make_active_slam,
    "mpc": make_interactive_mpc,
    "blocked": make_blocked_path_recovery,
    "localization": make_localization_uncertainty_recovery,
    "kitchen": make_goal_conditioned_minikitchen,
    "vla": make_tiny_vla_loop,
    "world_model": make_tiny_world_model_planning,
    "door": make_door_search_pomdp,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "names",
        nargs="*",
        choices=sorted(MAKERS),
        help="GIF names to build. Defaults to all.",
    )
    args = parser.parse_args()

    names = args.names or list(MAKERS)
    for name in names:
        path = MAKERS[name]()
        print(path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
