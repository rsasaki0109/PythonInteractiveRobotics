from __future__ import annotations

import numpy as np
import pytest

from pir.adapters.gymnasium_adapter import (
    GRID_ACTIONS,
    TABLETOP_ACTIONS,
    DynamicObstacleGridWorldGymnasiumAdapter,
    GridWorldGymnasiumAdapter,
    Tabletop2DGymnasiumAdapter,
    decode_grid_action,
    decode_tabletop_action,
    split_done,
)
from pir.core.types import Failure
from pir.worlds.grid_world import DynamicObstacleGridWorld, GridWorld2D
from pir.worlds.tabletop_2d import Tabletop2D


def test_grid_world_adapter_reset_returns_gymnasium_shape() -> None:
    env = GridWorldGymnasiumAdapter(seed=0)

    obs, info = env.reset(seed=0)

    assert set(obs) == {"time", "robot", "goal", "known_map"}
    assert obs["time"].shape == (1,)
    assert obs["robot"].shape == (2,)
    assert obs["goal"].shape == (2,)
    assert obs["known_map"].shape == (env.unwrapped.height, env.unwrapped.width)
    assert obs["known_map"].dtype == np.int8
    assert info["raw_obs"]["robot"] == env.unwrapped.start
    if env.observation_space is not None:
        assert env.observation_space.contains(obs)


def test_grid_world_adapter_step_decodes_discrete_action() -> None:
    env = GridWorldGymnasiumAdapter(seed=0)
    env.reset(seed=0)

    obs, reward, terminated, truncated, info = env.step(GRID_ACTIONS.index("north"))

    assert obs["robot"].tolist() == [7, 2]
    assert reward == pytest.approx(-0.01)
    assert terminated is False
    assert truncated is False
    assert info["decoded_action"] == "north"
    assert info["gymnasium_action"] == GRID_ACTIONS.index("north")
    assert info["raw_obs"]["robot"] == (7, 2)
    if env.action_space is not None:
        assert env.action_space.n == len(GRID_ACTIONS)


def test_grid_world_adapter_splits_success_as_terminated() -> None:
    world = GridWorld2D(seed=0, start=(2, 11), goal=(2, 12), max_steps=5)
    env = GridWorldGymnasiumAdapter(world)
    env.reset(seed=0)

    obs, reward, terminated, truncated, info = env.step("east")

    assert obs["robot"].tolist() == [2, 12]
    assert reward == pytest.approx(1.0)
    assert terminated is True
    assert truncated is False
    assert info["success"] is True


def test_grid_world_adapter_splits_timeout_as_truncated() -> None:
    world = GridWorld2D(seed=0, max_steps=1)
    env = GridWorldGymnasiumAdapter(world)
    env.reset(seed=0)

    _, _, terminated, truncated, info = env.step("east")

    assert terminated is False
    assert truncated is True
    assert isinstance(info["failure"], Failure)
    assert info["failure"].kind == "timeout"
    assert info["failure"].recoverable is False


def test_dynamic_obstacle_adapter_reset_returns_gymnasium_shape() -> None:
    env = DynamicObstacleGridWorldGymnasiumAdapter(seed=0)

    obs, info = env.reset(seed=0)

    assert set(obs) == {
        "time",
        "robot",
        "goal",
        "known_map",
        "dynamic_obstacles",
        "predicted_dynamic_obstacles",
    }
    assert obs["dynamic_obstacles"].shape == (1, 2)
    assert obs["predicted_dynamic_obstacles"].shape == (1, 2)
    assert obs["dynamic_obstacles"].tolist() == [[7, 4]]
    assert info["raw_obs"]["dynamic_obstacles"] == [(7, 4)]
    if env.observation_space is not None:
        assert env.observation_space.contains(obs)


def test_dynamic_obstacle_adapter_step_decodes_discrete_action() -> None:
    env = DynamicObstacleGridWorldGymnasiumAdapter(seed=0)
    env.reset(seed=0)

    obs, reward, terminated, truncated, info = env.step(GRID_ACTIONS.index("east"))

    assert obs["robot"].tolist() == [7, 2]
    assert obs["dynamic_obstacles"].tolist() == [[7, 5]]
    assert reward == pytest.approx(-0.01)
    assert terminated is False
    assert truncated is False
    assert info["decoded_action"] == "east"
    assert info["raw_obs"]["predicted_dynamic_obstacles"] == [(7, 6)]


def test_dynamic_obstacle_adapter_splits_success_as_terminated() -> None:
    world = DynamicObstacleGridWorld(seed=0, start=(1, 11), goal=(1, 12), max_steps=5)
    env = DynamicObstacleGridWorldGymnasiumAdapter(world)
    env.reset(seed=0)

    obs, reward, terminated, truncated, info = env.step("east")

    assert obs["robot"].tolist() == [1, 12]
    assert reward == pytest.approx(1.0)
    assert terminated is True
    assert truncated is False
    assert info["success"] is True


def test_dynamic_obstacle_adapter_splits_timeout_as_truncated() -> None:
    world = DynamicObstacleGridWorld(seed=0, max_steps=1)
    env = DynamicObstacleGridWorldGymnasiumAdapter(world)
    env.reset(seed=0)

    _, _, terminated, truncated, info = env.step("east")

    assert terminated is False
    assert truncated is True
    assert isinstance(info["failure"], Failure)
    assert info["failure"].kind == "timeout"
    assert info["failure"].recoverable is False


def test_dynamic_obstacle_adapter_keeps_recoverable_failure_nonterminal() -> None:
    world = DynamicObstacleGridWorld(seed=0, start=(7, 3), goal=(1, 12), max_steps=5)
    env = DynamicObstacleGridWorldGymnasiumAdapter(world)
    env.reset(seed=0)

    obs, reward, terminated, truncated, info = env.step("east")

    assert obs["robot"].tolist() == [7, 3]
    assert reward == pytest.approx(-0.16)
    assert terminated is False
    assert truncated is False
    assert isinstance(info["failure"], Failure)
    assert info["failure"].kind == "blocked_by_moving_obstacle"
    assert info["failure"].recoverable is True


def test_decode_grid_action_rejects_unknown_actions() -> None:
    with pytest.raises(ValueError):
        decode_grid_action(len(GRID_ACTIONS))

    with pytest.raises(ValueError):
        decode_grid_action("up")

    with pytest.raises(TypeError):
        decode_grid_action(1.2)  # type: ignore[arg-type]


def test_split_done_keeps_recoverable_failure_nonterminal() -> None:
    info = {"failure": Failure("collision", "recoverable collision", True)}

    assert split_done(False, info) == (False, False)
    assert split_done(True, info) == (True, False)


def test_tabletop_adapter_reset_returns_gymnasium_shape() -> None:
    env = Tabletop2DGymnasiumAdapter(
        seed=0,
        detector_noise=0.0,
        base_false_negative_rate=0.0,
    )

    obs, info = env.reset(seed=0)

    assert set(obs) == {
        "time",
        "camera",
        "detection_visible",
        "detection_position",
        "detection_confidence",
        "holding",
        "attempts",
    }
    assert obs["time"].shape == (1,)
    assert obs["camera"].shape == (2,)
    assert obs["detection_visible"].tolist() == [1]
    assert obs["detection_position"].shape == (2,)
    assert obs["detection_confidence"].shape == (1,)
    assert obs["holding"].tolist() == [0]
    assert info["raw_obs"]["attempts"] == 0
    if env.observation_space is not None:
        assert env.observation_space.contains(obs)


def test_tabletop_adapter_decodes_dict_action_and_updates_camera() -> None:
    env = Tabletop2DGymnasiumAdapter(seed=0)
    env.reset(seed=0)

    target = np.asarray([0.80, 0.20], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(
        {
            "action_type": TABLETOP_ACTIONS.index("look"),
            "target": target,
            "position": np.zeros(2, dtype=np.float32),
        }
    )

    assert obs["camera"].tolist() == pytest.approx(target.tolist())
    assert reward == pytest.approx(-0.01)
    assert terminated is False
    assert truncated is False
    assert info["decoded_action"]["type"] == "look"
    assert info["raw_obs"]["camera"].tolist() == pytest.approx(target.tolist())
    if env.action_space is not None:
        assert env.action_space.contains(
            {
                "action_type": TABLETOP_ACTIONS.index("look"),
                "target": target,
                "position": np.zeros(2, dtype=np.float32),
            }
        )


def test_tabletop_adapter_splits_success_as_terminated() -> None:
    world = Tabletop2D(
        seed=1,
        detector_noise=0.0,
        base_false_negative_rate=0.0,
        base_grasp_success=1.0,
    )
    env = Tabletop2DGymnasiumAdapter(world)
    env.reset(seed=1)

    obs, reward, terminated, truncated, info = env.step(
        {
            "action_type": TABLETOP_ACTIONS.index("pick"),
            "target": np.zeros(2, dtype=np.float32),
            "position": world.obj.position.astype(np.float32),
        }
    )

    assert obs["holding"].tolist() == [1]
    assert reward == pytest.approx(1.0)
    assert terminated is True
    assert truncated is False
    assert info["success"] is True


def test_tabletop_adapter_keeps_terminal_grasp_miss_as_terminated() -> None:
    world = Tabletop2D(seed=0, max_attempts=1)
    env = Tabletop2DGymnasiumAdapter(world)
    env.reset(seed=0)

    _, _, terminated, truncated, info = env.step(
        {
            "action_type": TABLETOP_ACTIONS.index("pick"),
            "target": np.zeros(2, dtype=np.float32),
            "position": np.asarray([0.0, 0.0], dtype=np.float32),
        }
    )

    assert terminated is True
    assert truncated is False
    assert isinstance(info["failure"], Failure)
    assert info["failure"].kind == "grasp_miss"
    assert info["failure"].recoverable is False


def test_decode_tabletop_action_rejects_unknown_actions() -> None:
    with pytest.raises(ValueError):
        decode_tabletop_action(len(TABLETOP_ACTIONS))

    with pytest.raises(ValueError):
        decode_tabletop_action("grasp")

    with pytest.raises(TypeError):
        decode_tabletop_action(1.2)  # type: ignore[arg-type]
