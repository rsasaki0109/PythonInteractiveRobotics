"""Gymnasium-style adapters for toy worlds.

The adapter module is importable without Gymnasium. Installing the `rl` extra
adds action and observation spaces and makes the wrapper a `gymnasium.Env`
subclass.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pir.core.types import Failure
from pir.worlds.grid_world import (
    OCCUPIED,
    UNKNOWN,
    DynamicObstacleGridWorld,
    GridWorld2D,
)
from pir.worlds.tabletop_2d import Tabletop2D

try:  # pragma: no cover - exercised only when the optional extra is installed.
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - core installs intentionally hit this.
    gym = None
    spaces = None


GRID_ACTIONS = ("stay", "north", "east", "south", "west")
TABLETOP_ACTIONS = ("look", "pick")
_GymEnv = gym.Env if gym is not None else object


class GridWorldGymnasiumAdapter(_GymEnv):
    """Expose `GridWorld2D` through the Gymnasium `reset` / `step` shape."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        env: GridWorld2D | None = None,
        *,
        render_mode: str | None = None,
        **world_kwargs: Any,
    ) -> None:
        self.env = env if env is not None else GridWorld2D(**world_kwargs)
        self.render_mode = render_mode
        self.action_space = self._make_action_space()
        self.observation_space = self._make_observation_space()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        del options
        raw_obs = self.env.reset(seed=seed)
        return self._encode_observation(raw_obs), self._make_info(raw_obs)

    def step(
        self,
        action: int | np.integer[Any] | str | dict[str, Any],
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        decoded_action = decode_grid_action(action)
        result = self.env.step(decoded_action)
        raw_obs, reward, done, info = result.as_tuple()
        terminated, truncated = split_done(done, info)
        gym_info = self._make_info(raw_obs, info)
        gym_info["gymnasium_action"] = action
        gym_info["decoded_action"] = decoded_action
        return self._encode_observation(raw_obs), reward, terminated, truncated, gym_info

    def render(self) -> None:
        self.env.render()

    def close(self) -> None:
        figure = getattr(self.env, "_fig", None)
        if figure is None:
            return
        import matplotlib.pyplot as plt

        plt.close(figure)
        self.env._fig = None
        if hasattr(self.env, "_ax"):
            self.env._ax = None

    @property
    def unwrapped(self) -> GridWorld2D:
        return self.env

    def _make_action_space(self) -> Any | None:
        if spaces is None:
            return None
        return spaces.Discrete(len(GRID_ACTIONS))

    def _make_observation_space(self) -> Any | None:
        if spaces is None:
            return None
        high_cell = max(self.env.height - 1, self.env.width - 1)
        return spaces.Dict(
            {
                "time": spaces.Box(
                    low=0,
                    high=self.env.max_steps,
                    shape=(1,),
                    dtype=np.int64,
                ),
                "robot": spaces.Box(low=0, high=high_cell, shape=(2,), dtype=np.int64),
                "goal": spaces.Box(low=0, high=high_cell, shape=(2,), dtype=np.int64),
                "known_map": spaces.Box(
                    low=UNKNOWN,
                    high=OCCUPIED,
                    shape=(self.env.height, self.env.width),
                    dtype=np.int8,
                ),
            }
        )

    def _encode_observation(self, raw_obs: dict[str, Any]) -> dict[str, np.ndarray]:
        return {
            "time": np.asarray([raw_obs["time"]], dtype=np.int64),
            "robot": np.asarray(raw_obs["robot"], dtype=np.int64),
            "goal": np.asarray(raw_obs["goal"], dtype=np.int64),
            "known_map": np.asarray(raw_obs["known_map"], dtype=np.int8),
        }

    def _make_info(
        self,
        raw_obs: dict[str, Any],
        info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        gym_info = {} if info is None else dict(info)
        gym_info["raw_obs"] = raw_obs
        gym_info["lidar"] = raw_obs.get("lidar")
        gym_info["trajectory"] = raw_obs.get("trajectory")
        return gym_info


class DynamicObstacleGridWorldGymnasiumAdapter(GridWorldGymnasiumAdapter):
    """Expose `DynamicObstacleGridWorld` through the Gymnasium API shape."""

    def __init__(
        self,
        env: DynamicObstacleGridWorld | None = None,
        *,
        render_mode: str | None = None,
        **world_kwargs: Any,
    ) -> None:
        super().__init__(
            env if env is not None else DynamicObstacleGridWorld(**world_kwargs),
            render_mode=render_mode,
        )

    @property
    def unwrapped(self) -> DynamicObstacleGridWorld:
        return self.env

    def _make_observation_space(self) -> Any | None:
        if spaces is None:
            return None
        high_cell = max(self.env.height - 1, self.env.width - 1)
        base = super()._make_observation_space()
        assert base is not None
        base.spaces["dynamic_obstacles"] = spaces.Box(
            low=0,
            high=high_cell,
            shape=(1, 2),
            dtype=np.int64,
        )
        base.spaces["predicted_dynamic_obstacles"] = spaces.Box(
            low=0,
            high=high_cell,
            shape=(1, 2),
            dtype=np.int64,
        )
        return base

    def _encode_observation(self, raw_obs: dict[str, Any]) -> dict[str, np.ndarray]:
        obs = super()._encode_observation(raw_obs)
        obs["dynamic_obstacles"] = _grid_cells_array(raw_obs["dynamic_obstacles"])
        obs["predicted_dynamic_obstacles"] = _grid_cells_array(
            raw_obs["predicted_dynamic_obstacles"]
        )
        return obs

    def _make_info(
        self,
        raw_obs: dict[str, Any],
        info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        gym_info = super()._make_info(raw_obs, info)
        gym_info["dynamic_obstacles"] = raw_obs.get("dynamic_obstacles", [])
        gym_info["predicted_dynamic_obstacles"] = raw_obs.get(
            "predicted_dynamic_obstacles",
            [],
        )
        return gym_info


class Tabletop2DGymnasiumAdapter(_GymEnv):
    """Expose `Tabletop2D` through the Gymnasium `reset` / `step` shape."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        env: Tabletop2D | None = None,
        *,
        render_mode: str | None = None,
        **world_kwargs: Any,
    ) -> None:
        self.env = env if env is not None else Tabletop2D(**world_kwargs)
        self.render_mode = render_mode
        self.action_space = self._make_action_space()
        self.observation_space = self._make_observation_space()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        del options
        raw_obs = self.env.reset(seed=seed)
        return self._encode_observation(raw_obs), self._make_info(raw_obs)

    def step(
        self,
        action: int | np.integer[Any] | str | dict[str, Any],
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        decoded_action = decode_tabletop_action(
            action,
            default_target=self.env.camera_pos,
            default_position=self.env.last_detection,
        )
        result = self.env.step(decoded_action)
        raw_obs, reward, done, info = result.as_tuple()
        terminated, truncated = split_done(done, info)
        gym_info = self._make_info(raw_obs, info)
        gym_info["gymnasium_action"] = action
        gym_info["decoded_action"] = decoded_action
        return self._encode_observation(raw_obs), reward, terminated, truncated, gym_info

    def render(self) -> None:
        self.env.render()

    def close(self) -> None:
        figure = getattr(self.env, "_fig", None)
        if figure is None:
            return
        import matplotlib.pyplot as plt

        plt.close(figure)
        self.env._fig = None
        if hasattr(self.env, "_ax"):
            self.env._ax = None

    @property
    def unwrapped(self) -> Tabletop2D:
        return self.env

    def _make_action_space(self) -> Any | None:
        if spaces is None:
            return None
        high = self.env.table_size.astype(np.float32)
        return spaces.Dict(
            {
                "action_type": spaces.Discrete(len(TABLETOP_ACTIONS)),
                "target": spaces.Box(low=0.0, high=high, shape=(2,), dtype=np.float32),
                "position": spaces.Box(low=0.0, high=high, shape=(2,), dtype=np.float32),
            }
        )

    def _make_observation_space(self) -> Any | None:
        if spaces is None:
            return None
        high = self.env.table_size.astype(np.float32)
        return spaces.Dict(
            {
                "time": spaces.Box(
                    low=0,
                    high=self.env.max_attempts * 4,
                    shape=(1,),
                    dtype=np.int64,
                ),
                "camera": spaces.Box(low=0.0, high=high, shape=(2,), dtype=np.float32),
                "detection_visible": spaces.MultiBinary(1),
                "detection_position": spaces.Box(
                    low=0.0,
                    high=high,
                    shape=(2,),
                    dtype=np.float32,
                ),
                "detection_confidence": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(1,),
                    dtype=np.float32,
                ),
                "holding": spaces.MultiBinary(1),
                "attempts": spaces.Box(
                    low=0,
                    high=self.env.max_attempts,
                    shape=(1,),
                    dtype=np.int64,
                ),
            }
        )

    def _encode_observation(self, raw_obs: dict[str, Any]) -> dict[str, np.ndarray]:
        detections = raw_obs.get("detections", [])
        detection = detections[0] if detections else None
        position = (
            np.asarray(detection["position"], dtype=np.float32)
            if detection is not None
            else np.zeros(2, dtype=np.float32)
        )
        confidence = float(detection.get("confidence", 0.0)) if detection is not None else 0.0
        holding = raw_obs.get("gripper", {}).get("holding") is not None
        return {
            "time": np.asarray([raw_obs["time"]], dtype=np.int64),
            "camera": np.asarray(raw_obs["camera"], dtype=np.float32),
            "detection_visible": np.asarray([detection is not None], dtype=np.int8),
            "detection_position": position,
            "detection_confidence": np.asarray([confidence], dtype=np.float32),
            "holding": np.asarray([holding], dtype=np.int8),
            "attempts": np.asarray([raw_obs["attempts"]], dtype=np.int64),
        }

    def _make_info(
        self,
        raw_obs: dict[str, Any],
        info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        gym_info = {} if info is None else dict(info)
        gym_info["raw_obs"] = raw_obs
        gym_info["detections"] = raw_obs.get("detections", [])
        gym_info["gripper"] = raw_obs.get("gripper", {})
        return gym_info


def decode_grid_action(action: int | np.integer[Any] | str | dict[str, Any]) -> str | dict[str, Any]:
    """Decode discrete Gymnasium actions while preserving dict actions."""

    if isinstance(action, np.integer):
        action = int(action)
    if isinstance(action, int):
        if action < 0 or action >= len(GRID_ACTIONS):
            raise ValueError(f"grid action index must be in [0, {len(GRID_ACTIONS) - 1}]")
        return GRID_ACTIONS[action]
    if isinstance(action, str):
        if action not in GRID_ACTIONS:
            raise ValueError(f"unknown grid action: {action}")
        return action
    if isinstance(action, dict):
        return action
    raise TypeError("grid action must be an int, direction string, or action dict")


def decode_tabletop_action(
    action: int | np.integer[Any] | str | dict[str, Any],
    *,
    default_target: np.ndarray | None = None,
    default_position: np.ndarray | None = None,
) -> dict[str, Any]:
    """Decode Gymnasium-style tabletop actions into `Tabletop2D` actions."""

    if isinstance(action, dict):
        if "type" in action:
            return dict(action)
        if "action_type" not in action:
            return dict(action)
        action_index = int(np.asarray(action["action_type"]).item())
        action_type = _decode_tabletop_action_type(action_index)
        if action_type == "look":
            target = action.get("target", default_target)
            return {"type": "look", "target": _position_or_zero(target)}
        position = action.get("position", default_position)
        return {"type": "pick", "position": _position_or_zero(position)}

    if isinstance(action, np.integer):
        action = int(action)
    if isinstance(action, int):
        action_type = _decode_tabletop_action_type(action)
    elif isinstance(action, str):
        if action not in TABLETOP_ACTIONS:
            raise ValueError(f"unknown tabletop action: {action}")
        action_type = action
    else:
        raise TypeError("tabletop action must be an int, action string, or action dict")

    if action_type == "look":
        return {"type": "look", "target": _position_or_zero(default_target)}
    return {"type": "pick", "position": _position_or_zero(default_position)}


def _decode_tabletop_action_type(action_index: int) -> str:
    if action_index < 0 or action_index >= len(TABLETOP_ACTIONS):
        raise ValueError(
            f"tabletop action index must be in [0, {len(TABLETOP_ACTIONS) - 1}]"
        )
    return TABLETOP_ACTIONS[action_index]


def _position_or_zero(position: Any) -> np.ndarray:
    if position is None:
        return np.zeros(2, dtype=float)
    return np.asarray(position, dtype=float)


def _grid_cells_array(cells: list[tuple[int, int]]) -> np.ndarray:
    return np.asarray(cells, dtype=np.int64).reshape((-1, 2))


def split_done(done: bool, info: dict[str, Any]) -> tuple[bool, bool]:
    """Split a legacy `done` flag into Gymnasium terminated/truncated flags."""

    if not done:
        return False, False
    failure = info.get("failure")
    if isinstance(failure, Failure) and failure.kind == "timeout":
        return False, True
    return True, False
