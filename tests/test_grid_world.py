from pir.core.types import Failure
from pir.worlds.grid_world import DynamicObstacleGridWorld, GridWorld2D, UNKNOWN


def test_grid_world_observation_reveals_lidar_cells() -> None:
    env = GridWorld2D(seed=0)
    obs = env.reset(seed=0)

    assert obs["robot"] == env.start
    assert obs["goal"] == env.goal
    assert obs["lidar"]["east"]["free_cells"] > 0
    assert (obs["known_map"] == UNKNOWN).any()


def test_grid_world_collision_is_reported_as_failure() -> None:
    env = GridWorld2D(seed=0)
    env.reset(seed=0)
    env.robot = (1, 1)

    result = env.step("north")

    assert result.done is False
    assert isinstance(result.info["failure"], Failure)
    assert result.info["failure"].kind == "collision"
    assert result.info["failure"].recoverable is True


def test_dynamic_grid_world_reports_moving_obstacle_in_observation() -> None:
    env = DynamicObstacleGridWorld(seed=0)
    obs = env.reset(seed=0)

    assert obs["dynamic_obstacles"] == [(7, 4)]
    assert obs["predicted_dynamic_obstacles"] == [(7, 5)]

    result = env.step("east")

    assert result.obs["dynamic_obstacles"] == [(7, 5)]
    assert result.info["dynamic_obstacles_after"] == [(7, 5)]
