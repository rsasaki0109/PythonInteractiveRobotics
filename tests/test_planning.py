from __future__ import annotations

import numpy as np

from pir.planning import (
    CARDINAL_DIRECTIONS,
    astar,
    bfs_reachable_count,
    manhattan,
)


def test_manhattan_basic() -> None:
    assert manhattan((0, 0), (3, 4)) == 7
    assert manhattan((2, 2), (2, 2)) == 0
    assert manhattan((1, 5), (4, 0)) == 8


def test_directions_are_cardinal_only() -> None:
    assert set(CARDINAL_DIRECTIONS) == {(-1, 0), (1, 0), (0, -1), (0, 1)}


def test_astar_open_grid_finds_shortest_path() -> None:
    walkable = np.ones((6, 6), dtype=bool)
    path = astar(walkable, (0, 0), (5, 5))
    assert path[0] == (0, 0)
    assert path[-1] == (5, 5)
    assert len(path) == 11  # 10 Manhattan steps + start cell


def test_astar_start_equals_goal_returns_singleton() -> None:
    walkable = np.ones((4, 4), dtype=bool)
    assert astar(walkable, (1, 1), (1, 1)) == [(1, 1)]


def test_astar_no_path_through_wall_returns_empty() -> None:
    walkable = np.ones((3, 3), dtype=bool)
    walkable[1, :] = False
    assert astar(walkable, (0, 0), (2, 2)) == []


def test_astar_start_or_goal_blocked_returns_empty() -> None:
    walkable = np.ones((3, 3), dtype=bool)
    walkable[0, 0] = False
    assert astar(walkable, (0, 0), (2, 2)) == []
    walkable = np.ones((3, 3), dtype=bool)
    walkable[2, 2] = False
    assert astar(walkable, (0, 0), (2, 2)) == []


def test_astar_extra_blocked_set_routes_around() -> None:
    walkable = np.ones((4, 4), dtype=bool)
    path_default = astar(walkable, (0, 0), (3, 3))
    path_routed = astar(
        walkable, (0, 0), (3, 3), blocked={(2, 2), (1, 2), (2, 1)}
    )
    assert path_default
    assert path_routed
    # the routed path must steer away from the blocked diagonal
    assert (2, 2) not in path_routed
    assert (1, 2) not in path_routed


def test_astar_edge_cost_steers_around_expensive_row() -> None:
    walkable = np.ones((4, 4), dtype=bool)
    edge_cost = np.ones((4, 4))
    edge_cost[1, :] = 5.0
    path = astar(walkable, (0, 0), (3, 3), edge_cost=edge_cost)
    # the path must avoid row 1 except where unavoidable
    row1_visits = sum(1 for r, _ in path if r == 1)
    assert row1_visits <= 1


def test_astar_endpoint_in_blocked_returns_empty() -> None:
    walkable = np.ones((3, 3), dtype=bool)
    assert astar(walkable, (0, 0), (2, 2), blocked={(2, 2)}) == []


def test_bfs_reachable_count_open_grid() -> None:
    walkable = np.ones((5, 5), dtype=bool)
    # k=1: 4 cardinal neighbors + start
    assert bfs_reachable_count(walkable, (2, 2), 1) == 5
    # k=2: 13 cells in a Manhattan ball of radius 2
    assert bfs_reachable_count(walkable, (2, 2), 2) == 13


def test_bfs_reachable_count_corner() -> None:
    walkable = np.ones((5, 5), dtype=bool)
    # at a corner only 3 cells reachable in k=1 (start + 2 valid neighbors)
    assert bfs_reachable_count(walkable, (0, 0), 1) == 3


def test_bfs_reachable_count_respects_walls() -> None:
    walkable = np.ones((5, 5), dtype=bool)
    walkable[1, :] = False
    # from (0, 0) the wall at row 1 traps the agent in row 0
    reachable = bfs_reachable_count(walkable, (0, 0), 5)
    assert reachable == 5  # row 0 has 5 cells


def test_bfs_reachable_count_out_of_bounds_start() -> None:
    walkable = np.ones((3, 3), dtype=bool)
    assert bfs_reachable_count(walkable, (-1, 0), 2) == 0
