"""Grid A* and reachability shared across navigation and embodied-AI examples.

The same A* skeleton is repeated across ten examples (04, 05, 06, 09,
10, 24, 27, 28, 32, 33) with small variations: some treat unknown
cells as free, some take an extra `blocked` set, some need a per-cell
shaping cost. This module exposes one canonical implementation.

The convention is that callers convert their own grid representation
(occupancy, known_map with FREE/OCCUPIED/UNKNOWN, etc.) into a bool
`walkable` array before calling `astar`. That keeps the planner
agnostic to the semantics of the surrounding world.
"""

from __future__ import annotations

import heapq
from typing import Iterable

import numpy as np


CARDINAL_DIRECTIONS: tuple[tuple[int, int], ...] = (
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
)


def manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    """Cardinal Manhattan distance between two grid cells."""

    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(
    walkable: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    *,
    edge_cost: np.ndarray | None = None,
    blocked: Iterable[tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    """Grid A* on a 2D bool `walkable` map.

    The path is a list of `(row, col)` tuples from `start` to `goal`,
    inclusive. Returns `[]` if no path exists or if either endpoint is
    not walkable. The expansion uses 4-neighborhood cardinal moves
    only.

    Parameters
    ----------
    walkable : np.ndarray
        2D bool array. `True` means the cell can be traversed.
    start : (row, col)
        Starting cell.
    goal : (row, col)
        Goal cell.
    edge_cost : optional 2D float array
        Cost of entering each target cell. When given, the A* uses
        `edge_cost[target]` as the step cost rather than a constant
        `1`. The heuristic stays Manhattan, so consistency holds as
        long as `edge_cost >= 1`. The caller is responsible for that.
    blocked : optional iterable of cells
        Extra cells the planner must avoid even though they appear
        walkable. Useful for marking recently observed blockages
        without mutating the underlying map.
    """

    height, width = walkable.shape
    walkable_bool = walkable.astype(bool, copy=False)
    if not _in_bounds(start, height, width) or not walkable_bool[start]:
        return []
    if not _in_bounds(goal, height, width) or not walkable_bool[goal]:
        return []
    blocked_set: set[tuple[int, int]] = set(blocked) if blocked is not None else set()
    if start in blocked_set or goal in blocked_set:
        return []

    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    g_score: dict[tuple[int, int], float] = {start: 0.0}
    open_heap: list[tuple[float, int, tuple[int, int]]] = [
        (float(manhattan(start, goal)), 0, start)
    ]
    counter = 1
    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        if current == goal:
            return _reconstruct_path(parent, goal)
        for dr, dc in CARDINAL_DIRECTIONS:
            neighbor = (current[0] + dr, current[1] + dc)
            if not _in_bounds(neighbor, height, width):
                continue
            if not walkable_bool[neighbor]:
                continue
            if neighbor in blocked_set:
                continue
            step = 1.0 if edge_cost is None else float(edge_cost[neighbor])
            tentative = g_score[current] + step
            if tentative < g_score.get(neighbor, float("inf")):
                g_score[neighbor] = tentative
                parent[neighbor] = current
                priority = tentative + manhattan(neighbor, goal)
                heapq.heappush(open_heap, (priority, counter, neighbor))
                counter += 1
    return []


def bfs_reachable_count(
    walkable: np.ndarray,
    start: tuple[int, int],
    k: int,
) -> int:
    """Number of cells reachable from `start` in at most `k` cardinal moves.

    Walls and out-of-bounds cells are not counted. `start` is counted
    even when not walkable - that matches the empowerment convention
    where you want to know "how many places am I in?" not "is this a
    legal place to stand?".
    """

    height, width = walkable.shape
    if not _in_bounds(start, height, width):
        return 0
    seen: set[tuple[int, int]] = {start}
    frontier: list[tuple[tuple[int, int], int]] = [(start, 0)]
    head = 0
    while head < len(frontier):
        (r, c), depth = frontier[head]
        head += 1
        if depth >= k:
            continue
        for dr, dc in CARDINAL_DIRECTIONS:
            nb = (r + dr, c + dc)
            if not _in_bounds(nb, height, width):
                continue
            if not walkable[nb]:
                continue
            if nb in seen:
                continue
            seen.add(nb)
            frontier.append((nb, depth + 1))
    return len(seen)


def _in_bounds(cell: tuple[int, int], height: int, width: int) -> bool:
    return 0 <= cell[0] < height and 0 <= cell[1] < width


def _reconstruct_path(
    parent: dict[tuple[int, int], tuple[int, int] | None],
    goal: tuple[int, int],
) -> list[tuple[int, int]]:
    path: list[tuple[int, int]] = [goal]
    while parent[path[-1]] is not None:
        path.append(parent[path[-1]])  # type: ignore[arg-type]
    path.reverse()
    return path
