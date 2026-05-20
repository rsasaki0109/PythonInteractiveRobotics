"""Fake range sensing for grid-based toy worlds."""

from __future__ import annotations

from typing import Any

import numpy as np


DIRECTIONS: dict[str, tuple[int, int]] = {
    "north": (-1, 0),
    "east": (0, 1),
    "south": (1, 0),
    "west": (0, -1),
}


def cast_cardinal_lidar(
    occupancy: np.ndarray,
    position: tuple[int, int],
    *,
    max_range: int = 5,
) -> dict[str, dict[str, Any]]:
    """Cast four grid rays and report free cells before the first obstacle."""

    row, col = position
    height, width = occupancy.shape
    scan: dict[str, dict[str, Any]] = {}

    for name, (dr, dc) in DIRECTIONS.items():
        free_cells: list[tuple[int, int]] = []
        hit: tuple[int, int] | None = None

        for distance in range(1, max_range + 1):
            cell = (row + dr * distance, col + dc * distance)
            r, c = cell

            if r < 0 or r >= height or c < 0 or c >= width:
                hit = cell
                break

            if occupancy[r, c]:
                hit = cell
                break

            free_cells.append(cell)

        scan[name] = {
            "free_cells": len(free_cells),
            "cells": free_cells,
            "hit": hit,
        }

    return scan
