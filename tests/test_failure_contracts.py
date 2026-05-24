from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from pir.core.types import Failure


ROOT = Path(__file__).resolve().parents[1]


def load_example(relative_path: str) -> ModuleType:
    path = ROOT / relative_path
    module_name = "failure_contract_" + "_".join(path.with_suffix("").parts[-3:])
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


FAILURE_EXAMPLES: list[tuple[str, dict[str, Any], set[str]]] = [
    (
        "examples/manipulation/01_pick_and_retry.py",
        {"seed": 3, "render": False, "max_steps": 20},
        {"grasp_miss"},
    ),
    (
        "examples/manipulation/04_moving_target_reaching.py",
        {"seed": 5, "render": False, "max_steps": 90},
        {"target_occluded"},
    ),
    (
        "examples/manipulation/05_object_search_and_pick.py",
        {"seed": 7, "render": False, "max_steps": 30},
        {"target_not_visible", "grasp_miss"},
    ),
    (
        "examples/manipulation/06_push_then_grasp.py",
        {"seed": 9, "render": False, "max_steps": 25},
        {"blocked_grasp"},
    ),
    (
        "examples/manipulation/07_probabilistic_suction_sorting.py",
        {"seed": 11, "render": False, "max_steps": 40},
        {"suction_miss"},
    ),
    (
        "examples/navigation/09_blocked_path_recovery.py",
        {"seed": 0, "render": False, "max_steps": 80},
        {"blocked_path"},
    ),
    (
        "examples/navigation/34_human_correction_replanning.py",
        {"seed": 0, "render": False, "max_steps": 60},
        {"human_correction"},
    ),
    (
        "examples/embodied_ai/10_door_search_pomdp.py",
        {"seed": 0, "render": False, "max_steps": 40},
        {"locked_door", "not_found"},
    ),
    (
        "examples/embodied_ai/18_goal_conditioned_minikitchen.py",
        {
            "command": "bring mug to table",
            "seed": 0,
            "render": False,
            "max_steps": 35,
        },
        {"target_not_found", "container_closed"},
    ),
    (
        "examples/embodied_ai/19_tiny_vla_loop.py",
        {
            "command": "place red block in blue bin",
            "seed": 0,
            "render": False,
            "max_steps": 25,
        },
        {"visual_pose_uncertain"},
    ),
    (
        "examples/world_models/20_tiny_world_model_planning.py",
        {"seed": 0, "render": False, "max_steps": 80},
        {"model_error"},
    ),
]


@pytest.mark.parametrize(("relative_path", "kwargs", "expected_kinds"), FAILURE_EXAMPLES)
def test_examples_report_structured_recoverable_failures(
    relative_path: str,
    kwargs: dict[str, Any],
    expected_kinds: set[str],
) -> None:
    module = load_example(relative_path)

    trace = module.run(**kwargs)

    infos_with_failure = [info for info in trace.infos if "failure" in info]
    failures = trace.failures()
    actual_kinds = {failure.kind for failure in failures}

    assert infos_with_failure
    assert len(failures) == len(infos_with_failure)
    assert expected_kinds <= actual_kinds

    for failure in failures:
        assert isinstance(failure, Failure)
        assert failure.kind
        assert failure.message
        assert isinstance(failure.recoverable, bool)

    for expected_kind in expected_kinds:
        matching = [failure for failure in failures if failure.kind == expected_kind]
        assert matching
        assert all(failure.recoverable for failure in matching)


def test_terminal_timeout_failure_is_not_recoverable() -> None:
    module = load_example("examples/navigation/09_blocked_path_recovery.py")
    env = module.BlockedPathWorld(max_steps=1)
    env.reset(seed=0)

    result = env.step("east")

    assert result.done is True
    assert isinstance(result.info["failure"], Failure)
    assert result.info["failure"].kind == "timeout"
    assert result.info["failure"].message
    assert result.info["failure"].recoverable is False
