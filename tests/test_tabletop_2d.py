import numpy as np

from pir.core.types import Failure
from pir.worlds.tabletop_2d import Tabletop2D


def test_bad_pick_reports_recoverable_failure() -> None:
    env = Tabletop2D(
        seed=0,
        detector_noise=0.0,
        base_false_negative_rate=0.0,
        max_attempts=2,
    )
    env.reset(seed=0)

    result = env.step({"type": "pick", "position": np.array([0.1, 0.1])})

    assert result.done is False
    assert isinstance(result.info["failure"], Failure)
    assert result.info["failure"].kind == "grasp_miss"
    assert result.info["failure"].recoverable is True


def test_exact_pick_can_succeed_when_success_probability_is_one() -> None:
    env = Tabletop2D(
        seed=1,
        detector_noise=0.0,
        base_false_negative_rate=0.0,
        base_grasp_success=1.0,
    )
    env.reset(seed=1)

    result = env.step({"type": "pick", "position": env.obj.position.copy()})

    assert result.done is True
    assert result.info["success"] is True
    assert result.reward == 1.0
