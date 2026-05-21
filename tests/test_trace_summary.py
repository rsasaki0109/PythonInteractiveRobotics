from __future__ import annotations

from pir.core.types import Failure, Trace, summarize_trace


def test_trace_summary_collects_rewards_failures_and_counters() -> None:
    trace = Trace()
    trace.append(
        {"position": 0},
        "try_pick",
        -0.2,
        {
            "failure": Failure("grasp_miss", "missed the object", True),
            "retry_count": 1,
        },
    )
    trace.append(
        {"position": 1},
        "retry_pick",
        1.0,
        {"success": True, "retry_count": 2, "search_count": 1},
    )

    summary = trace.summary()

    assert summary.steps == 2
    assert summary.total_reward == 0.8
    assert summary.success is True
    assert summary.failure_counts == {"grasp_miss": 1}
    assert summary.failure_kinds == ["grasp_miss"]
    assert summary.recoverable_failure_count == 1
    assert summary.terminal_failure_count == 0
    assert summary.retry_count == 2
    assert summary.counters["search_count"] == 1
    assert summary.final_info["success"] is True


def test_summarize_trace_counts_terminal_failures() -> None:
    trace = Trace()
    trace.append(
        {},
        "east",
        -0.1,
        {"failure": Failure("timeout", "maximum steps reached", False)},
    )

    summary = summarize_trace(trace)

    assert summary.success is False
    assert summary.failure_counts == {"timeout": 1}
    assert summary.recoverable_failure_count == 0
    assert summary.terminal_failure_count == 1


def test_empty_trace_summary_is_well_defined() -> None:
    summary = Trace().summary()

    assert summary.steps == 0
    assert summary.total_reward == 0
    assert summary.success is False
    assert summary.failure_counts == {}
    assert summary.counters == {}
    assert summary.final_info == {}
