# Trace Inspection

Every runnable example returns a `Trace`. A trace is the smallest shared record
of a closed-loop run: what the robot observed, what action it chose, what reward
it received, and what metadata the world reported.

## Shape

`Trace` stores four aligned lists. Index `i` describes one completed step:

| Field | Meaning |
| --- | --- |
| `observations[i]` | observation after the action was applied |
| `actions[i]` | action chosen by the agent or policy |
| `rewards[i]` | scalar reward returned by the world |
| `infos[i]` | metadata such as success, failure, counters, belief state, or plan state |

Examples append to a trace inside the control loop:

```python
result = env.step(action)
obs, reward, done, info = result.as_tuple()
trace.append(obs, action, reward, info)
```

The trace is intentionally in-memory and lightweight. It is not a logging
framework, file format, telemetry system, or simulator replay API.

## Failures

Failures should be reported through `info["failure"]` as a `Failure` object:

```python
from pir.core.types import Failure

info["failure"] = Failure(
    "blocked_path",
    "planned path is blocked by a newly observed obstacle",
    recoverable=True,
)
```

Use `trace.failures()` to extract structured failures without scanning every
info dictionary manually:

```python
failures = trace.failures()
failure_kinds = [failure.kind for failure in failures]
```

Recoverable failures should usually trigger a different next action: replan,
retry from a new belief, change viewpoint, push before grasping, or update a
world model. Terminal failures such as timeout or collision should use
`recoverable=False`.

## Summary

Use `trace.summary()` for compact headless inspection:

```python
trace = run(seed=0, render=False)
summary = trace.summary()

print(summary.steps)
print(summary.total_reward)
print(summary.success)
print(summary.failure_counts)
print(summary.retry_count)
print(summary.counters)
```

The summary includes:

| Field | Meaning |
| --- | --- |
| `steps` | number of recorded actions |
| `total_reward` | sum of all rewards |
| `success` | whether any step reported `info["success"]` |
| `failure_counts` | count by failure kind |
| `recoverable_failure_count` | number of recoverable failures |
| `terminal_failure_count` | number of nonrecoverable failures |
| `counters` | maximum numeric `*_count` values seen in `infos` |
| `retry_count` | shortcut for `counters.get("retry_count", 0)` |
| `final_info` | final info dictionary |

`counters` is deliberately simple. It picks up fields such as `retry_count`,
`replan_count`, `recovery_count`, `search_count`, or `model_error_count` when an
example reports them.

## Replay

`examples/runtime/26_trace_replay.py` records the smallest runtime loop
headless, then replays the recorded observations, actions, rewards, and infos:

```bash
python examples/runtime/26_trace_replay.py --no-render --max-steps 12
```

The replay example does not rerun the policy or world dynamics. It walks the
recorded trace, tracks cumulative reward, and optionally renders the observation
and action history. This keeps replay useful for teaching without turning the
project into a logging framework.

## Testing

Tests should assert the loop concept directly:

```python
trace = module.run(seed=0, render=False, max_steps=40)
summary = trace.summary()

assert summary.success is True
assert summary.retry_count >= 1
assert "grasp_miss" in summary.failure_counts
```

For failure contracts, prefer `trace.failures()` when the exact step is not the
thing being tested. Prefer direct `trace.infos[-1]` assertions when the final
state is the important behavior.
