# Runtime Examples

## `01_sense_act_loop.py`

### What this teaches

Robot intelligence is a loop, not a one-shot plan. The robot observes a noisy
state, acts, and uses the next observation to choose the next action.

### Run

```bash
python examples/runtime/01_sense_act_loop.py
```

### Key loop

```text
observe noisy pose -> choose velocity -> move -> observe again
```

### Simplifications

- point robot
- one circular obstacle
- no global planner
- no SLAM
- no physics simulator

### Things to try

- Increase the observation noise in `observe()`.
- Move the obstacle closer to the direct goal path.
- Reduce the action speed in `policy()`.
- Inspect `trace.infos` for clearance and goal-error history.

## `26_trace_replay.py`

### What this teaches

A `Trace` is enough to inspect a run after it finishes. The replay walks
recorded observations, actions, rewards, and infos without rerunning the policy
or environment dynamics.

### Run

```bash
python examples/runtime/26_trace_replay.py
```

### Key loop

```text
run source loop headless -> record Trace -> replay observations and actions -> summarize
```

### Simplifications

- replays the smallest runtime loop only
- uses recorded noisy observations, not true simulator state
- no file format or database
- no timeline controls beyond stride

### Things to try

- Run with `--no-render` and inspect `trace.summary()`.
- Increase `--max-steps`.
- Change `--stride` to skip rendered replay frames.
- Swap `record_trace()` to another example that returns a `Trace`.
