# Implementation Gap Audit

This audit ranks the most useful next implementation work after the initial
repository push, CI verification, Gymnasium adapter expansion, trace summaries,
and trace replay example.

## Current Stable Surface

- 22 runnable examples
- 59 smoke and regression tests
- CI green on Python 3.10, 3.11, and 3.12
- 21 generated README GIFs with nonblank checks
- Gymnasium-style adapters for `GridWorld2D`, `DynamicObstacleGridWorld`, and
  `Tabletop2D`
- `Trace.summary()` and `examples/runtime/26_trace_replay.py`

## Ranked Next Work

| Rank | Candidate | Impact | Effort | Risk | Recommendation |
| --- | --- | --- | --- | --- | --- |
| 1 | `docs/trace.md` | high | low | low | Do next |
| 2 | `BlockedPathWorld` Gymnasium adapter | medium | low-medium | medium | Do after trace docs |
| 3 | `examples/navigation/10_localization_uncertainty_recovery.py` | high | medium-high | medium | Best next major example |
| 4 | `MovingObstacleWorld` continuous-control adapter | medium | medium | medium | Good RL bridge follow-up |
| 5 | `examples/manipulation/08_belief_grasp_selection.py` | high | medium | medium | Good after navigation example |

## 1. Trace Documentation

Why it matters:

- Trace summaries and replay now exist, but their contract is only explained in
  scattered README snippets.
- A short dedicated document would make headless inspection discoverable.
- It finishes the remaining Priority 5 documentation task.

Suggested scope:

- Add `docs/trace.md`.
- Explain `observations`, `actions`, `rewards`, and `infos`.
- Explain `info["failure"]` and `Failure`.
- Show `trace.failures()` and `trace.summary()` examples.
- Mention that replay tooling is intentionally small and not a logging
  framework.

Acceptance checks:

- `python scripts/run_all_smoke_tests.py --check-gifs`
- Markdown asset checks remain green.

## 2. `BlockedPathWorld` Gymnasium Adapter

Why it matters:

- It is already listed as the next Gymnasium candidate.
- `BlockedPathWorld` has a clean `reset()` / `step()` world boundary.
- The action mapping can reuse `GRID_ACTIONS`.
- The observation is compact: time, robot, goal, known map, dynamic blocker,
  last blocked cell.

Implementation note:

- `BlockedPathWorld` currently lives inside
  `examples/navigation/09_blocked_path_recovery.py`, not `pir/worlds/`.
- Importing an example module from `pir/adapters` would blur the package
  boundary.
- Before adding the adapter, move only `BlockedPathWorld` into `pir/worlds/` or
  create a small wrapper module that does not pull in example-agent code.

Suggested tests:

- reset shape
- discrete action decoding
- success termination
- timeout truncation
- recoverable `blocked_path` failure remains nonterminal
- `info["raw_obs"]` preserves the original observation

## 3. `10_localization_uncertainty_recovery.py`

Why it matters:

- It is the top next-tier example in `docs/plan.md`.
- The existing navigation set already has belief-based navigation and active
  SLAM, but not a focused recovery loop for ambiguous localization.
- It would teach "I am not sure where I am, so I should take an information
  action before acting toward the goal."

Suggested loop:

```text
ambiguous pose belief -> detect high uncertainty -> move to landmark view ->
belief collapses -> resume goal navigation
```

Keep it small:

- reuse the style of `06_belief_based_navigation.py`
- use a fixed grid and a few landmarks
- force an initial ambiguous belief
- make recovery visible through entropy and a `localization_recovery_count`

Required updates:

- `examples/README.md`
- `examples/navigation/README.md`
- `tests/test_examples_smoke.py`
- `docs/status.md`
- optional GIF only if it becomes a major visible example

## 4. Continuous-Control Gymnasium Adapter

Why it matters:

- `MovingObstacleWorld` in `08_interactive_mpc.py` is the clearest continuous
  action-space candidate.
- It would demonstrate that optional RL compatibility is not limited to grid
  worlds.

Risk:

- `MovingObstacleWorld.step()` returns a tuple, not `StepResult`.
- The observation and action spaces need Gymnasium `Box` spaces.
- Like `BlockedPathWorld`, the world currently lives inside an example module.

Recommendation:

- Do this after the blocked-path adapter, using the same package-boundary
  decision.

## 5. Manipulation Belief Grasp Selection

Why it matters:

- The manipulation examples cover retry, visual servoing, IK, search, pushing,
  and suction sorting.
- A focused belief-to-grasp-choice example would fill a clear concept gap:
  multiple possible object poses leading to different grasp choices and failure
  updates.

Suggested loop:

```text
pose belief -> choose grasp with best expected success -> fail -> update belief
-> choose different grasp
```

Recommendation:

- Good next example after navigation localization recovery, unless the project
  wants manipulation growth first.

## Recommended Sequence

1. Add `docs/trace.md`.
2. Extract or package `BlockedPathWorld` cleanly, then add its Gymnasium
   adapter.
3. Add `examples/navigation/10_localization_uncertainty_recovery.py`.
4. Decide whether the next expansion should be continuous-control RL adapter or
   manipulation belief-grasp example.

This order keeps the current surface stable, finishes a mostly complete trace
story, then returns to the plan's highest-value new example work.
