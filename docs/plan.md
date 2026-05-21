# Project Plan

This document is the working execution plan for PythonInteractiveRobotics. It
turns the repository philosophy into concrete next steps while keeping the
first-run experience small, fast, and failure-aware.

## Current Baseline

The repository currently has:

- 32 runnable examples
- 20 numbered learning-path examples
- 12 extra examples outside the original learning-path roadmap
- 31 generated README GIFs
- 80 smoke and regression tests
- GitHub Actions CI for Python 3.10, 3.11, and 3.12
- core dependencies limited to `numpy` and `matplotlib`
- optional Gymnasium-style adapters for `GridWorld2D`,
  `DynamicObstacleGridWorld`, `Tabletop2D`, `BlockedPathWorld`, and
  `MovingObstacleWorld`
- bridge strategy docs for ROS2 and simulators
- trace inspection docs for headless run analysis

The root first-run command must remain:

```bash
python examples/manipulation/01_pick_and_retry.py
```

The full local verification command is:

```bash
python scripts/run_all_smoke_tests.py --check-gifs
```

## Handoff Snapshot

This section is written for the next coding agent. It should be enough to
resume without rereading the whole repository first.

Current repository state:

- Branch: `main`
- Check the latest pushed commit with `git log --oneline --decorate -1`.
- Latest known GitHub Actions run on `main`: green on Python 3.10, 3.11, and
  3.12
- Last local verification command run successfully:

```bash
python scripts/run_all_smoke_tests.py --check-gifs
```

Current high-value files to read before changing code:

1. `docs/implementation_gap_audit.md`
2. `docs/trace.md`
3. `docs/example_authoring.md`
4. `pir/adapters/gymnasium_adapter.py`
5. `tests/test_gymnasium_adapter.py`
6. `examples/navigation/09_blocked_path_recovery.py`

Recent completed work:

- Initial repository contents were committed and pushed.
- CI was confirmed green after the initial push.
- `DynamicObstacleGridWorldGymnasiumAdapter` was added.
- Trace summaries were added through `Trace.summary()` and
  `summarize_trace(trace)`.
- `examples/runtime/26_trace_replay.py` was added.
- `docs/trace.md` was added.
- `docs/implementation_gap_audit.md` was added.
- Category READMEs were tightened with more "What this teaches" and "Things to
  try" sections.
- `BlockedPathWorld` was extracted to `pir/worlds/blocked_path.py`; the
  example at `examples/navigation/09_blocked_path_recovery.py` imports the
  world from there while keeping its agent, A* policy, and run loop visible.
- `BlockedPathWorldGymnasiumAdapter` was added in
  `pir/adapters/gymnasium_adapter.py` and exported from
  `pir/adapters/__init__.py`. Optional `dynamic_blocker` / `last_blocked_cell`
  cells are encoded as `(-1, -1)` sentinels and preserved untouched in
  `info["raw_obs"]`.
- Focused adapter tests were added covering reset shape, step shape, action
  decoding, success termination, timeout truncation, and recoverable
  `blocked_path` failure.
- `examples/navigation/10_localization_uncertainty_recovery.py` was added.
  The agent wakes up with a bimodal pose belief, takes information actions
  toward the symmetric landmark, then switches one-way to goal navigation
  after the belief collapses. A GIF and smoke test cover the loop.
- `examples/manipulation/08_belief_grasp_selection.py` was added. The agent
  keeps a belief over three pose hypotheses, picks the grasp with the
  highest expected success across that belief, and runs a Bayes update on
  every miss until a grasp lands. GIF and smoke test cover the loop.
- `examples/manipulation/09_active_viewpoint_for_grasp.py` was added. The
  agent picks the viewpoint that maximally reduces expected occlusion under
  its pose belief, runs a Bayes update from each observation, then grasps
  with the type that maximizes expected success. GIF and smoke test cover
  the loop.
- `examples/embodied_ai/21_object_permanence_toy.py` was added. The agent
  sees an object once, watches it go behind an occluder, persists the last
  known position across the disappearance, walks to it, and peeks behind
  the occluder to recover the object. GIF and smoke test cover the loop.
- `examples/embodied_ai/22_where_did_i_see_it.py` was removed because its
  "explore -> memorize -> query memory -> revisit" loop overlapped
  `21_object_permanence_toy.py` without adding a clearly different lesson.
  The remaining cognitive-robotics example for memory under loss of
  observation is `21_object_permanence_toy.py`.
- `examples/navigation/29_safety_filter_cbf.py` was added. A naive
  go-to-goal nominal policy is paired with a separate runtime safety
  filter that, at every step, projects the nominal velocity onto the
  closest CBF-style safe half-space for each obstacle. The example
  exposes `u_nominal`, `u_safe`, `barrier_h_min`, `closest_approach`,
  `filter_active_count`, and `stuck_count` in `info`, and surfaces a
  recoverable `safety_filter_stuck` Failure when the projection clips
  the velocity below the stuck threshold. GIF and smoke test cover the
  loop.
- `examples/world_models/23_model_error_recovery.py` was added. The agent
  starts with an identity dynamics model, detects a regime shift when
  prediction error spikes, runs a short system-identification probe phase,
  averages observed offsets, updates the learned offset, and resumes goal
  navigation with the corrected model. GIF and smoke test cover the loop.
- `examples/navigation/24_information_gain_navigation.py` was added. The
  agent scouts an observation point to lidar-reveal an unknown gate state,
  then runs A* with full information to either a short route through the
  gate or a longer detour through the bottom opening. GIF and two smoke
  tests (open and closed gate) cover the loop.
- `examples/manipulation/25_clear_path_before_pick.py` was added. The
  agent tries to pick the target, hits a `precondition_blocked` failure,
  picks the obstacle, places it in a known clear zone, and retries the
  original pick. GIF and smoke test cover the loop.
- `examples/navigation/27_multi_agent_avoidance.py` was added. The robot
  shares the grid with two goal-seeking other agents, predicts each
  agent's next step, and runs A* over a map that treats current and
  predicted-next cells as occupied. GIF and smoke test cover the loop.
- `MovingObstacleWorld` was extracted from `examples/navigation/08_interactive_mpc.py`
  into `pir/worlds/moving_obstacle.py`, and `MovingObstacleWorldGymnasiumAdapter`
  was added in `pir/adapters/gymnasium_adapter.py`. The adapter exposes a
  continuous-control `Box(2,)` action space, splits `terminated` vs
  `truncated`, and preserves the raw observation in `info["raw_obs"]`. Five
  adapter tests were added.
- `examples/embodied_ai/28_curiosity_grid_exploration.py` was added. The agent
  keeps a visit-count map, picks the most novel reachable free cell using a
  novelty score, plans an A* path, commits to the path until target is reached
  or stale, and terminates when visited coverage of free cells crosses a
  threshold. GIF and smoke test cover the loop.

The next agent should not redo those items. If any of them seem missing, first
check the current branch and latest pulled commit.

Recommended next task:

1. Consolidate before expanding. The original Priority 4 list (28 numbered
   examples + extras) is complete. Strengthen `docs/example_authoring.md`,
   tighten category READMEs, and update `docs/implementation_gap_audit.md`
   before adding more examples.
2. If a new example is requested, prefer concepts that fill a clear gap (for
   example, a continuous-control manipulation primitive, a multi-step value-of-
   information loop, or a richer language-grounded recovery).
3. Keep the package/example boundary the same as the `BlockedPathWorld` and
   `MovingObstacleWorld` work: environment in `pir/worlds/`, agent + policy +
   run loop in the example.
4. Adapter tests should be added before broadening the adapter API further.

Do not start yet:

- Do not add Gymnasium as a core dependency.
- Do not add ROS2, MuJoCo, PyBullet, Torch, JAX, TensorFlow, Docker, or GPU
  requirements.
- Do not convert examples into a framework.
- Do not move shared code into `pir/` unless the move makes package boundaries
  cleaner or avoids three-or-more-example repetition.
- Do not regenerate GIFs unless a visible README GIF changes or a new major
  example is added.

## Product Boundary

This project is an educational collection of interactive robotics loops. It is
not a production robotics framework.

Keep in core:

- small toy worlds
- readable examples
- structured failures
- belief, memory, retry, recovery, and replanning loops
- deterministic smoke tests
- GIFs that reveal the loop and internal state

Keep out of core:

- ROS2
- MuJoCo
- PyBullet
- Isaac Sim
- Habitat
- Torch, JAX, TensorFlow
- pretrained models
- large datasets
- Docker or GPU requirements
- production safety claims

Optional bridge work is welcome only after the equivalent toy loop is already
clear and tested.

## Operating Rules

Every new example should satisfy these rules:

- runs headless with `render=False`
- starts rendering within 5 seconds in normal local use
- returns a `Trace`
- reports failures through `info["failure"]` as a `Failure`
- includes at least one of failure, uncertainty, retry, partial observation,
  memory, belief, recovery, or online replanning
- has a smoke test
- has a short GIF if it is a major example
- documents what is fake or simplified
- does not introduce mandatory heavy dependencies

Shared code should move into `pir/` only after it naturally repeats across
three or more examples.

## Near-Term Priorities

### Priority 1: Keep The Surface Stable

Goal: make the existing educational surface hard to break.

Tasks:

1. Keep `scripts/run_all_smoke_tests.py --check-gifs` green.
2. Keep README and category README GIF links checked by tests.
3. Add regression tests whenever a new failure contract appears.
4. Add small tests around shared adapters before extending them.
5. Keep CI aligned with the local smoke command.

Acceptance criteria:

- CI passes on Python 3.10, 3.11, and 3.12.
- local smoke plus GIF check reports all tests passed.
- no README GIF is missing, empty, or blank.

### Priority 2: Strengthen Example Documentation

Goal: make every example teach the loop without requiring source-code reading
first.

Tasks:

1. Add or tighten "What this teaches" sections in category READMEs.
2. Make every major example state its key loop in one line.
3. Add "Simplifications" and "Things to try" where missing.
4. Keep category GIF galleries updated when examples are added.
5. Add short captions that name internal state, not just motion.

Acceptance criteria:

- a learner can pick an example from `examples/README.md`
- the category README explains the loop before the code is opened
- the GIF shows the relevant belief, failure, retry, or replanning state

### Priority 3: Extend Optional Gymnasium Compatibility

Goal: let RL users touch toy loops without changing the examples or adding a
core dependency.

Already done:

- `GridWorldGymnasiumAdapter`
- `DynamicObstacleGridWorldGymnasiumAdapter`
- `Tabletop2DGymnasiumAdapter`
- `BlockedPathWorldGymnasiumAdapter`
- `MovingObstacleWorldGymnasiumAdapter` (continuous-control)

Next candidates:

1. a tiny embodied-AI wrapper for controlled language goals
2. an adapter for a curiosity / exploration world (e.g.
   `CuriosityGridWorld`) so RL agents can compare extrinsic and intrinsic
   reward signals on the same world

Rules:

- adapters must be importable without Gymnasium installed
- `pip install -e ".[rl]"` may add Gymnasium spaces
- adapters must preserve raw observations in `info["raw_obs"]`
- terminated vs truncated must be tested
- examples must not be rewritten around Gymnasium

Acceptance criteria:

- tests cover reset shape, step shape, action decoding, success termination,
  timeout truncation, and at least one failure path
- no new core dependency is added

### Priority 4: Add The Next Example Tier

Goal: grow from 24 examples toward 30 examples without losing readability.

Already done from the previous tier:

- `examples/navigation/10_localization_uncertainty_recovery.py` — pose
  uncertainty -> information action -> resume goal navigation.
- `examples/manipulation/08_belief_grasp_selection.py` — pose belief ->
  grasp choice -> failure update -> retry.
- `examples/manipulation/09_active_viewpoint_for_grasp.py` — pose belief ->
  active viewpoint -> Bayes update -> grasp.
- `examples/embodied_ai/21_object_permanence_toy.py` — see object ->
  memory persists across occlusion -> peek to recover.
- `examples/world_models/23_model_error_recovery.py` — regime-shift
  detection -> system_id probe -> model update -> resume.
- `examples/navigation/24_information_gain_navigation.py` — scout the
  gate -> reveal candidate state -> A* with full information.
- `examples/manipulation/25_clear_path_before_pick.py` — try target ->
  precondition fails -> pick obstacle -> place in clear zone -> retry.
- `examples/navigation/27_multi_agent_avoidance.py` — observe agents ->
  predict next -> A* around predictions.

The original "Priority 4: Add The Next Example Tier" target of growing toward
30 examples has been met. New examples should now come from new interaction
concepts rather than from the original table, and consolidation work
(documentation, adapter coverage, gap audit) should be considered first.

Selection rule:

- prefer examples that introduce a new interaction concept
- avoid adding another example that only changes a visual theme
- keep each example under roughly 500 lines unless there is a strong reason

### Priority 5: Trace And Replay Tools

Goal: make the internal loop easier to inspect after a run.

Done baseline:

- lightweight `Trace.summary()` and `summarize_trace(trace)` helpers
- tests for reward, success, failure-count, terminal/recoverable failure, and
  loop-counter summary fields
- `examples/runtime/26_trace_replay.py`
- smoke test for headless trace replay
- `docs/trace.md`

Tasks:

1. Keep trace tooling small enough that it does not become a logging framework.
2. Only add more trace APIs when at least two examples need the same inspection
   pattern.
3. Prefer small examples and docs over a generalized replay framework.

Acceptance criteria:

- a user can run an example headless and inspect what happened
- failures and retry counts can be summarized without rendering
- trace tooling does not become a full logging framework

## Bridge Roadmap

Bridge work should follow concept parity, not simulator ambition.

### Gymnasium

Current status:

- `GridWorld2D` adapter exists
- `DynamicObstacleGridWorld` adapter exists
- `Tabletop2D` adapter exists
- `BlockedPathWorld` adapter exists

Next:

- extend to other selected worlds only when the action and observation mapping
  stays clear
- keep examples independent from Gymnasium

### Simulator Bridge

Do later:

- MuJoCo or PyBullet version of `01_pick_and_retry.py`
- PyBullet or MuJoCo version of `03_closed_loop_ik.py`
- Habitat-style object search after the toy object-search loop is stable

Do not do yet:

- photorealistic assets
- large downloads
- universal simulator plugin system
- mandatory simulator dependency

### ROS2 Bridge

Do later:

- docs mapping toy costmaps to Nav2 concepts
- docs mapping `pick_and_retry` to action feedback and retry
- optional `pir_ros2` package shape

Do not do yet:

- add `rclpy` to core
- hide the teaching loop behind launch files
- claim production robotics support

## Documentation Plan

Keep these documents distinct:

| Document | Purpose |
| --- | --- |
| `README.md` | first impression, install, GIF gallery, first run |
| `docs/status.md` | current implementation snapshot |
| `docs/plan.md` | execution plan and priorities |
| `docs/trace.md` | trace fields, failures, summaries, and replay |
| `docs/example_roadmap.md` | numbered first 20 examples |
| `docs/learning_paths.md` | learner-facing paths through examples |
| `docs/example_authoring.md` | rules for adding examples |
| `docs/implementation_gap_audit.md` | ranked next implementation candidates |
| `docs/toy_worlds.md` | toy world roles and coverage |
| `docs/ros2_bridge_strategy.md` | optional ROS2 bridge direction |
| `docs/simulator_integration_strategy.md` | optional simulator bridge direction |

When a new example is added, update:

1. `examples/README.md`
2. the relevant category README
3. `docs/learning_paths.md` if it changes a path
4. `docs/status.md` counts
5. `README.md` if it is a major visible example
6. `scripts/make_gifs.py` if it needs a GIF
7. tests

When a new adapter is added, update:

1. `pir/adapters/gymnasium_adapter.py`
2. `pir/adapters/__init__.py`
3. `tests/test_gymnasium_adapter.py`
4. `README.md` if it changes the public adapter list
5. `docs/status.md`
6. `docs/plan.md`
7. `docs/simulator_integration_strategy.md` if bridge status changes

When package code is extracted from an example, keep the example's learner-facing
loop visible. The package module should make boundaries cleaner, not hide the
teaching logic.

## Testing Plan

Keep test types small and explicit:

| Test type | Purpose |
| --- | --- |
| world tests | direct environment behavior and failures |
| example smoke tests | examples run headless and return expected loop signals |
| failure contract tests | `info["failure"]` is structured and recoverable is meaningful |
| adapter tests | optional compatibility APIs preserve toy loop semantics |
| trace summary tests | headless traces expose compact success, failure, and counter summaries |
| Markdown asset tests | README image paths and category GIF galleries stay valid |
| GIF checks | generated GIFs have frames and nonblank pixels |

Avoid slow tests in the default path. Heavy simulator tests should live in
optional bridge jobs only after those bridges exist.

## Definition Of Done For A New Example

A new example is done when:

1. `run(seed=0, render=False, max_steps=...)` returns a `Trace`
2. `main()` runs as a script
3. the key loop is visible in code and docs
4. any failure is reported as `Failure`
5. smoke test covers success or intended terminal failure
6. category README includes the example
7. GIF exists if the example is a major teaching example
8. `python scripts/run_all_smoke_tests.py --check-gifs` passes

## Definition Of Done For A New Adapter

A new optional adapter is done when:

1. the adapter module imports without Gymnasium installed
2. `pip install -e ".[rl]"` provides action and observation spaces
3. `reset()` returns `(obs, info)`
4. `step()` returns `(obs, reward, terminated, truncated, info)`
5. `info["raw_obs"]` preserves the unencoded toy-world observation
6. action decoding is tested
7. success is tested as termination
8. timeout is tested as truncation
9. at least one domain failure path is tested
10. examples are not rewritten around Gymnasium
11. `python scripts/run_all_smoke_tests.py --check-gifs` passes

## Suggested Claude Work Plan

If Claude is taking over, the most efficient first pass is:

1. Run `git status --short --branch`.
2. Run `python scripts/run_all_smoke_tests.py --check-gifs` if local
   dependencies are already installed.
3. Pick the next adapter or example from "Next candidates" under
   `Priority 3: Extend Optional Gymnasium Compatibility` and
   `Priority 4: Add The Next Example Tier`.
4. Keep the package/example boundary the same as the `BlockedPathWorld`
   work: environment in `pir/worlds/`, agent + policy + run loop in the
   example.
5. Add adapter or smoke tests before broadening any public API.
6. Run `python -m pytest tests/test_gymnasium_adapter.py -q` and the full
   smoke plus GIF check.
7. Commit and push only after the worktree is cleanly understood.

## Do Not Start Yet

These are intentionally deferred:

- full ROS2 package
- real robot bridge
- MuJoCo or PyBullet required install
- Isaac Sim integration
- Habitat datasets
- neural VLA model
- benchmark leaderboard
- large policy training
- production autonomy claims

The project should stay useful because the learner can change robot behavior
and immediately see the environment respond.
