# Project Plan

This document is the working execution plan for PythonInteractiveRobotics. It
turns the repository philosophy into concrete next steps while keeping the
first-run experience small, fast, and failure-aware.

## Current Baseline

The repository currently has:

- 22 runnable examples
- 20 numbered learning-path examples
- 2 extra examples outside the original learning-path roadmap
- 21 generated README GIFs
- 59 smoke and regression tests
- GitHub Actions CI for Python 3.10, 3.11, and 3.12
- core dependencies limited to `numpy` and `matplotlib`
- optional Gymnasium-style adapters for `GridWorld2D`,
  `DynamicObstacleGridWorld`, and `Tabletop2D`
- bridge strategy docs for ROS2 and simulators

The root first-run command must remain:

```bash
python examples/manipulation/01_pick_and_retry.py
```

The full local verification command is:

```bash
python scripts/run_all_smoke_tests.py --check-gifs
```

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

Next candidates:

1. `BlockedPathWorld`
2. one continuous-control example such as `MovingObstacleWorld`
3. a tiny embodied-AI wrapper for controlled language goals

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

Goal: grow from 22 examples toward 30 examples without losing readability.

Recommended next examples:

| Priority | Example | Area | Loop |
| --- | --- | --- | --- |
| 1 | `examples/navigation/10_localization_uncertainty_recovery.py` | navigation | pose uncertainty -> information action -> recover |
| 2 | `examples/manipulation/08_belief_grasp_selection.py` | manipulation | pose belief -> grasp choice -> failure update |
| 3 | `examples/manipulation/09_active_viewpoint_for_grasp.py` | manipulation | choose view -> reduce occlusion -> grasp |
| 4 | `examples/embodied_ai/21_object_permanence_toy.py` | embodied AI | object disappears -> memory persists -> search |
| 5 | `examples/embodied_ai/22_where_did_i_see_it.py` | embodied AI | memory query -> revisit place -> act |
| 6 | `examples/world_models/23_model_error_recovery.py` | world model | prediction failure -> update model -> recover |
| 7 | `examples/navigation/24_information_gain_navigation.py` | navigation | goal progress vs information gain |
| 8 | `examples/manipulation/25_clear_path_before_pick.py` | manipulation | precondition failure -> clear obstacle -> pick |
| 9 | `examples/navigation/27_multi_agent_avoidance.py` | navigation | observe agents -> avoid -> replan |

Selection rule:

- prefer examples that introduce a new interaction concept
- avoid adding another example that only changes a visual theme
- keep each example under roughly 500 lines unless there is a strong reason

### Priority 5: Trace And Replay Tools

Goal: make the internal loop easier to inspect after a run.

Already done:

- lightweight `Trace.summary()` and `summarize_trace(trace)` helpers
- tests for reward, success, failure-count, terminal/recoverable failure, and
  loop-counter summary fields
- `examples/runtime/26_trace_replay.py`
- smoke test for headless trace replay

Tasks:

1. Document how a trace records observations, actions, rewards, and infos.
2. Keep trace tooling small enough that it does not become a logging framework.

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

Next:

- extend to selected worlds only when the action and observation mapping stays
  clear
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
| `docs/example_roadmap.md` | numbered first 20 examples |
| `docs/learning_paths.md` | learner-facing paths through examples |
| `docs/example_authoring.md` | rules for adding examples |
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
