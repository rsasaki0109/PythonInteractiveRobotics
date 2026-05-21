# Implementation Gap Audit

This audit ranks the most useful next implementation work after the original
20-example learning-path roadmap, the Priority 4 example tier, the
continuous-control Gymnasium adapter, and the curiosity exploration example.

## Current Stable Surface

- 34 runnable examples
- 20 numbered learning-path examples plus 14 extras
- 33 generated README GIFs with nonblank checks
- 84 smoke, adapter, and regression tests
- CI green on Python 3.10, 3.11, and 3.12
- Gymnasium-style adapters for `GridWorld2D`, `DynamicObstacleGridWorld`,
  `BlockedPathWorld`, `MovingObstacleWorld`, and `Tabletop2D`
- `Trace.summary()` and `examples/runtime/26_trace_replay.py`
- Trace contract docs in `docs/trace.md`
- Bridge strategy docs for ROS2 and simulators

## Already Closed Since Last Audit

| Closed | Notes |
| --- | --- |
| `docs/trace.md` | Trace contract is now documented and referenced from `examples/runtime/26_trace_replay.py`. |
| `BlockedPathWorld` Gymnasium adapter | Lives in `pir/adapters/gymnasium_adapter.py`. Tests cover reset, decode, terminated, truncated, and recoverable-failure paths. |
| `10_localization_uncertainty_recovery.py` | Information action then resume to goal. Smoke test asserts `info_gain_step_count >= 1`. |
| `08_belief_grasp_selection.py` | Belief over three pose hypotheses, Bayes update on miss, retries with a different grasp. |
| `09_active_viewpoint_for_grasp.py` | Viewpoint selection by expected reliability under pose belief. |
| `21_object_permanence_toy.py` | Memory persists across an occluder. |
| `22_where_did_i_see_it.py` (removed) | The "explore -> memorize -> query -> revisit" loop overlapped `21_object_permanence_toy.py` without adding a different lesson; removed after a GPT Pro curriculum review. |
| `04_online_replanning_astar.py` (kept) | GPT Pro suggested merging into 09. After review, kept because the trigger for replanning is *passive observation* of an unknown wall, while 09's trigger is *execution failure*. Both docstrings and READMEs were updated to make the distinction explicit. |
| `23_model_error_recovery.py` | Regime shift detection then short system-identification probe. |
| `24_information_gain_navigation.py` | Active scout to reveal a gate before A* with full info. |
| `25_clear_path_before_pick.py` | Precondition failure recovery by clearing an obstacle. |
| `27_multi_agent_avoidance.py` | A* over predicted-next cells of two goal-seeking agents. |
| `MovingObstacleWorld` extraction | Moved from the example file into `pir/worlds/moving_obstacle.py`, with `MovingObstacleWorldGymnasiumAdapter` and five adapter tests. |
| `28_curiosity_grid_exploration.py` | Visit-count-driven novelty target selection with A* commitment. |
| `29_safety_filter_cbf.py` | Runtime CBF safety filter on a nominal go-to-goal policy; `dh/dt >= -alpha * h` projection per obstacle. |
| `30_conformal_ask_for_help.py` | Offline conformal calibration; place when the prediction set is a singleton, defer to a toy oracle when not. Counters: `q_hat`, `help_request_count`, `coverage_violation_count`. |
| `31_options_with_interrupts.py` | Sutton-style options framework on a battery-aware navigation task; `go_to_goal` and `dock_and_charge` options with explicit `β` and a meta-policy interrupt rule. Counters: `option_start_count`, `option_interrupt_count`, `interrupts_due_to_battery_count`, `dock_count`, `recharge_step_count`. |

## Ranked Next Work

| Rank | Candidate | Impact | Effort | Risk | Recommendation |
| --- | --- | --- | --- | --- | --- |
| 1 | Tighten `docs/example_authoring.md` | medium | low | low | Do next; surface stability matters more than new examples now. |
| 2 | Trace summary helpers for failure tables | medium | low-medium | low | Small win for `docs/trace.md` readers. |
| 3 | Embodied-AI controlled-language Gymnasium adapter | medium | medium | medium | Useful RL bridge for language-conditioned policies. |
| 4 | `CuriosityGridWorld` Gymnasium adapter | medium | medium | low | Lets RL users compare extrinsic vs intrinsic reward. |
| 5 | One more failure-recovery example (e.g., recovery from wrong skill choice) | high | medium | medium | Only after consolidation is solid. |

## 1. Tighten Example Authoring Docs

Why it matters:

- New contributors should not need to read three examples to learn the
  "What this teaches / Run / Key loop / Simplifications / Things to try"
  pattern.
- Some category READMEs use slightly different headings.
- `docs/example_authoring.md` is the single contract for new examples.

Suggested scope:

- Pin the section headings every example README block must use.
- Note that the docstring header should declare success and failure conditions.
- Add a checklist for the `Failure` kinds an example introduces.

Acceptance checks:

- `python scripts/run_all_smoke_tests.py --check-gifs`
- Markdown asset checks remain green.

## 2. Trace Summary Helpers For Failure Tables

Why it matters:

- `Trace.summary()` already collects failures and counters.
- A small helper that prints a Markdown failure table from a trace would make
  README captions and `docs/trace.md` examples concrete and copyable.

Suggested scope:

- Add `summary.failure_table()` returning a list of `(kind, count, recoverable)`.
- Add an example block in `docs/trace.md`.
- Optional: surface the table in `examples/runtime/26_trace_replay.py`.

Risks:

- Keep this small. It is documentation glue, not a logging framework.

## 3. Controlled-language Gymnasium Adapter

Why it matters:

- The four worlds with adapters cover navigation, manipulation, and dynamic
  obstacles. Language-conditioned goals are still example-only.
- A thin wrapper around `examples/embodied_ai/01_goal_command_pick.py` or
  `examples/embodied_ai/18_goal_conditioned_minikitchen.py` would let RL users
  experiment with goal-conditioned reward shaping.

Risk:

- The action space mixes discrete skills with continuous coordinates.
- The goal string itself is part of the observation, which Gymnasium spaces do
  not represent naturally.
- Likely needs a small `Text` or `Discrete(goal_id)` projection rather than
  the raw string.

## 4. `CuriosityGridWorld` Gymnasium Adapter

Why it matters:

- The grid is small and discrete.
- Action and observation spaces map cleanly to existing grid adapter helpers.
- It would let RL users compare extrinsic reward (reaching a goal cell) with
  intrinsic reward (visit-count novelty) on the same world.

Recommendation:

- Do this after the controlled-language adapter so a clear pattern is in place
  for novelty-shaped observations.

## 5. One More Failure-Recovery Example

Why it matters:

- The current set covers blocked path, push-then-grasp, precondition failure,
  model error recovery, localization recovery, and information-gain detour.
- A "wrong-skill choice" recovery (try skill A → fails for a recoverable
  reason → switch to skill B) is still missing from the manipulation or
  embodied-AI categories.

Constraint:

- Only worth doing if it is meaningfully different from the existing failure
  examples. If it ends up restating `01_pick_and_retry.py`, drop it.

## Recommended Sequence

1. Tighten `docs/example_authoring.md` so new examples land with consistent
   docstring headers and category README blocks.
2. Add a small failure-table helper on `TraceSummary`.
3. Add the controlled-language Gymnasium adapter, then the
   `CuriosityGridWorld` adapter, in that order.
4. Decide whether the next example should be a new failure-recovery loop or
   should wait for a clear interaction-concept gap.

This order keeps the educational surface stable, finishes the documentation
and adapter expansion that was already in progress, and defers new examples
until they fill a clear gap rather than incrementing a counter.
