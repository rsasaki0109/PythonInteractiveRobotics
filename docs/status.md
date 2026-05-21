# Project Status

This document records the current implementation snapshot so contributors can
see what exists, what is verified, and what should come next.

## Snapshot

- Runnable examples: 32
- Learning-path roadmap examples: 21
- README GIFs: 31
- Smoke and regression tests: 80
- Core dependencies: `numpy`, `matplotlib`
- Contributor extra: `pip install -e ".[dev]"`
- CI: Python 3.10, 3.11, 3.12
- Execution plan: `docs/plan.md`
- Trace inspection docs: `docs/trace.md`
- Bridge docs: `docs/ros2_bridge_strategy.md`,
  `docs/simulator_integration_strategy.md`
- Optional adapters: `GridWorldGymnasiumAdapter`,
  `DynamicObstacleGridWorldGymnasiumAdapter`, `Tabletop2DGymnasiumAdapter`,
  `BlockedPathWorldGymnasiumAdapter`, `MovingObstacleWorldGymnasiumAdapter`

`examples/embodied_ai/01_goal_command_pick.py` is an extra flagship
goal-command example. The numbered 20-example roadmap is tracked in
`docs/example_roadmap.md`.

## Implemented Coverage

| Area | Examples | Main concepts |
| --- | ---: | --- |
| Runtime | 2 | smallest observe-act-observe loop, trace replay |
| Navigation | 11 | reactive avoidance, dynamic obstacles, replanning, exploration, belief, active SLAM, MPC, recovery, localization recovery, information-gain detour, multi-agent avoidance |
| Manipulation | 10 | retry, reactive grasping, IK servo, moving target reaching, search, push recovery, suction sorting, belief grasp selection, active viewpoint grasp, clear path before pick |
| Embodied AI | 7 | controlled goals, memory, POMDP search, tiny VLA loop, object permanence, memory query revisit, curiosity exploration |
| World models | 2 | action-conditioned dynamics, prediction error, model update, replanning, regime-shift detection, system identification |

## Verification

Fast smoke suite:

```bash
python scripts/run_all_smoke_tests.py
```

Full README GIF regeneration and validation:

```bash
python scripts/run_all_smoke_tests.py --gifs --check-gifs
```

The GIF check verifies frame count and nonblank pixels for all generated GIFs.
The regression suite also checks structured `Failure` objects for representative
recoverable and terminal failures.
The adapter suite checks the Gymnasium-style `reset()` / `step()` contract
without making Gymnasium a core dependency.
The trace docs and summary tests cover compact reward, success, failure, and
loop-counter extraction from headless runs.
The Markdown asset check verifies local README image links and category GIF
galleries.
GitHub Actions runs the smoke suite and GIF checks on Python 3.10, 3.11, and
3.12.

## Current Boundaries

Still intentionally out of core:

- ROS2
- MuJoCo
- PyBullet
- Isaac Sim
- Habitat
- PyTorch / JAX / TensorFlow
- pretrained models
- large datasets
- Docker or GPU requirements

These belong in optional bridge phases after the toy interaction loops are
stable.

## Next Phase

The next useful phase is still not adding heavy simulators. It is stabilizing
the educational surface and using `docs/plan.md` plus the bridge docs as
dependency boundaries:

1. tighten example docs and GIF captions as new examples are added
2. expand regression tests for shared contracts as new examples are added
3. extract shared helpers only where three or more examples naturally repeat
4. extend optional Gymnasium compatibility to the next clear toy-world mappings
5. keep trace replay tooling small enough that it does not become a logging
   framework

The root first-run experience should remain:

```bash
python examples/manipulation/01_pick_and_retry.py
```
