# PythonInteractiveRobotics

Interactive robotics and embodied intelligence with minimal Python examples.

PythonInteractiveRobotics is an educational open-source project for learning
closed-loop robotics, environment interaction, active perception, manipulation,
navigation, failure recovery, and embodied intelligence.

It is inspired by the clarity of PythonRobotics, but focuses on interactive
robotics loops rather than standalone algorithms.

## Current Status

- 32 runnable examples
- 20 learning-path roadmap examples
- 31 README GIFs generated from runnable examples
- 80 smoke and regression tests
- Core dependencies only: `numpy` and `matplotlib`

See `docs/status.md` for the implementation snapshot and `docs/plan.md` for
the working execution plan.

## Why this project?

Modern robotics is not just planning a path or running a controller once.
Robots observe, act, fail, retry, update beliefs, and replan in partially
observable environments.

This repository teaches those loops with small, readable, runnable Python
examples.

## Design goals

- Run in 5 seconds
- Minimal dependencies
- No ROS required
- No Docker required
- No GPU required
- No heavy simulator required
- Notebook friendly
- Interactive
- Closed-loop
- Failure-aware
- Educational

## Install

Minimal local install:

```bash
git clone <repo>
cd PythonInteractiveRobotics
pip install -e .
```

For contributors and GIF regeneration:

```bash
pip install -e ".[dev]"
```

## Run your first example

```bash
python examples/manipulation/01_pick_and_retry.py
```

You should see a tiny tabletop world where a robot tries to pick an object,
fails sometimes, updates its belief, and retries with a different strategy.

For a smaller first loop:

```bash
python examples/runtime/01_sense_act_loop.py
```

See `examples/README.md` for the complete runnable example index.

## See The Loops

These GIFs are generated from the runnable examples, not separate animations.

### Runtime and first manipulation loop

| Sense-act loop | Pick and retry |
| --- | --- |
| ![A point robot repeatedly observes noisy pose, acts, and observes again.](docs/assets/gifs/sense_act_loop.gif) | ![A tabletop robot misses grasps, updates belief, and retries.](docs/assets/gifs/pick_and_retry.gif) |

### Manipulation

| Reactive grasping | Closed-loop IK |
| --- | --- |
| ![A gripper servos toward an updated object belief, misses because of visual bias, corrects, and grasps.](docs/assets/gifs/reactive_grasping.gif) | ![A 2-link arm observes a noisy moving target and repeatedly servos with Jacobian IK until tracking stabilizes.](docs/assets/gifs/closed_loop_ik.gif) |

| Moving target reaching | Object search and pick |
| --- | --- |
| ![A 2-link arm predicts a briefly occluded moving target and keeps servoing until it reaches the target.](docs/assets/gifs/moving_target_reaching.gif) | ![A tabletop agent searches viewpoints, stores object memory, misses a low-confidence pick, then reobserves and succeeds.](docs/assets/gifs/object_search_and_pick.gif) |

| Push then grasp | Probabilistic suction sorting |
| --- | --- |
| ![A target starts under a shelf, the robot detects a blocked grasp, pushes it into open space, and then picks it.](docs/assets/gifs/push_then_grasp.gif) | ![A suction sorter estimates per-object success probabilities, recovers from a suction miss, prepares the seal, retries, and sorts into bins.](docs/assets/gifs/probabilistic_suction_sorting.gif) |

| Belief-guided grasp selection | Active viewpoint for grasp |
| --- | --- |
| ![A grasp agent keeps a belief over three pose hypotheses, picks the grasp with highest expected success, misses, runs a Bayes update, and tries a different grasp.](docs/assets/gifs/belief_grasp_selection.gif) | ![A grasp agent looks from the viewpoint that maximally reduces occlusion under its pose belief, updates the belief from each observation, then grasps with the type that maximizes expected success.](docs/assets/gifs/active_viewpoint_for_grasp.gif) |

| Clear path before pick |
| --- |
| ![A tabletop agent tries to pick the target, gets a precondition failure because an obstacle blocks the gripper path, picks the obstacle, places it in the clear zone, and retries the original pick.](docs/assets/gifs/clear_path_before_pick.gif) |

### Navigation and recovery

| Reactive obstacle avoidance | Dynamic obstacle avoidance |
| --- | --- |
| ![A grid robot uses fake lidar to avoid observed obstacles.](docs/assets/gifs/reactive_obstacle_avoidance.gif) | ![A grid robot avoids a moving obstacle with one-step prediction.](docs/assets/gifs/dynamic_obstacle_avoidance.gif) |

| Online A* replanning |
| --- |
| ![A grid robot plans through unknown space, observes a hidden wall, and replans.](docs/assets/gifs/online_replanning_astar.gif) |

| Frontier exploration | Belief-based navigation |
| --- | --- |
| ![A grid robot selects frontier cells to reveal unknown map space.](docs/assets/gifs/frontier_exploration.gif) | ![A grid robot maintains a belief heatmap, estimated pose, and true pose while navigating.](docs/assets/gifs/belief_based_navigation.gif) |

| Active SLAM toy | Interactive MPC |
| --- | --- |
| ![A grid robot reduces pose and map uncertainty with active sensing.](docs/assets/gifs/active_slam_toy.gif) | ![A point robot repeatedly replans short-horizon controls around a moving obstacle.](docs/assets/gifs/interactive_mpc.gif) |

| Blocked path recovery | Localization uncertainty recovery |
| --- | --- |
| ![A grid robot detects a newly blocked path, steps back, marks the blocked cell, and replans.](docs/assets/gifs/blocked_path_recovery.gif) | ![A grid robot starts with a bimodal pose belief, drives toward a landmark to break the symmetry, then navigates to the goal.](docs/assets/gifs/localization_uncertainty_recovery.gif) |

| Information-gain navigation | Multi-agent avoidance |
| --- | --- |
| ![A grid robot scouts an observation point to reveal an unknown gate state, then runs A* with full information to either the short route or the long detour.](docs/assets/gifs/information_gain_navigation.gif) | ![A grid robot shares the grid with two goal-seeking other agents, predicts each agent's next step, and A* around the predicted cells to reach its own goal.](docs/assets/gifs/multi_agent_avoidance.gif) |

| Safety filter (CBF) |
| --- |
| ![A point robot's naive go-to-goal nominal velocity is projected at each step onto a control-barrier-function half-space for each obstacle, sliding around them without the policy itself ever knowing they exist.](docs/assets/gifs/safety_filter_cbf.gif) |

### Embodied AI

| Goal command pick | Door search POMDP |
| --- | --- |
| ![A controlled language goal is parsed, then a tabletop robot searches, updates belief, misses grasps, and retries.](docs/assets/gifs/goal_command_pick.gif) | ![A room-search agent updates key-location belief after a locked door and an empty container, then finds the key.](docs/assets/gifs/door_search_pomdp.gif) |

| Goal-conditioned minikitchen | Tiny VLA loop |
| --- | --- |
| ![A kitchen agent parses a bring goal, searches containers, handles a closed cabinet, picks a mug, and places it on the table.](docs/assets/gifs/goal_conditioned_minikitchen.gif) | ![A toy VLA loop parses a language goal, reads visual tokens, picks from low confidence, recovers with a close view, and places the block.](docs/assets/gifs/tiny_vla_loop.gif) |

| Object permanence toy |
| --- |
| ![An embodied agent sees an object, watches it go behind an occluder, persists its memory, walks to the remembered position, and peeks behind the occluder to recover the object.](docs/assets/gifs/object_permanence_toy.gif) |

| Curiosity grid exploration |
| --- |
| ![A grid robot keeps a visit-count map, picks the least-visited reachable cell as an intrinsic curiosity target, walks to it on an A* path, and repeats until the visited coverage of free cells crosses a threshold.](docs/assets/gifs/curiosity_grid_exploration.gif) |

### World models

| Tiny world-model planning | Model error recovery |
| --- | --- |
| ![A point robot predicts action-conditioned dynamics, observes drift model error, updates a residual model, and replans to the goal.](docs/assets/gifs/tiny_world_model_planning.gif) | ![A point robot detects a sudden dynamics shift, switches to a short system-identification probe phase, updates the learned offset, and resumes goal navigation.](docs/assets/gifs/model_error_recovery.gif) |

Regenerate them with:

```bash
python scripts/make_gifs.py
```

Run the smoke suite and GIF checks with:

```bash
python scripts/run_all_smoke_tests.py --gifs --check-gifs
```

CI runs the same smoke suite and GIF checks on Python 3.10, 3.11, and 3.12.

## Core idea

```python
obs = env.reset(seed=0)
agent.reset()

for t in range(max_steps):
    action = agent.act(obs)
    obs, reward, done, info = env.step(action)
    agent.update(obs, reward, info)
    env.render()

    if done:
        break
```

The goal is not photorealism.
The goal is to understand the perception-action loop.

Every example returns a `Trace`, so headless runs can be inspected without
rendering. See `docs/trace.md` for the full trace contract.

```python
trace = run(seed=0, render=False)
summary = trace.summary()
print(summary.steps, summary.success, summary.failure_counts, summary.counters)
```

## Example categories

- Manipulation
- Navigation
- Active perception
- Failure recovery
- Belief-based decision making
- Embodied AI
- Tiny world models
- Robot runtime loops

## What this is not

This is not a production robotics framework.
This is not a replacement for ROS2, Nav2, MoveIt, MuJoCo, Isaac Sim, or Habitat.
This is a lightweight educational bridge toward them.

Bridge direction is documented separately:

- `docs/plan.md`
- `docs/trace.md`
- `docs/ros2_bridge_strategy.md`
- `docs/simulator_integration_strategy.md`

## Philosophy

Toy world, real concept.

A simplified 2D world is enough to teach:

- partial observability
- online replanning
- active perception
- retry
- collision
- uncertainty
- manipulation failure
- closed-loop intelligence

## Dependency policy

Core dependencies are intentionally small:

- Python >= 3.10
- numpy
- matplotlib

Optional extras are used for everything heavier:

```bash
pip install -e ".[dev]"      # pytest and GIF checks
pip install -e ".[viz]"      # GIF export only
pip install -e ".[pygame]"
pip install -e ".[rl]"
pip install -e ".[mujoco]"
pip install -e ".[pybullet]"
```

ROS2 and simulator integrations are optional bridges, not core dependencies.

`GridWorld2D`, `DynamicObstacleGridWorld`, `BlockedPathWorld`,
`MovingObstacleWorld`, and `Tabletop2D` also have lightweight Gymnasium-style
adapters:

```python
import numpy as np

from pir.adapters import (
    BlockedPathWorldGymnasiumAdapter,
    DynamicObstacleGridWorldGymnasiumAdapter,
    GridWorldGymnasiumAdapter,
    MovingObstacleWorldGymnasiumAdapter,
    Tabletop2DGymnasiumAdapter,
)

env = GridWorldGymnasiumAdapter(seed=0)
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(1)  # north

dynamic = DynamicObstacleGridWorldGymnasiumAdapter(seed=0)
obs, info = dynamic.reset(seed=0)
obs, reward, terminated, truncated, info = dynamic.step(2)  # east

blocked = BlockedPathWorldGymnasiumAdapter()
obs, info = blocked.reset(seed=0)
obs, reward, terminated, truncated, info = blocked.step(2)  # east

moving = MovingObstacleWorldGymnasiumAdapter(seed=0)
obs, info = moving.reset(seed=0)
obs, reward, terminated, truncated, info = moving.step(
    np.asarray([0.30, 0.10], dtype=np.float32)
)  # continuous velocity

tabletop = Tabletop2DGymnasiumAdapter(seed=0)
obs, info = tabletop.reset(seed=0)
obs, reward, terminated, truncated, info = tabletop.step(
    {"action_type": 0, "target": obs["camera"], "position": obs["detection_position"]}
)
```

Install `pip install -e ".[rl]"` when you want Gymnasium spaces for RL tooling.

## Contributing

See `CONTRIBUTING.md` and `docs/example_authoring.md` before adding examples.
Contributions should keep the loop readable, failure-aware, headless-testable,
and fast to run.

## License

MIT.
