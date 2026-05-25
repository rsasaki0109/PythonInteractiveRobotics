# PythonInteractiveRobotics

[![CI](https://github.com/rsasaki0109/PythonInteractiveRobotics/actions/workflows/ci.yml/badge.svg)](https://github.com/rsasaki0109/PythonInteractiveRobotics/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
![Core dependencies](https://img.shields.io/badge/core-numpy%20%2B%20matplotlib-orange)

**Robots observe, act, fail, retry, update beliefs, and replan.**
This repo shows that loop in small, readable Python — no ROS, no GPU, no
simulator. Just `numpy + matplotlib`.

[Open the example gallery](https://rsasaki0109.github.io/PythonInteractiveRobotics/)
or jump straight into the first runnable loop below. You can also run the
flagship loops directly in Colab:
[pick and retry](https://colab.research.google.com/github/rsasaki0109/PythonInteractiveRobotics/blob/main/notebooks/pick_and_retry.ipynb),
[safety filter](https://colab.research.google.com/github/rsasaki0109/PythonInteractiveRobotics/blob/main/notebooks/safety_filter_cbf.ipynb), and
[human correction replanning](https://colab.research.google.com/github/rsasaki0109/PythonInteractiveRobotics/blob/main/notebooks/human_correction_replanning.ipynb).
For language ambiguity, try
[clarifying question](https://colab.research.google.com/github/rsasaki0109/PythonInteractiveRobotics/blob/main/notebooks/clarifying_question.ipynb).
If the project helps you teach, prototype, or explain robotics loops, a GitHub
star helps others find it.

| Avoiding | Reaching under occlusion | Mapping while uncertain |
| --- | --- | --- |
| ![A point robot's naive go-to-goal velocity is projected onto a CBF safe set at every step. The policy itself never knows the obstacles exist - a separate runtime safety filter slides it around them.](docs/assets/gifs/safety_filter_cbf.gif) | ![A 2-link arm predicts a briefly occluded moving target, keeps servoing through the occlusion, and reaches the intercept point when the target reappears.](docs/assets/gifs/moving_target_reaching.gif) | ![A toy active-SLAM agent shrinks pose belief and occupancy belief at the same time, by picking moves that maximize expected entropy drop.](docs/assets/gifs/active_slam_toy.gif) |

## Try it

```bash
git clone https://github.com/rsasaki0109/PythonInteractiveRobotics.git
cd PythonInteractiveRobotics
python3 -m pip install -e .
python3 examples/manipulation/01_pick_and_retry.py
```

A tiny tabletop robot misses a grasp, updates its belief, and retries — in
under 5 seconds. Core dependencies are `numpy` and `matplotlib` only.

For an even smaller first loop:

```bash
python3 examples/runtime/01_sense_act_loop.py
```

## Start Here

| If you want to see | Run | What it teaches |
| --- | --- | --- |
| Failure recovery | `python3 examples/manipulation/01_pick_and_retry.py` | grasp miss -> belief update -> retry |
| Runtime safety | `python3 examples/navigation/29_safety_filter_cbf.py` | nominal controller -> CBF projection -> safe motion |
| Active perception | `python3 examples/navigation/07_active_slam_toy.py` | map and pose uncertainty -> information-seeking action |
| Human correction | [Open in Colab](https://colab.research.google.com/github/rsasaki0109/PythonInteractiveRobotics/blob/main/notebooks/human_correction_replanning.ipynb) | shortcut -> human correction -> cost update -> replan |
| Language ambiguity | [Open in Colab](https://colab.research.google.com/github/rsasaki0109/PythonInteractiveRobotics/blob/main/notebooks/clarifying_question.ipynb) | ambiguous command -> ask question -> answer -> act |

## Status

38 runnable examples · 37 README GIFs · 107 smoke / regression tests ·
5 Gymnasium-style adapters · CI green on Python 3.10, 3.11, and 3.12.

See `docs/status.md` for the implementation snapshot, `docs/plan.md` for the
working execution plan, and `examples/README.md` for the complete example
index. The GitHub Pages gallery is generated from `docs/index.html`, and
`docs/public_launch.md` keeps the public launch checklist.

## Why this project?

Modern robotics is not just planning a path or running a controller once.
Robots observe, act, fail, retry, update beliefs, and replan in partially
observable environments. This repository teaches those loops with small,
readable, runnable Python examples.

## Design goals

Run in 5 seconds · minimal dependencies · no ROS / Docker / GPU / heavy
simulator required · notebook friendly · interactive · closed-loop ·
failure-aware · educational.

## Install

```bash
git clone https://github.com/rsasaki0109/PythonInteractiveRobotics.git
cd PythonInteractiveRobotics
python3 -m pip install -e .
```

For contributors and GIF regeneration: `python3 -m pip install -e ".[dev]"`.

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

| Clear path before pick | Conformal ask-for-help |
| --- | --- |
| ![A tabletop agent tries to pick the target, gets a precondition failure because an obstacle blocks the gripper path, picks the obstacle, places it in the clear zone, and retries the original pick.](docs/assets/gifs/clear_path_before_pick.gif) | ![A sorter calibrates a conformal prediction set offline, then places items when the prediction set is a singleton and asks a toy oracle for help when it is ambiguous.](docs/assets/gifs/conformal_ask_for_help.gif) |

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

| Safety filter (CBF) | Options with interrupts |
| --- | --- |
| ![A point robot's naive go-to-goal nominal velocity is projected at each step onto a control-barrier-function half-space for each obstacle, sliding around them without the policy itself ever knowing they exist.](docs/assets/gifs/safety_filter_cbf.gif) | ![A battery-aware robot runs a go-to-goal option, gets interrupted mid-task when the battery drops below threshold, switches to dock-and-charge, then resumes go-to-goal once the battery is full.](docs/assets/gifs/options_with_interrupts.gif) |

| Human correction replanning |
| --- |
| ![A grid robot starts on a shortcut, receives a human correction before entering an unwanted zone, raises that zone's traversal cost, replans, and reaches the goal by a longer route.](docs/assets/gifs/human_correction_replanning.gif) |

### Embodied AI

| Goal command pick | Door search POMDP |
| --- | --- |
| ![A controlled language goal is parsed, then a tabletop robot searches, updates belief, misses grasps, and retries.](docs/assets/gifs/goal_command_pick.gif) | ![A room-search agent updates key-location belief after a locked door and an empty container, then finds the key.](docs/assets/gifs/door_search_pomdp.gif) |

| Goal-conditioned minikitchen | Tiny VLA loop |
| --- | --- |
| ![A kitchen agent parses a bring goal, searches containers, handles a closed cabinet, picks a mug, and places it on the table.](docs/assets/gifs/goal_conditioned_minikitchen.gif) | ![A toy VLA loop parses a language goal, reads visual tokens, picks from low confidence, recovers with a close view, and places the block.](docs/assets/gifs/tiny_vla_loop.gif) |

| Clarifying question |
| --- |
| ![A tabletop robot receives the ambiguous command pick the block, asks which block, receives a red answer, resolves the goal, and picks the red block.](docs/assets/gifs/clarifying_question.gif) |

| Object permanence toy |
| --- |
| ![An embodied agent sees an object, watches it go behind an occluder, persists its memory, walks to the remembered position, and peeks behind the occluder to recover the object.](docs/assets/gifs/object_permanence_toy.gif) |

| Curiosity grid exploration | Empowerment navigation |
| --- | --- |
| ![A grid robot keeps a visit-count map, picks the least-visited reachable cell as an intrinsic curiosity target, walks to it on an A* path, and repeats until the visited coverage of free cells crosses a threshold.](docs/assets/gifs/curiosity_grid_exploration.gif) | ![A grid robot prefers cells with many reachable successors by adding a k-step empowerment shaping term to its A* edge cost, sliding around narrow corridors even when the detour is slightly longer.](docs/assets/gifs/empowerment_navigation.gif) |

| Inverse reward from demo |
| --- |
| ![A grid robot watches one demo trajectory that detours through hidden scenic zones, learns linear reward weights from the demo's feature expectation versus a uniform random walk, then plans to a new goal with a shaped A* that reproduces the demonstrator's scenic preference.](docs/assets/gifs/inverse_reward_from_demo.gif) |

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
