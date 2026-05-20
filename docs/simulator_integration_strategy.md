# Simulator Integration Strategy

Simulator is a backend, not the product.

The first product is the toy world. MuJoCo, PyBullet, Habitat, Isaac Sim, and
ROS2 real robots are graduation paths after the loop is understood. The root
package must stay runnable without GPU drivers, large assets, Docker, ROS, or a
physics simulator.

## Purpose

Simulator bridges should preserve the same interaction loop learners already
see in the toy examples:

```text
observe -> update belief or memory -> choose action -> act
        -> observe failure or success -> retry, recover, or replan
```

The bridge is useful only when it makes that loop easier to transfer to a more
realistic stack. It should not replace the small examples as the first learning
surface.

## Concept Parity Before Fidelity

Start by recreating an existing toy loop in a higher-fidelity backend.

```text
Tabletop2D:
  pick -> fail -> update belief -> retry

MuJoCo or PyBullet:
  same loop, more realistic contact and kinematics
```

Do not add simulator-specific tasks until there is concept parity with a toy
example. A learner should be able to point to the same observation, belief,
action, failure, and retry step in both versions.

## Backend Roles

| Backend | Best role |
| --- | --- |
| Gymnasium | optional `reset()` / `step()` compatibility for toy worlds |
| MuJoCo | arm control, contact-rich tabletop manipulation, servo loops |
| PyBullet | URDF robots, IK, collision checks, robot learning exercises |
| Habitat | embodied navigation, object search, memory, instruction following |
| Isaac Sim | ROS2 bridge, photorealistic sensors, industrial integration |
| ROS2 real robot | optional hardware bridge after the toy and simulator loop match |

## Optional Package Shape

Keep bridge code out of core `pir/` until the dependency boundary is proven:

```text
pir_sim/
  adapters/
    gymnasium_adapter.py
    mujoco_tabletop_adapter.py
    pybullet_tabletop_adapter.py
    habitat_minikitchen_adapter.py
  examples/
    mujoco_pick_and_retry.py
    pybullet_closed_loop_ik.py
    habitat_object_search.py
```

Early adapters should be direct and readable:

```python
class MujocoTabletopAdapter:
    def reset(self): ...
    def step(self, action): ...
    def render(self): ...
```

Only introduce a shared simulator abstraction after at least three adapters
naturally share the same shape.

## Mapping From Current Toy Examples

| Toy example | Simulator bridge concept |
| --- | --- |
| `examples/runtime/01_sense_act_loop.py` | Gymnasium wrapper for the smallest `reset()` / `step()` loop |
| `examples/navigation/02_reactive_obstacle_avoidance.py` | PyBullet or Habitat range observations feeding a reactive policy |
| `examples/navigation/04_online_replanning_astar.py` | simulator map update -> planner update -> replan |
| `examples/navigation/05_frontier_exploration.py` | Habitat exploration with frontier-like goal selection |
| `examples/navigation/07_active_slam_toy.py` | higher-fidelity sensing while preserving explicit map uncertainty |
| `examples/navigation/08_interactive_mpc.py` | MuJoCo or PyBullet short-horizon control around moving actors |
| `examples/navigation/09_blocked_path_recovery.py` | collision or blocked motion result -> recovery state -> retry |
| `examples/manipulation/01_pick_and_retry.py` | MuJoCo or PyBullet grasp attempt -> contact failure -> retry |
| `examples/manipulation/03_closed_loop_ik.py` | PyBullet IK or MuJoCo servo loop with repeated observation updates |
| `examples/manipulation/05_object_search_and_pick.py` | object detection update -> planning scene update -> pick attempt |
| `examples/manipulation/06_push_then_grasp.py` | physics contact changes the world before the next grasp |
| `examples/manipulation/07_probabilistic_suction_sorting.py` | suction/contact probability estimated from repeated outcomes |
| `examples/embodied_ai/10_door_search_pomdp.py` | Habitat object/door search with explicit belief over hidden state |
| `examples/embodied_ai/18_goal_conditioned_minikitchen.py` | Habitat or Isaac task loop with language-like goals and feedback |
| `examples/embodied_ai/19_tiny_vla_loop.py` | visual tokens -> parsed goal -> simulator skill command |
| `examples/world_models/20_tiny_world_model_planning.py` | learned residual dynamics compared with simulator rollout error |

## Integration Stages

### Stage 1: Gymnasium compatibility

Expose selected toy worlds through an optional adapter:

```python
from pir.adapters import (
    DynamicObstacleGridWorldGymnasiumAdapter,
    GridWorldGymnasiumAdapter,
    Tabletop2DGymnasiumAdapter,
)

env = GridWorldGymnasiumAdapter(seed=0)
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(action)

dynamic = DynamicObstacleGridWorldGymnasiumAdapter(seed=0)
obs, info = dynamic.reset(seed=0)

tabletop = Tabletop2DGymnasiumAdapter(seed=0)
obs, info = tabletop.reset(seed=0)
```

`GridWorld2D`, `DynamicObstacleGridWorld`, and `Tabletop2D` currently have this
bridge. The adapters are importable without Gymnasium, and
`pip install -e ".[rl]"` adds Gymnasium spaces for RL tooling. This stage should
not change the teaching examples.

### Stage 2: Concept parity demos

Create one higher-fidelity demo that mirrors one existing toy example:

- `01_pick_and_retry.py` -> MuJoCo or PyBullet tabletop retry
- `03_closed_loop_ik.py` -> PyBullet IK or MuJoCo servo
- `05_frontier_exploration.py` -> Habitat object-search or frontier demo

The README should still point first to the toy example.

### Stage 3: Bridge documentation and traces

For every simulator demo, include:

- the toy example it mirrors
- what became more realistic
- what remains simplified or fake
- how failure is represented in `info["failure"]`
- a headless smoke path
- a short GIF or trace summary

### Stage 4: Optional ROS2 / simulator coupling

Only after the simulator bridge is stable should Isaac Sim or ROS2 coupling be
introduced. That belongs beside the ROS2 bridge strategy, not in core.

## Bridge Acceptance Criteria

A simulator bridge example should:

- keep simulator dependencies optional
- preserve the same closed-loop structure as a toy example
- show internal state, not only rendered physics
- expose failure, retry, belief, memory, or replanning explicitly
- run headless for CI where the backend permits it
- document asset, version, and installation requirements
- avoid large downloads in the first-run path
- avoid production robotics or safety claims

## What Not To Do

Avoid:

- adding MuJoCo, PyBullet, Habitat, Isaac Sim, or Torch to core dependencies
- hiding the loop behind a large framework wrapper
- starting with photorealistic assets before the toy loop is clear
- treating simulator fidelity as a substitute for visible belief or failure
- building a universal simulator plugin system too early
- requiring Docker, CUDA, or ROS for the first example

The bridge should let learners say: "this is the same interaction loop I saw in
the toy world, now with a more realistic backend."
