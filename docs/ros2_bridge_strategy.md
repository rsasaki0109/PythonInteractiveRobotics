# ROS2 Bridge Strategy

ROS2 should not be a core dependency.

The core package must stay importable without ROS, Docker, a colcon workspace,
or a system-level ROS install. ROS2 belongs in an optional bridge package after
the toy examples have made the interaction loop clear.

## Purpose

The bridge should not reimplement ROS2, Nav2, MoveIt, or Autoware. It should map
the small loops in this repository onto the concepts a learner will meet in a
real ROS2 stack.

```text
toy loop first -> ROS2 concept mapping -> optional bridge demo
```

## Concept Mapping

| PythonInteractiveRobotics | ROS2 |
| --- | --- |
| observation | topic subscription |
| action command | topic publication |
| long task | action |
| quick query | service |
| trace | bag or log |
| viewer | RViz or custom viewer |
| recovery state | behavior tree node or action feedback |
| failure object | action result / diagnostic message |
| belief / memory | node-local state, map layer, or planning scene |

## Optional Package Shape

Keep ROS2 code outside `pir/`:

```text
pir_ros2/
  nodes/
    pir_runtime_node.py
    toy_nav_node.py
    toy_manip_node.py
  bridges/
    obs_to_ros.py
    action_to_ros.py
    trace_to_bag.py
  examples/
    nav2_bridge_demo.py
    moveit_bridge_demo.py
```

The root package should still install and run with:

```bash
pip install -e .
python examples/manipulation/01_pick_and_retry.py
```

## Mapping From Current Toy Examples

| Toy example | ROS2 concept bridge |
| --- | --- |
| `examples/runtime/01_sense_act_loop.py` | minimal node loop: subscribe observation, publish command |
| `examples/navigation/02_reactive_obstacle_avoidance.py` | lidar topic -> velocity command |
| `examples/navigation/04_online_replanning_astar.py` | map update -> planner request -> path publication |
| `examples/navigation/05_frontier_exploration.py` | frontier goal selection -> navigation action goal |
| `examples/navigation/06_belief_based_navigation.py` | localization uncertainty -> recovery behavior |
| `examples/navigation/09_blocked_path_recovery.py` | behavior tree recovery node and action feedback |
| `examples/manipulation/01_pick_and_retry.py` | perception update -> grasp action -> failure result -> retry |
| `examples/manipulation/03_closed_loop_ik.py` | MoveIt Servo concept: repeated servo commands, not one-shot IK |
| `examples/manipulation/05_object_search_and_pick.py` | perception node -> planning scene update -> pick action |
| `examples/manipulation/06_push_then_grasp.py` | task-level recovery: change world state before retry |
| `examples/embodied_ai/18_goal_conditioned_minikitchen.py` | long-running action with feedback for goal-conditioned tasks |
| `examples/embodied_ai/19_tiny_vla_loop.py` | language goal -> perception tokens -> skill action dispatch |
| `examples/world_models/20_tiny_world_model_planning.py` | model error monitor -> planner update / diagnostic feedback |

## Nav2 Teaching Bridges

Useful docs or demos:

- toy costmap to Nav2 costmap concept
- toy frontier goal to Nav2 action goal
- toy blocked path recovery to Nav2 behavior tree recovery
- toy localization uncertainty to recovery behavior

Do not implement Nav2 from scratch. Show where the toy loop maps to Nav2
objects and actions.

## MoveIt Teaching Bridges

Useful docs or demos:

- `03_closed_loop_ik.py` -> MoveIt Servo concept
- `01_pick_and_retry.py` -> grasp action result and retry
- `05_object_search_and_pick.py` -> planning scene update
- `06_push_then_grasp.py` -> task-level manipulation recovery

Do not implement a motion planner in the bridge. The value is the connection
between perception, planning scene state, action feedback, and retry.

## Autoware Teaching Bridge

Autoware should be a later optional bridge. The nearest current toy concepts
are:

- `03_dynamic_obstacle_avoidance.py` for moving actors
- `08_interactive_mpc.py` for short-horizon control
- `09_blocked_path_recovery.py` for fail-safe recovery

Keep autonomous-driving claims conservative. These are educational concept
bridges, not production driving examples.

## Bridge Acceptance Criteria

A ROS2 bridge example should:

- live outside core `pir/`
- keep ROS2 dependencies optional
- preserve the same loop visible in the toy example
- expose failure and feedback, not only success
- have a headless or CI-friendly smoke path where possible
- document which toy simplifications disappear and which remain
- avoid requiring Docker, GPU, or a large simulator for the first bridge demo

## What Not To Do

Avoid:

- adding `rclpy` to core dependencies
- creating a universal robotics middleware abstraction too early
- hiding the toy loop behind ROS launch files
- treating RViz visualization as a replacement for explaining belief or failure
- claiming production readiness

The bridge should help learners recognize the same interaction loop in a real
ROS2 stack.
