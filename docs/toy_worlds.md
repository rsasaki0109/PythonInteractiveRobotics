# Toy Worlds

Toy worlds are the primary teaching surface. They are intentionally small so
true state, observation, belief, action, and failure can all be displayed.

## GridWorld2D

For navigation, frontier exploration, belief-based search, dynamic obstacle
avoidance, and active SLAM toys.

Core ideas:

- occupancy grid
- unknown cells
- fake lidar or range sensing
- noisy odometry
- dynamic obstacles
- collision checks
- replanning

## Tabletop2D

For pick and place, object search, occlusion, grasp retry, sorting, and active
camera movement.

Core ideas:

- 2D tabletop
- noisy object detections
- false negatives
- simplified contact
- probabilistic grasp success
- gripper state

## PlanarArm2D

For closed-loop IK, visual servoing, reaching under noisy observation, moving
target tracking, and reactive grasping.

Core ideas:

- 2-link or 3-link arm
- joint limits
- point end-effector
- approximate collision
- Jacobian IK
- target noise

## MiniHouseWorld

For object search, instruction following, memory, active perception, and
goal-conditioned behavior.

Core ideas:

- rooms
- doors
- containers
- hidden objects
- semantic labels
- egocentric partial observation
- simple language goals

## DriveWorld2D

For autonomous driving toys, lane following, obstacle avoidance, intersection
negotiation, and interactive MPC.

Core ideas:

- lanes
- waypoints
- moving actors
- fake lidar or bounding boxes
- speed control
- collision and near-miss metrics

## Current Example Coverage

The first implemented examples intentionally mix reusable toy worlds and
self-contained worlds. Keeping some worlds local to an example is acceptable
when it makes the teaching loop easier to read.

| Toy world family | Implemented examples |
| --- | --- |
| GridWorld2D | `02_reactive_obstacle_avoidance.py`, `03_dynamic_obstacle_avoidance.py`, `04_online_replanning_astar.py`, `05_frontier_exploration.py`, `06_belief_based_navigation.py`, `07_active_slam_toy.py`, `09_blocked_path_recovery.py` |
| Tabletop2D / tabletop local worlds | `01_pick_and_retry.py`, `02_reactive_grasping.py`, `05_object_search_and_pick.py`, `06_push_then_grasp.py`, `07_probabilistic_suction_sorting.py` |
| Planar arm local worlds | `03_closed_loop_ik.py`, `04_moving_target_reaching.py` |
| Mini house / kitchen local worlds | `10_door_search_pomdp.py`, `18_goal_conditioned_minikitchen.py`, `19_tiny_vla_loop.py` |
| Continuous planning local worlds | `08_interactive_mpc.py`, `20_tiny_world_model_planning.py` |
