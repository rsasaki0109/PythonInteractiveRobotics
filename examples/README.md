# Examples

Each example is a small closed-loop robotics lesson.  All examples provide a
`run(...)` function for notebooks/tests and a script entry point for direct use.

Run any example headless with its `--no-render` flag when available.

## Runtime

| Example | Run | Loop |
| --- | --- | --- |
| `runtime/01_sense_act_loop.py` | `python examples/runtime/01_sense_act_loop.py` | observe -> act -> observe |
| `runtime/26_trace_replay.py` | `python examples/runtime/26_trace_replay.py` | run loop -> record trace -> replay |

## Navigation

| Example | Run | Loop |
| --- | --- | --- |
| `navigation/02_reactive_obstacle_avoidance.py` | `python examples/navigation/02_reactive_obstacle_avoidance.py` | fake lidar -> reactive avoidance |
| `navigation/03_dynamic_obstacle_avoidance.py` | `python examples/navigation/03_dynamic_obstacle_avoidance.py` | observe moving obstacle -> avoid -> observe again |
| `navigation/04_online_replanning_astar.py` | `python examples/navigation/04_online_replanning_astar.py` | map update -> A* replanning |
| `navigation/05_frontier_exploration.py` | `python examples/navigation/05_frontier_exploration.py` | choose frontier -> observe unknown space |
| `navigation/06_belief_based_navigation.py` | `python examples/navigation/06_belief_based_navigation.py` | update pose belief -> choose action |
| `navigation/07_active_slam_toy.py` | `python examples/navigation/07_active_slam_toy.py` | act to reduce map and pose uncertainty |
| `navigation/08_interactive_mpc.py` | `python examples/navigation/08_interactive_mpc.py` | predict -> control -> replan |
| `navigation/09_blocked_path_recovery.py` | `python examples/navigation/09_blocked_path_recovery.py` | detect blocked path -> recover -> replan |
| `navigation/10_localization_uncertainty_recovery.py` | `python examples/navigation/10_localization_uncertainty_recovery.py` | ambiguous pose -> information action -> resume goal |
| `navigation/24_information_gain_navigation.py` | `python examples/navigation/24_information_gain_navigation.py` | scout observation -> reveal gate -> A* with full info |
| `navigation/27_multi_agent_avoidance.py` | `python examples/navigation/27_multi_agent_avoidance.py` | observe agents -> predict next -> A* around predictions |

## Manipulation

| Example | Run | Loop |
| --- | --- | --- |
| `manipulation/01_pick_and_retry.py` | `python examples/manipulation/01_pick_and_retry.py` | grasp miss -> belief update -> retry |
| `manipulation/02_reactive_grasping.py` | `python examples/manipulation/02_reactive_grasping.py` | visual servo -> miss -> correct -> grasp |
| `manipulation/03_closed_loop_ik.py` | `python examples/manipulation/03_closed_loop_ik.py` | observe target -> Jacobian step -> observe error |
| `manipulation/04_moving_target_reaching.py` | `python examples/manipulation/04_moving_target_reaching.py` | estimate velocity -> predict through occlusion -> reach |
| `manipulation/05_object_search_and_pick.py` | `python examples/manipulation/05_object_search_and_pick.py` | search viewpoint -> memory -> pick -> retry |
| `manipulation/06_push_then_grasp.py` | `python examples/manipulation/06_push_then_grasp.py` | blocked grasp -> push world -> grasp |
| `manipulation/07_probabilistic_suction_sorting.py` | `python examples/manipulation/07_probabilistic_suction_sorting.py` | suction miss -> update probability -> prepare -> sort |
| `manipulation/08_belief_grasp_selection.py` | `python examples/manipulation/08_belief_grasp_selection.py` | pose belief -> grasp choice -> miss -> update -> retry |
| `manipulation/09_active_viewpoint_for_grasp.py` | `python examples/manipulation/09_active_viewpoint_for_grasp.py` | choose view -> reduce occlusion -> grasp |
| `manipulation/25_clear_path_before_pick.py` | `python examples/manipulation/25_clear_path_before_pick.py` | try pick -> precondition fails -> clear obstacle -> retry |

## Embodied AI

| Example | Run | Loop |
| --- | --- | --- |
| `embodied_ai/01_goal_command_pick.py` | `python examples/embodied_ai/01_goal_command_pick.py "find the red block and pick it"` | parse goal -> search -> pick -> retry |
| `embodied_ai/10_door_search_pomdp.py` | `python examples/embodied_ai/10_door_search_pomdp.py` | room belief -> door/container action -> belief update |
| `embodied_ai/18_goal_conditioned_minikitchen.py` | `python examples/embodied_ai/18_goal_conditioned_minikitchen.py "bring mug to table"` | goal -> container search -> pick -> place |
| `embodied_ai/19_tiny_vla_loop.py` | `python examples/embodied_ai/19_tiny_vla_loop.py "place red block in blue bin"` | language -> visual tokens -> skill -> retry |
| `embodied_ai/21_object_permanence_toy.py` | `python examples/embodied_ai/21_object_permanence_toy.py` | see object -> memory persists across occlusion -> peek |
| `embodied_ai/28_curiosity_grid_exploration.py` | `python examples/embodied_ai/28_curiosity_grid_exploration.py` | visit counts -> novelty score -> A* to novel cell -> coverage |

## World Models

| Example | Run | Loop |
| --- | --- | --- |
| `world_models/20_tiny_world_model_planning.py` | `python examples/world_models/20_tiny_world_model_planning.py` | predict -> act -> observe model error -> update -> replan |
| `world_models/23_model_error_recovery.py` | `python examples/world_models/23_model_error_recovery.py` | predict -> error spike -> probe -> update model -> resume |

## Verification

```bash
pip install -e ".[dev]"
python scripts/run_all_smoke_tests.py
python scripts/run_all_smoke_tests.py --gifs --check-gifs
```
