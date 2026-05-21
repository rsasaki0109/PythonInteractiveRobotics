# Example Roadmap

The first examples should be ordered by the interaction loop they teach, not by
algorithm taxonomy.

This roadmap tracks the first 20 learning-path examples. The repository also
includes `examples/embodied_ai/01_goal_command_pick.py` as an extra flagship
controlled-language example.

| # | Example | Category | Loop |
| --- | --- | --- | --- |
| 1 | `examples/runtime/01_sense_act_loop.py` | runtime | observe -> act -> observe |
| 2 | `examples/navigation/02_reactive_obstacle_avoidance.py` | navigation | fake lidar -> reactive avoidance |
| 3 | `examples/navigation/03_dynamic_obstacle_avoidance.py` | navigation | repeated decisions around moving obstacles |
| 4 | `examples/navigation/04_online_replanning_astar.py` | navigation | map update -> A* replanning |
| 5 | `examples/navigation/05_frontier_exploration.py` | navigation | choose actions to observe unknown space |
| 6 | `examples/navigation/06_belief_based_navigation.py` | navigation | move with uncertain pose and map belief |
| 7 | `examples/navigation/07_active_slam_toy.py` | navigation / SLAM | act to improve map quality |
| 8 | `examples/navigation/08_interactive_mpc.py` | navigation / control | control with prediction and changing context |
| 9 | `examples/navigation/09_blocked_path_recovery.py` | recovery | detect blocked path -> recover |
| 10 | `examples/embodied_ai/10_door_search_pomdp.py` | embodied AI | search and remember under partial observation |
| 11 | `examples/manipulation/01_pick_and_retry.py` | manipulation | grasp miss -> belief update -> retry |
| 12 | `examples/manipulation/02_reactive_grasping.py` | manipulation | grasp under observation drift |
| 13 | `examples/manipulation/03_closed_loop_ik.py` | manipulation | servo instead of one-shot IK |
| 14 | `examples/manipulation/04_moving_target_reaching.py` | manipulation | track a moving target |
| 15 | `examples/manipulation/05_object_search_and_pick.py` | manipulation | search -> observe -> pick |
| 16 | `examples/manipulation/06_push_then_grasp.py` | manipulation | change the environment before grasping |
| 17 | `examples/manipulation/07_probabilistic_suction_sorting.py` | manipulation | success probability, retry, sorting |
| 18 | `examples/embodied_ai/18_goal_conditioned_minikitchen.py` | embodied AI | goal-conditioned interaction |
| 19 | `examples/embodied_ai/19_tiny_vla_loop.py` | embodied AI | language goal -> visual obs -> action |
| 20 | `examples/world_models/20_tiny_world_model_planning.py` | world model | plan with action-conditioned dynamics |

The flagship first manipulation example is
`examples/manipulation/01_pick_and_retry.py` because failure and retry are
visible immediately.
