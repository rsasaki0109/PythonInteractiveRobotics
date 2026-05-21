# Navigation Examples

## Learning Order

Start with reactive obstacle avoidance, then add dynamic obstacles, online
replanning, exploration, belief, active SLAM, short-horizon control, and
failure recovery.

## What This Teaches

These examples treat navigation as repeated estimation and correction. The
robot observes only part of the world, chooses an action, then uses the next
observation to update a map, pose belief, short-horizon plan, or recovery
state.

## GIF Gallery

| Reactive obstacle avoidance | Dynamic obstacle avoidance |
| --- | --- |
| ![A grid robot uses fake lidar to avoid observed obstacles.](../../docs/assets/gifs/reactive_obstacle_avoidance.gif) | ![A grid robot avoids a moving obstacle with one-step prediction.](../../docs/assets/gifs/dynamic_obstacle_avoidance.gif) |

| Online A* replanning | Frontier exploration |
| --- | --- |
| ![A grid robot plans through unknown space, observes a hidden wall, and replans.](../../docs/assets/gifs/online_replanning_astar.gif) | ![A grid robot selects frontier cells to reveal unknown map space.](../../docs/assets/gifs/frontier_exploration.gif) |

| Belief-based navigation | Active SLAM toy |
| --- | --- |
| ![A grid robot maintains a belief heatmap, estimated pose, and true pose while navigating.](../../docs/assets/gifs/belief_based_navigation.gif) | ![A grid robot reduces pose and map uncertainty with active sensing.](../../docs/assets/gifs/active_slam_toy.gif) |

| Interactive MPC | Blocked path recovery |
| --- | --- |
| ![A point robot repeatedly replans short-horizon controls around a moving obstacle.](../../docs/assets/gifs/interactive_mpc.gif) | ![A grid robot detects a newly blocked path, steps back, marks the blocked cell, and replans.](../../docs/assets/gifs/blocked_path_recovery.gif) |

| Localization uncertainty recovery | Information-gain navigation |
| --- | --- |
| ![A grid robot starts with a bimodal pose belief, drives toward a landmark to break the symmetry, then navigates to the goal.](../../docs/assets/gifs/localization_uncertainty_recovery.gif) | ![A grid robot scouts an observation point to reveal an unknown gate state, then runs A* with full information to either the short route or the long detour.](../../docs/assets/gifs/information_gain_navigation.gif) |

## `02_reactive_obstacle_avoidance.py`

### What this teaches

A robot can choose actions from the latest observation without building a full
global planner. Fake lidar reveals nearby free cells and obstacles. The agent
uses that observation to avoid blocked directions while still moving toward the
goal.

### Run

```bash
python examples/navigation/02_reactive_obstacle_avoidance.py
```

### Key loop

```text
observe lidar -> choose safe direction -> move -> reveal more map -> choose again
```

### Simplifications

- grid world
- four-direction lidar
- no continuous dynamics
- no A*
- no SLAM

### Things to try

- Reduce `lidar_range` in `GridWorld2D`.
- Add another wall to the default map.
- Change the fallback direction order.
- Count how often the agent enters `avoid_obstacle`.

## `03_dynamic_obstacle_avoidance.py`

### What this teaches

The obstacle map can change after every action. A robot should use the latest
observation and a small prediction to avoid stepping into a moving actor.

### Run

```bash
python examples/navigation/03_dynamic_obstacle_avoidance.py
```

### Key loop

```text
observe moving obstacle -> predict next cell -> choose safe move -> observe again
```

### Simplifications

- grid world
- one moving obstacle
- one-step prediction
- no velocity obstacle math
- no global planner

### Things to try

- Change the dynamic obstacle route in `DynamicObstacleGridWorld`.
- Remove `predicted_dynamic_obstacles` from the safety check.
- Increase the penalty for cells near the moving obstacle.
- Track how often the agent enters `wait_for_gap`.

## `04_online_replanning_astar.py`

### What this teaches

A plan is only valid relative to the robot's current map. The agent plans
through unknown cells, observes a hidden wall, invalidates the path, and runs
A* again.

### Run

```bash
python examples/navigation/04_online_replanning_astar.py
```

### Key loop

```text
observe map -> plan path -> move -> reveal obstacle -> invalidate path -> replan
```

### Simplifications

- grid world
- four-direction motion
- unknown cells are treated as traversable
- A* is implemented directly in the example
- no SLAM or probabilistic map update

### Things to try

- Reduce `lidar_range` in `GridWorld2D`.
- Treat unknown cells as blocked and compare path length.
- Add a second hidden wall to force another invalidation.
- Print `replan_count` after each step.

## `05_frontier_exploration.py`

### What this teaches

The robot can move to gather information, not just to reach a goal. Known free
cells next to unknown space become frontier goals.

### Run

```bash
python examples/navigation/05_frontier_exploration.py
```

### Key loop

```text
observe map -> find frontiers -> choose information goal -> move -> reveal map
```

### Simplifications

- grid world
- frontier target is a known-free cell
- no probabilistic occupancy update
- no global exploration optimality

### Things to try

- Change `coverage_goal`.
- Adjust the unknown-neighbor bonus in `choose_frontier()`.
- Reduce `lidar_range` and compare `frontier_switches`.
- Stop after the first frontier and inspect the partial map.

## `06_belief_based_navigation.py`

### What this teaches

The robot does not directly know its true pose. It keeps a belief over possible
grid cells, updates that belief from noisy landmark ranges, and uses the
estimated pose to choose actions.

### Run

```bash
python examples/navigation/06_belief_based_navigation.py
```

### Key loop

```text
predict belief -> observe landmark ranges -> update belief -> choose action
```

### Simplifications

- grid world
- known map
- range-only landmark measurements
- discrete Bayes filter
- no particle filter or continuous sensor model

### Things to try

- Increase `range_sigma`.
- Lower `entropy_threshold` in `BeliefNavigationAgent`.
- Move a landmark and watch the belief heatmap change.
- Disable localization actions and compare collision or timeout behavior.

## `07_active_slam_toy.py`

### What this teaches

Mapping and localization uncertainty can shape action choice. The robot chooses
moves that reduce pose entropy and occupancy-map entropy.

### Run

```bash
python examples/navigation/07_active_slam_toy.py
```

### Key loop

```text
predict pose belief -> scan map -> update map and pose -> choose information-gain action
```

### Simplifications

- grid world
- fake cardinal lidar
- categorical pose belief
- occupancy probability map
- one-step information-gain scoring
- not a production SLAM algorithm

### Things to try

- Change the weight between pose entropy and map entropy.
- Reduce the fake lidar range.
- Start with a more uncertain occupancy map.
- Compare greedy goal motion with information-gain action choice.

## `08_interactive_mpc.py`

### What this teaches

Control is also a closed loop. The robot repeatedly predicts short rollouts,
scores collision risk against a moving obstacle, executes one command, observes
again, and replans.

### Run

```bash
python examples/navigation/08_interactive_mpc.py
```

### Key loop

```text
observe obstacle -> roll out candidate controls -> choose lowest cost -> act -> observe again
```

### Simplifications

- continuous 2D point robot
- sampled controls instead of numerical optimization
- one moving obstacle
- short horizon
- no dynamics beyond velocity commands

### Things to try

- Change the control horizon.
- Add a stronger collision penalty.
- Increase the obstacle speed.
- Reduce the candidate controls and watch the trajectory degrade.

## `09_blocked_path_recovery.py`

### What this teaches

Failure is part of the API. A path can become blocked during execution, and the
robot should detect that failure, update memory, recover, and replan.

### Run

```bash
python examples/navigation/09_blocked_path_recovery.py
```

### Key loop

```text
plan path -> execute -> detect blocked path -> mark blocked -> recover -> replan
```

### Simplifications

- grid world
- one newly appearing blocker
- A* in the example file
- recovery is a single step-back action
- no long-horizon behavior tree

### Things to try

- Move the surprise blocker to a different corridor.
- Remove the step-back recovery and compare replanning.
- Count recoverable failures in the returned `Trace`.
- Make the blocked cell permanent only after two failed moves.

## `10_localization_uncertainty_recovery.py`

### What this teaches

The robot can wake up in a state where the pose belief is ambiguous: two mirror
cells give identical landmark distances. Before chasing the goal, the agent
takes an information action to break the symmetry, then resumes navigation
once the belief collapses.

### Run

```bash
python examples/navigation/10_localization_uncertainty_recovery.py
```

### Key loop

```text
bimodal belief -> entropy high -> move toward landmark -> belief collapses -> go to goal
```

### Simplifications

- grid world
- one landmark on the axis of symmetry
- prior is hand-built to be bimodal
- one-way state machine: `localize` -> `go_to_goal`
- no relocalization once goal seeking starts

### Things to try

- Change the two `candidate_cells` so the symmetry is column-aligned instead.
- Lower `entropy_threshold` and watch `localization_recovery_count` grow.
- Add a second landmark off the axis and remove the information loop.
- Raise `range_sigma` and see whether the belief still collapses cleanly.

## `24_information_gain_navigation.py`

### What this teaches

When the cost of a wrong route is high, an agent should spend a few steps on
information-gathering before committing. Here the wall between start and goal
has one candidate gate of unknown state. The agent first scouts an observation
point that lidar-reveals the gate, updates its belief, and only then runs A*
with full information. Compare to `04_online_replanning_astar.py`, which
discovers the same information passively while running greedy A* on an
optimistic map.

### Run

```bash
python examples/navigation/24_information_gain_navigation.py
```

### Key loop

```text
unknown gate -> scout to observation point -> lidar reveals gate -> A* with full info -> goal
```

### Simplifications

- grid world with one vertical wall and one candidate gate
- one bottom opening is always known free
- gate state is binary (open or closed) and revealed by direct lidar
- observation point is a fixed cell on the way to the gate
- A* treats unknown cells as free during scouting
- no continuous information value computation

### Things to try

- Run with `--candidate-closed` and compare path length and replan count.
- Move `observation_point` further from the gate and watch the scout cost rise.
- Lower `lidar_range` so the gate cannot be revealed from the scout point.
- Add a second candidate gate elsewhere and observe which one is scouted first.
