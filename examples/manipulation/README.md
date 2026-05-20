# Manipulation Examples

## Learning Order

Start with grasp retry, then move to reactive visual correction, arm servoing,
object search, world-changing recovery, and probabilistic sorting.

## What This Teaches

These examples show manipulation as feedback around contact, perception, and
world state. The robot observes a target, attempts a skill, checks whether the
world changed as expected, and uses failure information to retry, servo,
search, push, or prepare before trying again.

## GIF Gallery

| Pick and retry | Reactive grasping |
| --- | --- |
| ![A tabletop robot misses grasps, updates belief, and retries.](../../docs/assets/gifs/pick_and_retry.gif) | ![A gripper servos toward an updated object belief, misses because of visual bias, corrects, and grasps.](../../docs/assets/gifs/reactive_grasping.gif) |

| Closed-loop IK | Moving target reaching |
| --- | --- |
| ![A 2-link arm observes a noisy moving target and repeatedly servos with Jacobian IK until tracking stabilizes.](../../docs/assets/gifs/closed_loop_ik.gif) | ![A 2-link arm predicts a briefly occluded moving target and keeps servoing until it reaches the target.](../../docs/assets/gifs/moving_target_reaching.gif) |

| Object search and pick | Push then grasp |
| --- | --- |
| ![A tabletop agent searches viewpoints, stores object memory, misses a low-confidence pick, then reobserves and succeeds.](../../docs/assets/gifs/object_search_and_pick.gif) | ![A target starts under a shelf, the robot detects a blocked grasp, pushes it into open space, and then picks it.](../../docs/assets/gifs/push_then_grasp.gif) |

| Probabilistic suction sorting |
| --- |
| ![A suction sorter estimates per-object success probabilities, recovers from a suction miss, prepares the seal, retries, and sorts into bins.](../../docs/assets/gifs/probabilistic_suction_sorting.gif) |

## `01_pick_and_retry.py`

### What this teaches

A robot should not assume a grasp succeeds. It should observe the result,
update its belief, and retry differently.

### Run

```bash
python examples/manipulation/01_pick_and_retry.py
```

### Key loop

```text
observe object -> choose grasp -> attempt -> detect failure -> update belief -> retry
```

### Simplifications

- 2D tabletop
- fake object detector
- probabilistic grasp success
- simplified contact
- one object

### Things to try

- Increase detector noise in `Tabletop2D`.
- Change the retry offset schedule.
- Add a second object.
- Compare random retry with the current belief-based retry.

## `02_reactive_grasping.py`

### What this teaches

A robot should not close the gripper on a stale visual pose. It should keep
servoing toward updated observations, detect contact failure, correct its
belief, and retry from the new estimate.

### Run

```bash
python examples/manipulation/02_reactive_grasping.py
```

### Key loop

```text
observe moving object -> update belief -> servo gripper -> miss -> correct bias -> servo again -> grasp
```

### Simplifications

- 2D point gripper
- one moving object
- fake visual detector with a scripted calibration bias
- contact is reduced to distance threshold
- no arm kinematics or dynamics

### Things to try

- Increase `detector_noise`.
- Reduce `gripper_speed`.
- Change the close threshold in `ReactiveGraspAgent`.
- Remove the calibration correction and watch the retry keep failing.

## `03_closed_loop_ik.py`

### What this teaches

Inverse kinematics is more useful as a feedback loop than as a one-time solve.
The arm observes a noisy moving target, updates a target belief, takes one
damped Jacobian step, and observes again before choosing the next step.

### Run

```bash
python examples/manipulation/03_closed_loop_ik.py
```

### Key loop

```text
observe target -> update target belief -> Jacobian servo step -> observe new error -> repeat
```

### Simplifications

- 2-link planar arm
- point end-effector
- noisy target observation
- no torque, inertia, or collision
- damped least-squares Jacobian instead of a full motion planner

### Things to try

- Increase `observation_noise`.
- Lower `max_joint_delta` to make tracking lag visible.
- Increase `damping` in `ClosedLoopIKAgent`.
- Change the target path in `target_position()`.

## `04_moving_target_reaching.py`

### What this teaches

Reaching a moving target is not chasing the last visible pose. The arm observes
the target, estimates velocity, predicts through short occlusions, and corrects
again when observations return.

### Run

```bash
python examples/manipulation/04_moving_target_reaching.py
```

### Key loop

```text
observe target -> estimate velocity -> predict through occlusion -> servo -> observe again -> correct
```

### Simplifications

- 2-link planar arm
- point target and point end-effector
- rectangular occluder
- constant-velocity belief
- distance threshold as contact

### Things to try

- Move the occluder in `MovingTargetReachWorld`.
- Increase `lookahead_steps`.
- Increase `observation_noise`.
- Disable prediction during occlusion and compare the reach time.

## `05_object_search_and_pick.py`

### What this teaches

Picking starts before the gripper moves. The robot must search viewpoints,
remember what it has seen, choose the requested object instead of distractors,
and recover when a low-confidence pose causes a grasp miss.

### Run

```bash
python examples/manipulation/05_object_search_and_pick.py
```

### Key loop

```text
look from viewpoint -> update object memory -> choose target -> pick -> miss -> move camera -> retry
```

### Simplifications

- 2D tabletop
- scripted viewpoint-dependent visibility
- fake object detector
- object memory is a small dictionary
- grasp success is a confidence and distance check

### Things to try

- Change `VIEWPOINTS`.
- Add another distractor object.
- Require two high-confidence detections before picking.
- Try random viewpoint search instead of the fixed search order.

## `06_push_then_grasp.py`

### What this teaches

Sometimes retrying the same grasp is the wrong recovery. The robot should
recognize that the world state blocks grasping, push the object into a better
configuration, observe the changed state, and then grasp.

### Run

```bash
python examples/manipulation/06_push_then_grasp.py
```

### Key loop

```text
observe blocked target -> try grasp -> detect blocked grasp -> push object -> observe changed state -> grasp
```

### Simplifications

- 2D tabletop
- rectangular shelf blocks gripper closure
- point push with scripted displacement
- fake visual detector
- no force, friction, or full contact dynamics

### Things to try

- Change the push direction or distance.
- Make the shelf larger.
- Add a push miss condition.
- Compare "retry pick" with "push then pick".

## `07_probabilistic_suction_sorting.py`

### What this teaches

Suction is not a deterministic pick primitive. The robot should track
per-object success probability, observe suction failure, prepare a better seal,
retry, and continue sorting after the recovery.

### Run

```bash
python examples/manipulation/07_probabilistic_suction_sorting.py
```

### Key loop

```text
observe objects -> choose likely suction target -> pick -> sort to bin -> miss -> update probability -> prepare -> retry
```

### Simplifications

- 2D tabletop
- one suction cup
- color bins
- seeded probabilistic outcomes

### Things to try

- Change the initial success probabilities.
- Add a preparation cost before retrying suction.
- Sort the lowest-confidence object first and compare failures.
- Make one object require two preparation attempts.
- seal preparation is a simple probability boost

### Things to try

- Lower the blue object's `base_success`.
- Sort by lowest success first instead of highest.
- Add a bin placement error.
- Make preparation cost more time or reward.
