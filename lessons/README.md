# Lessons: a failure-first tour of robot loops

Most robotics tutorials stop at `action = plan(observation)` and assume the
action works. Real robots miss grasps, hit walls they could not see, lose track
of where they are, and misread ambiguous commands. **This course is ten short
loops, in order, about what a robot does _after_ something goes wrong.**

Each lesson is one runnable example you already have in this repo — no new
setup. Read the file top to bottom (every one fits on a screen or two), run it,
and watch the internal state change. Do them in order: each lesson assumes the
one before it.

```bash
python3 -m pip install -e .
```

| # | Lesson | Run | The loop it teaches |
| --: | --- | --- | --- |
| 1 | The bare loop | `python3 examples/runtime/01_sense_act_loop.py` | observe → act → observe |
| 2 | Fail and retry | `python3 examples/manipulation/01_pick_and_retry.py` | grasp miss → belief update → retry |
| 3 | Replan around a hidden wall | `python3 examples/navigation/04_online_replanning_astar.py` | plan → see new obstacle → replan |
| 4 | Recover from a blocked path | `python3 examples/navigation/09_blocked_path_recovery.py` | step into a dead end → back off → replan |
| 5 | Act on a belief, not the truth | `python3 examples/navigation/06_belief_based_navigation.py` | estimate pose → decide under uncertainty |
| 6 | Act to learn (active perception) | `python3 examples/navigation/07_active_slam_toy.py` | pick moves that shrink uncertainty |
| 7 | Stay safe at runtime | `python3 examples/navigation/29_safety_filter_cbf.py` | unsafe command → safety filter → safe motion |
| 8 | Know when you don't know | `python3 examples/manipulation/30_conformal_ask_for_help.py` | ambiguous → ask for help instead of guessing |
| 9 | Ask before acting | `python3 examples/embodied_ai/35_clarifying_question.py "pick the block" --answer red` | ambiguous command → clarify → act |
| 10 | Put it all together | `python3 examples/embodied_ai/36_household_task_agent.py "put the block away" --answer red` | clarify → plan → stay safe → retry → replan |

Most examples accept `--no-render` for a fast headless run, and every example
returns a `Trace` you can inspect (see [`docs/trace.md`](../docs/trace.md)).

---

## Lesson 1 — The bare loop

**Run:** `python3 examples/runtime/01_sense_act_loop.py`
([source](../examples/runtime/01_sense_act_loop.py))

The smallest closed-loop example: a robot observes a noisy state, acts, and
observes again. Nothing fails yet — this is the skeleton every later lesson
hangs on.

- **What it observes:** a noisy estimate of its own state, every step.
- **What changes:** nothing surprising — that's the point. Establish the
  `observe → act → observe` rhythm before failure enters.
- **Concept:** the perception–action loop is the unit of robotics, not a single
  planning call.

→ Next: a loop where the action can fail.

## Lesson 2 — Fail and retry

**Run:** `python3 examples/manipulation/01_pick_and_retry.py`
([source](../examples/manipulation/01_pick_and_retry.py))

A tabletop robot picks an object, **misses**, updates its belief about where the
object really is, and retries with a different grasp. This is the thesis of the
whole repo in one file.

- **What fails:** the grasp, because the perceived object pose is biased.
- **What it observes:** `info["failure"]` reports the miss; each attempt yields a
  fresh, noisy detection.
- **How belief changes:** a running estimate of the object pose is updated after
  every miss, so the next attempt aims somewhere new.
- **Concept:** failure is data. The interesting behaviour is the belief update
  between attempts.

→ Next: the same idea, but the failure is a wall instead of a grasp.

## Lesson 3 — Replan around a hidden wall

**Run:** `python3 examples/navigation/04_online_replanning_astar.py`
([source](../examples/navigation/04_online_replanning_astar.py))

The robot plans a path with A* through a map it only partially knows, drives
into a previously unknown wall, and **replans** from what it just learned.

- **What fails:** the current plan, when an obstacle is observed that the map did
  not contain.
- **What it observes:** new occupied cells as it moves.
- **How belief changes:** the occupancy map is updated, then A* is re-run on the
  new map.
- **Concept:** a plan is a hypothesis. Replanning is how a robot survives an
  imperfect map.

→ Next: what to do when replanning alone isn't enough and you're stuck.

## Lesson 4 — Recover from a blocked path

**Run:** `python3 examples/navigation/09_blocked_path_recovery.py`
([source](../examples/navigation/09_blocked_path_recovery.py))

The robot detects that the path ahead is blocked, **backs off**, marks the
blocked cell, and replans around it — recovery, not just replanning in place.

- **What fails:** forward progress; the robot would otherwise be wedged.
- **What it observes:** a blocked cell directly ahead.
- **How belief changes:** the blocked cell is committed to the map so the
  replan does not re-suggest it.
- **Concept:** recovery is an explicit behaviour — detect, undo, record, retry.

→ Next: stop assuming you even know where you are.

## Lesson 5 — Act on a belief, not the truth

**Run:** `python3 examples/navigation/06_belief_based_navigation.py`
([source](../examples/navigation/06_belief_based_navigation.py))

The robot no longer has direct access to its true pose. It maintains a belief
(a heatmap), estimates where it probably is, and navigates from that estimate.

- **What it observes:** noisy, partial cues about its position.
- **How belief changes:** a belief distribution over poses is updated each step;
  the chosen action depends on the estimate, not the ground truth.
- **Concept:** under partial observability you act on a belief. The true state is
  shown only so you can see how wrong the belief sometimes is.

→ Next: choose actions specifically to make that belief sharper.

## Lesson 6 — Act to learn (active perception)

**Run:** `python3 examples/navigation/07_active_slam_toy.py`
([source](../examples/navigation/07_active_slam_toy.py))

A toy active-SLAM agent picks moves that **reduce** both pose and map
uncertainty — it acts in order to perceive better, not just to reach a goal.

- **What it observes:** measurements whose value depends on where it chooses to go.
- **How belief changes:** moves are scored by expected uncertainty (entropy)
  reduction, so the agent seeks informative actions.
- **Concept:** perception is something you can plan for. Information is a reward.

→ Next: a hard runtime guarantee layered on top of any policy.

## Lesson 7 — Stay safe at runtime

**Run:** `python3 examples/navigation/29_safety_filter_cbf.py`
([source](../examples/navigation/29_safety_filter_cbf.py))

A naive go-to-goal controller produces velocities that would hit obstacles. A
**control-barrier-function** safety filter projects each command onto a safe
set — the policy never even knows the obstacles exist.

- **What fails (and is prevented):** collisions the nominal controller would
  cause.
- **What it observes:** obstacle geometry, used by the filter, not the policy.
- **How behaviour changes:** every command is minimally adjusted to stay safe.
- **Concept:** safety can be a separate runtime layer, independent of how smart
  the policy is.

→ Next: instead of guarding a confident policy, handle an unconfident one.

## Lesson 8 — Know when you don't know

**Run:** `python3 examples/manipulation/30_conformal_ask_for_help.py`
([source](../examples/manipulation/30_conformal_ask_for_help.py))

A sorter calibrates a **conformal prediction set** offline. At runtime it acts
only when the set is a single confident label, and **asks a toy oracle for help**
when the set is ambiguous.

- **What fails (and is avoided):** confident-but-wrong actions on ambiguous input.
- **What it observes:** a prediction set whose size reflects uncertainty.
- **How behaviour changes:** singleton → act; ambiguous → ask.
- **Concept:** calibrated uncertainty turns "I'm not sure" into a concrete
  request for help.

→ Next: the help request becomes a natural-language question.

## Lesson 9 — Ask before acting

**Run:** `python3 examples/embodied_ai/35_clarifying_question.py "pick the block" --answer red`
([source](../examples/embodied_ai/35_clarifying_question.py))

Given the ambiguous command *"pick the block"* with several blocks present, the
robot **asks which one**, takes the answer, resolves the goal, and acts.

- **What fails (and is avoided):** acting on an under-specified command.
- **What it observes:** multiple candidate objects matching the command.
- **How behaviour changes:** detect ambiguity → ask → bind the answer → act.
- **Concept:** language is just another noisy observation; clarification is a
  belief update driven by a human.

→ Next: combine clarify, plan, safety, retry, and replan in one agent.

## Lesson 10 — Put it all together

**Run:** `python3 examples/embodied_ai/36_household_task_agent.py "put the block away" --answer red`
([source](../examples/embodied_ai/36_household_task_agent.py))

The capstone. A household robot **clarifies** an ambiguous command, **plans**
through a room, **rejects an unsafe step**, **retries** a missed grasp, accepts a
**human correction** and **replans**, then stores the block.

- **What fails:** ambiguity, an unsafe floor step, and a grasp miss — all in one
  run.
- **How behaviour changes:** every earlier lesson shows up as one stage of this
  pipeline.
- **Concept:** "embodied intelligence" here is not one big model — it is these
  small failure-aware loops, composed.

→ Done. From here, branch into the full
[example index](../examples/README.md) or the deeper
[learning paths](../docs/learning_paths.md).

---

Found a failure mode these lessons don't cover yet? That's a great
contribution — see [`CONTRIBUTING.md`](../CONTRIBUTING.md). If this course helped
you learn or teach, a GitHub star helps other people find it.
