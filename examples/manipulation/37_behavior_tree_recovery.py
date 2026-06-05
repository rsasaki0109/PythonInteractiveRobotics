"""Drive a tabletop pick with a reactive behavior tree that recovers on failure.

`01_pick_and_retry.py` recovers from a grasp miss with imperative `if/else`
control flow. This example does the *same* tabletop task on the *same* world
(`Tabletop2D`), but the recovery is expressed **declaratively** as a reactive
behavior tree (BT) — the structure roboticists actually reach for when a robot
has to keep retrying and re-perceiving until it succeeds.

The tree is ticked once per control step. Each tick walks the tree and yields
exactly one environment action (the leaf that is `RUNNING`):

    Fallback "pick the block"
    ├── Condition  object_in_gripper?            # already holding -> whole tree SUCCESS
    └── Fallback   "grasp or recover"
        ├── Sequence "confident grasp"
        │   ├── Condition  belief_confident?      # localized and uncertainty small?
        │   └── Action     grasp_at_belief        # pick(belief_mean)
        └── Action     relook_to_refine           # recovery: move to a new viewpoint

A Fallback (a.k.a. Selector) ticks its children left to right and stops at the
first that does not FAIL, so it reads as "try the primary thing; if it isn't
possible, do the recovery." The single `relook_to_refine` recovery leaf covers
*both* failure modes the world throws:

  * **occlusion** — the object starts behind an occluder, so `belief_confident?`
    fails (nothing localized yet) and the tree falls through to `relook`, which
    scans a fresh viewpoint until the detector sees the block.
  * **grasp miss** — a missed grasp grows the belief radius back above the
    confidence threshold, so on the next tick `belief_confident?` fails again and
    the tree falls back to `relook` to gather a better view before re-grasping.

That is the lesson: a grasp miss is not handled by a special-case branch; it
simply makes a precondition false, and the *same* declarative fallback re-runs
active perception before retrying. Recovery is a property of the tree's shape,
not of a hand-written recovery routine.

Success: the block is lifted (`gripper.holding` set) before max_steps.
Failure: grasp_miss (recoverable until the world's attempt budget is spent,
then terminal) and tree_exhausted (terminal - ran out of steps still empty).

References:
  * M. Colledanchise and P. Ogren, "Behavior Trees in Robotics and AI: An
    Introduction," CRC Press, 2018. arXiv:1709.00084.
  * BehaviorTree.CPP (Faconti, Colledanchise) https://www.behaviortree.dev/ and
    py_trees (Stonier) https://github.com/splintered-reality/py_trees - the
    Sequence/Fallback tick semantics mirrored here, also used by the ROS 2
    Nav2 BT Navigator for reactive recovery.
"""

from __future__ import annotations

import argparse
import sys
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pir.core.types import Failure, Trace
from pir.worlds.tabletop_2d import Tabletop2D


# --- A minimal reactive behavior tree --------------------------------------
#
# Three return statuses and two composites (Sequence, Fallback) plus leaf
# Condition/Action nodes are enough to express the whole recovery policy. This
# mirrors the core of py_trees / BehaviorTree.CPP, kept small enough to read.


class Status(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    RUNNING = "running"


class Node:
    """Base class: a node is ticked and returns a Status."""

    name = "node"

    def tick(self, bb: "Blackboard") -> Status:  # pragma: no cover - overridden
        raise NotImplementedError


class Sequence(Node):
    """Tick children in order; FAIL/RUNNING short-circuits, all SUCCESS -> SUCCESS.

    Reactive: re-ticked from the first child every control step, so a condition
    that flips to FAILURE immediately re-routes the tree.
    """

    def __init__(self, name: str, children: list[Node]) -> None:
        self.name = name
        self.children = children

    def tick(self, bb: "Blackboard") -> Status:
        for child in self.children:
            status = child.tick(bb)
            if status is not Status.SUCCESS:
                return status
        return Status.SUCCESS


class Fallback(Node):
    """Tick children in order; first non-FAILURE wins, all FAILURE -> FAILURE.

    This is where recovery lives: the primary branch comes first, the recovery
    branch comes after, and the tree falls through to recovery only when the
    primary branch reports FAILURE.
    """

    def __init__(self, name: str, children: list[Node]) -> None:
        self.name = name
        self.children = children

    def tick(self, bb: "Blackboard") -> Status:
        for child in self.children:
            status = child.tick(bb)
            if status is not Status.FAILURE:
                return status
        return Status.FAILURE


class Condition(Node):
    """A leaf that reads the blackboard and returns SUCCESS or FAILURE at once."""

    def __init__(self, name: str, predicate: Callable[["Blackboard"], bool]) -> None:
        self.name = name
        self.predicate = predicate

    def tick(self, bb: "Blackboard") -> Status:
        return Status.SUCCESS if self.predicate(bb) else Status.FAILURE


class Action(Node):
    """A leaf that proposes one environment action and reports RUNNING.

    The environment executes the proposed action after the tick, so a leaf is
    "running" for exactly one control step; the outcome is observed on the next
    tick through the conditions above it.
    """

    def __init__(self, name: str, propose: Callable[["Blackboard"], dict[str, Any]]) -> None:
        self.name = name
        self.propose = propose

    def tick(self, bb: "Blackboard") -> Status:
        bb.action = self.propose(bb)
        bb.active_leaf = self.name
        return Status.RUNNING


class Blackboard:
    """Shared memory the tree reads and writes (py_trees-style)."""

    def __init__(self) -> None:
        self.obs: dict[str, Any] = {}
        self.action: dict[str, Any] | None = None
        self.active_leaf: str = ""


# --- The agent: belief tracking + the tree that decides what to do ----------


class BehaviorTreeAgent:
    """Tracks a spatial belief and lets a reactive BT choose look vs. grasp."""

    confidence_radius = 0.085

    def __init__(self) -> None:
        self.viewpoints = [
            np.array([0.84, 0.52]),  # right of the occluder -> object visible
            np.array([0.78, 0.22]),
            np.array([0.20, 0.84]),  # left but above the occluder -> still visible
        ]
        self.tree = self._build_tree()
        self.reset()

    def reset(self) -> None:
        self.belief_mean: np.ndarray | None = None
        self.belief_radius = 0.14
        self.look_count = 0
        self.retry_count = 0
        self._last_integrated_time: int | None = None

    # The tree is pure structure; all state lives on the agent / blackboard.
    def _build_tree(self) -> Node:
        confident_grasp = Sequence(
            "confident_grasp",
            [
                Condition("belief_confident?", self._belief_confident),
                Action("grasp_at_belief", self._grasp_at_belief),
            ],
        )
        grasp_or_recover = Fallback(
            "grasp_or_recover",
            [confident_grasp, Action("relook_to_refine", self._relook_to_refine)],
        )
        return Fallback(
            "pick_the_block",
            [Condition("object_in_gripper?", self._object_in_gripper), grasp_or_recover],
        )

    # --- conditions ---
    def _object_in_gripper(self, bb: Blackboard) -> bool:
        return (bb.obs.get("gripper") or {}).get("holding") is not None

    def _belief_confident(self, bb: Blackboard) -> bool:
        return self.belief_mean is not None and self.belief_radius <= self.confidence_radius

    # --- actions ---
    def _grasp_at_belief(self, bb: Blackboard) -> dict[str, Any]:
        assert self.belief_mean is not None
        return {"type": "pick", "position": np.clip(self.belief_mean, 0.0, 1.0)}

    def _relook_to_refine(self, bb: Blackboard) -> dict[str, Any]:
        target = self.viewpoints[self.look_count % len(self.viewpoints)]
        self.look_count += 1
        return {"type": "look", "target": target}

    # --- closed-loop hooks ---
    def act(self, obs: dict[str, Any]) -> dict[str, Any]:
        self._integrate_observation(obs)
        self.last_bb = Blackboard()
        self.last_bb.obs = obs
        status = self.tree.tick(self.last_bb)
        self.last_status = status
        # The root only reports SUCCESS once the block is held; until then a leaf
        # action is always RUNNING, so bb.action is set. Guard defensively.
        if self.last_bb.action is None:
            return {"type": "noop"}
        return self.last_bb.action

    def update(self, obs: dict[str, Any], reward: float, info: dict[str, Any]) -> None:
        self._integrate_observation(obs)

        failure = info.get("failure")
        if isinstance(failure, Failure) and failure.kind == "grasp_miss":
            # A miss does not trigger a special branch: it just grows uncertainty
            # back above the confidence threshold, so `belief_confident?` fails and
            # the Fallback re-runs `relook_to_refine` before the next grasp.
            self.retry_count += 1
            self.belief_radius = max(self.belief_radius, self.confidence_radius) + 0.04
            info["retry_count"] = self.retry_count
        elif info.get("success"):
            self.belief_radius = 0.025
            # Confirm the tree now short-circuits at `object_in_gripper?`, so the
            # root reports SUCCESS rather than the grasp leaf's RUNNING.
            confirm = Blackboard()
            confirm.obs = obs
            self.last_status = self.tree.tick(confirm)

        info["bt_status"] = getattr(self, "last_status", Status.RUNNING).value
        info["bt_leaf"] = getattr(self, "last_bb", Blackboard()).active_leaf
        info["belief_mean"] = None if self.belief_mean is None else self.belief_mean.copy()
        info["belief_radius"] = float(self.belief_radius)
        info["retry_count"] = int(self.retry_count)

    def _integrate_observation(self, obs: dict[str, Any]) -> None:
        obs_time = int(obs.get("time", -1))
        if obs_time == self._last_integrated_time:
            return
        self._last_integrated_time = obs_time

        detections = obs.get("detections", [])
        if not detections:
            return

        position = np.asarray(detections[0]["position"], dtype=float)
        confidence = float(detections[0].get("confidence", 0.5))
        if self.belief_mean is None:
            self.belief_mean = position.copy()
        else:
            alpha = float(np.clip(0.35 + 0.45 * confidence, 0.35, 0.80))
            self.belief_mean = alpha * self.belief_mean + (1.0 - alpha) * position
        self.belief_radius = max(0.035, self.belief_radius * 0.72)


def run(seed: int = 3, render: bool = True, max_steps: int = 40) -> Trace:
    env = Tabletop2D(seed=seed)
    obs = env.reset(seed=seed)
    agent = BehaviorTreeAgent()
    agent.reset()
    trace = Trace()

    for _ in range(max_steps):
        action = agent.act(obs)
        result = env.step(action)
        obs, reward, done, info = result.as_tuple()
        agent.update(obs, reward, info)
        trace.append(obs, action, reward, info)

        if render:
            env.render(agent=agent, info=info)

        if done:
            break

    if not (trace.infos and trace.infos[-1].get("success")):
        trace.infos[-1]["failure"] = trace.infos[-1].get("failure") or Failure(
            "tree_exhausted", "ran out of steps without lifting the object", False
        )

    return trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    trace = run(seed=args.seed, render=not args.no_render, max_steps=args.max_steps)
    picked = bool(trace.infos and trace.infos[-1].get("success"))
    leaves = [info.get("bt_leaf", "") for info in trace.infos]
    print(
        f"picked={picked} steps={len(trace.actions)} "
        f"retries={trace.summary().retry_count} "
        f"relooks={leaves.count('relook_to_refine')} grasps={leaves.count('grasp_at_belief')}"
    )

    if not args.no_render:
        import matplotlib.pyplot as plt

        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
