"""Ground a language model's plan in affordances: say what helps, do what is possible.

A language model is a good planner and a bad robot. Asked to "wipe the table" it
will confidently propose *pick up the sponge* — the right idea — without knowing
whether the robot is anywhere near the sponge. SayCan (Ahn et al., 2022, "Do As I
Can, Not As I Say") fixes this by scoring every skill twice and multiplying:

    score(skill) = p_LLM(skill furthers the instruction) * p_affordance(skill works now)

The language term ("Say") ranks skills by relevance to the goal; the affordance
term ("Can") is the robot's own estimate that the skill will succeed from the
current state. Their product is high only for a skill that is both useful *and*
executable, so the greedy argmax walks out a feasible plan with no separate
planner — and never commands a skill whose preconditions are unmet.

This example runs the same kitchen task two ways via the ``ground`` flag:

  * ``ground=True``  (SayCan): language x affordance -> go to the sponge, pick it
    up (retrying a slip), carry it to the table, wipe. Goal reached.
  * ``ground=False`` (language only): the argmax of the raw LLM scores commands
    "pick the sponge" while standing at the table, the precondition is unmet, and
    the robot repeats that affordance_violation until it times out. Ungrounded
    language is not executable.

The "LLM" here is a small, transparent stand-in for a language-model call: it
scores skills by relevance to the instruction given the running facts (what is
held, what is done), exactly the history-conditioned query SayCan makes — but it
is deliberately blind to physical preconditions, which is the whole point of
grounding it.

Success: the table is wiped clean.
Failure: affordance_violation (recoverable - a skill was commanded with its
precondition unmet), skill_slip (recoverable - an afforded skill stochastically
missed and is retried), and timeout (terminal).

References:
  * M. Ahn et al., "Do As I Can, Not As I Say: Grounding Language in Robotic
    Affordances," CoRL 2022. arXiv:2204.01691. https://say-can.github.io/
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pir.core.random import make_rng
from pir.core.types import Failure, StepResult, Trace

SKILLS = ("go_to_sponge", "go_to_table", "pick_sponge", "wipe_table", "done")


@dataclass
class KitchenState:
    location: str = "table"  # robot starts at the dirty table, sponge is elsewhere
    holding_sponge: bool = False
    table_clean: bool = False


@dataclass
class Skill:
    """A primitive with a precondition, an affordance (base success), and an effect."""

    name: str
    precondition: Any  # state -> bool
    base_success: float  # p(success) when the precondition is met
    effect: Any = None  # state -> None, applied on success


def _build_skills() -> dict[str, Skill]:
    def at(loc: str):
        return lambda s: s.location == loc

    skills = {
        "go_to_sponge": Skill("go_to_sponge", lambda s: True, 1.0,
                              lambda s: setattr(s, "location", "sponge")),
        "go_to_table": Skill("go_to_table", lambda s: True, 1.0,
                             lambda s: setattr(s, "location", "table")),
        "pick_sponge": Skill("pick_sponge", lambda s: at("sponge")(s) and not s.holding_sponge,
                            0.8, lambda s: setattr(s, "holding_sponge", True)),
        "wipe_table": Skill("wipe_table", lambda s: at("table")(s) and s.holding_sponge,
                           0.85, lambda s: setattr(s, "table_clean", True)),
        "done": Skill("done", lambda s: True, 1.0, None),
    }
    return skills


class KitchenWorld:
    """A two-location kitchen; skills enforce preconditions and may slip."""

    def __init__(self, *, seed: int | None = 0, max_steps: int = 20) -> None:
        self.skills = _build_skills()
        self.max_steps = max_steps
        self.seed = seed
        self.reset(seed=seed)

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            self.seed = seed
        self.rng = make_rng(self.seed)
        self.state = KitchenState()
        self.time = 0
        return self.observe()

    def observe(self) -> dict[str, Any]:
        s = self.state
        return {
            "time": self.time,
            "location": s.location,
            "holding_sponge": s.holding_sponge,
            "table_clean": s.table_clean,
            "affordances": {name: self.affordance(name) for name in SKILLS},
        }

    def affordance(self, skill_name: str) -> float:
        """The robot's estimate that the skill succeeds from the current state.

        High when the precondition holds (the skill's base success rate), near
        zero when it does not. This is the grounding signal SayCan multiplies in.
        """
        skill = self.skills[skill_name]
        return skill.base_success if skill.precondition(self.state) else 0.02

    def step(self, action: dict[str, Any]) -> StepResult:
        self.time += 1
        name = action.get("skill", "done")
        skill = self.skills[name]
        info: dict[str, Any] = {
            "time": self.time,
            "skill": name,
            "affordance": self.affordance(name),
            "success": False,
        }

        if name == "done":
            done = True
            info["success"] = self.state.table_clean
            return StepResult(self.observe(), 1.0 if self.state.table_clean else -0.2, done, info)

        if not skill.precondition(self.state):
            # The commanded skill is not executable here: the failure that
            # grounding is meant to prevent.
            info["failure"] = Failure(
                "affordance_violation", f"{name} precondition unmet in {self.state.location}", True
            )
            done = self.time >= self.max_steps
            if done:
                info["failure"] = Failure("timeout", "ran out of steps", False)
            return StepResult(self.observe(), -0.2, done, info)

        if self.rng.random() < skill.base_success:
            if skill.effect is not None:
                skill.effect(self.state)
            info["success"] = self.state.table_clean
            reward = 1.0 if self.state.table_clean else 0.05
            done = self.state.table_clean or self.time >= self.max_steps
            if not self.state.table_clean and self.time >= self.max_steps:
                info["failure"] = Failure("timeout", "ran out of steps", False)
            return StepResult(self.observe(), reward, done, info)

        # Afforded but stochastically slipped (e.g. the grasp missed): retry next.
        info["failure"] = Failure("skill_slip", f"{name} was afforded but missed", True)
        done = self.time >= self.max_steps
        if done:
            info["failure"] = Failure("timeout", "ran out of steps", False)
        return StepResult(self.observe(), -0.1, done, info)


def language_scores(instruction: str, obs: dict[str, Any]) -> dict[str, float]:
    """A transparent stand-in for an LLM call: p(skill furthers the instruction).

    It conditions on the running facts (held / clean) the way SayCan re-prompts
    the model with the plan so far, and ranks skills by *relevance to the goal* —
    but it never checks physical preconditions (it does not know where the robot
    is standing). That blindness is exactly what the affordance term grounds.
    """
    _ = instruction  # one task here; kept to mirror a real LLM prompt signature
    if obs["table_clean"]:
        scores = {"done": 0.70, "go_to_table": 0.10, "wipe_table": 0.08,
                  "go_to_sponge": 0.06, "pick_sponge": 0.06}
    elif obs["holding_sponge"]:
        # Has the sponge -> the model says "go wipe the table" (relevant, maybe
        # infeasible from here).
        scores = {"wipe_table": 0.45, "go_to_table": 0.30, "done": 0.10,
                  "pick_sponge": 0.08, "go_to_sponge": 0.07}
    else:
        # No sponge yet -> the model says "pick up the sponge" (relevant, and
        # infeasible unless already standing at it).
        scores = {"pick_sponge": 0.45, "go_to_sponge": 0.25, "wipe_table": 0.15,
                  "go_to_table": 0.10, "done": 0.05}
    return {name: scores.get(name, 0.0) for name in SKILLS}


class SayCanAgent:
    """Pick argmax over p_LLM(skill) * p_affordance(skill); drop the affordance to ablate."""

    def __init__(self, instruction: str = "wipe the table", ground: bool = True) -> None:
        self.instruction = instruction
        self.ground = ground

    def reset(self) -> None:
        self.last_scores: dict[str, dict[str, float]] = {}

    def act(self, obs: dict[str, Any]) -> dict[str, Any]:
        llm = language_scores(self.instruction, obs)
        affordance = obs["affordances"]
        if self.ground:
            combined = {name: llm[name] * affordance[name] for name in SKILLS}
        else:
            combined = dict(llm)  # language only: ignore whether the skill is possible
        chosen = max(SKILLS, key=lambda name: combined[name])
        self.last_scores = {"llm": llm, "affordance": affordance, "combined": combined}
        return {"skill": chosen}

    def update(self, obs: dict[str, Any], reward: float, info: dict[str, Any]) -> None:
        name = info.get("skill")
        if name and self.last_scores:
            info["llm_score"] = round(self.last_scores["llm"][name], 4)
            info["combined_score"] = round(self.last_scores["combined"][name], 4)
            info["grounded"] = self.ground


def run(
    seed: int = 0,
    render: bool = True,
    max_steps: int = 20,
    ground: bool = True,
    instruction: str = "wipe the table",
) -> Trace:
    world = KitchenWorld(seed=seed, max_steps=max_steps)
    obs = world.reset(seed=seed)
    agent = SayCanAgent(instruction=instruction, ground=ground)
    agent.reset()
    trace = Trace()

    for _ in range(max_steps):
        action = agent.act(obs)
        result = world.step(action)
        obs, reward, done, info = result.as_tuple()
        agent.update(obs, reward, info)
        trace.append(obs, action, reward, info)

        if render:
            _render(info)

        if done:
            break

    return trace


def _render(info: dict[str, Any]) -> None:
    failure = info.get("failure")
    tag = f" [{failure.kind}]" if failure else ""
    print(
        f"  t={info['time']:2d} skill={info['skill']:<13} "
        f"affordance={info['affordance']:.2f} combined={info.get('combined_score', 0):.3f}{tag}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--instruction", type=str, default="wipe the table")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument(
        "--no-ground", action="store_true", help="language only (no affordance grounding)"
    )
    args = parser.parse_args()

    if not args.no_render:
        print(f'instruction: "{args.instruction}"  (grounded={not args.no_ground})')
    trace = run(
        seed=args.seed,
        render=not args.no_render,
        max_steps=args.max_steps,
        ground=not args.no_ground,
        instruction=args.instruction,
    )
    final = trace.infos[-1]
    failures = sorted({f.kind for f in trace.failures()})
    print(
        f"cleaned={final.get('success', False)} steps={len(trace.actions)} "
        f"failures={failures} grounded={not args.no_ground}"
    )


if __name__ == "__main__":
    main()
