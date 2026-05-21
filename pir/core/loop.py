"""Tiny closed-loop runtime utilities."""

from __future__ import annotations

from typing import Any, Protocol

from pir.core.types import StepResult, Trace


class Env(Protocol):
    def reset(self, seed: int | None = None) -> Any:
        ...

    def step(self, action: Any) -> StepResult | tuple[Any, float, bool, dict[str, Any]]:
        ...


class Agent(Protocol):
    def reset(self) -> None:
        ...

    def act(self, obs: Any) -> Any:
        ...

    def update(self, obs: Any, reward: float, info: dict[str, Any]) -> None:
        ...


def _unpack_step(
    result: StepResult | tuple[Any, float, bool, dict[str, Any]],
) -> tuple[Any, float, bool, dict[str, Any]]:
    if isinstance(result, StepResult):
        return result.as_tuple()
    obs, reward, done, info = result
    return obs, float(reward), bool(done), dict(info)


def run_closed_loop(
    env: Env,
    agent: Agent,
    *,
    seed: int | None = None,
    max_steps: int = 300,
    render: bool = False,
    viewer: Any | None = None,
) -> Trace:
    """Run observe-act-update until done or max_steps."""

    obs = env.reset(seed=seed)
    agent.reset()
    trace = Trace()

    for _ in range(max_steps):
        action = agent.act(obs)
        obs, reward, done, info = _unpack_step(env.step(action))
        agent.update(obs, reward, info)
        trace.append(obs, action, reward, info)

        if render:
            if viewer is not None:
                viewer.render(env, agent, info)
            elif hasattr(env, "render"):
                env.render(agent=agent, info=info)

        if done:
            break

    return trace
