"""Conformal prediction as a runtime "ask for help" signal.

A toy sorter receives items one at a time. A noisy perception model emits
a softmax-like score over two classes. Naively committing to argmax on
every item silently fails on ambiguous items. Conformal prediction turns
the model's score into a *prediction set* whose marginal coverage is
calibrated against a small offline holdout. The agent's online rule is:

    if the prediction set is a singleton, place the item directly,
    otherwise ask a (toy) human oracle for the true label and then place.

Why this matters: the conformal calibration provides a distribution-free
coverage guarantee against the holdout distribution, so the agent's
*frequency of help requests* is tied to its *actual model error*, not to
an ad hoc confidence threshold the engineer picks by feel.

This example is intentionally structurally close to
`08_belief_grasp_selection.py` and `09_active_viewpoint_for_grasp.py`,
which both quantify uncertainty before acting. The difference is that
those examples reduce uncertainty by *acting in the world* (another
grasp attempt, another viewpoint), while this example reduces
uncertainty by *deferring to a human*.

Success: every item placed and zero wrong sorts on the test stream.
Failure: wrong_sort (recoverable - the agent committed without asking
and got the wrong bin), timeout (terminal - did not finish the stream
before max_steps).
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

from pir.core.types import Failure, StepResult, Trace


CLASS_NAMES = ("A", "B")


@dataclass(frozen=True)
class ConformalConfig:
    alpha: float = 0.10
    calibration_size: int = 30
    test_size: int = 16
    clear_min: float = 0.80
    clear_max: float = 0.95
    ambiguous_min: float = 0.45
    ambiguous_max: float = 0.58
    ambiguous_ratio: float = 0.25


@dataclass(frozen=True)
class Item:
    item_id: int
    true_label: int
    scores: tuple[float, float]


def _sample_item(
    item_id: int,
    true_label: int,
    ambiguous: bool,
    cfg: ConformalConfig,
    rng: np.random.Generator,
) -> Item:
    if ambiguous:
        s_true = float(rng.uniform(cfg.ambiguous_min, cfg.ambiguous_max))
    else:
        s_true = float(rng.uniform(cfg.clear_min, cfg.clear_max))
    s_other = 1.0 - s_true
    scores = [0.0, 0.0]
    scores[true_label] = s_true
    scores[1 - true_label] = s_other
    return Item(item_id=item_id, true_label=true_label, scores=(scores[0], scores[1]))


def _generate_stream(
    *,
    count: int,
    seed: int,
    cfg: ConformalConfig,
    ambiguous_ratio: float | None = None,
) -> tuple[Item, ...]:
    rng = np.random.default_rng(seed)
    ratio = cfg.ambiguous_ratio if ambiguous_ratio is None else ambiguous_ratio
    n_ambiguous = int(round(count * ratio))
    ambiguous_flags = np.zeros(count, dtype=bool)
    ambiguous_flags[:n_ambiguous] = True
    rng.shuffle(ambiguous_flags)
    items: list[Item] = []
    for index, is_ambiguous in enumerate(ambiguous_flags):
        true_label = int(rng.integers(0, 2))
        items.append(_sample_item(index, true_label, bool(is_ambiguous), cfg, rng))
    return tuple(items)


def conformal_calibrate(
    items: tuple[Item, ...],
    *,
    alpha: float,
) -> float:
    """Return the conformal threshold q_hat such that prediction sets
    `{c : score_c >= 1 - q_hat}` cover the true label with marginal
    probability `>= 1 - alpha` on the holdout distribution.
    """

    nonconformity = np.array(
        [1.0 - item.scores[item.true_label] for item in items],
        dtype=float,
    )
    n = len(nonconformity)
    q_level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
    return float(np.quantile(nonconformity, q_level, method="higher"))


def prediction_set(scores: tuple[float, float], q_hat: float) -> tuple[int, ...]:
    threshold = 1.0 - q_hat
    return tuple(c for c, s in enumerate(scores) if s >= threshold)


class ConformalSortingWorld:
    """Conveyor that delivers a fixed stream of items to be sorted into 2 bins."""

    def __init__(
        self,
        *,
        seed: int = 0,
        max_steps: int = 40,
        config: ConformalConfig | None = None,
    ) -> None:
        self.config = config or ConformalConfig()
        self.seed = seed
        self.max_steps = max_steps
        self.calibration_items = _generate_stream(
            count=self.config.calibration_size,
            seed=seed + 1000,
            cfg=self.config,
        )
        self.test_items = _generate_stream(
            count=self.config.test_size,
            seed=seed + 2000,
            cfg=self.config,
        )
        self.current_index = 0
        self.step_count = 0
        self.awaiting_place = False
        self.last_oracle: int | None = None
        self.placed: list[tuple[int, int, int]] = []
        self._fig: Any | None = None
        self._ax: Any | None = None

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            self.seed = seed
            self.calibration_items = _generate_stream(
                count=self.config.calibration_size,
                seed=seed + 1000,
                cfg=self.config,
            )
            self.test_items = _generate_stream(
                count=self.config.test_size,
                seed=seed + 2000,
                cfg=self.config,
            )
        self.current_index = 0
        self.step_count = 0
        self.awaiting_place = False
        self.last_oracle = None
        self.placed = []
        return self.observe()

    @property
    def remaining(self) -> int:
        return max(0, len(self.test_items) - self.current_index)

    def _current_item(self) -> Item | None:
        if self.current_index >= len(self.test_items):
            return None
        return self.test_items[self.current_index]

    def observe(self) -> dict[str, Any]:
        item = self._current_item()
        if item is None:
            return {
                "item_id": -1,
                "scores": (0.0, 0.0),
                "remaining": 0,
                "oracle_label": None,
                "awaiting_place": False,
            }
        return {
            "item_id": item.item_id,
            "scores": item.scores,
            "remaining": self.remaining,
            "oracle_label": self.last_oracle if self.awaiting_place else None,
            "awaiting_place": self.awaiting_place,
        }

    def step(self, action: int) -> StepResult:
        """Action: 0 = place class A, 1 = place class B, 2 = ask oracle."""

        self.step_count += 1
        item = self._current_item()
        info: dict[str, Any] = {
            "action": action,
            "remaining": self.remaining,
        }
        if item is None:
            info["success"] = True
            info["failure"] = Failure(
                kind="invalid_action",
                message="No more items; the agent should have terminated.",
                recoverable=False,
            )
            return StepResult(self.observe(), 0.0, True, info)

        if action == 2:
            if self.awaiting_place:
                info["failure"] = Failure(
                    kind="invalid_action",
                    message="Oracle already revealed the label; the agent must place.",
                    recoverable=True,
                )
                return StepResult(self.observe(), -0.5, False, info)
            self.last_oracle = item.true_label
            self.awaiting_place = True
            info["asked"] = True
            info["oracle_label"] = item.true_label
            return StepResult(self.observe(), -0.2, False, info)

        if action not in (0, 1):
            info["failure"] = Failure(
                kind="invalid_action",
                message=f"Unknown action {action!r}; expected 0, 1, or 2.",
                recoverable=True,
            )
            return StepResult(self.observe(), -0.5, False, info)

        used_help = self.awaiting_place
        chosen = int(action)
        correct = chosen == item.true_label
        self.placed.append((item.item_id, chosen, item.true_label))
        info["placed"] = chosen
        info["true_label"] = item.true_label
        info["used_help"] = used_help
        info["correct"] = correct
        if not correct:
            info["failure"] = Failure(
                kind="wrong_sort",
                message=(
                    f"Item {item.item_id} placed in bin {CLASS_NAMES[chosen]} "
                    f"but true class was {CLASS_NAMES[item.true_label]}."
                ),
                recoverable=True,
            )
        self.current_index += 1
        self.awaiting_place = False
        self.last_oracle = None

        finished = self.current_index >= len(self.test_items)
        timed_out = self.step_count >= self.max_steps and not finished
        if finished:
            info["success"] = all(p[1] == p[2] for p in self.placed)
        elif timed_out:
            info["failure"] = Failure(
                kind="timeout",
                message="Did not finish the stream before max_steps.",
                recoverable=False,
            )

        reward = 1.0 if correct else -1.0
        done = finished or timed_out
        return StepResult(self.observe(), reward, done, info)

    def render(self, agent: "ConformalAskForHelpAgent", info: dict[str, Any]) -> None:
        import matplotlib.pyplot as plt

        if self._fig is None or self._ax is None:
            plt.ion()
            self._fig, self._ax = plt.subplots(figsize=(6.4, 4.6))
        ax = self._ax
        ax.clear()
        _draw_scene(ax, self, agent, info)
        self._fig.canvas.draw_idle()
        plt.pause(0.001)


def _draw_scene(
    ax: Any,
    env: ConformalSortingWorld,
    agent: "ConformalAskForHelpAgent",
    info: dict[str, Any],
) -> None:
    import matplotlib.patches as mpatches

    ax.set_xlim(-0.2, 4.8)
    ax.set_ylim(-0.4, 2.4)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.set_title("Conformal ask-for-help: prediction set vs threshold")

    ax.add_patch(mpatches.Rectangle((0.6, 0.2), 0.5, 0.5, color="tab:red", alpha=0.25))
    ax.text(0.85, 0.05, "bin A", ha="center", fontsize=9, color="tab:red")
    ax.add_patch(mpatches.Rectangle((1.5, 0.2), 0.5, 0.5, color="tab:blue", alpha=0.25))
    ax.text(1.75, 0.05, "bin B", ha="center", fontsize=9, color="tab:blue")

    if env.current_index < len(env.test_items):
        item = env.test_items[env.current_index]
        color = "tab:red" if item.true_label == 0 else "tab:blue"
        ax.add_patch(mpatches.Circle((1.3, 1.6), 0.16, color=color, alpha=0.8))
        ax.text(1.3, 1.95, f"item {item.item_id}", ha="center", fontsize=9)

    bar_x = 2.6
    bar_w = 0.35
    threshold = 1.0 - agent.q_hat
    ax.plot([bar_x - 0.1, bar_x + 2 * bar_w + 0.1], [threshold, threshold], "--", color="black", alpha=0.6)
    ax.text(bar_x + 2 * bar_w + 0.15, threshold, f"1 - q̂ = {threshold:.2f}", va="center", fontsize=8)
    scores = info.get("scores")
    if scores is None and env.current_index < len(env.test_items):
        scores = env.test_items[env.current_index].scores
    if scores is not None:
        for idx, score in enumerate(scores):
            color = "tab:red" if idx == 0 else "tab:blue"
            ax.add_patch(
                mpatches.Rectangle(
                    (bar_x + idx * bar_w, 0.0),
                    bar_w * 0.85,
                    score,
                    color=color,
                    alpha=0.7,
                )
            )
            ax.text(
                bar_x + idx * bar_w + bar_w * 0.42,
                score + 0.04,
                f"{score:.2f}",
                ha="center",
                fontsize=8,
            )
            ax.text(
                bar_x + idx * bar_w + bar_w * 0.42,
                -0.15,
                CLASS_NAMES[idx],
                ha="center",
                fontsize=9,
            )

    status = (
        f"step={env.step_count}  state={agent.state}\n"
        f"sorted={agent.sorted_count}/{len(env.test_items)}  "
        f"help={agent.help_request_count}  "
        f"wrong={agent.wrong_sort_count}"
    )
    pred_set = info.get("prediction_set")
    if pred_set:
        status += f"\nset={tuple(CLASS_NAMES[c] for c in pred_set)}"
    if info.get("asked"):
        status += "  (asked oracle)"
    if "failure" in info:
        status += f"\nfailure={info['failure'].kind}"
    ax.text(0.05, 2.25, status, va="top", fontsize=9, family="monospace")


class ConformalAskForHelpAgent:
    """Place items per conformal prediction set; ask the oracle when ambiguous."""

    def __init__(self, env: ConformalSortingWorld) -> None:
        self.config = env.config
        self.q_hat = conformal_calibrate(
            env.calibration_items, alpha=self.config.alpha
        )
        self.reset()

    def reset(self) -> None:
        self.state = "calibrated"
        self.sorted_count = 0
        self.help_request_count = 0
        self.correct_no_help_count = 0
        self.correct_after_help_count = 0
        self.wrong_sort_count = 0
        self.coverage_violation_count = 0
        self.last_prediction_set: tuple[int, ...] = ()
        self.last_used_help: bool = False

    def act(self, obs: dict[str, Any]) -> int:
        if obs.get("awaiting_place"):
            self.state = "place_with_help"
            label = int(obs.get("oracle_label", 0))
            return label
        scores = obs.get("scores", (0.0, 0.0))
        pred_set = prediction_set(scores, self.q_hat)
        self.last_prediction_set = pred_set
        if len(pred_set) == 1:
            self.state = "place"
            return int(pred_set[0])
        self.state = "ask"
        return 2

    def update(
        self, obs: dict[str, Any], reward: float, info: dict[str, Any]
    ) -> None:
        del obs, reward
        info["prediction_set"] = self.last_prediction_set
        info["q_hat"] = float(self.q_hat)
        if info.get("asked"):
            self.help_request_count += 1
        if "placed" not in info:
            return
        used_help = bool(info.get("used_help"))
        correct = bool(info.get("correct"))
        true_label = int(info.get("true_label", -1))
        self.sorted_count += 1
        if correct and not used_help:
            self.correct_no_help_count += 1
        elif correct and used_help:
            self.correct_after_help_count += 1
        if not correct:
            self.wrong_sort_count += 1
            if true_label not in self.last_prediction_set:
                self.coverage_violation_count += 1
        info["agent_state"] = self.state

    def info(self) -> dict[str, Any]:
        return {
            "agent_state": self.state,
            "q_hat": float(self.q_hat),
            "sorted_count": int(self.sorted_count),
            "help_request_count": int(self.help_request_count),
            "correct_no_help_count": int(self.correct_no_help_count),
            "correct_after_help_count": int(self.correct_after_help_count),
            "wrong_sort_count": int(self.wrong_sort_count),
            "coverage_violation_count": int(self.coverage_violation_count),
            "prediction_set": self.last_prediction_set,
        }


def run(seed: int = 0, render: bool = True, max_steps: int = 40) -> Trace:
    env = ConformalSortingWorld(seed=seed, max_steps=max_steps)
    agent = ConformalAskForHelpAgent(env)
    obs = env.reset(seed=seed)
    agent.reset()
    trace = Trace()

    for _ in range(max_steps):
        action = agent.act(obs)
        obs, reward, done, info = env.step(action).as_tuple()
        agent.update(obs, reward, info)
        info.update(agent.info())
        trace.append(obs, action, reward, info)
        if render:
            env.render(agent=agent, info=info)
        if done:
            break

    return trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    trace = run(seed=args.seed, render=not args.no_render, max_steps=args.max_steps)
    final_info = trace.infos[-1] if trace.infos else {}
    failures = [failure.kind for failure in trace.failures()]
    print(
        f"success={final_info.get('success', False)} "
        f"steps={len(trace.actions)} "
        f"q_hat={final_info.get('q_hat', 0.0):.3f} "
        f"sorted={final_info.get('sorted_count', 0)} "
        f"help={final_info.get('help_request_count', 0)} "
        f"wrong={final_info.get('wrong_sort_count', 0)} "
        f"coverage_violations={final_info.get('coverage_violation_count', 0)} "
        f"failures={failures}"
    )

    if not args.no_render:
        import matplotlib.pyplot as plt

        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
