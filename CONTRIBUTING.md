# Contributing

PythonInteractiveRobotics is an educational repository. A good contribution is
small enough to read, fast enough to run, and explicit about the robot loop it
teaches.

The goal is not a general robotics framework. The goal is a set of clear,
failure-aware examples that teach interaction:

```text
observe -> update memory / belief -> act -> observe failure -> recover -> retry
```

See `docs/example_authoring.md` for the detailed example template.

## Contribution Scope

Good first contributions:

- improve an existing example without changing its teaching goal
- add a small missing failure case
- improve a visualization so belief, memory, or failure is easier to see
- add or tighten a smoke test
- improve docs that explain what is simplified or fake

Larger contributions should still keep the first run lightweight. Heavy
simulators, ROS2 bridges, learned policies, and large assets belong behind
optional extras and should not change the core experience.

## Example Checklist

Every example PR should satisfy:

- [ ] The example has a clear closed-loop interaction.
- [ ] The first visualization appears within 5 seconds on a normal laptop.
- [ ] It runs with core dependencies, or optional dependencies are documented.
- [ ] It includes at least one of failure, uncertainty, retry, partial observation, memory, or replanning.
- [ ] Failures are exposed through `info["failure"]` with a `Failure` object.
- [ ] It has a `run(...)` function that returns `Trace`.
- [ ] It has a script entry point with a `--no-render` mode.
- [ ] It has a focused smoke test in `tests/test_examples_smoke.py`.
- [ ] The relevant `examples/<category>/README.md` section is updated.
- [ ] `examples/README.md` is updated if the example is user-facing.
- [ ] `scripts/make_gifs.py` and `README.md` are updated if the example should appear in the GIF gallery.
- [ ] It states what is simplified, fake, or intentionally unrealistic.
- [ ] It does not require large assets, Docker, GPU, ROS, or heavy simulators for the core path.

## Naming and Placement

Use the smallest category that matches the teaching loop:

- `examples/runtime/` for the smallest loop patterns
- `examples/navigation/` for grid or driving-style navigation loops
- `examples/manipulation/` for grasping, reaching, pushing, sorting, and arm loops
- `examples/embodied_ai/` for goal-conditioned interaction, memory, language, and semantic search
- `examples/world_models/` for action-conditioned dynamics and planning with learned models

Name files with a stable numeric prefix when they are part of the main learning
path, for example:

```text
examples/manipulation/05_object_search_and_pick.py
examples/world_models/20_tiny_world_model_planning.py
```

Update `docs/example_roadmap.md` when adding or renaming a learning-path
example.

## Verification

Install contributor dependencies first:

```bash
pip install -e ".[dev]"
```

Run the fast smoke suite before opening a PR:

```bash
python scripts/run_all_smoke_tests.py
```

For changes that affect README visuals, regenerate and check GIFs:

```bash
python scripts/run_all_smoke_tests.py --gifs --check-gifs
```

If you add a new GIF, it should be generated from the runnable example, not
from a separate hand-made animation.

## Code Style

Prefer:

- readable over clever
- explicit loops over hidden pipelines
- dataclasses and functions over inheritance-heavy designs
- local example code before shared abstractions
- small deterministic seeds for repeatable failures

Avoid:

- YAML-heavy configuration
- plugin systems before there are multiple real backends
- adding shared code to `pir/` for a single example
- global framework abstractions that hide the loop being taught

Move shared code into `pir/` only after the same idea naturally appears in at
least three examples.

## Review Principles

Reviewers should ask:

- Is the loop easy to understand by reading top to bottom?
- Is the robot's observation visible?
- Is the robot's memory, belief, plan, or state visible?
- Does an action change the next observation?
- Is failure visible and recoverable when the concept calls for it?
- Can unnecessary abstraction be removed?
- Does it run quickly in headless mode?

Readable teaching examples are more valuable here than impressive but opaque
implementations.
