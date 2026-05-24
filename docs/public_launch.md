# Public Launch Checklist

This checklist tracks the work that makes PythonInteractiveRobotics easier to
discover, try, share, and contribute to. It is intentionally separate from
`docs/plan.md`, which is about implementation work.

## Current Baseline

- Repository: `https://github.com/rsasaki0109/PythonInteractiveRobotics`
- Gallery: `https://rsasaki0109.github.io/PythonInteractiveRobotics/`
- Public status: public GitHub repository
- Current package name: `python-interactive-robotics`
- Core promise: closed-loop robotics examples with `numpy + matplotlib`
- Core constraints: no ROS, Docker, GPU, heavy simulator, or neural framework
  in the first-run path
- Verification command:

```bash
python3 scripts/run_all_smoke_tests.py --check-gifs
```

## Launch Positioning

Use this short description in public posts and directory listings:

> Minimal Python robotics examples for planning, manipulation, active
> perception, and embodied AI. No ROS, GPU, or heavy simulator.

Use this longer description when space allows:

> PythonInteractiveRobotics teaches the robot loop in small, runnable Python:
> observe, act, fail, update belief or memory, retry, and replan. The examples
> cover navigation, manipulation, active perception, embodied AI, tiny world
> models, and runtime safety with only `numpy` and `matplotlib` in core.

## Release Readiness

- [x] Add GitHub issue templates for docs, examples, and bugs.
- [x] Seed public roadmap issues for the next launch tasks.
- [ ] Create a `v0.1.0` GitHub release after CI passes on `main`
      ([#1](https://github.com/rsasaki0109/PythonInteractiveRobotics/issues/1)).
- [ ] Publish `python-interactive-robotics` to PyPI
      ([#2](https://github.com/rsasaki0109/PythonInteractiveRobotics/issues/2)).
- [ ] Replace editable-install-first messaging with PyPI install messaging
      after the package is live.
- [x] Add a short GitHub Pages gallery for the strongest GIFs
      ([#3](https://github.com/rsasaki0109/PythonInteractiveRobotics/issues/3)).
- [ ] Enable GitHub Pages with the GitHub Actions source after the deployment
      workflow is on `main`.
- [ ] Add three Colab notebooks:
      `pick_and_retry`, `safety_filter_cbf`, and `active_slam_toy`
      ([#4](https://github.com/rsasaki0109/PythonInteractiveRobotics/issues/4)).
- [x] Add copyable public-launch post snippets
      ([#5](https://github.com/rsasaki0109/PythonInteractiveRobotics/issues/5)).

## Discovery Channels

Post one GIF and one concrete loop per post. Do not lead with the full README.
Link to the gallery first when the channel benefits from visual scanning, then
link to the exact example file for technical readers.

Good first posts:

- `grasp miss -> belief update -> retry`
- `unknown wall -> A* replanning`
- `nominal controller -> CBF safety filter`
- `pose and map uncertainty -> active SLAM action`
- `language goal -> visual tokens -> skill failure -> close-view retry`

Target channels:

- GitHub topic search
- X / LinkedIn short GIF posts
- robotics Discord or Slack communities
- Reddit `r/robotics` and `r/reinforcementlearning` when the post is
  educational and concrete
- personal blog or project page

## Copyable Launch Snippets

### Manipulation

Tiny robotics loop in plain Python:

```text
observe object -> choose grasp -> miss -> update belief -> retry
```

Run it:

```bash
python3 examples/manipulation/01_pick_and_retry.py
```

GIF: `docs/assets/gifs/pick_and_retry.gif`

### Runtime Safety

A nominal go-to-goal controller does not know about obstacles. A separate CBF
safety filter projects each velocity command back into the safe set:

```text
nominal u -> CBF projection -> safe u -> observe closest approach
```

Run it:

```bash
python3 examples/navigation/29_safety_filter_cbf.py
```

GIF: `docs/assets/gifs/safety_filter_cbf.gif`

### Active Perception

The robot does not just move toward the goal. It chooses actions that reduce
map and pose uncertainty:

```text
predict pose belief -> scan map -> update uncertainty -> choose information action
```

Run it:

```bash
python3 examples/navigation/07_active_slam_toy.py
```

GIF: `docs/assets/gifs/active_slam_toy.gif`

### Embodied AI

A tiny VLA-style loop without a neural model:

```text
language goal -> visual tokens -> skill call -> failure -> close-view retry
```

Run it:

```bash
python3 examples/embodied_ai/19_tiny_vla_loop.py "place red block in blue bin"
```

GIF: `docs/assets/gifs/tiny_vla_loop.gif`

## Contributor Funnel

Keep public issues small and specific. Good first issues should improve an
existing example, visualization, README block, or smoke test before adding a
new concept.

Issue labels to use:

- `good first issue`
- `documentation`
- `example`
- `visualization`
- `packaging`
- `release`

## Metrics To Check Weekly

Use GitHub traffic for the last 14 days:

```bash
gh api repos/rsasaki0109/PythonInteractiveRobotics/traffic/views
gh api repos/rsasaki0109/PythonInteractiveRobotics/traffic/clones
gh api repos/rsasaki0109/PythonInteractiveRobotics/traffic/popular/referrers
gh api repos/rsasaki0109/PythonInteractiveRobotics/traffic/popular/paths
```

Targets for the first 30 days after public launch:

- 1,000 unique repository views
- 50 stars
- 5 external referrers beyond `github.com`
- 1 external issue or pull request
- 3 public posts that each use a different GIF and loop
