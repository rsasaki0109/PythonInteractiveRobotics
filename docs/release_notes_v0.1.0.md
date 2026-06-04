# Release draft — v0.1.0 "Tiny Robot Failure Lab"

> Draft for the first public GitHub Release. Paste the body below into the
> release on the `v0.1.0` tag. Keep it a story, not a changelog. Cut anything
> that reads like an internal status report.

**Tag:** `v0.1.0` &nbsp;·&nbsp; **Title:** `v0.1.0 — Tiny Robot Failure Lab`

---

## Release body (copy from here)

### Robotics tutorials usually assume actions succeed. This one is about what happens when they don't.

Real robots miss grasps, drive into walls they couldn't see, lose track of where
they are, and misread ambiguous commands. PythonInteractiveRobotics is a tiny
lab for exactly that part of robotics — **observe, act, fail, update your
belief, replan, retry** — in readable Python with no ROS, no GPU, and no
simulator. Just `numpy + matplotlib`.

**What's in this first release**

A failure-first course of **10 short, runnable loops**, in order, plus 39 total
examples to branch into. Start here:

```bash
git clone https://github.com/rsasaki0109/PythonInteractiveRobotics.git
cd PythonInteractiveRobotics
python3 -m pip install -e .
python3 examples/manipulation/01_pick_and_retry.py
```

A tabletop robot misses a grasp, updates its belief, and retries — in under 5
seconds.

**Try it without installing**

- ▶ **Run in your browser:** the [live playground](https://rsasaki0109.github.io/PythonInteractiveRobotics/playground.html)
  with belief entropy, compare mode, and a failure timeline.
- 📓 **Open in Colab:** [pick and retry](https://colab.research.google.com/github/rsasaki0109/PythonInteractiveRobotics/blob/main/notebooks/pick_and_retry.ipynb),
  [safety filter](https://colab.research.google.com/github/rsasaki0109/PythonInteractiveRobotics/blob/main/notebooks/safety_filter_cbf.ipynb),
  [household task agent](https://colab.research.google.com/github/rsasaki0109/PythonInteractiveRobotics/blob/main/notebooks/household_task_agent.ipynb).
- 🎓 **Take the tour:** the [10-lesson failure-first course](https://github.com/rsasaki0109/PythonInteractiveRobotics/blob/main/lessons/README.md).

**Highlights**

- **Fail and retry** — grasp miss → belief update → retry (`manipulation/01`)
- **Replan around a hidden wall** — plan → see new obstacle → replan (`navigation/04`)
- **Act to learn** — toy active SLAM that moves to shrink uncertainty (`navigation/07`)
- **Stay safe at runtime** — a CBF safety filter the policy never knows about (`navigation/29`)
- **Ask before acting** — ambiguous command → clarify → act (`embodied_ai/35`)
- **Put it together** — a household agent that clarifies, plans, stays safe,
  retries, and replans in one run (`embodied_ai/36`)

Every example exposes failure through `info["failure"]` and returns an
inspectable `Trace`, so you can study a run headless without rendering.

**Under the hood**

- 39 runnable examples · 38 generated GIFs · 5 Colab notebooks
- 111 smoke / regression tests · CI green on Python 3.10, 3.11, and 3.12
- Core deps: `numpy` + `matplotlib` only; optional Gymnasium-style adapters and
  ROS2 / simulator bridge docs for when you outgrow the toy worlds

**Where this sits**

Not a replacement for ROS2, MoveIt, MuJoCo, Isaac Sim, or LeRobot — and not a
benchmark. Think of it as the missing closed-loop chapter you read *after*
algorithm textbooks like PythonRobotics and *before* a heavy stack: a small,
debuggable model of failure, belief, recovery, and replanning.

**Contribute**

Good first contributions are deliberately small:

- add a new **failure mode** to an existing world
- add a **one-file lesson** in the failure-first style
- improve a **trace story / GIF**

See [`CONTRIBUTING.md`](https://github.com/rsasaki0109/PythonInteractiveRobotics/blob/main/CONTRIBUTING.md).
If this helped you learn, teach, or prototype, a ⭐ helps others find it.

---

## Pre-publish checklist

Before tagging `v0.1.0`, confirm the launch surface is ready (see also
`docs/public_launch.md`):

- [ ] README top renders the hero GIF, the 8-line loop, and the browser link in
      the first screen
- [ ] `lessons/README.md` links resolve on GitHub and all 10 commands run
- [ ] CI green on `main` for 3.10 / 3.11 / 3.12
- [ ] GitHub repo "social preview" image set (Settings → General → Social preview)
- [ ] Playground page loads from GitHub Pages
- [ ] Tag the release: `git tag -a v0.1.0 -m "Tiny Robot Failure Lab" && git push origin v0.1.0`

## How to cut the release

```bash
# from a clean main with everything pushed
git tag -a v0.1.0 -m "Tiny Robot Failure Lab"
git push origin v0.1.0
# then create the GitHub Release on that tag and paste the body above,
# or:  gh release create v0.1.0 --title "v0.1.0 — Tiny Robot Failure Lab" --notes-file <(...)
```
