# Design memo: running the real Python loops in the browser (Pyodide)

Status: **proposal / not yet implemented.** This memo scopes follow-up item ②
from the launch plan — the highest-leverage growth hook GPT Pro flagged
("install-free / readable / no heavy stack" is what lands on Hacker News).

## Why now

The current [`docs/playground.js`](playground.js) is a ~1200-line **JavaScript
reimplementation** of just two scenarios (`clarifying`, `household`). It does not
run any of the repo's Python. Two consequences:

1. **Drift risk.** The JS dynamics can silently disagree with the tested Python
   examples; nothing keeps them in sync.
2. **Weak headline.** "An interactive web demo" is ordinary. "**The actual
   `numpy` example you'd run locally, running in your browser, no install**" is
   the share-worthy version — and it's true to the repo's whole premise.

Pyodide ([pyodide.org](https://pyodide.org)) runs CPython + NumPy in WebAssembly
in the browser, so we can execute the real example code with zero install.

## The key architecture decision: who renders?

The examples use `numpy` for the loop and `matplotlib` for rendering. Matplotlib
*is* available in Pyodide, but it is a large download and slow to first paint.
We do not need it in the browser, because **we already have a renderer**: the
existing playground draws scenes and belief panels from plain data.

**Recommended split — Python computes, JS draws:**

```
Pyodide (Python, numpy only)                Browser (existing JS)
  run(seed, render=False) -> Trace    ──▶   trace JSON  ──▶  current scene /
  trace.summary() / per-step records         (postMessage)    belief / timeline
                                                               renderer
```

- The Python side runs the **real** `examples/.../*.py` loop headless
  (`render=False`) and returns the `Trace` (already a first-class, tested object
  — see [`docs/trace.md`](trace.md)).
- A thin serializer turns the `Trace` (obs/action/info/reward per step) into the
  JSON shape the current `playground.js` renderer already consumes.
- The JS reimplementation of dynamics gets **deleted**; JS keeps only drawing.

This keeps the browser bundle small (no matplotlib), makes the playground a true
mirror of the tested code, and removes the drift problem instead of adding to it.

Fallback option (heavier): import `matplotlib` in Pyodide and blit Agg PNG frames
to a `<canvas>`. Simpler to wire (reuses `env.render`), but a multi-MB download
and janky first paint. Keep this only as a stopgap for an example whose JSON
renderer is not ready yet.

## Packaging

The loops only need `pir` + `numpy`. Options, simplest first:

1. **Load `pir` as source over the network.** `micropip` or Pyodide's
   `loadPackage("numpy")` for numpy, then fetch the handful of `pir/**.py` files
   (or a generated single-file bundle) and write them into Pyodide's virtual FS.
   No build step; works from GitHub Pages.
2. **Ship a wheel.** `python -m build`, host `pir-0.1.0-py3-none-any.whl` under
   `docs/`, `micropip.install("./pir-...whl")`. Cleaner import story; adds a
   release artifact to keep current.

Start with (1) for the first flagship, move to (2) if import wiring gets noisy.
`numpy` has a prebuilt Pyodide package, so no compilation is needed.

## The 5 flagship loops (and their render shape)

| Example | Renderer needed | Notes |
| --- | --- | --- |
| `manipulation/01_pick_and_retry` | tabletop (continuous) | best first target; the hero story, small state |
| `navigation/04_online_replanning_astar` | grid + path | reuses grid renderer; shows replanning |
| `navigation/29_safety_filter_cbf` | continuous + obstacles | needs vector overlay (nominal vs safe u) |
| `navigation/07_active_slam_toy` | grid + belief heatmap | reuses belief panel |
| `embodied_ai/35_clarifying_question` | already in JS today | swap JS dynamics for real Python first |

Two render families cover all five: **grid** (already drawn today) and
**tabletop/continuous** (small addition). Build those two renderers once.

## Phased plan

**Phase 0 — proof of concept. ✅ built (Python path verified; needs a browser
check).** [`docs/pyodide/poc.html`](pyodide/poc.html) loads Pyodide, loads numpy,
unpacks [`docs/pyodide/pir_bundle.zip`](pyodide/) (built by
[`scripts/build_pyodide_bundle.py`](../scripts/build_pyodide_bundle.py) — the
pure-Python `pir` package + `01_pick_and_retry.py`, ~24 KB), runs the
**unmodified** `run(seed, render=False)`, and prints `Trace.summary()` with
timings. It blocks `matplotlib` so the headless path can never silently pull it.

Packaging decision: went with a **zip + `unpackArchive`** (option close to (1)),
not micropip — it avoids resolving the declared matplotlib dependency and keeps
the download tiny. `tests/test_pyodide_bundle.py` pins the zip to the source so
it cannot drift, and `pages.yml` rebuilds it on deploy.

Verified locally (CPython simulating Pyodide's unpack-into-cwd, exact driver from
the HTML): `{"steps": 4, "success": true, "failure_counts": {"grasp_miss": 2},
"total_reward": 0.69}` — identical to `python3 .../01_pick_and_retry.py
--no-render`. **Still to confirm in a real browser:** Pyodide first-load time and
that `loadPyodide`/`unpackArchive`/`fetch` behave as expected. Open
`docs/pyodide/poc.html` via a local server (e.g. `python3 -m http.server` from
`docs/`) or on GitHub Pages and click “Run the real loop”.

**Phase 1 — one real loop on the page. ✅ built (Python path verified; needs a
browser check).** The playground ([`docs/playground.html`](playground.html))
has a **"Run real Python"** toggle for `clarifying_question`. When on, it lazily
boots Pyodide, unpacks the bundle, and runs the **unmodified**
`examples/embodied_ai/35_clarifying_question.py` `run(...)` headless; the real
`Trace` is serialized by [`pir/viz/playground_trace.py`](../pir/viz/playground_trace.py)
into the exact config the existing JS renderer consumes, and the page draws that.
The JS preview stays the default first paint (Pyodide is loaded only on toggle),
so the instant experience is preserved.

The trace→render JSON is now a **pinned contract**:
[`tests/test_playground_trace.py`](../tests/test_playground_trace.py) runs the
real loop, serializes it, and asserts every field the renderer reads is present
and plain-JSON (no numpy leaks) — exactly the drift guard the risk list calls for.

Verified locally (CPython simulating Pyodide's unpack-into-cwd, the exact driver
from `playground.js`): for `answer=red` the serializer returns the real
`ask → look → pick` trace (`ambiguous_goal` then resolved belief, pick at
`[32, 56]`), identical to the CLI loop. **Still to confirm in a real browser:**
toggling "Run real Python" boots Pyodide, runs the loop, and the scene/belief/
timeline redraw from the real trace.

Deliberately **deferred** (not yet done): deleting the JS `buildClarifyingScenario`
dynamics. It is kept as the no-Pyodide instant fallback so first paint never waits
on a multi-MB download. Full deletion waits until the real path is the verified
default — at which point the JS mock can be dropped and the contract test becomes
the single source of truth.

**Phase 2 — tabletop renderer + hero loop. ✅ built (Python path verified; needs
a browser check).** The playground has a **"Pick and retry (real Python)"**
scenario. It is real-Python-only: selecting it boots Pyodide and runs the
**unmodified** `examples/manipulation/01_pick_and_retry.py` `run(seed=3)` loop —
there is deliberately no JS mock, because the dynamics are stochastic and
belief-driven and a hand-faked version would reintroduce the exact drift Pyodide
removes. The README hero GIF now has a "run it yourself" twin.

A continuous `tabletop2d` renderer draws the real scene: the true object vs. the
agent's **spatial belief** (mean + a shrinking uncertainty radius), the occluder,
the camera, the last detection, and each pick attempt — mirroring the matplotlib
render in `pir/worlds/tabletop_2d.py`. The belief panel switches to a spatial
layout (uncertainty bar + attempts/retries/policy).

To make the loop's belief inspectable without the live agent object,
`01_pick_and_retry.py` now records `belief_mean`/`belief_radius`/`retry_count`
into the trace `info` each step (belief becomes first-class in the `Trace`).
Scene geometry (true object, occluder, camera) is ground truth the agent never
sees, so the driver reads it from a real `Tabletop2D` and passes it to
`pick_and_retry_trace_to_playground` rather than hard-coding world constants.
`tests/test_playground_trace.py` pins this second contract too.

Verified locally (unpacked-bundle sim, the exact `playground.js` driver):
`seed=3` yields the hero story — `scan → pick(miss) → pick(miss) → pick(done)`,
belief radius shrinking 10 → 9.8 → 9.5 → 2.5, `holding=True`, `retries=2`.
**Still to confirm in a real browser:** selecting the scenario boots Pyodide and
the tabletop/belief/timeline redraw from the real trace.

**Phase 3 — editable code cell. ✅ built (Python path verified; needs a browser
check).** The pick_and_retry scenario now shows an **"Agent brain"** editor
pre-filled with the *real* `PickAndRetryAgent` source (fetched via
`inspect.getsource` so it can never drift from the file). Editing it and clicking
**Run edited agent** execs the user's class in Pyodide and runs it against the
real `Tabletop2D` via the example's own `run_agent(...)` — the same loop the CLI
uses, so there is no second loop to drift. Syntax/runtime errors surface inline.

This is the "edit the robot's brain in the browser" moment. The edited code runs
entirely client-side (Pyodide is a WASM sandbox — exec'ing it is no more
privileged than a local REPL), `USER_SRC` is passed via `pyodide.globals` to
avoid string-escaping, and a custom agent that drops the belief attributes still
runs (`run_agent` reads them defensively).

Verified locally (unpacked-bundle sim, the exact drivers): the default source
reproduces `seed=3` (`4 steps, retries=2`); removing the deliberate first offset
makes it grab the belief mean immediately (`2 steps, retries=0`) — a live lesson
in *why* the retry schedule exists; malformed code raises a caught error.
`tests/test_playground_trace.py` exercises `run_agent` with a custom agent.
**Still to confirm in a real browser:** the editor populates with the real
source, edits re-run, and errors render inline.

All three phases are built and Python-verified; the remaining work is a single
browser pass over Phases 0–3 and (optionally) deleting the JS `clarifying`
preview once the real path is the trusted default.

## Risks / watch-list

- **First-load latency.** Pyodide core is a few MB. Lazy-load it only when the
  user clicks "Run real Python"; keep the instant JS-rendered preview as the
  default first paint. Cache aggressively.
- **No silent matplotlib import.** If any example imports matplotlib at module
  top level, the headless path drags it in. Keep example imports of matplotlib
  lazy (inside `render`/`main`), as `01_pick_and_retry.py` already does.
- **Trace serialization is the contract.** Add a `tests/` check that the JSON
  serializer covers every field the JS renderer reads, so Python and browser
  cannot drift — this is the guard that makes Pyodide *reduce* drift rather than
  add a new surface.
- **Keep it optional.** Pyodide is a `docs/` concern only. It must never become
  a core dependency or touch the 5-second local first-run.

## Definition of done (item ②)

- One flagship loop runs its **unmodified** Python `run(...)` in the browser.
- The JS reimplementation of that scenario's dynamics is deleted.
- First paint stays instant (Pyodide lazy-loaded on demand).
- A test pins the trace-JSON contract shared by Python and JS.
