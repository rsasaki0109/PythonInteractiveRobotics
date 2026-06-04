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

**Phase 1 — one real loop on the page (1–2 days).** Add a "Run real Python"
toggle to the existing playground for `clarifying_question` (its renderer already
exists). Python produces the trace; JS draws it; delete the JS dynamics for that
scenario. This is the first honest "real Python in your browser" claim.

**Phase 2 — tabletop renderer + hero loop (1–2 days).** Add the continuous
tabletop renderer and wire `pick_and_retry`. Now the README hero GIF has a
"run it yourself" twin.

**Phase 3 — editable code cell (1–2 days).** Expose the agent's `act()` in a
small editor so visitors can tweak the retry/belief logic and re-run. This is
the "wow, I can edit the robot's brain in the browser" moment that converts to
stars.

Ship Phase 0–1 behind the existing playground before any Hacker News launch;
Phases 2–3 can follow the launch.

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
