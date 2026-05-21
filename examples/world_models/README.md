# World Model Examples

## GIF Gallery

| Tiny world-model planning | Model error recovery |
| --- | --- |
| ![A point robot predicts action-conditioned dynamics, observes drift model error, updates a residual model, and replans to the goal.](../../docs/assets/gifs/tiny_world_model_planning.gif) | ![A point robot detects a sudden dynamics shift, switches to a short system-identification probe phase, updates the learned offset, and resumes goal navigation.](../../docs/assets/gifs/model_error_recovery.gif) |

## `20_tiny_world_model_planning.py`

### What this teaches

A world model is useful only inside a feedback loop. The robot predicts the
next state, acts, observes the real transition, measures model error, updates
an action-conditioned residual model, and replans.

### Run

```bash
python examples/world_models/20_tiny_world_model_planning.py
```

### Key loop

```text
predict next state -> plan action sequence -> act -> observe transition -> update dynamics model -> replan
```

### Simplifications

- 2D point robot
- discrete actions
- tiny residual dynamics table
- hidden drift field
- beam-search planner
- no neural network

### Things to try

- Increase the drift in `TinyWorldModelWorld`.
- Reduce the planning horizon.
- Remove residual learning and compare the path.
- Add a second hidden dynamics region.

## `23_model_error_recovery.py`

### What this teaches

A world model can be wrong all at once, not just slightly off. When the
underlying dynamics change, prediction error spikes. The agent should detect
that spike, stop trying to make progress toward the goal, run a short
system-identification probe phase, average the observed dynamics offset,
update the model, and then resume goal navigation with the corrected model.

This differs from `20_tiny_world_model_planning.py`, which fits a continuous
residual across a static drift region while moving. Here the failure is a
discrete regime shift, and recovery is an explicit probe state.

### Run

```bash
python examples/world_models/23_model_error_recovery.py
```

### Key loop

```text
predict next pos -> act -> observe error spike -> switch to system_id -> probe -> update offset -> resume goal
```

### Simplifications

- 2D continuous point robot
- single constant offset, applied after a fixed step
- 3 hand-picked probe actions, averaged
- identity initial model
- small Gaussian noise on every step
- no online residual fitting across regimes

### Things to try

- Change `regime_shift_at` and watch the recovery happen later.
- Increase `shift_offset` magnitude and confirm the spike still triggers.
- Lower `error_threshold` and watch noise alone trigger the recovery.
- Replace the probe action list with random probes and compare update quality.
