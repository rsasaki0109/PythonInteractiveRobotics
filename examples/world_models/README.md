# World Model Examples

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
