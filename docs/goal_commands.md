# Goal Commands

Use this document as the working instruction for goal-conditioned examples.

## Core Rule

Start with controlled natural language, not an LLM.

Goal commands should be parsed by small Python functions first. The teaching
target is the loop:

```text
goal -> observation -> memory / belief -> action -> environment change -> retry
```

## Implemented Flagship Command

The first goal-conditioned example implements:

```text
find the red block and pick it
```

This command should show:

- object search
- partial observation
- belief update
- pick attempt
- grasp failure
- retry with changed belief or action

## Implemented Controlled Commands

These commands are currently covered by runnable examples:

```text
find the red block and pick it
bring mug to table
place red block in blue bin
```

They map to:

- `examples/embodied_ai/01_goal_command_pick.py`
- `examples/embodied_ai/18_goal_conditioned_minikitchen.py`
- `examples/embodied_ai/19_tiny_vla_loop.py`

## Candidate Command Set

Keep the first parser limited to these commands:

```text
go to the red object
find the key
pick the red block
bring the key to the door
bring mug to table
find the red block and pick it
place red block in blue bin
```

Do not add open-ended language support yet.

## Implementation Order

1. Add a tiny command parser.
2. Add a goal-conditioned toy world or reuse `Tabletop2D`.
3. Add an agent with explicit memory / belief.
4. Implement a controlled command from this document.
5. Add failure-aware retry.
6. Add README GIF.
7. Add headless smoke test.

## Parser Policy

The parser should return structured goals, for example:

```python
{
    "intent": "find_and_pick",
    "object": "block",
    "color": "red",
}
```

Unknown commands should fail clearly:

```python
{
    "intent": "unknown",
    "message": "unsupported command",
}
```

## Do Not Add Yet

- LLM dependency
- VLM dependency
- speech recognition
- open vocabulary parsing
- large language datasets
- ROS action bridge

Those can come after the toy loop is clear.
