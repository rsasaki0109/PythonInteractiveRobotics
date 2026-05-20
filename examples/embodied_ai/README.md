# Embodied AI Examples

## Learning Order

Start with a controlled goal command, then move to hidden-state search,
goal-conditioned kitchen interaction, and the tiny VLA loop.

## What This Teaches

These examples keep language and embodied action small enough to inspect. A
command or hidden-state task becomes a structured goal, then the agent uses
observation, memory, belief, and recoverable failures to decide the next
physical action.

## GIF Gallery

| Goal command pick | Door search POMDP |
| --- | --- |
| ![A controlled language goal is parsed, then a tabletop robot searches, updates belief, misses grasps, and retries.](../../docs/assets/gifs/goal_command_pick.gif) | ![A room-search agent updates key-location belief after a locked door and an empty container, then finds the key.](../../docs/assets/gifs/door_search_pomdp.gif) |

| Goal-conditioned minikitchen | Tiny VLA loop |
| --- | --- |
| ![A kitchen agent parses a bring goal, searches containers, handles a closed cabinet, picks a mug, and places it on the table.](../../docs/assets/gifs/goal_conditioned_minikitchen.gif) | ![A toy VLA loop parses a language goal, reads visual tokens, picks from low confidence, recovers with a close view, and places the block.](../../docs/assets/gifs/tiny_vla_loop.gif) |

| Object permanence toy |
| --- |
| ![An embodied agent sees an object, watches it go behind an occluder, persists its memory, walks to the remembered position, and peeks behind the occluder to recover the object.](../../docs/assets/gifs/object_permanence_toy.gif) |

## `01_goal_command_pick.py`

### What this teaches

A simple language goal can be parsed into a structured intent, then executed as
a closed loop with search, memory, belief update, pick failure, and retry.

### Run

```bash
python examples/embodied_ai/01_goal_command_pick.py "find the red block and pick it"
```

### Key loop

```text
parse goal -> search object -> update belief -> pick -> observe failure -> retry
```

### Simplifications

- controlled natural language
- no LLM
- one object
- fake detector
- 2D tabletop
- probabilistic grasp success

### Things to try

- Try an unsupported command and inspect the `unsupported_goal` failure.
- Add a supported command to `SUPPORTED_COMMANDS`.
- Change the search viewpoint order.
- Increase the retry offsets and compare grasp misses.

## `10_door_search_pomdp.py`

### What this teaches

An embodied agent can search under partial observation by remembering visited
rooms, updating a belief over hidden object locations, and recovering from
failed actions such as locked doors or empty containers.

### Run

```bash
python examples/embodied_ai/10_door_search_pomdp.py
```

### Key loop

```text
observe room -> update memory -> choose door/container -> act -> update belief
```

### Simplifications

- small room graph
- one hidden key
- deterministic observations
- fixed search policy
- no LLM
- no full POMDP solver

### Things to try

- Move the hidden key to another container.
- Change the initial `key_belief`.
- Unlock the storage door and compare the search path.
- Add a second empty container to one room.

## `18_goal_conditioned_minikitchen.py`

### What this teaches

A goal-conditioned embodied agent should connect a parsed goal to observation,
memory, object search, container interaction, pick, and place. Failures such as
empty containers and closed containers update the next action.

### Run

```bash
python examples/embodied_ai/18_goal_conditioned_minikitchen.py "bring mug to table"
```

### Key loop

```text
parse goal -> observe station -> search container -> open on failure -> remember object -> pick -> place
```

### Simplifications

- controlled language
- tiny kitchen stations
- deterministic movement
- scripted containers
- no LLM
- no physics

### Things to try

- Change which container holds the target object.
- Add another supported bring goal.
- Start with a cabinet already open.
- Make a container empty and watch memory update.

## `19_tiny_vla_loop.py`

### What this teaches

A VLA-style loop can be understood before any neural model is involved:
language is parsed into a goal, vision produces object tokens, and action is a
discrete skill call. The robot still needs feedback because a low-confidence
visual token can lead to a failed skill.

### Run

```bash
python examples/embodied_ai/19_tiny_vla_loop.py "place red block in blue bin"
```

### Key loop

```text
language goal -> visual tokens -> pick/place skill -> observe failure -> change view -> retry skill
```

### Simplifications

- controlled language
- fake visual tokens
- discrete skills
- one target object and one target bin
- no LLM, VLM, or VLA model

### Things to try

- Lower the visual-token confidence threshold.
- Add a distractor block with a similar color.
- Remove the close-view retry and compare failures.
- Add a new discrete skill and route a goal to it.

## `21_object_permanence_toy.py`

### What this teaches

An embodied agent should not forget an object the moment it leaves the field of
view. The agent sees the object once, an occluder slides over it, and the
observation channel reports the object as gone. The agent persists the last
known position in memory, walks to it, and uses a short-range peek action to
recover the object behind the occluder.

### Run

```bash
python examples/embodied_ai/21_object_permanence_toy.py
```

### Key loop

```text
see object -> store memory -> object goes behind occluder -> persist memory -> walk to remembered position -> peek -> recover
```

### Simplifications

- 1x1 2D table
- one object and one rectangular occluder
- FOV is a fixed-radius circle around the agent
- observation either returns the true position or nothing
- peek is a close-range deterministic check
- no distractor objects

### Things to try

- Set `short_range_radius` above zero and watch the agent see through the
  occluder when very close.
- Move the object after the occluder activates and watch the peek miss.
- Disable memory storage in the agent and watch the agent never reach the
  object.
- Add a second object the agent should ignore.
