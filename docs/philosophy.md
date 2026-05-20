# Philosophy

PythonInteractiveRobotics is not a replacement for PythonRobotics. It is the
next educational layer after it.

PythonRobotics is a minimal Python textbook for understanding core robotics
algorithms. PythonInteractiveRobotics is a minimal Python textbook for
understanding intelligent loops that interact with an environment.

The core loop is:

```text
sense -> think -> act -> observe failure -> update belief -> retry
```

## Principles

Loop first, algorithm second.

Algorithms should appear inside an environment loop, not as isolated demos.

Failure is part of the API.

Collision, occlusion, grasp miss, localization drift, blocked paths, and retry
are teaching material. Examples should expose them through `info["failure"]`.

Toy world, real concept.

Physical realism can be low. Interaction, uncertainty, partial observation,
memory, and replanning should be real.

Five-second robotics.

The first run should not require ROS, Docker, GPU training, large assets, or a
heavy simulator.

## What This Repository Optimizes For

- readable examples
- visible internal state
- fast iteration
- failure-aware loops
- lightweight mental models that transfer to ROS2, MuJoCo, PyBullet, Habitat,
  Isaac Sim, and real robots later

## What It Does Not Optimize For

- production robotics middleware
- photorealistic simulation
- benchmark leaderboards
- large pretrained policies
- universal simulator abstractions
