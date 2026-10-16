# Multi-arm simulation

The released task implementations are:

| Module | Team | Task |
| --- | --- | --- |
| `multiarm_sim.envs.triangle_frame_insertion` | 3 | Triangular frame over peg |
| `multiarm_sim.envs.frame_insertion` | 4 | Quadrilateral frame over peg |
| `multiarm_sim.envs.arch_assembly` | 4 | Arch assembly |

Install this package in the simulator environment, independently of OpenPI:

```bash
python -m pip install -e ./simulation
```

The package contains project-owned environments, scripted demonstration
controllers, success checks, and HDF5 recording utilities. Robot models and
physics are provided by robosuite/MuJoCo.
The three-agent StackCube workflow in `pi05/` uses a separate RoboFactory
environment; its external dependency is documented in `docs/multiarm-hpc.md`.

Frame policy adapters and CLIs are in `pi05/src/pi05_fabric/data/multiarm_frame_tasks.py`
and `pi05/scripts/{convert,train,evaluate}_multiarm_frame.py`. All dataset,
checkpoint, and simulator paths are supplied by the caller. The evaluation CLI
can run the environment in a separate Python environment via its local IPC
server. Restrict the pickle-based IPC endpoint to trusted local evaluation
processes.
