# Multi-Arm pi0.5 Workflows

Each arm runs a complete local policy. The multi-agent implementation exchanges
Common, Private Prefix K/V, and peer-action representations while retaining
arm-local observations and action outputs.

## Environment and Data

Install OpenPI and the simulator dependencies as described in
[installation.md](installation.md). Set dataset, base-checkpoint, simulator,
and output paths explicitly. The examples in
`pi05/scripts/multiarm/env.example` describe the required environment variables;
GPU IDs must match the devices allocated to your job.

Conversion entry points generate training data and role-specific normalization
statistics. Use their `--help` output to select source and destination paths:

| Task family | Conversion | Training | Evaluation |
|---|---|---|---|
| Stack Cube | `multiarm/convert_stack_cube.py` | `multiarm/train_stack_cube.py` | `multiarm/evaluate_stack_cube.py` |
| Four-arm frame and arch | `multiarm/convert_four_arm_tasks.py` | `multiarm/train_four_arm.py` | `multiarm/evaluate_four_arm.py` |
| Multi-arm frame variants | `convert_multiarm_frame.py` | `train_multiarm_frame.py` | `evaluate_multiarm_frame.py` |

All paths in this table are relative to `pi05/scripts/`.
`multiarm/generate_four_arm_quantiles.py` provides the four-arm normalization
step. Task configurations are under `pi05/configs/multiarm/`.

## Launching and Evaluation

The generic launcher `pi05/scripts/multiarm/launch_agent_ranks.sh` accepts a
comma-separated list of three or four allocated GPU IDs, a coordinator port,
a log directory, and the training or evaluation command after `--`. It exposes
one device per rank and supplies the distributed process arguments.

Choose the interaction configuration, optimization budget, and checkpoint paths
through the task entry point. Common-first and Full single-stage adaptation are
supported; the appropriate recipe depends on the task. Select microbatch and
accumulation settings for the available memory rather than copying server-local
launch settings.

Evaluation loads a complete team checkpoint and connects to the external task
environment. `multiarm/four_arm_env_server.py` and `multiarm_frame_env_server.py`
provide the corresponding local IPC adapters. Run these services only in a
trusted environment. Keep evaluation conditions separate from training and
checkpoint selection, and retain success counts alongside reported rates.

## CPU Contract Tests

```bash
XLA_FLAGS=--xla_force_host_platform_device_count=4 \
JAX_PLATFORMS=cpu \
PYTHONPATH=pi05/src:$OPENPI_ROOT/src \
python -m pytest -q pi05/tests
```

These tests check interaction and checkpoint contracts. End-to-end GPU training
and simulation additionally require the datasets, base weights, and external
environments.
