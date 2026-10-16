# Datasets

The ALOHA two-agent experiments use the `Handover Box` and `Shoes Table` tasks.
This guide describes their observation/action layout and the multi-arm data
interfaces.

## Observation and action ownership

Each paired sample contains one shared global camera plus arm-local wrist and
proprioceptive observations. The 20-dimensional environment action is split in
role order into two 10-dimensional local actions:

| Agent | Local observations | Local action |
|---|---|---|
| Left | global camera, left wrist, left proprioception | action dimensions 0--9 |
| Right | global camera, right wrist, right proprioception | action dimensions 10--19 |

Each policy receives the shared global view and its own wrist and proprioceptive
observations. Peer information enters through the configured intermediate
interaction paths.

## Eagle layout

Set `dataset.root` in the task YAMLs or place the upstream RLDS data at
`eagle/external/aloha_rlds`. The task configurations are:

```text
eagle/configs/base/aloha_handover_box.yaml
eagle/configs/base/aloha_shoes_table.yaml
```

## pi0.5 converted layout

The pi0.5 loader consumes TensorFlow-free per-episode archives with:

```text
<task>/
  manifest.json
  normalization_q01_q99.json
  episodes/
    episode_00000.npz
    ...
```

Each archive stores `global_image`, `left_wrist_image`, `right_wrist_image`,
`proprioception`, `action`, and `instruction`. V2 training verifies that the
normalization record names the SHA-256 digest of the dataset manifest before it
starts. Quantile statistics can be generated with:

```bash
cd pi05
python scripts/generate_pi05_v2_quantile_stats.py --help
```

Record dataset provenance, conversion commands, and licenses alongside the
converted data.

## Multi-arm layouts

Three-agent StackCube uses one shared global RGB stream, one wrist RGB stream
per role, 9-D local state, and 8-D local actions. Conversion validates aligned
trajectories and produces role-specific q01/q99 statistics.

The four-agent tasks use one shared `agentview`, four role-local wrist streams,
9-D local state, and 7-D local actions. Only training-ready, success-only source
releases with four aligned roles are accepted. `frontview`, when present, is
used exclusively for data auditing. Conversion and quantile commands
are documented in [`multiarm-hpc.md`](multiarm-hpc.md).
