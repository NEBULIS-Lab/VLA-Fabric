# pi0.5 VLA-Fabric

This directory contains the pi0.5 realization used to test whether the
interaction functions studied with Eagle2 VLM-based SingleVLA agents transfer to a
structurally different flow-matching VLA. It keeps each arm's observations and
actions local while re-instantiating the three functions as:

- operator-level Raw Common fusion over the multimodal prefix;
- peer Private Prefix K/V consumed by the local action query;
- gated residual peer-action attention recomputed in each layer and flow step.

The primary model path is `pi_native_v2_residual_action`. The release also
includes independent fine-tuning, Full single-stage fine-tuning, task-local
continuation contracts, RoboTwin adapters, and inference path-removal profiles.
The shared N-agent core and focused three- and four-arm task workflows are
included under `scripts/multiarm`; `scripts/*_multiarm_frame.py` covers the
additional three-arm frame and independently trained frame controls.

## External dependencies

Install OpenPI separately and expose its source tree through `OPENPI_ROOT`.

```bash
export OPENPI_ROOT=/path/to/openpi
python -m pip install -e "$OPENPI_ROOT"
python -m pip install -e ./pi05
```

Dataset conversion expects the ALOHA RLDS data described in
[`docs/datasets.md`](../docs/datasets.md). Put converted task data under
`pi05/data/converted/<task>` and a pi0.5 base checkpoint under
`pi05/external/pi05_base`, or pass explicit paths to the CLIs.

## Training

`scripts/train_dual_pi05.py` supports Full single-stage, independently trained,
and Common-first settings. The example below initializes Common interaction
before enabling Full interaction. It is an example recipe, not a requirement
for every task. Use the trainer's `--help` output to configure data, checkpoints,
optimization, and execution.

```bash
cd pi05
./scripts/train_common_first.sh \
  data/converted/aloha_handover_box \
  external/pi05_base \
  outputs/handover
```

Set `GPUS` and `COORDINATOR_PORT` when the defaults conflict with other jobs.
The Python trainer is dry-run by default; the launcher adds `--execute`.

## Paired evaluation

Development uses condition IDs 0--199. The disjoint paired evaluation uses
IDs 200--399 after checkpoint selection.

```bash
cd pi05
export GPU_ID=1
CUDA_VISIBLE_DEVICES="$GPU_ID" ./scripts/evaluate_paired200.sh \
  outputs/handover/stage2/checkpoint_40000 \
  external/pi05_base \
  data/converted/aloha_handover_box \
  aloha_handover_box \
  outputs/handover/eval_fresh200
```

## Three- and Four-Agent Workflows

The multi-agent extension uses one process and one complete local policy per arm.
See [`docs/multiarm-hpc.md`](../docs/multiarm-hpc.md) for StackCube, Frame
Insertion, Arch Assembly, dataset preparation, training, and
strict evaluation. Set GPU IDs explicitly to the allocated devices. Select a
microbatch that fits those devices and
adjust accumulation to preserve the effective team batch.

## RoboTwin Evaluation

RoboTwin conversion and evaluation use the standalone CLIs
`convert_robotwin2_rlds.py`, `prepare_robotwin2_conditions.py`,
`serve_robotwin2_pi05.py`, and `evaluate_robotwin2_pi05.py` in `scripts/`.
Use `--help` for explicit dataset, checkpoint, and environment locations.
Inference removals (`no_common`, `no_private`, `core`, `all_off`) evaluate the
Full checkpoint with fixed weights and the selected interaction paths disabled.
Run the local IPC service within a trusted environment and restrict access to
the collaborating evaluation processes.

## Tests

```bash
XLA_FLAGS=--xla_force_host_platform_device_count=4 \
JAX_PLATFORMS=cpu \
PYTHONPATH=pi05/src:$OPENPI_ROOT/src \
python -m pytest -q pi05/tests
```
