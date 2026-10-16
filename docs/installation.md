# Installation

The Eagle and pi0.5 implementations each use a dedicated Python environment
with their corresponding PyTorch/JAX and Transformers dependencies.

## Eagle2 VLM-based SingleVLA environment

Requirements:

- Python 3.10 or 3.11;
- CUDA-capable PyTorch for full training and evaluation;
- an upstream TwinVLA/SingleVLA checkout;
- the ALOHA tabletop simulator used by that checkout.

```bash
python -m venv .venv-eagle
source .venv-eagle/bin/activate
python -m pip install --upgrade pip
python -m pip install -e './eagle[dev]'
```

Install the upstream libraries independently and expose their import paths.
Set `TWINVLA_ROOT` to the upstream checkout and configure the model, dataset,
and simulator paths in the run YAML. The examples use this layout:

```text
eagle/external/TwinVLA-base/
eagle/external/aloha_rlds/
eagle/external/tabletop_sim/
```

```bash
export TWINVLA_ROOT=/path/to/TwinVLA
export PYTHONPATH="$TWINVLA_ROOT:${PYTHONPATH:-}"
```

## pi0.5/OpenPI environment

Requirements:

- Python 3.11;
- a CUDA/JAX installation compatible with the selected accelerator;
- an upstream OpenPI checkout and pi0.5 base checkpoint.

Follow the upstream OpenPI installation instructions first, then install this
wrapper package:

```bash
python -m venv .venv-pi05
source .venv-pi05/bin/activate
python -m pip install --upgrade pip
python -m pip install -e /path/to/openpi
python -m pip install -e ./pi05
export OPENPI_ROOT=/path/to/openpi
```

Example model and dataset locations:

```text
pi05/external/pi05_base/
pi05/data/converted/aloha_handover_box/
pi05/data/converted/aloha_shoes_table/
```

For multi-arm execution, install the external task environment in a separate
compatible environment. StackCube requires RoboFactory and its IPC server
configuration. Frame Insertion and Arch Assembly require the released
`multiarm_sim` task package (`pip install -e ./simulation`), robosuite, MuJoCo assets, and an EGL-capable Python
environment. Their paths are passed through the variables documented in
[`multiarm-hpc.md`](multiarm-hpc.md).

## CPU Verification

Run the protocol tests on CPU. Eagle tests cover configuration, interaction, and
evaluation utilities; pi0.5 tests use the OpenPI Python sources to validate the
model boundary.

```bash
PYTHONPATH=eagle/src python -m pytest -q eagle/tests

XLA_FLAGS=--xla_force_host_platform_device_count=4 \
JAX_PLATFORMS=cpu \
PYTHONPATH=pi05/src:$OPENPI_ROOT/src \
python -m pytest -q pi05/tests
```
