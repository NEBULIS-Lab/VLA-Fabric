# VLA-Fabric

**Learning Interaction for Composable Multi-Arm VLA Agents**

Chenglong Zhang, Shuaijun Liu, Feiyang You, Shuyang Hao, Chengyu Wu, Ningxin Su

The Hong Kong University of Science and Technology (Guangzhou)

[Project website](https://nebulis-lab.com/VLA-Fabric/) · [Paper](docs/paper/VLA-Fabric.pdf)

This repository contains the code used for the ICLR 2027 VLA-Fabric study:
discovering how complete VLA policies learn to coordinate, re-instantiating
the resulting interaction functions in a structurally different VLA backbone,
and extending the same one-arm-one-agent organization to three- and four-arm
tasks.

## Implementations

| Directory | Backbone | Purpose |
|---|---|---|
| [`eagle/`](eagle/) | Eagle2 VLM-based SingleVLA | Controlled architecture and training study over 32 task-model combinations |
| [`pi05/`](pi05/) | pi0.5/OpenPI | Architecture-aware transfer and single-node 2/3/4-agent workflows |
| [`shared/pi05/`](shared/pi05/) | pi0.5/OpenPI | Shared-task fine-tuning and evaluation with one checkpoint per role |
| [`simulation/`](simulation/) | MuJoCo/robosuite | Three-arm frame, four-arm frame, and arch task environments |

Both implementations preserve one-arm-one-agent ownership: each policy receives
its local observation and produces its local action. Peer information crosses
only the interaction boundaries defined by the experiment configuration.

The code covers StackCube, three-/four-arm frame tasks, Arch Assembly,
RoboTwin bimanual task adapters, independently trained controls, and inference
path removals. The project website presents the architecture, experimental
results, and simulation and physical-robot videos.

## Quick start

1. Follow [`docs/installation.md`](docs/installation.md) for the two independent
   environments and upstream dependencies.
2. Prepare the ALOHA tasks as described in
   [`docs/datasets.md`](docs/datasets.md).
3. Use the package-level READMEs for training and paired evaluation.
4. For three- and four-agent execution, follow [`docs/multiarm-hpc.md`](docs/multiarm-hpc.md).
5. Consult [`docs/reproducibility.md`](docs/reproducibility.md) for the
   architecture matrix, stage protocol, and evaluation split.
6. Use the [shared-policy package](shared/pi05/README.md) in a separate
   environment for joint multi-task fine-tuning.

Independent is trained without interaction. All-off disables interaction at
inference in a trained Full policy; Full retains all three functions. Pretrained
models, datasets, and task simulators are installed separately. This release
includes model architectures, interaction modules, training, and simulation
workflows. Physical-robot acquisition, data export, and deployment code are
not included.

Check internal package dependencies with:

```bash
python -m pytest -q tests
```

Backbone-specific tests are documented in [`eagle/README.md`](eagle/README.md)
and [`pi05/README.md`](pi05/README.md).

## Citation and License

See [CITATION.cff](CITATION.cff) for authorship and project information.

Project-owned code is released under the MIT License in
[`LICENSE`](LICENSE). TwinVLA, OpenPI, ALOHA
datasets, simulators, and pretrained checkpoints remain subject to their own
licenses and must be obtained from their original sources.
The paper and its figures are not relicensed by the code license.
