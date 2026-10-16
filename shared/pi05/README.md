# Shared-Task pi0.5 Policies

This implementation fine-tunes one policy pair across Handover Box, Shoes Table,
Handover Mic, and optionally Scan Object, using one checkpoint per role across
the selected tasks. Its data interface follows `SharedMixtureSpec` in
`src/pi05_fabric/data/shared_mixture.py`.

## Environment

Use a separate OpenPI environment from the root `pi05/` package. Both packages
provide the `pi05_fabric` namespace; install only one in an environment.

```bash
python -m pip install -e ./shared/pi05
python shared/pi05/scripts/train_dual_pi05.py --help
python shared/pi05/scripts/evaluate_shared_four_task.py --help
```

The mixture manifest identifies the task datasets, per-task normalizers,
training configuration, and output locations. Set these to your own paths;
the manifest reader validates the data contracts before training. The trainer
supports shared Full and Independent settings through `--stage` and
`--mixture-manifest`.

Evaluation uses the same role checkpoints across tasks. The evaluator accepts
a task adapter, frozen condition file, normalization file, and checkpoint path.
Full and All-off evaluation retain the trained weights.

This package preserves the shared-policy training and checkpoint interface
separately from the task-specific continuation interface in the root `pi05/`.
Model weights, datasets, and machine-specific launch configurations are not
bundled.
