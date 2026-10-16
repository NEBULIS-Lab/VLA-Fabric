# Training and Evaluation

## Interaction Discovery

The SingleVLA configuration matrix varies Common aggregation, peer K/V, and
action-side interaction on Handover Box and Shoes Table. Configurations are in
`eagle/configs/architectures/` and `eagle/configs/runs/`. Single-stage fine-tunes
the configured interaction directly; Common-first initializes Common before
fine-tuning the full interaction.

## Cross-Backbone and Shared Policies

The pi0.5 realization places Common interaction in the multimodal prefix,
Private K/V at the prefix-to-action boundary, and gated peer-action attention
inside the Action Expert. Task-specific entry points are in `pi05/scripts/`.
The separate [shared-policy package](../shared/pi05/README.md) fine-tunes one
policy pair across multiple tasks.

## Multi-Arm Simulation Workflows

Use `pi05/scripts/multiarm/` for three- and four-agent execution, and
`simulation/` for the released manipulation environments. The
[multi-arm guide](multiarm-hpc.md) describes portable paths and launch commands.

## Evaluation Settings

Independent learns without interaction. All-off disables all three interaction
functions in a Full checkpoint without retraining. Individual removals disable
one function. Use a fixed checkpoint and the same condition file when comparing
these settings. The paper specifies the evaluation populations and result
definitions; the task evaluators validate condition IDs and aggregate outcomes.

Upstream pretrained models, task data, and simulator installations are external
dependencies. Use the task-appropriate normalizers and observation/action
interfaces when loading checkpoints.
