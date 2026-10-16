# Eagle2 VLM-Based SingleVLA Interaction Study

This package contains the two-agent Eagle2 VLM-based SingleVLA implementation used for the
controlled interaction study in the ICLR 2027 paper. Each arm owns a complete
SingleVLA policy, its wrist observation, proprioception, and local action head.
Only explicitly configured Common, Private K/V, and action representations
cross the policy boundary.

## Interaction Matrix

The `configs/matrix` manifests enumerate 16 models per task: nine Single-stage
architectures and seven Common-first counterparts. The architecture identifiers
are:

- `I0`: Independent policies;
- `I1`: Private K/V only;
- `I2`: one-way Common plus Private K/V;
- `I3-AVG` / `I3-RAW`: symmetric Common plus Private K/V;
- `I4-AVG` / `I4-RAW`: I3 plus Linear action interaction;
- `I5-AVG` / `I5-RAW`: I3 plus Transformer action interaction.

Common-first runs train Common interaction for 10k updates before enabling the
remaining channels in a separately initialized 40k Stage 2. The selected
I4-RAW Linear and I5-RAW Transformer families also have 100k follow-up
configurations for extended fine-tuning.

## External Dependencies

Place or link the upstream TwinVLA/SingleVLA checkout at
`external/TwinVLA-base`, the ALOHA RLDS data at `external/aloha_rlds`, and the
Tabletop-Sim checkout according to the top-level installation guide. Upstream
projects retain their original licenses.

## Training

```bash
cd eagle
python -m pip install -e .
scripts/train.sh configs/runs/handover_box/S-I4-RAW-stage2.yaml
```

Stage 2 configurations reference their matching Stage 1 outputs under
`outputs/`; run the listed Stage 1 configuration first.

## Paired Evaluation

```bash
scripts/evaluate_paired200.sh \
  checkpoints/i4_raw \
  aloha_handover_box \
  disjoint \
  outputs/evaluation/i4_raw_disjoint
```

Development uses rollout IDs 0--199. The disjoint paired evaluation uses IDs
200--399 after checkpoint selection is frozen.

## Unit Tests

```bash
PYTHONPATH=src python -m pytest -q tests
```
