"""Explicit schedule extension without resetting Adam, RNG, or data position."""
import copy
import hashlib
import json
from pathlib import Path

from pi05_fabric.data.focused_sampling import FocusedSampler
from pi05_fabric.training.stages import StageName, parameter_sha256


def prepare_extension(path, args, config, protocol):
    raw = Path(path).read_bytes()
    spec = json.loads(raw)
    for name, expected in spec['trainer_contract'].items():
        if getattr(args, name) != expected:
            raise ValueError(f'extension requires {name}={expected}')
    supported = {
        StageName.PI_NATIVE_V2_RESIDUAL_ACTION_STAGE2,
        StageName.PI_NATIVE_V2_EXPANDED_CONTINUATION,
    }
    if config.stage not in supported:
        raise ValueError('extension requires V2 Full or expanded V2')
    if config.weights_checkpoint is not None or config.stage1_checkpoint is not None or config.resume_checkpoint is None:
        raise ValueError('extension requires full-state resume, not weight-only loading')
    source = json.loads((Path(spec['source_checkpoint']) / 'manifest.json').read_text())
    if (source['step'] != spec['source_step'] or source['schedule_step'] != spec['source_step']
            or source['model_sha256'] != spec['source_model_sha256']
            or source['state_sha256'] != spec['source_state_sha256']):
        raise ValueError('extension source checkpoint changed')
    if source['stage'] != config.stage.value:
        raise ValueError('source stage changed')
    if source['training_seed'] != config.resolved_training_seed or source['model_seed'] != config.resolved_model_seed:
        raise ValueError('extension must preserve source seeds')
    source_protocol = source['protocol']
    for key in ('action_horizon', 'execution_horizon', 'flow_steps',
                'normalization_file', 'normalization'):
        if source_protocol.get(key) != protocol.get(key):
            raise ValueError(f'extension normalization or action protocol changed: {key}')
    contract = copy.deepcopy(source_protocol['continuation'])
    if config.stage is StageName.PI_NATIVE_V2_RESIDUAL_ACTION_STAGE2:
        sampler = FocusedSampler(spec['focus_index'], config.dataset,
                                 microbatch=config.batch_size,
                                 accumulation=config.gradient_accumulation)
        if sampler.sha256 != spec['focus_index_sha256'] or sampler.sha256 != contract['focus_index_sha256']:
            raise ValueError('extension sampler changed')
        source_global_batch = contract['global_batch_size']
    else:
        sampler = None
        if contract.get('kind') != 'scan_expanded_low_rewarm':
            raise ValueError('expanded extension source contract changed')
        source_global_batch = source_protocol.get('training_contract', {}).get('global_batch_size')
    if config.global_batch_size != source_global_batch or config.global_batch_size != 16:
        raise ValueError('extension must preserve global batch 16')
    if source['parent_model_sha256'] != spec['parent_model_sha256']:
        raise ValueError('extension lineage changed')
    if args.warmup_steps != 0 or args.schedule_offset != source['step'] or config.steps <= source['step']:
        raise ValueError('extension requires offset cosine with no rewarm')
    rates = {
        'action_expert': [args.action_expert_peak, args.action_expert_final],
        'paligemma': [args.paligemma_peak, args.paligemma_final],
        'adapters': [args.learning_rate, args.final_learning_rate],
    }
    if config.stage is StageName.PI_NATIVE_V2_EXPANDED_CONTINUATION:
        rates.update({
            'action_ffw': [args.action_ffw_peak, args.action_ffw_final],
            'paligemma_qo': [args.paligemma_qo_peak, args.paligemma_qo_final],
        })
    contract['low_lr_extension'] = {
        'spec_sha256': hashlib.sha256(raw).hexdigest(),
        'source_model_sha256': source['model_sha256'],
        'source_state_sha256': source['state_sha256'],
        'source_stage': source['stage'],
        'source_step': source['step'], 'end_step': config.steps,
        'optimizer': 'preserve Adam moments/counts; cosine(count-source_step); no warmup',
        'rates': rates,
    }
    return sampler, contract


def validate_extension_resume(snapshot, *, protocol, training_seed, model_seed):
    extension = protocol['continuation']['low_lr_extension']
    expected_stage = StageName(extension.get(
        'source_stage', StageName.PI_NATIVE_V2_RESIDUAL_ACTION_STAGE2.value))
    if snapshot.stage is not expected_stage:
        raise ValueError('extension stage mismatch')
    if snapshot.training_seed != training_seed or snapshot.model_seed != model_seed:
        raise ValueError('extension seed changed')
    if snapshot.parent_model_sha256 != protocol['continuation']['parent_model_sha256']:
        raise ValueError('extension lineage changed')
    if snapshot.step != snapshot.schedule_step or not extension['source_step'] <= snapshot.step <= extension['end_step']:
        raise ValueError('extension step/schedule mismatch')
    if snapshot.protocol_metadata == protocol:
        return
    source_protocol = copy.deepcopy(protocol)
    del source_protocol['continuation']['low_lr_extension']
    target_stable = {k: v for k, v in source_protocol.items() if k != 'training_contract'}
    snapshot_stable = {
        k: v for k, v in snapshot.protocol_metadata.items() if k != 'training_contract'
    }
    if (snapshot_stable != target_stable or snapshot.step != extension['source_step']
            or parameter_sha256(snapshot.params) != extension['source_model_sha256']):
        raise ValueError('extension source contract or parameter hash changed')
