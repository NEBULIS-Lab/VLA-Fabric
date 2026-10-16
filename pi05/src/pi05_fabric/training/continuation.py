"""Explicit continuation contract, distinct from a same-stage optimizer resume."""
import hashlib
import json
from pathlib import Path

import jax
import numpy as np

from pi05_fabric.data.focused_sampling import FocusedSampler
from pi05_fabric.training.engine_checkpoint import _replace_selected_params
from pi05_fabric.training.stages import StageName, parameter_sha256


def prepare_continuation(path, args, config, protocol):
    raw=Path(path).read_bytes(); spec=json.loads(raw)
    if spec.get('kind') == 'scan_expanded_low_rewarm':
        from pi05_fabric.training.expanded_continuation import prepare_expanded_continuation
        return None, prepare_expanded_continuation(path, args, config, protocol)
    if spec.get('kind') == 'low_lr_extension':
        from pi05_fabric.training.low_lr_extension import prepare_extension
        return prepare_extension(path, args, config, protocol)
    for name, expected in spec['trainer_contract'].items():
        if getattr(args, name) != expected:
            raise ValueError(f'continuation requires {name}={expected}')
    if config.stage is not StageName.PI_NATIVE_V2_RESIDUAL_ACTION_STAGE2:
        raise ValueError('continuation supports the fixed V2 Full architecture only')
    if config.stage1_checkpoint is not None or (config.weights_checkpoint is None and config.resume_checkpoint is None):
        raise ValueError('continuation requires a weight-only parent or its own resume')
    if config.weights_checkpoint is not None and config.weights_checkpoint.resolve()!=Path(spec['parent_checkpoint']).resolve():
        raise ValueError('wrong continuation parent path')
    sampler=FocusedSampler(spec['focus_index'],config.dataset,microbatch=config.batch_size,accumulation=config.gradient_accumulation)
    if sampler.sha256 != spec['focus_index_sha256']:
        raise ValueError('focus index hash changed')
    parent=json.loads((Path(spec['parent_checkpoint'])/'manifest.json').read_text())
    if parent['model_sha256'] != spec['parent_model_sha256'] or parent['step'] != 20000:
        raise ValueError('parent manifest mismatch')
    if parent['protocol'] != protocol:
        raise ValueError('parent normalization or action protocol changed')
    if config.resolved_training_seed == parent['training_seed']:
        raise ValueError('continuation requires an independent training seed')
    contract={'phase':'focused_continuation','spec_sha256':hashlib.sha256(raw).hexdigest(),
              'focus_index_sha256':sampler.sha256,'parent_model_sha256':parent['model_sha256'],
              'global_batch_size':config.global_batch_size,'training_seed':config.resolved_training_seed,
              'sampler':'uniform microbatch2 then focus microbatch2; class/episode/start uniform',
              'optimizer':'fresh grouped AdamW; 500 warmup from final; cosine to10000'}
    return sampler, contract


def apply_continuation_weights(engine, snapshot, *, contract, protocol, model_seed):
    if contract.get('kind') == 'scan_expanded_low_rewarm':
        from pi05_fabric.training.expanded_continuation import apply_expanded_continuation_weights
        return apply_expanded_continuation_weights(
            engine,
            snapshot,
            contract=contract,
            protocol=protocol,
            model_seed=model_seed,
        )
    if engine.current_step()!=0:
        raise ValueError('weight-only continuation requires a fresh engine')
    if snapshot.stage is not StageName.PI_NATIVE_V2_RESIDUAL_ACTION_STAGE2 or snapshot.step!=20000:
        raise ValueError('expected original Stage2-20k parent')
    if snapshot.model_seed != model_seed:
        raise ValueError('frozen base/model initialization seed changed')
    base_protocol={k:v for k,v in protocol.items() if k!='continuation'}
    if snapshot.protocol_metadata != base_protocol:
        raise ValueError('parent protocol changed')
    if parameter_sha256(snapshot.params) != contract['parent_model_sha256']:
        raise ValueError('parent parameter hash mismatch')
    current=engine.selected_params().to_pure_dict()
    def schema(tree):
        return [(jax.tree_util.keystr(p), np.shape(v), str(v.dtype))
                for p,v in jax.tree_util.tree_flatten_with_path(tree)[0]]
    if schema(current)!=schema(snapshot.params):
        raise ValueError('trainable boundary changed')
    # The newly constructed engine already owns fresh optimizer buffers; do not
    # allocate a second Adam tree or retain the parent's optimizer on the GPU.
    _replace_selected_params(engine,snapshot.params)


def validate_continuation_resume(snapshot, *, protocol, training_seed, model_seed):
    if 'low_lr_extension' in protocol['continuation']:
        from pi05_fabric.training.low_lr_extension import validate_extension_resume
        return validate_extension_resume(snapshot, protocol=protocol, training_seed=training_seed, model_seed=model_seed)
    if protocol['continuation'].get('kind') == 'scan_expanded_low_rewarm':
        from pi05_fabric.training.expanded_continuation import validate_expanded_resume
        return validate_expanded_resume(
            snapshot,
            protocol=protocol,
            training_seed=training_seed,
            model_seed=model_seed,
        )
    if snapshot.protocol_metadata != protocol:
        raise ValueError('resume continuation contract changed')
    if snapshot.training_seed != training_seed or snapshot.model_seed != model_seed:
        raise ValueError('resume RNG/model seed changed')
    if snapshot.parent_model_sha256 != protocol['continuation']['parent_model_sha256']:
        raise ValueError('resume parent lineage changed')
