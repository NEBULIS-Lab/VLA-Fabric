import hashlib

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as model_api
from openpi.models import pi0_config

from pi05_fabric.agents.dual_pi05 import DualPi05
from pi05_fabric.agents.dual_pi05 import DualPi05Mode
from pi05_fabric.agents.dual_pi05 import SharedInferenceMode
from pi05_fabric.agents.dual_pi05 import resolve_shared_interaction
from pi05_fabric.agents.dual_pi05 import shared_effective_channels


def _hash_state(model) -> str:
    digest = hashlib.sha256()
    for path, value in nnx.state(model, nnx.Param).flat_state().items():
        digest.update("/".join(str(item) for item in path).encode())
        array = np.asarray(value.value)
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _toy_pair():
    config = pi0_config.Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=2,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
    )
    left = config.create(jax.random.key(1))
    right = config.create(jax.random.key(2))
    pair = DualPi05(
        left,
        right,
        mode=DualPi05Mode.PI_NATIVE_V2_RESIDUAL_ACTION,
        rngs=nnx.Rngs(3),
    )
    observation = model_api.preprocess_observation(
        None, config.fake_obs(batch_size=1), train=False
    )
    prefix, _, _ = left.embed_prefix(observation)
    ownership = tuple(np.zeros(prefix.shape[1], dtype=np.int8))
    noise = jnp.zeros((1, 2, 32), dtype=jnp.float32)
    return pair, observation, ownership, noise


def test_shared_modes_resolve_three_effective_channels() -> None:
    full = resolve_shared_interaction(SharedInferenceMode.FULL)
    all_off = resolve_shared_interaction(SharedInferenceMode.ALL_OFF)

    assert (full.common, full.private_kv) == (True, True)
    assert (all_off.common, all_off.private_kv) == (False, False)
    assert shared_effective_channels(SharedInferenceMode.FULL) == {
        "common": True,
        "private_kv": True,
        "action": True,
    }
    assert shared_effective_channels(SharedInferenceMode.ALL_OFF) == {
        "common": False,
        "private_kv": False,
        "action": False,
    }


def test_full_and_all_off_use_one_parameter_tree_and_return_finite_actions() -> None:
    pair, observation, ownership, noise = _toy_pair()
    before = _hash_state(pair)

    full = pair.sample_actions(
        jax.random.key(4),
        observation,
        observation,
        ownership=ownership,
        mode=DualPi05Mode.PI_NATIVE_V2_RESIDUAL_ACTION,
        interaction_override=resolve_shared_interaction(SharedInferenceMode.FULL),
        num_steps=1,
        left_noise=noise,
        right_noise=noise,
        preprocessed=True,
    )
    all_off = pair.sample_actions(
        jax.random.key(4),
        observation,
        observation,
        ownership=ownership,
        mode=DualPi05Mode.PI_NATIVE_V2_RESIDUAL_ACTION,
        interaction_override=resolve_shared_interaction(SharedInferenceMode.ALL_OFF),
        num_steps=1,
        left_noise=noise,
        right_noise=noise,
        preprocessed=True,
    )

    assert _hash_state(pair) == before
    for actions in (*full, *all_off):
        assert actions.shape == (1, 2, 32)
        assert bool(jnp.isfinite(actions).all())
