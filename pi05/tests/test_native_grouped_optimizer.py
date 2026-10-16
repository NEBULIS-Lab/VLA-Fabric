from __future__ import annotations

import jax.numpy as jnp
import optax
from flax import nnx
import jax

from openpi.models import pi0_config
from openpi.models import model as model_api

from pi05_fabric.agents.dual_pi05 import DualPi05
from pi05_fabric.agents.dual_pi05 import DualPi05Mode

from pi05_fabric.agents.pi05_strong import ADAPTER_GROUP
from pi05_fabric.agents.pi05_strong import ACTION_EXPERT_GROUP
from pi05_fabric.agents.pi05_strong import PALIGEMMA_GROUP
from pi05_fabric.training.optimizer import GroupedOptimizerSettings
from pi05_fabric.training.optimizer import create_grouped_optimizer
from pi05_fabric.training.engine import create_native_engine
from pi05_fabric.training.engine import parameter_values


def test_grouped_optimizer_updates_each_group_and_clips_once():
    params = {
        "action": jnp.asarray([1.0]),
        "paligemma": jnp.asarray([1.0]),
        "adapter": jnp.asarray([1.0]),
    }
    labels = {
        "action": ACTION_EXPERT_GROUP,
        "paligemma": PALIGEMMA_GROUP,
        "adapter": ADAPTER_GROUP,
    }
    settings = GroupedOptimizerSettings(
        total_steps=10,
        warmup_steps=2,
        action_expert_peak=5e-6,
        action_expert_final=5e-7,
        paligemma_peak=3e-6,
        paligemma_final=3e-7,
        adapter_peak=5e-5,
        adapter_final=5e-6,
    )
    tx = create_grouped_optimizer(settings, labels)
    updates, _ = tx.update(
        {name: jnp.asarray([100.0]) for name in params},
        tx.init(params),
        params,
    )
    updated = optax.apply_updates(params, updates)

    assert all(float(updated[name][0]) < 1.0 for name in params)
    assert abs(float(updates["adapter"][0])) > abs(float(updates["action"][0]))


def test_expanded_optimizer_keeps_five_distinct_learning_rate_groups():
    params = {
        "action": jnp.asarray([1.0]),
        "action_ffw": jnp.asarray([1.0]),
        "paligemma": jnp.asarray([1.0]),
        "paligemma_qo": jnp.asarray([1.0]),
        "adapter": jnp.asarray([1.0]),
    }
    labels = {
        "action": ACTION_EXPERT_GROUP,
        "action_ffw": "action_expert_ffw_base",
        "paligemma": PALIGEMMA_GROUP,
        "paligemma_qo": "paligemma_qo_base",
        "adapter": ADAPTER_GROUP,
    }
    settings = GroupedOptimizerSettings(
        total_steps=10_000,
        warmup_steps=500,
        action_expert_peak=1.5e-6,
        action_expert_final=5e-7,
        action_ffw_peak=5e-7,
        action_ffw_final=1e-7,
        paligemma_peak=9e-7,
        paligemma_final=3e-7,
        paligemma_qo_peak=3e-7,
        paligemma_qo_final=5e-8,
        adapter_peak=2e-5,
        adapter_final=5e-6,
    )
    groups = settings.group_settings()

    assert set(groups) == set(labels.values())
    assert groups["action_expert_ffw_base"].peak_learning_rate == 5e-7
    assert groups["paligemma_qo_base"].peak_learning_rate == 3e-7

    tx = create_grouped_optimizer(settings, labels)
    updates, _ = tx.update(
        {name: jnp.asarray([1.0]) for name in params},
        tx.init(params),
        params,
    )
    magnitudes = {name: abs(float(value[0])) for name, value in updates.items()}
    assert magnitudes["adapter"] > magnitudes["action"]
    assert magnitudes["action"] > magnitudes["action_ffw"]
    assert magnitudes["paligemma"] > magnitudes["paligemma_qo"]


def test_grouped_optimizer_supports_optax_0_2_4_api(monkeypatch):
    monkeypatch.delattr(optax, "partition", raising=False)
    params = {"action": jnp.asarray([1.0])}
    labels = {"action": ACTION_EXPERT_GROUP}
    tx = create_grouped_optimizer(
        GroupedOptimizerSettings(total_steps=10, warmup_steps=2),
        labels,
    )

    updates, _ = tx.update(
        {"action": jnp.asarray([1.0])},
        tx.init(params),
        params,
    )

    assert float(updates["action"][0]) < 0


def test_native_engine_selects_strong_boundary_without_linear_action():
    config = pi0_config.Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=2,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
    )
    model = DualPi05(
        config.create(jax.random.key(1)),
        config.create(jax.random.key(2)),
        mode=DualPi05Mode.PI_NATIVE_THREE_PATH,
        rngs=nnx.Rngs(3),
    )
    settings = GroupedOptimizerSettings(total_steps=10, warmup_steps=2)

    engine = create_native_engine(
        model,
        mode=DualPi05Mode.PI_NATIVE_THREE_PATH,
        settings=settings,
    )
    paths = set(parameter_values(engine.selected_params()))

    assert any("kv_einsum/w" in path and "kv_einsum_1" not in path for path in paths)
    assert any("q_einsum_1/w" in path for path in paths)
    assert any("mlp_1/gating_einsum" in path for path in paths)
    assert any("action_out_proj/kernel" in path for path in paths)
    assert all("fabric_linear_action" not in path for path in paths)

    observation = model_api.Observation(
        images={},
        image_masks={},
        state=jnp.zeros((1, 32), dtype=jnp.float32),
        tokenized_prompt=jnp.ones((1, 4), dtype=jnp.int32),
        tokenized_prompt_mask=jnp.ones((1, 4), dtype=bool),
    )
    actions = config.fake_act(batch_size=1)
    ownership = jnp.asarray([[0, 0, 1, 1]], dtype=jnp.int8)
    engine.state, info = engine.step(
        engine.state,
        jax.random.key(4),
        observation,
        observation,
        actions,
        actions,
        ownership,
        preprocessed=True,
    )
    assert engine.current_step() == 1
    assert bool(jnp.asarray(info.finite).all())
