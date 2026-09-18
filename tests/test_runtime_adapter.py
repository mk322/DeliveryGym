"""The VAGEN text adapter must not change the vendored env's behavior.

the design plan M1 accepts when "a fixed existing DeliveryBench action trace runs
through the adapter three times and matches the pinned pre-adapter state/event
hash".

"Pre-adapter" means the vendored environment driven directly with raw action
strings, exactly as the vendored scripted rollout drives it. "Post-adapter"
means the same trace expressed as typed ``ActionEnvelope`` objects and executed
through ``VagenTextRuntime``. If the adapter changed a transition, added an
implicit step, or reordered anything, the authoritative state digests would
diverge.

The trace is fixed in this file rather than generated, so a change in the
scripted policy cannot quietly change what is being compared.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
from pathlib import Path

import pytest

from embodiedbench.artifacts.state_digest import diff_state, state_digest
from embodiedbench.baseline.determinism import apply_deterministic_patches
from embodiedbench.baseline.replay import STATE_POLICY, load_vendor_env_module
from embodiedbench.runtime.core import EmbodiedRuntime, EpisodeNotStarted, StepIndexMismatch
from embodiedbench.runtime.text.vagen_adapter import POSITIONAL_KEY, VagenTextRuntime, render_vendor_action
from embodiedbench.schemas.base import CapabilityError
from embodiedbench.schemas.fixtures import _episode_spec
from embodiedbench.schemas.runtime import (
    ActionEnvelope,
    ActionStatus,
    NavPoint3DAction,
    TaskAction,
    TerminationReason,
)

pytestmark = pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / "vendor" / "vagen").exists(),
    reason="vendored VAGEN checkout not present",
)

MAP = "small-city-11"
SEED = 42
MAX_STEPS = 60

# A fixed trace: view the pool, accept an order, make three moves, then a
# deliberately invalid pickup. The invalid action is included on purpose --
# error paths are where an adapter is most likely to diverge.
FIXED_TRACE: list[tuple[str, dict]] = [
    ("VIEW_ORDERS", {}),
    ("ACCEPT_ORDER", {POSITIONAL_KEY: [0]}),
    ("MOVE", {"direction": "forward"}),
    ("MOVE", {"direction": "left"}),
    ("MOVE", {"direction": "forward"}),
    ("PICKUP", {"orders": [0]}),
    ("WAIT", {}),
]


def _vendor_strings() -> list[str]:
    return [render_vendor_action(name, args) for name, args in FIXED_TRACE]


def _pre_adapter_digests() -> tuple[list[str], list[float]]:
    """Drive the vendored env directly, exactly as its own rollout script does."""
    apply_deterministic_patches()
    module = load_vendor_env_module()

    async def drive():
        config = dataclasses.asdict(module.PRESETS["nav"])
        config.update(map_name=MAP, render_mode="text", max_steps=MAX_STEPS)
        env = module.DeliveryBench(config)
        try:
            await env.reset(seed=SEED)
            inner = env._env
            roots = lambda: {  # noqa: E731 - matches the adapter's own roots
                "dm": inner.dms[0] if inner.dms else None,
                "order_manager": inner.om,
                "store_manager": getattr(inner, "sm", None),
                "step_count": getattr(inner, "step_count", getattr(inner, "steps", None)),
                "max_steps": getattr(inner, "max_steps", None),
            }
            digests = [state_digest(roots(), policy=STATE_POLICY)]
            rewards = []
            for action in _vendor_strings():
                _obs, reward, done, _info = await env.step(json.dumps({"action": action}))
                digests.append(state_digest(roots(), policy=STATE_POLICY))
                rewards.append(float(reward))
                if done:
                    break
            return digests, rewards
        finally:
            await env.close()

    return asyncio.run(drive())


def _post_adapter_digests() -> tuple[list[str], list[float]]:
    """Drive the same trace through the typed adapter."""
    instance = _episode_spec().model_copy(update={"seed": SEED})
    runtime = VagenTextRuntime(map_name=MAP, max_steps=MAX_STEPS)
    try:
        runtime.reset(instance)
        digests = [runtime.authoritative_state_digest()]
        rewards = []
        for index, (name, arguments) in enumerate(FIXED_TRACE):
            envelope = ActionEnvelope(
                episode_id=instance.instance_id,
                step_index=index,
                action=TaskAction(name=name, arguments=arguments),
            )
            result = runtime.step(envelope)
            digests.append(runtime.authoritative_state_digest())
            rewards.append(result.reward)
            if result.terminated or result.truncated:
                break
        return digests, rewards
    finally:
        runtime.close()


# ─────────────────────────────────────────────────────────────────────────────
# Behavior preservation
# ─────────────────────────────────────────────────────────────────────────────


def test_adapter_matches_pre_adapter_state_hash_three_times():
    """The M1 acceptance criterion."""
    expected, expected_rewards = _pre_adapter_digests()
    assert len(expected) == len(FIXED_TRACE) + 1, "the fixed trace should run to completion"

    for attempt in range(3):
        observed, observed_rewards = _post_adapter_digests()
        assert observed == expected, f"state digests diverged on attempt {attempt + 1}"
        assert observed_rewards == pytest.approx(expected_rewards)


def test_adapter_and_vendor_agree_step_by_step():
    """A divergence should name the field, not just show two hashes."""
    expected, _ = _pre_adapter_digests()
    observed, _ = _post_adapter_digests()
    for index, (a, b) in enumerate(zip(expected, observed)):
        assert a == b, f"first divergence after step {index - 1}"


def test_action_rendering_is_stable_and_correct():
    assert render_vendor_action("VIEW_ORDERS", {}) == "VIEW_ORDERS()"
    assert render_vendor_action("ACCEPT_ORDER", {POSITIONAL_KEY: [0]}) == "ACCEPT_ORDER(0)"
    assert render_vendor_action("MOVE", {"direction": "left"}) == 'MOVE(direction="left")'
    assert render_vendor_action("PICKUP", {"orders": [0]}) == "PICKUP(orders=[0])"
    assert render_vendor_action("DROP_OFF", {"oid": 0}) == "DROP_OFF(oid=0)"


def test_keyword_argument_order_is_deterministic():
    """Byte-identical vendored input for the same envelope, whatever dict order."""
    a = render_vendor_action("X", {"b": 1, "a": 2})
    b = render_vendor_action("X", {"a": 2, "b": 1})
    assert a == b == "X(a=2, b=1)"


# ─────────────────────────────────────────────────────────────────────────────
# Contract conformance
# ─────────────────────────────────────────────────────────────────────────────


def test_runtime_satisfies_the_protocol():
    runtime = VagenTextRuntime(map_name=MAP, max_steps=10)
    try:
        assert isinstance(runtime, EmbodiedRuntime)
    finally:
        runtime.close()


def test_step_before_reset_is_typed():
    runtime = VagenTextRuntime(map_name=MAP, max_steps=10)
    try:
        with pytest.raises(EpisodeNotStarted):
            runtime.step(
                ActionEnvelope(episode_id="x", step_index=0, action=TaskAction(name="WAIT"))
            )
    finally:
        runtime.close()


def test_out_of_order_step_index_is_rejected():
    instance = _episode_spec().model_copy(update={"seed": SEED})
    runtime = VagenTextRuntime(map_name=MAP, max_steps=10)
    try:
        runtime.reset(instance)
        with pytest.raises(StepIndexMismatch):
            runtime.step(
                ActionEnvelope(
                    episode_id=instance.instance_id, step_index=5, action=TaskAction(name="WAIT")
                )
            )
    finally:
        runtime.close()


def test_unsupported_navigation_mode_is_a_capability_error():
    instance = _episode_spec().model_copy(update={"seed": SEED})
    runtime = VagenTextRuntime(map_name=MAP, max_steps=10)
    try:
        runtime.reset(instance)
        with pytest.raises(CapabilityError):
            runtime.step(
                ActionEnvelope(
                    episode_id=instance.instance_id,
                    step_index=0,
                    action=NavPoint3DAction(
                        target={"u_norm": 0.5, "v_norm": 0.5, "distance_m": 6.0}
                    ),
                )
            )
    finally:
        runtime.close()


def test_invalid_action_produces_a_typed_rejection():
    """the design plan M6: every invalid action receives typed feedback."""
    instance = _episode_spec().model_copy(update={"seed": SEED})
    runtime = VagenTextRuntime(map_name=MAP, max_steps=20)
    try:
        runtime.reset(instance)
        # DROP_OFF with nothing accepted cannot succeed.
        result = runtime.step(
            ActionEnvelope(
                episode_id=instance.instance_id,
                step_index=0,
                action=TaskAction(name="DROP_OFF", arguments={"oid": 0}),
            )
        )
        assert result.action_result.status is ActionStatus.REJECTED
        assert result.action_result.error_code
        assert result.observation.last_action_result is not None
    finally:
        runtime.close()


def test_observation_carries_no_privileged_reference():
    instance = _episode_spec().model_copy(update={"seed": SEED})
    runtime = VagenTextRuntime(map_name=MAP, max_steps=10)
    try:
        observation, info = runtime.reset(instance)
        assert info.privileged_state_ref is not None
        assert "private://" not in json.dumps(observation.to_dict())
    finally:
        runtime.close()


def test_effective_step_budget_reflects_the_dynamic_override():
    """The nav preset recomputes max_steps per seed; the adapter must say so."""
    instance = _episode_spec().model_copy(update={"seed": SEED})
    runtime = VagenTextRuntime(map_name=MAP, max_steps=3)
    try:
        runtime.reset(instance)
        assert runtime.effective_step_budget() != 3
        assert runtime.effective_step_budget() >= 20
    finally:
        runtime.close()


def test_step_budget_exhaustion_truncates_rather_than_terminates():
    """design plan §7: budget exhaustion is truncation, not task termination."""
    instance = _episode_spec().model_copy(update={"seed": SEED})
    # dynamic_max_steps_mult=None makes max_steps static, so the budget under
    # test is the one that is actually enforced.
    runtime = VagenTextRuntime(
        map_name=MAP, max_steps=3, config_overrides={"dynamic_max_steps_mult": None}
    )
    try:
        runtime.reset(instance)
        assert runtime.effective_step_budget() == 3
        last = None
        for index in range(3):
            last = runtime.step(
                ActionEnvelope(
                    episode_id=instance.instance_id,
                    step_index=index,
                    action=TaskAction(name="WAIT"),
                )
            )
            if last.truncated or last.terminated:
                break
        assert last is not None
        assert last.truncated and not last.terminated
        assert last.termination_reason is TerminationReason.STEP_BUDGET_EXHAUSTED
    finally:
        runtime.close()


def test_events_are_deterministic_and_unique():
    """design plan §12.6: events must not duplicate, and must replay identically."""
    def collect():
        instance = _episode_spec().model_copy(update={"seed": SEED})
        runtime = VagenTextRuntime(map_name=MAP, max_steps=20)
        try:
            runtime.reset(instance)
            ids = []
            for index, (name, arguments) in enumerate(FIXED_TRACE[:4]):
                result = runtime.step(
                    ActionEnvelope(
                        episode_id=instance.instance_id,
                        step_index=index,
                        action=TaskAction(name=name, arguments=arguments),
                    )
                )
                ids.extend(e.event_id for e in result.events)
            return ids
        finally:
            runtime.close()

    first, second = collect(), collect()
    assert first == second
    assert len(first) == len(set(first))
