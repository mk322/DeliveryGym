"""Text and cached runtimes must agree on everything except pixels.

design plan §7.2: for a seeded action sequence the two runtimes must match on action
acceptance and error code, authoritative pose, simulation clock, task state and
inventory, economy and constraint state, event sequence, reward components, and
the terminated/truncated reason. "Pixels do not need to match."

the design plan schedules this at M5, but the cached runtime exists at R1 because the
trainer gate needs images on multiple turns, so the conformance suite comes with
it rather than after it. Conformance that is written once the two runtimes have
already diverged is conformance that gets weakened to fit.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from embodiedbench.artifacts.state_digest import diff_state
from embodiedbench.runtime.cached import VagenCachedRuntime
from embodiedbench.runtime.text import VagenTextRuntime
from embodiedbench.schemas.fixtures import _episode_spec
from embodiedbench.schemas.runtime import ActionEnvelope, RuntimeMode, TaskAction

REPO_ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(
    not (REPO_ROOT / "vendor" / "vagen").exists(),
    reason="vendored VAGEN checkout not present",
)

MAP = "small-city-11"
SEED = 42

TRACE: list[tuple[str, dict]] = [
    ("VIEW_ORDERS", {}),
    ("ACCEPT_ORDER", {"_args": [0]}),
    ("MOVE", {"direction": "forward"}),
    ("MOVE", {"direction": "left"}),
    ("MOVE", {"direction": "forward"}),
    ("WAIT", {}),
    ("PICKUP", {"orders": [0]}),
    ("MOVE", {"direction": "backward"}),
]


def _pair(tmp_path):
    instance = _episode_spec().model_copy(update={"seed": SEED})
    text = VagenTextRuntime(map_name=MAP, max_steps=40)
    cached = VagenCachedRuntime(map_name=MAP, media_root=tmp_path / "media", max_steps=40)
    return instance, text, cached


def test_state_agrees_at_reset(tmp_path):
    instance, text, cached = _pair(tmp_path)
    try:
        text.reset(instance)
        cached.reset(instance)
        differences = diff_state(text.state_tree(), cached.state_tree(), limit=10)
        assert differences == [], differences
    finally:
        text.close()
        cached.close()


def test_full_trace_conformance(tmp_path):
    """The design plan §7.2 comparison, run over a seeded action sequence."""
    instance, text, cached = _pair(tmp_path)
    try:
        text.reset(instance)
        cached.reset(instance)
        assert text.authoritative_state_digest() == cached.authoritative_state_digest()

        for index, (name, arguments) in enumerate(TRACE):
            envelope = ActionEnvelope(
                episode_id=instance.instance_id,
                step_index=index,
                action=TaskAction(name=name, arguments=arguments),
            )
            text_result = text.step(envelope)
            cached_result = cached.step(envelope)

            # action acceptance and error code
            assert text_result.action_result.status is cached_result.action_result.status
            assert text_result.action_result.error_code == cached_result.action_result.error_code
            # authoritative state, inventory, economy, clock
            differences = diff_state(text.state_tree(), cached.state_tree(), limit=8)
            assert differences == [], f"step {index}: {differences}"
            # reward and its components
            assert text_result.reward == pytest.approx(cached_result.reward)
            assert text_result.reward_components == pytest.approx(cached_result.reward_components)
            # termination
            assert text_result.terminated == cached_result.terminated
            assert text_result.truncated == cached_result.truncated
            assert text_result.termination_reason == cached_result.termination_reason
            # event sequence
            assert [e.kind for e in text_result.events] == [e.kind for e in cached_result.events]
            assert [e.event_id for e in text_result.events] == [
                e.event_id for e in cached_result.events
            ]
            if text_result.terminated or text_result.truncated:
                break
    finally:
        text.close()
        cached.close()


def test_pose_and_clock_agree_exactly(tmp_path):
    """In-process, the same engine should be exact, not merely within tolerance."""
    instance, text, cached = _pair(tmp_path)
    try:
        text.reset(instance)
        cached.reset(instance)
        for index, (name, arguments) in enumerate(TRACE[:5]):
            envelope = ActionEnvelope(
                episode_id=instance.instance_id,
                step_index=index,
                action=TaskAction(name=name, arguments=arguments),
            )
            a = text.step(envelope).observation.agent_pose
            b = cached.step(envelope).observation.agent_pose
            assert a is not None and b is not None
            assert a.position.distance_cm(b.position) == 0.0
            assert a.yaw_difference_deg(b) == 0.0
        assert text._current_sim_time() == cached._current_sim_time()
    finally:
        text.close()
        cached.close()


# ─────────────────────────────────────────────────────────────────────────────
# What the cached runtime adds
# ─────────────────────────────────────────────────────────────────────────────


def test_cached_runtime_declares_itself_and_its_channels(tmp_path):
    instance, text, cached = _pair(tmp_path)
    try:
        _observation, info = cached.reset(instance)
        assert cached.capabilities.mode is RuntimeMode.CACHED
        assert "rgb" in cached.capabilities.observation_channels
        assert info.runtime_mode is RuntimeMode.CACHED
        assert text.capabilities.mode is RuntimeMode.TEXT
    finally:
        text.close()
        cached.close()


def test_images_appear_on_every_turn(tmp_path):
    """design plan §11.1's gate needs images on multiple environment turns."""
    instance, text, cached = _pair(tmp_path)
    try:
        observation, _ = cached.reset(instance)
        assert observation.media, "reset produced no image"
        for index, (name, arguments) in enumerate(TRACE[:4]):
            result = cached.step(
                ActionEnvelope(
                    episode_id=instance.instance_id,
                    step_index=index,
                    action=TaskAction(name=name, arguments=arguments),
                )
            )
            assert result.observation.media, f"step {index} produced no image"
            assert len(cached.last_images) == len(result.observation.media)
    finally:
        text.close()
        cached.close()


def test_media_refs_are_relative_and_content_addressed(tmp_path):
    instance, text, cached = _pair(tmp_path)
    try:
        observation, _ = cached.reset(instance)
        for ref in observation.media:
            assert not ref.path.startswith("/")
            assert ref.sha256 in ref.path
            assert cached.media.path_for(ref.path).exists()
    finally:
        text.close()
        cached.close()


def test_repeated_frames_are_deduplicated(tmp_path):
    """A revisited pose must not write the same frame twice."""
    instance, text, cached = _pair(tmp_path)
    try:
        cached.reset(instance)
        for index in range(6):
            cached.step(
                ActionEnvelope(
                    episode_id=instance.instance_id,
                    step_index=index,
                    action=TaskAction(name="WAIT"),
                )
            )
        stats = cached.media_stats()
        assert stats["deduplicated"] > 0, stats
        assert stats["distinct_images"] == stats["writes"]
    finally:
        text.close()
        cached.close()


def test_text_runtime_produces_no_media(tmp_path):
    instance, text, cached = _pair(tmp_path)
    try:
        observation, _ = text.reset(instance)
        assert observation.media == []
    finally:
        text.close()
        cached.close()


def test_traffic_lights_are_disabled_where_this_exclusion_applies():
    """The precondition for excluding ``visible_signal_views`` from the digest.

    That field is album-derived and genuinely read by the traffic-light rules
    (traffic_lights.py:293), so excluding it is only safe while traffic lights
    are off. If a profile ever turns them on, text and cached would diverge for
    a real reason and the exclusion would be hiding it -- so the precondition is
    asserted here rather than assumed.
    """
    import dataclasses

    from embodiedbench.baseline.replay import load_vendor_env_module

    module = load_vendor_env_module()
    for preset_name in ("nav",):
        config = dataclasses.asdict(module.PRESETS[preset_name])
        assert config.get("enable_pedestrian_traffic_lights") is False, (
            f"preset {preset_name!r} enables pedestrian traffic lights; the "
            "visible_signal_views digest exclusion is no longer safe (BASELINE-F2)"
        )


def test_cached_runtime_resolves_the_real_album(tmp_path):
    """The album the EnvSpec advertises must be the one the runtime loads.

    The first zero-shot baseline ran with vision nominally on and received only
    the map image, because the engine was pointed at the album root's 12-row
    stub manifest instead of the 665-row one below it.
    """
    from embodiedbench.compiler.env_spec_builder import find_album
    if not find_album(MAP).get("found"):
        pytest.skip(f"the {MAP} first-person album (deliverybench_fpv) is not present here")
    instance, text, cached = _pair(tmp_path)
    try:
        assert cached.album.get("found")
        assert cached.album["waypoints"] > 100, cached.album
        observation, _info = cached.reset(instance)
        # Both a first-person frame and a map, not just the map.
        assert len(observation.media) == 2, [m.width_px for m in observation.media]
        sizes = {(m.width_px, m.height_px) for m in observation.media}
        assert any(w == 640 for w, _h in sizes), sizes
    finally:
        text.close()
        cached.close()
