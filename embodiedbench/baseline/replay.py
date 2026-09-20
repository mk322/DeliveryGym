"""M0 deterministic replay of the vendored DeliveryBench environment.

the design plan M0 acceptance: "One existing procgen text episode and one visual episode
replay twice from recorded actions with identical transition, terminal-state,
and score hashes."

Three hashes are produced per run, matching those three words:

``transition_hash``
    Folded over every step: the action, the authoritative post-step state
    digest, reward, terminated/truncated, and the error/event surface. Any
    divergence anywhere in the episode changes it.

``terminal_hash``
    The authoritative state digest after the final step alone.

``score_hash``
    The scalar outcome surface — cumulative reward, earnings, deliveries,
    success, and step count.

Visual runs additionally fold an ``observation_media_hash`` over the sha256 of
every returned image, so a cached-album regression is caught even though
design plan §7.2 does not require pixel equality between *runtimes*. Within one
runtime replayed twice, the pixels must match.

This module drives the vendored engine through its public async surface only.
It does not modify vendor/.
"""

from __future__ import annotations

import asyncio
import dataclasses
import importlib
import io
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from embodiedbench.artifacts.hashing import sha256_bytes
from embodiedbench.artifacts.state_digest import (
    ExtractionPolicy,
    ExtractionStats,
    digest_of,
    extract_state,
    state_digest,
)

VENDOR_VAGEN = Path(__file__).resolve().parents[2] / "vendor" / "vagen"


def _ensure_vendor_on_path() -> None:
    path = str(VENDOR_VAGEN)
    if path not in sys.path:
        sys.path.insert(0, path)


def load_vendor_env_module():
    """Import the vendored DeliveryBench env module without copying it."""
    _ensure_vendor_on_path()
    return importlib.import_module("vagen.envs.deliverybench.deliverybench_env")


# ─────────────────────────────────────────────────────────────────────────────
# Episode records
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class StepRecord:
    index: int
    action: str
    reward: float
    done: bool
    state_digest: str
    obs_digest: str
    media_digest: str | None
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class EpisodeRecord:
    """One recorded or replayed episode and its three acceptance hashes."""

    label: str
    map_name: str
    preset: str
    render_mode: str
    seed: int
    actions: list[str]
    steps: list[StepRecord]
    reset_state_digest: str
    transition_hash: str
    terminal_hash: str
    score_hash: str
    score: dict[str, Any]
    observation_media_hash: str | None
    extraction_stats: dict[str, Any] = field(default_factory=dict)

    def hashes(self) -> dict[str, Any]:
        return {
            "reset_state_digest": self.reset_state_digest,
            "transition_hash": self.transition_hash,
            "terminal_hash": self.terminal_hash,
            "score_hash": self.score_hash,
            "observation_media_hash": self.observation_media_hash,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "map_name": self.map_name,
            "preset": self.preset,
            "render_mode": self.render_mode,
            "seed": self.seed,
            "step_count": len(self.steps),
            "actions": self.actions,
            "hashes": self.hashes(),
            "score": self.score,
            "extraction_stats": self.extraction_stats,
            "steps": [s.to_dict() for s in self.steps],
        }


# ─────────────────────────────────────────────────────────────────────────────
# State access
# ─────────────────────────────────────────────────────────────────────────────

# The vendored engine keeps per-process handles (a run recorder, a bus manager
# holding timers) alongside genuine state. Digesting the delivery agent, the
# order manager, and the store manager captures pose, inventory, economy,
# energy, order lifecycle, and clock -- the surface design plan §7.2 names.
#
# One exclusion is needed and is stated rather than buried. It is not a
# workaround for a flaky test: the field is provably inert.
STATE_POLICY = ExtractionPolicy(
    documented_exclusions={
        "start_time": (
            "vlm_delivery/entities/order.py:117 declares "
            "`start_time: float = field(init=False, default_factory=time.time)`. "
            "It samples host wall-clock at Order construction, so two identical "
            "episodes never agree on it. A search of the whole DeliveryBench tree "
            "finds no read of Order.start_time -- it is written and never used, so "
            "it cannot influence a transition, a reward, or a termination. "
            "Excluding it keeps the digest a statement about simulation state "
            "instead of about when the process ran. Tracked as M0-F2; if a future "
            "revision starts reading it, the field must become sim-clock derived "
            "and this exclusion must be deleted."
        ),
        "run_dir": (
            "vlm_delivery/gym_like_interface/text_env.py:216-219 builds a per-episode "
            "output folder named `run_%Y%m%d_%H%M%S` from host wall-clock and stores "
            "its absolute path on the delivery agent. It is an artifact sink, not "
            "simulation state: nothing reads it to decide a transition, and the "
            "trajectory recorder only writes into it. It is also an absolute "
            "machine-specific path, which design plan §5.1 bars from portable artifacts, "
            "so it could not appear in a cross-machine conformance hash even if it "
            "were deterministic. Tracked as M0-F3."
        ),
        "map_exportor": (
            "A renderer handle the engine attaches to the delivery agent when "
            "`enable_map_images` is on (text_env.py:277,281). It is only ever "
            "invoked as `.export(...)` to produce a map image "
            "(deliverybench_env.py:1457-1460); nothing reads it to decide a "
            "transition, and it holds no task, economy, inventory, or clock state. "
            "Its presence therefore depends solely on the observation channel, so "
            "including it would make the text and cached runtimes differ by "
            "construction -- precisely the comparison design plan §7.2 excludes when it "
            "says pixels need not match while state must. Tracked as M1-F5."
        ),
        "visible_signal_views": (
            "Album-derived observation metadata: deliverybench_env.py:791 fills "
            "cfg['traffic_lights']['visible_signal_views'] from the FPV manifest's "
            "traffic-light rows, listing the (x, y, yaw) views in which a signal is "
            "actually visible. A text runtime has no album and so cannot have this "
            "by construction, which is why it appears on the cached side only.\n\n"
            "This exclusion is CONDITIONAL and narrower than the others. Unlike "
            "start_time, run_dir and map_exportor, this field is genuinely read -- "
            "traffic_lights.py:293 consults it when deciding whether a signal "
            "constrains a crossing. It is inert here only because "
            "enable_pedestrian_traffic_lights is False in every profile we run, and "
            "test_traffic_lights_are_disabled_where_this_exclusion_applies checks "
            "that precondition rather than trusting it.\n\n"
            "If traffic lights are ever enabled, text and cached genuinely diverge: "
            "the text runtime cannot know which signals are visible, so it would "
            "apply a different rule. At that point this exclusion must be deleted "
            "and one of two things done -- declare text unsupported for "
            "traffic-light configurations, or move visible_signal_views out of the "
            "album and into the map artifact so both runtimes share it. Tracked as "
            "BASELINE-F2."
        ),
    }
)


def authoritative_state(env: Any) -> dict[str, Any]:
    """The state roots that define a DeliveryBench transition."""
    inner = env._env
    dm = inner.dms[0] if inner.dms else None
    return {
        "dm": dm,
        "order_manager": inner.om,
        "store_manager": getattr(inner, "sm", None),
        "step_count": getattr(inner, "step_count", getattr(inner, "steps", None)),
        "max_steps": getattr(inner, "max_steps", None),
    }


def score_surface(env: Any, cumulative_reward: float, step_index: int) -> dict[str, Any]:
    """Scalar outcome surface used for the score hash."""
    inner = env._env
    dm = inner.dms[0] if inner.dms else None
    om = inner.om
    # The engine records completions on the delivery agent, not the order
    # manager; fall back across both so a vendor refactor is visible as a None
    # rather than a silent zero.
    delivered = None
    for holder in (dm, om):
        for attr in ("completed_orders", "delivered_orders", "finished_orders"):
            value = getattr(holder, attr, None)
            if value is not None:
                delivered = len(value) if hasattr(value, "__len__") else value
                break
        if delivered is not None:
            break
    return {
        "cumulative_reward": round(float(cumulative_reward), 9),
        "steps_taken": step_index,
        "earnings_total": round(float(getattr(dm, "earnings_total", 0.0)), 9) if dm else None,
        "energy_pct": round(float(getattr(dm, "energy_pct", 0.0)), 9) if dm else None,
        "delivered_count": delivered,
        "sim_hours": round(float(env._get_sim_hours()), 9) if hasattr(env, "_get_sim_hours") else None,
        "success": bool(env._check_success()) if hasattr(env, "_check_success") else None,
    }


def _media_digest(obs: dict[str, Any]) -> str | None:
    """sha256 over every image in an observation, in a stable order."""
    mmi = obs.get("multi_modal_input") or {}
    if not mmi:
        return None
    parts: list[str] = []
    for key in sorted(mmi):
        value = mmi[key]
        images = value if isinstance(value, list) else [value]
        for image in images:
            if hasattr(image, "tobytes"):
                buf = io.BytesIO()
                image.save(buf, format="PNG")
                parts.append(f"{key}:{sha256_bytes(buf.getvalue())}")
            else:
                parts.append(f"{key}:{sha256_bytes(repr(image).encode())}")
    return sha256_bytes("|".join(parts).encode()) if parts else None


def _obs_digest(obs: dict[str, Any]) -> str:
    """Digest of the textual observation only (images handled separately)."""
    return digest_of({k: v for k, v in obs.items() if k != "multi_modal_input"})


# ─────────────────────────────────────────────────────────────────────────────
# Drive
# ─────────────────────────────────────────────────────────────────────────────


async def run_episode(
    *,
    label: str,
    map_name: str,
    preset: str,
    render_mode: str,
    seed: int,
    max_steps: int,
    policy: Callable[[Any, dict[str, Any], int], str] | None = None,
    actions: list[str] | None = None,
    config_overrides: dict[str, Any] | None = None,
    deterministic: bool = True,
) -> EpisodeRecord:
    """Run one episode, either driven by ``policy`` or replaying ``actions``.

    Exactly one of ``policy`` / ``actions`` must be given. Replay stops early if
    the environment terminates before the recorded action list is exhausted;
    that shortfall shows up as a transition-hash mismatch rather than silently
    passing.
    """
    if (policy is None) == (actions is None):
        raise ValueError("pass exactly one of policy= or actions=")

    module = load_vendor_env_module()
    if deterministic:
        from embodiedbench.baseline.determinism import apply_deterministic_patches

        apply_deterministic_patches()
    config = dataclasses.asdict(module.PRESETS[preset])
    config.update(map_name=map_name, render_mode=render_mode, max_steps=max_steps)
    if config_overrides:
        config.update(config_overrides)

    env = module.DeliveryBench(config)
    stats = ExtractionStats()
    try:
        obs, _reset_info = await env.reset(seed=seed)
        reset_digest = sha256_bytes(
            digest_of(extract_state(authoritative_state(env), policy=STATE_POLICY, stats=stats)).encode()
        )

        steps: list[StepRecord] = []
        issued: list[str] = []
        media_parts: list[str] = []
        transition_acc = reset_digest
        cumulative_reward = 0.0
        last_state = reset_digest

        step_limit = len(actions) if actions is not None else max_steps
        for index in range(step_limit):
            action = actions[index] if actions is not None else policy(env, obs, index)
            if action is None:
                break
            issued.append(action)

            obs, reward, done, info = await env.step(action)
            cumulative_reward += float(reward)

            last_state = state_digest(
                authoritative_state(env), policy=STATE_POLICY, stats=stats
            )
            media = _media_digest(obs)
            if media:
                media_parts.append(media)

            record = StepRecord(
                index=index,
                action=action,
                reward=round(float(reward), 9),
                done=bool(done),
                state_digest=last_state,
                obs_digest=_obs_digest(obs),
                media_digest=media,
                error=(info or {}).get("error"),
            )
            steps.append(record)
            transition_acc = sha256_bytes(
                (transition_acc + "|" + digest_of(record.to_dict())).encode()
            )
            if done:
                break

        score = score_surface(env, cumulative_reward, len(steps))
        return EpisodeRecord(
            label=label,
            map_name=map_name,
            preset=preset,
            render_mode=render_mode,
            seed=seed,
            actions=issued,
            steps=steps,
            reset_state_digest=reset_digest,
            transition_hash=transition_acc,
            terminal_hash=last_state,
            score_hash=digest_of(score),
            score=score,
            observation_media_hash=(
                sha256_bytes("|".join(media_parts).encode()) if media_parts else None
            ),
            extraction_stats=stats.to_dict(),
        )
    finally:
        await env.close()


def run_episode_sync(**kwargs: Any) -> EpisodeRecord:
    return asyncio.run(run_episode(**kwargs))
