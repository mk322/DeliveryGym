"""Turn a courier episode into training samples.

The training package had everything except a way to reach *this* environment.
``core.py`` holds the mask and reward invariants, ``policy_update.py`` does a
masked REINFORCE step, and ``r1_gate.py`` proved both against the old vagen
runtime with a ``MOVE_TO``/``PICKUP`` action space. None of it could see
``CourierEnv``, so the environment the benchmark actually ships was the one
thing nothing could train on.

This is that adapter, and it is deliberately thin. It drives the *normal*
harness -- ``CourierSession`` renders the prompt and dispatches the reply, the
same code a scored evaluation runs -- and records token provenance as it goes.
That matters more than it sounds: if training drove a different loop from
evaluation, a policy would be optimised against a prompt it is never scored on.

What it must get right, and what the tests pin:

* **the mask.** Only the tokens the policy sampled carry gradient. The
  observation, the photographs, and the chat template's own markers are all
  appended as non-assistant tokens by the adapter, so the mask comes from
  provenance rather than from re-scanning text for delimiters.
* **the reward.** ``CourierRun.total_reward`` is the episode return: +1.0 a
  delivery, ±0.5 for punctuality, +0.1 a collection, −1.0 for crossing on red.
  Splitting one episode into shards must preserve it exactly once, which
  ``check_reward_conservation`` verifies.
* **the images.** A courier turn can carry nine pictures -- four street views,
  four pedestrian lamps and the phone's map. Every one of them is image tokens
  in the context, so the count is capped and the cap is reported rather than
  silently applied.

The adapter is passed in rather than constructed. Any object with
``start_episode``/``observe``/``generate``/``close_assistant_turn`` works, which
is what lets the tests run the whole path -- masks, reward accounting, fan-out --
on a stub, with no model and no GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from embodiedbench.training.core import (
    TrainingSample,
    build_training_sample,
    check_reward_conservation,
    split_for_fanout,
)
from embodiedbench.training.image_groups import (
    required_image_groups,
    validate_atomic_image_capacity,
)

# A turn can offer nine pictures. Qwen3-VL spends on the order of a few hundred
# tokens on each, so an unbounded turn is how a rollout runs out of context in
# the middle of an episode. The street views come first because they are the
# only place a barrier appears; the lamps and the map follow.
DEFAULT_MAX_IMAGES = 5

# One "unit" of shaped progress. A city block on this map is on the order of
# 60-110 m, so 100 m makes a typical good block worth about the same as the
# format term and a delivery still worth ten of them.
PROGRESS_SCALE_CM = 10_000.0


class RolloutAdapter(Protocol):
    """What a rollout needs from a model. Deliberately small."""

    def start_episode(self, system_prompt: str) -> Any: ...
    def observe(self, state: Any, text: str, images: list[Any] | None = None) -> Any: ...
    def generate(self, state: Any, *, max_new_tokens: int = 64,
                 temperature: float = 0.0, seed: int | None = None) -> tuple[str, Any]: ...
    def close_assistant_turn(self, state: Any) -> None: ...


@dataclass
class CourierRollout:
    """One episode: the token sequence, what happened in it, and the samples."""

    episode_id: str
    state: Any
    samples: list[TrainingSample]
    reward: float                 # what the policy is optimised against
    env_return: float             # what the benchmark scores. Never shaped.
    format_score: float
    progress_score: float         # net metres closed on the target, in units
    summary: dict[str, Any]
    turns: int = 0
    format_errors: int = 0
    rejected: int = 0
    images_shown: int = 0
    images_dropped: int = 0
    turn_rewards: list[float] = field(default_factory=list)

    def token_report(self) -> dict[str, int]:
        return self.state.summary() if hasattr(self.state, "summary") else {}


def _load_images(observation: Any, *, max_images: int, scratch: Path | None) -> tuple[list[Any], int]:
    """The pictures for one turn, as PIL images, in caption order.

    Street views first, then lamps, then the map -- the order the captions are
    written in, so a model matching captions to images by position is not misled
    by the trainer. The map is SVG and has to be rasterised; if that is not
    possible it is dropped rather than sent as text, because a drawing described
    in words is a different observation from a drawing.
    """
    from PIL import Image

    validate_atomic_image_capacity(observation, max_images)
    groups = required_image_groups(observation)
    required_ids = {id(frame) for frames in groups.values() for frame in frames}
    required = [frame for frame in observation.frames if id(frame) in required_ids]
    photographs = required + [
        frame for frame in observation.frames
        if (id(frame) not in required_ids
            and frame.kind == "photograph" and frame.path)
    ]
    drawings = [f for f in observation.frames if f.kind == "map" and f.svg]

    chosen: list[Any] = []
    dropped = 0
    for frame in photographs:
        if len(chosen) >= max_images:
            dropped += 1
            continue
        try:
            chosen.append(Image.open(frame.path).convert("RGB"))
        except Exception as exc:  # noqa: BLE001 - required frames fail the turn
            if id(frame) in required_ids:
                raise RuntimeError(
                    f"required image {frame.view_id} could not be loaded"
                ) from exc
            dropped += 1
    for frame in drawings:
        if len(chosen) >= max_images or scratch is None:
            dropped += 1
            continue
        try:
            import cairosvg

            out = scratch / f"map_{len(chosen)}.png"
            cairosvg.svg2png(bytestring=frame.svg.encode(), write_to=str(out),
                             output_width=720, output_height=540)
            chosen.append(Image.open(out).convert("RGB"))
        except Exception:  # noqa: BLE001
            dropped += 1
    return chosen, dropped


def _remaining_cm(env: Any) -> tuple[float | None, str | None]:
    """How far the courier still has to walk, and what it is walking to.

    Privileged information -- ``route_length_cm`` is the environment's own
    bookkeeping and never appears in an observation. It is legitimate for a
    *training* signal for the same reason a simulator may compute a reward it
    does not show: the policy is scored on ``env_return``, which this never
    touches.
    """
    order = env.active_order()
    if order is None:
        return None, None
    target = order.target.kerb_node
    return env.route_length_cm(env.node_id, target), target


def rollout_courier_episode(
    adapter: RolloutAdapter,
    env: Any,
    *,
    episode_id: str,
    city: str = "Paris",
    max_turns: int = 40,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    seed: int | None = None,
    format_weight: float = 0.0,
    progress_weight: float = 0.0,
    max_images: int = DEFAULT_MAX_IMAGES,
    scratch: Path | None = None,
    shards: int = 1,
) -> CourierRollout:
    """Run one episode through the real harness and return training samples.

    ``env`` must already be ``reset()``. The caller owns it, because a trainer
    wants to choose the tier, the seed and the embodiment.
    """
    from embodiedbench.agent.courier.session import CourierSession

    session = CourierSession(env, city=city)
    state = adapter.start_episode(session.system_prompt())

    turns = format_errors = rejected = shown = dropped = 0
    turn_rewards: list[float] = []
    progress_cm = 0.0

    for turn in range(max_turns):
        if session.finished:
            break
        before_cm, before_target = _remaining_cm(env)
        observation = session.observe()
        images, lost = _load_images(observation, max_images=max_images, scratch=scratch)
        shown += len(images)
        dropped += lost

        adapter.observe(state, observation.text, images)
        reply, _ = adapter.generate(
            state, max_new_tokens=max_new_tokens, temperature=temperature,
            seed=None if seed is None else seed + turn,
        )
        adapter.close_assistant_turn(state)

        log = session.step(reply)
        turns += 1
        turn_rewards.append(float(log.reward or 0.0))

        # Potential-based shaping, accumulated per turn rather than measured
        # end-to-end. The target switches from the pickup to the dropoff the
        # moment a parcel is collected, and the two are far apart, so an
        # end-to-end difference would book that switch as a huge gain or loss
        # that the policy did not earn. Turns where the target changed are
        # skipped; the collection itself is already worth +0.1 in env_return.
        after_cm, after_target = _remaining_cm(env)
        if (before_cm is not None and after_cm is not None
                and before_target == after_target):
            progress_cm += before_cm - after_cm
        if log.status == "format_error":
            format_errors += 1
        elif log.status == "rejected":
            rejected += 1

    env_return = float(session.run.total_reward)
    # How often the model emitted something the world could execute at all.
    format_score = (turns - format_errors) / turns if turns else 0.0

    # The training objective, which is NOT the benchmark score.
    #
    # A courier that never delivers gets a return of exactly 0.0 on every seed,
    # so a batch has no variance, every advantage is zero, and REINFORCE has
    # nothing to push on -- measured on Qwen2-VL-2B: four seeds, four zeros.
    # The task reward is too sparse to bootstrap a small model from.
    #
    # ``format_weight`` adds a dense term for emitting an executable action.
    # That is a real part of the job -- a courier that cannot express a move
    # cannot make one -- and it is the standard first rung of a curriculum. It
    # is kept in a separate field from ``env_return`` and defaults to off, so a
    # shaped run can never be quoted as a benchmark score by accident.
    #
    # ``progress_weight`` adds the second rung. Qwen3-VL-4B scores 1.0 on the
    # format term before any training -- it has nothing left to learn there, so
    # every episode in a batch earns the same, every advantage is zero and the
    # optimiser is handed a no-op. Distance closed on the target is the next
    # signal that is dense enough to have variance and still points at the job.
    # It is potential-based (Ng, Harada & Russell 1999), so it cannot change
    # which policy is optimal -- it only says *earlier* what the delivery would
    # have said later.
    progress_score = progress_cm / PROGRESS_SCALE_CM
    reward = (env_return + format_weight * format_score
              + progress_weight * progress_score)
    summary = env.summary()
    metadata = {
        "tier": summary.get("difficulty"),
        "embodiment": summary.get("embodiment"),
        "viewpoint_served": summary.get("viewpoint_served"),
        "viewpoint_matches_embodiment": summary.get("viewpoint_matches_embodiment"),
        "delivered": summary.get("delivered"),
        "orders_issued": summary.get("orders_issued"),
        "on_time": summary.get("on_time"),
        "turns": turns,
        "format_errors": format_errors,
        "rejected": rejected,
        "images_shown": shown,
        "images_dropped": dropped,
        "termination": session.run.termination_reason,
        "env_return": env_return,
        "format_score": round(format_score, 4),
        "format_weight": format_weight,
        "progress_score": round(progress_score, 4),
        "progress_weight": progress_weight,
        "unfenced_actions": session.spend.unfenced_actions,
    }

    sample = build_training_sample(
        state, episode_id=episode_id, reward=reward, metadata=metadata
    )
    # Park the vision tensors on the CPU. Every rollout in a batch is alive at
    # once during the update, so keeping their pixels resident capped the batch
    # at two episodes on a 24 GB card -- and a batch of two is most of why the
    # advantage estimate is noise. ``build_model_inputs`` moves them back per
    # sample, one at a time.
    if sample.pixel_values:
        sample.pixel_values = [p.to("cpu") for p in sample.pixel_values]
    if sample.image_grid_thw:
        sample.image_grid_thw = [g.to("cpu") for g in sample.image_grid_thw]
    samples = split_for_fanout(sample, shards) if shards > 1 else [sample]
    # Cheap, and it is the invariant most likely to be broken by a later change
    # to how episodes are cut up.
    check_reward_conservation(reward, samples)

    return CourierRollout(
        episode_id=episode_id, state=state, samples=samples, reward=reward,
        env_return=env_return, format_score=format_score,
        progress_score=progress_score,
        summary=summary, turns=turns, format_errors=format_errors,
        rejected=rejected, images_shown=shown, images_dropped=dropped,
        turn_rewards=turn_rewards,
    )


def rollout_batch(
    adapter: RolloutAdapter,
    make_env: Any,
    seeds: list[int],
    **kwargs: Any,
) -> list[CourierRollout]:
    """One rollout per seed. ``make_env(seed)`` returns a reset environment."""
    out = []
    for seed in seeds:
        env = make_env(seed)
        out.append(rollout_courier_episode(
            adapter, env, episode_id=f"courier-{seed}", seed=seed, **kwargs))
    return out


def batch_advantages(rollouts: list[CourierRollout]) -> list[float]:
    """Reward minus the batch mean -- the baseline the REINFORCE step expects.

    A batch of one has no baseline and gets zero advantage rather than its own
    reward: with nothing to compare against, the sign of the gradient would be
    decided by whether the reward happened to be positive, which is not a
    learning signal.
    """
    if len(rollouts) < 2:
        return [0.0 for _ in rollouts]
    rewards = [r.reward for r in rollouts]
    mean = sum(rewards) / len(rewards)
    return [r - mean for r in rewards]
