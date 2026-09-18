"""The RL path, on the environment the benchmark actually ships.

The training package was written against the old vagen runtime and nothing in
the suite touched it -- 975 tests, zero references to ``embodiedbench.training``.
So the two failure modes design plan §2.1 calls out as silently invalidating an RL
run had no regression protection on the courier environment at all: observation
tokens leaking into the policy loss, and reward being lost or duplicated when a
trajectory is split.

These run on a stub adapter. That is the point: the invariants are properties of
the *rollout*, not of any model, so they can be checked without a GPU, without a
checkpoint, and in under a second. The one test that needs a real model is
marked and skipped when it is not there.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any

import pytest

from embodiedbench.compiler.road_network import build_road_network
from embodiedbench.agent.courier.session import Frame, Observation
from embodiedbench.runtime.city.courier_env import CourierEnv
from embodiedbench.training.core import (
    MaskViolation,
    RewardAccountingError,
    check_reward_conservation,
    split_for_fanout,
    validate_sample,
)
from embodiedbench.training.courier_rollout import (
    _load_images,
    batch_advantages,
    rollout_courier_episode,
)
from embodiedbench.training.image_groups import required_image_groups

MAPS = Path("vendor/vagen/vagen/envs/deliverybench/maps/citycore-paris")
STREETS = Path(os.environ.get("ALBUMS_DIR", "/data/albums")) / "paris_streets_v2/citycore-paris"


@pytest.fixture(scope="module")
def paris():
    return build_road_network(MAPS, map_name="citycore-paris")


# ── a model-free stand-in ────────────────────────────────────────────────────
#
# It tokenises by character code, which is enough to exercise every invariant:
# the ids are integers, the provenance is per token, and the assistant spans
# carry one log-prob each. What it must not do is guess which tokens were the
# policy's -- that has to come from where they were appended.


@dataclass
class StubSpan:
    role: str
    start: int
    end: int
    is_assistant: bool
    logprobs: list[float] = field(default_factory=list)


@dataclass
class StubState:
    input_ids: list[int] = field(default_factory=list)
    assistant_mask: list[bool] = field(default_factory=list)
    spans: list[StubSpan] = field(default_factory=list)
    pixel_values: list[Any] = field(default_factory=list)
    image_grid_thw: list[Any] = field(default_factory=list)
    image_token_positions: list[int] = field(default_factory=list)

    def assistant_logprobs(self) -> list[float]:
        return [lp for span in self.spans if span.is_assistant for lp in span.logprobs]

    def summary(self) -> dict[str, int]:
        return {"total_tokens": len(self.input_ids),
                "assistant_tokens": sum(self.assistant_mask),
                "turns": len(self.spans)}


class StubAdapter:
    """Appends tokens with honest provenance and replies with a legal action."""

    IMAGE_TOKEN = 999_999

    def __init__(self, reply: str = "THOUGHT: go\n```\nwalk_to(\"Rue de Grenelle\", \"east\")\n```"):
        self.reply = reply
        self.generated = 0

    def _append(self, state, ids, *, is_assistant, logprobs=None, image_positions=()):
        start = len(state.input_ids)
        state.input_ids.extend(ids)
        state.assistant_mask.extend([is_assistant] * len(ids))
        state.image_token_positions.extend(start + i for i in image_positions)
        state.spans.append(StubSpan(
            "assistant" if is_assistant else "env", start, len(state.input_ids),
            is_assistant, list(logprobs or []),
        ))

    def start_episode(self, system_prompt: str) -> StubState:
        state = StubState()
        self._append(state, [ord(c) % 5000 for c in system_prompt], is_assistant=False)
        return state

    def observe(self, state, text, images=None):
        images = images or []
        ids = [self.IMAGE_TOKEN] * len(images) + [ord(c) % 5000 for c in text]
        self._append(state, ids, is_assistant=False,
                     image_positions=range(len(images)))

    def generate(self, state, *, max_new_tokens=64, temperature=0.0, seed=None):
        ids = [ord(c) % 5000 for c in self.reply][:max_new_tokens]
        self._append(state, ids, is_assistant=True, logprobs=[-0.5] * len(ids))
        self.generated += 1
        return self.reply, None

    def close_assistant_turn(self, state) -> None:
        self._append(state, [2], is_assistant=False)


def make_env(paris, seed=0, tier="solo"):
    kwargs = {}
    if STREETS.exists():
        kwargs["album_root"] = STREETS
    env = CourierEnv(paris, seed=seed, difficulty=tier, stride="block", **kwargs)
    env.reset()
    return env


def _rgb_frame(tmp_path: Path, name: str, colour: tuple[int, int, int], **kwargs):
    from PIL import Image

    path = tmp_path / name
    Image.new("RGB", (2, 2), colour).save(path)
    return Frame(label=name, path=str(path), **kwargs)


class TestRequiredImageGroupsAtRolloutIngress:
    def _pair(self, tmp_path: Path) -> list[Frame]:
        binding = {"capture_group_id": "pair-1", "required_group": True}
        return [
            _rgb_frame(tmp_path, "front.png", (255, 0, 0),
                       view_id="front", **binding),
            _rgb_frame(tmp_path, "rear.png", (0, 0, 255),
                       view_id="rear", **binding),
        ]

    def _interleaved_pairs(self, tmp_path: Path) -> list[Frame]:
        return [
            _rgb_frame(tmp_path, "a-front.png", (255, 0, 0), view_id="front",
                       capture_group_id="pair-a", required_group=True),
            _rgb_frame(tmp_path, "b-front.png", (0, 255, 0), view_id="front",
                       capture_group_id="pair-b", required_group=True),
            _rgb_frame(tmp_path, "a-rear.png", (0, 0, 255), view_id="rear",
                       capture_group_id="pair-a", required_group=True),
            _rgb_frame(tmp_path, "b-rear.png", (255, 255, 0), view_id="rear",
                       capture_group_id="pair-b", required_group=True),
        ]

    def test_capacity_one_fails_before_reading_either_required_view(self, tmp_path):
        observation = Observation("turn", frames=[
            Frame("front", path=str(tmp_path / "missing-front.png"),
                  view_id="front", capture_group_id="pair-1",
                  required_group=True),
            Frame("rear", path=str(tmp_path / "missing-rear.png"),
                  view_id="rear", capture_group_id="pair-1",
                  required_group=True),
        ])

        with pytest.raises(ValueError, match="pair-1 requires 2 photographs; max_images=1"):
            _load_images(observation, max_images=1, scratch=tmp_path)

    def test_capacity_two_keeps_the_ordered_pair_and_drops_the_optional_map(self, tmp_path):
        observation = Observation("turn", frames=[
            *self._pair(tmp_path),
            Frame("[map]", kind="map", svg="<svg/>")
        ])

        images, dropped = _load_images(observation, max_images=2, scratch=tmp_path)

        assert [image.getpixel((0, 0)) for image in images] == [
            (255, 0, 0), (0, 0, 255)
        ]
        assert dropped == 1

    def test_capacity_three_keeps_the_pair_then_the_normal_optional_photo(self, tmp_path):
        observation = Observation("turn", frames=[
            *self._pair(tmp_path),
            _rgb_frame(tmp_path, "legacy.png", (0, 255, 0)),
        ])

        images, dropped = _load_images(observation, max_images=3, scratch=tmp_path)

        assert [image.getpixel((0, 0)) for image in images] == [
            (255, 0, 0), (0, 0, 255), (0, 255, 0)
        ]
        assert dropped == 0

    @pytest.mark.parametrize("invalid", [True, "2", 2.0, -1])
    def test_capacity_must_be_an_exact_nonnegative_integer(self, tmp_path, invalid):
        observation = Observation("turn", frames=self._pair(tmp_path))

        with pytest.raises(ValueError, match="max_images must be a non-negative int"):
            _load_images(observation, max_images=invalid, scratch=tmp_path)

    def test_invalid_capacity_is_rejected_before_malformed_group_metadata(self, tmp_path):
        malformed = Observation("turn", frames=[
            Frame("front", required_group=True, view_id="front")
        ])

        with pytest.raises(ValueError, match="max_images must be a non-negative int"):
            _load_images(malformed, max_images="2", scratch=tmp_path)

    def test_total_required_capacity_is_rejected_before_any_group_file_is_read(
            self, tmp_path):
        frames = self._interleaved_pairs(tmp_path)
        missing = [
            Frame(**{**frame.to_dict(), "path": str(tmp_path / f"missing-{index}.png")})
            for index, frame in enumerate(frames)
        ]

        with pytest.raises(
                ValueError,
                match="required image groups require 4 photographs; max_images=3"):
            _load_images(Observation("turn", frames=missing),
                         max_images=3, scratch=tmp_path)

    def test_multiple_groups_keep_interleaved_observation_and_caption_order(self, tmp_path):
        frames = self._interleaved_pairs(tmp_path)
        captions = "\n".join(frame.label for frame in frames)
        observation = Observation(captions, frames=frames)

        images, dropped = _load_images(observation, max_images=4, scratch=tmp_path)

        assert [image.getpixel((0, 0)) for image in images] == [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)
        ]
        assert observation.text.splitlines() == [frame.label for frame in frames]
        assert dropped == 0

    def test_group_validation_accepts_duck_typed_frames_in_observation_order(
            self, tmp_path):
        from types import SimpleNamespace

        frames = [SimpleNamespace(**frame.to_dict())
                  for frame in self._interleaved_pairs(tmp_path)]

        groups = required_image_groups(SimpleNamespace(frames=frames))

        assert list(groups) == ["pair-a", "pair-b"]
        assert groups["pair-a"] == [frames[0], frames[2]]
        assert groups["pair-b"] == [frames[1], frames[3]]

    @pytest.mark.parametrize("malformation", [
        "missing_group_id", "rear_first", "partial", "duplicate_rear",
    ])
    def test_malformed_required_group_metadata_is_rejected(self, tmp_path, malformation):
        pair = self._pair(tmp_path)
        if malformation == "missing_group_id":
            pair[0] = Frame(**{**pair[0].to_dict(), "capture_group_id": ""})
        elif malformation == "rear_first":
            pair.reverse()
        elif malformation == "partial":
            pair.pop()
        else:
            pair.append(pair[-1])

        message = ("missing capture_group_id" if malformation == "missing_group_id"
                   else "required image group pair-1 is not front, rear")
        with pytest.raises(ValueError, match=message):
            _load_images(Observation("turn", frames=pair),
                         max_images=3, scratch=tmp_path)

    @pytest.mark.parametrize("view", ["front", "rear"])
    @pytest.mark.parametrize("failure", ["missing", "corrupt"])
    def test_a_required_view_load_failure_is_not_counted_as_an_optional_drop(
            self, tmp_path, view, failure):
        pair = self._pair(tmp_path)
        index = 0 if view == "front" else 1
        bad_path = tmp_path / f"bad-{view}.png"
        if failure == "corrupt":
            bad_path.write_bytes(b"not an image")
        pair[index] = Frame(**{**pair[index].to_dict(), "path": str(bad_path)})

        with pytest.raises(
                RuntimeError,
                match=f"required image {view} could not be loaded") as exc_info:
            _load_images(Observation("turn", frames=pair),
                         max_images=2, scratch=tmp_path)
        assert exc_info.value.__cause__ is not None

    @pytest.mark.parametrize("view", ["front", "rear"])
    def test_a_required_view_conversion_failure_keeps_its_chained_cause(
            self, tmp_path, monkeypatch, view):
        from PIL import Image

        pair = self._pair(tmp_path)
        real_open = Image.open

        class ConversionFailure:
            def convert(self, _mode):
                raise OSError("synthetic conversion failure")

        def fail_selected(path):
            if Path(path).name == f"{view}.png":
                return ConversionFailure()
            return real_open(path)

        monkeypatch.setattr(Image, "open", fail_selected)
        with pytest.raises(
                RuntimeError,
                match=f"required image {view} could not be loaded") as exc_info:
            _load_images(Observation("turn", frames=pair),
                         max_images=2, scratch=tmp_path)
        assert isinstance(exc_info.value.__cause__, OSError)

    def test_an_optional_decode_failure_keeps_existing_drop_accounting(self, tmp_path):
        bad = tmp_path / "bad-optional.png"
        bad.write_bytes(b"not an image")
        observation = Observation("turn", frames=[
            *self._pair(tmp_path),
            Frame("legacy", path=str(bad)),
        ])

        images, dropped = _load_images(observation, max_images=3, scratch=tmp_path)

        assert len(images) == 2
        assert dropped == 1


class TestTheRolloutReachesTheCourierEnvironment:
    def test_an_episode_becomes_a_valid_training_sample(self, paris):
        rollout = rollout_courier_episode(
            StubAdapter(), make_env(paris), episode_id="e0", max_turns=6)
        assert rollout.turns > 0
        assert len(rollout.samples) == 1
        validate_sample(rollout.samples[0])
        assert rollout.samples[0].response_token_count > 0

    def test_the_policy_drives_the_same_harness_evaluation_uses(self, paris):
        """Training against a different prompt from the one scored would optimise
        a policy for an observation it never sees."""
        adapter = StubAdapter()
        rollout = rollout_courier_episode(
            adapter, make_env(paris), episode_id="e0", max_turns=4)
        assert adapter.generated == rollout.turns
        # the actions really went into the world
        assert rollout.summary["turns"] > 0

    def test_a_malformed_reply_is_recorded_not_crashed_on(self, paris):
        rollout = rollout_courier_episode(
            StubAdapter(reply="I will just walk north."), make_env(paris),
            episode_id="e0", max_turns=4)
        assert rollout.format_errors > 0
        validate_sample(rollout.samples[0])


class TestOnlySampledTokensCarryGradient:
    def test_no_observation_token_is_inside_the_mask(self, paris):
        rollout = rollout_courier_episode(
            StubAdapter(), make_env(paris), episode_id="e0", max_turns=5)
        sample = rollout.samples[0]
        assistant = set(sample.response_indices)
        spans = rollout.state.spans
        for span in spans:
            positions = set(range(span.start, span.end))
            if span.is_assistant:
                assert positions <= assistant
            else:
                assert not (positions & assistant), f"{span.role} tokens in the loss"

    @pytest.mark.skipif(not STREETS.exists(), reason="albums not mounted")
    def test_no_image_token_is_inside_the_mask(self, paris):
        rollout = rollout_courier_episode(
            StubAdapter(), make_env(paris), episode_id="e0", max_turns=5)
        sample = rollout.samples[0]
        assert sample.image_token_positions, "no images were shown at all"
        assert not (set(sample.image_token_positions) & set(sample.response_indices))

    def test_every_sampled_token_has_a_logprob(self, paris):
        rollout = rollout_courier_episode(
            StubAdapter(), make_env(paris), episode_id="e0", max_turns=5)
        sample = rollout.samples[0]
        assert len(sample.logprobs) == sample.response_token_count

    def test_a_mask_that_lies_is_rejected(self, paris):
        rollout = rollout_courier_episode(
            StubAdapter(), make_env(paris), episode_id="e0", max_turns=3)
        sample = rollout.samples[0]
        sample.response_mask[0] = True          # claim a system token was sampled
        with pytest.raises(MaskViolation):
            validate_sample(sample)


class TestRewardSurvivesBeingSplitUp:
    def test_fanout_preserves_the_episode_return_exactly(self, paris):
        rollout = rollout_courier_episode(
            StubAdapter(), make_env(paris), episode_id="e0", max_turns=6, shards=4)
        assert len(rollout.samples) == 4
        check_reward_conservation(rollout.reward, rollout.samples)

    def test_losing_a_shard_is_caught(self, paris):
        """With a reward to lose. A stub policy that never delivers returns 0.0,
        and dropping a shard of nothing still sums to nothing -- so the check is
        vacuous exactly when the policy is failing, which is most of training.
        The reward is set here so the test measures the invariant rather than
        the stub's competence."""
        rollout = rollout_courier_episode(
            StubAdapter(), make_env(paris), episode_id="e0", max_turns=4)
        sample = rollout.samples[0]
        sample.reward = 1.5
        shards = split_for_fanout(sample, 3)
        check_reward_conservation(1.5, shards)
        with pytest.raises(RewardAccountingError):
            check_reward_conservation(1.5, shards[:-1])

    def test_the_reward_is_the_environment_return(self, paris):
        """Not a proxy invented by the trainer: the same number the harness
        accumulated from deliveries, punctuality and red lights."""
        env = make_env(paris)
        rollout = rollout_courier_episode(
            StubAdapter(), env, episode_id="e0", max_turns=6)
        assert rollout.reward == pytest.approx(sum(rollout.turn_rewards), abs=1e-6)


class TestTheShapedTermsStayOutOfTheBenchmarkScore:
    """A shaped run must never be quotable as a benchmark result.

    ``progress_weight`` exists because Qwen3-VL-4B already scores 1.0 on the
    format term, so that rung yields a zero advantage on every batch. Distance
    closed on the target is dense enough to have variance -- but it is
    privileged information (``route_length_cm`` is never shown to the agent),
    so the one thing that must hold is that it moves ``reward`` and leaves
    ``env_return`` alone.
    """

    def test_progress_weight_changes_the_objective_and_not_the_score(self, paris):
        plain = rollout_courier_episode(
            StubAdapter(), make_env(paris), episode_id="e0", max_turns=5)
        shaped = rollout_courier_episode(
            StubAdapter(), make_env(paris), episode_id="e0", max_turns=5,
            progress_weight=1.0)
        assert shaped.env_return == plain.env_return
        assert shaped.reward == pytest.approx(
            plain.env_return + shaped.progress_score, abs=1e-6)

    def test_the_shaped_reward_is_still_conserved_across_shards(self, paris):
        rollout = rollout_courier_episode(
            StubAdapter(), make_env(paris), episode_id="e0", max_turns=6,
            progress_weight=1.0, format_weight=1.0, shards=3)
        check_reward_conservation(rollout.reward, rollout.samples)

    def test_collecting_does_not_book_the_target_switch_as_progress(self, paris):
        """The target jumps from the pickup to the dropoff on collection, and
        the two are streets apart. Booking that switch would hand the policy a
        large reward or penalty it did not walk for."""
        rollout = rollout_courier_episode(
            StubAdapter(), make_env(paris), episode_id="e0", max_turns=8,
            progress_weight=1.0)
        # A stub that only ever walks cannot close more ground than it covered.
        assert abs(rollout.progress_score) < 5.0


class TestTheBaseline:
    def test_advantages_are_centred_on_the_batch(self, paris):
        rollouts = [rollout_courier_episode(StubAdapter(), make_env(paris, seed=s),
                                            episode_id=f"e{s}", max_turns=3)
                    for s in range(3)]
        advantages = batch_advantages(rollouts)
        assert sum(advantages) == pytest.approx(0.0, abs=1e-6)

    def test_a_single_rollout_has_no_baseline_and_so_no_advantage(self, paris):
        one = [rollout_courier_episode(StubAdapter(), make_env(paris),
                                       episode_id="e0", max_turns=3)]
        assert batch_advantages(one) == [0.0]


class TestTheReplyFormatDoesNotDiscardCorrectActions:
    """A bare call is an action; a call inside prose is narration.

    Qwen2-VL-2B ends every reply with exactly the right call and no code fence.
    Rejecting those made all three of its opening turns format errors, the
    three-strikes rule ended the episode before it acted once, and every seed
    scored 0.0 for a reason that had nothing to do with navigation. The fence
    exists to make the action unambiguous, not to test markdown.
    """

    def test_a_bare_call_on_its_own_line_is_accepted(self, paris):
        from embodiedbench.agent.courier.loop import parse_reply

        action = parse_reply(
            "THOUGHT: the route says south-west.\nI will take it.\nwalk_to(\"Rue de Grenelle\", \"east\")",
            {"walk_to"})
        assert action.render() == "walk_to(\"Rue de Grenelle\", \"east\")"
        assert action.unfenced is True

    def test_a_fenced_call_is_still_not_flagged(self, paris):
        from embodiedbench.agent.courier.loop import parse_reply

        action = parse_reply("THOUGHT: x\n```\nwalk_to(\"Rue de Grenelle\", \"east\")\n```", {"walk_to"})
        assert action.unfenced is False

    def test_a_call_inside_a_sentence_is_not_an_action(self, paris):
        """Executing the model's narration is the guessing the parser exists to
        avoid."""
        from embodiedbench.agent.courier.loop import FormatError, parse_reply

        with pytest.raises(FormatError):
            parse_reply("THOUGHT: I will walk_to(\"Rue de Grenelle\", \"east\") once I am past the barrier.",
                        {"walk_to"})

    def test_leniency_is_counted_not_hidden(self, paris):
        from embodiedbench.agent.courier.session import CourierSession

        env = make_env(paris)
        session = CourierSession(env, city="Paris")
        session.step("THOUGHT: go\nwalk_to(\"Rue de Grenelle\", \"east\")")
        assert session.spend.unfenced_actions == 1


class TestTheLaunchScriptIsOneCommand:
    """A comment inside a backslash-continued command ends the command there.

    A note was added in the middle of the trainer.* arguments and silently
    dropped the last three: the run started with no checkpoint directory, no
    rollout dump and no validation dump, and nothing failed -- the training
    just quietly stopped writing the files the analysis depends on. Nothing in
    the log says so except the resolved config, ten thousand lines in.
    """

    def scripts(self):
        from pathlib import Path

        root = Path(__file__).resolve().parents[1] / "embodiedbench" / "training"
        return sorted(root.rglob("*.sh"))

    def test_no_comment_interrupts_a_continued_command(self):
        assert self.scripts(), "no launch scripts found"
        for path in self.scripts():
            lines = path.read_text().splitlines()
            for i, line in enumerate(lines[:-1]):
                if not line.rstrip().endswith("\\"):
                    continue
                following = lines[i + 1].strip()
                assert not following.startswith("#"), (
                    f"{path.name}:{i + 2} is a comment inside a continued "
                    f"command, which ends it there and drops every argument "
                    f"after it:\n  {lines[i]}\n  {lines[i + 1]}")

    def test_the_grpo_launcher_still_passes_its_output_directories(self):
        """The three that were dropped, named so a later edit cannot drop them
        again without a test saying which."""
        text = next(p for p in self.scripts()
                    if p.name == "train_grpo_courier.sh").read_text()
        for key in ("trainer.default_local_dir",
                    "trainer.rollout_data_dir",
                    "trainer.validation_data_dir"):
            assert key in text, f"{key} is missing from the launcher"
