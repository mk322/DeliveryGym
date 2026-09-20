"""The VAGEN adapter, checked without ray, verl, or a GPU.

The point of testing this separately is that the environment side is the half
that can be wrong quietly. A PPO run with a broken observation still produces a
loss curve; it just trains on the wrong thing. So the invariants pinned here
are the ones the agent loop assumes and does not re-check:

* one ``<image>`` placeholder per image, always;
* the text is the harness's own, not a training-only variant;
* ``done`` really ends the episode;
* the benchmark's return is carried in ``info`` unshaped.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from embodiedbench.agent.courier.session import Frame, Observation
from embodiedbench.training.vagen_courier_env import (
    IMAGE_PLACEHOLDER,
    CourierGymEnv,
)
from embodiedbench.training.image_groups import required_image_groups

STREETS = Path(os.environ.get("ALBUMS_DIR", "/data/albums")) / "paris_streets_v2/citycore-paris"
def walk(env):
    """A reply that takes this junction's first street, whatever it is called.

    Streets are named rather than numbered now, so no fixed string is a legal
    move at every junction. This reads the name off the environment the same
    way a courier reads it off the observation.
    """
    street, heading = env._env.street_at(1)
    return f'THOUGHT: go\n```\nwalk_to("{street}", "{heading}")\n```'



def run(coro):
    """asyncio.run per call.

    get_event_loop() passed in isolation and failed under the full suite --
    "There is no current event loop in thread 'MainThread'" -- because another
    test had already closed the implicit loop. A helper that only works when
    its file is run alone is worse than no helper.
    """
    return asyncio.run(coro)


def _rgb_frame(tmp_path: Path, name: str, colour: tuple[int, int, int], **kwargs):
    from PIL import Image

    path = tmp_path / name
    Image.new("RGB", (2, 2), colour).save(path)
    return Frame(label=name, path=str(path), **kwargs)


@pytest.fixture(scope="module")
def config():
    base = {"difficulty": "solo", "stride": "block", "max_turns": 6}
    if STREETS.exists():
        base["album_root"] = str(STREETS)
    return base


class TestRequiredImageGroupsAtVagenIngress:
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

    def _env(self, tmp_path: Path, max_images: int) -> CourierGymEnv:
        from PIL import Image

        env = CourierGymEnv({"max_images": max_images})
        env.album_root = tmp_path
        env._rasterise = lambda _svg, _index: Image.new("RGB", (2, 2), (0, 255, 0))
        return env

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
            self._env(tmp_path, 1)._load_images(observation)

    @pytest.mark.parametrize(("max_images", "expected_colours", "dropped"), [
        (2, [(255, 0, 0), (0, 0, 255)], 1),
        (3, [(255, 0, 0), (0, 0, 255), (0, 255, 0)], 0),
    ])
    def test_the_pair_is_first_and_the_optional_map_needs_a_third_slot(
            self, tmp_path, max_images, expected_colours, dropped):
        observation = Observation("turn", frames=[
            *self._pair(tmp_path),
            Frame("[map]", kind="map", svg="<svg/>")
        ])

        images, actual_dropped, labels = self._env(
            tmp_path, max_images)._load_images(observation)

        assert [image.getpixel((0, 0)) for image in images] == expected_colours
        assert labels == ["front.png", "rear.png"] + (["[map]"] if max_images == 3 else [])
        assert actual_dropped == dropped

    @pytest.mark.parametrize("invalid", [True, "2", 2.0, -1])
    def test_capacity_must_be_an_exact_nonnegative_integer(self, tmp_path, invalid):
        observation = Observation("turn", frames=self._pair(tmp_path))

        with pytest.raises(ValueError, match="max_images must be a non-negative int"):
            self._env(tmp_path, invalid)._load_images(observation)

    def test_invalid_capacity_is_rejected_before_malformed_group_metadata(self, tmp_path):
        malformed = Observation("turn", frames=[
            Frame("front", required_group=True, view_id="front")
        ])

        with pytest.raises(ValueError, match="max_images must be a non-negative int"):
            self._env(tmp_path, "2")._load_images(malformed)

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
            self._env(tmp_path, 3)._load_images(Observation("turn", frames=missing))

    def test_multiple_groups_keep_interleaved_image_label_and_caption_order(
            self, tmp_path):
        from types import SimpleNamespace

        frames = self._interleaved_pairs(tmp_path)
        observation = Observation(
            "turn\n### photographs\n  stale\n### after",
            frames=frames,
        )
        env = self._env(tmp_path, 4)
        env._session = SimpleNamespace(observe=lambda: observation, lamp_legs={})
        env._env = SimpleNamespace(show_only_these_signals=lambda _signals: None)

        obs, dropped = env._observation()

        images = obs["multi_modal_input"][IMAGE_PLACEHOLDER]
        assert [image.getpixel((0, 0)) for image in images] == [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)
        ]
        caption = obs["obs_str"].split("### photographs\n", 1)[1].split(
            "\n###", 1)[0]
        assert [line.strip() for line in caption.splitlines()] == [
            frame.label for frame in frames
        ]
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

    def test_vagen_rejects_malformed_pair_order_before_loading(self, tmp_path):
        pair = self._pair(tmp_path)
        pair.reverse()

        with pytest.raises(ValueError, match="required image group pair-1 is not front, rear"):
            self._env(tmp_path, 2)._load_images(Observation("turn", frames=pair))

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
            self._env(tmp_path, 2)._load_images(Observation("turn", frames=pair))
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
            self._env(tmp_path, 2)._load_images(Observation("turn", frames=pair))
        assert isinstance(exc_info.value.__cause__, OSError)

    def test_an_optional_decode_failure_keeps_existing_drop_accounting(self, tmp_path):
        bad = tmp_path / "bad-optional.png"
        bad.write_bytes(b"not an image")
        observation = Observation("turn", frames=[
            *self._pair(tmp_path),
            Frame("legacy", path=str(bad)),
        ])

        images, dropped, labels = self._env(tmp_path, 3)._load_images(observation)

        assert len(images) == 2
        assert dropped == 1
        assert labels == ["front.png", "rear.png"]


class TestTheObservationContract:
    def test_placeholders_match_the_number_of_images(self, config):
        """The agent loop treats a mismatch as fatal, and it is the failure a
        caption-reordering or image-capping change introduces first."""
        env = CourierGymEnv(config)
        obs, _ = run(env.reset(0))
        images = obs.get("multi_modal_input", {}).get(IMAGE_PLACEHOLDER, [])
        assert obs["obs_str"].count(IMAGE_PLACEHOLDER) == len(images)

        for _ in range(3):
            obs, _, done, _ = run(env.step(walk(env)))
            if done:
                break
            images = obs.get("multi_modal_input", {}).get(IMAGE_PLACEHOLDER, [])
            assert obs["obs_str"].count(IMAGE_PLACEHOLDER) == len(images)
        run(env.close())

    def test_a_text_only_episode_carries_no_placeholder(self):
        """Without an album there are no pictures, and a stray placeholder
        would make the loop expect an image that does not exist."""
        env = CourierGymEnv({"difficulty": "solo", "stride": "block"})
        obs, _ = run(env.reset(0))
        assert IMAGE_PLACEHOLDER not in obs["obs_str"]
        assert "multi_modal_input" not in obs
        run(env.close())

    def test_the_prompt_is_the_harness_prompt(self, config):
        """Training against a different prompt from the one scored optimises a
        policy for an observation it never sees."""
        from embodiedbench.agent.courier.session import CourierSession

        env = CourierGymEnv(config)
        run(env.reset(0))
        assert run(env.system_prompt())["obs_str"] == env._session.system_prompt()
        # and the per-turn text, placeholders aside
        obs, _ = env._observation()
        # Everything but the photographs block, which the adapter rewrites to
        # name the frames it actually sent and in the order it sends them --
        # interleaved street/lamp, where the harness lists all streets then all
        # lamps. See TestTrainingSeesTheSameWorldAsEvaluation.
        import re as _re

        def body(text):
            return _re.sub(r"### photographs\n.*?(?=\n###|\Z)", "", text,
                           flags=_re.S).replace(IMAGE_PLACEHOLDER, "").strip()

        assert body(env._session.observe().text) == body(obs["obs_str"])
        run(env.close())

    def test_the_image_cap_is_reported_not_hidden(self, config):
        if not STREETS.exists():
            pytest.skip("albums not mounted")
        env = CourierGymEnv({**config, "max_images": 1})
        obs, info = run(env.reset(0))
        images = obs["multi_modal_input"][IMAGE_PLACEHOLDER]
        # One street view plus the phone's map, which is lit from the first
        # turn now that the direction to walk exists nowhere else.
        assert len(images) == 2
        assert info["images_dropped"] >= 1
        run(env.close())


class TestTheEpisode:
    def test_reset_step_done(self, config):
        env = CourierGymEnv(config)
        run(env.reset(0))
        done = False
        steps = 0
        while not done and steps < 10:
            _, reward, done, info = run(env.step(walk(env)))
            steps += 1
            assert isinstance(reward, float)
        assert done, "max_turns did not end the episode"
        run(env.close())

    def test_the_seed_decides_the_episode(self, config):
        a = CourierGymEnv(config)
        b = CourierGymEnv(config)
        c = CourierGymEnv(config)
        first, _ = run(a.reset(0))
        same, _ = run(b.reset(0))
        other, _ = run(c.reset(7))
        assert first["obs_str"] == same["obs_str"]
        assert first["obs_str"] != other["obs_str"]
        for env in (a, b, c):
            run(env.close())

    def test_a_malformed_reply_is_a_recorded_turn_not_a_crash(self, config):
        env = CourierGymEnv(config)
        run(env.reset(0))
        _, reward, _, info = run(env.step("I will just walk north."))
        assert info["status"] == "format_error"
        assert reward == 0.0
        run(env.close())

    def test_info_carries_the_unshaped_benchmark_return(self, config):
        """A trainer may optimise whatever it likes; the number it *reports*
        has to be the one the harness accumulated."""
        env = CourierGymEnv(config)
        run(env.reset(0))
        total = 0.0
        for _ in range(4):
            _, reward, done, info = run(env.step(walk(env)))
            total += reward
            assert info["env_return"] == pytest.approx(total, abs=1e-6)
            if done:
                break
        run(env.close())


class TestTheShapedTermStaysOutOfTheBenchmarkScore:
    """GRPO's first step on this env reported reward_variance 0.0, pg_loss 0.0
    and grad_norm 0.0: a policy that never delivers earns the same 0.0 on every
    sample of the group, so there is no advantage to push on. progress_weight
    is the dense term that fixes it -- and the thing that must never happen is
    it leaking into the number the benchmark reports."""

    def test_shaping_moves_reward_and_leaves_env_return_alone(self, config):
        plain = CourierGymEnv(config)
        shaped = CourierGymEnv({**config, "progress_weight": 1.0})
        run(plain.reset(0))
        run(shaped.reset(0))
        pr = sr = 0.0
        for _ in range(4):
            _, r, d1, i1 = run(plain.step(walk(plain)))
            _, s, d2, i2 = run(shaped.step(walk(shaped)))
            pr, sr = pr + r, sr + s
            assert i1["env_return"] == pytest.approx(i2["env_return"], abs=1e-6)
            if d1 or d2:
                break
        assert sr != pr or i2["progress_score"] == 0.0
        run(plain.close()); run(shaped.close())

    def test_off_by_default(self, config):
        env = CourierGymEnv(config)
        run(env.reset(0))
        _, reward, _, info = run(env.step(walk(env)))
        assert info["progress_weight"] == 0.0
        assert reward == pytest.approx(info["env_return"], abs=1e-6)
        run(env.close())

    def test_the_dense_term_actually_varies_across_seeds(self, config):
        """The whole point is variance. If every episode scored the same the
        group advantage would still be zero and nothing would be fixed."""
        scores = []
        for seed in (0, 1, 2, 3):
            env = CourierGymEnv({**config, "progress_weight": 1.0})
            run(env.reset(seed))
            for _ in range(4):
                _, _, done, info = run(env.step(walk(env)))
                if done:
                    break
            scores.append(info["progress_score"])
            run(env.close())
        assert len(set(scores)) > 1, f"no variance across seeds: {scores}"


class TestImagesCanBeMadeCheaperToBuyTurns:
    """Token budget is turns. A 640x480 frame is ~380 tokens and the four-turn
    window it forced covered two of the ten successful collections this model
    managed, which fell on turns 2, 3, 5, 5, 8, 14, 14, 17, 26 and 36."""

    def test_frames_are_downscaled_to_the_configured_side(self, config):
        if not STREETS.exists():
            pytest.skip("albums not mounted")
        env = CourierGymEnv({**config, "image_max_side": 320})
        obs, _ = run(env.reset(0))
        for image in obs["multi_modal_input"][IMAGE_PLACEHOLDER]:
            assert max(image.size) <= 320
        run(env.close())

    def test_zero_leaves_the_frame_alone(self, config):
        if not STREETS.exists():
            pytest.skip("albums not mounted")
        env = CourierGymEnv({**config, "image_max_side": 0})
        obs, _ = run(env.reset(0))
        assert max(obs["multi_modal_input"][IMAGE_PLACEHOLDER][0].size) == 640
        run(env.close())

    def test_aspect_ratio_survives(self, config):
        if not STREETS.exists():
            pytest.skip("albums not mounted")
        env = CourierGymEnv({**config, "image_max_side": 320})
        obs, _ = run(env.reset(0))
        w, h = obs["multi_modal_input"][IMAGE_PLACEHOLDER][0].size
        assert abs(w / h - 640 / 480) < 0.02
        run(env.close())


class TestTheJobIsMeasuredInMoney:
    """Earnings are what a courier is actually judged on.

    The fee is 3.00 plus a cent a metre, paid in full on time and in part when
    late, so punctuality and job size are inside the number already. Nothing
    about it was chosen here, unlike env_return's +1.0/+/-0.5/+0.1/-1.0.

    It is reported, not optimised. Across 40 evaluation episodes earnings were
    non-zero in exactly the 8 that delivered, so a policy that has never
    delivered sees a constant zero -- no variance, no gradient.
    """

    def test_earnings_are_reported_every_turn(self, config):
        env = CourierGymEnv(config)
        run(env.reset(0))
        _, _, _, info = run(env.step(walk(env)))
        assert "earnings" in info and "earnings_per_hour" in info
        run(env.close())

    def test_earnings_are_zero_until_something_is_delivered(self, config):
        env = CourierGymEnv(config)
        run(env.reset(0))
        for _ in range(4):
            _, _, done, info = run(env.step(walk(env)))
            if done:
                break
            assert info["earnings"] == 0.0 or info["delivered"]
        run(env.close())

    def test_shaping_never_touches_earnings(self, config):
        """The training objective may be anything; the money may not move."""
        plain = CourierGymEnv(config)
        shaped = CourierGymEnv({**config, "progress_weight": 1.0})
        run(plain.reset(0)); run(shaped.reset(0))
        for _ in range(4):
            _, _, d1, i1 = run(plain.step(walk(plain)))
            _, _, d2, i2 = run(shaped.step(walk(shaped)))
            assert i1["earnings"] == i2["earnings"]
            if d1 or d2:
                break
        run(plain.close()); run(shaped.close())


class TestEarningsCanBeTheObjectiveItself:
    """Optimising money is the point of the project, so it has to be selectable as
    the reward, not merely reported beside one.

    env_return is four constants picked in this repository. Earnings are
    defined by the job: 3.00 plus a cent a metre, in full on time and in part
    late, so punctuality and job size are already inside it. Progress shaping
    is potential-based and therefore cannot change which policy is optimal --
    under this basis the optimum is the best-earning courier.
    """

    def test_the_episode_rewards_sum_to_what_the_shift_earned(self, config):
        env = CourierGymEnv({**config, "reward_basis": "earnings"})
        run(env.reset(0))
        total = 0.0
        for _ in range(6):
            _, reward, done, info = run(env.step(walk(env)))
            total += reward
            assert total == pytest.approx(info["earnings"], abs=1e-6)
            if done:
                break
        run(env.close())

    def test_the_default_basis_is_unchanged(self, config):
        """Switching the objective must be deliberate; every earlier number was
        measured against env_return."""
        env = CourierGymEnv(config)
        run(env.reset(0))
        _, reward, _, info = run(env.step(walk(env)))
        assert info["reward_basis"] == "env_return"
        assert reward == pytest.approx(info["env_return"], abs=1e-6)
        run(env.close())

    def test_an_unknown_basis_is_refused_rather_than_ignored(self):
        with pytest.raises(ValueError, match="reward_basis"):
            CourierGymEnv({"reward_basis": "vibes"})

    def test_shaping_rides_on_top_without_touching_the_money(self, config):
        env = CourierGymEnv({**config, "reward_basis": "earnings",
                             "progress_weight": 1.0})
        run(env.reset(0))
        for _ in range(4):
            _, reward, done, info = run(env.step(walk(env)))
            # the shaped reward may be non-zero while nothing has been paid
            assert info["earnings"] >= 0.0
            if done:
                break
        run(env.close())


class TestTrainingSeesTheSameCityEvaluationDoes:
    """The albums are named, not guessed.

    They used to be derived as album_root.parent/"signals"/name, which resolves
    to a directory that does not exist, and an `if path.exists()` guard skipped
    it without a word. Training ran with no pedestrian lamps, no barrier frames,
    no pavement views -- an on-foot courier shown the carriageway -- and with
    enforce_signals silently False, while evaluation had all of them. Two
    different environments, one set of conclusions.
    """

    def test_every_album_reaches_the_environment(self, config):
        if not STREETS.exists():
            pytest.skip("albums not mounted")
        env = CourierGymEnv({**config, "hazards": True})
        run(env.reset(0))
        inner = env._env
        for key in ("album_root", "pavement_album_root", "signal_album_root",
                    "obstacle_album_root"):
            assert getattr(inner, key, None) is not None, f"{key} was dropped"
        assert inner.enforce_signals, "red lights were not being enforced"
        run(env.close())

    def test_a_missing_album_is_refused_rather_than_skipped(self, config):
        env = CourierGymEnv({**config, "signal_album_root": "/nonexistent/album"})
        with pytest.raises(FileNotFoundError, match="signal_album_root"):
            run(env.reset(0))

    def test_hazards_false_still_drops_the_hazard_albums(self, config):
        if not STREETS.exists():
            pytest.skip("albums not mounted")
        env = CourierGymEnv({**config, "hazards": False})
        run(env.reset(0))
        assert env._env.signal_album_root is None
        run(env.close())


class TestSuccessIsActuallyReported:
    """VAGEN reads success from info["success"]; the key has to be there.

    Without it extract_success returns False forever, traj_success is 0.0 in
    every log, and that zero gets read as evidence the policy never delivers.
    It also gates the agent loop's early exit, so a completed delivery could
    not end its own episode.
    """

    def test_the_key_exists_every_turn(self, config):
        env = CourierGymEnv(config)
        run(env.reset(0))
        _, _, _, info = run(env.step(walk(env)))
        assert "success" in info
        assert isinstance(info["success"], bool)
        run(env.close())

    def test_it_is_false_while_nothing_is_delivered(self, config):
        env = CourierGymEnv(config)
        run(env.reset(0))
        for _ in range(4):
            _, _, done, info = run(env.step(walk(env)))
            if info["delivered"]:
                break
            assert info["success"] is False
            if done:
                break
        run(env.close())

    def test_vagen_would_read_it(self, config):
        """Exactly the function the agent loop calls."""
        def extract_success(info, success_keys="success|is_success"):
            for key in success_keys.split("|"):
                if key in info:
                    return bool(info[key])
            return False

        env = CourierGymEnv(config)
        run(env.reset(0))
        _, _, _, info = run(env.step(walk(env)))
        assert extract_success(info) == info["success"]
        run(env.close())


class TestTheCaptionDescribesWhatWasActuallySent:
    """The turn's words must not promise pictures the policy never receives.

    The harness captions every frame the turn offers -- "[1], [2] -- the view
    down each of those streets, in that order" -- and the image cap here then
    sends one. A model told it can see two streets while holding one picture
    has no way to know which is missing, so a reply reasoning about "the
    photograph of street 2" reasons about an image it never got.
    """

    def test_the_caption_names_only_the_frames_that_survived(self, config):
        if not STREETS.exists():
            pytest.skip("albums not mounted")
        import re

        env = CourierGymEnv({**config, "max_images": 1})
        obs, info = run(env.reset(0))
        assert info["images_dropped"] >= 1, "need a turn where something was cut"
        caption = re.search(r"### photographs(.*?)(\n###|\Z)", obs["obs_str"], re.S)
        named = re.findall(r"^\s*\[([^\]]+)\]", caption.group(1), re.M)
        sent = len(obs["multi_modal_input"][IMAGE_PLACEHOLDER])
        assert len(named) == sent, f"caption names {named} but {sent} were sent"
        run(env.close())

    def test_an_uncapped_turn_keeps_the_original_caption(self, config):
        if not STREETS.exists():
            pytest.skip("albums not mounted")
        env = CourierGymEnv({**config, "max_images": 16})
        obs, info = run(env.reset(0))
        assert info["images_dropped"] == 0
        assert "in that order" in obs["obs_str"] or "the view down" in obs["obs_str"]
        run(env.close())


class TestALampTravelsWithItsStreet:
    """A red light is charged for, so the lamp must be sent with its crossing.

    The environment charges time for crossing on red exactly where the album
    can show the lamp -- ``signal_frame_for`` refuses to render one it cannot
    see, and that gate exists so a mechanic is only charged when it is
    visible. The image cap then broke the same rule from the other end: at
    ``max_images=1`` it sent the first street view and no lamp at all, so
    training penalised a crossing the policy had no picture of. Streets are
    what the cap counts; a lamp rides along with the street it governs.
    """

    def _signalised_turn(self, config, max_images):
        env = CourierGymEnv({**config, "max_images": max_images})
        run(env.reset(0))
        for node in sorted(env._env.signalised):
            env._env.node_id = node
            # Ask the observation, not the frame list. A junction can have a
            # lamp frame that does not survive ``max_images`` -- the lamp rides
            # with its street, and if that street is third in a two-street turn
            # both are dropped. Checking the raw frames found such a junction
            # and handed back a turn with no lamp caption in it at all, which
            # read as a pairing bug in the tests below rather than as this
            # helper picking the wrong junction.
            obs = env._observation()
            if any(line.startswith("[light:") for line in self._caption(obs[0])):
                return env, obs
        run(env.close())
        pytest.skip("no signalised junction whose lamp survives this image budget")

    def _caption(self, obs):
        import re  # noqa: F811 - the module-level import is inside another test
        block = re.search(r"### photographs\n(.*?)(\n###|\Z)",
                          obs["obs_str"], re.S)
        # The placeholder line is appended after the caption and the
        # photographs block is last, so it falls inside this match.
        return [line.strip() for line in block.group(1).splitlines()
                if line.strip() and IMAGE_PLACEHOLDER not in line]

    def test_the_lamp_is_sent_with_the_street_it_governs(self, config):
        if not STREETS.exists():
            pytest.skip("albums not mounted")
        import re

        env, (obs, _dropped) = self._signalised_turn(config, 2)
        lines = self._caption(obs)
        # Whichever street carries the lamp, the lamp comes directly after it.
        # Not necessarily the first street: a junction shows one lamp now, and
        # it belongs to whichever approach reads it best, not to street one.
        lamp = next(i for i, line in enumerate(lines) if line.startswith("[light:"))
        assert lamp > 0, lines
        governs = re.match(r"\[light:\s*([^\],]+),\s*([^\]]+)\]", lines[lamp])
        assert governs, lines
        assert lines[lamp - 1].startswith(
            f"[{governs.group(1)}, {governs.group(2)}]"), lines
        assert len(obs["multi_modal_input"][IMAGE_PLACEHOLDER]) == len(lines)
        run(env.close())

    def test_a_lamp_is_never_sent_without_its_street(self, config):
        if not STREETS.exists():
            pytest.skip("albums not mounted")
        import re

        env, (obs, _dropped) = self._signalised_turn(config, 2)
        lines = self._caption(obs)
        streets = {re.match(r"\[([^\]]+)\]", line).group(1)
                   for line in lines if not line.startswith("[light")}
        lamps = {re.match(r"\[light:\s*([^\]]+)\]", line).group(1)
                 for line in lines if line.startswith("[light")}
        assert lamps <= streets, f"lamps {lamps} without streets {streets}"
        assert len(obs["multi_modal_input"][IMAGE_PLACEHOLDER]) == len(lines)
        run(env.close())

    def test_the_cap_counts_streets_not_lamps(self, config):
        """Otherwise a signalised junction shows half as far as a plain one."""
        if not STREETS.exists():
            pytest.skip("albums not mounted")
        import re

        env, (obs, _dropped) = self._signalised_turn(config, 2)
        lines = self._caption(obs)
        streets = [line for line in lines
                   if not line.startswith(("[light", "[map"))]
        assert len(streets) == 2, lines
        run(env.close())


class TestTrainingSeesTheSameWorldAsEvaluation:
    """Every conclusion in this project is drawn across both paths at once.

    Training and evaluation have now drifted apart three separate times -- a
    different city (the hazard albums were silently skipped), a different
    caption (frames promised but not sent), a different set of charged
    crossings (lamps not sent while red was charged). Each time the numbers
    stayed comparable-looking and stopped being comparable.

    The image budget is a deliberate difference: evaluation sends every frame,
    training sends one street per turn because the whole trajectory shares one
    context. Everything else must be word for word identical, and that is what
    these pin.
    """

    def test_the_adapter_changes_nothing_but_the_photographs_block(self, config):
        if not STREETS.exists():
            pytest.skip("albums not mounted")
        import re

        env = CourierGymEnv({**config, "max_images": 1})
        obs, _info = run(env.reset(0))
        harness = env._session.observe().text

        def without_photographs(text):
            text = re.sub(r"### photographs\n.*?(?=\n###|\Z)", "", text, flags=re.S)
            return text.replace(IMAGE_PLACEHOLDER, "").strip()

        assert without_photographs(obs["obs_str"]) == without_photographs(harness)
        run(env.close())

    def test_the_charged_crossings_are_the_ones_the_frames_can_show(self, config):
        """served_long_edge narrows the gate; it must never widen it, and the
        environment must not charge an approach the harness downscales past
        legibility."""
        if not STREETS.exists():
            pytest.skip("albums not mounted")
        small = CourierGymEnv({**config, "image_max_side": 320})
        big = CourierGymEnv({**config, "image_max_side": 0})
        run(small.reset(0))
        run(big.reset(0))
        narrow = small._env.visible_signals
        wide = big._env.visible_signals
        assert narrow <= wide

        # It used to assert that 320 px *costs* approaches, which was true of
        # the kerb album: its camera looked down the street, the lamp was 43
        # model pixels, and 34 of 130 approaches fell under the floor. The
        # lamps read out of the scene are framed from 2.5 to 7 m away and
        # centred, so they survive the downscale and none are lost. Asserting
        # the old album's loss would be asserting the old album's defect.
        #
        # What must hold is that the gate is wired and monotone: shrink the
        # frame far enough and it has to bite, or the field it reads is being
        # ignored -- which is exactly what happened when the new sidecar was
        # first written without lamp_px.
        tiny = CourierGymEnv({**config, "image_max_side": 24})
        run(tiny.reset(0))
        assert tiny._env.visible_signals < wide, (
            "the resolution gate reads lamp_px; if nothing is lost at 24 px "
            "the album is not reporting lamp sizes and the gate is inert"
        )
        run(tiny.close())
        run(small.close())
        run(big.close())


class TestPayingByTheHourGivesGRPOSomethingToPushOn:
    """The first real training step scored 4.075 with within-group variance of
    exactly 0.0, and so pg_loss 0.0 and grad_norm 0.0, while scores across the
    batch ranged 0.0 to 5.43. The fee is 3.00 plus a cent a metre of the order,
    so four samples of one order either all delivered for the identical fee or
    none did for 0.0: all the variance was between orders, where GRPO -- which
    normalises inside the group -- cannot use it."""

    def test_the_rate_is_paid_once_at_the_end(self, config):
        """A rate is not a sum of per-turn deltas, so every turn but the last
        pays nothing."""
        env = CourierGymEnv({**config, "reward_basis": "earnings_per_hour",
                             "max_turns": 3})
        run(env.reset(0))
        rewards = []
        for _ in range(3):
            _, reward, done, _ = run(env.step(walk(env)))
            rewards.append(reward)
            if done:
                break
        assert all(r == pytest.approx(0.0) for r in rewards[:-1]), (
            "an unfinished turn paid something; the rate is a terminal reward")
        run(env.close())

    def test_a_shift_that_delivered_nothing_pays_nothing(self, config):
        """Otherwise a charge for time would pay a policy to give up early."""
        env = CourierGymEnv({**config, "reward_basis": "earnings_per_hour",
                             "max_turns": 2})
        run(env.reset(0))
        total = 0.0
        for _ in range(2):
            _, reward, done, info = run(env.step(walk(env)))
            total += reward
            if done:
                break
        assert not info["delivered"], "this seed delivers in two turns; pick another"
        assert total == pytest.approx(0.0)
        run(env.close())

    def test_it_is_a_basis_the_env_admits(self, config):
        env = CourierGymEnv({**config, "reward_basis": "earnings_per_hour"})
        run(env.reset(0))
        _, _, _, info = run(env.step(walk(env)))
        assert info["reward_basis"] == "earnings_per_hour"
        run(env.close())

    def test_an_unknown_basis_is_refused_rather_than_silently_zeroed(self, config):
        with pytest.raises(ValueError, match="unknown reward_basis"):
            CourierGymEnv({**config, "reward_basis": "per_parcel"})


class TestEndlessDispatchOneOrderAtATime:
    """difficulty: endless + queue_depth: 1 is the fixed-time-total-earnings
    shift: the dispatcher refills after every delivery, the clock and the turn
    cap end the episode, and total earnings can differ within a GRPO group."""

    def test_the_depth_reaches_the_env_and_the_dispatch_never_runs_out(self):
        env = CourierGymEnv({"difficulty": "endless", "queue_depth": 1,
                             "reward_basis": "earnings", "max_turns": 3})
        run(env.reset(0))
        inner = env._env
        assert inner.unbounded, "endless tier should leave the dispatcher unbounded"
        assert inner.queue_depth == 1, "the explicit depth must win over the tier's 3"
        assert len(inner.live_orders()) == 1
        run(env.close())

    def test_the_episode_does_not_end_with_the_first_delivery(self):
        env = CourierGymEnv({"difficulty": "endless", "queue_depth": 1,
                             "reward_basis": "earnings", "max_turns": 2})
        run(env.reset(0))
        inner = env._env
        order = inner.active_order()
        # Deliver by teleport: the wrapper's step is not under test here.
        inner.node_id = order.pickup.kerb_node
        assert inner.collect().ok
        inner.node_id = order.dropoff.kerb_node
        assert inner.hand_over().ok
        assert not inner.finished, (
            "endless dispatch must refill instead of ending the shift")
        assert len(inner.live_orders()) == 1, "a fresh order should be live"
        assert inner.summary()["earnings"] > 0
        run(env.close())
