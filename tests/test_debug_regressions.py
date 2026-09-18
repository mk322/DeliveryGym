"""Regressions for the defects two independent reviews found in the courier env.

Each test names the thing that was wrong and asserts the property that makes it
stay fixed, rather than asserting the current wording. The two in the first class
are the ones that invalidated the measurement: both let a policy score as though
it could see without ever decoding a pixel.
"""

from __future__ import annotations

import re
import os
from pathlib import Path

import pytest

from embodiedbench.agent.courier.frame_alias import FrameAliases
from embodiedbench.agent.courier.session import CourierSession
from embodiedbench.agent.courier.tools import TOOLS_BY_NAME
from embodiedbench.compiler.road_network import build_road_network
from embodiedbench.runtime.city.courier_env import CourierEnv, Stride

MAPS = Path("vendor/vagen/vagen/envs/deliverybench/maps/citycore-paris")
STREETS = Path(os.environ.get("ALBUMS_DIR", "/data/albums")) / Path("paris_streets_v2/citycore-paris")
SIGNALS = Path(os.environ.get("ALBUMS_DIR", "/data/albums")) / Path("paris_signals_kerb/citycore-paris")
OBSTACLES = Path(os.environ.get("ALBUMS_DIR", "/data/albums")) / Path("paris_obstacles/citycore-paris")

# The hazard vocabulary the albums bake into their filenames.
HAZARD_IN_NAME = re.compile(r"road_block|slow_pedestrian|_red\b|_green\b")


@pytest.fixture(scope="module")
def paris():
    return build_road_network(MAPS, map_name="citycore-paris")


def make(paris, *, tier="solo", stride="block", seed=0, albums=True):
    kwargs = {}
    if albums and STREETS.exists():
        kwargs = {"album_root": STREETS,
                  "signal_album_root": SIGNALS,
                  "obstacle_album_root": OBSTACLES}
    env = CourierEnv(paris, seed=seed, difficulty=tier, stride=stride, **kwargs)
    env.reset()
    return env


def drive(session, env, turns=25):
    """Walk the first candidate repeatedly, yielding each observation."""
    for _ in range(turns):
        yield session.observe()
        rows = env.candidates()
        if not rows or session.finished:
            return
        session.step(f'THOUGHT: step\n```\n'
                     f'walk_to("{rows[0]["street"]}", "{rows[0]["heading"]}")\n```')


class TestTheMeasurementCannotBeShortCircuited:
    """A policy must not be able to score without looking."""

    def test_a_refusal_never_quotes_a_distance(self, paris):
        """It was a rangefinder that beat walking on price.

        A block costs 13-26 s to walk and a refusal cost 5 s, so walk-probe-walk
        read the sign of the change and found any door without a photograph.
        """
        env = make(paris, albums=False)
        for call, order_end in ((env.collect, "pickup"), (env.hand_over, "dropoff")):
            outcome = call()
            if outcome.ok:
                continue
            assert "m away" not in outcome.message
            assert not re.search(r"\b\d+\s*m\b", outcome.message), outcome.message

    @pytest.mark.skipif(not STREETS.exists(), reason="albums not mounted")
    def test_no_frame_handed_to_a_policy_names_its_hazard(self, paris):
        """39.9% of served frames used to say ``road_block`` in the path, and
        ``RUNNING.md`` passes ``frame.path`` straight to the model."""
        seen = 0
        for seed in range(3):
            env = make(paris, seed=seed)
            session = CourierSession(env, city="Paris")
            for observation in drive(session, env):
                for frame in observation.frames:
                    if frame.kind != "photograph" or not frame.path:
                        continue
                    seen += 1
                    assert not HAZARD_IN_NAME.search(Path(frame.path).name), frame.path
        assert seen > 100, "drove too little to be evidence"

    @pytest.mark.skipif(not STREETS.exists(), reason="albums not mounted")
    def test_the_privileged_reference_can_still_read_frame_names(self, paris):
        """The rename is at the harness boundary only.

        ``ObservationOnlyCourier(sighted=True)`` is documented to read the
        frame's *name* as a stand-in for perfect recognition; it reads
        ``env.candidates()``, which must keep its readable album paths or the
        ceiling arm stops measuring anything.
        """
        env = make(paris)
        named = [r for r in env.candidates() if r.get("image")]
        assert named, "no candidate carried an image"
        assert all("/" in str(r["image"]) for r in named)

    def test_aliases_are_stable_and_shared(self, tmp_path):
        """Same picture, same name -- so replay and dedup still work."""
        aliases = FrameAliases(root=tmp_path / "frames")
        source = tmp_path / "toward_s1_n2_road_block.png"
        source.write_bytes(b"not really a png, but bytes are bytes")
        first, second = aliases.alias(str(source)), aliases.alias(str(source))
        assert first == second
        assert "road_block" not in first
        assert Path(first).exists()
        assert aliases.alias("") == ""
        assert aliases.alias(str(tmp_path / "missing.png")) == ""


class TestTheNumbersTheCourierIsToldAreTrue:
    def test_consecutive_house_numbers_are_near_each_other(self, paris):
        """N and N+1 must be across the road, not twelve junctions apart.

        Two per-side counters each stepping by 2 kept each side monotone and
        still desynchronised them, because the sides carry different numbers of
        buildings. The courier reads both sides at once, so it saw the
        interleaving: ``2 | 4,6 | 1,8 | 3,5,10 | 7`` on Rue Oberkampf, and a
        Rue du Bac whose whole even side was one No. 2 past No. 13.
        """
        from embodiedbench.compiler.road_network import project_to_polyline

        env = make(paris, albums=False)
        worst = 0
        for street in paris.streets:
            nodes = [n for n in paris.nodes.values() if n.street_index == street.index]
            if len(nodes) < 5:
                continue
            nodes.sort(key=lambda n: project_to_polyline(n.position, street.polyline)[1])
            first_seen: dict[int, int] = {}
            for i, node in enumerate(nodes):
                for number in re.findall(r"\d+", env.house_numbers_near(node.id) or ""):
                    first_seen.setdefault(int(number), i)
            for number, index in first_seen.items():
                if number + 1 in first_seen:
                    worst = max(worst, abs(first_seen[number + 1] - index))
        # Was 12 before the shared cursor. Some offset is inherent -- the two
        # sides are read from one spot -- but it has to stay small enough that
        # "walk the way the numbers climb" is sound advice.
        assert worst <= 6, f"No. N and N+1 are {worst} junctions apart"


    def test_the_quoted_reach_is_what_walk_to_actually_walks(self, paris):
        """At block stride the row quoted the next *waypoint* while the action
        walked the whole block, so three numbers described one leg: the row said
        18 m, the route said "1 junction, 29 m", the outcome said "29 m, through
        2 junctions"."""
        checked = 0
        for seed in range(6):
            env = make(paris, stride=Stride.BLOCK, seed=seed, albums=False)
            for _ in range(20):
                rows = env.candidates()
                if not rows:
                    break
                row = rows[0]
                quoted, junctions = row["reach_m"], row["reach_junctions"]
                outcome = env.walk_to(*env.street_at(row["k"]))
                if not outcome.ok:
                    break
                assert outcome.walked_m == pytest.approx(quoted, abs=0.6)
                assert f"through {junctions} junction" in outcome.message
                checked += 1
        assert checked > 50

    def test_waypoint_stride_still_quotes_the_waypoint(self, paris):
        """The block reach is a block-stride concept; at waypoint stride one call
        is one waypoint and the old label is the correct one."""
        env = make(paris, stride=Stride.WAYPOINT, albums=False)
        for row in env.candidates():
            assert "reach_m" not in row
            assert row["distance_m"] > 0


class TestThePromptDescribesTheWorldItActuallyHas:
    def test_no_tool_is_advertised_that_cannot_be_called(self, paris):
        """``follow_street`` was named in the skills prose at block stride, where
        it is not dispatchable, so a policy following its own instructions lost a
        turn to a format error. The symmetry check covered the tool list and not
        the prose around it."""
        for stride in Stride.ALL:
            env = make(paris, stride=stride, albums=False)
            session = CourierSession(env, city="Paris")
            prompt = session.system_prompt()
            allowed = set(env.allowed_tool_names())
            mentioned = {
                name for name in TOOLS_BY_NAME
                if re.search(rf"\b{re.escape(name)}\s*\(", prompt)
            }
            assert mentioned <= allowed, (
                f"{stride}: prompt advertises {sorted(mentioned - allowed)}"
            )


class TestTheBodyDoingTheDelivering:
    """Embodiment: speed, stamina, what it costs to stop, and whose viewpoint."""

    def test_a_faster_body_spends_less_of_the_clock(self, paris):
        from embodiedbench.tasks.courier_oracle import ObservationOnlyCourier

        times = {}
        for name in ("human_on_foot", "human_on_scooter"):
            env = make(paris, tier="pair", albums=False)
            env = CourierEnv(paris, seed=0, difficulty="pair", stride="block",
                             embodiment=name)
            env.reset()
            ObservationOnlyCourier(env, max_steps=9000).run(0)
            summary = env.summary()
            times[name] = summary["sim_seconds"]
            assert summary["walked_m"] > 0
        assert times["human_on_scooter"] < times["human_on_foot"], times

    def test_stamina_drains_with_distance_not_time(self, paris):
        """A scooter covering the same ground has done less work, not more."""
        spent = {}
        for name in ("human_on_foot", "human_on_scooter", "human_in_car"):
            env = CourierEnv(paris, seed=0, difficulty="solo", stride="block",
                             embodiment=name)
            env.reset()
            for _ in range(6):
                rows = env.candidates()
                if not rows or not env.walk_to(*env.street_at(rows[0]["k"])).ok:
                    break
            spent[name] = env.summary()["stamina_spent"]
        assert spent["human_on_foot"] > spent["human_on_scooter"] > 0
        assert spent["human_in_car"] == 0.0

    def test_a_tired_body_slows_down_rather_than_stopping(self, paris):
        """A hard stop turns one bad estimate into an unfinishable episode."""
        env = CourierEnv(paris, seed=0, difficulty="solo", embodiment="human_on_foot")
        env.reset()
        fresh = env.travel_speed_cm_s()
        env.stamina = 0.0
        assert env.tired
        assert env.travel_speed_cm_s() < fresh
        assert env.travel_speed_cm_s() > 0, "a spent courier must still move"

    def test_a_declared_but_unmeasured_body_refuses_to_run(self, paris):
        for name in ("robot_dog", "humanoid_robot"):
            with pytest.raises(ValueError, match="not characterised"):
                CourierEnv(paris, seed=0, difficulty="solo", embodiment=name)

    def test_an_unknown_body_is_refused(self, paris):
        with pytest.raises(ValueError, match="unknown embodiment"):
            CourierEnv(paris, seed=0, difficulty="solo", embodiment="unicycle")

    def test_walking_reports_when_it_was_shown_the_wrong_viewpoint(self, paris):
        """Until the pavement album is baked a walker is served carriageway
        frames. That is a debt, and the summary has to carry it rather than let
        a run look like evidence about a pedestrian policy."""
        env = CourierEnv(paris, seed=0, difficulty="solo", embodiment="human_on_foot")
        env.reset()
        summary = env.summary()
        assert summary["viewpoint_expected"] == "pavement"
        assert summary["viewpoint_matches_embodiment"] == (
            summary["viewpoint_served"] == "pavement"
        )

    def test_a_rider_is_entitled_to_the_carriageway_album(self, paris):
        env = CourierEnv(paris, seed=0, difficulty="solo", embodiment="human_on_scooter")
        env.reset()
        assert env.summary()["viewpoint_matches_embodiment"] is True


PAVEMENT = Path(os.environ.get("ALBUMS_DIR", "/data/albums")) / Path("paris_streets_pavement/citycore-paris")


class TestTheWalkerSeesThePavement:
    """The viewpoint a body is shown has to be the one it stands at.

    Every album in the repository was baked from the carriageway centreline --
    measured offset 0.00 m, including the one called ``paris_signals_kerb`` --
    so a courier on foot was being trained to recognise a place it can never
    stand. ``paris_streets_pavement`` is the same 856 approaches shot from the
    footway.
    """

    @pytest.mark.skipif(not PAVEMENT.exists(), reason="pavement album not baked")
    def test_walking_with_the_pavement_album_is_a_matched_viewpoint(self, paris):
        env = CourierEnv(paris, seed=0, difficulty="solo", stride="block",
                         album_root=STREETS, pavement_album_root=PAVEMENT,
                         embodiment="human_on_foot")
        env.reset()
        summary = env.summary()
        assert summary["viewpoint_expected"] == "pavement"
        assert summary["viewpoint_served"] == "pavement"
        assert summary["viewpoint_matches_embodiment"] is True
        # and the frames really come from that album
        served = [r["image"] for r in env.candidates() if r.get("image")]
        assert served and all("paris_streets_pavement" in p for p in served), served

    @pytest.mark.skipif(not PAVEMENT.exists(), reason="pavement album not baked")
    def test_a_rider_still_gets_the_carriageway_album(self, paris):
        """Handing a scooter the footway frames would be the same error the
        other way round."""
        env = CourierEnv(paris, seed=0, difficulty="solo", stride="block",
                         album_root=STREETS, pavement_album_root=PAVEMENT,
                         embodiment="human_on_scooter")
        env.reset()
        assert env.summary()["viewpoint_served"] == "carriageway"
        served = [r["image"] for r in env.candidates() if r.get("image")]
        assert served and all("paris_streets_pavement" not in p for p in served)

    @pytest.mark.skipif(not PAVEMENT.exists(), reason="pavement album not baked")
    def test_the_pavement_album_covers_every_approach_the_other_one_does(self):
        """A partial album silently drops the courier's eyes on some streets."""
        carriage = {p.relative_to(STREETS / "images")
                    for p in (STREETS / "images").rglob("*.png")}
        footway = {p.relative_to(PAVEMENT / "images")
                   for p in (PAVEMENT / "images").rglob("*.png")}
        missing = carriage - footway
        assert not missing, f"{len(missing)} approaches missing, e.g. {sorted(missing)[:3]}"


PAVEMENT_OBSTACLES = Path(os.environ.get("ALBUMS_DIR", "/data/albums")) / Path("paris_obstacles_pavement/citycore-paris")


class TestTheViewpointDoesNotAnnounceTheHazard:
    """Both albums move together, or the viewpoint becomes the tell.

    With a pavement street album and a carriageway obstacle album, a walker got
    footway frames on clear streets and centreline frames wherever an obstacle
    stood. Measured: 19.5% of its frames came from the carriageway album and
    *every one of them was an obstacle*, so a policy could read the hazard off
    the camera position without decoding the picture -- the filename leak again
    in a different disguise.
    """

    @pytest.mark.skipif(not (PAVEMENT.exists() and PAVEMENT_OBSTACLES.exists()),
                        reason="pavement albums not baked")
    def test_a_walker_never_sees_a_carriageway_frame(self, paris):
        seen = 0
        for seed in range(3):
            env = CourierEnv(paris, seed=seed, difficulty="shift", stride="block",
                             embodiment="human_on_foot",
                             album_root=STREETS, pavement_album_root=PAVEMENT,
                             signal_album_root=SIGNALS,
                             obstacle_album_root=OBSTACLES,
                             pavement_obstacle_album_root=PAVEMENT_OBSTACLES)
            env.reset()
            for _ in range(30):
                for row in env.candidates():
                    path = row.get("image")
                    if not path:
                        continue
                    seen += 1
                    assert "pavement" in path, path
                rows = env.candidates()
                if not rows or not env.walk_to(*env.street_at(rows[0]["k"])).ok:
                    break
        assert seen > 200, "walked too little to be evidence"

    @pytest.mark.skipif(not PAVEMENT.exists(), reason="pavement album not baked")
    def test_mixing_the_two_viewpoints_is_refused(self, paris):
        with pytest.raises(ValueError, match="pavement obstacle album"):
            CourierEnv(paris, seed=0, difficulty="solo",
                       embodiment="human_on_foot",
                       album_root=STREETS, pavement_album_root=PAVEMENT,
                       obstacle_album_root=OBSTACLES)


class TestAReasoningModelCanBeEvaluatedAtAll:
    """The reply contract must not reject a model for thinking out loud.

    Qwen3.5-9B writes a candidate call inside its <think> block and the real one
    after it. Parsing the whole reply finds two fenced blocks and refuses, which
    scored 82.5% format errors against a model that was never malformed. Since
    reasoning models are most of what a benchmark like this now has to measure,
    that was a hole in the harness, not a property of the policy.
    """

    def test_only_the_answer_after_the_thought_is_parsed(self):
        from embodiedbench.agent.courier.loop import parse_reply

        reply = (
            "<think>\nI could go north.\n```\nwalk_to(\"Rue de Grenelle\", \"east\")\n```\nNo, south.\n</think>\n\n"
            "THOUGHT: south it is.\n```\nwalk_to(\"Rue de Grenelle\", \"east\")\n```"
        )
        assert parse_reply(reply, {"walk_to"}).render() == "walk_to(\"Rue de Grenelle\", \"east\")"

    def test_a_reply_without_a_thought_block_is_untouched(self):
        from embodiedbench.agent.courier.loop import parse_reply

        assert parse_reply("THOUGHT: go\n```\nwalk_to(\"Rue de Grenelle\", \"east\")\n```",
                           {"walk_to"}).render() == "walk_to(\"Rue de Grenelle\", \"east\")"

    def test_a_thought_that_never_closes_is_still_a_format_error(self):
        """An unterminated thought has no answer in it, and inventing one from
        the rehearsal is the guessing the parser exists to avoid."""
        from embodiedbench.agent.courier.loop import FormatError, parse_reply

        with pytest.raises(FormatError):
            parse_reply("<think>\nmaybe\n```\nwalk_to(\"Rue de Grenelle\", \"east\")\n```\nor maybe not",
                        {"walk_to"})
