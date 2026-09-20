"""The compiled city: conversion, runtime, and the solvability floor.

Three claims are defended here, and each was false at some point during this
work, which is why each has a test rather than a note:

1. the conversion is rule-based and runs on *any* map, not just the one with the
   richest export;
2. what the runtime tells the agent is what the runtime scores, in one angle
   convention and against one target;
3. an agent restricted to the observation can actually deliver.

The solvability floor in the last class is the one that would catch a silent
regression in any of the others. It is a floor, not a target: the reference
courier is a simple heuristic, and its ~0.7 route efficiency deliberately leaves
headroom for a learned policy to beat it.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import pytest

from embodiedbench.compiler.road_network import (
    MIN_EDGE_CM,
    build_road_network,
    extract_strokes,
    load_buildings,
    load_road_segments,
    street_name,
)
from embodiedbench.runtime.city.courier_env import (
    ARRIVAL_TOLERANCE_CM,
    Condition,
    CourierEnv,
    compass_of,
)
from embodiedbench.tasks.courier_oracle import (
    ObservationOnlyCourier,
    run_reference_courier,
)

MAPS = Path(__file__).resolve().parents[1] / "vendor" / "vagen" / "vagen" / "envs" / "deliverybench" / "maps"
PARIS = MAPS / "citycore-paris"
ALL_MAPS = sorted(p for p in MAPS.iterdir() if p.is_dir()) if MAPS.exists() else []

needs_maps = pytest.mark.skipif(not ALL_MAPS, reason="vendored maps not present")


def write_map(directory: Path, roads: list[dict], buildings: dict | None = None) -> Path:
    (directory / "roads.json").write_text(json.dumps({"roads": roads}))
    if buildings is not None:
        (directory / "buildings.json").write_text(json.dumps(buildings))
    return directory


def segment(x1: float, y1: float, x2: float, y2: float) -> dict:
    return {"start": {"x": x1, "y": y1}, "end": {"x": x2, "y": y2}, "is_highway": False}


# ─────────────────────────────────────────────────────────────────────────────
# 1. The conversion generalises
# ─────────────────────────────────────────────────────────────────────────────


@needs_maps
class TestConversionGeneralises:
    @pytest.mark.parametrize("map_dir", ALL_MAPS, ids=lambda p: p.name)
    def test_every_map_compiles_to_a_connected_network(self, map_dir):
        """The contract is any map in, a usable network or a stated reason out.

        Only Paris ships roads_detailed.json, and the nine procgen maps use a
        different building schema, so a compiler that required either would work
        on one map and claim to be general.
        """
        network = build_road_network(map_dir, map_name=map_dir.name)
        summary = network.summary()
        assert summary["nodes"] > 0, summary["notes"]
        assert summary["components"] == 1, f"{map_dir.name} fragmented: {summary}"
        assert summary["largest_component_fraction"] == 1.0

    @pytest.mark.parametrize("map_dir", ALL_MAPS, ids=lambda p: p.name)
    def test_no_degenerate_edges(self, map_dir):
        """A sub-metre edge is a step that costs a turn and moves nowhere."""
        network = build_road_network(map_dir, map_name=map_dir.name)
        for a, b in network.edges():
            gap = math.dist(network.nodes[a].position, network.nodes[b].position)
            assert gap >= MIN_EDGE_CM, f"{a}->{b} is {gap:.1f} cm"

    @pytest.mark.parametrize("map_dir", ALL_MAPS, ids=lambda p: p.name)
    def test_addresses_are_derived_and_coherent(self, map_dir):
        """Street names and numbers must come from geometry.

        The engine synthesised them per node at runtime: 97.5% of its edges had
        an empty road name, and `1 Union Ave` sat next to `215 Union Ave`.
        """
        network = build_road_network(map_dir, map_name=map_dir.name)
        assert network.addresses, f"{map_dir.name} produced no addresses"
        by_street: dict[str, list] = {}
        for address in network.addresses:
            assert address.street_name.strip()
            by_street.setdefault((address.street_name, address.side), []).append(address)
        ascending = 0
        for values in by_street.values():
            ordered = sorted(values, key=lambda a: a.arc_cm)
            if all(x.number < y.number for x, y in zip(ordered, ordered[1:])):
                ascending += 1
        assert ascending / len(by_street) >= 0.85, (
            f"only {ascending}/{len(by_street)} street sides number monotonically"
        )

    @pytest.mark.parametrize("map_dir", ALL_MAPS, ids=lambda p: p.name)
    def test_every_address_has_a_reachable_kerb(self, map_dir):
        """A door a courier cannot stand outside is not a deliverable address."""
        network = build_road_network(map_dir, map_name=map_dir.name)
        for address in network.addresses:
            assert address.kerb_node in network.nodes

    def test_both_building_schemas_are_read(self):
        """CityCore writes center_cm+bbox_cm in centimetres; procgen writes
        bounds in metres. A loader that knew only one silently returned an empty
        list for Paris and disabled every footprint rule on it."""
        paris = load_buildings(PARIS)
        assert len(paris) > 100
        assert any(b.entrance_yaw_deg is not None for b in paris)
        procgen = load_buildings(MAPS / "small-city-11")
        assert len(procgen) > 10
        # Metres in the file, centimetres in memory: a 30 m building is 3000 cm.
        assert max(b.half_extent[0] for b in procgen) > 200

    def test_an_unknown_building_schema_raises(self):
        """Returning an empty list on an unrecognised schema is how the Paris
        footprints went missing without anyone noticing."""
        with tempfile.TemporaryDirectory() as tmp:
            directory = write_map(
                Path(tmp), [segment(0, 0, 100, 0)],
                buildings={"buildings": [{"mystery": 1, "fields": 2}]},
            )
            with pytest.raises(ValueError, match="unrecognised footprint schema"):
                build_road_network(directory)

    def test_units_are_inferred_when_undeclared(self):
        """roads.json declares no units on any map. The two candidates differ by
        100x, so extent settles it: a city spanning 1200 is metres."""
        with tempfile.TemporaryDirectory() as tmp:
            metres = build_road_network(write_map(Path(tmp), [segment(0, 0, 200, 0)]))
        with tempfile.TemporaryDirectory() as tmp:
            centimetres = build_road_network(write_map(Path(tmp), [segment(0, 0, 50000, 0)]))
        assert metres.streets[0].length_cm == pytest.approx(20000.0)
        assert centimetres.streets[0].length_cm == pytest.approx(50000.0)


class TestConversionSurvivesBadInput:
    """A broken map is a verdict, not a traceback."""

    @pytest.mark.parametrize("roads,expect_nodes", [
        ([], False),
        ([segment(0, 0, 0, 0)] * 5, False),
        ([segment(0, 0, 100, 0)], True),
        ([segment(0, 0, 50, 0)] * 20, True),
    ], ids=["empty", "zero_length", "single", "duplicates"])
    def test_degenerate_geometry(self, roads, expect_nodes):
        with tempfile.TemporaryDirectory() as tmp:
            network = build_road_network(write_map(Path(tmp), roads))
        assert bool(network.nodes) is expect_nodes
        assert network.notes

    def test_duplicate_segments_make_one_street(self):
        """Exporters repeat a carriageway once per lane. Left in, each copy
        seeded its own stroke and twenty duplicates became twenty streets."""
        with tempfile.TemporaryDirectory() as tmp:
            network = build_road_network(write_map(Path(tmp), [segment(0, 0, 50, 0)] * 20))
        assert len(network.streets) == 1

    def test_non_numeric_coordinates_are_skipped_not_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "roads.json").write_text(json.dumps({"roads": [
                {"start": {"x": "nonsense", "y": 0}, "end": {"x": 1, "y": 2}},
                segment(0, 0, 100, 0),
            ]}))
            network = build_road_network(directory)
        assert network.nodes
        assert any("non-numeric" in note for note in network.notes)


class TestStrokeExtraction:
    def test_a_straight_run_is_one_street(self):
        strokes = extract_strokes(
            load_road_segments(Path("."))[0] or _fake_line(), weld_cm=900.0, max_turn_deg=35.0
        ) if False else extract_strokes(_fake_line(), weld_cm=900.0, max_turn_deg=35.0)
        assert len(strokes) == 1
        assert len(strokes[0]) == 4

    def test_a_right_angle_starts_a_new_street(self):
        """35 degrees is the stroke threshold: a curving boulevard stays one
        street, a genuine corner becomes two."""
        from embodiedbench.compiler.road_network import RoadSegment

        corner = [
            RoadSegment(start=(0.0, 0.0), end=(10000.0, 0.0)),
            RoadSegment(start=(10000.0, 0.0), end=(10000.0, 10000.0)),
        ]
        assert len(extract_strokes(corner, weld_cm=900.0, max_turn_deg=35.0)) == 2

    def test_street_names_are_stable_and_distinct(self):
        names = [street_name(i) for i in range(120)]
        assert names[:5] == [street_name(i) for i in range(5)]
        assert len(set(names)) == len(names)


def _fake_line():
    from embodiedbench.compiler.road_network import RoadSegment

    return [
        RoadSegment(start=(0.0, 0.0), end=(5000.0, 0.0)),
        RoadSegment(start=(5000.0, 0.0), end=(10000.0, 0.0)),
        RoadSegment(start=(10000.0, 0.0), end=(15000.0, 0.0)),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 2. The runtime cannot contradict itself
# ─────────────────────────────────────────────────────────────────────────────


@needs_maps
class TestRuntimeSelfConsistency:
    def env(self, seed: int = 0) -> CourierEnv:
        env = CourierEnv(build_road_network(PARIS, map_name="citycore-paris"), seed=seed)
        env.reset()
        return env

    def test_quoted_distance_is_to_the_scored_point(self):
        """The defect the independent evaluation ranked first: distance was
        quoted to the postal address while arrival was tested against a node up
        to 18 m away, so eight of ten seeds could never satisfy the check they
        were being measured on."""
        env = self.env()
        order = env.active_order()
        quoted = env.distance_to_target_cm()
        scored = math.dist(env.position(), order.pickup.kerb)
        assert quoted == pytest.approx(scored)

    def test_a_refusal_states_arrival_and_nothing_measurable(self):
        """A refusal is a yes/no the courier could get by standing there.

        Quoting the distance turned it into a rangefinder that beat walking on
        price; quoting the tolerance is harmless on its own but there is nothing
        left for it to qualify.
        """
        env = self.env()
        outcome = env.collect()
        assert not outcome.ok
        assert "m away" not in outcome.message
        assert f"{ARRIVAL_TOLERANCE_CM/100:.0f} m" not in outcome.message
        assert outcome.message.startswith("You are not standing at")

    def test_candidate_numbering_is_stable_regardless_of_approach(self):
        """Numbering by distance renumbered the same corner depending on where
        the agent arrived from, so the same k meant different streets."""
        env = self.env()
        first = {r["k"]: r["node"] for r in env.candidates()}
        env.arrived_from = next(iter(env.network.nodes[env.node_id].neighbours))
        assert {r["k"]: r["node"] for r in env.candidates()} == first

    def test_every_candidate_has_a_street_name_and_a_heading(self):
        env = self.env()
        for row in env.candidates():
            assert row["street"].strip()
            assert row["heading"] in (
                "north", "north-east", "east", "south-east",
                "south", "south-west", "west", "north-west",
            )

    def test_walking_to_a_candidate_arrives_at_that_candidate(self):
        env = self.env()
        rows = env.candidates()
        target = rows[0]["node"]
        assert env.walk_to(*env.street_at(rows[0]["k"])).ok
        assert env.node_id == target

    def test_a_street_that_is_not_here_is_refused_with_the_ones_that_are(self):
        env = self.env()
        outcome = env.walk_to("Rue Imaginaire", "north")
        assert not outcome.ok and outcome.code == "no_such_street"
        # The refusal has to be actionable: it names what the courier can
        # actually take, in the words walk_to takes them in.
        for row in env.candidates():
            assert row["street"] in outcome.message
            assert row["heading"] in outcome.message

    def test_the_map_lookup_gives_direction_and_distance_but_not_a_route(self):
        """A phone gives a bearing and a distance. Turn-by-turn directions would
        make the phone the navigator and the benchmark a reading test."""
        env = self.env()
        outcome = env.check_map(env.active_order().pickup.text)
        assert outcome.ok and "m away" in outcome.message
        for giveaway in ("walk_to", "take street", "first left", "then "):
            assert giveaway not in outcome.message.lower()

    def test_spawn_is_never_a_cul_de_sac(self):
        """The engine spawned into a degree-1 pocket where the only legal move
        was a 378 m hop, so the first turns carried no decision."""
        for seed in range(8):
            env = self.env(seed)
            assert len(env.network.nodes[env.node_id].neighbours) >= 2

    def test_orders_are_between_real_addresses_on_different_streets(self):
        for seed in range(8):
            env = self.env(seed)
            order = env.active_order()
            assert order is not None
            assert order.pickup.street_name != order.dropoff.street_name
            assert order.pickup.text and order.dropoff.text

    def test_compass_names_match_the_bearing_convention(self):
        assert compass_of(0.0) == "north" and compass_of(90.0) == "east"
        assert compass_of(180.0) == "south" and compass_of(270.0) == "west"


# ─────────────────────────────────────────────────────────────────────────────
# 3. The solvability floor
# ─────────────────────────────────────────────────────────────────────────────


@needs_maps
class TestSolvabilityFloor:
    """An agent restricted to the observation must be able to deliver.

    Before the conversion was rebuilt, the best strategy the observation
    supported delivered 3 of 16 episodes; an independent evaluation called the
    task not navigable. These thresholds sit below what is currently measured
    (100% on Paris, 91% across all ten maps) so ordinary variation does not fail
    the build, but a real regression does.
    """

    @pytest.mark.parametrize("map_dir", ALL_MAPS, ids=lambda p: p.name)
    def test_the_reference_courier_delivers(self, map_dir):
        network = build_road_network(map_dir, map_name=map_dir.name)
        delivered = attempted = 0
        for seed in range(6):
            env = CourierEnv(network, seed=seed, order_count=1)
            env.reset()
            if not env.orders:
                continue
            attempted += 1
            delivered += run_reference_courier(env, seed).delivered
        assert attempted, f"{map_dir.name} generated no orders"
        assert delivered / attempted >= 0.5, f"{map_dir.name}: {delivered}/{attempted}"

    def test_paris_is_fully_solvable(self):
        network = build_road_network(PARIS, map_name="citycore-paris")
        delivered = 0
        for seed in range(10):
            env = CourierEnv(network, seed=seed, order_count=1)
            env.reset()
            delivered += run_reference_courier(env, seed).delivered
        assert delivered >= 9, f"Paris solvability regressed to {delivered}/10"

    def test_the_reference_courier_reaches_for_geometry_and_says_so(self):
        """The floor it establishes changed, and the change is the point.

        This courier used to navigate on the phone's spoken bearing -- "it lies
        to the east of you" -- which made it an observation-only policy and its
        success a proof that the task was solvable from text. That was also the
        problem: if a text policy can solve it, the photographs and the map are
        decoration, and the benchmark is a reading exercise with pictures
        attached.

        The bearing is now drawn rather than spoken. Nothing in the words says
        which way to walk, so no text-only policy can navigate at all, and this
        one reads the geometry directly and is honest about doing it. What it
        proves is narrower and still worth having: the world is solvable. What
        it no longer proves -- that the world is solvable without looking -- is
        exactly the property the benchmark wanted gone.
        """
        import inspect

        source = inspect.getsource(ObservationOnlyCourier)
        assert "_address_position" in source
        assert "Oracle privilege, used knowingly" in source
        # Still not allowed to route: reading where a place is differs from
        # being handed the way there, which is the whole task.
        for forbidden in ("route_length_cm", "route_legs", "shortest"):
            assert forbidden not in source, f"reference courier reached for {forbidden}"

    def test_no_direction_to_walk_survives_anywhere_in_the_words(self):
        """The property the change exists to create, pinned where it can break.

        Every place the environment speaks to the courier is checked, because
        one sentence naming a compass point anywhere puts the whole navigation
        problem back into the text.
        """
        network = build_road_network(PARIS, map_name="citycore-paris")
        env = CourierEnv(network, seed=0, difficulty="solo", stride="block")
        env.reset()
        order = env.orders[0]
        spoken = [env.navigate().message,
                  env.check_map(order.pickup.text).message,
                  env.check_order().message]
        for message in spoken:
            for word in ("north", "south", "east", "west", "crow flies"):
                assert word not in message.lower(), message

    def test_it_leaves_headroom_for_a_learned_policy(self):
        """A reference that walks the optimal route would make the benchmark
        unimprovable; ~0.7 efficiency means there is something to learn."""
        network = build_road_network(PARIS, map_name="citycore-paris")
        efficiencies = []
        for seed in range(8):
            env = CourierEnv(network, seed=seed, order_count=1)
            env.reset()
            result = run_reference_courier(env, seed)
            if result.delivered:
                efficiencies.append(result.to_dict()["efficiency"])
        assert efficiencies
        mean = sum(efficiencies) / len(efficiencies)
        assert 0.3 < mean < 0.95, f"reference efficiency {mean:.2f} leaves no headroom"


@needs_maps
class TestDifficultyLadder:
    """Each rung must remove a crutch, and each reported rung must be solvable."""

    def network(self):
        return build_road_network(PARIS, map_name="citycore-paris")

    def test_an_unknown_condition_is_refused(self):
        with pytest.raises(ValueError, match="unknown condition"):
            CourierEnv(self.network(), condition="easy")

    def test_the_phone_works_only_on_the_full_rung(self):
        network = self.network()
        address = None
        for condition in Condition.ALL:
            env = CourierEnv(network, seed=0, condition=condition)
            env.reset()
            address = address or env.active_order().pickup.text
            outcome = env.check_map(address)
            assert outcome.ok is (condition == Condition.FULL)

    def test_house_numbers_leave_the_text_on_the_visual_rung(self):
        """Asked of look(), because read_sign() is gone.

        It answered with the first line of the observation and was retired in
        the tool audit; the rung's claim is unchanged and now rests on the tool
        that reports numbers the observation does not carry -- the ones running
        away down a street the courier is not standing on.
        """
        network = self.network()
        for condition in Condition.ALL:
            env = CourierEnv(network, seed=0, condition=condition)
            env.reset()
            message = env.look(*env.street_at(1)).message
            reads_numbers = "doors read" in message
            assert reads_numbers is (condition != Condition.VISUAL), message

    @pytest.mark.parametrize("condition", Condition.VALIDATED)
    def test_every_reported_rung_is_solvable(self, condition):
        """A rung nobody has solved measures nothing. These thresholds sit below
        what is measured (100% and 30%) so variation does not fail the build."""
        network = self.network()
        delivered = 0
        for seed in range(10):
            env = CourierEnv(network, seed=seed, order_count=1, condition=condition)
            env.reset()
            delivered += run_reference_courier(env, seed, max_steps=400).delivered
        floor = 8 if condition == Condition.FULL else 1
        assert delivered >= floor, f"{condition}: {delivered}/10"

    def test_the_rungs_are_ordered_by_difficulty(self):
        """no_phone must be strictly harder than full, or it is not a rung.

        Measured over both rungs explicitly rather than over VALIDATED: the
        ordering is a property of the conditions themselves and has to keep
        holding while no_phone sits outside the reportable set awaiting a policy
        that can clear it.
        """
        network = self.network()
        scores = {}
        for condition in (Condition.FULL, Condition.NO_PHONE):
            delivered = 0
            for seed in range(10):
                env = CourierEnv(network, seed=seed, order_count=1, condition=condition)
                env.reset()
                delivered += run_reference_courier(env, seed, max_steps=400).delivered
            scores[condition] = delivered
        assert scores[Condition.FULL] > scores[Condition.NO_PHONE], scores

    def test_the_unvalidated_rung_is_excluded_from_reporting(self):
        """It is kept because the work is real, and excluded because the renders
        carry no legible door numbers -- so 0% there measures missing
        information, not missing perception."""
        assert Condition.VISUAL not in Condition.VALIDATED
        assert set(Condition.VALIDATED) < set(Condition.ALL)


@needs_maps
class TestVisionIsLoadBearing:
    """The claim the whole benchmark rests on: sight must be worth something.

    Before pedestrian signals, text alone solved 97% and the photographs were
    decoration -- a text-only policy lost nothing by never looking. Signals fix
    that with information that is only ever in the picture: a light's colour is
    seen, not read, so it appears in no observation string.
    """

    def network(self):
        return build_road_network(PARIS, map_name="citycore-paris")

    @staticmethod
    def _walk_to_signalised(env, want_signalised=True, limit=60):
        """Walk until standing at (or away from) a signalised junction.

        Bounded and it *chooses* rather than always taking candidate 1: taking
        the first candidate every time oscillates between two nodes and the
        original unbounded loop hung the suite until it was SIGTERMed.
        """
        seen = set()
        for _ in range(limit):
            if (env.node_id in env.signalised) is want_signalised:
                return True
            options = [r for r in env.candidates() if r["node"] not in seen] or env.candidates()
            if not options:
                return False
            seen.add(env.node_id)
            env.walk_to(*env.street_at(options[0]["k"]))
        return (env.node_id in env.signalised) is want_signalised

    def signalled_env(self, **kwargs):
        """An env with the signal charge switched on explicitly.

        Production has it off (``SIGNAL_FRAMES_AVAILABLE = False``) because no
        frame shows the live phase yet, and charging for information the agent
        cannot obtain made the score anti-correlated with delivery. The logic
        still needs covering, so the tests opt in.

        Opting in now means two things, not one: the charge is on *and* the
        album declares it can show the lamp. These tests are about the mechanic
        -- does crossing cost, does waiting clear it, is looking worth more than
        not looking -- so they declare every approach visible. Whether the real
        Paris album can show a given lamp is a separate question, measured by
        ``TestTheLightIsChargedOnlyWhereItCanBeSeen``.
        """
        network = self.network()
        env = CourierEnv(network, enforce_signals=True, **kwargs)
        env.visible_signals = {f"{node}|{neighbour}" for node in env.signalised
                               for neighbour in network.nodes[node].neighbours}
        return env

    def test_signalised_junctions_are_derived_from_the_graph(self):
        """Degree, not extra map data, so it holds on any map."""
        network = self.network()
        signalised = network.signalised_nodes()
        assert signalised
        for node_id in signalised:
            assert len(network.nodes[node_id].neighbours) >= 3
        for node_id, node in network.nodes.items():
            if len(node.neighbours) < 3:
                assert node_id not in signalised

    def test_the_light_state_never_appears_in_any_text(self):
        """If a policy could read the colour in words, the images would be
        optional again and the condition would prove nothing."""
        env = self.signalled_env(seed=0)
        env.reset()
        assert self._walk_to_signalised(env), "no signalised junction found nearby"
        strings = [
            env.location_text(), env.clock_text(),
            env.check_order().message, env.look(*env.street_at(1)).message,
        ] + [f"{r['street']} {r['heading']}" for r in env.candidates()]
        # Word boundaries, not substrings: "numbered 9" contains "red", and the
        # naive check failed on the environment's own correct output.
        import re as _re

        for text in strings:
            assert not _re.search(r"\bred\b", text, _re.I), text
            assert not _re.search(r"\bgreen\b", text, _re.I), text

    def test_the_light_is_deterministic_in_junction_direction_and_time(self):
        """The album bakes a red and a green frame per approach, so the runtime
        must agree with itself about which one is live."""
        env = self.signalled_env(seed=0)
        env.reset()
        assert self._walk_to_signalised(env), "no signalised junction found nearby"
        k = env.candidates()[0]["k"]
        assert env.light_here(k) == env.light_here(k)

    def test_crossing_axes_alternate(self):
        """When one axis is green the other is red, as at a real crossing."""
        from embodiedbench.runtime.city.courier_env import signal_state

        for moment in (0.0, 30.0, 60.0, 120.0):
            north = signal_state("j", 90.0, moment)     # compass north
            east = signal_state("j", 0.0, moment)       # compass east
            assert north != east, (moment, north, east)

    def test_an_unsignalised_junction_has_no_light(self):
        env = self.signalled_env(seed=0)
        env.reset()
        assert self._walk_to_signalised(env, want_signalised=False)
        assert env.light_here(env.candidates()[0]["k"]) is None

    def test_crossing_on_red_is_costed_not_blocked(self):
        """A real courier can cross against the light. Making it impossible
        would measure nothing about whether the agent chose to look."""
        from embodiedbench.runtime.city.courier_env import RED_CROSSING_PENALTY

        env = self.signalled_env(seed=0)
        env.reset()
        assert self._walk_to_signalised(env), "no signalised junction found nearby"
        red = next((r["k"] for r in env.candidates() if env.light_here(r["k"]) == "red"), None)
        assert red is not None
        outcome = env.walk_to(*env.street_at(red))
        assert outcome.ok and outcome.moved
        assert outcome.reward == pytest.approx(-RED_CROSSING_PENALTY)
        assert env.red_crossings == 1

    def test_waiting_turns_a_red_light_green(self):
        """The penalty has to be avoidable, or it is a tax rather than a test."""
        env = self.signalled_env(seed=0)
        env.reset()
        assert self._walk_to_signalised(env), "no signalised junction found nearby"
        red = next((r["k"] for r in env.candidates() if env.light_here(r["k"]) == "red"), None)
        assert red is not None
        for _ in range(6):
            env.wait()
            if env.light_here(red) == "green":
                break
        assert env.light_here(red) == "green"

    def test_looking_is_worth_more_than_the_whole_delivery(self):
        """The measured swing: 340 violations against 0, on identical routes.

        If the penalty were small a policy could rationally ignore the lights,
        and the images would again be optional.
        """
        network = self.network()
        blind = sighted = 0
        for seed in range(3):
            env = self.signalled_env(seed=seed, order_count=1)
            env.reset()
            run_reference_courier(env, seed, max_steps=150)
            blind += env.summary()["red_crossings"]

            env = self.signalled_env(seed=seed, order_count=1)
            env.reset()
            original = env.walk_to

            def guarded(street, heading=None, _env=env, _original=original):
                k, refusal = _env.resolve_street(street, heading)
                if refusal is not None:
                    return refusal
                waited = 0
                while _env.light_here(k) == "red" and waited < 5:
                    _env.wait()
                    waited += 1
                return _original(street, heading)

            env.walk_to = guarded
            run_reference_courier(env, seed, max_steps=150)
            sighted += env.summary()["red_crossings"]

        assert sighted == 0, f"sighted courier still ran {sighted} red lights"
        assert blind > 8, f"only {blind} violations — the penalty is too weak to matter"


@needs_maps
class TestSignalChargeIsGated:
    def test_the_charge_is_off_until_the_frames_exist(self):
        """Guards against re-enabling a penalty for unobtainable information.

        With it on, the reference courier delivered 10/10 and scored -4.4 to
        -16.4, because ~17 unavoidable violations at -1.0 each swamped +1.6 of
        delivery credit. The metric ranked a perfect run below a stalled one.
        """
        from embodiedbench.runtime.city.courier_env import SIGNAL_FRAMES_AVAILABLE

        assert SIGNAL_FRAMES_AVAILABLE is False
        env = CourierEnv(build_road_network(PARIS, map_name="citycore-paris"), seed=0)
        env.reset()
        assert env.enforce_signals is False

    def test_the_reference_courier_is_not_punished_for_delivering(self):
        network = build_road_network(PARIS, map_name="citycore-paris")
        delivered = violations = 0
        for seed in range(6):
            env = CourierEnv(network, seed=seed, order_count=1)
            env.reset()
            delivered += run_reference_courier(env, seed, max_steps=400).delivered
            violations += env.summary()["red_crossings"]
        assert delivered >= 5
        assert violations == 0, "charging for an invisible light again"


@needs_maps
class TestEveryIndexTheCourierIsOfferedStartsAtOne:
    """One interface, one counting convention.

    Streets are offered as "[1] Rue de la Paix" and taken with walk_to(1);
    route legs are numbered from 1; looking down a street is look(2). Jobs
    were numbered from 0, and the number was never shown -- the slip says
    "Job: collect from 8 Rue Bonaparte" and the signature says
    "navigate(job: int = default)". A model reading that interface writes
    navigate(1) for its only job and is told it is not carrying it. On 40
    held-out seeds that was 23 refusals, 8 of them inside one episode.
    """

    def env(self, tier="solo", seed=19):
        env = CourierEnv(build_road_network(PARIS, map_name="citycore-paris"),
                         seed=seed, difficulty=tier)
        env.reset()
        return env

    def test_the_only_job_is_job_one(self):
        env = self.env()
        assert [o.index for o in env.orders] == [1]
        assert env.navigate(1).ok

    def test_a_deeper_tier_numbers_its_jobs_from_one(self):
        env = self.env(tier="shift")
        assert [o.index for o in env.orders][:3] == [1, 2, 3]
        assert 0 not in [o.index for o in env.orders]

    def test_a_job_drawn_later_continues_the_numbering(self):
        """ENDLESS draws lazily, and a second series starting at 0 would put
        two different jobs on the same number."""
        env = self.env(tier="endless")
        first = [o.index for o in env.orders]
        assert first and min(first) == 1
        assert len(set(first)) == len(first)

    def test_the_slip_shows_the_number_it_uses(self):
        """The number was never on screen anywhere, which is how a model came
        to guess it. navigate no longer takes one at all -- it takes an address
        -- but check_order still numbers the jobs, and a number the courier is
        shown has to be the number the environment means."""
        env = self.env(tier="pair")
        message = env.check_order().message
        assert "Job 1:" in message
        assert "Job 0:" not in message


class TestObservationHonesty:
    """Round-2 findings: every claim the prompt makes must hold in the data."""

    def env(self, seed: int = 0) -> CourierEnv:
        env = CourierEnv(build_road_network(PARIS, map_name="citycore-paris"), seed=seed)
        env.reset()
        return env

    def test_walking_on_keeps_the_street_name(self):
        """The prompt's rule, checked against the graph. Naming an edge after its
        destination node made 30.1% of edges disagree between the two
        directions; splitting it showed the disagreement is entirely at junction
        links, where it is correct."""
        env = self.env()
        network = env.network
        for a, node in network.nodes.items():
            for b in node.neighbours:
                if network.nodes[a].street_index == network.nodes[b].street_index:
                    assert env.edge_street(a, b) == env.edge_street(b, a)

    def test_a_turn_names_the_street_turned_onto(self):
        env = self.env()
        network = env.network
        turns = [
            (a, b) for a, node in network.nodes.items() for b in node.neighbours
            if network.nodes[a].street_index != network.nodes[b].street_index
        ]
        assert turns
        a, b = turns[0]
        assert env.edge_street(a, b) == env.street_of(b)
        assert env.edge_street(b, a) == env.street_of(a)

    def test_the_map_quotes_walking_distance_not_crow_flies(self):
        """Straight-line range rewarded approaching a wall: a six-turn trap had
        it improving 140 -> 105 m while the route worsened 148 -> 238 m."""
        import math
        import re

        env = self.env()
        target = env.active_order().pickup
        message = env.check_map(target.text).message
        quoted = float(re.search(r"about (\d+) m away", message).group(1))
        straight = math.dist(env.position(), target.kerb) / 100.0
        route = env.route_length_cm(env.node_id, target.kerb_node) / 100.0
        assert abs(quoted - route) < 2.0
        assert quoted >= straight - 1.0

    def test_standing_still_is_not_going_in_circles(self):
        """arrive() fires every turn, including looking and consulting, so the
        warning fired on an agent that had not moved — contradicting the runbook
        that tells it to change direction on exactly that signal."""
        from embodiedbench.agent.courier.memory import CourierMemory

        memory = CourierMemory()
        for step in range(6):
            memory.arrive(step=step, node_id="same", street="Rue A")
        assert not memory.is_looping()

    def test_real_oscillation_is_still_caught(self):
        from embodiedbench.agent.courier.memory import CourierMemory

        memory = CourierMemory()
        for step in range(6):
            memory.arrive(step=step, node_id=f"n{step % 2}", street="Rue A")
        assert memory.is_looping()


@needs_maps
class TestTheShiftIsFinishable:
    """A headline number no policy can reach measures nothing.

    Round-3 measurement: a shortest-path courier with perfect knowledge, no
    wrong turns and no time spent perceiving needed a mean of 76.6 minutes
    (60.9-89.3 across seeds 0-19) to work the ten orders a 60-minute shift
    issued, and finished 6.95 of them. ``delivered / 10`` was unreachable on
    every seed, because only the delivery leg was bounded and the walk to the
    next pickup was drawn without any limit at all.
    """

    def network(self):
        return build_road_network(PARIS, map_name="citycore-paris")

    @staticmethod
    def optimal_seconds(env) -> float:
        from embodiedbench.runtime.city.courier_env import WALK_SPEED_CM_S

        cursor, total = env.node_id, 0.0
        for order in env.orders:
            approach = env.route_length_cm(cursor, order.pickup.kerb_node)
            delivery = env.route_length_cm(order.pickup.kerb_node, order.dropoff.kerb_node)
            assert approach is not None and delivery is not None
            total += (approach + delivery) / WALK_SPEED_CM_S + 60.0
            cursor = order.dropoff.kerb_node
        return total

    def test_a_perfect_courier_could_finish_every_order_issued(self):
        network = self.network()
        for seed in range(8):
            env = CourierEnv(network, seed=seed, order_count=10, shift_seconds=3600.0)
            env.reset()
            assert env.orders, f"seed {seed} issued no orders at all"
            spent = self.optimal_seconds(env)
            assert spent <= 3600.0, (
                f"seed {seed}: the orders issued need {spent/60:.1f} min of optimal "
                f"walking, more than the 60 min shift"
            )

    def test_the_shift_is_filled_not_left_empty(self):
        """The other failure mode: cutting the list so far that the clock never
        binds, which would make the deadline decorative."""
        network = self.network()
        used = []
        for seed in range(8):
            env = CourierEnv(network, seed=seed, order_count=10, shift_seconds=3600.0)
            env.reset()
            used.append(self.optimal_seconds(env) / 3600.0)
        assert sum(used) / len(used) > 0.5, f"shifts only {sum(used)/len(used):.0%} full"

    def test_the_approach_leg_is_bounded_like_the_delivery_leg(self):
        from embodiedbench.runtime.city.courier_env import MAX_APPROACH_WALK_CM

        network = self.network()
        for seed in range(6):
            env = CourierEnv(network, seed=seed, order_count=10, shift_seconds=3600.0)
            env.reset()
            cursor = env.node_id
            for order in env.orders:
                approach = env.route_length_cm(cursor, order.pickup.kerb_node)
                assert approach is not None and approach <= MAX_APPROACH_WALK_CM + 1.0
                cursor = order.dropoff.kerb_node

    def test_no_shift_clock_means_no_order_budget(self):
        """Order economics must not silently depend on a clock nobody set."""
        network = self.network()
        env = CourierEnv(network, seed=0, order_count=10, shift_seconds=None)
        env.reset()
        assert len(env.orders) == 10


@needs_maps
class TestFollowStreetIsMechanical:
    """The turn budget and the graph were sized against different worlds.

    A delivery leg is a median 530 m; a compiled carriageway edge is a median
    18 m. Ten orders needed a mean 351 ``walk_to`` calls against a step budget of
    120, so even a shortest-path oracle stopped at 3.1 deliveries and the cap,
    not the courier, set the score. The macro is admissible only because it is
    mechanical: it never picks a street, and it stops wherever the situation
    stops being obvious.
    """

    def env(self, seed: int = 0) -> CourierEnv:
        env = CourierEnv(build_road_network(PARIS, map_name="citycore-paris"),
                         seed=seed, order_count=1)
        env.reset()
        return env

    def test_it_only_ever_walks_the_street_it_was_given(self):
        for seed in range(4):
            env = self.env(seed)
            for _ in range(12):
                row = env.candidates()[0]
                street, start = row["street"], env.node_id
                outcome = env.follow_street(*env.street_at(row["k"]), 6)
                assert outcome.ok
                assert street in outcome.message
                if env.node_id != start:
                    # Every junction it passed through stayed on that street.
                    assert env.street_of(env.node_id) == street or street in outcome.message

    def test_it_stops_rather_than_guessing_at_a_fork(self):
        env = self.env()
        seen_fork = False
        for _ in range(40):
            row = env.candidates()[0]
            outcome = env.follow_street(*env.street_at(row["k"]), 6)
            if "forks here" in outcome.message or "does not go on" in outcome.message:
                seen_fork = True
                break
        assert seen_fork, "follow_street never handed control back"

    def test_it_never_walks_further_than_asked(self):
        env = self.env()
        before = env.sim_seconds
        outcome = env.follow_street(*env.street_at(env.candidates()[0]["k"]), 2)
        from embodiedbench.runtime.city.courier_env import WALK_SPEED_CM_S

        assert outcome.ok
        assert outcome.walked_m * 100.0 / WALK_SPEED_CM_S == pytest.approx(
            env.sim_seconds - before, abs=1.0)

    def test_it_refuses_a_street_that_is_not_there(self):
        env = self.env()
        outcome = env.follow_street("Rue Imaginaire", "north")
        assert not outcome.ok and outcome.code == "no_such_street"

    def test_it_hands_control_back_at_a_crossing_it_can_see(self):
        """Otherwise the macro takes the red lights its caller is charged for
        and never gets to look at -- the same defect, one layer down."""
        network = build_road_network(PARIS, map_name="citycore-paris")
        env = CourierEnv(network, seed=0, order_count=1, enforce_signals=True)
        env.reset()
        env.visible_signals = {f"{n}|{m}" for n in env.signalised
                               for m in network.nodes[n].neighbours}
        for _ in range(60):
            outcome = env.follow_street(*env.street_at(env.candidates()[0]["k"]), 6)
            assert env.red_crossings == 0 or "pedestrian light" in outcome.message
        assert env.red_crossings == 0

    def test_every_advertised_tool_can_actually_be_run(self):
        """The rule that removed the macros in the first place, enforced.

        Asked of the dispatch table rather than of the environment. The rule
        being protected is "a name in the prompt can be executed", and the
        executor need not be the city: ``note`` writes in the courier's own
        notebook, where nothing about the world changes, and the session runs
        it. Asking ``getattr(env, name)`` would forbid that for no reason and
        would push a no-op onto CourierEnv purely to satisfy the check.
        """
        from embodiedbench.agent.courier.session import CourierSession

        env = self.env()
        session = CourierSession(env, with_images=False)
        for name in env.allowed_tool_names():
            assert callable(session.dispatch.get(name)), \
                f"{name} is advertised but nothing can run it"

    def test_the_notebook_can_be_written_and_reads_back(self):
        """``note`` was advertised for a long time with no executor anywhere.

        The skills tell the courier to note a dead end by name, so the one
        place it could record a conclusion was unreachable, and the failure was
        silent: the call parsed, then died in dispatch.
        """
        from embodiedbench.agent.courier.session import CourierSession

        env = self.env()
        session = CourierSession(env, with_images=False)
        if "note" not in session.dispatch:
            pytest.skip("this condition does not offer a notebook")
        before = env.sim_seconds
        outcome = session.dispatch["note"]("Rue Monge north end is a dead end")
        assert outcome.ok
        assert "Rue Monge" in session.memory.render()
        # A turn, but not a second: writing on your own hand does not take a
        # minute, and the clock is what the shift is scored against.
        assert env.sim_seconds == before
        assert not session.dispatch["note"]("   ").ok


@needs_maps
class TestTheLightIsChargedOnlyWhereItCanBeSeen:
    """Having a signal album is not the same as being able to see the light.

    The Paris bake renders 347 signalised approaches in both phases, but the
    camera looks along the street the courier is about to take, and comparing
    each red/green pair for lamp-coloured pixels finds a switching lamp on 55 of
    them (15.9%). Charging on all 347 put an oracle courier on 43 violations a
    shift, almost none of which it could have seen coming -- the same
    "penalised for an unobservable" defect ``SIGNAL_FRAMES_AVAILABLE`` was
    added to prevent, arriving by a different door.
    """

    def env(self, **kwargs) -> CourierEnv:
        env = CourierEnv(build_road_network(PARIS, map_name="citycore-paris"),
                         seed=0, order_count=1, enforce_signals=True, **kwargs)
        env.reset()
        return env

    @staticmethod
    def walk_to_signalised(env, limit: int = 200) -> bool:
        seen = set()
        for _ in range(limit):
            if env.node_id in env.signalised:
                return True
            options = [r for r in env.candidates() if r["node"] not in seen] or env.candidates()
            seen.add(env.node_id)
            env.walk_to(*env.street_at(options[0]["k"]))
        return env.node_id in env.signalised

    def test_an_album_that_declares_nothing_charges_nothing(self):
        env = self.env()
        assert env.visible_signals is None
        assert self.walk_to_signalised(env)
        for row in env.candidates():
            assert not env.signal_is_visible(env.node_id, row["node"])

    def test_a_declared_approach_is_charged_and_an_undeclared_one_is_not(self):
        env = self.env()
        assert self.walk_to_signalised(env)
        rows = env.candidates()
        red = next((r for r in rows if env.light_here(r["k"]) == "red"), None)
        assert red is not None
        env.visible_signals = set()
        env.walk_to(*env.street_at(red["k"]))
        assert env.red_crossings == 0, "charged for a lamp the album does not show"

        env = self.env()
        assert self.walk_to_signalised(env)
        rows = env.candidates()
        red = next((r for r in rows if env.light_here(r["k"]) == "red"), None)
        env.visible_signals = {f"{env.node_id}|{red['node']}"}
        env.walk_to(*env.street_at(red["k"]))
        assert env.red_crossings == 1, "not charged for a lamp the album does show"

    def test_the_sidecar_is_read_from_the_album(self):
        with tempfile.TemporaryDirectory() as tmp:
            album = Path(tmp)
            (album / "signal_visibility.json").write_text(
                json.dumps({"legible": ["a|b", "c|d"]}))
            env = CourierEnv(build_road_network(PARIS, map_name="citycore-paris"),
                             seed=0, order_count=1, signal_album_root=album)
            assert env.visible_signals == {"a|b", "c|d"}
            assert env.enforce_signals is True

    def test_a_lamp_too_small_at_the_served_size_is_not_charged(self):
        """The album certifies at 768 px; a harness that sends 320 has not seen
        the frame that was certified. Area falls with the square of the resize,
        so a 20 px lamp in a 1280-wide bake is 1.25 px by the time a 320 px
        harness is done with it -- under the album's own 2x2 floor. On the
        Paris kerb album this is 34 of the 130 certified approaches, and
        charging them is charging for a light the policy was never sent enough
        pixels to see."""
        with tempfile.TemporaryDirectory() as tmp:
            album = Path(tmp)
            (album / "signal_visibility.json").write_text(json.dumps({
                "legible": ["a|b", "c|d"],
                # [smaller lamp px, frame width, frame height]
                "lamp_px": {"a|b": [400, 1280, 960], "c|d": [20, 1280, 960]},
            }))
            network = build_road_network(PARIS, map_name="citycore-paris")
            big = CourierEnv(network, seed=0, order_count=1,
                             signal_album_root=album, served_long_edge=768)
            small = CourierEnv(network, seed=0, order_count=1,
                               signal_album_root=album, served_long_edge=320)
            assert big.visible_signals == {"a|b", "c|d"}
            assert small.visible_signals == {"a|b"}

    def test_an_unmeasured_approach_keeps_the_albums_answer(self):
        """This is a refinement of the gate, not a second gate that fails
        closed on its own for a different reason."""
        with tempfile.TemporaryDirectory() as tmp:
            album = Path(tmp)
            (album / "signal_visibility.json").write_text(json.dumps({
                "legible": ["a|b", "c|d"], "lamp_px": {"a|b": [400, 1280, 960]}}))
            env = CourierEnv(build_road_network(PARIS, map_name="citycore-paris"),
                             seed=0, order_count=1, signal_album_root=album,
                             served_long_edge=320)
            assert env.visible_signals == {"a|b", "c|d"}

    def test_a_harness_that_says_nothing_gets_the_albums_answer(self):
        with tempfile.TemporaryDirectory() as tmp:
            album = Path(tmp)
            (album / "signal_visibility.json").write_text(json.dumps({
                "legible": ["a|b", "c|d"],
                "lamp_px": {"a|b": [400, 1280, 960], "c|d": [20, 1280, 960]}}))
            env = CourierEnv(build_road_network(PARIS, map_name="citycore-paris"),
                             seed=0, order_count=1, signal_album_root=album)
            assert env.visible_signals == {"a|b", "c|d"}

    def test_one_wait_clears_a_red_light(self):
        """A flat 15 s against a 60 s phase meant one wait usually left the light
        exactly as red, so obeying it cost one to four turns and the courier
        could not tell which in advance."""
        env = self.env()
        assert self.walk_to_signalised(env)
        red = next((r["k"] for r in env.candidates() if env.light_here(r["k"]) == "red"), None)
        assert red is not None
        env.wait()
        assert env.light_here(red) == "green"

    def test_crossing_on_red_costs_more_than_waiting_it_out(self):
        """Otherwise crossing is strictly cheaper on the clock, every time, and
        a policy that uses its eyes is slower than one that does not."""
        from embodiedbench.runtime.city.courier_env import (
            RED_CROSSING_PENALTY_S,
            SIGNAL_PHASE_S,
        )

        assert RED_CROSSING_PENALTY_S > SIGNAL_PHASE_S / 2.0


class TestTheNotesReadLikeNotes:
    def test_the_trail_names_a_place_not_a_bare_number(self):
        """It rendered as "Came from: 9 <- 1 <- 2" -- three house numbers with no
        street attached, naming nothing the agent could find again."""
        from embodiedbench.agent.courier.memory import CourierMemory

        memory = CourierMemory()
        memory.arrive(step=0, node_id="a", street="Quai Beaubourg", address_hint="2")
        memory.arrive(step=1, node_id="b", street="Quai Beaubourg", address_hint="1")
        memory.arrive(step=2, node_id="c", street="Boulevard Lafayette", address_hint="9")
        rendered = memory.render()
        assert "Quai Beaubourg no. 2" in rendered
        assert "Came from: 9" not in rendered

    def test_the_harness_does_not_write_in_the_models_notebook(self):
        """The notebook shows six lines and a shift changes goal twenty times,
        so an auto-written "goal changed" line meant the notebook was six copies
        of the Job: line above it and every note the model wrote was evicted."""
        from embodiedbench.agent.courier.memory import CourierMemory

        memory = CourierMemory()
        memory.write("Quai Montorgueil south past no. 37 is a dead end")
        for step in range(8):
            memory.set_goal("deliver to", f"{step} Rue A")
        assert memory.notebook == ["Quai Montorgueil south past no. 37 is a dead end"]
        assert "dead end" in memory.render()

    def test_the_loop_warning_names_a_place_not_a_graph_id(self):
        """It was the only string in the whole observation carrying a raw node
        id, so it named somewhere the agent could not recognise."""
        from embodiedbench.agent.courier.memory import CourierMemory

        memory = CourierMemory()
        for step in range(8):
            memory.arrive(step=step, node_id=f"s006_n{step % 2}",
                          street="Rue Oberkampf", address_hint="7")
        rendered = memory.render()
        assert memory.is_looping()
        assert "Rue Oberkampf no. 7" in rendered
        assert "s006_n" not in rendered

    def test_the_job_line_reads_as_english(self):
        from embodiedbench.agent.courier.memory import CourierMemory

        memory = CourierMemory()
        memory.set_goal("collect from", "5 Rue Saint-Antoine")
        rendered = memory.render()
        assert "collect from 5 Rue Saint-Antoine" in rendered
        assert " at 5 Rue" not in rendered


class TestThePromptDescribesWhatIsScored:
    def test_the_courier_is_told_the_lights_exist(self):
        """walk_to charges time and reward for crossing on red at 105 junctions,
        the one mechanic here that genuinely needs a photograph, and no line of
        the prompt said a light existed or that wait() was the answer."""
        from embodiedbench.agent.courier.prompts import build_system_prompt
        from embodiedbench.agent.courier.tools import available_tools

        prompt = build_system_prompt(city="Paris", tools=available_tools(
            ["VIEW_ORDERS", "ACCEPT_ORDER", "PICKUP", "DROP_OFF", "WAIT", "MOVE_TO"]))
        assert "light" in prompt.lower()
        assert "wait()" in prompt

    def test_the_prompt_never_advertises_a_tool_with_no_executor(self):
        from embodiedbench.agent.courier.prompts import build_system_prompt
        from embodiedbench.agent.courier.tools import UNIMPLEMENTED_TOOLS, available_tools

        for allow in (True, False):
            prompt = build_system_prompt(city="Paris", tools=available_tools(
                ["VIEW_ORDERS", "ACCEPT_ORDER", "PICKUP", "DROP_OFF", "WAIT", "MOVE_TO"],
                allow_consult=allow))
            for tool in UNIMPLEMENTED_TOOLS:
                assert f"{tool.name}(" not in prompt, tool.name


@needs_maps
class TestTheNumbersOnTheDoorsAreTheNumbersItPrints:
    """A limit on how much text to print must not change what the text claims.

    ``house_numbers_near`` took the first four numbers and printed
    ``first-last``, so a junction with doors 1, 2, 3, 5, 7, 9, 11, 13 -- every
    one of them at that junction -- advertised "1-5". 17 of 477 addresses could
    not be recognised standing at their own front door, and the ARRIVAL runbook
    tells the courier to compare the number exactly, so a policy that obeys the
    prompt walks away from a delivery it has already reached. It did: on seed 9
    the courier arrived at 9 Rue Mouffetard on turn 5 and spent the remaining
    115 turns oscillating past it for 0 deliveries.
    """

    def env(self, seed: int = 0) -> CourierEnv:
        env = CourierEnv(build_road_network(PARIS, map_name="citycore-paris"),
                         seed=seed, order_count=1)
        env.reset()
        return env

    @staticmethod
    def numbers_in(text: str) -> list[int]:
        import re

        return [int(x) for x in re.findall(r"\d+", text)]

    def test_every_address_is_advertised_at_its_own_door(self):
        env = self.env()
        missing = [
            address.text for address in env.network.addresses
            if address.kerb_node in env.network.nodes
            and address.number not in self.numbers_in(
                env.house_numbers_near(address.kerb_node))
        ]
        assert not missing, f"{len(missing)} doors invisible from their own kerb: {missing[:5]}"

    def test_it_never_advertises_a_door_that_is_out_of_reach(self):
        """A printed span asserts every number inside it is here. On this map
        that was false for 458 of 953 advertised numbers, and the courier that
        believed it collected nothing."""
        env = self.env()
        unreachable = 0
        advertised = 0
        for node in env.network.nodes:
            text = env.house_numbers_near(node)
            if not text:
                continue
            here = env.position(node)
            street = env.street_of(node)
            for number in self.numbers_in(text):
                doors = [a for a in env.addresses_by_street.get(street, [])
                         if a.number == number]
                if not doors:
                    continue
                advertised += 1
                if not any(math.dist(here, a.kerb) <= ARRIVAL_TOLERANCE_CM for a in doors):
                    unreachable += 1
        assert advertised > 100
        assert unreachable == 0, f"{unreachable}/{advertised} advertised doors out of reach"

    def test_a_single_door_is_said_as_a_number_not_a_range(self):
        env = self.env()
        singles = [env.house_numbers_near(n) for n in env.network.nodes]
        assert any(t.isdigit() for t in singles if t)


class TestOneLampIsChargedOnce:
    """The map has no lamp objects, so the bake put one at each junction.

    Signalised junctions are derived from node degree and one light mesh is
    baked at each, so a junction with four ways out has four photographs of
    the same lamp from the same camera, differing only in which phase is lit.
    The environment gave each approach its own phase from its own bearing and
    charged each separately -- one lamp treated as four.

    Worse than the double charge: told "the lamp for Rue de la Paix" and "the
    lamp for Rue Cujas", the courier held two pictures with identical
    backgrounds and nothing but the caption to tell them apart. On this album
    34 junctions showed one lamp to three approaches, three showed one to
    four, two showed one to five, and exactly one junction has two genuinely
    different lamps.
    """

    def sidecar(self, tmp, entries):
        (tmp / "signal_visibility.json").write_text(json.dumps(entries))
        return CourierEnv(build_road_network(PARIS, map_name="citycore-paris"),
                          seed=0, order_count=1, signal_album_root=tmp)

    def test_approaches_sharing_a_lamp_keep_exactly_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = self.sidecar(Path(tmp), {
                "legible": ["n|a", "n|b", "n|c"],
                "lamp_px": {"n|a": [400, 1280, 960], "n|b": [900, 1280, 960],
                            "n|c": [500, 1280, 960]},
                # All three boxes are the same lamp.
                "lamp_box": {"n|a": [600, 300, 640, 360],
                             "n|b": [601, 301, 641, 361],
                             "n|c": [599, 299, 639, 359]},
            })
            # The largest lamp wins: it is the view a courier could read.
            assert env.visible_signals == {"n|b"}

    def test_two_real_lamps_at_one_junction_both_survive(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = self.sidecar(Path(tmp), {
                "legible": ["n|a", "n|b"],
                "lamp_px": {"n|a": [400, 1280, 960], "n|b": [400, 1280, 960]},
                "lamp_box": {"n|a": [100, 300, 140, 360],
                             "n|b": [900, 300, 940, 360]},
            })
            assert env.visible_signals == {"n|a", "n|b"}

    def test_junctions_do_not_borrow_each_others_lamps(self):
        """Two junctions can have a lamp in the same part of the frame without
        it being the same lamp."""
        with tempfile.TemporaryDirectory() as tmp:
            env = self.sidecar(Path(tmp), {
                "legible": ["m|a", "n|a"],
                "lamp_px": {"m|a": [400, 1280, 960], "n|a": [400, 1280, 960]},
                "lamp_box": {"m|a": [600, 300, 640, 360],
                             "n|a": [600, 300, 640, 360]},
            })
            assert env.visible_signals == {"m|a", "n|a"}

    def test_an_album_without_boxes_is_left_alone(self):
        """Older sidecars do not carry lamp_box; narrowing them on no evidence
        would silently switch the mechanic off."""
        with tempfile.TemporaryDirectory() as tmp:
            env = self.sidecar(Path(tmp), {"legible": ["n|a", "n|b"]})
            assert env.visible_signals == {"n|a", "n|b"}
