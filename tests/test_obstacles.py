"""Obstacles: the mechanic that makes vision necessary for *walking*.

Before them the environment needed eyes at signalised junctions and nowhere
else, so a policy could ignore every photograph and still walk perfectly. The
claims defended here are the ones that make that untrue, and each of them is a
claim someone could break by accident:

1. placement is rule-based, deterministic in ``(map, seed)``, and runs on any
   map -- not a hand-authored list for Paris;
2. a ``road_block`` really stops a walk, a ``slow_pedestrian`` really costs
   time, and neither can strand an address;
3. **nothing in the observation, and no tool, ever says an obstacle is there.**
   This is the load-bearing one: the moment a sentence leaks it, the picture
   stops mattering and the mechanic is decoration;
4. and the charge is gated on the album actually showing it, so the environment
   never bills for information it withheld.
"""

from __future__ import annotations

import json
import math
import tempfile
import os
from pathlib import Path

import pytest

from embodiedbench.compiler.road_network import build_road_network
from embodiedbench.runtime.city.courier_env import (
    Difficulty,
    CourierEnv,
    Stride,
)
from embodiedbench.runtime.city.obstacles import (
    BLOCKED_SECONDS,
    OBSTACLE_TYPES,
    OBSTACLE_VISIBILITY_FILE,
    ROAD_BLOCK,
    SLOW_PEDESTRIAN,
    SLOW_SECONDS,
    ObstacleField,
    approach_key,
    bridges,
    obstacle_sites,
    site_key,
)
from embodiedbench.tasks.courier_oracle import run_reference_courier

MAPS = (Path(__file__).resolve().parents[1] / "vendor" / "vagen" / "vagen"
        / "envs" / "deliverybench" / "maps")
PARIS = MAPS / "citycore-paris"
ALL_MAPS = sorted(p for p in MAPS.iterdir() if p.is_dir()) if MAPS.exists() else []
needs_maps = pytest.mark.skipif(not ALL_MAPS, reason="vendored maps not present")

STREETS = Path(os.environ.get("ALBUMS_DIR", "/data/albums")) / Path("paris_streets_v2/citycore-paris")
OBSTACLES = Path(os.environ.get("ALBUMS_DIR", "/data/albums")) / Path("paris_obstacles/citycore-paris")
needs_album = pytest.mark.skipif(
    not (OBSTACLES / OBSTACLE_VISIBILITY_FILE).exists(),
    reason="obstacle album not baked here",
)


@pytest.fixture(scope="module")
def paris():
    return build_road_network(PARIS, map_name="citycore-paris")


def neighbours_of(network):
    return {n: sorted(node.neighbours) for n, node in network.nodes.items()}


def stub_album(directory: Path, network, *, keys=None) -> Path:
    """An album that declares every site visible, with placeholder frames.

    The mechanic and the gate are separable and the tests want them separately:
    this exercises the mechanic without needing a renderer, and the real album's
    frames are measured by ``obstacle_visibility``.
    """
    sites = obstacle_sites(neighbours_of(network), network.map_name)
    visible = []
    for a, b in sites:
        for src, dst in ((a, b), (b, a)):
            if keys is not None and approach_key(src, dst) not in keys:
                continue
            for kind in OBSTACLE_TYPES:
                path = directory / "images" / src / f"toward_{dst}_{kind}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"png")
            visible.append(approach_key(src, dst))
    (directory / OBSTACLE_VISIBILITY_FILE).write_text(
        json.dumps({"map": network.map_name, "visible": sorted(visible)}))
    return directory


# ─────────────────────────────────────────────────────────────────────────────


@needs_maps
class TestPlacementIsARule:
    def test_the_same_map_and_seed_give_the_same_obstacles(self, paris):
        neighbours = neighbours_of(paris)
        first = ObstacleField.generate(neighbours, "citycore-paris", 7)
        second = ObstacleField.generate(neighbours, "citycore-paris", 7)
        assert first.by_site == second.by_site
        assert first.by_site  # and it placed something

    def test_a_different_seed_gives_a_different_arrangement(self, paris):
        neighbours = neighbours_of(paris)
        arrangements = {
            frozenset(ObstacleField.generate(neighbours, "citycore-paris", s).by_site.items())
            for s in range(8)
        }
        assert len(arrangements) == 8, "seeds are not independent"

    def test_the_sites_do_not_depend_on_the_seed(self, paris):
        """The album is baked once per map, so the *places* must be seed-free.

        If they were not, an episode could place an obstacle at a spot with no
        photograph of it, and the gate would silently switch the mechanic off.
        """
        neighbours = neighbours_of(paris)
        sites = {site_key(a, b) for a, b in obstacle_sites(neighbours, "citycore-paris")}
        for seed in range(6):
            placed = set(ObstacleField.generate(neighbours, "citycore-paris", seed).by_site)
            assert placed <= sites

    def test_it_runs_on_every_vendored_map(self):
        """Rule-based means rule-based: no map is special-cased."""
        for map_dir in ALL_MAPS:
            network = build_road_network(map_dir, map_name=map_dir.name)
            if len(network.nodes) < 20:
                continue
            field = ObstacleField.generate(neighbours_of(network), map_dir.name, 0)
            assert field.by_site, f"{map_dir.name}: no obstacles placed anywhere"
            for kind in field.by_site.values():
                assert kind in OBSTACLE_TYPES

    def test_a_block_never_cuts_the_city_in_two(self, paris):
        """An unreachable address is an unsolvable episode, not a hard one."""
        neighbours = neighbours_of(paris)
        cut = bridges(neighbours)
        for seed in range(10):
            field = ObstacleField.generate(neighbours, "citycore-paris", seed)
            blocked = {k for k, kind in field.by_site.items() if kind == ROAD_BLOCK}
            assert not (blocked & cut)

    def test_every_address_is_still_reachable_from_the_spawn(self, paris):
        """The stronger version of the claim, checked on the real graph."""
        with tempfile.TemporaryDirectory() as tmp:
            album = stub_album(Path(tmp), paris)
            for seed in range(4):
                env = CourierEnv(paris, seed=seed, difficulty=Difficulty.SHIFT,
                                 obstacle_album_root=album)
                env.reset()
                for order in env.orders:
                    for stop in (order.pickup, order.dropoff):
                        assert env.route_cost(env.node_id, stop.kerb_node,
                                              obstacles=True) is not None


@needs_maps
class TestTheMechanicBites:
    def blocked_edge(self, env):
        """Some edge with a live, visible road block, and a node beside it."""
        for key, kind in sorted(env.obstacles.by_site.items()):
            if kind != ROAD_BLOCK:
                continue
            a, b = key.split("|")
            if env.obstacles.in_effect(a, b):
                return a, b
        pytest.skip("no visible road block on this seed")

    def slow_edge(self, env):
        for key, kind in sorted(env.obstacles.by_site.items()):
            if kind != SLOW_PEDESTRIAN:
                continue
            a, b = key.split("|")
            if env.obstacles.in_effect(a, b):
                return a, b
        pytest.skip("no visible congestion on this seed")

    def env(self, paris, tmp, seed=0):
        env = CourierEnv(paris, seed=seed, obstacle_album_root=stub_album(Path(tmp), paris))
        env.reset()
        return env

    def test_a_road_block_refuses_the_walk_and_costs_the_time(self, paris):
        with tempfile.TemporaryDirectory() as tmp:
            env = self.env(paris, tmp)
            a, b = self.blocked_edge(env)
            env.node_id, env.arrived_from = a, None
            k = next(r["k"] for r in env.candidates() if r["node"] == b)
            before = env.sim_seconds
            outcome = env.walk_to(*env.street_at(k))
            assert not outcome.ok and outcome.code == "way_blocked"
            assert env.node_id == a, "the courier moved through a barrier"
            assert env.sim_seconds - before == pytest.approx(BLOCKED_SECONDS)
            assert env.summary()["blocked_attempts"] == 1

    def test_it_blocks_from_both_ends(self, paris):
        """A barrier is an object, not a one-way rule."""
        with tempfile.TemporaryDirectory() as tmp:
            env = self.env(paris, tmp)
            a, b = self.blocked_edge(env)
            assert env.obstacles.blocks(a, b) and env.obstacles.blocks(b, a)

    def test_congestion_costs_time_but_lets_you_through(self, paris):
        with tempfile.TemporaryDirectory() as tmp:
            env = self.env(paris, tmp)
            a, b = self.slow_edge(env)
            env.node_id, env.arrived_from = a, None
            row = next(r for r in env.candidates() if r["node"] == b)
            plain = row["distance_m"] * 100.0 / 140.0
            outcome = env.walk_to(*env.street_at(row["k"]))
            assert outcome.ok and env.node_id == b
            assert outcome.sim_seconds == pytest.approx(plain + SLOW_SECONDS)
            assert env.summary()["slow_passages"] == 1

    def test_the_photograph_changes_and_it_is_the_only_thing_that_does(self, paris):
        """The whole design in one assertion.

        The frame for an obstructed street is a different file; every word of
        text about that street is identical to the clear case.
        """
        with tempfile.TemporaryDirectory() as tmp:
            clean = CourierEnv(paris, seed=0)
            clean.reset()
            env = self.env(paris, tmp)
            a, b = self.blocked_edge(env)
            for world in (clean, env):
                world.node_id, world.arrived_from = a, None
            rows = {r["node"]: r for r in env.candidates()}
            plain_rows = {r["node"]: r for r in clean.candidates()}
            assert rows[b]["image"] != plain_rows[b]["image"]
            assert rows[b]["image"].endswith(f"_{ROAD_BLOCK}.png")
            for field in ("street", "bearing", "distance_m", "heading", "relative", "k"):
                assert rows[b][field] == plain_rows[b][field]

    def test_the_clock_is_priced_on_the_obstructed_city(self, paris):
        """Otherwise obstacles would tighten the deadline instead of adding work.

        The tiers exist to vary the *task*, and the one thing the ladder is not
        allowed to do is get harder because the stopwatch got meaner.
        """
        with tempfile.TemporaryDirectory() as tmp:
            album = stub_album(Path(tmp), paris)
            longer = 0
            for seed in range(6):
                clean = CourierEnv(paris, seed=seed, difficulty=Difficulty.SHIFT)
                clean.reset()
                env = CourierEnv(paris, seed=seed, difficulty=Difficulty.SHIFT,
                                 obstacle_album_root=album)
                env.reset()
                assert env.shift_seconds >= clean.shift_seconds - 1.0
                longer += env.shift_seconds > clean.shift_seconds + 1.0
            assert longer >= 4, "the clock did not follow the obstacles at all"


@needs_maps
class TestNothingSaysItIsThere:
    """The claim the whole mechanic rests on.

    If any sentence the agent can read names the obstacle, a policy that never
    looks at a photograph can route around barriers perfectly, and the images go
    back to being decoration. So this checks the observation and *every* tool
    that returns text, at every junction that has an obstacle on it, over many
    seeds -- not one hand-picked corner.
    """

    LEAK = __import__("re").compile(
        r"\b(block(ed|age)?|barrier|barriers|fence|fenced|obstacle|obstacles|"
        r"obstruct\w*|roadwork\w*|closed|closure|shut|skip|bin|bins|congest\w*|"
        r"crowd\w*|stand|hoarding|scaffold\w*|impassable|cordon\w*)\b",
        __import__("re").IGNORECASE,
    )

    def observations(self, paris, album, seeds=range(6)):
        """Every text an agent could read while standing at an obstructed corner."""
        from embodiedbench.agent.courier.session import CourierSession

        out = []
        for seed in seeds:
            env = CourierEnv(paris, seed=seed, difficulty=Difficulty.TRIPLE,
                             album_root=STREETS, obstacle_album_root=album)
            env.reset()
            corners = [
                key.split("|")[0] for key in sorted(env.obstacles.by_site)
                if env.obstacles.in_effect(*key.split("|"))
            ]
            for node in corners[:6]:
                env.node_id, env.arrived_from = node, None
                session = CourierSession(env, with_images=True)
                out.append(session.observe().text)
                for k in (r["k"] for r in env.candidates()):
                    out.append(env.look(*env.street_at(k)).message)
                out.append(env.check_order().message)
                out.append(env.navigate().message)
                target = env.target_address()
                if target is not None:
                    out.append(env.check_map(target.text).message)
        return out

    def test_no_text_the_agent_can_read_ever_mentions_an_obstacle(self, paris):
        with tempfile.TemporaryDirectory() as tmp:
            album = stub_album(Path(tmp), paris)
            texts = self.observations(paris, album)
            assert len(texts) >= 60, "not enough text to make the claim"
            for text in texts:
                found = self.LEAK.findall(text or "")
                assert not found, f"the environment leaked {found} in: {text!r}"

    def test_the_candidate_rows_carry_no_obstacle_field(self, paris):
        """A field is a leak too: the prompt renderer would print it."""
        with tempfile.TemporaryDirectory() as tmp:
            env = CourierEnv(paris, seed=0, obstacle_album_root=stub_album(Path(tmp), paris))
            env.reset()
            for key in sorted(env.obstacles.by_site):
                a, b = key.split("|")
                if not env.obstacles.in_effect(a, b):
                    continue
                env.node_id, env.arrived_from = a, None
                row = next(r for r in env.candidates() if r["node"] == b)
                for value in row.values():
                    assert ROAD_BLOCK not in str(value).replace(str(row.get("image")), "")
                assert not any(k in row for k in ("obstacle", "blocked", "hazard", "type"))
                break

    def test_the_route_still_goes_straight_through_a_barrier(self, paris):
        """A map app cannot see a skip, and this one must not pretend to.

        If ``navigate()`` routed around obstacles it would be doing the seeing,
        and the photographs would again be optional.
        """
        with tempfile.TemporaryDirectory() as tmp:
            env = CourierEnv(paris, seed=0, obstacle_album_root=stub_album(Path(tmp), paris))
            env.reset()
            routed_through = 0
            for seed in range(8):
                env = CourierEnv(paris, seed=seed, difficulty=Difficulty.SHIFT,
                                 obstacle_album_root=stub_album(Path(tmp), paris))
                env.reset()
                for order in env.orders:
                    path = env.route_nodes(env.node_id, order.dropoff.kerb_node) or []
                    routed_through += sum(
                        1 for a, b in zip(path, path[1:]) if env.obstacles.blocks(a, b)
                    )
            assert routed_through > 0, (
                "the phone avoided every barrier -- it is navigating by sight"
            )


@needs_maps
class TestTheRouteCanBeActedOn:
    """The first instruction has to describe the street it names.

    Not an obstacle test, but found while playing one: on a curving street the
    route announced a leg by its *overall* direction and put a left/right beside
    it computed from its *first* hop, so "Take Rue Saint-Antoine — north-west, on
    your left" named a street the corner lists as south-east. The two disagree by
    more than 90 degrees on 10.1% of legs, and an agent matching the compass
    against the street list rejects the street it was told to take.
    """

    def test_the_route_never_names_a_direction_to_walk(self, paris):
        """It used to name one per leg, which is the navigation given away.

        Walked across eight shifts and asked for a route from wherever the
        courier happens to be, because the sentence that leaks is the one
        produced in some state nobody thought to check.
        """
        for seed in range(8):
            env = CourierEnv(paris, seed=seed, difficulty=Difficulty.SHIFT)
            env.reset()
            for _ in range(10):
                rows = env.candidates()
                if env.target_address() is None or not rows:
                    break
                message = env.navigate().message
                for word in ("north", "south", "east", "west"):
                    assert word not in message.lower(), (seed, message)
                row = rows[0]
                env.walk_to(row["street"], row["heading"])

    def test_no_album_means_no_obstacle_bites(self, paris):
        """The default. An unphotographed obstacle is not an obstacle."""
        env = CourierEnv(paris, seed=0)
        env.reset()
        assert env.enforce_obstacles is False
        assert env.obstacles.by_site, "placement should still be deterministic"
        assert all(env.obstacles.in_effect(*k.split("|")) is None
                   for k in env.obstacles.by_site)

    def test_an_album_that_declares_nothing_charges_nothing(self, paris):
        """Silence is not consent: an unmeasured album is an unchecked one."""
        with tempfile.TemporaryDirectory() as tmp:
            env = CourierEnv(paris, seed=0, obstacle_album_root=Path(tmp))
            env.reset()
            assert all(env.obstacles.in_effect(*k.split("|")) is None
                       for k in env.obstacles.by_site)

    def test_an_obstacle_seen_from_only_one_side_is_inert(self, paris):
        """Half a photograph would make an edge shut one way and open the other."""
        with tempfile.TemporaryDirectory() as tmp:
            network = paris
            sites = obstacle_sites(neighbours_of(network), "citycore-paris")
            one_way = {approach_key(a, b) for a, b in sites}
            album = stub_album(Path(tmp), network, keys=one_way)
            env = CourierEnv(network, seed=0, obstacle_album_root=album)
            env.reset()
            assert all(env.obstacles.in_effect(*k.split("|")) is None
                       for k in env.obstacles.by_site)


@needs_maps
class TestVisionIsNecessaryForWalking:
    def test_looking_at_the_street_is_worth_a_measurable_amount(self, paris):
        """The proof, and the number that makes the mechanic real.

        Both couriers are the same policy running the same seeds; the only
        difference is that one is allowed to notice what the photograph of a
        street shows before it takes it. The blind one discovers barriers by
        walking into them, at a turn and 45 s each.
        """
        with tempfile.TemporaryDirectory() as tmp:
            album = stub_album(Path(tmp), paris)
            blind = sighted = 0
            for seed in range(8):
                for sighted_run in (False, True):
                    env = CourierEnv(paris, seed=seed, difficulty=Difficulty.TRIPLE,
                                     album_root=STREETS, obstacle_album_root=album)
                    env.reset()
                    run_reference_courier(env, seed, max_steps=400, sighted=sighted_run)
                    if sighted_run:
                        sighted += env.summary()["blocked_attempts"]
                    else:
                        blind += env.summary()["blocked_attempts"]
            assert sighted == 0, f"the sighted courier still walked into {sighted} barriers"
            assert blind >= 8, (
                f"only {blind} collisions over 8 shifts -- the obstacles are too "
                "rare to make looking worth anything"
            )


@needs_maps
class TestABlockIsSeenWholeOrNotAtAll:
    """What the block stride nearly cost, kept as a standing check.

    A block is walked in one call, so the photograph at its corner has to answer
    about the block. Its first working version answered about the first
    eighteen metres, and a courier that read every frame it was given collided
    almost as often as one that never looked -- 59 barriers to 49. A benchmark
    where reading the picture does not pay is the one thing this must not be.
    """

    def blocked_block(self, network, album):
        """A corner, a street off it, and a barrier somewhere along that block."""
        env = CourierEnv(network, seed=0, difficulty=Difficulty.TRIPLE,
                         stride=Stride.BLOCK, album_root=STREETS,
                         obstacle_album_root=album)
        env.reset()
        for node in sorted(network.nodes):
            for first in sorted(network.nodes[node].neighbours):
                chain = env.block_chain(node, first)
                if len(chain) < 2:
                    continue
                deep = [hop for hop in chain[1:] if env.obstacles.blocks(*hop)]
                if deep:
                    return env, node, first, deep[0]
        return None

    def test_a_barrier_deep_in_a_block_is_in_the_corners_photograph(self, paris):
        with tempfile.TemporaryDirectory() as tmp:
            found = self.blocked_block(paris, stub_album(Path(tmp), paris))
            if found is None:
                pytest.skip("no seed placed a barrier past the first hop of a block")
            env, node, first, deep = found
            frame = env.obstacle_frame_for(node, first)
            assert frame is not None, "the block's photograph showed a clear road"
            assert frame.endswith("_road_block.png")
            # The same question at waypoint stride is about the first hop alone,
            # and there the answer is correctly nothing.
            env.stride = Stride.WAYPOINT
            assert env.obstacle_frame_for(node, first) is None

    def test_a_barrier_outranks_the_furniture_in_front_of_it(self, paris):
        """The 11 collisions that survived the first fix, all on one street.

        A block can hold a stall on its first hop and a barrier four hops later.
        Serving the nearest served the stall, so a courier that read the frame
        walked past it into the barrier. A barrier is what changes the decision.
        """
        with tempfile.TemporaryDirectory() as tmp:
            album = stub_album(Path(tmp), paris)
            env = CourierEnv(paris, seed=0, difficulty=Difficulty.TRIPLE,
                             stride=Stride.BLOCK, album_root=STREETS,
                             obstacle_album_root=album)
            env.reset()
            for node in sorted(paris.nodes):
                for first in sorted(paris.nodes[node].neighbours):
                    chain = env.block_chain(node, first)
                    kinds = [env.obstacles.in_effect(*hop) for hop in chain]
                    present = [k for k in kinds if k]
                    if ROAD_BLOCK in present and len(set(present)) > 1:
                        frame = env.obstacle_frame_for(node, first)
                        assert frame.endswith("_road_block.png"), (
                            f"{node}->{first} showed {Path(frame).name} with a "
                            "barrier further down the same block")
                        return
            pytest.skip("no block held both kinds at this seed")

    def test_there_is_no_way_to_tell_the_phone_anything(self, paris):
        """The channel is gone, and its absence is what the block stride tests.

        A barrier deep in a block has to be seen from the corner, because seeing
        it is now the only thing that can help -- there is no report to file and
        the route will name the street again next time it is asked.
        """
        with tempfile.TemporaryDirectory() as tmp:
            found = self.blocked_block(paris, stub_album(Path(tmp), paris))
            if found is None:
                pytest.skip("no seed placed a barrier past the first hop of a block")
            env, node, first, deep = found
            assert not hasattr(env, "report_blocked")
            assert env.obstacle_frame_for(node, first).endswith("_road_block.png")
