"""Is the difficulty ladder real, and is the hardest rung a real objective?

The ladder used to be a stopwatch in costume. Each tier carried its own clock
multiple -- 6.0, 4.5, 2.6, 1.9 -- and the published 100 / 100 / 79 / 72 came
entirely from those four numbers. Holding the multiple fixed and varying only
the order count, the reference courier scored 52 / 58 / 63 / 71 at 1.9 and
100 / 98 / 99 / 100 at 4.5: order count did not make the task harder, it made
it *easier*, because a per-order clock averages over more orders and one bad leg
stops deciding the episode.

So the tests here defend two things that a comment cannot:

  the clock is not the difficulty    raise it and the ladder does not move
  the queue is                       depth 1 -> 2 -> 3 is where the score falls

and, for the autonomous rung, that profit is an objective rather than a label:
it has no ceiling, a late delivery is worth less than a punctual one, an
abandoned order is worth nothing, and no order length is a better deal per
second than any other -- so there is nothing to cherry-pick.

Runs use the reference courier because it is the declared solvability floor. A
tier it cannot clear is a tier nobody should be scored on.
"""

from __future__ import annotations

import math
import statistics
import os
from pathlib import Path

import pytest

from embodiedbench.compiler.road_network import build_road_network
from embodiedbench.runtime.city import courier_env as courier_module
from embodiedbench.runtime.city.courier_env import (
    ARRIVAL_TOLERANCE_CM,
    HANDLING_SECONDS,
    LATE_FEE_FRACTION,
    MAX_APPROACH_WALK_CM,
    MAX_ORDER_WALK_CM,
    MIN_APPROACH_WALK_CM,
    ORDER_EXPIRY_MULTIPLE,
    REJECTED_ACTION_SECONDS,
    TIME_BUDGET_MULTIPLE,
    WALK_SPEED_CM_S,
    CourierEnv,
    Difficulty,
    Stride,
)
from embodiedbench.tasks import courier_router as courier_router_module
from embodiedbench.tasks.courier_oracle import (
    ObservationOnlyCourier,
    run_reference_courier,
)
from embodiedbench.tasks.courier_router import (
    ShortestPathCourier,
    run_shortest_path_courier,
)

MAPS = Path(__file__).resolve().parents[1] / "vendor" / "vagen" / "vagen" / "envs" / "deliverybench" / "maps"
PARIS = MAPS / "citycore-paris"
needs_maps = pytest.mark.skipif(not PARIS.exists(), reason="vendored maps not present")

BOUNDED = (Difficulty.SOLO, Difficulty.PAIR, Difficulty.TRIPLE, Difficulty.SHIFT)


@pytest.fixture(scope="module")
def paris():
    return build_road_network(PARIS, map_name="citycore-paris")


def episode(network, tier: str, seed: int, **kwargs) -> CourierEnv:
    env = CourierEnv(network, seed=seed, difficulty=tier, **kwargs)
    env.reset()
    run_reference_courier(env, seed, max_steps=20000)
    return env


def random_walk(env: CourierEnv, seed: int, max_steps: int = 20000) -> None:
    """A courier with no sense of direction, and nothing else missing.

    It is handed the same arrival cue the reference courier gets and the same
    tools; only the choice of street is random. That isolates navigation, which
    is what the ladder claims to measure -- if this scores like the reference
    courier, the tier is measuring something else.
    """
    import random as _random

    rng = _random.Random(seed * 7919 + 13)
    previous = None
    for _ in range(max_steps):
        if env.shift_over:
            return
        order = env.active_order()
        if order is None:
            return
        if math.dist(env.position(), order.target.kerb) <= ARRIVAL_TOLERANCE_CM:
            if (env.hand_over() if order.picked_up else env.collect()).ok:
                previous = None
                continue
        rows = env.candidates()
        if not rows:
            return
        row = rng.choice([r for r in rows if r["node"] != previous] or rows)
        previous = env.node_id
        env.walk_to(*env.street_at(row["k"]))


@needs_maps
class TestTheClockIsPrincipled:
    """One rule, stated in the units it is stated in, on every tier."""

    def test_the_clock_is_the_stated_multiple_of_a_perfect_run(self, paris):
        """It used to be a multiple of ``sum(deadline)/slack``, which is not the
        optimum: it counts 24 s of handling per order where a courier spends 60.
        So SPEC said 6.0 and the courier got 5.3, said 1.9 and got 1.6, and the
        one number a reader could check was the one number that was wrong.
        """
        for tier in BOUNDED:
            for seed in range(6):
                env = CourierEnv(paris, seed=seed, difficulty=tier)
                env.reset()
                cursor, optimal = env.node_id, 0.0
                for order in env.orders:
                    approach = env.route_length_cm(cursor, order.pickup.kerb_node)
                    delivery = env.route_length_cm(order.pickup.kerb_node,
                                                   order.dropoff.kerb_node)
                    optimal += ((approach + delivery) / WALK_SPEED_CM_S
                                + 2.0 * HANDLING_SECONDS)
                    cursor = order.dropoff.kerb_node
                assert env.shift_seconds == pytest.approx(
                    optimal * TIME_BUDGET_MULTIPLE, rel=1e-6), (
                    f"{tier} seed {seed}: clock {env.shift_seconds:.0f}s is not "
                    f"{TIME_BUDGET_MULTIPLE}x the {optimal:.0f}s a perfect run costs"
                )

    def test_every_tier_gives_the_same_proportional_room(self, paris):
        """Per-tier multiples made the tiers incomparable by construction: a
        score on solo and a score on shift were measured against clocks 3.2x
        apart in generosity, so the difference between them was mostly the
        difference between the two stopwatches."""
        for tier in BOUNDED:
            for seed in range(6):
                env = CourierEnv(paris, seed=seed, difficulty=tier)
                env.reset()
                assert env.shift_seconds / env.optimal_seconds == pytest.approx(
                    TIME_BUDGET_MULTIPLE, rel=1e-6)

    def test_the_autonomous_tier_gets_a_fixed_hour(self, paris):
        """ENDLESS is scored on money against a wall clock. Deriving its clock
        from its own order list would make a policy that draws more work also
        get more time to do it in."""
        for seed in range(4):
            env = CourierEnv(paris, seed=seed, difficulty=Difficulty.ENDLESS)
            env.reset()
            assert env.shift_seconds == Difficulty.ENDLESS_SECONDS

    def test_no_seed_gets_an_absurd_clock(self, paris):
        """Sized from the seed's own work, so the spread across seeds is the
        spread of the work and nothing else."""
        for tier in BOUNDED:
            minutes = []
            for seed in range(12):
                env = CourierEnv(paris, seed=seed, difficulty=tier)
                env.reset()
                minutes.append(env.shift_seconds / 60.0)
            per_order = [m / Difficulty.order_count(tier) for m in minutes]
            assert min(per_order) > 5.0, f"{tier}: {min(per_order):.1f} min an order"
            assert max(per_order) / min(per_order) < 2.0, (
                f"{tier}: clock varies {max(per_order)/min(per_order):.1f}x across seeds"
            )


@needs_maps
class TestDifficultyIsNotAStopwatch:
    """The claim the old ladder could not survive."""

    def test_one_clock_governs_every_rung(self, paris):
        """The bug this class exists for: a *per-tier* stopwatch.

        The old ladder gave each rung its own multiple -- 6.0, 4.5, 2.6, 1.9 --
        so what looked like "more orders is harder" was "less time is harder"
        wearing its costume. One multiple for every tier is what makes a rung a
        statement about queue depth, and it is the property to hold on to.
        """
        for tier in BOUNDED:
            for seed in range(4):
                env = CourierEnv(paris, seed=seed, difficulty=tier)
                env.reset()
                assert env.shift_seconds == pytest.approx(
                    env.optimal_seconds * TIME_BUDGET_MULTIPLE
                ), f"{tier} does not use the shared multiple"

    def test_the_deep_rungs_are_limited_by_the_queue_not_the_clock(
        self, paris, monkeypatch
    ):
        """Doubling the clock must not rescue the tiers that test sequencing.

        This used to assert that doubling changed *nothing anywhere*, which was
        true only because the clock sat far above where it binds -- and that was
        exactly why perfect sight bought nothing: a courier could walk into every
        closure on the map and still finish. The multiple is now 3.5, chosen so
        the hazards cost something, so the shallow rungs do move with the clock.
        That is deliberate and it is where vision is measured.

        What must not move is the deep end. ``triple`` and ``shift`` are there to
        test holding several jobs at once, and if doubling their clock lifted
        their scores they would be measuring the stopwatch again. Measured over
        10 seeds: triple 11/30 and shift 19/100 at both 3.5 and 7.0, unchanged.
        """
        seeds = range(10)
        deep = [t for t in BOUNDED if Difficulty.queue_depth(t) > 1]
        assert deep, "no deep rung to check"
        baseline = {
            tier: sum(episode(paris, tier, s).delivered_count for s in seeds)
            for tier in deep
        }
        monkeypatch.setattr(courier_module, "TIME_BUDGET_MULTIPLE",
                            TIME_BUDGET_MULTIPLE * 2.0)
        for tier in deep:
            doubled = sum(episode(paris, tier, s).delivered_count for s in seeds)
            issued = Difficulty.order_count(tier) * len(seeds)
            # Not exact equality. One delivery in thirty moves on ``triple``
            # because a doubled clock occasionally lets one already-collected
            # parcel land before its window shuts, and that is seed noise rather
            # than the clock setting the difficulty. The failure this guards
            # against is the old ladder's, where solo went 24% -> 100% on the
            # same sweep; a tenth of the orders is far below that and far above
            # the noise.
            drift = abs(doubled - baseline[tier]) / issued
            assert drift <= 0.10, (
                f"{tier}: doubling the shift clock moved the score "
                f"{baseline[tier]} -> {doubled} of {issued} ({drift:.0%}); "
                "the clock is setting the difficulty"
            )

    def test_the_queue_is_what_deepens(self, paris):
        """The axis, stated as data so it cannot drift from the docstring."""
        depths = [Difficulty.queue_depth(t) for t in Difficulty.ALL]
        assert depths == sorted(depths), depths
        assert depths[0] == 1 and depths[-1] > 1

    def test_a_deeper_queue_costs_a_sequential_courier(self, paris):
        """Ten orders one at a time against ten orders three at a time. Same
        list, same clock, same policy: the only difference is that the windows
        overlap, and that is the whole of the difficulty."""
        seeds = range(10)
        shallow = sum(episode(paris, Difficulty.SHIFT, s, queue_depth=1).delivered_count
                      for s in seeds)
        deep = sum(episode(paris, Difficulty.SHIFT, s).delivered_count for s in seeds)
        assert deep < shallow * 0.85, (
            f"queue depth {Difficulty.queue_depth(Difficulty.SHIFT)} scored {deep} "
            f"against {shallow} at depth 1 -- the depth is not doing any work"
        )

    def test_the_windows_of_a_deep_queue_really_do_overlap(self, paris):
        env = CourierEnv(paris, seed=0, difficulty=Difficulty.SHIFT)
        env.reset()
        live = env.live_orders()
        assert len(live) == Difficulty.queue_depth(Difficulty.SHIFT)
        assert all(o.issued_at_s == 0.0 for o in live), (
            "a queue whose deadlines start one after another is a queue of one"
        )


@needs_maps
class TestEasyTiersAreEasy:
    """A tier the competent fail is a broken tier, not a hard one."""

    def test_the_reference_courier_clears_solo_on_every_seed(self, paris):
        failures = [s for s in range(25) if episode(paris, Difficulty.SOLO, s).delivered_count < 1]
        assert not failures, f"solo failed on seeds {failures}"

    def test_the_reference_courier_nearly_clears_pair(self, paris):
        delivered = [episode(paris, Difficulty.PAIR, s).delivered_count for s in range(25)]
        assert sum(delivered) / (2 * len(delivered)) >= 0.9, (
            f"pair delivered {sum(delivered)}/{2*len(delivered)}"
        )

    def test_the_hard_tiers_are_still_solvable(self, paris):
        """Hard must mean "needs a better policy", not "needs luck". A queue-aware
        shortest-path courier finishes 97-100% of every rung; if that ever stops
        being true the tier has become unreachable rather than difficult."""
        import heapq

        def walk_shortest(env, max_steps=20000):
            for _ in range(max_steps):
                if env.shift_over:
                    return
                live = env.live_orders()
                if not live:
                    return
                best = None
                for order in live:
                    stop = order.dropoff if order.picked_up else order.pickup
                    distance = env.route_length_cm(env.node_id, stop.kerb_node)
                    if distance is not None and (best is None or distance < best[0]):
                        best = (distance, stop, order)
                if best is None:
                    return
                _, stop, order = best
                if math.dist(env.position(), stop.kerb) <= ARRIVAL_TOLERANCE_CM:
                    if not (env.hand_over() if order.picked_up else env.collect()).ok:
                        return
                    continue
                # one shortest-path hop
                previous, queue, seen = {env.node_id: None}, [(0.0, env.node_id)], set()
                goal = stop.kerb_node
                while queue:
                    cost, node = heapq.heappop(queue)
                    if node == goal:
                        break
                    if node in seen:
                        continue
                    seen.add(node)
                    for nb in env.network.nodes[node].neighbours:
                        if nb not in seen:
                            previous.setdefault(nb, node)
                            heapq.heappush(queue, (
                                cost + math.dist(env.position(node), env.position(nb)), nb))
                path, node = [], goal
                while node is not None:
                    path.append(node)
                    node = previous.get(node)
                path.reverse()
                if len(path) < 2:
                    return
                rows = {r["node"]: r for r in env.candidates()}
                if path[1] not in rows:
                    return
                row = rows[path[1]]
                env.walk_to(row["street"], row["heading"])

        for tier in (Difficulty.TRIPLE, Difficulty.SHIFT):
            delivered = issued = 0
            for seed in range(8):
                env = CourierEnv(paris, seed=seed, difficulty=tier)
                env.reset()
                walk_shortest(env)
                delivered += env.delivered_count
                issued += env.issued_count
            assert delivered / issued >= 0.9, f"{tier}: perfect play only got {delivered}/{issued}"


@needs_maps
class TestTheLadderDiscriminates:
    """A rung that a random walker scores like a courier is not a rung."""

    def test_random_street_choice_is_far_worse_on_every_tier(self, paris):
        for tier in Difficulty.ALL:
            reference = blind = 0.0
            for seed in range(10):
                reference += episode(paris, tier, seed).earnings
                env = CourierEnv(paris, seed=seed, difficulty=tier)
                env.reset()
                random_walk(env, seed)
                blind += env.earnings
            assert blind < reference * 0.35, (
                f"{tier}: random walk earned {blind:.2f} against the reference "
                f"courier's {reference:.2f} -- the tier is not measuring navigation"
            )


@needs_maps
class TestNoActionIsFree:
    """Both currencies have to see everything the courier does."""

    def env(self, paris, tier=Difficulty.SOLO):
        env = CourierEnv(paris, seed=0, difficulty=tier)
        env.reset()
        return env

    def test_a_refused_collect_is_not_a_free_rangefinder(self, paris):
        """A refusal must not quote the distance to the door.

        Charging 5 s for it was necessary and not sufficient: a block costs
        13-26 s to walk, so a priced refusal was still cheaper than moving, and
        walk-probe-walk gave a gradient oracle for the door that needed no
        photograph. It has to cost *and* say nothing.
        """
        env = self.env(paris)
        turns, seconds = env.turns, env.sim_seconds
        outcome = env.collect()
        assert not outcome.ok
        assert outcome.message == (
            f"You are not standing at {env.active_order().pickup.text}."
        )
        assert "m away" not in outcome.message
        assert env.turns == turns + 1
        assert env.sim_seconds == pytest.approx(seconds + REJECTED_ACTION_SECONDS)

    def test_walking_into_a_wall_costs_something(self, paris):
        env = self.env(paris)
        turns, seconds = env.turns, env.sim_seconds
        assert not env.walk_to("Rue Imaginaire", "north").ok
        assert env.turns == turns + 1 and env.sim_seconds > seconds

    def test_every_action_tool_counts_as_a_turn(self, paris):
        """``collect``, ``hand_over`` and ``wait`` never touched the counter, so
        three of the seven things a courier can do were invisible to a metric the
        benchmark reports as a headline."""
        env = self.env(paris)
        order = env.orders[0]
        for call, expected in (
            (lambda: env.wait(), 1),
            (lambda: env.collect(), 1),
            (lambda: env.hand_over(), 1),
            (lambda: env.look(*env.street_at(1)), 1),
            (lambda: env.check_order(), 1),
        ):
            before = env.turns
            call()
            assert env.turns == before + expected, call

    def test_a_successful_delivery_counts_as_a_turn(self, paris):
        env = self.env(paris)
        order = env.orders[0]
        env.node_id = order.pickup.kerb_node
        before = env.turns
        assert env.collect().ok
        assert env.turns == before + 1
        env.node_id = order.dropoff.kerb_node
        before = env.turns
        assert env.hand_over().ok
        assert env.turns == before + 1

    def test_following_a_street_is_one_decision(self, paris):
        """Its whole argument is that four junctions down one street is one
        decision for a rider. Each inner step charged a turn, so the macro was
        billed as four and the turn budget refused to reward what it was added
        to allow."""
        env = self.env(paris)
        before = env.turns
        outcome = env.follow_street(*env.street_at(1), 6)
        assert outcome.ok
        assert env.turns == before + 1
        assert outcome.walked_m > 0


@needs_maps
class TestOrdersAreRealJourneys:
    def test_no_order_is_degenerate(self, paris):
        for tier in BOUNDED:
            for seed in range(8):
                env = CourierEnv(paris, seed=seed, difficulty=tier)
                env.reset()
                cursor = env.node_id
                for order in env.orders:
                    approach = env.route_length_cm(cursor, order.pickup.kerb_node)
                    delivery = env.route_length_cm(order.pickup.kerb_node,
                                                   order.dropoff.kerb_node)
                    assert approach is not None and delivery is not None, "unreachable"
                    assert order.pickup.street_name != order.dropoff.street_name
                    assert order.pickup.text != order.dropoff.text
                    # A job that starts at the door is not a job. 2% of chained
                    # orders had a 0 m approach before this floor existed.
                    assert MIN_APPROACH_WALK_CM <= approach <= MAX_APPROACH_WALK_CM
                    assert 8000.0 <= delivery <= MAX_ORDER_WALK_CM
                    cursor = order.dropoff.kerb_node

    def test_the_work_is_varied_rather_than_one_journey_repeated(self, paris):
        env = CourierEnv(paris, seed=0, difficulty=Difficulty.SHIFT)
        env.reset()
        streets = {o.pickup.street_name for o in env.orders} | {
            o.dropoff.street_name for o in env.orders}
        assert len(streets) >= 6, streets
        legs = [env.route_length_cm(o.pickup.kerb_node, o.dropoff.kerb_node)
                for o in env.orders]
        assert statistics.pstdev(legs) > 2000.0, "every leg is the same length"


@needs_maps
class TestProfitIsAnObjective:
    """ENDLESS is scored on money, so money has to mean something."""

    def test_a_late_delivery_pays_less_than_a_punctual_one(self, paris):
        env = CourierEnv(paris, seed=0, difficulty=Difficulty.SOLO)
        env.reset()
        order = env.orders[0]
        env.node_id = order.pickup.kerb_node
        env.collect()
        env.node_id = order.dropoff.kerb_node
        env.sim_seconds = order.due_at() + 1.0
        assert env.hand_over().ok
        assert order.paid == pytest.approx(round(order.fee * LATE_FEE_FRACTION, 2))
        assert env.earnings == pytest.approx(order.paid)
        assert 0.0 < order.paid < order.fee

    def test_an_abandoned_order_pays_nothing_and_cannot_be_delivered(self, paris):
        """Without this the deadline was decorative under an objective that is
        literally "maximise profit": a delivery three hours late paid exactly
        what an on-time one did."""
        env = CourierEnv(paris, seed=0, difficulty=Difficulty.SOLO)
        env.reset()
        order = env.orders[0]
        env.node_id = order.pickup.kerb_node
        env.collect()
        env.node_id = order.dropoff.kerb_node
        env.sim_seconds = order.expires_at() + 1.0
        assert env.live_orders() == [] or order not in env.live_orders()
        assert order.expired and order.paid == 0.0
        assert not env.hand_over().ok
        assert env.earnings == 0.0

    def test_the_dispatcher_replaces_an_order_it_took_back(self, paris):
        """Being bad at the job must not shorten it. Before this a random walker
        was issued 3 of a ten-order shift and the episode simply stopped."""
        env = CourierEnv(paris, seed=0, difficulty=Difficulty.SHIFT)
        env.reset()
        env.sim_seconds = max(o.expires_at() for o in env.live_orders()) + 1.0
        live = env.live_orders()
        assert len(live) == Difficulty.queue_depth(Difficulty.SHIFT)
        assert env.expired_count >= 1
        assert all(o.issued_at_s == env.sim_seconds for o in live)

    def test_profit_has_no_ceiling(self, paris, monkeypatch):
        """A 40-order list was the whole of ENDLESS, so a policy four times
        better than the reference courier would have hit a wall the benchmark
        put there. Orders are drawn on demand instead: twice the clock, twice
        the money, and no list to exhaust."""
        def earn(hours: int) -> float:
            monkeypatch.setattr(Difficulty, "ENDLESS_SECONDS", 3600.0 * hours)
            total = 0.0
            for seed in range(6):
                env = CourierEnv(paris, seed=seed, difficulty=Difficulty.ENDLESS)
                env.reset()
                run_reference_courier(env, seed, max_steps=40000)
                total += env.earnings
            return total

        one, four = earn(1), earn(4)
        assert four > one * 2.5, f"four hours earned {four:.2f} against {one:.2f} in one"

    def test_no_order_length_is_a_free_lunch(self, paris):
        """If short orders paid better per second, the profit objective would
        reward cherry-picking rather than couriering. Measured over the drawn
        distribution the rate is flat to within a tenth."""
        rates = []
        for seed in range(20):
            env = CourierEnv(paris, seed=seed, difficulty=Difficulty.SHIFT)
            env.reset()
            cursor = env.node_id
            for order in env.orders:
                approach = env.route_length_cm(cursor, order.pickup.kerb_node)
                delivery = env.route_length_cm(order.pickup.kerb_node,
                                               order.dropoff.kerb_node)
                seconds = (approach + delivery) / WALK_SPEED_CM_S + 2.0 * HANDLING_SECONDS
                rates.append((delivery, order.fee / seconds))
                cursor = order.dropoff.kerb_node
        rates.sort()
        quarter = len(rates) // 4
        short = statistics.mean(r for _, r in rates[:quarter])
        long = statistics.mean(r for _, r in rates[-quarter:])
        assert abs(short - long) / short < 0.15, (
            f"short orders pay {short:.4f}/s and long ones {long:.4f}/s -- "
            "a profit-seeking courier should take only the short ones"
        )

    def test_the_courier_cannot_be_paid_twice_for_one_parcel(self, paris):
        env = CourierEnv(paris, seed=0, difficulty=Difficulty.SOLO)
        env.reset()
        order = env.orders[0]
        env.node_id = order.pickup.kerb_node
        env.collect()
        env.node_id = order.dropoff.kerb_node
        assert env.hand_over().ok
        paid = env.earnings
        assert not env.hand_over().ok
        assert env.earnings == paid


@needs_maps
class TestTheQueueIsVisibleAndServable:
    """A demand the observation hides is a demand that measures luck."""

    def test_every_live_job_is_in_the_observation(self, paris):
        env = CourierEnv(paris, seed=0, difficulty=Difficulty.SHIFT)
        env.reset()
        text = env.clock_text() + "\n" + env.check_order().message
        for order in env.live_orders():
            assert order.pickup.text in text or order.dropoff.text in text

    def test_a_job_can_be_served_out_of_the_order_it_arrived_in(self, paris):
        """The queue is a scheduling problem only if the schedule is the
        courier's to choose. Standing at any live pickup collects it, whichever
        job the observation happens to be focused on."""
        env = CourierEnv(paris, seed=0, difficulty=Difficulty.SHIFT)
        env.reset()
        live = env.live_orders()
        assert len(live) >= 2
        last = live[-1]
        assert env.active_order() is not last
        env.node_id = last.pickup.kerb_node
        assert env.collect().ok
        assert last.picked_up and not live[0].picked_up

    def test_the_focus_does_not_flip_under_a_committed_courier(self, paris):
        """A min-by-deadline default made the focus change as the clock ticked,
        so a policy steering by it thrashed between two targets and lost half its
        deliveries. Whatever is in the bag stays in the bag."""
        env = CourierEnv(paris, seed=0, difficulty=Difficulty.SHIFT)
        env.reset()
        live = env.live_orders()
        carried = live[-1]
        env.node_id = carried.pickup.kerb_node
        assert env.collect().ok
        assert env.active_order() is carried
        env.sim_seconds += 600.0
        assert env.active_order() is carried


@needs_maps
class TestThePhoneStatesItsOwnFrame:
    """Two numbers, two frames, and the sentence has to say which is which."""

    def test_the_phone_says_how_far_and_never_which_way(self, paris):
        """A spoken bearing made choosing a street a reading exercise.

        The lookup used to end "it lies to the east of you as the crow flies",
        which a policy can act on without looking at anything -- and once the
        phone speaks a direction, the photographs and the map are decoration.
        The distance is a number a courier could ask for; the direction is on
        the map, which is a picture.
        """
        env = CourierEnv(paris, seed=0, difficulty=Difficulty.SOLO)
        env.reset()
        message = env.check_map(env.orders[0].dropoff.text).message
        assert "on foot" in message
        for word in ("north", "south", "east", "west", "crow flies"):
            assert word not in message.lower(), message

    def test_the_courier_recognises_being_lost_and_goes_back(self, paris):
        """The recovery that every easy-tier failure needed. Without it the
        reference courier took one wrong turn and walked away from the door for
        the rest of the shift; with it the same policy scores 92% on TRIPLE
        against 88%, because a wrong turn stops being an absorbing state."""
        from embodiedbench.tasks.courier_oracle import ObservationOnlyCourier

        env = CourierEnv(paris, seed=0, difficulty=Difficulty.SOLO)
        env.reset()
        courier = ObservationOnlyCourier(env)
        assert not courier.lost, "no range seen yet is not the same as lost"
        courier.target_distance_m = 100.0
        courier.note_distance(100.0)
        assert not courier.lost
        courier.target_distance_m = 400.0
        assert courier.lost
        courier.forget_distances()
        assert not courier.lost, "a new door must not inherit the old one's record"


@needs_maps
class TestTheSummaryCanJudgeALongHorizon:
    def test_it_reports_both_currencies_and_the_route(self, paris):
        env = episode(paris, Difficulty.SHIFT, 0)
        summary = env.summary()
        for field in (
            "delivered", "on_time", "late", "expired", "orders_issued",
            "turns", "turns_per_delivery", "sim_seconds", "minutes_per_delivery",
            "profit", "earnings_per_hour", "deliveries_per_hour",
            "walked_m", "optimal_walk_m", "walk_ratio", "time_ratio",
            "rejected_actions", "queue_depth",
        ):
            assert field in summary, field
        assert summary["walk_ratio"] > 1.0, (
            "the reference courier does not walk the perfect route; a ratio of 1 "
            "means the comparison is against itself"
        )

    def test_orders_issued_counts_what_was_handed_over(self, paris):
        """``len(self.orders)`` was the denominator, and ENDLESS drew 40 up
        front, so a perfect courier scored 39%."""
        env = episode(paris, Difficulty.ENDLESS, 0)
        summary = env.summary()
        assert summary["orders_issued"] == sum(
            1 for o in env.orders if o.issued_at_s is not None)
        assert summary["delivered"] <= summary["orders_issued"]

    def test_a_deliberately_worse_policy_reads_as_worse_in_the_summary(self, paris):
        env = CourierEnv(paris, seed=0, difficulty=Difficulty.SHIFT)
        env.reset()
        random_walk(env, 0)
        blind = env.summary()
        good = episode(paris, Difficulty.SHIFT, 0).summary()
        assert blind["delivered"] < good["delivered"]
        assert blind["profit"] < good["profit"]
        assert blind["expired"] > good["expired"]


class TestStride:
    """One call, one waypoint -- or one call, one block. Same city either way.

    The block stride exists because a delivery cost the reference courier 76 to
    109 turns, 47% of them ``walk_to`` calls pressing the same button down the
    same street. It must not become a second, easier world: the same metres, the
    same seconds, the same lights and the same doors, asked less often.
    """

    def env(self, paris, stride, **kw):
        env = CourierEnv(paris, seed=3, difficulty="solo", stride=stride,
                         enforce_signals=False, enforce_obstacles=False, **kw)
        env.reset()
        return env

    def test_an_unknown_stride_is_refused_at_construction(self, paris):
        with pytest.raises(ValueError, match="stride"):
            CourierEnv(paris, seed=0, stride="teleport")

    def test_a_block_covers_more_ground_per_call_than_a_waypoint(self, paris):
        fine, blocks = self.env(paris, "waypoint"), self.env(paris, "block")
        assert fine.node_id == blocks.node_id, "same seed, same start"
        one = fine.walk_to(*fine.street_at(1))
        many = blocks.walk_to(*blocks.street_at(1))
        assert one.ok and many.ok
        assert many.walked_m >= one.walked_m

    def test_a_block_walk_costs_the_time_it_takes(self, paris):
        """The stride changes how often the courier is asked, not the clock.

        A block that took one turn and one waypoint's worth of seconds would be
        a shortcut through the city rather than a coarser view of it.
        """
        env = self.env(paris, "block")
        before = env.sim_seconds
        out = env.walk_to(*env.street_at(1))
        assert out.ok and out.walked_m > 0
        # Walking speed is the same constant either way, so the seconds charged
        # follow the metres covered rather than the number of calls.
        assert env.sim_seconds - before == pytest.approx(out.sim_seconds)
        assert out.sim_seconds >= out.walked_m * 100.0 / WALK_SPEED_CM_S - 1e-6

    def test_one_block_is_one_turn(self, paris):
        env = self.env(paris, "block")
        before = env.turns
        env.walk_to(*env.street_at(1))
        assert env.turns == before + 1

    def test_a_block_stops_where_a_choice_exists(self, paris):
        """The rule that separates the stride from the follow_street macro.

        The macro may walk through a side turning because its caller said "stay
        on this street". The stride may not, or the courier is never offered a
        turn it was standing on.
        """
        env = self.env(paris, "block")
        for _ in range(30):
            rows = env.candidates()
            if len(rows) > 2 and env.turns:
                break
            if not env.walk_to(*env.street_at(1)).ok:
                break
        else:
            pytest.skip("no junction reached within the walk")
        assert len(env.candidates()) > 2

    def test_a_block_stops_at_a_door_it_was_sent_to(self, paris):
        """Walking past the pickup would make the stride worse than not having it."""
        env = self.env(paris, "block")
        order = env.active_order()
        for _ in range(60):
            if math.dist(env.position(), order.target.kerb) <= ARRIVAL_TOLERANCE_CM:
                break
            rows = env.candidates()
            if not rows or not env.walk_to(*env.street_at(rows[0]["k"])).ok:
                break
        # Whether this seed's greedy walk reaches the door is not the claim; the
        # claim is that if it is standing at one, the walk stopped there.
        if math.dist(env.position(), order.target.kerb) <= ARRIVAL_TOLERANCE_CM:
            assert env.collect().ok or env.hand_over().ok

    def test_the_macro_is_not_offered_beside_the_stride_that_replaced_it(self, paris):
        """Two names for one action is the redundancy the tool audit forbids."""
        assert "follow_street" in self.env(paris, "waypoint").allowed_tool_names()
        assert "follow_street" not in self.env(paris, "block").allowed_tool_names()

    def test_both_strides_are_solvable_from_the_observation_alone(self, paris):
        """The measurement the stride was added for, at one seed to keep it quick.

        Full figures, five seeds a tier: the reference courier delivers 5/5,
        10/10 and 13/15 on SOLO, PAIR and TRIPLE at block stride against 5/5,
        10/10 and 14/15 at waypoint stride, for 44-49 turns a delivery instead of
        76-92. Two ways of spending fewer turns still were measured and rejected:
        consulting the phone every N moves (0/5 to 4/5 delivered, worse at every
        N) and skipping it on the target street (3/5 SOLO, 5/10 PAIR). Both break
        the same thing -- this policy steers by a range that shrinks, and a stale
        range is not a smaller one.
        """
        for stride in ("waypoint", "block"):
            env = CourierEnv(paris, seed=3, difficulty="solo", stride=stride)
            env.reset()
            courier = ObservationOnlyCourier(env, max_steps=2000)
            courier.run(3)
            assert env.delivered_count == 1, f"{stride} stride could not be solved"


class TestTheTurnCostOfADelivery:
    """What one delivery costs in turns, bracketed at both ends.

    The reference courier answers "is the observation sufficient" and nothing
    else. Asked how many turns a delivery costs it answers about its own habit:
    half of every episode it runs is phone lookups, because it navigates by a
    range that has to be re-read after every move. So the question needs the
    other bracket, and ``ShortestPathCourier`` is it -- privileged, never scored,
    and useful only for saying what the map costs once the navigating is given
    away.
    """

    def test_a_block_costs_about_half_the_turns_of_the_waypoints_in_it(self, paris):
        cost = {}
        for stride in ("waypoint", "block"):
            turns = delivered = 0
            for seed in range(3):
                env = CourierEnv(paris, seed=seed, difficulty="pair", stride=stride)
                result = run_shortest_path_courier(env, seed)
                turns += result.turns
                delivered += result.delivered
            assert delivered == 6, f"{stride}: the router should deliver everything"
            cost[stride] = turns / delivered
        assert cost["block"] < cost["waypoint"] * 0.75
        # The band the block stride exists to reach. Measured over five seeds a
        # tier it is 13.8 to 15.4 turns a delivery against 24.3 to 27.4.
        assert 10 <= cost["block"] <= 30

    def test_the_stride_changes_the_asking_not_the_walking(self, paris):
        """The property that makes the two strides one world rather than two.

        Same seed, same jobs, same route: the same metres walked and the same
        seconds spent. If a block were cheaper in anything but turns it would be
        a shortcut, and a score at one stride could not be compared with a score
        at the other.
        """
        runs = {}
        for stride in ("waypoint", "block"):
            env = CourierEnv(paris, seed=1, difficulty="pair", stride=stride)
            result = run_shortest_path_courier(env, 1)
            runs[stride] = result
        fine, blocks = runs["waypoint"], runs["block"]
        assert fine.delivered == blocks.delivered == 2
        assert blocks.walked_m == pytest.approx(fine.walked_m, rel=0.02)
        assert blocks.sim_seconds == pytest.approx(fine.sim_seconds, rel=0.02)
        assert blocks.turns < fine.turns

    def test_the_privileged_router_is_not_the_reference(self, paris):
        """It reads the graph, so it may never stand in for a policy score.

        Kept as a test rather than a comment because the failure it guards
        against is quiet: a table that quotes this courier's delivery rate as a
        baseline is reporting the map, not a model.
        """
        env = CourierEnv(paris, seed=0, difficulty="solo")
        router = ShortestPathCourier(env)
        env.reset()
        assert router._next_step(env.active_order().target.kerb_node) is not None
        # Its one privilege, named: the road network. Everything else it uses is
        # a tool the prompt advertises.
        source = Path(courier_router_module.__file__).read_text()
        assert "route_nodes" in source
        assert "self.env.network.nodes" not in source or "position" in source


class TestTheRouteSpeaksTheStrideYouWalk:
    """Found by playing the environment by hand rather than by reading it.

    The phone said "take Avenue des Rosiers north-east, 6 junctions, 108 m", one
    ``walk_to`` covered 18 m and stopped, and there was no way to tell from
    inside whether the courier had gone the wrong way or simply been counted in
    different units. A route whose units are not the caller's units is a route
    the caller cannot check itself against.
    """

    def legs(self, paris, stride):
        env = CourierEnv(paris, seed=7, difficulty="pair", stride=stride)
        env.reset()
        target = env.active_order().target
        return env, env.route_legs(env.node_id, target.kerb_node)

    def test_a_leg_counts_the_calls_it_takes_at_this_stride(self, paris):
        fine_env, fine = self.legs(paris, "waypoint")
        block_env, blocks = self.legs(paris, "block")
        assert fine and blocks
        # Same route, same metres -- only the counting changes.
        assert sum(leg["metres"] for leg in blocks) == pytest.approx(
            sum(leg["metres"] for leg in fine), rel=1e-6)
        assert sum(leg["junctions"] for leg in blocks) < sum(
            leg["junctions"] for leg in fine)

    def test_walking_the_first_leg_takes_the_calls_the_leg_promised(self, paris):
        """The check the courier could not make. One call per counted junction."""
        env, legs = self.legs(paris, "block")
        first = legs[0]
        rows = {row["street"]: row for row in env.candidates()}
        assert first["street"] in rows
        calls = 0
        for _ in range(first["junctions"] + 3):
            row = next((r for r in env.candidates()
                        if r["street"] == first["street"]
                        and r["node"] != env.arrived_from), None)
            if row is None or env.node_id == first["end"]:
                break
            if not env.walk_to(*env.street_at(row["k"])).ok:
                break
            calls += 1
            if env.node_id == first["end"]:
                break
        assert calls <= first["junctions"], (
            f"the leg promised {first['junctions']} calls and took {calls}")

    def test_the_route_is_drawn_rather_than_dictated(self, paris):
        """Turn-by-turn text is the whole navigation problem, given away.

        Every leg used to be spelled out -- "Take Rue de Grenelle, east, 1
        junction, 18 m" -- so a courier holding that text never had to look at
        anything. What survives is what a phone tells you at a glance: how far,
        how long, how many streets. Which way is on the map.
        """
        env = CourierEnv(paris, seed=7, difficulty="pair", stride="block")
        env.reset()
        message = env.navigate().message
        assert "m," in message and "min on foot" in message
        for word in ("north", "south", "east", "west", "left", "right"):
            assert word not in message.lower(), message
        assert not any(line.strip().startswith(("1.", "2.", "3."))
                       for line in message.splitlines()), message

    def blocked_env(self, paris, stride):
        env = CourierEnv(paris, seed=0, difficulty=Difficulty.TRIPLE, stride=stride,
                         album_root=Path(os.environ.get("ALBUMS_DIR", "/data/albums")) / Path("paris_streets_v2/citycore-paris"),
                         obstacle_album_root=Path(os.environ.get("ALBUMS_DIR", "/data/albums")) / Path("paris_obstacles/citycore-paris"))
        env.reset()
        return env

    def test_the_tool_is_gone_and_says_why(self):
        from embodiedbench.agent.courier.tools import ALL_TOOLS, RETIRED_TOOLS

        assert "report_blocked" not in {t.name for t in ALL_TOOLS}
        assert "report_blocked" in RETIRED_TOOLS
        assert len(RETIRED_TOOLS["report_blocked"]) > 60, "a reason, not a tombstone"

    @pytest.mark.parametrize("stride", ["waypoint", "block"])
    def test_the_route_keeps_naming_a_street_it_has_been_walked_into(self, paris, stride):
        """The property the whole change rests on.

        Walking into a barrier teaches the courier and teaches the phone
        nothing. If the route quietly stopped offering the street, the
        photographs would be decoration again.
        """
        env = self.blocked_env(paris, stride)
        assert not hasattr(env, "report_blocked")
        for _ in range(60):
            rows = env.candidates()
            if not rows:
                break
            target = env.target_address()
            before = env.route_nodes(env.node_id, target.kerb_node) if target else None
            row = rows[0]
            outcome = env.walk_to(*env.street_at(row["k"]))
            if not outcome.ok and outcome.code == "way_blocked":
                after = env.route_nodes(env.node_id, target.kerb_node)
                assert after == before, (
                    "the survey learned something from the courier walking into a wall")
                return
        pytest.skip("no barrier reached within the walk")

    def test_the_ceiling_still_reaches_it_by_remembering_and_replanning(self, paris):
        """Solvable without the crutch -- which is what makes removing it fair.

        The router keeps its own record of what it has walked into and searches
        the graph minus those edges. Asking the survey each turn and guessing
        greedily when the answer was unwalkable delivered 4 of 12 over six seeds
        and hit 235 barriers; this delivers everything.
        """
        delivered = issued = 0
        for seed in range(3):
            env = CourierEnv(paris, seed=seed, difficulty=Difficulty.PAIR,
                             stride=Stride.BLOCK,
                             album_root=Path(os.environ.get("ALBUMS_DIR", "/data/albums")) / Path("paris_streets_v2/citycore-paris"),
                             obstacle_album_root=Path(os.environ.get("ALBUMS_DIR", "/data/albums")) / Path("paris_obstacles/citycore-paris"))
            result = run_shortest_path_courier(env, seed, max_steps=9000)
            delivered += result.delivered
            issued += result.issued
        assert delivered == issued, f"{delivered}/{issued} without a way to tell the phone"
