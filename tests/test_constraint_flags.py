"""The optional constraints are optional: off is the benchmark, exactly.

Five flags were added on top of a benchmark whose arms were already training,
so the property these tests defend first is *absence*: with every flag at its
default, an environment built after this feature is the same environment,
order for order and fee for fee, as one built before it. Each flag draws any
randomness it needs from its own stream, keyed by (seed, order index), so a
flagged arm and a control arm walk the very same city and differ only in what
the slip says and what the door pays.

Then each mechanic, on: jitter moves fees and nothing else; cold food pays
``COLD_FEE_FRACTION`` of what warm food would; a note changes what the door
costs; walking energy off means the courier never tires and is never offered
a rest; the phone's battery runs out, the screen goes dark, and the dispatcher
stops relighting it.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from embodiedbench.compiler.road_network import build_road_network
from embodiedbench.runtime.city.courier_env import (
    ARRIVAL_TOLERANCE_CM,
    COLD_FEE_FRACTION,
    FOOD_WARM_SECONDS,
    HANDLING_SECONDS,
    LATE_FEE_FRACTION,
    NOTE_DOOR_HANDLING_S,
    NOTE_LEAVE_AT_DOOR,
    NOTE_RING_EXTRA_S,
    NOTE_RING_FIRST,
    ORDER_NOTES,
    PHONE_BATTERY_NAVIGATE_PCT,
    CourierEnv,
)

MAPS = (Path(__file__).resolve().parents[1] / "vendor" / "vagen" / "vagen"
        / "envs" / "deliverybench" / "maps")
PARIS = MAPS / "citycore-paris"
needs_maps = pytest.mark.skipif(not PARIS.exists(), reason="vendored maps not present")

ALL_ON = dict(
    enable_earning_jitter=True,
    enable_food_temperature=True,
    enable_special_notes=True,
    enable_phone_battery=True,
)


@pytest.fixture(scope="module")
def paris():
    return build_road_network(PARIS, map_name="citycore-paris")


def fresh(network, seed=3, **kwargs) -> CourierEnv:
    env = CourierEnv(network, seed=seed, difficulty="endless", **kwargs)
    env.reset()
    return env


def stand_at_pickup(env: CourierEnv):
    """Teleport to the active order's pickup door, as a test may and a courier may not."""
    order = env.active_order()
    env.node_id = order.pickup.kerb_node
    assert math.dist(env.position(), order.pickup.kerb) <= ARRIVAL_TOLERANCE_CM
    return order


def stand_at_dropoff(env: CourierEnv, order):
    env.node_id = order.dropoff.kerb_node
    assert math.dist(env.position(), order.dropoff.kerb) <= ARRIVAL_TOLERANCE_CM


@needs_maps
class TestOffIsTheBenchmark:
    def test_flags_on_leave_the_geometry_alone(self, paris):
        """Same seed, flags on and off: same doors, same deadlines, same city."""
        base = fresh(paris, seed=5)
        flagged = fresh(paris, seed=5, **ALL_ON)
        assert len(base.orders) == len(flagged.orders)
        for a, b in zip(base.orders, flagged.orders):
            assert a.pickup.text == b.pickup.text
            assert a.dropoff.text == b.dropoff.text
            assert a.deadline_s == b.deadline_s
        assert base.node_id == flagged.node_id

    def test_defaults_report_themselves(self, paris):
        env = fresh(paris, seed=5)
        summary = env.summary()
        assert summary["constraints"] == {
            "earning_jitter": False, "food_temperature": False,
            "special_notes": False, "walking_energy": True,
            "phone_battery": False, "food_categories": False,
            "phone_recharge": False,
        }
        assert summary["phone_battery_left"] is None
        assert summary["cold_deliveries"] == 0
        assert all(o.note == "" for o in env.orders)

    def test_default_fee_is_the_published_formula(self, paris):
        env = fresh(paris, seed=5)
        for order in env.orders:
            walk = env.route_length_cm(order.pickup.kerb_node, order.dropoff.kerb_node)
            assert order.fee == env.fee_for(walk)


@needs_maps
class TestEarningJitter:
    def test_jitter_moves_fees_and_reproduces(self, paris):
        base = fresh(paris, seed=7)
        jittered = fresh(paris, seed=7, enable_earning_jitter=True)
        again = fresh(paris, seed=7, enable_earning_jitter=True)
        fees = [(a.fee, b.fee) for a, b in zip(base.orders, jittered.orders)]
        assert any(a != b for a, b in fees), "jitter never moved a single fee"
        assert [o.fee for o in jittered.orders] == [o.fee for o in again.orders]
        # Within the declared band, and never below the floor.
        for a, b in fees:
            assert b >= 1.0
            assert abs(b - a) <= a * 0.2 + 0.01

    def test_different_seeds_draw_different_swings(self, paris):
        one = fresh(paris, seed=7, enable_earning_jitter=True)
        two = fresh(paris, seed=8, enable_earning_jitter=True)
        # Not a theorem, but 2x40 independent draws agreeing would be one.
        assert [o.fee for o in one.orders[:5]] != [o.fee for o in two.orders[:5]]


@needs_maps
class TestFoodTemperature:
    def test_cold_delivery_pays_the_fraction(self, paris):
        env = fresh(paris, seed=3, enable_food_temperature=True)
        order = stand_at_pickup(env)
        assert env.collect().ok
        assert order.picked_up_at_s is not None
        # Dawdle past the warm window (and probably the deadline; both
        # discounts are asserted together so the arithmetic is pinned).
        env.sim_seconds = order.picked_up_at_s + FOOD_WARM_SECONDS + 120.0
        stand_at_dropoff(env, order)
        outcome = env.hand_over()
        assert outcome.ok
        late = order.delivered_at_s > order.due_at()
        expected = order.fee * (LATE_FEE_FRACTION if late else 1.0) * COLD_FEE_FRACTION
        assert order.paid == pytest.approx(round(expected, 2))
        assert env.cold_deliveries == 1
        assert "cold" in outcome.message

    def test_prompt_delivery_pays_in_full(self, paris):
        env = fresh(paris, seed=3, enable_food_temperature=True)
        order = stand_at_pickup(env)
        assert env.collect().ok
        stand_at_dropoff(env, order)
        outcome = env.hand_over()
        assert outcome.ok
        assert order.paid == order.fee
        assert env.cold_deliveries == 0
        assert "cold" not in outcome.message

    def test_collect_warns_only_under_the_flag(self, paris):
        cold_world = fresh(paris, seed=3, enable_food_temperature=True)
        stand_at_pickup(cold_world)
        assert "hot" in cold_world.collect().message
        plain_world = fresh(paris, seed=3)
        stand_at_pickup(plain_world)
        assert "hot" not in plain_world.collect().message


@needs_maps
class TestSpecialNotes:
    def test_notes_are_drawn_and_reproduce(self, paris):
        one = fresh(paris, seed=11, enable_special_notes=True)
        two = fresh(paris, seed=11, enable_special_notes=True)
        assert [o.note for o in one.orders] == [o.note for o in two.orders]
        assert all(o.note in ORDER_NOTES for o in one.orders)

    def _deliver_with_note(self, paris, note: str) -> tuple[CourierEnv, float]:
        env = fresh(paris, seed=3, enable_special_notes=True)
        order = stand_at_pickup(env)
        order.note = note   # pin the note; the draw is tested separately
        assert env.collect().ok
        stand_at_dropoff(env, order)
        before = env.sim_seconds
        assert env.hand_over().ok
        return env, env.sim_seconds - before

    def test_door_drop_is_quicker(self, paris):
        env, spent = self._deliver_with_note(paris, NOTE_LEAVE_AT_DOOR)
        assert spent == NOTE_DOOR_HANDLING_S
        assert env.notes_followed == 1

    def test_ringing_first_is_slower(self, paris):
        env, spent = self._deliver_with_note(paris, NOTE_RING_FIRST)
        assert spent == HANDLING_SECONDS + NOTE_RING_EXTRA_S
        assert env.notes_followed == 1

    def test_no_note_is_the_plain_door(self, paris):
        env, spent = self._deliver_with_note(paris, "")
        assert spent == HANDLING_SECONDS
        assert env.notes_followed == 0

    def test_the_slip_prints_the_note(self, paris):
        env = fresh(paris, seed=3, enable_special_notes=True)
        order = env.active_order()
        order.note = NOTE_RING_FIRST
        assert NOTE_RING_FIRST in env._check_order_impl().message


@needs_maps
class TestWalkingEnergy:
    def walk_once(self, env: CourierEnv) -> None:
        row = env.candidates()[0]
        assert env.walk_to(*env.street_at(row["k"])).ok

    def test_on_by_default_and_draining(self, paris):
        env = fresh(paris, seed=3)
        full = env.stamina
        self.walk_once(env)
        assert env.stamina < full

    def test_off_never_tires_and_never_rests(self, paris):
        env = fresh(paris, seed=3, enable_walking_energy=False)
        full = env.stamina
        self.walk_once(env)
        assert env.stamina == full
        refusal = env.rest()
        assert not refusal.ok and refusal.code == "never_tires"
        assert "REST" not in " ".join(env.allowed_tool_names()).upper()


@needs_maps
class TestPhoneBattery:
    def test_routes_cost_charge(self, paris):
        env = fresh(paris, seed=3, enable_phone_battery=True)
        before = env.phone_battery
        assert env.navigate().ok
        assert env.phone_battery == before - PHONE_BATTERY_NAVIGATE_PCT

    def test_the_screen_drains_while_lit(self, paris):
        env = fresh(paris, seed=3, enable_phone_battery=True)
        assert env.screen_target is not None, "the screen starts lit"
        before = env.phone_battery
        row = env.candidates()[0]
        assert env.walk_to(*env.street_at(row["k"])).ok
        assert env.phone_battery < before

    def test_a_dead_phone_stays_dead(self, paris):
        env = fresh(paris, seed=3, enable_phone_battery=True)
        env.phone_battery = 0.3
        row = env.candidates()[0]
        assert env.walk_to(*env.street_at(row["k"])).ok
        assert env.phone_battery == 0.0
        assert env.screen_target is None and env.screen_route == []
        assert env.phone_died_at_s is not None
        refusal = env.navigate()
        assert not refusal.ok and refusal.code == "phone_dead"
        # The dispatcher must not relight a dead screen on the next job.
        env._issue()
        assert env.screen_target is None

    def test_off_means_no_battery_at_all(self, paris):
        env = fresh(paris, seed=3)
        assert env.phone_battery is None
        assert env.navigate().ok
        assert env.phone_battery is None


@needs_maps
class TestTheAgentIsTold:
    """A rule the world charges for has to be in the prompt, and only then."""

    def session_of(self, paris, **flags):
        from embodiedbench.agent.courier.session import CourierSession
        env = fresh(paris, seed=3, **flags)
        return CourierSession(env, city="Paris", with_images=False), env

    def test_no_flags_no_special_rules(self, paris):
        session, _ = self.session_of(paris)
        assert "SPECIAL RULES" not in session.system_prompt()
        assert "Phone battery" not in session.observe().text

    def test_every_active_flag_has_its_paragraph(self, paris):
        session, _ = self.session_of(paris, **ALL_ON)
        prompt = session.system_prompt()
        assert "SPECIAL RULES THIS SHIFT" in prompt
        for phrase in ("FEES VARY", "HOT FOOD GOES COLD",
                       "READ THE NOTE ON THE SLIP", "BATTERY IS FINITE"):
            assert phrase in prompt, phrase

    def test_one_flag_renders_one_rule(self, paris):
        session, _ = self.session_of(paris, enable_food_temperature=True)
        prompt = session.system_prompt()
        assert "HOT FOOD GOES COLD" in prompt
        assert "BATTERY IS FINITE" not in prompt

    def test_the_observation_reads_out_the_battery(self, paris):
        session, env = self.session_of(paris, enable_phone_battery=True)
        assert "Phone battery: 100%." in session.observe().text
        env.phone_battery = 0.0
        # The world changed behind the session's back; observe() is memoised
        # within a turn, so the test says so the way a harness would.
        session.refresh()
        assert "phone is dead" in session.observe().text

    def test_the_clock_line_carries_the_note(self, paris):
        session, env = self.session_of(paris, enable_special_notes=True)
        order = env.active_order()
        order.note = NOTE_LEAVE_AT_DOOR
        assert NOTE_LEAVE_AT_DOOR in session.observe().text

    def test_the_clock_line_carries_the_food_timer(self, paris):
        session, env = self.session_of(paris, enable_food_temperature=True)
        order = stand_at_pickup(env)
        assert env.collect().ok
        assert "The food has been out 0 min; it goes cold at 8." in session.observe().text
        env.sim_seconds = order.picked_up_at_s + FOOD_WARM_SECONDS + 60.0
        session.refresh()
        assert "has gone cold" in session.observe().text


@needs_maps
class TestComplianceIsReported:
    """Every flag that is on reports its execution rate to the trainer.

    Through the real training glue (CourierGymEnv.step), because that is the
    only path wandb ever sees: an env summary nobody forwards is a dashboard
    that lies by omission.
    """

    METRIC_KEYS = {"cold_deliveries", "warm_delivery_rate",
                   "melted_deliveries", "intact_delivery_rate",
                   "notes_followed", "noted_delivery_rate",
                   "phone_battery_left", "phone_alive_rate", "phone_recharges",
                   "mean_fee_paid", "rests", "stamina_left"}

    def step_info(self, **config):
        import asyncio
        from embodiedbench.training.vagen_courier_env import CourierGymEnv
        env = CourierGymEnv(env_config={
            "difficulty": "endless", "hazards": False, **config})
        asyncio.run(env.reset(seed=3))
        _, _, _, info = asyncio.run(env.step("THOUGHT: waiting.\n```\nwait()\n```"))
        return info

    def test_the_key_set_never_depends_on_the_flags(self):
        """The invariant DataProto.concat enforces with an assert.

        The constraint val yaml mixes flags-on and flags-off blocks in one
        validation pass; verl's concat requires every episode dict to carry
        identical keys, so a key that appears only under its flag kills the
        whole batch -- that is exactly how courier-c2-battery died at step 0.
        """
        off = self.step_info()
        on = self.step_info(enable_earning_jitter=True,
                            enable_food_temperature=True,
                            enable_special_notes=True,
                            enable_phone_battery=True)
        one = self.step_info(enable_food_temperature=True)
        assert set(off) == set(on) == set(one)
        assert self.METRIC_KEYS <= set(off)

    def test_flags_off_reports_the_inert_constants(self):
        info = self.step_info()
        assert info["warm_delivery_rate"] == 1.0
        assert info["cold_deliveries"] == 0
        assert info["noted_delivery_rate"] == 0.0
        assert info["phone_battery_left"] == 100.0
        assert info["phone_alive_rate"] == 1.0

    def test_every_open_flag_reports_its_rate(self):
        info = self.step_info(enable_earning_jitter=True,
                              enable_food_temperature=True,
                              enable_special_notes=True,
                              enable_phone_battery=True)
        # No deliveries yet: warm compliance is vacuous (1.0), exposure and
        # fee are 0, the phone is alive on a full battery.
        assert info["warm_delivery_rate"] == 1.0
        assert info["noted_delivery_rate"] == 0.0
        assert info["mean_fee_paid"] == 0.0
        assert info["phone_alive_rate"] == 1.0
        assert info["phone_battery_left"] > 0.0
        assert "rests" in info and "stamina_left" in info

    def test_the_loops_forward_the_same_keys(self):
        """The forwarding lists in both agent-loop variants cover every key.

        Parsed rather than imported: the vendored loops import verl, which is
        a training-node dependency this test machine deliberately lacks.
        """
        import ast
        loop = (Path(__file__).resolve().parents[1] / "vendor" / "vagen"
                / "vagen" / "agent_loop" / "gym_agent_loop.py")
        tree = ast.parse(loop.read_text())
        def literal(name):
            return next(
                ast.literal_eval(node.value)
                for node in ast.walk(tree)
                if isinstance(node, ast.Assign)
                and any(getattr(t, "id", "") == name for t in node.targets))

        keys = set(literal("CONSTRAINT_METRIC_KEYS"))
        assert keys == self.METRIC_KEYS
        # The error-path defaults must cover every key, or a trajectory whose
        # env died before its first report re-creates the missing-key crash.
        assert set(literal("CONSTRAINT_METRIC_DEFAULTS")) == keys
        no_concat = loop.with_name("gym_agent_loop_no_concat.py").read_text()
        assert "CONSTRAINT_METRIC_KEYS" in no_concat
        assert "CONSTRAINT_METRIC_DEFAULTS" in no_concat


@needs_maps
class TestFoodCategories:
    def test_categories_draw_and_reproduce(self, paris):
        one = fresh(paris, seed=11, enable_food_categories=True)
        two = fresh(paris, seed=11, enable_food_categories=True)
        assert [o.category for o in one.orders] == [o.category for o in two.orders]
        assert all(o.category in ("hot meal", "ice cream", "groceries")
                   for o in one.orders)

    def test_flag_off_no_categories(self, paris):
        env = fresh(paris, seed=11)
        assert all(o.category == "" for o in env.orders)

    def _deliver_category(self, paris, category, dawdle_s):
        from embodiedbench.runtime.city.courier_env import (
            ICECREAM_MELT_SECONDS, MELT_FEE_FRACTION)
        env = fresh(paris, seed=3, enable_food_categories=True)
        order = stand_at_pickup(env)
        order.category = category
        assert env.collect().ok
        env.sim_seconds = order.picked_up_at_s + dawdle_s
        stand_at_dropoff(env, order)
        outcome = env.hand_over()
        assert outcome.ok
        return env, order, outcome

    def test_ice_cream_melts_on_a_short_fuse(self, paris):
        from embodiedbench.runtime.city.courier_env import (
            ICECREAM_MELT_SECONDS, LATE_FEE_FRACTION, MELT_FEE_FRACTION)
        env, order, outcome = self._deliver_category(
            paris, "ice cream", ICECREAM_MELT_SECONDS + 60.0)
        late = order.delivered_at_s > order.due_at()
        expected = order.fee * (LATE_FEE_FRACTION if late else 1.0) * MELT_FEE_FRACTION
        assert order.paid == pytest.approx(round(expected, 2))
        assert env.melted_deliveries == 1 and env.cold_deliveries == 0
        assert "melted" in outcome.message

    def test_ice_cream_survives_a_fast_leg(self, paris):
        from embodiedbench.runtime.city.courier_env import ICECREAM_MELT_SECONDS
        env, order, outcome = self._deliver_category(
            paris, "ice cream", ICECREAM_MELT_SECONDS - 120.0)
        assert env.melted_deliveries == 0
        assert "melted" not in outcome.message

    def test_groceries_never_spoil(self, paris):
        # 15 minutes: past both spoil windows, short of the expiry sweep
        # (deadline x3) that would take the order back before the door.
        env, order, outcome = self._deliver_category(paris, "groceries", 900.0)
        assert env.melted_deliveries == 0 and env.cold_deliveries == 0
        assert "cold" not in outcome.message and "melted" not in outcome.message

    def test_the_slip_names_the_contents(self, paris):
        env = fresh(paris, seed=3, enable_food_categories=True)
        order = env.active_order()
        order.category = "ice cream"
        assert "Contents: ice cream." in env._check_order_impl().message
        assert "Carrying: ice cream." in env.clock_text()

    def test_categories_get_their_prompt_paragraph(self, paris):
        from embodiedbench.agent.courier.session import CourierSession
        env = fresh(paris, seed=3, enable_food_categories=True)
        prompt = CourierSession(env, city="Paris", with_images=False).system_prompt()
        assert "WHAT IS IN THE BAG" in prompt
        assert "HOT FOOD GOES COLD" not in prompt   # subsumed, not duplicated


@needs_maps
class TestPhoneRecharge:
    def test_recharge_requires_battery(self, paris):
        with pytest.raises(ValueError):
            fresh(paris, seed=3, enable_phone_recharge=True)

    def test_charge_costs_time_and_buys_charge(self, paris):
        from embodiedbench.runtime.city.courier_env import (
            PHONE_RECHARGE_PCT, PHONE_RECHARGE_SECONDS)
        env = fresh(paris, seed=3, enable_phone_battery=True,
                    enable_phone_recharge=True)
        env.phone_battery = 30.0
        before = env.sim_seconds
        outcome = env.charge_phone()
        assert outcome.ok
        assert env.phone_battery == 30.0 + PHONE_RECHARGE_PCT
        assert env.sim_seconds - before == PHONE_RECHARGE_SECONDS
        assert env.phone_recharges == 1

    def test_charge_revives_a_dead_phone(self, paris):
        env = fresh(paris, seed=3, enable_phone_battery=True,
                    enable_phone_recharge=True)
        env.phone_battery = 0.0
        env.screen_target = None
        env.screen_route = []
        outcome = env.charge_phone()
        assert outcome.ok and "comes back on" in outcome.message
        assert env.phone_battery > 0
        assert env.screen_target is not None, "the screen relights on the job"

    def test_tool_enters_the_menu_with_the_flag(self, paris):
        with_bank = fresh(paris, seed=3, enable_phone_battery=True,
                          enable_phone_recharge=True)
        assert "charge_phone" in with_bank.allowed_tool_names()
        without = fresh(paris, seed=3, enable_phone_battery=True)
        assert "charge_phone" not in without.allowed_tool_names()
        refusal = without.charge_phone()
        assert not refusal.ok and refusal.code == "nothing_to_charge"

    def test_power_bank_gets_its_prompt_paragraph(self, paris):
        from embodiedbench.agent.courier.session import CourierSession
        env = fresh(paris, seed=3, enable_phone_battery=True,
                    enable_phone_recharge=True)
        prompt = CourierSession(env, city="Paris", with_images=False).system_prompt()
        assert "POWER BANK" in prompt


@needs_maps
class TestDifficultyKnobs:
    def test_deadline_slack_scales_the_windows(self, paris):
        loose = fresh(paris, seed=3, deadline_slack=3.5)
        tight = fresh(paris, seed=3, deadline_slack=1.8)
        # Same seed, same orders; only the windows differ.
        assert [o.pickup.text for o in loose.orders] == [o.pickup.text for o in tight.orders]
        assert all(l.deadline_s > t.deadline_s
                   for l, t in zip(loose.orders, tight.orders))

    def test_warm_seconds_knob_moves_the_fuse(self, paris):
        env = fresh(paris, seed=3, enable_food_temperature=True,
                    food_warm_seconds=300.0)
        order = stand_at_pickup(env)
        assert env.collect().ok
        env.sim_seconds = order.picked_up_at_s + 360.0   # past 5 min, under 8
        stand_at_dropoff(env, order)
        assert env.hand_over().ok
        assert env.cold_deliveries == 1
