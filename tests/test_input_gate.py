"""The input-quality gate and the observation rewrite it drove.

Every case here is a defect the gate actually found on Paris, kept as a test so a
change to the observation layer that reintroduces one fails loudly. The ordering
matters: the static checks are properties of a string and an EnvSpec, so they
need no engine, and only the end-to-end case needs the vendored checkout.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from embodiedbench.schemas.env_spec import EnvSpec
from embodiedbench.tasks.input_gate import (
    MAX_WORKABLE_CHOICES,
    ObservationOnlyOracle,
    angular_gap,
    check_observation,
    objective,
    parse_marks,
)
from embodiedbench.tasks.observation_rewrite import (
    Candidate,
    Guidance,
    bearing_between,
    compass_of,
    rewrite_observation,
)

VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "vagen"
PARIS_SPEC = (
    Path(__file__).resolve().parents[1]
    / "artifacts" / "verification" / "MAPS" / "citycore-paris.envspec.json"
)

# The observation exactly as the vendored engine renders it on Paris, including
# the two defects: a MOVE block on a map that does not enable MOVE, and a
# routing hint in directional language.
RAW_PARIS = """### agent_state
You are Agent 1, you are facing North, you are at 115 Union Ave. Inventory: empty.
### reachable_waypoints
From here, MOVE(direction=...) reaches (relative to your facing):
  - forward : (blocked)
  - left    : (blocked)
  - right   : (blocked)
  - backward: reachable
### active_orders
You have accepted the following active orders:

[Order #0]
  Pickup : 131 Union Ave
  Dropoff: 46 Hill St
  Status : Ready for pickup
### ephemeral_context
[navigation]
mode: walk
from: 115 Union Ave
to: 90 Main St (restaurant 1)
distance_m: 202.2
next_move: move backward

### waypoint_marks
1 reachable adjacent waypoint(s), numbered below. Move with MOVE_TO(<number>).
- MOVE_TO(1): dock_275 (55 Cherry St), 153.7 m, behind you
"""

REWRITTEN = """### agent_state
You are Agent 1, you are facing North, you are at 131 Union Ave. Inventory: empty.
### active_orders
[Order #0]
  Pickup : 131 Union Ave
  Dropoff: 46 Hill St
  Status : Ready for pickup

### objective
Collect the order from its restaurant.
Your destination is 444 m away, to the west (bearing 258 deg).
Choose the numbered waypoint that moves you toward it.

### waypoint_marks (2 of 2 shown)
- MOVE_TO(1): 55 Cherry St [dock_275], 154 m to the west
- MOVE_TO(2): 20 Park St [int_467], 20 m to the east
"""


def paris_spec() -> EnvSpec:
    return EnvSpec.from_dict(json.loads(PARIS_SPEC.read_text()))


needs_spec = pytest.mark.skipif(not PARIS_SPEC.exists(), reason="no compiled Paris EnvSpec")


# ─────────────────────────────────────────────────────────────────────────────
# Static checks
# ─────────────────────────────────────────────────────────────────────────────


@needs_spec
class TestStaticChecks:
    def test_move_block_on_a_move_to_map_is_blocking(self):
        """The defect that motivated the whole gate.

        Paris disables MOVE because only 17.4% of its edges are near-cardinal,
        yet the observation instructs the agent to use it. A policy that obeys
        the text is rejected on every turn.
        """
        findings = check_observation(RAW_PARIS, paris_spec())
        codes = {f.code for f in findings}
        assert "observation_advertises_disabled_action" in codes
        assert any(f.severity == "blocking" for f in findings)

    def test_directional_routing_hint_is_blocking(self):
        findings = check_observation(RAW_PARIS, paris_spec())
        assert "routing_hint_uses_disabled_move" in {f.code for f in findings}

    def test_the_rewritten_observation_has_no_blocking_finding(self):
        """The fix has to actually clear the checks, not merely look different."""
        findings = check_observation(REWRITTEN, paris_spec())
        blocking = [f for f in findings if f.severity == "blocking"]
        assert blocking == [], [f.to_dict() for f in blocking]

    def test_missing_numbered_choices_is_blocking(self):
        stripped = REWRITTEN.split("### waypoint_marks")[0]
        codes = {f.code for f in check_observation(stripped, paris_spec())}
        assert "no_enumerated_choices" in codes

    def test_an_unstated_objective_is_blocking(self):
        codes = {f.code for f in check_observation("### agent_state\nYou are here.", paris_spec())}
        assert "no_stated_objective" in codes

    def test_too_many_choices_is_reported_but_not_blocking(self):
        """Paris reaches degree 35. That is a real degradation -- it measures
        list indexing rather than navigation -- but it does not make the input
        unusable, so it must not stop an otherwise valid environment."""
        lines = "\n".join(
            f"- MOVE_TO({i}): Street {i} [n_{i}], {i * 3} m to the north"
            for i in range(1, MAX_WORKABLE_CHOICES + 6)
        )
        text = REWRITTEN.split("### waypoint_marks")[0] + "### waypoint_marks\n" + lines
        findings = check_observation(text, paris_spec())
        codes = {f.code for f in findings}
        assert "too_many_choices" in codes
        assert all(f.severity != "blocking" for f in findings if f.code == "too_many_choices")

    def test_a_single_choice_is_reported_as_no_decision(self):
        codes = {f.code for f in check_observation(RAW_PARIS, paris_spec())}
        assert "no_real_choice" in codes

    def test_a_compass_bearing_counts_as_a_bearing(self):
        """The rewritten line gives an absolute direction rather than a
        facing-relative one. Checking only the facing-relative form reported
        every rewritten candidate as defective."""
        codes = {f.code for f in check_observation(REWRITTEN, paris_spec())}
        assert "candidate_without_bearing" not in codes


# ─────────────────────────────────────────────────────────────────────────────
# Parsing
# ─────────────────────────────────────────────────────────────────────────────


class TestParsing:
    def test_both_observation_forms_are_understood(self):
        """The gate must judge the raw observation too, or a regression in the
        vendored rendering would go unseen behind our own rewrite."""
        assert len(parse_marks(RAW_PARIS)) == 1
        assert len(parse_marks(REWRITTEN)) == 2

    def test_the_rewritten_form_yields_absolute_bearings(self):
        marks = {m.index: m for m in parse_marks(REWRITTEN)}
        assert marks[1].bearing_deg == 270.0
        assert marks[2].bearing_deg == 90.0
        assert marks[1].node_id == "dock_275"

    def test_the_vendored_form_yields_relative_directions(self):
        mark = parse_marks(RAW_PARIS)[0]
        assert mark.direction == "backward"
        assert mark.bearing_deg is None

    def test_the_objective_is_readable(self):
        distance, bearing = objective(REWRITTEN)
        assert distance == 444.0
        assert bearing == 258.0

    def test_no_objective_reads_as_none_rather_than_zero(self):
        assert objective(RAW_PARIS) is None


class TestGeometryHelpers:
    def test_angular_gap_wraps(self):
        assert angular_gap(350.0, 10.0) == pytest.approx(20.0)
        assert angular_gap(10.0, 350.0) == pytest.approx(20.0)
        assert angular_gap(0.0, 180.0) == pytest.approx(180.0)

    @pytest.mark.parametrize(
        "bearing,name",
        [(0, "north"), (44, "north-east"), (90, "east"), (180, "south"),
         (270, "west"), (359, "north")],
    )
    def test_compass_names_are_right(self, bearing, name):
        assert compass_of(bearing) == name

    def test_bearing_between_matches_the_axes(self):
        assert bearing_between(0, 0, 100, 0) == pytest.approx(0.0)
        assert bearing_between(0, 0, 0, 100) == pytest.approx(90.0)


# ─────────────────────────────────────────────────────────────────────────────
# The rewrite
# ─────────────────────────────────────────────────────────────────────────────


@needs_spec
class TestRewrite:
    def guidance(self) -> Guidance:
        return Guidance(
            objective="Collect the order from its restaurant.",
            target_distance_m=444.0,
            target_bearing_deg=258.0,
            candidates=[
                Candidate(1, "dock_275", "55 Cherry St", 153.7, 270.0),
                Candidate(2, "int_467", "20 Park St", 20.0, 90.0),
            ],
        )

    def test_the_move_block_is_removed(self):
        out = rewrite_observation(
            RAW_PARIS, self.guidance(), enabled_actions=list(paris_spec().enabled_actions)
        )
        assert "MOVE(direction" not in out
        assert "reachable_waypoints" not in out

    def test_the_stale_routing_block_is_removed(self):
        """It is deleted rather than corrected. Leaving a fixed copy alongside
        the recomputed objective would give the policy two answers, and the
        stale one read 202.2 m for 22 consecutive steps."""
        out = rewrite_observation(
            RAW_PARIS, self.guidance(), enabled_actions=list(paris_spec().enabled_actions)
        )
        assert "202.2" not in out
        assert "next_move" not in out

    def test_the_move_block_survives_on_a_map_that_enables_move(self):
        """The rewrite is driven by the environment's own action space, not by a
        hardcoded assumption that MOVE is always wrong."""
        out = rewrite_observation(
            RAW_PARIS, self.guidance(), enabled_actions=["MOVE", "MOVE_TO"]
        )
        assert "reachable_waypoints" in out

    def test_goal_distance_and_bearing_are_stated(self):
        out = rewrite_observation(
            RAW_PARIS, self.guidance(), enabled_actions=list(paris_spec().enabled_actions)
        )
        assert "444 m away" in out
        assert "west" in out

    def test_the_shortest_path_answer_is_not_given_away(self):
        """A line naming the correct MOVE_TO would make the gate pass and the
        benchmark meaningless: the policy would learn to copy one token."""
        out = rewrite_observation(
            RAW_PARIS, self.guidance(), enabled_actions=list(paris_spec().enabled_actions)
        )
        lowered = out.lower()
        for giveaway in ("shortest path", "recommended", "best move", "you should move"):
            assert giveaway not in lowered

    def test_candidate_numbers_are_the_engines_own(self):
        """Renumbering would make MOVE_TO(k) move somewhere other than the line
        the policy read -- a corruption invisible from outside the runtime."""
        out = rewrite_observation(
            RAW_PARIS, self.guidance(), enabled_actions=list(paris_spec().enabled_actions)
        )
        assert "MOVE_TO(1): 55 Cherry St [dock_275]" in out
        assert "MOVE_TO(2): 20 Park St [int_467]" in out

    def test_candidates_are_capped_and_the_cap_is_disclosed(self):
        guidance = Guidance(
            objective="go", target_distance_m=10.0, target_bearing_deg=0.0,
            candidates=[Candidate(i, f"n_{i}", f"St {i}", float(i), 0.0) for i in range(1, 30)],
        )
        out = rewrite_observation(RAW_PARIS, guidance, enabled_actions=["MOVE_TO"], max_candidates=8)
        assert "(8 of 29 shown)" in out

    def test_no_candidates_is_stated_plainly(self):
        out = rewrite_observation(
            RAW_PARIS, Guidance(objective="go"), enabled_actions=["MOVE_TO"]
        )
        assert "No reachable waypoint from here." in out


# ─────────────────────────────────────────────────────────────────────────────
# The oracle
# ─────────────────────────────────────────────────────────────────────────────


class TestObservationOnlyOracle:
    def test_it_steers_toward_the_goal_bearing(self):
        """Goal bears 258 (west); candidate 1 is west, candidate 2 is east.

        The agent is moved off the pickup address first: REWRITTEN puts it
        standing on it, where collecting the order is the right move and steering
        is not tested at all.
        """
        away = REWRITTEN.replace("you are at 131 Union Ave", "you are at 9 Other Rd")
        oracle = ObservationOnlyOracle(["MOVE_TO", "PICKUP", "WAIT"])
        name, arguments, _reason = oracle.act(away)
        assert name == "MOVE_TO"
        assert arguments["_args"] == [1]

    def test_it_picks_up_when_standing_at_the_pickup_address(self):
        """The observation never says "you are at the restaurant"; it prints two
        addresses and leaves the comparison to the reader. Missing that is why
        the oracle walked to the door and never collected the order."""
        oracle = ObservationOnlyOracle(["MOVE_TO", "PICKUP"])
        name, _arguments, reason = oracle.act(REWRITTEN)
        assert name == "PICKUP", reason

    def test_it_does_not_pick_up_at_the_wrong_address(self):
        elsewhere = REWRITTEN.replace("you are at 131 Union Ave", "you are at 9 Other Rd")
        oracle = ObservationOnlyOracle(["MOVE_TO", "PICKUP"])
        name, _arguments, _reason = oracle.act(elsewhere)
        assert name == "MOVE_TO"

    def test_it_accepts_an_order_when_holding_none(self):
        text = "### active_orders\nYou currently have no accepted orders\n[Order #0]\n  Pickup : X"
        oracle = ObservationOnlyOracle(["VIEW_ORDERS", "ACCEPT_ORDER", "MOVE_TO"])
        name, _arguments, _reason = oracle.act(text)
        assert name == "ACCEPT_ORDER"

    def test_it_refuses_to_reverse_when_there_is_another_option(self):
        """Greedy bearing-following walks into two-cycles: each of two nodes
        looks best seen from the other, and the agent oscillates until the
        budget runs out."""
        oracle = ObservationOnlyOracle(["MOVE_TO"])
        oracle.act(REWRITTEN)  # commits to dock_275
        # Now dock_275 is behind; both candidates still point west-ish.
        text = REWRITTEN.replace(
            "- MOVE_TO(2): 20 Park St [int_467], 20 m to the east",
            "- MOVE_TO(2): 20 Park St [int_467], 20 m to the west",
        ).replace("you are at 131 Union Ave", "you are at 55 Cherry St")
        _name, arguments, _reason = oracle.act(text)
        assert arguments["_args"] == [2], "should not step straight back to dock_275"

    def test_it_will_reverse_at_a_dead_end(self):
        """Forbidding the reversal unconditionally would strand the agent."""
        oracle = ObservationOnlyOracle(["MOVE_TO"])
        oracle.act(REWRITTEN)
        only_back = (
            REWRITTEN.split("### waypoint_marks")[0]
            + "### waypoint_marks (1 of 1 shown)\n"
            + "- MOVE_TO(1): 55 Cherry St [dock_275], 154 m to the west\n"
        ).replace("you are at 131 Union Ave", "you are at 9 Other Rd")
        name, arguments, _reason = oracle.act(only_back)
        assert name == "MOVE_TO" and arguments["_args"] == [1]

    def test_it_uses_no_privileged_state(self):
        """Its only input is the string. If it ever grew a runtime handle it
        could navigate an observation that tells a real policy nothing, and the
        gate would certify an unusable input."""
        import inspect

        source = inspect.getsource(ObservationOnlyOracle)
        for forbidden in ("_dm(", "city_map", "waypoint_graph", "shortest_path", "runtime"):
            assert forbidden not in source, f"oracle reached for {forbidden}"


@pytest.mark.skipif(not VENDOR.exists(), reason="vendored VAGEN checkout not present")
@needs_spec
class TestGateEndToEnd:
    def test_the_rewritten_input_lets_the_oracle_reach_and_work_the_pickup(self):
        """The claim #12 rests on, run against the live engine.

        Before the rewrite the oracle made no progress at all: the frozen
        distance held at 202.2 m while it oscillated between two nodes. After it,
        the agent closes 444 m, arrives, and collects the order without a single
        rejected action.
        """
        from embodiedbench.tasks.input_gate_run import run_gate

        report = run_gate(map_name="citycore-paris", env_spec=paris_spec(), max_steps=200)
        assert not report.blocking, [f.to_dict() for f in report.blocking]
        assert report.rejected_actions == 0
        assert report.made_progress
        assert report.distance_start_m > 400
        assert report.distance_end_m < 120
        actions = {step.action.split("{")[0] for step in report.steps}
        assert "PICKUP" in actions, "the oracle never collected the order"
