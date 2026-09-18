"""An observation-only courier, used to measure whether the input is solvable.

The rule is the same as before and it is the whole point: this policy may read
only what the environment says to it. It never touches the road network, never
asks for a route, never sees a coordinate. If it can deliver, the observation
carries enough for a model to deliver. If it cannot, no amount of model quality
will help, and the environment is what needs fixing.

It navigates the way a courier without satnav does, and nothing cleverer:

  1. ask the phone which street the address is on and roughly which way
  2. read the sign to learn which street it is standing on
  3. if those match, walk along the street watching the phone's range, and
     do not take a turning that makes it worse
  4. if they do not, take the street heading nearest the phone's direction,
     preferring one it has not already walked
  5. if it is getting further away than it ever was, go back to the junction
     where it was closest and try a different street
  6. when the phone says the address is close, or the number on the slip is
     on a door in front of it, try the door

The door numbers are read (step 6) but do not steer. An earlier version of
this courier steered along the target street by them, on the reasoning that
numbers are monotone along a real street; on this map's doors they are not
(one junction reads "8, 11, 13", the next nothing, the next "9", the next
"6"), and when the repaired reader made that rule fire for the first time
the courier turned round at every parity change and lost two easy shifts
it had cleared. The phone's range, checked after every walk, is the signal
that is monotone along the right street, and it is what the released
numbers were measured with.

Step 5 is what stops a wrong turn from being permanent. The phone's bearing is
a pin, not a direction of travel -- straight-line and route heading disagree by
more than 90 degrees on 22.7% of lookups -- so a courier steering by it will
sometimes set off into a loop or a dead end. Without a way back that was the end
of the episode: every easy-tier failure measured was one wrong turn out of a
pickup followed by 250 m in the wrong direction and no recovery. Retracing to
the closest junction it has seen takes TRIPLE from 88% to 92% and SHIFT from 55%
to 64%, and it uses nothing the observation does not already say.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from embodiedbench.runtime.city.courier_env import (
    COMPASS,
    CourierEnv,
    bearing_deg,
    compass_of,
)

# Two fields, read separately, because they are measured in two frames: the
# distance is along the road and the bearing is as the crow flies. Reading them
# out of one pattern is what let a wording change take the whole phone away.
_MAP_ANSWER = re.compile(r"is on (.+?), about (\d+) m away")
_MAP_HEADING = re.compile(r"lies to the (\S+?) of you")
_ADDRESS = re.compile(r"^(\d+)\s+(.*)$")
_ON_STREET = re.compile(r"You are on ([^,.]+)")
# The doors readable from where the courier stands: "outside number 3",
# "outside numbers 28, 30, 35". An earlier pattern knew only the one-door
# and the "6-8" forms, so on the multi-door line -- most junctions -- the
# courier read no numbers at all and never tried a door it was standing
# at unless the phone said so; the released table was measured that way
# and withdrawn.
_OUTSIDE_NUMBERS = re.compile(r"outside numbers? ([0-9][0-9,\s-]*)")
_NUMBER = re.compile(r"\d+")


def _numbers_in(text: str | None) -> list[int]:
    return [int(n) for n in _NUMBER.findall(text or "")]
# Above this range, cover ground a block at a time; below it, step by step so
# the door is not overshot.
# On the target street, re-check the phone only this often; between checks the
# door numbers carry the progress signal.
MAP_CHECK_EVERY = 4
# Rough metres per house number, for estimating range from numbers alone.
NUMBER_TO_METRES = 8.0


def compass_degrees(name: str) -> float | None:
    try:
        return COMPASS.index(name.strip().lower()) * 45.0
    except ValueError:
        return None


def angular_gap(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


@dataclass
class OracleResult:
    seed: int
    delivered: bool = False
    deliveries: int = 0
    collected: bool = False
    steps: int = 0
    sim_seconds: float = 0.0
    rejected: int = 0
    closest_pickup_m: float = 1e9
    closest_dropoff_m: float = 1e9
    route_optimal_m: float = 0.0
    walked_m: float = 0.0
    trace: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed, "delivered": self.delivered,
            "deliveries": self.deliveries, "collected": self.collected,
            "steps": self.steps, "rejected": self.rejected,
            "sim_minutes": round(self.sim_seconds / 60.0, 1),
            "closest_pickup_m": round(self.closest_pickup_m, 1),
            "closest_dropoff_m": round(self.closest_dropoff_m, 1),
            "walked_m": round(self.walked_m, 1),
            "route_optimal_m": round(self.route_optimal_m, 1),
            "efficiency": (
                round(self.route_optimal_m / self.walked_m, 3) if self.walked_m else 0.0
            ),
        }


class ObservationOnlyCourier:
    """A courier that reads only what it is told."""

    # ── the sighted variant ──────────────────────────────────────────────────
    #
    # Everything above is a policy that reads only *text*, and that is the point
    # of it: if it delivers, the words are sufficient. ``sighted=True`` gives the
    # same policy the two things the words deliberately never say -- what is
    # standing in a street, and what colour its light is -- and nothing else. The
    # gap between the two runs is what the photographs are worth, and it is the
    # measurement the whole benchmark turns on.
    #
    # Both are read privileged rather than from pixels: the frame's *name* for
    # the obstacle, ``light_here`` for the lamp. So these numbers are the value
    # of the information in the photograph with recognition assumed perfect --
    # the ceiling on what looking can buy, not a claim about how hard it is to
    # look. A real vision policy lands between the two runs.
    OBSTACLE_SUFFIXES = {"_road_block.png": "road_block",
                         "_slow_pedestrian.png": "slow_pedestrian"}

    def __init__(self, env: CourierEnv, *, max_steps: int = 120, sighted: bool = False,
                 stop_at_turns: int | None = None):
        self.env = env
        self.max_steps = max_steps
        #: Stop once the world has counted this many turns. A model is scored
        #: on turns and every action is one -- a phone check as much as a
        #: walk -- so a floor run to the same number of turns reads on the
        #: model's scale; ``max_steps`` alone counts only this courier's own
        #: loop, in which one step is a walk and a phone check together.
        self.stop_at_turns = stop_at_turns
        # The address the courier is working towards, so a run that resumes
        # (the reference tool stops at each earnings checkpoint and goes on)
        # does not spend a turn asking the phone what it already knows.
        self._target_text: str | None = None
        # Off by default, because the reference courier's whole job is to show
        # the *text* is sufficient. Turned on, it shows what the text is not.
        self.sighted = sighted
        self.visited: dict[str, int] = {}
        self.target_street: str | None = None
        self.target_number: int | None = None
        self.target_bearing: float | None = None
        self.target_distance_m: float | None = None
        self.last_numbers: tuple[int, int] | None = None
        self.previous_node: str | None = None
        # Moves that made the phone's distance worse. A courier who walks a
        # block, checks the phone and sees the destination is further away turns
        # round and does not try that turning again. Without this the oracle
        # reached the right street on every failing seed and then bounced
        # between two nodes -- one was revisited 121 times.
        self.bad_moves: set[tuple[str, str]] = set()
        # Streets it has walked into and found shut. This courier reads only
        # what the world says to it and the world says nothing about an obstacle
        # until you are standing at one -- so this is a record of bumps, not of
        # perception, and that is exactly what makes it the *blind* baseline
        # against which looking at the photograph is worth something. Without it
        # the policy re-picked the same barrier every turn until the clock ran
        # out, which measures a missing memory rather than a missing eye.
        self.blocked: set[tuple[str, str]] = set()
        # Steps of grace left after finding the way shut. See ``start_detour``.
        self.detour_steps = 0
        self.streets_seen: set[str] = set()
        self.turns_since_map = 0
        # The closest the phone has said this target is, and where the courier
        # was standing when it said so. Together they are the way back.
        self.node_distance: dict[str, float] = {}
        # Where each choice actually led, learned by taking it. Under the block
        # stride the candidate a courier picks and the corner it arrives at are
        # different nodes; see ``distance_behind``.
        self.reached: dict[tuple[str, str], str] = {}
        self.best_distance = 1e9

    # ── reading what it is told ──────────────────────────────────────────────

    def read_slip(self, address: str) -> None:
        """Take the street and number straight off the order slip.

        The address itself names the street -- "1 Rue Saint-Jacques" says which
        street as plainly as any map does. Learning the target street only from
        the phone made the courier helpless the moment the phone was taken away,
        when in fact it had been told the street all along.
        """
        parsed = _ADDRESS.match(str(address).strip())
        if parsed:
            self.target_number = int(parsed.group(1))
            self.target_street = parsed.group(2).strip()

    def _address_position(self, address: str):
        """Where an address actually is. Oracle privilege, used knowingly.

        Resolved through the environment's own lookup rather than by matching
        text, so the courier and the phone agree on which door is meant.
        """
        match = self.env._find_address(address)
        return match.kerb if match is not None else None

    def consult_map(self, address: str) -> None:
        self.read_slip(address)
        outcome = self.env.check_map(address)
        if not outcome.ok:
            # No signal. The street name from the slip still stands; only the
            # bearing and range are lost.
            self.target_bearing = None
            self.target_distance_m = None
            return
        match = _MAP_ANSWER.search(outcome.message)
        if match:
            self.target_street = match.group(1)
            self.target_distance_m = float(match.group(2))
            self.note_distance(self.target_distance_m)
        # The bearing is no longer in the phone's words -- it is drawn on the
        # map, because a direction stated in text made choosing a street a
        # reading exercise and the photographs decoration. This courier is an
        # oracle, not an agent: it is entitled to the geometry directly, and
        # what it establishes is that the world is solvable, which was always
        # the claim. It no longer establishes that the world is solvable *from
        # the text alone*, and that is the point of the change.
        target = self._address_position(address)
        # Quantised to the same eight points the phone used to speak, so this
        # is the identical signal by a different route rather than a sharper
        # one. An exact bearing changes which street wins a near-tie and this
        # courier is a solvability floor, not a competitor.
        self.target_bearing = (
            compass_degrees(compass_of(bearing_deg(self.env.position(), target)))
            if target else None)

    # How much worse than the best range seen counts as "I have gone wrong".
    # A courier who has walked half as far again as their closest approach and
    # is still getting further away does not keep going; they go back to the
    # last place they knew where they were. Without this the courier had no
    # recovery at all: on pair seed 20 it took one wrong turn out of a pickup --
    # the phone's pin lies east, the road east is a loop, the route goes
    # south-west -- and then walked 250 m the wrong way for the rest of the
    # shift, delivering 1 of 2. Every single easy-tier failure was this.
    LOST_FACTOR = 1.5
    LOST_MARGIN_M = 40.0

    def note_distance(self, distance: float) -> None:
        """Remember how close this junction was to the target."""
        node = self.env.node_id
        seen = self.node_distance.get(node)
        if seen is None or distance < seen:
            self.node_distance[node] = distance
        if distance < self.best_distance:
            self.best_distance = distance

    def distance_behind(self, here: str, toward: str) -> float | None:
        """How close the target was, last time this choice was taken.

        The distinction this exists for is the whole of what changes between the
        two strides. A candidate row names *the next waypoint down a street*, and
        under the waypoint stride that is also where the courier ends up, so a
        junction's remembered distance could be looked up by the candidate's own
        node. Under the block stride it is not: one call walks past several
        waypoints to the far end of the block, so a table keyed by the first one
        never matched and both the tabu and the retrace silently stopped firing.
        The reference courier then oscillated between two corners 83 m apart for
        the rest of the shift, on every seed.

        Keying on "where this choice led when I took it" is also the honest
        model. A courier standing on a corner does not know what is at the far
        end of a block they have never walked -- and once they have walked it,
        they do.
        """
        node = self.reached.get((here, toward), toward)
        return self.node_distance.get(node)

    def was_the_way_on(self, row: dict[str, Any], options: list[dict[str, Any]]) -> bool:
        """Would the courier have taken this street if it were open?

        Only then is the barrier on it worth abandoning the range for. The test
        is the courier's own rule read back: the target street if it is standing
        on the wrong one, otherwise the option nearest the phone's bearing.
        """
        if self.target_street is not None and row["street"] == self.target_street:
            return True
        if self.target_bearing is None:
            return len(options) <= 1
        best = min(options, key=lambda r: angular_gap(r["bearing"], self.target_bearing))
        return best["node"] == row["node"]

    def forget_distances(self) -> None:
        self.node_distance = {}
        self.best_distance = 1e9

    # How many moves the courier is allowed to get further away after finding a
    # street shut. A detour round a closed block is 3-6 junctions on this map,
    # and each of them makes the phone's number *worse* -- the phone routes
    # through the barrier, because a map app does not know about it -- so twelve
    # is a detour and a wrong turn out of one, with room to come back.
    DETOUR_GRACE = 12
    # Escalating to the phone after N visits to the same junction was tried as a
    # cure for the one remaining easy-tier livelock (SOLO seed 14, a two-cycle
    # 470 m from the door) and rejected: revisiting a junction four times is also
    # what a courier does while working out which end of a street the numbers
    # run from, so at N=4 it fired constantly and took SOLO from 95% to 85% and
    # PAIR from 98% to 83%. One documented failing seed is cheaper than a rule
    # that misfires on nineteen good ones.

    def start_detour(self) -> None:
        """The way is shut: stop trusting the distances, and allow a way round.

        Both of this policy's recovery rules are built on the phone's range
        shrinking: a move that lengthens it is added to ``bad_moves``, and a
        range half again as bad as the best seen means "lost, go back". Round a
        barrier every single move breaks both rules, because the range is
        measured along a route the courier cannot walk. Left alone the two rules
        fight each other -- retrace to the junction beside the barrier, refuse
        the only ways on because they made things worse, retrace again -- and on
        SOLO seed 3 that turned a 604 m delivery into 2142 m of thrashing and
        the first easy-tier failures the obstacles caused.

        So on learning the street is shut the courier drops what it thought it
        knew about distance -- which was about a road that is not open -- and
        gets a few moves during which getting further away is allowed.
        """
        self.detour_steps = self.DETOUR_GRACE
        self.bad_moves.clear()
        self.forget_distances()

    @property
    def detouring(self) -> bool:
        return self.detour_steps > 0

    @property
    def budget_spent(self) -> bool:
        """The world has counted every turn this run was given. Checked
        before each action, not each step: a step is a walk and a phone
        check, and a model scored to the same budget gets no 61st action."""
        return self.stop_at_turns is not None and self.env.turns >= self.stop_at_turns

    @property
    def lost(self) -> bool:
        if self.detouring:
            return False
        if self.target_distance_m is None or self.best_distance >= 1e9:
            return False
        return self.target_distance_m > self.best_distance * self.LOST_FACTOR + self.LOST_MARGIN_M


    def read_here(self) -> tuple[str | None, tuple[int, ...] | None]:
        """Street and door numbers where the courier stands.

        Taken from the observation, which states both, and no longer from a
        tool. There used to be a read_sign() fallback here, guarded by "if the
        observation did not parse" -- and read_sign answered with the very
        sentence the observation had just failed to parse, at three simulated
        seconds and a turn. The tool has since been retired for exactly that
        reason (see ``RETIRED_TOOLS``), and the fallback with it: a courier does
        not stop to squint at a sign they are already looking at.
        """
        message = self.env.location_text()
        sign = _ON_STREET.search(message)
        numbers = _OUTSIDE_NUMBERS.search(message)
        found = tuple(_numbers_in(numbers.group(1))) if numbers else ()
        return (sign.group(1).strip() if sign else None), (found or None)

    # ── deciding ─────────────────────────────────────────────────────────────

    def seen_obstacle(self, row: dict[str, Any]) -> str | None:
        """What the photograph of this street shows standing in it, if anything.

        Deliberately the only place the policy touches an image at all, and it
        touches the file *name*: the album stores an obstructed approach as
        ``toward_<node>_<kind>.png``, so this is a perfect-recognition stand-in
        for a model looking at the frame. Nothing else about the row changes --
        the text is identical whether or not something is there, which is the
        property the no-leak test asserts.
        """
        path = str(row.get("image") or "")
        for suffix, kind in self.OBSTACLE_SUFFIXES.items():
            if path.endswith(suffix):
                return kind
        return None

    # There is no ``ask_the_route`` any more. The phone used to speak its
    # route ("1. Turn left onto Rue Monge ...") and this courier read the
    # first street off it while detouring round a barrier; the phone now
    # draws the route and says only how far and how long, and a text-only
    # policy may read where a place is (the pin's bearing, above) but not
    # be handed the way there -- ``tests/test_city_pipeline.py`` forbids
    # this class the route calls by name. The regex outlived the words it
    # read and matched nothing, which is one of the two dead rules the
    # released floor table was withdrawn for; now the rule is gone with the
    # words. On a detour the courier steers by the pin's bearing with its
    # range rules suspended (``start_detour``), and nothing else.

    def red_light(self, k: int, toward: str) -> bool:
        """Is the lamp governing this crossing red, and can it be seen?

        Both halves matter. The colour is the fact; ``signal_is_visible`` is the
        promise that a frame exists showing it. Acting on a lamp the album
        cannot show would make this policy's advantage an artefact of privileged
        access rather than a measurement of what the pictures carry.
        """
        env = self.env
        if not env.signal_is_visible(env.node_id, toward):
            return False
        return env.light_here(k) == "red"

    def choose(self, street_here: str | None, numbers_here: tuple[int, ...] | None) -> int | None:
        rows = self.env.candidates()
        # A street it has already found shut is not a candidate. Dropped here,
        # before any rule looks at the list, so no branch below can pick it --
        # the earlier version filtered in three of the five branches and the two
        # it missed were the two that fire on the target street, which is where
        # the barriers matter most.
        here = self.env.node_id
        open_rows = [r for r in rows if (here, r["node"]) not in self.blocked]
        if self.sighted:
            seen_shut = [r for r in open_rows if self.seen_obstacle(r) == "road_block"]
            for row in seen_shut:
                if (here, row["node"]) not in self.blocked:
                    # Seen, so avoided. There is nobody to tell -- the phone
                    # routes on a survey and cannot learn -- so the whole of what
                    # vision buys here is that the courier does not walk into it,
                    # and remembers, and goes round. The route will keep naming
                    # this street every time it is asked; overruling it is the
                    # courier's job.
                    self.blocked.add((here, row["node"]))
                    self.blocked.add((row["node"], here))
                    # A detour is a suspension of the courier's own navigation:
                    # for a dozen moves the range is allowed to get worse,
                    # because the range is measured through a barrier. That is
                    # right when the shut street was the way on, and wrong when
                    # it was merely in sight. At block stride the difference
                    # stopped being academic -- one barrier is visible from
                    # every corner of every block that contains it, so a
                    # courier that detoured on sight was in permanent detour and
                    # spent 2,618 turns on the six seeds it had been doing in
                    # 1,045, without delivering any more.
                    if self.was_the_way_on(row, open_rows):
                        self.start_detour()
            # Precedence, and it has to be this way round. The obvious version
            # -- drop what is visibly shut, and if that empties the list put it
            # back -- picks a barrier the courier is looking at, because the
            # list it falls back to has already had every remembered barrier
            # removed from it. Measured at block stride on TRIPLE that was all
            # 32 remaining collisions: an open street existed every time, and
            # the policy walked into a photographed barrier instead of taking a
            # street it had merely once found shut.
            #
            # A street remembered as shut is a better bet than one that can be
            # seen to be shut, so the fallback goes there first and only then to
            # everything.
            visible = [r for r in open_rows if r not in seen_shut]
            remembered = [r for r in rows if self.seen_obstacle(r) != "road_block"]
            open_rows = visible or remembered or open_rows
        rows = open_rows or rows
        if not rows:
            return None

        # On a detour the range is lying (it is measured through the barrier),
        # so the rules that read it -- the tabu and the retrace -- are
        # suspended by ``start_detour`` and the bearing rule below steers.

        # Lost, and there is a junction behind that was closer: go back to it.
        # This is checked before anything else because every other rule here is
        # a way of making progress, and a courier who has gone wrong does not
        # need a better way of making progress -- it needs to stop.
        if self.lost:
            known = [(distance, r) for r in rows
                     for distance in (self.distance_behind(here, r["node"]),)
                     if distance is not None]
            if known:
                closest = min(known, key=lambda pair: pair[0])
                if closest[0] < (self.target_distance_m or 1e9):
                    return closest[1]["k"]

        on_target_street = (
            street_here is not None
            and self.target_street is not None
            and street_here == self.target_street
        )

        # On the right street the phone's range decides which way (the
        # worsening-move tabu below and the retrace above); the door numbers
        # are read for the door, not for steering -- see the module
        # docstring for the shift they lost when they steered.

        # Not on the target street yet: the first thing a courier does at a
        # junction is read the street names and look for the one they want. The
        # candidate list carries them, so check before falling back to the
        # compass. Omitting this was the single biggest gap -- the oracle walked
        # past the street it was looking for because it was only comparing
        # angles.
        if not on_target_street and self.target_street is not None:
            onto = [r for r in rows if r["street"] == self.target_street]
            if onto:
                onto.sort(key=lambda r: (self.visited.get(r["node"], 0), r["distance_m"]))
                return onto[0]["k"]

        # No bearing: explore. Prefer somewhere unvisited, then a street whose
        # name is new, and only then repeat -- which is how a courier without a
        # phone searches an unfamiliar district. Freezing on rows[0] here was
        # what turned "no signal" into a 0% condition.
        if self.target_bearing is None:
            here = self.env.node_id
            options = [r for r in rows if (here, r["node"]) not in self.bad_moves] or rows
            forward = [r for r in options if r["node"] != self.previous_node]
            options = forward or options
            options.sort(key=lambda r: (
                self.visited.get(r["node"], 0),
                0 if r["street"] not in self.streets_seen else 1,
                r["distance_m"],
            ))
            self.streets_seen.add(options[0]["street"])
            return options[0]["k"]
        here = self.env.node_id
        options = [r for r in rows if (here, r["node"]) not in self.bad_moves] or rows
        forward = [r for r in options if r["node"] != self.previous_node]
        options = forward or options
        options.sort(key=lambda r: (
            self.visited.get(r["node"], 0),
            round(angular_gap(r["bearing"], self.target_bearing) / 45.0),
            r["distance_m"],
        ))
        return options[0]["k"]

    # ── the run ──────────────────────────────────────────────────────────────

    def run(self, seed: int) -> OracleResult:
        env = self.env
        result = OracleResult(seed=seed)
        order = env.active_order()
        if order is None:
            result.trace.append("no order was generated")
            return result
        target_text = order.target.text
        if target_text != self._target_text:
            if self.budget_spent:
                return result
            self.consult_map(target_text)
            self.visited[env.node_id] = 1
            self._target_text = target_text

        for step in range(self.max_steps):
            result.steps = step + 1
            order = env.active_order()
            if order is None or env.shift_over:
                break
            if self.budget_spent:
                break
            if order.target.text != target_text:
                target_text = order.target.text
                self._target_text = target_text
                # A new door means the whole record of what was close is about a
                # different place. Keeping it would have the courier "retrace"
                # toward a junction that was near the last address.
                self.forget_distances()
                self.consult_map(target_text)

            street_here, numbers_here = self.read_here()
            distance = self.target_distance_m if self.target_distance_m is not None else 1e9
            if order.picked_up:
                result.closest_dropoff_m = min(result.closest_dropoff_m, distance)
            else:
                result.closest_pickup_m = min(result.closest_pickup_m, distance)

            # Close enough to try the door. With a phone that is a range; with
            # or without one, the slip's number on a door in front of it --
            # exactly the cue a person uses. (A bracket round the readable
            # numbers was tried and tries doors it is not at: the line reads
            # "28, 30, 35" at one junction, and 32 is a walk away.)
            on_street = street_here is not None and street_here == self.target_street
            door_here = (
                numbers_here is not None and self.target_number is not None
                and self.target_number in numbers_here
            )
            if distance <= 10.0 or (on_street and door_here):
                outcome = env.hand_over() if order.picked_up else env.collect()
                result.trace.append(f"{step}: try door -> {outcome.message[:70]}")
                if outcome.ok:
                    if order.picked_up and order.delivered:
                        # One job done; a shift is a queue, so take the next.
                        # Returning here made every episode a single delivery and
                        # reported 1/10 on a ten-order shift.
                        result.delivered = True
                        result.deliveries += 1
                        result.sim_seconds = env.sim_seconds
                        self.visited.clear()
                        self.bad_moves.clear()
                        self.previous_node = None
                        self.forget_distances()
                    else:
                        result.collected = True
                        self.visited.clear()
                        self.previous_node = None
                        self.forget_distances()
                    # Re-read the job from the world rather than assuming which
                    # one it is. With a live queue the courier may have just
                    # served an order it was passing rather than the one it was
                    # sent to, and an order it was carrying may have expired
                    # while it walked -- so ``active_order`` can be a different
                    # job, or none at all.
                    nxt = env.active_order()
                    if nxt is None or env.shift_over:
                        result.sim_seconds = env.sim_seconds
                        return result
                    target_text = nxt.target.text
                    if self.budget_spent:
                        # the next job's phone check would be the action
                        # after the last one this run was given
                        result.sim_seconds = env.sim_seconds
                        return result
                    self._target_text = target_text
                    self.consult_map(target_text)
                    continue
                result.rejected += 1

            choice = self.choose(street_here, numbers_here)
            if self.budget_spent:
                break
            if choice is None:
                result.trace.append(f"{step}: nowhere to go from {env.node_id}")
                break
            before = env.node_id
            before_distance = self.target_distance_m
            # Spent, and there is a rest available: take it. A minute buys the
            # tank back and the courier walks the rest of the shift at full
            # speed, so on anything longer than a short tier this is strictly
            # cheaper than dragging along at the tired fraction. A floor policy
            # that ignored a tool it has been given would understate the floor.
            if getattr(env, "tired", False) and "rest" in env.allowed_tool_names():
                env.rest()
                continue
            toward = next((r["node"] for r in env.candidates() if r["k"] == choice), None)
            if self.sighted and toward is not None and self.red_light(choice, toward):
                # Read the lamp, see out the phase. One wait always changes the
                # light, so this is a turn and 0-60 s against a 45 s charge and
                # the same crossing a moment later -- which is the trade the
                # penalty exists to create, and the reason a text-only courier
                # loses whole shifts to it.
                env.wait()
                continue
            # Take the whole block when the chosen street is the one already
            # underfoot. A leg is a median 530 m over 18 m edges, so stepping
            # junction by junction costs ~35 turns an order and ~351 a shift
            # against a 120-turn budget -- which is why this courier delivered 3
            # of 10 while spending its shift walking one edge at a time.
            # Deliberately NOT follow_street. Covering a block per turn sounds
            # like the fix for a 351-call leg, but this courier steers by the
            # distance the phone reports after each move, and committing to
            # several junctions blind throws that feedback away: measured 3.12
            # -> 0.00 deliveries, and still only 1.33 with the macro gated to
            # long range. Turn count is not this policy's binding constraint.
            outcome = env.walk_to(*env.street_at(choice))
            if not outcome.ok:
                result.rejected += 1
                if outcome.code == "way_blocked" and toward is not None:
                    # Learned the hard way, and remembered in both directions:
                    # a barrier that stopped you walking north will stop you
                    # walking back south through it too.
                    self.blocked.add((before, toward))
                    self.blocked.add((toward, before))
                    self.start_detour()
                result.trace.append(f"{step}: {outcome.message[:70]}")
                continue
            result.walked_m += outcome.walked_m
            if toward is not None:
                self.reached[(before, toward)] = env.node_id
            self.previous_node = before
            self.visited[env.node_id] = self.visited.get(env.node_id, 0) + 1
            if self.detour_steps:
                self.detour_steps -= 1

            # One phone check per turn, after moving. Skipping it on the target
            # street and substituting a range estimated from door numbers was
            # tried and is far worse -- 5.50 -> 0.75 deliveries. The estimate is
            # not on the same scale as the real range, so the arrival test and
            # the worsening-move tabu, which both compare against it, stopped
            # meaning anything. Consulting costs simulated time and that cost is
            # real; the fix is a policy that needs fewer checks, not a policy
            # that invents the number.
            if self.budget_spent:
                break
            self.consult_map(target_text)
            after = self.target_distance_m
            if (not self.detouring and before_distance is not None and after is not None
                    and after > before_distance + 2.0):
                # Keyed on the choice, not on where it landed -- the filters in
                # ``choose`` see candidates, and a block-strided move does not
                # end at the candidate it names.
                self.bad_moves.add((before, toward if toward is not None else env.node_id))

        result.sim_seconds = env.sim_seconds
        return result


def score_run(env: CourierEnv, result: OracleResult, start_node: str) -> OracleResult:
    """Fill in the privileged measurements, after the fact.

    Deliberately not a method on the courier. Route optimality needs the graph,
    and a policy that can reach the graph to *report* on itself is one edit away
    from reaching it to *decide*. Keeping the measurement out here means the
    restriction the reference courier exists to demonstrate is enforced by
    structure rather than by discipline.
    """
    order = env.orders[0] if env.orders else None
    if order is None:
        return result
    legs = (
        (env.route_length_cm(start_node, order.pickup.kerb_node) or 0.0)
        + (env.route_length_cm(order.pickup.kerb_node, order.dropoff.kerb_node) or 0.0)
    )
    result.route_optimal_m = legs / 100.0
    return result


def run_reference_courier(
    env: CourierEnv, seed: int, *, max_steps: int = 250, sighted: bool = False
) -> OracleResult:
    """Run the reference courier and score it. The usual entry point."""
    start_node = env.node_id
    result = ObservationOnlyCourier(env, max_steps=max_steps, sighted=sighted).run(seed)
    return score_run(env, result, start_node)
