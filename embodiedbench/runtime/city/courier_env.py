"""A delivery runtime built on the compiled city, not on the engine's graph.

Every blocker the independent evaluation found traces to the same thing: the
simulation and the description of it were built from different data. The agent
was told a distance to one point and scored at another 12 m away. It was told a
bearing in one angle convention and shown a photograph in a second. It was
offered waypoints on a graph whose edges missed the road by a median of 11 m.
Patching those one at a time keeps the two descriptions in sync only until the
next one drifts.

So this runtime has exactly one source of truth -- the ``RoadNetwork`` the
conversion layer compiles -- and everything the agent reads is derived from it:

  where it stands        a node on the carriageway, on a named street
  where it can go        that node's neighbours, numbered
  the picture it sees    the street-view frame baked for that exact (node, neighbour)
  the address it wants   a door derived from a real building footprint
  the distance quoted    to the node the success check uses, not near it
  the bearing quoted     one convention, shared with the renderer

Because there is one source, a claim in the text cannot disagree with the world:
the number the agent is told to walk to is the number the arrival check tests.
That is also what makes the task deployable -- "walk to 42 Rue de Rivoli and
ring the bell" is an instruction a real courier robot could be given, and the
success condition is standing at a door, not being within tolerance of an
abstract point.

The vendored engine remains the reference implementation for order economics and
is not used here; the two are compared in the benchmark rather than stacked.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from embodiedbench.compiler.road_network import (
    Address,
    RoadNetwork,
    bearing_deg,
    build_road_network,
)
from embodiedbench.runtime.city.embodiment import Embodiment, Viewpoint
from embodiedbench.runtime.city.embodiment import get as embodiment_for
from embodiedbench.runtime.city.map_image import MapDrawing, render_map
from embodiedbench.runtime.city.street_names import (
    StreetAmbiguous,
    StreetNotHere,
    match_street,
    resolve_relative,
)
from embodiedbench.runtime.city.obstacles import (
    BLOCKED_SECONDS,
    ROAD_BLOCK,
    SLOW_PEDESTRIAN,
    ObstacleField,
    load_visibility as load_obstacle_visibility,
    obstacle_sites,
    site_key as obstacle_site_key,
)

COMPASS = ("north", "north-east", "east", "south-east",
           "south", "south-west", "west", "north-west")
# How a street leaving this junction lies relative to the way the courier is
# already facing. A compass bearing is what a map gives; "on your left" is what
# a person standing on the corner uses, and a courier following spoken
# directions ("turn left onto Rue X") needs the second to act on the first.
# Indexed the same way as COMPASS: 45-degree sectors, clockwise from straight on.
RELATIVE = ("straight ahead", "half right", "on your right", "sharp right",
            "behind you", "sharp left", "on your left", "half left")


class Difficulty:
    """How many jobs are live at once is the difficulty axis. The clock is not.

    The previous ladder claimed order count was the axis and gave each tier its
    own clock multiple -- 6.0, 4.5, 2.6, 1.9 -- and the measurement says the
    claim was false. Holding the multiple fixed and varying only the order
    count, the reference courier scores (25 seeds, delivered/issued):

        multiple   solo    pair   triple   shift
          1.9      52%     58%     63%      71%
          2.6      64%     78%     79%      92%
          4.5     100%     98%     99%     100%

    Order count does not make the task harder. It makes it *easier*, because a
    per-order clock averages over more orders and one bad leg stops deciding the
    episode. Every step of the published 100 / 100 / 79 / 72 ladder came from
    the hand-set multiples and nothing else -- a stopwatch wearing the costume
    of a task demand, which is exactly what the ladder was not supposed to be.

    What genuinely gets harder as a shift lengthens is holding several jobs at
    once. A courier with three live orders cannot walk them one at a time: their
    windows run concurrently, so serving them in the order they arrived means
    the last one is cold before it is collected. The work is to *sequence* --
    to batch stops that lie near each other, and to give up the order the
    dispatcher handed you. That is a real demand and it scales smoothly. With
    one uniform clock and only the depth varying (30 seeds, delivered/issued and
    on-time/issued):

        tier      orders  depth   reference        queue-aware optimum
        solo         1      1     100%   67% ot    100%   100% ot
        pair         2      1      98%   64% ot    100%   100% ot
        triple       3      2      92%   22% ot    100%    84% ot
        shift       10      3      64%    7% ot     97%    55% ot

    Every rung stays solvable -- perfect routing finishes 97-100% of all of them
    -- and every rung leaves room above the reference courier, in deliveries and
    in punctuality both. Raising the shift clock by any amount changes none of
    these numbers, which is the property the old ladder did not have.

    ``ENDLESS`` is the autonomous-courier tier: a fixed hour, a queue that never
    empties because orders are generated on demand rather than drawn from a
    finite list, and profit as the score. There is no order to run out of and no
    ratio to saturate, so a better policy always shows up as more money -- the
    reference courier takes 12.00 an hour and a perfect router 50.88, and
    doubling the hour doubles both.
    """

    SOLO = "solo"
    PAIR = "pair"
    TRIPLE = "triple"
    SHIFT = "shift"
    ENDLESS = "endless"
    ALL = (SOLO, PAIR, TRIPLE, SHIFT, ENDLESS)

    # tier: (orders in the shift, how many may be live at once)
    #
    # ``0`` orders means unbounded: the dispatcher keeps handing out work for as
    # long as the clock runs. Only ENDLESS uses it.
    #
    # The lower rungs are deliberately shallow: a tier the competent fail is a
    # broken tier, and solo/pair exist to prove an agent can read an address,
    # find a street and recognise a door at all. Depth rises from TRIPLE, so
    # what separates policies is scheduling rather than luck.
    SPEC: dict[str, tuple[int, int]] = {
        SOLO: (1, 1),
        PAIR: (2, 1),
        TRIPLE: (3, 2),
        SHIFT: (10, 3),
        ENDLESS: (0, 3),
    }
    ENDLESS_SECONDS = 3600.0

    @classmethod
    def order_count(cls, tier: str) -> int:
        return cls.spec(tier)[0]

    @classmethod
    def queue_depth(cls, tier: str) -> int:
        return cls.spec(tier)[1]

    @classmethod
    def spec(cls, tier: str) -> tuple[int, int]:
        if tier not in cls.SPEC:
            raise ValueError(f"unknown difficulty {tier!r}; expected {cls.ALL}")
        return cls.SPEC[tier]


class Condition:
    """How much the world tells the agent, as a difficulty ladder.

    The three rungs are not arbitrary knobs: they remove, in order, the crutches
    that let a text-only policy succeed without looking at anything.

    ``full``     signs, house numbers and a phone that gives bearing and range.
                 The reference courier solves this from text alone, which is why
                 it establishes a solvability floor and nothing more.
    ``no_phone`` the phone is gone. The agent must read street signs, remember
                 where it has been, and follow house numbers -- the way a courier
                 works in a district they do not know.
    ``visual``   **declared, not validated -- do not report scores on it.**
                 The intent was to move house numbers out of the text so the
                 photograph carried the only copy. Inspecting the renders killed
                 that: the CityCore facades have no legible door numbers, so the
                 condition is not vision-dependent, it is impossible. The
                 reference courier scores 0/20, which measures the absence of
                 information rather than the absence of perception.

                 What the renders *do* distinguish is storefronts -- the map
                 carries 18 restaurants and 11 stores with a ``poi_type``, and
                 their shopfronts are visible and signed. A sound visual rung
                 would ask the courier to find the restaurant by its frontage.
                 That needs a vision policy to establish it is solvable, so it
                 stays unvalidated until one has, rather than being shipped as a
                 hard condition and quietly inflating the difficulty range.

    Stating them this way keeps the benchmark honest about what it is measuring:
    a score on ``full`` is a claim about planning, a score on ``no_phone`` is a
    claim about search and memory, and no rung yet supports a claim about
    perception.
    """

    FULL = "full"
    NO_PHONE = "no_phone"
    VISUAL = "visual"
    ALL = (FULL, NO_PHONE, VISUAL)
    # The rungs whose solvability has been demonstrated. A benchmark run should
    # report these; VISUAL is present so the work is not lost, not so it can be
    # scored.
    # Round-2 measurement put the reference courier at 1/10 on NO_PHONE, not the
    # 30% an earlier and looser floor accepted. A rung nobody has solved cannot
    # be reported, so it joins VISUAL as declared-but-unvalidated until a policy
    # clears it.
    VALIDATED = (FULL,)


class Stride:
    """How far one ``walk_to`` carries. The same city at two resolutions.

    The compiled carriageway carries a waypoint every 18 m, which is the right
    spacing for a camera and the wrong one for a decision. Measured over 25
    recorded runs, a single delivery cost the reference courier between 75 and
    104 turns; 47% of them were ``walk_to`` calls pressing the same button down
    the same street, and 49% were phone lookups in between. A courier standing on
    a corner does not make eighteen decisions to walk one block. They make one.

    ``waypoint``  one call, one waypoint. Nothing between the courier and the
                  ground: every 18 m is a place to stop, look and change its
                  mind. This is what the trajectories, the albums and every
                  measurement so far were taken at, and it stays the default.
    ``block``     one call, one block -- along the named street until a choice
                  exists: a side turning, a fork, a dead end, a crossing with a
                  light the courier can read, a door it was sent to, or something
                  in the way. About 12 to 25 turns per delivery instead of 90.

    Neither is a simplification of the world. The same metres are walked, the
    same seconds spent, the same photographs shown, the same lights obeyed and
    the same obstacles hit. What differs is how often the courier is asked. They
    measure different things, which is why both are kept: the waypoint stride
    asks whether a policy can follow a street, the block stride whether it can
    plan a route, and running only the first is how a navigation benchmark ends
    up mostly measuring patience.
    """

    WAYPOINT = "waypoint"
    BLOCK = "block"
    ALL = (WAYPOINT, BLOCK)


# Standing this close to a door counts as being at it. A doorway is about a
# metre wide and a courier stops on the pavement outside, so a few metres is
# generous without being meaningless. It is quoted to the agent, not hidden.
ARRIVAL_TOLERANCE_CM = 800.0
WALK_SPEED_CM_S = 140.0          # a brisk walk, 1.4 m/s
# Signal timing follows the vendored DeliveryBench convention exactly
# (vlm_delivery/utils/traffic_lights.py signal_state_for_axis): the phase is one
# minute, and on odd minutes the south-north axis is red while east-west is
# green. Matching it rather than inventing a period is what lets the light
# frames the procgen maps already carry -- baked as yaw_000_red.png /
# yaw_000_green.png -- be read by this runtime unchanged, and keeps a score here
# comparable with one from the reference environment.
SIGNAL_PHASE_S = 60.0
# A light governs a crossing within this radius (their DEFAULT_CONTROL_RADIUS_CM).
SIGNAL_CONTROL_RADIUS_CM = 650.0
# Crossing against the light costs time as well as reward. Time is the honest
# unit -- the vendored default is 15 s -- because it makes the violation trade
# against the deadline the way it does for a real courier, instead of being a
# flat fee a policy can ignore.
#
# 15 s was the wrong number, and the direction of the error matters: the phase is
# 60 s, so waiting out a red costs a uniform 0-60 s, mean 30 s. At 15 s crossing
# on red was *strictly cheaper on the clock than obeying the light*, every time,
# and the only thing arguing the other way was a reward term ``summary()`` did
# not even report. A sighted policy that used its eyes was slower than a blind
# one. The charge has to exceed the expected wait or looking cannot pay.
#
# 45 s was still not enough, and the tier that showed it is the one the whole
# design points at. Crossing on red is only taken half the time -- the light is
# green the other half -- so the expected charge is half the penalty, while
# obeying the light costs the expected wait (half of the 60 s phase, so 30 s,
# halved again because half the crossings are green: 15 s) *plus a turn*. At 45 s
# the two are 22.5 s against 15 s and a turn, which is a rounding error, and on
# ENDLESS -- where the score is money against a fixed hour, so a turn spent
# waiting is a delivery not made -- the courier that read every lamp earned
# 4.75/h against 5.17/h for the one that read none. The mechanic the photographs
# exist for was actively unprofitable.
#
# Swept on 20 seeds, blind reference against the same policy allowed to read the
# lamp (delivered %, and profit per hour on ENDLESS):
#
#     penalty   SOLO          PAIR          ENDLESS
#       45 s    85 / 95       90 / 97.5     5.17 / 4.75   looking loses
#       75 s    80 / 95       85 / 97.5     4.15 / 4.75   looking wins everywhere
#      105 s    80 / 95       85 / 97.5     3.22 / 4.75
#
# 75 s is where the sign flips and the sighted courier is unchanged by it -- 95%,
# 97.5%, 4.75/h at every setting, because it never pays the charge. Only the
# courier that does not look is worse off, which is what a penalty for not
# looking is supposed to mean.
RED_CROSSING_PENALTY_S = 75.0
RED_CROSSING_PENALTY = 1.0
# Charging for the light requires the light to be *visible*. The album currently
# bakes one static frame per approach, so no frame shows the live phase, and the
# colour appears in no text either. Penalising it anyway made the score
# anti-correlated with success: the reference courier delivers 10/10 and scores
# -4.4 to -16.4, because 17 unavoidable violations at -1.0 swamp +1.6 of
# delivery credit. A metric that punishes information the environment withholds
# measures nothing, so the charge is off until time-matched red/green frames are
# served -- at which point this becomes True and the penalty is earned.
SIGNAL_FRAMES_AVAILABLE = False
# The same argument, one mechanic over: an obstacle is charged for exactly where
# the album can show it. ``obstacles.py`` holds the placement rule and the gate;
# what lives here is only how the runtime spends the two currencies on it.
OBSTACLE_FRAMES_AVAILABLE = False
# The sidecar that says *which* approaches actually show a lamp.
#
# Having a signal album is not the same as being able to see the light, and
# treating them as the same brought back the exact defect the flag above was
# added to kill. The Paris bake renders 347 signalised approaches in both phases,
# but the camera looks along the street the courier is about to take and the
# lamp is behind or beside it on most corners: comparing each red/green pair for
# lamp-coloured pixels finds a switching lamp on 55 of 347 approaches (15.9%),
# and a blind visual sample of twelve approaches found the light readable in at
# most two. Charging on all 347 is charging for an invisible light again, one
# layer down -- an oracle courier takes 43 violations a shift on routes where it
# could have seen and avoided six.
#
# So the album must *declare* what it shows. ``signal_visibility.json`` lists the
# approaches whose two frames differ in a lamp; anything absent from that list is
# an approach where the courier cannot see the light, and is not charged.
SIGNAL_VISIBILITY_FILE = "signal_visibility.json"
# A door number is legible from about this far along the pavement. Beyond it the
# courier is being told about a building it cannot see.
# A number is only worth printing if standing here counts as arriving. Reading
# doors 3x further than the arrival tolerance produced 76.2% false arrivals and a
# reproducible 14-turn livelock: the agent was told "numbers 6-8" and collect()
# refused every turn because door 7 was 18 m off. Tied to the tolerance so the
# two cannot drift apart again.
READABLE_NUMBER_CM = ARRIVAL_TOLERANCE_CM
# A delivery leg. 250 m each way is about four minutes' walking plus handling,
# so a ten-order shift fits an hour with room to make mistakes.
MAX_ORDER_WALK_CM = 25000.0
# ...except it did not, because only the *delivery* leg was bounded. The walk
# from the last drop-off to the next pickup was drawn without any limit and came
# out at a median 320 m and a maximum 1148 m, so the real leg -- approach plus
# delivery -- ran to a median 530 m. A shortest-path courier with perfect
# knowledge, no wrong turns and no time spent looking needed a mean of 76.6
# minutes (60.9-89.3 across seeds 0-19) to work ten orders, and delivered 6.95 of
# them inside the hour. Ten out of ten was not merely hard, it was arithmetically
# impossible on every seed, which makes ``delivered / 10`` a metric no policy can
# score well on and no policy can be compared by.
MAX_APPROACH_WALK_CM = MAX_ORDER_WALK_CM
# How much of the shift the generated orders are allowed to consume at optimal
# play, when a clock is *imposed from outside*. Below 1.0 so a competent courier
# finishes the list and a wandering one does not.
#
# It applies to nothing on the difficulty ladder, and saying so is the point:
# the tiers derive their clock from the list they drew, so cutting the list to
# fit that clock is circular. It silently did nothing on every tier anyway --
# ``_make_orders`` read ``self.shift_seconds`` before ``reset`` had computed it,
# so the budget was always ``None``. A constant whose effect is zero is worse
# than no constant, because the comment above it reads as a guarantee.
SHIFT_FILL = 0.85
# An order is never allowed to be a formality. Only the *delivery* leg had a
# floor, so 2% of chained orders had the next pickup at the very node the last
# drop-off used -- an approach of 0 m, an order that begins with the courier
# already standing at the door.
MIN_APPROACH_WALK_CM = 3000.0
# Two ends of every job, thirty seconds each: the time a courier spends at a
# door that is not spent walking. Named because the clock is derived from it and
# a number that sets the clock should not be a literal buried in three places.
HANDLING_SECONDS = 30.0
# ── optional constraints, every one behind a flag that defaults to the ──────
# ── benchmark as it is. A flag that is off must not move a single number. ───
#
# Fee jitter: the same route pays a little more or less, order to order, drawn
# from its own RNG stream so the order geometry is untouched. ±20% keeps the
# fee-per-metre band fee_for() argues for while giving a dispatcher's pricing
# the day-to-day variance real platforms have.
FEE_JITTER_FRACTION = 0.2
# Food cools from the moment it is collected. Eight minutes is between the
# optimal delivery leg (about 3-4 minutes walking on the longest orders) and a
# sloppy one, so a courier that dawdles between door and door pays for it and
# one that walks straight there never does.
FOOD_WARM_SECONDS = 480.0
# A cold delivery is still a delivery, like a late one: it pays less, not
# nothing. Multiplies with the late discount -- late AND cold is both.
COLD_FEE_FRACTION = 0.7
# Special notes, each with a real mechanical effect at the door. Leaving at
# the door skips the wait for the customer; ringing first adds one. Half of
# orders carry no note, so reading the slip is informative, not ritual.
NOTE_LEAVE_AT_DOOR = "Leave it at the door."
NOTE_RING_FIRST = "Ring when you arrive; the customer is slow to the door."
ORDER_NOTES = ("", "", NOTE_LEAVE_AT_DOOR, NOTE_RING_FIRST)
NOTE_DOOR_HANDLING_S = 15.0
NOTE_RING_EXTRA_S = 20.0
# Food categories: what is in the bag decides which clock it is on. A hot
# meal cools (the warm window above); ice cream melts on a shorter fuse and
# discounts harder; groceries do not care. Half of orders are meals so the
# temperature mechanic stays the common case, and the category is printed on
# the slip -- reading it is what the flag pays for.
ORDER_CATEGORIES = ("hot meal", "hot meal", "ice cream", "groceries")
ICECREAM_MELT_SECONDS = 300.0
MELT_FEE_FRACTION = 0.6
# The phone's battery. Requesting a route costs a real look at the screen;
# the live map sips charge every block walked with it lit. 4% a route and
# 0.5% a block means a courier that re-routes constantly kills the phone
# mid-shift and walks the rest by street names alone -- the map goes dark and
# navigate() is refused. There is no charger on shift.
PHONE_BATTERY_NAVIGATE_PCT = 4.0
PHONE_BATTERY_SCREEN_PCT = 0.5
# A power bank, if the shift carries one: ninety seconds standing still buys
# forty points of charge. That prices a route request at about nine seconds
# of charging time -- cheap enough to save a dead-phone shift, expensive
# enough that burning charge still loses real clock.
PHONE_RECHARGE_SECONDS = 90.0
PHONE_RECHARGE_PCT = 40.0
# One clock rule for the whole ladder: the time a courier who never puts a foot
# wrong would need for the whole list, times this.
#
# It is uniform on purpose. Per-tier multiples (6.0 / 4.5 / 2.6 / 1.9) were the
# entire published difficulty ladder -- see ``Difficulty`` -- and a stopwatch is
# not a task demand. With one multiple the tiers become genuinely comparable:
# the same proportional room everywhere, so a difference between tiers is a
# difference in the work. 4.5 is set from measurement, not taste: at 4.5 the
# reference courier finishes every order of the shallow tiers (100% and 98%),
# which is what "an easy tier must be near-perfect" requires, and the score is
# insensitive to the multiple thereafter -- 3.5 and 4.5 give the same ladder to
# within noise, because at that point the clock is no longer what binds.
# 4.5 was that number for a city with nothing in the way. Obstacles moved it, and
# the direction is the informative part: a barrier lengthens the *optimal* route
# by 1.21x, and this multiple scales with the optimum, so if the reference
# courier's overhead scaled the same way nothing would need to change. It does
# not. Its overhead is proportional to the distance it actually walks, and going
# round closures it walks 2.0x the optimum instead of 1.66x, so the same multiple
# bought it proportionally less room. Measured on 20 seeds with the obstacle
# album live:
#
#     multiple   solo (blind / sighted)   pair (blind / sighted)
#       4.5           65% / 65%               85% / 90%
#       6.75          85% / 95%               90% / 98%
#       9.0           85% / 95%               90% / 98%
#      13.5           85% / 95%               90% / 98%
#
# Two things are worth reading off that table. The easy tiers come back to
# near-perfect, which is what they are for. And past about 6.75 the clock stops
# being what decides anything at all -- doubling and trebling it again changes
# not one delivery, because what binds then is whether the courier can find the
# door, which is the demand the ladder is supposed to measure. 7.0 sits inside
# that flat region rather than on its edge.
#
# ---------------------------------------------------------------------------
# THAT TABLE DOES NOT REPRODUCE, and the half of it that fails is the half the
# choice of 7.0 rests on. Re-measured with this code -- same policy, same
# albums, 12 seeds, both strides, mean over solo and pair:
#
#     multiple   blind    sighted   gain   cells where sight wins
#       2.0      27.3%     36.9%    +9.6           4/4
#       2.6      39.4%     50.1%   +10.8           4/4
#       3.5      52.2%     63.4%   +11.2           4/4
#       4.5      66.0%     69.1%    +3.2           2/4
#       6.75     69.8%     72.9%    +3.1           2/4
#       9.0      69.8%     72.9%    +3.1           2/4
#
# The saturation claim holds exactly: 6.75 and 9.0 are identical to the
# delivery. What does not hold is "sight separates there". It separates by
# +3.1 points at 6.75 and by +11.2 at 3.5, and at 7.0 the sighted arm is level
# with or behind the blind one on half the cells -- perfect recognition removes
# every barrier collision and every red crossing and buys nothing, because at
# seven times optimal a courier can walk into every closure on the map and
# still finish.
#
# So the multiple is 3.5, chosen off the second table rather than the first.
# It is where the gain from looking is largest, and it is the largest multiple
# at which looking wins in every cell tested rather than half of them.
#
# What that costs, stated plainly because it is a real cost: the blind floor on
# solo and pair falls from about 70% to about 52%, so those tiers are no longer
# ones a text-only courier passes comfortably. The ``Difficulty`` docstring
# describes them as rungs that exist "to prove an agent can read an address,
# find a street and recognise a door at all", and at 3.5 a courier that cannot
# see fails about half of them -- which is the point: the half it fails are the
# ones with something in the way. A *sighted* agent still passes comfortably,
# and that is the property the ladder should have had all along.
#
# The reference-policy sweeps (an internal tool) regenerate the second table
# and the figures in
# docs/RUNNING.md. Both must be re-run if this number moves again.
TIME_BUDGET_MULTIPLE = 3.5
# When the dispatcher gives up on an order, as a multiple of the window it
# quoted. A job nobody can be bothered to finish has to stop being worth points,
# or "ignore the deadline" is a free strategy: before this, a delivery three
# hours late paid exactly what an on-time one did.
ORDER_EXPIRY_MULTIPLE = 3.0
# A late delivery still gets paid, but not in full. Half, because the two
# degenerate ends are both worse: pay full and the deadline is decorative, pay
# nothing and a courier running late is better off abandoning a bag it has
# already collected -- which is a strategy no dispatcher would design for.
LATE_FEE_FRACTION = 0.5
# What a refused action costs. It was zero on both currencies, and that made
# ``collect()`` a free unlimited rangefinder: standing anywhere, a policy could
# call it, be told "It is 140 m away", and pay neither a turn nor a second --
# the same information ``check_map`` charges a turn and five seconds for. An
# action the benchmark cannot see is an action outside the measurement.
#
# Charging for it was necessary and not sufficient, and the arithmetic says why:
# a block costs 13-26 s to walk, so at 5 s a refusal was *still strictly cheaper
# than moving*. The optimal endgame became walk-probe-walk -- read the sign of
# the change in the quoted distance and you have a gradient oracle for the door,
# needing no photograph, no door number and no street sign. Two independent
# reviewers found it and both used it to finish an episode. So the distance is
# gone from the refusal entirely: a refusal now reports arrival, which is a
# yes/no the courier could get by standing there, and nothing else. Distance to
# an address is what ``check_map`` sells.
REJECTED_ACTION_SECONDS = 5.0


def compass_of(bearing: float) -> str:
    return COMPASS[int((bearing % 360.0) / 45.0 + 0.5) % 8]


def _leading_number(numbers: str) -> int | None:
    """The first house number out of a rendered range like ``"12, 14, … 40"``."""
    head = numbers.split(",", 1)[0].strip()
    return int(head) if head.isdigit() else None


def relative_of(bearing: float, facing: float | None) -> str:
    """Where a bearing lies relative to the way the courier is facing."""
    if facing is None:
        return ""
    return RELATIVE[int(((bearing - facing) % 360.0) / 45.0 + 0.5) % 8]


def turn_word(from_bearing: float, to_bearing: float) -> str:
    """How a route describes the change of heading at a junction."""
    delta = (to_bearing - from_bearing) % 360.0
    if delta < 20.0 or delta > 340.0:
        return "continue"
    if delta < 70.0:
        return "bear right"
    if delta < 160.0:
        return "turn right"
    if delta < 200.0:
        return "turn back"
    if delta < 290.0:
        return "turn left"
    return "bear left"


def movement_axis(bearing: float) -> str:
    """Which crossing axis a heading belongs to."""
    return "north-south" if compass_of(bearing) in ("north", "south") else "east-west"


def signal_state_for_axis(seconds: float, axis: str) -> str:
    """Deterministic two-phase signal, matching the vendored controller.

    Reimplemented rather than imported so this runtime carries no import-time
    dependency on the vendored engine, and cross-checked against it by test:
    ``test_signal_phase_matches_the_vendored_controller`` fails if the two ever
    disagree. Copying the convention without pinning it would drift silently and
    make every baked light frame wrong by one phase.
    """
    minute = int(max(0.0, float(seconds)) // 60.0)
    south_north_red = bool(minute % 2 == 1)
    normalised = str(axis or "").strip().lower().replace("_", "-")
    if normalised in {"south-north", "north-south", "vertical", "sn", "ns"}:
        return "red" if south_north_red else "green"
    return "green" if south_north_red else "red"


def signal_state(node_id: str, bearing: float, sim_seconds: float) -> str:
    """The pedestrian light facing a courier about to cross, at this moment.

    The whole city switches together, as it does in the reference environment --
    an earlier version offset each junction by a hash of its id, which looked
    more realistic and made every baked frame disagree with the engine.

    Nothing about this reaches the observation text. The colour of a light is
    something you *see*, and making it readable in words would hand a text-only
    policy the one signal this environment has that genuinely requires looking.
    """
    return signal_state_for_axis(sim_seconds, movement_axis(bearing))


@dataclass
class Order:
    """One job, expressed entirely in addresses a person could read out."""

    index: int
    pickup: Address
    dropoff: Address
    fee: float
    deadline_s: float
    picked_up: bool = False
    delivered: bool = False
    delivered_at_s: float | None = None
    # When this job was handed over. A deadline measured from clock-in makes one
    # slow first order doom every later one -- the reference courier went 0/10
    # on-time for exactly that reason -- and no real dispatcher works that way.
    issued_at_s: float | None = None
    # Given up on: past ``ORDER_EXPIRY_MULTIPLE`` windows and taken back by the
    # dispatcher. It pays nothing and it cannot be delivered.
    expired: bool = False
    # What it actually paid. Full fee on time, ``LATE_FEE_FRACTION`` of it late,
    # nothing if it expired -- so the ledger and the deadline are one number
    # rather than two that can disagree.
    paid: float = 0.0
    # When the parcel came off the restaurant counter. Only the food-temperature
    # flag reads it; None everywhere the flag is off.
    picked_up_at_s: float | None = None
    # The customer's note, verbatim on the slip. Empty unless special notes are
    # on and this order drew one.
    note: str = ""
    # What is in the bag ("hot meal" / "ice cream" / "groceries"). Empty unless
    # food categories are on; the category picks the spoil window and discount.
    category: str = ""

    def spoil_window_s(self, default_warm_s: float) -> float | None:
        """Seconds from collect until this order pays its spoil discount.

        None means this parcel does not spoil. With no category (categories
        off) the caller decides via the plain temperature flag.
        """
        if self.category == "ice cream":
            return ICECREAM_MELT_SECONDS
        if self.category == "hot meal" or not self.category:
            return default_warm_s
        return None    # groceries

    def spoil_fraction(self) -> float:
        return MELT_FEE_FRACTION if self.category == "ice cream" else COLD_FEE_FRACTION

    def spoil_word(self) -> str:
        return "melted" if self.category == "ice cream" else "cold"

    def due_at(self, fallback_issue_s: float = 0.0) -> float:
        return (self.issued_at_s if self.issued_at_s is not None else fallback_issue_s) + self.deadline_s

    def expires_at(self, fallback_issue_s: float = 0.0) -> float:
        issued = self.issued_at_s if self.issued_at_s is not None else fallback_issue_s
        return issued + self.deadline_s * ORDER_EXPIRY_MULTIPLE

    def minutes_left(self, now_s: float) -> float:
        return (self.due_at() - now_s) / 60.0

    @property
    def live(self) -> bool:
        """Issued, not delivered, not given up on -- a job the courier still has."""
        return self.issued_at_s is not None and not self.delivered and not self.expired

    @property
    def target(self) -> Address:
        return self.dropoff if self.picked_up else self.pickup

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index, "pickup": self.pickup.text, "dropoff": self.dropoff.text,
            "fee": round(self.fee, 2), "deadline_s": round(self.deadline_s, 1),
            "picked_up": self.picked_up, "delivered": self.delivered,
            "expired": self.expired, "paid": round(self.paid, 2),
            "issued_at_s": (None if self.issued_at_s is None else round(self.issued_at_s, 1)),
            "note": self.note,
            "category": self.category,
        }


@dataclass
class StepOutcome:
    """What one tool call did."""

    ok: bool
    message: str = ""
    code: str = ""
    reward: float = 0.0
    sim_seconds: float = 0.0
    moved: bool = False
    finished: bool = False
    # Reported by the world, so a policy can account for its own effort without
    # computing distances from coordinates it should not have.
    walked_m: float = 0.0
    # Set by ``_refuse`` once it has booked the turn, the rejection and the
    # seconds, so the ``_charge`` wrapper around the consulting tools does not
    # book them a second time. A refused ``look``/``navigate`` used to count
    # two turns, two rejections and ten seconds against the one it took.
    charged: bool = False
    # A picture the tool produced, as SVG. Only the phone's map has one, and it
    # is deliberately not called an "image": the photographs come from the
    # world and this comes from the map, and a harness that put them in one list
    # would let a courier believe the phone can see the street.
    drawing: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class CourierEnv:
    """The delivery world, over a compiled road network."""

    SCREEN_ROUTE_NONE = "none"
    SCREEN_ROUTE_AUTOMATIC = "automatic"
    SCREEN_ROUTE_EXPLICIT = "explicit"

    def __init__(
        self,
        network: RoadNetwork,
        *,
        seed: int = 0,
        order_count: int = 1,
        album_root: Path | None = None,
        signal_album_root: Path | None = None,
        obstacle_album_root: Path | None = None,
        deadline_slack: float = 2.5,
        shift_seconds: float | None = None,
        condition: str = Condition.FULL,
        enforce_signals: bool | None = None,
        narration: str = "none",
        enforce_obstacles: bool | None = None,
        difficulty: str | None = None,
        queue_depth: int | None = None,
        stride: str = Stride.WAYPOINT,
        embodiment: str | Embodiment | None = None,
        pavement_album_root: Path | None = None,
        pavement_obstacle_album_root: Path | None = None,
        served_long_edge: float | None = None,
        order_bias: dict[str, float] | None = None,
        # ── optional constraints. Defaults are the benchmark as it is: a flag
        # left alone must not move one number of an existing run. Walking
        # energy defaults ON because the stamina drain has always been live;
        # the other four are new mechanics and default OFF.
        enable_earning_jitter: bool = False,
        enable_food_temperature: bool = False,
        enable_special_notes: bool = False,
        enable_walking_energy: bool = True,
        enable_phone_battery: bool = False,
        enable_food_categories: bool = False,
        enable_phone_recharge: bool = False,
        food_warm_seconds: float = FOOD_WARM_SECONDS,
        # Pixel-goal (live UE) runs may print the pawn's pose in the
        # observation for audit; the album benchmark never does.
        show_pose: bool = False,
    ):
        # A tier sets the list, the queue and the clock; they are not
        # independent, and the tier is the only place they are chosen together.
        # An explicit depth still wins, because the claim "the depth is what
        # makes this hard" is only testable if the depth can be held at 1 with
        # everything else unchanged.
        if difficulty is not None:
            order_count, tier_depth = Difficulty.spec(difficulty)
            if queue_depth is None:
                queue_depth = tier_depth
            # The clock is derived in reset(), once the orders exist.
            shift_seconds = None
        if queue_depth is None:
            queue_depth = 1
        # What is doing the delivering. Speed, stamina, the cost of stopping and
        # -- the one that matters most for transfer -- which viewpoint's album it
        # is entitled to see. See ``embodiment.py``.
        self.embodiment = embodiment_for(embodiment)
        self.embodiment.require_defined()
        self.pavement_album_root = (
            Path(pavement_album_root) if pavement_album_root else None
        )
        self.pavement_obstacle_album_root = (
            Path(pavement_obstacle_album_root) if pavement_obstacle_album_root else None
        )
        self.difficulty = difficulty
        # How many jobs the dispatcher lets the courier hold at once. This is
        # the difficulty axis: their windows run concurrently, so a deep queue
        # has to be *sequenced* rather than worked in the order it arrived.
        self.queue_depth = max(1, int(queue_depth))
        # A tier of 0 orders means the dispatcher never runs out -- ENDLESS.
        self.unbounded = difficulty is not None and order_count == 0
        # Off unless the album can show the light. Overridable so the mechanic
        # stays testable before the frames land.
        # Charge for the light exactly when the light can be seen. Passing a
        # signal album is the evidence; without one the penalty would punish
        # information the environment withholds, which is what made the score
        # anti-correlated with delivery before this existed.
        # Which facts the text is allowed to state outright. An axis of its own,
        # orthogonal to Condition: that ladder takes tools away, this one moves
        # information between the pictures and the words while the tools stay
        # exactly the same.
        #
        #   "none"   direction, lights and barriers are in the images only.
        #            The benchmark as designed.
        #   "route"  the route's next street is named in the text; lights and
        #            barriers stay in the pictures. Isolates route-reading --
        #            the one thing measurement says current models cannot do --
        #            from the two visual jobs they have never been tested on.
        #   "all"    every one of the three is stated in the text. A text-only
        #            policy can solve this, which is the point: it is the
        #            solvability floor the other two are measured against.
        if narration not in ("none", "route", "all"):
            raise ValueError(f"narration is none, route or all, not {narration!r}")
        self.narration = narration
        if enforce_signals is None:
            enforce_signals = signal_album_root is not None or SIGNAL_FRAMES_AVAILABLE
        self.enforce_signals = bool(enforce_signals)
        if enforce_obstacles is None:
            enforce_obstacles = obstacle_album_root is not None or OBSTACLE_FRAMES_AVAILABLE
        self.enforce_obstacles = bool(enforce_obstacles)
        if condition not in Condition.ALL:
            raise ValueError(f"unknown condition {condition!r}; expected one of {Condition.ALL}")
        self.condition = condition
        if stride not in Stride.ALL:
            raise ValueError(f"unknown stride {stride!r}; expected one of {Stride.ALL}")
        self.stride = stride
        self.network = network
        self.seed = seed
        # Adaptive-curriculum sampling weights over order features, or None.
        # None is the benchmark: _draw_order takes the branch that existed
        # before this parameter did and consumes the RNG identically, so a
        # baseline run is byte-for-byte the same shift with or without this
        # code in the tree. See _order_features for the axes.
        self.order_bias = dict(order_bias) if order_bias else None
        self.enable_earning_jitter = bool(enable_earning_jitter)
        self.enable_food_temperature = bool(enable_food_temperature)
        self.enable_special_notes = bool(enable_special_notes)
        self.enable_walking_energy = bool(enable_walking_energy)
        self.enable_phone_battery = bool(enable_phone_battery)
        # Categories imply the temperature machinery: a meal that cools is the
        # common case of "what is in the bag matters", so the flag switches the
        # superset on rather than requiring two flags that can disagree.
        self.enable_food_categories = bool(enable_food_categories)
        if self.enable_food_categories:
            self.enable_food_temperature = True
        if enable_phone_recharge and not enable_phone_battery:
            raise ValueError(
                "enable_phone_recharge without enable_phone_battery: there is "
                "no battery to charge, and a tool that can only be refused is "
                "a turn the agent is invited to lose.")
        self.enable_phone_recharge = bool(enable_phone_recharge)
        self.food_warm_seconds = float(food_warm_seconds)
        self.phone_recharges: int = 0
        self.melted_deliveries: int = 0
        # None means "this shift has no battery mechanic", so every reader can
        # tell "off" from "flat" -- a 0.0 with the flag off would look like a
        # dead phone in the summary.
        self.phone_battery: float | None = 100.0 if self.enable_phone_battery else None
        self.phone_died_at_s: float | None = None
        self.cold_deliveries: int = 0
        self.notes_followed: int = 0
        self.order_count = order_count
        # The album this body is entitled to. A person on foot is on the
        # pavement and a rider is in the carriageway, and showing either the
        # other one's frames trains a policy to recognise a world it will never
        # stand in. Only the carriageway album has been baked; until the
        # pavement bake lands, a walking courier is served carriageway frames and
        # ``viewpoint_served`` in the summary says so, so the debt is measurable
        # rather than silent.
        if (self.embodiment.viewpoint == Viewpoint.PAVEMENT
                and self.pavement_album_root is not None):
            album_root = self.pavement_album_root
            self.viewpoint_served = Viewpoint.PAVEMENT
            # The obstacle album has to move with it. Serving footway frames for
            # clear streets and centreline frames wherever an obstacle stands
            # makes the *viewpoint itself* announce the hazard: measured at 19.5%
            # of a walker's frames, every one of them an obstacle. That is the
            # filename leak again in a different disguise, so the two albums are
            # switched together or not at all.
            if pavement_obstacle_album_root is not None:
                obstacle_album_root = pavement_obstacle_album_root
            elif obstacle_album_root is not None:
                raise ValueError(
                    "a pavement album was given without a pavement obstacle "
                    "album: the obstacle frames would come from the carriageway "
                    "and the change of viewpoint alone would tell the policy an "
                    "obstacle is there. Pass pavement_obstacle_album_root, or "
                    "drop obstacle_album_root."
                )
        else:
            self.viewpoint_served = Viewpoint.CARRIAGEWAY
        self.viewpoint_matches_embodiment = (
            self.viewpoint_served == self.embodiment.viewpoint
        )
        self.album_root = Path(album_root) if album_root else None
        # Frames baked in both signal states, one pair per signalised approach.
        self.signal_album_root = Path(signal_album_root) if signal_album_root else None
        # Frames baked with something in the way, one per (approach, kind).
        self.obstacle_album_root = Path(obstacle_album_root) if obstacle_album_root else None
        self.deadline_slack = deadline_slack
        # A shift runs on a clock, not a step count. Ten deliveries in an hour is
        # a courier's day; a step cap measures how chatty the policy is instead.
        self.shift_seconds = shift_seconds

        self.streets = {s.index: s for s in network.streets}
        self.addresses_by_street: dict[str, list[Address]] = {}
        for address in network.addresses:
            self.addresses_by_street.setdefault(address.street_name, []).append(address)
        for values in self.addresses_by_street.values():
            values.sort(key=lambda a: a.number)

        self.node_id: str = ""
        self.arrived_from: str | None = None
        self.sim_seconds: float = 0.0
        self.orders: list[Order] = []
        self.earnings: float = 0.0
        self.finished = False
        self.red_crossings: int = 0
        # Which lamps the sender managed to put in front of the model this
        # turn. None means "no one is limiting them".
        self.signals_shown: set[str] | None = None
        self.waits_at_red: int = 0
        # Long-horizon is measured in both currencies: a policy can be cheap in
        # turns and slow on the clock, or the reverse, and only reporting one
        # hides half of what went wrong.
        self.turns: int = 0
        # How far the courier actually walked, against how far a perfect one
        # would have. Only the oracle used to count this, so route quality was
        # unmeasurable for every other policy -- and route quality is most of
        # what separates a good courier from a lucky one.
        self.walked_cm: float = 0.0
        self.optimal_seconds: float = 0.0
        self.stamina: float = float(self.embodiment.stamina or 0.0)
        self.rests: int = 0
        self.optimal_walk_cm: float = 0.0
        # Actions the world refused. A policy that spends a third of its turns
        # walking into walls is not the same as one that spends none, and the
        # score alone cannot tell them apart.
        self.rejected_actions: int = 0
        self.expired_count: int = 0
        # Walking into a barrier, and squeezing past a congested pavement. Both
        # are time a courier that read the photograph never spends, so they are
        # the two numbers that say whether looking paid.
        self.blocked_attempts: int = 0
        # Edges this courier has personally walked into and been turned back
        # from. Not the phone's knowledge -- the courier's own eyes.
        self.witnessed_blocks: set[tuple[str, str]] = set()
        self.slow_passages: int = 0
        # What is on the phone's screen. A map app does not switch off when you
        # put the phone in your pocket: the route it drew stays drawn and the
        # dot showing where you are keeps moving. Showing the map for one turn
        # made asking for it a tax -- playing an episode by hand, 45% of the
        # turns went on ``navigate()``, 37 lookups at 15 s each, nine minutes of
        # an hour spent standing still reading a phone.
        #
        # The *route* is frozen at the moment it was asked for and the *position*
        # is live. That is the whole design: walking is free to watch, but a
        # better route costs a call, so a courier that wanders off the blue line
        # can see that it has and must decide whether the answer is worth 15 s.
        self.screen_route: list[tuple[float, float]] = []
        self.screen_target: Address | None = None
        # Automatic routes follow the active job when dispatch state changes;
        # an explicit navigate() route belongs to the courier and survives
        # ordinary queue reads/refills until a task action changes the job in
        # hand. Target identity alone cannot distinguish those two cases.
        self._screen_route_owner = self.SCREEN_ROUTE_NONE
        # The last door number read on the street the courier is on, so the
        # observation can say which way the numbers run rather than making the
        # policy remember across turns.
        self._last_numbers: tuple[str | None, int | None] = (None, None)
        # Nothing here records what the courier knows about barriers, and that
        # is the design. ``report_blocked`` used to let the courier tell its
        # phone a street was shut, and the phone would route round it -- which
        # is not a thing a rider does. You see a skip, you take the next street;
        # you do not open the map app and file a report. The phone is a survey
        # and stays permanently blind, so the route keeps pointing through the
        # barrier and the courier has to overrule it from what it can see. That
        # is what makes looking necessary rather than merely rewarded.

        self.signalised = network.signalised_nodes()
        # The long edge the harness actually sends, if it says. The album
        # certifies legibility after a resize to MODEL_LONG_EDGE_PX; a harness
        # that downscales further is not looking at the frame that was
        # certified, and the gate has to be asked again at the size the policy
        # gets. Silence means the album's own answer stands.
        self.served_long_edge = (
            float(served_long_edge) if served_long_edge else None)
        # Whether the observation states the courier's own coordinates. Off by
        # default: it is information the street action space cannot use and
        # never had, and switching it on for everyone would move the baseline
        # the coordinate space is being compared against. The coordinate space
        # turns it on for itself, and it stays a separate flag so the
        # controlled version of that comparison -- street action space, pose
        # shown -- is one config away rather than a code change.
        self.show_pose = bool(show_pose)
        self.visible_signals = self._load_signal_visibility()
        # Where an obstacle can stand on this map. Seed-free and computed once:
        # it is a property of the road network, which is what lets one bake of
        # the album serve every episode of every seed.
        self._neighbours = {n: sorted(node.neighbours) for n, node in network.nodes.items()}
        self._obstacle_sites = obstacle_sites(self._neighbours, network.map_name)
        self._visible_obstacles = load_obstacle_visibility(self.obstacle_album_root)
        self.obstacles = ObstacleField()


    def _load_signal_visibility(self) -> set[str] | None:
        """Which approaches the album can actually show a lamp on.

        ``None`` means the album made no claim, and then nothing is charged --
        the same default as having no album at all. Silence is not consent: an
        album that does not say what it shows is an album that has not been
        checked, and the failure mode of guessing is a penalty on an invisible
        light, which is the defect this whole gate exists to prevent.
        """
        if self.signal_album_root is None:
            return None
        path = self.signal_album_root / SIGNAL_VISIBILITY_FILE
        if not path.exists():
            return None
        import json

        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            return None
        legible = {str(k) for k in data.get("legible", [])}
        legible &= self._readable_at_served_size(data, legible)
        return self._one_lamp_per_lamp(data, legible)

    def _one_lamp_per_lamp(self, data: dict, legible: set[str]) -> set[str]:
        """Stop charging one lamp several times over.

        The map has no lamp objects. Signalised junctions are derived from
        node degree and the bake puts one light mesh at each of them, so a
        junction with four ways out has four photographs of THE SAME LAMP,
        from the same camera, differing only in which phase is lit. The
        environment then gave each approach its own phase from its own bearing
        and charged each one separately -- one lamp treated as four.

        What that did to the courier is worse than the double-counting. Told
        "the lamp for Rue de la Paix" and "the lamp for Rue Cujas", it was
        handed two pictures with identical backgrounds and no way to tell
        which was which; the caption was the only thing distinguishing them.
        Measured on this album: 34 junctions show one lamp to three
        approaches, three show one to four, two show one to five, and only a
        single junction in the map has two genuinely different lamps. Of 130
        charged approaches about 61 were the same lamp counted again.

        So each group of approaches sharing a lamp keeps exactly one -- the
        view where the lamp is largest, which is the one a courier could
        actually read -- and the rest are treated as having no visible lamp at
        all: no frame, no charge. That is the same rule the visibility gate
        already applies, said about a lamp rather than about an album.
        """
        # An album that gives every approach its own lamp says so, and must
        # not be folded back together: composited lamps sit at the same place
        # in every frame, so their boxes coincide although the lamps are
        # genuinely separate. The rendered album's boxes coincide for the
        # opposite reason -- one lamp photographed repeatedly -- and only the
        # album knows which case it is.
        if data.get("lamps_are_per_approach"):
            return legible
        boxes = data.get("lamp_box")
        sizes = data.get("lamp_px")
        if not isinstance(boxes, dict):
            return legible

        def overlap(a: list, b: list) -> float:
            ax0, ay0, ax1, ay1 = a
            bx0, by0, bx1, by1 = b
            wide = max(0, min(ax1, bx1) - max(ax0, bx0))
            tall = max(0, min(ay1, by1) - max(ay0, by0))
            inter = wide * tall
            union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
            return inter / union if union > 0 else 0.0

        def lamp_pixels(key: str) -> int:
            row = (sizes or {}).get(key)
            return int(row[0]) if row else 0

        by_node: dict[str, list[str]] = {}
        for key in legible:
            by_node.setdefault(key.split("|")[0], []).append(key)

        kept: set[str] = set()
        for node, keys in by_node.items():
            groups: list[list[str]] = []
            for key in sorted(keys):
                box = boxes.get(key)
                if not box:
                    groups.append([key])
                    continue
                for group in groups:
                    other = boxes.get(group[0])
                    if other and overlap(box, other) > 0.5:
                        group.append(key)
                        break
                else:
                    groups.append([key])
            for group in groups:
                kept.add(max(group, key=lambda k: (lamp_pixels(k), k)))
        return kept

    # A 2x2 patch after the resize -- the album's own floor, restated here so
    # the runtime is not silently more permissive than the measurement was.
    SERVED_MIN_PIXELS = 4.0

    def _readable_at_served_size(self, data: dict, legible: set[str]) -> set[str]:
        """Of the certified approaches, those still readable at the served size.

        The album records each lamp's area in the frame as baked. Area falls
        with the square of the resize, so a lamp certified at 768 px can be
        under a 2x2 patch by the time a 320 px harness has finished with it --
        on the Paris kerb album that is 34 of 130 approaches. Charging those is
        charging for a light the policy was never sent enough pixels to see,
        which is the same defect the visibility gate exists to prevent, one
        stage further down the pipe.
        """
        sizes = data.get("lamp_px")
        if not self.served_long_edge:
            return legible
        if not isinstance(sizes, dict):
            # Album baked before lamp_px existed: the gate cannot run. Say so once.
            if not getattr(self, "_warned_no_lamp_px", False):
                self._warned_no_lamp_px = True
                import logging

                logging.getLogger(__name__).warning(
                    "served_long_edge=%s but this album has no lamp_px "
                    "metadata; the served-size visibility gate is OFF and "
                    "red-light charging follows the bake resolution. Re-bake "
                    "the album to get served-size gating.",
                    self.served_long_edge,
                )
            return legible
        readable = set()
        for key in legible:
            row = sizes.get(key)
            if not row:
                # Unmeasured. Keep it: the album certified it and this check is
                # a refinement, not a second gate with a different default.
                readable.add(key)
                continue
            px, width, height = row
            scale = min(1.0, self.served_long_edge / max(width, height))
            if px * scale * scale >= self.SERVED_MIN_PIXELS:
                readable.add(key)
        return readable

    def _blocked_along(self, row: dict[str, Any]) -> bool:
        """Is anything shut on the stretch this call would actually walk?

        At block stride ``walk_to`` runs the whole block, so asking only about
        the first 18 m of it says "clear" and then stops at a barrier six
        junctions along -- 3.4% of rows did exactly that. The narrated setting
        has no photograph to catch it, so the sentence has to cover the same
        ground the call does.
        """
        if self.stride != Stride.BLOCK:
            return self.obstacles.blocks(self.node_id, row["node"])
        return any(self.obstacles.blocks(a, b)
                   for a, b in self.block_chain(self.node_id, row["node"]))

    def _is_route_step(self, toward: str) -> bool:
        """Is walking to ``toward`` the next step of the shortest route?

        Uses the same routing the phone draws, so the narrated setting says
        exactly what the picture would have shown -- otherwise the two settings
        would be different tasks rather than the same task told two ways.
        """
        order = next((o for o in self.orders if o.live), None)
        if order is None:
            return False
        # kerb_node, not nearest_node. Everything else that decides where a
        # stop *is* -- the drawn map, collect, hand_over, arrival, the optimal
        # route -- uses the kerb. They differ on 14% of addresses and by more
        # than the arrival tolerance on most of those, so a marker aimed at the
        # nearest node vanishes at a junction where hand_over still refuses,
        # leaving the courier on the doorstep with nothing left to read.
        target = (order.dropoff if order.picked_up else order.pickup).kerb_node
        # Obstacle-aware, deliberately unlike the drawn route. Under narration
        # the words are the only source of direction, so they must never point
        # through a barrier; the map is allowed to, because the photograph is
        # there to contradict it.
        route = self.route_nodes(self.node_id, target, obstacles=True)
        return bool(route and len(route) > 1 and route[1] == toward)

    def signal_is_visible(self, node_id: str, toward: str) -> bool:
        """Can the courier standing at ``node_id`` see the lamp for this crossing?

        Two gates, and the second exists because the first is not enough. The
        album says which crossings *have* a lamp it can show. The harness then
        decides how many pictures fit in one turn, and when it runs out it drops
        street views and their lamps together -- so a crossing the album can
        show is not necessarily a crossing the courier was shown. Charging on
        the album alone penalises a policy for a lamp that never arrived, which
        is the same defect the album gate was added to prevent, arriving one
        layer further out.

        ``signals_shown`` is set by whatever is doing the sending, each turn.
        Left as None it means "everything the album has", which is right for the
        evaluation harness that sends them all.
        """
        key = f"{node_id}|{toward}"
        if self.signals_shown is not None and key not in self.signals_shown:
            return False
        if self.visible_signals is None:
            return bool(SIGNAL_FRAMES_AVAILABLE)
        return key in self.visible_signals

    def signal_in_album(self, node_id: str, toward: str) -> bool:
        """Can the album show this crossing's lamp at all?

        The first of ``signal_is_visible``'s two gates on its own. A walk that
        runs through several junctions -- ``follow_street``, the block stride
        -- must stop at every lamp the album can show, whether or not the
        sender put that lamp in *this turn's* pictures: ``signals_shown``
        describes the junction the turn started at, and a lamp three
        junctions down was never a candidate for it. Stopping on the sender's
        list let the macro walk through red lights it was never charged for;
        the charge on the hop itself (``_step_to``) still reads both gates.
        """
        key = f"{node_id}|{toward}"
        if self.visible_signals is None:
            return bool(SIGNAL_FRAMES_AVAILABLE)
        return key in self.visible_signals

    def show_only_these_signals(self, keys: "set[str] | None") -> None:
        """Declare which lamps actually reached the model this turn."""
        self.signals_shown = None if keys is None else set(keys)

    # ── lifecycle ────────────────────────────────────────────────────────────

    def reset(self) -> None:
        rng = random.Random(self.seed)
        # Spawn only where the courier has a real choice. A degree-1 node is a
        # cul-de-sac, and starting in one wastes turns on a decision that is not
        # a decision -- the engine's own spawn was such a node.
        self.node_id = self._choose_spawn_node(rng)
        # Where the shift began. ``_optimal_route`` chains from here; reading
        # ``node_id`` at summary time priced the approach to the first pickup
        # from the last drop-off instead, so walk_ratio/time_ratio drifted with
        # wherever the courier happened to stop.
        self.spawn_node = self.node_id
        self.arrived_from = None
        self.sim_seconds = 0.0
        self.earnings = 0.0
        self.finished = False
        self.stamina = float(self.embodiment.stamina or 0.0)
        self.rests = 0
        self.red_crossings = 0
        self.signals_shown = None
        self.waits_at_red = 0
        self.walked_cm = 0.0
        self.rejected_actions = 0
        self.expired_count = 0
        self.turns = 0
        self.blocked_attempts = 0
        self.witnessed_blocks = set()
        self.slow_passages = 0
        self.screen_route = []
        self.screen_target = None
        self._screen_route_owner = self.SCREEN_ROUTE_NONE
        self._last_numbers = (None, None)
        self.phone_battery = 100.0 if self.enable_phone_battery else None
        self.phone_died_at_s = None
        self.cold_deliveries = 0
        self.notes_followed = 0
        self.phone_recharges = 0
        self.melted_deliveries = 0

        # Which sites are live this shift. Deterministic in (map, seed), so a
        # replay of a seed meets the same city; different every seed, so the
        # arrangement cannot be learned once and reused.
        self.obstacles = ObstacleField.generate(
            self._neighbours, self.network.map_name, self.seed,
            sites=self._obstacle_sites,
            visible=self._visible_obstacles if self.enforce_obstacles else None,
        )
        # Kept so ENDLESS can keep drawing after the initial list runs out. The
        # cursor is where the *last drawn* order ended, not where the courier
        # is, so a lazily drawn job chains onto the list exactly as a
        # pre-drawn one would and the two paths cannot diverge.
        self._order_rng = rng
        self._draw_cursor = self.node_id
        self._usable_addresses = [
            a for a in self.network.addresses if a.kerb_node in self.network.nodes
        ]
        self.orders = self._make_orders(rng)
        self.optimal_seconds, self.optimal_walk_cm = self._optimal_route()
        if self.difficulty is not None:
            if self.unbounded:
                self.shift_seconds = Difficulty.ENDLESS_SECONDS
            else:
                # One rule, every tier: the work this seed actually drew, times
                # the room a policy is allowed to waste. See TIME_BUDGET_MULTIPLE
                # for why the multiple is uniform.
                self.shift_seconds = self.optimal_seconds * TIME_BUDGET_MULTIPLE
        self._issue()

    def _choose_spawn_node(self, rng: random.Random) -> str:
        """Choose the graph node at which a new shift starts.

        The stock benchmark remains seed-random over genuine junction choices.
        Region-backed embodied environments override this one seam when UE has
        already been configured to spawn at a certified pedestrian point.  The
        hook keeps graph state and the physical pawn aligned without generating
        and then overwriting a temporary order book during ``reset()``.
        """

        candidates = sorted(
            n for n, node in self.network.nodes.items() if len(node.neighbours) >= 2
        ) or sorted(self.network.nodes)
        if not candidates:
            raise ValueError("cannot start a courier shift on an empty road network")
        return rng.choice(candidates)

    def _optimal_route(self, orders: list[Order] | None = None) -> tuple[float, float]:
        """Seconds and centimetres for a courier who never puts a foot wrong.

        Serving the given list in the given order, chained from the spawn. Used
        two ways: over the whole drawn list it sizes the shift clock, and over
        the orders a run actually delivered it says how much further than
        necessary that run walked. A queue-aware policy can beat it by batching
        stops, which is the point -- it is a reference, not a bound.
        """
        cursor = getattr(self, "spawn_node", None) or self.node_id
        seconds, walk = 0.0, 0.0
        for order in (self.orders if orders is None else orders):
            # Obstacle-aware, and that is not a detail. The clock every tier
            # derives from this number, so costing the shift on a route the
            # obstacles forbid would make the barriers a *stopwatch* penalty --
            # the tier would get harder because the deadline no longer fits,
            # which is precisely the failure the uniform TIME_BUDGET_MULTIPLE
            # was introduced to end. Priced here, a detour costs what a detour
            # costs and nothing more.
            approach = self.route_cost(cursor, order.pickup.kerb_node, obstacles=True)
            delivery = self.route_cost(order.pickup.kerb_node, order.dropoff.kerb_node,
                                       obstacles=True)
            for legs in (approach, delivery):
                if legs is None:
                    continue
                seconds += legs[0]
                walk += legs[1]
            seconds += 2.0 * HANDLING_SECONDS
            cursor = order.dropoff.kerb_node
        return seconds, walk

    def delivered_optimal(self) -> tuple[float, float]:
        """The perfect route through the stops this run actually served.

        Compared against what it really walked, this is the only route-quality
        number that survives a partial shift and an unbounded queue: a courier
        that delivers three of ten cannot be judged against the walk for ten,
        and ENDLESS has no fixed list to judge against at all.
        """
        done = sorted(
            (o for o in self.orders if o.delivered and o.delivered_at_s is not None),
            key=lambda o: o.delivered_at_s,
        )
        return self._optimal_route(done)

    def route_nodes(self, start: str, goal: str) -> list[str] | None:
        """The node sequence of the best walk. Privileged, like route_cost.

        Exists for the adaptive curriculum's feature extraction, which needs
        to know what a candidate order's route passes -- signals, junctions --
        not merely what it costs. Never used to build an observation.
        """
        if start not in self.network.nodes or goal not in self.network.nodes:
            return None
        import heapq

        parents: dict[str, str] = {}
        seen: set[str] = set()
        queue = [(0.0, start)]
        while queue:
            cost, node = heapq.heappop(queue)
            if node == goal:
                path = [node]
                while path[-1] != start:
                    path.append(parents[path[-1]])
                return path[::-1]
            if node in seen:
                continue
            seen.add(node)
            here = self.position(node)
            for neighbour in self.network.nodes[node].neighbours:
                if neighbour in seen:
                    continue
                if neighbour not in parents:
                    parents[neighbour] = node
                heapq.heappush(
                    queue, (cost + math.dist(here, self.position(neighbour)), neighbour))
        return None

    # The adaptive curriculum's feature axes, and the weight keys a profile may
    # set. Each maps a candidate order to buckets whose weights multiply its
    # acceptance probability; 1.0 everywhere is exactly the base distribution.
    _BIAS_FLOOR = 0.15   # no candidate class is ever starved entirely

    def _order_features(self, walk_cm: float, pickup, dropoff) -> list[str]:
        """Which weight keys apply to this candidate order."""
        feats = ["len_short" if walk_cm < 15000.0
                 else "len_long" if walk_cm > 25000.0 else "len_mid"]
        path = self.route_nodes(pickup.kerb_node, dropoff.kerb_node)
        if path:
            junctions = sum(
                1 for n in path if len(self.network.nodes[n].neighbours) >= 3)
            if junctions >= 6:
                feats.append("junction_dense")
            if self.visible_signals is not None:
                on_signals = sum(1 for n in path
                                 if any(k.startswith(n) for k in self.visible_signals))
                if on_signals >= 1:
                    feats.append("signalled")
        return feats

    def order_class_counts(self) -> dict[str, int]:
        """How many ISSUED orders fall in each curriculum class this shift.

        The learnability-driven profile builder (frontier mode) needs to know
        which classes a seed's shift actually contained before it can credit
        that seed's group-variance to a class. Computed from the same feature
        function the bias uses, so the two cannot disagree about what a class
        means.
        """
        counts: dict[str, int] = {}
        for order in self.orders:
            if order.issued_at_s is None:
                continue
            walk = self.route_length_cm(order.pickup.kerb_node,
                                        order.dropoff.kerb_node)
            if walk is None:
                continue
            for feat in self._order_features(walk, order.pickup, order.dropoff):
                counts[feat] = counts.get(feat, 0) + 1
        return counts

    def _bias_accepts(self, rng: random.Random, walk_cm: float,
                      pickup, dropoff) -> bool:
        """Soft rejection sampling toward the profile's weights.

        Multiplicative over the candidate's features, floored so the base
        support is preserved -- the curriculum reweights the distribution, it
        never removes orders the benchmark could draw. Uses its own RNG draw,
        which is fine here and only here: with the bias on, the stream is
        allowed to diverge from baseline, because the distribution already
        has.
        """
        weight = 1.0
        for feat in self._order_features(walk_cm, pickup, dropoff):
            weight *= float(self.order_bias.get(feat, 1.0))
        # Normalised against the largest weight the profile can produce, so a
        # profile of {len_long: 3.0} means "a long order is 3x as likely to be
        # accepted as anything else", not "everything else is rejected".
        ceiling = max((float(v) for v in self.order_bias.values()), default=1.0)
        ceiling = max(ceiling, 1.0)
        accept = max(self._BIAS_FLOOR, min(1.0, weight / ceiling))
        return rng.random() < accept

    def _draw_order(self, index: int, rng: random.Random,
                    budget_left: float | None = None) -> Order | None:
        """One job, chained onto the last one drawn.

        Chaining matters: drawing every pickup from the spawn makes each order an
        independent teleport-and-fetch, while a real shift is a sequence where
        the last drop-off is the next journey's start. It is also what keeps a
        deep queue in one district rather than scattered across the map, so the
        sequencing problem the tier poses is a real one and not a forced march.
        """
        usable = self._usable_addresses
        if len(usable) < 2:
            return None
        # The bias rejects candidates the base filters would accept, so it
        # must buy those rejections back with attempts -- the floor bounds
        # the thinning at 1/_BIAS_FLOOR, and the budget scales by exactly
        # that. The unbiased path keeps its original 400.
        attempts = 400 if self.order_bias is None else int(400 / self._BIAS_FLOOR)
        for _ in range(attempts):
            pickup, dropoff = rng.sample(usable, 2)
            if pickup.street_name == dropoff.street_name:
                continue
            walk = self.route_length_cm(pickup.kerb_node, dropoff.kerb_node)
            # Long enough to be a journey, short enough that ten fit in an
            # hour. Unbounded walks averaged 20 minutes a leg, which made a
            # ten-order shift a 197-minute queue nobody could finish.
            if walk is None or not (8000.0 <= walk <= MAX_ORDER_WALK_CM):
                continue
            start = self.route_length_cm(self._draw_cursor, pickup.kerb_node)
            # The approach leg counts against the shift exactly like the
            # delivery leg does, so it is bounded at both ends exactly like it:
            # a 0 m approach is a job that begins at the door.
            if start is None or not (MIN_APPROACH_WALK_CM <= start <= MAX_APPROACH_WALK_CM):
                continue
            # A deadline generous enough that a competent courier makes it
            # and a wandering one does not.
            # 2.5x the optimal walk plus a minute of handling. The multiple
            # has to cover perception, not just walking: an agent that has to
            # read signs and check a map to find an address spends real time
            # doing it, and at 1.6x the reference courier went 0/10 on time
            # while still delivering. At 3.0x nothing was ever late, which
            # measures nothing either.
            #
            # Priced on the city as it is *today*, obstacles included, for the
            # same reason the shift clock is. The two clocks were split for a
            # while and the split was the whole regression: the shift budget
            # followed the detours and the per-job window did not, so a job whose
            # only route ran round a barrier arrived inside the shift and outside
            # its own window. That is difficulty arriving as a tighter stopwatch,
            # which is exactly what the ladder is not allowed to do -- measured
            # on SOLO it took the reference courier from 100% to 70%, and the
            # failures were orders expiring with a third of the shift unspent.
            approach = self.route_cost(self._draw_cursor, pickup.kerb_node, obstacles=True)
            delivery = self.route_cost(pickup.kerb_node, dropoff.kerb_node, obstacles=True)
            if approach is None or delivery is None:
                continue
            walking_seconds = approach[0] + delivery[0]
            leg = walking_seconds * self.deadline_slack + 2.0 * HANDLING_SECONDS
            # What this job costs a courier who never puts a foot wrong: the
            # two legs at walking pace plus the handling at each end.
            optimal = walking_seconds + 2.0 * HANDLING_SECONDS
            # Draw again rather than give up: one long pair landing late in
            # the shift must not truncate a list that a shorter pair would
            # still fit. Giving up on the first over-budget draw cost four of
            # twenty seeds more than half their orders.
            if budget_left is not None and optimal > budget_left:
                continue
            # Adaptive curriculum, applied LAST: the candidate has already
            # passed every base-validity filter, so the bias reweights the
            # benchmark's own distribution and can never admit an order the
            # benchmark could not draw. Behind the guard so that with the
            # bias off this branch costs nothing and -- the part that
            # matters -- consumes no RNG, leaving baseline shifts
            # byte-identical.
            if self.order_bias is not None and not self._bias_accepts(
                    rng, walk, pickup, dropoff):
                continue
            self._draw_cursor = dropoff.kerb_node
            order = Order(
                index=index, pickup=pickup, dropoff=dropoff,
                fee=self.fee_for(walk),
                # Per-job allowance, not a cumulative clock.
                deadline_s=leg,
            )
            # Both extras draw from their own streams, keyed by (seed, index),
            # never from ``rng``: the order geometry above must be byte-identical
            # with these flags on or off, so a jittered arm and a control arm
            # walk the very same city and differ only in what the slip says.
            if self.enable_earning_jitter:
                swing = random.Random(f"fee-jitter-{self.seed}-{index}").uniform(
                    -FEE_JITTER_FRACTION, FEE_JITTER_FRACTION)
                order.fee = round(max(1.0, order.fee * (1.0 + swing)), 2)
            if self.enable_special_notes:
                order.note = random.Random(
                    f"order-note-{self.seed}-{index}").choice(ORDER_NOTES)
            if self.enable_food_categories:
                order.category = random.Random(
                    f"order-category-{self.seed}-{index}").choice(ORDER_CATEGORIES)
            return order
        return None

    @staticmethod
    def fee_for(walk_cm: float) -> float:
        """What a job pays, from the distance the parcel travels.

        A flat call-out plus a rate per metre, and the rate is what stops the
        courier from cherry-picking. Measured over the drawn distribution the
        marginal pay of an extra metre is 0.01, and an extra metre costs 0.71 s
        of walking; against the mean earning rate of about 0.016/s that makes a
        long job worth 0.0143/s at the margin against 0.0157/s on average --
        within 10%, so there is no length a profit-seeking courier should prefer
        on rate alone. ``test_no_order_length_is_a_free_lunch`` pins the band.
        """
        return round(3.0 + walk_cm / 100.0 * 0.01, 2)

    def _make_orders(self, rng: random.Random) -> list[Order]:
        """The shift's list, drawn up front. ENDLESS draws its as it goes."""
        orders: list[Order] = []
        # Only an externally imposed clock can bound the list; a derived one is
        # computed *from* the list, so cutting the list to fit it is circular.
        # Saying so is the point -- SHIFT_FILL silently did nothing on every
        # difficulty tier because ``reset`` had not set ``shift_seconds`` yet.
        budget = (self.shift_seconds * SHIFT_FILL
                  if self.shift_seconds is not None and self.difficulty is None else None)
        elapsed, cursor = 0.0, self.node_id
        # Jobs are numbered from 1, like everything else the courier is offered.
        # They used to start at 0 while streets started at 1, and nothing ever
        # showed the number -- the slip says "Job: collect from 8 Rue Bonaparte"
        # and the tool signature says "navigate(job: int = default)". A model
        # reading an interface whose every other index is 1-based writes
        # navigate(1) for its only job, and got "You are not carrying job 1. In
        # hand: 0." Measured on 40 held-out seeds: 23 refusals, 8 of them in one
        # episode, which is a fifth of that episode's forty turns spent on an
        # off-by-one in the interface rather than on the city.
        for index in range(1, self.order_count + 1):
            left = None if budget is None else budget - elapsed
            order = self._draw_order(index, rng, budget_left=left)
            if order is None:
                break
            orders.append(order)
            if budget is not None:
                # Two shortest paths an order, and only the externally clocked
                # case has anything to spend them on.
                elapsed += ((self.route_length_cm(cursor, order.pickup.kerb_node) or 0.0)
                            + (self.route_length_cm(order.pickup.kerb_node,
                                                    order.dropoff.kerb_node) or 0.0)
                            ) / WALK_SPEED_CM_S + 2.0 * HANDLING_SECONDS
            cursor = order.dropoff.kerb_node
        return orders

    # ── the live queue ───────────────────────────────────────────────────────

    def _expire_overdue(self) -> None:
        """Take back every order the dispatcher has given up on.

        Without this a deadline is decorative: a delivery three hours late paid
        exactly what an on-time one did, so "ignore the clock" cost nothing at
        all and the on-time count was a statistic rather than a stake.
        """
        for order in self.orders:
            if order.live and self.sim_seconds > order.expires_at():
                order.expired = True
                self.expired_count += 1

    def _issue(self) -> None:
        """Top the queue back up to its depth, drawing more work if need be.

        Sweeps first, so an order taken back is replaced in the same breath. A
        dispatcher that took an order back and gave nothing in return let a
        policy end its own episode by being slow: the reference courier was
        issued 8.1 of a ten-order shift and a random walker 3 of 10, because a
        queue that emptied through expiry was never refilled and the run simply
        stopped. Being bad at the job must not shorten it.
        """
        self._expire_overdue()
        live = sum(1 for o in self.orders if o.live)
        for order in self.orders:
            if live >= self.queue_depth:
                self._light_the_screen()
                return
            if order.issued_at_s is None:
                order.issued_at_s = self.sim_seconds
                live += 1
        self._light_the_screen()
        while self.unbounded and live < self.queue_depth:
            order = self._draw_order(len(self.orders) + 1, self._order_rng)
            if order is None:
                self._light_the_screen()
                return
            order.issued_at_s = self.sim_seconds
            self.orders.append(order)
            live += 1
        self._light_the_screen()

    def _drain_screen(self) -> None:
        """The lit map sips battery for every block walked under it.

        Only with the phone-battery flag on, and only while a route is
        actually on the screen. When the charge runs out the screen goes
        dark mid-shift: the route disappears, ``navigate()`` starts refusing,
        and the courier finishes the shift on street names and photographs
        alone. The moment it died is recorded, because "how long did the map
        last" is the number that says whether the drain rate binds.
        """
        if self.phone_battery is None or self.screen_target is None:
            return
        if self.phone_battery <= 0.0:
            return
        self.phone_battery = max(0.0, self.phone_battery - PHONE_BATTERY_SCREEN_PCT)
        if self.phone_battery <= 0.0:
            self.screen_route = []
            self.screen_target = None
            # A dead screen owns nothing: whatever navigate() had put there
            # is gone, and the revival must be free to light the job in hand.
            self._screen_route_owner = self.SCREEN_ROUTE_NONE
            self.phone_died_at_s = self.sim_seconds

    def _light_the_screen(self, *, reclaim: bool = False) -> None:
        """Put the active job in hand on the phone's map, without being asked.

        The direction to walk lives only on the map now -- nothing in any
        sentence the environment speaks says which way to go. That made the map
        load-bearing and left it behind a tool call: the courier had to spend a
        turn on navigate() before it had any direction at all, and on 40
        held-out episodes 25 never called it. Those 25 walked the whole shift
        with no source of direction whatsoever, which is not a hard task, it is
        an unanswerable one.

        A courier who has just been given a job is looking at it on their
        phone. So the screen starts lit, on the job in hand, and navigate()
        goes back to being what it is for: re-centring the route after the
        courier has moved, or pointing the phone at some other address.
        """
        if self.condition in (Condition.NO_PHONE, Condition.VISUAL):
            return
        # A dead phone does not light itself for a new job. The courier walks
        # the rest of the shift by street names, which is the decision the
        # battery exists to make real.
        if self.phone_battery is not None and self.phone_battery <= 0.0:
            return
        # A route the courier chose with navigate() is theirs until a state
        # transition (collection) reclaims the screen for the carried job.
        if (self._screen_route_owner == self.SCREEN_ROUTE_EXPLICIT
                and not reclaim):
            return
        # Read straight off the list rather than through active_order(), which
        # goes back through live_orders() and _issue() -- and this is called
        # from _issue. The first version recursed until the stack ran out.
        live = [o for o in self.orders if o.live]
        carried = next((o for o in live if o.picked_up), None)
        order = carried or (live[0] if live else None)
        if order is None:
            self.screen_target = None
            self.screen_route = []
            self._screen_route_owner = self.SCREEN_ROUTE_NONE
            return
        if order.target is self.screen_target:
            self._screen_route_owner = self.SCREEN_ROUTE_AUTOMATIC
            return
        if order.target.kerb_node:
            # Computing the drawing is what stores the route on the screen.
            self.map_drawing(order.target)
            self._screen_route_owner = self.SCREEN_ROUTE_AUTOMATIC

    def live_orders(self) -> list[Order]:
        """The jobs in hand right now, oldest first.

        Sweeping and refilling here rather than only on hand-over is what makes
        the queue a property of the clock instead of a property of how often the
        policy happens to succeed.
        """
        self._issue()
        return [o for o in self.orders if o.live]

    # ── geometry ─────────────────────────────────────────────────────────────

    def position(self, node_id: str | None = None) -> tuple[float, float]:
        node = self.network.nodes[node_id or self.node_id]
        return (node.x_cm, node.y_cm)

    def street_of(self, node_id: str) -> str:
        return self.streets[self.network.nodes[node_id].street_index].name

    def edge_street(self, a: str, b: str) -> str:
        """The street an edge runs along.

        Naming a candidate after its *destination node* made 30.1% of edges
        report different names in the two directions -- walking A to B was "Rue
        Monge" and B back to A was "Rue Jacob" -- which contradicts the system
        prompt's own rule that a street keeps its name from one junction to the
        next. When both ends share a street, the edge runs along it; otherwise
        the step is a turn onto the neighbour's street, which is what a courier
        would say.
        """
        street_a = self.network.nodes[a].street_index
        street_b = self.network.nodes[b].street_index
        return self.streets[street_a if street_a == street_b else street_b].name

    def route_length_cm(self, start: str, goal: str) -> float | None:
        """Shortest walking distance, for the environment's own bookkeeping.

        Privileged: it sets deadlines and measures how far off optimal a run
        was. It is never placed in an observation, because handing the agent a
        shortest path would replace navigation with reading a number.

        Streets the courier has *reported* shut are avoided, and only those. The
        phone does not know a barrier is there until it is told, so this is the
        courier's own knowledge being used to route, not the environment's.
        """
        if start not in self.network.nodes or goal not in self.network.nodes:
            return None
        import heapq

        seen: set[str] = set()
        queue = [(0.0, start)]
        while queue:
            cost, node = heapq.heappop(queue)
            if node == goal:
                return cost
            if node in seen:
                continue
            seen.add(node)
            here = self.position(node)
            for neighbour in sorted(self.network.nodes[node].neighbours):
                if neighbour in seen:
                    continue
                heapq.heappush(
                    queue, (cost + math.dist(here, self.position(neighbour)), neighbour)
                )
        return None

    def route_cost(self, start: str, goal: str, *,
                   obstacles: bool = False) -> tuple[float, float] | None:
        """``(seconds, centimetres)`` for the best walk, optionally avoiding obstacles.

        Two separate currencies because with obstacles they stop agreeing: the
        quickest way past a congested pavement can be the longer way round. The
        search is therefore on *seconds* and the metres are carried along the
        path it picks, rather than the reverse.

        ``obstacles=False`` is the plain graph and is what the phone uses -- a
        map app does not know a street is shut. ``obstacles=True`` is what the
        environment uses to price a shift and to judge how much further than
        necessary a run walked; it is privileged and never observable.
        """
        if start not in self.network.nodes or goal not in self.network.nodes:
            return None
        import heapq

        best: dict[str, float] = {start: 0.0}
        walked: dict[str, float] = {start: 0.0}
        seen: set[str] = set()
        queue = [(0.0, start)]
        while queue:
            cost, node = heapq.heappop(queue)
            if node == goal:
                return cost, walked[node]
            if node in seen:
                continue
            seen.add(node)
            here = self.position(node)
            for neighbour in sorted(self.network.nodes[node].neighbours):
                if neighbour in seen:
                    continue
                if obstacles and self.obstacles.blocks(node, neighbour):
                    continue
                distance = math.dist(here, self.position(neighbour))
                step = cost + distance / WALK_SPEED_CM_S
                if obstacles:
                    step += self.obstacles.delay_seconds(node, neighbour)
                if step < best.get(neighbour, float("inf")):
                    best[neighbour] = step
                    walked[neighbour] = walked[node] + distance
                    heapq.heappush(queue, (step, neighbour))
        return None

    def route_nodes(self, start: str, goal: str, *,
                    obstacles: bool = False) -> list[str] | None:
        """The shortest walk from ``start`` to ``goal``, as the nodes along it.

        Same search as ``route_length_cm``, keeping the predecessors. Privileged
        in the same way: it is the raw material the navigation tool turns into
        spoken directions, and it never reaches an observation as node ids.
        """
        if start not in self.network.nodes or goal not in self.network.nodes:
            return None
        import heapq

        # The predecessor has to be updated whenever a cheaper way in is found,
        # not fixed the first time a node is pushed. Keeping the first pusher
        # reconstructs a path made of edges the search never chose, and on a
        # dense grid that path can be longer than the distance quoted beside it.
        best: dict[str, float] = {start: 0.0}
        came: dict[str, str] = {}
        seen: set[str] = set()
        queue = [(0.0, start)]
        while queue:
            cost, node = heapq.heappop(queue)
            if node == goal:
                path = [node]
                while path[-1] != start:
                    path.append(came[path[-1]])
                return list(reversed(path))
            if node in seen:
                continue
            seen.add(node)
            here = self.position(node)
            for neighbour in sorted(self.network.nodes[node].neighbours):
                if obstacles and self.obstacles.blocks(node, neighbour):
                    # A route that walks through a barrier is not a route. The
                    # drawn map may show one -- a map has never seen the skip --
                    # but a route stated in words has no picture to correct it,
                    # so the courier would be told to walk into a wall and told
                    # nothing else.
                    continue
                if neighbour in seen:
                    continue
                step = cost + math.dist(here, self.position(neighbour))
                if step < best.get(neighbour, float("inf")):
                    best[neighbour] = step
                    came[neighbour] = node
                    heapq.heappush(queue, (step, neighbour))
        return None

    def route_legs(self, start: str, goal: str) -> list[dict[str, Any]] | None:
        """A route as a person would say it: one leg per street, with the turn.

        Consecutive edges on the same street are one instruction -- "keep going
        along Rue Monge for four junctions" -- because that is one decision for
        a courier, and because a turn-by-turn list of 30 identical 18 m hops is
        not directions, it is the graph.
        """
        path = self.route_nodes(start, goal)
        if path is None or len(path) < 2:
            return [] if path is not None else None
        legs: list[dict[str, Any]] = []
        # The turn at a junction is between the edge you arrive on and the edge
        # you leave on, so this tracks the *last* edge walked, not the first edge
        # of the previous leg. Streets curve, and on a leg of six 18 m hops the
        # two differ by enough to call a left turn a right one.
        last_bearing: float | None = None
        for a, b in zip(path, path[1:]):
            street = self.edge_street(a, b)
            bearing = bearing_deg(self.position(a), self.position(b))
            length = math.dist(self.position(a), self.position(b)) / 100.0
            # A junction the courier will be *asked about*, which is not the
            # same thing at both strides. Playing the environment by hand caught
            # this: the phone said "take Avenue des Rosiers north-east, 6
            # junctions, 108 m", one walk_to covered 18 m and stopped, and the
            # courier had no way to tell whether it had gone the wrong way or
            # simply been counted in different units. Under the block stride a
            # leg's count is the number of calls it takes.
            counts = (self.stride != Stride.BLOCK
                      or len(self._neighbours.get(b, ())) != 2)
            if legs and legs[-1]["street"] == street:
                legs[-1]["junctions"] += int(counts)
                legs[-1]["metres"] += length
                legs[-1]["end"] = b
            else:
                legs.append({
                    "street": street, "start": a, "end": b,
                    "junctions": max(1, int(counts)),
                    "metres": length, "bearing": bearing,
                    "heading": compass_of(bearing),
                    "turn": ("head" if last_bearing is None
                             else turn_word(last_bearing, bearing)),
                })
            last_bearing = bearing
        for leg in legs:
            # The heading a leg is *announced* by is the direction it goes
            # overall, not the direction of its first 5 m. On a curving boulevard
            # the two disagree: one leg was announced "west" from a 5 m stub and
            # then ran north-east for 144 m, so a courier who re-asked at the
            # fork was told to walk the opposite way from the instruction it was
            # already following. ``bearing`` stays the first edge, because that
            # is the step the courier takes now.
            leg["heading"] = compass_of(
                bearing_deg(self.position(leg["start"]), self.position(leg["end"]))
            )
        return legs

    def album_coverage(self) -> dict[str, Any]:
        """How much of the walkable network the album actually covers.

        The album is baked against a specific compiled network. Rebuilding the
        graph -- a weld-tolerance change was enough -- renames every node and the
        frames become orphans while every manifest row still says ``status: ok``.
        That happened once and went unnoticed until a policy reported 86% of its
        candidates had no picture. Coverage is measured, not assumed.
        """
        total = covered = signalled = 0
        for node_id, node in self.network.nodes.items():
            for neighbour in node.neighbours:
                total += 1
                if self._plain_frame(node_id, neighbour):
                    covered += 1
                if self.signal_album_root is not None and node_id in self.signalised:
                    path = (self.signal_album_root / "images" / node_id
                            / f"toward_{neighbour}_red.png")
                    if path.exists():
                        signalled += 1
        return {
            "directed_edges": total, "with_frame": covered,
            "fraction": round(covered / total, 4) if total else 0.0,
            "signalised_with_frame": signalled,
            "album_root": str(self.album_root) if self.album_root else None,
        }

    def facing(self) -> float | None:
        """The bearing the courier is facing: the way the last step brought it.

        ``None`` at the very start of a shift, when the courier has not walked
        anywhere yet and so has no back to its head. Every relative direction in
        the observation is derived from this one number, so it lives here rather
        than being recomputed by each caller.
        """
        if self.arrived_from is None or self.arrived_from not in self.network.nodes:
            return None
        return bearing_deg(self.position(self.arrived_from), self.position())

    def _raw_candidates(self) -> list[dict[str, Any]]:
        """Neighbours with geometry but no images, so frame_for cannot recurse."""
        here = self.position()
        facing = self.facing()
        rows: list[dict[str, Any]] = []
        for neighbour in sorted(self.network.nodes[self.node_id].neighbours):
            there = self.position(neighbour)
            rows.append({
                "node": neighbour,
                "street": self.edge_street(self.node_id, neighbour),
                "bearing": bearing_deg(here, there),
                "distance_m": math.dist(here, there) / 100.0,
                "back": neighbour == self.arrived_from,
            })
        rows.sort(key=lambda r: (r["bearing"], r["node"]))
        for number, row in enumerate(rows, start=1):
            row["k"] = number
            row["heading"] = compass_of(row["bearing"])
            # A courier turns left and right, not to 214 degrees. Compass alone
            # made every route instruction a two-step conversion the agent had to
            # do in its head from a bearing it was never given.
            row["relative"] = relative_of(row["bearing"], facing)
        return rows

    def _block_preview(self, first: str) -> tuple[float, int, str]:
        """How far ``walk_to`` will actually carry the courier down this street.

        The candidate line quoted ``distance_m``, the distance to the *next
        waypoint*, at both strides. At block stride that is not what the action
        does, and on a short stub into a bend it is not even the right direction:
        a reviewer took a candidate labelled "on your left (south-east) — next
        junction 7 m" and was carried 61 m north-west. Three numbers described
        one leg -- the row said 18 m, the route said "1 junction, 29 m", the
        outcome said "29 m, through 2 junctions" -- and a policy budgeting from
        the row was wrong every time.

        Walks the same stop rules as ``_run_street`` with no side effects.
        Obstacles are deliberately not consulted: a preview that shortened
        itself at a barrier would announce the barrier in the text, which is the
        one fact the photographs are supposed to hold alone.
        """
        street = self.edge_street(self.node_id, first)
        here, previous, node = self.node_id, self.node_id, first
        distance = math.dist(self.position(here), self.position(first)) / 100.0
        junctions = 1
        for _ in range(len(self.network.nodes)):
            if self._standing_at_a_door(node):
                break
            neighbours = sorted(self.network.nodes[node].neighbours)
            onward = [n for n in neighbours
                      if self.edge_street(node, n) == street and n != previous]
            if len(onward) != 1 or len(neighbours) > 2:
                break
            if node in self.signalised and self.signal_is_visible(node, onward[0]):
                break
            distance += math.dist(self.position(node), self.position(onward[0])) / 100.0
            previous, node = node, onward[0]
            junctions += 1
        return distance, junctions, node

    def candidates(self) -> list[dict[str, Any]]:
        """The numbered streets leaving this junction, with their pictures.

        Ordering is stable and geographic -- clockwise from north -- so the same
        junction always numbers the same way and the numbers mean something a
        person could re-derive. Ordering by distance, as an earlier version did,
        renumbered the same corner depending on where the agent came from.
        """
        rows = self._raw_candidates()
        for row in rows:
            row["image"] = self.frame_for(self.node_id, row["node"])
            row["signal_image"] = self.signal_frame_for(self.node_id, row["node"])
            row["blocked_seen"] = (self.node_id, row["node"]) in self.witnessed_blocks
            if self.narration in ("route", "all"):
                # Named, not merely hinted: under this setting the text has to
                # be enough on its own, or the setting measures nothing.
                row["on_route"] = self._is_route_step(row["node"])
            if self.narration == "all":
                row["told_blocked"] = self._blocked_along(row)
                row["told_signal"] = (
                    signal_state(self.node_id, row["bearing"], self.sim_seconds)
                    if (self.enforce_signals and self.node_id in self.signalised
                        and self.signal_is_visible(self.node_id, row["node"]))
                    else None)
            if self.stride == Stride.BLOCK:
                # What one call actually buys, so the courier can budget from the
                # number it is shown. ``distance_m`` stays as it was -- the step
                # to the next waypoint -- because the geometry and the reference
                # policies are written against it.
                reach, junctions, end = self._block_preview(row["node"])
                row["reach_m"] = reach
                row["reach_junctions"] = junctions
                row["reach_heading"] = compass_of(
                    bearing_deg(self.position(), self.position(end))
                ) if end != self.node_id else row["heading"]
        return rows

    def frame_for(self, node_id: str, toward: str) -> str | None:
        """The view from ``node_id`` looking down the street toward ``toward``.

        Always the street frame, and that is the whole point. This used to serve
        the *signal* frame at the 105 signalised junctions, and the signal bake
        aims the camera at the lamp rather than along the street: on 88 of those
        105 nodes every approach shares one yaw, so all of a junction's
        candidate photographs were the same picture, pointing somewhere the
        courier was not about to walk. The system prompt promises "the
        photograph labelled k is the view down that street", and at 30% of
        junctions it was not.

        The lamp is a second thing to look at, not a replacement for the first,
        so it is served by ``signal_frame_for`` and shown beside this one.

        An obstacle is different again: it is *on* the street being looked down,
        so it replaces the frame rather than being shown beside it. That
        substitution is the entire mechanism by which the obstacle exists for
        the agent -- there is no field, no sentence and no tool that says it is
        there, so a courier that does not compare this picture with the one it
        expected walks into it.
        """
        return self.obstacle_frame_for(node_id, toward) or self._plain_frame(node_id, toward)

    def block_chain(self, node_id: str, toward: str,
                    limit: int | None = None) -> list[tuple[str, str]]:
        """The (from, to) hops one ``walk_to`` covers at the current stride.

        One hop at waypoint stride. At block stride, the hops along the named
        street as far as the next corner -- the same walk ``_run_street`` takes,
        read rather than performed, so a question can be asked about the whole
        block before committing a turn to it.
        """
        if self.stride != Stride.BLOCK:
            return [(node_id, toward)]
        street = self.edge_street(node_id, toward)
        hops = [(node_id, toward)]
        previous, current = node_id, toward
        bound = limit if limit is not None else len(self.network.nodes)
        for _ in range(bound):
            neighbours = self._neighbours.get(current, [])
            if len(neighbours) != 2:
                break                       # a corner: the block ends here
            onward = [n for n in neighbours
                      if n != previous and self.edge_street(current, n) == street]
            if len(onward) != 1:
                break
            hops.append((current, onward[0]))
            previous, current = current, onward[0]
        return hops

    def obstacle_frame_for(self, node_id: str, toward: str) -> str | None:
        """The view down this street with what is standing in it, if anything is.

        At block stride the question is asked of the whole block, not of the
        first eighteen metres of it, and the difference is the difference
        between vision mattering and not. Measured on TRIPLE over six seeds, a
        courier that reads the frames avoids every barrier at waypoint stride
        (21 collisions to 0) and almost none at block stride (59 to 49) when
        only the first hop is consulted -- because the barrier it walks into is
        four hops down a street whose photograph showed a clear road.

        Serving the obstacle's own frame for the whole block is honest here
        because of what a block is: the contraction rule breaks a block at any
        bend over 35 degrees, at any change of street name and at every junction,
        so a barrier standing on it is standing in the courier's line of sight.
        The photograph is taken a few tens of metres further along than the
        courier's feet, and it shows the thing that is really there.
        """
        if self.obstacle_album_root is None:
            return None
        found: list[tuple[str, str, str]] = []
        for start, step in self.block_chain(node_id, toward):
            kind = self.obstacles.in_effect(start, step)
            if kind is None:
                continue
            path = (self.obstacle_album_root / "images" / start
                    / f"toward_{step}_{kind}.png")
            if path.exists():
                found.append((kind, str(path), step))
        if not found:
            return None
        # A block can hold more than one thing, and then which one is shown
        # decides whether looking was any use. Taking the nearest served the
        # stall on the first hop and hid the barrier four hops behind it, so a
        # courier that read the frame walked into the barrier anyway -- 11 of
        # the 11 remaining collisions at block stride were this exact shape, all
        # on one street. A barrier is what changes the decision, and on a block
        # straight to within 35 degrees it is in view past the furniture.
        blocking = next((f for f in found if f[0] == ROAD_BLOCK), None)
        return (blocking or found[0])[1]

    def signal_frame_for(self, node_id: str, toward: str) -> str | None:
        """The pedestrian lamp facing this crossing, in the phase now running.

        Only where the album can actually show it. An approach outside
        ``signal_visibility.json`` has a lamp somewhere off frame, and handing
        the agent a picture of a light it cannot see -- then charging it for
        crossing -- is the defect the visibility gate exists to prevent.
        """
        if self.signal_album_root is None or node_id not in self.signalised:
            return None
        if not self.signal_is_visible(node_id, toward):
            return None
        rows = {row["node"]: row for row in self._raw_candidates()}
        row = rows.get(toward)
        if row is None:
            return None
        state = signal_state(node_id, row["bearing"], self.sim_seconds)
        path = (self.signal_album_root / "images" / node_id
                / f"toward_{toward}_{state}.png")
        return str(path) if path.exists() else None

    def _plain_frame(self, node_id: str, toward: str) -> str | None:
        """The baked view from ``node_id`` looking at ``toward``.

        One frame per walkable direction, so the picture beside ``walk_to(k)``
        is the view down the street ``walk_to(k)`` takes. The earlier album
        rendered four fixed compass yaws, which put the median walkable street
        20 degrees off the centre of the frame it was attached to.
        """
        if self.album_root is None:
            return None
        path = self.album_root / "images" / node_id / f"toward_{toward}.png"
        return str(path) if path.exists() else None

    # ── what the agent is told ───────────────────────────────────────────────

    def house_numbers_near(self, node_id: str, limit: int = 10) -> str:
        """The numbers a courier could read off the doors *at this spot*.

        Only doors whose kerb is this node, or close enough to read from it.
        Taking the two spatially nearest regardless of range advertised numbers
        from a median 50 m and up to 196 m away, so a junction could read
        "outside number 6-8" while door 7 was 107 m down the road -- which is
        exactly how a policy following the "compare the number exactly" rule
        walked away from its delivery.
        """
        street = self.street_of(node_id)
        here = self.position(node_id)
        near = [
            a for a in self.addresses_by_street.get(street, [])
            if a.kerb_node == node_id or math.dist(here, a.kerb) <= READABLE_NUMBER_CM
        ]
        if not near:
            return ""
        numbers = sorted(a.number for a in near)
        # The span of what is actually here, not the span of the first four.
        #
        # ``[:limit]`` was applied before taking min and max, so a node with
        # doors 1, 2, 3, 5, 7, 9, 11, 13 -- all of them at that node, 0.0 m away
        # -- advertised "1-5" and denied the existence of six doors the courier
        # was standing in front of. 17 of the map's 477 addresses could not be
        # recognised at their own doorstep, and the ARRIVAL runbook tells the
        # courier to "compare the number and street to the slip, exactly", so a
        # policy that obeys the prompt walks away from a delivery it has already
        # reached. On seed 9 that cost the whole episode: the courier arrived at
        # 9 Rue Mouffetard on turn 5, read "1-5", left, and spent the remaining
        # 115 turns oscillating past the door for 0 deliveries.
        #
        # The doors are named, not summarised as a span. A span asserts that
        # every number inside it is here, and on this map that is false about
        # half the time: a corner showing "2-8" from doors 2 and 8 has doors 4
        # and 6 fifty metres down the road, so a courier told to compare the
        # number exactly reads a match and collects nothing. Measured over the
        # map, 458 of 953 advertised numbers were for a door out of reach, and
        # naming them instead takes that to 15. Only 8 of 366 junctions have
        # more doors than ``limit`` names, so the ellipsis is the rare case
        # rather than the usual one.
        if len(numbers) == 1:
            return str(numbers[0])
        if len(numbers) <= limit:
            return ", ".join(str(n) for n in numbers)
        return ", ".join(str(n) for n in numbers[:limit]) + f", … {numbers[-1]}"

    def photo_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Which of this turn's rows get a photograph, in caption order.

        The candidate rows themselves, here: one picture per street leaving
        the junction, each captioned with the street the courier would name to
        take it. That pairing is the whole design of the street action space
        -- the caption is in the words ``walk_to`` accepts, so reading a
        picture and acting on it need no translation.

        It is a hook because the pairing stops being the design when the
        courier no longer takes streets by name. See
        ``EmbodiedCourierEnv.photo_rows``.
        """
        return rows

    def location_text(self) -> str:
        street = self.street_of(self.node_id)
        numbers = self.house_numbers_near(self.node_id)
        text = f"You are on {street}"
        if numbers:
            noun = "number" if numbers.isdigit() else "numbers"
            text += f", outside {noun} {numbers}"
        text += "."

        # Say out loud whether this street is the one on the slip.
        #
        # This is not privileged information: it is string equality between two
        # things already printed a few lines apart, the job line and this one.
        # It is here because the comparison is the step policies skip. Measured
        # on Qwen3-VL-4B over 40 episodes: 42 collect() attempts, 32 of them
        # refused because it was not at the door -- 32 turns spent asking a
        # question the observation had already answered. Nothing here says
        # which way the address is; finding it is still the task.
        order = self.active_order()
        if order is not None:
            target = order.target
            if target.street_name == street:
                text += (f" This is the street on the slip; the slip says "
                         f"{target.number}.")
                trend = self._number_trend(street, numbers)
                if trend:
                    text += " " + trend
            else:
                text += f" The slip says {target.street_name}, which is not this street."
        if self.show_pose:
            text += " " + self.pose_text()
        return text

    def pose_text(self) -> str:
        """Where the courier is standing, and which way it is facing.

        The fairness line for the coordinate action space, and the reason it
        is a whole sentence rather than a number: a policy asked to name a
        point has to be told the point it is naming *from*, or every
        coordinate it writes is a guess dressed as arithmetic. What it is NOT
        told is where the delivery is -- the pin is on the map to be measured
        like anything else, and handing over its coordinates would replace the
        task with subtraction.

        Both numbers are metres, north first, on the axes the whole
        environment speaks (north is +x, east is +y -- see
        ``map_image.MapView.to_px``) so that what is printed here and what
        ``walk_to_xy`` accepts are the same two numbers in the same order.

        One decimal: the pawn lands a median 38 cm from the node it aimed at,
        so a second decimal would be printing noise, and whole metres would
        make two adjacent positions read as one.
        """
        x_m, y_m = (v / 100.0 for v in self.position())
        text = f"You are standing at ({x_m:.1f}, {y_m:.1f}), north then east."
        facing = self.facing()
        if facing is not None:
            text += f" You are facing {compass_of(facing)}."
        return text

    def _number_trend(self, street: str, numbers: str) -> str:
        """Whether the doors counted up or down on the way here.

        This is the signal a person actually uses to find a door: not the
        number on this building, but which way the numbers are going. Two
        thirds of failed episodes never reach the pickup at all, and the
        observation was giving the courier a number with nothing to compare it
        against -- it had to hold the last one in its head across a turn, and
        across forty turns of context it did not.

        Facts only, no advice. It says the numbers rose or fell; it does not
        say to turn around. Which way to walk is still the decision under test.
        """
        here = _leading_number(numbers)
        last_street, last_number = self._last_numbers
        self._last_numbers = (street, here if here is not None else last_number)
        if here is None or last_street != street or last_number is None:
            return ""
        if here == last_number:
            return ""
        return ("The numbers rose as you walked here."
                if here > last_number else
                "The numbers fell as you walked here.")

    def clock_text(self) -> str:
        """Every job in hand and how long each has left.

        The whole queue, not just one job. Showing the focused order alone would
        hand the agent a deep queue and hide the thing that makes it deep: a
        courier cannot sequence windows it cannot see, and a tier whose demand
        is invisible in the observation measures luck.
        """
        live = self.live_orders()
        if not live:
            return ""
        if len(live) == 1:
            order = live[0]
            left = order.minutes_left(self.sim_seconds)
            text = (f"You are {abs(left):.0f} min past the deadline." if left < 0
                    else f"{left:.0f} min left before the deadline.")
            return text + self._order_extras(order)
        lines = [f"You are carrying {len(live)} jobs. Their deadlines run at the same time:"]
        for order in live:
            left = order.minutes_left(self.sim_seconds)
            stage = ("deliver to " + order.dropoff.text if order.picked_up
                     else "collect from " + order.pickup.text)
            when = (f"{left:.0f} min left" if left >= 0 else f"{abs(left):.0f} min overdue")
            lines.append(f"  job {order.index}: {stage} — {when}"
                         + self._order_extras(order))
        return "\n".join(lines)

    def _order_extras(self, order: Order) -> str:
        """What the optional constraints append to a job's clock line.

        The note, because the clock block is the one place every job is
        already listed each turn -- a note only readable through a tool call
        is a note most policies never read. And the food's own clock, because
        the warm window runs beside the deadline and a courier told only one
        of two concurrent timers cannot trade them off. Both flags off, this
        is the empty string and the clock line is the old clock line.
        """
        extras = ""
        if order.note:
            extras += f' Note: "{order.note}"'
        if order.category:
            extras += f" Carrying: {order.category}."
        window = order.spoil_window_s(self.food_warm_seconds)
        if (self.enable_food_temperature and window is not None
                and order.picked_up
                and not order.delivered and order.picked_up_at_s is not None):
            out_min = (self.sim_seconds - order.picked_up_at_s) / 60.0
            window_min = window / 60.0
            if order.category == "ice cream":
                verb, done = "melts", "has melted"
            else:
                verb, done = "goes cold", "has gone cold"
            extras += (f" The {order.category or 'food'} has been out {out_min:.0f} min"
                       + (f"; it {verb} at {window_min:.0f}."
                          if out_min <= window_min else f" and {done}."))
        return extras

    def active_order(self) -> Order | None:
        """The job the observation talks about by default.

        Whatever is already in the bag, otherwise the oldest job in the queue.
        Both halves matter. Collecting a parcel and then being told about a
        different address is how a policy ends up carrying a bag around the
        district; and defaulting to the *oldest* rather than the tightest keeps
        the focus from flipping under a policy every time the clock ticks -- a
        min-by-deadline default made the reference courier thrash between two
        targets and cost it half its deliveries.

        It is a default, not a constraint. ``collect`` and ``hand_over`` serve
        whichever live order the courier is actually standing at, so a policy
        that plans a better sequence is free to walk it, and that is precisely
        the capability the deeper tiers exist to measure.
        """
        live = self.live_orders()
        if not live:
            return None
        carried = [o for o in live if o.picked_up]
        return (carried or live)[0]

    @property
    def shift_over(self) -> bool:
        return (
            self.shift_seconds is not None and self.sim_seconds >= self.shift_seconds
        )

    @property
    def delivered_count(self) -> int:
        return sum(1 for o in self.orders if o.delivered)

    @property
    def issued_count(self) -> int:
        """Jobs the dispatcher actually handed over.

        Not ``len(self.orders)``. ENDLESS draws work on demand, and a list that
        keeps growing is not a denominator -- scoring ``delivered / 40`` against
        a queue nobody can finish reported a perfect courier at 39%.
        """
        return sum(1 for o in self.orders if o.issued_at_s is not None)

    @property
    def on_time_count(self) -> int:
        return sum(
            1 for o in self.orders
            if o.delivered and o.delivered_at_s is not None
            and o.delivered_at_s <= o.due_at()
        )

    @property
    def late_count(self) -> int:
        return self.delivered_count - self.on_time_count

    def target_address(self) -> Address | None:
        order = self.active_order()
        return order.target if order else None

    def distance_to_target_cm(self) -> float | None:
        """Straight-line distance to the *door the arrival check tests*.

        The evaluation found the previous runtime quoting distance to one point
        while validating arrival against another, with a gap over the tolerance
        in eight of ten seeds. Here both are this door.
        """
        target = self.target_address()
        if target is None:
            return None
        return math.dist(self.position(), target.kerb)

    # ── tools ────────────────────────────────────────────────────────────────

    def _charge(self, outcome: StepOutcome) -> StepOutcome:
        """Advance the clock by what the tool declared it cost.

        Every tool returned a ``sim_seconds`` and only walking and waiting ever
        applied it, so looking things up was free against the deadline and the
        optimal policy was to consult on every turn forever. Charging here, in
        one place, means a tool cannot declare a cost it does not pay.
        """
        if outcome.charged:
            return outcome
        self.turns += 1
        if not outcome.ok:
            # A looking or consulting tool that was refused is still a refused
            # action, and it is still a turn the courier did not spend walking.
            # The declared cost stands where there is one -- a phone with no
            # signal says so in a second -- but a refusal that declared nothing
            # pays the same floor as any other.
            self.rejected_actions += 1
            if outcome.sim_seconds <= 0.0:
                outcome.sim_seconds = REJECTED_ACTION_SECONDS
        self.sim_seconds += outcome.sim_seconds
        return outcome

    def _refuse(self, outcome: StepOutcome, seconds: float = REJECTED_ACTION_SECONDS) -> StepOutcome:
        """A refused action still happened, so it still costs.

        It cost nothing before -- not a turn, not a second -- and that made
        ``collect()`` an unlimited free rangefinder: its refusal states the exact
        distance to the door, which is what ``check_map`` charges a turn and five
        seconds to say. Walking up to a door and finding it is the wrong one is
        a thing a courier *does*, and both currencies this benchmark reports
        have to see it happen.
        """
        self.turns += 1
        self.rejected_actions += 1
        self.sim_seconds += seconds
        outcome.sim_seconds = seconds
        outcome.charged = True
        return outcome

    # ── the body ─────────────────────────────────────────────────────────────

    def travel_speed_cm_s(self) -> float:
        """How fast this body is moving *now*.

        A tired courier does not stop, it slows: a hard stop turns one bad
        estimate about stamina into an episode nobody can finish, and that is not
        what running out of energy does to a rider.
        """
        speed = self.embodiment.speed_cm_s or WALK_SPEED_CM_S
        if self.stamina <= 0.0:
            speed *= self.embodiment.tired_speed_fraction
        return speed

    def _spend_stamina(self, metres: float) -> None:
        """Stamina goes on distance, not on time.

        Charging by time would make the slowest body the most tired one, which
        is backwards -- a scooter covering the same ground in a third of the
        time has done less work, not more.
        """
        if not self.enable_walking_energy:
            return
        drain = self.embodiment.stamina_per_m or 0.0
        if drain:
            self.stamina = max(0.0, self.stamina - metres * drain)

    @property
    def tired(self) -> bool:
        return self.stamina <= 0.0

    def _standing_at_a_door(self, node_id: str | None = None) -> bool:
        """At the kerb of any live job -- not merely the one in hand.

        A walk that ran past a pickup because the courier happened to be
        carrying a different job would make the block stride worse than walking
        the same ground one waypoint at a time, which is the one thing it must
        never be.

        ``node_id`` lets ``_block_preview`` ask the question about a node the
        courier has not reached yet.
        """
        here = self.position(node_id)
        return any(
            math.dist(here, order.target.kerb)
            <= self._task_action_tolerance_cm(collected=order.picked_up)
            for order in self.live_orders()
        )

    def resolve_street(self, street: str, heading: str | None = None):
        """Which street leaving this junction the courier named.

        Returns ``(k, None)`` or ``(None, refusal)``. Streets are chosen by
        name and bearing rather than by a number the observation assigns,
        because the number is only stable within one junction: Rue de Grenelle
        is street 3 here, street 1 at the next corner and absent at the one
        after. A policy given numbers cannot carry a single fact about a
        street from one corner to the next -- "I have already tried that one"
        is not expressible. A name and a bearing are the same everywhere.

        The refusals are written to be acted on. Naming a street that is not
        here lists the ones that are; naming one that is here twice asks for
        the bearing and says which two are available.
        """
        rows = self.candidates()
        # "left"/"right" mean relative to facing, not west/east.
        heading = resolve_relative(heading, self.facing())
        try:
            row = match_street(rows, street, heading)
        except StreetNotHere:
            return None, self._refuse(StepOutcome(
                ok=False, code="no_such_street",
                message=(
                    f"There is no {street} leaving this junction. From here you "
                    f"can take: {self._street_menu(rows)}."
                ),
            ))
        except StreetAmbiguous as error:
            return None, self._refuse(StepOutcome(
                ok=False, code="which_way",
                message=(
                    f"{street} leaves this junction in more than one direction "
                    f"({' and '.join(error.headings)}). Say which: "
                    f'walk_to("{street}", "{error.headings[0]}").'
                ),
            ))
        return row["k"], None

    def street_at(self, k: int) -> tuple[str, str]:
        """The name and bearing of this junction's k-th street, clockwise from north.

        The courier no longer sees these numbers -- it names streets -- but the
        reference policies and the tests still need a way to say "the first
        street here" without knowing the map. Nothing the policy can reach
        calls this.
        """
        row = next(row for row in self.candidates() if row["k"] == k)
        return row["street"], row["heading"]

    def _street_menu(self, rows: list[dict[str, Any]] | None = None) -> str:
        """The streets here, as a courier would say them back."""
        rows = self.candidates() if rows is None else rows
        return ", ".join(f'"{row["street"]}" {row["heading"]}' for row in rows)

    def walk_to(self, street: str, heading: str | None = None) -> StepOutcome:
        """Take the named street. How far one call carries is the stride.

        Two resolutions of the same city, and the difference is only where the
        courier is asked to stop and choose. See ``Stride``.
        """
        k, refusal = self.resolve_street(street, heading)
        if refusal is not None:
            return refusal
        if self.stride == Stride.BLOCK:
            return self._run_street(k, steps=None, to_corner=True)
        return self._step_to(k)

    def _step_to(self, k: int) -> StepOutcome:
        """One waypoint along street ``k``: the atomic move, whatever the stride."""
        rows = {row["k"]: row for row in self.candidates()}
        if k not in rows:
            message = (f"That street does not leave this junction. From here "
                       f"you can take: {self._street_menu()}.")
            if len(rows) == 1:
                # Naming the legal street was not enough. On three of forty
                # episodes the courier stood at a dead end and asked for street
                # 2 twenty-five, twenty-six and thirty-three times in a row --
                # the whole episode -- because the only legal move went back the
                # way it came and it would not take it. The refusal now says
                # that going back is the move, not a mistake.
                only = rows[next(iter(rows))]
                message += (f' This is a dead end. walk_to("{only["street"]}", '
                            f'"{only["heading"]}") goes back the way you came, '
                            "and here that is the only way on: take it rather "
                            "than asking again.")
            return self._refuse(StepOutcome(
                ok=False, code="no_such_street", message=message,
            ))
        row = rows[k]
        # A barrier is found the way a courier finds one: by walking up to it.
        # The refusal names it, because someone standing at a barrier can see it
        # -- but by then the turn and the time are gone, and that gap is exactly
        # what the photograph was worth. Nothing before this moment mentions it.
        if self.obstacles.blocks(self.node_id, row["node"]):
            self.blocked_attempts += 1
            # The courier is standing at the barrier. The phone still does not
            # know -- a survey does not learn -- but a person who has just been
            # stopped by a barrier can still see it on the next turn, and
            # re-offering the street as though nothing happened is not realism,
            # it is the opposite. Measured on Qwen3-VL-4B: 96 of 153 refused
            # actions were way_blocked, and in 52 of 93 cases the very next
            # action was the same street again, because the menu was identical.
            self.witnessed_blocks.add((self.node_id, row["node"]))
            # Nothing is recorded. Whatever the courier now knows about this
            # street it knows the way a person does -- it was just standing at
            # the barrier -- and remembering it is the agent's job. The phone is
            # not told, because there is nobody to tell: the route is computed
            # from a survey, and a survey does not learn.
            return self._refuse(StepOutcome(
                ok=False, code="way_blocked",
                message=(
                    # "You walk back to the junction" read as "you are where you
                    # started", and at block stride that is flatly wrong: the
                    # metres already walked are kept and the courier is standing
                    # at the barrier, which the position header says and this
                    # sentence contradicted.
                    f"{row['street']} is blocked and you cannot get past. You are "
                    "at the last junction before it. You will have to go round."
                ),
            ), seconds=BLOCKED_SECONDS)
        penalty = 0.0
        if (self.enforce_signals and self.node_id in self.signalised
                and self.signal_is_visible(self.node_id, row["node"])):
            if signal_state(self.node_id, row["bearing"], self.sim_seconds) == "red":
                # Crossing anyway. Allowed, costed, and counted -- a courier who
                # never looks will do this about half the time.
                self.red_crossings += 1
                penalty = RED_CROSSING_PENALTY
                # Held up at the kerb, and the delay counts against the deadline.
                self.sim_seconds += RED_CROSSING_PENALTY_S
        self.turns += 1
        seconds = row["distance_m"] * 100.0 / self.travel_speed_cm_s()
        self._spend_stamina(row["distance_m"])
        # A congested pavement is passable, so this is not a refusal: it is the
        # same walk, slower. The message stays silent about why, because saying
        # "you were held up by the stand on the pavement" would put the obstacle
        # into the text and hand a blind policy a free map of the city's
        # obstructions after one lap. The courier sees the clock move; working
        # out what it was is what the photograph is for.
        delay = self.obstacles.delay_seconds(self.node_id, row["node"])
        if delay:
            self.slow_passages += 1
            seconds += delay
        self.arrived_from = self.node_id
        self.node_id = row["node"]
        self.sim_seconds += seconds
        self.walked_cm += row["distance_m"] * 100.0
        self._drain_screen()
        self._issue()
        crossed_on_red = penalty > 0.0
        return StepOutcome(
            ok=True, moved=True, sim_seconds=seconds, walked_m=row["distance_m"],
            reward=-penalty,
            message=(
                f"You walk {row['distance_m']:.0f} m {row['heading']} along {row['street']}."
                + (" You crossed against the pedestrian light." if crossed_on_red else "")
            ),
        )

    # How many junctions one ``follow_street`` may cover. Matches the macro
    # declared in ``skills.py``, which is where the argument for it is written.
    MAX_FOLLOW = 6

    def follow_street(self, street: str, heading: str | None = None,
                      n: int = MAX_FOLLOW) -> StepOutcome:
        """Take the named street and keep going straight, up to ``n`` junctions.

        The turn budget and the graph were sized against different worlds. A
        delivery leg is a median 530 m; an edge on the compiled carriageway is a
        median 18 m. So one order costs about 35 ``walk_to`` calls and a ten-order
        shift about 351 -- against a step budget of 120. Even a shortest-path
        oracle with no perception cost delivered 3.1 orders before the budget ran
        out, so the cap, not the courier, was setting the score.

        Walking four junctions down one street is one decision for a rider, not
        four, and ``skills.py`` already argued the case: a macro is admissible
        exactly when it is mechanical. This one never chooses a street -- the
        caller names it -- and it stops the moment the situation stops being
        obvious: at a fork, at a dead end, when the street changes name, and at
        the door it was sent to. It cannot walk a lost courier anywhere its
        caller could not have walked one step at a time, and it cannot walk it
        past the turn it should have taken.
        """
        k, refusal = self.resolve_street(street, heading)
        if refusal is not None:
            return refusal
        return self._run_street(k, steps=max(1, min(int(n), self.MAX_FOLLOW)),
                                to_corner=False)

    def _run_street(self, k: int, *, steps: int | None, to_corner: bool) -> StepOutcome:
        """Walk street ``k`` until something worth a decision happens.

        One body for both the macro and the block stride, because they stop for
        the same reasons and only disagree about two of them. ``steps`` bounds
        the macro's reach; ``None`` is the stride, which runs to the end of the
        block however long it is. ``to_corner`` adds the stride's extra rule:
        stop wherever a choice exists, not merely where this street stops going.

        That extra rule is the whole difference between the two resolutions. The
        macro is allowed to walk straight through a side turning, because its
        caller said "stay on this street"; the stride is not, because at block
        resolution a courier who is not offered the turning cannot take it.
        """
        rows = {row["k"]: row for row in self.candidates()}
        if k not in rows:
            return self._refuse(StepOutcome(
                ok=False, code="no_such_street",
                message=(
                    f"That street does not leave this junction. From here you "
                    f"can take: {self._street_menu()}."
                ),
            ))
        street = rows[k]["street"]
        before_turns = self.turns
        before_rejected = self.rejected_actions
        first = self._step_to(k)
        if not first.ok:
            return first
        walked, seconds, reward, taken, stop = first.walked_m, first.sim_seconds, first.reward, 1, ""
        # A block on any map is bounded by the graph, but a ring road with no
        # junction on it is not, and a stride with no bound would walk it for
        # ever. One step per node is past any real block and short of a hang.
        limit = steps if steps is not None else len(self.network.nodes)
        while taken < limit:
            if self._standing_at_a_door():
                stop = " You are at the address you were looking for."
                break
            here = self.candidates()
            onward = [
                row for row in here
                if row["street"] == street and row["node"] != self.arrived_from
            ]
            if not onward:
                stop = (" The street ends here." if len(here) <= 1
                        else f" {street} does not go on from here.")
                break
            if len(onward) > 1:
                stop = f" {street} forks here."
                break
            # A crossing whose light the courier can see is a decision, so the
            # macro hands control back rather than walking through it. Without
            # this the macro quietly took the red lights its caller was being
            # charged for and never got to look at -- reintroducing, inside one
            # tool, exactly the "penalised for something you could not observe"
            # defect the visibility gate exists to remove.
            if (self.node_id in self.signalised
                    and self.signal_in_album(self.node_id, onward[0]["node"])):
                stop = " There is a pedestrian light at this crossing."
                break
            if to_corner and len(here) > 2:
                # A side turning. The macro may pass it; the stride may not, or
                # the courier is never offered a turn it was standing on.
                stop = f" A street leaves {street} here."
                break
            # Obstacles are deliberately *not* handled here, and the reason is a
            # leak rather than an oversight. Every other early stop announces
            # itself -- a fork, a dead end, a light -- so an unexplained one
            # would tell a policy that never looks at anything that there is
            # something in the road ahead, which is the one fact the pictures
            # are supposed to hold alone. Instead the walk simply runs into it
            # below and reports it from the kerb, the way it happens.
            outcome = self._step_to(onward[0]["k"])
            if not outcome.ok:
                # Whatever stopped it is reported in its own words, from where
                # the courier now stands. A macro that ended silently would be
                # the leak described above. The inner refusal booked its
                # seconds on the clock; the macro's own outcome carries them
                # too, and the walk it interrupted is an accepted action --
                # the barrier is counted in blocked_attempts, not here.
                seconds += outcome.sim_seconds
                self.rejected_actions = before_rejected
                stop = f" {outcome.message}"
                break
            walked += outcome.walked_m
            seconds += outcome.sim_seconds
            reward += outcome.reward
            taken += 1
        # One decision, one turn. Each inner ``walk_to`` charged a turn of its
        # own, so a macro whose entire argument is that four junctions down one
        # street is *one* decision for a rider was billed as four -- which made
        # the turn budget refuse to reward the thing it was introduced to allow.
        # Time is untouched: the walking still takes exactly as long.
        self.turns = before_turns + 1
        return StepOutcome(
            ok=True, moved=True, sim_seconds=seconds, walked_m=walked, reward=reward,
            message=(
                f"You walk {walked:.0f} m along {street}, through {taken} junction"
                f"{'' if taken == 1 else 's'}.{stop}"
            ),
        )

    def _look_impl(self, street: str, heading: str | None = None) -> StepOutcome:
        """The door numbers down street k, and nothing the turn already said.

        It used to open with the street's name, compass heading and distance to
        the next junction -- which is the candidate line for k, word for word,
        already on screen. Two of its three sentences could not change what the
        courier knew, so the courier's only reason to spend a turn here was the
        third, and it was buried at the end behind the restatement.

        What is left is the one thing this junction's text does not carry: how
        the numbers run *down a street the courier is not on*. That is what
        chooses a direction along a road when no phone will, and it is worth two
        seconds precisely because it is not free every turn for every street.
        """
        k, refusal = self.resolve_street(street, heading)
        if refusal is not None:
            return refusal
        row = next(row for row in self.candidates() if row["k"] == k)
        if self.condition == Condition.VISUAL:
            # The numbers are meant to be on the doors under this condition. They
            # are not legible in these renders -- see ``Condition`` -- so this
            # says what it can see rather than pretending.
            return StepOutcome(
                ok=True, sim_seconds=2.0,
                message=(
                    f"You cannot make out door numbers down {row['street']} from "
                    "here. You would have to walk it."
                ),
            )
        numbers = self.house_numbers_near(row["node"])
        here = self.house_numbers_near(self.node_id)
        if not numbers:
            return StepOutcome(
                ok=True, sim_seconds=2.0,
                message=f"No door numbers are visible down {row['street']}.",
            )
        way = ""
        # Which way the numbers run is the whole reason to look, so say it rather
        # than leaving two number ranges for the agent to difference.
        first, second = _leading_number(here), _leading_number(numbers)
        if first is not None and second is not None and first != second:
            way = " climbing" if second > first else " falling"
        return StepOutcome(
            ok=True, sim_seconds=2.0,
            message=f"Down {row['street']} the doors read {numbers}{way}.",
        )

    def _check_order_impl(self) -> StepOutcome:
        """The whole slip stack, not one slip.

        A courier holding three jobs reads all three. Reporting only the focused
        one hid the entire demand of the deeper tiers -- which job to serve next
        is the decision being scored, and it cannot be made from a description
        of one job.
        """
        live = self.live_orders()
        if not live:
            return StepOutcome(ok=True, sim_seconds=1.0, message="You have no job in hand.")
        lines = []
        for order in live:
            stage = "deliver to" if order.picked_up else "collect from"
            lines.append(
                f"Job {order.index}: {stage} {order.target.text}. "
                f"Pickup {order.pickup.text}, dropoff {order.dropoff.text}. "
                f"Fee {order.fee:.2f}. "
                f"{order.minutes_left(self.sim_seconds):.0f} min left."
                + (f" Contents: {order.category}." if order.category else "")
                + (f" Note: {order.note}" if order.note else "")
                + (" The food is cooling; it pays less cold."
                   if (self.enable_food_temperature and order.picked_up
                       and order.spoil_window_s(self.food_warm_seconds) is not None)
                   else "")
            )
        return StepOutcome(ok=True, sim_seconds=1.0 + 1.0 * len(live),
                           message="\n".join(lines))

    def _check_map_impl(self, address: str) -> StepOutcome:
        """Phone lookup: which street, how far, roughly which way.

        A direction and a distance, which is what a map app gives. Not a route:
        turn-by-turn directions would make the phone the navigator.
        """
        if self.condition in (Condition.NO_PHONE, Condition.VISUAL):
            return StepOutcome(
                ok=False, code="no_phone", sim_seconds=1.0,
                message="Your phone has no signal here. You will have to find it by the streets.",
            )
        match = self._find_address(address)
        if match is None:
            known = sorted(self.addresses_by_street)[:4]
            return StepOutcome(
                ok=False, code="unknown_address", sim_seconds=5.0,
                message=(
                    f"Your phone cannot find {address!r}. Streets you know of include "
                    f"{', '.join(known)}."
                ),
            )
        here = self.position()
        # Walking distance, not crow-flies. Straight-line range was actively
        # misleading: a six-turn trap had it improving 140 -> 105 m while the
        # route the courier would have to walk worsened 148 -> 238 m, so the
        # agent was rewarded for approaching a wall. A map app quotes route
        # distance, and so does this.
        route = self.route_length_cm(self.node_id, match.kerb_node) if match.kerb_node else None
        distance = (route if route is not None else math.dist(here, match.kerb)) / 100.0
        # Straight-line, and now *said* to be straight-line.
        #
        # The two halves of this sentence are measured in different frames and
        # always were: the distance is along the road and the bearing is as the
        # crow flies. Over 1495 sampled lookups the straight-line bearing and the
        # direction the route actually leaves in differ by a median of 46
        # degrees, by more than 90 on 22.7%, and by more than 135 -- the pin
        # lying broadly behind the courier -- on 8.4%. Only 27% named the compass
        # point the courier should walk in.
        #
        # That is not a bug in the bearing, it is a pin: a map app shows you
        # where a place *is*, and working out which street gets you there is the
        # job. It became a bug because the sentence read like an instruction. On
        # pair seed 20 -- the one easy-tier failure in 40 seeds -- the phone said
        # "to the east" for a dropoff whose route leaves south-west, and a
        # courier following the bearing walked away from the door for the rest of
        # the shift.
        #
        # Quoting the first hop of the route instead was tried and rejected: it
        # is what ``navigate()`` is for, it lifts the reference courier from 53%
        # to 70% on SHIFT by doing the navigation for it, and it would leave the
        # benchmark with two tools that both route. So the wording carries the
        # frame instead, and the courier is told which of the two numbers is a
        # direction to walk in -- neither.
        # No bearing in the words. A phone shows you where a place is by
        # drawing it, and every direction this benchmark spoke aloud was a
        # direction the policy could act on without looking at anything --
        # which made choosing a street a text problem and the photographs
        # decoration. The bearing is on the map, where a person reads it.
        return StepOutcome(
            ok=True, sim_seconds=5.0,
            message=(
                f"{match.text} is on {match.street_name}, about {distance:.0f} m "
                f"away on foot. Your phone is showing you where it is."
            ),
        )

    # How many legs of the route the phone reads out before it stops. A map app
    # shows the next few turns, not the whole itinerary, and a courier who is
    # told fourteen turns will not remember the fourteenth. Asking again is a
    # turn and 15 s, which is the cost of not paying attention.
    NAVIGATE_LEGS_SHOWN = 5
    # What a phone lookup plus reading the route off the screen costs a courier
    # standing on the pavement. Set above ``check_map``'s 5 s deliberately: the
    # route is worth more, and a policy that calls it every turn instead of
    # walking should lose the race to one that calls it once a leg.
    NAVIGATE_SECONDS = 15.0

    def _navigate_impl(self, where: str | None = None) -> StepOutcome:
        """Put a route to an address on the phone's screen.

        Takes the address, the way a person types one in:
        ``navigate("13 Avenue Dauphine")``. It used to take a job number, which
        is a thing the dispatcher knows and a phone does not, and which the
        courier had to be told separately. An address is on the slip in front
        of it. A bare ``navigate()`` still routes to the job in hand, and a
        number still selects among several jobs, because at the deeper tiers
        sequencing is the task and "route to my second job" is a real thing to
        want.

        This is the phone's map app, and it is deliberately the *only* thing in
        the environment that will tell a courier which way to go -- and it now
        does that by drawing rather than by speaking. Everything a rider does
        with their eyes stays with their eyes: it says nothing about the
        pedestrian light at the next crossing and nothing about what is in the
        way.

        A route is not a solution. The courier still has to read it off the
        screen, execute it, watch the crossings, and recognise the door.
        """
        # Addresses only. The signature says text and the implementation used
        # to also take a job number, which is the shape of mismatch this
        # benchmark keeps finding in itself: a contract stated in one place and
        # quietly widened in another. Selecting among several jobs is still
        # expressible, because each one has its own address on the slip.
        # Condition gate first, before the address branch's early returns.
        if self.condition in (Condition.NO_PHONE, Condition.VISUAL):
            return StepOutcome(
                ok=False, code="no_phone", sim_seconds=1.0,
                message="Your phone has no signal here. You will have to find it by the streets.",
            )
        if self.phone_battery is not None:
            if self.phone_battery <= 0.0:
                return self._refuse(StepOutcome(
                    ok=False, code="phone_dead", sim_seconds=1.0,
                    message=("Your phone is dead. The screen stays dark; you "
                             "will have to find it by the streets."),
                ))
            # A route request is a real look at a lit screen. Charged before
            # the route is computed, like walking charges before arriving --
            # and a phone that dies on the request dies before answering it.
            self.phone_battery = max(
                0.0, self.phone_battery - PHONE_BATTERY_NAVIGATE_PCT)
            if self.phone_battery <= 0.0:
                self.screen_route = []
                self.screen_target = None
                self._screen_route_owner = self.SCREEN_ROUTE_NONE
                self.phone_died_at_s = self.sim_seconds
                return self._refuse(StepOutcome(
                    ok=False, code="phone_dead", sim_seconds=1.0,
                    message="Your phone dies as you open the map.",
                ))
        job: int | None = None
        if isinstance(where, str) and where.strip():
            match = self._find_address(where)
            if match is None:
                known = sorted(self.addresses_by_street)[:4]
                return self._refuse(StepOutcome(
                    ok=False, code="unknown_address", sim_seconds=5.0,
                    message=(
                        f"Your phone cannot find {where}. Streets it knows "
                        f"include {', '.join(known)}."
                    ),
                ))
            order = next(
                (o for o in self.live_orders()
                 if o.target.text.strip().lower() == match.text.strip().lower()),
                None)
            if order is None:
                # A real map app routes anywhere; it does not check your job
                # list first. The screen goes to the address asked for.
                return self._route_to(match)
            job = order.index
        live = self.live_orders()
        if not live:
            return StepOutcome(ok=True, sim_seconds=1.0, message="You have no job in hand.")
        if job is None:
            order = self.active_order()
        else:
            # Which job to route to is the courier's decision, not the phone's.
            # With several live orders the sequencing *is* the task at the deeper
            # tiers, and a navigation tool that only ever routes to the oldest
            # job would quietly make that decision on the agent's behalf.
            order = next((o for o in live if o.index == int(job)), None)
            if order is None:
                return StepOutcome(
                    ok=False, code="no_such_job", sim_seconds=1.0,
                    message=(
                        f"You are not carrying job {job}. In hand: "
                        f"{', '.join('job ' + str(o.index) for o in live)}."
                    ),
                )
        if order is None:
            return StepOutcome(ok=True, sim_seconds=1.0, message="You have no job in hand.")
        return self._route_to(order.target, collected=order.picked_up)

    def _route_to(
        self, target: Address, *, collected: bool | None = None,
    ) -> StepOutcome:
        """Put a route to one address on the screen, whoever asked for it."""
        gap = math.dist(self.position(), target.kerb)
        tolerance_cm = (
            ARRIVAL_TOLERANCE_CM
            if collected is None
            else self._task_action_tolerance_cm(collected=collected)
        )
        if gap <= tolerance_cm:
            return StepOutcome(
                ok=True, sim_seconds=self.NAVIGATE_SECONDS,
                message=f"You have arrived: {target.text} is right here.",
            )
        legs = self.route_legs(self.node_id, target.kerb_node) if target.kerb_node else None
        if not legs:
            return StepOutcome(
                ok=False, code="no_route", sim_seconds=self.NAVIGATE_SECONDS,
                message=(
                    f"Your phone cannot plot a route to {target.text} from here. "
                    "Walk to a bigger street and ask again."
                ),
            )
        metres = sum(leg["metres"] for leg in legs)
        minutes = metres * 100.0 / WALK_SPEED_CM_S / 60.0
        # The route is drawn, not dictated. Every leg used to be spelled out
        # -- "Take Rue de Grenelle, east, 1 junction, 18 m" -- and a courier
        # holding that text never had to look at anything: the street to take
        # was named, the bearing was named, and the photographs and the map
        # were both decoration. Choosing a street was a reading exercise.
        #
        # What the phone says now is what a phone says when you glance at it
        # without stopping: how far, how long, and that it is on the screen.
        # Which way to go is on the map, which is a picture. The names of the
        # streets are on the corner, which is text. Whether the way is open,
        # and whether the light is red, are in the photographs. No one of them
        # is enough.
        lines = [
            f"Route to {target.text} — {metres:.0f} m, about {minutes:.0f} min "
            f"on foot, {len(legs)} street{'' if len(legs) == 1 else 's'} to walk.",
            "  Your phone is showing the route. Read it off the map: the line "
            "runs from where you are to where you are going.",
        ]
        drawing = self.map_drawing(target, legs).svg
        self._screen_route_owner = self.SCREEN_ROUTE_EXPLICIT
        return StepOutcome(
            ok=True, sim_seconds=self.NAVIGATE_SECONDS, message="\n".join(lines),
            drawing=drawing,
        )

    def map_drawing(self, target: Address | None = None,
                    legs: list[dict[str, Any]] | None = None) -> MapDrawing:
        """The picture on the phone's screen: streets, the route, the pin, you.

        Drawn from the compiled network, so it says what a survey knows and
        nothing else. It cannot show a light, a barrier or a shopfront -- not
        because that would be hard, but because the phone cannot see the street,
        and that separation is the reason ``report_blocked`` exists at all. The
        Nothing on it came from the courier's eyes. There is no way to put a
        barrier on this map, because there is no way to tell the map about one.

        With no ``target``, this is the screen as it stands: the last route the
        courier asked for, still drawn where it was drawn, with the courier's
        current position and heading on it. Pass a target to compute a fresh
        route -- which is what ``navigate`` does, and what it charges for.
        """
        route: list[tuple[float, float]] = []
        if target is None:
            # Re-snapped to where the courier is standing, the way a navigation
            # app does every second. It used to redraw the route exactly as it
            # was when navigate() was last called, so the line stayed put while
            # the courier walked along it -- harmless while the picture was
            # only a line, and wrong the moment a banner started naming the
            # next street off it: after one step it named the street behind.
            target = self.screen_target
            if target is not None and target.kerb_node:
                path = self.route_nodes(self.node_id, target.kerb_node) or []
                route = self._map_route_points(path)
                self.screen_route = list(route)
            else:
                route = list(self.screen_route)
        elif target.kerb_node:
            path = self.route_nodes(self.node_id, target.kerb_node) or []
            route = self._map_route_points(path)
            self.screen_target, self.screen_route = target, list(route)
        return render_map(
            self.network,
            here=self.position(),
            facing_deg=self.facing(),
            route=route,
            destination=(target.kerb if target is not None else None),
            destination_label=(target.text if target is not None else ""),
            here_label="you are here",
            # The banner names the street the route takes next. A navigation
            # app puts it there because a bearing read off a drawn line is the
            # hardest thing on the screen; measured here, a model reads a
            # street name off this map 100% of the time and a direction 25%.
            **self._next_instruction(route),
        )

    def _map_route_points(self, path: list[str]) -> list[tuple[float, float]]:
        """Turn a topological path into the line drawn from the courier.

        Named-node lookups deliberately stay graph geometry, but the first
        point is the courier's current position.  Those are identical in the
        stock street action space.  In pose-tracked spaces the pawn can stand
        between nodes; leaving the graph node at the start made the blue line
        begin metres away from the ``you are here`` puck even though both were
        rendered in the same SVG.  Replacing only the first point preserves
        the routed graph edges and the next-node banner while joining the line
        to the position the camera and task actions actually use.
        """
        if not path:
            return []
        return [self.position(), *(self.position(node) for node in path[1:])]

    def _next_instruction(self, route: list[tuple[float, float]]) -> dict[str, str]:
        """The banner: the street to take next, and the bearing to take it at.

        Both copied off the candidate row the courier will act on, not
        derived a second way. Deriving the bearing separately -- as the compass
        of the step to the route's next node -- disagreed with the list on 6
        frames in 67, and on one of them it said south-east where the list
        offered the same street going north-west. A banner the courier cannot
        copy verbatim into walk_to is worse than no banner, because it reads
        as the map contradicting the corner.
        """
        if len(route) < 2:
            return {"next_street": "", "next_heading": ""}
        # The bearing has to be the one the candidate line quotes -- at block
        # stride that is the whole block's, not the first hop's -- but it must
        # be computed here rather than read off candidates(). candidates()
        # looks up a street view and a lamp for every way out, and under the
        # live renderer a lookup is a render request: drawing the map, which is
        # a survey drawing that has never seen the street, was issuing a batch
        # of renders. Three live tests caught it; the cost is real either way.
        for row in self._raw_candidates():
            if math.dist(self.position(row["node"]), route[1]) < 1.0:
                # The row's own first-edge bearing, exactly as the candidate
                # line prints it and match_street accepts it. This recomputed
                # the whole block's compass for a while, to match a candidate
                # line that then printed the block's -- but the block bearing
                # differs from the accepted one on 5.6% of rows, so the banner
                # was verbatim-copyable except when it was not. One string,
                # everywhere.
                heading = row.get("heading") or ""
                return {"next_street": str(row["street"]),
                        "next_heading": str(heading)}
        nearest = min(self.network.nodes,
                      key=lambda n: math.dist(self.position(n), route[1]))
        return {"next_street": self.street_of(nearest) or "",
                "next_heading": compass_of(
                    bearing_deg(self.position(), route[1]))}

    def _find_address(self, text: str) -> Address | None:
        wanted = " ".join(str(text).split()).lower()
        for address in self.network.addresses:
            if address.text.lower() == wanted:
                return address
        for address in self.network.addresses:
            if wanted and wanted in address.text.lower():
                return address
        return None

    def _task_action_tolerance_cm(self, *, collected: bool) -> float:
        """Distance at which pickup/drop-off task actions become valid.

        The stock benchmark uses one shared tolerance. Specialised embodied
        environments may tighten one phase without duplicating the collection
        and payment state machines.
        """
        return ARRIVAL_TOLERANCE_CM

    def _order_at_hand(self, collected: bool) -> Order | None:
        """A live order whose next stop is the door the courier is standing at.

        Serving *whichever* job is here, rather than only the focused one, is
        what makes a queue a queue: the courier's route is its plan, and a plan
        that batches two nearby stops has to be executable without a tool for
        saying so. Nearest first, so two doors inside the tolerance resolve the
        way a person would resolve them.
        """
        here = self.position()
        candidates = [
            (math.dist(here, (o.dropoff if collected else o.pickup).kerb), o.index, o)
            for o in self.live_orders() if o.picked_up == collected
        ]
        tolerance_cm = self._task_action_tolerance_cm(collected=collected)
        candidates = [c for c in candidates if c[0] <= tolerance_cm]
        return min(candidates)[2] if candidates else None

    def _nearest_live(self, *, collected: bool) -> Order | None:
        """The live job whose door is closest, for a refusal that names it.

        Falling back to ``active_order`` meant a courier standing one junction
        short of job 1's pickup was told "you are not at 19 Boulevard du Temple"
        -- job 0's address, 71 m the other way. The message named a place the
        courier was not going and said nothing about the one it was, which is
        the opposite of what a refusal is for.
        """
        live = [o for o in self.live_orders() if o.picked_up == collected]
        if not live:
            return None
        here = self.position()
        return min(live, key=lambda o: math.dist(
            here, (o.dropoff if collected else o.pickup).kerb))

    def collect(self) -> StepOutcome:
        order = (self._order_at_hand(collected=False) or self._nearest_live(collected=False)
                 or self.active_order())
        if order is None:
            return self._refuse(StepOutcome(
                ok=False, code="no_order", message="You have no job in hand."))
        if order.picked_up:
            return self._refuse(StepOutcome(
                ok=False, code="already_collected",
                message="You already have this order."))
        gap = math.dist(self.position(), order.pickup.kerb)
        if gap > self._task_action_tolerance_cm(collected=False):
            return self._refuse(StepOutcome(
                ok=False, code="not_at_pickup",
                message=f"You are not standing at {order.pickup.text}.",
            ))
        self.turns += 1
        order.picked_up = True
        # Handling time is charged here, like walking charges its own. Both
        # ``collect`` and ``hand_over`` declared 30 s and neither advanced the
        # clock, so 60 s a delivery -- 10 minutes over a ten-order shift -- was
        # free, and ``_make_orders`` had already budgeted for it when it sized
        # the deadlines. A tool must not declare a cost it does not pay; that is
        # what ``_charge`` exists to guarantee, and these two bypass it.
        # Handling, plus whatever this body costs to stop with: nothing on
        # foot, a stand and a lock on a scooter, somewhere to leave a car.
        # Without that a car strictly dominates and the vehicle is not a
        # choice.
        self.sim_seconds += HANDLING_SECONDS + self.embodiment.stop_overhead_s
        # The counter clock for the food, started once the parcel is actually
        # in the bag -- i.e. after the handling that put it there.
        order.picked_up_at_s = self.sim_seconds
        self._issue()
        # Collection changes which carried job active_order() focuses on.
        # Refresh for that state transition, not from ordinary queue reads:
        # those must preserve a route the courier explicitly chose with
        # navigate(), and a newly carried job may not be the queue's first one.
        self._light_the_screen(reclaim=True)
        if not self.enable_food_temperature:
            freshness = ""
        elif order.category == "ice cream":
            freshness = " The ice cream is frozen; it will not stay that way."
        elif order.category == "groceries":
            freshness = " Groceries -- nothing in the bag spoils."
        else:
            freshness = " The food is hot; it will not stay that way."
        return StepOutcome(
            ok=True, sim_seconds=HANDLING_SECONDS, reward=0.1,
            message=f"You collect the order from {order.pickup.text}." + freshness)

    def hand_over(self) -> StepOutcome:
        order = (self._order_at_hand(collected=True) or self._nearest_live(collected=True)
                 or self.active_order())
        if order is None:
            return self._refuse(StepOutcome(
                ok=False, code="no_order", message="You have no job in hand."))
        if not order.picked_up:
            return self._refuse(StepOutcome(
                ok=False, code="not_collected",
                message=f"You have not collected it yet, from {order.pickup.text}."))
        gap = math.dist(self.position(), order.dropoff.kerb)
        if gap > self._task_action_tolerance_cm(collected=True):
            return self._refuse(StepOutcome(
                ok=False, code="not_at_dropoff",
                message=f"You are not standing at {order.dropoff.text}.",
            ))
        self.turns += 1
        # The handover takes 30 s and the customer is not served until it is
        # done, so the clock moves before lateness is judged. See ``collect``.
        # Handling, plus whatever this body costs to stop with: nothing on
        # foot, a stand and a lock on a scooter, somewhere to leave a car.
        # Without that a car strictly dominates and the vehicle is not a
        # choice.
        #
        # A note changes what the door costs. The effect is applied by the
        # world, not chosen by the courier -- reading the note buys the
        # *planning* information (this door is quick, that one is slow), which
        # is what a slip is for.
        handling = HANDLING_SECONDS
        note_line = ""
        if self.enable_special_notes and order.note == NOTE_LEAVE_AT_DOOR:
            handling = NOTE_DOOR_HANDLING_S
            self.notes_followed += 1
            note_line = " You leave it at the door, as the note asked."
        elif self.enable_special_notes and order.note == NOTE_RING_FIRST:
            handling = HANDLING_SECONDS + NOTE_RING_EXTRA_S
            self.notes_followed += 1
            note_line = " You ring and wait for the customer to come down."
        self.sim_seconds += handling + self.embodiment.stop_overhead_s
        order.delivered = True
        order.delivered_at_s = self.sim_seconds
        on_time = self.sim_seconds <= order.due_at()
        # A late parcel is still a parcel, and it is still worth something --
        # but not what an on-time one is worth. Paying the full fee whatever the
        # clock said made the deadline decorative under an objective that is
        # explicitly "maximise profit": a policy could ignore every window and
        # lose nothing it was scored on.
        paid = order.fee if on_time else order.fee * LATE_FEE_FRACTION
        # Spoilage multiplies with late rather than replacing it: they are two
        # different failures with two different clocks, and a delivery can
        # commit both. The window and the discount come from what is in the
        # bag -- the plain temperature flag treats everything as a hot meal;
        # categories give ice cream a shorter fuse and groceries none.
        window = order.spoil_window_s(self.food_warm_seconds)
        cold = (self.enable_food_temperature
                and window is not None
                and order.picked_up_at_s is not None
                and self.sim_seconds - order.picked_up_at_s > window)
        spoil_line = ""
        if cold:
            paid *= order.spoil_fraction()
            if order.category == "ice cream":
                self.melted_deliveries += 1
            else:
                self.cold_deliveries += 1
            spoil_line = (f", and the {order.category or 'food'} has gone "
                          f"{order.spoil_word()}")
        order.paid = round(paid, 2)
        self.earnings += order.paid
        # Take another job the moment a hand is free, so a queue that is meant
        # to be ``queue_depth`` deep actually stays that deep.
        self._issue()
        self._light_the_screen(reclaim=True)
        self.finished = not self.live_orders() or self.shift_over
        return StepOutcome(
            ok=True, sim_seconds=handling,
            reward=1.0 + (0.5 if on_time else -0.5),
            finished=self.finished,
            message=(
                f"You hand the order to the customer at {order.dropoff.text}"
                f"{'' if on_time else ', late'}"
                f"{spoil_line}."
                f" You are paid {order.paid:.2f}.{note_line}"
            ),
        )

    WAIT_SECONDS = 15.0

    def wait(self) -> StepOutcome:
        """Wait where you are -- at a crossing, for the light.

        At a signalised junction this waits *for the phase*, not for a fixed 15
        seconds. A flat 15 s against a 60 s phase meant one ``wait()`` usually
        left the light exactly as red as it was, so obeying a light cost between
        one and four turns and the courier could not tell in advance which. That
        made the only tool the photographs exist to support unusable in practice:
        a policy that could read the lamp still had no reliable way to act on it.
        Waiting out the phase is one turn, always works, and costs the time it
        really costs -- 0 to 60 s, 30 s on average.
        """
        if self.node_id in self.signalised:
            self.waits_at_red += 1
            phase_end = (math.floor(self.sim_seconds / SIGNAL_PHASE_S) + 1) * SIGNAL_PHASE_S
            seconds = max(phase_end - self.sim_seconds, 1.0)
        else:
            seconds = self.WAIT_SECONDS
        self.turns += 1
        self.sim_seconds += seconds
        self._issue()
        return StepOutcome(ok=True, sim_seconds=seconds,
                           message=f"You wait {seconds:.0f} s.")

    REST_SECONDS = 60.0

    def rest(self) -> StepOutcome:
        """Spend a minute to get energy back.

        The counterpart to the stamina drain, and without it the drain is not a
        resource but a decay: an hour-long shift simply got slower and there was
        nothing for the courier to decide. With it, a small tank and a fast drain
        cost *turns* -- which is what makes one body different from another in
        the only currency the benchmark reports.

        A full tank's worth is recovered per rest, so one call is always enough,
        for the same reason ``wait()`` sees the whole phase out: a tool the agent
        has to guess the repeat count of is a tool it cannot plan with.
        """
        capacity = float(self.embodiment.stamina or 0.0)
        # Refuse on the same condition the tool is gated on, or a body that is
        # not offered a rest can still take one by guessing the name.
        if not (capacity and self.embodiment.stamina_per_m
                and self.enable_walking_energy):
            return self._refuse(StepOutcome(
                ok=False, code="never_tires",
                message="You are not the one doing the work; there is nothing to rest.",
            ))
        before = self.stamina
        self.stamina = capacity
        self.turns += 1
        self.sim_seconds += self.REST_SECONDS
        self.rests += 1
        self._issue()
        return StepOutcome(
            ok=True, sim_seconds=self.REST_SECONDS,
            message=(
                f"You stop for a minute. You feel ready again."
                if before < capacity else
                "You stop for a minute, though you were not tired."
            ),
        )

    def charge_phone(self) -> StepOutcome:
        """Ninety seconds on the power bank, forty points of charge back.

        The counterpart to the battery drain, exactly as ``rest`` is to the
        stamina drain and for the same reason: a resource that only empties is
        a countdown, not a decision. A dead phone comes back to life -- and the
        screen relights on the job in hand, because a courier who just revived
        their phone is looking at it.
        """
        if not (self.enable_phone_battery and self.enable_phone_recharge):
            return self._refuse(StepOutcome(
                ok=False, code="nothing_to_charge",
                message="You have nothing to charge it with.",
            ))
        was_dead = (self.phone_battery or 0.0) <= 0.0
        before = self.phone_battery or 0.0
        self.phone_battery = min(100.0, before + PHONE_RECHARGE_PCT)
        self.turns += 1
        self.sim_seconds += PHONE_RECHARGE_SECONDS
        self.phone_recharges += 1
        self._issue()
        if was_dead:
            # Reclaim: the screen is being revived, not merely refreshed, so
            # an explicit route the dead phone still remembered must not veto it.
            self._light_the_screen(reclaim=True)
        return StepOutcome(
            ok=True, sim_seconds=PHONE_RECHARGE_SECONDS,
            message=(
                f"You stand with the power bank for a minute and a half. "
                f"The phone is at {self.phone_battery:.0f}%."
                + (" The screen comes back on." if was_dead else "")
            ),
        )

    def light_here(self, k: int) -> str | None:
        """Ground truth for the light facing candidate ``k``.

        Privileged: it scores the run and picks which baked frame to serve. It is
        never rendered into the observation text.
        """
        if self.node_id not in self.signalised:
            return None
        rows = {row["k"]: row for row in self.candidates()}
        if k not in rows:
            return None
        return signal_state(self.node_id, rows[k]["bearing"], self.sim_seconds)

    def allowed_tool_names(self) -> list[str]:
        """Tool names this condition permits, for building the prompt.

        The condition gated ``check_map`` at call time but never reached the tool
        menu, so a no-phone episode still opened by telling the courier to use a
        phone it did not have -- and the policy duly wasted its first turn on it.
        """
        from embodiedbench.agent.courier.tools import available_tools

        env_actions = ["VIEW_ORDERS", "ACCEPT_ORDER", "PICKUP", "DROP_OFF", "WAIT"]
        env_actions += list(self.movement_env_actions())
        # A body that never tires must not be offered a rest: a tool that
        # can only be refused is a turn the agent is invited to lose. The
        # walking-energy flag turns the tiring off the same way an untiring
        # body would, so the menu follows it.
        if (self.embodiment.stamina and self.embodiment.stamina_per_m
                and self.enable_walking_energy):
            env_actions.append("REST")
        # The power bank enters the menu with its flag, like REST with its
        # drain: a chargeable battery without the tool is a countdown, and the
        # tool without the battery is a guaranteed refusal.
        if self.enable_phone_battery and self.enable_phone_recharge:
            env_actions.append("CHARGE_PHONE")
        allow_consult = self.condition == Condition.FULL
        offered = available_tools(env_actions, allow_consult=allow_consult,
                                  limits=self.tool_limits())
        if self.stride == Stride.BLOCK:
            # ``follow_street`` exists to spend one turn on several waypoints of
            # one street. Under this stride ``walk_to`` already does that, and
            # offering both would put two names on one action -- the same defect
            # the tool audit retired ``read_sign`` for, arrived at from the other
            # direction.
            return [t.name for t in offered if t.name != "follow_street"]
        # ``note`` is back, and this time it runs. It was removed because it
        # was advertised with no executor anywhere -- the defect the
        # UNIMPLEMENTED_TOOLS split exists to make impossible. CourierSession
        # now executes it itself, because a notebook is the courier's and not
        # the city's, so the name is dispatchable again.
        names = [t.name for t in offered]
        if allow_consult and "note" not in names:
            names.append("note")
        return names

    def movement_env_actions(self) -> tuple[str, ...]:
        """The environment actions that move the courier.

        A hook rather than a constant because the coordinate action space is
        chosen at construction and has to reach the menu: an environment
        offers ``MOVE_TO`` or ``MOVE_TO_XY``, never both, and everything that
        follows from that -- which tools are described, which worked example
        the prompt shows, which line sits under the list of streets -- falls
        out of this one answer. See ``EmbodiedCourierEnv``.
        """
        return ("MOVE_TO",)

    def tool_limits(self) -> dict[str, Any]:
        """The environment's own numbers, for the manuals that quote them."""
        return {}

    def tools_for_prompt(self) -> list[Any]:
        """The tool objects behind ``allowed_tool_names``, manuals filled in.

        The session used to look each allowed name up in ``TOOLS_BY_NAME``,
        which hands back the module-level declaration -- the one whose limits
        are still ``{placeholders}``. Names alone were enough while no tool's
        manual quoted a number the environment owned.
        """
        from embodiedbench.agent.courier.tools import TOOLS_BY_NAME

        limits = self.tool_limits()
        return [TOOLS_BY_NAME[name].filled(**limits)
                for name in self.allowed_tool_names()]

    def summary(self) -> dict[str, Any]:
        """Everything needed to judge a run, including how it was judged.

        Three things were missing and each hid a different failure. Route
        quality: only the oracle counted the metres it walked, so a policy that
        delivered by wandering twice as far scored the same as one that went
        straight there. Wasted effort: refused actions were invisible, so a
        policy spending a third of its turns walking into walls read as clean.
        And rate: ENDLESS is scored on money against a fixed hour, and money per
        hour is the objective itself -- reporting the total without the clock it
        was earned against makes two runs of different lengths look comparable.
        """
        hours = self.sim_seconds / 3600.0 if self.sim_seconds > 0 else 0.0
        walked_m = self.walked_cm / 100.0
        done_seconds, done_walk_cm = self.delivered_optimal()
        return {
            "node": self.node_id, "street": self.street_of(self.node_id) if self.node_id else "",
            "sim_seconds": round(self.sim_seconds, 1), "earnings": round(self.earnings, 2),
            # The score for ENDLESS, and an honest ledger everywhere else: what
            # the shift actually paid, after late jobs are discounted and
            # abandoned ones pay nothing.
            "profit": round(self.earnings, 2),
            "orders": [o.to_dict() for o in self.orders],
            # The body, and whether it was shown its own viewpoint. A run where
            # ``viewpoint_matches_embodiment`` is false is still a run, but it is
            # not evidence about a policy that has to work from a pavement.
            "embodiment": self.embodiment.name,
            "viewpoint_expected": self.embodiment.viewpoint,
            "viewpoint_served": self.viewpoint_served,
            "viewpoint_matches_embodiment": self.viewpoint_matches_embodiment,
            "rests": self.rests,
            "stamina_left": round(self.stamina, 2),
            "stamina_spent": round(
                float(self.embodiment.stamina or 0.0) - self.stamina, 2),
            "tired": self.tired,
            "difficulty": self.difficulty, "turns": self.turns,
            "queue_depth": self.queue_depth,
            "minutes_per_delivery": (
                round(self.sim_seconds / 60.0 / self.delivered_count, 2)
                if self.delivered_count else None
            ),
            "turns_per_delivery": (
                round(self.turns / self.delivered_count, 1)
                if self.delivered_count else None
            ),
            "delivered": self.delivered_count, "on_time": self.on_time_count,
            "late": self.late_count, "expired": self.expired_count,
            "orders_issued": self.issued_count,
            "rejected_actions": self.rejected_actions,
            "walked_m": round(walked_m, 1),
            # Optimal for the deliveries *actually completed*, not for the shift
            # as issued -- so it is only a yardstick once something has arrived.
            # It read 0.0 mid-episode, which looks like a measured bound of zero
            # and made ``walked_m vs optimal_walk_m`` nonsense on any live
            # dashboard. ``None`` says "not yet defined", which is the truth.
            "optimal_walk_m": (round(done_walk_cm / 100.0, 1)
                               if done_walk_cm else None),
            # Metres walked against metres a perfect courier would have walked
            # for the very same deliveries. 1.0 is a straight line to every
            # door; 2.0 means half the shift was spent lost. Below 1.0 is legal
            # and informative -- it is a queue served in a better order than the
            # one it was drawn in.
            "walk_ratio": (round(walked_m / (done_walk_cm / 100.0), 2)
                           if done_walk_cm else None),
            "optimal_seconds": round(done_seconds, 1),
            "time_ratio": (round(self.sim_seconds / done_seconds, 2)
                           if done_seconds else None),
            "shift_optimal_seconds": round(self.optimal_seconds, 1),
            "deliveries_per_hour": (round(self.delivered_count / hours, 2) if hours else None),
            "earnings_per_hour": (round(self.earnings / hours, 2) if hours else None),
            "shift_seconds": self.shift_seconds, "shift_over": self.shift_over,
            "red_crossings": self.red_crossings, "waits_at_red": self.waits_at_red,
            "signalised_junctions": len(self.signalised),
            # The obstacle ledger. ``blocked_attempts`` is the number a policy
            # that looks at its photographs drives to zero and one that does not
            # cannot: every one of them is a wasted turn and 45 s that the same
            # route, chosen from the same information, would not have cost.
            "blocked_attempts": self.blocked_attempts,
            "slow_passages": self.slow_passages,
            "obstacles": self.obstacles.counts(),
            # The optional-constraint ledger: which flags this run was scored
            # under, and what each one cost. All-defaults reads exactly like a
            # run from before the flags existed, except for these keys.
            "constraints": {
                "earning_jitter": self.enable_earning_jitter,
                "food_temperature": self.enable_food_temperature,
                "special_notes": self.enable_special_notes,
                "walking_energy": self.enable_walking_energy,
                "phone_battery": self.enable_phone_battery,
                "food_categories": self.enable_food_categories,
                "phone_recharge": self.enable_phone_recharge,
            },
            "cold_deliveries": self.cold_deliveries,
            "melted_deliveries": self.melted_deliveries,
            "phone_recharges": self.phone_recharges,
            "notes_followed": self.notes_followed,
            "phone_battery_left": (None if self.phone_battery is None
                                   else round(self.phone_battery, 1)),
            "phone_died_at_s": (None if self.phone_died_at_s is None
                                else round(self.phone_died_at_s, 1)),
            "finished": self.finished,
        }


def load_city(map_dir: Path, **kwargs: Any) -> CourierEnv:
    """Compile a map and open a courier world on it."""
    network = build_road_network(Path(map_dir), map_name=Path(map_dir).name)
    return CourierEnv(network, **kwargs)


def _charged(name: str):
    """Public tool that charges its own declared time."""

    def call(self, *args, **kwargs):
        return self._charge(getattr(self, f"_{name}_impl")(*args, **kwargs))

    call.__name__ = name
    return call


for _name in ("look", "check_order", "check_map", "navigate"):
    setattr(CourierEnv, _name, _charged(_name))
