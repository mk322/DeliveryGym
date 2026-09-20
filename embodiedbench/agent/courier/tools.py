"""The courier's tools: everything the agent can *do* or *ask*.

mini-SWE-agent gives its agent exactly one tool -- bash -- because a shell is
already a universal interface to a computer. An embodied courier has no such
universal verb, so the tool set has to be designed, and the design question is
what a real delivery rider actually has on the job.

A rider has three distinct kinds of affordance, and conflating them is what made
the first version of this environment unusable:

**Act.** Things that change the world and consume time: walking to the next
junction, picking up a bag, handing it over. These are irreversible, they cost
the step budget, and they can fail for physical reasons.

**Look.** Things that change only what the rider knows: turning to look down a
street, reading the shopfront in front of them. A rider can look around for free
before committing to a turn, and denying that is what forced the earlier policy
to guess.

**Consult.** Things that query knowledge the rider carries: the order slip in
their pocket, the map app on their phone. A phone lookup is not perception --
it works around a corner and in the dark -- but it is also not free, and a
simulation that makes it free trains an agent that never looks up.

Each tool declares which kind it is, what it costs, and what it can return,
because the harness needs that to charge budgets correctly and the task layer
needs it to decide which tools an episode allows. A ``consult`` tool is the one
a benchmark may want to switch off to make a condition harder; an ``act`` tool
never can be.

Tools are *declared here and dispatched by name*, so the prompt the model sees
and the code that executes are generated from one source. A tool the environment
does not enable is not described in the prompt, which is what stopped the old
observation telling the agent to use MOVE on a map where MOVE was disabled.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Callable


class _Blanks(dict):
    """Leaves a placeholder nobody supplied exactly as it was written.

    So that filling in one number does not turn every other brace in the prose
    into a ``KeyError`` -- and so that an unsupplied one survives to be caught
    by the check in ``available_tools`` rather than vanishing.
    """

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _fill(text: str, values: dict[str, Any]) -> str:
    return text.format_map(_Blanks(values)) if text else text


class ToolKind(str, Enum):
    """What a tool does to the world, the agent's knowledge, or neither."""

    ACT = "act"          # changes the world, costs simulated time
    LOOK = "look"        # changes what the agent knows, from where it stands
    CONSULT = "consult"  # queries carried knowledge: the order slip, the phone


# ─────────────────────────────────────────────────────────────────────────────
# What the free observation already says
# ─────────────────────────────────────────────────────────────────────────────
#
# An audit of the 25 recorded trajectories found the reference courier using
# four of eleven tools, and two of the seven it never touched turned out to be
# unable to tell it anything:
#
#     observation   "You are on Quai Beaubourg, outside number 2."
#     read_sign()   "The sign says Quai Beaubourg. The doors here are numbered 2."
#
# read_sign was a three-second, one-turn way to be told what the turn already
# said. look(k) was three quarters of the same thing -- its street name, compass
# and distance are the candidate line verbatim -- with one genuinely new fact
# buried at the end.
#
# A menu entry that cannot change what the agent knows is worse than useless:
# it costs prompt, it costs a turn when taken, and it teaches a policy that
# actions need not pay. So the redundancy is made checkable rather than
# remembered. Each knowledge tool declares the facts a call returns, the
# observation declares the facts it states for free, and ``available_tools``
# drops any tool that has nothing left. ``tests/test_courier_harness.py``
# asserts the menu contains no such tool, so the next one cannot be added
# quietly.
#
# Names are facts, not sentences: two tools that both return "how far the next
# junction is" must use one name, or the check cannot see they overlap.
FACT_STREET_HERE = "street_here"          # which street the courier stands on
FACT_NUMBERS_HERE = "numbers_here"        # door numbers at this junction
FACT_STREET_NAMES = "street_names"        # the names of the streets leaving it
FACT_HEADINGS = "headings"                # compass and left/right for each
FACT_DISTANCES = "distances"              # metres to each next junction
FACT_NUMBERS_AHEAD = "numbers_ahead"      # door numbers down a street not taken
FACT_ORDER_ADDRESSES = "order_addresses"  # both ends of every live job
FACT_ORDER_FEE = "order_fee"
FACT_DEADLINES = "deadlines"
FACT_ROUTE_DISTANCE = "route_distance"    # how far, on foot, to a named address
FACT_TARGET_BEARING = "target_bearing"    # which way it lies, as the crow flies
FACT_TURN_BY_TURN = "turn_by_turn"        # street, turn and distance, leg by leg

# Stated every turn, in every condition, without spending anything. Note what is
# NOT here: nothing about a light, an obstacle, or a shopfront. Those are in the
# photographs and in no tool at all, which is what makes looking load-bearing.
OBSERVATION_PROVIDES: frozenset[str] = frozenset({
    FACT_STREET_HERE, FACT_NUMBERS_HERE, FACT_STREET_NAMES,
    FACT_HEADINGS, FACT_DISTANCES, FACT_DEADLINES,
})


@dataclass(frozen=True)
class ToolParam:
    name: str
    type: str
    description: str
    required: bool = True


@dataclass
class Tool:
    """One thing the courier can do, described once for both prompt and dispatch."""

    name: str
    kind: ToolKind
    summary: str
    params: tuple[ToolParam, ...] = ()
    example: str = ""
    # Simulated seconds a call costs. Looking is quick but not free -- a rider
    # who spins on the spot every turn is not delivering.
    time_cost_s: float = 0.0
    # Whether a call counts against the step budget. Consulting the phone does
    # not move the rider, but it must still cost something or the optimal policy
    # is to query forever.
    counts_as_step: bool = True
    requires_env_action: str | None = None
    # The facts a call returns, from the vocabulary above. Empty for a tool that
    # acts rather than informs -- ``collect`` earns its place by what it does.
    provides: frozenset[str] = frozenset()

    # ── the manual ───────────────────────────────────────────────────────────
    #
    # A one-line summary and an example told the courier what a tool is called
    # and not what it does with what it is given. Measured over 1206 turns of
    # Qwen3-VL-4B: 1139 of them were walk_to and three were look, against 47
    # no_such_street refusals -- a courier with one hammer, repeatedly told the
    # street it named is not here. The three things it was never told are the
    # three below: what a call gives back, what a refusal means and what to do
    # about it, and when the tool is the wrong one.
    #
    # They live on the tool rather than in the prose because the prose is
    # generated per condition: a tool the environment has taken away must take
    # its manual with it, exactly as it takes its menu line.
    returns: str = ""
    # (refusal wording as the courier sees it, what to do about it)
    refusals: tuple[tuple[str, str], ...] = ()
    use_when: str = ""
    not_for: str = ""
    # A second worked call, for the chunked prompt's two-line example. It has to
    # come off the tool for the same reason the first one does: the example is
    # part of the manual, and a hard-coded second line demonstrates a tool the
    # environment may not have.
    example2: str = ""

    def filled(self, **values: Any) -> "Tool":
        """This manual with the environment's own numbers written into it.

        A tool whose behaviour is bounded by a configured number has to state
        that number, and stating it twice -- once in the prose here, once in
        the env that enforces it -- is how a prompt comes to promise a limit
        the runtime does not keep. So the prose carries ``{placeholder}`` and
        the env supplies the value; ``available_tools`` refuses to hand back a
        manual with a placeholder still in it.
        """
        if not values:
            return self
        return replace(
            self,
            summary=_fill(self.summary, values),
            returns=_fill(self.returns, values),
            use_when=_fill(self.use_when, values),
            not_for=_fill(self.not_for, values),
            refusals=tuple((wording, _fill(remedy, values))
                           for wording, remedy in self.refusals),
        )

    def manual(self) -> str:
        """The full entry for this tool: call, result, refusals, judgement.

        Every fact stays -- what to pass, what comes back, every refusal and
        its remedy, when to reach for it and when not to -- in the fewest
        words that still say it. The entries are ~45% of the system prompt
        and ride in front of every request of every turn, so each word here
        is paid for hundreds of times per episode.
        """
        out = [f"{self.typed_signature()}", f"    {self.summary}"]
        for param in self.params:
            out.append(f"    {param.name} — {param.description}")
        if self.example:
            out.append(f"    e.g. {self.example}")
        if self.returns:
            out.append(f"    returns: {self.returns}")
        for wording, remedy in self.refusals:
            out.append(f'    refused "{wording}" — {remedy}')
        if self.use_when:
            out.append(f"    use when: {self.use_when}")
        if self.not_for:
            out.append(f"    not for: {self.not_for}")
        # Walking's cost is the distance, not a constant, so quoting a fixed
        # number for it would be a figure the courier could not reconcile with
        # its own clock.
        if self.kind is ToolKind.ACT and not self.time_cost_s:
            cost = "cost: the walk, and the turn"
        elif not self.time_cost_s:
            cost = "cost: no clock time, but still the turn"
        else:
            cost = f"cost: ~{self.time_cost_s:.0f} s, and the turn"
        out.append(f"    {cost}")
        return "\n".join(out)

    def informative(self, already_known: frozenset[str] = OBSERVATION_PROVIDES) -> bool:
        """Can this call tell the courier something the turn has not already?"""
        return not self.provides or bool(self.provides - already_known)

    def signature(self) -> str:
        inner = ", ".join(
            p.name if p.required else f"{p.name}=None" for p in self.params
        )
        return f"{self.name}({inner})"

    def typed_signature(self) -> str:
        """The signature with argument types, so the menu needs no second block.

        The menu used to spend three lines on every tool -- signature, one line
        per parameter, and an example -- which is 1179 tokens of system prompt
        for ten tools whose names say most of it. Folding the type into the
        signature says the same thing in one line, and the parameter's own
        description is only worth its line when the name does not carry it.
        """
        inner = ", ".join(
            f"{p.name}: {p.type}" + ("" if p.required else " = default")
            for p in self.params
        )
        return f"{self.name}({inner})"

    def describe(self) -> str:
        lines = [f"{self.typed_signature()} — {self.summary}"]
        # Only parameters the signature does not already explain. ``k: int`` next
        # to "the number beside the street you want" is the same sentence twice.
        for param in self.params:
            if param.name in self.summary or param.type in ("int",):
                continue
            lines.append(f"    {param.name} — {param.description}")
        if self.example and any(p.type == "str" for p in self.params):
            lines.append(f"    e.g. {self.example}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# The courier's tool set
# ─────────────────────────────────────────────────────────────────────────────

WALK_TO = Tool(
    name="walk_to",
    kind=ToolKind.ACT,
    summary=(
        "Walk down a street leaving this junction. The bearing is only "
        "needed when the same name leaves here twice -- most junctions."
    ),
    params=(
        ToolParam("street", "str", "the name of the street, as it is written"),
        ToolParam("heading", "str",
                  "which way along it: north, north-east, east, and so on",
                  required=False),
    ),
    example='walk_to("Rue de Grenelle", "east")',
    example2='walk_to("Rue du Bac", "north")',
    time_cost_s=0.0,
    requires_env_action="MOVE_TO",
    returns=(
        "the next junction: street, door numbers, ways out, a photograph "
        "down each. If the numbers moved, the turn says which way."
    ),
    refusals=(
        ("that street does not leave this junction",
         "the name is not on this turn's list; you have not reached it "
         "yet. Take the listed street nearest the map line's way."),
        ("which way along it",
         "that street leaves here twice. Say the bearing as well: "
         'walk_to("Rue de Grenelle", "east").'),
        ("blocked and you cannot get past",
         "you are still at the junction; that street stays shut all "
         "shift. Take a different one."),
    ),
    use_when="you know which street you want and which way along it",
    not_for=(
        "finding out what is down a street; the photograph is already in "
        "front of you, and look() reads numbers without walking"
    ),
)

WALK_TO_XY = Tool(
    name="walk_to_xy",
    kind=ToolKind.ACT,
    summary=(
        "Take one step toward a point on the map, naming the point as a pair "
        "of coordinates. Every turn tells you the point you are standing on "
        "and which way you face, in the same two numbers. A step is at most "
        "{max_step_m} m, so the point you name is a step away and not your "
        "destination: name the next {max_step_m} m of the way, walk it, and "
        "name the next from where you land."
    ),
    # The axes are the city's, not the ones a reader would assume: this map's
    # north is +x and its east is +y, which is what ``bearing_deg`` and the
    # drawing both speak (see ``map_image.MapView.to_px``). Stating it the
    # other way round -- as the first draft of this tool did -- would send
    # every coordinate the policy names ninety degrees off.
    params=(
        ToolParam("x", "number", "how far north, in metres, on the same scale "
                                 "your own position is given in"),
        ToolParam("y", "number", "how far east, in metres, on the same scale"),
    ),
    example="walk_to_xy(-267.1, 97.8)",
    example2="walk_to_xy(-266.2, 98.9)",
    time_cost_s=0.0,
    requires_env_action="MOVE_TO_XY",
    returns=(
        "how far you actually walked and the point you are standing on now, "
        "then the same look at where you have arrived that any other arrival "
        "gives: the streets leaving it, and the view down them."
    ),
    refusals=(
        ("there is no way to walk there",
         "there is no pavement between you and that point -- a building, a "
         "wall, a river. You are left where the way ran out. Read the map "
         "again and aim at somewhere a person could walk to."),
        ("you are already standing there",
         "the point you named is the one you are on -- the same two numbers "
         "the turn gave you. Decide how far north and how far east you want "
         "to go, add that to each of your numbers, and name the result."),
        ("that is further than one step",
         "you named somewhere more than {max_step_m} m off. Nothing moves. "
         "Name a point on the way there instead; several steps all arrive."),
    ),
    use_when=(
        "you can see on the map where you want to be, and would rather aim "
        "straight at it than work out which streets lead there"
    ),
    not_for=(
        "somewhere you have not located on the map. You are told your own "
        "position and nothing else's, so a coordinate you guessed at walks "
        "you into a wall and costs you the turn"
    ),
)

WALK_TO_PIXEL = Tool(
    name="walk_to_pixel",
    kind=ToolKind.ACT,
    summary=(
        "Take one step toward a point you pick IN the photograph you were "
        "just shown, not on a map. Name the point as two numbers between 0 "
        "and 1: how far across (0 is the left edge, 1 the right) and how "
        "far down (0 is the top, 1 the bottom). You are not told your own "
        "position in metres or coordinates; you point at ground you can "
        "see."
    ),
    params=(
        ToolParam("u", "number", "how far across the photograph, left (0) "
                                 "to right (1)"),
        ToolParam("v", "number", "how far down the photograph, top (0) to "
                                 "bottom (1)"),
    ),
    example="walk_to_pixel(0.50, 0.80)",
    example2="walk_to_pixel(0.35, 0.85)",
    time_cost_s=0.0,
    requires_env_action="MOVE_TO_PIXEL",
    returns=(
        "how far you actually walked, then a new photograph taken from "
        "where you land -- pick your next point on that one, not the one "
        "you just used"
    ),
    refusals=(
        ("outside the photograph",
         "both numbers have to be between 0 and 1. Pick a point inside "
         "the picture."),
        ("not somewhere you can walk",
         "that point is not walkable ground -- it may be sky, a wall, an "
         "obstacle, or too far off the pavement. Look at the same "
         "photograph again and pick a different visible point of open "
         "ground, below the horizon."),
        ("there is no way to walk there",
         "the point resolved to somewhere real but nothing led to it. You "
         "are left where the way ran out; a new photograph is waiting."),
    ),
    use_when=(
        "you can see, in the photograph in front of you, ground you could "
        "walk to"
    ),
    not_for=(
        "anywhere outside the photograph, or the sky, a wall, a vehicle, "
        "or another obstacle -- pick visible open ground only"
    ),
)

# The dual-view action intentionally has the same policy-facing name as the
# legacy pixel action and a different required shape. It therefore cannot join
# ``PIXEL_GOAL_TOOLS`` or ``TOOLS_BY_NAME``: those are the global declarations
# used by legacy environments and name-based audits. The active environment
# substitutes this object in ``tools_for_prompt()``, and a session retains the
# exact objects it was handed for parsing and error messages.
WALK_TO_PIXEL_FRONT_REAR = Tool(
    name="walk_to_pixel",
    kind=ToolKind.ACT,
    summary=("Take one step toward a point in one of the two photographs "
             "shown this turn. Select the photograph and the point in one call."),
    params=(
        ToolParam("view", "str", 'the photograph label: "front" or "rear"'),
        ToolParam("u", "number", "how far across the selected photograph"),
        ToolParam("v", "number", "how far down the selected photograph"),
    ),
    example='walk_to_pixel(view="front", u=0.50, v=0.80)',
    example2='walk_to_pixel(view="rear", u=0.35, v=0.85)',
    time_cost_s=0.0,
    requires_env_action="MOVE_TO_PIXEL",
    returns=("how far you actually walked, then a fresh simultaneous "
             "front/rear photograph pair from where you arrived"),
    refusals=(
        ("outside the photograph",
         "the selected point is outside the selected photograph"),
        ("photograph unavailable",
         "the selected photograph is not part of this turn"),
        ("not somewhere you can walk",
         "the selected point did not resolve to walkable ground"),
        ("there is no way to walk there",
         "the point resolved, but the movement controller could not complete a route"),
    ),
    use_when=("you want to step toward visible ground in either photograph "
              "from this turn"),
    not_for=("a point not visible in the selected photograph, or an object "
             "that is not walkable ground"),
)

# The four-view variant: the same call, four photograph labels, and a walk
# that follows pedestrian ways (pavement, marked crossings) to the point.
WALK_TO_PIXEL_QUAD = Tool(
    name="walk_to_pixel",
    kind=ToolKind.ACT,
    summary=("Walk to a point on the pavement in one of the four photographs "
             "shown this turn, along pedestrian ways. Select the photograph "
             "and the point in one call."),
    params=(
        ToolParam("view", "str",
                  'the photograph label: "front", "left", "right" or "rear"'),
        ToolParam("u", "number", "how far across the selected photograph"),
        ToolParam("v", "number", "how far down the selected photograph"),
    ),
    example='walk_to_pixel(view="front", u=0.50, v=0.80)',
    example2='walk_to_pixel(view="left", u=0.35, v=0.85)',
    time_cost_s=0.0,
    requires_env_action="MOVE_TO_PIXEL",
    returns=("how far you walked, then four fresh photographs from where "
             "you arrived"),
    refusals=(
        ("outside the photograph",
         "the selected point is outside the selected photograph"),
        ("photograph unavailable",
         "the selected photograph is not part of this turn"),
        ("not on the pedestrian way",
         "the selected point is not on or beside a pavement or marked crossing"),
        ("on the carriageway",
         "the selected point is on the road, away from any pavement"),
        ("there is no way to walk there",
         "no pedestrian way leads from here to that point"),
    ),
    use_when=("you want to walk to visible paving in any of the four "
              "photographs from this turn"),
    not_for=("a point up a wall, in the sky, or on the road away from the "
             "pavement"),
)

FOLLOW_STREET = Tool(
    name="follow_street",
    kind=ToolKind.ACT,
    summary=(
        "Take a street and keep going along it for up to n junctions (6 at "
        "most). Stops early at a fork, a dead end, a crossing with a light, "
        "anything blocking the way, or your address."
    ),
    params=(
        ToolParam("street", "str", "the name of the street"),
        ToolParam("heading", "str", "which way along it", required=False),
        ToolParam("n", "int", "how many junctions to walk, at most 6", required=False),
    ),
    example='follow_street("Rue de Grenelle", "east", 6)',
    time_cost_s=0.0,
    requires_env_action="MOVE_TO",
    returns=(
        "wherever it stopped, and why: a fork, a dead end, a crossing with a "
        "light, something blocking the way, or your address"
    ),
    refusals=(
        ("that street does not leave this junction",
         "same as walk_to -- take a street that is on this turn's list"),
    ),
    use_when=(
        "the route says stay on this street for several junctions. It is one "
        "turn instead of six"
    ),
    not_for="the junction where you mean to turn off; it may carry you past it",
)

LOOK = Tool(
    name="look",
    kind=ToolKind.LOOK,
    # It used to answer with the street's name, compass and distance as well --
    # the candidate line word for word, already on screen, for two seconds and a
    # turn. What it alone can say is the numbers running away down a street the
    # courier is not standing on, which is the gradient that finds a door when no
    # phone will. So that is all it says now, and the summary promises only that.
    summary=(
        "Read the door numbers down a street without walking it, and which "
        "way they climb."
    ),
    params=(
        ToolParam("street", "str", "the name of the street to look down"),
        ToolParam("heading", "str", "which way along it", required=False),
    ),
    example='look("Rue de Grenelle", "east")',
    # A rider glances down a street in a couple of seconds. Charging something
    # keeps looking honest without making it precious.
    time_cost_s=2.0,
    counts_as_step=True,
    provides=frozenset({FACT_NUMBERS_AHEAD}),
    returns="that street's door numbers, and which way they climb",
    refusals=(
        ("that street does not leave this junction",
         "look only sees streets on this turn's list"),
    ),
    use_when=(
        "you are on the right street at the wrong number and cannot tell "
        "which way the numbers rise; one turn here beats a walk the wrong way"
    ),
    not_for="seeing a red light or a barrier",
)

# ``read_sign`` was here, and it is gone. It answered "The sign says Quai
# Beaubourg. The doors here are numbered 2." to a turn that had already opened
# with "You are on Quai Beaubourg, outside number 2." -- the same two facts, in
# every condition, for three seconds and a turn. There is nothing to weaken or
# reprice: a tool whose entire output is a restatement of the prompt is not a
# cheap tool, it is not a tool. Anything that wants to re-anchor the courier on
# which street it is on reads the observation.

CHECK_ORDER = Tool(
    name="check_order",
    kind=ToolKind.CONSULT,
    summary="Re-read the order slip: pickup address, dropoff address, deadline, fee.",
    example="check_order()",
    time_cost_s=2.0,
    # The turn header names the end the courier is walking to and when it is due.
    # The slip is where both ends and the money are.
    provides=frozenset({FACT_ORDER_ADDRESSES, FACT_ORDER_FEE, FACT_DEADLINES}),
    returns="the slip again: pickup, dropoff, deadline, fee",
    use_when="you have lost track of which address you are heading for",
    not_for=(
        "every turn -- the job is already written at the top of your notes"
    ),
)

CHECK_MAP = Tool(
    name="check_map",
    kind=ToolKind.CONSULT,
    summary="Look up one address on your phone: which street, how far, roughly which way.",
    params=(ToolParam("address", "str", "the address to look up, as written on the slip"),),
    example='check_map("42 Rue de Rivoli")',
    time_cost_s=5.0,
    # Search, not directions: where a place is and how far, for any address the
    # courier can name -- including one no live job mentions. ``navigate`` routes
    # only to a job in hand, and says how to get there rather than where it is.
    provides=frozenset({FACT_ROUTE_DISTANCE, FACT_TARGET_BEARING}),
    returns=(
        "one address looked up: which street it is on, roughly how far, and "
        "roughly which way"
    ),
    refusals=(
        ("no such address",
         "the address has to be one the city has -- copy it from the slip"),
    ),
    use_when="you want to know where an address is without routing to it",
    not_for="reading the route; that is drawn on the map picture",
)

NAVIGATE = Tool(
    name="navigate",
    kind=ToolKind.CONSULT,
    summary=(
        "Put a route on the phone's screen; leave the address out to route to "
        "the job in hand. The route is DRAWN on the map, not written out. The "
        "phone cannot see crossings, traffic or doors."
    ),
    params=(ToolParam("where", "str",
                      "the address to route to, or leave out for the job in hand",
                      required=False),),
    example='navigate("13 Avenue Dauphine")',
    # A phone lookup and reading the route off the screen, standing still. Dearer
    # than check_map because it answers a bigger question: a policy that calls it
    # every turn instead of walking should lose to one that calls it once a leg.
    time_cost_s=15.0,
    provides=frozenset({FACT_TURN_BY_TURN, FACT_ROUTE_DISTANCE}),
    returns=(
        "a route drawn on the map, on screen every turn after; read which way "
        "to go off the drawing"
    ),
    refusals=(
        ("no such address",
         "copy the address from the slip exactly"),
        ("no route",
         "nothing walkable reaches it; go a different way and ask again"),
    ),
    use_when=(
        "the screen is not already showing the way, or has stopped matching "
        "what you see"
    ),
    not_for=(
        "getting past a barrier; the map cannot see it and will route you "
        "into the same street again"
    ),
)


COLLECT = Tool(
    name="collect",
    kind=ToolKind.ACT,
    summary="Collect the order. Only works standing at the pickup address.",
    example="collect()",
    time_cost_s=30.0,
    requires_env_action="PICKUP",
    returns="the parcel in your bag, and the job becomes a delivery",
    refusals=(
        ("you are not at the pickup",
         "you are somewhere else. It costs a turn and points nowhere; "
         "compare the street and number at the top of the turn against the "
         "slip instead of probing."),
    ),
    use_when="street and door number both match the pickup on the slip",
    not_for="checking whether you have arrived",
)

HAND_OVER = Tool(
    name="hand_over",
    kind=ToolKind.ACT,
    summary="Hand the order to the customer. Only works standing at the dropoff address.",
    example="hand_over()",
    time_cost_s=30.0,
    requires_env_action="DROP_OFF",
    returns="the fee, and the order is done",
    refusals=(
        ("you are not at the dropoff",
         "same as collect: it is not a way to search, it is a way to finish"),
        ("you have not collected it yet",
         "go to the pickup first"),
    ),
    use_when="street and door number both match the dropoff",
    not_for="checking whether you have arrived",
)

ACCEPT_JOB = Tool(
    name="accept_job",
    kind=ToolKind.ACT,
    summary="Accept an offered job.",
    params=(ToolParam("k", "int", "the number of the job to accept"),),
    example="accept_job(0)",
    requires_env_action="ACCEPT_ORDER",
)

LIST_JOBS = Tool(
    name="list_jobs",
    kind=ToolKind.CONSULT,
    summary="See the jobs currently on offer.",
    example="list_jobs()",
    requires_env_action="VIEW_ORDERS",
)

WAIT = Tool(
    name="wait",
    kind=ToolKind.ACT,
    # "Wait where you are for a moment" described the only tool that answers the
    # only mechanic the photographs exist for, and never mentioned either. At a
    # crossing this waits out the phase, so one call always changes the light --
    # which is the fact a policy needs in order to use it at all.
    summary=(
        "Wait where you are; at a crossing this sees the light change, so one "
        "call is always enough."
    ),
    example="wait()",
    time_cost_s=10.0,
    requires_env_action="WAIT",
    returns="the crossing, with the light having changed",
    use_when="the pedestrian light for the street you want is red",
    not_for="anywhere else; nothing changes because you stood still",
)

REST = Tool(
    name="rest",
    kind=ToolKind.ACT,
    # A tank that only empties is not a resource, it is a decay: over a long
    # shift the courier simply gets slower and there is nothing to decide. With
    # somewhere to spend time for energy back, stamina becomes the trade the
    # tiers are meant to pose -- rest now and walk fast, or push on tired.
    summary="Stop and catch your breath: time spent, energy back.",
    example="rest()",
    time_cost_s=60.0,
    requires_env_action="REST",
    returns="energy back, at the cost of clock",
    refusals=(
        ("you never tire", "this body has no stamina to recover"),
    ),
    use_when="walking has slowed because you are tired",
    not_for="a pause to think; thinking is free and this is not",
)

CHARGE_PHONE = Tool(
    name="charge_phone",
    kind=ToolKind.ACT,
    # The battery makes the map mortal; the power bank makes that a decision
    # rather than a countdown. Ninety seconds buys forty points, so saving a
    # shift is possible but never free -- standing still charging is time not
    # spent walking, exactly like resting.
    summary="Plug the phone into your power bank: time spent, charge back.",
    example="charge_phone()",
    time_cost_s=90.0,
    requires_env_action="CHARGE_PHONE",
    returns="battery back, at the cost of clock",
    refusals=(
        ("nothing to charge", "this shift's phone has no battery mechanic"),
    ),
    use_when="the battery is low and you still need the map",
    not_for="topping up a phone that is nearly full",
)

NOTE = Tool(
    name="note",
    kind=ToolKind.CONSULT,
    summary=(
        "Write a line in your notebook, shown back to you every turn — for "
        "what you must not forget."
    ),
    params=(ToolParam("text", "str", "what to remember, in a few words"),),
    example='note("Rue Monge north end is a dead end")',
    time_cost_s=0.0,
    returns="the line, written into your notes, where it stays every turn",
    refusals=(
        ("a note needs something written on it", "give it some text"),
    ),
    use_when=(
        "you worked out something the turn will not tell you again -- a dead "
        "end, a corner already searched"
    ),
    not_for="what is already printed every turn: streets, numbers, the job",
)

# Only tools the runtime dispatches. `accept_job`, `list_jobs` and `note` are
# defined above and have no executor anywhere, exactly like the macros that were
# already removed -- the earlier fix was applied to the macros alone and missed
# these three. A tool in the menu that raises FormatError costs a turn and, three
# times running, the episode.
#
# ``follow_street`` was removed from the menu on that same rule, correctly, when
# nothing could run it. ``CourierEnv.follow_street`` now can, and it belongs back
# in: a leg is a median 530 m over 18 m edges, so a courier without it spends
# about 35 turns per order pressing the same button.
ALL_TOOLS: tuple[Tool, ...] = (
    WALK_TO, FOLLOW_STREET, LOOK, CHECK_ORDER, CHECK_MAP, NAVIGATE,
    COLLECT, HAND_OVER, WAIT, REST, CHARGE_PHONE, NOTE,
)

# Offered only when the environment enables coordinate walking, because the
# two are different tasks: naming a street is a choice among the handful this
# junction offers, while naming a point is a free coordinate the policy has to
# derive from the map and its own pose. Mixing them in one menu would let a
# run fall back to the easy one and report the hard one's number.
COORDINATE_TOOLS: tuple[Tool, ...] = (WALK_TO_XY,)

# Offered only when the environment enables pixel-goal walking. A third task,
# not a variant of the coordinate one: the policy names a point IN a picture
# rather than deriving a metric position, and gets no coordinates at all --
# so it belongs apart from ``COORDINATE_TOOLS`` for the same reason that one
# is kept apart from the street menu.
PIXEL_GOAL_TOOLS: tuple[Tool, ...] = (WALK_TO_PIXEL,)

#: The environment actions that move the courier. An environment enables
#: exactly one of these -- which one is the action space a run is measuring,
#: and the prompt is built around whichever it finds.
MOVEMENT_ACTIONS: frozenset[str] = frozenset(
    {"MOVE_TO", "MOVE_TO_XY", "MOVE_TO_PIXEL"})
UNIMPLEMENTED_TOOLS: tuple[Tool, ...] = (LIST_JOBS, ACCEPT_JOB)
TOOLS_BY_NAME: dict[str, Tool] = {
    t.name: t for t in
    ALL_TOOLS + COORDINATE_TOOLS + PIXEL_GOAL_TOOLS + UNIMPLEMENTED_TOOLS
}
# Tools that were in the menu and are not any more, with the reason, so a
# reviewer can tell a deliberate retirement from an oversight.
RETIRED_TOOLS: dict[str, str] = {
    "report_blocked": (
        "it let the courier tell its phone a street was shut, and the phone "
        "would route round it. That is not a thing a rider does -- you see a "
        "skip and you take the next street, you do not file a report with the "
        "map app. With it gone the phone stays permanently blind, the route "
        "keeps pointing through the barrier, and going round is something the "
        "courier has to work out from what it can see"
    ),
    "read_sign": (
        "every fact it returned -- the street here and the numbers here -- is "
        "stated in the first two lines of every observation, in every condition"
    ),
}

#: A ``{placeholder}`` no environment filled in. See ``available_tools``.
_UNFILLED = re.compile(r"\{[a-z_][a-z0-9_]*\}")


def available_tools(
    enabled_env_actions: list[str],
    *,
    allow_consult: bool = True,
    observation_provides: frozenset[str] = OBSERVATION_PROVIDES,
    limits: dict[str, Any] | None = None,
) -> list[Tool]:
    """The tools this environment can actually execute and that can say something.

    Three filters, and they fail in different ways if omitted. A tool backed by
    an environment action the map does not enable is dropped rather than
    described and then refused. ``allow_consult`` exists because the phone is the
    natural difficulty knob: switch it off and the courier has to navigate by
    house numbers and memory alone. And a tool whose facts the observation
    already states is dropped as well -- not because it would fail, but because
    it would succeed at telling the courier what it was just told, and charge a
    turn for it.

    ``observation_provides`` is a parameter rather than a constant because it is
    the same knob as ``allow_consult`` pointed at the other half: a condition
    that stops stating house numbers in the header gives a tool that reports
    house numbers something to do again.

    ``limits`` are the environment's own numbers, written into the manuals that
    quote them. The coordinate action space is the reason it exists: its step
    cap is configurable, the manual has to state it, and the two would drift
    the first time either was changed alone. The check below is the point --
    an unfilled placeholder reaches the model as the literal characters
    ``{max_step_m}``, which is the same defect as the ``{street}`` this
    module's docstring was written about.

    The coordinate and pixel-goal tools are considered here alongside the
    street ones, and the ``requires_env_action`` filter is what keeps all
    three apart: an environment enables exactly one of ``MOVE_TO``,
    ``MOVE_TO_XY``, or ``MOVE_TO_PIXEL``, so the menu describes one action
    space and the other two are not mentioned.
    """
    enabled = set(enabled_env_actions)
    out: list[Tool] = []
    for tool in ALL_TOOLS + COORDINATE_TOOLS + PIXEL_GOAL_TOOLS:
        if tool.requires_env_action and tool.requires_env_action not in enabled:
            continue
        if tool.kind is ToolKind.CONSULT and not allow_consult:
            continue
        if not tool.informative(observation_provides):
            continue
        chosen = tool.filled(**(limits or {}))
        left = _UNFILLED.search(chosen.manual())
        if left:
            raise ValueError(
                f"{tool.name}'s manual still carries {left.group(0)}: the "
                "environment offering this tool has to supply that value in "
                "`limits`, or the model is shown the placeholder."
            )
        out.append(chosen)
    return out


def render_tool_menu(tools: list[Tool]) -> str:
    """The tool section of the system prompt, grouped by what each kind does."""
    groups = {
        ToolKind.ACT: ("ACTIONS — these change the world and take time",
                       "Each entry says what the call gives back, what a "
                       "refusal means and what to do about it, and when the "
                       "tool is the wrong one."),
        ToolKind.LOOK: ("LOOKING — these tell you about where you are standing",
                        ""),
        ToolKind.CONSULT: ("CONSULTING — these query what you carry, not what "
                           "you see", ""),
    }
    lines: list[str] = []
    for kind, (heading, note) in groups.items():
        chosen = [t for t in tools if t.kind is kind]
        if not chosen:
            continue
        lines.append(heading + ":")
        if note:
            lines.append("  " + note)
        for tool in chosen:
            lines.append("")
            lines.append("  " + tool.manual().replace("\n", "\n  "))
        lines.append("")
    return "\n".join(lines).rstrip()
