"""Prompt templates for the courier harness. Data, not code.

Kept as templates for the same reason mini-SWE-agent keeps its prompts in YAML:
the wording is the part most often changed and least often tested, so it must be
inspectable and diffable without reading control flow. Every field the templates
interpolate is listed in ``REQUIRED_FIELDS`` and checked, so a renamed field
fails loudly instead of rendering the literal ``{street}`` into the model's
context.

The system prompt is built from the environment's own tool set, so a tool this
map cannot execute is never described. That is not tidiness: the first Paris
observation told the agent to use ``MOVE(direction=...)`` on a map where MOVE was
disabled, and a policy that obeyed it was rejected on every single turn.
"""

from __future__ import annotations

import logging

from embodiedbench.agent.courier.skills import render_procedures
from embodiedbench.agent.courier.tools import (
    MOVEMENT_ACTIONS,
    Tool,
    ToolKind,
    render_tool_menu,
)

_log = logging.getLogger("courier.prompts")

SYSTEM_TEMPLATE = """You are a delivery courier working on foot in {city}. You collect parcels and
hand them to customers at street addresses, against a clock.

Each turn you are shown your notes, where you are standing, the streets leaving
this junction by name and bearing, and a photograph looking down them. Where the
next junction is metres off, or the street turns, the photograph is mostly the
building opposite — a view that does not reach, not an empty street.

{sources}

{steps}

{map_reading}

Repeating a call that was just refused will be refused again for the same
reason. Nothing about the world changed in between. Read what the refusal
listed, and choose from that.

{hazards}

{tool_menu}

{streets}

What you have been trained to do:
{procedures}

HOW TO REPLY — exactly this shape, every turn:

THOUGHT: one line saying what you read and what you concluded.
```
{reply_example}
```

  - {call_count_rule}
  - The name must be one of the tools listed above.
  - {quoting_rule}
  - Whole numbers go bare.
  - A tool with no arguments still needs its brackets: {no_arg_example}
  - Keep the THOUGHT to one line. A reply that runs too long is cut off before
    it reaches the action, and a cut-off reply loses the turn."""

# ─────────────────────────────────────────────────────────────────────────────
# The three action spaces
# ─────────────────────────────────────────────────────────────────────────────
#
# Naming a street, naming a coordinate, and naming a point in the photograph
# are three different tasks, and the paragraphs below are where the prompt
# says which one is being asked for. They are split out rather than written
# into the template because everything else about the prompt -- the hazards,
# the reply format, the procedures, the clock -- is the same task in all
# three, and a second copy of the whole system prompt with a few paragraphs
# different is how a run comes to differ from its own baseline in ways
# nobody chose. See ``tools.COORDINATE_TOOLS``/``tools.PIXEL_GOAL_TOOLS`` for
# why no two of the three are ever offered together.

MAP_READING = """READING THE MAP. It is a picture of the streets around you, north up. On it:

  - a BLUE LINE is your route, from where you are to where you are going;
  - a THICK BLUE ARROW leaves your position along the first stretch of that
    route — it points at the street you want next;
  - a small circle labelled "you are here" is you;
  - a red pin is the address you are heading for;
  - the streets are named on the map, written along each street.

The surest way to use it is by NAME, not by angle — but only ever a name that
is ALSO IN THE LIST above. Do it in this order:

  1. read the names the blue line runs along;
  2. go down the list of streets leaving this junction and find one of those
     names in it;
  3. walk that one, spelling it exactly as THE LIST spells it.

If none of the line's names is in the list, you have not reached them yet —
THE MAP IS NOT WRONG AND NEITHER IS THE LIST. Typing a name off the map that
is not in the list is refused and the turn is gone. Take the listed street
that runs most nearly the way the arrow points, and read the map again from
the next corner. The arrow and compass are a check on the name you picked,
not a substitute.

The route is walked one junction at a time. The line crosses several streets;
you can only take one that leaves the corner you are on. When the street you
had in mind is not in the list, THE MAP IS NOT WRONG AND NEITHER IS THE LIST —
you have not reached that street yet."""

STREET_RULES = """How the streets work here:
  - You take a street by naming it and the way you are going:
    walk_to("Rue de Grenelle", "east"). Both are written on its line, and the
    photograph captioned with the same two words is the view down it.
  - The same name usually leaves a junction twice — two directions along one
    street, which is why the bearing is part of naming one.
  - A street name means the same thing everywhere: one you have already tried
    is one you can recognise and rule out.
  - Each street also says where it lies relative to the way you face, so
    "turn left onto Rue X" is actable.
  - House numbers run in order, odd one side and even the other; falling when
    you want higher means turn around.
  - A street keeps its name junction to junction; a new name at the top of the
    turn means you have left the street you were on."""

QUOTING_RULE = ("Street names and bearings go in double quotes, spelled as "
                "they are written\n    in the list: {number_example}.")

# The coordinate action space. What changes is the whole middle of the job:
# the courier no longer picks a name off a list, it reads a position off the
# map and names one. So the map section stops being "find the name in both
# places" and becomes "measure", the street rules become coordinate rules, and
# the quoting rule inverts -- these are the only arguments in this grammar
# that must NOT be quoted.
MAP_READING_XY = """READING THE MAP. It is a picture of the streets around you, north up. On it:

  - a BLUE LINE is your route, from where you are to where you are going;
  - a THICK BLUE ARROW leaves your position along the first stretch of that
    route — it points the way you want to go next;
  - a small circle labelled "you are here" is you, and every turn tells you the
    two numbers for that circle;
  - a red pin is the address you are heading for;
  - the streets are named on the map, written along each street.

YOU MOVE BY NAMING A POINT, NOT A STREET. The two numbers are metres: the first
counts NORTH, which is up the map, and the second counts EAST, which is right
across it. Both can be negative. Your own position is given in exactly those
two numbers every turn, so the way to name a point is to start from where you
are and count:

  1. find yourself on the map, and read the two numbers the turn gives you;
  2. look along the blue line and pick somewhere on it you want to reach —
     a corner it turns at, or simply a stretch of it ahead of you;
  3. work out how far north and how far east that point is FROM YOU, using the
     scale bar, and add each to your own two numbers;
  4. walk to the result.

Aim at the road. A point inside a building or across a wall has no pavement
leading to it, and asking for one costs you the turn and leaves you where the
way ran out. When you are unsure, name a nearer point on the line rather than a
further one: several short walks along a route all arrive, and one long walk
through a building does not.

Nothing tells you the coordinates of the address you are delivering to. Its pin
is on the map and you can measure it like anything else, but the numbers for it
are not written anywhere, and a guessed coordinate is a walk into a wall."""

STREET_RULES_XY = """How getting about works here:
  - You move by naming a point: walk_to_xy(-267.1, 97.8). North first, east
    second, both in metres, both bare numbers.
  - The streets leaving where you stand are still listed, with their names,
    bearings and how far the next junction is. You do not take one by name —
    but the list and the photographs are how you tell which way is walkable
    from here, and how far it is worth aiming.
  - Where you end up is a real position, and it is where your next call counts
    from. Read your own two numbers again every turn rather than adding up the
    walks you meant to take: a walk that was cut short or stopped at a wall
    leaves you somewhere you did not plan.
  - House numbers run in order along a street, odd one side and even the other;
    if they are falling and you want a higher one, turn around.
  - A street keeps its name from junction to junction, so a new name means you
    have left the street you were on."""

QUOTING_RULE_XY = ("Coordinates are bare numbers — no quotes, no units, no "
                   "brackets of\n    their own: {number_example}.")

# The narrated settings keep their own first step -- the route marker still
# says which way, and taking that away would make the coordinate arm harder
# than the street arm at something other than the action space. What changes
# is only the last clause of the last step, where "walk it" is a call this
# space does not have. At narration=route the two arms then differ in exactly
# one thing: whether the move is expressed as a name or as a point.
STEPS_XY_ALL = """SO EVERY MOVE IS THE SAME TWO STEPS:
  1. Find the street in the list marked *** THE ROUTE GOES THIS WAY ***. Its
     bearing is the way to go and its distance is how far the next junction is.
  2. If its line says the pedestrian light is RED, wait(). If its line says
     BLOCKED, that street is shut — aim along another and the marker will move.
     Otherwise name a point that far along it, and walk there."""

STEPS_XY_ROUTE = """SO EVERY MOVE IS THE SAME TWO STEPS:
  1. Find the street in the list marked *** THE ROUTE GOES THIS WAY ***. Its
     bearing is the way to go and its distance is how far the next junction is.
  2. Look at that street's photograph. Red light, or blocked? Then wait() or
     aim along a different one. Otherwise turn that bearing and that distance
     into a point — how far north, how far east — add it to your own position,
     and walk there."""

THREE_STEPS_XY = """SO EVERY MOVE IS THE SAME THREE STEPS:
  1. Look at the map. Where does the line go from here — how far north and how
     far east of the point you are standing on?
  2. Add that to your own two numbers, and keep the point on a street. Do not
     aim past a corner the route turns at; aim at the corner.
  3. Look at the photograph down the way you are about to walk. Red light, or
     blocked? Then wait() or aim somewhere else. Otherwise walk it."""

THREE_STEPS = """SO EVERY MOVE IS THE SAME THREE STEPS:
  1. Look at the map. Which way does the line leave you — which compass point?
  2. Look at the list of streets here. Which one goes that way? Take the one
     whose bearing is nearest the line, even if its name is not one you were
     expecting; street names change from junction to junction and the route
     runs through several of them.
  3. Look at that street's photograph. Red light, or blocked? Then wait() or
     take a different street. Otherwise walk it."""

HAZARD_RULES = """LOOK AT THE PHOTOGRAPHS BEFORE YOU ACT. The text will never tell you the colour
of a pedestrian light, what is standing in your way, or what a shopfront says.
Those are in the pictures and nowhere else. Where a crossing has a pedestrian
light you can see, a separate photograph of it is shown, captioned
[light: street name, bearing], and that lamp — not any light in the street views,
which are older photographs — is the one governing your crossing.
Crossing while it is red costs you time and counts against you; wait() sees the
phase out.

Streets get blocked and streets get congested. Roadworks, a barrier or a skip
can shut a street completely — you cannot walk it at all, and finding that out
by trying costs you a turn and about three quarters of a minute before you are
back where you started. Furniture crowding the pavement does not stop you but
slows you down by about the same. Which streets, and where, changes from shift
to shift, and is written in no list, no order slip and no route: look down each
street's photograph before you take it.{blocked_advice}"""
# How many calls a turn may carry. One sentence, held in a constant rather than
# written into the template, because the chunked session changes this rule and
# nothing else about the prompt: a second copy of the whole system prompt with
# one paragraph different is how a training prompt and an evaluation prompt
# drift apart without anyone deciding they should.
ONE_CALL_RULE = ("Exactly one fenced block, containing exactly one call, and "
                 "nothing after it.")

# The multi-call rule, for K > 1. It states the cost as well as the permission:
# the calls run in order and the run stops at the first refusal, so a chunk is
# a bet that every call after the first will still make sense. A model told it
# may issue K calls and not told what a refusal does to the rest will chain
# optimistically and lose the turn.
CHUNK_CALL_RULE = """Exactly one fenced block, holding UP TO {calls} calls, one per line, and
    nothing after it. ONLY THE FIRST ONE HAPPENS. The rest are your plan for
    after it: write them so you are thinking a few steps ahead, and expect to
    write them again next turn, because the first step will not land exactly
    where you predicted and the turn after this one starts from where it
    actually landed. So make the FIRST call the one you are surest of. More
    than {calls} calls is refused outright and nothing is carried out."""

# What a chunked turn gets back: every call it made, in order, with what the
# world said to each. A single summarising sentence would lose exactly the
# thing the courier needs -- which call it was that stopped the turn.
CHUNK_FEEDBACK_TEMPLATE = """You made {count} calls. In order, this is what happened:

{lines}
{tail}"""

OBSERVATION_TEMPLATE = """{memory}

### where you are
{location}
{clock}

### streets leaving this junction
{candidates}
{take_hint}

### photographs
{photographs}
{extra}"""


def render_photographs(rows: list[dict], *, phone_map: bool = False) -> str:
    """The caption list for the images attached to this turn.

    The images arrive as an ordered list beside the text, and a model that
    cannot tell which picture is which street will read them in whatever order
    it likes. So every frame gets a caption here, in the same order the frames
    are attached, and the caption names the street it belongs to -- in the
    same words ``walk_to`` takes, so reading the picture and acting on it do
    not require a translation step.

    The phone's map, when there is one, is captioned last and captioned as a
    *drawing*. It is the one picture in the list that did not come through the
    courier's eyes, and a caption that let it pass for a photograph would be the
    harness telling the courier its phone can see the street.
    """
    lines: list[str] = []
    # One caption per frame, naming the street the way walk_to takes it. This
    # used to be a single index list ("[1], [2] — the view down each of those
    # streets, in that order") because a per-street caption repeated the
    # candidate line above it word for word. With the streets named rather
    # than numbered there is no index to carry the ordering contract, so the
    # caption carries it by naming, which also survives the image budget
    # dropping some of them.
    for row in rows:
        if row.get("image"):
            view = row.get("view")
            # A row the environment marked as the view ahead is captioned as
            # what it is. Naming a street the courier is not being offered
            # would read as a menu it cannot order from.
            lines.append(
                f'  [front, {row["heading"]}] the view in front of you'
                if view == "front" else
                f'  [rear, {row["heading"]}] the view behind you'
                if view == "rear" else
                f'  [left, {row["heading"]}] the view to your left'
                if view == "left" else
                f'  [right, {row["heading"]}] the view to your right'
                if view == "right" else
                f'  [ahead, {row["heading"]}] the view straight ahead of you'
                if row.get("ahead") else
                f'  [{row["street"]}, {row["heading"]}] the view down it from here')
    for row in rows:
        if row.get("signal_image"):
            lines.append(f'  [light: {row["street"]}, {row["heading"]}] '
                         "the pedestrian light for that crossing")
    if phone_map:
        lines.append(
            "  [map] your phone's map — a drawing, not a photograph: it has the "
            "streets and your route on it and cannot see anything in them"
        )
    if not lines:
        return "  (no photographs here)"
    return "\n".join(lines)

FORMAT_ERROR_TEMPLATE = """Your last reply could not be read as an action.

{error}

Reply with a short THOUGHT, then exactly one action in a fenced block:
```
{example}
```"""

TRUNCATED_TEMPLATE = """{error}

Your reply is cut off when it gets too long, and a cut-off reply loses the
turn. Lead with the action and keep the reasoning to a single line."""

# A refusal that only says no gets repeated at. Measured on 40 held-out
# episodes: after being refused, the model stopped reasoning entirely -- four
# turns in a row of a bare `walk_to("Rue de Mazarine", "north-west")` with no
# THOUGHT at all, the same call each time, until the session ended stuck. It
# had not thought the wrong thing, it had stopped thinking.
#
# So the refusal asks for the reasoning back, in the order the move is made,
# and asks for it in words before the call. It gives no answer away: the
# bearing has to be read off the map, and which street matches it is still the
# decision under test.
REJECTED_TEMPLATE = """That did not work: {reason}

You are still where you were, and nothing about the junction has changed, so
the same call will fail the same way. Work it out again in your THOUGHT before
you act:
  1. Which way does the route line leave you on the map — which compass point?
  2. Of the streets listed above, which one goes most nearly that way?
  3. Is that street's photograph clear — light green, nothing across the road?"""

# One paragraph per optional constraint, stating the rule and its numbers.
# Rendered only for the flags a shift actually runs under: a rule the world
# does not enforce must not be taught, and with every flag off this renders
# the empty string and the prompt is byte-identical to the flagless one.
# No tool is named in any of these lines, so they are safe under every
# Condition (the battery rule matters most under FULL, but describing the
# phone without naming navigate() keeps the prompt/tool symmetry check clean).
SPECIAL_RULE_TEXTS = {
    "earning_jitter": (
        "  - FEES VARY. Two jobs of the same length can pay differently; the "
        "fee on the\n    slip is what this job pays, so read it before "
        "deciding which job to serve."),
    "food_temperature": (
        "  - HOT FOOD GOES COLD. From the moment you collect it you have "
        "about 8\n    minutes; delivered cold it pays only 70% of what it "
        "would have. The clock\n    line tells you how long the food has "
        "been out. Between pickup and door,\n    hurry."),
    "special_notes": (
        "  - READ THE NOTE ON THE SLIP. Some customers leave one and it "
        "changes what\n    happens at the door: \"leave it at the door\" is "
        "quick, \"ring first\" is\n    slow. It costs nothing to know and "
        "changes which job to do first."),
    "phone_battery": (
        "  - YOUR PHONE'S BATTERY IS FINITE. Asking it for a route costs "
        "charge, and\n    every block walked with the map lit sips a little "
        "more. At zero the screen\n    goes dark for the rest of the shift "
        "-- no map, no routes, only the street\n    names and the "
        "photographs. The charge reads out beside the clock."),
    "phone_recharge": (
        "  - YOU CARRY A POWER BANK. charge_phone() spends a minute and a "
        "half standing\n    still and puts 40% back; it even revives a dead "
        "phone. Charging is time not\n    spent walking -- top up before "
        "you are desperate, not after."),
    "food_categories": (
        "  - WHAT IS IN THE BAG DECIDES THE CLOCK. The slip names it: a HOT "
        "MEAL goes\n    cold about 8 minutes after collection and then pays "
        "70%; ICE CREAM melts in\n    about 5 and then pays 60%; GROCERIES "
        "never spoil. The clock line counts it\n    down for you. Read the "
        "slip before you plan the leg."),
}


def render_special_rules(active: "list[str]") -> str:
    """The shift's special-rule block, or the empty string when none apply."""
    lines = [SPECIAL_RULE_TEXTS[name] for name in active if name in SPECIAL_RULE_TEXTS]
    if not lines:
        return ""
    return "SPECIAL RULES THIS SHIFT:\n" + "\n".join(lines)


# Pixel-goal is a visual benchmark. Its feedback states the observed cause,
# supplied by the environment, and does not turn that cause into instructions
# about where or how to aim next.
REJECTED_TEMPLATE_PIXEL = """That did not work: {reason}"""

# Non-navigation tools have their own truthful environment messages. They do
# not inherit either street-selection coaching or pixel-selection language.
REJECTED_TEMPLATE_ACTION = """That did not work: {reason}"""

REQUIRED_FIELDS = {
    "system": {"city", "tool_menu", "procedures", "blocked_advice",
               "map_reading", "streets", "quoting_rule"},
    "observation": {"memory", "location", "clock", "candidates", "photographs",
                    "extra", "take_hint"},
}

# The line under the list of streets, naming the call that acts on it. It is a
# parameter of the observation for the same reason the menu is a parameter of
# the system prompt: on a map with no ``walk_to`` it was still telling the
# courier, every single turn, to take a street with ``walk_to``.
TAKE_STREET_HINT = (
    '(take one with walk_to("street name", "bearing") — the name and the '
    'bearing\n exactly as they are written above; nothing else is a street)')
# Under the coordinate action space the list is still worth printing -- it is
# how the courier tells which way is walkable and how far -- but it is no
# longer a menu, and saying so is the whole of the difference.
#: ``{max_step_m}`` is filled from the environment, the same way the tool
#: manual's copy is -- see ``tools.available_tools``.
AIM_HINT = (
    "(you do not take these by name — they are what is walkable from here, "
    "and how far.\n Move with walk_to_xy(north, east), counting from your own "
    "position above.\n ONE CALL CARRIES YOU UP TO {max_step_m} m: a point a "
    "metre or two off spends the\n turn to go almost nowhere. Aim the whole "
    "step.)")


# The pixel-goal action space. A third middle, not a variant of the
# coordinate one: the courier is never given its own position or told to
# measure anything, on purpose --
# it names a point IN the photograph it was just shown, the way a person
# walking looks at the pavement ahead rather than at a survey. So the map
# section drops the whole "measure and add" procedure and keeps only the
# one thing a map still contributes here: which general way to head.
MAP_READING_PIXEL = """READING THE MAP. It is a picture of the streets around you, north up. On it:

  - a BLUE LINE is your route, from where you are to where you are going;
  - a THICK BLUE ARROW leaves your position along the first stretch of that
    route — it points the general direction to head in next;
  - a small circle labelled "you are here" is you;
  - a red pin is the address you are heading for;
  - the streets are named on the map, written along each street.

YOU DO NOT MEASURE ANYTHING FROM IT. There is no coordinate to work out and
no distance to add up. The map only tells you which general direction to
head — read the arrow, then look at the photograph in front of you and pick
a point that carries you that way.

The photograph, not the map, is what you act on: you move by pointing at a
spot in it directly, below the horizon, the way a person walking looks at
the pavement just ahead and places their next few steps on it rather than
reading a survey."""

STREET_RULES_PIXEL = """How getting about works here:
  - You move by pointing at the photograph you were just shown, not by
    naming a street or a coordinate: walk_to_pixel(0.50, 0.80). The first
    number is how far ACROSS the picture (0 is the left edge, 1 the right),
    the second is how far DOWN it (0 is the top, 1 the bottom).
  - Pick visible ground, not sky, not a wall, not a vehicle, not a person —
    open pavement below the horizon line, roughly v between 0.6 and 0.9 (not
    right at the bottom edge of the picture, and not up near the horizon).
  - Each step is short and lands you somewhere new. The next turn's
    photograph is taken fresh from where you landed; pick a new point on
    THAT one. The same two numbers typed again are almost never still
    pointing at open ground.
  - The streets leaving where you stand are still listed, with their names
    and bearings, so you can tell which general direction is walkable and
    roughly how far off the next junction is — but you do not take one by
    name."""

QUOTING_RULE_PIXEL = ("u and v are bare numbers between 0 and 1 — no quotes, "
                      "no units,\n    no brackets of their own: "
                      "{number_example}.")

# Each variant's first step is the same arrival check, ahead of any route
# reasoning -- and for the same reason ``DECIDE`` in skills.py puts it first
# for the street/waypoint action space: five real-engine full-delivery runs
# walked straight through both the pickup and the dropoff, never once calling
# collect() or hand_over(), because nothing in the pixel-goal procedure gave
# arrival a turn to be checked -- only "wait or walk". ``DECIDE`` itself
# cannot cover this: it is gated on ``walk_to``, the street tool, and its own
# wording ("keep its name, walk the way the numbers must go") does not fit a
# courier that moves by pointing at a photograph. See
# `the first real-engine runs (August 2026, in the branch history)`.
STEPS_PIXEL_ALL = """SO EVERY MOVE IS THE SAME THREE STEPS:
  1. Check where you are against the slip, first, before anything else. If
     the street and door number in "where you are" already match the pickup
     (or the dropoff, once you are carrying it), collect() or hand_over() now
     — walking on past it to be sure only carries you further away.
  2. Otherwise find the street in the list marked *** THE ROUTE GOES THIS
     WAY ***. Its bearing is the general direction to head in.
  3. If its line says the pedestrian light is RED, wait(). If its line says
     BLOCKED, that street is shut — head a different way and the marker will
     move. Otherwise look at the photograph and point at open ground that
     carries you that way."""

STEPS_PIXEL_ROUTE = """SO EVERY MOVE IS THE SAME THREE STEPS:
  1. Check where you are against the slip, first. Street and number both
     match the pickup (or the dropoff, once carried)? collect() or
     hand_over() now, rather than walking past to double check.
  2. Otherwise find the street in the list marked *** THE ROUTE GOES THIS
     WAY ***. Its bearing is the general direction to head in.
  3. Look at the photograph. Red light, or blocked? Then wait() or head a
     different way next turn. Otherwise pick a point of open pavement in the
     photograph that carries you toward that bearing, and walk_to_pixel it."""

THREE_STEPS_PIXEL = """SO EVERY MOVE IS THE SAME FOUR STEPS:
  1. Check where you are against the slip, first, before reading the map.
     Street and door number already matching the pickup (or the dropoff,
     once carried)? collect() or hand_over() now — do not walk past it to be
     sure.
  2. Otherwise look at the map. Which way does the line leave you — which
     compass point?
  3. Look at the list of streets here. Which one goes most nearly that way?
     You are not taking it by name, only reading off its bearing.
  4. Look at the photograph. Red light ahead, or blocked? Then wait() or
     let the next photograph aim a different way. Otherwise pick a point of
     open pavement that carries you toward that bearing, below the horizon,
     roughly v between 0.6 and 0.9, and walk_to_pixel it."""

# Under pixel-goal the list is still worth printing, for the same reason it
# is under coordinate mode -- but nothing in it is a step cap to quote, so
# unlike ``AIM_HINT`` this needs no ``{placeholder}`` filled from the
# environment.
AIM_HINT_PIXEL = (
    "(you do not take these by name — they tell you which general direction "
    "is\n walkable and how far the next junction is. Move by pointing at "
    "the photograph\n below: walk_to_pixel(u, v), u across and v down, both "
    "between 0 and 1. Pick\n visible pavement below the horizon, roughly v "
    "between 0.6 and 0.9.)")


# The dual-view observation needs only to explain what the street list is for.
# The system prompt below owns the complete visual movement contract once;
# repeating the same mechanics under the list on every turn made them easier to
# imitate than the image itself.
AIM_HINT_PIXEL_FRONT_REAR = (
    "(street names and bearings are orientation context, not movement calls; "
    "move using the current labelled photographs.)")


FRONT_REAR_REPLY_EXAMPLE = (
    'walk_to_pixel(view="front", u=0.37, v=0.82)')
FRONT_REAR_REPLY_EXAMPLE_2 = (
    'walk_to_pixel(view="rear", u=0.63, v=0.84)')


FRONT_REAR_SYSTEM_TEMPLATE = """You are a delivery courier working on foot in {city}. Complete pickup and
hand-over jobs against the clock.

EVIDENCE AND PRIORITY
{evidence_contract}

A late delivery still earns less; abandoning it earns nothing. Every tool uses
a turn, and its clock cost is listed below.

VISUAL MOVEMENT CONTRACT
  - One walk_to_pixel call selects one current photograph and one point local
    to it in the same call. `view` is exactly "front" or "rear". In that
    selected image, u runs left (0) to right (1), and v runs top (0) to bottom
    (1).
  - The pixel resolves to the first visible surface at that exact image point.
    It cannot pass through a plant, wall, person, vehicle, or other object to
    reach pavement hidden behind it.
  - A valid target is visibly on connected pedestrian pavement/sidewalk, or on
    a marked crosswalk when entering a roadway. Facades, objects, sky, and
    roadway outside a marked crosswalk are not pedestrian targets.
  - One locomotion action may travel several metres. A new labelled pair is
    captured at the resulting current pose. A surface-resolution refusal does
    not move you; controller feedback that reports distance or timeout may mean
    partial movement, so trust the next `where you are` and photographs.

CHOOSING A GOOD PIXEL
  - Find the pavement first: the paved band along the base of the buildings
    on your side, bounded by the kerb. The wider, darker band through the
    middle of the picture is the road, refused except on the white stripes of
    a marked crosswalk.
  - Aim a few metres ahead along that band -- a point in the lower third of
    the image (v roughly 0.65 to 0.9) that is plainly on paving. The bottom
    edge is the ground under you; the horizon is far away and rarely paving.
  - Where the band bends or continues past a corner, put the point where the
    paving visibly continues. Do not jump across the road to shorten the
    map's line; to cross, aim at the stripes of the crosswalk where the blue
    route crosses the road -- the only crossing you may use -- or walk on
    until it is in view. Stripes anywhere else are refused.
  - The rear view is a real option: select it when the route goes back the
    way you came, when you have walked past the door, or when the front view
    is filled by a wall or a window a step away.
  - After "non-walkable surface", the point hit an object or the road: move it
    onto plain paving, clear of planters, poles, vehicles and building fronts.
    After "outside a marked crosswalk", stay on this pavement, or aim at the
    stripes where the route crosses.
  - Read `where you are` after every walk; the door is reached when it names
    the slip's number. Then collect() or hand_over() -- do not walk past it.

VISUAL STATE AND REFUSALS
{state_contract}
  - "non-walkable surface" means the first visible hit was not a pedestrian
    surface. "ground ... not reachable" means it was ground outside the
    connected pedestrian navigation surface. "no way" means the point resolved
    but the movement controller could not complete a route.
  - A numerically different pixel on the same invalid object is still invalid.
    After a refusal, re-examine both current views rather than mechanically
    scanning one coordinate. Repeating the same refused call cannot help.

TOOLS
{tool_menu}

HOW TO REPLY
Reply in exactly this shape. The coordinates below demonstrate syntax only;
they are not a recommended point. Derive view, u, and v from the current images.

THOUGHT: one concise line naming the evidence you used and your conclusion.
```
{reply_example}
```

  - {call_count_rule}
  - Use exactly one listed tool name per call.
  - Put string arguments in double quotes. Write numeric arguments as bare
    numbers. No-argument tools still require parentheses.
  - Put nothing after the fenced block. Keep THOUGHT to one line so the action
    is not truncated."""


QUAD_EVIDENCE_NONE = """Each turn gives you the current job and notes; a `where you are` street and
door number; local street names and bearings for orientation; four
photographs taken at the same moment -- `front`, `left`, `right` and `rear`,
covering the whole horizon; the last result, when there is one; and a
phone-map drawing while navigation is active.

Use that evidence in this order:
  1. If `where you are` matches the pickup, collect(). If it matches the
     dropoff and you carry the parcel, hand_over(). Do not use either call to
     probe whether you have arrived.
  2. Otherwise read the phone's blue pedestrian route for its general
     direction, and find that direction among the four photographs: the
     street list's bearings say which way each photograph looks.
  3. In that photograph, pick a point plainly on the pavement a few metres
     along. You are walked there along pedestrian ways."""


QUAD_MOVEMENT_CONTRACT = """VISUAL MOVEMENT CONTRACT
  - One walk_to_pixel call selects one of this turn's four photographs and one
    point in it. `view` is exactly "front", "left", "right" or "rear". In the
    selected image, u runs left (0) to right (1), and v runs top (0) to bottom
    (1).
  - The point resolves to the first visible surface at that exact image point.
    It cannot pass through a plant, wall, person or vehicle to reach paving
    hidden behind it.
  - A good point is plainly on the pavement, or on the stripes of a marked
    crossing. You are then walked there along pedestrian ways: along the
    pavement, and over the marked crossing when the point is across the road.
    A point within about a metre of the pavement -- a kerb stone, the foot of
    a wall, the base of a planter -- counts as the paving beside it.
  - Refused, without moving you: the sky; walls and windows above the ground;
    the road away from any pavement; points too far from any pedestrian way.
  - One walk may cover many metres. Four new photographs are taken where you
    arrive; `front` is the way you were last walking.

CHOOSING A GOOD POINT
  - Look at all four photographs first. The pavement you want is often to
    your left or right, not ahead.
  - Aim a few metres along the pavement in the direction the route goes: a
    point in the lower half of the picture that is plainly paving. Do not
    aim at the paving under your feet.
  - To cross the road, aim at the far pavement or at the crossing's stripes;
    the walk takes the marked crossing for you.
  - Read `where you are` after every walk; the door is reached when it names
    the slip's number. Then collect() or hand_over() -- do not walk past it.

VISUAL STATE AND REFUSALS
{state_contract}
  - "not on the pedestrian way" means the point was not on or beside a
    pavement or marked crossing. "on the carriageway" means it was on the road
    away from any pavement. "no way" means no pedestrian way leads there.
  - After a refusal, re-examine all four photographs rather than nudging one
    coordinate. Repeating the same refused call cannot help."""


FRONT_REAR_EVIDENCE_NONE = """Each turn gives you the current job and notes; a `where you are` street and
door number; local street names and bearings for orientation; simultaneous
`front` and `rear` photographs; the last result, when there is one; and a
phone-map drawing while navigation is active.

Use that evidence in this order:
  1. If `where you are` matches the pickup, collect(). If it matches the
     dropoff and you carry the parcel, hand_over(). Do not use either call to
     probe whether you have arrived.
  2. Otherwise read the phone's blue pedestrian route only for its general
     direction. The route follows pedestrian connectivity and real marked
     crosswalks, but it cannot see a current signal phase, temporary obstacle,
     or what surface a camera pixel will hit.
  3. In the photographs, identify a visible pedestrian-walkable surface that
     carries you broadly along that route. The image decides whether a pixel is
     physically valid; a blue line on the map does not."""


FRONT_REAR_EVIDENCE_ROUTE = """Each turn states the route direction by marking one local street
`*** THE ROUTE GOES THIS WAY ***`. It also gives `where you are` and a
simultaneous `front`/`rear` photograph pair.

Use that evidence in this order:
  1. If `where you are` matches the pickup, collect(). If it matches the
     dropoff and you carry the parcel, hand_over(). Do not use either call to
     probe whether you have arrived.
  2. Otherwise use the marked street's bearing for the general direction; you
     do not need to infer it from the phone map.
  3. Use the photographs to select an exact visible pedestrian-walkable
     surface and to read current signals and obstacles. Text route guidance
     cannot make an invalid camera pixel valid."""


FRONT_REAR_EVIDENCE_ALL = """Each turn states the route direction, pedestrian-signal state, and blockage
state in the local street list. It also gives `where you are` and a simultaneous
`front`/`rear` photograph pair.

Use that evidence in this order:
  1. If `where you are` matches the pickup, collect(). If it matches the
     dropoff and you carry the parcel, hand_over(). Do not use either call to
     probe whether you have arrived.
  2. Otherwise follow the street marked `*** THE ROUTE GOES THIS WAY ***`,
     obey its stated RED or BLOCKED status, and use its bearing as the general
     direction.
  3. Still inspect the photographs to select an exact visible
     pedestrian-walkable surface. Narrated route and hazard facts cannot make
     an invalid camera pixel valid."""


FRONT_REAR_STATE_NONE = """  - A separate [light: street, bearing] photograph, when present, is the
    authoritative current pedestrian signal. If it is red, wait() once.
  - The photographs, not the phone, show current barriers and congestion. When
    live visual evidence conflicts with the route drawing, believe the images."""


FRONT_REAR_STATE_ALL = """  - The street list's stated pedestrian-signal and BLOCKED status is
    authoritative. If the route street says RED, wait() once; if it says
    BLOCKED, do not enter it.
  - The photograph still determines whether the exact selected pixel is a
    visible, connected pedestrian surface."""


_FRONT_REAR_EVIDENCE_BY_NARRATION = {
    "none": FRONT_REAR_EVIDENCE_NONE,
    "route": FRONT_REAR_EVIDENCE_ROUTE,
    "all": FRONT_REAR_EVIDENCE_ALL,
}


_FRONT_REAR_TOOL_SUMMARIES = {
    "walk_to_pixel": "Execute the visual movement contract above.",
    "collect": (
        "Collect only when `where you are` matches the pickup; a refusal means "
        "you are elsewhere."),
    "hand_over": (
        "Finish only when `where you are` matches the dropoff and the parcel "
        "has been collected."),
    "wait": (
        "Use only for a red pedestrian signal; one call advances its phase."),
    "rest": "Recover stamina at the cost of clock time; do not use it to think.",
    "look": (
        "Read house numbers along a locally listed street without walking it; "
        "it does not reveal signals or barriers."),
    "check_order": "Re-read the pickup, dropoff, deadline, and fee.",
    "check_map": (
        "Look up one address's street, walking distance, and rough bearing; "
        "this does not draw a route."),
    "navigate": (
        "Draw or refresh the pedestrian route to an address, or to the current "
        "job when omitted."),
    "note": "Keep a short fact in the notes shown on later turns.",
}


def _front_rear_tool_cost(tool: Tool) -> str:
    if tool.kind is ToolKind.ACT and not tool.time_cost_s:
        return "Cost: the walk and one turn."
    if not tool.time_cost_s:
        return "Cost: one turn, no clock time."
    return f"Cost: about {tool.time_cost_s:.0f} s and one turn."


def _render_front_rear_tool_menu(tools: list[Tool]) -> str:
    """Render the dual-view menu without repeating every tool's full manual."""

    headings = {
        ToolKind.ACT: "ACTIONS",
        ToolKind.LOOK: "LOOKING",
        ToolKind.CONSULT: "CONSULTING",
    }
    lines: list[str] = []
    for kind in (ToolKind.ACT, ToolKind.LOOK, ToolKind.CONSULT):
        chosen = [tool for tool in tools if tool.kind is kind]
        if not chosen:
            continue
        lines.append(headings[kind] + ":")
        for tool in chosen:
            summary = _FRONT_REAR_TOOL_SUMMARIES.get(tool.name, tool.summary)
            lines.append(
                f"  {tool.typed_signature()} — {summary} "
                f"{_front_rear_tool_cost(tool)}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _quad_system_template() -> str:
    """The four-view prompt: the dual-view template with its movement
    contract, pixel guidance and refusal glossary replaced."""
    head, _sep, tail = FRONT_REAR_SYSTEM_TEMPLATE.partition("VISUAL MOVEMENT CONTRACT")
    _middle, _sep2, rest = tail.partition("TOOLS\n{tool_menu}")
    return head + QUAD_MOVEMENT_CONTRACT + "\n\nTOOLS\n{tool_menu}" + rest


def _build_front_rear_system_prompt(
    *, city: str, tools: list[Tool], narration: str, action_chunk: int,
    quad: bool = False,
) -> str:
    chunk = max(1, int(action_chunk))
    call_count_rule = (
        ONE_CALL_RULE if chunk == 1
        else CHUNK_CALL_RULE.format(calls=chunk))
    examples = [FRONT_REAR_REPLY_EXAMPLE, FRONT_REAR_REPLY_EXAMPLE_2]
    reply_example = "\n".join(
        examples[index % len(examples)] for index in range(chunk))
    template = _quad_system_template() if quad else FRONT_REAR_SYSTEM_TEMPLATE
    evidence = _FRONT_REAR_EVIDENCE_BY_NARRATION.get(
        narration, FRONT_REAR_EVIDENCE_NONE)
    if quad and narration not in ("route", "all"):
        evidence = QUAD_EVIDENCE_NONE
    return template.format(
        city=city,
        evidence_contract=evidence,
        state_contract=(FRONT_REAR_STATE_ALL if narration == "all"
                        else FRONT_REAR_STATE_NONE),
        tool_menu=_render_front_rear_tool_menu(tools),
        reply_example=reply_example,
        call_count_rule=call_count_rule,
    )


def build_system_prompt(*, city: str, tools: list[Tool],
                        narration: str = "none",
                        action_chunk: int = 1,
                        special_rules: str = "") -> str:
    """Compose the system prompt from the tools this environment really has.

    ``action_chunk`` is how many calls one turn may carry. At 1 -- the default,
    and every condition that existed before chunking -- this renders exactly
    the bytes it always did; above 1 the reply rule is replaced (never
    appended to, never duplicated) by the multi-call one, because a prompt that
    said "exactly one call" *and* "up to three calls" would be a contradiction
    the policy has to guess its way out of.
    """
    # Macros are described in skills.py but no executor dispatches them, and
    # parse_reply rightly rejects a name it cannot run -- three of those in a row
    # truncates the episode. Advertising a tool that does not exist is the same
    # defect as advertising a disabled one, so they stay out of the prompt until
    # something can run them.
    names = {t.name for t in tools}
    # Advice that names a tool this condition has taken away is the same defect
    # as a menu that does: under ``no_phone`` there is no phone to tell, so the
    # sentence about telling it must go with the tool.
    # Your phone cannot see, there is nobody to tell, and it will not learn.
    # A courier that keeps taking the street the route names will keep walking
    # into the same barrier for the rest of the shift.
    blocked_advice = (
        " Your phone routes on a map. A map does not know about a skip, a"
        " barrier or roadworks, there is no way to tell it, and it will go on"
        " sending you down a street that is shut every time you ask. When the"
        " picture and the route disagree, believe the picture: take another"
        " street yourself, get past, and ask again from there."
        if "navigate" in names else
        " If a street is shut, remember it and go round; nothing will remind you."
    )
    # Even the formatting examples have to come from the live tool set. This line
    # read "walk_to(2), follow_street(2, 4)" at every stride, so the block-stride
    # prompt demonstrated the syntax of a tool the runtime would refuse.
    # One call only: the fence it lands in says "exactly one call", and a
    # second comma-joined call made the shown example a reply the parser rejects.
    #
    # And it went on being written out by hand here, which is how the same
    # defect reached the coordinate action space: a prompt whose only worked
    # example was walk_to(...) on a map where walk_to does not exist. The
    # example is the movement tool's own, whichever movement tool this
    # environment has. At MOVE_TO that is the string it always was.
    mover = next((t for t in tools if t.requires_env_action in MOVEMENT_ACTIONS
                  and t.example), None)
    coordinates = mover is not None and mover.name == "walk_to_xy"
    pixel_goal = mover is not None and mover.name == "walk_to_pixel"
    front_rear_pixel = pixel_goal and any(
        param.name == "view" for param in mover.params)
    # Dual-view pixel control has its own compact contract. Keeping this return
    # here scopes the wording change to that action space: the street,
    # coordinate, and legacy single-camera prompts below retain their exact
    # rendered bytes and benchmark baselines.
    if front_rear_pixel:
        quad = "left" in next(
            param.description for param in mover.params if param.name == "view")
        return _build_front_rear_system_prompt(
            city=city,
            tools=tools,
            narration=narration,
            action_chunk=action_chunk,
            quad=quad,
        )
    # Likewise the no-argument example: it named check_order(), which no_phone
    # takes away, so that condition's prompt demonstrated a tool it had removed.
    no_arg = next((t.name for t in tools if not t.params), "")
    no_arg_example = f"{no_arg}()" if no_arg else "a call with empty brackets"
    # A map with no movement action at all leaves nothing to demonstrate; take
    # any tool that has arguments rather than name one this map lacks.
    with_args = next((t for t in tools if t.params and t.example), None)
    number_example = (mover.example if mover else
                      with_args.example if with_args else no_arg_example)
    # Each setting is told the truth about itself and nothing else. A prompt
    # that keeps telling a narrated courier the colour is only in the picture
    # is teaching a rule that does not hold, and one that leaves the sentence
    # out of the visual setting removes the only warning that it does.
    if narration == "all":
        sources = (
            "EVERYTHING YOU NEED IS IN THE WORDS. The list of streets below\n"
            "carries all of it: which street the route takes, marked *** THE ROUTE\n"
            "GOES THIS WAY ***; whether a pedestrian light is red or green; and\n"
            "whether a street is blocked. The photographs are there to look at and\n"
            "you are not required to read anything out of them.")
    elif narration == "route":
        sources = (
            "WHERE EACH THING COMES FROM. Two sources, and you need both.\n\n"
            "  - THE LIST OF STREETS tells you which way to go. The street the\n"
            "    route takes is marked *** THE ROUTE GOES THIS WAY ***. You do not\n"
            "    have to work the direction out of the map.\n"
            "  - THE PHOTOGRAPHS are the only place a red light or a barrier\n"
            "    appears. Nothing in the words will ever tell you the colour of a\n"
            "    light or that a street is shut. Look before you commit.")
    else:
        sources = (
            "WHERE EACH THING YOU NEED COMES FROM. Three sources, and no one of "
            "them is\nenough to reach a door.\n\n"
            "  The map on your phone tells you WHICH WAY. It draws a line from "
            "where you are\n  to where you are going, and the streets are named "
            "on it.\n\n"
            "  The junction you are standing at tells you WHAT THE STREETS ARE "
            "CALLED. Only\n  the streets in that list exist for you this turn. A "
            "street anywhere else in\n  the city — including one further along "
            "your route — cannot be walked from\n  here, however clearly the "
            "line passes through it.\n\n"
            "  The photographs tell you WHETHER YOU CAN GO. Whether the "
            "pedestrian light is\n  red, whether a barrier is across the road, "
            "whether the pavement is choked:\n  none of that is in any text, "
            "here or anywhere.")
    # The three steps and the hazard rules were written for the visual world
    # and stayed put when only {sources} was swapped, so the narrated prompt
    # said "you are not required to read anything out of them" and then told
    # the courier six more times that the colour is in no text. A prompt that
    # contradicts itself teaches nothing; worse, it teaches the courier to
    # spend turns looking for a fact the words already gave it.
    if narration == "all":
        steps = (
            "SO EVERY MOVE IS THE SAME TWO STEPS:\n"
            "  1. Find the street in the list marked *** THE ROUTE GOES THIS "
            "WAY ***.\n"
            "  2. If its line says the pedestrian light is RED, wait(). If its "
            "line says\n     BLOCKED, that street is shut — take another and "
            "the marker will move.\n     Otherwise walk it.")
        hazards = (
            "Streets get blocked and streets get congested, and the line for "
            "that street\nsays so: BLOCKED means you cannot walk it at all. A "
            "crowded pavement is not\ncalled out and only costs you a little "
            "time. Crossing on a stated RED costs you\ntime and counts against "
            "you; wait() sees the phase out, and one wait is enough.")
    elif narration == "route":
        steps = (
            "SO EVERY MOVE IS THE SAME TWO STEPS:\n"
            "  1. Find the street in the list marked *** THE ROUTE GOES THIS "
            "WAY ***.\n"
            "  2. Look at that street's photograph. Red light, or blocked? Then "
            "wait() or\n     take a different street. Otherwise walk it.")
        hazards = HAZARD_RULES.format(blocked_advice=blocked_advice)
    else:
        steps = THREE_STEPS
        hazards = HAZARD_RULES.format(blocked_advice=blocked_advice)
    # The coordinate space replaces the steps rather than adding to them: the
    # last clause of every variant is "otherwise walk it", naming a call this
    # space does not have. Narration and action space are different axes --
    # what the WORDS may state, and what a CALL may name -- so all three
    # narration settings have a coordinate form.
    if coordinates:
        steps = {"all": STEPS_XY_ALL,
                 "route": STEPS_XY_ROUTE}.get(narration, THREE_STEPS_XY)
    elif pixel_goal:
        steps = {"all": STEPS_PIXEL_ALL,
                 "route": STEPS_PIXEL_ROUTE}.get(narration, THREE_STEPS_PIXEL)

    # The worked example is the rule in miniature, so it has to carry the same
    # number of calls the rule permits: a chunked prompt whose only example is
    # a single call teaches the shape it is trying to move away from. The
    # second line is another walk for the same reason the first one is -- it
    # comes out of the live tool set, not out of a tool this map may not have.
    # Optional-constraint rules ride with the hazard section: they are shift
    # rules of the same kind, and appending only when non-empty keeps the
    # default prompt byte-identical to one from before the flags existed.
    if special_rules:
        hazards = hazards + "\n\n" + special_rules
    chunk = max(1, int(action_chunk))
    call_count_rule = (ONE_CALL_RULE if chunk == 1
                       else CHUNK_CALL_RULE.format(calls=chunk))
    second = (mover.example2 or mover.example) if mover else number_example
    reply_example = (number_example if chunk == 1 else
                     f"{number_example}\n{second}")
    return SYSTEM_TEMPLATE.format(
        city=city,
        sources=sources,
        steps=steps,
        hazards=hazards,
        map_reading=(MAP_READING_PIXEL if pixel_goal else
                    MAP_READING_XY if coordinates else MAP_READING),
        streets=(STREET_RULES_PIXEL if pixel_goal else
                 STREET_RULES_XY if coordinates else STREET_RULES),
        quoting_rule=(QUOTING_RULE_PIXEL if pixel_goal else
                     QUOTING_RULE_XY if coordinates else QUOTING_RULE).format(
            number_example=number_example),
        tool_menu=render_tool_menu(tools),
        procedures=render_procedures(available=names, narration=narration),
        blocked_advice=blocked_advice,
        number_example=number_example,
        no_arg_example=no_arg_example,
        call_count_rule=call_count_rule,
        reply_example=reply_example,
    )


def build_observation(
    *,
    memory: str,
    location: str,
    clock: str = "",
    candidates: str,
    photographs: str = "",
    extra: str = "",
    take_hint: str = TAKE_STREET_HINT,
) -> str:
    """Compose one turn's observation."""
    text = OBSERVATION_TEMPLATE.format(
        memory=memory, location=location, clock=clock,
        candidates=candidates or "There is no way on from here.",
        take_hint=take_hint,
        photographs=photographs or "  (no photographs here)",
        extra=extra,
    ).rstrip() + "\n"
    # Which part of a turn actually grows. `max_tokens must be at least 1,
    # got -N` overran by a DIFFERENT N each time (-37, then -259), so the
    # growth is content-driven, and images are capped at max_images -- which
    # leaves the text. Sized in characters, per part, so the answer is a
    # measurement rather than the fifth guess.
    # warning, not info: verl configures the root logger and info from a
    # library logger does not survive it -- a measurement that is not printed
    # is not a measurement.
    _log.warning(
        "obs_size total=%d memory=%d location=%d clock=%d candidates=%d "
        "photographs=%d extra=%d",
        len(text), len(memory), len(location), len(clock),
        len(candidates or ""), len(photographs or ""), len(extra or ""),
    )
    return text


def render_candidates(rows: list[dict]) -> str:
    """The streets leaving this junction, named as a courier would name them.

    Each line carries what a rider reads off a corner: the street's name, how
    far the next junction is, which way it heads, and the house numbers that
    way. It deliberately does not say which one is correct -- that is the
    decision under test.

    The lines used to be numbered, and the courier chose by number. The number
    is stable within one junction and meaningless across junctions, so nothing
    the policy learned about a street survived walking to the next corner. A
    name and a bearing are the same everywhere, which is what makes "I have
    already tried that one" a thought the policy can have.
    """
    if not rows:
        return "There is no way on from here."
    lines = []
    for row in rows:
        parts = [f'  "{row["street"]}"']
        # Relative first, compass second. A courier on a corner decides in left
        # and right; the compass is what the phone speaks, and both are needed to
        # act on a route instruction, but only one of them is what the body does.
        #
        # The compass printed is the FIRST-EDGE bearing, for one reason that
        # outranks every other: it is the string ``match_street`` exact-matches
        # when the name is ambiguous, and this line ends by telling the model
        # to type the bearing "exactly as written". It printed the block-end
        # bearing for a while, which reads better on a curved street, and on
        # 5.6% of rows differed from the accepted one -- the same row then
        # carried one bearing here, another under its photograph, and a third
        # in the refusal that names the headings on offer. One string,
        # everywhere: the photograph caption, this line, the map banner and the
        # matcher now all quote ``row["heading"]``.
        heading = row.get("heading", "")
        # The bearing is the second half of the street's name here: it is what
        # walk_to needs, so it is printed in the words walk_to takes rather
        # than only as scenery. The relative direction stays alongside it,
        # because a courier on a corner thinks in left and right while the
        # phone speaks compass, and acting on a route needs both.
        if heading:
            parts.append(f"going {heading}"
                         + (f", {row['relative']}" if row.get("relative") else ""))
        elif row.get("relative"):
            parts.append(str(row["relative"]))
        if row.get("reach_m") is not None:
            junctions = row.get("reach_junctions", 1)
            parts.append(
                f"{row['reach_m']:.0f} m on, {junctions} junction"
                f"{'' if junctions == 1 else 's'}, to the next choice"
            )
        elif row.get("distance_m") is not None:
            parts.append(f"next junction {row['distance_m']:.0f} m")
        if row.get("numbers"):
            parts.append(f"numbers {row['numbers']}")
        if row.get("blocked_seen"):
            # What the courier saw with its own eyes last time it tried. The
            # phone is still not told; this is memory, not routing. Without it
            # the menu after a refusal is byte-identical to the menu before,
            # and a policy re-picks the barrier -- 52 of 93 times, measured.
            parts.append("(BLOCKED — you tried this and could not get past)")
        if row.get("on_route"):
            # Under the narrated settings this is the whole of the direction
            # information, so it is stated plainly rather than hinted at.
            parts.append("*** THE ROUTE GOES THIS WAY ***")
        if row.get("told_signal") == "red":
            parts.append("pedestrian light: RED")
        elif row.get("told_signal") == "green":
            parts.append("pedestrian light: green")
        if row.get("told_blocked"):
            parts.append("BLOCKED — there is a barrier across it")
        if row.get("refused"):
            # The strongest place to put a refusal is the line being chosen
            # from. Told only in prose, it was ignored: the identical call was
            # made again immediately in 63 of 128 attempts.
            parts.append(f"(REFUSED ALREADY — {row['refused']})")
        if row.get("seen"):
            parts.append("(you have walked this before)")
        lines.append(" — ".join(parts))

    body = "\n".join(lines)
    if len(rows) == 1:
        # A dead end reads as an ordinary one-line menu, and a policy used to
        # two or three choices asks for street 2. Twenty of twenty-five
        # no_such_street refusals were exactly that.
        body += "\nThis is a dead end: street 1 is the only way on."
    return body
