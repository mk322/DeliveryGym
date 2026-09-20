"""The courier's skills: procedures, and the small set the harness may execute.

"Skill" is doing two jobs in most agent frameworks, and separating them matters
here more than usual, because one of the two can quietly destroy the benchmark.

**Procedural skills** are written guidance -- a runbook the rider was trained on.
mini-SWE-agent carries the same thing as numbered workflow steps in its instance
template ("reproduce first, then edit, then verify"). They cost nothing, they
constrain nothing, and the model may ignore them. Everything genuinely about
*navigating* lives here, because that is the capability under test: a harness
that computes the route and hands over a waypoint number is measuring itself.

**Executable skills** are macros the harness runs. Each one is a fixed expansion
into tool calls that a competent rider would not think about individually --
walking four junctions down the same street is one decision, not four. Admitting
a macro requires it to be *mechanical*: it must not choose a direction, rank a
candidate, or consult the graph. ``follow_street`` walks the street the model
named until the street ends or something changes; it never picks which street.
``retrace`` walks back along a trail the agent already made. Neither can turn a
lost agent into a found one, which is the test each had to pass to be included.

The split is enforced, not merely documented: an executable skill declares the
tools it expands into, and a skill that tried to expand into a route computation
would have nothing to declare.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from embodiedbench.agent.courier.tools import FOLLOW_STREET as FOLLOW_STREET_TOOL
from embodiedbench.agent.courier.tools import Tool, ToolParam


# ─────────────────────────────────────────────────────────────────────────────
# Procedural skills: guidance, not code
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Step:
    """One line of a runbook, optionally gated on a tool being callable.

    Gating whole procedures was not enough. ``Finding an address`` requires
    ``navigate`` and ``walk_to``, both of which exist at every stride, but one of
    its *steps* named ``follow_street``, which exists only at waypoint stride --
    so the block-stride prompt advertised a tool the runtime would refuse, and a
    policy that followed its own instructions lost a turn to a format error.
    A step names a tool, so a step is what has to be gated.
    """

    text: str
    requires: tuple[str, ...] = ()


@dataclass(frozen=True)
class Procedure:
    """A named runbook the courier is expected to know."""

    name: str
    when: str
    steps: tuple[str | Step, ...]
    # Tools this runbook tells the courier to use. A procedure whose tools are
    # not available is not shown: guidance that names an action the environment
    # will refuse is the same defect as a tool menu that does, and it slipped
    # back in here after being fixed one layer up.
    requires: tuple[str, ...] = ()
    # Tools whose presence makes this runbook redundant. Two runbooks for one job
    # is worse than one: the courier has to decide which it is in, and the prompt
    # pays for both.
    excluded_by: tuple[str, ...] = ()

    def render(self, available: set[str] | None = None) -> str:
        lines = [
            step.text if isinstance(step, Step) else step
            for step in self.steps
            if available is None
            or not isinstance(step, Step)
            or all(tool in available for tool in step.requires)
        ]
        body = "\n".join(f"    {i}. {s}" for i, s in enumerate(lines, 1))
        return f"  {self.name} — {self.when}\n{body}"


DECIDE = Procedure(
    name="Working out your move",
    when="every turn, before anything else — stop at the first line that fits",
    # The other runbooks each answer one situation well and say nothing about
    # which situation you are in. Measured on Qwen3-VL-4B over 40 episodes, that
    # gap is where the task is lost: it reached the slip's street in 22 of 40
    # episodes and then walked off it 81 times, and once on the right street its
    # next move was toward the right number only 37 times against 25 away --
    # barely better than a coin. Ordering the questions is the missing piece.
    steps=(
        Step("At the address? The \"where you are\" line says whether this is the "
             "street on the slip and which number it wants. If the street matches "
             "and the numbers here include that one, act now — do not walk on to "
             "be sure.",
             requires=("collect",)),
        "On the slip's street at the wrong number? The door is along this "
        "street; keep its name, walk the way the numbers must go. Leaving the "
        "right street is the most expensive mistake available to you.",
        Step("If you cannot tell which way the numbers run, "
             "look(\"street\", \"bearing\") reads them without walking it.",
             requires=("look",)),
        Step("Not on the slip's street? Then you are travelling: navigate() once, "
             "take the street its first instruction names, and do not ask again "
             "until you have made that turn.",
             requires=("navigate",)),
        "Before you commit to a street, look at its photograph. A street already "
        "marked (BLOCKED) stays blocked for the rest of the shift.",
    ),
    requires=("walk_to",),
)

FIND_ADDRESS = Procedure(
    name="Finding an address",
    when="you know the address but not where it is",
    steps=(
        "navigate() for the route: which street, which turn, how far.",
        "Read which way the line leaves you and take the street here whose "
        "bearing is nearest it; only streets in the list can be walked.",
        Step("To stay on one street for several junctions, "
             "follow_street(\"street\", \"bearing\", n) does them in one turn.",
             requires=("follow_street",)),
        "navigate() again when you have made the turn, or when what you see stops "
        "matching what it said.",
    ),
    requires=("navigate", "walk_to"),
)

FIND_ADDRESS_NO_PHONE = Procedure(
    name="Finding an address without a route",
    when="you know the address but the phone will not give you a route",
    steps=(
        "The turn already names the street you are standing on and the doors at your "
        "feet; the numbers run in order, odd one side and even the other.",
        "If you are on the right street, look() down it to see whether the "
        "numbers climb or fall, then walk the way they climb toward yours.",
        "If you are not, take a street heading the right way and read the numbers "
        "again at the next junction to check you are closer.",
    ),
    # Shown exactly when the route tool is missing, so the courier is never given
    # two runbooks for one job.
    requires=("look", "walk_to"),
    excluded_by=("navigate",),
)

DEAD_END = Procedure(
    name="When a street ends",
    when="a street runs out or the only way on is back",
    steps=(
        "note() that this way was a dead end, with the street name.",
        "Walk back to the last junction that had another street leaving it.",
        "Take a different street — not the one you arrived by.",
    ),
    requires=("note", "walk_to"),
)

LOST = Procedure(
    name="When you are going in circles",
    # The old wording -- "your notes say you have passed somewhere more than
    # twice" -- cited a signal the observation does not carry. The notes hold a
    # "Came from" trail of recent places; nothing counts visits or flags a
    # repeat. A runbook that tells the courier to read something that is not
    # written anywhere is worse than no runbook: it reads as followed advice
    # while nothing was checked. Measured on Qwen3-VL-4B: 26 of 41 moves
    # returned to a junction already visited, 12 distinct junctions in a whole
    # episode.
    when="a street name in \"Came from\" is one you are about to walk again",
    steps=(
        "Stop repeating the last turn; it is the one that brought you back.",
        "Re-anchor on the street named at the top of the turn and the doors beside it.",
        "Prefer a street you have not walked yet, even if it looks less direct.",
        "Walking back the way you came is legal and at a dead end it is the "
        "only move; refusing to backtrack spends the shift standing still.",
    ),
    requires=("walk_to",),
)

WAY_SHUT = Procedure(
    name="When the way is shut",
    when="a photograph shows a barrier across the street the route wants",
    steps=(
        "Believe the picture: the map cannot see the barrier and will send "
        "you the same way again.",
        "Take another street yourself. The map on your phone shows the layout: "
        "pick one that runs the same way and rejoin further along.",
        "Ask for a route again once you are past it, from where you now are.",
    ),
    requires=("walk_to",),
)

DEADLINE = Procedure(
    name="Watching the clock",
    when="an order has a deadline",
    steps=(
        "A late delivery still scores, but less. An abandoned one scores nothing.",
        Step("wait() is for a red pedestrian light and nothing else.",
             requires=("wait",)),
        "Every tool costs time, looking and consulting included. Do not spend turns on "
        "them when the way is already clear.",
    ),
    requires=(),
)

ARRIVAL = Procedure(
    name="Knowing you have arrived",
    when="you think you are at the pickup or the dropoff",
    steps=(
        "Compare the street and door numbers at the top of the turn to the slip, "
        "exactly.",
        "Only collect() at the pickup and hand_over() at the dropoff; a "
        "refusal is not a search tool.",
        "If the number is close but wrong, walk one more junction the way the numbers "
        "are going.",
    ),
    requires=("collect", "hand_over"),
)

CROSSINGS = Procedure(
    name="Crossing at a light",
    when="a photograph of a pedestrian light is shown for the street you want",
    steps=(
        "Read the lamp in the [light: street, bearing] photograph, not one in a street view.",
        "Red — wait(), once; it sees the phase out.",
        "Green, or no lamp photograph for that street — walk on.",
    ),
    requires=("wait", "walk_to"),
)

# The lights were scored and never mentioned. ``walk_to`` charged 45 s and a
# reward penalty for crossing on red at 105 junctions, the only mechanic in this
# environment that genuinely requires looking at a picture, and no line of the
# system prompt told the courier that a light existed, that the photograph was
# where to find it, or that ``wait()`` -- described as "wait for a moment" -- was
# what to do about it. A rule the agent is graded on and never told is not a
# difficulty, it is a scoring error.
PROCEDURES: tuple[Procedure, ...] = (
    DECIDE, FIND_ADDRESS, FIND_ADDRESS_NO_PHONE, ARRIVAL, CROSSINGS, WAY_SHUT,
    DEAD_END, LOST, DEADLINE,
)


# Runbooks whose steps are about reading a photograph. Under narration="all"
# the same facts arrive in the text, so these would send the courier looking
# for something it has already been told -- the prompt telling it twice, in
# two places, to do two different things.
PHOTOGRAPH_KEYED = ("When the way is shut", "Crossing at a light")

NARRATED_HAZARDS = """  Crossing at a light — the street's line says the pedestrian light is RED
    1. wait(). One wait sees the phase out, so one is always enough.
    2. When no colour is stated for a street, there is no light on it: walk on.
  When the way is shut — the street's line says BLOCKED
    1. That street cannot be walked at all. Do not try it to make sure.
    2. Take another street; the route marker will move to whichever way is
       open, so there is always something to follow."""


def render_procedures(
    procedures: tuple[Procedure, ...] = PROCEDURES,
    available: set[str] | None = None,
    narration: str = "none",
) -> str:
    """Render only the runbooks whose tools this environment can execute."""
    chosen = [
        p for p in procedures
        if available is None or (
            all(tool in available for tool in p.requires)
            and not any(tool in available for tool in p.excluded_by)
        )
    ]
    if narration == "all":
        chosen = [p for p in chosen if p.name not in PHOTOGRAPH_KEYED]
        return "\n".join([p.render(available) for p in chosen]
                         + [NARRATED_HAZARDS])
    return "\n".join(p.render(available) for p in chosen)


# ─────────────────────────────────────────────────────────────────────────────
# Executable skills: macros the harness runs
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Macro:
    """A fixed expansion into tool calls.

    ``expands_to`` is the whole safety argument: a macro may only be built from
    tools the agent could have called itself, in an order fixed in advance. A
    macro that needed to decide something would have to name a tool that decides,
    and there is none.
    """

    tool: Tool
    expands_to: tuple[str, ...]
    rationale: str
    max_expansion: int


FOLLOW_STREET = Macro(
    # One declaration, shared with the tool menu and the dispatcher, so the macro
    # described here and the tool the runtime executes cannot drift apart.
    tool=FOLLOW_STREET_TOOL,
    expands_to=("walk_to",),
    rationale=(
        "Walking four junctions down one street is one decision for a rider, not four. "
        "The macro never chooses the street -- the model names it -- and it stops the "
        "moment the situation changes, so it cannot walk an agent past the turn it "
        "should have taken."
    ),
    max_expansion=6,
)

RETRACE = Macro(
    tool=Tool(
        name="retrace",
        kind=__import__(
            "embodiedbench.agent.courier.tools", fromlist=["ToolKind"]
        ).ToolKind.ACT,
        summary="Walk back the way you came, up to n junctions.",
        params=(ToolParam("n", "int", "how many junctions to walk back, at most 4"),),
        example="retrace(2)",
    ),
    expands_to=("walk_to",),
    rationale=(
        "Backtracking is mechanical: the trail is already in memory and the agent "
        "walked it itself. It buys nothing the agent could not do one step at a time, "
        "and it stops a dead-end recovery from eating the whole step budget."
    ),
    max_expansion=4,
)

MACROS: tuple[Macro, ...] = (FOLLOW_STREET, RETRACE)
MACROS_BY_NAME: dict[str, Macro] = {m.tool.name: m for m in MACROS}


def render_macros(macros: tuple[Macro, ...] = MACROS) -> str:
    return "\n".join("  " + m.tool.describe().replace("\n", "\n  ") for m in macros)
