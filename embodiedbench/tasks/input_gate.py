"""Is the observation enough to act on? An input-quality gate. Rule-based, no AI.

A reward function can be perfect and an environment still untrainable, because
the policy is never told what it needs to decide. That failure is invisible from
the environment side: episodes run, rewards come back zero, and the natural
reading is "the model is weak" when the truth is "the prompt never contained the
answer".

So this gate judges the *input*, and it does it two ways, because the cheap way
alone is not convincing.

**Static checks** read one rendered observation and test properties that can be
settled by inspection. The one that matters most on Paris is agreement between
what the observation tells the agent to do and what the environment will accept:
the vendored ``NAVIGATE`` helper answers with ``next_move: move backward``, and
Paris does not enable ``MOVE`` at all -- only 17.4% of its edges are near-cardinal,
so the compiler offers ``MOVE_TO`` instead. A policy that follows the routing hint
literally emits an action the environment rejects, every turn, forever.

**An observation-only oracle** is the real test. It is a policy with perfect
reasoning and no privileged access: it may read the rendered observation text and
nothing else -- no graph, no coordinates, no internal state. If that policy
cannot make progress, the observation is missing information, and no VLM will
recover it. If it can, the input is sufficient and any remaining failure belongs
to the model.

The oracle is deliberately not a strong planner. It follows the routing hint and
the numbered candidates, which is the intended loop; making it cleverer would let
it compensate for a bad observation and hide the very defect the gate exists to
find.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from embodiedbench.schemas.env_spec import EnvSpec

# Bearings the vendored engine prints beside each numbered candidate.
_BEARING_WORDS = {
    "ahead": "forward",
    "in front": "forward",
    "front": "forward",
    "behind you": "backward",
    "behind": "backward",
    "to your left": "left",
    "left": "left",
    "to your right": "right",
    "right": "right",
}
# Every action name the observation might tell the agent to use.
_ACTION_MENTION = re.compile(r"\b([A-Z][A-Z_]{2,})\s*\(")
# Two shapes are accepted: the vendored line, which gives a facing-relative
# bearing ("behind you"), and the rewritten line, which gives an absolute compass
# direction. The gate has to be able to judge either, since judging only the one
# we produce would never catch a regression in the raw observation.
_MARK_LINE = re.compile(r"MOVE_TO\((\d+)\):\s*(\S+)\s*\(([^)]*)\),\s*([\d.]+)\s*m,\s*(.+)")
_REWRITTEN_MARK = re.compile(
    r"MOVE_TO\((\d+)\):\s*(.+?)\s*\[([^\]]+)\],\s*([\d.]+)\s*m to the (\S+)"
)
_OBJECTIVE = re.compile(
    r"Your destination is\s+([\d.]+)\s*m away, to the (\S+)\s*\(bearing\s*([\d.]+)"
)
COMPASS_DEG = {
    "north": 0.0, "north-east": 45.0, "east": 90.0, "south-east": 135.0,
    "south": 180.0, "south-west": 225.0, "west": 270.0, "north-west": 315.0,
}
_AT_ADDRESS = re.compile(r"you are at\s+([^.]+?)\.")
_PICKUP_ADDRESS = re.compile(r"Pickup\s*:\s*(.+)")
_DROPOFF_ADDRESS = re.compile(r"Dropoff\s*:\s*(.+)")
_NEXT_MOVE = re.compile(r"next_move:\s*(.+)")
_NAV_DISTANCE = re.compile(r"distance_m:\s*([\d.]+)")

# A policy asked to pick among more than this many numbered candidates is being
# set an indexing problem rather than a navigation one. Paris post-repair reaches
# degree 35 at its worst junction.
MAX_WORKABLE_CHOICES = 12


@dataclass
class Finding:
    """One thing wrong with the input, and why it matters."""

    code: str
    severity: str  # "blocking" | "degrading"
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "severity": self.severity, "detail": self.detail}


@dataclass
class MarkCandidate:
    """One numbered choice as the observation presents it."""

    index: int
    node_id: str
    address: str
    distance_m: float
    bearing: str

    @property
    def direction(self) -> str | None:
        text = self.bearing.strip().lower()
        for phrase, direction in _BEARING_WORDS.items():
            if text.startswith(phrase):
                return direction
        return None

    @property
    def bearing_deg(self) -> float | None:
        """Absolute bearing, when the observation states one."""
        return COMPASS_DEG.get(self.bearing.strip().lower())


def parse_marks(observation: str) -> list[MarkCandidate]:
    """Read the numbered candidates out of the observation, as a policy would."""
    candidates: list[MarkCandidate] = []
    for match in _REWRITTEN_MARK.finditer(observation or ""):
        try:
            candidates.append(
                MarkCandidate(
                    index=int(match.group(1)),
                    node_id=match.group(3),
                    address=match.group(2),
                    distance_m=float(match.group(4)),
                    bearing=match.group(5).strip().rstrip("."),
                )
            )
        except ValueError:
            continue
    if candidates:
        return candidates
    for match in _MARK_LINE.finditer(observation or ""):
        try:
            candidates.append(
                MarkCandidate(
                    index=int(match.group(1)),
                    node_id=match.group(2),
                    address=match.group(3),
                    distance_m=float(match.group(4)),
                    bearing=match.group(5).strip(),
                )
            )
        except ValueError:
            continue
    return candidates


def mentioned_actions(observation: str) -> set[str]:
    """Action names the observation tells the agent to use."""
    return {m.group(1) for m in _ACTION_MENTION.finditer(observation or "")}


def objective(observation: str) -> tuple[float, float] | None:
    """``(distance_m, bearing_deg)`` to the goal, as the observation states it."""
    match = _OBJECTIVE.search(observation or "")
    if not match:
        return None
    return (float(match.group(1)), float(match.group(3)))


def angular_gap(a_deg: float, b_deg: float) -> float:
    """Smallest angle between two bearings, so 350 and 10 are 20 apart."""
    return abs((a_deg - b_deg + 180.0) % 360.0 - 180.0)


def routing_hint(observation: str) -> tuple[str | None, float | None]:
    """The ``next_move`` and ``distance_m`` from a NAVIGATE response, if present."""
    move = _NEXT_MOVE.search(observation or "")
    distance = _NAV_DISTANCE.search(observation or "")
    return (
        move.group(1).strip() if move else None,
        float(distance.group(1)) if distance else None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Static checks
# ─────────────────────────────────────────────────────────────────────────────


def check_observation(observation: str, env: EnvSpec) -> list[Finding]:
    """Judge a single rendered observation against the environment's action space."""
    findings: list[Finding] = []
    enabled = set(env.enabled_actions)

    # 1. Everything the observation tells the agent to do must be executable.
    mentioned = mentioned_actions(observation)
    unexecutable = {a for a in mentioned if a not in enabled and a.isupper()}
    if unexecutable:
        findings.append(Finding(
            code="observation_advertises_disabled_action",
            severity="blocking",
            detail=(
                f"the observation names {sorted(unexecutable)} but this environment "
                f"enables {sorted(enabled)}. A policy that follows the observation "
                "literally emits a rejected action every turn."
            ),
        ))

    # 2. The routing hint has to speak the enabled action's language. This is the
    #    Paris defect: NAVIGATE answers "move backward" on a map where MOVE is
    #    disabled precisely because its streets are not cardinal.
    hint, _distance = routing_hint(observation)
    if hint and "MOVE" not in enabled:
        if re.search(r"\bmove\s+(forward|backward|left|right)\b", hint, re.IGNORECASE):
            findings.append(Finding(
                code="routing_hint_uses_disabled_move",
                severity="blocking",
                detail=(
                    f"the routing hint says {hint!r}, a directional MOVE, but this map "
                    "does not enable MOVE. The hint has to be translated to a numbered "
                    "MOVE_TO candidate before a policy can follow it."
                ),
            ))

    # 3. If MOVE_TO is the action, the observation must enumerate the choices.
    marks = parse_marks(observation)
    if "MOVE_TO" in enabled and not marks:
        findings.append(Finding(
            code="no_enumerated_choices",
            severity="blocking",
            detail=(
                "MOVE_TO is the primary action but the observation lists no numbered "
                "waypoints, so the policy has nothing to name. Set "
                "enable_waypoint_marks on the runtime."
            ),
        ))

    # 4. A choice the policy cannot reliably index is not a choice.
    if len(marks) > MAX_WORKABLE_CHOICES:
        findings.append(Finding(
            code="too_many_choices",
            severity="degrading",
            detail=(
                f"{len(marks)} numbered candidates exceeds the {MAX_WORKABLE_CHOICES} "
                "a policy can index reliably; this measures list handling rather than "
                "navigation."
            ),
        ))
    if len(marks) == 1:
        findings.append(Finding(
            code="no_real_choice",
            severity="degrading",
            detail=(
                "exactly one candidate, so the step carries no decision. Common at "
                "dead-end nodes and harmless in isolation, but an episode made of "
                "these measures nothing."
            ),
        ))

    # 5. Each candidate must be distinguishable from the others.
    if marks:
        if len({m.index for m in marks}) != len(marks):
            findings.append(Finding(
                code="duplicate_choice_numbers",
                severity="blocking",
                detail="two candidates share a number, so MOVE_TO(k) is ambiguous.",
            ))
        # Either form of bearing counts: facing-relative from the vendored
        # block, or absolute compass from the rewritten one. Checking only the
        # facing-relative one reported every rewritten candidate as defective.
        without_bearing = [
            m.index for m in marks if m.direction is None and m.bearing_deg is None
        ]
        if without_bearing:
            findings.append(Finding(
                code="candidate_without_bearing",
                severity="degrading",
                detail=(
                    f"candidates {without_bearing} carry no recognisable bearing, so a "
                    "routing hint cannot be matched to them."
                ),
            ))

    # 6. The agent must be told what it is trying to do.
    if "### active_orders" not in observation and "Pickup" not in observation:
        findings.append(Finding(
            code="no_stated_objective",
            severity="blocking",
            detail="the observation states no objective, so no action can be judged better than another.",
        ))
    return findings


# ─────────────────────────────────────────────────────────────────────────────
# The observation-only oracle
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class OracleStep:
    step: int
    action: str
    reason: str
    distance_m: float | None = None
    choices: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step, "action": self.action, "reason": self.reason,
            "distance_m": self.distance_m, "choices": self.choices,
        }


class ObservationOnlyOracle:
    """A perfect reasoner restricted to what the observation says.

    Its whole value is the restriction. It may read the rendered text and nothing
    else -- given the graph or the agent's coordinates it would navigate happily
    on an observation that tells a real policy nothing, and the gate would pass
    an unusable input.

    The strategy is the loop the observation is designed for and no more: ask for
    a route, read the recommended direction, pick the numbered candidate whose
    bearing matches it, and act on the task when the observation says to. Making
    it a better planner would let it compensate for a poor observation, which is
    the one thing this must not do.
    """

    def __init__(self, enabled_actions: list[str]):
        self.enabled = set(enabled_actions)
        self.have_navigated = False
        self.last_distance: float | None = None
        self.visited_marks: set[str] = set()
        # The node stepped away from last turn. Greedy bearing-following walks
        # into 2-cycles: at 13 m from the goal the oracle bounced between two
        # nodes for the rest of the episode because each looked best from the
        # other. Refusing an immediate reversal is the smallest fix that is
        # still about *planning* rather than about information -- it adds no
        # knowledge the observation lacks, so it cannot mask an input defect.
        self.previous_node: str | None = None
        self.current_node: str | None = None

    def act(self, observation: str) -> tuple[str, dict[str, Any], str]:
        """Return ``(action_name, arguments, reason)`` from the observation alone."""
        text = observation or ""
        marks = parse_marks(text)
        hint, distance = routing_hint(text)

        # Accept work when there is none in hand.
        if "You currently have no accepted orders" in text:
            if not self.have_navigated and "VIEW_ORDERS" in self.enabled:
                pass
            if "ACCEPT_ORDER" in self.enabled and "[Order #" in text:
                return "ACCEPT_ORDER", {"_args": [0]}, "an order is offered and none is held"
            if "VIEW_ORDERS" in self.enabled:
                return "VIEW_ORDERS", {}, "no orders held and none listed"

        # Act on arrival. The observation never says "you are at the
        # restaurant" -- it prints the agent's street address and the order's,
        # and leaves the comparison to the reader. Matching those two strings is
        # something a policy can do from the text alone, which is the only kind
        # of rule allowed here.
        here = _AT_ADDRESS.search(text)
        if here:
            location = here.group(1).strip().lower()
            pickup = _PICKUP_ADDRESS.search(text)
            dropoff = _DROPOFF_ADDRESS.search(text)
            if (
                "PICKUP" in self.enabled
                and "Ready for pickup" in text
                and pickup
                and pickup.group(1).strip().lower() == location
            ):
                return "PICKUP", {"orders": [0]}, f"standing at the pickup address {location!r}"
            if (
                "DROP_OFF" in self.enabled
                and dropoff
                and dropoff.group(1).strip().lower() == location
                and "Ready for pickup" not in text
            ):
                return "DROP_OFF", {"oid": 0}, f"standing at the dropoff address {location!r}"

        # Preferred rule: steer by bearing. The rewritten observation states
        # where the goal lies and where each candidate leads, so a policy that
        # can compare two directions can navigate. This is the check that
        # matters -- it needs no routing helper and no privileged state.
        goal = objective(text)
        if goal is not None and marks:
            _goal_distance, goal_bearing = goal
            self.last_distance = _goal_distance
            aligned = [m for m in marks if m.bearing_deg is not None]
            # Only forbid an immediate reversal, and only when there is
            # somewhere else to go; at a genuine dead end, turning back is
            # correct. Preferring *unvisited* nodes instead was tried and is
            # worse -- it pushes the agent away from the goal down side streets
            # and ended 119 m out against this rule's 72 m.
            forward_only = [m for m in aligned if m.node_id != self.previous_node]
            if forward_only:
                aligned = forward_only
            if aligned:
                # Prefer the candidate pointing most nearly at the goal; break
                # ties toward the nearer one so the agent does not commit to a
                # long edge when a short one heads the same way.
                chosen = min(
                    aligned,
                    key=lambda m: (round(angular_gap(m.bearing_deg, goal_bearing) / 45.0),
                                   m.distance_m),
                )
                gap = angular_gap(chosen.bearing_deg, goal_bearing)
                self.previous_node = self.current_node
                self.current_node = chosen.node_id
                return (
                    "MOVE_TO", {"_args": [chosen.index]},
                    f"goal bears {goal_bearing:.0f} deg; candidate {chosen.index} "
                    f"bears {chosen.bearing_deg:.0f} deg (off by {gap:.0f})",
                )

        # Ask for a route when the observation carries no current one.
        if hint is None and "NAVIGATE" in self.enabled:
            self.have_navigated = True
            return "NAVIGATE", {"target": "restaurant 1"}, "no routing hint in view"

        if distance is not None:
            self.last_distance = distance

        # Follow the hint by matching its direction to a candidate's bearing.
        if marks and hint:
            wanted = None
            for word in ("forward", "backward", "left", "right"):
                if word in hint.lower():
                    wanted = word
                    break
            if wanted:
                matching = [m for m in marks if m.direction == wanted]
                if matching:
                    chosen = min(matching, key=lambda m: m.distance_m)
                    return (
                        "MOVE_TO", {"_args": [chosen.index]},
                        f"hint {hint!r} matches candidate {chosen.index} bearing "
                        f"{chosen.bearing!r}",
                    )
            # The hint named a direction no candidate offers. That is exactly the
            # information gap this oracle exists to expose, so it is reported
            # rather than papered over with a guess.
            unvisited = [m for m in marks if m.node_id not in self.visited_marks]
            if unvisited:
                chosen = min(unvisited, key=lambda m: m.distance_m)
                self.visited_marks.add(chosen.node_id)
                return (
                    "MOVE_TO", {"_args": [chosen.index]},
                    f"hint {hint!r} matches no candidate bearing; fell back to nearest unvisited",
                )

        if marks:
            chosen = min(marks, key=lambda m: m.distance_m)
            return "MOVE_TO", {"_args": [chosen.index]}, "no hint; took the nearest candidate"
        if "WAIT" in self.enabled:
            return "WAIT", {}, "observation offers no actionable information"
        return "VIEW_ORDERS", {}, "nothing else is available"


@dataclass
class GateReport:
    """Whether this environment's input can be acted on."""

    map_name: str
    static_findings: list[Finding] = field(default_factory=list)
    steps: list[OracleStep] = field(default_factory=list)
    rejected_actions: int = 0
    hint_match_failures: int = 0
    distance_start_m: float | None = None
    distance_end_m: float | None = None
    reward_total: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def blocking(self) -> list[Finding]:
        return [f for f in self.static_findings if f.severity == "blocking"]

    @property
    def made_progress(self) -> bool:
        if self.distance_start_m is None or self.distance_end_m is None:
            return False
        return self.distance_end_m < self.distance_start_m

    @property
    def passed(self) -> bool:
        # Progress alone is not enough: an input that produces rejected actions
        # is broken even if the oracle stumbles forward anyway.
        return not self.blocking and self.rejected_actions == 0 and self.made_progress

    def to_dict(self) -> dict[str, Any]:
        return {
            "map": self.map_name,
            "static_findings": [f.to_dict() for f in self.static_findings],
            "blocking_count": len(self.blocking),
            "oracle_steps": len(self.steps),
            "rejected_actions": self.rejected_actions,
            "hint_match_failures": self.hint_match_failures,
            "distance_start_m": self.distance_start_m,
            "distance_end_m": self.distance_end_m,
            "made_progress": self.made_progress,
            "reward_total": round(self.reward_total, 4),
            "trace": [s.to_dict() for s in self.steps[:40]],
            "notes": self.notes,
            "status": "pass" if self.passed else "fail",
        }
