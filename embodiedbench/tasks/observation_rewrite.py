"""Rewrite the vendored observation into one a policy can actually act on.

The input-quality gate found three defects in what DeliveryBench shows an agent
on Paris, and all three are presentation, not simulation -- the environment knows
the right answer and renders the wrong thing:

1.  The ``### reachable_waypoints`` block is written in ``MOVE(direction=...)``,
    which Paris does not enable. Only 17.4% of its edges are near-cardinal, so
    the compiler offers ``MOVE_TO`` instead. Three of the four directions read
    "(blocked)" and the fourth is a lie about an action that will be rejected.

2.  The ``[navigation]`` block from ``NAVIGATE`` is computed once and never
    refreshed. Its ``distance_m`` read 202.2 at every one of 22 steps while the
    agent moved through twenty different addresses, and its ``next_move``
    alternated forward/backward because it was still describing the original
    pose. A policy following it oscillates between two nodes forever. This is
    worse than missing information: it is confident and wrong.

3.  Nothing in the observation relates a candidate to the goal. The agent is
    told each candidate's distance *from itself*, never whether going there
    helps, so no amount of reasoning over the text can choose.

**What this deliberately does not do.** It would be easy to print "the shortest
path is MOVE_TO(7)", and the gate would pass immediately. That replaces
navigation with copying a number: the policy would learn to find one token, and
the benchmark would measure nothing. So the rewrite supplies the two facts a
person with a compass and a street sign has -- which way the goal is, and which
way each candidate leads -- and leaves the matching to the policy. That matching
is a real decision, it is what the first-person view also supports, and it is
verifiable: a correct choice reduces the distance to the goal.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

COMPASS = ("north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west")

_REACHABLE_BLOCK = re.compile(
    r"### reachable_waypoints\n(?:.*\n)*?(?=###|\Z)", re.MULTILINE
)
_NAV_CONTEXT_BLOCK = re.compile(
    r"### ephemeral_context\n\[navigation\]\n(?:.*\n)*?(?=###|\Z)", re.MULTILINE
)
_MARK_BLOCK = re.compile(r"### waypoint_marks\n(?:.*\n?)*?(?=###|\Z)", re.MULTILINE)


def compass_of(bearing_deg: float) -> str:
    """Name a bearing, so the observation reads like directions rather than numbers."""
    index = int((bearing_deg % 360.0) / 45.0 + 0.5) % 8
    return COMPASS[index]


@dataclass
class Candidate:
    """One numbered choice, with what the policy needs to judge it."""

    index: int
    node_id: str
    address: str
    distance_m: float
    bearing_deg: float

    def render(self) -> str:
        return (
            f"- MOVE_TO({self.index}): {self.address} [{self.node_id}], "
            f"{self.distance_m:.0f} m to the {compass_of(self.bearing_deg)}"
        )


@dataclass
class Guidance:
    """Freshly computed goal-relative facts, recomputed every single turn."""

    objective: str = ""
    target_address: str = ""
    target_distance_m: float | None = None
    target_bearing_deg: float | None = None
    candidates: list[Candidate] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def render(self, max_candidates: int) -> str:
        lines = ["### objective", self.objective or "No active objective."]
        if self.target_distance_m is not None and self.target_bearing_deg is not None:
            lines.append(
                f"Your destination is {self.target_distance_m:.0f} m away, "
                f"to the {compass_of(self.target_bearing_deg)} "
                f"(bearing {self.target_bearing_deg:.0f} deg)."
            )
            lines.append(
                "Choose the numbered waypoint that moves you toward it. Each waypoint "
                "below lists the direction it lies in."
            )
        if self.candidates:
            # Nearest first. A policy scanning a list reads the top; putting the
            # plausible short hops there keeps a long tail from burying them.
            shown = sorted(self.candidates, key=lambda c: c.distance_m)[:max_candidates]
            lines.append("")
            lines.append(f"### waypoint_marks ({len(shown)} of {len(self.candidates)} shown)")
            lines.extend(candidate.render() for candidate in shown)
        else:
            lines.append("")
            lines.append("### waypoint_marks")
            lines.append("No reachable waypoint from here.")
        return "\n".join(lines)


def bearing_between(from_x: float, from_y: float, to_x: float, to_y: float) -> float:
    return math.degrees(math.atan2(to_y - from_y, to_x - from_x)) % 360.0


# The engine's own mark line. Parsing it rather than renumbering is deliberate:
# `_mark_candidates` numbers from 0 and sorts by direction key, while the printed
# block numbers from 1. Any independent numbering risks MOVE_TO(k) moving
# somewhere other than the line the policy read, which is unfalsifiable from the
# outside and would corrupt every episode silently.
_VENDOR_MARK = re.compile(
    r"MOVE_TO\((\d+)\):\s*(\S+)\s*\(([^)]*)\),\s*([\d.]+)\s*m"
)


def compute_guidance(
    runtime: Any, observation_text: str, *, max_candidates: int = 8
) -> Guidance:
    """Recompute goal-relative facts from the live environment state.

    The *environment* may consult the graph; the policy may not. That asymmetry
    is the point -- the defect being fixed is that the environment knew the agent
    had moved and kept printing a stale answer anyway.
    """
    guidance = Guidance()
    agent = runtime._dm()
    if agent is None:
        guidance.notes.append("no delivery agent in this environment")
        return guidance

    pose = runtime._agent_pose()
    if pose is None:
        guidance.notes.append("agent pose unavailable")
        return guidance
    x_cm, y_cm = pose.position.x_cm, pose.position.y_cm

    orders = list(getattr(agent, "active_orders", None) or [])
    if not orders:
        guidance.objective = "You have no accepted order. Accept one to begin."
    else:
        order = orders[0]
        carrying = bool(getattr(agent, "carrying", None) or getattr(agent, "inventory", None))
        target = (
            getattr(order, "delivery_address", None)
            if carrying
            else getattr(order, "pickup_address", None)
        )
        guidance.objective = (
            "Deliver the order to its customer." if carrying
            else "Collect the order from its restaurant."
        )
        if target is not None:
            tx, ty = float(target.x), float(target.y)
            guidance.target_distance_m = math.hypot(tx - x_cm, ty - y_cm) / 100.0
            guidance.target_bearing_deg = bearing_between(x_cm, y_cm, tx, ty)
            guidance.target_address = "your destination"

    positions = _node_positions(agent)
    for match in _VENDOR_MARK.finditer(observation_text or ""):
        node_id = match.group(2)
        position = positions.get(node_id)
        if position is None:
            # A candidate whose node cannot be located gets no bearing rather
            # than a fabricated one; the gate reports it as a defect.
            guidance.notes.append(f"candidate {node_id} is not in the graph")
            continue
        guidance.candidates.append(Candidate(
            index=int(match.group(1)),
            node_id=node_id,
            address=match.group(3),
            distance_m=float(match.group(4)),
            bearing_deg=bearing_between(x_cm, y_cm, position[0], position[1]),
        ))
    return guidance


def _node_positions(agent: Any) -> dict[str, tuple[float, float]]:
    adjacency = agent.city_map.waypoint_graph.adjacency_list
    return {
        str(getattr(node, "waypoint_id", "") or ""): (
            float(node.position.x), float(node.position.y)
        )
        for node in adjacency
    }


def rewrite_observation(
    text: str,
    guidance: Guidance,
    *,
    enabled_actions: list[str],
    max_candidates: int = 8,
) -> str:
    """Strip the misleading blocks and substitute freshly computed guidance."""
    out = text or ""
    # The directional block describes an action this map does not enable.
    if "MOVE" not in set(enabled_actions):
        out = _REACHABLE_BLOCK.sub("", out)
    # The stale routing block is removed outright rather than corrected: it is
    # recomputed below, and leaving both would give the policy two answers.
    out = _NAV_CONTEXT_BLOCK.sub("", out)
    out = _MARK_BLOCK.sub("", out)
    out = re.sub(r"\n{3,}", "\n\n", out).rstrip()
    return f"{out}\n\n{guidance.render(max_candidates)}\n"
