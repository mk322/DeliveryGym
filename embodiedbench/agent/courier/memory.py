"""What the courier remembers between turns.

A VLM sees one frame at a time and forgets everything the moment the context
window rolls. On a street network that is fatal in a specific, measurable way:
the observation-only oracle circled a three-node loop 21 m from its destination
for the rest of an episode, because from each node the next looked best, and it
had no way to know it had already been there.

So memory is a harness responsibility, not a model one. It is also the part of
the design most easily got wrong by making it too clever: a harness that plans
the route for the model turns the benchmark into a test of the harness. The rule
followed here is that memory may record **what happened**, never **what to do**.

Four stores, each with a reason a rider would recognise:

trail       where I have been, in order, and how many times. A rider notices
            they are going in circles.
streets     which streets I have seen, and where they led. A rider builds a
            sketch of the neighbourhood as they ride it.
dead_ends   turns that led nowhere. Worth remembering precisely because they
            look identical to good turns from the junction.
notebook    free-text lines the model wrote itself, shown back verbatim. This is
            the one store the model controls, and it is deliberately unstructured
            so the harness never has to interpret it.

Nothing here consults the graph. Everything recorded is something the agent
observed through a tool, which is what keeps memory honest: replaying the
trajectory reproduces the memory exactly.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

# How many notebook lines to show back. Long enough to hold a plan, short enough
# that the model cannot bury the current observation under its own history.
MAX_NOTES_SHOWN = 6
# A node visited more than this is a loop worth flagging in the prompt.
REVISIT_WARN = 2


@dataclass
class Visit:
    """One arrival, as the agent experienced it."""

    step: int
    node_id: str
    street: str
    address_hint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"step": self.step, "node": self.node_id, "street": self.street,
                "address": self.address_hint}


@dataclass
class CourierMemory:
    """The rider's working memory for one shift."""

    trail: list[Visit] = field(default_factory=list)
    visit_counts: Counter = field(default_factory=Counter)
    streets_seen: dict[str, str] = field(default_factory=dict)
    dead_ends: set[str] = field(default_factory=set)
    # junction -> the (street, heading) pairs already walked out of it
    taken: dict[str, set] = field(default_factory=dict)
    # junction -> {(street, heading): why it was refused}
    turned_down: dict[str, dict] = field(default_factory=dict)
    notebook: list[str] = field(default_factory=list)
    goal: str = ""
    goal_kind: str = ""

    # ── recording ────────────────────────────────────────────────────────────

    def arrive(self, *, step: int, node_id: str, street: str, address_hint: str = "") -> None:
        """Record an arrival -- only when the agent actually moved.

        Called on every turn, including looking and consulting, the visit count
        climbed while the courier stood still and the "going in circles" warning
        fired on an agent that had gone nowhere. That contradicted the LOST
        runbook, which tells it to change direction on exactly that signal.
        """
        if self.trail and self.trail[-1].node_id == node_id:
            return
        self.trail.append(Visit(step=step, node_id=node_id, street=street,
                                address_hint=address_hint))
        self.visit_counts[node_id] += 1

    def saw_street(self, name: str, description: str) -> None:
        # First impression wins: a later glance from a different angle should not
        # silently overwrite what the agent already committed to memory.
        self.streets_seen.setdefault(name, description)

    def mark_dead_end(self, node_id: str) -> None:
        self.dead_ends.add(node_id)

    def took(self, node_id: str, street: str, heading: str) -> None:
        """Record that this exact way out of this exact junction was walked.

        Keyed on the junction as well as the street, because "I have walked
        Rue Monge" is not a useful thought -- a street is walked in both
        directions and from several corners -- while "I have already left this
        corner by Rue Monge going east" is exactly the fact that stops a loop.
        """
        self.taken.setdefault(node_id, set()).add((street, heading))

    def has_taken(self, node_id: str, street: str, heading: str) -> bool:
        return (street, heading) in self.taken.get(node_id, set())

    def refused(self, node_id: str, street: str, heading: str, why: str) -> None:
        """Record a call this junction turned down, so it is never offered blind.

        Nothing about the world changes between a refusal and the next turn, so
        the same call is refused for the same reason -- and it was made again
        immediately in 63 of 128 attempts. The courier is told that in the
        prompt and does not act on it. Being told is not the same as being
        shown: the fact belongs on the line the courier is choosing from.
        """
        self.turned_down.setdefault(node_id, {})[(street, heading)] = why

    def refusal_at(self, node_id: str, street: str, heading: str) -> str:
        here = self.turned_down.get(node_id, {})
        return here.get((street, heading)) or here.get((street, "")) or ""

    def write(self, text: str) -> None:
        line = " ".join(str(text).split())[:160]
        if line and line not in self.notebook:
            self.notebook.append(line)

    def set_goal(self, kind: str, address: str) -> None:
        """Change the job. Deliberately silent in the notebook.

        This used to write a line every time the goal moved. The notebook shows
        the last six lines and a shift changes goal twenty times, so by the
        middle of an episode the notebook was six copies of a fact the ``Job:``
        line above it already states, and every note the model had actually
        written for itself had been evicted. The one store the model controls is
        not somewhere the harness gets to talk.
        """
        self.goal_kind, self.goal = kind, address

    # ── reporting back ───────────────────────────────────────────────────────

    def place_label(self, node_id: str) -> str:
        """A node, said the way the observation says it.

        Falls back to the id only when the agent has genuinely never been there
        and so has no words for it.
        """
        for visit in reversed(self.trail):
            if visit.node_id == node_id:
                if visit.street and visit.address_hint:
                    return f"{visit.street} no. {visit.address_hint}"
                return visit.street or visit.node_id
        return node_id

    @property
    def current_node(self) -> str | None:
        return self.trail[-1].node_id if self.trail else None

    @property
    def previous_node(self) -> str | None:
        return self.trail[-2].node_id if len(self.trail) > 1 else None

    def is_looping(self) -> bool:
        return any(count > REVISIT_WARN for count in self.visit_counts.values())

    def recent_trail(self, limit: int = 5) -> list[str]:
        """Where I have just been, as a rider would say it.

        The label used to prefer ``address_hint`` on its own, and that field
        holds a house-number range, so the trail rendered as
        ``Came from: 9 <- 1 <- 2`` -- three bare integers with no street
        attached, which name nothing the agent can find again and read as a list
        of choices rather than of places. A place is a street *and* a number.
        """
        seen: list[str] = []
        for visit in reversed(self.trail[:-1]):
            if visit.street and visit.address_hint:
                label = f"{visit.street} no. {visit.address_hint}"
            else:
                label = visit.street or visit.address_hint or visit.node_id
            if label not in seen:
                seen.append(label)
            if len(seen) >= limit:
                break
        return seen

    def render(self) -> str:
        """The memory block shown to the model each turn.

        It states facts and never a recommendation. The loop warning is the one
        judgement it makes, and it is a judgement about the *past* -- you have
        been here three times -- not about what to do next.
        """
        lines: list[str] = ["### your notes"]
        if self.goal:
            # "collect from at 5 Rue Saint-Antoine" -- the kind already carries
            # its own preposition, so the template must not add a second one.
            lines.append(f"Job: {self.goal_kind} {self.goal}".replace("  ", " "))
        trail = self.recent_trail()
        if trail:
            lines.append("Came from: " + " <- ".join(trail))
        if self.is_looping():
            # Named as places, not as node ids. "You have passed through
            # s006_n016 more than twice" was the only string in the observation
            # containing a raw graph id: it appears nowhere else the agent can
            # see, so it names somewhere the agent cannot recognise and cannot
            # act on, which makes the one judgement the memory block offers
            # unusable.
            repeated: list[str] = []
            for node, count in self.visit_counts.most_common():
                label = self.place_label(node)
                # Two nodes on one street share a label, and naming the street
                # three times reads as three places.
                if count > REVISIT_WARN and label not in repeated:
                    repeated.append(label)
                if len(repeated) >= 3:
                    break
            lines.append(
                f"You have passed {', '.join(repeated)} more than twice — "
                "you are going in circles."
            )
        if self.dead_ends:
            lines.append("Dead ends you found: "
                         + ", ".join(sorted(self.place_label(n) for n in self.dead_ends)))
        # ``Streets you have seen`` used to be rendered here: the last four
        # street names with where each was seen from. It is dropped, and the
        # reason is that it was measured rather than argued about. Across 95
        # turns of hand play it never once changed a decision, because a street
        # the courier has *seen* is either on the candidate list in front of it
        # -- where it appears with a number, a bearing and a photograph -- or it
        # is somewhere the courier cannot act on from here. It grew every turn
        # and pushed the two notes that do earn their place, ``Came from`` and
        # the circling warning, further from the top. The data is still in
        # ``to_dict`` for anyone analysing a trajectory.
        for note in self.notebook[-MAX_NOTES_SHOWN:]:
            lines.append(f"- {note}")
        if len(lines) == 1:
            lines.append("(nothing yet)")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal, "goal_kind": self.goal_kind,
            "trail": [v.to_dict() for v in self.trail],
            "visit_counts": dict(self.visit_counts),
            "streets_seen": dict(self.streets_seen),
            "dead_ends": sorted(self.dead_ends),
            "taken": {node: sorted(ways) for node, ways in self.taken.items()},
            "turned_down": {node: {f"{k[0]}|{k[1]}": v for k, v in ways.items()}
                            for node, ways in self.turned_down.items()},
            "notebook": list(self.notebook),
            "looping": self.is_looping(),
        }
