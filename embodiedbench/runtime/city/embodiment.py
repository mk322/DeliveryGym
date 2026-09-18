"""What is doing the delivering, and what that costs it.

The courier used to be a disembodied speed constant. Every episode moved at
1.4 m/s, tired never, and was shown the world from the middle of the
carriageway -- which is a viewpoint a person on foot never occupies, and which
made the whole album wrong for the only embodiment the benchmark had.

An embodiment answers four questions, and each one has to show up in the
environment or it is decoration:

``speed_cm_s``       how fast it covers ground, so a faster body spends less of
                     the clock on the same route
``stamina``          how much work it can do before it tires, and how fast that
                     drains with distance. A body with a bigger tank and a
                     slower drain can run a longer shift without slowing down
``viewpoint``        where its eyes are. A person on foot is on the pavement; a
                     scooter and a car are in the carriageway. The album a
                     policy is shown must match, or it is being trained to
                     recognise a world it will never stand in
``stop_overhead_s``  what it costs to stop and deal with the vehicle at each
                     door -- a rider parks and locks, a driver finds a space.
                     Without it a car strictly dominates and the choice of
                     vehicle is not a choice

Only the human is defined. The robot dog and the humanoid are declared with
their fields left at ``None`` on purpose: they are a stated intention, not a
tier anyone can run, and a config full of invented numbers would let them be
scored as though someone had measured them.
"""

from __future__ import annotations

from dataclasses import dataclass


class Viewpoint:
    """Where the camera stands. The album has to have been baked from here."""

    # On the footway, offset from the centreline toward the kerb. What a person
    # on foot sees: parked cars, café tables, bollards, doorways at reading
    # distance.
    PAVEMENT = "pavement"
    # On the carriageway centreline. What a rider or a driver sees. This is the
    # viewpoint every album in the repository was baked from, including the one
    # named ``paris_signals_kerb``, whose camera is 0.00 m from the centreline.
    CARRIAGEWAY = "carriageway"
    ALL = (PAVEMENT, CARRIAGEWAY)


@dataclass(frozen=True)
class Embodiment:
    """One body, and the four things about it the environment can feel."""

    name: str
    speed_cm_s: float | None = None
    # The tank, in the same units the drain is quoted in. ``None`` means this
    # body has not been characterised -- see the TODO entries below.
    stamina: float | None = None
    # Drain per metre travelled. Distance, not time: a courier is tired by the
    # ground it covers, and charging by time would make the slow body the tired
    # one, which is backwards.
    stamina_per_m: float | None = None
    # What a tired body loses. It does not stop -- a courier who has run out
    # keeps going, slower -- because a hard stop turns one bad estimate into an
    # unfinishable episode.
    tired_speed_fraction: float = 0.6
    viewpoint: str = Viewpoint.PAVEMENT
    stop_overhead_s: float = 0.0
    # Whether this body may use a pedestrian crossing. A car may not, and that
    # is what the signal album is about for it.
    uses_pedestrian_crossings: bool = True
    note: str = ""

    @property
    def defined(self) -> bool:
        """Has anyone actually measured this body?"""
        return None not in (self.speed_cm_s, self.stamina, self.stamina_per_m)

    def require_defined(self) -> None:
        if not self.defined:
            raise ValueError(
                f"embodiment {self.name!r} is declared but not characterised "
                f"({self.note or 'no measurements yet'}). Running it would score "
                "a body nobody has measured."
            )

    def range_m(self) -> float | None:
        """How far it can go before it tires. The number that makes the tank mean something."""
        if self.stamina is None or not self.stamina_per_m:
            return None
        return self.stamina / self.stamina_per_m


# ── the human, and the three ways it can travel ─────────────────────────────
#
# The speeds are ordinary urban ones rather than vehicle top speeds: a courier
# in traffic is not doing 50 km/h, and quoting a top speed would make the car
# beat the clock by a margin no city delivers.

ON_FOOT = Embodiment(
    name="human_on_foot",
    speed_cm_s=140.0,                 # 1.4 m/s, a brisk walk -- unchanged
    stamina=100.0,
    stamina_per_m=0.02,               # 5 km before it tires, about a long shift
    viewpoint=Viewpoint.PAVEMENT,
    stop_overhead_s=0.0,              # nothing to park
    uses_pedestrian_crossings=True,
)

ON_SCOOTER = Embodiment(
    name="human_on_scooter",
    speed_cm_s=420.0,                 # 15 km/h, a shared-scheme scooter in traffic
    stamina=100.0,
    stamina_per_m=0.004,              # 25 km: standing, not walking
    viewpoint=Viewpoint.CARRIAGEWAY,
    stop_overhead_s=20.0,             # stop, kick the stand down, lock it
    uses_pedestrian_crossings=False,
)

IN_CAR = Embodiment(
    name="human_in_car",
    speed_cm_s=700.0,                 # 25 km/h door to door in a dense city
    stamina=100.0,
    stamina_per_m=0.0,                # the car does the work
    viewpoint=Viewpoint.CARRIAGEWAY,
    stop_overhead_s=75.0,             # finding somewhere to leave it is the cost
    uses_pedestrian_crossings=False,
)

# ── declared, deliberately not characterised ────────────────────────────────
#
# Left empty on purpose. Both need their own albums as well as their own
# numbers: a quadruped's eyes are about 50 cm off the ground and a humanoid's
# are near a person's, and neither album exists. Filling these in with plausible
# guesses is how an unmeasured tier ends up in a results table.

ROBOT_DOG = Embodiment(
    name="robot_dog",
    viewpoint=Viewpoint.PAVEMENT,
    note="TODO: gait speed, battery as stamina, ~50 cm camera height, own album",
)

HUMANOID = Embodiment(
    name="humanoid_robot",
    viewpoint=Viewpoint.PAVEMENT,
    note="TODO: walking speed, battery as stamina, own album at its eye height",
)

EMBODIMENTS: dict[str, Embodiment] = {
    e.name: e for e in (ON_FOOT, ON_SCOOTER, IN_CAR, ROBOT_DOG, HUMANOID)
}
DEFAULT = ON_FOOT.name


def get(name: str | Embodiment | None) -> Embodiment:
    if name is None:
        return EMBODIMENTS[DEFAULT]
    if isinstance(name, Embodiment):
        return name
    try:
        return EMBODIMENTS[name]
    except KeyError:
        raise ValueError(
            f"unknown embodiment {name!r}; expected one of {sorted(EMBODIMENTS)}"
        ) from None
