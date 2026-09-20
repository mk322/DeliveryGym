"""The task plugin protocol (design plan §8.1).

A task may read a ``WorldBundle`` and call a ``Runtime``, but must never import
UE or cached-runtime implementation code (design plan §4.1). Nothing in this module
imports either, and the protocol is stated in terms of schema objects only, so
a task written against it cannot reach into a simulator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from embodiedbench.schemas.episode import EpisodeSpec
from embodiedbench.schemas.runtime import ActionEnvelope, Event
from embodiedbench.schemas.trajectory import ScoreReport, Trajectory
from embodiedbench.schemas.world import WorldBundle


@dataclass
class TaskRequirements:
    """What a task needs a world to provide before it can run (design plan §6.1 P4)."""

    affordances: dict[str, int] = field(default_factory=dict)
    routes: dict[str, int] = field(default_factory=dict)

    def deficit(self, available: dict[str, int]) -> dict[str, int]:
        """Missing counts, so a compiler can report before modifying anything."""
        return {
            name: needed - available.get(name, 0)
            for name, needed in self.affordances.items()
            if needed - available.get(name, 0) > 0
        }


@dataclass
class SolvabilityVerdict:
    """design plan §8.3: episode generation is followed by an oracle feasibility check."""

    feasible: bool
    reasons: list[str] = field(default_factory=list)

    def require(self) -> None:
        if not self.feasible:
            raise ValueError("instance rejected as unsolvable: " + "; ".join(self.reasons))


@runtime_checkable
class TaskPlugin(Protocol):
    """design plan §8.1's plugin surface."""

    id: str
    version: str

    def requirements(self, config: dict[str, Any]) -> TaskRequirements: ...

    def generate(self, world: WorldBundle, seed: int, config: dict[str, Any]) -> EpisodeSpec: ...

    def action_schema(self, state: Any) -> list[str]: ...

    def evaluate(self, trajectory: Trajectory, privileged: dict[str, Any]) -> ScoreReport: ...
