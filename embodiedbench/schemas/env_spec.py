"""``EnvSpec``: the standardized output of the map -> env pipeline.

One environment supports many tasks, so this describes what an environment *is*
and what it *can support*, and says nothing about any particular task. A task
layer reads it to decide whether it can run here and how; it never re-derives
these facts from the map, because two consumers deriving the same fact
independently is how they come to disagree.

It carries four things a task layer actually needs:

``navigation``   which actions exist on this map, and why that was decided
``affordances`` what POI types exist and in what quantity, so a task can check
                its own requirements before generating an episode
``observation``  which channels are really available (not which are configured)
``certificate``  the grade, the flags behind it, and the evidence

The "and why" matters. The navigation decision is a measurement plus a
threshold, both recorded, so a reader can disagree with the conclusion without
re-running the pipeline.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import Field, model_validator

from embodiedbench.schemas.base import SchemaModel
from embodiedbench.schemas.environment import CertificationGrade, NavigationMode
from embodiedbench.schemas.world import Sha256, StableId


class NavigationStyle(str, Enum):
    """How an agent is allowed to move on this map.

    Derived from geometry, never assumed. design plan §3.3.4 requires graph-neighbour
    navigation to be the default for Paris; the pipeline generalizes that to any
    map whose streets are not near-cardinal.
    """

    GRAPH = "graph"
    CARDINAL_AND_GRAPH = "cardinal+graph"


class GraphSummary(SchemaModel):
    """The measurements the navigation decision and the grade rest on."""

    node_count: int = Field(ge=0)
    edge_count: int = Field(ge=0)
    mean_degree: float = Field(ge=0.0)
    max_degree: int = Field(ge=0)
    dock_nodes: int = Field(default=0, ge=0)
    junction_nodes: int = Field(default=0, ge=0)
    cardinal_fraction: float = Field(ge=0.0, le=1.0)
    largest_component_fraction: float = Field(ge=0.0, le=1.0)
    component_count: int = Field(default=1, ge=0)
    median_edge_m: float = Field(default=0.0, ge=0.0)
    longest_edge_m: float = Field(default=0.0, ge=0.0)
    long_edge_threshold_m: float = Field(default=0.0, ge=0.0)


class GraphRepairSummary(SchemaModel):
    """What noding changed, so a consumer knows it is not looking at raw output."""

    applied: bool = False
    converged: bool = False
    passes_note: str = ""
    edges_before: int = Field(default=0, ge=0)
    edges_after: int = Field(default=0, ge=0)
    edges_split: int = Field(default=0, ge=0)
    skipped_nodes_recovered: int = Field(default=0, ge=0)
    mean_degree_before: float = Field(default=0.0, ge=0.0)
    mean_degree_after: float = Field(default=0.0, ge=0.0)
    longest_edge_before_m: float = Field(default=0.0, ge=0.0)
    longest_edge_after_m: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def _repair_never_invents_reachability(self) -> "GraphRepairSummary":
        # Noding splits edges; it must never reduce the node-to-node reach of the
        # graph, and it must never leave the graph longer-edged than it found it.
        if self.applied and self.longest_edge_after_m > self.longest_edge_before_m + 1e-6:
            raise ValueError(
                "noding cannot lengthen the longest edge: "
                f"{self.longest_edge_before_m} -> {self.longest_edge_after_m}"
            )
        return self


class AffordanceInventory(SchemaModel):
    """What a task can build on: POI types and how many of each exist.

    A task declares requirements against this rather than against a map name,
    which is what lets one environment support many tasks.
    """

    counts: dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _counts_are_sane(self) -> "AffordanceInventory":
        for name, count in self.counts.items():
            if count < 0:
                raise ValueError(f"affordance {name!r} has a negative count")
            if not name:
                raise ValueError("affordance names must be non-empty")
        return self

    def total(self) -> int:
        return sum(self.counts.values())

    def satisfies(self, requirements: dict[str, int]) -> bool:
        return all(self.counts.get(name, 0) >= need for name, need in requirements.items())

    def deficit(self, requirements: dict[str, int]) -> dict[str, int]:
        out = {}
        for name, need in requirements.items():
            short = need - self.counts.get(name, 0)
            if short > 0:
                out[name] = short
        return out


class ObservationSupport(SchemaModel):
    """Channels this environment can actually produce.

    ``text`` is always available. ``rgb`` requires a cached album bound to this
    map; a configuration flag alone is not support, which is exactly how Paris
    looked ready for vision while having no album at all.
    """

    channels: list[str] = Field(default_factory=lambda: ["text"])
    has_cached_album: bool = False
    album_waypoints: int = Field(default=0, ge=0)
    album_headings: int = Field(default=0, ge=0)
    album_coverage_fraction: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _rgb_requires_an_album(self) -> "ObservationSupport":
        if "text" not in self.channels:
            raise ValueError("every environment supports text observations")
        if "rgb" in self.channels and not self.has_cached_album:
            raise ValueError(
                "an environment cannot declare the rgb channel without a cached album; "
                "a config flag is not an album"
            )
        if self.has_cached_album and self.album_waypoints <= 0:
            raise ValueError("a cached album with no waypoints is not an album")
        return self


class QualityFlag(SchemaModel):
    """One rule-based concern, with the stage that found it."""

    code: str = Field(min_length=1)
    count: int = Field(ge=0)
    detail: str = ""
    stage: str = Field(default="graph", pattern="^(source|graph|load|analyze|validate)$")


class SolvabilityEvidence(SchemaModel):
    """Scripted-oracle proof that the environment is playable."""

    episodes: int = Field(ge=0)
    delivered_episodes: int = Field(ge=0)
    solvability_rate: float = Field(ge=0.0, le=1.0)
    mean_steps: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def _rate_matches_counts(self) -> "SolvabilityEvidence":
        if self.delivered_episodes > self.episodes:
            raise ValueError("more delivered episodes than episodes")
        if self.episodes:
            expected = self.delivered_episodes / self.episodes
            if abs(expected - self.solvability_rate) > 1e-6:
                raise ValueError(
                    f"solvability rate {self.solvability_rate} does not match "
                    f"{self.delivered_episodes}/{self.episodes}"
                )
        return self


class EnvSpec(SchemaModel):
    """A compiled, standardized environment description."""

    SCHEMA_ID = "embodiedbench/env_spec"
    SCHEMA_VERSION = "0.1.0"
    VERSIONED_ENVELOPE = True

    env_id: StableId
    map_name: StableId
    compiler_version: str = Field(default="0.1.0", pattern=r"^\d+\.\d+\.\d+$")

    # ── what an agent may do here ────────────────────────────────────────────
    navigation_style: NavigationStyle
    navigation_modes: list[NavigationMode] = Field(min_length=1)
    enabled_actions: list[str] = Field(min_length=1)
    enable_waypoint_marks: bool = True
    navigation_rationale: str = Field(min_length=1)

    # ── what exists here ─────────────────────────────────────────────────────
    graph: GraphSummary
    graph_repair: GraphRepairSummary = Field(default_factory=GraphRepairSummary)
    affordances: AffordanceInventory = Field(default_factory=AffordanceInventory)
    observation: ObservationSupport = Field(default_factory=ObservationSupport)

    # ── how good it is ───────────────────────────────────────────────────────
    grade: CertificationGrade
    quality_flags: list[QualityFlag] = Field(default_factory=list)
    solvability: SolvabilityEvidence | None = None
    usable: bool = True
    failure_code: str | None = None
    failure_explanation: str | None = None

    # ── provenance ───────────────────────────────────────────────────────────
    world_bundle_sha256: Sha256 | None = None
    thresholds: dict[str, Any] = Field(default_factory=dict)
    runtime_config: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _internally_consistent(self) -> "EnvSpec":
        # The action space must match the declared style, or a task layer reading
        # one and executing the other silently produces unreachable steps -- the
        # exact Paris failure this schema exists to prevent.
        has_cardinal = "MOVE" in self.enabled_actions
        if self.navigation_style is NavigationStyle.GRAPH and has_cardinal:
            raise ValueError(
                "graph navigation must not expose directional MOVE: on a non-cardinal map "
                "most shortest-path steps are unreachable by direction"
            )
        if self.navigation_style is NavigationStyle.CARDINAL_AND_GRAPH and not has_cardinal:
            raise ValueError("cardinal+graph navigation must expose MOVE")
        if "MOVE_TO" not in self.enabled_actions:
            raise ValueError(
                "MOVE_TO is required: graph-neighbour navigation is defined on every map "
                "and is the only action guaranteed to be executable"
            )
        if self.enable_waypoint_marks is False and "MOVE_TO" in self.enabled_actions:
            raise ValueError("MOVE_TO requires enable_waypoint_marks")

        if not self.usable:
            if self.grade is not CertificationGrade.FAIL:
                raise ValueError("an unusable environment cannot carry a passing grade")
            if not self.failure_code:
                raise ValueError("an unusable environment must state why")
        else:
            if self.failure_code:
                raise ValueError("a usable environment must not carry a failure code")

        # design plan §6.2: grade A means every check passed with nothing unreviewed.
        if self.grade is CertificationGrade.A and self.quality_flags:
            raise ValueError(
                "grade A cannot carry quality flags: "
                + ", ".join(sorted({f.code for f in self.quality_flags}))
            )
        if self.grade in (CertificationGrade.A, CertificationGrade.B):
            if self.solvability is None:
                raise ValueError("a graded environment must carry solvability evidence")
            if self.solvability.solvability_rate <= 0.0:
                raise ValueError("a graded environment must be demonstrably playable")
        return self

    # ── convenience for the task layer ───────────────────────────────────────

    def supports(self, requirements: dict[str, int]) -> bool:
        """Whether this environment can host a task with these requirements."""
        return self.usable and self.affordances.satisfies(requirements)

    def supports_vision(self) -> bool:
        return "rgb" in self.observation.channels and self.observation.has_cached_album

    def flag_codes(self) -> list[str]:
        return sorted({flag.code for flag in self.quality_flags})
