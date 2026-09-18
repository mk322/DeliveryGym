"""Contract and stress tests for ``EnvSpec``.

EnvSpec is the boundary between the compiler and every task layer, so the tests
that matter are the ones proving it cannot express an environment that would
break a task. The Paris failure is the model for all of them: a spec that said
"cardinal MOVE is available" on a non-cardinal map would send a task layer
straight into unreachable steps, so the schema refuses to represent it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from embodiedbench.compiler.env_spec_builder import build_env_spec, find_album
from embodiedbench.schemas.base import PayloadError, SchemaError, VersionError
from embodiedbench.schemas.env_spec import (
    AffordanceInventory,
    EnvSpec,
    GraphRepairSummary,
    GraphSummary,
    NavigationStyle,
    ObservationSupport,
    QualityFlag,
    SolvabilityEvidence,
)
from embodiedbench.schemas.environment import CertificationGrade, NavigationMode

REPO_ROOT = Path(__file__).resolve().parents[1]


def _graph(**overrides) -> GraphSummary:
    base = dict(
        node_count=136, edge_count=153, mean_degree=2.25, max_degree=4,
        dock_nodes=60, junction_nodes=76, cardinal_fraction=1.0,
        largest_component_fraction=1.0, component_count=1,
        median_edge_m=30.6, longest_edge_m=131.5, long_edge_threshold_m=184.0,
    )
    base.update(overrides)
    return GraphSummary(**base)


def _spec(**overrides) -> EnvSpec:
    base = dict(
        env_id="demo-env",
        map_name="demo",
        navigation_style=NavigationStyle.CARDINAL_AND_GRAPH,
        navigation_modes=[NavigationMode.NAV_WAYPOINT],
        enabled_actions=["VIEW_ORDERS", "MOVE_TO", "MOVE"],
        enable_waypoint_marks=True,
        navigation_rationale="100.0% of edges are near-cardinal",
        graph=_graph(),
        grade=CertificationGrade.B,
        quality_flags=[QualityFlag(code="degenerate_graph_edges", count=1, stage="graph")],
        solvability=SolvabilityEvidence(
            episodes=5, delivered_episodes=5, solvability_rate=1.0, mean_steps=19.4
        ),
    )
    base.update(overrides)
    return EnvSpec(**base)


# ─────────────────────────────────────────────────────────────────────────────
# The Paris invariant: the action space must match the geometry
# ─────────────────────────────────────────────────────────────────────────────


def test_graph_navigation_cannot_expose_cardinal_move():
    """The exact Paris failure, made unrepresentable."""
    with pytest.raises(ValueError, match="must not expose directional MOVE"):
        _spec(
            navigation_style=NavigationStyle.GRAPH,
            enabled_actions=["VIEW_ORDERS", "MOVE_TO", "MOVE"],
        )


def test_cardinal_style_must_actually_offer_move():
    with pytest.raises(ValueError, match="must expose MOVE"):
        _spec(
            navigation_style=NavigationStyle.CARDINAL_AND_GRAPH,
            enabled_actions=["VIEW_ORDERS", "MOVE_TO"],
        )


def test_move_to_is_always_required():
    """Graph-neighbour navigation is defined on every map; nothing else is."""
    with pytest.raises(ValueError, match="MOVE_TO is required"):
        _spec(enabled_actions=["VIEW_ORDERS", "MOVE"])


def test_move_to_requires_waypoint_marks():
    with pytest.raises(ValueError, match="requires enable_waypoint_marks"):
        _spec(enable_waypoint_marks=False)


def test_graph_style_spec_is_valid_without_move():
    spec = _spec(
        navigation_style=NavigationStyle.GRAPH,
        enabled_actions=["VIEW_ORDERS", "MOVE_TO", "NAVIGATE"],
        graph=_graph(cardinal_fraction=0.231),
        navigation_rationale="only 23.1% of edges are near-cardinal",
    )
    assert spec.navigation_style is NavigationStyle.GRAPH


# ─────────────────────────────────────────────────────────────────────────────
# Grades must be backed by evidence
# ─────────────────────────────────────────────────────────────────────────────


def test_grade_a_cannot_carry_flags():
    with pytest.raises(ValueError, match="grade A cannot carry quality flags"):
        _spec(grade=CertificationGrade.A)


def test_grade_a_is_valid_with_no_flags():
    assert _spec(grade=CertificationGrade.A, quality_flags=[]).grade is CertificationGrade.A


def test_graded_environment_needs_solvability_evidence():
    with pytest.raises(ValueError, match="must carry solvability evidence"):
        _spec(solvability=None)


def test_graded_environment_must_be_playable():
    with pytest.raises(ValueError, match="demonstrably playable"):
        _spec(
            solvability=SolvabilityEvidence(
                episodes=5, delivered_episodes=0, solvability_rate=0.0
            )
        )


def test_solvability_rate_must_match_its_counts():
    with pytest.raises(ValueError, match="does not match"):
        SolvabilityEvidence(episodes=5, delivered_episodes=5, solvability_rate=0.5)


def test_cannot_deliver_more_episodes_than_were_run():
    with pytest.raises(ValueError, match="more delivered episodes"):
        SolvabilityEvidence(episodes=2, delivered_episodes=3, solvability_rate=1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Usable / unusable coherence
# ─────────────────────────────────────────────────────────────────────────────


def test_unusable_environment_cannot_carry_a_passing_grade():
    with pytest.raises(ValueError, match="cannot carry a passing grade"):
        _spec(usable=False, failure_code="no_pois", grade=CertificationGrade.B)


def test_unusable_environment_must_state_why():
    with pytest.raises(ValueError, match="must state why"):
        _spec(usable=False, grade=CertificationGrade.FAIL, solvability=None)


def test_usable_environment_must_not_carry_a_failure_code():
    with pytest.raises(ValueError, match="must not carry a failure code"):
        _spec(failure_code="no_pois")


# ─────────────────────────────────────────────────────────────────────────────
# Vision support is measured, not configured
# ─────────────────────────────────────────────────────────────────────────────


def test_rgb_channel_requires_an_actual_album():
    """Paris looked vision-capable by config and had no album at all."""
    with pytest.raises(ValueError, match="not an album"):
        ObservationSupport(channels=["text", "rgb"], has_cached_album=True, album_waypoints=0)
    with pytest.raises(ValueError, match="without a cached album"):
        ObservationSupport(channels=["text", "rgb"], has_cached_album=False)


def test_text_channel_is_mandatory():
    with pytest.raises(ValueError, match="supports text observations"):
        ObservationSupport(channels=["rgb"], has_cached_album=True, album_waypoints=10)


def test_supports_vision_reflects_the_album():
    assert not _spec().supports_vision()
    assert _spec(
        observation=ObservationSupport(
            channels=["text", "rgb"], has_cached_album=True, album_waypoints=136
        )
    ).supports_vision()


# ─────────────────────────────────────────────────────────────────────────────
# Repair summary cannot claim the impossible
# ─────────────────────────────────────────────────────────────────────────────


def test_noding_cannot_lengthen_the_longest_edge():
    with pytest.raises(ValueError, match="cannot lengthen"):
        GraphRepairSummary(
            applied=True, longest_edge_before_m=100.0, longest_edge_after_m=200.0
        )


def test_noding_shortening_is_accepted():
    summary = GraphRepairSummary(
        applied=True, longest_edge_before_m=640.0, longest_edge_after_m=313.0
    )
    assert summary.longest_edge_after_m < summary.longest_edge_before_m


# ─────────────────────────────────────────────────────────────────────────────
# Affordances: the one-env-many-tasks contract
# ─────────────────────────────────────────────────────────────────────────────


def test_affordance_requirements_are_checkable():
    inventory = AffordanceInventory(counts={"restaurant": 18, "store": 11, "building": 414})
    assert inventory.satisfies({"restaurant": 10})
    assert not inventory.satisfies({"customer": 1})
    assert inventory.deficit({"restaurant": 20, "customer": 5}) == {"restaurant": 2, "customer": 5}
    assert inventory.total() == 443


def test_negative_affordance_counts_are_rejected():
    with pytest.raises(ValueError, match="negative count"):
        AffordanceInventory(counts={"restaurant": -1})


def test_supports_is_false_for_an_unusable_environment():
    spec = _spec(
        usable=False,
        failure_code="no_pois",
        grade=CertificationGrade.FAIL,
        quality_flags=[],
        solvability=None,
        affordances=AffordanceInventory(counts={"restaurant": 5}),
    )
    assert not spec.supports({"restaurant": 1})


# ─────────────────────────────────────────────────────────────────────────────
# Envelope behaviour, matching every other contract
# ─────────────────────────────────────────────────────────────────────────────


def test_round_trip_and_hash_stability():
    spec = _spec()
    assert spec.round_trip().to_dict() == spec.to_dict()
    assert spec.round_trip().content_hash() == spec.content_hash()


def test_unknown_field_is_rejected():
    payload = _spec().to_dict()
    payload["from_the_future"] = 1
    with pytest.raises(PayloadError):
        EnvSpec.from_dict(payload)


def test_wrong_version_is_rejected():
    payload = _spec().to_dict()
    payload["schema_version"] = "9.0.0"
    with pytest.raises(VersionError):
        EnvSpec.from_dict(payload)


# ─────────────────────────────────────────────────────────────────────────────
# Against real pipeline output
# ─────────────────────────────────────────────────────────────────────────────

SPEC_DIR = REPO_ROOT / "artifacts" / "verification" / "MAPS"
REAL_SPECS = sorted(SPEC_DIR.glob("*.envspec.json")) if SPEC_DIR.exists() else []


@pytest.mark.skipif(not REAL_SPECS, reason="no compiled EnvSpecs on disk")
@pytest.mark.parametrize("path", REAL_SPECS, ids=lambda p: p.name.split(".")[0])
def test_real_env_specs_revalidate(path):
    """Every spec the pipeline wrote must load back through the schema."""
    payload = json.loads(path.read_text())
    spec = EnvSpec.from_dict(payload)
    assert spec.to_dict() == payload


@pytest.mark.skipif(not REAL_SPECS, reason="no compiled EnvSpecs on disk")
def test_paris_spec_reports_its_known_gaps():
    """Paris must advertise the gaps we know it has, not hide them."""
    paris = [p for p in REAL_SPECS if "citycore-paris" in p.name]
    if not paris:
        pytest.skip("Paris not compiled")
    spec = EnvSpec.from_dict(json.loads(paris[0].read_text()))
    # Non-cardinal, so directional MOVE must be absent.
    assert spec.navigation_style is NavigationStyle.GRAPH
    assert "MOVE" not in spec.enabled_actions
    # No FPV album exists for Paris yet.
    assert not spec.supports_vision()
    # design plan §3.3.7's missing Delivery affordances.
    assert spec.affordances.counts.get("customer", 0) == 0
    assert spec.grade is not CertificationGrade.A


def test_album_discovery_prefers_the_largest_manifest():
    """small-city-11 keeps a 12-row stub at the album root and the real one below."""
    album = find_album("small-city-11")
    if not album.get("found"):
        pytest.skip("procgen album not present")
    assert album["waypoints"] > 100, album
    assert album["headings"] == 4


def test_album_discovery_reports_absence_for_paris():
    album = find_album("citycore-paris")
    assert not album.get("found")
    assert "reason" in album


def test_builder_produces_a_valid_spec_for_an_unusable_map():
    """A 'no' must arrive through the same contract as a 'yes'."""
    verdict = {
        "map": "broken",
        "unusable": True,
        "grade": "fail",
        "analysis": {},
        "quality_findings": [
            {"code": "no_pois", "count": 1, "detail": "no POIs", "stage": "load"}
        ],
        "validation": {"passes": False, "episodes": []},
        "failure": {"code": "no_pois", "explanation": "the map declares no POI nodes"},
        "thresholds": {},
    }
    spec = build_env_spec(verdict)
    assert not spec.usable
    assert spec.grade is CertificationGrade.FAIL
    assert spec.failure_code == "no_pois"
    assert not spec.supports({"restaurant": 1})
    assert spec.round_trip().to_dict() == spec.to_dict()
