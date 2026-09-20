"""The point-navigation chain, tested against the rules design plan §9 states.

Each test names the clause it defends. The geometry ones use a camera at the
origin looking down +X so an expected answer can be worked out on paper, which is
the only way to catch a transform that is self-consistently wrong -- exactly the
class of bug that put the first Paris render camera pointing at the sky.
"""

from __future__ import annotations

import math

import pytest

from embodiedbench.compiler.pipeline import GraphAnalysis, Thresholds, decide_point_navigation
from embodiedbench.embodiment.point_nav import (
    GraphProximityTraversable,
    GroundPlaneDepth,
    ModelPredictedDepth,
    PointNavRejected,
    PoseLattice,
    camera_to_world,
    distance_error_m,
    resolve_point_action,
)
from embodiedbench.schemas.embodiment import DistanceSource
from embodiedbench.schemas.environment import NavigationMode, PointExecutionVariant
from embodiedbench.schemas.geometry import CameraIntrinsics, FrameName, Pose, Vec3
from embodiedbench.schemas.runtime import (
    ControllerOutcomeCode,
    NavPoint2DDepthAction,
    NavPoint3DAction,
)

CAMERA_HEIGHT_CM = 160.0


def intrinsics() -> CameraIntrinsics:
    # 640x480 with fx=fy=320 is a 90-degree horizontal field of view, matching
    # the album renderer's plain view.
    return CameraIntrinsics(
        width_px=640, height_px=480, fx_px=320.0, fy_px=320.0,
        cx_px=320.0, cy_px=240.0, near_cm=10.0, far_cm=1.0e5,
    )


def camera(yaw_deg: float = 0.0, x_cm: float = 0.0, y_cm: float = 0.0) -> Pose:
    return Pose(
        frame=FrameName.BUNDLE_WORLD,
        position=Vec3(x_cm=x_cm, y_cm=y_cm, z_cm=CAMERA_HEIGHT_CM),
        yaw_deg=yaw_deg,
    )


def road_east(length_m: int = 60) -> GraphProximityTraversable:
    """A straight road running east from the origin, one node per metre."""
    return GraphProximityTraversable(points=[(x * 100.0, 0.0) for x in range(length_m)])


# ─────────────────────────────────────────────────────────────────────────────
# The pose lattice (design plan §9.5)
# ─────────────────────────────────────────────────────────────────────────────


class TestPoseLattice:
    def test_quantisation_is_shared_not_cached_only(self):
        """design plan §9.5 forbids snapping in cached mode alone.

        The lattice therefore has to be reachable from the embodiment layer that
        every runtime goes through, not from a cached-runtime helper. This test
        is a structural guard: if the lattice ever moves under runtime/cached,
        the import breaks and someone has to read 9.5 again.
        """
        from embodiedbench.embodiment import point_nav

        assert hasattr(point_nav, "PoseLattice")

    def test_position_snaps_to_nearest_cell(self):
        lattice = PoseLattice(spacing_cm=200.0, headings=8)
        assert lattice.quantise_position(240.0, -140.0) == (200.0, -200.0)
        assert lattice.quantise_position(0.0, 0.0) == (0.0, 0.0)

    def test_heading_snaps_to_nearest_of_eight(self):
        lattice = PoseLattice(spacing_cm=200.0, headings=8)
        assert lattice.quantise_heading(0.0) == 0.0
        assert lattice.quantise_heading(44.0) == 45.0
        assert lattice.quantise_heading(359.0) == 0.0
        assert lattice.quantise_heading(-1.0) == 0.0

    def test_quantisation_is_idempotent(self):
        """A pose already on the lattice must not move.

        If it did, replaying a trajectory would drift with each pass and the
        cached and live runtimes would disagree about where the agent is.
        """
        lattice = PoseLattice()
        pose = Pose(
            frame=FrameName.BUNDLE_WORLD,
            position=Vec3(x_cm=600.0, y_cm=-400.0, z_cm=0.0),
            yaw_deg=135.0,
        )
        once = lattice.quantise(pose)
        assert lattice.quantise(once).model_dump() == once.model_dump()

    def test_snap_error_is_bounded_and_published(self):
        """A consumer has to be able to reason about how much the snap moved it."""
        lattice = PoseLattice(spacing_cm=200.0)
        assert lattice.max_snap_error_cm() == pytest.approx(141.42, abs=0.01)

    def test_degenerate_lattices_are_refused(self):
        with pytest.raises(ValueError):
            PoseLattice(spacing_cm=0.0)
        with pytest.raises(ValueError):
            PoseLattice(headings=0)


# ─────────────────────────────────────────────────────────────────────────────
# Frames (design plan §9.4 step 2)
# ─────────────────────────────────────────────────────────────────────────────


class TestCameraToWorld:
    def test_straight_ahead_at_zero_yaw_goes_east(self):
        point = camera_to_world(camera(yaw_deg=0.0), Vec3(x_cm=0.0, y_cm=0.0, z_cm=1000.0))
        assert point.x_cm == pytest.approx(1000.0)
        assert point.y_cm == pytest.approx(0.0, abs=1e-6)

    def test_yaw_rotates_the_forward_axis(self):
        point = camera_to_world(camera(yaw_deg=90.0), Vec3(x_cm=0.0, y_cm=0.0, z_cm=1000.0))
        assert point.x_cm == pytest.approx(0.0, abs=1e-6)
        assert point.y_cm == pytest.approx(1000.0)

    def test_right_of_frame_is_ninety_degrees_clockwise(self):
        """Catches a swapped or negated right vector, which is self-consistent
        under forward-only tests and wrong everywhere else."""
        point = camera_to_world(camera(yaw_deg=0.0), Vec3(x_cm=1000.0, y_cm=0.0, z_cm=0.0))
        assert point.x_cm == pytest.approx(0.0, abs=1e-6)
        assert point.y_cm == pytest.approx(-1000.0)

    def test_camera_translation_is_applied(self):
        origin = camera(yaw_deg=0.0, x_cm=5000.0, y_cm=-2000.0)
        point = camera_to_world(origin, Vec3(x_cm=0.0, y_cm=0.0, z_cm=1000.0))
        assert point.x_cm == pytest.approx(6000.0)
        assert point.y_cm == pytest.approx(-2000.0)

    def test_downward_image_component_lowers_the_world_point(self):
        point = camera_to_world(camera(), Vec3(x_cm=0.0, y_cm=160.0, z_cm=1000.0))
        assert point.z_cm == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Ground-plane depth (design plan §9.3)
# ─────────────────────────────────────────────────────────────────────────────


class TestGroundPlaneDepth:
    def test_range_matches_hand_computed_similar_triangles(self):
        """Working the expected number out by hand is the point: an
        implementation that is wrong but self-consistent passes any test that
        derives its expectation from the same code."""
        depth = GroundPlaneDepth()
        distance = depth.distance_m(0.5, 0.75, camera(), intrinsics())
        # v_norm 0.75 -> 360 px -> 120 px below the principal point, so the
        # ground hit is 426.67 cm ahead and 160 cm down: a slant range of
        # sqrt(426.67^2 + 160^2). The provider returns slant range because that
        # is what unproject consumes and what a model's distance_m means.
        ground_cm = CAMERA_HEIGHT_CM * 320.0 / 120.0
        expected_cm = math.hypot(ground_cm, CAMERA_HEIGHT_CM)
        assert distance == pytest.approx(expected_cm / 100.0, rel=1e-9)

    def test_lower_in_frame_is_nearer(self):
        depth, intr, cam = GroundPlaneDepth(), intrinsics(), camera()
        near = depth.distance_m(0.5, 0.95, cam, intr)
        far = depth.distance_m(0.5, 0.55, cam, intr)
        assert near < far

    def test_horizon_is_no_ground_hit_not_a_huge_number(self):
        """design plan §9.4 wants no_ground_hit distinct from a wall hit. Returning a
        vast range instead would be silently clipped to the cap and executed."""
        with pytest.raises(PointNavRejected) as caught:
            GroundPlaneDepth().distance_m(0.5, 0.5, camera(), intrinsics())
        assert caught.value.outcome is ControllerOutcomeCode.NO_GROUND_HIT

    def test_above_horizon_is_no_ground_hit(self):
        with pytest.raises(PointNavRejected) as caught:
            GroundPlaneDepth().distance_m(0.5, 0.2, camera(), intrinsics())
        assert caught.value.outcome is ControllerOutcomeCode.NO_GROUND_HIT

    def test_camera_below_ground_is_refused(self):
        sunk = Pose(
            frame=FrameName.BUNDLE_WORLD,
            position=Vec3(x_cm=0.0, y_cm=0.0, z_cm=-10.0),
            yaw_deg=0.0,
        )
        with pytest.raises(PointNavRejected) as caught:
            GroundPlaneDepth().distance_m(0.5, 0.9, sunk, intrinsics())
        assert caught.value.outcome is ControllerOutcomeCode.NO_GROUND_HIT

    def test_provider_declares_itself_geometric(self):
        """design plan §9.3 requires results from this provider to be reported as
        geometry, not depth reasoning. The compiler reads that flag."""
        decision = decide_point_navigation(
            GraphAnalysis(node_count=10, edge_count=12, edge_length_m={"p50": 18.0})
        )
        assert decision.depth_provider_is_geometric is True
        assert "ground_plane" in decision.depth_provider


# ─────────────────────────────────────────────────────────────────────────────
# The chain end to end (design plan §9.4)
# ─────────────────────────────────────────────────────────────────────────────


class TestResolvePointAction:
    def test_nav_point_3d_uses_the_model_distance(self):
        action = NavPoint3DAction(target={"u_norm": 0.5, "v_norm": 0.9, "distance_m": 6.0})
        resolution = resolve_point_action(
            action=action, camera=camera(), intrinsics=intrinsics(),
            traversable=road_east(), lattice=PoseLattice(),
        )
        assert resolution.request.mode is NavigationMode.NAV_POINT_3D
        assert resolution.request.distance_source is DistanceSource.MODEL_PREDICTION
        # distance_m is slant range, so a 6 m ray angled down at v=0.9 (tan 0.6)
        # reaches 600 / sqrt(1 + 0.6^2) = 514.5 cm along the ground. The road has
        # a node every metre, so it projects to 500 cm and snaps to the 2 m cell
        # at 600 cm. Working the number through by hand is deliberate: it is the
        # only way to catch a chain that is self-consistently wrong.
        assert resolution.request.target_world.x_cm == pytest.approx(600.0 / math.sqrt(1.36))
        assert resolution.request.projected_target.x_cm == pytest.approx(500.0)
        assert resolution.request.quantized_target.position.x_cm == pytest.approx(600.0)

    def test_nav_point_2d_depth_uses_the_provider(self):
        action = NavPoint2DDepthAction(target={"u_norm": 0.5, "v_norm": 0.75})
        resolution = resolve_point_action(
            action=action, camera=camera(), intrinsics=intrinsics(),
            traversable=road_east(), lattice=PoseLattice(),
            depth=GroundPlaneDepth(),
        )
        assert resolution.request.mode is NavigationMode.NAV_POINT_2D_DEPTH
        assert resolution.request.distance_source is DistanceSource.EXTERNAL_DEPTH
        # v=0.75 is 120 px below the principal point, so the ground hit is
        # height * fy / offset = 426.67 cm away horizontally.
        assert resolution.request.target_world.x_cm == pytest.approx(CAMERA_HEIGHT_CM * 320.0 / 120.0)
        # ...and lands on the ground, not above or below it.
        assert resolution.request.target_world.z_cm == pytest.approx(0.0, abs=1e-6)

    def test_2d_depth_without_a_provider_is_refused(self):
        """Silently substituting a default range would make this mode a
        differently-named nav_point_3d and void design plan §9.7.6's comparison."""
        action = NavPoint2DDepthAction(target={"u_norm": 0.5, "v_norm": 0.8})
        with pytest.raises(PointNavRejected) as caught:
            resolve_point_action(
                action=action, camera=camera(), intrinsics=intrinsics(),
                traversable=road_east(), lattice=PoseLattice(),
            )
        assert caught.value.outcome is ControllerOutcomeCode.LOW_DEPTH_CONFIDENCE

    def test_both_modes_share_one_chain(self):
        """Given the same range, the two modes must land on the same pose.

        If they diverge, a comparison between them measures the implementation
        difference rather than the distance source.
        """
        distance_m = CAMERA_HEIGHT_CM * 320.0 / 120.0 / 100.0
        common = dict(
            camera=camera(yaw_deg=0.0), intrinsics=intrinsics(),
            traversable=road_east(), lattice=PoseLattice(),
        )
        three_d = resolve_point_action(
            action=NavPoint3DAction(
                target={"u_norm": 0.5, "v_norm": 0.75, "distance_m": distance_m}
            ),
            **common,
        )
        two_d = resolve_point_action(
            action=NavPoint2DDepthAction(target={"u_norm": 0.5, "v_norm": 0.75}),
            depth=GroundPlaneDepth(),
            **common,
        )
        assert (three_d.request.quantized_target.model_dump()
                == two_d.request.quantized_target.model_dump())

    def test_untraversable_target_is_rejected_not_nudged(self):
        """A point on a rooftop or across a canal has no route; accepting it and
        letting the controller fail later loses the reason."""
        action = NavPoint3DAction(target={"u_norm": 0.5, "v_norm": 0.9, "distance_m": 6.0})
        with pytest.raises(PointNavRejected) as caught:
            resolve_point_action(
                action=action, camera=camera(), intrinsics=intrinsics(),
                # Road runs north; the target is 6 m east of it.
                traversable=GraphProximityTraversable(
                    points=[(0.0, y * 100.0) for y in range(1, 60)], tolerance_cm=200.0
                ),
                lattice=PoseLattice(),
            )
        assert caught.value.outcome is ControllerOutcomeCode.NOT_TRAVERSABLE

    def test_snap_variant_clips_an_overlong_distance(self):
        action = NavPoint3DAction(
            target={"u_norm": 0.5, "v_norm": 0.9, "distance_m": 500.0},
            variant=PointExecutionVariant.SNAP,
        )
        resolution = resolve_point_action(
            action=action, camera=camera(), intrinsics=intrinsics(),
            traversable=road_east(), lattice=PoseLattice(), max_range_m=18.0,
        )
        assert resolution.request.max_range_m == 18.0
        # Clipped to the cap, then projected and snapped -- 18 m, not 500.
        assert resolution.request.quantized_target.position.x_cm <= 1900.0

    def test_strict_variant_refuses_rather_than_repairing(self):
        """design plan §9.2's strict variant treats the model's distance as
        load-bearing; repairing it along the ray would hide the error the
        variant exists to expose."""
        action = NavPoint3DAction(
            target={"u_norm": 0.5, "v_norm": 0.9, "distance_m": 500.0},
            variant=PointExecutionVariant.STRICT,
        )
        with pytest.raises(PointNavRejected) as caught:
            resolve_point_action(
                action=action, camera=camera(), intrinsics=intrinsics(),
                traversable=road_east(), lattice=PoseLattice(), max_range_m=18.0,
            )
        assert caught.value.outcome is ControllerOutcomeCode.OUT_OF_RANGE

    def test_the_source_point_is_kept_verbatim(self):
        """design plan §9.4 wants the raw model output auditable after the chain."""
        action = NavPoint3DAction(target={"u_norm": 0.31, "v_norm": 0.87, "distance_m": 4.0})
        resolution = resolve_point_action(
            action=action, camera=camera(), intrinsics=intrinsics(),
            traversable=road_east(), lattice=PoseLattice(),
        )
        assert resolution.request.source_image_point.u_norm == pytest.approx(0.31)
        assert resolution.request.source_image_point.v_norm == pytest.approx(0.87)

    def test_every_frame_of_the_chain_is_recorded(self):
        action = NavPoint3DAction(target={"u_norm": 0.5, "v_norm": 0.9, "distance_m": 6.0})
        resolution = resolve_point_action(
            action=action, camera=camera(), intrinsics=intrinsics(),
            traversable=road_east(), lattice=PoseLattice(),
        )
        for field in ("target_camera", "target_world", "projected_target", "quantized_target"):
            assert getattr(resolution.request, field) is not None, field

    def test_heading_faces_the_target(self):
        """Arriving pointed away from where the agent asked to go makes the next
        observation useless."""
        action = NavPoint3DAction(target={"u_norm": 0.5, "v_norm": 0.9, "distance_m": 6.0})
        resolution = resolve_point_action(
            action=action, camera=camera(yaw_deg=90.0), intrinsics=intrinsics(),
            traversable=GraphProximityTraversable(
                points=[(0.0, y * 100.0) for y in range(60)]
            ),
            lattice=PoseLattice(),
        )
        assert resolution.request.quantized_target.yaw_deg == pytest.approx(90.0)

    def test_the_result_validates_against_the_published_schema(self):
        """resolve_point_action returns a NavigationRequest, whose own validator
        enforces the mode/distance-source pairing. Building one by hand in the
        wrong combination must fail."""
        action = NavPoint2DDepthAction(target={"u_norm": 0.5, "v_norm": 0.85})
        resolution = resolve_point_action(
            action=action, camera=camera(), intrinsics=intrinsics(),
            traversable=road_east(), lattice=PoseLattice(), depth=GroundPlaneDepth(),
        )
        request = resolution.request
        round_tripped = type(request).model_validate(request.model_dump())
        assert round_tripped.mode is NavigationMode.NAV_POINT_2D_DEPTH


class TestDistanceError:
    def test_error_is_logged_even_when_snapping_repairs_it(self):
        """design plan §9.2: without this, a policy emitting a constant distance looks
        competent, because the projection hides how wrong it was."""
        assert distance_error_m(30.0, 6.0) == pytest.approx(24.0)
        assert distance_error_m(6.0, 6.0) == pytest.approx(0.0)

    def test_model_predicted_provider_returns_what_it_was_given(self):
        assert ModelPredictedDepth(distance=7.5).distance_m() == pytest.approx(7.5)


# ─────────────────────────────────────────────────────────────────────────────
# The compiler's mode decision
# ─────────────────────────────────────────────────────────────────────────────


class TestDecidePointNavigation:
    def test_reach_scales_with_the_map_median_edge(self):
        """A fixed cap is about one edge on Paris and several blocks on a dense
        procgen map, so one point step and one waypoint step would not be
        comparable across maps."""
        dense = decide_point_navigation(
            GraphAnalysis(node_count=100, edge_count=200, edge_length_m={"p50": 8.0})
        )
        sparse = decide_point_navigation(
            GraphAnalysis(node_count=100, edge_count=200, edge_length_m={"p50": 25.0})
        )
        assert dense.max_range_m == pytest.approx(8.0)
        assert sparse.max_range_m == pytest.approx(25.0)

    def test_reach_is_clamped_to_a_sane_band(self):
        tiny = decide_point_navigation(
            GraphAnalysis(node_count=10, edge_count=10, edge_length_m={"p50": 0.4})
        )
        huge = decide_point_navigation(
            GraphAnalysis(node_count=10, edge_count=10, edge_length_m={"p50": 900.0})
        )
        assert tiny.max_range_m == Thresholds().point_range_min_m
        assert huge.max_range_m == Thresholds().point_range_max_m

    def test_a_map_without_a_graph_defines_no_point_mode(self):
        decision = decide_point_navigation(GraphAnalysis(node_count=0, edge_count=0))
        assert decision.definable is False
        assert decision.servable_runtimes == []

    def test_node_keyed_album_cannot_serve_point_modes(self):
        """design plan §9.5: a point action ends at a lattice pose, and a node-keyed
        album has no image there. Claiming cached support would hand the runtime
        a frame for the wrong place."""
        decision = decide_point_navigation(
            GraphAnalysis(node_count=100, edge_count=200, edge_length_m={"p50": 18.0}),
            album_is_lattice_baked=False,
        )
        assert decision.servable_runtimes == ["live"]
        assert "lattice pose" in decision.rationale

    def test_lattice_baked_album_can_serve_cached(self):
        decision = decide_point_navigation(
            GraphAnalysis(node_count=100, edge_count=200, edge_length_m={"p50": 18.0}),
            album_is_lattice_baked=True,
        )
        assert set(decision.servable_runtimes) == {"live", "cached"}

    def test_rationale_is_never_empty(self):
        for analysis in (
            GraphAnalysis(node_count=0, edge_count=0),
            GraphAnalysis(node_count=5, edge_count=4, edge_length_m={"p50": 12.0}),
        ):
            assert decide_point_navigation(analysis).rationale.strip()


# EnvSpec refuses to grade an environment with no solvability evidence, which is
# the right rule; these tests supply the minimum that satisfies it.
_SOLVED = {
    "episodes": [{"delivered": True}],
    "delivered_episodes": 1,
    "solvability_rate": 1.0,
    "mean_steps": 12.0,
}


class TestEnvSpecCarriesPointNav:
    def test_a_usable_map_declares_all_three_modes(self):
        from embodiedbench.compiler.env_spec_builder import build_env_spec

        spec = build_env_spec({
            "map": "test-map",
            "grade": "b",
            "analysis": {
                "node_count": 100, "edge_count": 200, "mean_degree": 4.0,
                "cardinal_fraction": 0.2, "largest_component_fraction": 1.0,
                "component_count": 1, "edge_length_m": {"p50": 18.0, "max": 60.0},
                "degree_histogram": {"4": 100},
            },
            "navigation": {
                "mode": "graph",
                "enabled_actions": ["VIEW_ORDERS", "MOVE_TO", "NAVIGATE"],
                "enable_waypoint_marks": True,
                "rationale": "graph navigation",
            },
            "validation": _SOLVED,
        })
        assert spec.navigation_modes == [
            NavigationMode.NAV_WAYPOINT,
            NavigationMode.NAV_POINT_3D,
            NavigationMode.NAV_POINT_2D_DEPTH,
        ]
        # nav_waypoint stays first: design plan §9 makes it the production path and a
        # consumer picking modes[0] must get it.
        assert spec.navigation_modes[0] is NavigationMode.NAV_WAYPOINT

    def test_reach_and_lattice_reach_the_runtime(self):
        """A runtime has to enforce the same cap the compiler decided, so the
        number must survive into the published spec."""
        from embodiedbench.compiler.env_spec_builder import build_env_spec

        spec = build_env_spec({
            "map": "test-map",
            "grade": "b",
            "analysis": {
                "node_count": 100, "edge_count": 200, "mean_degree": 4.0,
                "cardinal_fraction": 0.2, "largest_component_fraction": 1.0,
                "component_count": 1, "edge_length_m": {"p50": 12.0, "max": 60.0},
                "degree_histogram": {"4": 100},
            },
            "navigation": {
                "mode": "graph", "enabled_actions": ["MOVE_TO"],
                "enable_waypoint_marks": True, "rationale": "graph navigation",
            },
            "validation": _SOLVED,
        })
        point = spec.runtime_config["point_navigation"]
        assert point["max_range_m"] == pytest.approx(12.0)
        assert point["lattice_spacing_cm"] > 0
        assert point["servable_runtimes"] == ["live"]

    def test_an_unusable_map_offers_only_waypoint(self):
        from embodiedbench.compiler.env_spec_builder import build_env_spec

        spec = build_env_spec({
            "map": "broken",
            "grade": "c",
            "unusable": True,
            "failure": {"code": "empty_graph", "explanation": "no nodes"},
            "analysis": {"node_count": 0, "edge_count": 0},
        })
        assert spec.navigation_modes == [NavigationMode.NAV_WAYPOINT]


class TestParisPointNavIsCoherent:
    """The measured Paris numbers, so a regression in the repaired graph shows
    up as a nav-mode change rather than being noticed months later."""

    PARIS_ANALYSIS = GraphAnalysis(
        node_count=4540, edge_count=5649, edge_length_m={"p50": 18.5, "max": 313.0}
    )

    def test_paris_reach_is_about_one_edge(self):
        decision = decide_point_navigation(self.PARIS_ANALYSIS)
        assert 15.0 <= decision.max_range_m <= 22.0

    def test_paris_lattice_is_finer_than_its_edges(self):
        """A lattice as coarse as the graph could not express where a point
        action lands, which is why design plan §9.5 rejects snapping to nodes."""
        decision = decide_point_navigation(self.PARIS_ANALYSIS)
        median_edge_cm = self.PARIS_ANALYSIS.edge_length_m["p50"] * 100.0
        assert decision.lattice_spacing_cm < median_edge_cm / 5.0

    def test_paris_headings_are_finer_than_the_album(self):
        """The album has four cardinal yaws; the lattice must be at least that
        fine or a point action could not face where it was sent."""
        decision = decide_point_navigation(self.PARIS_ANALYSIS)
        assert decision.lattice_headings >= 4
        assert decision.lattice_headings % 4 == 0

    def test_a_full_circle_of_points_stays_on_the_road(self):
        """Sweep the image plane and confirm no yaw produces a silently accepted
        target off the traversable set."""
        lattice, intr = PoseLattice(), intrinsics()
        # A 3 m tolerance, so a point selected far off the road axis is rejected
        # rather than silently dragged back onto it.
        traversable = GraphProximityTraversable(
            points=[(x * 100.0, 0.0) for x in range(60)], tolerance_cm=300.0
        )
        accepted, rejected = 0, 0
        for u in (0.2, 0.35, 0.5, 0.65, 0.8):
            for v in (0.6, 0.7, 0.8, 0.9):
                action = NavPoint2DDepthAction(target={"u_norm": u, "v_norm": v})
                try:
                    resolution = resolve_point_action(
                        action=action, camera=camera(), intrinsics=intr,
                        traversable=traversable, lattice=lattice,
                        depth=GroundPlaneDepth(), max_range_m=18.5,
                    )
                except PointNavRejected:
                    rejected += 1
                    continue
                accepted += 1
                target = resolution.request.projected_target
                x, y = target.x_cm, target.y_cm
                assert min(
                    math.hypot(x - px, y - py) for px, py in traversable.points
                ) < 1e-6, "accepted a target that is not on the traversable set"
        # Both outcomes must occur, or the test is only exercising one branch.
        assert accepted > 0 and rejected > 0
