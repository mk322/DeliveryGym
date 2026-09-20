"""design plan §9.7.2's oracle, and proof that the oracle can fail.

An oracle nobody has seen reject anything is not evidence. The first version of
``run_oracle`` gave a deliberately sign-flipped world transform a clean pass:
the flip pushed 90% of targets off the road, they came back as
``not_traversable``, and only the handful the flip happened not to move were
scored. So the bulk of this file injects known bugs and requires the oracle to
notice each one.

The four injected bugs are chosen to cover the ways a coordinate chain actually
breaks in this codebase:

y-flip      a sign error, the class that put the Paris render camera at the sky
xy-swap     an axis transposition, which survives any test using symmetric data
2% scale    a units or calibration slip too small to be visible by eye
1 degree    a heading offset, which no single-direction test can detect at all
"""

from __future__ import annotations

import importlib
import random
from pathlib import Path

import pytest

import embodiedbench.embodiment.oracle as oracle_module
import embodiedbench.embodiment.point_nav as point_nav
from embodiedbench.embodiment.oracle import (
    ALBUM_INTRINSICS,
    OracleCase,
    _rejection_is_legitimate,
    run_oracle,
    world_to_camera,
)
from embodiedbench.schemas.geometry import FrameName, Pose, Vec3

MAX_RANGE_M = 18.0

# Captured at import, before any test swaps the symbol out, so an injected bug
# always wraps the genuine implementation rather than a previous injection.
_original = point_nav.camera_to_world


def jittered_grid(seed: int = 7, size: int = 12) -> list[tuple[str, float, float]]:
    """A grid deliberately off the 2 m lattice, so quantisation really moves things.

    An aligned grid would leave every quantised residual at exactly zero and the
    lattice bound check would never be exercised.
    """
    rng = random.Random(seed)
    return [
        (f"n_{i}_{j}", i * 370.0 + rng.uniform(-40, 40), j * 430.0 + rng.uniform(-40, 40))
        for i in range(size)
        for j in range(size)
    ]


def run_with_broken_transform(replacement, nodes) -> dict:
    """Swap in a broken ``camera_to_world`` and run the oracle against it."""
    original = point_nav.camera_to_world
    point_nav.camera_to_world = replacement
    try:
        # The oracle binds the symbol at import, so it has to be re-resolved.
        importlib.reload(oracle_module)
        return oracle_module.run_oracle(
            map_name="injected-bug", node_positions=nodes,
            max_range_m=MAX_RANGE_M, max_cameras=20,
        ).to_dict()
    finally:
        point_nav.camera_to_world = original
        importlib.reload(oracle_module)


class TestWorldToCamera:
    """The oracle's own transform, checked independently of the one it tests."""

    def test_a_point_straight_ahead_is_on_the_optical_axis(self):
        camera = Pose(
            frame=FrameName.BUNDLE_WORLD,
            position=Vec3(x_cm=0.0, y_cm=0.0, z_cm=160.0),
            yaw_deg=0.0,
        )
        point = world_to_camera(camera, Vec3(x_cm=1000.0, y_cm=0.0, z_cm=160.0))
        assert point.z_cm == pytest.approx(1000.0)
        assert point.x_cm == pytest.approx(0.0, abs=1e-9)
        assert point.y_cm == pytest.approx(0.0, abs=1e-9)

    def test_ground_level_target_is_below_the_axis(self):
        camera = Pose(
            frame=FrameName.BUNDLE_WORLD,
            position=Vec3(x_cm=0.0, y_cm=0.0, z_cm=160.0),
            yaw_deg=0.0,
        )
        point = world_to_camera(camera, Vec3(x_cm=1000.0, y_cm=0.0, z_cm=0.0))
        assert point.y_cm == pytest.approx(160.0)

    def test_it_inverts_camera_to_world_without_being_derived_from_it(self):
        """Both directions are written out by hand; this checks they agree.

        That agreement is meaningful precisely because neither was obtained by
        inverting the other.
        """
        camera = Pose(
            frame=FrameName.BUNDLE_WORLD,
            position=Vec3(x_cm=1234.0, y_cm=-567.0, z_cm=160.0),
            yaw_deg=37.0,
        )
        for world in (
            Vec3(x_cm=1600.0, y_cm=-200.0, z_cm=0.0),
            Vec3(x_cm=900.0, y_cm=-900.0, z_cm=50.0),
        ):
            camera_point = world_to_camera(camera, world)
            back = point_nav.camera_to_world(camera, camera_point)
            assert back.x_cm == pytest.approx(world.x_cm, abs=1e-6)
            assert back.y_cm == pytest.approx(world.y_cm, abs=1e-6)
            assert back.z_cm == pytest.approx(world.z_cm, abs=1e-6)


class TestProjectIsTheInverseOfUnproject:
    def test_round_trip_recovers_the_image_point(self):
        for u, v in ((0.5, 0.75), (0.12, 0.9), (0.88, 0.6)):
            point = ALBUM_INTRINSICS.unproject(u, v, 800.0)
            recovered = ALBUM_INTRINSICS.project(point)
            assert recovered is not None
            assert recovered[0] == pytest.approx(u, abs=1e-9)
            assert recovered[1] == pytest.approx(v, abs=1e-9)
            assert recovered[2] == pytest.approx(800.0, abs=1e-6)

    def test_a_point_behind_the_camera_has_no_image_coordinate(self):
        assert ALBUM_INTRINSICS.project(Vec3(x_cm=0.0, y_cm=0.0, z_cm=-100.0)) is None

    def test_a_point_outside_the_sensor_is_not_clamped(self):
        """Clamping would invent an image point the camera never saw, and the
        oracle would then 'validate' a target that is not in frame."""
        assert ALBUM_INTRINSICS.project(Vec3(x_cm=10000.0, y_cm=0.0, z_cm=100.0)) is None


class TestOraclePassesOnACorrectChain:
    def test_residual_is_zero_on_a_synthetic_grid(self):
        report = run_oracle(
            map_name="grid",
            node_positions=[(f"n_{i}_{j}", i * 400.0, j * 400.0)
                            for i in range(12) for j in range(12)],
            max_range_m=MAX_RANGE_M, max_cameras=20,
        )
        assert report.passed
        assert report.max_unprojected_residual_cm == pytest.approx(0.0, abs=1e-6)
        # A pass with nothing visible would be vacuous.
        assert report.visible > 100

    def test_ground_plane_depth_recovers_the_true_range_exactly(self):
        """Both point modes must agree on how far away a ground-level target is.

        If they do not, comparing the two modes measures the disagreement rather
        than the thing design plan §9.7.6 wants to compare.
        """
        report = run_oracle(
            map_name="grid", node_positions=jittered_grid(),
            max_range_m=MAX_RANGE_M, max_cameras=20,
        )
        assert report.max_depth_error_m == pytest.approx(0.0, abs=1e-9)

    def test_quantisation_stays_inside_its_published_bound(self):
        report = run_oracle(
            map_name="jittered", node_positions=jittered_grid(),
            max_range_m=MAX_RANGE_M, max_cameras=20,
        )
        assert report.max_quantised_residual_cm <= report.lattice_snap_bound_cm
        # ...and actually exercises the snap, rather than sitting at zero
        # because the grid happened to be lattice-aligned.
        assert report.max_quantised_residual_cm > 50.0

    def test_no_known_good_target_is_rejected(self):
        report = run_oracle(
            map_name="jittered", node_positions=jittered_grid(),
            max_range_m=MAX_RANGE_M, max_cameras=20,
        )
        assert report.rejected == {}


class TestOracleDetectsInjectedBugs:
    """Each case proves the oracle can fail; without them a pass means nothing."""

    def test_a_sign_flip_is_caught(self):
        report = run_with_broken_transform(
            lambda camera, point: (
                lambda v: Vec3(x_cm=v.x_cm, y_cm=-v.y_cm, z_cm=v.z_cm)
            )(_original(camera, point)),
            jittered_grid(),
        )
        assert report["status"] == "fail"
        assert report["failure_count"] > 100

    def test_an_axis_swap_is_caught(self):
        report = run_with_broken_transform(
            lambda camera, point: (
                lambda v: Vec3(x_cm=v.y_cm, y_cm=v.x_cm, z_cm=v.z_cm)
            )(_original(camera, point)),
            jittered_grid(),
        )
        assert report["status"] == "fail"

    def test_a_two_percent_scale_error_is_caught(self):
        """This is the one that motivated measuring the residual *before* the
        graph projection: snapping to the nearest node absorbed the error
        entirely and the oracle reported a clean pass."""
        report = run_with_broken_transform(
            lambda camera, point: (
                lambda v: Vec3(x_cm=v.x_cm * 1.02, y_cm=v.y_cm * 1.02, z_cm=v.z_cm)
            )(_original(camera, point)),
            jittered_grid(),
        )
        assert report["status"] == "fail"
        # Every target still projects onto a road node, so the post-projection
        # residual stays at zero. Only the raw residual shows the bug.
        assert report["max_residual_cm"] == pytest.approx(0.0, abs=1e-6)
        assert report["max_unprojected_residual_cm"] > 10.0

    def test_a_one_degree_heading_error_is_caught(self):
        report = run_with_broken_transform(
            lambda camera, point: _original(
                Pose(frame=camera.frame, position=camera.position,
                     yaw_deg=camera.yaw_deg + 1.0),
                point,
            ),
            jittered_grid(),
        )
        assert report["status"] == "fail"


class TestRejectionJudgement:
    def test_out_of_range_is_legitimate_only_when_really_out_of_range(self):
        inside = OracleCase(
            camera_x_cm=0, camera_y_cm=0, camera_yaw_deg=0, target_x_cm=0, target_y_cm=0,
            mode="nav_point_3d", visible=True, accepted=False,
            rejection="out_of_range", true_distance_m=5.0,
        )
        outside = OracleCase(
            camera_x_cm=0, camera_y_cm=0, camera_yaw_deg=0, target_x_cm=0, target_y_cm=0,
            mode="nav_point_3d", visible=True, accepted=False,
            rejection="out_of_range", true_distance_m=25.0,
        )
        assert _rejection_is_legitimate(inside, MAX_RANGE_M) is False
        assert _rejection_is_legitimate(outside, MAX_RANGE_M) is True

    @pytest.mark.parametrize(
        "reason", ["not_traversable", "no_ground_hit", "low_depth_confidence", "unknown"]
    )
    def test_no_other_rejection_is_ever_legitimate(self, reason):
        """The oracle aims at a real graph node on the ground and in frame, so
        there is no honest reason for the chain to refuse it."""
        case = OracleCase(
            camera_x_cm=0, camera_y_cm=0, camera_yaw_deg=0, target_x_cm=0, target_y_cm=0,
            mode="nav_point_3d", visible=True, accepted=False,
            rejection=reason, true_distance_m=5.0,
        )
        assert _rejection_is_legitimate(case, MAX_RANGE_M) is False


class TestVacuousRunsAreNotPasses:
    def test_an_empty_graph_does_not_pass(self):
        report = run_oracle(
            map_name="empty", node_positions=[], max_range_m=MAX_RANGE_M
        )
        assert not report.passed
        assert report.notes

    def test_a_graph_with_no_visible_neighbours_does_not_pass(self):
        """Nodes further apart than the range cap produce no round trips at all.
        Reporting that as success would certify an unvalidated chain."""
        far_apart = [(f"n_{i}", i * 100_000.0, 0.0) for i in range(5)]
        report = run_oracle(
            map_name="sparse", node_positions=far_apart, max_range_m=MAX_RANGE_M
        )
        assert not report.passed
        assert report.visible == 0


class TestParisOracle:
    """The real graph, so a regression in graph repair or the album's camera
    convention shows up here rather than in a training run weeks later."""

    @pytest.mark.skipif(
        not (Path(__file__).resolve().parents[1] / "vendor" / "vagen").exists(),
        reason="vendored VAGEN checkout not present",
    )
    def test_paris_round_trips_exactly(self):
        from embodiedbench.compiler.fpv_render import graph_node_positions

        nodes = graph_node_positions("citycore-paris")
        assert len(nodes) > 500, "Paris graph shrank unexpectedly"
        report = run_oracle(
            map_name="citycore-paris", node_positions=nodes,
            max_range_m=18.5, max_cameras=60,
        )
        assert report.passed, report.failures[:3]
        assert report.max_unprojected_residual_cm == pytest.approx(0.0, abs=1e-6)
        assert report.rejected == {}
        assert report.visible > 200
