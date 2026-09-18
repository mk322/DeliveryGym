"""Regression tests for the unified traffic-light + obstacle runtime.

These guard the 2026-06 unification (see TRAFFIC_LIGHT_AUDIT.md): one runtime
TrafficController built from the FPV manifest, count-only red lights, the
two-axis "front red / right green" behaviour, and the directed ObstacleField +
the seeded dock-obstacle sampler.

Run: python -m pytest vagen/envs/deliverybench/test_hazards_unified.py
"""

from pathlib import Path

from vagen.envs.deliverybench.vlm_delivery.utils.hazards import (
    ObstacleField,
    TrafficController,
    normalize_axis,
)

_HERE = Path(__file__).resolve().parent
_MANIFEST = (
    _HERE
    / "deliverybench_fpv"
    / "small-city-11-new"
    / "main_base_floor_road_full_1280x960"
    / "manifest.jsonl"
)
_MAP_DIR = _HERE / "maps" / "small-city-11"


def test_normalize_axis():
    assert normalize_axis("south-north") == "NS"
    assert normalize_axis("north-south") == "NS"
    assert normalize_axis("east-west") == "EW"
    assert normalize_axis("EW") == "EW"


def test_axis_of_cardinals():
    assert TrafficController.axis_of(0) == "NS"
    assert TrafficController.axis_of(180) == "NS"
    assert TrafficController.axis_of(90) == "EW"
    assert TrafficController.axis_of(270) == "EW"


def test_minute_parity_flips():
    tc = TrafficController([{"x_cm": 0, "y_cm": 0}])
    # even minute (t in [0,60)) -> NS green / EW red ; odd minute -> flipped
    assert tc.light("NS", 30.0) == "green" and tc.light("EW", 30.0) == "red"
    assert tc.light("NS", 90.0) == "red" and tc.light("EW", 90.0) == "green"


def test_two_axis_front_red_right_green():
    """The headline 'front red / right green' case at an intersection."""
    tc = TrafficController([{"x_cm": 0, "y_cm": 0}])
    # odd minute: travelling N (forward, NS axis) is red; E (right, EW axis) green
    assert tc.light_for_bearing(0.0, 90.0) == "red"
    assert tc.light_for_bearing(90.0, 90.0) == "green"
    # even minute flips both
    assert tc.light_for_bearing(0.0, 30.0) == "green"
    assert tc.light_for_bearing(90.0, 30.0) == "red"


def test_wait_advances_into_next_minute():
    tc = TrafficController([{"x_cm": 0, "y_cm": 0}])
    assert abs(tc.seconds_to_next_minute(30.0) - 30.0) < 1e-6
    assert abs(tc.seconds_to_next_minute(60.0) - 60.0) < 1e-6  # boundary -> full period


def test_controller_from_manifest_matches_images():
    if not _MANIFEST.exists():
        return  # dataset not present in this checkout
    tc = TrafficController.load_manifest(_MANIFEST)
    assert len(tc) == 34  # 34 signalised intersections in small-city-11-new
    import json
    row = next(
        json.loads(line)
        for line in _MANIFEST.read_text(encoding="utf-8").splitlines()
        if '"render_kind": "traffic_light"' in line
    )
    assert tc.is_signalised(row["x_cm"], row["y_cm"])
    assert not tc.is_signalised(row["x_cm"] + 99999, row["y_cm"])
    # the recorded face axis round-trips
    assert tc.face_axis(row["x_cm"], row["y_cm"], row["yaw"]) == normalize_axis(row["signal_axis"])


def test_obstacle_field_is_directed():
    of = ObstacleField([
        {"src_x_cm": 100, "src_y_cm": 200, "dst_x_cm": 100, "dst_y_cm": 700, "type": "road_block"},
    ])
    assert of.obstacle_on(100, 200, 100, 700) == "road_block"
    assert of.obstacle_on(100, 700, 100, 200) is None  # reverse edge is clear (one-sided)
    assert len(of) == 1


def test_dock_obstacle_sampler_is_seeded_one_sided_and_dock_to_dock():
    if not (_MAP_DIR / "progen_world_enriched.json").exists():
        return
    from vagen.envs.deliverybench.tools.render_dock_obstacle_ue import build_jobs

    out = _HERE / "deliverybench_fpv" / "small-city-11-obstacles"
    jobs_a = build_jobs(_MAP_DIR, out, seed=5)
    jobs_b = build_jobs(_MAP_DIR, out, seed=5)
    assert [j.dock_id for j in jobs_a] == [j.dock_id for j in jobs_b]  # deterministic
    assert jobs_a, "expected at least one sampled dock"
    for j in jobs_a:
        assert j.dst_id.startswith("dock_"), f"obstacle must point dock->next-dock, got {j.dst_id}"
    of = ObstacleField([
        {"src_x_cm": j.dock_x_cm, "src_y_cm": j.dock_y_cm,
         "dst_x_cm": j.dst_x_cm, "dst_y_cm": j.dst_y_cm, "type": j.obstacle_type}
        for j in jobs_a
    ])
    j = jobs_a[0]
    assert of.obstacle_on(j.dock_x_cm, j.dock_y_cm, j.dst_x_cm, j.dst_y_cm) is not None
    assert of.obstacle_on(j.dst_x_cm, j.dst_y_cm, j.dock_x_cm, j.dock_y_cm) is None  # one-sided
    # stored yaw matches the FPV fetch convention (90 - bearing) for this facing
    assert j.stored_yaw == int(round((90.0 - j.bearing_deg) % 360.0))


def test_unified_driver_plan_covers_all_kinds():
    if not (_MAP_DIR / "progen_world_enriched.json").exists():
        return
    from vagen.envs.deliverybench.tools.render_fpv_dataset_ue import (
        build_jobs, _to_waypoint_light_job, _phase_for,
    )

    out = _HERE / "deliverybench_fpv" / "small-city-11"
    view_jobs, obstacle_jobs, sidecar = build_jobs(_MAP_DIR, out, seed=5)
    kinds = {}
    for j in view_jobs:
        kinds[j.render_kind] = kinds.get(j.render_kind, 0) + 1
    # plain = every waypoint x 4 yaws
    n_wp = len({(j.x_cm, j.y_cm) for j in view_jobs if j.render_kind == "plain"})
    assert kinds["plain"] == n_wp * 4
    # lights come in green+red pairs
    assert kinds["traffic_light"] % 2 == 0 and kinds["traffic_light"] > 0
    # canonical naming
    for j in view_jobs:
        name = j.image_path.rsplit("/", 1)[-1]
        if j.render_kind == "plain":
            assert name == f"yaw_{j.stored_yaw:03d}.png"
        else:
            assert name in (f"yaw_{j.stored_yaw:03d}_green.png", f"yaw_{j.stored_yaw:03d}_red.png")
    for o in obstacle_jobs:
        assert o.blocked_path.endswith(f"yaw_{o.stored_yaw:03d}_blocked.png")
    # every obstacle blocked view has a plain clear twin at the same (pos, yaw)
    plain_keys = {(j.x_cm, j.y_cm, j.stored_yaw) for j in view_jobs if j.render_kind == "plain"}
    for o in obstacle_jobs:
        assert (o.dock_x_cm, o.dock_y_cm, o.stored_yaw) in plain_keys
    assert len(sidecar) == len(obstacle_jobs)
    # rendering delegates to the proven UE script: every view job must translate
    # to a valid WaypointLightJob, and the phase must encode the right colour.
    assert _phase_for("south-north", "green") == "even" and _phase_for("east-west", "green") == "odd"
    for j in view_jobs[:50]:
        wlj = _to_waypoint_light_job(j, "small-city-11")
        assert wlj.output_path == j.image_path
        assert wlj.render_kind == ("traffic_light" if j.render_kind == "traffic_light" else "normal_scene")


def test_env_selects_correct_fpv_image_from_canonical_manifest():
    """End-to-end: the env's own _load_fpv_lookup + _select_fpv_image return the
    right panel image for the canonical naming, every call.

    Builds a tiny tagged dataset (one solid colour per file), then checks the
    front/right panels show opposite signals at one observation, the light flips
    when the clock advances a minute, the obstacle blocked view shows only on the
    blocked edge, and with mechanics off everything falls back to plain.
    """
    import json
    import tempfile
    try:
        from PIL import Image
        from vagen.envs.deliverybench.deliverybench_env import DeliveryBench
    except Exception:
        return  # env deps unavailable in this checkout

    colors = {
        "int_000/yaw_090.png": (10, 10, 10),
        "int_000/yaw_090_green.png": (0, 200, 0), "int_000/yaw_090_red.png": (200, 0, 0),
        "int_000/yaw_000.png": (11, 11, 11),
        "int_000/yaw_000_green.png": (0, 150, 0), "int_000/yaw_000_red.png": (150, 0, 0),
        "dock_000/yaw_000.png": (20, 20, 20), "dock_000/yaw_000_blocked.png": (0, 0, 200),
        "dock_000/yaw_090.png": (30, 30, 30),
    }
    with tempfile.TemporaryDirectory(prefix="fpvtest_") as td:
        tmp = Path(td)
        for rel, c in colors.items():
            p = tmp / "images" / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 8), c).save(p)

        def row(wp, kind, x, y, yaw, rk, img, **extra):
            d = {"waypoint_id": wp, "waypoint_kind": kind, "x_cm": x, "y_cm": y,
                 "yaw": yaw, "render_kind": rk, "image_path": f"x/{img}", "status": "ok"}
            d.update(extra)
            return d

        rows = [
            row("int_0", "intersection", 0.0, 0.0, 90.0, "plain", "int_000/yaw_090.png"),
            row("int_0", "intersection", 0.0, 0.0, 90.0, "traffic_light", "int_000/yaw_090_green.png", signal_state="green", signal_axis="south-north"),
            row("int_0", "intersection", 0.0, 0.0, 90.0, "traffic_light", "int_000/yaw_090_red.png", signal_state="red", signal_axis="south-north"),
            row("int_0", "intersection", 0.0, 0.0, 0.0, "plain", "int_000/yaw_000.png"),
            row("int_0", "intersection", 0.0, 0.0, 0.0, "traffic_light", "int_000/yaw_000_green.png", signal_state="green", signal_axis="east-west"),
            row("int_0", "intersection", 0.0, 0.0, 0.0, "traffic_light", "int_000/yaw_000_red.png", signal_state="red", signal_axis="east-west"),
            row("dock_0", "dock", 1000.0, 0.0, 0.0, "plain", "dock_000/yaw_000.png"),
            row("dock_0", "dock", 1000.0, 0.0, 0.0, "obstacle", "dock_000/yaw_000_blocked.png", obstacle_type="road_block", obstacle_state="blocked"),
            row("dock_0", "dock", 1000.0, 0.0, 90.0, "plain", "dock_000/yaw_090.png"),
        ]
        (tmp / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

        env = DeliveryBench({"fpv_dir": str(tmp)})
        env._load_fpv_lookup()

        class _Clock:
            def __init__(self, t): self.t = t
            def now_sim(self): return self.t

        class _DM:
            pass

        dm = _DM()
        dm._traffic = TrafficController.load_manifest(tmp / "manifest.jsonl")
        dm._obstacle_field = ObstacleField([
            {"src_x_cm": 1000.0, "src_y_cm": 0.0, "dst_x_cm": 1500.0, "dst_y_cm": 0.0, "type": "road_block"},
        ])
        dm.clock = _Clock(30.0)
        INT, DOCK = (0.0, 0.0), (1000.0, 0.0)

        def tag(img):
            return None if img is None else img.getpixel((0, 0))

        # even minute: NS green, EW red -> front(N) green, right(E) red at one obs
        assert tag(env._select_fpv_image(INT, 0.0, 90.0, dm)) == (0, 200, 0)
        assert tag(env._select_fpv_image(INT, 90.0, 0.0, dm)) == (150, 0, 0)
        assert tag(env._select_fpv_image(INT, 0.0, 90.0, dm)) == (0, 200, 0)  # deterministic

        # odd minute: flips
        dm.clock.t = 90.0
        assert tag(env._select_fpv_image(INT, 0.0, 90.0, dm)) == (200, 0, 0)
        assert tag(env._select_fpv_image(INT, 90.0, 0.0, dm)) == (0, 150, 0)

        # obstacle shows only on the blocked edge; the other facing is plain
        assert tag(env._select_fpv_image(DOCK, 90.0, 0.0, dm)) == (0, 0, 200)
        assert tag(env._select_fpv_image(DOCK, 0.0, 90.0, dm)) == (30, 30, 30)

        # mechanics off -> plain everywhere (no cone, no signal)
        dm.clock.t = 90.0
        dm._traffic = None
        dm._obstacle_field = None
        assert tag(env._select_fpv_image(INT, 0.0, 90.0, dm)) == (10, 10, 10)
        assert tag(env._select_fpv_image(DOCK, 90.0, 0.0, dm)) == (20, 20, 20)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
    print("ALL PASSED")
