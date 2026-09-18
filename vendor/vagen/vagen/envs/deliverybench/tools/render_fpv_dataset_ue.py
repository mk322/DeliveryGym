"""Unified DeliveryBench FPV dataset renderer (plain + traffic-light + obstacle).

One driver that builds the whole FPV dataset for a map with a single canonical
naming protocol and a single manifest, so the runtime has exactly one source of
truth aligned to the images. It produces three kinds of view:

1. **plain**         — every waypoint, all 4 cardinal yaws.
2. **traffic_light** — at each *intersection*, for every direction that crosses a
   vehicle road ("facing the other side of the road"), both **green** and **red**.
3. **obstacle**      — a seeded sample of docks gets a `RoadCone_C` 5 m ahead on
   the dock→next-dock edge; the blocked front view is rendered (its clear twin is
   the plain view at that yaw, so no separate clear shot is needed).

Rendering is **delegated to the proven per-kind UE scripts** (plain + light ->
`render_waypoint_pedestrian_light_ue._ue_script`; obstacle ->
`render_dock_obstacle_ue._ue_script`); this driver only *plans* jobs, assigns
canonical filenames, and writes one manifest + `obstacles.json`.

Defaults reproduce `deliverybench_fpv/<map>/ue_render_setup_record.md`, the config
that is actually hosted on the MCP/editor server:
  - traffic lights -> **face-aligned** camera 1600 cm in front of the signal
    face, z 170, 1280x720; light render-location = `crossing_center +
    (json_location - crossing_center) * 10` (matters at non-origin intersections);
  - plain        -> z 160, 640x480;
  - world assets are spawned from progen JSON (the .md spawned them) — pass
    `--no-spawn-map-assets` if the loaded level already holds the full city.

Canonical image naming (under `<out>/images/<waypoint_id>/`), joined to the
runtime by `(round(x_cm,1), round(y_cm,1), yaw)`:

    yaw_<NNN>.png            # plain (the agent's front/side/back view)
    yaw_<NNN>_green.png      # intersection crossing face, light green
    yaw_<NNN>_red.png        # intersection crossing face, light red
    yaw_<NNN>_blocked.png    # sampled dock, road block present

`<NNN>` is the stored FPV yaw; the runtime fetches `yaw = (90 - compass_dir)`, so
yaw 000→E, 090→N, 180→W, 270→S. The manifest `render_kind` is `plain` /
`traffic_light` / `obstacle`; light rows add `signal_state` + `signal_axis`,
obstacle rows add `obstacle_type`/`obstacle_state` + `edge_dst_*` and the
`obstacles.json` sidecar (loaded by the runtime `ObstacleField`).

Examples:
    # plan only (no UE server) — writes manifest.jsonl + obstacles.json:
    python -m vagen.envs.deliverybench.tools.render_fpv_dataset_ue \
        vagen/envs/deliverybench/maps/small-city-11 --dry-run

    # full render against a SimWorld Studio MCP server:
    python -m vagen.envs.deliverybench.tools.render_fpv_dataset_ue \
        vagen/envs/deliverybench/maps/small-city-11 --mcp-port 55565
"""

from __future__ import annotations

import os
import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from vagen.envs.deliverybench.tools.render_pedestrian_light_ue import (
    _send_mcp_script,
    _wait_for_output,
)
from vagen.envs.deliverybench.tools.render_waypoint_pedestrian_light_ue import (
    PED_LIGHT_RENDER_XY_SCALE,
    WaypointLightJob,
    _load_map,
    _minimal_light,
    _select_visible_light_for_edge,
    _ue_script as _wp_ue_script,
    _yaw_for_abs_direction,
)
from vagen.envs.deliverybench.tools.render_dock_obstacle_ue import (
    DEFAULT_CONE_ASSET,
    DEFAULT_CONE_SCALE,
    DEFAULT_OBSTACLE_DIST_CM,
    DEFAULT_SAMPLE_FRAC,
    DEFAULT_SEED,
    ObstacleJob,
    build_jobs as build_obstacle_jobs,
    _ue_script as _cone_ue_script,
)

FPV_YAW_OFFSET_DEG = 90.0
STORED_YAWS = (0, 90, 180, 270)

# Defaults reproduce the ue_render_setup_record.md whole-dataset config so the
# MCP/editor render matches what is hosted:
#   traffic lights -> face-aligned camera 1600 cm in front of the signal face,
#     z 170, 1280x720, render-location = crossing_center + (json-center)*10.
#   normal/plain   -> z 160, 640x480.
# (The .md asset-scale 2.0 is visibility-only; it is not threaded here.)
DEFAULT_LIGHT_Z = 170.0             # traffic_light_z_cm
DEFAULT_NORMAL_Z = 160.0            # normal_z_cm
DEFAULT_CAMERA_DISTANCE = 1600.0    # face camera: 1600 cm in front of the signal face
DEFAULT_CAMERA_BACKOFF_CM = 900.0   # waypoint mode (plain) only
DEFAULT_FACE_CAMERA_SIGN = 1.0
DEFAULT_FOV = 90.0
DEFAULT_LIGHT_IMG_W = 1280          # traffic_light_resolution
DEFAULT_LIGHT_IMG_H = 720
DEFAULT_NORMAL_IMG_W = 640          # normal_resolution
DEFAULT_NORMAL_IMG_H = 480
DEFAULT_LIGHT_RENDER_XY_SCALE = 10.0  # render_location_fix multiplier (center-relative)
DEFAULT_LIGHT_ASSET_SCALE = 2.0       # traffic_light_asset_scale (.md)
DEFAULT_CONE_Z = 0.0

_CARDINALS = (("north", 0.0), ("east", 90.0), ("south", 180.0), ("west", 270.0))


def _yaw_to_compass(stored_yaw: float) -> float:
    return (FPV_YAW_OFFSET_DEG - float(stored_yaw)) % 360.0


def _abs_for_compass(bearing_deg: float) -> str:
    b = float(bearing_deg) % 360.0
    best, best_d = "north", 999.0
    for name, deg in _CARDINALS:
        d = min(abs(b - deg), 360.0 - abs(b - deg))
        if d < best_d:
            best, best_d = name, d
    return best


def _stored_yaw_for_compass(bearing_deg: float) -> int:
    return int(round((FPV_YAW_OFFSET_DEG - float(bearing_deg)) % 360.0))


def _signal_axis_for_abs(abs_dir: str) -> str:
    return "south-north" if abs_dir in ("north", "south") else "east-west"


def _phase_for(axis_verbose: str, state: str) -> str:
    """Phase string the proven UE script uses so the rendered face shows ``state``.

    Proven convention (_state_for_phase): even -> NS green / EW red; odd flips.
    """
    is_ns = str(axis_verbose).lower().startswith(("south", "north"))
    if is_ns:
        return "even" if state == "green" else "odd"
    return "odd" if state == "green" else "even"


def _md_light_payload(node: Dict[str, Any], xy_scale: float) -> Dict[str, Any]:
    """Minimal light dict with the ue_render_setup_record.md render-location fix
    applied: location = crossing_center + (json_location - crossing_center)*scale.

    The proven _ue_script then spawns at this location with ped_light_xy_scale=1,
    so the signal lands where the recorded dataset put it (matters at non-origin
    intersections, where a plain location*scale would be far off).
    """
    ml = _minimal_light(node)
    props = node.get("properties", {}) or {}
    center = (props.get("controlled_crossing", {}) or {}).get("center", {}) or {}
    loc = ml["properties"].get("location", {}) or {}
    try:
        cx, cy = float(center["x"]), float(center["y"])
    except (KeyError, TypeError):
        return ml  # no crossing center recorded -> leave as-is
    lx = float(loc.get("x", cx)); ly = float(loc.get("y", cy)); lz = float(loc.get("z", 0.0))
    ml["properties"]["location"] = {
        "x": cx + (lx - cx) * float(xy_scale),
        "y": cy + (ly - cy) * float(xy_scale),
        "z": lz,
    }
    return ml


@dataclass
class ViewJob:
    """A planned plain or traffic-light view (obstacles are ObstacleJobs)."""
    render_kind: str            # "plain" | "traffic_light"
    waypoint_id: str
    waypoint_kind: str
    waypoint_name: str
    x_cm: float
    y_cm: float
    stored_yaw: int
    abs_direction: str          # compass direction the camera looks
    image_path: str
    # traffic-light extras
    signal_state: str = ""      # "green" | "red"
    signal_axis: str = ""       # "south-north" | "east-west"
    dst_id: str = ""
    dst_x_cm: float = 0.0
    dst_y_cm: float = 0.0
    light_id: str = ""
    light_face: str = ""
    light_face_direction: str = ""
    light_node: Dict[str, Any] = field(default_factory=dict)   # _minimal_light(node)


def _img_dir(out_root: Path, waypoint_id: str) -> Path:
    kind, num = waypoint_id.rsplit("_", 1)
    return out_root / "images" / f"{kind}_{int(num):03d}"


def build_jobs(
    scenario_dir: Path,
    out_root: Path,
    *,
    sample_frac: float = DEFAULT_SAMPLE_FRAC,
    seed: int = DEFAULT_SEED,
    obstacle_dist_cm: float = DEFAULT_OBSTACLE_DIST_CM,
    cone_z: float = DEFAULT_CONE_Z,
    obstacle_type: str = "road_block",
    light_render_xy_scale: float = DEFAULT_LIGHT_RENDER_XY_SCALE,
    max_waypoints: Optional[int] = None,
) -> Tuple[List[ViewJob], List[ObstacleJob], List[Dict[str, Any]]]:
    """Return (plain+light view jobs, obstacle jobs, obstacles.json entries)."""
    city_map = _load_map(scenario_dir)
    lights = list(getattr(city_map, "traffic_lights", []) or [])
    by_id = {str(l.get("id")): l for l in lights}
    map_name = scenario_dir.name

    waypoints = sorted(getattr(city_map, "waypoints_by_id", {}).items(), key=lambda kv: kv[0])
    if max_waypoints is not None:
        waypoints = waypoints[:max_waypoints]

    view_jobs: List[ViewJob] = []
    for wp_id, node in waypoints:
        wp_kind = str(getattr(node, "waypoint_kind", "")).lower()
        wp_name = str(getattr(node, "waypoint_name", ""))
        x_cm, y_cm = round(float(node.position.x), 1), round(float(node.position.y), 1)
        img_dir = _img_dir(out_root, wp_id)

        # 1) plain: every waypoint, all 4 yaws.
        for yaw in STORED_YAWS:
            abs_dir = _abs_for_compass(_yaw_to_compass(yaw))
            view_jobs.append(ViewJob(
                render_kind="plain", waypoint_id=wp_id, waypoint_kind=wp_kind,
                waypoint_name=wp_name, x_cm=x_cm, y_cm=y_cm, stored_yaw=yaw,
                abs_direction=abs_dir, image_path=str(img_dir / f"yaw_{yaw:03d}.png"),
            ))

        # 2) traffic_light: intersection crossing faces, green + red (only when a
        #    controlling pedestrian light is found for the edge).
        if wp_kind == "intersection":
            for adj in city_map.adjacents(node):
                if not adj.get("crosses_vehicle_road"):
                    continue
                target = adj.get("node")
                if target is None:
                    continue
                bearing = float(adj.get("bearing_deg", 0.0))
                abs_dir = _abs_for_compass(bearing)
                yaw = _stored_yaw_for_compass(bearing)
                axis = _signal_axis_for_abs(abs_dir)
                lid, face, face_dir = _select_visible_light_for_edge(
                    lights,
                    source_x=float(node.position.x), source_y=float(node.position.y),
                    target_x=float(target.position.x), target_y=float(target.position.y),
                    movement_abs_direction=abs_dir, fallback_light_id="",
                )
                light_node = by_id.get(lid)
                if not light_node:
                    continue  # no renderable signal for this crossing -> plain only
                for state in ("green", "red"):
                    view_jobs.append(ViewJob(
                        render_kind="traffic_light", waypoint_id=wp_id, waypoint_kind=wp_kind,
                        waypoint_name=wp_name, x_cm=x_cm, y_cm=y_cm, stored_yaw=yaw,
                        abs_direction=abs_dir,
                        image_path=str(img_dir / f"yaw_{yaw:03d}_{state}.png"),
                        signal_state=state, signal_axis=axis,
                        dst_id=str(adj.get("id", "")),
                        dst_x_cm=round(float(target.position.x), 1),
                        dst_y_cm=round(float(target.position.y), 1),
                        light_id=str(lid), light_face=str(face), light_face_direction=str(face_dir),
                        light_node=_md_light_payload(light_node, light_render_xy_scale),
                    ))

    # 3) obstacle: seeded dock sample (dock -> next dock), reusing the obstacle
    #    sampler so the rule + obstacles.json schema stay single-sourced.
    obstacle_jobs = build_obstacle_jobs(
        scenario_dir, out_root, sample_frac=sample_frac, seed=seed,
        obstacle_dist_cm=obstacle_dist_cm, cone_z=cone_z, obstacle_type=obstacle_type,
    )
    sidecar = [{
        "src_id": oj.dock_id, "src_x_cm": oj.dock_x_cm, "src_y_cm": oj.dock_y_cm,
        "dst_id": oj.dst_id, "dst_x_cm": oj.dst_x_cm, "dst_y_cm": oj.dst_y_cm,
        "bearing_deg": oj.bearing_deg, "type": oj.obstacle_type,
    } for oj in obstacle_jobs]
    return view_jobs, obstacle_jobs, sidecar


def _to_waypoint_light_job(vj: ViewJob, map_name: str) -> WaypointLightJob:
    """Translate a plain/light ViewJob into the proven renderer's job dataclass."""
    is_light = vj.render_kind == "traffic_light"
    tx, ty = (vj.dst_x_cm, vj.dst_y_cm) if is_light else (vj.x_cm, vj.y_cm)
    return WaypointLightJob(
        map_name=map_name,
        source_waypoint_id=vj.waypoint_id,
        source_waypoint_name=vj.waypoint_name,
        source_x=float(vj.x_cm), source_y=float(vj.y_cm),
        source_render_x=float(vj.x_cm), source_render_y=float(vj.y_cm),
        target_waypoint_id=vj.dst_id or vj.waypoint_id,
        target_waypoint_name="",
        target_x=float(tx), target_y=float(ty),
        target_render_x=float(tx), target_render_y=float(ty),
        movement_abs_direction=vj.abs_direction,
        camera_abs_direction=vj.abs_direction,
        camera_yaw_deg=float(_yaw_for_abs_direction(vj.abs_direction)),
        agent_facing_direction=vj.abs_direction,
        view_direction="front",
        action_direction="forward",
        phase=_phase_for(vj.signal_axis, vj.signal_state) if is_light else "even",
        signal_state=vj.signal_state if is_light else "normal",
        signal_axis=vj.signal_axis,
        light_id=vj.light_id,
        light_face=vj.light_face,
        light_face_direction=vj.light_face_direction,
        render_kind="traffic_light" if is_light else "normal_scene",
        output_path=vj.image_path,
    )


def _manifest_row_view(
    vj: ViewJob,
    map_name: str,
    *,
    normal_z: float = DEFAULT_NORMAL_Z,
    light_z: float = DEFAULT_LIGHT_Z,
) -> Dict[str, Any]:
    is_light = vj.render_kind == "traffic_light"
    row: Dict[str, Any] = {
        "map_name": map_name, "waypoint_id": vj.waypoint_id,
        "waypoint_kind": vj.waypoint_kind, "waypoint_name": vj.waypoint_name,
        "x_cm": vj.x_cm, "y_cm": vj.y_cm, "yaw": float(vj.stored_yaw),
        "z_cm": float(light_z) if is_light else float(normal_z),
        "camera_id": 0, "capture_mode": "camera",
        "render_kind": vj.render_kind, "image_path": vj.image_path,
        "status": "ok", "error": None,
    }
    if is_light:
        row.update({"signal_state": vj.signal_state, "signal_axis": vj.signal_axis,
                    "light_id": vj.light_id, "light_face": vj.light_face,
                    "light_face_direction": vj.light_face_direction})
    return row


def _manifest_row_obstacle(
    oj: ObstacleJob,
    map_name: str,
    *,
    normal_z: float = DEFAULT_NORMAL_Z,
) -> Dict[str, Any]:
    return {
        "map_name": map_name, "waypoint_id": oj.dock_id, "waypoint_kind": "dock",
        "waypoint_name": oj.dock_name, "x_cm": oj.dock_x_cm, "y_cm": oj.dock_y_cm,
        "yaw": float(oj.stored_yaw), "z_cm": float(normal_z), "camera_id": 0,
        "capture_mode": "camera", "render_kind": "obstacle",
        "obstacle_type": oj.obstacle_type, "obstacle_state": "blocked",
        "bearing_deg": oj.bearing_deg, "edge_dst_id": oj.dst_id,
        "edge_dst_x_cm": oj.dst_x_cm, "edge_dst_y_cm": oj.dst_y_cm,
        "image_path": oj.blocked_path, "status": "ok", "error": None,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("scenario_dir", type=Path)
    p.add_argument("--out-dir", type=Path, default=None,
                   help="FPV output dir (default: deliverybench_fpv/<map>)")
    p.add_argument("--mcp-port", type=int, default=55565)
    p.add_argument("--sample-frac", type=float, default=DEFAULT_SAMPLE_FRAC)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--obstacle-dist-cm", type=float, default=DEFAULT_OBSTACLE_DIST_CM)
    p.add_argument("--obstacle-type", default="road_block", choices=["road_block", "slow_pedestrian"])
    p.add_argument("--cone-asset", default=DEFAULT_CONE_ASSET)
    p.add_argument("--cone-z", type=float, default=DEFAULT_CONE_Z)
    p.add_argument("--cone-scale", type=float, default=DEFAULT_CONE_SCALE,
                   help="actor scale for the road cone (default 5x)")
    # camera / world — defaults reproduce ue_render_setup_record.md
    p.add_argument("--light-camera-mode", choices=["waypoint", "face"], default="face",
                   help="'face' (.md default) = camera 1600 cm in front of the signal "
                        "face, looking at it; 'waypoint' = the agent's forward view.")
    p.add_argument("--light-z", type=float, default=DEFAULT_LIGHT_Z,
                   help="traffic_light_z_cm (.md: 170)")
    p.add_argument("--normal-z", type=float, default=DEFAULT_NORMAL_Z,
                   help="normal_z_cm (.md: 160)")
    p.add_argument("--camera-backoff-cm", type=float, default=DEFAULT_CAMERA_BACKOFF_CM,
                   help="waypoint-mode (plain) backoff only")
    p.add_argument("--normal-camera-backoff-cm", type=float, default=None,
                   help="plain waypoint camera backoff override; defaults to --camera-backoff-cm")
    p.add_argument("--light-camera-backoff-cm", type=float, default=None,
                   help="traffic-light waypoint camera backoff override; defaults to --camera-backoff-cm")
    p.add_argument("--obstacle-camera-backoff-cm", type=float, default=None,
                   help="obstacle camera backoff override; defaults to --camera-backoff-cm")
    p.add_argument("--camera-distance", type=float, default=DEFAULT_CAMERA_DISTANCE,
                   help="face-mode distance in front of the signal (.md: 1600)")
    p.add_argument("--face-camera-sign", type=float, default=DEFAULT_FACE_CAMERA_SIGN)
    p.add_argument("--camera-fov", type=float, default=DEFAULT_FOV)
    p.add_argument("--light-render-xy-scale", type=float, default=DEFAULT_LIGHT_RENDER_XY_SCALE,
                   help="render_location_fix: center + (json-center)*scale (.md: 10)")
    p.add_argument("--light-asset-scale", type=float, default=DEFAULT_LIGHT_ASSET_SCALE,
                   help="actor scale for the spawned signal (.md: 2.0)")
    p.add_argument("--light-image-width", type=int, default=DEFAULT_LIGHT_IMG_W)
    p.add_argument("--light-image-height", type=int, default=DEFAULT_LIGHT_IMG_H)
    p.add_argument("--normal-image-width", type=int, default=DEFAULT_NORMAL_IMG_W)
    p.add_argument("--normal-image-height", type=int, default=DEFAULT_NORMAL_IMG_H)
    p.add_argument("--spawn-map-assets", action=argparse.BooleanOptionalAction, default=True,
                   help="Spawn progen world assets before rendering (.md spawned them; "
                        "default on). Use --no-spawn-map-assets if the level already has "
                        "the full city.")
    p.add_argument("--clear-existing-scene", action="store_true",
                   help="Before the first render, destroy existing visible/map actors in "
                        "the loaded UE level and then spawn only the map assets from "
                        "the JSON world. Lighting/sky/fog/postprocess/camera actors are "
                        "kept so screenshots remain illuminated.")
    p.add_argument("--world-json", type=Path, default=None)
    p.add_argument("--ue-assets-json", type=Path,
                   default=Path(os.environ.get("UE_ASSETS_JSON", "simworld/data/ue_assets.json")))
    p.add_argument("--max-waypoints", type=int, default=None)
    p.add_argument("--only", choices=["plain", "traffic_light", "obstacle"], default=None)
    p.add_argument("--mcp-timeout-s", type=float, default=120.0,
                   help="socket wait for each MCP Python execution; raise this for first-time UE asset compilation")
    p.add_argument("--screenshot-timeout-s", type=float, default=60.0)
    p.add_argument("--skip-existing-newer-than", type=Path, default=None,
                   help="Resume helper: skip a screenshot only when its output "
                        "exists and is newer than this marker file.")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    scenario_dir = args.scenario_dir.resolve()
    map_name = scenario_dir.name
    out_root = (args.out_dir or (scenario_dir.parent.parent / "deliverybench_fpv" / map_name)).resolve()
    world_json = (args.world_json or (scenario_dir / "progen_world_enriched.json")).resolve()

    view_jobs, obstacle_jobs, sidecar = build_jobs(
        scenario_dir, out_root, sample_frac=args.sample_frac, seed=args.seed,
        obstacle_dist_cm=args.obstacle_dist_cm, cone_z=args.cone_z,
        obstacle_type=args.obstacle_type, light_render_xy_scale=args.light_render_xy_scale,
        max_waypoints=args.max_waypoints,
    )
    if args.only == "obstacle":
        view_jobs = []
    elif args.only in ("plain", "traffic_light"):
        view_jobs = [v for v in view_jobs if v.render_kind == args.only]
        obstacle_jobs, sidecar = [], sidecar  # keep sidecar for the env even if not rendering

    n_plain = sum(1 for v in view_jobs if v.render_kind == "plain")
    n_light = sum(1 for v in view_jobs if v.render_kind == "traffic_light")
    print(f"[plan] map={map_name} -> {out_root}")
    print(f"[plan] jobs: plain={n_plain} traffic_light={n_light} obstacle={len(obstacle_jobs)} "
          f"(seed={args.seed}, frac={args.sample_frac})")

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "obstacles.json").write_text(json.dumps({
        "map_name": map_name, "seed": args.seed, "sample_frac": args.sample_frac,
        "obstacle_dist_cm": args.obstacle_dist_cm, "obstacles": sidecar,
    }, indent=2), encoding="utf-8")

    def _write_manifest():
        with (out_root / "manifest.jsonl").open("w", encoding="utf-8") as fh:
            for v in view_jobs:
                fh.write(json.dumps(_manifest_row_view(
                    v, map_name, normal_z=args.normal_z, light_z=args.light_z
                )) + "\n")
            for o in obstacle_jobs:
                fh.write(json.dumps(_manifest_row_obstacle(
                    o, map_name, normal_z=args.normal_z
                )) + "\n")

    def _write_waypoints():
        by_id: Dict[str, Dict[str, Any]] = {}
        for v in view_jobs:
            by_id.setdefault(v.waypoint_id, {
                "id": v.waypoint_id,
                "kind": v.waypoint_kind,
                "name": v.waypoint_name,
                "x_cm": v.x_cm,
                "y_cm": v.y_cm,
                "z_cm": float(args.normal_z),
            })
        for o in obstacle_jobs:
            by_id.setdefault(o.dock_id, {
                "id": o.dock_id,
                "kind": "dock",
                "name": o.dock_name,
                "x_cm": o.dock_x_cm,
                "y_cm": o.dock_y_cm,
                "z_cm": float(args.normal_z),
            })
        def sort_key(item: Dict[str, Any]) -> Tuple[str, int]:
            ident = str(item.get("id") or "")
            try:
                kind, num = ident.rsplit("_", 1)
                return kind, int(num)
            except Exception:
                return ident, 0
        payload = {
            "map_name": map_name,
            "source": str(scenario_dir),
            "waypoints": sorted(by_id.values(), key=sort_key),
        }
        (out_root / "waypoints.json").write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )

    if args.dry_run:
        _write_manifest()
        _write_waypoints()
        print(f"[dry-run] wrote obstacles.json + manifest.jsonl "
              f"+ waypoints.json ({len(view_jobs) + len(obstacle_jobs)} rows). No UE contact.")
        return

    # --- render: delegate to the proven per-kind UE scripts, with per-kind
    #     camera config matching ue_render_setup_record.md ---
    common = dict(
        face_camera_sign=args.face_camera_sign, camera_backoff_cm=args.camera_backoff_cm,
        camera_distance=args.camera_distance, camera_fov=args.camera_fov,
        world_json_path=world_json, ue_assets_json_path=args.ue_assets_json,
        spawn_map_assets=args.spawn_map_assets, clear_existing_scene=False,
        debug_markers=False,
    )
    normal_camera_backoff_cm = (
        args.camera_backoff_cm if args.normal_camera_backoff_cm is None
        else args.normal_camera_backoff_cm
    )
    light_camera_backoff_cm = (
        args.camera_backoff_cm if args.light_camera_backoff_cm is None
        else args.light_camera_backoff_cm
    )
    obstacle_camera_backoff_cm = (
        args.camera_backoff_cm if args.obstacle_camera_backoff_cm is None
        else args.obstacle_camera_backoff_cm
    )
    # traffic-light: face-aligned, z 170, 1280x720; the render-location fix is
    # already baked into light_node, so spawn at scale 1; asset scale per .md (2.0).
    light_kwargs = dict(common, camera_z=args.light_z,
                        image_width=args.light_image_width, image_height=args.light_image_height,
                        camera_backoff_cm=light_camera_backoff_cm,
                        ped_light_xy_scale=1.0, light_asset_scale=args.light_asset_scale)
    # plain: waypoint eye view, z 160, 640x480 (no light spawned).
    plain_kwargs = dict(common, camera_z=args.normal_z,
                        image_width=args.normal_image_width, image_height=args.normal_image_height,
                        camera_backoff_cm=normal_camera_backoff_cm,
                        ped_light_xy_scale=1.0)
    cone_kwargs = dict(
        cone_asset=args.cone_asset, camera_z=args.normal_z,
        camera_backoff_cm=obstacle_camera_backoff_cm, camera_fov=args.camera_fov,
        image_width=args.normal_image_width, image_height=args.normal_image_height,
        spawn_map_assets=args.spawn_map_assets, clear_existing_scene=False,
        world_json_path=str(world_json),
        ue_assets_json_path=str(args.ue_assets_json),
        cone_scale=args.cone_scale,
    )

    rendered = 0
    skipped = 0
    total = len(view_jobs) + len(obstacle_jobs)
    scene_reset_done = False
    script_path = out_root / "_last_fpv_render_ue_job.py"
    skip_marker_mtime = None
    if args.skip_existing_newer_than is not None:
        marker = args.skip_existing_newer_than.resolve()
        if not marker.exists():
            raise SystemExit(f"skip marker does not exist: {marker}")
        skip_marker_mtime = marker.stat().st_mtime

    def _send(script: str, out_path: str):
        nonlocal rendered, skipped
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if (
            skip_marker_mtime is not None
            and out.exists()
            and out.stat().st_mtime >= skip_marker_mtime
        ):
            rendered += 1
            skipped += 1
            if rendered % 25 == 0:
                print(f"  ... {rendered}/{total} processed ({skipped} skipped)")
            return
        script_path.write_text(script, encoding="utf-8")
        started = time.time()
        # Keep the MCP JSON payload small; this MCP build can crash parsing
        # large one-shot script strings.
        result = _send_mcp_script(
            args.mcp_port,
            f"exec(open({str(script_path)!r}, 'r', encoding='utf-8').read())",
            timeout_s=args.mcp_timeout_s,
        )
        if result.get("status") != "success" or not result.get("result", {}).get("success", True):
            raise SystemExit(json.dumps(result, indent=2))
        _wait_for_output(Path(out_path), started, timeout_s=args.screenshot_timeout_s)
        rendered += 1
        if rendered % 25 == 0:
            if skipped:
                print(f"  ... {rendered}/{total} processed ({skipped} skipped)")
            else:
                print(f"  ... {rendered}/{total} rendered")

    for v in view_jobs:
        do_clear = bool(args.clear_existing_scene and not scene_reset_done)
        if do_clear:
            scene_reset_done = True
        wlj = _to_waypoint_light_job(v, map_name)
        if v.render_kind == "traffic_light":
            kwargs = dict(light_kwargs, clear_existing_scene=do_clear)
            script = _wp_ue_script(light=v.light_node or {}, job=wlj,
                                   camera_mode=args.light_camera_mode, **kwargs)
        else:
            kwargs = dict(plain_kwargs, clear_existing_scene=do_clear)
            script = _wp_ue_script(light={}, job=wlj, camera_mode="waypoint", **kwargs)
        _send(script, v.image_path)

    for o in obstacle_jobs:
        do_clear = bool(args.clear_existing_scene and not scene_reset_done)
        if do_clear:
            scene_reset_done = True
        kwargs = dict(cone_kwargs, clear_existing_scene=do_clear)
        script = _cone_ue_script(job=o, spawn_cone=True, out_path=o.blocked_path, **kwargs)
        _send(script, o.blocked_path)

    _write_manifest()
    _write_waypoints()
    suffix = f", skipped {skipped} fresh existing" if skipped else ""
    print(f"rendered {rendered - skipped} images{suffix} -> {out_root} "
          f"(manifest.jsonl + obstacles.json + waypoints.json written)")


if __name__ == "__main__":
    main()
