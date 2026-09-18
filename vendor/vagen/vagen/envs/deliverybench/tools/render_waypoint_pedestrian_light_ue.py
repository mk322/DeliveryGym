"""Render UE waypoint-view pedestrian-light cache images.

This renders first-person waypoint camera views for pedestrian lights that are
visible before crossing a road.  Images are organized as:

    <map>/ue_pedestrian_light_waypoint_views/
      wp_<waypoint_id>/
        even/
          front|left|right|backward/*.png
        odd/
          front|left|right|backward/*.png

The ``even`` and ``odd`` folders correspond to the runtime traffic-light phase:
even minutes make north/south crossings green, odd minutes make east/west
crossings green.

Example small smoke test:
    python -m vagen.envs.deliverybench.tools.render_waypoint_pedestrian_light_ue \\
        vagen/envs/deliverybench/maps/small-city-11 --mcp-port 55560 \\
        --max-waypoints 2
"""

from __future__ import annotations

import os
import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from PIL import Image, ImageDraw

from vagen.envs.deliverybench.tools.render_pedestrian_light_ue import (
    _send_mcp_script,
    _wait_for_output,
)
from vagen.envs.deliverybench.tools.render_gmaps import (
    MARGIN_PX,
    OUT_LONG,
    OUT_MIN,
    STYLE,
    View,
    _draw_text_with_halo,
    _font,
    load_world,
)
from vagen.envs.deliverybench.vlm_delivery.map.map import Map


RELATIVE_DIRECTIONS = ("front", "left", "right", "backward")
PHASE_SECONDS = {"even": 0.0, "odd": 60.0}
ABS_DIRS = ("north", "east", "south", "west")
DEFAULT_AGENT_FACING = "north"
DEFAULT_UE_ASSETS_JSON = Path(os.environ.get("UE_ASSETS_JSON", "simworld/data/ue_assets.json"))
PED_LIGHT_RENDER_XY_SCALE = 10.0
DIR_TO_VEC = {
    "north": (0.0, 1.0),
    "east": (1.0, 0.0),
    "south": (0.0, -1.0),
    "west": (-1.0, 0.0),
}
LEFT_OF = {"north": "west", "west": "south", "south": "east", "east": "north"}
RIGHT_OF = {v: k for k, v in LEFT_OF.items()}
BACK_OF = {"north": "south", "south": "north", "east": "west", "west": "east"}
ABS_DIR_FROM_COMPASS = {
    "N": "north",
    "NE": "north",
    "E": "east",
    "SE": "south",
    "S": "south",
    "SW": "south",
    "W": "west",
    "NW": "north",
}


@dataclass
class WaypointLightJob:
    map_name: str
    source_waypoint_id: str
    source_waypoint_name: str
    source_x: float
    source_y: float
    source_render_x: float
    source_render_y: float
    target_waypoint_id: str
    target_waypoint_name: str
    target_x: float
    target_y: float
    target_render_x: float
    target_render_y: float
    movement_abs_direction: str
    camera_abs_direction: str
    camera_yaw_deg: float
    agent_facing_direction: str
    view_direction: str
    action_direction: str
    phase: str
    signal_state: str
    signal_axis: str
    light_id: str
    light_face: str
    light_face_direction: str
    render_kind: str
    output_path: str


def _slug(text: Any) -> str:
    out = []
    for ch in str(text):
        if ch.isalnum() or ch in {"-", "_"}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "unknown"


def _abs_direction_from_delta(dx: float, dy: float) -> str:
    if abs(dx) >= abs(dy):
        return "east" if dx >= 0 else "west"
    return "north" if dy >= 0 else "south"


def _abs_direction_from_compass(compass: Any, dx: float, dy: float) -> str:
    """Match MOVE(direction=...)'s coarse compass reduction in step_to.py."""

    coarse = ABS_DIR_FROM_COMPASS.get(str(compass or "").strip().upper())
    return coarse or _abs_direction_from_delta(dx, dy)


def _abs_direction_for_relative(facing: str, relative: str) -> str:
    facing = str(facing or DEFAULT_AGENT_FACING).strip().lower()
    if facing not in ABS_DIRS:
        facing = DEFAULT_AGENT_FACING
    if relative in {"front", "forward"}:
        return facing
    if relative == "left":
        return LEFT_OF[facing]
    if relative == "right":
        return RIGHT_OF[facing]
    if relative == "backward":
        return BACK_OF[facing]
    raise ValueError(f"unknown relative direction {relative!r}")


def _relative_direction_for_abs(facing: str, absolute: str) -> str:
    for relative in RELATIVE_DIRECTIONS:
        if _abs_direction_for_relative(facing, relative) == absolute:
            return relative
    raise ValueError(f"{absolute!r} is not reachable from facing {facing!r}")


def _yaw_for_abs_direction(direction: str) -> float:
    vx, vy = DIR_TO_VEC[str(direction).lower()]
    return math.degrees(math.atan2(vy, vx))


def _dist_point_to_segment(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> float:
    vx, vy = bx - ax, by - ay
    den = vx * vx + vy * vy
    if den <= 1e-9:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * vx + (py - ay) * vy) / den
    t = max(0.0, min(1.0, t))
    qx, qy = ax + vx * t, ay + vy * t
    return math.hypot(px - qx, py - qy)


def _select_visible_light_for_edge(
    lights: Iterable[Mapping[str, Any]],
    *,
    source_x: float,
    source_y: float,
    target_x: float,
    target_y: float,
    movement_abs_direction: str,
    fallback_light_id: str,
) -> Tuple[str, str, str]:
    """Pick the destination-side pole face for a crossing edge.

    Runtime movement only needs to know whether the edge is red/green, so the
    map-level check can use the nearest pole to the crossing.  Rendering should
    show the signal across the road near the next waypoint.  Since the signal
    is on the far sidewalk side, the visible face points back toward the source:
    a northbound crossing should pick a south-facing face near the north target
    waypoint.
    """

    move_dir = str(movement_abs_direction).lower()
    fallback_face_direction = BACK_OF.get(move_dir, move_dir)
    best: Optional[Tuple[float, float, str, str, str]] = None
    for light in lights or []:
        props = light.get("properties", {}) or {}
        faces = props.get("faces", {}) or {}
        try:
            lx, ly = float(light["x"]), float(light["y"])
        except Exception:
            loc = props.get("location", {}) or {}
            try:
                lx, ly = float(loc["x"]), float(loc["y"])
            except Exception:
                continue
        for face_name, face in faces.items():
            crosswalk_dir = str(face.get("crosswalk_side_direction") or "").lower()
            facing_dir = str(face.get("facing_direction") or "").lower()
            if crosswalk_dir == move_dir:
                priority_penalty = 0.0
            elif facing_dir == fallback_face_direction:
                priority_penalty = 5000.0
            else:
                continue
            target_dist = math.hypot(lx - target_x, ly - target_y)
            segment_dist = _dist_point_to_segment(
                lx,
                ly,
                source_x,
                source_y,
                target_x,
                target_y,
            )
            score = target_dist + segment_dist * 0.05 + priority_penalty
            light_id = str(light.get("id") or fallback_light_id)
            selected_face_direction = facing_dir or fallback_face_direction
            candidate = (score, target_dist, light_id, str(face_name), selected_face_direction)
            if best is None or candidate < best:
                best = candidate

    if best is None:
        return str(fallback_light_id or "unknown_light"), "", fallback_face_direction
    return best[2], best[3], best[4]


def _state_for_phase(axis: str, phase: str) -> str:
    is_ns = str(axis).lower() in {"south-north", "north-south", "vertical", "sn", "ns"}
    if phase == "even":
        return "green" if is_ns else "red"
    if phase == "odd":
        return "red" if is_ns else "green"
    raise ValueError(f"unknown phase {phase!r}")


def _load_map(scenario_dir: Path) -> Map:
    config_path = Path(__file__).resolve().parents[1] / "vlm_delivery" / "input" / "game_mechanics_config.json"
    try:
        cfg = json.loads(config_path.read_text(encoding="utf-8")).get("map", {})
    except Exception:
        cfg = {}
    m = Map(cfg)
    m.import_roads(str(scenario_dir / "roads.json"))
    m.import_pois(str(scenario_dir / "progen_world_enriched.json"))
    return m


def _load_fpv_waypoint_coords(scenario_dir: Path) -> Dict[str, Tuple[float, float]]:
    """Load UE-scale waypoint coordinates from the existing FPV capture manifest."""

    fpv_manifest = scenario_dir.parent.parent / "deliverybench_fpv" / scenario_dir.name / "manifest.jsonl"
    coords: Dict[str, Tuple[float, float]] = {}
    if not fpv_manifest.exists():
        return coords
    with fpv_manifest.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("status") != "ok":
                continue
            waypoint_id = str(entry.get("waypoint_id") or "")
            if not waypoint_id or waypoint_id in coords:
                continue
            try:
                coords[waypoint_id] = (float(entry["x_cm"]), float(entry["y_cm"]))
            except Exception:
                continue
    return coords


def _render_xy(
    waypoint_id: str,
    x: float,
    y: float,
    fpv_coords: Mapping[str, Tuple[float, float]],
) -> Tuple[float, float]:
    return fpv_coords.get(str(waypoint_id), (float(x), float(y)))


def _build_jobs(
    scenario_dir: Path,
    *,
    max_waypoints: Optional[int],
    one_edge_per_waypoint: bool,
    agent_facings: Tuple[str, ...],
    waypoint_ids: Optional[Tuple[str, ...]],
    out_root: Path,
    include_normal_views: bool = False,
) -> List[WaypointLightJob]:
    m = _load_map(scenario_dir)
    fpv_coords = _load_fpv_waypoint_coords(scenario_dir)
    map_name = scenario_dir.name
    jobs: List[WaypointLightJob] = []
    seen_sources = set()

    waypoint_items = sorted(
        m.waypoints_by_id.items(),
        key=lambda item: item[0],
    )
    requested_waypoints = set(waypoint_ids or ())
    for source_id, source in waypoint_items:
        if requested_waypoints and source_id not in requested_waypoints:
            continue
        source_edges = []
        for adj in m.adjacents(source):
            # The runtime no longer attaches a light-state dict to edges (lights
            # are vision-only; state is TrafficController's job). A crossing face
            # is now any MOVE-legal edge that crosses a vehicle road; its axis is
            # derived from the travel direction.
            if not adj.get("crosses_vehicle_road"):
                continue
            target = adj.get("node")
            if target is None:
                continue
            source_edges.append((adj, target))
        if not source_edges:
            continue
        if max_waypoints is not None and len(seen_sources) >= max_waypoints:
            break
        seen_sources.add(source_id)
        if one_edge_per_waypoint:
            source_edges = source_edges[:1]

        for adj, target in source_edges:
            target_id = str(adj.get("id") or getattr(target, "waypoint_id", "unknown"))
            dx = float(target.position.x) - float(source.position.x)
            dy = float(target.position.y) - float(source.position.y)
            move_abs = _abs_direction_from_compass(adj.get("compass"), dx, dy)
            axis = "south-north" if move_abs in ("north", "south") else "east-west"
            source_rx, source_ry = _render_xy(source_id, source.position.x, source.position.y, fpv_coords)
            target_rx, target_ry = _render_xy(target_id, target.position.x, target.position.y, fpv_coords)
            light_id, light_face, light_face_direction = _select_visible_light_for_edge(
                getattr(m, "traffic_lights", []),
                source_x=float(source.position.x),
                source_y=float(source.position.y),
                target_x=float(target.position.x),
                target_y=float(target.position.y),
                movement_abs_direction=move_abs,
                fallback_light_id="unknown_light",
            )
            for agent_facing in agent_facings:
                rel = _relative_direction_for_abs(agent_facing, move_abs)
                camera_abs = _abs_direction_for_relative(agent_facing, rel)
                camera_yaw_deg = _yaw_for_abs_direction(camera_abs)
                for phase in ("even", "odd"):
                    state = _state_for_phase(axis, phase)
                    stem = (
                        f"wp_{_slug(source_id)}"
                        f"__to_{_slug(target_id)}"
                        f"__facing_{agent_facing}"
                        f"__view_{rel}"
                        f"__action_{'forward' if rel == 'front' else rel}"
                        f"__move_{move_abs}"
                        f"__axis_{_slug(axis)}"
                        f"__light_{_slug(light_id)}"
                        f"__face_{_slug(light_face or light_face_direction)}"
                        f"__state_{state}"
                    )
                    out_path = out_root / f"wp_{_slug(source_id)}" / phase / rel / f"{stem}.png"
                    jobs.append(
                        WaypointLightJob(
                            map_name=map_name,
                            source_waypoint_id=str(source_id),
                            source_waypoint_name=str(getattr(source, "waypoint_name", "")),
                            source_x=float(source.position.x),
                            source_y=float(source.position.y),
                            source_render_x=float(source_rx),
                            source_render_y=float(source_ry),
                            target_waypoint_id=target_id,
                            target_waypoint_name=str(getattr(target, "waypoint_name", "")),
                            target_x=float(target.position.x),
                            target_y=float(target.position.y),
                            target_render_x=float(target_rx),
                            target_render_y=float(target_ry),
                            movement_abs_direction=move_abs,
                            camera_abs_direction=camera_abs,
                            camera_yaw_deg=float(camera_yaw_deg),
                            agent_facing_direction=agent_facing,
                            view_direction=rel,
                            action_direction="forward" if rel == "front" else rel,
                            phase=phase,
                            signal_state=state,
                            signal_axis=axis,
                            light_id=light_id,
                            light_face=light_face,
                            light_face_direction=light_face_direction,
                            render_kind="traffic_light",
                            output_path=str(out_path),
                        )
                    )
        if include_normal_views:
            crossing_rels = {
                (job.agent_facing_direction, job.view_direction, job.phase)
                for job in jobs
                if job.source_waypoint_id == str(source_id)
            }
            source_rx, source_ry = _render_xy(source_id, source.position.x, source.position.y, fpv_coords)
            for agent_facing in agent_facings:
                for rel in RELATIVE_DIRECTIONS:
                    camera_abs = _abs_direction_for_relative(agent_facing, rel)
                    camera_yaw_deg = _yaw_for_abs_direction(camera_abs)
                    for phase in ("even", "odd"):
                        if (agent_facing, rel, phase) in crossing_rels:
                            continue
                        stem = (
                            f"wp_{_slug(source_id)}"
                            f"__facing_{agent_facing}"
                            f"__view_{rel}"
                            f"__normal_scene"
                        )
                        out_path = out_root / f"wp_{_slug(source_id)}" / phase / rel / f"{stem}.png"
                        jobs.append(
                            WaypointLightJob(
                                map_name=map_name,
                                source_waypoint_id=str(source_id),
                                source_waypoint_name=str(getattr(source, "waypoint_name", "")),
                                source_x=float(source.position.x),
                                source_y=float(source.position.y),
                                source_render_x=float(source_rx),
                                source_render_y=float(source_ry),
                                target_waypoint_id="",
                                target_waypoint_name="",
                                target_x=float(source.position.x),
                                target_y=float(source.position.y),
                                target_render_x=float(source_rx),
                                target_render_y=float(source_ry),
                                movement_abs_direction=camera_abs,
                                camera_abs_direction=camera_abs,
                                camera_yaw_deg=float(camera_yaw_deg),
                                agent_facing_direction=agent_facing,
                                view_direction=rel,
                                action_direction="forward" if rel == "front" else rel,
                                phase=phase,
                                signal_state="normal",
                                signal_axis="",
                                light_id="",
                                light_face="",
                                light_face_direction="",
                                render_kind="normal_scene",
                                output_path=str(out_path),
                            )
                        )
    return jobs


def _ue_script(
    *,
    light: Mapping[str, Any],
    job: WaypointLightJob,
    camera_mode: str,
    camera_z: float,
    camera_distance: float,
    face_camera_sign: float,
    camera_backoff_cm: float,
    camera_fov: float,
    image_width: int,
    image_height: int,
    world_json_path: Path,
    ue_assets_json_path: Optional[Path],
    spawn_map_assets: bool,
    clear_existing_scene: bool,
    debug_markers: bool,
    ped_light_xy_scale: float,
    light_asset_scale: float = 1.0,
) -> str:
    light_json = json.dumps(light)
    job_json = json.dumps(asdict(job))
    ue_assets_json = str(ue_assets_json_path) if ue_assets_json_path else ""
    return f"""
import json
import math
import unreal

LIGHT = json.loads({light_json!r})
JOB = json.loads({job_json!r})
WORLD_JSON_PATH = {str(world_json_path)!r}
UE_ASSETS_JSON_PATH = {ue_assets_json!r}
CAMERA_MODE = {camera_mode!r}
SPAWN_MAP_ASSETS = {bool(spawn_map_assets)!r}
CLEAR_EXISTING_SCENE = {bool(clear_existing_scene)!r}
DEBUG_MARKERS = {bool(debug_markers)!r}
CAMERA_Z = float({camera_z!r})
CAMERA_DISTANCE = float({camera_distance!r})
FACE_CAMERA_SIGN = float({face_camera_sign!r})
CAMERA_BACKOFF_CM = float({camera_backoff_cm!r})
CAMERA_FOV = float({camera_fov!r})
IMAGE_WIDTH = int({image_width!r})
IMAGE_HEIGHT = int({image_height!r})
PED_LIGHT_XY_SCALE = float({ped_light_xy_scale!r})
LIGHT_ASSET_SCALE = float({light_asset_scale!r})

DIR_TO_VEC = {{
    "north": (0.0, 1.0),
    "east": (1.0, 0.0),
    "south": (0.0, -1.0),
    "west": (-1.0, 0.0),
}}
BACK_OF = {{
    "north": "south",
    "east": "west",
    "south": "north",
    "west": "east",
}}


def log(msg):
    print("VAGEN_WP_PED_LIGHT_UE " + str(msg))


def phase_state_for_direction(direction):
    direction = str(direction or "").lower()
    phase = str(JOB["phase"]).lower()
    is_ns = direction in ("north", "south")
    if phase == "even":
        return "green" if is_ns else "red"
    return "red" if is_ns else "green"


def clean_generated():
    subsys = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    deleted = 0
    for actor in list(subsys.get_all_level_actors()):
        try:
            label = actor.get_actor_label()
            cls = actor.get_class().get_name()
            if (
                label.startswith("VAGEN_WaypointPedLight_")
                or label.startswith("VAGEN_WaypointMarker_")
                or label.startswith("VAGEN_WaypointStage_")
                or cls == "RT_BP_street_light_ped_C"
            ):
                subsys.destroy_actor(actor)
                deleted += 1
        except Exception:
            pass
    log("deleted_count=%d" % deleted)


def clear_existing_scene():
    if not CLEAR_EXISTING_SCENE:
        return
    subsys = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    keep_class_fragments = (
        "WorldSettings",
        "LevelScriptActor",
        "DirectionalLight",
        "SkyLight",
        "SkyAtmosphere",
        "ExponentialHeightFog",
        "AtmosphericFog",
        "VolumetricCloud",
        "PostProcessVolume",
        "CameraActor",
        "CineCameraActor",
        "PlayerStart",
    )
    keep_label_prefixes = (
        "Floor",
        "Road_",
        "RoadY_",
        "CrossPatch",
    )
    deleted = 0
    kept = 0
    for actor in list(subsys.get_all_level_actors()):
        try:
            label = actor.get_actor_label()
            cls = actor.get_class().get_name()
            if (
                any(fragment in cls for fragment in keep_class_fragments)
                or any(label.startswith(prefix) for prefix in keep_label_prefixes)
            ):
                kept += 1
                continue
            subsys.destroy_actor(actor)
            deleted += 1
        except Exception as exc:
            log("clear_scene_skip actor=%s error=%s" % (actor, exc))
    log("clear_existing_scene deleted=%d kept=%d" % (deleted, kept))


def load_json_file(path, default):
    if not path:
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        log("json_load_failed path=%s error=%s" % (path, exc))
        return default


def is_pedestrian_light_node(node):
    props = node.get("properties", {{}}) or {{}}
    return (
        props.get("poi_type") == "pedestrian_light"
        or props.get("type") == "pedestrian_light"
        or str(node.get("instance_name") or "").lower() == "rt_bp_street_light_ped"
    )


def load_blueprint_class_any(asset_path):
    if not asset_path:
        return None
    cls = None
    try:
        cls = unreal.EditorAssetLibrary.load_blueprint_class(asset_path)
    except Exception:
        cls = None
    if cls is None:
        try:
            cls = unreal.load_class(None, asset_path)
        except Exception:
            cls = None
    if cls is None:
        try:
            asset = unreal.load_asset(asset_path)
            cls = getattr(asset, "generated_class", None)
        except Exception:
            cls = None
    return cls


def map_asset_exists(prefix):
    subsys = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    for actor in list(subsys.get_all_level_actors()):
        try:
            if actor.get_actor_label().startswith(prefix):
                return True
        except Exception:
            pass
    return False


def asset_path_for_node(node, assets):
    props = node.get("properties", {{}}) or {{}}
    explicit = props.get("ue_asset_path")
    if explicit:
        return explicit
    inst = str(node.get("instance_name") or "")
    entry = assets.get(inst, {{}}) if isinstance(assets, dict) else {{}}
    if isinstance(entry, dict):
        return entry.get("asset_path") or entry.get("path")
    if isinstance(entry, str):
        return entry
    return ""


def ensure_map_assets():
    if not SPAWN_MAP_ASSETS:
        return
    prefix = "VAGEN_WorldAsset_%s_" % str(JOB.get("map_name", "map"))
    if map_asset_exists(prefix):
        log("map_assets_cached prefix=%s" % prefix)
        return
    world = load_json_file(WORLD_JSON_PATH, {{}})
    assets = load_json_file(UE_ASSETS_JSON_PATH, {{}})
    spawned = 0
    missing = 0
    failed = 0
    for node in world.get("nodes", []) or []:
        if is_pedestrian_light_node(node):
            continue
        asset_path = asset_path_for_node(node, assets)
        if not asset_path:
            missing += 1
            continue
        cls = load_blueprint_class_any(asset_path)
        if cls is None:
            missing += 1
            continue
        props = node.get("properties", {{}}) or {{}}
        loc = props.get("location", {{}}) or {{}}
        ori = props.get("orientation", {{}}) or {{}}
        scale = props.get("scale", {{}}) or {{}}
        try:
            actor = unreal.EditorLevelLibrary.spawn_actor_from_class(
                cls,
                unreal.Vector(float(loc.get("x", 0.0)), float(loc.get("y", 0.0)), float(loc.get("z", 0.0))),
                unreal.Rotator(
                    pitch=float(ori.get("pitch", 0.0)),
                    yaw=float(ori.get("yaw", 0.0)),
                    roll=float(ori.get("roll", 0.0)),
                ),
            )
            actor.set_actor_label(prefix + str(node.get("id") or spawned))
            actor.set_actor_scale3d(unreal.Vector(
                float(scale.get("x", 1.0)),
                float(scale.get("y", 1.0)),
                float(scale.get("z", 1.0)),
            ))
            spawned += 1
        except Exception as exc:
            failed += 1
            log("map_asset_spawn_failed id=%s asset=%s error=%s" % (node.get("id"), asset_path, exc))
    log("map_assets_spawned=%d missing=%d failed=%d" % (spawned, missing, failed))


def spawn_basic_mesh(label, mesh_name, loc, scale, color=None):
    mesh = unreal.load_asset("/Engine/BasicShapes/%s.%s" % (mesh_name, mesh_name))
    actor = unreal.EditorLevelLibrary.spawn_actor_from_class(
        unreal.StaticMeshActor,
        unreal.Vector(float(loc[0]), float(loc[1]), float(loc[2])),
        unreal.Rotator(pitch=0.0, yaw=0.0, roll=0.0),
    )
    actor.set_actor_label(label)
    actor.set_actor_scale3d(unreal.Vector(float(scale[0]), float(scale[1]), float(scale[2])))
    comp = actor.get_component_by_class(unreal.StaticMeshComponent)
    if comp and mesh:
        comp.set_static_mesh(mesh)
        if color:
            try:
                base_mat = unreal.load_asset("/Engine/BasicShapes/BasicShapeMaterial.BasicShapeMaterial")
                dyn_mat = None
                if base_mat and hasattr(unreal, "KismetMaterialLibrary"):
                    dyn_mat = unreal.KismetMaterialLibrary.create_dynamic_material_instance(actor, base_mat)
                if dyn_mat:
                    linear = unreal.LinearColor(
                        float(color[0]),
                        float(color[1]),
                        float(color[2]),
                        float(color[3]),
                    )
                    for param_name in ("Color", "BaseColor", "Base Color"):
                        try:
                            dyn_mat.set_vector_parameter_value(param_name, linear)
                        except Exception:
                            pass
                    comp.set_material(0, dyn_mat)
            except Exception as exc:
                log("marker_color_failed=%s" % exc)
    return actor


def spawn_context_stage():
    sx = float(JOB["source_x"])
    sy = float(JOB["source_y"])
    tx = float(JOB["target_x"])
    ty = float(JOB["target_y"])
    mx = (sx + tx) * 0.5
    my = (sy + ty) * 0.5
    dx = tx - sx
    dy = ty - sy
    horizontal = abs(dx) >= abs(dy)

    # Simple debug context inside UE: road band, walking path, source and target
    # waypoint markers.  These are deliberately labeled so clean_generated()
    # can remove them without touching the real scene.
    if horizontal:
        spawn_basic_mesh("VAGEN_WaypointStage_vehicle_road", "Cube", (mx, my, -5.0), (1.35, 5.0, 0.03), (0.12, 0.12, 0.12, 1.0))
        spawn_basic_mesh("VAGEN_WaypointStage_crosswalk_path", "Cube", (mx, my, 3.0), (max(abs(dx), 80.0) / 100.0, 0.16, 0.025), (1.0, 0.82, 0.16, 1.0))
    else:
        spawn_basic_mesh("VAGEN_WaypointStage_vehicle_road", "Cube", (mx, my, -5.0), (5.0, 1.35, 0.03), (0.12, 0.12, 0.12, 1.0))
        spawn_basic_mesh("VAGEN_WaypointStage_crosswalk_path", "Cube", (mx, my, 3.0), (0.16, max(abs(dy), 80.0) / 100.0, 0.025), (1.0, 0.82, 0.16, 1.0))

    spawn_basic_mesh(
        "VAGEN_WaypointMarker_source_" + str(JOB.get("source_waypoint_id", "source")),
        "Cylinder",
        (sx, sy, 92.0),
        (0.24, 0.24, 1.45),
        (0.05, 0.35, 1.0, 1.0),
    )
    spawn_basic_mesh(
        "VAGEN_WaypointMarker_target_" + str(JOB.get("target_waypoint_id", "target")),
        "Cylinder",
        (tx, ty, 135.0),
        (0.32, 0.32, 2.25),
        (0.0, 0.95, 0.25, 1.0),
    )
    spawn_basic_mesh(
        "VAGEN_WaypointMarker_target_head_" + str(JOB.get("target_waypoint_id", "target")),
        "Sphere",
        (tx, ty, 265.0),
        (0.62, 0.62, 0.62),
        (0.0, 0.95, 0.25, 1.0),
    )


def desired_for_component(component_name, states):
    name = component_name.lower()
    if "_l_" in name:
        return states.get("left", "red")
    if "_r_" in name:
        return states.get("right", "red")
    return "off"


def set_signal_components(signal, states, face_sides):
    target = None
    target_forward = None
    target_face = str(JOB.get("light_face") or "").lower()
    target_side = str(face_sides.get(target_face) or "").lower()
    target_state = str(JOB.get("signal_state") or "").lower()
    for comp in signal.get_components_by_class(unreal.StaticMeshComponent):
        name = comp.get_name().lower()
        if "crossing_light_stop" not in name and "crossing_light_walk" not in name:
            continue
        desired = desired_for_component(name, states)
        selected_component = False
        if target_side and target_side in name:
            selected_component = True
        elif target_face == "left" and "_l_" in name:
            selected_component = True
        elif target_face == "right" and "_r_" in name:
            selected_component = True
        if target_face and target_state in ("green", "red"):
            desired = target_state
            visible = ("stop" in name and target_state == "red") or ("walk" in name and target_state == "green")
        else:
            visible = ("stop" in name and desired == "red") or ("walk" in name and desired == "green")
        comp.set_visibility(visible, True)
        comp.set_hidden_in_game(not visible, True)
        if visible:
            loc = comp.get_world_location()
            if selected_component or (not target_face and target is None):
                target = loc
                try:
                    target_forward = comp.get_forward_vector()
                except Exception:
                    try:
                        target_forward = comp.get_component_rotation().get_forward_vector()
                    except Exception:
                        target_forward = None
    return target, target_forward


def look_at(camera, target):
    dx = target.x - camera.x
    dy = target.y - camera.y
    dz = target.z - camera.z
    return unreal.Rotator(
        pitch=math.degrees(math.atan2(dz, math.sqrt(dx * dx + dy * dy))),
        yaw=math.degrees(math.atan2(dy, dx)),
        roll=0.0,
    )


def spawn_light(light):
    if str(JOB.get("render_kind") or "") != "traffic_light" or not light.get("id"):
        return None, None, None
    props = light.get("properties", {{}})
    loc = props.get("location", {{}})
    ori = props.get("orientation", {{}})
    asset = props.get("ue_asset_path") or "/Game/RealTimeBench/Traffic/RT_BP_street_light_ped"
    ped_cls = load_blueprint_class_any(asset)
    if ped_cls is None:
        raise RuntimeError("could not load pedestrian-light Blueprint: " + asset)
    actor = unreal.EditorLevelLibrary.spawn_actor_from_class(
        ped_cls,
        unreal.Vector(
            float(loc.get("x", 0.0)) * PED_LIGHT_XY_SCALE,
            float(loc.get("y", 0.0)) * PED_LIGHT_XY_SCALE,
            float(loc.get("z", 0.0)),
        ),
        unreal.Rotator(
            pitch=float(ori.get("pitch", 0.0)),
            yaw=float(ori.get("yaw", 0.0)),
            roll=float(ori.get("roll", 0.0)),
        ),
    )
    actor.set_actor_label("VAGEN_WaypointPedLight_" + light.get("id", "unnamed"))
    if LIGHT_ASSET_SCALE != 1.0:
        actor.set_actor_scale3d(unreal.Vector(LIGHT_ASSET_SCALE, LIGHT_ASSET_SCALE, LIGHT_ASSET_SCALE))
    states = {{}}
    face_sides = {{}}
    for face_name, face in (props.get("faces", {{}}) or {{}}).items():
        states[face_name] = phase_state_for_direction(
            face.get("crosswalk_side_direction") or face.get("facing_direction", "")
        )
        face_sides[str(face_name).lower()] = str(face.get("component_side") or "").lower()
    target, target_forward = set_signal_components(actor, states, face_sides)
    if CAMERA_MODE == "waypoint" and target is not None and target_forward is not None:
        view_dx, view_dy = DIR_TO_VEC[str(JOB["camera_abs_direction"]).lower()]
        camera_x = float(JOB.get("source_render_x", JOB["source_x"])) - float(view_dx) * CAMERA_BACKOFF_CM
        camera_y = float(JOB.get("source_render_y", JOB["source_y"])) - float(view_dy) * CAMERA_BACKOFF_CM
        desired_yaw = math.degrees(math.atan2(camera_y - target.y, camera_x - target.x))
        current_yaw = math.degrees(math.atan2(float(target_forward.y), float(target_forward.x)))
        delta_yaw = (desired_yaw - current_yaw + 180.0) % 360.0 - 180.0
        rot = actor.get_actor_rotation()
        new_rot = unreal.Rotator(pitch=float(rot.pitch), yaw=float(rot.yaw) + float(delta_yaw), roll=float(rot.roll))
        try:
            actor.set_actor_rotation(new_rot, False)
        except TypeError:
            actor.set_actor_rotation(new_rot)
        target, target_forward = set_signal_components(actor, states, face_sides)
        log("rotated_light_for_waypoint_actual face=%s current_yaw=%.1f desired_yaw=%.1f delta=%.1f yaw=%.1f" % (
            JOB.get("light_face"), current_yaw, desired_yaw, delta_yaw, new_rot.yaw
        ))
    return actor, target, target_forward


def yaw_for_direction(direction):
    vx, vy = DIR_TO_VEC[str(direction).lower()]
    return math.degrees(math.atan2(vy, vx))


clear_existing_scene()
clean_generated()
ensure_map_assets()
if DEBUG_MARKERS:
    spawn_context_stage()
_actor, target, target_forward = spawn_light(LIGHT)

if CAMERA_MODE == "face":
    if target is None:
        raise RuntimeError("selected visible pedestrian-light face not found: %s %s" % (
            JOB.get("light_id"), JOB.get("light_face")
        ))
    face = (LIGHT.get("properties", {{}}).get("faces", {{}}) or {{}}).get(str(JOB.get("light_face") or ""), {{}})
    facing = face.get("facing", {{}})
    fx = float(facing.get("x", 0.0))
    fy = float(facing.get("y", 0.0))
    if target_forward is not None:
        fx = float(target_forward.x)
        fy = float(target_forward.y)
    camera_loc = unreal.Vector(
        target.x + fx * CAMERA_DISTANCE * FACE_CAMERA_SIGN,
        target.y + fy * CAMERA_DISTANCE * FACE_CAMERA_SIGN,
        CAMERA_Z,
    )
    camera_rot = look_at(camera_loc, target)
else:
    view_dx, view_dy = DIR_TO_VEC[str(JOB["camera_abs_direction"]).lower()]
    camera_loc = unreal.Vector(
        float(JOB.get("source_render_x", JOB["source_x"])) - float(view_dx) * CAMERA_BACKOFF_CM,
        float(JOB.get("source_render_y", JOB["source_y"])) - float(view_dy) * CAMERA_BACKOFF_CM,
        CAMERA_Z,
    )
    camera_rot = unreal.Rotator(
        pitch=0.0,
        yaw=float(JOB["camera_yaw_deg"]),
        roll=0.0,
    )
subsys = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)
subsys.set_level_viewport_camera_info(camera_loc, camera_rot)
try:
    unreal.SystemLibrary.execute_console_command(None, "fov %.1f" % CAMERA_FOV)
except Exception:
    pass
log("camera mode=%s loc=(%.1f,%.1f,%.1f) yaw=%.1f backoff=%.1f phase=%s state=%s out=%s" % (
    CAMERA_MODE,
    camera_loc.x, camera_loc.y, camera_loc.z, camera_rot.yaw,
    CAMERA_BACKOFF_CM, JOB["phase"], JOB["signal_state"], JOB["output_path"],
))
task = unreal.AutomationLibrary.take_high_res_screenshot(
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    JOB["output_path"],
)
log("HIGHRES_CALLED %s" % task)
"""


def _load_lights(scenario_dir: Path) -> List[Dict[str, Any]]:
    with (scenario_dir / "progen_world_enriched.json").open("r", encoding="utf-8") as f:
        world = json.load(f)
    return [
        node for node in world.get("nodes", [])
        if (node.get("properties") or {}).get("poi_type") == "pedestrian_light"
    ]


def _minimal_light(node: Mapping[str, Any]) -> Dict[str, Any]:
    props = dict(node.get("properties", {}) or {})
    return {
        "id": str(node.get("id") or ""),
        "properties": {
            "ue_asset_path": props.get("ue_asset_path"),
            "location": props.get("location", {}),
            "orientation": props.get("orientation", {}),
            "faces": props.get("faces", {}) or {},
        },
    }


def _node_xy(node: Mapping[str, Any]) -> Optional[Tuple[float, float]]:
    props = node.get("properties", {}) or {}
    loc = props.get("location", {}) or {}
    try:
        return float(loc["x"]), float(loc["y"])
    except Exception:
        return None


def _is_pedestrian_light_node(node: Mapping[str, Any]) -> bool:
    props = node.get("properties", {}) or {}
    return (
        props.get("poi_type") == "pedestrian_light"
        or props.get("type") == "pedestrian_light"
        or str(node.get("instance_name") or "").lower() == "rt_bp_street_light_ped"
    )


def _focused_map_assets_world(
    scenario_dir: Path,
    jobs: List[WaypointLightJob],
    lights: Iterable[Mapping[str, Any]],
    *,
    radius_cm: float,
    ped_light_xy_scale: float,
) -> Dict[str, Any]:
    """Return a progen-world-shaped JSON containing only assets near jobs.

    UE rendering often only needs context around one waypoint/crossing.  Loading
    every building in a city can spend minutes compiling assets before the first
    screenshot.  The focused file keeps the same node schema and asset paths but
    filters non-light map instances to a local neighborhood.
    """

    with (scenario_dir / "progen_world_enriched.json").open("r", encoding="utf-8") as f:
        world = json.load(f)

    light_ids = {job.light_id for job in jobs if job.light_id}
    anchor_points: List[Tuple[float, float]] = []
    for job in jobs:
        anchor_points.append((float(job.source_render_x), float(job.source_render_y)))
        if job.target_waypoint_id:
            anchor_points.append((float(job.target_render_x), float(job.target_render_y)))
    for light in lights:
        if str(light.get("id") or "") not in light_ids:
            continue
        xy = _node_xy(light)
        if xy is None:
            continue
        anchor_points.append((xy[0] * ped_light_xy_scale, xy[1] * ped_light_xy_scale))

    if not anchor_points:
        return {"nodes": []}

    selected_nodes: List[Mapping[str, Any]] = []
    for node in world.get("nodes", []) or []:
        if _is_pedestrian_light_node(node):
            continue
        xy = _node_xy(node)
        if xy is None:
            continue
        if min(math.hypot(xy[0] - ax, xy[1] - ay) for ax, ay in anchor_points) <= radius_cm:
            selected_nodes.append(node)

    return {
        "nodes": selected_nodes,
        "focused_asset_source": str(scenario_dir / "progen_world_enriched.json"),
        "focused_asset_radius_cm": float(radius_cm),
    }


def _write_manifest(out_root: Path, jobs: List[WaypointLightJob]) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "description": "UE waypoint-view pedestrian-light render cache",
        "layout": "wp_<waypoint_id>/<even|odd>/<front|left|right|backward>/*.png",
        "job_count": len(jobs),
        "jobs": [asdict(job) for job in jobs],
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _light_xy(light: Mapping[str, Any]) -> Tuple[float, float]:
    try:
        return float(light["x"]), float(light["y"])
    except Exception:
        loc = (light.get("properties", {}) or {}).get("location", {}) or {}
        return float(loc.get("x", 0.0)), float(loc.get("y", 0.0))


def _make_focused_locator_view(
    source: Any,
    jobs: List[WaypointLightJob],
    lights_by_id: Mapping[str, Mapping[str, Any]],
) -> View:
    points: List[Tuple[float, float]] = [
        (float(source.position.x), float(source.position.y)),
    ]
    for job in jobs:
        points.append((float(job.target_x), float(job.target_y)))
        light = lights_by_id.get(job.light_id)
        if light is not None:
            lx, ly = _light_xy(light)
            points.append((lx, ly))
            vx, vy = DIR_TO_VEC.get(job.light_face_direction, (0.0, 0.0))
            points.append((lx + vx * 900.0, ly + vy * 900.0))
        vx, vy = DIR_TO_VEC.get(job.camera_abs_direction, (0.0, 0.0))
        points.append((float(source.position.x) + vx * 1200.0, float(source.position.y) + vy * 1200.0))

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    span = max(xmax - xmin, ymax - ymin, 2600.0)
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    pad = span * 0.22
    half = span / 2.0 + pad
    return View(cx - half, cx + half, cy - half, cy + half, 1600, 1200, 70)


def _draw_locator_star(draw: ImageDraw.ImageDraw, cx: float, cy: float) -> None:
    outer = 12.0
    inner = 5.5
    points = []
    for i in range(10):
        radius = outer if i % 2 == 0 else inner
        angle = -math.pi / 2.0 + i * math.pi / 5.0
        points.append((cx + math.cos(angle) * radius, cy + math.sin(angle) * radius))
    draw.polygon(points, fill="#8e24aa", outline="white")


def _draw_locator_arrow(
    draw: ImageDraw.ImageDraw,
    start: Tuple[float, float],
    end: Tuple[float, float],
    *,
    fill: str,
    width: int,
) -> None:
    sx, sy = start
    ex, ey = end
    draw.line([start, end], fill=fill, width=width)
    dx, dy = ex - sx, ey - sy
    length = math.hypot(dx, dy)
    if length <= 1e-6:
        return
    ux, uy = dx / length, dy / length
    lx, ly = -uy, ux
    head_len = 18.0
    head_w = 10.0
    draw.polygon(
        [
            (ex, ey),
            (ex - ux * head_len + lx * head_w, ey - uy * head_len + ly * head_w),
            (ex - ux * head_len - lx * head_w, ey - uy * head_len - ly * head_w),
        ],
        fill=fill,
    )


def _draw_world_arrow(
    draw: ImageDraw.ImageDraw,
    view: View,
    start_x: float,
    start_y: float,
    direction: str,
    *,
    length_cm: float,
    fill: str,
    width: int,
) -> None:
    vx, vy = DIR_TO_VEC.get(str(direction).lower(), (0.0, 0.0))
    start = view.to_px(start_x, start_y)
    end = view.to_px(start_x + vx * length_cm, start_y + vy * length_cm)
    _draw_locator_arrow(draw, start, end, fill=fill, width=width)


def _draw_view_cone(
    draw: ImageDraw.ImageDraw,
    view: View,
    origin_x: float,
    origin_y: float,
    direction: str,
    *,
    length_cm: float = 1200.0,
    fov_deg: float = 62.0,
) -> None:
    vx, vy = DIR_TO_VEC.get(str(direction).lower(), (0.0, 0.0))
    if vx == 0.0 and vy == 0.0:
        return
    theta = math.atan2(vy, vx)
    half = math.radians(fov_deg / 2.0)
    p0 = view.to_px(origin_x, origin_y)
    p1 = view.to_px(
        origin_x + math.cos(theta - half) * length_cm,
        origin_y + math.sin(theta - half) * length_cm,
    )
    p2 = view.to_px(
        origin_x + math.cos(theta + half) * length_cm,
        origin_y + math.sin(theta + half) * length_cm,
    )
    draw.polygon([p0, p1, p2], fill=(26, 115, 232, 42), outline=(26, 115, 232, 150))
    _draw_world_arrow(
        draw,
        view,
        origin_x,
        origin_y,
        direction,
        length_cm=length_cm * 0.82,
        fill="#1a73e8",
        width=4,
    )


def _edge_key(job: WaypointLightJob) -> Tuple[str, str, str, str]:
    return (
        job.source_waypoint_id,
        job.target_waypoint_id,
        job.light_id,
        job.light_face,
    )


def _unique_crossing_jobs(jobs: List[WaypointLightJob]) -> List[WaypointLightJob]:
    unique: Dict[Tuple[str, str, str, str], WaypointLightJob] = {}
    for job in jobs:
        unique.setdefault(_edge_key(job), job)
    return list(unique.values())


def _draw_focused_locator_map(
    scenario_dir: Path,
    city_map: Map,
    source_id: str,
    source: Any,
    jobs: List[WaypointLightJob],
    out_path: Path,
    *,
    title_suffix: str,
) -> None:
    lights_by_id = {str(light.get("id")): light for light in city_map.traffic_lights}
    unique_jobs = _unique_crossing_jobs(jobs)
    view = _make_focused_locator_view(source, unique_jobs, lights_by_id)
    label_font = _font(22)
    small_font = _font(17)
    title_font = _font(26)
    img = Image.new("RGBA", (view.out_w, view.out_h), STYLE["city_bg"])
    draw = ImageDraw.Draw(img)

    def in_view(x: float, y: float, margin_cm: float = 1800.0) -> bool:
        return (
            view.xmin - margin_cm <= x <= view.xmax + margin_cm
            and view.ymin - margin_cm <= y <= view.ymax + margin_cm
        )

    # Local road skeleton.
    for edge in city_map.graph_skel.edges:
        a = edge.node1.position
        b = edge.node2.position
        if not (in_view(float(a.x), float(a.y)) or in_view(float(b.x), float(b.y))):
            continue
        meta = city_map.graph_skel.get_edge_meta(edge.node1, edge.node2) or {}
        width = 7 if meta.get("kind") == "road" else 3
        fill = "#f4f4ef" if meta.get("kind") == "road" else "#cfe3cf"
        outline = "#b8bdb9" if meta.get("kind") == "road" else "#8fb98f"
        pa = view.to_px(float(a.x), float(a.y))
        pb = view.to_px(float(b.x), float(b.y))
        draw.line([pa, pb], fill=outline, width=width + 3)
        draw.line([pa, pb], fill=fill, width=width)

    target_ids = {job.target_waypoint_id for job in unique_jobs}

    # Context waypoint edges.
    selected_edges = {tuple(sorted((source_id, job.target_waypoint_id))) for job in unique_jobs}
    for edge in city_map.waypoint_graph.edges:
        u = edge.node1
        v = edge.node2
        uid = city_map._waypoint_id_by_node.get(u, "")
        vid = city_map._waypoint_id_by_node.get(v, "")
        if not (
            in_view(float(u.position.x), float(u.position.y))
            or in_view(float(v.position.x), float(v.position.y))
        ):
            continue
        key = tuple(sorted((uid, vid)))
        color = "#ff8c00" if key in selected_edges else "#7ea07e"
        width = 7 if key in selected_edges else 3
        draw.line(
            [
                view.to_px(float(u.position.x), float(u.position.y)),
                view.to_px(float(v.position.x), float(v.position.y)),
            ],
            fill=color,
            width=width,
        )

    # Agent camera/view cones.
    drawn_dirs = set()
    for job in unique_jobs:
        key = (job.camera_abs_direction, job.view_direction)
        if key in drawn_dirs:
            continue
        drawn_dirs.add(key)
        _draw_view_cone(
            draw,
            view,
            float(source.position.x),
            float(source.position.y),
            job.camera_abs_direction,
        )

    # Source and target/pedestrian waiting points.
    sx, sy = view.to_px(float(source.position.x), float(source.position.y))
    draw.ellipse([sx - 18, sy - 18, sx + 18, sy + 18], fill="white")
    draw.ellipse([sx - 14, sy - 14, sx + 14, sy + 14], fill="#1a73e8", outline="#0b3d75")
    _draw_text_with_halo(
        draw,
        (sx + 17, sy - 22),
        f"agent {source_id}",
        label_font,
        "#0b3d75",
        "#ffffff",
        anchor="la",
    )

    for job in unique_jobs:
        tx, ty = view.to_px(job.target_x, job.target_y)
        draw.ellipse([tx - 16, ty - 16, tx + 16, ty + 16], fill="white")
        draw.ellipse([tx - 12, ty - 12, tx + 12, ty + 12], fill="#ff8c00", outline="#8a4b00")
        _draw_text_with_halo(
            draw,
            (tx + 14, ty + 8),
            f"ped/target {job.target_waypoint_id}",
            small_font,
            "#7a3e00",
            "#ffffff",
            anchor="la",
        )
        _draw_locator_arrow(
            draw,
            view.to_px(float(source.position.x), float(source.position.y)),
            view.to_px(job.target_x, job.target_y),
            fill="#ff8c00",
            width=3,
        )

    # Destination-side pedestrian lights and face direction arrows.
    for job in unique_jobs:
        light = lights_by_id.get(job.light_id)
        if light is None:
            continue
        lx, ly = _light_xy(light)
        px, py = view.to_px(lx, ly)
        _draw_locator_star(draw, px, py)
        _draw_world_arrow(
            draw,
            view,
            lx,
            ly,
            job.light_face_direction,
            length_cm=850.0,
            fill="#8e24aa",
            width=4,
        )
        _draw_text_with_halo(
            draw,
            (px + 12, py - 28),
            f"light {job.light_id.split('_crossing_')[-1]}\nface {job.light_face}/{job.light_face_direction}",
            small_font,
            "#6a1b9a",
            "#ffffff",
            anchor="la",
        )

    _draw_text_with_halo(
        draw,
        (55, 30),
        f"{scenario_dir.name} {source_id} {title_suffix}",
        title_font,
        "#202124",
        "#ffffff",
        anchor="la",
    )
    _draw_text_with_halo(
        draw,
        (55, 64),
        "blue cone = waypoint camera/view; orange = crossing move; purple arrow = light face",
        small_font,
        "#202124",
        "#ffffff",
        anchor="la",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(out_path)


def _write_waypoint_locator_maps(
    scenario_dir: Path,
    out_root: Path,
    jobs: List[WaypointLightJob],
) -> None:
    """Write focused locator maps in each ``wp_<id>`` render folder."""

    if not jobs:
        return
    city_map = _load_map(scenario_dir)

    jobs_by_source: Dict[str, List[WaypointLightJob]] = {}
    for job in jobs:
        jobs_by_source.setdefault(job.source_waypoint_id, []).append(job)

    for source_id, source_jobs in jobs_by_source.items():
        source = city_map.waypoints_by_id.get(source_id)
        if source is None:
            continue
        wp_dir = out_root / f"wp_{_slug(source_id)}"
        _draw_focused_locator_map(
            scenario_dir,
            city_map,
            source_id,
            source,
            source_jobs,
            wp_dir / "waypoint_2d_map.png",
            title_suffix="crossing overview",
        )
        jobs_by_target: Dict[str, List[WaypointLightJob]] = {}
        for job in source_jobs:
            jobs_by_target.setdefault(job.target_waypoint_id, []).append(job)
        for target_id, target_jobs in jobs_by_target.items():
            _draw_focused_locator_map(
                scenario_dir,
                city_map,
                source_id,
                source,
                target_jobs,
                wp_dir / f"crossing_2d_map__to_{_slug(target_id)}.png",
                title_suffix=f"to {target_id}",
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario_dir", type=Path)
    parser.add_argument("--mcp-port", type=int, default=55560)
    parser.add_argument("--out-root", type=Path)
    parser.add_argument("--max-waypoints", type=int, default=2)
    parser.add_argument(
        "--waypoint-id",
        action="append",
        default=[],
        help="Specific source waypoint id to render. Can be passed multiple times.",
    )
    parser.add_argument("--one-edge-per-waypoint", action="store_true")
    parser.add_argument(
        "--include-normal-views",
        action="store_true",
        help="Also render normal rollout views for directions without a traffic-light-checked move.",
    )
    parser.add_argument(
        "--agent-facing",
        default=DEFAULT_AGENT_FACING,
        choices=("north", "east", "south", "west", "all"),
        help=(
            "Agent facing used to convert crossing edges into front/left/right/backward "
            "views. Use 'all' only when building a facing-aware cache."
        ),
    )
    parser.add_argument(
        "--phase",
        choices=("even", "odd", "both"),
        default="both",
        help="Limit rendered light states. Use 'even' for the north/south-green phase.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--camera-mode",
        choices=("waypoint", "face"),
        default="waypoint",
        help=(
            "'waypoint' uses the VAGEN relative-view yaw from the waypoint; "
            "'face' renders a close pedestrian-facing signal view for the same "
            "waypoint/direction job."
        ),
    )
    parser.add_argument("--camera-z", type=float, default=170.0)
    parser.add_argument("--camera-distance", type=float, default=450.0)
    parser.add_argument(
        "--face-camera-sign",
        type=float,
        default=1.0,
        help=(
            "Multiplier for face-mode camera placement along the selected UE "
            "component forward vector. Use -1 if a particular asset exposes "
            "the back side as its forward vector."
        ),
    )
    parser.add_argument(
        "--camera-backoff-cm",
        type=float,
        default=900.0,
        help=(
            "Move the camera this far opposite the crossing direction before "
            "capturing the waypoint view, so source-side pedestrian-light faces "
            "are visible before entering the crosswalk."
        ),
    )
    parser.add_argument("--camera-fov", type=float, default=90.0)
    parser.add_argument("--image-width", type=int, default=1280)
    parser.add_argument("--image-height", type=int, default=720)
    parser.add_argument(
        "--spawn-map-assets",
        action="store_true",
        help="Spawn actual map instances from progen_world_enriched.json before rendering.",
    )
    parser.add_argument(
        "--focus-map-assets",
        action="store_true",
        help=(
            "When --spawn-map-assets is set, spawn only actual map JSON assets near "
            "the selected waypoint/light jobs instead of the full city."
        ),
    )
    parser.add_argument(
        "--focus-map-asset-radius-cm",
        type=float,
        default=10000.0,
        help="Neighborhood radius for --focus-map-assets.",
    )
    parser.add_argument(
        "--ue-assets-json",
        type=Path,
        default=DEFAULT_UE_ASSETS_JSON,
        help="Path to the instance_name -> UE asset_path catalog used by map JSON assets.",
    )
    parser.add_argument(
        "--debug-markers",
        action="store_true",
        help="Add temporary waypoint/crossing debug markers in the UE view.",
    )
    parser.add_argument(
        "--ped-light-xy-scale",
        type=float,
        default=PED_LIGHT_RENDER_XY_SCALE,
        help="Render-time x/y scale for generated pedestrian lights to match the UE city scale.",
    )
    parser.add_argument(
        "--light-asset-scale",
        type=float,
        default=1.0,
        help="Actor scale for the spawned pedestrian-light asset "
             "(ue_render_setup_record.md whole-dataset value: 2.0).",
    )
    parser.add_argument("--timeout-s", type=float, default=45.0)
    parser.add_argument(
        "--skip-waypoint-maps",
        action="store_true",
        help="Do not write wp_<id>/waypoint_2d_map.png locator maps.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Write the manifest and focused 2D maps, then stop before UE rendering.",
    )
    args = parser.parse_args()

    scenario_dir = args.scenario_dir.resolve()
    out_root = (
        args.out_root
        or scenario_dir / "ue_pedestrian_light_waypoint_views"
    ).resolve()
    agent_facings = ABS_DIRS if args.agent_facing == "all" else (str(args.agent_facing),)
    jobs = _build_jobs(
        scenario_dir,
        max_waypoints=args.max_waypoints,
        one_edge_per_waypoint=bool(args.one_edge_per_waypoint),
        agent_facings=agent_facings,
        waypoint_ids=tuple(args.waypoint_id or ()),
        out_root=out_root,
        include_normal_views=bool(args.include_normal_views),
    )
    if args.phase != "both":
        jobs = [job for job in jobs if job.phase == args.phase]
    print(f"built {len(jobs)} waypoint pedestrian-light render jobs")
    if args.dry_run:
        print("dry run: manifest/images were not written")
        for job in jobs[:12]:
            print(json.dumps(asdict(job), indent=2))
        return

    _write_manifest(out_root, jobs)
    print(f"wrote manifest: {out_root / 'manifest.json'}")
    if not args.skip_waypoint_maps:
        _write_waypoint_locator_maps(scenario_dir, out_root, jobs)
        print("wrote per-waypoint 2D locator maps")
    if args.prepare_only:
        print("prepare only: UE images were not rendered")
        return

    lights = _load_lights(scenario_dir)
    if not lights:
        raise SystemExit(f"no pedestrian lights found in {scenario_dir / 'progen_world_enriched.json'}")
    lights_by_id = {str(light.get("id")): _minimal_light(light) for light in lights}
    script_path = out_root / "_last_waypoint_pedestrian_light_ue_job.py"
    world_json_path = scenario_dir / "progen_world_enriched.json"
    if args.spawn_map_assets and args.focus_map_assets:
        focused_world = _focused_map_assets_world(
            scenario_dir,
            jobs,
            lights,
            radius_cm=float(args.focus_map_asset_radius_cm),
            ped_light_xy_scale=float(args.ped_light_xy_scale),
        )
        world_json_path = out_root / "_focused_map_assets_world.json"
        world_json_path.write_text(json.dumps(focused_world, indent=2) + "\n", encoding="utf-8")
        print(
            "wrote focused map-asset world: "
            f"{world_json_path} ({len(focused_world.get('nodes', []))} nodes)"
        )

    rendered: List[str] = []
    for job in jobs:
        if job.render_kind == "traffic_light":
            selected_light = lights_by_id.get(job.light_id)
        else:
            selected_light = {"id": "", "properties": {}}
        if selected_light is None:
            raise SystemExit(f"light {job.light_id!r} not found in {scenario_dir / 'progen_world_enriched.json'}")
        out_path = Path(job.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.is_absolute() and out_path.parts[:2] == ("/", "tmp"):
            # The live SimWorld UE session may run under a different Unix user
            # from this Python process.  Keep /tmp render targets writable so
            # AutomationLibrary can save the screenshots there.
            out_path.parent.chmod(0o777)
        script = _ue_script(
            light=selected_light,
            job=job,
            camera_mode=args.camera_mode,
            camera_z=args.camera_z,
            camera_distance=args.camera_distance,
            face_camera_sign=args.face_camera_sign,
            camera_backoff_cm=args.camera_backoff_cm,
            camera_fov=args.camera_fov,
            image_width=args.image_width,
            image_height=args.image_height,
            world_json_path=world_json_path,
            ue_assets_json_path=args.ue_assets_json if args.ue_assets_json.exists() else None,
            spawn_map_assets=bool(args.spawn_map_assets),
            clear_existing_scene=False,
            debug_markers=bool(args.debug_markers),
            ped_light_xy_scale=float(args.ped_light_xy_scale),
            light_asset_scale=float(args.light_asset_scale),
        )
        script_path.write_text(script, encoding="utf-8")
        started_at = time.time()
        # The current MCP plugin can crash when a single incoming JSON command
        # exceeds one socket read chunk. Keep the command tiny and let UE read
        # the real Python from disk.
        result = _send_mcp_script(
            args.mcp_port,
            f"exec(open({str(script_path)!r}, 'r', encoding='utf-8').read())",
        )
        status = result.get("status")
        if status != "success" or not result.get("result", {}).get("success", False):
            if "timed out" not in str(result.get("error", "")).lower():
                raise SystemExit(json.dumps(result, indent=2))
            print(f"MCP timeout reported; waiting for screenshot anyway: {out_path}")
        _wait_for_output(out_path, started_at, timeout_s=args.timeout_s)
        rendered.append(str(out_path))
        print(f"rendered {len(rendered)}/{len(jobs)}: {out_path}")

    print(f"rendered {len(rendered)} waypoint pedestrian-light UE images")


if __name__ == "__main__":
    main()
