"""Render UE dock-obstacle FPV variants (road blocks the agent must BYPASS).

This is the obstacle counterpart of ``render_waypoint_pedestrian_light_ue.py``.
It samples a fixed fraction of **dock** waypoints (seeded, so the set is
reproducible and shared with the runtime), places a road block ~5 m in front of
each sampled dock along the edge toward its **next dock** (i.e. down the street,
not across toward the intersection), and renders the **front** view twice from
the identical camera:

  * ``images/<dock_id>/yaw_<NNN>_blocked.png`` — with the cone spawned
  * ``images/<dock_id>/yaw_<NNN>.png``         — after the cone is deleted (clear)

The obstacle is **one-sided** (only ``dock -> road`` is blocked; the reverse is
clear) and matches the runtime ``ObstacleField`` schema, so the same
``obstacles.json`` this script writes is what the env loads. At runtime the
front FPV shows the cone, ``MOVE("forward")`` into it is rejected
(``collision_count += 1``) and the agent must ``BYPASS()`` to pass. See
``OBSTACLE_TRAFFIC_DESIGN.md`` and ``deliverybench_fpv/FPV_CAPTURE_PLAN.md``.

The cone asset defaults to ``/Game/CityDatabase/blueprints/RoadCone.RoadCone_C``.

Example (smoke test, no UE server — just write the sidecar + job list):
    python -m vagen.envs.deliverybench.tools.render_dock_obstacle_ue \
        vagen/envs/deliverybench/maps/small-city-11 --dry-run

Example (real render against a SimWorld Studio MCP server):
    python -m vagen.envs.deliverybench.tools.render_dock_obstacle_ue \
        vagen/envs/deliverybench/maps/small-city-11 --mcp-port 55565 \
        --spawn-map-assets
"""

from __future__ import annotations

import os
import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from vagen.envs.deliverybench.tools.render_pedestrian_light_ue import (
    _send_mcp_script,
    _wait_for_output,
)
from vagen.envs.deliverybench.tools.render_waypoint_pedestrian_light_ue import (
    DIR_TO_VEC,
    _load_map,
    _yaw_for_abs_direction,
)

# Runtime FPV yaw convention: the env fetches the stored yaw for a compass
# travel direction D as ``(FPV_YAW_OFFSET_DEG - D) % 360`` (see
# deliverybench_env._build_fpv_cross). We must record the *same* stored yaw so
# the obstacle image is fetched when the agent stands at the dock facing the
# road.
FPV_YAW_OFFSET_DEG = 90.0

DEFAULT_CONE_ASSET = "/Game/CityDatabase/blueprints/RoadCone.RoadCone_C"
DEFAULT_UE_ASSETS_JSON = os.environ.get("UE_ASSETS_JSON", "simworld/data/ue_assets.json")
DEFAULT_OBSTACLE_DIST_CM = 300.0   # 3 m ahead of the dock, along the road edge
DEFAULT_CONE_SCALE = 5.0           # actor scale for the road cone (bigger = clearer)
DEFAULT_SAMPLE_FRAC = 0.05         # 5% of dock waypoints
DEFAULT_SEED = 5
DEFAULT_CAMERA_Z = 170.0           # matches the proven waypoint-view camera
DEFAULT_CAMERA_BACKOFF_CM = 900.0  # 9 m behind the dock, so the edge ahead is visible
DEFAULT_CONE_Z = 0.0               # cone on the ground
DEFAULT_IMG_W = 1280
DEFAULT_IMG_H = 720

_CARDINALS = (("north", 0.0), ("east", 90.0), ("south", 180.0), ("west", 270.0))


def _abs_dir_for_bearing(bearing_deg: float) -> str:
    """Snap a compass bearing (0=N,90=E,180=S,270=W) to the nearest cardinal name."""
    b = float(bearing_deg) % 360.0
    best, best_d = "north", 999.0
    for name, deg in _CARDINALS:
        d = min(abs(b - deg), 360.0 - abs(b - deg))
        if d < best_d:
            best, best_d = name, d
    return best


def _stored_yaw_for_bearing(bearing_deg: float) -> int:
    """The runtime FPV lookup yaw for facing ``bearing_deg`` (compass)."""
    return int(round((FPV_YAW_OFFSET_DEG - float(bearing_deg)) % 360.0))


@dataclass
class ObstacleJob:
    map_name: str
    dock_id: str
    dock_name: str
    dock_x_cm: float            # runtime/sim position (FPV join key)
    dock_y_cm: float
    dock_render_x: float        # UE-scale camera position
    dock_render_y: float
    dst_id: str                 # the next-dock neighbour the block sits in front of
    dst_x_cm: float
    dst_y_cm: float
    bearing_deg: float          # compass bearing dock -> next dock
    abs_direction: str          # cardinal name of that edge
    stored_yaw: int             # FPV lookup yaw (front view facing the next dock)
    camera_yaw_deg: float       # UE camera yaw
    cone_x: float               # UE-scale cone position (5 m ahead)
    cone_y: float
    cone_z: float
    obstacle_type: str
    blocked_path: str
    clear_path: str


def _next_dock_neighbour(city_map: Any, dock_node: Any) -> Optional[Dict[str, Any]]:
    """The next **dock** down the street (nearest MOVE-legal dock-kind neighbour).

    The block is placed on the dock -> next-dock edge so the front view shows it
    blocking travel along the street (not the crossing toward the intersection).
    Docks with no dock neighbour are skipped.
    """
    adj = city_map.adjacents(dock_node) or []
    best = None
    best_d = 1e18
    for a in adj:
        if not a.get("legal_move", True):
            continue
        if str(a.get("kind", "")).lower() != "dock":
            continue
        d = float(a.get("dist_m", 0.0))
        if d < best_d:
            best, best_d = a, d
    return best


def _dock_nodes(city_map: Any) -> List[Tuple[str, Any]]:
    """(waypoint_id, node) for every dock waypoint, sorted by id for determinism."""
    out: List[Tuple[str, Any]] = []
    by_id = getattr(city_map, "waypoints_by_id", {}) or {}
    for wp_id, node in by_id.items():
        if str(getattr(node, "waypoint_kind", "")).lower() == "dock":
            out.append((str(wp_id), node))
    out.sort(key=lambda t: t[0])
    return out


def build_jobs(
    scenario_dir: Path,
    out_root: Path,
    *,
    sample_frac: float = DEFAULT_SAMPLE_FRAC,
    seed: int = DEFAULT_SEED,
    obstacle_dist_cm: float = DEFAULT_OBSTACLE_DIST_CM,
    cone_z: float = DEFAULT_CONE_Z,
    obstacle_type: str = "road_block",
    max_docks: Optional[int] = None,
) -> List[ObstacleJob]:
    """Sample docks (seeded) and build one obstacle job per sampled dock.

    The sampling is a deterministic function of ``seed`` over the sorted dock id
    list, so the env and the renderer agree on which docks are blocked.
    """
    city_map = _load_map(scenario_dir)
    map_name = scenario_dir.name

    docks = _dock_nodes(city_map)
    # Only docks that actually have a next-dock neighbour can host a block.
    eligible: List[Tuple[str, Any, Dict[str, Any]]] = []
    for wp_id, node in docks:
        nb = _next_dock_neighbour(city_map, node)
        if nb is not None:
            eligible.append((wp_id, node, nb))

    rng = random.Random(seed)
    k = max(1, int(round(len(eligible) * float(sample_frac)))) if eligible else 0
    k = min(k, len(eligible))
    sampled = sorted(rng.sample(eligible, k), key=lambda t: t[0]) if k else []
    if max_docks is not None:
        sampled = sampled[: max_docks]

    jobs: List[ObstacleJob] = []
    for wp_id, node, nb in sampled:
        dock_x, dock_y = float(node.position.x), float(node.position.y)
        nb_node = nb["node"]
        dst_x, dst_y = float(nb_node.position.x), float(nb_node.position.y)
        bearing = float(nb.get("bearing_deg", 0.0))
        abs_dir = _abs_dir_for_bearing(bearing)
        vx, vy = DIR_TO_VEC[abs_dir]
        # UE world coords == runtime/sim node position (cm) for these maps — the
        # world is spawned from progen (m -> cm) and the FPV join keys on the same
        # cm positions. Do NOT look up by waypoint_id: capture ids have drifted
        # from the runtime graph, so an id-join fetches the wrong dock.
        rx, ry = dock_x, dock_y
        stored_yaw = _stored_yaw_for_bearing(bearing)
        img_dir = out_root / "images" / f"{wp_id.rsplit('_', 1)[0]}_{int(wp_id.rsplit('_', 1)[1]):03d}"
        jobs.append(
            ObstacleJob(
                map_name=map_name,
                dock_id=wp_id,
                dock_name=str(getattr(node, "waypoint_name", "")),
                dock_x_cm=round(dock_x, 1),
                dock_y_cm=round(dock_y, 1),
                dock_render_x=float(rx),
                dock_render_y=float(ry),
                dst_id=str(nb.get("id", "")),
                dst_x_cm=round(dst_x, 1),
                dst_y_cm=round(dst_y, 1),
                bearing_deg=round(bearing, 1),
                abs_direction=abs_dir,
                stored_yaw=stored_yaw,
                camera_yaw_deg=round(_yaw_for_abs_direction(abs_dir), 1),
                cone_x=round(float(rx) + vx * obstacle_dist_cm, 1),
                cone_y=round(float(ry) + vy * obstacle_dist_cm, 1),
                cone_z=float(cone_z),
                obstacle_type=obstacle_type,
                blocked_path=str(img_dir / f"yaw_{stored_yaw:03d}_blocked.png"),
                clear_path=str(img_dir / f"yaw_{stored_yaw:03d}.png"),
            )
        )
    return jobs


def _ue_script(
    *,
    job: ObstacleJob,
    cone_asset: str,
    spawn_cone: bool,
    out_path: str,
    camera_z: float,
    camera_backoff_cm: float,
    camera_fov: float,
    image_width: int,
    image_height: int,
    spawn_map_assets: bool,
    clear_existing_scene: bool,
    world_json_path: str,
    ue_assets_json_path: str,
    cone_scale: float = DEFAULT_CONE_SCALE,
) -> str:
    job_json = json.dumps(asdict(job))
    return f"""
import json
import math
import unreal

JOB = json.loads({job_json!r})
CONE_ASSET = {cone_asset!r}
SPAWN_CONE = {bool(spawn_cone)!r}
CONE_SCALE = float({cone_scale!r})
OUT = {out_path!r}
CAMERA_Z = float({camera_z!r})
CAMERA_BACKOFF_CM = float({camera_backoff_cm!r})
CAMERA_FOV = float({camera_fov!r})
IMAGE_WIDTH = int({image_width!r})
IMAGE_HEIGHT = int({image_height!r})
SPAWN_MAP_ASSETS = {bool(spawn_map_assets)!r}
CLEAR_EXISTING_SCENE = {bool(clear_existing_scene)!r}
WORLD_JSON_PATH = {world_json_path!r}
UE_ASSETS_JSON_PATH = {ue_assets_json_path!r}

DIR_TO_VEC = {{"north": (0.0, 1.0), "east": (1.0, 0.0), "south": (0.0, -1.0), "west": (-1.0, 0.0)}}


def log(msg):
    print("VAGEN_DOCK_OBSTACLE_UE " + str(msg))


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
        # Strip a trailing ".Name_C" generated-class suffix and retry as blueprint.
        try:
            base = asset_path.split(".")[0]
            cls = unreal.EditorAssetLibrary.load_blueprint_class(base)
        except Exception:
            cls = None
    if cls is None:
        try:
            asset = unreal.load_asset(asset_path)
            cls = getattr(asset, "generated_class", None)
        except Exception:
            cls = None
    return cls


def load_json_file(path, default):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return default


def clean_obstacles():
    subsys = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    deleted = 0
    for actor in list(subsys.get_all_level_actors()):
        try:
            if actor.get_actor_label().startswith("VAGEN_Obstacle_"):
                subsys.destroy_actor(actor)
                deleted += 1
        except Exception:
            pass
    log("cleaned_obstacles=%d" % deleted)


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


def map_asset_exists(prefix):
    subsys = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    for actor in list(subsys.get_all_level_actors()):
        try:
            if actor.get_actor_label().startswith(prefix):
                return True
        except Exception:
            pass
    return False


def is_pedestrian_light_node(node):
    props = node.get("properties", {{}}) or {{}}
    kind = str(props.get("poi_type") or props.get("type") or "").lower()
    inst = str(node.get("instance_name") or "").lower()
    return kind in ("pedestrian_light", "traffic_light") or "street_light_ped" in inst


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
    for node in world.get("nodes", []) or []:
        if is_pedestrian_light_node(node):
            continue
        asset_path = asset_path_for_node(node, assets)
        if not asset_path:
            continue
        cls = load_blueprint_class_any(asset_path)
        if cls is None:
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
                float(scale.get("x", 1.0)), float(scale.get("y", 1.0)), float(scale.get("z", 1.0)),
            ))
            spawned += 1
        except Exception as exc:
            log("map_asset_spawn_failed id=%s err=%s" % (node.get("id"), exc))
    log("map_assets_spawned=%d" % spawned)


def spawn_cone():
    cls = load_blueprint_class_any(CONE_ASSET)
    if cls is None:
        raise RuntimeError("could not load road-block blueprint: " + str(CONE_ASSET))
    direction = str(JOB.get("abs_direction") or "").lower()
    # BP_RoadBlocker's striped face points opposite the old RoadCone-facing
    # convention. Keep the striped side visible to the agent/camera.
    yaw_by_dir = {{"north": 180.0, "south": 0.0, "west": 270.0, "east": 90.0}}
    yaw = yaw_by_dir.get(direction, 0.0)
    actor = unreal.EditorLevelLibrary.spawn_actor_from_class(
        cls,
        unreal.Vector(float(JOB["cone_x"]), float(JOB["cone_y"]), float(JOB["cone_z"])),
        unreal.Rotator(pitch=0.0, yaw=yaw, roll=0.0),
    )
    actor.set_actor_label("VAGEN_Obstacle_" + str(JOB.get("dock_id", "dock")))
    if CONE_SCALE != 1.0:
        actor.set_actor_scale3d(unreal.Vector(CONE_SCALE, CONE_SCALE, CONE_SCALE))
    loc = actor.get_actor_location()
    log("spawned_cone=%s loc=(%.1f,%.1f,%.1f) yaw=%.1f scale=%.1f" % (
        actor.get_actor_label(), loc.x, loc.y, loc.z, yaw, CONE_SCALE))
    return actor


clear_existing_scene()
clean_obstacles()
ensure_map_assets()
if SPAWN_CONE:
    spawn_cone()

view_dx, view_dy = DIR_TO_VEC[str(JOB["abs_direction"]).lower()]
camera_loc = unreal.Vector(
    float(JOB["dock_render_x"]) - float(view_dx) * CAMERA_BACKOFF_CM,
    float(JOB["dock_render_y"]) - float(view_dy) * CAMERA_BACKOFF_CM,
    CAMERA_Z,
)
camera_rot = unreal.Rotator(pitch=0.0, yaw=float(JOB["camera_yaw_deg"]), roll=0.0)
subsys = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)
subsys.set_level_viewport_camera_info(camera_loc, camera_rot)
try:
    unreal.SystemLibrary.execute_console_command(None, "fov %.1f" % CAMERA_FOV)
except Exception:
    pass
log("camera loc=(%.1f,%.1f,%.1f) yaw=%.1f spawn_cone=%s out=%s" % (
    camera_loc.x, camera_loc.y, camera_loc.z, camera_rot.yaw, SPAWN_CONE, OUT,
))
task = unreal.AutomationLibrary.take_high_res_screenshot(IMAGE_WIDTH, IMAGE_HEIGHT, OUT)
log("HIGHRES_CALLED %s" % task)
"""


def _manifest_rows(job: ObstacleJob) -> List[Dict[str, Any]]:
    """Two FPV manifest rows for a dock: the blocked variant and the clear twin.

    Keyed by the runtime join fields (x_cm/y_cm/yaw). The blocked row carries
    ``render_kind="obstacle"`` so the env loads it into the obstacle lookup; the
    clear row is a plain capture that refreshes the dock's front view.
    """
    common = {
        "map_name": job.map_name,
        "waypoint_id": job.dock_id,
        "waypoint_kind": "dock",
        "waypoint_name": job.dock_name,
        "x_cm": job.dock_x_cm,
        "y_cm": job.dock_y_cm,
        "yaw": float(job.stored_yaw),
        "z_cm": DEFAULT_CAMERA_Z,
        "camera_id": 0,
        "capture_mode": "camera",
        "edge_dst_id": job.dst_id,
        "edge_dst_x_cm": job.dst_x_cm,
        "edge_dst_y_cm": job.dst_y_cm,
        "bearing_deg": job.bearing_deg,
        "status": "ok",
        "error": None,
    }
    blocked = dict(common)
    blocked.update({
        "render_kind": "obstacle",
        "obstacle_type": job.obstacle_type,
        "obstacle_state": "blocked",
        "image_path": job.blocked_path,
    })
    clear = dict(common)
    clear.update({
        "render_kind": "plain",
        "obstacle_state": "clear",
        "image_path": job.clear_path,
    })
    return [blocked, clear]


def _obstacles_sidecar(jobs: List[ObstacleJob], *, map_name: str, seed: int,
                       sample_frac: float, obstacle_dist_cm: float) -> Dict[str, Any]:
    """The ObstacleField sidecar the runtime loads (directed dock -> road edges)."""
    return {
        "map_name": map_name,
        "seed": seed,
        "sample_frac": sample_frac,
        "obstacle_dist_cm": obstacle_dist_cm,
        "obstacles": [
            {
                "src_id": j.dock_id,
                "src_x_cm": j.dock_x_cm,
                "src_y_cm": j.dock_y_cm,
                "dst_id": j.dst_id,
                "dst_x_cm": j.dst_x_cm,
                "dst_y_cm": j.dst_y_cm,
                "bearing_deg": j.bearing_deg,
                "type": j.obstacle_type,
            }
            for j in jobs
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario_dir", type=Path, help="map dir (has progen_world_enriched.json, roads.json)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="FPV output dir (default: deliverybench_fpv/<map>-obstacles)")
    parser.add_argument("--mcp-port", type=int, default=55565)
    parser.add_argument("--sample-frac", type=float, default=DEFAULT_SAMPLE_FRAC)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--obstacle-dist-cm", type=float, default=DEFAULT_OBSTACLE_DIST_CM)
    parser.add_argument("--obstacle-type", default="road_block", choices=["road_block", "slow_pedestrian"])
    parser.add_argument("--cone-asset", default=DEFAULT_CONE_ASSET)
    parser.add_argument("--cone-scale", type=float, default=DEFAULT_CONE_SCALE,
                        help="actor scale for the road cone (default 5x)")
    parser.add_argument("--camera-z", type=float, default=DEFAULT_CAMERA_Z)
    parser.add_argument("--camera-backoff-cm", type=float, default=DEFAULT_CAMERA_BACKOFF_CM)
    parser.add_argument("--cone-z", type=float, default=DEFAULT_CONE_Z)
    parser.add_argument("--camera-fov", type=float, default=90.0)
    parser.add_argument("--image-width", type=int, default=DEFAULT_IMG_W)
    parser.add_argument("--image-height", type=int, default=DEFAULT_IMG_H)
    parser.add_argument("--spawn-map-assets", action="store_true",
                        help="spawn generated world roads/buildings before rendering")
    parser.add_argument("--world-json", type=Path, default=None)
    parser.add_argument("--ue-assets-json", type=Path, default=Path(DEFAULT_UE_ASSETS_JSON))
    parser.add_argument("--max-docks", type=int, default=None)
    parser.add_argument("--screenshot-timeout-s", type=float, default=60.0)
    parser.add_argument("--dry-run", action="store_true",
                        help="build jobs + write obstacles.json/manifest without contacting UE")
    args = parser.parse_args()

    scenario_dir = args.scenario_dir.resolve()
    map_name = scenario_dir.name
    out_root = (args.out_dir or (scenario_dir.parent.parent / "deliverybench_fpv" / f"{map_name}-obstacles")).resolve()
    world_json = (args.world_json or (scenario_dir / "progen_world_enriched.json")).resolve()

    jobs = build_jobs(
        scenario_dir, out_root,
        sample_frac=args.sample_frac, seed=args.seed,
        obstacle_dist_cm=args.obstacle_dist_cm, cone_z=args.cone_z,
        obstacle_type=args.obstacle_type, max_docks=args.max_docks,
    )
    if not jobs:
        raise SystemExit(f"no eligible dock waypoints sampled in {scenario_dir}")

    out_root.mkdir(parents=True, exist_ok=True)
    # Single source of truth for the runtime ObstacleField + the FPV lookup.
    sidecar = _obstacles_sidecar(
        jobs, map_name=map_name, seed=args.seed,
        sample_frac=args.sample_frac, obstacle_dist_cm=args.obstacle_dist_cm,
    )
    (out_root / "obstacles.json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")

    manifest_rows: List[Dict[str, Any]] = []
    for j in jobs:
        manifest_rows.extend(_manifest_rows(j))

    print(f"[plan] map={map_name} sampled_docks={len(jobs)} "
          f"(seed={args.seed}, frac={args.sample_frac}) -> {out_root}")
    for j in jobs:
        print(f"  dock {j.dock_id} ({j.dock_name}) face={j.abs_direction} "
              f"yaw={j.stored_yaw} cone@({j.cone_x:.0f},{j.cone_y:.0f}) -> next dock {j.dst_id}")

    if args.dry_run:
        # Still write the manifest so the sidecar + intended renders are inspectable.
        with (out_root / "manifest.jsonl").open("w", encoding="utf-8") as fh:
            for row in manifest_rows:
                fh.write(json.dumps(row) + "\n")
        print(f"[dry-run] wrote obstacles.json + manifest.jsonl ({len(manifest_rows)} rows). "
              f"No UE contact.")
        return

    common_kwargs = dict(
        cone_asset=args.cone_asset, camera_z=args.camera_z,
        camera_backoff_cm=args.camera_backoff_cm, camera_fov=args.camera_fov,
        image_width=args.image_width, image_height=args.image_height,
        spawn_map_assets=args.spawn_map_assets,
        clear_existing_scene=False,
        world_json_path=str(world_json), ue_assets_json_path=str(args.ue_assets_json),
        cone_scale=args.cone_scale,
    )

    rendered = 0
    for j in jobs:
        Path(j.blocked_path).parent.mkdir(parents=True, exist_ok=True)
        # 1) blocked: spawn cone, shoot. 2) clear: delete cone (clean), shoot.
        for spawn_cone, out_path in ((True, j.blocked_path), (False, j.clear_path)):
            script = _ue_script(job=j, spawn_cone=spawn_cone, out_path=out_path, **common_kwargs)
            started_at = time.time()
            result = _send_mcp_script(args.mcp_port, script)
            if result.get("status") != "success" or not result.get("result", {}).get("success", True):
                raise SystemExit(json.dumps(result, indent=2))
            _wait_for_output(Path(out_path), started_at, timeout_s=args.screenshot_timeout_s)
            rendered += 1
            print(f"  rendered {'BLOCKED' if spawn_cone else 'clear  '} {j.dock_id}: {out_path}")

    with (out_root / "manifest.jsonl").open("w", encoding="utf-8") as fh:
        for row in manifest_rows:
            fh.write(json.dumps(row) + "\n")
    print(f"rendered {rendered} images for {len(jobs)} docks; "
          f"wrote obstacles.json + manifest.jsonl to {out_root}")


if __name__ == "__main__":
    main()
