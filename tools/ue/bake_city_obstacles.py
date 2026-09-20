"""Bake the obstacle albums for a procgen city: every site, both ways, both kinds.

Site selection, camera placement and album layout are the Paris obstacle
baker's, unchanged: ``obstacle_sites`` is a property of the road graph (no
seed in the hash), each site is photographed from both directions in both
kinds, and the pavement edition moves the camera and nothing else -- the
19.5%-of-frames viewpoint leak that motivated it is documented over in
tools/ue/bake_obstacles.py. What changes here is the world: the city is
spawned (shared SPAWN_PREAMBLE -- demo_2, floor-keeping clear, registry
wait), the props are SimWorld's own street furniture rather than CityCore
Paris models, and the renderer is the tick-loop editor drive every other
album in this family uses instead of a live MCP session.

    python tools/ue/bake_city_obstacles.py --map small-city-11 --render --gpu 5
    python tools/ue/bake_city_obstacles.py --map small-city-11 --render --viewpoint pavement --gpu 5
    python tools/ue/bake_city_obstacles.py --map small-city-11 --manifest [--viewpoint pavement]
"""

from __future__ import annotations

import os
import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from embodiedbench.compiler.road_network import bearing_deg, build_road_network  # noqa: E402
from embodiedbench.runtime.city.obstacles import OBSTACLE_TYPES, obstacle_sites  # noqa: E402
from tools.ue.bake_city_streets import (  # noqa: E402
    CAMERA_Z_CM, HEIGHT, MAPS_DIR, PROJECT, SPAWN_PREAMBLE, UE_ASSETS, WIDTH,
    launch_editor,
)

# Same standoff geometry as the Paris bake, same reasons (near enough to be
# unmistakable, far enough to read as "down the street").
STANDOFF_CM = 650.0
MIN_STANDOFF_CM = 380.0
EDGE_FRACTION = 0.45
KERB_MARGIN_CM = 60.0

# SimWorld street furniture standing in for the CityCore props. A
# ``road_block`` spans the carriageway: four barrier panels with traffic
# drums at the ends, visibly no way past. A ``slow_pedestrian`` crowds the
# right-hand footway: an advertising stand, a bin, boxes and bags -- passable,
# not quickly. Distinguishable at a glance and from a distance, which is the
# discrimination the benchmark asks for.
P = "/Game/city_props/Assets/props"
LAYOUTS: dict[str, list[dict]] = {
    "road_block": [
        # Kerb to kerb, double depth, with a cone picket in front: from the
        # camera's standoff this must read as "the street is CLOSED", not as
        # street furniture that happens to be in the way.
        {"asset": f"{P}/road_blocker/SM_road_blocker_b", "along": 0.0, "across": -270.0, "yaw": 90.0, "scale": 1.0},
        {"asset": f"{P}/road_blocker/SM_road_blocker_b", "along": 0.0, "across": -180.0, "yaw": 90.0, "scale": 1.0},
        {"asset": f"{P}/road_blocker/SM_road_blocker_b", "along": 0.0, "across": -90.0, "yaw": 90.0, "scale": 1.0},
        {"asset": f"{P}/road_blocker/SM_road_blocker_b", "along": 0.0, "across": 0.0, "yaw": 90.0, "scale": 1.0},
        {"asset": f"{P}/road_blocker/SM_road_blocker_b", "along": 0.0, "across": 90.0, "yaw": 90.0, "scale": 1.0},
        {"asset": f"{P}/road_blocker/SM_road_blocker_b", "along": 0.0, "across": 180.0, "yaw": 90.0, "scale": 1.0},
        {"asset": f"{P}/road_blocker/SM_road_blocker_b", "along": 0.0, "across": 270.0, "yaw": 90.0, "scale": 1.0},
        {"asset": f"{P}/road_blocker/SM_road_blocker_b", "along": 60.0, "across": -135.0, "yaw": 90.0, "scale": 1.0},
        {"asset": f"{P}/road_blocker/SM_road_blocker_b", "along": 60.0, "across": 45.0, "yaw": 90.0, "scale": 1.0},
        {"asset": f"{P}/road_blocker/SM_road_blocker_b", "along": 60.0, "across": 225.0, "yaw": 90.0, "scale": 1.0},
        {"asset": f"{P}/traffic_drum/SM_traffic_drum", "along": 0.0, "across": -330.0, "yaw": 0.0, "scale": 1.0},
        {"asset": f"{P}/traffic_drum/SM_traffic_drum", "along": 0.0, "across": 330.0, "yaw": 0.0, "scale": 1.0},
        {"asset": f"{P}/road_cone/SM_road_cone", "along": -110.0, "across": -200.0, "yaw": 0.0, "scale": 1.0},
        {"asset": f"{P}/road_cone/SM_road_cone", "along": -110.0, "across": -65.0, "yaw": 0.0, "scale": 1.0},
        {"asset": f"{P}/road_cone/SM_road_cone", "along": -110.0, "across": 65.0, "yaw": 0.0, "scale": 1.0},
        {"asset": f"{P}/road_cone/SM_road_cone", "along": -110.0, "across": 200.0, "yaw": 0.0, "scale": 1.0},
    ],
    "slow_pedestrian": [
        {"asset": f"{P}/advertising/SM_advert_small_a", "along": 0.0, "across": 360.0, "yaw": 90.0, "scale": 1.0},
        {"asset": f"{P}/trash_bin/SM_trash_bin_a", "along": 60.0, "across": 330.0, "yaw": 20.0, "scale": 1.0},
        {"asset": f"{P}/carton_boxes/SM_carton_box_a", "along": 120.0, "across": 350.0, "yaw": 40.0, "scale": 1.0},
        {"asset": f"{P}/carton_boxes/SM_carton_box_b", "along": 140.0, "across": 400.0, "yaw": 70.0, "scale": 1.0},
        {"asset": f"{P}/garbage_bag/SM_garbage_bag_a", "along": 200.0, "across": 340.0, "yaw": 0.0, "scale": 1.0},
        {"asset": f"{P}/garbage_bag/SM_garbage_bag_b", "along": 220.0, "across": 385.0, "yaw": 55.0, "scale": 1.0},
    ],
}


def props_for(kind: str, origin: tuple[float, float], bearing: float) -> list[dict]:
    """One layout in world space. Right is screen-right (UE yaw is left-handed
    with Z up), the same quarter-turn the Paris baker settled on after putting
    a whole cluster on the wrong pavement."""
    radians = math.radians(bearing)
    forward = (math.cos(radians), math.sin(radians))
    right = (-math.sin(radians), math.cos(radians))
    out = []
    for item in LAYOUTS[kind]:
        along, across = float(item["along"]), float(item["across"])
        out.append({
            "asset": item["asset"],
            "x": origin[0] + forward[0] * along + right[0] * across,
            "y": origin[1] + forward[1] * along + right[1] * across,
            "z": 0.0,
            "yaw": bearing + float(item["yaw"]),
            "scale": float(item["scale"]),
        })
    return out


def plan(map_name: str, out: Path, viewpoint: str) -> list[dict]:
    """Identical shape to the Paris obstacle plan, per job and per field."""
    network = build_road_network(MAPS_DIR / map_name, map_name=map_name)
    neighbours = {n: sorted(node.neighbours) for n, node in network.nodes.items()}
    jobs = []
    for a, b in obstacle_sites(neighbours, map_name):
        for src, dst in ((a, b), (b, a)):
            here = network.nodes[src].position
            there = network.nodes[dst].position
            length = math.dist(here, there)
            standoff = max(MIN_STANDOFF_CM, min(STANDOFF_CM, length * EDGE_FRACTION))
            bearing = bearing_deg(here, there)
            radians = math.radians(bearing)
            origin = (here[0] + math.cos(radians) * standoff,
                      here[1] + math.sin(radians) * standoff)
            camera = here
            if viewpoint == "pavement":
                street = network.streets[network.nodes[src].street_index]
                offset = street.width_cm / 2.0 + KERB_MARGIN_CM
                right = math.radians(bearing + 90.0)
                camera = (here[0] + offset * math.cos(right),
                          here[1] + offset * math.sin(right))
            for kind in OBSTACLE_TYPES:
                # Aim at the obstacle, not down the street. With the camera
                # on the kerb (pavement view) or the cluster on the footway
                # (slow_pedestrian), a street-axis yaw put the subject at the
                # edge of the frame; murray wants it dead centre. Camera
                # POSITION is unchanged, so the viewpoint semantics hold.
                props = props_for(kind, origin, bearing)
                cxm = sum(pr["x"] for pr in props) / len(props)
                cym = sum(pr["y"] for pr in props) / len(props)
                aim = bearing_deg(camera, (cxm, cym))
                jobs.append({
                    "site": f"{a}|{b}", "node": src, "toward": dst, "type": kind,
                    "viewpoint": viewpoint,
                    "yaw": round(aim, 1),
                    "x_cm": round(camera[0], 2), "y_cm": round(camera[1], 2),
                    "z_cm": CAMERA_Z_CM, "standoff_cm": round(standoff, 1),
                    "edge_length_m": round(length / 100.0, 1),
                    "origin": [round(origin[0], 1), round(origin[1], 1)],
                    "street_bearing": round(bearing, 1),
                    "props": props,
                    "image_path": str(out / "images" / src / f"toward_{dst}_{kind}.png"),
                })
    return jobs


OBSTACLE_TAIL = r'''
SHOTS = json.load(open({shots_json!r}))
W, H = {w}, {h}
WARMUP_S = {warmup!r}

def clear_props():
    subsys = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    for actor in subsys.get_all_level_actors():
        try:
            if actor and actor.actor_has_tag("EB_OBSTACLE"):
                subsys.destroy_actor(actor)
        except Exception:
            pass

def place_props(props):
    subsys = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    placed = 0
    for item in props:
        asset = unreal.load_asset(item["asset"])
        if asset is None:
            note("prop missing: %s" % item["asset"])
            continue
        actor = subsys.spawn_actor_from_object(
            asset,
            unreal.Vector(item["x"], item["y"], item["z"]),
            unreal.Rotator(pitch=0.0, yaw=item["yaw"], roll=0.0))
        if actor is None:
            continue
        actor.tags = ["EB_OBSTACLE"]
        actor.set_actor_scale3d(
            unreal.Vector(item["scale"], item["scale"], item["scale"]))
        placed += 1
    return placed

STATE = {{"i": 0, "pending": None, "waited": 0, "done": 0, "failed": 0,
          "busy": False, "warmup_start": None}}

def _tick(delta):
    if STATE["busy"]:
        return
    STATE["busy"] = True
    try:
        _tick_body(delta)
    finally:
        STATE["busy"] = False

def _tick_body(delta):
    if WARMUP_S > 0:
        if STATE["warmup_start"] is None:
            STATE["warmup_start"] = time.time()
            note("warmup %ds" % WARMUP_S)
        if time.time() - STATE["warmup_start"] < WARMUP_S:
            try:
                INVALIDATE()
            except Exception:
                pass
            return
    if STATE["pending"] is not None:
        path = STATE["pending"]
        if os.path.exists(path) and os.path.getsize(path) > 0:
            STATE["pending"], STATE["waited"] = None, 0
            STATE["done"] += 1
            if STATE["done"] % 25 == 0:
                note("done=%d failed=%d of %d"
                     % (STATE["done"], STATE["failed"], len(SHOTS)))
        else:
            try:
                INVALIDATE()
            except Exception:
                pass
            STATE["waited"] += 1
            if STATE["waited"] > 900:
                note("timeout %s" % path)
                STATE["pending"], STATE["waited"] = None, 0
                STATE["failed"] += 1
        return
    while STATE["i"] < len(SHOTS):
        shot = SHOTS[STATE["i"]]
        STATE["i"] += 1
        out = shot["image_path"]
        if os.path.exists(out) and os.path.getsize(out) > 0:
            STATE["done"] += 1
            continue
        os.makedirs(os.path.dirname(out), exist_ok=True)
        clear_props()
        place_props(shot["props"])
        try:
            unreal.get_editor_subsystem(
                unreal.UnrealEditorSubsystem).set_level_viewport_camera_info(
                unreal.Vector(shot["x_cm"], shot["y_cm"], shot["z_cm"]),
                unreal.Rotator(pitch=0.0, yaw=shot["yaw"], roll=0.0))
            unreal.AutomationLibrary.take_high_res_screenshot(W, H, out)
        except Exception as error:
            STATE["failed"] += 1
            note("ERROR on %s: %r" % (out, error))
        STATE["pending"] = out
        return
    clear_props()
    note("FINISHED done=%d failed=%d" % (STATE["done"], STATE["failed"]))
    LOG.close()
    unreal.SystemLibrary.quit_editor()

note("starting: %d shots" % len(SHOTS))
unreal.register_slate_post_tick_callback(_tick)
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--map", required=True)
    parser.add_argument("--gpu", default="5")
    parser.add_argument("--viewpoint", default="carriageway",
                        choices=("carriageway", "pavement"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--manifest", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=240)
    args = parser.parse_args()

    # The runtime's album discovery walks these exact sibling names.
    stem = ("paris_obstacles_pavement" if args.viewpoint == "pavement"
            else "paris_obstacles")
    out = args.out or Path(os.environ.get("ALBUMS_DIR", "/data/albums")) / stem / args.map
    out.mkdir(parents=True, exist_ok=True)
    jobs = plan(args.map, out, args.viewpoint)
    print(f"planned {len(jobs)} frames ({args.viewpoint})")

    if args.render:
        shots = jobs[:args.limit] if args.limit else jobs
        shots_json = out / ("shots_trial.json" if args.limit else "shots.json")
        shots_json.write_text(json.dumps(shots, indent=1))
        tag = f"{args.map}_{args.viewpoint}"
        script = PROJECT / "Saved" / f"eb_obstacles_{tag}.py"
        script.write_text(SPAWN_PREAMBLE.format(
            map_name=args.map, skip=[], drop_fog=False, base_plate=False,
            world_json=str(MAPS_DIR / args.map / "progen_world_enriched.json"),
            assets_json=str(UE_ASSETS),
            progress=str(out / "progress.log"),
        ) + OBSTACLE_TAIL.format(
            shots_json=str(shots_json), w=WIDTH, h=HEIGHT,
            warmup=int(args.warmup)))
        rc = launch_editor(script, args.gpu, out / "bake.log",
                           out / "progress.log")
        print(f"render rc={rc}; progress in {out/'progress.log'}")
        if rc != 0:
            return rc

    if args.manifest:
        ok = 0
        with (out / "manifest.jsonl").open("w") as handle:
            for job in jobs:
                exists = (Path(job["image_path"]).exists()
                          and Path(job["image_path"]).stat().st_size > 0)
                ok += int(exists)
                row = {k: v for k, v in job.items() if k != "props"}
                row.update({"status": "ok" if exists else "failed",
                            "error": None if exists else "missing",
                            "map_name": args.map, "render_kind": "obstacle",
                            "image_size": [WIDTH, HEIGHT]})
                handle.write(json.dumps(row) + "\n")
        print(f"manifest: {ok}/{len(jobs)} frames present")

    if not (args.render or args.manifest):
        parser.error("one of --render / --manifest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
