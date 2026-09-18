"""Bake the obstacle album: every obstacle site, both ways, both kinds.

The runtime decides where an obstacle *can* be from the road network alone
(``runtime/city/obstacles.obstacle_sites``), with no seed in the hash, so the
site list is a property of the map. That is what makes one bake enough: every
episode of every seed draws its obstacles from this list, and this list is what
gets photographed.

For each site edge and each direction along it, two frames are captured from
the identical camera the street album used -- pedestrian eye level at the node,
looking down the street -- one with a line of barriers across the carriageway
(``road_block``) and one with the pavement crowded by street furniture
(``slow_pedestrian``). Same camera as the clear frame, because that is what
makes the difference between them *be* the obstacle and nothing else, which is
what the visibility measurement then relies on.

    python tools/ue/bake_obstacles.py --out $ALBUMS_DIR/paris_obstacles/citycore-paris

Afterwards, measure what the frames actually show and write the gate:

    python -m embodiedbench.compiler.obstacle_visibility \\
        $ALBUMS_DIR/paris_obstacles/citycore-paris \\
        --clear $ALBUMS_DIR/paris_streets_v2/citycore-paris --write-sidecar
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from embodiedbench.compiler.fpv_render import UeMcp, ensure_map
from embodiedbench.compiler.road_network import bearing_deg, build_road_network
from embodiedbench.runtime.city.obstacles import OBSTACLE_TYPES, obstacle_sites

# The street album's camera, exactly. Changing either number here would make an
# obstructed frame differ from its clear counterpart in ways that are not the
# obstacle, and the visibility measurement is a comparison of the two.
CAMERA_Z_CM = 160.0
WIDTH, HEIGHT = 640, 480

# How far down the edge the obstacle stands. Near enough to fill enough of the
# frame to be unmistakable, far enough that it reads as "across the street
# ahead" rather than "pressed against the lens". Clamped into the edge so a
# short hop does not put the barrier past the next junction.
STANDOFF_CM = 650.0
# Below this the obstacle is against the lens rather than down the street: on a
# short hop the barrier filled the bottom third of the frame and the 6.6 m
# advertising column ran off the top of it.
MIN_STANDOFF_CM = 380.0
EDGE_FRACTION = 0.45

BARRIER = "/Game/CityCore_Paris/Models/Props/StreetProps/SM_PR_Fence_01"
BIN = "/Game/CityCore_Paris/Models/Props/StreetProps/SM_PR_WheelieBin_01"
STAND = "/Game/CityCore_Paris/Models/Props/StreetProps/SM_PR_AdvertismentStand"
CHALKBOARD = "/Game/CityCore_Paris/Models/Props/StreetProps/SM_FrameChalkboard"
POT = "/Game/CityCore_Paris/Models/Props/StreetProps/SM_PlantPot_01"

# Where each prop goes, relative to the point on the edge the obstacle sits at:
# ``along`` is metres-forward down the edge, ``across`` is metres to the right
# of travel, ``yaw`` is added to the travel bearing, ``scale`` is uniform.
#
# A ``road_block`` spans the carriageway: five barrier panels, 1.34 m each, laid
# corner to corner with bins at the ends, so there is visibly no way past. A
# ``slow_pedestrian`` sits to one side: the column, a chalkboard and pots
# crowding the footway, which you get round but not quickly. The two are meant
# to be distinguishable at a glance and from a distance, because that is the
# discrimination the benchmark is asking a model to make.
LAYOUTS: dict[str, list[dict[str, float | str]]] = {
    "road_block": [
        {"asset": BARRIER, "along": 0.0, "across": -260.0, "yaw": 90.0, "scale": 1.0},
        {"asset": BARRIER, "along": 0.0, "across": -130.0, "yaw": 90.0, "scale": 1.0},
        {"asset": BARRIER, "along": 0.0, "across": 0.0, "yaw": 90.0, "scale": 1.0},
        {"asset": BARRIER, "along": 0.0, "across": 130.0, "yaw": 90.0, "scale": 1.0},
        {"asset": BARRIER, "along": 0.0, "across": 260.0, "yaw": 90.0, "scale": 1.0},
        {"asset": BIN, "along": -60.0, "across": -330.0, "yaw": 0.0, "scale": 1.0},
        {"asset": BIN, "along": -60.0, "across": 330.0, "yaw": 0.0, "scale": 1.0},
    ],
    "slow_pedestrian": [
        {"asset": STAND, "along": 0.0, "across": 260.0, "yaw": 0.0, "scale": 1.0},
        {"asset": CHALKBOARD, "along": -140.0, "across": 150.0, "yaw": 35.0, "scale": 1.0},
        {"asset": POT, "along": -260.0, "across": 220.0, "yaw": 0.0, "scale": 1.0},
        {"asset": BIN, "along": 120.0, "across": 190.0, "yaw": 15.0, "scale": 1.0},
        {"asset": POT, "along": 240.0, "across": 250.0, "yaw": 0.0, "scale": 1.0},
    ],
}

# Spawn, point the camera, shoot, and clean up -- in one call, because the MCP
# server runs a script and returns while the editor keeps ticking, so anything
# split across calls can be photographed half-built.
SCRIPT = """
import unreal

tag = "EB_OBSTACLE"
eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
for actor in eas.get_all_level_actors():
    if actor and actor.actor_has_tag(tag):
        eas.destroy_actor(actor)

spawned = []
for item in {props!r}:
    asset = unreal.EditorAssetLibrary.load_asset(item["asset"])
    if asset is None:
        continue
    actor = eas.spawn_actor_from_object(
        asset,
        unreal.Vector(item["x"], item["y"], item["z"]),
        unreal.Rotator(pitch=0.0, yaw=item["yaw"], roll=0.0),
    )
    if actor is None:
        continue
    actor.tags = [tag]
    actor.set_actor_scale3d(unreal.Vector(item["scale"], item["scale"], item["scale"]))
    spawned.append(actor)

unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).set_level_viewport_camera_info(
    unreal.Vector({x!r}, {y!r}, {z!r}),
    unreal.Rotator(pitch=0.0, yaw={yaw!r}, roll=0.0),
)
unreal.AutomationLibrary.take_high_res_screenshot({w}, {h}, {out!r})
"""

CLEANUP = """
import unreal
eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
for actor in eas.get_all_level_actors():
    if actor and actor.actor_has_tag("EB_OBSTACLE"):
        eas.destroy_actor(actor)
"""


def props_for(kind: str, origin: tuple[float, float], bearing: float,
              ground_z: float) -> list[dict]:
    """Place one layout in world space, given where and which way it faces."""
    radians = math.radians(bearing)
    # UE yaw 0 is +X and increases toward +Y, and ``bearing_deg`` is written in
    # the same convention, so forward is (cos, sin) and right is a quarter turn
    # clockwise from it.
    forward = (math.cos(radians), math.sin(radians))
    # Screen-right, not maths-right. UE's yaw is left-handed with Z up, so the
    # camera's right is the forward vector turned a further +90 degrees of yaw.
    # Taking it the other way put the whole ``slow_pedestrian`` cluster on the
    # wrong side of the street, which on a narrow one meant the column stood
    # half out of frame instead of on the pavement being walked past.
    right = (-math.sin(radians), math.cos(radians))
    out = []
    for item in LAYOUTS[kind]:
        along, across = float(item["along"]), float(item["across"])
        out.append({
            "asset": item["asset"],
            "x": origin[0] + forward[0] * along + right[0] * across,
            "y": origin[1] + forward[1] * along + right[1] * across,
            "z": ground_z,
            "yaw": bearing + float(item["yaw"]),
            "scale": float(item["scale"]),
        })
    return out


def plan(map_dir: Path, out: Path, map_name: str, viewpoint: str = "carriageway") -> list[dict]:
    """Every obstacle site, both ways, both kinds.

    ``viewpoint`` moves the *camera* and nothing else. The props stay exactly
    where they are, because the obstacle is a fact about the street rather than
    about who is looking at it.

    A pavement bake is not a nicety. The street album has a pavement edition and
    this one did not, so a courier on foot was served footway frames on clear
    streets and centreline frames wherever an obstacle stood -- and since the
    obstacle album is the only source of those, the *viewpoint itself* told the
    policy a hazard was present, with no need to look at the picture. Measured
    before this existed: 19.5% of a walker's frames came from the centreline
    album and every one of them was an obstacle. That is the filename leak
    again, wearing a different hat.
    """
    network = build_road_network(map_dir, map_name=map_name)
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
                # Same rule as tools/ue/bake_pavement_views.py: half the street's
                # own width plus a margin, to the right of travel.
                street = network.streets[network.nodes[src].street_index]
                offset = street.width_cm / 2.0 + 60.0
                right = math.radians(bearing + 90.0)
                camera = (here[0] + offset * math.cos(right),
                          here[1] + offset * math.sin(right))
            for kind in OBSTACLE_TYPES:
                jobs.append({
                    "site": f"{a}|{b}", "node": src, "toward": dst, "type": kind,
                    "viewpoint": viewpoint,
                    "yaw": round(bearing, 1), "x_cm": camera[0], "y_cm": camera[1],
                    "z_cm": CAMERA_Z_CM, "standoff_cm": round(standoff, 1),
                    "edge_length_m": round(length / 100.0, 1),
                    "origin": [round(origin[0], 1), round(origin[1], 1)],
                    "ground_z_cm": round(network.nodes[src].position and 0.0, 1),
                    "image_path": str(out / "images" / src / f"toward_{dst}_{kind}.png"),
                })
    return jobs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--map-dir", type=Path, default=(
        Path(__file__).resolve().parents[2]
        / "vendor/vagen/vagen/envs/deliverybench/maps/citycore-paris"))
    parser.add_argument("--map-name", default="citycore-paris")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scene", default="/Game/CityCore_Paris/Scenes/ParisCity_FinalBlueprints")
    parser.add_argument("--scene-token", default="Paris")
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--limit", type=int, default=0, help="render only the first N jobs")
    parser.add_argument("--viewpoint", default="carriageway",
                        choices=("carriageway", "pavement"))
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--timeout", type=float, default=90.0)
    args = parser.parse_args(argv)

    started = time.time()
    jobs = plan(args.map_dir, args.out, args.map_name, args.viewpoint)
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"planned {len(jobs)} frames", flush=True)

    mcp = UeMcp(port=args.port)
    ensure_map(mcp, args.scene, args.scene_token)
    time.sleep(5)

    rendered = skipped = failed = 0
    for index, job in enumerate(jobs):
        path = Path(job["image_path"])
        if path.exists() and path.stat().st_size > 0:
            skipped += 1
            job["status"] = "ok"
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        props = props_for(job["type"], tuple(job["origin"]), job["yaw"], 0.0)
        try:
            mcp.python(SCRIPT.format(
                props=props, x=float(job["x_cm"]), y=float(job["y_cm"]),
                z=CAMERA_Z_CM, yaw=float(job["yaw"]),
                w=WIDTH, h=HEIGHT, out=str(path),
            ))
        except Exception as exc:  # noqa: BLE001 - a failed capture is a result
            failed += 1
            job["status"], job["error"] = "failed", f"{type(exc).__name__}: {exc}"
            continue
        deadline = time.time() + args.timeout
        while time.time() < deadline:
            if path.exists() and path.stat().st_size > 0:
                break
            time.sleep(0.25)
        if path.exists() and path.stat().st_size > 0:
            rendered += 1
            job["status"] = "ok"
        else:
            failed += 1
            job["status"], job["error"] = "failed", "screenshot never appeared"
        if (index + 1) % 25 == 0:
            rate = (rendered + skipped) / max(time.time() - started, 1e-6)
            print(f"  {index+1}/{len(jobs)} rendered={rendered} skipped={skipped} "
                  f"failed={failed} {rate:.2f} img/s", flush=True)

    mcp.python(CLEANUP)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for job in jobs:
            job.setdefault("status", "failed")
            job["map_name"] = args.map_name
            job["render_kind"] = "obstacle"
            job["image_size"] = [WIDTH, HEIGHT]
            handle.write(json.dumps(job) + "\n")

    report = {
        "map": args.map_name, "planned": len(jobs), "rendered": rendered,
        "skipped_existing": skipped, "failed": failed,
        "camera_z_cm": CAMERA_Z_CM, "image_size": [WIDTH, HEIGHT],
        "seconds": round(time.time() - started, 1),
        "status": "pass" if failed == 0 and jobs else "fail",
    }
    if args.report:
        args.report.write_text(json.dumps(report, indent=1))
    print(json.dumps(report), flush=True)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
