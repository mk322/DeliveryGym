"""Which crossings actually have a lamp, read out of the scene.

The environment used to decide this by counting: a node with three or more ways
out was signalised, and every approach to it was charged for crossing on red.
Three separate things were wrong with the data underneath that, each found by
going one level deeper into the scene:

1. **Degree is not a lamp.** 105 junctions have degree three or more; only 34 of
   them have a lamp. Two thirds of the charged crossings had no lamp in the
   world at all.
2. **``poi_type: pedestrian_light`` is not a lamp either.** The exporter wrote
   that on all 125 ``BP_TrafficLightsPoles_C`` actors, but 51 of them carry no
   lamp mesh -- they are signposts, WrongWay and Towing and NoStanding -- and
   six more carry only a vehicle head, which a courier on foot does not read.
   The real count is 75 pedestrian lamp heads on 68 poles. Charging a crossing
   because a *No Parking* sign stands near it is the same mistake as reading a
   red sign as a red light, one level further up.
3. **The pole's yaw is not the lamp's yaw.** The head is a component on the
   pole and it is usually rotated relative to it: of 75 heads only 27 sit at the
   pole's own yaw, 24 are turned 90 degrees, 12 are turned 270, and the rest are
   odd angles. Measured against the orientation the exporter recorded, 64% of
   lamps were wrong by more than 20 degrees, with a median error of 90. That is
   why the baked lamps looked like they were pointing the wrong way: they were.

And one thing that cannot be read from any export, only from a render: the lit
face points along the head's **right** axis, not its forward one. Four cameras
placed around a lamp at 90 degree intervals show the figure from exactly one of
them, at ``yaw + 90``. Checked on four lamps across both housing variants.

So: a crossing is signalised when a lamp stands near the junction, lies in the
direction of the leg being crossed, and has its lit face turned towards the
courier. Anything else is a lamp the courier cannot read, and an unreadable lamp
must not be charged.

The lamp facts come from ``lamp_heads.json``, exported by
``CityCore_Paris/Scripts/export_lamp_heads.py``, which searches every actor in
the level rather than every actor whose class name happens to say TrafficLight.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
DEFAULT_MAP = REPO / "vendor/vagen/vagen/envs/deliverybench/maps/citycore-paris"

# A lamp belongs to the junction it stands at, not to whichever node happens to
# be nearest: a lamp 6 m from a corner is that corner's, and the mid-block node
# 4 m away in the other direction has nothing to do with it. 25 m is wide
# enough to cover a Haussmann corner and narrow enough not to reach the next.
JUNCTION_RADIUS_CM = 2500.0
# How far off the leg's own bearing a lamp can sit and still be the lamp a
# person crossing that leg reads. A junction's lamps sit one per corner, so the
# arc has to be generous; beyond 50 degrees the lamp belongs to the next leg.
LEG_ARC_DEG = 60.0
# How far off the lamp's lit face the courier can stand and still read it.
# This started at 70 as a guess and is now what the bake measured. Rendering
# every candidate and reading the figure at served size draws a sharp line: at
# 43 degrees and beyond the housing's own side wall clips the aperture, and it
# clips the thin walking figure far more than the fat standing one, so the
# green phase goes unreadable while the red still looks fine. Everything at 38
# and below reads in both phases. 40 sits in that gap.
#
# The rule still is not trusted on its own -- every frame is measured, and one
# lamp at 3 degrees is dropped because a vehicle head stands in front of it,
# which no angle could have predicted.
FACING_ARC_DEG = 40.0
# The lit face is turned this far from the head's own forward direction. Not a
# guess -- see the module docstring. Confirmed twice over: four cameras around a
# lamp show the figure only from yaw+90, and the lens quad's own normal in the
# mesh is +Y, which is the same direction.
LIT_FACE_OFFSET_DEG = 90.0
# Where the lit lens sits relative to the mesh pivot, in centimetres of the
# mesh's own space. The pivot is on the pole and the housing hangs off a
# bracket, so a camera aimed at the pivot puts the figure off to one side --
# 24 cm of offset is 5 degrees at three metres, which is most of the way out of
# a centre box. Read off the lens quad in both housing variants; they agree.
LENS_OFFSET_CM = (0.5, 24.0, -2.4)
# The aperture is 24 cm square, so half of it is what a check should look at.
LENS_HALF_CM = 12.0
# A junction is only a junction if you can choose there.
MIN_DEGREE = 3


def _bearing(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 360.0


def _gap(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _lens(x: float, y: float, z: float, yaw: float) -> tuple[float, float, float]:
    """The centre of the lit aperture, given the head's pivot and yaw."""
    ox, oy, oz = LENS_OFFSET_CM
    angle = math.radians(yaw)
    return (x + ox * math.cos(angle) - oy * math.sin(angle),
            y + ox * math.sin(angle) + oy * math.cos(angle),
            z + oz)


def scene_lamps(map_dir: Path) -> list[dict[str, Any]]:
    """Every pedestrian lamp head in the level, with its head position and lit face.

    Reads ``lamp_heads.json`` if it is there. Falls back to the old POI list
    only so that a checkout without the export still runs -- and says so, since
    that list is the one with the signposts in it.
    """
    export = map_dir / "lamp_heads.json"
    if export.exists():
        data = json.loads(export.read_text())
        return [{
            "id": f"{head['actor']}|{head['component']}",
            "x": float(head["x"]),
            "y": float(head["y"]),
            "z": float(head["z"]),
            "yaw": float(head["yaw"]) % 360.0,
            "lit_face": (float(head["yaw"]) + LIT_FACE_OFFSET_DEG) % 360.0,
            "lens": _lens(float(head["x"]), float(head["y"]), float(head["z"]),
                          float(head["yaw"])),
            "variant": head.get("variant"),
            "from_scene": True,
        } for head in data["heads"]]

    world = json.loads((map_dir / "progen_world_enriched.json").read_text())
    out = []
    for node in world.get("nodes", []):
        props = node.get("properties") or {}
        if props.get("poi_type") != "pedestrian_light":
            continue
        location = props.get("location") or {}
        yaw = float((props.get("orientation") or {}).get("yaw", 0.0)) % 360.0
        out.append({
            "id": node.get("id"),
            "x": float(location.get("x", 0.0)),
            "y": float(location.get("y", 0.0)),
            "z": float(location.get("z", 0.0)),
            "yaw": yaw,
            "lit_face": (yaw + LIT_FACE_OFFSET_DEG) % 360.0,
            "lens": _lens(float(location.get("x", 0.0)),
                          float(location.get("y", 0.0)),
                          float(location.get("z", 0.0)), yaw),
            "variant": None,
            # This list includes 51 signposts and records the pole's yaw rather
            # than the head's. Anything built on it is approximate at best.
            "from_scene": False,
        })
    return out


def lamps_by_leg(network, lamps: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Map ``"node|toward"`` to the lamp a courier crossing that leg reads.

    Three tests, all of which a real lamp passes and each of which removes a
    different kind of false positive:

    * it stands at this junction, not at the next one;
    * it lies along the leg being crossed, not along a different arm;
    * its lit face is turned towards the courier. This is the test the old code
      could not make, because it had the pole's yaw rather than the head's, and
      it is what stops a lamp being charged when all the courier can see of it
      is its back.
    """
    positions = {nid: (node.x_cm, node.y_cm) for nid, node in network.nodes.items()}
    junctions = {nid: p for nid, p in positions.items()
                 if len(network.nodes[nid].neighbours) >= MIN_DEGREE}
    if not junctions:
        return {}

    at_junction: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for lamp in lamps:
        here = (lamp["x"], lamp["y"])
        nid, distance = min(
            ((n, math.dist(here, p)) for n, p in junctions.items()),
            key=lambda pair: pair[1])
        if distance <= JUNCTION_RADIUS_CM:
            at_junction[nid].append({**lamp, "distance_cm": round(distance, 1)})

    out: dict[str, dict[str, Any]] = {}
    for nid, here_lamps in at_junction.items():
        legs = [(nb, _bearing(positions[nid], positions[nb]))
                for nb in sorted(network.nodes[nid].neighbours)]
        for lamp in here_lamps:
            towards = _bearing(positions[nid], (lamp["x"], lamp["y"]))
            # Can the courier, standing at the junction, see the lit face?
            back = _bearing((lamp["x"], lamp["y"]), positions[nid])
            facing = _gap(back, lamp["lit_face"])
            if facing > FACING_ARC_DEG:
                continue
            # Which crossing does this lamp govern? Not "the leg the lamp lies
            # down" -- that proxy breaks exactly where lamps actually stand. A
            # corner lamp at a four-way sits 45 degrees from two legs at once,
            # so the arc test either accepts both or rejects both, and the two
            # readings disagree for 22 of the 36 lamps here.
            #
            # A pedestrian lamp faces along the crossing it governs: it is
            # mounted either on the far kerb looking back across, or on the
            # near kerb looking back at the person waiting. Both put the
            # crossing on the lamp's own facing axis, which is why that axis
            # is what the leg is matched against, in either direction.
            leg, offset = min(
                ((nb, min(_gap(lamp["lit_face"], lb),
                          _gap((lamp["lit_face"] + 180.0) % 360.0, lb)))
                 for nb, lb in legs),
                key=lambda pair: pair[1])
            if offset > LEG_ARC_DEG:
                continue
            key = f"{nid}|{leg}"
            # Two lamps can fall on one leg -- the pair either side of a
            # crossing. Keep the one turned most squarely towards the courier;
            # that is the one they actually read, and it photographs best.
            if key in out and out[key]["facing_deg"] <= facing:
                continue
            out[key] = {**lamp, "offset_deg": round(offset, 1),
                        "facing_deg": round(facing, 1),
                        "bearing_from_node": round(towards, 2)}
    return out


def sidecar(network, map_dir: Path) -> dict[str, Any]:
    lamps = scene_lamps(map_dir)
    from_scene = bool(lamps and lamps[0].get("from_scene"))
    by_leg = lamps_by_leg(network, lamps)
    junctions = [n for n, node in network.nodes.items()
                 if len(node.neighbours) >= MIN_DEGREE]
    lamped = {key.split("|")[0] for key in by_leg}
    return {
        "map": map_dir.name,
        "method": (
            "read from the scene. Every pedestrian lamp head in the level is "
            "attached to the junction it stands at, then to the leg it faces, "
            "and kept only if its lit face is turned towards the courier. The "
            "head's own yaw is used, not the pole's -- they differ for 64% of "
            "lamps, by 90 degrees at the median -- and the lit face is the "
            "head's right axis, which was established by rendering one lamp "
            "from four directions rather than assumed."
        ),
        "from_scene": from_scene,
        "lamps_are_per_approach": True,
        "junctions": len(junctions),
        "junctions_with_a_lamp": len(lamped),
        "lamps_in_scene": len(lamps),
        "legible_count": len(by_leg),
        "legible": sorted(by_leg),
        # Where the lamp is, which way it faces, and which way it is lit --
        # what a camera aimed at a lamp needs, and what tells a checker where
        # in the frame to look rather than trusting any red pixel it finds.
        "lamp_pose": {key: {"x": v["x"], "y": v["y"], "z": v["z"],
                            # Aim here, not at x/y/z: that is the pole.
                            "lens": [round(c, 1) for c in v["lens"]],
                            "lens_half_cm": LENS_HALF_CM,
                            "yaw": v["yaw"], "lit_face": v["lit_face"],
                            "distance_cm": v["distance_cm"],
                            "facing_deg": v["facing_deg"],
                            "variant": v.get("variant")}
                      for key, v in sorted(by_leg.items())},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--map", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--lamps", type=Path,
                        help="directory holding lamp_heads.json, if not --map")
    parser.add_argument("--out", type=Path,
                        help="write signal_visibility.json here")
    args = parser.parse_args()

    from embodiedbench.compiler.road_network import build_road_network

    network = build_road_network(args.map, map_name=args.map.name)
    data = sidecar(network, args.lamps or args.map)
    if not data["from_scene"]:
        print("WARNING: no lamp_heads.json; falling back to the POI list, "
              "which contains 51 signposts and the wrong yaws")
    print(f"{data['lamps_in_scene']} pedestrian lamp heads in the scene")
    print(f"{data['junctions_with_a_lamp']} of {data['junctions']} junctions "
          f"have one facing a leg")
    print(f"{data['legible_count']} crossings are signalised "
          f"(was: every approach to all {data['junctions']})")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(data, indent=1))
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
