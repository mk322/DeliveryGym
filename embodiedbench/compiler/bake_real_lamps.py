"""Plan the camera for every signalised crossing, and check what came back.

Two halves of one job. ``poses`` turns the lamp list into camera placements for
the renderer; ``verify`` reads the rendered frames and decides which crossings
the album may actually charge for.

The verification is the point. Every earlier version of this album declared a
lamp legible because a constant said it should be, and the constants were wrong
in ways nobody could see from the code: lamps that were signposts, lamps facing
away, lamps 43 pixels across shared between three streets. Here a crossing
enters ``signal_visibility.json`` only if its own frame was measured and the lit
figure was found **at the pixel the lamp projects to**. Red paint on a No Entry
sign cannot pass that test, because the test does not look at the sign.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

# The courier reads a lamp from the kerb, not from the middle of the junction
# and not from across the street. Frame each lamp from within this range and
# both the figure and the crossing behind it stay in shot.
NEAR_CM = 250.0
FAR_CM = 700.0
EYE_CM = 165.0
FOV_DEG = 40.0
# The harness serves frames at this width; legibility is judged there, not at
# render size, because that is what the model is given.
SERVED_LONG_EDGE = 320
# What the frames were rendered at, which is the frame ``lamp_px`` is expressed
# in so the runtime can scale it to whatever it serves.
RENDER_LONG_EDGE = 1280
# Half the lit aperture, in centimetres, measured off the rendered lamps. The
# check looks at this and nothing else: a sign on the same pole is outside it.
LENS_HALF_W_CM = 12.0
LENS_HALF_H_CM = 12.0


def _standoff(distance: float) -> float:
    return max(NEAR_CM, min(distance, FAR_CM))


def poses(sidecar: dict[str, Any], network) -> list[dict[str, Any]]:
    """One camera per signalised crossing, aimed at that crossing's lamp."""
    positions = {nid: (node.x_cm, node.y_cm) for nid, node in network.nodes.items()}
    out = []
    for key, lamp in sorted(sidecar["lamp_pose"].items()):
        node = key.split("|")[0]
        if node not in positions:
            continue
        nx, ny = positions[node]
        # Aim at the lens, not at x/y/z. Those are the mesh pivot, which sits
        # on the pole; the housing hangs 24 cm off it on a bracket, and at
        # three metres that is five degrees -- enough to put the lit figure
        # outside a centre box and make a perfectly good lamp read as unlit.
        lx, ly, lz = lamp.get("lens", [lamp["x"], lamp["y"], lamp["z"]])
        lamp = {**lamp, "x": lx, "y": ly, "z": lz}
        distance = math.dist((nx, ny), (lamp["x"], lamp["y"]))
        standoff = _standoff(distance)
        # Back off along the line from the lamp towards the junction, so the
        # camera stands where a courier waiting at that crossing would.
        towards_node = math.atan2(ny - lamp["y"], nx - lamp["x"])
        cx = lamp["x"] + standoff * math.cos(towards_node)
        cy = lamp["y"] + standoff * math.sin(towards_node)
        yaw = math.degrees(math.atan2(lamp["y"] - cy, lamp["x"] - cx)) % 360.0
        # Aim at the head, which is between 1.4 and 2.2 m up depending on the
        # pole. Aiming at the pole base -- which is what the exported position
        # gives -- photographs the pavement.
        pitch = math.degrees(math.atan2(lamp["z"] - EYE_CM, standoff))
        out.append({
            "key": key,
            "label": key,
            "cam": [round(cx, 1), round(cy, 1), EYE_CM],
            # roll, yaw, pitch -- the order the sequence builder keys them in
            "rot": [0.0, round(yaw, 2), round(pitch, 2)],
            "lamp": [lamp["x"], lamp["y"], lamp["z"]],
            "standoff_cm": round(standoff, 1),
            "lamp_distance_cm": round(distance, 1),
            "facing_deg": lamp.get("facing_deg"),
            "variant": lamp.get("variant"),
        })
    return out


def _project(pose: dict[str, Any], width: int, height: int) -> tuple[int, int]:
    """Where the lamp head lands in the frame, in pixels.

    The camera is aimed straight at it, so this is the centre by construction --
    computing it anyway is what makes the check a check rather than an
    assumption, and it catches a pose whose yaw or pitch was written wrong.
    """
    cx, cy, cz = pose["cam"]
    lx, ly, lz = pose["lamp"]
    yaw = math.radians(pose["rot"][1])
    pitch = math.radians(pose["rot"][2])
    dx, dy, dz = lx - cx, ly - cy, lz - cz
    # Into camera space: forward along the yaw, right perpendicular to it.
    forward = dx * math.cos(yaw) + dy * math.sin(yaw)
    right = -dx * math.sin(yaw) + dy * math.cos(yaw)
    if forward <= 1e-6:
        return width // 2, height // 2
    focal = (width / 2.0) / math.tan(math.radians(FOV_DEG) / 2.0)
    u = width / 2.0 + focal * (right / forward)
    # Pitch rotates the view up; the lamp's height above the camera is already
    # in dz, so subtract the aim to get the residual.
    v = height / 2.0 - focal * (dz / forward - math.tan(pitch))
    return int(round(u)), int(round(v))


def measure(frame_path: Path, pose: dict[str, Any],
            served_long_edge: int = SERVED_LONG_EDGE) -> dict[str, Any]:
    """Is the lit figure there, in the right colour, at the size the model sees?

    Looks in a box around where the lamp projects -- never at the whole frame.
    A red sign elsewhere in shot is invisible to this measurement, which is the
    entire reason it is written this way.
    """
    from PIL import Image
    import numpy as np

    image = Image.open(frame_path).convert("RGB")
    scale = served_long_edge / max(image.size)
    served = image.resize((max(1, round(image.width * scale)),
                           max(1, round(image.height * scale))),
                          Image.LANCZOS)
    u, v = _project(pose, image.width, image.height)
    u, v = round(u * scale), round(v * scale)

    # The box is the lamp's own projected size, not a fixed fraction of the
    # frame. That distinction is the whole check. A flat box of 10% of frame
    # height reached far enough above the housing to take in the No Entry sign
    # mounted on the same pole, and counted its paint as a red lamp -- the
    # exact confusion this album exists to avoid, committed by the thing
    # measuring it. Sized from the housing, the sign is outside the box.
    focal_served = (served.width / 2.0) / math.tan(math.radians(FOV_DEG) / 2.0)
    distance = max(1.0, pose["standoff_cm"])
    half_w = max(2, round(focal_served * (LENS_HALF_W_CM / distance)))
    half_h = max(3, round(focal_served * (LENS_HALF_H_CM / distance)))
    left, top = max(0, u - half_w), max(0, v - half_h)
    right, bottom = min(served.width, u + half_w), min(served.height, v + half_h)
    patch = np.asarray(served.crop((left, top, right, bottom))).astype(float)
    if patch.size == 0:
        return {"ok": False, "why": "lamp projects outside the frame"}

    # Judge the colour on the brightest pixels in the box, not on all of them
    # and not on a per-pixel threshold. Two things defeat a threshold here:
    # the box necessarily contains housing and background as well as figure,
    # and a lit LED blows out towards white at its core, so its brightest
    # pixels are the *least* saturated ones. The lamps added to the level are
    # brighter than the authored ones -- they wear the LED material directly,
    # while the authored ones sit behind a dynamic instance -- and a
    # green-versus-blue test failed every one of them at (220, 255, 249)
    # while passing the dimmer authored lamps at (188, 255, 246). Red minus
    # green over the brightest pixels separates both cleanly, because bloom
    # lifts every channel but does not change which hue is dominant.
    level = patch.max(axis=2)
    flat = patch.reshape(-1, 3)
    order = np.argsort(level.reshape(-1))
    keep = max(8, int(0.15 * flat.shape[0]))
    brightest = flat[order[-keep:]]
    mean = brightest.mean(axis=0)
    return {
        "ok": True,
        "at": [u, v],
        "box": [left, top, right, bottom],
        "box_px": [right - left, bottom - top],
        # Positive means red-dominant, negative green-dominant.
        "red_over_green": round(float(mean[0] - mean[1]), 1),
        "peak_level": float(level.max()),
        "mean_rgb": [round(float(c), 1) for c in mean],
        "served_size": list(served.size),
    }


# How far red has to beat green, on the brightest pixels, for a phase to count
# as read. Squarely-seen lamps score +67 to +151 red and -38 to -73 green,
# against roughly zero for background. 12 clears background comfortably without
# failing a bright lamp whose lit core has bloomed towards white.
COLOUR_MARGIN = 12.0
# A lamp hidden behind a vehicle head leaves the box showing only facade, which
# is dimmer than any lit lens.
MIN_PEAK_LEVEL = 90.0


def verdict(red: dict, green: dict) -> list[str]:
    """Why this crossing cannot be charged, or an empty list if it can."""
    why = []
    if not red.get("ok") or not green.get("ok"):
        return ["lamp projects outside its own frame"]
    if red["peak_level"] < MIN_PEAK_LEVEL or green["peak_level"] < MIN_PEAK_LEVEL:
        why.append("nothing lit in the box")
    if red["red_over_green"] < COLOUR_MARGIN:
        why.append("red phase does not read red")
    if -green["red_over_green"] < COLOUR_MARGIN:
        why.append("green phase does not read green")
    return why


def verified_sidecar(sidecar: dict, rows: list[dict]) -> dict:
    """The sidecar the runtime reads: only crossings whose frame was checked.

    Everything dropped stays in the file under ``rejected``, with the reason.
    An album that quietly forgets what it could not photograph is how a lamp
    that does not exist ends up being charged for.
    """
    keep = {row["key"] for row in rows if row["pass"]}
    dropped = {row["key"]: row["why"] for row in rows if not row["pass"]}
    # The lit area of each lamp at render size, as [pixels, width, height], so
    # the runtime can re-decide legibility for whatever size it actually
    # serves. Area falls with the square of the resize, so a lamp checked at
    # 320 px can be under a 2x2 patch at 160. Without this the runtime's
    # resolution gate has nothing to read and silently passes everything --
    # which is what the previous sidecar's absence of the field did.
    lamp_px = {}
    for row in rows:
        if not row["pass"]:
            continue
        box = row["red"]["box_px"]
        served = row["red"]["served_size"]
        # Back out to render size: the box was measured on the served frame.
        factor = RENDER_LONG_EDGE / max(served)
        lamp_px[row["key"]] = [
            round(box[0] * factor * box[1] * factor),
            RENDER_LONG_EDGE,
            round(RENDER_LONG_EDGE * min(served) / max(served)),
        ]
    return {
        **sidecar,
        "method": sidecar["method"] + (
            " Every crossing listed here was then rendered in both phases and "
            "measured at the 320 px the harness serves, inside a box that is "
            "the lamp's own 24 cm aperture projected into the frame. A "
            "crossing that could not be read in both phases is not listed, "
            "whatever the geometry said about it."
        ),
        "verified_against_renders": True,
        "legible": sorted(keep),
        "legible_count": len(keep),
        "lamp_pose": {k: v for k, v in sidecar["lamp_pose"].items() if k in keep},
        "lamp_px": lamp_px,
        "rejected": dropped,
    }


def assemble(bake: Path, album: Path, rows: list[dict],
             plan: list[dict]) -> dict:
    """Lay the rendered frames out the way the runtime reads them.

    ``images/<node>/toward_<neighbour>_<state>.png`` -- the same layout the
    earlier albums use, so nothing in the runtime has to learn a new one. Only
    crossings that passed are copied: a frame on disk that the sidecar does not
    list is a trap for the next person, and a frame the sidecar lists that is
    not on disk is worse.
    """
    import shutil

    images = album / "images"
    written = 0
    for row in rows:
        if not row["pass"]:
            continue
        node, toward = row["key"].split("|")
        target = images / node
        target.mkdir(parents=True, exist_ok=True)
        for atlas, state in (("E01", "red"), ("E02", "green")):
            source = bake / f"real/{atlas}/{row['i']:04d}.png"
            shutil.copyfile(source, target / f"toward_{toward}_{state}.png")
            written += 1
    return {"frames": written, "crossings": sum(r["pass"] for r in rows)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--measured", type=Path,
                        help="measured.json; with it, --out is a verified sidecar")
    parser.add_argument("--album", type=Path,
                        help="lay the passing frames out as an album here")
    parser.add_argument("--bake", type=Path, default=Path("lamp_bake"))
    parser.add_argument("--map", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from embodiedbench.compiler.road_network import build_road_network

    map_dir = args.map or (Path(__file__).resolve().parents[2]
                           / "vendor/vagen/vagen/envs/deliverybench/maps/citycore-paris")
    network = build_road_network(map_dir, map_name=map_dir.name)
    sidecar = json.loads(args.sidecar.read_text())
    if args.measured:
        rows = json.loads(args.measured.read_text())
        if args.album:
            plan = json.loads((args.bake / "real_poses.json").read_text())["poses"]
            laid = assemble(args.bake, args.album, rows, plan)
            print(f"{laid['frames']} frames for {laid['crossings']} crossings "
                  f"-> {args.album}")
        verified = verified_sidecar(sidecar, rows)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(verified, indent=1))
        print(f"{verified['legible_count']} crossings verified against their "
              f"own frames, {len(verified['rejected'])} dropped")
        print("wrote", args.out)
        return 0
    plan = poses(sidecar, network)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"fov_deg": FOV_DEG, "poses": plan}, indent=1))
    distances = sorted(p["lamp_distance_cm"] for p in plan)
    print(f"{len(plan)} camera poses")
    if distances:
        print(f"lamp distance: min {distances[0]/100:.1f} m  "
              f"median {distances[len(distances)//2]/100:.1f} m  "
              f"max {distances[-1]/100:.1f} m")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
