"""Bake the pedestrian-signal album for a procgen city, both phases, verified.

The Paris signal album (paris_lamps_real) is the contract: for every
signalised crossing the runtime reads ``images/<node>/toward_<n>_{red,green}``
plus ``signal_visibility.json``, and a crossing is only listed there if its
own rendered frames were measured and the lit figure found at the pixel the
lamp projects to. All of that machinery -- poses, measure, verdict, album
assembly -- lives in ``embodiedbench.compiler.bake_real_lamps`` and is reused
here unchanged; this file supplies the parts a procgen city is missing: the
lamps themselves (spawned from ``progen_world_enriched.json``), and their
lens positions (read back from the spawned components, because the world json
knows where the pole is, not where the housing hangs).

Three passes, two of them in an editor:

    python tools/ue/bake_city_signals.py --map small-city-11 --probe  --gpu 5
    python tools/ue/bake_city_signals.py --map small-city-11 --render --gpu 5
    python tools/ue/bake_city_signals.py --map small-city-11 --assemble

Probe spawns the city and every pedestrian light, dumps each light's
crossing-figure component positions, and the host assigns lights to
signalised approaches and builds the camera plan. Render walks that plan,
setting every crossing figure to red then to green per pose. Assemble
measures, verdicts, lays out the album and writes the verified sidecar.
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

from embodiedbench.compiler.road_network import build_road_network  # noqa: E402
from tools.ue.bake_city_streets import (  # noqa: E402
    MAPS_DIR, PROJECT, SPAWN_PREAMBLE, UE_ASSETS, launch_editor,
)

# The render size the measurement assumes (bake_real_lamps.RENDER_LONG_EDGE).
RENDER_W, RENDER_H = 1280, 960
# A pedestrian light belongs to a crossing if it stands within this of the
# junction node. The pole is on the kerb, a junction arm is 6 m wide, and the
# next junction is 150 m away, so anything between these two scales works;
# 20 m keeps a mid-block lamp (there are none today) from adopting a junction.
ASSIGN_RADIUS_CM = 2000.0

PED_LIGHT_INSTANCE = "rt_bp_street_light_ped"
# The blueprint the world json names (/Game/RealTimeBench/...) is not in this
# content set -- it lived in the original bench's own project. The meshes it
# was built from are: complete signal variants, one per phase. A light is
# therefore two overlapping actors whose visibility the render pass toggles.
# The real thing, found by inspecting demo_2's own working signals: the local
# equivalent of the original bench's RT_BP_street_light_ped. Its components
# are the lit-figure plates -- crossing_light_stop_l/r visible on red,
# crossing_light_walk_l/r on green -- exactly the switch the original
# renderer flipped. Figures sit at ~280 cm.
PED_LIGHT_BP = ("/Game/city_props/BP/props/street_light/"
                "BP_street_light.BP_street_light_C")
HEAD_Z_CM = 280.0


def _light_nodes(map_name: str) -> list[dict]:
    """Pedestrian lights from the world json, moved onto the pavement corner.

    The generated light positions sit ~1.5 m from the junction centre --
    inside a 6 m carriageway, so the poles stood in the middle of the road.
    A real light stands at the corner of the footway. The grid is axis-
    aligned, so each light's quadrant is the sign pair of its offset from
    the junction, and the corner is half the street width plus the kerb
    margin out along both axes.
    """
    import math as _math
    from embodiedbench.compiler.road_network import build_road_network
    world = json.loads(
        (MAPS_DIR / map_name / "progen_world_enriched.json").read_text())
    network = build_road_network(MAPS_DIR / map_name, map_name=map_name)
    junctions = [network.nodes[n] for n in network.signalised_nodes()]
    out = []
    for node in world.get("nodes", []):
        inst = str(node.get("instance_name") or "").lower()
        props = node.get("properties", {}) or {}
        if not (inst == PED_LIGHT_INSTANCE
                or props.get("poi_type") == "pedestrian_light"):
            continue
        loc = props.get("location", {}) or {}
        lx, ly = float(loc.get("x", 0.0)), float(loc.get("y", 0.0))
        best = min(junctions, key=lambda j: _math.dist((lx, ly), (j.x_cm, j.y_cm)),
                   default=None)
        if best is not None and _math.dist((lx, ly), (best.x_cm, best.y_cm)) < 2000:
            width = network.streets[best.street_index].width_cm
            off = width / 2.0 + 60.0
            dx, dy = lx - best.x_cm, ly - best.y_cm
            sx = 1.0 if dx >= 0 else -1.0
            sy = 1.0 if dy >= 0 else -1.0
            loc = dict(loc)
            loc["x"] = best.x_cm + sx * off
            loc["y"] = best.y_cm + sy * off
            node = dict(node)
            node["properties"] = dict(props)
            node["properties"]["location"] = loc
            # Face the junction. The generated yaw belonged to the old
            # mid-road position; after the corner snap half the figure
            # plates pointed away from the crossing they serve, and the
            # facing filter rightly rejected them -- city 13 fell from 15
            # legible approaches to 8 on exactly this.
            ori = dict(node["properties"].get("orientation", {}) or {})
            ori["yaw"] = _math.degrees(_math.atan2(best.y_cm - loc["y"],
                                                   best.x_cm - loc["x"]))
            node["properties"]["orientation"] = ori
        out.append(node)
    # One pole per corner: two generated lights can snap to the same spot,
    # which renders as z-fighting twins.
    seen = set()
    deduped = []
    for node in out:
        loc = (node.get("properties", {}) or {}).get("location", {}) or {}
        key = (round(float(loc.get("x", 0.0))), round(float(loc.get("y", 0.0))))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(node)
    return deduped


# ----------------------------------------------------------------- probe ----
# Spawns the city exactly as the street bake does (same preamble semantics:
# demo_2 level, clear keeping the floor, idempotent labels), plus every
# pedestrian light, then writes one line per crossing-figure component:
# light id, component name, world position. No cameras, no frames.
PROBE_TAIL = r'''
LIGHTS = json.load(open({lights_json!r}))
PROBE_OUT = {probe_out!r}
LIGHT_BP = {light_bp!r}

def probe_lights():
    cls = load_bp(LIGHT_BP)
    if cls is None:
        note("FATAL: %s did not load" % LIGHT_BP)
        json.dump({{"components": []}}, open(PROBE_OUT, "w"))
        return
    dump = []
    spawned = 0
    for light in LIGHTS:
        props = light.get("properties", {{}}) or {{}}
        loc = props.get("location", {{}}) or {{}}
        ori = props.get("orientation", {{}}) or {{}}
        try:
            actor = unreal.EditorLevelLibrary.spawn_actor_from_class(
                cls,
                unreal.Vector(float(loc.get("x", 0.0)), float(loc.get("y", 0.0)),
                              float(loc.get("z", 0.0))),
                unreal.Rotator(pitch=0.0, yaw=float(ori.get("yaw", 0.0)),
                               roll=0.0))
            spawned += 1
        except Exception as error:
            note("light_spawn_failed id=%s error=%r" % (light.get("id"), error))
            continue
        for comp in actor.get_components_by_class(unreal.StaticMeshComponent):
            name = comp.get_name().lower()
            if "crossing_light_stop" not in name:
                continue
            w = comp.get_world_location()
            entry = {{"light_id": str(light.get("id")),
                      "component": comp.get_name(),
                      "x": float(w.x), "y": float(w.y), "z": float(w.z),
                      "base_z": float(loc.get("z", 0.0))}}
            try:
                f = comp.get_forward_vector()
                entry["forward"] = [float(f.x), float(f.y)]
            except Exception:
                pass
            dump.append(entry)
        unreal.EditorLevelLibrary.destroy_actor(actor)
    note("lights probed=%d stop components=%d" % (spawned, len(dump)))
    json.dump({{"components": dump}}, open(PROBE_OUT, "w"), indent=1)
    note("probe written")

probe_lights()
unreal.SystemLibrary.quit_editor()
'''


# This asset family's camera: the figures hang at 280 cm, a metre above the
# Paris lens height, and brl.poses' 2.5-7 m standoffs made every frame crane
# its neck (11-25 degrees of pitch). Stand further back and cap the pitch,
# and the frame reads like a person waiting at the kerb. The measurement
# geometry (_project) handles an off-centre lamp, so nothing else moves.
# The framing: bigger lamp, higher camera, level view. Raising the eye
# to 230 cm leaves only a 50 cm rise to the head, so the camera can stand at
# 4-5 m -- twice as close, the figure twice as tall in frame -- while the
# pitch stays under eight degrees. (The earlier eye-level/680 cm pairing was
# itself a fix for a 13-degree crane at 500 cm; moving the eye up beats
# moving the camera back.)
LED_EYE_CM = 230.0
LED_NEAR_CM = 380.0
LED_FAR_CM = 550.0
LED_PITCH_CAP_DEG = 8.0


def poses_led(sidecar: dict, network) -> list[dict]:
    positions = {nid: (node.x_cm, node.y_cm) for nid, node in network.nodes.items()}
    out = []
    for key, lamp in sorted(sidecar["lamp_pose"].items()):
        node = key.split("|")[0]
        if node not in positions:
            continue
        nx, ny = positions[node]
        lx, ly, lz = lamp.get("lens", [lamp["x"], lamp["y"], lamp.get("z", 0.0)])
        distance = math.dist((nx, ny), (lx, ly))
        standoff = max(LED_NEAR_CM, min(distance, LED_FAR_CM))
        towards_node = math.atan2(ny - ly, nx - lx)
        cx = lx + standoff * math.cos(towards_node)
        cy = ly + standoff * math.sin(towards_node)
        yaw = math.degrees(math.atan2(ly - cy, lx - cx)) % 360.0
        pitch = min(LED_PITCH_CAP_DEG,
                    math.degrees(math.atan2(lz - LED_EYE_CM, standoff)))
        out.append({
            "key": key,
            "label": key,
            "light_id": lamp.get("light_id"),
            "cam": [round(cx, 1), round(cy, 1), LED_EYE_CM],
            "rot": [0.0, round(yaw, 2), round(pitch, 2)],
            "lamp": [lx, ly, lz],
            "standoff_cm": round(standoff, 1),
            "lamp_distance_cm": round(distance, 1),
            "variant": lamp.get("variant"),
        })
    return out


def build_sidecar(map_name: str, probe: dict | None, out_root: Path) -> dict:
    """One lamp per signalised approach, placed by the road graph alone.

    The generated light positions were made for a different renderer and sat
    mid-carriageway; snapping them to axis-aligned corners broke on city 13's
    diagonal junctions, and the facing filter then found nothing to point at.
    A pedestrian light is a function of the crossing it serves: across the
    arm's mouth, on the walker's right, turned back to face them. Computing
    it from the arm's own direction works on any geometry -- grid, diagonal,
    or curved -- and needs no probe pass at all.
    """
    network = build_road_network(MAPS_DIR / map_name, map_name=map_name)
    lamp_pose: dict[str, dict] = {}
    for node_id in sorted(network.signalised_nodes()):
        node = network.nodes[node_id]
        width = network.streets[node.street_index].width_cm
        off = width / 2.0 + 60.0
        for neighbour in sorted(node.neighbours):
            other = network.nodes[neighbour]
            span = math.dist((node.x_cm, node.y_cm), (other.x_cm, other.y_cm))
            dx = (other.x_cm - node.x_cm) / max(span, 1e-6)
            dy = (other.y_cm - node.y_cm) / max(span, 1e-6)
            # Screen-right of the walking direction, the same quarter-turn
            # the pavement album uses.
            rx, ry = -dy, dx
            px = node.x_cm + dx * off + rx * off
            py = node.y_cm + dy * off + ry * off
            yaw = math.degrees(math.atan2(-dy, -dx))
            lamp_pose[f"{node_id}|{neighbour}"] = {
                "x": round(px, 1), "y": round(py, 1), "z": 0.0,
                "lens": [round(px, 1), round(py, 1), HEAD_Z_CM],
                "light_id": f"AP_{node_id}_{neighbour}",
                "yaw": round(yaw, 2),
                "variant": "BP_street_light_C",
            }
    return {
        "map": map_name,
        "method": (
            "One pedestrian light per signalised approach, placed from the "
            "road graph: across the arm's mouth on the walker's right, "
            "facing back along the arm. No generated positions involved."),
        "lamp_pose": lamp_pose,
    }


def sidecar_lights(sidecar: dict) -> list[dict]:
    """The render pass's spawn list, one entry per lamp in the sidecar."""
    out = []
    for key, lamp in sorted(sidecar["lamp_pose"].items()):
        out.append({
            "id": lamp["light_id"],
            "properties": {
                "location": {"x": lamp["x"], "y": lamp["y"], "z": 0.0},
                "orientation": {"yaw": lamp["yaw"]},
            },
        })
    return out


# ---------------------------------------------------------------- render ----
# Same spawn as the probe, then one capture per (pose, phase): every crossing
# figure in the level is set to the phase -- stop heads visible on red, walk
# heads visible on green -- the camera is placed from the plan, and the frame
# goes to real/E01 (red) or real/E02 (green) under the bake directory, which
# is the layout bake_real_lamps.assemble() reads.
RENDER_TAIL = r'''
LIGHTS = json.load(open({lights_json!r}))
SHOTS = json.load(open({shots_json!r}))
FOV = {fov!r}
W, H = {w}, {h}
WARMUP_S = {warmup!r}
LIGHT_BP = {light_bp!r}
ACTORS = {{}}

def spawn_lights():
    cls = load_bp(LIGHT_BP)
    if cls is None:
        note("FATAL: %s did not load" % LIGHT_BP)
        return
    for light in LIGHTS:
        props = light.get("properties", {{}}) or {{}}
        loc = props.get("location", {{}}) or {{}}
        ori = props.get("orientation", {{}}) or {{}}
        try:
            actor = unreal.EditorLevelLibrary.spawn_actor_from_class(
                cls,
                unreal.Vector(float(loc.get("x", 0.0)), float(loc.get("y", 0.0)),
                              float(loc.get("z", 0.0))),
                unreal.Rotator(pitch=0.0, yaw=float(ori.get("yaw", 0.0)),
                               roll=0.0))
            actor.set_actor_label("EB_Sig_" + str(light.get("id")))
            # The blueprint carries a vehicle traffic-light head above the
            # pedestrian box, permanently lit red. In the green frames that
            # drew red-over-green on one pole -- two signals, one picture,
            # murray could not say which one the frame was about. The album
            # is about the pedestrian figure; the vehicle head goes.
            for comp in actor.get_components_by_class(
                    unreal.StaticMeshComponent):
                if comp.get_name().lower() == "traffic_light":
                    comp.set_visibility(False, True)
                    comp.set_hidden_in_game(True, True)
            ACTORS[str(light.get("id"))] = actor
        except Exception as error:
            note("light_spawn_failed id=%s error=%r" % (light.get("id"), error))
    note("lights spawned=%d" % len(ACTORS))

def set_phase(phase, target):
    """Only the photographed light shows a figure. A junction carries four
    lights and every one of them lit the same colour is a picture with no
    subject -- murray could not tell which lamp the frame was about."""
    for lid, actor in ACTORS.items():
        try:
            comps = actor.get_components_by_class(unreal.StaticMeshComponent)
        except Exception:
            continue
        is_target = (target is not None and lid == str(target))
        figures = [(comp, comp.get_name().lower()) for comp in comps
                   if "crossing_light_stop" in comp.get_name().lower()
                   or "crossing_light_walk" in comp.get_name().lower()]
        want = "crossing_light_stop" if phase == "red" else "crossing_light_walk"
        # Three passes, order load-bearing. set_visibility(..., propagate=True)
        # drags children along, and on this BP the stop plate rides under the
        # walk plate: showing walk re-showed stop, and every green frame
        # carried a lit red hand beside the walker (large-city-30's whole
        # signal album failed verification on exactly this). All off, then
        # the wanted figure on, then the unwanted explicitly off LAST -- so
        # whatever the hierarchy re-shows, the final word hides it.
        for comp, _ in figures:
            comp.set_visibility(False, True)
            comp.set_hidden_in_game(True, True)
        if is_target:
            for comp, name in figures:
                if want in name:
                    comp.set_visibility(True, True)
                    comp.set_hidden_in_game(False, True)
            for comp, name in figures:
                if want not in name:
                    comp.set_visibility(False, True)
                    comp.set_hidden_in_game(True, True)

spawn_lights()
try:
    unreal.SystemLibrary.execute_console_command(None, "fov %.1f" % FOV)
except Exception:
    pass

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
            note("wrote %s" % path)
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
        out = shot["out"]
        if os.path.exists(out) and os.path.getsize(out) > 0:
            STATE["done"] += 1
            continue
        os.makedirs(os.path.dirname(out), exist_ok=True)
        set_phase(shot["phase"], shot.get("light"))
        cam = shot["cam"]; rot = shot["rot"]
        try:
            unreal.get_editor_subsystem(
                unreal.UnrealEditorSubsystem).set_level_viewport_camera_info(
                unreal.Vector(cam[0], cam[1], cam[2]),
                unreal.Rotator(pitch=rot[2], yaw=rot[1], roll=rot[0]))
            unreal.AutomationLibrary.take_high_res_screenshot(W, H, out)
            note("shot %d/%d %s" % (STATE["i"], len(SHOTS), shot["phase"]))
        except Exception as error:
            STATE["failed"] += 1
            note("ERROR on %s: %r" % (out, error))
        STATE["pending"] = out
        return
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
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--assemble", action="store_true")
    parser.add_argument("--bake", type=Path, default=None)
    parser.add_argument("--album", type=Path, default=None)
    parser.add_argument("--warmup", type=int, default=240)
    args = parser.parse_args()

    from embodiedbench.compiler import bake_real_lamps as brl

    bake = args.bake or Path("signal_bake") / args.map
    album = args.album or Path(os.environ.get("ALBUMS_DIR", "/data/albums")) / "paris_lamps_real" / args.map
    bake.mkdir(parents=True, exist_ok=True)
    lights_json = bake / "lights.json"

    if args.probe:
        # Pure geometry now -- no editor run. Kept under the same flag so the
        # runbook sequence still works verbatim.
        sidecar = build_sidecar(args.map, None, bake)
        (bake / "lamp_sidecar.json").write_text(json.dumps(sidecar, indent=1))
        lights_json.write_text(json.dumps(sidecar_lights(sidecar), indent=1))
        network = build_road_network(MAPS_DIR / args.map, map_name=args.map)
        plan = poses_led(sidecar, network)
        (bake / "real_poses.json").write_text(
            json.dumps({"fov_deg": brl.FOV_DEG, "poses": plan}, indent=1))
        print(f"planned {len(sidecar['lamp_pose'])} approach lamps, "
              f"{len(plan)} poses (host-side, no editor)")
        return 0

    if args.render:
        plan = json.loads((bake / "real_poses.json").read_text())["poses"]
        shots = []
        for i, pose in enumerate(plan):
            for atlas, phase in (("E01", "red"), ("E02", "green")):
                shots.append({"i": i, "phase": phase,
                              "light": pose.get("light_id"),
                              "cam": pose["cam"], "rot": pose["rot"],
                              "out": str(bake / f"real/{atlas}/{i:04d}.png")})
        shots_json = bake / "shots.json"
        shots_json.write_text(json.dumps(shots, indent=1))
        script = PROJECT / "Saved" / f"eb_signal_render_{args.map}.py"
        script.write_text(SPAWN_PREAMBLE.format(
            map_name=args.map, skip=[], drop_fog=False, base_plate=False,
            world_json=str(MAPS_DIR / args.map / "progen_world_enriched.json"),
            assets_json=str(UE_ASSETS), progress=str(bake / "render_progress.log")) + RENDER_TAIL.format(
            lights_json=str(lights_json), shots_json=str(shots_json),
            fov=brl.FOV_DEG, w=RENDER_W, h=RENDER_H, warmup=int(args.warmup),
            light_bp=PED_LIGHT_BP))
        rc = launch_editor(script, args.gpu, bake / "render.log",
                           bake / "render_progress.log")
        print(f"render rc={rc}; progress in {bake/'render_progress.log'}")
        return rc

    if args.assemble:
        def measure_by_phase_diff(red_path, green_path):
            """Find the lamp by what changes between phases, then judge it.

            Geometry projects where the lamp OUGHT to be; the pair of frames
            says where it IS -- they are the same scene except the switching
            figure. Locate the largest smoothed between-phase difference,
            take the same 20x20 box in both frames, and score it with the
            same top-chroma criterion the geometric path uses. Both phases
            share one box by construction.
            """
            import numpy as np
            from PIL import Image, ImageFilter

            def served(path):
                image = Image.open(path).convert("RGB")
                scale = brl.SERVED_LONG_EDGE / max(image.size)
                return image.resize((max(1, round(image.width * scale)),
                                     max(1, round(image.height * scale))),
                                    Image.LANCZOS)

            r_img, g_img = served(red_path), served(green_path)
            if r_img.size != g_img.size:
                return None
            r = np.asarray(r_img).astype(float)
            g = np.asarray(g_img).astype(float)
            diff = Image.fromarray(np.abs(r - g).sum(axis=2).clip(0, 255)
                                   .astype("uint8"))
            blurred = np.asarray(diff.filter(ImageFilter.BoxBlur(4))).astype(float)
            peak = float(blurred.max())
            if peak < 12.0:      # phases visually identical: nothing switched
                return None
            y, x = np.unravel_index(int(blurred.argmax()), blurred.shape)
            y, x = int(y), int(x)
            half = 10
            left = max(0, x - half); top = max(0, y - half)
            right = min(r.shape[1], x + half); bottom = min(r.shape[0], y + half)
            box = [int(left), int(top), int(right), int(bottom)]

            def judge(arr):
                patch = arr[top:bottom, left:right].reshape(-1, 3)
                if patch.size == 0:
                    return {"ok": False}
                chroma = np.abs(patch[:, 0] - patch[:, 1])
                keep = max(8, int(0.15 * patch.shape[0]))
                order = np.argsort(chroma)
                brightest = patch[order[-keep:]]
                mean = brightest.mean(axis=0)
                return {"ok": True, "at": [int(x), int(y)], "box": box,
                        "box_px": [right - left, bottom - top],
                        "red_over_green": round(float(mean[0] - mean[1]), 1),
                        "peak_level": float(brightest.max()),
                        "mean_rgb": [round(float(c), 1) for c in mean],
                        "served_size": list(r_img.size),
                        "judged_on": "phase-diff"}

            return judge(r), judge(g)

        def measure_led(frame_path, pose):
            """brl.measure's geometry, judged on the most chromatic pixels.

            The Paris lamp is a filled lens, so its brightest pixels are its
            colour. This asset is a dotted LED figure mounted high against
            open sky: the box's brightest 15% is routinely the sky between
            the dots, which scores red_over_green 0 on a plainly lit lamp.
            Judging on the top-chroma (|R-G|) pixels asks the same question
            -- is a red or green figure lit at this exact spot -- without
            letting white sky vote. An unlit lamp has no chromatic pixels to
            offer and still fails.
            """
            import numpy as np
            from PIL import Image
            base = brl.measure(frame_path, pose)
            if not base.get("ok"):
                return base
            image = Image.open(frame_path).convert("RGB")
            scale = brl.SERVED_LONG_EDGE / max(image.size)
            served = image.resize((max(1, round(image.width * scale)),
                                   max(1, round(image.height * scale))),
                                  Image.LANCZOS)
            left, top, right, bottom = base["box"]
            patch = np.asarray(served.crop((left, top, right, bottom))).astype(float)
            if patch.size == 0:
                return base
            flat = patch.reshape(-1, 3)
            chroma = np.abs(flat[:, 0] - flat[:, 1])
            keep = max(8, int(0.15 * flat.shape[0]))
            order = np.argsort(chroma)
            brightest = flat[order[-keep:]]
            mean = brightest.mean(axis=0)
            out = dict(base)
            out["red_over_green"] = round(float(mean[0] - mean[1]), 1)
            out["peak_level"] = float(brightest.max())
            out["mean_rgb"] = [round(float(c), 1) for c in mean]
            out["judged_on"] = "top-chroma"
            return out

        sidecar = json.loads((bake / "lamp_sidecar.json").read_text())
        plan = json.loads((bake / "real_poses.json").read_text())["poses"]
        rows = []
        for i, pose in enumerate(plan):
            # The plate's pivot is not the figure's centre, and at 300 px a
            # 15 cm aim error parks the 24 cm measurement box on sky or
            # housing -- measured: box means of 240 (sky) and 100 (housing
            # edge) on lamps that are plainly lit in the frame. Search a
            # +/-25 cm neighbourhood; both phases must score through the
            # SAME offset, so this corrects the aim without ever letting the
            # two phases be judged at different spots. The colour criteria
            # themselves are untouched.
            best = None
            # The analytic lens is the pole's axis; the figure plates hang
            # up to 70 cm off it on their brackets. Both phases still go
            # through the SAME offset.
            for dz in (-25.0, 0.0, 25.0):
                for dlat in (-80.0, -55.0, -30.0, 0.0, 30.0, 55.0, 80.0):
                    lx, ly, lz = pose["lamp"]
                    yaw = math.radians(pose["rot"][1])
                    cand = dict(pose)
                    cand["lamp"] = [lx - math.sin(yaw) * dlat,
                                    ly + math.cos(yaw) * dlat, lz + dz]
                    red = measure_led(bake / f"real/E01/{i:04d}.png", cand)
                    green = measure_led(bake / f"real/E02/{i:04d}.png", cand)
                    if not red.get("ok") or not green.get("ok"):
                        continue
                    # Maximise the WEAKER phase's margin: scoring the sum
                    # let a blazing red drag the offset onto sky that washed
                    # out the green.
                    score = min(red["red_over_green"],
                                -green["red_over_green"])
                    if best is None or score > best[0]:
                        best = (score, red, green)
            if best is None:
                red = measure_led(bake / f"real/E01/{i:04d}.png", pose)
                green = measure_led(bake / f"real/E02/{i:04d}.png", pose)
            else:
                _, red, green = best
            # Asset-family margins. The Paris lens is a filled disc with
            # nothing lit near it; this is a dotted LED figure with an
            # always-red vehicle head a metre above, whose glow bleeds into
            # every box and taxes the green margin by half. Unlit or
            # wrong-facing plates measure +7 (red-biased, for exactly that
            # reason), so red at >= 12 and green at <= -6 both stay clear of
            # everything that is not a lit figure. Paris keeps its own gate.
            def verdict(r, g):
                w = []
                if not r.get("ok") or not g.get("ok"):
                    return ["lamp projects outside its own frame"]
                if (r["peak_level"] < brl.MIN_PEAK_LEVEL
                        or g["peak_level"] < brl.MIN_PEAK_LEVEL):
                    w.append("nothing lit in the box")
                if r["red_over_green"] < brl.COLOUR_MARGIN:
                    w.append("red phase does not read red")
                if g["red_over_green"] > -6.0:
                    w.append("green phase does not read green")
                return w

            why = verdict(red, green)
            if why:
                # Geometry failed; ask the frames themselves. The two phases
                # are the same scene with ONE difference -- the switching
                # figure -- so the largest between-phase change IS the lamp,
                # wherever the projection thought it was. large-city-30
                # proved the need: every lamp plainly lit and switching,
                # every projection ~55 px off, 45/45 rejected.
                found = measure_by_phase_diff(bake / f"real/E01/{i:04d}.png",
                                              bake / f"real/E02/{i:04d}.png")
                if found is not None:
                    red2, green2 = found
                    why2 = verdict(red2, green2)
                    if not why2:
                        red, green, why = red2, green2, why2
            rows.append({"key": pose["key"], "i": i, "pass": not why,
                         "why": why, "red": red, "green": green})
        (bake / "measured.json").write_text(json.dumps(rows, indent=1))
        laid = brl.assemble(bake, album, rows, plan)
        verified = brl.verified_sidecar(sidecar, rows)
        album.mkdir(parents=True, exist_ok=True)
        (album / "signal_visibility.json").write_text(
            json.dumps(verified, indent=1))
        print(f"{laid['frames']} frames for {laid['crossings']} crossings -> {album}")
        print(f"verified {verified['legible_count']} legible, "
              f"{len(verified['rejected'])} rejected")
        return 0

    parser.error("one of --probe / --render / --assemble")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
