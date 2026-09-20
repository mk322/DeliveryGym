"""Bake the pavement album: the street seen from the footway, not the middle of it.

Every album in this repository was shot from the carriageway centreline. The
manifests say so and the measurement confirms it -- street, signal and obstacle
cameras all sit 0.00 m from the node, which *is* the centreline. That is the
right viewpoint for a scooter or a car and the wrong one for the only embodiment
the benchmark had: a courier on foot is on the pavement, and a policy trained on
road-centre frames has learned to recognise a place it will never stand.

This bakes the same 856 approaches from where a pedestrian actually is: offset
perpendicular to the direction of travel, toward the right-hand kerb, by half the
street's own width plus a margin, so the camera stands on the footway. Right-hand
because France drives on the right and a pedestrian keeps to the right pavement.

Everything else is held identical to the carriageway bake -- 160 cm eye height,
640x480, same yaw, same map -- so the two albums differ by the thing being
varied and nothing else. That is what lets a run at one viewpoint be compared
with a run at the other.

    python tools/ue/bake_pavement_views.py --plan          # write the job list
    python tools/ue/bake_pavement_views.py --render        # drive the editor

The render step launches UnrealEditor offscreen against the CityCore Paris
project, installs a slate tick callback and captures one frame per tick. A
commandlet cannot do it: ``take_high_res_screenshot`` needs the editor to tick
before the file appears, and a commandlet never ticks.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from embodiedbench.compiler.road_network import bearing_deg, build_road_network

REPO = Path(__file__).resolve().parents[2]
MAPS = REPO / "vendor/vagen/vagen/envs/deliverybench/maps/citycore-paris"
MAP_NAME = "citycore-paris"

# Held identical to the carriageway bake. See fpv_render.PLAIN_* and
# tools/ue/bake_obstacles.py, which pins the same numbers for the same reason.
CAMERA_Z_CM = 160.0
WIDTH, HEIGHT = 640, 480
SCENE = "/Game/CityCore_Paris/Scenes/ParisCity_FinalBlueprints"

PROJECT = Path(os.environ.get("CITYCORE_PARIS_PROJECT", "/opt/CityCore_Paris"))
ENGINE = Path(os.environ.get("UE_ROOT", "/opt/UnrealEngine-5.8"))
DEFAULT_OUT = Path(os.environ.get("ALBUMS_DIR", "/data/albums")) / "paris_streets_pavement" / MAP_NAME

# How far onto the footway. Half the carriageway puts the camera exactly on the
# kerbstone; a little further puts it where a person walks. Streets here are
# 6 m wide with a few at 10 m, so this is between 3.6 m and 5.6 m from the
# centreline depending on the street -- which is why it is taken per street
# rather than as one number.
KERB_MARGIN_CM = 60.0

# Chosen by calibration against the carriageway album; see RENDER_SCRIPT.
EXPOSURE_EV = 10.0


def plan_jobs(out_root: Path) -> list[dict]:
    """One job per approach, camera moved off the centreline to the footway."""
    network = build_road_network(MAPS, map_name=MAP_NAME)
    jobs: list[dict] = []
    for node_id, node in sorted(network.nodes.items()):
        here = (node.x_cm, node.y_cm)
        street = network.streets[node.street_index]
        offset = street.width_cm / 2.0 + KERB_MARGIN_CM
        for neighbour in sorted(node.neighbours):
            other = network.nodes[neighbour]
            yaw = bearing_deg(here, (other.x_cm, other.y_cm))
            # Perpendicular, to the right of the direction of travel. UE yaw is
            # degrees clockwise from +X, so the right-hand normal is yaw + 90.
            right = math.radians(yaw + 90.0)
            jobs.append({
                "waypoint_id": node_id,
                "toward_node": neighbour,
                "yaw": yaw,
                "x_cm": here[0] + offset * math.cos(right),
                "y_cm": here[1] + offset * math.sin(right),
                "z_cm": CAMERA_Z_CM,
                "centreline_x_cm": here[0],
                "centreline_y_cm": here[1],
                "kerb_offset_cm": offset,
                "street": street.name,
                "street_width_cm": street.width_cm,
                "image_path": str(out_root / "images" / node_id
                                  / f"toward_{neighbour}.png"),
                "map_name": MAP_NAME,
                "render_kind": "street_view_pavement",
                "viewpoint": "pavement",
                "image_size": [WIDTH, HEIGHT],
            })
    return jobs


# How the frames are actually taken, and why it took four attempts.
#
# 1. ``SceneCapture2D``. Synchronous, needs no live editor, and its output is
#    unusable: everything outside direct sunlight renders pure black, because
#    Lumen global illumination does not accumulate in a scene capture. Setting
#    ``always_persist_rendering_state``, forcing the Lumen scene-capture cvars
#    and letting it converge over 96 captures moved the frame mean from 24.3 to
#    24.8 against the reference album's 64.0. Not an exposure problem, and not
#    fixable from the capture component.
#
# 2. Editor viewport screenshots under ``-ExecutePythonScript``. This is the
#    path the carriageway album was taken on, and it is the right one -- but
#    that switch means "run this script, then quit", so the editor exited before
#    ticking and no screenshot was ever written. Directories created, zero
#    frames.
#
# 3. Driving a living editor over UE's Python remote execution. The editor
#    stayed up and ticked 923 frames, but never announced itself on multicast,
#    on 0.0.0.0 or pinned to loopback.
#
# 4. What works, and what is used here: ``-ExecCmds="py <script>"``. It runs the
#    script *after* startup without implying a quit, so the editor keeps
#    ticking, a slate post-tick callback can drive the whole album one frame per
#    tick, and the script quits the editor when the list is done. Same
#    ``set_level_viewport_camera_info`` and ``take_high_res_screenshot`` the
#    carriageway album used, so the two albums differ by camera position and
#    nothing else.
RENDER_SCRIPT = r'''
import json, os, unreal

JOBS = json.load(open({jobs!r}))
W, H = {w}, {h}
STATE = {{"i": 0, "pending": None, "waited": 0, "done": 0, "failed": 0}}
LOG = open({progress!r}, "w", buffering=1)

def _note(message):
    LOG.write(message + "\n")

try:
    unreal.SystemLibrary.execute_console_command(
        unreal.EditorLevelLibrary.get_editor_world(), "DisableAllScreenMessages")
except Exception as error:
    _note("could not silence screen messages: %s" % error)

def _tick(delta):
    if STATE["pending"] is not None:
        path = STATE["pending"]
        if os.path.exists(path) and os.path.getsize(path) > 0:
            STATE["pending"], STATE["waited"] = None, 0
            STATE["done"] += 1
            if STATE["done"] % 25 == 0:
                _note("done=%d failed=%d of %d" % (
                    STATE["done"], STATE["failed"], len(JOBS)))
        else:
            STATE["waited"] += 1
            if STATE["waited"] > 900:
                _note("timeout %s" % path)
                STATE["pending"], STATE["waited"] = None, 0
                STATE["failed"] += 1
        return

    while STATE["i"] < len(JOBS):
        job = JOBS[STATE["i"]]
        STATE["i"] += 1
        out = job["image_path"]
        if os.path.exists(out) and os.path.getsize(out) > 0:
            STATE["done"] += 1
            continue
        os.makedirs(os.path.dirname(out), exist_ok=True)
        loc = unreal.Vector(job["x_cm"], job["y_cm"], job["z_cm"])
        # Keyword arguments are mandatory: unreal.Rotator is positionally
        # (roll, pitch, yaw), and getting it wrong points the camera at the sky.
        rot = unreal.Rotator(pitch=0.0, yaw=job["yaw"], roll=0.0)
        try:
            unreal.get_editor_subsystem(
                unreal.UnrealEditorSubsystem).set_level_viewport_camera_info(loc, rot)
            unreal.AutomationLibrary.take_high_res_screenshot(W, H, out)
        except Exception as error:
            # Swallowed silently, this looked like "3 jobs, 0 captured, 0
            # failed" -- the callback kept advancing while every frame threw.
            STATE["failed"] += 1
            _note("ERROR on %s: %r" % (out, error))
            return
        STATE["pending"] = out
        return

    _note("FINISHED done=%d failed=%d" % (STATE["done"], STATE["failed"]))
    LOG.close()
    unreal.SystemLibrary.quit_editor()

_note("starting: %d jobs" % len(JOBS))
unreal.register_slate_post_tick_callback(_tick)
'''


PROGRESS = PROJECT / "Saved" / "pavement_progress.log"


def render(jobs_path: Path, gpu: str, log: Path | None = None) -> int:
    """Launch the editor and let it drive itself through the job list."""
    log = log or PROJECT / "Saved" / "pavement_bake.log"
    script = PROJECT / "Saved" / "eb_pavement_render.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(RENDER_SCRIPT.format(
        jobs=str(jobs_path), w=WIDTH, h=HEIGHT, progress=str(PROGRESS)))

    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
    command = [
        str(ENGINE / "Engine/Binaries/Linux/UnrealEditor"),
        str(PROJECT / "CityCore_Paris.uproject"),
        SCENE,
        "-RenderOffscreen", "-Unattended", "-NoSplash", "-NoSound",
        f"-UserDir={PROJECT / 'Saved/User'}",
        "-DDC=NoZenLocalFallback",
        f"-LocalDataCachePath={PROJECT / 'Saved/DerivedDataCache'}",
        # Not -ExecutePythonScript: that means "run it, then quit", and the
        # editor was gone before the first screenshot landed.
        # No inner quotes: the args go to execve directly, and quoting them
        # produced -ExecCmds="py "/path"" which the engine parsed as nothing.
        f"-ExecCmds=py {script}",
        "-stdout",
    ]
    print(" ".join(command), flush=True)
    with log.open("w") as handle:
        return subprocess.call(command, env=env, stdout=handle, stderr=handle)


def write_manifest(jobs: list[dict], out_root: Path) -> dict:
    rows, missing = [], 0
    for job in jobs:
        path = Path(job["image_path"])
        ok = path.exists() and path.stat().st_size > 0
        missing += 0 if ok else 1
        rows.append({**job, "status": "ok" if ok else "missing"})
    out_root.mkdir(parents=True, exist_ok=True)
    with (out_root / "manifest.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return {"rows": len(rows), "missing": missing}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--gpu", default="7")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--manifest", action="store_true")
    args = parser.parse_args()

    out_root = Path(args.out)
    jobs = plan_jobs(out_root)
    jobs_path = out_root / "jobs.json"
    out_root.mkdir(parents=True, exist_ok=True)
    jobs_path.write_text(json.dumps(jobs))
    print(f"planned {len(jobs)} approaches -> {jobs_path}")

    if args.plan and not (args.render or args.manifest):
        return 0
    if args.render:
        code = render(jobs_path, args.gpu)
        print(f"editor exited {code}")
    if args.render or args.manifest:
        print(write_manifest(jobs, out_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
