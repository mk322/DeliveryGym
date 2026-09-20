"""Bake the street-view album for a procgen city, SimWorld edition.

The Paris albums were shot inside a hand-authored UE scene, so their bakers
(`bake_pavement_views.py` and kin) could assume the world already existed.
The nine procgen cities have no scene: the city is *spawned* -- every
building, road piece and POI placed from ``progen_world_enriched.json`` via
the SimWorld asset table -- and then photographed. This does both, with the
same camera, the same per-edge job plan and the same manifest schema as the
Paris street album, so a second city plugs into the runtime without the
runtime learning anything new.

The coordinate question that decides whether this can work at all was
measured before it was assumed: the world json's spawn locations span
[-19956, 79859] x [-19845, 39845] cm and ``build_road_network``'s nodes span
[-20000, 80000] x [-20000, 40000] -- the same frame, no transform. Metres in
``roads.json`` are the compiler's own x100.

    python tools/ue/bake_city_streets.py --map small-city-11 --plan
    python tools/ue/bake_city_streets.py --map small-city-11 --render --gpu 5 --limit 24
    python tools/ue/bake_city_streets.py --map small-city-11 --manifest

Render mechanics are inherited wholesale from bake_pavement_views.py, four
failed attempts and all: a live editor via ``-ExecCmds="py ..."`` (never
``-ExecutePythonScript``, which quits before the first tick), a slate
post-tick callback taking one frame per tick, and the file's existence -- not
the call's return -- as the definition of done.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from embodiedbench.compiler.road_network import build_road_network  # noqa: E402

MAPS_DIR = REPO / "vendor/vagen/vagen/envs/deliverybench/maps"
ENGINE = Path(os.environ.get("UE_ENGINE", "/opt/UnrealEngine-5.8"))
PROJECT = Path(os.environ.get("SIMWORLD_PROJECT", "/opt/SimWorld"))
UPROJECT = PROJECT / "SimWorld.uproject"
# The SimWorld asset table: instance_name -> blueprint path. A copy lives
# beside the bakes so the bake does not depend on someone else's checkout
# staying readable.
UE_ASSETS = Path(os.environ.get(
    "UE_ASSETS_JSON", str(PROJECT / "ue_assets.json")))

WIDTH, HEIGHT = 640, 480
CAMERA_Z_CM = 160.0            # pedestrian eye level, same as every album


def bearing_deg(a: tuple[float, float], b: tuple[float, float]) -> float:
    """UE yaw: degrees clockwise from +X."""
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))


# How far onto the footway a pavement camera stands. Same constant as the
# Paris pavement album (tools/ue/bake_pavement_views.py): half the street's
# own width puts the camera on the kerbstone, the margin puts it where a
# person walks.
KERB_MARGIN_CM = 60.0


def plan_jobs(map_name: str, out_root: Path,
              viewpoint: str = "street") -> list[dict]:
    """One job per directed street edge, camera at the node.

    The schema is the Paris albums', field for field: the runtime joins on
    (x_cm, y_cm, yaw) and reads street/toward_street for captions, and a
    second city that renames anything is a second format, not a second city.
    ``viewpoint="street"`` shoots from the carriageway centreline;
    ``"pavement"`` moves the camera perpendicular to the direction of travel,
    onto the right-hand footway, by half the street's width plus the margin --
    the same numbers, fields and right-hand rule as the Paris pavement album,
    because the on-foot embodiment stands there and nowhere else.
    """
    network = build_road_network(MAPS_DIR / map_name, map_name=map_name)
    jobs: list[dict] = []
    for node_id, node in sorted(network.nodes.items()):
        here = (node.x_cm, node.y_cm)
        street = network.streets[node.street_index]
        offset = street.width_cm / 2.0 + KERB_MARGIN_CM
        for neighbour in sorted(node.neighbours):
            other = network.nodes[neighbour]
            yaw = bearing_deg(here, (other.x_cm, other.y_cm))
            job = {
                "waypoint_id": node_id,
                "toward_node": neighbour,
                "yaw": round(yaw, 1),
                "x_cm": round(here[0], 2),
                "y_cm": round(here[1], 2),
                "z_cm": CAMERA_Z_CM,
                "street": street.name,
                "toward_street": network.streets[other.street_index].name,
                "edge_length_m": round(
                    math.dist(here, (other.x_cm, other.y_cm)) / 100.0, 1),
                "image_path": str(out_root / "images" / node_id
                                  / f"toward_{neighbour}.png"),
                "map_name": map_name,
                "render_kind": "street_view",
                "image_size": [WIDTH, HEIGHT],
            }
            if viewpoint == "pavement":
                # Right of the direction of travel: UE yaw is degrees
                # clockwise from +X, so the right-hand normal is yaw + 90.
                right = math.radians(yaw + 90.0)
                job.update({
                    "x_cm": round(here[0] + offset * math.cos(right), 2),
                    "y_cm": round(here[1] + offset * math.sin(right), 2),
                    "centreline_x_cm": round(here[0], 2),
                    "centreline_y_cm": round(here[1], 2),
                    "kerb_offset_cm": offset,
                    "street_width_cm": street.width_cm,
                    "render_kind": "street_view_pavement",
                    "viewpoint": "pavement",
                })
            jobs.append(job)
    return jobs


# The spawn preamble, adapted from the recorded job that rendered the original
# small-city albums (vendor/.../deliverybench_fpv/*/\_last_fpv_render_ue_job.py).
# Idempotent by actor-label prefix, so a resumed bake does not spawn the city
# twice on top of itself. Pedestrian lights are skipped here exactly as the
# original did: the light album is its own bake with its own phases.
SPAWN_PREAMBLE = r'''
import json, math, os, time, unreal

MAP_NAME = {map_name!r}
SKIP_INSTANCES = set({skip!r})
DROP_FOG = {drop_fog!r}
BASE_PLATE = {base_plate!r}
WORLD_JSON = {world_json!r}
ASSETS_JSON = {assets_json!r}
LOG = open({progress!r}, "w", buffering=1)

def note(message):
    LOG.write("%s %s\n" % (time.strftime("%H:%M:%S"), message))

# take_high_res_screenshot only queues a request; the queue is serviced when a
# viewport actually redraws, and an offscreen editor sitting idle does not
# redraw. Twelve requests produced exactly four files -- the first, plus the
# last three flushed at shutdown, all stamped the same second. The original
# pipeline never hit this because it ran one editor process per frame and let
# process exit do the flushing. One invalidate per tick makes the redraw
# happen on purpose instead.
def _find_invalidator():
    try:
        sub = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
        if sub and hasattr(sub, "editor_invalidate_viewports"):
            return sub.editor_invalidate_viewports
    except Exception:
        pass
    try:
        if hasattr(unreal.EditorLevelLibrary, "editor_invalidate_viewports"):
            return unreal.EditorLevelLibrary.editor_invalidate_viewports
    except Exception:
        pass
    return lambda: None

INVALIDATE = _find_invalidator()

def load_bp(asset_path):
    for loader in (
        lambda p: unreal.EditorAssetLibrary.load_blueprint_class(p),
        lambda p: unreal.load_class(None, p),
        lambda p: getattr(unreal.load_asset(p), "generated_class", None),
    ):
        try:
            cls = loader(asset_path)
            if cls:
                return cls
        except Exception:
            pass
    return None

def is_ped_light(node):
    props = node.get("properties", {{}}) or {{}}
    return (props.get("poi_type") == "pedestrian_light"
            or props.get("type") == "pedestrian_light"
            or str(node.get("instance_name") or "").lower() == "rt_bp_street_light_ped")

def asset_path_for(node, assets):
    props = node.get("properties", {{}}) or {{}}
    if props.get("ue_asset_path"):
        return props["ue_asset_path"]
    entry = assets.get(str(node.get("instance_name") or ""), {{}})
    if isinstance(entry, dict):
        return entry.get("asset_path") or entry.get("path")
    return entry if isinstance(entry, str) else None

def clear_default_scene(subsys):
    keep = ("WorldSettings", "LevelScriptActor", "DirectionalLight", "SkyLight",
            "SkyAtmosphere", "ExponentialHeightFog", "AtmosphericFog",
            "VolumetricCloud", "PostProcessVolume", "PlayerStart",
            "CameraActor", "CineCameraActor")
    if DROP_FOG:
        keep = tuple(k for k in keep if "Fog" not in k and "Cloud" not in k)
    # The ground itself, kept by label -- the same four prefixes every original
    # deliverybench render tool kept, because they are the demo level's floor
    # and road surface. Deleting them is how eleven attempts stood a city on a
    # white void.
    keep_labels = ("Floor", "Road_", "RoadY_", "CrossPatch")
    deleted = kept = 0
    for actor in list(subsys.get_all_level_actors()):
        try:
            cls = actor.get_class().get_name()
            label = actor.get_actor_label()
            if any(k in cls for k in keep) \
                    or any(label.startswith(k) for k in keep_labels):
                kept += 1
                continue
            if label.startswith("EB_City_"):
                continue
            subsys.destroy_actor(actor)
            deleted += 1
        except Exception:
            pass
    note("cleared_default_scene deleted=%d kept=%d" % (deleted, kept))

STATE_DIAG = set()

def spawn_city():
    # The asset registry scans asynchronously at editor start, and an
    # -ExecCmds script runs before it finishes: every load_blueprint_class
    # then returns None and the whole city reports "missing" while sitting
    # on disk. Wait for the scan; it is the difference between spawned=0
    # missing=113 and a city.
    try:
        registry = unreal.AssetRegistryHelpers.get_asset_registry()
        note("asset_registry loading=%s -- waiting" % registry.is_loading_assets())
        registry.wait_for_completion()
        note("asset_registry ready")
    except Exception as error:
        note("asset_registry wait failed: %r" % error)
    subsys = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    prefix = "EB_City_%s_" % MAP_NAME
    for actor in subsys.get_all_level_actors():
        try:
            if actor.get_actor_label().startswith(prefix):
                note("city_already_spawned")
                return
        except Exception:
            pass
    clear_default_scene(subsys)
    # The ground itself. progen placed every building against base_map
    # 'map_1' -- a 1000 m blank plate with roads on a 150 m pitch -- and that
    # plate lives in the content as one blueprint (SM_Template_Map_Floor and
    # the roadlines meshes inside blank_map-roads150_C). The original
    # deliverybench renders kept actors labelled Floor/Road_/roadlines from
    # their base level; SimBlank has none, which is why the first full render
    # of this city stood on a white void.
    base_cls = (load_bp("/Game/blank_map-roads150.blank_map-roads150_C")
                if BASE_PLATE else None)
    if base_cls is not None:
        try:
            base = unreal.EditorLevelLibrary.spawn_actor_from_class(
                base_cls, unreal.Vector(0.0, 0.0, 0.0), unreal.Rotator())
            base.set_actor_label(prefix + "BaseMap")
            note("base_map spawned (blank_map-roads150)")
        except Exception as error:
            note("base_map spawn FAILED: %r" % error)
    elif BASE_PLATE:
        note("base_map class missing: /Game/blank_map-roads150")
    world = json.load(open(WORLD_JSON))
    assets = json.load(open(ASSETS_JSON))
    spawned = missing = failed = 0
    for node in world.get("nodes", []) or []:
        if is_ped_light(node):
            continue
        if str(node.get("instance_name")) in SKIP_INSTANCES:
            continue
        path = asset_path_for(node, assets)
        cls = load_bp(path) if path else None
        if cls is None:
            missing += 1
            if missing <= 5:
                note("miss inst=%r path=%r exists=%s" % (
                    str(node.get("instance_name")), path,
                    unreal.EditorAssetLibrary.does_asset_exist(path)
                    if path else "n/a"))
            continue
        props = node.get("properties", {{}}) or {{}}
        loc = props.get("location", {{}}) or {{}}
        ori = props.get("orientation", {{}}) or {{}}
        scale = props.get("scale", {{}}) or {{}}
        try:
            actor = unreal.EditorLevelLibrary.spawn_actor_from_class(
                cls,
                unreal.Vector(float(loc.get("x", 0.0)), float(loc.get("y", 0.0)),
                              float(loc.get("z", 0.0))),
                unreal.Rotator(pitch=float(ori.get("pitch", 0.0)),
                               yaw=float(ori.get("yaw", 0.0)),
                               roll=float(ori.get("roll", 0.0))))
            actor.set_actor_label(prefix + str(node.get("id") or spawned))
            actor.set_actor_scale3d(unreal.Vector(
                float(scale.get("x", 1.0)), float(scale.get("y", 1.0)),
                float(scale.get("z", 1.0))))
            spawned += 1
        except Exception as error:
            failed += 1
            note("spawn_failed id=%s asset=%s error=%r"
                 % (node.get("id"), path, error))
    note("city_spawned spawned=%d missing=%d failed=%d" % (spawned, missing, failed))
    # What is the ground actually made of? Name every static-mesh component
    # and its materials for the plate and one road piece -- the answer to a
    # grey road is in this list, not in another theory.
    def dump_actor(actor, tag):
        try:
            comps = actor.get_components_by_class(unreal.StaticMeshComponent)
            note("diag %s: %d mesh comps" % (tag, len(comps)))
            for comp in list(comps)[:12]:
                mesh = comp.static_mesh
                mats = []
                for mi in range(comp.get_num_materials()):
                    m = comp.get_material(mi)
                    mats.append(m.get_path_name() if m else "None")
                note("diag %s comp=%s mesh=%s mats=%s" % (
                    tag, comp.get_name(),
                    mesh.get_path_name() if mesh else None, mats))
        except Exception as error:
            note("diag %s failed: %r" % (tag, error))
    for actor in subsys.get_all_level_actors():
        try:
            label = actor.get_actor_label()
        except Exception:
            continue
        if label == prefix + "BaseMap":
            dump_actor(actor, "basemap")
        elif "Road1" in str(actor.get_class().get_name()) and "dumped_road" not in STATE_DIAG:
            STATE_DIAG.add("dumped_road")
            dump_actor(actor, "road1")

spawn_city()
'''

# The street album's capture tail; other albums append their own to the same
# preamble.
STREET_TAIL = r'''
WARMUP_S = {warmup!r}
JOBS = json.load(open({jobs!r}))
W, H = {w}, {h}

STATE = {{"i": 0, "pending": None, "waited": 0, "done": 0, "failed": 0,
          "busy": False, "warmup_start": None}}

# take_high_res_screenshot pumps slate internally, and slate runs post-tick
# callbacks -- so the call re-enters this function before it returns. The
# evidence: nine requests fired inside one visual tick, their notes unwound
# in reverse (stack) order all labelled i=9, and eight of the nine stacked
# requests produced no file, no error and no timeout. With the guard the
# recursive entries bounce off and the one-shot-in-flight design is real.
def _tick(delta):
    if STATE["busy"]:
        return
    STATE["busy"] = True
    try:
        _tick_body(delta)
    finally:
        STATE["busy"] = False

def _tick_body(delta):
    # Shader warm-up. The editor JIT-compiles material shaders after the city
    # spawns, and while a material's shaders are compiling UE renders it with
    # the default grey -- the whole road read as untextured because the
    # compile thread only finished 16 seconds AFTER the last capture. Hold
    # the captures (while forcing redraws, which also streams the virtual
    # textures) until the compilers have had their window.
    if WARMUP_S > 0:
        if STATE["warmup_start"] is None:
            STATE["warmup_start"] = time.time()
            note("warmup: holding captures %ds for shader compilation" % WARMUP_S)
        if time.time() - STATE["warmup_start"] < WARMUP_S:
            try:
                INVALIDATE()
            except Exception:
                pass
            return
        if STATE["warmup_start"] != -1:
            note("warmup done")
            STATE["warmup_start"] = -1
    if STATE["pending"] is not None:
        path = STATE["pending"]
        if os.path.exists(path) and os.path.getsize(path) > 0:
            STATE["pending"], STATE["waited"] = None, 0
            STATE["done"] += 1
            note("wrote %s" % path)
            if STATE["done"] % 25 == 0:
                note("done=%d failed=%d of %d"
                     % (STATE["done"], STATE["failed"], len(JOBS)))
        else:
            try:
                INVALIDATE()
            except Exception:
                pass
            STATE["waited"] += 1
            if STATE["waited"] % 120 == 0:
                note("still waiting (%d ticks) for %s" % (STATE["waited"], path))
            if STATE["waited"] > 900:
                note("timeout %s" % path)
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
        rot = unreal.Rotator(pitch=0.0, yaw=job["yaw"], roll=0.0)
        try:
            unreal.get_editor_subsystem(
                unreal.UnrealEditorSubsystem).set_level_viewport_camera_info(loc, rot)
            task = unreal.AutomationLibrary.take_high_res_screenshot(W, H, out)
            note("shot %d/%d requested %s task=%r"
                 % (STATE["i"], len(JOBS), out, task))
        except Exception as error:
            STATE["failed"] += 1
            note("ERROR on %s: %r" % (out, error))
            return
        STATE["pending"] = out
        return
    note("FINISHED done=%d failed=%d" % (STATE["done"], STATE["failed"]))
    LOG.close()
    unreal.SystemLibrary.quit_editor()

note("starting: %d jobs" % len(JOBS))
unreal.register_slate_post_tick_callback(_tick)
'''

SPAWN_AND_RENDER_SCRIPT = SPAWN_PREAMBLE + STREET_TAIL


def render(map_name: str, jobs_path: Path, gpu: str, log: Path,
           progress: Path, skip: list[str] | None = None,
           drop_fog: bool = False, warmup: int = 240,
           level: str = "/Game/Maps/demo_2",
           base_plate: bool = False) -> int:
    script = PROJECT / "Saved" / f"eb_bake_{map_name}.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(SPAWN_AND_RENDER_SCRIPT.format(
        map_name=map_name,
        skip=sorted(skip or []),
        drop_fog=bool(drop_fog),
        base_plate=bool(base_plate),
        warmup=int(warmup),
        world_json=str(MAPS_DIR / map_name / "progen_world_enriched.json"),
        assets_json=str(UE_ASSETS),
        jobs=str(jobs_path), w=WIDTH, h=HEIGHT, progress=str(progress)))
    return launch_editor(script, gpu, log, progress, level=level)


def launch_editor(script: Path, gpu: str, log: Path, progress: Path,
                  level: str = "/Game/Maps/demo_2") -> int:
    """Drive one editor run of ``script``: verified GPU attach, silent-hang
    babysitting, and retries. Shared by every album baker for this project."""

    # Steering an editor onto one specific card took three wrong theories:
    # CUDA_VISIBLE_DEVICES by index filters Vulkan but through CUDA's
    # FASTEST_FIRST order, which on identical cards is arbitrary; by UUID it
    # does not filter Vulkan at all (measured: the editor landed on the full
    # card regardless); and -graphicsadapter indexes Vulkan's own enumeration,
    # whose order nothing guarantees. So nothing is trusted: the editor is
    # launched with an adapter index, the card it actually attached to is read
    # back from nvidia-smi by UUID, and a wrong card is killed and the next
    # index tried. Empiricism, because every prior belief here was wrong.
    import subprocess as sp
    import time as _t

    def _uuid_of(index: str) -> str | None:
        try:
            out = sp.run(["nvidia-smi", "--query-gpu=index,uuid",
                          "--format=csv,noheader"],
                         capture_output=True, text=True, timeout=10).stdout
            return dict(l.split(", ") for l in out.strip().splitlines()).get(str(index))
        except Exception:
            return None

    target_uuid = _uuid_of(gpu) if str(gpu).isdigit() else None
    print(f"target: nvidia-smi index {gpu} = {target_uuid}")

    def _command(adapter: int) -> list[str]:
        return [
            str(PROJECT / "Binaries/Linux/SimWorldEditor"),
            str(UPROJECT),
            os.environ.get("UE_BAKE_LEVEL", level),
            # The spear plugin binds an RPC server on port 30000 at module
            # startup, every other SimWorld editor on the machine wants the
            # same port, and after four failed binds its "graceful exit" path
            # SIGSEGVs -- which presented as three different crashes before
            # the port line was read. Role=None skips the bind entirely; it
            # is how the resident editor fleet coexists, read straight off a
            # live process's command line.
            "-SpServicesRole=None",
            "-RenderOffscreen", "-Unattended", "-NoSplash", "-NoSound",
            f"-UserDir={PROJECT / 'Saved/User'}",
            "-DDC=NoZenLocalFallback",
            f"-LocalDataCachePath={os.environ.get('UE_DDC', str(PROJECT / 'Saved/DDC'))}",
            f"-graphicsadapter={adapter}",
            f"-ExecCmds=py {script}",
            "-stdout",
        ]

    env = dict(os.environ)
    env.pop("CUDA_VISIBLE_DEVICES", None)   # full Vulkan enumeration, on purpose

    order = [int(gpu)] + [i for i in range(8) if i != int(gpu)] \
        if str(gpu).isdigit() else list(range(8))
    # Three rounds over the adapter order: a startup hang kills one launch,
    # not the bake.
    order = order * 3
    _LAUNCH_T0 = _t.time()
    for adapter in order:
        command = _command(adapter)
        print(" ".join(command), flush=True)
        with log.open("w") as handle:
            proc = sp.Popen(command, env=env, stdout=handle, stderr=handle)
            # Wait for the editor to attach to a card, then read WHICH.
            attached = None
            for _ in range(240):            # up to 4 min to appear on a GPU
                _t.sleep(1)
                if proc.poll() is not None:
                    break
                out = sp.run(["nvidia-smi",
                              "--query-compute-apps=pid,gpu_uuid",
                              "--format=csv,noheader"],
                             capture_output=True, text=True).stdout
                for line in out.strip().splitlines():
                    pid, _, uu = line.partition(", ")
                    if pid.strip() == str(proc.pid):
                        attached = uu.strip()
                        break
                if attached:
                    break
            if attached and target_uuid and attached != target_uuid:
                print(f"adapter {adapter} attached to {attached}, want "
                      f"{target_uuid}; killing and trying the next index",
                      flush=True)
                proc.kill(); proc.wait()
                continue
            if attached:
                print(f"adapter {adapter} attached to the requested card; "
                      f"rendering", flush=True)
            # Babysit the run: twice now the editor has hung silently in
            # early module startup (UdpMessaging once, LiveLinkHub once) --
            # zero CPU, zero log growth, forever. One such hang cost 3.5
            # unattended hours. A stalled editor is killed and the next
            # launch attempt made; progress.log going fresh ends the special
            # watch because from there the capture loop reports for itself.
            last_size = -1
            stalled = 0
            fresh = False
            while proc.poll() is None:
                _t.sleep(30)
                if not fresh:
                    try:
                        fresh = progress.stat().st_mtime > _LAUNCH_T0
                    except OSError:
                        pass
                if fresh:
                    continue
                size = log.stat().st_size if log.exists() else 0
                if size == last_size:
                    stalled += 1
                    if stalled >= 16:       # 8 minutes of a silent log
                        print(f"adapter {adapter}: editor silent for 8 min "
                              f"during startup; killing and relaunching",
                              flush=True)
                        proc.kill(); proc.wait()
                        break
                else:
                    stalled, last_size = 0, size
            if proc.poll() is not None:
                rc = proc.wait()
                if fresh or rc == 0:
                    return rc
                print(f"adapter {adapter}: editor exited {rc} before the "
                      f"python stage; retrying", flush=True)
                continue
    print("no launch attempt survived to the python stage", flush=True)
    return 1


def write_manifest(jobs: list[dict], out_root: Path) -> None:
    ok = 0
    with (out_root / "manifest.jsonl").open("w") as handle:
        for job in jobs:
            exists = (Path(job["image_path"]).exists()
                      and Path(job["image_path"]).stat().st_size > 0)
            ok += int(exists)
            handle.write(json.dumps({**job,
                                     "status": "ok" if exists else "failed",
                                     "error": None if exists else "missing"})
                         + "\n")
    print(f"manifest: {ok}/{len(jobs)} frames present")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--map", required=True)
    parser.add_argument("--viewpoint", choices=("street", "pavement"),
                        default="street")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--manifest", action="store_true")
    parser.add_argument("--gpu", default="5")
    parser.add_argument("--limit", type=int, default=0,
                        help="render only the first N jobs (a trial)")
    parser.add_argument("--skip", action="append", default=[],
                        help="instance_name values not to spawn (diagnosis)")
    parser.add_argument("--drop-fog", action="store_true",
                        help="delete fog/cloud actors from the base level")
    parser.add_argument("--warmup", type=int, default=240,
                        help="seconds to hold captures for shader compilation")
    parser.add_argument("--level", default="/Game/Maps/demo_2",
                        help="level to load; demo_2 is the map the original"
                             " renders stood on, floor and sky included")
    parser.add_argument("--base-plate", action="store_true",
                        help="spawn blank_map-roads150 (for levels with no"
                             " ground of their own)")
    args = parser.parse_args()

    if args.out:
        out_root = args.out
    elif args.viewpoint == "pavement":
        # The runtime discovers sibling albums by these exact directory names
        # (vagen_courier_env: paris_streets_pavement/<map>), Paris legacy
        # prefix and all. A different name here is an album it cannot see.
        out_root = Path(os.environ.get("ALBUMS_DIR", "/data/albums")) / "paris_streets_pavement" / args.map
    else:
        out_root = Path(os.environ.get("ALBUMS_DIR", "/data/albums")) / "city_streets_v1" / args.map
    out_root.mkdir(parents=True, exist_ok=True)
    jobs = plan_jobs(args.map, out_root, viewpoint=args.viewpoint)
    jobs_path = out_root / "jobs.json"

    if args.plan or not jobs_path.exists():
        jobs_path.write_text(json.dumps(jobs, indent=1))
        streets = len({j["street"] for j in jobs})
        print(f"planned {len(jobs)} directed views over "
              f"{len({j['waypoint_id'] for j in jobs})} nodes, {streets} streets"
              f" -> {jobs_path}")

    if args.render:
        subset = jobs[:args.limit] if args.limit else jobs
        trial_path = out_root / ("jobs_trial.json" if args.limit else "jobs.json")
        if args.limit:
            trial_path.write_text(json.dumps(subset, indent=1))
        rc = render(args.map, trial_path, args.gpu,
                    log=out_root / "bake.log",
                    progress=out_root / "progress.log", skip=args.skip,
                    drop_fog=args.drop_fog, warmup=args.warmup,
                    level=args.level, base_plate=args.base_plate)
        print(f"editor exited {rc}; progress in {out_root/'progress.log'}")

    if args.manifest:
        write_manifest(jobs, out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
