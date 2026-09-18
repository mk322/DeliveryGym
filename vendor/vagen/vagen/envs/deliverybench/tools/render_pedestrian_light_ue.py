"""Render UE checks for DeliveryBench pedestrian lights.

This replaces the earlier root-level tmp render scripts.  It reads
``progen_world_enriched.json`` pedestrian-light metadata, sends an in-memory
Python script to the SimWorld Studio MCP socket, spawns the pedestrian-light
Blueprint, applies the configured red/green face state, and captures a direct
view from the requested approach.

Example:
    python -m vagen.envs.deliverybench.tools.render_pedestrian_light_ue \
        vagen/envs/deliverybench/maps/small-city-11 --mcp-port 55592 --checks 2
"""

from __future__ import annotations

import argparse
import json
import socket
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from vagen.envs.deliverybench.utils.pedestrian_lights import (
    is_pedestrian_light_node,
)


def _load_lights(scenario_dir: Path) -> List[Dict[str, Any]]:
    with (scenario_dir / "progen_world_enriched.json").open("r", encoding="utf-8") as f:
        world = json.load(f)
    return [node for node in world.get("nodes", []) if is_pedestrian_light_node(node)]


def _send_mcp_script(port: int, script: str, *, timeout_s: float = 120.0) -> Dict[str, Any]:
    msg = json.dumps({"type": "execute_python_script", "params": {"script": script}}) + "\n"
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(float(timeout_s))
    try:
        sock.connect(("127.0.0.1", int(port)))
        sock.sendall(msg.encode("utf-8"))
        data = ""
        while True:
            chunk = sock.recv(4096).decode("utf-8", errors="replace")
            if not chunk:
                break
            data += chunk
            try:
                return json.loads(data)
            except json.JSONDecodeError:
                continue
        return json.loads(data)
    finally:
        sock.close()


def _wait_for_output(path: Path, started_at: float, *, timeout_s: float = 30.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            if path.exists() and path.stat().st_mtime >= started_at - 0.25:
                return
        except FileNotFoundError:
            # UE can briefly create/move the screenshot while the filesystem is
            # being polled; just retry until the timeout.
            pass
        time.sleep(0.25)
    raise TimeoutError(f"UE screenshot did not appear within {timeout_s:.1f}s: {path}")


def _ue_script(
    *,
    lights: Iterable[Mapping[str, Any]],
    selected_light_id: str,
    face: str,
    out_path: Path,
    camera_z: float,
    camera_distance: float,
) -> str:
    lights_json = json.dumps(list(lights))
    selected_json = json.dumps(selected_light_id)
    face_json = json.dumps(face)
    out_json = json.dumps(str(out_path))
    return f"""
import json
import math
import unreal

LIGHTS = json.loads({lights_json!r})
SELECTED_LIGHT_ID = json.loads({selected_json!r})
FACE = json.loads({face_json!r})
OUT = json.loads({out_json!r})
CAMERA_Z = float({camera_z!r})
CAMERA_DISTANCE = float({camera_distance!r})


def log(msg):
    print("VAGEN_PED_LIGHT_UE " + str(msg))


def clean_generated():
    subsys = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    deleted = 0
    for actor in list(subsys.get_all_level_actors()):
        label = actor.get_actor_label()
        cls = actor.get_class().get_name()
        if label.startswith("VAGEN_PedLight_") or label.startswith("VAGEN_PedLightStage_") or cls == "RT_BP_street_light_ped_C":
            subsys.destroy_actor(actor)
            deleted += 1
    log("deleted_count=%d" % deleted)


def cube(label, loc, scale):
    mesh = unreal.load_asset("/Engine/BasicShapes/Cube.Cube")
    mat = unreal.load_asset("/Engine/BasicShapes/BasicShapeMaterial.BasicShapeMaterial")
    actor = unreal.EditorLevelLibrary.spawn_actor_from_class(
        unreal.StaticMeshActor,
        unreal.Vector(float(loc[0]), float(loc[1]), float(loc[2])),
        unreal.Rotator(pitch=0, yaw=0, roll=0),
    )
    actor.set_actor_label(label)
    actor.set_actor_scale3d(unreal.Vector(float(scale[0]), float(scale[1]), float(scale[2])))
    comp = actor.get_component_by_class(unreal.StaticMeshComponent)
    if comp:
        comp.set_static_mesh(mesh)
        if mat:
            comp.set_material(0, mat)
    return actor


def draw_crosswalk_stage(light):
    props = light.get("properties", {{}})
    crossing = props.get("controlled_crossing", {{}})
    center = crossing.get("center", {{}})
    cx = float(center.get("x", props.get("location", {{}}).get("x", 0.0)))
    cy = float(center.get("y", props.get("location", {{}}).get("y", 0.0)))
    prefix = "VAGEN_PedLightStage_" + light.get("id", "selected")
    cube(prefix + "_Road", (cx, cy, -6), (3.6, 2.25, 0.025))
    cube(prefix + "_NearCurb", (cx - 125, cy, 1), (0.22, 2.35, 0.05))
    cube(prefix + "_FarCurb", (cx + 125, cy, 1), (0.22, 2.35, 0.05))
    for i, xoff in enumerate([-84, -56, -28, 0, 28, 56, 84]):
        cube(prefix + "_Zebra_%02d" % i, (cx + xoff, cy, 4), (0.07, 1.60, 0.018))


def look_at(camera, target):
    dx = target.x - camera.x
    dy = target.y - camera.y
    dz = target.z - camera.z
    return unreal.Rotator(
        pitch=math.degrees(math.atan2(dz, math.sqrt(dx * dx + dy * dy))),
        yaw=math.degrees(math.atan2(dy, dx)),
        roll=0.0,
    )


def desired_for_component(component_name, states, active_face):
    name = component_name.lower()
    if active_face in states:
        active_state = states.get(active_face, "red")
        if "_l_" in name or "_r_" in name:
            return active_state
    if "_l_" in name:
        return states.get("left", states.get(active_face, "red"))
    if "_r_" in name:
        return states.get("right", states.get(active_face, "red"))
    return "off"


def set_signal_components(signal, states, target_face):
    target = None
    for comp in signal.get_components_by_class(unreal.StaticMeshComponent):
        name = comp.get_name().lower()
        if "crossing_light_stop" not in name and "crossing_light_walk" not in name:
            continue
        desired = desired_for_component(name, states, target_face)
        visible = ("stop" in name and desired == "red") or ("walk" in name and desired == "green")
        comp.set_visibility(visible, True)
        comp.set_hidden_in_game(not visible, True)
        loc = comp.get_world_location()
        if visible:
            log("visible_component=%s state=%s loc=(%.1f,%.1f,%.1f)" % (
                comp.get_name(), desired, loc.x, loc.y, loc.z
            ))
            if (
                ("_l_" in name and target_face in ("left", "front"))
                or ("_r_" in name and target_face == "right")
            ):
                target = loc
    return target


def spawn_light(light):
    props = light.get("properties", {{}})
    loc = props.get("location", {{}})
    ori = props.get("orientation", {{}})
    asset = props.get("ue_asset_path") or "/Game/RealTimeBench/Traffic/RT_BP_street_light_ped"
    ped_cls = unreal.EditorAssetLibrary.load_blueprint_class(asset)
    if ped_cls is None:
        raise RuntimeError("could not load pedestrian-light Blueprint: " + asset)
    actor = unreal.EditorLevelLibrary.spawn_actor_from_class(
        ped_cls,
        unreal.Vector(float(loc.get("x", 0.0)), float(loc.get("y", 0.0)), float(loc.get("z", 0.0))),
        unreal.Rotator(
            pitch=float(ori.get("pitch", 0.0)),
            yaw=float(ori.get("yaw", 0.0)),
            roll=float(ori.get("roll", 0.0)),
        ),
    )
    actor.set_actor_label("VAGEN_PedLight_" + light.get("id", "unnamed"))
    actor.set_actor_scale3d(unreal.Vector(1.0, 1.0, 1.0))
    states = props.get("face_states", {{}})
    target = set_signal_components(actor, states, FACE if light.get("id") == SELECTED_LIGHT_ID else "")
    loc2 = actor.get_actor_location()
    log("spawned=%s loc=(%.1f,%.1f,%.1f)" % (actor.get_actor_label(), loc2.x, loc2.y, loc2.z))
    return actor, target


clean_generated()
selected = None
target = None
for light in LIGHTS:
    if light.get("id") == SELECTED_LIGHT_ID:
        selected = light
        draw_crosswalk_stage(light)
    _, maybe_target = spawn_light(light)
    if light.get("id") == SELECTED_LIGHT_ID:
        target = maybe_target

if selected is None:
    raise RuntimeError("selected pedestrian light not found: " + SELECTED_LIGHT_ID)
if target is None:
    raise RuntimeError("selected visible pedestrian-light face not found: " + SELECTED_LIGHT_ID + " " + FACE)

face = selected.get("properties", {{}}).get("faces", {{}}).get(FACE, {{}})
facing = face.get("facing", {{}})
fx = float(facing.get("x", 0.0))
fy = float(facing.get("y", 0.0))
camera_loc = unreal.Vector(
    target.x + fx * CAMERA_DISTANCE,
    target.y + fy * CAMERA_DISTANCE,
    CAMERA_Z,
)
camera_rot = look_at(camera_loc, target)
unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).set_level_viewport_camera_info(camera_loc, camera_rot)
log("selected=%s face=%s camera=(%.1f,%.1f,%.1f) target=(%.1f,%.1f,%.1f) rot=(%.2f,%.2f,%.2f)" % (
    SELECTED_LIGHT_ID, FACE,
    camera_loc.x, camera_loc.y, camera_loc.z,
    target.x, target.y, target.z,
    camera_rot.pitch, camera_rot.yaw, camera_rot.roll,
))
task = unreal.AutomationLibrary.take_high_res_screenshot(1920, 1080, OUT)
log("HIGHRES_CALLED %s %s" % (task, OUT))
"""


def _jobs(
    lights: List[Mapping[str, Any]],
    *,
    light_id: Optional[str],
    face: Optional[str],
    checks: int,
    out_dir: Path,
    out: Optional[Path],
) -> List[Tuple[str, str, Path]]:
    def face_names(light: Mapping[str, Any]) -> List[str]:
        faces = light.get("properties", {}).get("faces", {}) or {}
        names = list(faces.keys())
        return names or ["left"]

    if light_id:
        selected = next((light for light in lights if str(light.get("id")) == str(light_id)), None)
        available = face_names(selected) if selected else ["left"]
        chosen_face = face or available[0]
        if out is None:
            out = out_dir / f"ue_pedestrian_light_{light_id}_{chosen_face}.png"
        return [(light_id, chosen_face, out)]

    jobs: List[Tuple[str, str, Path]] = []
    for light in lights[: max(1, checks)]:
        lid = str(light.get("id"))
        for chosen_face in face_names(light):
            jobs.append((lid, chosen_face, out_dir / f"ue_pedestrian_light_{lid}_{chosen_face}.png"))
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario_dir", type=Path)
    parser.add_argument("--mcp-port", type=int, default=55592)
    parser.add_argument("--light-id")
    parser.add_argument("--face")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--checks", type=int, default=2)
    parser.add_argument("--camera-z", type=float, default=120.0)
    parser.add_argument("--camera-distance", type=float, default=350.0)
    args = parser.parse_args()

    scenario_dir = args.scenario_dir.resolve()
    out_dir = (args.out_dir or scenario_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    lights = _load_lights(scenario_dir)
    if not lights:
        raise SystemExit(f"no pedestrian lights found in {scenario_dir / 'progen_world_enriched.json'}")

    rendered = []
    for lid, face, out_path in _jobs(
        lights,
        light_id=args.light_id,
        face=args.face,
        checks=args.checks,
        out_dir=out_dir,
        out=args.out,
    ):
        script = _ue_script(
            lights=lights,
            selected_light_id=lid,
            face=face,
            out_path=out_path,
            camera_z=args.camera_z,
            camera_distance=args.camera_distance,
        )
        started_at = time.time()
        result = _send_mcp_script(args.mcp_port, script)
        status = result.get("status")
        if status != "success" or not result.get("result", {}).get("success", False):
            raise SystemExit(json.dumps(result, indent=2))
        _wait_for_output(out_path, started_at)
        rendered.append(out_path)
        print(f"rendered {lid} {face}: {out_path}")

    print(f"rendered {len(rendered)} UE pedestrian-light checks")


if __name__ == "__main__":
    main()
