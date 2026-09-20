"""Rerender large DeliveryBench city FPV outputs with JSON map assets present.

This helper keeps the same render settings used for the cleaned-floor FPV
datasets, but avoids an empty UE scene by spawning the non-light assets from
``progen_world_enriched.json`` before running the dataset renderer.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

from PIL import Image, ImageDraw, ImageOps

from vagen.envs.deliverybench.tools.render_pedestrian_light_ue import _send_mcp_script


DEFAULT_MAPS = ("large-city-26", "large-city-28", "large-city-30")
DEFAULT_OUT_PROFILE = "main_base_floor_road_full_1280x960_clean_floor"
DEFAULT_UE_ASSETS_JSON = Path(os.environ.get("UE_ASSETS_JSON", "simworld/data/ue_assets.json"))
DEFAULT_CONE_ASSET = "/Game/CityDatabase/blueprints/BP_RoadBlocker.BP_RoadBlocker_C"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _is_pedestrian_light_node(node: Dict[str, Any]) -> bool:
    props = node.get("properties", {}) or {}
    kind = str(props.get("poi_type") or props.get("type") or "").lower()
    inst = str(node.get("instance_name") or "").lower()
    return kind in ("pedestrian_light", "traffic_light") or "street_light_ped" in inst


def _asset_path_for_node(node: Dict[str, Any], assets: Dict[str, Any]) -> str:
    props = node.get("properties", {}) or {}
    explicit = props.get("ue_asset_path")
    if explicit:
        return str(explicit)
    inst = str(node.get("instance_name") or "")
    entry = assets.get(inst, {}) if isinstance(assets, dict) else {}
    if isinstance(entry, dict):
        return str(entry.get("asset_path") or entry.get("path") or "")
    if isinstance(entry, str):
        return entry
    return ""


def _chunks(items: List[Dict[str, Any]], size: int) -> Iterable[List[Dict[str, Any]]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _clear_scene_script() -> str:
    return r"""
import unreal

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
        if any(fragment in cls for fragment in keep_class_fragments) or any(
            label.startswith(prefix) for prefix in keep_label_prefixes
        ):
            kept += 1
            continue
        subsys.destroy_actor(actor)
        deleted += 1
    except Exception as exc:
        print("VAGEN_JSON_ASSET_RERENDER clear_skip actor=%s error=%s" % (actor, exc))
print("VAGEN_JSON_ASSET_RERENDER clear_existing_scene deleted=%d kept=%d" % (deleted, kept))
"""


def _spawn_assets_script(map_name: str, nodes: List[Dict[str, Any]]) -> str:
    payload = json.dumps(nodes)
    return f"""
import json
import unreal

MAP_NAME = {map_name!r}
NODES = json.loads({payload!r})

def log(msg):
    print("VAGEN_JSON_ASSET_RERENDER " + str(msg))

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

prefix = "VAGEN_WorldAsset_%s_" % MAP_NAME
spawned = 0
missing = 0
failed = 0
for node in NODES:
    asset_path = node.get("_asset_path") or ""
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
        log("spawn_failed id=%s asset=%s error=%s" % (node.get("id"), asset_path, exc))
log("spawn_chunk map=%s spawned=%d missing=%d failed=%d" % (MAP_NAME, spawned, missing, failed))
"""


def _spawn_json_assets(
    *,
    scenario_dir: Path,
    map_name: str,
    ue_assets_json: Path,
    mcp_port: int,
    timeout_s: float,
    chunk_size: int,
) -> int:
    world = json.loads((scenario_dir / "progen_world_enriched.json").read_text(encoding="utf-8"))
    assets = json.loads(ue_assets_json.read_text(encoding="utf-8"))
    nodes: List[Dict[str, Any]] = []
    for node in world.get("nodes", []) or []:
        if _is_pedestrian_light_node(node):
            continue
        asset_path = _asset_path_for_node(node, assets)
        if not asset_path:
            continue
        enriched = dict(node)
        enriched["_asset_path"] = asset_path
        nodes.append(enriched)

    result = _send_mcp_script(mcp_port, _clear_scene_script(), timeout_s=timeout_s)
    if result.get("status") != "success" or not result.get("result", {}).get("success", True):
        raise RuntimeError(json.dumps(result, indent=2))

    total = 0
    chunks = list(_chunks(nodes, chunk_size))
    for idx, chunk in enumerate(chunks, start=1):
        result = _send_mcp_script(
            mcp_port,
            _spawn_assets_script(map_name, chunk),
            timeout_s=timeout_s,
        )
        if result.get("status") != "success" or not result.get("result", {}).get("success", True):
            raise RuntimeError(json.dumps(result, indent=2))
        total += len(chunk)
        print(f"[setup] {map_name}: spawned chunk {idx}/{len(chunks)} ({total}/{len(nodes)} requested)")
    return len(nodes)


def _run_renderer(args: argparse.Namespace, map_name: str, scenario_dir: Path, out_dir: Path) -> None:
    cmd = [
        sys.executable,
        "-m",
        "vagen.envs.deliverybench.tools.render_fpv_dataset_ue",
        str(scenario_dir),
        "--out-dir",
        str(out_dir),
        "--mcp-port",
        str(args.mcp_port),
        "--mcp-timeout-s",
        str(args.mcp_timeout_s),
        "--light-camera-mode",
        "waypoint",
        "--normal-camera-backoff-cm",
        "0",
        "--light-camera-backoff-cm",
        "-900",
        "--obstacle-camera-backoff-cm",
        "500",
        "--normal-z",
        "160",
        "--light-z",
        "160",
        "--normal-image-width",
        "1280",
        "--normal-image-height",
        "960",
        "--light-image-width",
        "1280",
        "--light-image-height",
        "960",
        "--light-asset-scale",
        "3",
        "--cone-asset",
        args.cone_asset,
        "--cone-scale",
        "4",
        "--screenshot-timeout-s",
        str(args.screenshot_timeout_s),
        "--no-spawn-map-assets",
    ]
    print(f"[render] {map_name}: running {' '.join(cmd)}")
    subprocess.run(cmd, cwd=_repo_root(), check=True)


def _read_manifest(out_dir: Path) -> List[Dict[str, Any]]:
    rows = []
    manifest = out_dir / "manifest.jsonl"
    if not manifest.exists():
        return rows
    with manifest.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_thumb(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail(size, Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", size, (245, 245, 245))
        x = (size[0] - im.width) // 2
        y = (size[1] - im.height) // 2
        canvas.paste(im, (x, y))
        return canvas


def _write_contact_sheet(items: List[tuple[Path, str]], out_path: Path, *, columns: int = 5) -> None:
    if not items:
        return
    thumb = (256, 192)
    label_h = 28
    margin = 10
    columns = max(1, columns)
    rows = int(math.ceil(len(items) / columns))
    width = margin + columns * (thumb[0] + margin)
    height = margin + rows * (thumb[1] + label_h + margin)
    sheet = Image.new("RGB", (width, height), (235, 235, 235))
    draw = ImageDraw.Draw(sheet)
    for idx, (path, label) in enumerate(items):
        r = idx // columns
        c = idx % columns
        x = margin + c * (thumb[0] + margin)
        y = margin + r * (thumb[1] + label_h + margin)
        try:
            img = _load_thumb(path, thumb)
        except Exception:
            img = Image.new("RGB", thumb, (80, 80, 80))
        sheet.paste(ImageOps.expand(img, border=1, fill=(40, 40, 40)), (x, y))
        draw.text((x, y + thumb[1] + 4), label[:38], fill=(15, 15, 15))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def _make_contact_sheets(out_dir: Path) -> Dict[str, int]:
    rows = _read_manifest(out_dir)
    counts = {"obstacle": 0, "green": 0, "red": 0}
    obstacle: List[tuple[Path, str]] = []
    green: List[tuple[Path, str]] = []
    red: List[tuple[Path, str]] = []
    for row in rows:
        path = Path(row["image_path"])
        label = f'{row.get("waypoint_id", "")} yaw {row.get("stored_yaw", "")}'
        kind = row.get("render_kind")
        if kind == "obstacle":
            obstacle.append((path, label))
        elif kind == "traffic_light" and row.get("signal_state") == "green":
            green.append((path, label))
        elif kind == "traffic_light" and row.get("signal_state") == "red":
            red.append((path, label))
    _write_contact_sheet(obstacle, out_dir / "obstacle_contact_sheet.png", columns=5)
    _write_contact_sheet(green, out_dir / "traffic_light_green_contact_sheet.png", columns=6)
    _write_contact_sheet(red, out_dir / "traffic_light_red_contact_sheet.png", columns=6)
    counts["obstacle"] = len(obstacle)
    counts["green"] = len(green)
    counts["red"] = len(red)
    return counts


def _verify(out_dir: Path, marker: Path) -> Dict[str, int]:
    rows = _read_manifest(out_dir)
    fresh = 0
    missing = 0
    plain = 0
    light = 0
    obstacle = 0
    marker_mtime = marker.stat().st_mtime if marker.exists() else 0.0
    for row in rows:
        path = Path(row["image_path"])
        if not path.exists():
            missing += 1
            continue
        if path.stat().st_mtime >= marker_mtime:
            fresh += 1
        kind = row.get("render_kind")
        if kind == "plain":
            plain += 1
        elif kind == "traffic_light":
            light += 1
        elif kind == "obstacle":
            obstacle += 1
    return {
        "rows": len(rows),
        "fresh": fresh,
        "missing": missing,
        "plain": plain,
        "traffic_light": light,
        "obstacle": obstacle,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("maps", nargs="*", default=list(DEFAULT_MAPS))
    parser.add_argument("--mcp-port", type=int, default=55566)
    parser.add_argument("--mcp-timeout-s", type=float, default=1800.0)
    parser.add_argument("--screenshot-timeout-s", type=float, default=240.0)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--ue-assets-json", type=Path, default=DEFAULT_UE_ASSETS_JSON)
    parser.add_argument("--cone-asset", default=DEFAULT_CONE_ASSET)
    args = parser.parse_args()

    root = _repo_root()
    summaries = {}
    for map_name in args.maps:
        scenario_dir = root / "vagen" / "envs" / "deliverybench" / "maps" / map_name
        out_dir = (
            root
            / "vagen"
            / "envs"
            / "deliverybench"
            / "deliverybench_fpv"
            / map_name
            / DEFAULT_OUT_PROFILE
        )
        marker = Path("/tmp") / f"{map_name}_json_asset_rerender_marker"
        marker.write_text(str(time.time()) + "\n", encoding="utf-8")
        print(f"[setup] {map_name}: clearing scene and spawning JSON assets")
        requested_assets = _spawn_json_assets(
            scenario_dir=scenario_dir,
            map_name=map_name,
            ue_assets_json=args.ue_assets_json,
            mcp_port=args.mcp_port,
            timeout_s=args.mcp_timeout_s,
            chunk_size=args.chunk_size,
        )
        print(f"[setup] {map_name}: requested {requested_assets} JSON assets")
        _run_renderer(args, map_name, scenario_dir, out_dir)
        sheets = _make_contact_sheets(out_dir)
        verify = _verify(out_dir, marker)
        summaries[map_name] = {"requested_assets": requested_assets, **verify, **sheets}
        print(f"[verify] {map_name}: {json.dumps(summaries[map_name], sort_keys=True)}")
    print("[done] " + json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
