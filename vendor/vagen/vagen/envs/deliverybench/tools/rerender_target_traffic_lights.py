"""Targeted DeliveryBench traffic-light rerender helper.

Use this when a small set of pedestrian-light FPV images needs to be replaced
after orientation/color fixes. The helper loads the map JSON assets into a clean
Main.umap scene, renders only the requested green/red images, and rebuilds the
contact sheets for the output folder.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from vagen.envs.deliverybench.tools.render_fpv_dataset_ue import (
    DEFAULT_LIGHT_RENDER_XY_SCALE,
    _to_waypoint_light_job,
    build_jobs,
)
from vagen.envs.deliverybench.tools.render_pedestrian_light_ue import (
    _send_mcp_script,
    _wait_for_output,
)
from vagen.envs.deliverybench.tools.render_waypoint_pedestrian_light_ue import (
    _ue_script as _wp_ue_script,
)
from vagen.envs.deliverybench.tools.rerender_large_city_json_assets import (
    DEFAULT_OUT_PROFILE,
    DEFAULT_UE_ASSETS_JSON,
    _make_contact_sheets,
    _repo_root,
    _spawn_json_assets,
)


TARGETS: Dict[str, List[Tuple[str, int]]] = {
    "small-city-15": [
        ("int_005", 90),
        ("int_018", 270),
        ("int_016", 270),
        ("int_014", 270),
        ("int_025", 90),
        ("int_027", 90),
        ("int_023", 90),
        ("int_033", 270),
    ],
    "medium-city-18": [
        ("int_009", 270),
        ("int_018", 270),
        ("int_021", 90),
        ("int_027", 270),
        ("int_040", 90),
        ("int_045", 270),
        ("int_019", 90),
        ("int_011", 90),
        ("int_003", 90),
    ],
    "medium-city-20": [
        ("int_009", 270),
        ("int_018", 270),
        ("int_021", 90),
        ("int_027", 270),
        ("int_042", 90),
        ("int_049", 270),
        ("int_040", 90),
        ("int_019", 90),
        ("int_011", 90),
        ("int_003", 90),
    ],
    "medium-city-22": [
        ("int_008", 90),
        ("int_018", 270),
        ("int_022", 90),
        ("int_024", 90),
        ("int_039", 90),
        ("int_032", 270),
        ("int_016", 270),
        ("int_004", 90),
    ],
}


def _out_dir(root: Path, map_name: str) -> Path:
    return (
        root
        / "vagen"
        / "envs"
        / "deliverybench"
        / "deliverybench_fpv"
        / map_name
        / DEFAULT_OUT_PROFILE
    )


def _scenario_dir(root: Path, map_name: str) -> Path:
    return root / "vagen" / "envs" / "deliverybench" / "maps" / map_name


def _select_jobs(scenario_dir: Path, out_dir: Path, targets: Iterable[Tuple[str, int]]) -> List[Any]:
    def norm_waypoint_id(waypoint_id: str) -> str:
        kind, num = str(waypoint_id).rsplit("_", 1)
        return f"{kind}_{int(num)}"

    wanted = {
        (norm_waypoint_id(wp), int(yaw), state)
        for wp, yaw in targets
        for state in ("green", "red")
    }
    view_jobs, _, _ = build_jobs(
        scenario_dir,
        out_dir,
        light_render_xy_scale=DEFAULT_LIGHT_RENDER_XY_SCALE,
    )
    selected = [
        job
        for job in view_jobs
        if job.render_kind == "traffic_light"
        and (job.waypoint_id, int(job.stored_yaw), job.signal_state) in wanted
    ]
    found = {(job.waypoint_id, int(job.stored_yaw), job.signal_state) for job in selected}
    missing = sorted(wanted - found)
    if missing:
        raise SystemExit(f"missing requested traffic-light jobs in {scenario_dir.name}: {missing}")
    selected.sort(key=lambda j: (j.waypoint_id, int(j.stored_yaw), j.signal_state))
    return selected


def _render_jobs(
    *,
    map_name: str,
    jobs: List[Any],
    out_dir: Path,
    scenario_dir: Path,
    mcp_port: int,
    mcp_timeout_s: float,
    screenshot_timeout_s: float,
) -> None:
    script_path = out_dir / "_last_target_light_rerender_ue_job.py"
    world_json = scenario_dir / "progen_world_enriched.json"
    for idx, job in enumerate(jobs, start=1):
        waypoint_job = _to_waypoint_light_job(job, map_name)
        script = _wp_ue_script(
            light=job.light_node or {},
            job=waypoint_job,
            camera_mode="waypoint",
            face_camera_sign=1.0,
            camera_backoff_cm=-900.0,
            camera_distance=1600.0,
            camera_fov=90.0,
            world_json_path=world_json,
            ue_assets_json_path=DEFAULT_UE_ASSETS_JSON,
            spawn_map_assets=False,
            clear_existing_scene=False,
            debug_markers=False,
            camera_z=160.0,
            image_width=1280,
            image_height=960,
            ped_light_xy_scale=1.0,
            light_asset_scale=3.0,
        )
        out_path = Path(job.image_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text(script, encoding="utf-8")
        started = time.time()
        result = _send_mcp_script(
            mcp_port,
            f"exec(open({str(script_path)!r}, 'r', encoding='utf-8').read())",
            timeout_s=mcp_timeout_s,
        )
        if result.get("status") != "success" or not result.get("result", {}).get("success", True):
            raise SystemExit(json.dumps(result, indent=2))
        _wait_for_output(out_path, started, timeout_s=screenshot_timeout_s)
        print(f"[render] {map_name}: {idx}/{len(jobs)} {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("maps", nargs="*", default=list(TARGETS))
    parser.add_argument("--mcp-port", type=int, default=55566)
    parser.add_argument("--mcp-timeout-s", type=float, default=1800.0)
    parser.add_argument("--screenshot-timeout-s", type=float, default=240.0)
    parser.add_argument("--chunk-size", type=int, default=4)
    args = parser.parse_args()

    root = _repo_root()
    summary = {}
    for map_name in args.maps:
        scenario_dir = _scenario_dir(root, map_name)
        out_dir = _out_dir(root, map_name)
        marker = Path("/tmp") / f"{map_name}_target_light_rerender_marker"
        marker.write_text(str(time.time()) + "\n", encoding="utf-8")
        print(f"[setup] {map_name}: clearing scene and spawning JSON assets")
        requested_assets = _spawn_json_assets(
            scenario_dir=scenario_dir,
            map_name=map_name,
            ue_assets_json=DEFAULT_UE_ASSETS_JSON,
            mcp_port=args.mcp_port,
            timeout_s=args.mcp_timeout_s,
            chunk_size=args.chunk_size,
        )
        jobs = _select_jobs(scenario_dir, out_dir, TARGETS[map_name])
        print(f"[plan] {map_name}: requested_assets={requested_assets} target_images={len(jobs)}")
        _render_jobs(
            map_name=map_name,
            jobs=jobs,
            out_dir=out_dir,
            scenario_dir=scenario_dir,
            mcp_port=args.mcp_port,
            mcp_timeout_s=args.mcp_timeout_s,
            screenshot_timeout_s=args.screenshot_timeout_s,
        )
        sheets = _make_contact_sheets(out_dir)
        fresh = sum(1 for job in jobs if Path(job.image_path).stat().st_mtime >= marker.stat().st_mtime)
        summary[map_name] = {
            "target_images": len(jobs),
            "fresh_target_images": fresh,
            "requested_assets": requested_assets,
            "contact_sheets": sheets,
        }
        print(f"[verify] {map_name}: {json.dumps(summary[map_name], sort_keys=True)}")
    print("[done] " + json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
