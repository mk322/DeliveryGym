"""Render a cached FPV album for any map, driving UE over its MCP server.

VAGEN's own renderer speaks a raw JSON-line socket to a SimWorld Studio bridge
on port 55565. UE 5.8's editor instead exposes the official MCP server over
HTTP, whose ``ExecutePythonScript`` tool runs editor Python. This module targets
that, which matters for a reason beyond transport: each MCP call runs a short
script and returns, so the editor keeps ticking *between* calls. A blocking
in-editor script starves the loop that performs the work, which is why an
earlier navmesh build reported "not building" and sampled zero points.

The capture itself is the same pair of calls the vendored renderer makes:

    UnrealEditorSubsystem.set_level_viewport_camera_info(location, rotation)
    AutomationLibrary.take_high_res_screenshot(width, height, output_path)

``take_high_res_screenshot`` is asynchronous, so the driver issues one capture
per call and then waits for the file to appear rather than assuming it did.

The job list comes from the map's own navigation graph — every certified node by
every cardinal yaw — so an album covers exactly the positions the runtime can
stand on. The runbook is explicit that images join to the runtime by
``(round(x_cm,1), round(y_cm,1), yaw)`` and "never by waypoint_id, those drift",
so position is what the manifest keys on.

Rendering is resumable and idempotent: a job whose output already exists is
skipped, so an interrupted bake continues instead of restarting.
"""

from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

# Matches the vendored album's plain-view configuration.
PLAIN_CAMERA_Z_CM = 160.0
PLAIN_WIDTH = 640
PLAIN_HEIGHT = 480
YAWS = (0.0, 90.0, 180.0, 270.0)


class UeMcp:
    """Minimal client for the UE editor's official MCP server."""

    TOOL = "SimWorldRuntime.SimWorldStudioToolset.ExecutePythonScript"

    def __init__(self, port: int = 8000, host: str = "127.0.0.1", timeout: float = 600.0):
        self.base = f"http://{host}:{port}/mcp"
        self.timeout = timeout
        self.session: str | None = None
        self._rpc_id = 0
        self._initialise()

    def _post(self, payload: dict[str, Any]) -> Any:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self.session:
            headers["Mcp-Session-Id"] = self.session
        request = urllib.request.Request(
            self.base, data=json.dumps(payload).encode(), headers=headers, method="POST"
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            session = response.headers.get("Mcp-Session-Id")
            if session and not self.session:
                self.session = session
            body = response.read().decode("utf-8", "replace")
        if not body.strip():
            return None
        if "data:" in body:  # server-sent-events framing
            for line in reversed(body.splitlines()):
                if line.startswith("data:"):
                    body = line[5:].strip()
                    break
        try:
            return json.loads(body)
        except ValueError:
            return None

    def _initialise(self) -> None:
        self._post({
            "jsonrpc": "2.0", "id": 0, "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05", "capabilities": {},
                "clientInfo": {"name": "embodiedbench", "version": "0.1"},
            },
        })
        self._post({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def python(self, script: str) -> Any:
        self._rpc_id += 1
        return self._post({
            "jsonrpc": "2.0", "id": self._rpc_id, "method": "tools/call",
            "params": {"name": self.TOOL, "arguments": {
                "Script": script, "ExecutionMode": "ExecuteFile"}},
        })

    def ping(self) -> bool:
        try:
            self._post({"jsonrpc": "2.0", "id": 999, "method": "tools/list", "params": {}})
            return True
        except Exception:  # noqa: BLE001
            return False


@dataclass
class RenderJob:
    """One image to capture."""

    waypoint_id: str
    x_cm: float
    y_cm: float
    yaw: float
    output_path: Path

    @property
    def key(self) -> tuple[float, float, float]:
        return (round(self.x_cm, 1), round(self.y_cm, 1), self.yaw)


@dataclass
class RenderReport:
    map_name: str
    planned: int = 0
    skipped_existing: int = 0
    rendered: int = 0
    failed: int = 0
    failures: list[dict[str, Any]] = field(default_factory=list)
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "map": self.map_name,
            "planned": self.planned,
            "skipped_existing": self.skipped_existing,
            "rendered": self.rendered,
            "failed": self.failed,
            "failures": self.failures[:40],
            "seconds": round(self.seconds, 1),
            "status": "pass" if self.failed == 0 and self.planned else "fail",
        }


def plan_jobs(node_positions: list[tuple[str, float, float]], album_root: Path) -> list[RenderJob]:
    """Every node by every cardinal yaw, in a stable order."""
    jobs: list[RenderJob] = []
    for waypoint_id, x_cm, y_cm in sorted(node_positions, key=lambda n: (n[0], n[1], n[2])):
        # Directory naming follows the engine's own convention: int_0 -> int_000.
        if "_" in waypoint_id:
            kind, number = waypoint_id.rsplit("_", 1)
            try:
                directory = f"{kind}_{int(number):03d}"
            except ValueError:
                directory = waypoint_id
        else:
            directory = waypoint_id
        for yaw in YAWS:
            jobs.append(
                RenderJob(
                    waypoint_id=waypoint_id,
                    x_cm=x_cm,
                    y_cm=y_cm,
                    yaw=yaw,
                    output_path=album_root / "images" / directory / f"yaw_{int(yaw):03d}.png",
                )
            )
    return jobs


CAPTURE_SCRIPT = """
import unreal
loc = unreal.Vector({x!r}, {y!r}, {z!r})
# Keyword arguments are mandatory here: unreal.Rotator is positionally
# (roll, pitch, yaw), so Rotator(0.0, yaw, 0.0) sets the PITCH and tips the
# camera at the sky. The first Paris capture looked straight up because of it.
rot = unreal.Rotator(pitch=0.0, yaw={yaw!r}, roll=0.0)
unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).set_level_viewport_camera_info(loc, rot)
unreal.AutomationLibrary.take_high_res_screenshot({w}, {h}, {out!r})
unreal.log("[EB-FPV] {out}")
"""

LOAD_MAP_SCRIPT = """
import unreal
w = unreal.EditorLevelLibrary.get_editor_world()
if w is None or {map_token!r} not in w.get_name():
    unreal.EditorLoadingAndSavingUtils.load_map({map_path!r})
    w = unreal.EditorLevelLibrary.get_editor_world()
unreal.log("[EB-FPV] world=%s" % (w.get_name() if w else None))
"""


def ensure_map(mcp: UeMcp, map_path: str, map_token: str) -> None:
    mcp.python(LOAD_MAP_SCRIPT.format(map_path=map_path, map_token=map_token))


def render_jobs(
    mcp: UeMcp,
    jobs: list[RenderJob],
    *,
    map_name: str,
    width: int = PLAIN_WIDTH,
    height: int = PLAIN_HEIGHT,
    camera_z: float = PLAIN_CAMERA_Z_CM,
    file_timeout_s: float = 60.0,
    progress_every: int = 100,
) -> RenderReport:
    """Capture every job, skipping any whose output already exists."""
    report = RenderReport(map_name=map_name, planned=len(jobs))
    started = time.time()

    for index, job in enumerate(jobs):
        if job.output_path.exists() and job.output_path.stat().st_size > 0:
            report.skipped_existing += 1
            continue
        job.output_path.parent.mkdir(parents=True, exist_ok=True)

        script = CAPTURE_SCRIPT.format(
            x=float(job.x_cm), y=float(job.y_cm), z=float(camera_z),
            yaw=float(job.yaw), w=width, h=height, out=str(job.output_path),
        )
        try:
            mcp.python(script)
        except Exception as exc:  # noqa: BLE001 - a failed capture is a result
            report.failed += 1
            report.failures.append({"key": str(job.key), "error": f"{type(exc).__name__}: {exc}"})
            continue

        # take_high_res_screenshot is asynchronous; wait for the file rather
        # than assuming the call implies the image.
        deadline = time.time() + file_timeout_s
        while time.time() < deadline:
            if job.output_path.exists() and job.output_path.stat().st_size > 0:
                break
            time.sleep(0.25)
        if job.output_path.exists() and job.output_path.stat().st_size > 0:
            report.rendered += 1
        else:
            report.failed += 1
            report.failures.append({"key": str(job.key), "error": "screenshot never appeared"})

        if progress_every and (index + 1) % progress_every == 0:
            elapsed = time.time() - started
            rate = (report.rendered + report.skipped_existing) / max(elapsed, 1e-6)
            print(
                f"  {index + 1}/{len(jobs)} rendered={report.rendered} "
                f"skipped={report.skipped_existing} failed={report.failed} "
                f"{rate:.1f} img/s",
                flush=True,
            )

    report.seconds = time.time() - started
    return report


def write_manifest(jobs: list[RenderJob], album_root: Path, map_name: str) -> Path:
    """Write the album manifest in the schema the runtime already reads."""
    album_root.mkdir(parents=True, exist_ok=True)
    manifest = album_root / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for job in jobs:
            exists = job.output_path.exists() and job.output_path.stat().st_size > 0
            handle.write(json.dumps({
                "map_name": map_name,
                "waypoint_id": job.waypoint_id,
                "yaw": job.yaw,
                "x_cm": job.x_cm,
                "y_cm": job.y_cm,
                "z_cm": PLAIN_CAMERA_Z_CM,
                "render_kind": "plain",
                "camera_mode": "waypoint",
                "camera_backoff_cm": 0.0,
                "capture_mode": "camera",
                "image_path": str(job.output_path),
                "image_size": [PLAIN_WIDTH, PLAIN_HEIGHT],
                "status": "ok" if exists else "failed",
                "error": None if exists else "image missing after render",
            }) + "\n")
    return manifest


def graph_node_positions(map_name: str, base_dir: str | None = None) -> list[tuple[str, float, float]]:
    """Every navigation-graph node, as the album's coverage target."""
    import asyncio
    import dataclasses

    from embodiedbench.baseline.compat import apply_map_compatibility_patches
    from embodiedbench.baseline.determinism import apply_deterministic_patches
    from embodiedbench.baseline.replay import load_vendor_env_module

    apply_deterministic_patches()
    apply_map_compatibility_patches()
    module = load_vendor_env_module()

    async def load():
        config = dataclasses.asdict(module.PRESETS["nav"])
        if base_dir:
            config["base_dir"] = base_dir
        config.update(map_name=map_name, render_mode="text", max_steps=8)
        env = module.DeliveryBench(config)
        try:
            await env.reset(seed=0)
            adjacency = env._env.dms[0].city_map.waypoint_graph.adjacency_list
            return [
                (
                    str(getattr(node, "waypoint_id", "") or f"n_{index}"),
                    float(node.position.x),
                    float(node.position.y),
                )
                for index, node in enumerate(adjacency)
            ]
        finally:
            await env.close()

    return asyncio.run(load())
