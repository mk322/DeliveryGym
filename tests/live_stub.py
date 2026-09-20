"""A real HTTP nav-render/v0 service for the live-backend tests.

Not a mock of the client -- an actual ``ThreadingHTTPServer`` on a real port,
because the thing the tests defend is the *wire*: a client bug that only a
socket can show (timeouts, HTTP error bodies, JSON framing) passes straight
through a mocked transport. The frames are labelled PNGs drawn with PIL, so a
test can open what came back and a human debugging one can see which request
produced it.

The service is deliberately deterministic: the same key always draws the same
pixels, hence the same PNG bytes -- the property the album cache and
FrameAliases assume of a real service process, reproduced here so the tests
can assert on it.

Failure injection is explicit state, flipped by the test that needs it:
``healthz_ok`` (health probes fail), ``fail_keys`` (per-item failures inside
a 200 response), ``reject_code`` (the whole request dies with a wire error;
``busy`` goes out as HTTP 503, the way the real service sends it),
``busy_batches`` (the next N render batches answer 503 busy, then service
resumes -- the transient the busy taxonomy exists for), ``delay_s`` (every
render sleeps first, so a test can hold a request in flight on purpose).
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import threading
import time
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

PROTOCOL = "nav-render/v0"


def draw_frame(key: str, width: int, height: int) -> bytes:
    """A labelled PNG whose pixels are a pure function of (key, size)."""
    from PIL import Image, ImageDraw

    digest = hashlib.sha256(key.encode()).digest()
    background = (digest[0], digest[1], digest[2])
    image = Image.new("RGB", (width, height), background)
    ImageDraw.Draw(image).text((8, 8), key, fill=(255, 255, 255))
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


class FakeRenderService:
    """One in-process UE stand-in speaking nav-render/v0 over HTTP."""

    def __init__(self, out_dir: Path, *, instance_id: str = "fake-0",
                 map_name: str = "citycore-paris"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.instance_id = instance_id
        self.map_name = map_name
        # Test-visible state.
        self.healthz_ok = True
        self.fail_keys: set[str] = set()
        self.reject_code: str | None = None
        self.busy_batches = 0
        self.busy_hits = 0
        self.delay_s = 0.0
        self.render_counts: Counter[str] = Counter()
        self.batches: list[dict[str, Any]] = []      # raw request bodies
        self.health_probes = 0
        self._lock = threading.Lock()

        service = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:  # keep pytest output clean
                pass

            def do_GET(self) -> None:
                if self.path != "/healthz":
                    self._send(404, {"error": {"code": "bad_request",
                                               "message": f"no route {self.path}"}})
                    return
                with service._lock:
                    service.health_probes += 1
                    ok = service.healthz_ok
                if not ok:
                    self._send(500, {"error": {"code": "engine_down",
                                               "message": "engine not attached"}})
                    return
                self._send(200, {
                    "status": "ok", "protocol": PROTOCOL,
                    "instance_id": service.instance_id,
                    "map_name": service.map_name,
                    "engine_connected": True, "episodes_active": 0,
                    "uptime_s": 1.0,
                })

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(
                    int(self.headers.get("Content-Length", "0"))))
                if self.path != "/render":
                    # Anything that is not a render is a Track B endpoint --
                    # answered by the subclass that implements them, 404 here.
                    answer = service.handle_track_b(self.path, body)
                    if answer is None:
                        self._send(404, {"error": {"code": "bad_request",
                                                   "message": f"no route {self.path}"}})
                    else:
                        self._send(*answer)
                    return
                if body.get("protocol") != PROTOCOL:
                    self._send(400, {"error": {"code": "bad_request",
                                               "message": "wrong protocol"}})
                    return
                with service._lock:
                    reject = service.reject_code
                    if service.busy_batches > 0:
                        service.busy_batches -= 1
                        service.busy_hits += 1
                        reject = reject or "busy"
                    delay = service.delay_s
                    service.batches.append(body)
                if reject:
                    # busy is 503 on the real service; other injected codes
                    # keep the generic 500 the earlier tests were written for.
                    self._send(503 if reject == "busy" else 500,
                               {"error": {"code": reject,
                                          "message": "injected by test"}})
                    return
                if delay:
                    time.sleep(delay)
                self._send(200, {"results": [self._result(body, item)
                                             for item in body.get("requests", [])]})

            def _result(self, body: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
                key = item["key"]
                with service._lock:
                    service.render_counts[key] += 1
                    failed = key in service.fail_keys
                if failed:
                    return {"key": key, "status": "failed",
                            "error": "injected per-item failure"}
                camera = item.get("camera") or body["camera"]
                png = draw_frame(key, int(camera["width"]), int(camera["height"]))
                digest = hashlib.sha256(png).hexdigest()
                out = {"key": key, "status": "ok", "path": None, "png_base64": None,
                       "sha256": digest,
                       "width": int(camera["width"]), "height": int(camera["height"])}
                if body.get("return_mode") == "base64":
                    out["png_base64"] = base64.b64encode(png).decode("ascii")
                else:
                    target = service.out_dir / f"{digest}.png"
                    if not target.exists():
                        target.write_bytes(png)
                    out["path"] = str(target)
                return out

            def _send(self, code: int, payload: dict[str, Any]) -> None:
                raw = json.dumps(payload).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def handle_track_b(self, path: str, body: dict[str, Any]):
        """Track B endpoints live in the subclass; the base speaks Track A only.

        Returns ``(status, payload)`` or ``None`` for "no such route".
        """
        return None

    # ── lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> "FakeRenderService":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    @property
    def port(self) -> int:
        return self._server.server_address[1]

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


class FakeTrackBService(FakeRenderService):
    """The Track A stub plus the four stateful Track B endpoints (spec 3b).

    The walk is a kinematic straight-line integrator at fixed dt: each tick
    moves the agent ``speed * dt`` toward the target (never past it), and the
    walk ends the moment the remaining distance is within ``arrive_cm`` -- so
    tick counts are exact and predictable from geometry alone:

        ticks = ceil((distance - arrive_cm) / (speed * fixed_dt))

    ``sim_seconds`` is always ``ticks * fixed_dt``, which is the property the
    embodied env's clock assertions rest on.

    Failure injection, all explicit test state:

    * ``wall_after_cm`` -- every walk stops making progress after this many
      cm and reports ``stuck`` (ticks still count the movement that
      happened, including the final partial step into the wall);
    * ``busy_track_b`` -- every Track B request answers 503 busy, the "some
      other episode holds this instance" signal;
    * a walk whose ``max_sim_seconds`` runs out reports ``timeout`` with the
      full tick budget burned, no knob needed.
    """

    FIXED_DT = 0.0333

    def __init__(self, out_dir: Path, *, instance_id: str = "fake-b0",
                 map_name: str = "citycore-paris"):
        super().__init__(out_dir, instance_id=instance_id, map_name=map_name)
        self.fixed_dt = self.FIXED_DT
        self.agent: dict[str, Any] | None = None
        self.wall_after_cm: float | None = None
        self.busy_track_b = False
        # Test-visible transcripts, one row per request.
        self.episodes: list[dict[str, Any]] = []
        self.walks: list[dict[str, Any]] = []
        self.walk_pixels: list[dict[str, Any]] = []
        self.observes: list[dict[str, Any]] = []
        self.view_pairs: list[dict[str, Any]] = []
        self.snapshots: dict[str, dict[str, Any]] = {}
        self._observe_seq = 0
        self.episode_ends: list[str] = []

    # ── routing ──────────────────────────────────────────────────────────────

    def handle_track_b(self, path: str, body: dict[str, Any]):
        routes = {"/episode": self._episode, "/walk": self._walk,
                  "/walk_pixel": self._walk_pixel,
                  "/observe": self._observe,
                  "/observe_views": self._observe_views,
                  "/episode_end": self._episode_end}
        if path not in routes:
            return None
        if body.get("protocol") != PROTOCOL:
            return 400, {"error": {"code": "bad_request",
                                   "message": "wrong protocol"}}
        with self._lock:
            if self.busy_track_b:
                return 503, {"error": {"code": "busy",
                                       "message": "another episode holds this instance"}}
            return routes[path](body)

    # ── the four endpoints (called under the lock) ───────────────────────────

    def _episode(self, body: dict[str, Any]):
        spawn = body["spawn"]
        self.agent = {
            "episode_id": body["episode_id"],
            "x": float(spawn["x_cm"]), "y": float(spawn["y_cm"]),
            "z": float(spawn["z_cm"]), "yaw": float(spawn["yaw_deg"]),
            "speed_cm_s": float(body["agent"]["speed_cm_s"]),
            "eye_z_cm": float(body["agent"]["eye_z_cm"]),
            "camera": dict(body["agent"]["camera"]),
        }
        self.episodes.append(body)
        return 200, {"episode_id": body["episode_id"],
                     "pose": self._pose(), "fixed_dt": self.fixed_dt}

    def _walk(self, body: dict[str, Any]):
        agent = self._active(body)
        if agent is None:
            return 400, {"error": {"code": "bad_request",
                                   "message": "no such active episode"}}
        target = body["target"]
        tx, ty = float(target["x_cm"]), float(target["y_cm"])
        arrive_cm = float(body.get("arrive_cm", 50.0))
        max_sim_seconds = float(body.get("max_sim_seconds", 120.0))
        response = self._kinematic_walk(agent, tx, ty, arrive_cm, max_sim_seconds)
        start = response.pop("start_xy")
        self.walks.append({"start_xy": start, "target_xy": (tx, ty),
                           "arrive_cm": arrive_cm, **response})
        return 200, response

    def _kinematic_walk(self, agent: dict[str, Any], tx: float, ty: float,
                        arrive_cm: float, max_sim_seconds: float) -> dict[str, Any]:
        """The reference integrator, factored out so /walk and /walk_pixel
        share one notion of what walking to a world point does -- the two
        endpoints differ only in how the target got picked, never in how it
        is walked to."""
        max_ticks = int(max_sim_seconds / self.fixed_dt)
        step = agent["speed_cm_s"] * self.fixed_dt
        start = (agent["x"], agent["y"])
        wall = self.wall_after_cm
        ticks = 0
        walked = 0.0
        arrived = stuck = timeout = False
        while True:
            dist = math.hypot(tx - agent["x"], ty - agent["y"])
            if dist <= max(arrive_cm, 1e-9):
                arrived = True
                break
            if ticks >= max_ticks:
                timeout = True
                break
            move = min(step, dist)
            if wall is not None:
                move = min(move, wall - walked)
            if move <= 0.0:
                stuck = True
                break
            agent["x"] += (tx - agent["x"]) / dist * move
            agent["y"] += (ty - agent["y"]) / dist * move
            walked += move
            ticks += 1
        if walked > 0.0:
            agent["yaw"] = math.degrees(math.atan2(ty - start[1], tx - start[0]))
        return {"start_xy": start, "arrived": arrived, "stuck": stuck,
                "timeout": timeout, "ticks": ticks,
                "sim_seconds": ticks * self.fixed_dt,
                "pose": self._pose(), "walked_cm": walked}

    # Fake pixel resolution. Not a raycast -- there is no ground mesh here --
    # but deterministic and geometrically sane, so tests can assert both the
    # "resolves and walks" and "does not resolve" paths without an engine:
    #
    #   * v <= HORIZON_V is above the fake horizon: nothing to hit, the same
    #     shape of failure a raycast into the sky would be;
    #   * v in (HORIZON_V, 1] maps to a forward distance, nearest at v=1;
    #   * u maps to a lateral offset off the agent's current heading;
    #   * a lateral offset past LATERAL_LIMIT_CM is "off the sidewalk", the
    #     fake's stand-in for a NavMesh projection that gives up.
    HORIZON_V = 0.5
    FORWARD_RANGE_CM = 2000.0
    LATERAL_LIMIT_CM = 500.0

    def _resolve_pixel(self, agent: dict[str, Any], u: float, v: float,
                       yaw_deg: float | None = None
                       ) -> tuple[tuple[float, float] | None, str | None]:
        if v <= self.HORIZON_V:
            return None, "no_ground_hit"
        forward_cm = self.FORWARD_RANGE_CM * (1.0 - v) / (1.0 - self.HORIZON_V)
        lateral_cm = (u - 0.5) * 2.0 * self.LATERAL_LIMIT_CM * 1.5
        if abs(lateral_cm) > self.LATERAL_LIMIT_CM:
            return None, "off_navmesh"
        yaw = math.radians(agent["yaw"] if yaw_deg is None else yaw_deg)
        forward = (math.cos(yaw), math.sin(yaw))
        right = (math.sin(yaw), -math.cos(yaw))
        x = agent["x"] + forward[0] * forward_cm + right[0] * lateral_cm
        y = agent["y"] + forward[1] * forward_cm + right[1] * lateral_cm
        return (x, y), None

    def _walk_pixel(self, body: dict[str, Any]):
        binding_keys = ("view", "capture_group_id", "camera_snapshot_id")
        binding = {key: body[key] for key in binding_keys if key in body}
        if binding and len(binding) != len(binding_keys):
            return 400, {"error": {"code": "bad_request",
                                   "message": (
                                       "a bound pixel needs view, "
                                       "capture_group_id, and "
                                       "camera_snapshot_id")},
                         **binding}
        agent = self._active(body)
        if agent is None:
            return 400, {"error": {"code": "bad_request",
                                   "message": "no such active episode"},
                         **binding}
        pixel = body["pixel"]
        u, v = float(pixel["u"]), float(pixel["v"])
        arrive_cm = float(body.get("arrive_cm", 50.0))
        max_sim_seconds = float(body.get("max_sim_seconds", 120.0))
        is_bound = bool(binding)
        snapshot = None
        if is_bound:
            snapshot = self.snapshots.get(body["camera_snapshot_id"])
            if (snapshot is None or snapshot["episode_id"] != body["episode_id"]
                    or snapshot["group"] != body["capture_group_id"]
                    or snapshot["view"] != body["view"]):
                return 400, {"error": {"code": "bad_request",
                                       "message": "view/group/snapshot mismatch"},
                             **binding}
        if snapshot is None:
            target, rejection = self._resolve_pixel(agent, u, v)
        else:
            target, rejection = self._resolve_pixel(agent, u, v, snapshot["yaw"])
        if target is None:
            response = {"arrived": False, "stuck": False, "timeout": False,
                        "ticks": 0, "sim_seconds": 0.0, "pose": self._pose(),
                        "walked_cm": 0.0,
                        "resolved": {"raw_world_hit_cm": None,
                                    "accepted_target_cm": None,
                                    "navmesh_adjustment_cm": None,
                                    "rejection_reason": rejection},
                        **binding}
            self.walk_pixels.append({"pixel": (u, v), "target_xy": None, **response})
            return 200, response
        # No real NavMesh here, so the accepted target IS the raw hit and the
        # adjustment is exactly zero -- a real service reports a nonzero
        # distance whenever the raycast landed off the walkable surface but
        # still close enough to snap.
        walk = self._kinematic_walk(agent, target[0], target[1],
                                    arrive_cm, max_sim_seconds)
        start = walk.pop("start_xy")
        response = {**walk,
                   "resolved": {"raw_world_hit_cm": [target[0], target[1]],
                               "accepted_target_cm": [target[0], target[1]],
                               "navmesh_adjustment_cm": 0.0,
                               "rejection_reason": None},
                   **binding}
        self.walk_pixels.append({"pixel": (u, v), "start_xy": start,
                                 "target_xy": target, **response})
        return 200, response

    def _observe(self, body: dict[str, Any]):
        agent = self._active(body)
        if agent is None:
            return 400, {"error": {"code": "bad_request",
                                   "message": "no such active episode"}}
        if body["yaw_deg"] is not None:
            agent["yaw"] = float(body["yaw_deg"])
        camera = body["camera"]
        label = (f"pose({agent['x']:.0f},{agent['y']:.0f})"
                 f"@yaw{agent['yaw']:.1f}")
        png = draw_frame(label, int(camera["width"]), int(camera["height"]))
        digest = hashlib.sha256(png).hexdigest()
        # The /render-item result shape with a service-assigned key (the
        # caller still owns album naming and ignores it), plus the pose echo.
        self._observe_seq += 1
        result = {"key": f"observe/{self._observe_seq:06d}",
                  "status": "ok", "path": None, "png_base64": None,
                  "sha256": digest,
                  "width": int(camera["width"]), "height": int(camera["height"]),
                  "pose": self._pose()}
        if body.get("return_mode") == "base64":
            result["png_base64"] = base64.b64encode(png).decode("ascii")
        else:
            target = self.out_dir / f"{digest}.png"
            if not target.exists():
                target.write_bytes(png)
            result["path"] = str(target)
        self.observes.append(dict(body))
        return 200, result

    def _observed_view(self, body: dict[str, Any], view: str,
                       yaw_offset_deg: float, snapshot: str, yaw_deg: float):
        camera = body["camera"]
        label = (f"{view} pose({self.agent['x']:.0f},{self.agent['y']:.0f})"
                 f"@yaw{yaw_deg:.1f}")
        png = draw_frame(label, int(camera["width"]), int(camera["height"]))
        digest = hashlib.sha256(png).hexdigest()
        result = {
            "view": view,
            "yaw_offset_deg": yaw_offset_deg,
            "camera_snapshot_id": snapshot,
            "camera_intrinsics_id": (
                f"cam-{int(camera['width'])}x{int(camera['height'])}"
                f"-fov{float(camera['fov_deg']):g}"),
            "status": "ok",
            "path": None,
            "png_base64": None,
            "sha256": digest,
            "width": int(camera["width"]),
            "height": int(camera["height"]),
        }
        if body.get("return_mode") == "base64":
            result["png_base64"] = base64.b64encode(png).decode("ascii")
        else:
            target = self.out_dir / f"{digest}.png"
            if not target.exists():
                target.write_bytes(png)
            result["path"] = str(target)
        return result

    def _observe_views(self, body: dict[str, Any]):
        agent = self._active(body)
        if agent is None:
            return 400, {"error": {"code": "bad_request",
                                   "message": "no such active episode"}}
        group = f"view-pair-{self._observe_seq + 1}"
        base_yaw = float(agent["yaw"])
        views = []
        for view, offset in (("front", 0.0), ("rear", 180.0)):
            yaw = (base_yaw + offset) % 360.0
            snapshot = f"{group}-{view}"
            self.snapshots[snapshot] = {
                "episode_id": body["episode_id"], "group": group,
                "view": view, "pose": dict(self._pose()), "yaw": yaw,
            }
            views.append(self._observed_view(body, view, offset, snapshot, yaw))
        self._observe_seq += 1
        response = {"capture_group_id": group, "pose": self._pose(), "views": views}
        self.view_pairs.append({"request": dict(body), **response})
        return 200, response

    def _episode_end(self, body: dict[str, Any]):
        self.episode_ends.append(body["episode_id"])
        self.agent = None
        return 200, {"ok": True}

    # ── helpers ──────────────────────────────────────────────────────────────

    def _active(self, body: dict[str, Any]) -> dict[str, Any] | None:
        if self.agent is None or self.agent["episode_id"] != body.get("episode_id"):
            return None
        return self.agent

    def _pose(self) -> dict[str, float]:
        return {"x_cm": self.agent["x"], "y_cm": self.agent["y"],
                "z_cm": self.agent["z"], "yaw_deg": self.agent["yaw"]}


def write_endpoints(path: Path, services: list[Any], seats: int | None = None) -> Path:
    """An endpoints.json for these services (or (id, base_url) pairs).

    ``seats`` writes the fleet's ``max_episodes`` per instance; left None the
    key is omitted entirely, which is the pre-seats file shape the pool must
    still read (defaulting to one courier per instance).
    """
    instances = []
    for entry in services:
        if isinstance(entry, FakeRenderService):
            row = {"id": entry.instance_id, "base_url": entry.base_url,
                   "map_name": entry.map_name, "gpu_uuid": ""}
        else:
            member_id, base_url = entry
            row = {"id": member_id, "base_url": base_url,
                   "map_name": "citycore-paris", "gpu_uuid": ""}
        if seats is not None:
            row["max_episodes"] = seats
        instances.append(row)
    path.write_text(json.dumps({"version": 0, "instances": instances}, indent=2))
    return path
