"""
AgentGym-compatible HTTP server for DeliveryBench-text-only.

This server wraps `vlm_delivery.gym_like_interface.text_env.DeliveryBenchGymEnvText`
and exposes a minimal set of endpoints compatible with AgentGym env clients:

- POST /create
- POST /reset
- POST /step
- GET  /observation
- POST /close

Notes:
- Designed for *external-agent control*: the environment will NOT initialize
  an internal VLM client (enable_vlm=False). Actions must be provided by the
  caller via /step.
- Multimodal observation returns exactly TWO images:
  [global_map_png, local_map_png]
"""

from __future__ import annotations

import argparse
import base64
import json
import threading
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse


def _json_loads(b: bytes) -> dict[str, Any]:
    if not b:
        return {}
    return json.loads(b.decode("utf-8"))


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _error(handler: BaseHTTPRequestHandler, status: int, msg: str):
    _json_response(handler, status, {"error": msg})


@dataclass
class _EnvSession:
    env: Any
    last_text: str = ""
    last_reward: float = 0.0
    last_done: bool = False


_LOCK = threading.Lock()
_SESSIONS: dict[str, _EnvSession] = {}


def _default_base_dir() -> str:
    # This file lives at <repo>/vlm_delivery/agentgym_http_server.py
    # so repo root is parent of `vlm_delivery/`.
    return str(Path(__file__).resolve().parent.parent)


def _pyqt5_available() -> bool:
    try:
        import PyQt5  # noqa: F401

        return True
    except Exception:
        return False


def _get_dm(env: Any) -> Any:
    dms = getattr(env, "dms", None) or []
    if not dms:
        raise RuntimeError("env has no delivery-man instance; did you call reset()?")
    return dms[0]


def _build_text_obs(env: Any) -> str:
    dm = _get_dm(env)
    if hasattr(dm, "build_vlm_input") and callable(getattr(dm, "build_vlm_input")):
        return str(dm.build_vlm_input())
    # Fallback: keep it readable even if something changes upstream.
    if hasattr(dm, "to_text") and callable(getattr(dm, "to_text")):
        return str(dm.to_text())
    return "observation: N/A"


def _collect_images_b64(env: Any) -> list[str]:
    dm = _get_dm(env)
    from vlm_delivery.utils.vlm_runtime import vlm_collect_images

    imgs = vlm_collect_images(dm) or []
    # Expect exactly two images (global/local), but keep it robust.
    out: list[str] = []
    for b in imgs[:2]:
        out.append(base64.b64encode(b).decode("ascii"))
    return out


class _Handler(BaseHTTPRequestHandler):
    server_version = "DeliveryBenchAgentGymHTTP/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        # Keep server quiet by default (AgentGym spawns many calls).
        return

    def _read_json(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length") or "0")
        except Exception:
            length = 0
        data = self.rfile.read(length) if length > 0 else b""
        try:
            return _json_loads(data)
        except Exception as e:
            raise ValueError(f"Invalid JSON body: {e}") from e

    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/observation":
            qs = parse_qs(parsed.query or "")
            env_id = (qs.get("id", [""])[0] or "").strip()
            if not env_id:
                return _error(self, 400, "Missing query param: id")
            with _LOCK:
                sess = _SESSIONS.get(env_id)
            if sess is None:
                return _error(self, 404, f"Unknown env id: {env_id}")
            try:
                images_b64 = _collect_images_b64(sess.env)
                text = _build_text_obs(sess.env)
                sess.last_text = text
                return _json_response(
                    self,
                    200,
                    {
                        "text": text,
                        "images": images_b64,
                        "reward": sess.last_reward,
                        "done": sess.last_done,
                    },
                )
            except Exception as e:
                return _error(self, 500, f"Failed to build observation: {e}")

        if parsed.path in ("/health", "/"):
            return _json_response(self, 200, {"status": "ok"})

        return _error(self, 404, f"Unknown path: {parsed.path}")

    def do_POST(self):  # noqa: N802
        parsed = urlparse(self.path)
        try:
            data = self._read_json()
        except Exception as e:
            return _error(self, 400, str(e))

        if parsed.path == "/create":
            try:
                from vlm_delivery.gym_like_interface import DeliveryBenchGymEnvText

                base_dir = str((data.get("base_dir") or "").strip() or _default_base_dir())
                map_name = str((data.get("map_name") or "medium-city-22")).strip() or "medium-city-22"
                max_steps = int(data.get("max_steps") or 200)
                time_scale = float(data.get("time_scale") or 1.0)
                enable_map_images = bool(True if data.get("enable_map_images") is None else data.get("enable_map_images"))
                # Renderer selection:
                # - If caller specifies map_renderer, respect it.
                # - Else default to Qt when available (pixel-identical-ish), and
                #   fall back to PIL when PyQt5 is not installed.
                if data.get("map_renderer") is not None:
                    map_renderer = str(data.get("map_renderer")).strip().lower() or "qt"
                else:
                    map_renderer = "qt" if _pyqt5_available() else "pil"

                env = DeliveryBenchGymEnvText(
                    base_dir=base_dir,
                    map_name=map_name,
                    max_steps=max_steps,
                    time_scale=time_scale,
                    enable_map_images=enable_map_images,
                    map_renderer=map_renderer,
                    enable_vlm=False,  # external-agent control
                )
                env_id = str(uuid.uuid4())
                sess = _EnvSession(env=env, last_text="", last_reward=0.0, last_done=False)
                with _LOCK:
                    _SESSIONS[env_id] = sess
                return _json_response(self, 200, {"id": env_id})
            except Exception as e:
                return _error(self, 500, f"Failed to create env: {e}")

        if parsed.path == "/reset":
            env_id = str((data.get("id") or "")).strip()
            if not env_id:
                return _error(self, 400, "Missing field: id")
            with _LOCK:
                sess = _SESSIONS.get(env_id)
            if sess is None:
                return _error(self, 404, f"Unknown env id: {env_id}")

            # AgentGym uses data_idx as seed-like index.
            data_idx = int(data.get("data_idx") or 0)
            try:
                _obs, info = sess.env.reset(seed=data_idx)
                sess.last_reward = 0.0
                sess.last_done = False
                sess.last_text = _build_text_obs(sess.env)
                return _json_response(
                    self,
                    200,
                    {
                        "observation": sess.last_text,
                        "reward": sess.last_reward,
                        "done": sess.last_done,
                        "info": info or {},
                    },
                )
            except Exception as e:
                return _error(self, 500, f"Failed to reset env: {e}")

        if parsed.path == "/step":
            env_id = str((data.get("id") or "")).strip()
            if not env_id:
                return _error(self, 400, "Missing field: id")
            with _LOCK:
                sess = _SESSIONS.get(env_id)
            if sess is None:
                return _error(self, 404, f"Unknown env id: {env_id}")

            action = data.get("action")
            if action is None:
                return _error(self, 400, "Missing field: action")
            action_str = str(action)

            try:
                _obs, reward, terminated, truncated, info = sess.env.step(action_str)
                done = bool(terminated) or bool(truncated)
                sess.last_reward = float(reward)
                sess.last_done = bool(done)
                sess.last_text = _build_text_obs(sess.env)
                return _json_response(
                    self,
                    200,
                    {
                        "observation": sess.last_text,
                        "reward": sess.last_reward,
                        "done": sess.last_done,
                        "info": info or {},
                    },
                )
            except Exception as e:
                return _error(self, 500, f"Failed to step env: {e}")

        if parsed.path == "/close":
            env_id = str((data.get("id") or "")).strip()
            if not env_id:
                return _error(self, 400, "Missing field: id")
            with _LOCK:
                sess = _SESSIONS.pop(env_id, None)
            if sess is None:
                return _json_response(self, 200, {"status": "closed"})
            try:
                sess.env.close()
            except Exception:
                pass
            return _json_response(self, 200, {"status": "closed"})

        return _error(self, 404, f"Unknown path: {parsed.path}")


class _AgentGymHTTPServer(HTTPServer):
    # Make restarts reliable (avoid TIME_WAIT bind issues).
    allow_reuse_address = True


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8009)
    args = parser.parse_args(argv)

    # NOTE: We intentionally use a non-threaded server here.
    # The Qt renderer path (PyQt5) is not thread-safe and expects all Qt objects
    # (including QApplication) to be created/used on the main thread.
    httpd = _AgentGymHTTPServer((args.host, args.port), _Handler)
    print(f"[DeliveryBenchAgentGymHTTP] listening on http://{args.host}:{args.port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            httpd.server_close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

