"""
AgentGym Gateway HTTP server for DeliveryBench-text-only.

Goal:
- Present ONE `env_server_base` that supports many concurrent env_ids, matching
  AgentGym's typical usage pattern.

Approach:
- Each env_id is backed by a dedicated *worker process* running
  `vlm_delivery.agentgym_http_server` on its own localhost port.
- The gateway proxies AgentGym-style endpoints to the correct worker based on id.

Why process-per-env:
- DeliveryBench uses Qt (thread affinity) and process-level singletons (e.g. Comms),
  which are hard to safely isolate in a single Python process across multiple envs.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

import requests


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


def _repo_root() -> str:
    # This file lives at <repo>/vlm_delivery/agentgym_gateway_server.py
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _pick_free_port(host: str) -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind((host, 0))
        return int(s.getsockname()[1])
    finally:
        try:
            s.close()
        except Exception:
            pass


@dataclass
class _Worker:
    env_id: str
    host: str
    port: int
    proc: subprocess.Popen
    created_at: float
    last_used_at: float

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


_LOCK = threading.Lock()
_ENV_TO_WORKER: dict[str, _Worker] = {}


def _terminate_process(proc: subprocess.Popen, timeout_s: float = 5.0) -> None:
    try:
        if proc.poll() is not None:
            return
        proc.terminate()
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            if proc.poll() is not None:
                return
            time.sleep(0.05)
        proc.kill()
    except Exception:
        pass


def _cleanup_all_workers():
    with _LOCK:
        workers = list(_ENV_TO_WORKER.values())
        _ENV_TO_WORKER.clear()
    for w in workers:
        _terminate_process(w.proc)


atexit.register(_cleanup_all_workers)


def _install_signal_handlers():
    def _handler(_signum, _frame):
        _cleanup_all_workers()
        raise SystemExit(0)

    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(s, _handler)
        except Exception:
            pass


def _spawn_worker(
    *,
    worker_host: str,
    worker_port: int,
    startup_timeout_s: float,
) -> subprocess.Popen:
    env = dict(os.environ)
    # Ensure Qt can run headlessly.
    env.setdefault("QT_QPA_PLATFORM", "offscreen")

    cmd = [
        sys.executable,
        "-m",
        "vlm_delivery.agentgym_http_server",
        "--host",
        worker_host,
        "--port",
        str(worker_port),
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=_repo_root(),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    # Wait for health
    deadline = time.time() + float(startup_timeout_s)
    url = f"http://{worker_host}:{worker_port}/health"
    last_err: Optional[Exception] = None
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("worker exited during startup")
        try:
            r = requests.get(url, timeout=1.0)
            if r.status_code == 200:
                return proc
        except Exception as e:
            last_err = e
        time.sleep(0.1)
    raise RuntimeError(f"worker did not become healthy in time: {last_err}")


def _worker_request(
    method: str,
    url: str,
    *,
    json_body: Optional[dict[str, Any]] = None,
    params: Optional[dict[str, Any]] = None,
    timeout_s: float = 2400.0,
) -> tuple[int, dict[str, Any]]:
    try:
        r = requests.request(method, url, json=json_body, params=params, timeout=timeout_s)
    except Exception as e:
        return 502, {"error": f"Upstream worker request failed: {e}"}

    try:
        payload = r.json()
        if not isinstance(payload, dict):
            payload = {"error": f"Worker returned non-dict JSON: {payload!r}"}
    except Exception:
        payload = {"error": f"Worker returned non-JSON response: {r.text[:200]}"}
    return int(r.status_code), payload


class _GatewayHandler(BaseHTTPRequestHandler):
    server_version = "DeliveryBenchAgentGymGateway/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
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
        if parsed.path in ("/health", "/"):
            return _json_response(self, 200, {"status": "ok"})

        if parsed.path == "/observation":
            qs = parse_qs(parsed.query or "")
            env_id = (qs.get("id", [""])[0] or "").strip()
            if not env_id:
                return _error(self, 400, "Missing query param: id")

            with _LOCK:
                worker = _ENV_TO_WORKER.get(env_id)
                if worker:
                    worker.last_used_at = time.time()
            if worker is None:
                return _error(self, 404, f"Unknown env id: {env_id}")

            status, payload = _worker_request(
                "GET",
                f"{worker.base_url}/observation",
                params={"id": env_id},
                timeout_s=2400.0,
            )
            return _json_response(self, status, payload)

        return _error(self, 404, f"Unknown path: {parsed.path}")

    def do_POST(self):  # noqa: N802
        parsed = urlparse(self.path)
        try:
            data = self._read_json()
        except Exception as e:
            return _error(self, 400, str(e))

        if parsed.path == "/create":
            # Unlimited policy: every create spawns a new worker process.
            worker_host = "127.0.0.1"
            retries = 8
            last_err: Optional[Exception] = None
            for _ in range(retries):
                port = _pick_free_port(worker_host)
                try:
                    proc = _spawn_worker(worker_host=worker_host, worker_port=port, startup_timeout_s=30.0)
                    # Create env inside worker and use the worker's env_id as global id.
                    status, payload = _worker_request(
                        "POST",
                        f"http://{worker_host}:{port}/create",
                        json_body=data,
                        timeout_s=2400.0,
                    )
                    if status != 200:
                        raise RuntimeError(f"worker /create failed: {payload}")
                    env_id = str(payload.get("id") or "").strip()
                    if not env_id:
                        raise RuntimeError(f"worker /create missing id: {payload}")

                    w = _Worker(
                        env_id=env_id,
                        host=worker_host,
                        port=port,
                        proc=proc,
                        created_at=time.time(),
                        last_used_at=time.time(),
                    )
                    with _LOCK:
                        _ENV_TO_WORKER[env_id] = w
                    return _json_response(self, 200, {"id": env_id})
                except Exception as e:
                    last_err = e
                    try:
                        _terminate_process(proc)  # type: ignore[name-defined]
                    except Exception:
                        pass
                    continue
            return _error(self, 500, f"Failed to create env after retries: {last_err}")

        # All other POST endpoints expect an id in JSON.
        env_id = str((data.get("id") or "")).strip()
        if not env_id:
            return _error(self, 400, "Missing field: id")

        with _LOCK:
            worker = _ENV_TO_WORKER.get(env_id)
            if worker:
                worker.last_used_at = time.time()
        if worker is None:
            # Keep close idempotent-ish.
            if parsed.path == "/close":
                return _json_response(self, 200, {"status": "closed"})
            return _error(self, 404, f"Unknown env id: {env_id}")

        if parsed.path == "/reset":
            status, payload = _worker_request(
                "POST",
                f"{worker.base_url}/reset",
                json_body=data,
                timeout_s=2400.0,
            )
            return _json_response(self, status, payload)

        if parsed.path == "/step":
            status, payload = _worker_request(
                "POST",
                f"{worker.base_url}/step",
                json_body=data,
                timeout_s=2400.0,
            )
            return _json_response(self, status, payload)

        if parsed.path == "/close":
            # Proxy close then terminate worker process.
            status, payload = _worker_request(
                "POST",
                f"{worker.base_url}/close",
                json_body=data,
                timeout_s=30.0,
            )
            with _LOCK:
                _ENV_TO_WORKER.pop(env_id, None)
            _terminate_process(worker.proc)
            return _json_response(self, status, payload)

        return _error(self, 404, f"Unknown path: {parsed.path}")


class _GatewayHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    args = parser.parse_args(argv)

    _install_signal_handlers()

    httpd = _GatewayHTTPServer((args.host, args.port), _GatewayHandler)
    print(f"[DeliveryBenchAgentGymGateway] listening on http://{args.host}:{args.port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            httpd.server_close()
        except Exception:
            pass
        _cleanup_all_workers()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

