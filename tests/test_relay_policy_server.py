"""The relay endpoint: a served model whose replies come from a file."""
from __future__ import annotations

import base64
import io
import json
import socket
import sys
import threading
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import relay_policy_server as relay  # noqa: E402


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _png_bytes(width: int = 8, height: int = 6) -> bytes:
    from PIL import Image
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (200, 30, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_a_request_becomes_files_and_the_answer_file_becomes_the_reply(tmp_path):
    port = _free_port()
    server = relay.ThreadingHTTPServer(
        ("127.0.0.1", port), relay.make_handler(relay.Relay(tmp_path, "relay-test", 30.0)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as response:
            models = json.loads(response.read())
        assert [m["id"] for m in models["data"]] == ["relay-test"]

        image = base64.b64encode(_png_bytes()).decode()
        payload = {
            "model": "relay-test",
            "messages": [
                {"role": "system", "content": "You are a courier."},
                {"role": "user", "content": [
                    {"type": "text", "text": "### where you are\nRue X"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}},
                ]},
            ],
        }
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        result: dict = {}

        def call():
            with urllib.request.urlopen(request, timeout=30) as response:
                result["body"] = json.loads(response.read())
        caller = threading.Thread(target=call)
        caller.start()

        turn_dir = tmp_path / "turn-0001"
        deadline = time.monotonic() + 10
        while not (turn_dir / "prompt.txt").exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert (turn_dir / "prompt.txt").exists()
        assert (tmp_path / "pending").read_text().strip() == "turn-0001"
        prompt = (turn_dir / "prompt.txt").read_text()
        assert "===== system =====\nYou are a courier." in prompt
        assert "[image 1: img-1.png]" in prompt
        assert (turn_dir / "img-1.png").read_bytes() == _png_bytes()
        assert (turn_dir / "composite.png").exists()

        (turn_dir / "answer.txt").write_text('THOUGHT: go.\n```\nwalk_to("Rue X", "east")\n```\n')
        caller.join(timeout=15)
        assert not caller.is_alive()
        body = result["body"]
        assert body["choices"][0]["message"]["content"].startswith("THOUGHT: go.")
        assert body["choices"][0]["finish_reason"] == "stop"
        assert (tmp_path / "pending").read_text().strip() == ""
    finally:
        server.shutdown()
        server.server_close()


def test_an_unanswered_turn_times_out_as_a_gateway_error(tmp_path):
    port = _free_port()
    server = relay.ThreadingHTTPServer(
        ("127.0.0.1", port), relay.make_handler(relay.Relay(tmp_path, "relay-test", 0.6)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode(),
            headers={"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(request, timeout=10)
        except urllib.error.HTTPError as error:
            assert error.code == 504
        else:
            raise AssertionError("an unanswered turn must fail, not hang or succeed")
    finally:
        server.shutdown()
        server.server_close()
