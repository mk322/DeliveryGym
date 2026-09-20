#!/usr/bin/env python3
"""An OpenAI-compatible endpoint whose "model" is whoever answers a file.

Both evaluators in this repository -- the waypoint benchmark
(``python -m embodiedbench.eval run --base-url ...``) and the any-point
runner (``QWEN_ENDPOINT=...``) -- talk to a chat-completions endpoint and
nothing else. This server is that endpoint with no model behind it: every
request is written to a directory as the text the policy would read and
the images it would see, and the reply is whatever appears in
``answer.txt`` there. It is how a human, or an interactive model that has
no API, is run as the policy under exactly the rules a served model gets:
same prompt, same images, same turn budget, same refusals. Nothing about
the harness changes; the harness cannot tell.

    python tools/relay_policy_server.py --dir /tmp/relay --port 8600 --model relay

    # in another shell, the evaluator or the any-point launcher pointed at it:
    python -m embodiedbench.eval run --model relay --base-url http://127.0.0.1:8600/v1 ...
    QWEN_ENDPOINT=http://127.0.0.1:8600/v1/chat/completions QWEN_MODEL_NAME=relay ...

Each request becomes ``<dir>/turn-NNNN/`` holding ``prompt.txt`` (the
system prompt and every message, images replaced by their file names),
``img-K.png`` for each image in the request, ``composite.png`` (the images
of the last user message stacked, each captioned with its index -- nothing
the message text does not already say), and ``request.json``. The server
then waits for ``<dir>/turn-NNNN/answer.txt`` and returns its contents as
the assistant message. ``<dir>/pending`` names the turn being waited on.

What the answering party sees is exactly what a served model would see;
the composite is a convenience for a viewer that opens one file per turn,
not extra information.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def _split_content(content) -> tuple[str, list[tuple[str, bytes]]]:
    """Return (text, [(mime, bytes)]) for an OpenAI message content field."""
    if isinstance(content, str):
        return content, []
    text_parts: list[str] = []
    images: list[tuple[str, bytes]] = []
    for part in content or []:
        kind = part.get("type")
        if kind == "text":
            text_parts.append(part.get("text", ""))
        elif kind == "image_url":
            url = (part.get("image_url") or {}).get("url", "")
            match = re.match(r"data:([^;]+);base64,(.*)$", url, re.S)
            if match:
                images.append((match.group(1), base64.b64decode(match.group(2))))
                text_parts.append(f"[image {len(images)}]")
            else:
                text_parts.append(f"[image url {url[:60]}]")
    return "\n".join(text_parts), images


def _composite(images: list[bytes], out: Path) -> bool:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return False
    frames = []
    for raw in images:
        try:
            frames.append(Image.open(io.BytesIO(raw)).convert("RGB"))
        except Exception:
            continue
    if not frames:
        return False
    width = max(f.width for f in frames)
    label_h = 18
    height = sum(f.height + label_h for f in frames)
    canvas = Image.new("RGB", (width, height), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    y = 0
    for index, frame in enumerate(frames, 1):
        draw.text((4, y + 2), f"image {index}  ({frame.width}x{frame.height})", fill=(255, 255, 255))
        y += label_h
        canvas.paste(frame, (0, y))
        y += frame.height
    canvas.save(out)
    return True


class Relay:
    def __init__(self, root: Path, model: str, answer_timeout: float) -> None:
        self.root = root
        self.model = model
        self.answer_timeout = answer_timeout
        self.lock = threading.Lock()
        self.root.mkdir(parents=True, exist_ok=True)
        existing = [int(p.name.split("-")[1]) for p in self.root.glob("turn-*") if p.name.split("-")[1].isdigit()]
        self.counter = max(existing, default=0)

    def next_turn_dir(self) -> Path:
        with self.lock:
            self.counter += 1
            path = self.root / f"turn-{self.counter:04d}"
        path.mkdir(parents=True, exist_ok=False)
        return path

    def handle(self, payload: dict) -> str:
        turn_dir = self.next_turn_dir()
        (turn_dir / "request.json").write_text(json.dumps(payload, indent=1))
        lines: list[str] = []
        image_index = 0
        last_user_images: list[bytes] = []
        for message in payload.get("messages", []):
            role = message.get("role", "?")
            text, images = _split_content(message.get("content"))
            names = []
            for mime, raw in images:
                image_index += 1
                ext = "png" if "png" in mime else "jpg"
                name = f"img-{image_index}.{ext}"
                (turn_dir / name).write_bytes(raw)
                names.append(name)
            if names:
                for k, name in enumerate(names, 1):
                    text = text.replace(f"[image {k}]", f"[image {k}: {name}]", 1)
            if role == "user":
                last_user_images = [raw for _, raw in images]
            lines.append(f"===== {role} =====\n{text}\n")
        (turn_dir / "prompt.txt").write_text("\n".join(lines))
        if last_user_images:
            _composite(last_user_images, turn_dir / "composite.png")
        (self.root / "pending").write_text(turn_dir.name + "\n")
        answer_path = turn_dir / "answer.txt"
        deadline = time.monotonic() + self.answer_timeout
        while not answer_path.exists():
            if time.monotonic() > deadline:
                raise TimeoutError(f"no answer for {turn_dir.name} within {self.answer_timeout:.0f} s")
            time.sleep(0.5)
        time.sleep(0.2)  # let a writer finish
        answer = answer_path.read_text()
        (self.root / "pending").write_text("")
        (turn_dir / "response.json").write_text(json.dumps({"content": answer}, indent=1))
        return answer


def make_handler(relay: Relay):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quiet
            sys.stderr.write("relay: " + (fmt % args) + "\n")

        def _json(self, status: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.rstrip("/") in ("/v1/models", "/models"):
                self._json(200, {"object": "list", "data": [
                    {"id": relay.model, "object": "model", "created": 0, "owned_by": "relay"}]})
            else:
                self._json(404, {"error": {"message": "not found"}})

        def do_POST(self):
            if self.path.rstrip("/") not in ("/v1/chat/completions", "/chat/completions"):
                self._json(404, {"error": {"message": "not found"}})
                return
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            try:
                answer = relay.handle(payload)
            except TimeoutError as error:
                self._json(504, {"error": {"message": str(error), "type": "timeout"}})
                return
            self._json(200, {
                "id": f"relay-{int(time.time() * 1000)}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": payload.get("model") or relay.model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": answer},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            })
    return Handler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", type=Path, required=True, help="where turns are written and answers read")
    parser.add_argument("--port", type=int, default=8600)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--model", default="relay", help="the served model name")
    parser.add_argument("--answer-timeout", type=float, default=3600.0,
                        help="seconds to wait for answer.txt before failing the request")
    args = parser.parse_args(argv)
    relay = Relay(args.dir.resolve(), args.model, args.answer_timeout)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(relay))
    print(f"relay policy server: http://{args.host}:{args.port}/v1  model={args.model}  dir={relay.root}",
          flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
