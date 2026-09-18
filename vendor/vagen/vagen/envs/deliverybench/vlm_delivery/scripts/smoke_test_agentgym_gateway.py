"""
Smoke test for DeliveryBench AgentGym Gateway server.

Usage:
  python -m vlm_delivery.scripts.smoke_test_agentgym_gateway --gateway http://127.0.0.1:8010 --n 8

Expected:
  - create N env_ids
  - reset each
  - observation returns 2 images (global/local)
  - one simple step succeeds
  - close all
"""

from __future__ import annotations

import argparse
import base64
import time

import requests


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gateway", default="http://127.0.0.1:8010")
    parser.add_argument("--n", type=int, default=8)
    args = parser.parse_args(argv)

    base = args.gateway.rstrip("/")
    n = int(args.n)

    ids: list[str] = []

    # Create envs
    for i in range(n):
        r = requests.post(
            base + "/create",
            json={
                "map_name": "medium-city-22",
                "max_steps": 10,
                # force Qt if available; fallback is handled worker-side
                "map_renderer": "qt",
            },
            timeout=120,
        )
        r.raise_for_status()
        env_id = r.json()["id"]
        ids.append(env_id)
        print("created", i, env_id)

    # Interleave reset + observation + step
    for i, env_id in enumerate(ids):
        r = requests.post(base + "/reset", json={"id": env_id, "data_idx": i}, timeout=2400)
        r.raise_for_status()
        print("reset", env_id, "ok")

    for env_id in ids:
        r = requests.get(base + "/observation", params={"id": env_id}, timeout=2400)
        r.raise_for_status()
        obs = r.json()
        imgs = obs.get("images") or []
        assert len(imgs) == 2, f"expected 2 images, got {len(imgs)}"
        # validate PNG headers
        for j, b64 in enumerate(imgs):
            raw = base64.b64decode(b64)
            assert raw[:8] == b"\x89PNG\r\n\x1a\n", f"image {j} not PNG"
        print("observation", env_id, "ok (2 images)")

        r = requests.post(base + "/step", json={"id": env_id, "action": "VIEW_ORDERS()"}, timeout=2400)
        r.raise_for_status()
        print("step", env_id, "ok")
        time.sleep(0.05)

    # Close all
    for env_id in ids:
        try:
            r = requests.post(base + "/close", json={"id": env_id}, timeout=60)
            r.raise_for_status()
        except Exception as e:
            print("close failed", env_id, e)
        else:
            print("closed", env_id)

    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

