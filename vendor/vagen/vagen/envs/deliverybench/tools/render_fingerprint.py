"""Render-fingerprint check for DeliveryBench observations.

Motivation: a silent renderer change (July 2026) once shifted the map image
for identical env state and broke a trained checkpoint 89% -> 22%. This tool
pins the observation channels to content hashes at a fixed, teleported state
so any drift is caught BEFORE data generation, training, or evaluation.

Fingerprinted channels (map small-city-15, seed 9000, teleport to the first
manifest waypoint, facing 0):
  * classic env (flag off):  map image pixels, plain FPV cross pixels, obs text
  * waypoint-marks env:      marked FPV cross pixels, obs text (### waypoint_marks)

Usage:
  PYTHONPATH=. python -m vagen.envs.deliverybench.tools.render_fingerprint --write
  PYTHONPATH=. python -m vagen.envs.deliverybench.tools.render_fingerprint --check

--write stores baselines in render_fingerprint_baseline.json next to this
file; --check exits 1 with a per-channel diff when anything moved. Re-run
--write deliberately (and say so in the commit) after an INTENDED render
change.
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict

BASELINE_PATH = Path(__file__).resolve().parent / "render_fingerprint_baseline.json"
MAP = "small-city-15"
SEED = 9000


def _h(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:12]


def _img_hash(img) -> str:
    return _h(img.tobytes())


async def collect() -> Dict[str, Any]:
    from ..deliverybench_env import DeliveryBench
    from .build_balanced_visual_sft_data import make_env_config
    from .fpv_waypoint_marks import default_manifest, load_fpv_lookup

    fp: Dict[str, Any] = {"map": MAP, "seed": SEED}
    anchor = sorted(load_fpv_lookup(default_manifest(MAP)).keys())[0]
    fp["anchor_pos"] = list(anchor)

    async def obs_at_anchor(cfg):
        env = DeliveryBench(cfg)
        await env.system_prompt()
        obs, _ = await env.reset(seed=SEED)
        dm = env._env.dms[0]
        dm.x, dm.y = anchor
        dm.facing_deg = 0.0
        obs = await env._build_observation({}, {}, False)
        sp = (await env.system_prompt())
        sp = sp["obs_str"] if isinstance(sp, dict) else str(sp)
        await env.close()
        return obs, sp

    # classic env (enable_waypoint_marks OFF) — regression dimension
    cfg = make_env_config(map_name=MAP, max_steps=25,
                          feasible_order_step_budget=20, enable_fpv=True)
    cfg = dataclasses.replace(
        cfg, fpv_dir=str(default_manifest(MAP).parent))
    obs, sp = await obs_at_anchor(cfg)
    imgs = (obs.get("multi_modal_input") or {}).get("<image>", [])
    fp["classic_n_images"] = len(imgs)
    fp["classic_fpv_cross"] = _img_hash(imgs[0]) if imgs else None
    fp["classic_map"] = _img_hash(imgs[1]) if len(imgs) > 1 else None
    fp["classic_obs_text"] = _h(obs["obs_str"].encode())
    fp["classic_sys_prompt"] = _h(sp.encode())

    # waypoint-marks env (flag ON)
    cfg_m = make_env_config(map_name=MAP, max_steps=25,
                            feasible_order_step_budget=20, enable_fpv=True,
                            waypoint_marks=True)
    obs_m, sp_m = await obs_at_anchor(cfg_m)
    imgs_m = (obs_m.get("multi_modal_input") or {}).get("<image>", [])
    fp["marks_n_images"] = len(imgs_m)
    fp["marks_fpv_cross"] = _img_hash(imgs_m[0]) if imgs_m else None
    fp["marks_obs_text"] = _h(obs_m["obs_str"].encode())
    fp["marks_sys_prompt"] = _h(sp_m.encode())
    fp["marks_block_present"] = "### waypoint_marks" in obs_m["obs_str"]
    return fp


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true",
                      help="record the current hashes as the new baseline")
    mode.add_argument("--check", action="store_true",
                      help="compare against the stored baseline; exit 1 on drift")
    args = ap.parse_args(argv)

    fp = asyncio.run(collect())
    if args.write:
        BASELINE_PATH.write_text(json.dumps(fp, indent=2) + "\n")
        print(f"baseline written: {BASELINE_PATH}")
        print(json.dumps(fp, indent=2))
        return

    if not BASELINE_PATH.exists():
        print(f"no baseline at {BASELINE_PATH}; run --write first", file=sys.stderr)
        sys.exit(2)
    base = json.loads(BASELINE_PATH.read_text())
    drift = {k: (base.get(k), fp.get(k))
             for k in sorted(set(base) | set(fp)) if base.get(k) != fp.get(k)}
    if drift:
        print("RENDER FINGERPRINT DRIFT — do not train/eval until explained:")
        for k, (b, c) in drift.items():
            print(f"  {k}: baseline={b}  current={c}")
        sys.exit(1)
    print(f"render fingerprint OK ({len(base)} channels match baseline)")


if __name__ == "__main__":
    main()
