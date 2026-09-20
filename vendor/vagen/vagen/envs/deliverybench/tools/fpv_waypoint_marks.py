"""Numbered waypoint markers on baked DeliveryBench FPV (F0 feasibility support).

Renders the agent's reachable waypoints (one-hop `city_map.adjacents()`) as
numbered glowing markers on the baked 4-yaw FPV photos, composed into the SAME
labelled cross layout as `DeliveryBench._build_fpv_cross` (BACK top, LEFT |
FRONT | RIGHT middle, front ½×½, sides ¼×¼, panel-direction swap included).

Standalone by design: reads the FPV manifest directly and uses each entry's
absolute `image_path` (the baked images live in a sibling checkout on the same
shared disk; the env's own local-`images/` lookup does not apply here). No
existing env code is modified — this backs the F0 probe only.

Conventions mirrored from deliverybench_env.py:
  * stored_yaw = (fpv_yaw_offset_deg − compass_dir) % 360   (reflection, off=90)
  * panel → compass dir: front=+0, right=+270, back=+180, left=+90 (the swap)
  * one unknown is empirically calibrated, not assumed: whether in-photo +x
    (right) corresponds to compass-clockwise (`mirror=False`) or ccw
    (`mirror=True`) relative bearing. Run `--calibrate` and eyeball the output
    against the top-down map before trusting either value.

Demo / calibration:
  PYTHONPATH=. python -m vagen.envs.deliverybench.tools.fpv_waypoint_marks \
      --map small-city-15 --seed 9000 --out /tmp/marks_demo
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

# Geometry + marker drawing were promoted into the env package for F1
# (single source of truth, shared with deliverybench_env._build_fpv_cross).
# Re-exported here so F0 probe/gallery scripts keep working unchanged.
from ..fpv_marks import (  # noqa: F401
    CAM_H_CM, FONT_BOLD, HFOV_DEG, PANEL_DIRS, overlay_marks,
    project_in_photo, wrap180)
from ..fpv_marks import draw_marker as _draw_marker  # noqa: F401

_FPV_ROOT = Path(__file__).resolve().parent.parent / "deliverybench_fpv"
_DATASET_NAMES = ("main_base_floor_road_full_1280x960_clean_floor",
                  "main_base_floor_road_full_1280x960")


def default_manifest(map_name: str) -> Path:
    for name in _DATASET_NAMES:
        p = _FPV_ROOT / map_name / name / "manifest.jsonl"
        if p.exists():
            return p
    raise FileNotFoundError(f"no FPV manifest for {map_name} under {_FPV_ROOT}")


def load_fpv_lookup(manifest: Path) -> Dict[Tuple[float, float], Dict[float, Path]]:
    """{(x_cm, y_cm): {stored_yaw: image_path}} for plain renders, via the
    manifest's absolute image_path (files live on the shared /data disk)."""
    lookup: Dict[Tuple[float, float], Dict[float, Path]] = {}
    for line in open(manifest, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        e = json.loads(line)
        if e.get("status") != "ok":
            continue
        kind = str(e.get("render_kind") or "plain").lower()
        if kind != "plain":
            continue
        pos = (round(float(e["x_cm"]), 1), round(float(e["y_cm"]), 1))
        lookup.setdefault(pos, {})[float(e["yaw"]) % 360.0] = Path(e["image_path"])
    return lookup


def nearest_pos_key(lookup, x_cm: float, y_cm: float, tol_cm: float = 5.0):
    key = (round(float(x_cm), 1), round(float(y_cm), 1))
    if key in lookup:
        return key
    best, best_d = None, tol_cm
    for k in lookup:
        d = math.hypot(k[0] - x_cm, k[1] - y_cm)
        if d <= best_d:
            best, best_d = k, d
    return best


def build_marked_cross(
    lookup: Dict[Tuple[float, float], Dict[float, Path]],
    pos_key: Tuple[float, float],
    facing_deg: float,
    candidates: List[Dict[str, Any]],
    *,
    yaw_offset_deg: float = 90.0,
    mirror: bool = False,
    draw_marks: bool = True,
) -> Optional[Image.Image]:
    """Compose the env-identical labelled FPV cross; overlay numbered markers.

    candidates: [{"index": 1, "bearing_deg": <compass>, "dist_cm": <float>}, ...]
    Markers are drawn on the FINAL canvas (fixed pixel size, legible in the
    ¼-scale side panels too), using each panel's paste offset + scale.
    """
    yaws = lookup.get(pos_key)
    if not yaws:
        return None
    facing = float(facing_deg) % 360.0

    raw: Dict[str, Optional[Image.Image]] = {}
    for label, doff in PANEL_DIRS.items():
        cdir = (facing + doff) % 360.0
        stored = (yaw_offset_deg - cdir) % 360.0
        p = yaws.get(stored)
        raw[label] = Image.open(p).convert("RGB") if p and p.exists() else None
    ref = next((im for im in raw.values() if im is not None), None)
    if ref is None:
        return None
    w, h = ref.size
    fw, fh = w // 2, h // 2
    sw, sh = w // 4, h // 4

    def _panel(label: str, size):
        im = raw.get(label)
        tile = im.resize(size).convert("RGB") if im is not None else Image.new(
            "RGB", size, (40, 40, 40))
        d = ImageDraw.Draw(tile)
        txt = f"{label.upper()} VIEW"
        tw = d.textlength(txt) if hasattr(d, "textlength") else 8 * len(txt)
        d.rectangle([0, 0, tw + 8, 16], fill=(0, 0, 0))
        d.text((4, 2), txt, fill=(255, 255, 255))
        return tile

    # panel -> (paste_x, paste_y, panel_w, panel_h)  — identical to env layout
    geom = {
        "back": ((w - sw) // 2, 0, sw, sh),
        "left": (0, sh + (fh - sh) // 2, sw, sh),
        "front": (sw, sh, fw, fh),
        "right": (sw + fw, sh + (fh - sh) // 2, sw, sh),
    }
    canvas = Image.new("RGB", (w, sh + fh), (255, 255, 255))
    for label in ("back", "left", "front", "right"):
        px, py, pw, ph = geom[label]
        canvas.paste(_panel(label, (pw, ph)), (px, py))

    if draw_marks and candidates:
        canvas, _n = overlay_marks(canvas, geom, (w, h), candidates, facing,
                                   mirror=mirror)
    return canvas


# Promoted into the env proper for F1: the same enumeration now backs the
# MOVE_TO validator, the FPV marker renderer, and the per-step candidate
# text. Re-exported here so F0 probe/gallery scripts keep working unchanged.
from ..vlm_delivery.actions.move import enumerate_candidates  # noqa: E402,F401


def _demo() -> None:  # calibration harness (F0-2)
    import argparse, asyncio
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--map", default="small-city-15")
    ap.add_argument("--seed", type=int, default=9000)
    ap.add_argument("--out", default="/tmp/marks_demo")
    ap.add_argument("--mirror", action="store_true")
    args = ap.parse_args()

    from ..deliverybench_env import DeliveryBench
    from .build_balanced_visual_sft_data import make_env_config

    async def run():
        cfg = make_env_config(map_name=args.map, max_steps=25,
                              feasible_order_step_budget=20, enable_fpv=False)
        env = DeliveryBench(cfg)
        await env.system_prompt()
        obs, _ = await env.reset(seed=args.seed)
        out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
        dm = env._env.dms[0]
        lookup = load_fpv_lookup(default_manifest(args.map))
        cands = enumerate_candidates(dm)
        pos = nearest_pos_key(lookup, float(dm.x), float(dm.y))
        print(f"dm at ({dm.x:.1f},{dm.y:.1f}) facing={dm.facing_deg} pos_key={pos}")
        for c in cands:
            print(f"  mark {c['index']}: {c['id']:12s} {c['name'][:28]:28s} "
                  f"bearing={c['bearing_deg']:6.1f} dist={c['dist_cm']/100:5.1f}m")
        # save the top-down map for cross-checking
        pil = [im for v in (obs.get("multi_modal_input") or {}).values()
               for im in (v if isinstance(v, list) else [v])]
        if pil:
            pil[0].save(out / "topdown_map.png")
        for facing in (0.0, 90.0, 180.0, 270.0):
            img = build_marked_cross(lookup, pos, facing, cands, mirror=args.mirror)
            if img is None:
                print(f"facing {facing}: no FPV photos at pos_key {pos}")
                continue
            img.save(out / f"cross_f{int(facing):03d}{'_m' if args.mirror else ''}.png")
        print("wrote", out)
        await env.close()

    asyncio.run(run())


if __name__ == "__main__":
    _demo()
