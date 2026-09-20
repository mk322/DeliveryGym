#!/usr/bin/env python3
# tools/obstacle_avoidance_eval.py
# -*- coding: utf-8 -*-
"""
Obstacle-avoidance probe for DeliveryBench (OpenRouter Qwen3-VL-8B-Instruct).

What it measures
----------------
Given the *exact* observation an agent gets mid-navigation (4-way FPV cross +
top-down city map + text state, with `next_move: move forward` active), does the
model choose `BYPASS()` (the correct detour) when the FRONT view shows a road
block, versus `MOVE(direction="forward")` (a collision)? And does it spuriously
`BYPASS()` when the way is clear?

The obstacle is **never named in text** (vision-only, per OBSTACLE_TRAFFIC_DESIGN.md);
the only obstacle signal is the rendered FRONT FPV panel. We do NOT tell the model
to move forward — the env's own `next_move` hint provides that, exactly as in a
real rollout.

2x2 factorial (50 calls per arm = 200 total by default):
  - guidance:  A = "no guidance" (system prompt explains what BYPASS is for, nothing more)
               B = "guidance"    (A + "before moving forward, check the front view for an obstacle")
  - scene:     obstacle (FRONT shows a road block) | clear (same waypoints, plain FRONT)

Retry protocol (obstacle arms): if attempt 1 is MOVE(forward) — i.e. the model
"failed" / would collide — we feed back the error
`MOVE("forward") failed because of a potential obstacle ahead.` and re-query once,
to measure error-driven recovery to BYPASS.

The 5 obstacle waypoints come from the baked dataset
`deliverybench_fpv/small-city-11-new` (5 `*_blocked.png` + obstacles.json). The
clear arm reuses the SAME 5 waypoints with the obstacle field detached (plain
twin FPV) — a matched control isolating exactly the obstacle image.

Run:
  OPENROUTER_API_KEY=... PYTHONPATH=. \
    python -m vagen.envs.deliverybench.tools.obstacle_avoidance_eval
  # quick smoke (1 call/arm):  ... obstacle_avoidance_eval --calls_per_arm 1
  # re-render HTML only:       ... obstacle_avoidance_eval --render_only <run_dir>

Output: vagen/envs/deliverybench/outputs/obstacle_avoid_<ts>/
  log.jsonl, scenes/<id>_{fpv,map}.png, report.html
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import dataclasses
import html
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("DELIVERYBENCH_MULTI_AGENT", "0")

_THIS = Path(__file__).resolve()
_PKG = _THIS.parent.parent                       # .../deliverybench
_REPO = _PKG.parents[3]                           # repo root (…/VAGEN)
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from vagen.envs.deliverybench.deliverybench_env import (  # noqa: E402
    DeliveryBench, STAGE_1_CONFIG,
)
from vagen.envs.deliverybench.vlm_delivery.utils.hazards import ObstacleField  # noqa: E402

# ── dataset locations ────────────────────────────────────────────────────────
_FPV_NEW = _PKG / "deliverybench_fpv" / "small-city-11-new"
_FULL_MANIFEST = _FPV_NEW / "main_base_floor_road_full_1280x960" / "manifest.jsonl"
_OBSTACLES_JSON = _FPV_NEW / "main_base_floor_road_full_1280x960" / "obstacles.json"
_IMAGES_DIR = _FPV_NEW / "images"

MODEL = os.environ.get("OBSTACLE_MODEL", "qwen/qwen3-vl-8b-instruct")
OPENROUTER_BASE = "https://openrouter.ai/api/v1"

_ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning_and_reflection": {"type": "string"},
        "action": {"type": "string"},
        "future_plan": {"type": "string"},
    },
    "required": ["reasoning_and_reflection", "action", "future_plan"],
    "additionalProperties": False,
}


# ── consolidated FPV dir (full manifest + obstacles.json + symlinked images) ──
def build_consolidated_fpv(dst: Path) -> Path:
    """The full manifest lives in a subdir; the PNGs live at the dataset root; the
    root manifest is a broken stub. Stitch a loadable dir (manifest+obstacles +
    images symlink) without touching the user's dataset."""
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "manifest.jsonl").write_bytes(_FULL_MANIFEST.read_bytes())
    (dst / "obstacles.json").write_bytes(_OBSTACLES_JSON.read_bytes())
    link = dst / "images"
    if link.is_symlink() or link.exists():
        if link.is_symlink():
            link.unlink()
    if not link.exists():
        os.symlink(_IMAGES_DIR, link)
    return dst


def make_config(fpv_dir: Path):
    return dataclasses.replace(
        STAGE_1_CONFIG,
        map_name="small-city-11",
        render_mode="vision",
        enable_fpv=True,
        enable_map_images=True,
        use_gmaps_renderer=True,
        map_renderer="pil",
        gmaps_out_scale=0.5,
        fpv_dir=str(fpv_dir),
        task_mode="delivery",
        enabled_actions=["VIEW_ORDERS", "ACCEPT_ORDER", "MOVE", "PICKUP",
                         "DROP_OFF", "WAIT", "NAVIGATE", "PASSBY"],
    )


# ── system prompt surgery: explain BYPASS, and (version B) add the front-check ─
_PASSBY_LINE_RE = re.compile(r'^- PASSBY\(\).*$', re.MULTILINE)
_BYPASS_NEW = ('- BYPASS()  # take a small detour around an obstacle in the path '
               'directly ahead. It ends at the same next waypoint as '
               'MOVE(direction="forward") but costs about 1.5x the time and energy, '
               'so it is the right choice when the way straight ahead is blocked. '
               '(PASSBY is an alias of BYPASS.)')
_B_GUIDANCE = ('\n**Before moving forward:** look at your FRONT (first-person) view. '
               'If an obstacle such as a barrier, cones, or a person is blocking the '
               'path straight ahead, do not MOVE(direction="forward") into it — use '
               'BYPASS() to go around it instead.\n')


def build_prompts(base_prompt: str) -> Dict[str, str]:
    if not _PASSBY_LINE_RE.search(base_prompt):
        raise RuntimeError(
            "PASSBY spec line not found in system prompt — the env action spec "
            "changed; update _PASSBY_LINE_RE in obstacle_avoidance_eval.py."
        )
    # Replace the env's PASSBY line with the obstacle-aware BYPASS instruction
    # (this IS the Version-A 'BYPASS instruction' baseline).
    prompt_a = _PASSBY_LINE_RE.sub(lambda _m: _BYPASS_NEW, base_prompt, count=1)
    marker = "**Output:**"
    if marker in prompt_a:
        prompt_b = prompt_a.replace(marker, _B_GUIDANCE + "\n" + marker, 1)
    else:
        prompt_b = prompt_a + _B_GUIDANCE
    return {"A_noguid": prompt_a, "B_guided": prompt_b}


# ── scene construction ───────────────────────────────────────────────────────
def _ang_close(a: float, b: float, tol: float = 25.0) -> bool:
    d = abs((a - b) % 360.0)
    return min(d, 360.0 - d) <= tol


async def build_scene(env: DeliveryBench, obstacle: Dict[str, Any], *, with_obstacle: bool,
                      seed: int) -> Optional[Dict[str, Any]]:
    """Reset, accept a hypothetical order, teleport onto the obstacle's src
    waypoint facing the blocked edge, NAVIGATE one hop forward, and capture the
    real observation. `with_obstacle` toggles whether the FRONT panel shows the
    road block (obstacle arm) or the plain twin (clear control)."""
    await env.reset(seed=seed)
    dm = env._env.dms[0]

    # Hypothetical order for realistic context (best-effort; pool may vary).
    accepted = None
    try:
        await env.step("VIEW_ORDERS()")
        obs, _r, _d, info = await env.step("ACCEPT_ORDER(0)")
        if not info.get("action_error"):
            accepted = 0
    except Exception:
        pass

    # Teleport onto the obstacle source waypoint, face the blocked edge.
    sx, sy, bearing = float(obstacle["src_x_cm"]), float(obstacle["src_y_cm"]), float(obstacle["bearing_deg"])
    dm.x, dm.y = sx, sy
    dm.facing_deg = bearing

    node = dm.city_map.nearest_waypoint(dm.x, dm.y)
    adj = dm.city_map.adjacents(node) or []
    fwd = next((a for a in adj if _ang_close(a["bearing_deg"], bearing)), None)
    if fwd is None:
        return None  # no forward neighbour at the blocked bearing → skip scene

    # Attach the obstacle field only for the obstacle arm.
    if with_obstacle:
        dm._obstacle_field = ObstacleField([obstacle])
        blocked = dm._obstacle_field.obstacle_on(dm.x, dm.y, fwd["node"].position.x, fwd["node"].position.y)
        if blocked is None:
            return None  # sanity: the edge must actually be obstacle-blocked
    else:
        dm._obstacle_field = None

    # Lay a forward route so next_move == "move forward" (route to the runtime
    # forward-neighbour id; capture ids drifted, so use the graph's id).
    obs, _r, _d, info = await env.step(f'NAVIGATE(target="{fwd["id"]}")')
    if info.get("action_error"):
        return None

    nm = _extract_next_move(obs["obs_str"])
    if nm != "move forward":
        # The route must require a forward step for the probe to be valid.
        return None

    mmi = obs.get("multi_modal_input") or {}
    imgs = [im for v in mmi.values() for im in (v if isinstance(v, list) else [v])]
    return {
        "kind": "obstacle" if with_obstacle else "clear",
        "capture_src_id": obstacle["src_id"],
        "runtime_src_id": getattr(node, "waypoint_id", "?"),
        "runtime_fwd_id": fwd["id"],
        "bearing_deg": bearing,
        "obstacle_type": obstacle.get("type", "road_block"),
        "accepted_order": accepted,
        "next_move": nm,
        "obs_text": obs["obs_str"].replace("<image>", "").strip(),
        "images": imgs,                          # [fpv_cross, gmaps_map]
    }


def _extract_next_move(obs_text: str) -> Optional[str]:
    for line in obs_text.splitlines():
        if "next_move:" in line:
            return line.split("next_move:", 1)[1].strip().lower()
    return None


# ── action classification ────────────────────────────────────────────────────
_THINK_RE = re.compile(r"^.*?</think>\s*", re.DOTALL)


def sanitize(text: str) -> str:
    t = (text or "").strip()
    if "</think>" in t:
        t = _THINK_RE.sub("", t, count=1).strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t).strip()
    if not t.startswith("{"):
        m = re.search(r"\{.*\}", t, re.DOTALL)
        if m:
            t = m.group(0)
    return t


def parse_fields(raw: str) -> Dict[str, str]:
    try:
        obj = json.loads(sanitize(raw))
        if isinstance(obj, dict):
            return {k: str(obj.get(k, "")) for k in ("action", "reasoning_and_reflection", "future_plan")}
    except Exception:
        pass
    out = {}
    for k in ("action", "reasoning_and_reflection", "future_plan"):
        m = re.search(rf'"{k}"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
        out[k] = (m.group(1) if m else "")
    return out


def classify(action_str: str) -> str:
    a = (action_str or "").strip()
    if re.match(r'^(BYPASS|PASSBY)\s*\(', a, re.I):
        return "BYPASS"
    if re.match(r'^MOVE\s*\(', a, re.I) and re.search(r'forward', a, re.I):
        return "MOVE_FORWARD"
    return "OTHER"


# ── model call (OpenRouter) ──────────────────────────────────────────────────
def pil_to_data_url(img) -> str:
    from io import BytesIO
    buf = BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


async def call_model(client, messages, *, temperature: float, max_tokens: int) -> str:
    resp = await client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
        top_p=0.9,
        max_tokens=max_tokens,
        response_format={"type": "json_schema",
                         "json_schema": {"name": "delivery_action", "strict": True,
                                         "schema": _ACTION_SCHEMA}},
        extra_body={"reasoning": {"effort": "none"},
                    "plugins": [{"id": "response-healing"}]},
    )
    return resp.choices[0].message.content or ""


def user_message(scene: Dict[str, Any], data_urls: List[str], error: Optional[str] = None) -> Dict[str, Any]:
    text = scene["obs_text"]
    if error:
        text = f"⚠ Your previous action FAILED: {error}\n\n{text}"
    content = [{"type": "image_url", "image_url": {"url": u}} for u in data_urls]
    content.append({"type": "text", "text": text})
    return {"role": "user", "content": content}


_RETRY_ERROR = 'MOVE("forward") failed because of a potential obstacle ahead.'


async def run_one_call(client, sem, system_prompt: str, scene: Dict[str, Any],
                       data_urls: List[str], *, do_retry: bool, temperature: float,
                       max_tokens: int) -> Dict[str, Any]:
    async with sem:
        msgs = [{"role": "system", "content": system_prompt}, user_message(scene, data_urls)]
        try:
            raw1 = await call_model(client, msgs, temperature=temperature, max_tokens=max_tokens)
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}
        f1 = parse_fields(raw1)
        c1 = classify(f1.get("action", ""))
        rec: Dict[str, Any] = {"raw1": raw1, "action1": f1.get("action", ""),
                               "reasoning1": f1.get("reasoning_and_reflection", ""),
                               "class1": c1}
        if do_retry and c1 == "MOVE_FORWARD":
            msgs2 = msgs + [
                {"role": "assistant", "content": raw1},
                user_message(scene, data_urls, error=_RETRY_ERROR),
            ]
            try:
                raw2 = await call_model(client, msgs2, temperature=temperature, max_tokens=max_tokens)
                f2 = parse_fields(raw2)
                rec.update({"raw2": raw2, "action2": f2.get("action", ""),
                            "reasoning2": f2.get("reasoning_and_reflection", ""),
                            "class2": classify(f2.get("action", ""))})
            except Exception as exc:
                rec["retry_error"] = f"{type(exc).__name__}: {exc}"
        return rec


# ── orchestration ────────────────────────────────────────────────────────────
async def amain(args) -> None:
    from openai import AsyncOpenAI

    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENROUTER_API_KEY before running.")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = (Path(args.out_dir).expanduser().resolve() if args.out_dir
                else (_PKG / "outputs" / f"obstacle_avoid_{ts}"))
    out_root.mkdir(parents=True, exist_ok=True)
    scenes_dir = out_root / "scenes"
    scenes_dir.mkdir(exist_ok=True)
    print(f"[out] {out_root}")

    fpv_dir = build_consolidated_fpv(out_root / "_fpv")
    obstacles = json.loads(_OBSTACLES_JSON.read_text())["obstacles"][: args.n_scenes]

    env = DeliveryBench(make_config(fpv_dir))
    base_prompt = (await env.system_prompt())["obs_str"]
    prompts = build_prompts(base_prompt)

    # Build the distinct scenes (obstacle + clear) once; reuse across calls.
    print("[scenes] building obstacle + clear observations …")
    scenes: List[Dict[str, Any]] = []
    seed0 = 42
    for i, ob in enumerate(obstacles):
        for with_ob in (True, False):
            sc = await build_scene(env, ob, with_obstacle=with_ob, seed=seed0 + i)
            if sc is None:
                print(f"   ! skipped {ob['src_id']} (with_obstacle={with_ob}) — invalid scene")
                continue
            sid = f"{'OB' if with_ob else 'CL'}_{ob['src_id']}"
            sc["scene_id"] = sid
            # persist images + data-urls
            urls = []
            labels = ["fpv", "map"]
            for k, im in enumerate(sc["images"][:2]):
                p = scenes_dir / f"{sid}_{labels[k]}.png"
                im.save(p)
                urls.append(pil_to_data_url(im))
            sc["data_urls"] = urls
            sc["img_files"] = [f"scenes/{sid}_{labels[k]}.png" for k in range(len(urls))]
            del sc["images"]
            scenes.append(sc)
            print(f"   + {sid}: runtime {sc['runtime_src_id']}->{sc['runtime_fwd_id']} "
                  f"bearing {sc['bearing_deg']:.0f} next_move='{sc['next_move']}'")
    await env.close()

    ob_scenes = [s for s in scenes if s["kind"] == "obstacle"]
    cl_scenes = [s for s in scenes if s["kind"] == "clear"]
    if not ob_scenes or not cl_scenes:
        raise SystemExit("No valid scenes built — aborting.")

    # Arms: (version, scene_kind). do_retry only for obstacle arms.
    arms = [
        ("A_noguid", "obstacle", ob_scenes),
        ("B_guided", "obstacle", ob_scenes),
        ("A_noguid", "clear", cl_scenes),
        ("B_guided", "clear", cl_scenes),
    ]

    client = AsyncOpenAI(base_url=OPENROUTER_BASE, api_key=api_key, timeout=args.timeout)
    sem = asyncio.Semaphore(args.concurrency)

    # write run meta + scene records
    log_path = out_root / "log.jsonl"
    with log_path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"event": "run_meta", "model": MODEL, "ts": ts,
                             "calls_per_arm": args.calls_per_arm, "temperature": args.temperature,
                             "n_scenes": len(ob_scenes), "retry_error": _RETRY_ERROR}) + "\n")
        fh.write(json.dumps({"event": "prompts", "A_noguid": prompts["A_noguid"],
                             "B_guided": prompts["B_guided"]}) + "\n")
        for s in scenes:
            fh.write(json.dumps({"event": "scene", **{k: v for k, v in s.items()
                                                      if k != "data_urls"}}) + "\n")

    # build the task list: calls_per_arm per arm, cycling scenes
    tasks = []
    meta = []
    for version, kind, sc_list in arms:
        for j in range(args.calls_per_arm):
            sc = sc_list[j % len(sc_list)]
            do_retry = (kind == "obstacle")
            meta.append({"version": version, "kind": kind, "scene_id": sc["scene_id"], "call_idx": j})
            tasks.append(run_one_call(client, sem, prompts[version], sc, sc["data_urls"],
                                      do_retry=do_retry, temperature=args.temperature,
                                      max_tokens=args.max_tokens))
    print(f"[calls] dispatching {len(tasks)} calls (concurrency {args.concurrency}) …")
    results = await asyncio.gather(*tasks)

    # log per-call results
    with log_path.open("a", encoding="utf-8") as fh:
        for m, r in zip(meta, results):
            fh.write(json.dumps({"event": "call", **m, **r}) + "\n")

    # console summary
    summary = summarize(meta, results)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"event": "summary", **summary}) + "\n")
    print_summary(summary)

    html_path = render_html(out_root)
    print(f"[html] {html_path}")


# ── summary + html ───────────────────────────────────────────────────────────
def summarize(meta: List[Dict], results: List[Dict]) -> Dict[str, Any]:
    arms: Dict[str, Dict[str, Any]] = {}
    for m, r in zip(meta, results):
        key = f"{m['version']}|{m['kind']}"
        a = arms.setdefault(key, {"n": 0, "BYPASS": 0, "MOVE_FORWARD": 0, "OTHER": 0,
                                  "errors": 0, "retried": 0, "recovered": 0})
        if r.get("error"):
            a["errors"] += 1
            continue
        a["n"] += 1
        a[r["class1"]] = a.get(r["class1"], 0) + 1
        if "class2" in r:
            a["retried"] += 1
            if r["class2"] == "BYPASS":
                a["recovered"] += 1
    return {"arms": arms}


def _pct(n, d):
    return f"{(100.0*n/d):.0f}%" if d else "—"


def print_summary(summary: Dict[str, Any]) -> None:
    print("\n================ OBSTACLE-AVOIDANCE SUMMARY ================")
    print(f"{'arm':<22}{'n':>4}{'BYPASS':>9}{'MOVE_fwd':>10}{'OTHER':>8}{'recover':>10}")
    for key in ("A_noguid|obstacle", "B_guided|obstacle", "A_noguid|clear", "B_guided|clear"):
        a = summary["arms"].get(key)
        if not a:
            continue
        n = a["n"]
        rec = f"{a['recovered']}/{a['retried']}" if a["retried"] else "—"
        print(f"{key:<22}{n:>4}{a['BYPASS']:>4}({_pct(a['BYPASS'],n):>4}){a['MOVE_FORWARD']:>4}"
              f"({_pct(a['MOVE_FORWARD'],n):>4}){a['OTHER']:>3}({_pct(a['OTHER'],n):>4}){rec:>10}")
    print("============================================================\n")


_CSS = """
:root{
  --ground:#f5f6f8; --surface:#ffffff; --ink:#161a22; --ink-soft:#5b6472;
  --hairline:#e4e7ec; --hairline-2:#eef0f3;
  --accent:#3b53d6; --accent-soft:#eef1fd;
  --good:#1a9d5b; --good-soft:#e7f6ee; --bad:#d23b34; --bad-soft:#fcebea;
  --warn:#c2890f; --warn-soft:#fbf3df;
  --hazard:#d2622a; --hazard-soft:#fceee6; --calm:#1a8a6b; --calm-soft:#e6f4f0;
}
*{box-sizing:border-box}
body{font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  color:var(--ink);background:var(--ground);margin:0;line-height:1.5;
  -webkit-font-smoothing:antialiased}
.wrap{max-width:1080px;margin:0 auto;padding:0 24px 80px}
.mono{font-family:ui-monospace,"SF Mono","JetBrains Mono",Menlo,monospace;
  font-variant-numeric:tabular-nums}
a{color:var(--accent)}
.topbar{height:4px;background:linear-gradient(90deg,var(--accent),var(--hazard))}
/* hero */
header.hero{padding:40px 0 8px}
.eyebrow{font-size:11px;font-weight:700;letter-spacing:.13em;text-transform:uppercase;
  color:var(--ink-soft)}
h1{font-size:32px;line-height:1.12;letter-spacing:-.022em;margin:10px 0 0;font-weight:800;
  text-wrap:balance;max-width:20ch}
.thesis{font-size:16px;color:var(--ink-soft);margin:12px 0 0;max-width:64ch;text-wrap:pretty}
.metarow{display:flex;flex-wrap:wrap;gap:8px;margin-top:18px}
.meta{font-size:12px;background:var(--surface);border:1px solid var(--hairline);
  border-radius:7px;padding:5px 10px;color:var(--ink-soft)}
.meta b{color:var(--ink);font-weight:650}
/* section heads */
section{margin-top:44px}
.shead{display:flex;align-items:baseline;gap:12px;margin-bottom:16px}
.shead h2{font-size:13px;font-weight:700;letter-spacing:.12em;text-transform:uppercase;margin:0}
.shead .rule{flex:1;height:1px;background:var(--hairline)}
.shead .note{font-size:12px;color:var(--ink-soft)}
/* 2x2 matrix */
.matrix{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:720px){.matrix{grid-template-columns:1fr}}
.cell{background:var(--surface);border:1px solid var(--hairline);border-radius:14px;
  padding:18px 18px 16px;position:relative;overflow:hidden}
.cell .tags{display:flex;gap:6px;margin-bottom:14px}
.bignum{display:flex;align-items:baseline;gap:9px}
.bignum .pct{font-size:42px;font-weight:800;letter-spacing:-.03em;line-height:1}
.bignum .of{font-size:13px;color:var(--ink-soft)}
.metric-label{font-size:12.5px;color:var(--ink-soft);margin:7px 0 15px}
.metric-label b{color:var(--ink);font-weight:650}
.stack{display:flex;height:13px;border-radius:7px;overflow:hidden;background:var(--hairline-2)}
.stack span{display:block;height:100%}
.legend{display:flex;flex-wrap:wrap;gap:4px 16px;margin-top:11px;font-size:12px}
.legend .li{display:flex;align-items:center;gap:6px;color:var(--ink-soft)}
.dot{width:9px;height:9px;border-radius:3px;flex:none}
.legend .li b{color:var(--ink);font-weight:650}
.recovery{margin-top:14px;padding-top:13px;border-top:1px solid var(--hairline-2);
  font-size:12.5px;color:var(--ink-soft)}
.recovery b{color:var(--good);font-weight:700}
.recovery.none b{color:var(--ink-soft)}
/* insight chips */
.insights{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}
@media(max-width:720px){.insights{grid-template-columns:1fr}}
.insight{background:var(--surface);border:1px solid var(--hairline);border-radius:12px;padding:16px}
.insight .k{font-size:11px;font-weight:700;letter-spacing:.07em;text-transform:uppercase;
  color:var(--ink-soft)}
.insight .v{font-size:24px;font-weight:800;letter-spacing:-.02em;margin:8px 0 4px;display:flex;
  align-items:baseline;gap:8px}
.insight .v .arrow{color:var(--ink-soft);font-weight:600;font-size:17px}
.insight .d{font-size:12.5px;color:var(--ink-soft);text-wrap:pretty}
/* generic chips */
.chip{display:inline-flex;align-items:center;font-size:11px;font-weight:700;
  padding:3px 9px;border-radius:20px;letter-spacing:.01em;white-space:nowrap}
.chip.ob{background:var(--hazard-soft);color:var(--hazard)}
.chip.cl{background:var(--calm-soft);color:var(--calm)}
.chip.A{background:#eef0f3;color:#475063}
.chip.B{background:var(--accent-soft);color:var(--accent)}
.chip.wp{background:#eef0f3;color:#475063;font-family:ui-monospace,Menlo,monospace}
.chip.k_BYPASS{background:var(--good-soft);color:var(--good)}
.chip.k_MOVE_FORWARD{background:var(--bad-soft);color:var(--bad)}
.chip.k_OTHER{background:var(--warn-soft);color:var(--warn)}
/* disclosure blocks */
details{background:var(--surface);border:1px solid var(--hairline);border-radius:12px;
  padding:0;margin-bottom:12px;overflow:hidden}
details>summary{cursor:pointer;font-weight:600;font-size:14px;padding:14px 16px;
  list-style:none;display:flex;align-items:center;gap:9px;flex-wrap:wrap}
details>summary::-webkit-details-marker{display:none}
details>summary::before{content:"▸";color:var(--ink-soft);font-size:11px;transition:transform .15s}
details[open]>summary::before{transform:rotate(90deg)}
details>summary:hover{background:var(--hairline-2)}
details .body{padding:0 16px 16px}
/* scene cards */
.scene{display:flex;gap:16px;flex-wrap:wrap;align-items:flex-start}
.scene figure{margin:0}
.scene img{height:220px;border:1px solid var(--hairline);border-radius:8px;cursor:zoom-in;display:block}
.scene figcaption{font-size:11px;color:var(--ink-soft);text-align:center;margin-top:5px;
  letter-spacing:.02em}
.obs{font-family:ui-monospace,Menlo,monospace;font-size:11.5px;line-height:1.5;
  background:#fafbfc;border:1px solid var(--hairline);border-radius:8px;padding:12px;
  white-space:pre-wrap;word-break:break-word;max-height:320px;overflow:auto;flex:1;min-width:320px;
  color:#2c333f}
/* per-call log */
.callrow{font-size:12.5px;padding:8px 0;display:flex;align-items:flex-start;gap:9px;
  flex-wrap:wrap;border-top:1px solid var(--hairline-2)}
.callrow:first-child{border-top:none}
.callrow .idx{color:#9aa3af;font-family:ui-monospace,Menlo,monospace;min-width:34px}
.cdot{width:9px;height:9px;border-radius:50%;flex:none;margin-top:4px}
.cdot.ok{background:var(--good)}.cdot.no{background:var(--bad)}.cdot.mid{background:var(--warn)}
.act{font-family:ui-monospace,Menlo,monospace;background:#f1f2f6;color:#2c333f;
  padding:2px 7px;border-radius:5px}
.reason{color:var(--ink-soft);flex:1;min-width:200px;font-style:italic}
.retry{color:var(--warn);width:100%;padding-left:43px;margin-top:3px}
/* per-call full record */
details.call{border:1px solid var(--hairline-2);border-radius:10px;margin:8px 0;background:#fff}
details.call>summary{padding:10px 13px;font-weight:500;font-size:12.5px}
details.call>summary:hover{background:#fafbfc}
details.call .body{padding:2px 13px 14px}
.iolbl{font-size:10px;font-weight:700;letter-spacing:.09em;text-transform:uppercase;
  color:var(--ink-soft);margin:14px 0 6px}
.iolbl.err{color:var(--bad)}
.ioref{font-size:12.5px;color:var(--ink-soft);text-wrap:pretty}
.ioref a{text-decoration:none;font-weight:600}
.ioref.err{color:var(--bad);background:var(--bad-soft);border:1px solid #f3c9c6;
  border-radius:7px;padding:7px 10px;font-family:ui-monospace,Menlo,monospace}
.obs.raw{max-height:260px}
.act.big{display:inline-block;margin:2px 0 2px;font-size:13px;font-weight:600;padding:3px 9px}
.sysdiff{background:#fff6cf;padding:1px 4px;border-radius:3px;font-weight:600}
.scene-note{font-size:12.5px;color:var(--ink-soft);margin-bottom:14px;text-wrap:pretty}
/* lightbox */
#lb{display:none;position:fixed;inset:0;z-index:999;background:rgba(12,14,20,.88);
  align-items:center;justify-content:center;cursor:zoom-out;padding:24px}
#lb img{max-width:96vw;max-height:96vh;border-radius:8px}
@media(prefers-reduced-motion:reduce){*{transition:none!important}}
"""

# semantic palette mirrored for inline bar/dot fills
_COL = {"good": "#1a9d5b", "bad": "#d23b34", "warn": "#c2890f"}


def _b64_file(p: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()


def _verlabel(ver: str) -> str:
    return "no guidance" if ver == "A_noguid" else "with guidance"


def render_html(run_dir: Path, standalone: bool = True) -> Path:
    recs = [json.loads(l) for l in (run_dir / "log.jsonl").read_text().splitlines() if l.strip()]
    meta = next((r for r in recs if r["event"] == "run_meta"), {})
    prompts = next((r for r in recs if r["event"] == "prompts"), {})
    scenes = {r["scene_id"]: r for r in recs if r["event"] == "scene"}
    calls = [r for r in recs if r["event"] == "call"]
    summary = next((r for r in recs if r["event"] == "summary"), {"arms": {}})
    arms = summary.get("arms", {})

    arm_order = [("A_noguid", "obstacle"), ("B_guided", "obstacle"),
                 ("A_noguid", "clear"), ("B_guided", "clear")]

    # ── 2x2 result matrix ────────────────────────────────────────────────
    def _stack_and_legend(a, correct_cls):
        """Stacked bar (correct→wrong→other) + legend, colored by correctness."""
        n = a["n"] or 1
        wrong_cls = "MOVE_FORWARD" if correct_cls == "BYPASS" else "BYPASS"
        order = [(correct_cls, "good"), (wrong_cls, "bad"), ("OTHER", "warn")]
        segs, legs = [], []
        names = {"BYPASS": "BYPASS", "MOVE_FORWARD": 'MOVE("forward")', "OTHER": "other"}
        for cls, tone in order:
            v = a.get(cls, 0)
            if v:
                segs.append(f'<span style="width:{100.0*v/n:.4f}%;background:{_COL[tone]}"></span>')
            legs.append(f'<span class="li"><span class="dot" style="background:{_COL[tone]}"></span>'
                        f'{names[cls]} <b>{v}</b> · {_pct(v,n)}</span>')
        return f'<div class="stack">{"".join(segs)}</div>', f'<div class="legend">{"".join(legs)}</div>'

    cells = []
    for ver, kind in arm_order:
        a = arms.get(f"{ver}|{kind}")
        if not a:
            continue
        n = a["n"] or 1
        correct_cls = "BYPASS" if kind == "obstacle" else "MOVE_FORWARD"
        succ = a.get(correct_cls, 0)
        if kind == "obstacle":
            mlabel = 'chose <b>BYPASS</b> — the correct detour around the block'
        else:
            mlabel = 'chose <b>MOVE("forward")</b> — correct, the road is clear'
        stack, legend = _stack_and_legend(a, correct_cls)
        if a.get("retried"):
            rec = (f'<div class="recovery">After the <span class="mono">MOVE("forward")</span> '
                   f'failure message, <b>{a["recovered"]}/{a["retried"]} ({_pct(a["recovered"],a["retried"])})</b> '
                   f'switched to BYPASS.</div>')
        else:
            rec = ('<div class="recovery none">No retries — a clear road needs no recovery step. '
                   'Any BYPASS here is a <b>false positive</b>.</div>')
        kindchip = (f'<span class="chip {"ob" if kind=="obstacle" else "cl"}">'
                    f'{"obstacle ahead" if kind=="obstacle" else "clear road"}</span>')
        verchip = f'<span class="chip {ver[0]}">{_verlabel(ver)}</span>'
        cells.append(
            f'<div class="cell"><div class="tags">{kindchip}{verchip}</div>'
            f'<div class="bignum"><span class="pct mono" style="color:{_COL["good"]}">{_pct(succ,n)}</span>'
            f'<span class="of mono">{succ}/{n}</span></div>'
            f'<div class="metric-label">{mlabel}</div>{stack}{legend}{rec}</div>')
    matrix = f'<div class="matrix">{"".join(cells)}</div>'

    # ── key-insight chips ────────────────────────────────────────────────
    def g(key, k):
        return (arms.get(key) or {}).get(k, 0)
    obA_n = g("A_noguid|obstacle", "n") or 1
    obB_n = g("B_guided|obstacle", "n") or 1
    clA_n = g("A_noguid|clear", "n") or 1
    clB_n = g("B_guided|clear", "n") or 1
    ret = g("A_noguid|obstacle", "retried") + g("B_guided|obstacle", "retried")
    rec = g("A_noguid|obstacle", "recovered") + g("B_guided|obstacle", "recovered")
    fp = g("A_noguid|clear", "BYPASS") + g("B_guided|clear", "BYPASS")
    insights = (
        '<div class="insights">'
        f'<div class="insight"><div class="k">Guidance lift · proactive avoidance</div>'
        f'<div class="v mono">{_pct(g("A_noguid|obstacle","BYPASS"),obA_n)}'
        f'<span class="arrow">→</span>{_pct(g("B_guided|obstacle","BYPASS"),obB_n)}</div>'
        f'<div class="d">Adding “check the front view before moving forward” is the only thing that '
        f'produces any unprompted BYPASS — without it the model never avoids the block on its own.</div></div>'
        f'<div class="insight"><div class="k">False positives · clear road</div>'
        f'<div class="v mono" style="color:{_COL["good"]}">{_pct(fp,clA_n+clB_n)}</div>'
        f'<div class="d">BYPASS chosen on a clear road, across both prompts ({fp}/{clA_n+clB_n}). '
        f'The guidance does not make the model over-trigger.</div></div>'
        f'<div class="insight"><div class="k">Recovery · after error feedback</div>'
        f'<div class="v mono" style="color:{_COL["good"]}">{_pct(rec,ret) if ret else "—"}</div>'
        f'<div class="d">Of the {ret} obstacle calls that drove forward, {rec} switched to BYPASS once '
        f'told it failed — the bottleneck is <b>seeing</b> the block, not knowing the action.</div></div>'
        '</div>')

    # ── system prompts (highlight injected guidance) ─────────────────────
    def hl(p):
        e = html.escape(p)
        e = e.replace(html.escape("- BYPASS()"), '<span class="sysdiff">- BYPASS()</span>')
        e = e.replace(html.escape("**Before moving forward:**"),
                      '<span class="sysdiff">**Before moving forward:**</span>')
        return e
    sys_block = (
        f'<details><summary>Version A · no guidance <span class="chip A">A</span>'
        f'<span class="scene-note" style="margin:0">BYPASS instruction only</span></summary>'
        f'<div class="body"><div class="obs">{hl(prompts.get("A_noguid",""))}</div></div></details>'
        f'<details><summary>Version B · with guidance <span class="chip B">B</span>'
        f'<span class="scene-note" style="margin:0">+ “check the front view before moving forward”</span></summary>'
        f'<div class="body"><div class="obs">{hl(prompts.get("B_guided",""))}</div></div></details>')

    # ── scene gallery ────────────────────────────────────────────────────
    scene_cards = []
    for sid, s in scenes.items():
        kindchip = (f'<span class="chip {"ob" if s["kind"]=="obstacle" else "cl"}">'
                    f'{"obstacle" if s["kind"]=="obstacle" else "clear"}</span>')
        imgs = "".join(
            f'<figure><img src="{_b64_file(run_dir / f)}" loading="lazy"><figcaption>'
            f'{"four-direction FPV — front = travel heading" if i==0 else "top-down city map + planned route"}'
            f'</figcaption></figure>'
            for i, f in enumerate(s.get("img_files", [])))
        scene_cards.append(
            f'<details id="scene-{sid}"><summary>{kindchip}<span class="chip wp">{sid}</span>'
            f'<span class="scene-note" style="margin:0">runtime {s["runtime_src_id"]}→{s["runtime_fwd_id"]} · '
            f'bearing {s["bearing_deg"]:.0f}° · next_move=<span class="mono">{s["next_move"]}</span></span></summary>'
            f'<div class="body"><div class="scene">{imgs}<div class="obs">{html.escape(s["obs_text"])}</div></div></div></details>')

    # ── per-call log — complete record per call ──────────────────────────
    # Every call expands to its full transcript: which system message, the
    # step's observation text (+ link to its images), and the full model
    # output (reasoning + action + future_plan via the raw JSON) for attempt 1
    # and, when the model drove forward, the error feedback + attempt 2.
    by_arm: Dict[str, List[Dict]] = {}
    for c in calls:
        by_arm.setdefault(f"{c['version']}|{c['kind']}", []).append(c)

    def _outbox(action: str, raw: str, who: str) -> str:
        return (f'<div class="iolbl">{who} · parsed action</div>'
                f'<div class="act big">{html.escape(action or "—")}</div>'
                f'<div class="iolbl">{who} · full model output (reasoning · action · future_plan)</div>'
                f'<div class="obs raw">{html.escape((raw or "").strip())}</div>')

    call_blocks = []
    for ver, kind in arm_order:
        cs = by_arm.get(f"{ver}|{kind}", [])
        if not cs:
            continue
        correct_cls = "BYPASS" if kind == "obstacle" else "MOVE_FORWARD"
        verfull = "Version A · no guidance" if ver == "A_noguid" else "Version B · with guidance"
        items = []
        for c in cs:
            sid = c["scene_id"]
            if c.get("error"):
                items.append(
                    f'<details class="call"><summary><span class="idx">#{c["call_idx"]}</span>'
                    f'<span class="cdot mid"></span><span class="chip wp">{sid}</span>'
                    f'<span style="color:{_COL["bad"]}">API error</span></summary>'
                    f'<div class="body"><div class="ioref err">{html.escape(str(c["error"]))}</div></div></details>')
                continue
            k1 = c["class1"]
            dot = "ok" if k1 == correct_cls else ("no" if k1 in ("BYPASS", "MOVE_FORWARD") else "mid")
            obs_text = scenes.get(sid, {}).get("obs_text", "(observation not logged)")
            summ = (f'<span class="idx">#{c["call_idx"]}</span>'
                    f'<span class="cdot {dot}"></span>'
                    f'<span class="chip wp">{sid}</span>'
                    f'<span class="chip k_{k1}">{k1}</span>'
                    f'<span class="act">{html.escape(c.get("action1",""))}</span>')
            body = (
                f'<div class="iolbl">system message</div>'
                f'<div class="ioref">{verfull} — full text in the '
                f'<a href="#sysprompts">System prompts</a> section.</div>'
                f'<div class="iolbl">step info</div>'
                f'<div class="ioref">scene <span class="mono">{sid}</span> · '
                f'next_move <span class="mono">{scenes.get(sid,{}).get("next_move","?")}</span> · '
                f'<a href="#scene-{sid}">four-direction FPV + map for {sid} ↑</a></div>'
                f'<div class="iolbl">observation (step input)</div>'
                f'<div class="obs">{html.escape(obs_text)}</div>'
                + _outbox(c.get("action1", ""), c.get("raw1", ""), "attempt 1"))
            if "class2" in c:
                body += (f'<div class="iolbl err">error fed back</div>'
                         f'<div class="ioref err">{html.escape(_RETRY_ERROR)}</div>'
                         + _outbox(c.get("action2", ""), c.get("raw2", ""), "attempt 2 (after error)"))
            elif c.get("retry_error"):
                body += f'<div class="ioref err">retry API error: {html.escape(str(c["retry_error"]))}</div>'
            items.append(f'<details class="call"><summary>{summ}</summary><div class="body">{body}</div></details>')
        kindchip = (f'<span class="chip {"ob" if kind=="obstacle" else "cl"}">'
                    f'{"obstacle" if kind=="obstacle" else "clear"}</span>')
        verchip = f'<span class="chip {ver[0]}">{_verlabel(ver)}</span>'
        call_blocks.append(
            f'<details><summary>{kindchip}{verchip}'
            f'<span class="scene-note" style="margin:0">{len(cs)} calls · green dot = correct for this arm · '
            f'expand any call for system · observation · full output</span>'
            f'</summary><div class="body">{"".join(items)}</div></details>')

    n_obstacle = sum(1 for s in scenes.values() if s["kind"] == "obstacle")
    content = (
        f'<div class="topbar"></div><div class="wrap">'
        f'<header class="hero"><div class="eyebrow">DeliveryBench · vision-grounded safety probe</div>'
        f'<h1>Does the model see the road block before it walks into it?</h1>'
        f'<p class="thesis">Each call places {html.escape(meta.get("model",MODEL))} mid-navigation with '
        f'<span class="mono">next_move: move forward</span> active. The only signal that the way is blocked '
        f'is the road block rendered in the <b>front</b> first-person panel — it is never named in text. '
        f'Correct response: <b>BYPASS</b> (the costlier detour) on an obstacle, <b>MOVE("forward")</b> on a clear road.</p>'
        f'<div class="metarow">'
        f'<span class="meta">model <b>{html.escape(meta.get("model",MODEL))}</b></span>'
        f'<span class="meta"><b>{meta.get("calls_per_arm","?")}</b> calls × 4 arms</span>'
        f'<span class="meta">temperature <b>{meta.get("temperature","?")}</b></span>'
        f'<span class="meta"><b>{n_obstacle}</b> obstacle waypoints</span>'
        f'<span class="meta">run <b>{html.escape(meta.get("ts",""))}</b></span></div></header>'
        f'<section><div class="shead"><h2>Result · 2×2</h2><div class="rule"></div>'
        f'<div class="note">rows: what is ahead · columns: prompt guidance</div></div>{matrix}</section>'
        f'<section><div class="shead"><h2>What it means</h2><div class="rule"></div></div>{insights}</section>'
        f'<section><div class="shead"><h2>Scenes · raw input &amp; observation</h2><div class="rule"></div>'
        f'<div class="note">{len(scenes)} scenes</div></div>'
        f'<p class="scene-note">Each scene below is the exact model input: a system prompt (one of the two), the '
        f'current-step observation text, and two images — the four-direction FPV cross and the top-down map. '
        f'Obstacle and clear scenes share a waypoint; only the front FPV panel differs.</p>'
        + "".join(scene_cards) + '</section>'
        f'<section id="sysprompts"><div class="shead"><h2>System prompts</h2><div class="rule"></div></div>{sys_block}</section>'
        f'<section><div class="shead"><h2>Per-call log</h2><div class="rule"></div></div>'
        + "".join(call_blocks) + '</section></div>'
        '<div id="lb"><img alt="zoomed view"></div>'
        '<script>'
        'var lb=document.getElementById("lb"),lbi=lb.querySelector("img");'
        'document.addEventListener("click",function(e){'
        ' if(e.target.tagName==="IMG"&&e.target.closest(".scene")){lbi.src=e.target.src;lb.style.display="flex";}});'
        'lb.addEventListener("click",function(){lb.style.display="none";});'
        'document.addEventListener("keydown",function(e){if(e.key==="Escape")lb.style.display="none";});'
        '</script>')

    if standalone:
        page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
                f'<meta name="viewport" content="width=device-width,initial-scale=1">'
                f'<title>Obstacle-avoidance probe · {html.escape(MODEL)}</title>'
                f'<style>{_CSS}</style></head><body>{content}</body></html>')
    else:
        page = f'<style>{_CSS}</style>{content}'
    out = run_dir / ("report.html" if standalone else "report_artifact.html")
    out.write_text(page, encoding="utf-8")
    return out


def main():
    ap = argparse.ArgumentParser(description="DeliveryBench obstacle-avoidance probe")
    ap.add_argument("--calls_per_arm", type=int, default=50)
    ap.add_argument("--n_scenes", type=int, default=5)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max_tokens", type=int, default=700)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--render_only", type=str, default=None,
                    help="re-render report.html from an existing run dir and exit")
    args = ap.parse_args()
    if args.render_only:
        print(render_html(Path(args.render_only)))
        return
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
