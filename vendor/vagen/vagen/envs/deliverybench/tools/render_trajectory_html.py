"""
Render an agent<->env interaction trajectory as a single HTML page.

Matches the log schema emitted by the current rollout_qwen.py. It expects:
  - one {event:"system", system_text} record (the system message), and
  - per step {step, obs_in_text, model_response, action, reasoning,
    action_error, obs_out_text, images:[fpv,map], sim_hours, earnings,
    deliveries, reward, latency_s}.

The page shows the system message once at the top, then per turn:
  1. INPUT / OBSERVATION the model received this turn (text + images)
  2. MODEL RESPONSE (raw output + parsed action)
  3. OBSERVATION AFTER the action executed (text + resulting images)

Usage:
    python -m vagen.envs.deliverybench.tools.render_trajectory_html
    python -m vagen.envs.deliverybench.tools.render_trajectory_html \\
        --run_dir vagen/envs/deliverybench/outputs/rollout_20260616_011611
"""

from __future__ import annotations

import base64
import html
import json
import mimetypes
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

_PKG_OUTPUTS = Path(__file__).resolve().parent.parent / "outputs"


def _metric_badges(traj: Dict[str, Any], metric_names: List[str]) -> str:
    """Render a ✓/✗ badge per metric for a single rollout trajectory."""
    try:
        from ..benchmark import CHECKS
    except Exception:
        return ""
    names = [n for n in (metric_names or list(CHECKS)) if n in CHECKS]
    parts: List[str] = []
    for name in names:
        try:
            ok = bool(CHECKS[name](traj))
        except Exception:
            continue
        cls, sym = ("mok", "✓") if ok else ("mno", "✗")
        parts.append(f'<span class="mbadge {cls}">{html.escape(name)} {sym}</span>')
    return "".join(parts)


def _latest_run_dir() -> Path:
    runs = sorted(
        (p for p in _PKG_OUTPUTS.glob("rollout_*") if p.is_dir()),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    if not runs:
        raise FileNotFoundError(f"no rollout_* dirs under {_PKG_OUTPUTS}")
    return runs[0]


_CSS = """
  body { font-family: ui-sans-serif, system-ui; margin: 24px auto; max-width: 1200px;
         color: #1f2328; background: #f6f7f9; }
  h1 { margin: 0 0 4px; font-size: 20px; }
  /* rollout selector */
  .tabbar { position: sticky; top: 0; z-index: 50; display: flex; flex-wrap: wrap; gap: 6px;
            padding: 10px 0; margin-bottom: 12px; background: #f6f7f9; border-bottom: 1px solid #d8dee4; }
  .tab { font: inherit; font-size: 13px; padding: 5px 11px; border: 1px solid #d0d7de;
         border-radius: 16px; background: #fff; color: #24292f; cursor: pointer; }
  .tab:hover { background: #eef1f4; }
  .tab.active { background: #1f6feb; border-color: #1f6feb; color: #fff; font-weight: 600; }
  .sub { color: #656d76; font-size: 13px; margin-bottom: 16px; }
  .sub code { background: #eaeef2; padding: 1px 5px; border-radius: 3px; }
  .metrics-row { margin-top: 6px; display: flex; flex-wrap: wrap; gap: 6px; }
  .mbadge { font-size: 12px; padding: 2px 8px; border-radius: 12px; font-weight: 600; }
  .mbadge.mok { background: #dafbe1; color: #1a7f37; }
  .mbadge.mno { background: #ffebe9; color: #cf222e; }
  details.sys { background: #fff; border: 1px solid #d8dee4; border-radius: 10px;
                padding: 10px 14px; margin-bottom: 20px; }
  details.sys summary { cursor: pointer; font-weight: 700; }
  .turn { background: #fff; border: 1px solid #d8dee4; border-radius: 10px;
          margin-bottom: 18px; overflow: hidden; }
  .hdr { display: flex; align-items: center; gap: 12px; padding: 9px 14px;
         background: #f0f2f5; border-bottom: 1px solid #e2e6ea; font-size: 13px; }
  .hdr .step { font-weight: 700; font-size: 15px; }
  .hdr .state { color: #57606a; margin-left: auto; font-variant-numeric: tabular-nums; }
  .badge { padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 700; letter-spacing: .4px; }
  .ok  { background: #dafbe1; color: #1a7f37; }
  .err { background: #ffebe9; color: #cf222e; }
  .sec { padding: 12px 14px; border-top: 1px solid #eef1f4; }
  .sec:first-child { border-top: none; }
  .lbl { font-size: 10px; letter-spacing: 1px; text-transform: uppercase; font-weight: 700; margin: 0 0 7px; }
  .sec.inp > .lbl { color: #0969da; }
  .sec.out > .lbl { color: #6639ba; }
  .sec.res > .lbl { color: #bc8b00; }
  .imgs { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 8px; }
  .imgs figure { margin: 0; }
  .imgs figcaption { font-size: 11px; color: #8a929b; text-align: center; margin-top: 3px; }
  .imgs img { height: 220px; border: 1px solid #d8dee4; border-radius: 6px; cursor: zoom-in; }
  /* click-to-zoom lightbox */
  #lb { display: none; position: fixed; inset: 0; z-index: 1000;
        background: rgba(0,0,0,.85); align-items: center; justify-content: center;
        overflow: auto; cursor: zoom-out; }
  #lb img { max-width: 96vw; max-height: 96vh; border-radius: 6px; cursor: zoom-in; }
  #lb img.z { max-width: none; max-height: none; cursor: zoom-out; }  /* 1:1 native size */
  #lbhint { position: fixed; top: 12px; left: 50%; transform: translateX(-50%);
            color: #fff; font-size: 12px; opacity: .8; z-index: 1001; pointer-events: none; }
  .obs { font-family: ui-monospace, Menlo, monospace; font-size: 12px; line-height: 1.45;
         background: #f6f8fa; border: 1px solid #e2e6ea; border-radius: 6px; padding: 10px;
         white-space: pre-wrap; word-break: break-word; max-height: 340px; overflow: auto; }
  .action { font-family: ui-monospace, Menlo, monospace; background: #eef0fb; color: #3b2e80;
            padding: 5px 9px; border-radius: 5px; display: inline-block; font-weight: 600; margin-bottom: 8px; }
  .errbox { margin-top: 8px; background: #fff0ef; border: 1px solid #ffcecb; color: #b32128;
            padding: 7px 10px; border-radius: 5px; font-family: ui-monospace, Menlo, monospace; font-size: 12px; }
  .traffic { margin-top: 8px; border: 1px solid #d8dee4; border-radius: 6px; overflow: hidden;
             font-family: ui-monospace, Menlo, monospace; font-size: 12px; }
  .traffic .trow { display: flex; flex-wrap: wrap; gap: 8px; padding: 7px 10px; background: #f6f8fa; }
  .traffic .trow + .trow { border-top: 1px solid #eaeef2; }
  .traffic .pill { padding: 1px 7px; border-radius: 999px; font-weight: 700; }
  .traffic .green { background: #dafbe1; color: #1a7f37; }
  .traffic .red { background: #ffebe9; color: #cf222e; }
  .traffic .plain { background: #eaeef2; color: #57606a; }
"""


def _make_src_of(rd: Path, embed: bool) -> Callable[[str], str]:
    """Return a function mapping a relative image path to an <img> src.

    When `embed` is True, each image is read from the run dir and inlined as a
    base64 data URI (cached per path) so the HTML is fully self-contained.
    Otherwise the relative path is returned unchanged.
    """
    cache: Dict[str, str] = {}

    def src_of(rel: str) -> str:
        if not embed:
            return rel
        if rel in cache:
            return cache[rel]
        path = (rd / rel)
        try:
            data = path.read_bytes()
        except OSError:
            cache[rel] = rel
            return rel
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        b64 = base64.b64encode(data).decode("ascii")
        cache[rel] = f"data:{mime};base64,{b64}"
        return cache[rel]

    return src_of


def _img_figs(images: List[str], src_of: Callable[[str], str]) -> str:
    labels = ["first-person view", "city map"]
    figs = []
    for i, rel in enumerate(images or []):
        lbl = labels[i] if i < len(labels) else f"img{i}"
        figs.append(
            f'<figure><img src="{html.escape(src_of(rel))}" loading="lazy"/>'
            f'<figcaption>{html.escape(lbl)}</figcaption></figure>'
        )
    return f'<div class="imgs">{"".join(figs)}</div>' if figs else ""


def _traffic_light_html(sr: Dict[str, Any]) -> str:
    check = sr.get("traffic_light_check")
    is_move_action = str(sr.get("action") or "").strip().upper().startswith("MOVE")
    checks_total = sr.get("pedestrian_traffic_light_checks")
    violations_total = sr.get("pedestrian_traffic_light_violations")
    legacy_violations = sr.get("traffic_violations")
    if not check and not checks_total and not violations_total and not legacy_violations:
        return ""

    rows: List[str] = []
    if is_move_action and isinstance(check, dict):
        controlled = bool(check.get("controlled"))
        violation = bool(check.get("violation"))
        state = str(check.get("state") or ("uncontrolled" if not controlled else "unknown")).lower()
        state_cls = "red" if violation or state == "red" else ("green" if state == "green" else "plain")
        parts = [
            f'<span>edge={html.escape(str(check.get("from_waypoint_id", "?")))}'
            f'→{html.escape(str(check.get("to_waypoint_id", "?")))}</span>',
            f'<span>controlled={html.escape(str(controlled))}</span>',
            f'<span class="pill {state_cls}">state={html.escape(state)}</span>',
            f'<span>violation={html.escape(str(violation))}</span>',
        ]
        for key in ("axis", "minute", "movement_direction", "light_id", "penalty_s"):
            val = check.get(key)
            if val is not None:
                parts.append(f'<span>{html.escape(key)}={html.escape(str(val))}</span>')
        rows.append(f'<div class="trow">{"".join(parts)}</div>')

    totals = []
    if checks_total is not None:
        totals.append(f'<span>checks_total={html.escape(str(checks_total))}</span>')
    if violations_total is not None:
        totals.append(f'<span>violations_total={html.escape(str(violations_total))}</span>')
    if legacy_violations is not None:
        totals.append(f'<span>all_traffic_violations={html.escape(str(legacy_violations))}</span>')
    if totals:
        rows.append(f'<div class="trow">{"".join(totals)}</div>')

    return f'<div class="traffic">{"".join(rows)}</div>'


def _turn_card(sr: Dict[str, Any], in_imgs: List[str], out_imgs: List[str],
               src_of: Callable[[str], str]) -> str:
    step = sr["step"]
    err = sr.get("action_error")
    valid = sr.get("action_valid", err is None)
    badge = '<span class="badge ok">OK</span>' if (valid and not err) else '<span class="badge err">FAILED</span>'
    state = " · ".join([
        f"sim={sr.get('sim_hours', 0)}h", f"earn=${sr.get('earnings', 0)}",
        f"deliv={sr.get('deliveries', 0)}", f"reward={sr.get('reward', 0)}",
        f"lat={sr.get('latency_s', 0)}s",
    ])

    obs_in = html.escape(sr.get("obs_in_text") or "(not logged)")
    raw = html.escape(sr.get("model_response") or "")
    action = html.escape(str(sr.get("action") or "?"))
    traffic = _traffic_light_html(sr)

    return f"""
<div class="turn">
  <div class="hdr"><span class="step">turn {step}</span>{badge}<span class="state">{state}</span></div>

  <div class="sec inp">
    <div class="lbl">1 · observation</div>
    {_img_figs(in_imgs, src_of)}
    <div class="obs">{obs_in}</div>
  </div>

  <div class="sec out">
    <div class="lbl">2 · model response</div>
    <div class="action">parsed action: {action}</div>
    {traffic}
    <div class="obs">{raw}</div>
  </div>
</div>"""


def render(run_dir: Optional[str] = None, embed_images: bool = True) -> Path:
    rd = Path(run_dir).expanduser() if run_dir else _latest_run_dir()
    if not rd.is_absolute():
        rd = (Path.cwd() / rd).resolve()
    log_path = rd / "log.jsonl"
    if not log_path.exists():
        raise FileNotFoundError(log_path)
    src_of = _make_src_of(rd, embed_images)

    recs = [json.loads(l) for l in log_path.read_text().splitlines() if l.strip()]
    by_rollout: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    starts: Dict[int, Dict[str, Any]] = {}
    ends: Dict[int, Dict[str, Any]] = {}
    system_text = ""
    metric_names: List[str] = []
    for r in recs:
        rid = r.get("rollout_id", 0)
        ev = r.get("event")
        if ev == "step":
            by_rollout[rid].append(r)
        elif ev == "rollout_start":
            starts[rid] = r
        elif ev == "rollout_end":
            ends[rid] = r
        elif ev == "system" and not system_text:
            system_text = r.get("system_text", "")
        elif ev == "metrics" and not metric_names:
            metric_names = r.get("metrics") or []

    sys_block = (
        f'<details class="sys"><summary>system message (shown once, every turn)</summary>'
        f'<div class="obs">{html.escape(system_text)}</div></details>'
        if system_text else ""
    )

    rids = sorted(by_rollout)
    tabs: List[str] = []
    blocks: List[str] = []
    for idx, rid in enumerate(rids):
        steps = sorted(by_rollout[rid], key=lambda s: s["step"])
        end = ends.get(rid, {})
        seed = starts.get(rid, {}).get("seed", end.get("seed", "?"))

        cards = []
        # input images for turn N = images saved after turn N-1 (reset = step_000)
        prev_imgs = [f"images/rollout_{rid}/step_000_img{i}.png" for i in (0, 1)]
        for sr in steps:
            out_imgs = sr.get("images") or []
            cards.append(_turn_card(sr, prev_imgs, out_imgs, src_of))
            prev_imgs = out_imgs or prev_imgs

        delivered = end.get("deliveries", 0) or 0
        summary = (
            f"seed=<code>{seed}</code> · steps=<code>{end.get('steps', len(steps))}</code> · "
            f"deliveries=<code>{end.get('deliveries', '?')}</code> · "
            f"net=<code>${end.get('net_profit', '?')}</code> · "
            f"$/h=<code>{end.get('hourly_profit', '?')}</code>"
            + (" · <code>ABORTED</code>" if end.get("aborted") else "")
        )
        badges = _metric_badges({"steps": steps, "end": end}, metric_names)
        if badges:
            summary += f'<div class="metrics-row">{badges}</div>'

        # tab label: rollout id + seed (+ delivery check / abort marker)
        mark = " ✓" if delivered else (" ✗" if end.get("aborted") else "")
        active = " active" if idx == 0 else ""
        tabs.append(
            f'<button class="tab{active}" data-rid="{rid}" onclick="showRollout({rid})">'
            f'#{rid} · seed {seed}{mark}</button>'
        )
        disp = "" if idx == 0 else ' style="display:none"'
        blocks.append(
            f'<section class="rollout" id="rollout-{rid}"{disp}>'
            f'<h1>rollout {rid}</h1><div class="sub">{summary}</div>'
            + "".join(cards) + '</section>'
        )

    # Show the selector only when there is more than one rollout.
    tabbar = f'<div class="tabbar">{"".join(tabs)}</div>' if len(rids) > 1 else ""

    select_js = (
        '<script>'
        'function showRollout(rid){'
        ' document.querySelectorAll(".rollout").forEach(function(s){'
        '   s.style.display = (s.id === "rollout-"+rid) ? "" : "none";});'
        ' document.querySelectorAll(".tab").forEach(function(b){'
        '   b.classList.toggle("active", b.dataset.rid === String(rid));});'
        ' window.scrollTo(0, 0);'
        '}'
        '</script>'
    )

    lightbox = (
        '<div id="lb"><div id="lbhint">click image: toggle 1:1 · click backdrop: close</div>'
        '<img id="lbimg"></div>'
        '<script>'
        'const lb=document.getElementById("lb"),lbi=document.getElementById("lbimg");'
        'document.addEventListener("click",function(e){'
        ' if(e.target.tagName==="IMG"&&e.target.closest(".imgs")){'
        '  lbi.src=e.target.src;lbi.classList.remove("z");lb.style.display="flex";}});'
        'lb.addEventListener("click",function(){lb.style.display="none";});'
        'lbi.addEventListener("click",function(e){e.stopPropagation();lbi.classList.toggle("z");});'
        'document.addEventListener("keydown",function(e){if(e.key==="Escape")lb.style.display="none";});'
        '</script>'
    )
    body = (
        f'<!doctype html><html><head><meta charset="utf-8">'
        f'<title>{html.escape(rd.name)} trajectory</title><style>{_CSS}</style></head>'
        f'<body>{tabbar}{sys_block}{"".join(blocks)}{select_js}{lightbox}</body></html>'
    )
    out = rd / "trajectory.html"
    out.write_text(body, encoding="utf-8")
    return out


def main(run_dir: Optional[str] = None, embed_images: bool = True) -> None:
    print(f"wrote {render(run_dir=run_dir, embed_images=embed_images)}")


if __name__ == "__main__":
    import fire
    fire.Fire(main)
