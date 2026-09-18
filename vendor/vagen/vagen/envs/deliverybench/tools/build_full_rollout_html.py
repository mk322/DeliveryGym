"""Render fully-autonomous DeliveryBench rollouts as a self-contained HTML.

Reads the summary.json from qwen35_lora_rollout_full.py + the per-step images,
and shows, per episode, the full action sequence the MODEL chose (not just MOVE):
each step's observation, the action it emitted, whether the env accepted it, and
a milestone checklist + where the episode broke. Diagnostic for "does prompt-only
instruction-following survive the MOVE-only LoRA".

Run:
  PYTHONPATH=. python -m vagen.envs.deliverybench.tools.build_full_rollout_html \
    --summary <run>/rollout_eval/full_close/summary.json \
    --trace-dir <run>/rollout_eval/full_close \
    --output outputs/deliverybench_sft/full_rollout_close.html
"""
from __future__ import annotations
import argparse, base64, html, json
from pathlib import Path

ATYPE_CLASS = {"VIEW_ORDERS": "a-view", "ACCEPT_ORDER": "a-acc", "NAVIGATE": "a-nav",
               "PICKUP": "a-pick", "DROP_OFF": "a-drop", "MOVE": "a-move"}
STAGE_LABEL = {
    "delivered": "delivered ✅",
    "never_viewed_or_accepted": "never viewed/accepted an order",
    "viewed_but_never_accepted": "viewed orders but never accepted",
    "accepted_but_never_navigated": "accepted but never navigated",
    "never_reached_pickup": "navigated but never reached pickup",
    "reached_pickup_but_no_PICKUP": "reached pickup but never called PICKUP",
    "never_reached_dropoff": "picked up but never reached drop-off",
    "reached_dropoff_but_no_DROPOFF": "reached drop-off but never called DROP_OFF",
    "other": "other",
}


def _img_b64(p: Path) -> str:
    if not p.exists():
        return ""
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode("ascii")


def _milestones(ms: dict) -> str:
    order = [("accepted", "accept"), ("navigated", "navigate"), ("reached_pickup", "reach pickup"),
             ("picked", "PICKUP"), ("reached_dropoff", "reach dropoff"), ("dropped", "DROP_OFF")]
    chips = []
    for k, lbl in order:
        on = ms.get(k)
        chips.append(f"<span class='ms {'on' if on else 'off'}'>{'✓' if on else '✗'} {html.escape(lbl)}</span>")
    return " ".join(chips)


def _episode_panel(ep: dict, trace_dir: Path) -> str:
    city = html.escape(ep.get("map", "?")); seed = ep.get("seed", "?")
    success = ep.get("success")
    badge = "<span class='ok'>SUCCESS</span>" if success else "<span class='fail'>FAILED</span>"
    stage = STAGE_LABEL.get(ep.get("failure_stage", ""), ep.get("failure_stage", ""))
    steps = ep.get("steps", [])
    ac = ep.get("action_counts", {})
    ac_str = ", ".join(f"{k}×{v}" for k, v in ac.items() if k)
    meta = (f"where it broke: <b>{html.escape(stage)}</b> &nbsp;·&nbsp; steps: {ep.get('n_steps')} "
            f"&nbsp;·&nbsp; actions: {html.escape(ac_str) or '—'} &nbsp;·&nbsp; format-fails: {ep.get('format_fail', 0)}")
    if ep.get("error"):
        meta += f" &nbsp;·&nbsp; <span class='fail'>error: {html.escape(str(ep['error']))}</span>"
    ms = _milestones(ep.get("milestones", {}))

    cards = []
    for t in steps:
        img = _img_b64(trace_dir / (t.get("image") or "")) if t.get("image") else ""
        at = t.get("atype") or "?"
        cls = ATYPE_CLASS.get(at, "a-unk")
        if not t.get("ok"):
            cls += " err"
        imgtag = f"<img src='{img}'/>" if img else "<div class='noimg'>no image</div>"
        act = html.escape(str(t.get("action") or "(unparsed)"))
        hint = ""
        if t.get("obs_pickup_hint"):
            hint += "<span class='hint'>pickup_hint shown</span>"
        if t.get("obs_dropoff_hint"):
            hint += "<span class='hint'>dropoff_hint shown</span>"
        err = f"<div class='row err'>⛔ {html.escape(str(t.get('err')))}</div>" if not t.get("ok") else ""
        cards.append(
            f"<div class='card {cls}'>"
            f"<div class='stepno'>step {t.get('step')}<span class='at'>{html.escape(at)}</span></div>"
            f"{imgtag}"
            f"{hint}"
            f"<div class='act'>{act}</div>"
            f"{err}"
            f"</div>")
    return (f"<section class='ep'>"
            f"<h2>{city} &nbsp;<span class='seed'>seed {seed}</span> &nbsp;{badge}</h2>"
            f"<div class='meta'>{meta}</div>"
            f"<div class='ms-row'>{ms}</div>"
            f"<div class='strip'>{''.join(cards)}</div>"
            f"</section>")


def build(summary_path: Path, trace_dir: Path, out: Path):
    data = json.loads(summary_path.read_text())
    eps = data.get("episodes", [])
    ov = data.get("overall", {})
    thinking = data.get("thinking", "?")
    stages = data.get("failure_stages", {})
    stage_rows = "".join(
        f"<tr><td>{html.escape(STAGE_LABEL.get(k, k))}</td><td class='num'>{v}</td></tr>"
        for k, v in sorted(stages.items(), key=lambda kv: -kv[1]))
    panels = "\n".join(_episode_panel(e, trace_dir) for e in eps)
    css = """
    body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:0;background:#0f1115;color:#e6e6e6}
    .wrap{max-width:1500px;margin:0 auto;padding:28px}
    h1{font-size:24px;margin:0 0 4px} .sub{color:#9aa4b2;margin:0 0 18px;font-size:14px}
    .legend{font-size:13px;color:#9aa4b2;margin:0 0 18px;line-height:1.6}
    table.brk{border-collapse:collapse;margin:0 0 22px;font-size:13px;min-width:340px}
    table.brk td{border:1px solid #2a3340;padding:5px 10px} table.brk .num{text-align:right;color:#e0b64a;font-weight:700}
    section.ep{background:#161a21;border:1px solid #232a35;border-radius:12px;padding:18px 18px 6px;margin:0 0 22px}
    h2{font-size:18px;margin:0 0 6px} .seed{color:#8b95a4;font-weight:400;font-size:14px}
    .meta{color:#aeb6c2;font-size:13px;margin:0 0 8px}
    .ms-row{margin:0 0 14px} .ms{font-size:11px;padding:2px 7px;border-radius:10px;margin-right:4px}
    .ms.on{background:#173a22;color:#7fe0a0} .ms.off{background:#2a1c1c;color:#e08a8a}
    .strip{display:flex;flex-wrap:wrap;gap:10px;align-items:flex-start}
    .card{width:158px;background:#1c2129;border:1px solid #2a3340;border-radius:9px;padding:7px;border-left-width:4px}
    .card img{width:100%;border-radius:5px;display:block;background:#000}
    .card.a-view{border-left-color:#7a8bd0} .card.a-acc{border-left-color:#5fb0d0} .card.a-nav{border-left-color:#c78ad0}
    .card.a-pick{border-left-color:#5fd07a} .card.a-drop{border-left-color:#5fd07a} .card.a-move{border-left-color:#5a6473}
    .card.a-unk{border-left-color:#5a5320} .card.err{border-color:#7d3030;background:#241a1a}
    .stepno{font-size:11px;color:#9aa4b2;display:flex;justify-content:space-between;margin-bottom:5px;align-items:center}
    .at{font-weight:700;font-size:10px;color:#c8cfd9;background:#242b34;padding:1px 5px;border-radius:4px}
    .act{font-size:11px;margin-top:5px;color:#dfe4ea;word-break:break-word;font-family:ui-monospace,Menlo,monospace}
    .hint{display:inline-block;font-size:10px;color:#e0b64a;background:#2a2515;padding:1px 5px;border-radius:4px;margin-top:5px}
    .row.err{color:#ff9d9d;font-size:10px;margin-top:4px}
    .noimg{height:110px;display:flex;align-items:center;justify-content:center;color:#6b7382;font-size:12px;background:#000;border-radius:5px}
    .ok{color:#5fd07a;font-weight:700;font-size:13px} .fail{color:#ff7676;font-weight:700;font-size:13px}
    """
    n = ov.get("episodes", len(eps)); n_ok = ov.get("success", 0)
    htmltext = f"""<!doctype html><html><head><meta charset="utf-8">
<title>DeliveryBench — fully-autonomous rollout (thinking={html.escape(str(thinking))})</title>
<style>{css}</style></head><body><div class="wrap">
<h1>DeliveryBench — fully-autonomous rollout</h1>
<p class="sub">LoRA checkpoint-{html.escape(str(data.get('checkpoint','?')).split('-')[-1])} · thinking=<b>{html.escape(str(thinking))}</b> · the MODEL drives <b>every</b> action (no oracle scaffold) · held-out cities · overall success <b>{n_ok}/{n}</b></p>
<p class="legend">This isolates one question: with only a MOVE-only LoRA, can the model still follow the system prompt's workflow (VIEW_ORDERS → ACCEPT_ORDER → NAVIGATE → MOVE → PICKUP → NAVIGATE → DROP_OFF)?
Each card = one step: the observation, the action the model emitted (left-border colour = action type), and whether the env accepted it (<b style="color:#ff7676">red ⛔</b> = rejected/illegal). <span class="hint">hint shown</span> marks a step where the observation already contained the exact PICKUP/DROP_OFF command to copy.</p>
<h3 style="font-size:15px;margin:0 0 6px">Where episodes break</h3>
<table class="brk"><tr><td><b>failure stage</b></td><td class="num"><b>#</b></td></tr>{stage_rows}</table>
{panels}
</div></body></html>"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(htmltext)
    print(f"wrote {out} ({len(htmltext.encode())//1024} KB, {len(eps)} episodes)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--trace-dir", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    build(Path(args.summary), Path(args.trace_dir), Path(args.output))


if __name__ == "__main__":
    main()
