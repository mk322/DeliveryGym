"""Render representative DeliveryBench rollout traces as a self-contained HTML.

Reads <trace-dir>/trace.json + the step PNGs written by qwen35_lora_rollout_trace.py,
embeds each step image as base64, and lays out one panel per episode: a strip of
steps showing the observation, the model's chosen MOVE direction, the oracle's
correct direction, and a match/miss badge. No external assets — openable / shareable
as a single file.

Run:
  PYTHONPATH=. python -m vagen.envs.deliverybench.tools.build_rollout_trace_html \
    --trace-dir <run>/rollout_eval/trace \
    --output outputs/deliverybench_sft/rollout_trace.html
"""
from __future__ import annotations
import argparse, base64, html, json
from pathlib import Path

ARROW = {"forward": "↑ forward", "backward": "↓ backward", "left": "← left", "right": "→ right"}


def _img_b64(p: Path) -> str:
    if not p.exists():
        return ""
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode("ascii")


def _dir_label(d):
    if not d:
        return "<span class='na'>—</span>"
    return html.escape(ARROW.get(d, d))


# Honest per-episode captions (all figures verified from the enriched trace).
NOTES = {
    "ep0_small-city-15_9000":
        "Clean short delivery. 9 legal moves; agreed with the oracle on 8/9 — the one "
        "divergence (step 3: left vs forward) was still a legal move and the delivery completed.",
    "ep1_medium-city-22_10001":
        "Longer route with a 10-step drop-off leg. Agreed with the oracle on 11/12; the single "
        "divergence still reached the target. Delivered.",
    "ep2_large-city-30_11000":
        "Largest map, still solved. 14 legal moves, agreed with the oracle on 12/14. Delivered.",
    "ep3_large-city-30_11002":
        "Failure — but instructive. All 22 moves were legal and accepted (the model did NOT get "
        "stuck on illegal actions); it simply didn't reach the far pickup before the env's 25-step "
        "budget ran out (3 workflow steps + 22 moves). The other large-city episode solved its "
        "route in 14 moves, so this is a hard long-path instance under a tight budget, not invalid "
        "behaviour. The oracle route-helper produced no per-step reference for this seed, so "
        "per-step agreement is n/a here (a helper limitation, not 22 wrong moves).",
}


def _episode_panel(ep: dict, trace_dir: Path) -> str:
    tag = ep.get("tag", "")
    city = html.escape(ep.get("map", "?"))
    seed = ep.get("seed", "?")
    success = ep.get("success")
    badge = "<span class='ok'>SUCCESS</span>" if success else "<span class='fail'>FAILED</span>"
    steps = ep.get("trace", [])
    n = len(steps)
    ref = [t for t in steps if t.get("oracle_direction")]        # steps that HAVE an oracle reference
    nref = len(ref)
    nmatch = sum(1 for t in ref if t.get("match") is True)
    nblock = sum(1 for t in steps if t.get("move_ok") is False)
    pickup = ep.get("arrived_pickup")
    dropoff = ep.get("arrived_dropoff")
    mv1 = ep.get("moves_leg1"); mv2 = ep.get("moves_leg2")
    agree = f"{nmatch}/{nref} ({100*nmatch//nref}%)" if nref else "n/a (no oracle reference for this seed)"
    meta = (f"reached pickup: {'✅' if pickup else '❌'} &nbsp;·&nbsp; "
            f"reached drop-off: {'✅' if dropoff else ('—' if dropoff is None else '❌')} &nbsp;·&nbsp; "
            f"moves: leg1={mv1}, leg2={mv2 if mv2 is not None else '—'} &nbsp;·&nbsp; "
            f"legal moves: {n-nblock}/{n} &nbsp;·&nbsp; "
            f"model↔oracle agreement: {agree}")
    if ep.get("error"):
        meta += f" &nbsp;·&nbsp; <span class='fail'>error: {html.escape(str(ep['error']))}</span>"
    note = NOTES.get(tag, "")
    note_html = f"<div class='note'>{html.escape(note)}</div>" if note else ""

    cards = []
    cur_leg = None
    for t in steps:
        leg = t.get("leg")
        if leg != cur_leg:
            cur_leg = leg
            legname = "Leg 1 — walk to pickup" if leg == "to_pickup" else "Leg 2 — walk to drop-off"
            cards.append(f"<div class='legsep'>{html.escape(legname)}</div>")
        img = _img_b64(trace_dir / (t.get("image") or "")) if t.get("image") else ""
        md = t.get("model_direction"); od = t.get("oracle_direction"); m = t.get("match")
        if m is True:
            cls, mark, mtxt = "cmatch", "✓", "matched"
        elif m is False:
            cls, mark, mtxt = "cdiv", "≠", "diverged"
        else:
            cls, mark, mtxt = "cunk", "·", "no ref"
        blocked = t.get("move_ok") is False
        if blocked:
            cls = "cblock"
        imgtag = f"<img src='{img}'/>" if img else "<div class='noimg'>no image</div>"
        block_html = (f"<div class='row blk'>⛔ {html.escape(str(t.get('err') or 'blocked'))}</div>"
                      if blocked else "")
        cards.append(
            f"<div class='card {cls}'>"
            f"<div class='stepno'>step {t.get('step')}<span class='mark' title='{mtxt}'>{mark}</span></div>"
            f"{imgtag}"
            f"<div class='row'><span class='k'>model</span> {_dir_label(md)}</div>"
            f"<div class='row'><span class='k'>oracle</span> {_dir_label(od)}</div>"
            f"{block_html}"
            f"</div>")
    return (f"<section class='ep'>"
            f"<h2>{city} &nbsp;<span class='seed'>seed {seed}</span> &nbsp;{badge}</h2>"
            f"<div class='meta'>{meta}</div>"
            f"{note_html}"
            f"<div class='strip'>{''.join(cards)}</div>"
            f"</section>")


def build(trace_dir: Path, out: Path):
    src = trace_dir / "trace_enriched.json"          # prefer the replay-enriched trace (has move_ok)
    if not src.exists():
        src = trace_dir / "trace.json"
    data = json.loads(src.read_text())
    eps = data.get("episodes", [])
    step = data.get("checkpoint_step", "?")
    n_ok = sum(1 for e in eps if e.get("success"))
    panels = "\n".join(_episode_panel(e, trace_dir) for e in eps)
    css = """
    body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:0;background:#0f1115;color:#e6e6e6}
    .wrap{max-width:1500px;margin:0 auto;padding:28px}
    h1{font-size:24px;margin:0 0 4px} .sub{color:#9aa4b2;margin:0 0 22px;font-size:14px}
    .legend{font-size:13px;color:#9aa4b2;margin:0 0 22px}
    .legend b{color:#e6e6e6}
    section.ep{background:#161a21;border:1px solid #232a35;border-radius:12px;padding:18px 18px 6px;margin:0 0 22px}
    h2{font-size:18px;margin:0 0 6px} .seed{color:#8b95a4;font-weight:400;font-size:14px}
    .meta{color:#aeb6c2;font-size:13px;margin:0 0 8px}
    .note{color:#c8cfd9;font-size:13px;line-height:1.5;margin:0 0 14px;padding:9px 12px;background:#12161d;border-left:3px solid #3a4556;border-radius:0 6px 6px 0;max-width:1100px}
    .strip{display:flex;flex-wrap:wrap;gap:10px;align-items:flex-start}
    .legsep{flex-basis:100%;font-size:12px;letter-spacing:.04em;text-transform:uppercase;color:#7f8a99;margin:8px 0 2px;border-top:1px dashed #2a3340;padding-top:8px}
    .card{width:150px;background:#1c2129;border:1px solid #2a3340;border-radius:9px;padding:7px}
    .card img{width:100%;border-radius:5px;display:block;background:#000}
    .card.cmatch{border-color:#2f6b3a} .card.cdiv{border-color:#6b5a2f} .card.cunk{border-color:#38414f} .card.cblock{border-color:#7d3030}
    .stepno{font-size:12px;color:#9aa4b2;display:flex;justify-content:space-between;margin-bottom:5px}
    .mark{font-weight:700} .cmatch .mark{color:#5fd07a} .cdiv .mark{color:#e0b64a} .cunk .mark{color:#6b7382} .cblock .mark{color:#ff7676}
    .row{font-size:12px;margin-top:3px} .k{display:inline-block;width:42px;color:#7f8a99}
    .row.blk{color:#ff9d9d;font-size:11px}
    .na{color:#6b7382} .noimg{height:120px;display:flex;align-items:center;justify-content:center;color:#6b7382;font-size:12px;background:#000;border-radius:5px}
    .ok{color:#5fd07a;font-weight:700;font-size:13px} .fail{color:#ff7676;font-weight:700;font-size:13px}
    """
    htmltext = f"""<!doctype html><html><head><meta charset="utf-8">
<title>DeliveryBench MOVE rollout — representative trajectories</title>
<style>{css}</style></head><body><div class="wrap">
<h1>DeliveryBench MOVE-only rollout — representative trajectories</h1>
<p class="sub">LoRA checkpoint-{html.escape(str(step))} · held-out cities · scaffolded rollout (oracle drives the workflow; the model drives every MOVE) · {n_ok}/{len(eps)} episodes shown succeeded</p>
<p class="legend">Each card is one navigation step: the <b>observation</b> the model saw, the <b>model</b>'s chosen MOVE, and the <b>oracle</b>'s reference MOVE for that step.
Border colour — <b style="color:#5fd07a">green ✓</b> matched the oracle · <b style="color:#e0b64a">amber ≠</b> diverged (still a legal move; the episode can still succeed) · <b style="color:#8a94a3">grey ·</b> no oracle reference for that step · <b style="color:#ff7676">red ⛔</b> the env rejected the move as illegal.
The model runs with close-thinking, so its output is the action only (no free-text reasoning). Agreement is counted only over steps that have an oracle reference.</p>
{panels}
</div></body></html>"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(htmltext)
    kb = len(htmltext.encode()) // 1024
    print(f"wrote {out} ({kb} KB, {len(eps)} episodes)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace-dir", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    build(Path(args.trace_dir), Path(args.output))


if __name__ == "__main__":
    main()
