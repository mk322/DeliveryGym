"""Case studies: what did RL actually teach the courier?

Three subcommands, meant to be run in order, on whatever machine holds the
models (a workstation with one GPU is enough):

  rollout   play N unseen tasks x K rollouts against one served model and
            record EVERYTHING -- observation text, frame captions and paths,
            the model's full reply, what the world said back, the running
            ledger. One JSONL per episode. The eval harness records outcomes;
            this records behaviour, which is what a case study is made of.

  analyze   turn transcripts into behaviour rates per model: does it wait at
            a shown red lamp, does it walk into the same barrier twice, does
            it unstick after a refusal, what does a delivery cost it in turns
            and navigate() calls. Pure string-matching against messages this
            repository owns, so the detectors cannot drift.

  html      one self-contained page: headline table, behaviour-rate table
            with base-vs-checkpoint deltas, auto-picked "moments" (waited out
            a red, went round a barrier, recovered from a refusal), and every
            transcript expandable turn by turn with thumbnails at the moments
            that matter. The page is the case study; the JSONL is its data.

Seeds default to 5000+: training draws 0-999 and 1200-4199, validation owns
1000-1063, so 5000+ has never been seen by anything and a difference between
base and checkpoint on it is learning, not memory.
"""

from __future__ import annotations

import argparse
import base64
import html as html_lib
import io
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# ── the strings the detectors key on (owned by this repo, pinned by tests) ──
S_CROSSED_RED = "You crossed against the pedestrian light"
S_BLOCKED = "is blocked and you cannot get past"
S_PAID = "You are paid"
S_COLLECT = "You collect the order"
S_LIGHT_CAPTION = "[light:"
S_PHONE_DEAD = "phone is dead"


# ═══════════════════════════════════════════════════════════ rollout ═══════

def cmd_rollout(args) -> int:
    from embodiedbench.agent.courier.loop import parse_reply
    from embodiedbench.agent.courier.model_io import ModelClient
    from embodiedbench.agent.courier.session import CourierSession
    from embodiedbench.compiler.road_network import build_road_network
    from embodiedbench.eval.run import _albums, data_url, load_task_config, rasterise
    from embodiedbench.runtime.city.courier_env import CourierEnv

    out_root = Path(args.out) / args.tag
    out_root.mkdir(parents=True, exist_ok=True)
    scratch = out_root / "maps"
    scratch.mkdir(exist_ok=True)

    task = load_task_config()          # the real val block: albums, hazards
    task["max_turns"] = args.max_turns
    for kv in args.flag or []:
        key, _, value = kv.partition("=")
        task[key] = value.lower() in ("1", "true", "yes")

    network = build_road_network(Path(task["map_dir"]) if task.get("map_dir")
                                 else REPO / "vendor/vagen/vagen/envs/deliverybench/maps/citycore-paris",
                                 map_name="citycore-paris")
    seeds = [int(s) for s in range(args.seed_base, args.seed_base + args.tasks)]

    jobs = [(seed, r) for seed in seeds for r in range(args.rollouts)]
    print(f"[{args.tag}] {len(jobs)} episodes "
          f"({args.tasks} tasks x {args.rollouts} rollouts) -> {out_root}")

    def one(seed: int, ridx: int) -> dict:
        client = ModelClient(args.base_url.rstrip("/") + "/chat/completions",
                             args.model, max_tokens=args.max_tokens,
                             max_requeries=3,
                             max_tokens_ceiling=max(args.max_tokens * 4, 8192),
                             api_key=args.api_key)
        kwargs = dict(seed=seed,
                      difficulty=task.get("difficulty", "endless"),
                      queue_depth=task.get("queue_depth", 1),
                      stride=task.get("stride", "block"),
                      embodiment=task.get("embodiment", "human_on_foot"),
                      **_albums(task))
        kwargs.update({k: bool(task[k]) for k in (
            "enable_earning_jitter", "enable_food_temperature",
            "enable_special_notes", "enable_walking_energy",
            "enable_phone_battery") if k in task})
        env = CourierEnv(network, **kwargs)
        env.reset()
        session = CourierSession(env, city="Paris")
        system = session.system_prompt()

        path = out_root / f"seed{seed}_r{ridx}.jsonl"
        rows: list[dict] = [{
            "kind": "meta", "tag": args.tag, "model": args.model, "seed": seed,
            "rollout": ridx, "temperature": args.temperature,
            "max_turns": args.max_turns,
        }]
        history: list[dict] = []
        for turn in range(args.max_turns):
            if session.finished:
                break
            observation = session.observe()
            content = [{"type": "text", "text": observation.text}]
            frames = []
            for frame in observation.frames:
                if frame.kind == "photograph" and frame.path:
                    content.append({"type": "image_url",
                                    "image_url": {"url": data_url(Path(frame.path))}})
                    frames.append({"caption": frame.label, "path": frame.path})
                elif frame.kind == "map" and frame.svg:
                    png = rasterise(frame.svg, scratch / f"{seed}_{ridx}_{turn}.png")
                    if png:
                        content.append({"type": "image_url",
                                        "image_url": {"url": data_url(png)}})
                        frames.append({"caption": frame.label, "path": str(png)})
            history.append({"role": "user", "content": content})
            keep = args.history_turns * 2
            recent = history[-keep:]
            stripped = []
            for i, m in enumerate(recent):
                if (m["role"] == "user" and isinstance(m["content"], list)
                        and i < len(recent) - 1):
                    text = next((c["text"] for c in m["content"]
                                 if c["type"] == "text"), "")
                    stripped.append({"role": "user", "content": text})
                else:
                    stripped.append(m)
            messages = [{"role": "system", "content": system}, *stripped]

            started = time.time()
            try:
                reply, parsed, rejected = client.act(
                    messages, lambda t: parse_reply(t, set(session.allowed)),
                    temperature=args.temperature)
                if parsed is None:
                    reply = reply or "(no parseable action)"
            except Exception as error:  # noqa: BLE001 - transport
                rows.append({"kind": "turn", "turn": turn + 1,
                             "status": "infra_error",
                             "error": f"{type(error).__name__}: {error}"})
                break

            history.append({"role": "assistant", "content": reply})
            log = session.step(reply)
            rows.append({
                "kind": "turn", "turn": turn + 1,
                "obs": observation.text,
                "frames": frames,
                "reply": reply,
                "action": log.action,
                "status": log.status,
                "error": log.error,
                "result": session.feedback,
                "sim_seconds": round(env.sim_seconds, 1),
                "earnings": round(env.earnings, 2),
                "street": env.street_of(env.node_id) if env.node_id else "",
                "latency_s": round(time.time() - started, 1),
                "requeries": len(rejected),
            })

        summary = env.summary()
        summary["termination"] = (session.run.termination_reason
                                  or ("out_of_turns" if not session.finished else ""))
        rows.append({"kind": "summary", **summary})
        with path.open("w") as fh:
            for row in rows:
                fh.write(json.dumps(row, default=str) + "\n")
        return {"seed": seed, "rollout": ridx,
                "earnings": summary.get("earnings"),
                "delivered": summary.get("delivered")}

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for done in pool.map(lambda sr: one(*sr), jobs):
            print(f"  seed {done['seed']} r{done['rollout']}: "
                  f"${done['earnings']} / {done['delivered']} delivered")
    print(f"[{args.tag}] complete")
    return 0


# ═══════════════════════════════════════════════════════════ analyze ═══════

def load_episode(path: Path) -> dict:
    meta, turns, summary = {}, [], {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row["kind"] == "meta":
            meta = row
        elif row["kind"] == "turn":
            turns.append(row)
        else:
            summary = row
    return {"meta": meta, "turns": turns, "summary": summary, "path": path}


def behaviour(ep: dict) -> dict:
    """Behaviour rates for one episode, from the strings the world speaks."""
    turns = ep["turns"]
    s = ep["summary"]
    lamp_shown = crossed_red = waited_at_lamp = 0
    blocked_events = []          # (turn, street)
    reblocked = 0
    refused_runs = 0             # runs of the same refused call, length >= 2
    navigates = sum(1 for t in turns if (t.get("action") or "").startswith("navigate"))
    prev_refused = None
    run_len = 0
    for t in turns:
        obs = t.get("obs") or ""
        result = t.get("result") or ""
        action = t.get("action") or ""
        shown = S_LIGHT_CAPTION in obs
        if shown:
            lamp_shown += 1
            if action.startswith("wait("):
                waited_at_lamp += 1
        if S_CROSSED_RED in result:
            crossed_red += 1
        if S_BLOCKED in result:
            street = (re.match(r"(.+?) is blocked", result.strip()) or [None, ""])[1]
            if any(b[1] == street for b in blocked_events):
                reblocked += 1
            blocked_events.append((t["turn"], street))
        if t.get("status") == "rejected":
            if action == prev_refused:
                run_len += 1
                if run_len == 2:
                    refused_runs += 1
            else:
                prev_refused, run_len = action, 1
        else:
            prev_refused, run_len = None, 0
    delivered = int(s.get("delivered") or 0)
    return {
        "earnings": float(s.get("earnings") or 0.0),
        "delivered": delivered,
        "turns": len(turns),
        "lamp_shown_turns": lamp_shown,
        "waited_at_lamp": waited_at_lamp,
        "wait_rate_at_lamp": waited_at_lamp / lamp_shown if lamp_shown else None,
        "red_crossings": int(s.get("red_crossings") or 0),
        "blocked_attempts": int(s.get("blocked_attempts") or 0),
        "walked_into_same_barrier_again": reblocked,
        "stuck_repeat_runs": refused_runs,
        "rejected_actions": int(s.get("rejected_actions") or 0),
        "navigates": navigates,
        "navigates_per_delivery": navigates / delivered if delivered else None,
        "walk_ratio": s.get("walk_ratio"),
        "turns_per_delivery": s.get("turns_per_delivery"),
        "termination": s.get("termination"),
    }


def moments(ep: dict) -> list[dict]:
    """Turns worth looking at, with a one-line reason each."""
    out = []
    turns = ep["turns"]
    for i, t in enumerate(turns):
        obs, result = t.get("obs") or "", t.get("result") or ""
        action = t.get("action") or ""
        if S_LIGHT_CAPTION in obs and action.startswith("wait("):
            nxt = turns[i + 1] if i + 1 < len(turns) else None
            crossed_clean = bool(nxt and S_CROSSED_RED not in (nxt.get("result") or "")
                                 and (nxt.get("action") or "").startswith("walk_to"))
            out.append({"turn": t["turn"], "why": (
                "waited at a shown pedestrian lamp"
                + (", then crossed clean" if crossed_clean else ""))})
        if S_CROSSED_RED in result:
            out.append({"turn": t["turn"], "why": "crossed against a red light (charged 75 s)"})
        if S_BLOCKED in result:
            nxt = turns[i + 1] if i + 1 < len(turns) else None
            went_round = bool(nxt and (nxt.get("status") == "accepted")
                              and (nxt.get("action") or "").startswith("walk_to"))
            out.append({"turn": t["turn"], "why": (
                "walked into a barrier" + (", went round next turn" if went_round else ""))})
        if S_PAID in result:
            out.append({"turn": t["turn"], "why": f"delivery completed ({result.strip().splitlines()[0][:90]})"})
        if S_PHONE_DEAD in obs.lower():
            out.append({"turn": t["turn"], "why": "playing on with a dead phone (no map)"})
    return out


def aggregate(episodes: list[dict]) -> dict:
    rows = [behaviour(e) for e in episodes]
    def mean(key, only_not_none=True):
        vals = [r[key] for r in rows if r[key] is not None] if only_not_none \
            else [r[key] for r in rows]
        return round(sum(vals) / len(vals), 3) if vals else None
    lamp = sum(r["lamp_shown_turns"] for r in rows)
    waited = sum(r["waited_at_lamp"] for r in rows)
    return {
        "episodes": len(rows),
        "mean_earnings": mean("earnings"),
        "mean_delivered": mean("delivered"),
        "wait_rate_at_shown_lamp": round(waited / lamp, 3) if lamp else None,
        "red_crossings_per_ep": mean("red_crossings"),
        "blocked_attempts_per_ep": mean("blocked_attempts"),
        "repeat_barrier_walks_per_ep": mean("walked_into_same_barrier_again"),
        "stuck_repeat_runs_per_ep": mean("stuck_repeat_runs"),
        "rejected_actions_per_ep": mean("rejected_actions"),
        "navigates_per_delivery": mean("navigates_per_delivery"),
        "mean_walk_ratio": mean("walk_ratio"),
        "mean_turns_per_delivery": mean("turns_per_delivery"),
    }


def cmd_analyze(args) -> int:
    for run_dir in args.runs:
        eps = [load_episode(p) for p in sorted(Path(run_dir).glob("seed*_r*.jsonl"))]
        agg = aggregate(eps)
        print(f"\n== {run_dir} ({agg['episodes']} episodes)")
        for key, value in agg.items():
            print(f"  {key:32s} {value}")
    return 0


# ══════════════════════════════════════════════════════════════ html ═══════

# Which aggregate keys improve upward / downward, for the delta colouring.
UP_GOOD = {"mean_earnings", "mean_delivered", "wait_rate_at_shown_lamp"}
DOWN_GOOD = {"red_crossings_per_ep", "blocked_attempts_per_ep",
             "repeat_barrier_walks_per_ep", "stuck_repeat_runs_per_ep",
             "rejected_actions_per_ep", "navigates_per_delivery",
             "mean_walk_ratio", "mean_turns_per_delivery"}


def thumb(path: str, long_edge: int = 288) -> str:
    try:
        from PIL import Image
        im = Image.open(path)
        im.thumbnail((long_edge, long_edge))
        buf = io.BytesIO()
        im.convert("RGB").save(buf, format="JPEG", quality=70)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""


def esc(text) -> str:
    return html_lib.escape(str(text if text is not None else ""))


def render_episode(ep: dict, with_images: bool) -> str:
    meta, s = ep["meta"], ep["summary"]
    marks = {m["turn"]: m["why"] for m in moments(ep)}
    head = (f"seed {meta['seed']} r{meta['rollout']} — "
            f"${s.get('earnings', 0)} / {s.get('delivered', 0)} delivered, "
            f"{len(ep['turns'])} turns, {esc(s.get('termination'))}")
    parts = [f"<details class='ep'><summary>{esc(head)}</summary>"]
    for t in ep["turns"]:
        why = marks.get(t["turn"])
        cls = " moment" if why else ""
        parts.append(f"<div class='turn{cls}'>")
        parts.append(f"<div class='tno'>turn {t['turn']}"
                     + (f" — <b>{esc(why)}</b>" if why else "") + "</div>")
        if why and with_images:
            for fr in (t.get("frames") or [])[:4]:
                data = thumb(fr["path"])
                if data:
                    parts.append(
                        f"<figure><img src='{data}'/>"
                        f"<figcaption>{esc(fr['caption'])}</figcaption></figure>")
        parts.append(f"<details><summary>observation</summary>"
                     f"<pre>{esc(t.get('obs'))}</pre></details>")
        parts.append(f"<pre class='reply'>{esc(t.get('reply'))}</pre>")
        status = t.get("status")
        result = t.get("result") or t.get("error") or ""
        parts.append(f"<div class='res {esc(status)}'>[{esc(status)}] {esc(result)}</div>")
        parts.append("</div>")
    parts.append("</details>")
    return "".join(parts)


CSS = """
body{font-family:system-ui,sans-serif;margin:1.2rem;max-width:1200px}
table{border-collapse:collapse;margin:.8rem 0}
td,th{border:1px solid #ccc;padding:.25rem .6rem;text-align:right}
th:first-child,td:first-child{text-align:left}
.good{background:#e6f6e6}.bad{background:#fbe3e3}
.ep{margin:.4rem 0;border:1px solid #ddd;border-radius:6px;padding:.3rem .6rem}
.turn{border-top:1px dashed #ddd;padding:.35rem 0}
.turn.moment{background:#fffbe6}
.tno{color:#555;font-size:.85rem}
pre{white-space:pre-wrap;background:#f7f7f7;padding:.4rem;border-radius:4px;font-size:.8rem}
pre.reply{background:#eef3fb}
.res{font-size:.85rem;color:#333}.res.rejected{color:#a33}.res.format_error{color:#a33}
figure{display:inline-block;margin:.2rem}figcaption{font-size:.7rem;color:#666;max-width:290px}
h2{margin-top:2rem;border-bottom:2px solid #eee}
.note{color:#555;font-size:.9rem}
"""


def cmd_html(args) -> int:
    groups: dict[str, list[dict]] = {}
    for spec in args.runs:
        tag, _, run_dir = spec.partition("=")
        groups[tag] = [load_episode(p)
                       for p in sorted(Path(run_dir).glob("seed*_r*.jsonl"))]
    tags = list(groups)
    base_tag = args.base if args.base in groups else tags[0]
    aggs = {tag: aggregate(eps) for tag, eps in groups.items()}

    out = [f"<style>{CSS}</style>",
           "<h1>Courier case studies</h1>",
           f"<p class='note'>{' · '.join(f'{t}: {len(groups[t])} episodes' for t in tags)}"
           f" — baseline column: <b>{esc(base_tag)}</b>. Unseen seeds; same seed = same city and orders,"
           " so any behavioural difference is the policy.</p>"]

    # headline + behaviour table
    out.append("<h2>Behaviour rates</h2><table><tr><th>metric</th>")
    out += [f"<th>{esc(t)}</th>" for t in tags]
    out.append("</tr>")
    for key in aggs[base_tag]:
        if key == "episodes":
            continue
        out.append(f"<tr><td>{esc(key)}</td>")
        base_val = aggs[base_tag][key]
        for tag in tags:
            val = aggs[tag][key]
            cls = ""
            if tag != base_tag and val is not None and base_val is not None:
                better = (val > base_val and key in UP_GOOD) or \
                         (val < base_val and key in DOWN_GOOD)
                worse = (val < base_val and key in UP_GOOD) or \
                        (val > base_val and key in DOWN_GOOD)
                cls = " class='good'" if better else (" class='bad'" if worse else "")
            out.append(f"<td{cls}>{esc(val)}</td>")
        out.append("</tr>")
    out.append("</table>")

    # auto-written emergence notes
    out.append("<h2>What changed (auto-read)</h2><ul>")
    for tag in tags:
        if tag == base_tag:
            continue
        a, b = aggs[base_tag], aggs[tag]
        notes = []
        def cmp(key, label, fmt="{:.2f}"):
            if a[key] is None or b[key] is None:
                return
            notes.append(f"{label}: {fmt.format(a[key])} → {fmt.format(b[key])}")
        cmp("mean_earnings", "earnings", "${:.2f}")
        cmp("mean_delivered", "deliveries")
        cmp("wait_rate_at_shown_lamp", "waits when a lamp is shown", "{:.0%}")
        cmp("red_crossings_per_ep", "red crossings /ep")
        cmp("blocked_attempts_per_ep", "barrier walk-ins /ep")
        cmp("repeat_barrier_walks_per_ep", "same-barrier repeats /ep")
        cmp("stuck_repeat_runs_per_ep", "stuck loops /ep")
        cmp("navigates_per_delivery", "navigate() per delivery")
        out.append(f"<li><b>{esc(tag)}</b> vs {esc(base_tag)} — " + "; ".join(notes) + "</li>")
    out.append("</ul><p class='note'>Yellow turns inside the transcripts are the "
               "auto-picked moments behind these numbers.</p>")

    # per-task grid
    out.append("<h2>Per-task outcomes (mean over rollouts)</h2><table><tr><th>seed</th>")
    out += [f"<th>{esc(t)} $ / del</th>" for t in tags]
    out.append("</tr>")
    seeds = sorted({e["meta"]["seed"] for eps in groups.values() for e in eps})
    for seed in seeds:
        out.append(f"<tr><td>{seed}</td>")
        for tag in tags:
            eps = [e for e in groups[tag] if e["meta"]["seed"] == seed]
            if eps:
                money = sum(float(e["summary"].get("earnings") or 0) for e in eps) / len(eps)
                dels = sum(int(e["summary"].get("delivered") or 0) for e in eps) / len(eps)
                out.append(f"<td>{money:.2f} / {dels:.1f}</td>")
            else:
                out.append("<td>—</td>")
        out.append("</tr>")
    out.append("</table>")

    # transcripts
    for tag in tags:
        out.append(f"<h2>Transcripts — {esc(tag)}</h2>")
        for ep in groups[tag]:
            out.append(render_episode(ep, with_images=not args.no_images))

    Path(args.out).write_text("\n".join(out))
    print(f"wrote {args.out} ({Path(args.out).stat().st_size/1e6:.1f} MB)")
    return 0


# ═══════════════════════════════════════════════════════════════ cli ═══════

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("rollout")
    r.add_argument("--base-url", default="http://127.0.0.1:8200/v1")
    r.add_argument("--model", required=True)
    r.add_argument("--api-key", default="EMPTY")
    r.add_argument("--tag", required=True, help="label for this model (base / step200 ...)")
    r.add_argument("--out", default="case_studies")
    r.add_argument("--tasks", type=int, default=20)
    r.add_argument("--rollouts", type=int, default=4)
    r.add_argument("--seed-base", type=int, default=5000,
                   help="first seed; 5000+ is unseen by training and validation")
    r.add_argument("--temperature", type=float, default=0.7)
    r.add_argument("--max-turns", type=int, default=60)
    r.add_argument("--max-tokens", type=int, default=400)
    r.add_argument("--history-turns", type=int, default=8)
    r.add_argument("--workers", type=int, default=8)
    r.add_argument("--flag", action="append",
                   help="constraint override, e.g. enable_phone_battery=true")
    r.set_defaults(fn=cmd_rollout)

    a = sub.add_parser("analyze")
    a.add_argument("runs", nargs="+")
    a.set_defaults(fn=cmd_analyze)

    h = sub.add_parser("html")
    h.add_argument("--runs", nargs="+", required=True,
                   help="tag=dir pairs, e.g. base=case_studies/base step200=case_studies/step200")
    h.add_argument("--base", default="base", help="which tag is the comparison baseline")
    h.add_argument("--out", default="case_study.html")
    h.add_argument("--no-images", action="store_true")
    h.set_defaults(fn=cmd_html)

    args = p.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
