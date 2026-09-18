"""Fully-autonomous DeliveryBench rollout for the MOVE-only LoRA model.

Unlike qwen35_lora_rollout_eval.py (which oracle-drives VIEW_ORDERS/ACCEPT/
NAVIGATE/PICKUP/DROP_OFF and lets the model drive only MOVE), here the MODEL
drives EVERY action. The env config and observation format are identical to the
SFT/scaffold setup — crucially, the system prompt already describes the whole
workflow and the "workflow gates" (VIEW_ORDERS -> ACCEPT_ORDER -> NAVIGATE with
the exact pickup address -> MOVE -> PICKUP -> NAVIGATE dropoff -> MOVE ->
DROP_OFF). So this tests one thing: does prompt-based instruction-following for
the non-MOVE actions survive a MOVE-only LoRA, or does the model default to MOVE?

Each step is a single turn (system + current obs -> action), matching the SFT
distribution. `env.step()` parses the raw model text itself. We log every action
+ whether the env accepted it, and classify where each episode breaks.

Run:
  CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. python -m vagen.envs.deliverybench.tools.qwen35_lora_rollout_full \
    --checkpoint <run>/checkpoint-280 --thinking close \
    --trace-dir <run>/rollout_eval/full_close --output <run>/rollout_eval/full_close/summary.json
"""
from __future__ import annotations
import argparse, asyncio, json, re, tempfile
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

from ..deliverybench_env import DeliveryBench
from .build_balanced_visual_sft_data import make_env_config
from .qwen35_visual_sft_dataset import to_qwen_structured_messages, load_pil_images, IMAGE_PLACEHOLDER
from .qwen35_lora_rollout_eval import _load_model, DEFAULT_MODEL


def _load_base_model(model_path: Path, device):
    """Load the untuned base model (NO LoRA adapter). Tier-0 test: does the base
    Qwen3.5-VL follow the full-workflow prompt on its own — i.e. is the workflow
    capability still in the base weights, just masked by the MOVE-only adapter?"""
    torch.backends.cudnn.enabled = False
    torch.backends.cuda.enable_cudnn_sdp(False)
    processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        str(model_path), dtype=torch.bfloat16, trust_remote_code=True,
        local_files_only=True, low_cpu_mem_usage=True, attn_implementation="sdpa")
    model.to(device); model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = True
    return model, processor

# The same 9 held-out (city, seed) pairs the scaffolded rollout used (8/9), so
# results are directly comparable instance-by-instance.
DEFAULT_PAIRS = [
    ("small-city-15", 9000), ("small-city-15", 9002), ("small-city-15", 9003),
    ("medium-city-22", 10001), ("medium-city-22", 10002), ("medium-city-22", 10004),
    ("large-city-30", 11000), ("large-city-30", 11002), ("large-city-30", 11003),
]


def _atype(action: str | None) -> str | None:
    if not action:
        return None
    m = re.match(r"\s*([A-Za-z_]+)", action)
    return m.group(1).upper() if m else None


def _is_move_phase(obs_str: str) -> bool:
    """MOVE leg = an active [navigation] route with no arrival hint yet -> use the MOVE adapter.
    Everything else (no order / order pool / just accepted / at a pickup/dropoff hint) is a
    workflow decision -> use the base model. Verified against the env's per-phase obs markers."""
    return ("[navigation]" in obs_str) and ("[pickup_hint]" not in obs_str) and ("[dropoff_hint]" not in obs_str)


def _save_first_image(obs: Dict[str, Any], trace_dir: Path, name: str) -> str | None:
    pil = [im for v in (obs.get("multi_modal_input") or {}).values()
           for im in (v if isinstance(v, list) else [v])]
    if not pil:
        return None
    p = trace_dir / f"{name}.png"
    pil[0].save(p)
    return p.name


def _generate_action(model, processor, system_text, obs, device, tmpdir: Path, step: int,
                     thinking: str, max_new_tokens: int) -> str:
    """Single-turn: system + current obs -> raw model text (env parses it).

    Mirrors qwen35_lora_rollout_eval._model_move exactly (save PILs -> real paths ->
    load_pil_images), so the prompt is byte-identical to the SFT/eval path; only the
    thinking mode and token budget differ, and we return the raw text unparsed.
    """
    user_text = obs.get("obs_str", "")
    pil = [im for v in (obs.get("multi_modal_input") or {}).values() for im in (v if isinstance(v, list) else [v])]
    entries = []
    for i, im in enumerate(pil):
        p = tmpdir / f"s{step}_{i}.png"
        im.save(p)
        entries.append({"image": str(p)})
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": ""},
    ]
    qmsgs = to_qwen_structured_messages(messages, entries)
    pil_imgs = load_pil_images(entries)
    prompt = processor.apply_chat_template(qmsgs[:-1], tokenize=False, add_generation_prompt=True)
    if thinking == "close":
        prompt = f"{prompt}\n</think>\n\n"
    inputs = processor(text=[prompt], images=pil_imgs or None, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
    plen = int(inputs["input_ids"].shape[-1])
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, min_new_tokens=1,
                             do_sample=False, use_cache=True)
    new = out[:, plen:] if int(out.shape[-1]) > plen else out
    return processor.batch_decode(new, skip_special_tokens=True)[0]


def _generate_action_mt(model, processor, system_text, history, obs, device, tmpdir, step,
                        thinking, max_new_tokens) -> str:
    """Multi-turn: system + FULL text history of prior (obs, action) + current obs -> action.

    Gives the model memory of what it already did (fixes the 'keep re-viewing orders' loop
    the single-turn harness caused). To avoid a vision-token blow-up we keep only the CURRENT
    turn's image; past observations are carried as text with their <image> stripped, and past
    assistant turns are the compact action the model took."""
    user_text = obs.get("obs_str", "")
    pil = [im for v in (obs.get("multi_modal_input") or {}).values() for im in (v if isinstance(v, list) else [v])]
    entries = []
    for i, im in enumerate(pil):
        p = tmpdir / f"mt{step}_{i}.png"; im.save(p); entries.append({"image": str(p)})
    msgs = [{"role": "system", "content": system_text}]
    for h in history:
        msgs.append({"role": "user", "content": h["user"]})       # <image> already stripped
        msgs.append({"role": "assistant", "content": h["assistant"]})
    msgs.append({"role": "user", "content": user_text})           # current turn keeps its <image>
    msgs.append({"role": "assistant", "content": ""})
    qmsgs = to_qwen_structured_messages(msgs, entries)             # only current turn has a placeholder
    pil_imgs = load_pil_images(entries)
    prompt = processor.apply_chat_template(qmsgs[:-1], tokenize=False, add_generation_prompt=True)
    if thinking == "close":
        prompt = f"{prompt}\n</think>\n\n"
    inputs = processor(text=[prompt], images=pil_imgs or None, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
    plen = int(inputs["input_ids"].shape[-1])
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, min_new_tokens=1,
                             do_sample=False, use_cache=True)
    new = out[:, plen:] if int(out.shape[-1]) > plen else out
    return processor.batch_decode(new, skip_special_tokens=True)[0]


async def run_episode(cfg, model, processor, device, seed, budget, trace_dir, tmpdir, ep_tag,
                      thinking, max_new_tokens, multi_turn=False, hierarchical=False) -> Dict[str, Any]:
    env = DeliveryBench(cfg)
    rec: Dict[str, Any] = {"seed": seed, "map": cfg.map_name, "tag": ep_tag, "thinking": thinking, "steps": []}
    try:
        system_text = (await env.system_prompt())["obs_str"]
        obs, _ = await env.reset(seed=seed)
        success = False
        history: List[Dict[str, str]] = []
        for step in range(budget):
            obs_str = obs.get("obs_str", "")
            saw_pu = "[pickup_hint]" in obs_str
            saw_do = "[dropoff_hint]" in obs_str
            img = _save_first_image(obs, trace_dir, f"{ep_tag}_step{step:02d}")
            move_phase = _is_move_phase(obs_str)
            if hierarchical:
                if move_phase:  # MOVE leg -> adapter, single-turn close-thinking (its SFT/88.9% path)
                    route = "adapter"
                    raw = _generate_action(model, processor, system_text, obs, device, tmpdir, step, "close", 96)
                else:           # workflow decision -> BASE model (adapter off), multi-turn open-thinking
                    route = "base"
                    ctx = model.disable_adapter() if hasattr(model, "disable_adapter") else nullcontext()
                    with ctx:
                        raw = _generate_action_mt(model, processor, system_text, history, obs, device, tmpdir,
                                                  step, "open", 896)
            elif multi_turn:
                route = "mt"
                raw = _generate_action_mt(model, processor, system_text, history, obs, device, tmpdir, step,
                                          thinking, max_new_tokens)
            else:
                route = "single"
                raw = _generate_action(model, processor, system_text, obs, device, tmpdir, step,
                                       thinking, max_new_tokens)
            obs, _r, done, info = await env.step(raw)
            parsed = info.get("parsed") or {}
            action = parsed.get("action")
            err = info.get("action_error")
            if hierarchical or multi_turn:  # accumulate memory for the base's workflow decisions
                hist_user = "[navigation in progress]" if (hierarchical and move_phase) \
                    else obs_str.replace(IMAGE_PLACEHOLDER, "[map image]")
                hist_act = json.dumps({"action": action}) if action else (raw.strip()[:200] or "{}")
                history.append({"user": hist_user, "assistant": hist_act})
            rec["steps"].append({
                "step": step, "image": img, "route": route,
                "action": action, "atype": _atype(action),
                "format_correct": bool(parsed.get("format_correct")),
                "ok": not err, "err": (str(err)[:120] if err else None),
                "is_tool": bool(info.get("is_tool")),
                "obs_pickup_hint": saw_pu, "obs_dropoff_hint": saw_do,
                "reasoning": (str(parsed.get("reasoning"))[:200] if parsed.get("reasoning") else None),
                "raw": raw.strip()[:400],
            })
            success = bool(info.get("success"))
            _last = rec["steps"][-1]
            print(f"  {ep_tag} step{step:02d}: [{route:7s}] {(_last['atype'] or '(unparsed)'):12s} "
                  f"ok={_last['ok']} pu={_last['obs_pickup_hint']} do={_last['obs_dropoff_hint']} "
                  f"| act={str(_last['action'])[:48]}", flush=True)
            if done:
                break
        dm = env._env.dms[0] if (env._env and env._env.dms) else None
        rec["deliveries"] = len(getattr(dm, "completed_orders", []) or []) if dm else 0
        rec["success"] = success
    except Exception as exc:  # noqa: BLE001
        rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["success"] = False
        rec.setdefault("deliveries", 0)
    finally:
        await env.close()
    _classify(rec)
    return rec


def _classify(rec: Dict[str, Any]) -> None:
    steps = rec.get("steps", [])
    def ok_type(t):  # a successfully-executed action of this type ever happened
        return any(s["atype"] == t and s["ok"] for s in steps)
    accepted = ok_type("ACCEPT_ORDER")
    navigated = ok_type("NAVIGATE")
    picked = ok_type("PICKUP")
    dropped = ok_type("DROP_OFF")
    saw_pu = any(s["obs_pickup_hint"] for s in steps)
    saw_do = any(s["obs_dropoff_hint"] for s in steps)
    if rec.get("success"):
        stage = "delivered"
    elif not ok_type("VIEW_ORDERS") and not accepted:
        stage = "never_viewed_or_accepted"
    elif not accepted:
        stage = "viewed_but_never_accepted"
    elif not navigated:
        stage = "accepted_but_never_navigated"
    elif not saw_pu:
        stage = "never_reached_pickup"
    elif not picked:
        stage = "reached_pickup_but_no_PICKUP"
    elif not saw_do:
        stage = "never_reached_dropoff"
    elif not dropped:
        stage = "reached_dropoff_but_no_DROPOFF"
    else:
        stage = "other"
    from collections import Counter
    rec["failure_stage"] = stage
    rec["action_counts"] = dict(Counter(s["atype"] for s in steps))
    rec["ok_action_counts"] = dict(Counter(s["atype"] for s in steps if s["ok"]))
    rec["n_steps"] = len(steps)
    rec["format_fail"] = sum(1 for s in steps if not s["format_correct"])
    rec["milestones"] = {"accepted": accepted, "navigated": navigated,
                         "reached_pickup": saw_pu, "picked": picked,
                         "reached_dropoff": saw_do, "dropped": dropped}


async def amain(args):
    device = torch.device(args.device)
    if args.hierarchical:
        assert args.checkpoint, "--hierarchical needs --checkpoint (the MOVE adapter, toggled off for workflow steps)"
        model, processor = _load_model(Path(args.checkpoint), Path(args.model_path), device)
        ckpt_label = f"HIERARCHICAL (base workflow + {Path(args.checkpoint).name} for MOVE)"
    elif args.no_adapter:
        model, processor = _load_base_model(Path(args.model_path), device)
        ckpt_label = "BASE (no adapter)"
    else:
        model, processor = _load_model(Path(args.checkpoint), Path(args.model_path), device)
        ckpt_label = str(args.checkpoint)
    trace_dir = Path(args.trace_dir); trace_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output); out_path.parent.mkdir(parents=True, exist_ok=True)
    max_new = args.max_new_tokens or (256 if args.thinking == "open" else 96)
    pairs = DEFAULT_PAIRS if not args.pairs else [(c, int(s)) for c, s in (p.rsplit(":", 1) for p in args.pairs)]
    episodes: List[Dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        for i, (city, seed) in enumerate(pairs):
            cfg = make_env_config(map_name=city, max_steps=args.max_steps,
                                  feasible_order_step_budget=args.feasible_budget, enable_fpv=args.enable_fpv)
            ep_tag = f"ep{i}_{city}_{seed}"
            rec = await run_episode(cfg, model, processor, device, seed, args.budget, trace_dir, tmpdir, ep_tag,
                                    args.thinking, max_new, multi_turn=args.multi_turn,
                                    hierarchical=args.hierarchical)
            ac = rec.get("action_counts", {})
            print(f"[{ep_tag}] success={rec.get('success')} stage={rec.get('failure_stage')} "
                  f"steps={rec.get('n_steps')} actions={ac} fmtfail={rec.get('format_fail')} "
                  f"err={rec.get('error','')}", flush=True)
            episodes.append(rec)

    # aggregate
    from collections import Counter
    n = len(episodes)
    n_ok = sum(1 for e in episodes if e.get("success"))
    stages = Counter(e.get("failure_stage") for e in episodes)
    route_counts = Counter(s.get("route") for e in episodes for s in e.get("steps", []))
    per_city: Dict[str, Any] = {}
    for c in sorted({e["map"] for e in episodes}):
        eps = [e for e in episodes if e["map"] == c]
        per_city[c] = {"episodes": len(eps), "success": sum(1 for e in eps if e.get("success"))}
    summary = {
        "checkpoint": ckpt_label, "thinking": args.thinking, "multi_turn": args.multi_turn,
        "hierarchical": args.hierarchical, "route_counts": dict(route_counts),
        "overall": {"episodes": n, "success": n_ok, "success_rate": (n_ok / n) if n else None},
        "failure_stages": dict(stages), "per_city": per_city, "episodes": episodes,
    }
    out_path.write_text(json.dumps(summary, indent=2))
    print("\n=== SUMMARY (full-autonomous, thinking=%s) ===" % args.thinking)
    print(f"overall success: {n_ok}/{n}")
    print("per-city:", json.dumps(per_city))
    print("where it breaks:", json.dumps(dict(stages), indent=2))
    print("wrote", out_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=None, help="LoRA checkpoint (ignored with --no-adapter)")
    ap.add_argument("--no-adapter", action="store_true", default=False,
                    help="run the BASE model with NO LoRA adapter (Tier-0 workflow-capability test)")
    ap.add_argument("--model-path", default=DEFAULT_MODEL)
    ap.add_argument("--pairs", nargs="*", default=None, help="override city:seed pairs")
    ap.add_argument("--thinking", choices=["open", "close"], default="close")
    ap.add_argument("--multi-turn", action="store_true", default=False,
                    help="carry conversation history across steps (gives the model memory of prior actions)")
    ap.add_argument("--hierarchical", action="store_true", default=False,
                    help="phase-routed policy: base model (adapter off, multi-turn) for workflow, "
                         "MOVE adapter (single-turn) for navigation legs")
    ap.add_argument("--max-new-tokens", type=int, default=0, help="override generation budget (0=auto by thinking mode)")
    ap.add_argument("--trace-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--budget", type=int, default=40, help="max model turns per episode")
    ap.add_argument("--feasible-budget", type=int, default=20)
    ap.add_argument("--enable-fpv", action="store_true", default=False)
    args = ap.parse_args()
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
