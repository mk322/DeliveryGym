"""Evaluate any OpenAI-compatible model on the courier benchmark.

This is the benchmark's front door: no trainer, no Ray, no vLLM checkout --
one process, one endpoint, the immutable validation seeds, and a JSON of
scores with confidence intervals.

The episode is driven through ``CourierGymEnv`` -- the very adapter the RL
trainer rolls out with -- so the observation a model is scored on here is
the observation it would be trained on: the same prompt builder, the same
image budget (``max_images``), the same downscale (``image_max_side``), the
same lamp-visibility gating, all read from the **validation yaml** rather
than restated. Two copies of "the task" is how they quietly become two
tasks; this file keeps none.

    python -m embodiedbench.eval run --model qwen3-vl-4b \\
        --base-url http://127.0.0.1:8000/v1 --split val --out results/
    python -m embodiedbench.eval compare results/a.json results/b.json

A number this evaluator prints is a statement about the model: the loop
decides nothing, it posts what the harness shows and hands back what the
model said.
"""

from __future__ import annotations

import asyncio
import base64
import concurrent.futures as cf
import io
import json
import logging
import os
import time
from pathlib import Path

from embodiedbench.eval import stats

REPO = Path(__file__).resolve().parents[2]
VAL_YAML = REPO / "embodiedbench/training/vagen/val_courier.yaml"

# The benchmark's split protocol, in one place. ``val`` is the number;
# ``trainprobe`` exists for train/val gap analysis. Both name exactly the seed
# sets the RL harness validates on (the yaml's ranges are inclusive and sized
# to their ``n_envs``, so the two cannot drift apart by a seed).
SPLITS = {"val": range(1000, 1064), "trainprobe": range(0, 32)}

#: Turns at which a shift's running earnings are recorded, matching the
#: trainer's ``earnings_at_N`` validation metrics.
EARNINGS_CHECKPOINTS = (20, 40, 60, 80, 100)


def load_task_config() -> dict:
    """The val block of the validation yaml -- the single source of the task."""
    import yaml
    spec = yaml.safe_load(VAL_YAML.read_text())["envs"][0]
    cfg = dict(spec["config"])
    cfg["max_turns"] = int(spec.get("max_turns", 60))
    return cfg


def _png_data_url(image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def _content(obs: dict) -> list[dict]:
    """One user message from one adapter observation.

    The adapter appends one ``<image>`` placeholder per picture after the
    text, and the chat template puts the pictures there; this sends the
    text, then the same pictures in the same order. The pictures arrive
    already sized to the yaml's ``image_max_side`` and already trimmed to
    its ``max_images``, with captions that name exactly what was sent.
    """
    from embodiedbench.training.vagen_courier_env import IMAGE_PLACEHOLDER

    text = str(obs.get("obs_str", "")).replace(IMAGE_PLACEHOLDER, "").rstrip()
    images = (obs.get("multi_modal_input") or {}).get(IMAGE_PLACEHOLDER) or []
    content: list[dict] = [{"type": "text", "text": text}]
    for image in images:
        content.append({"type": "image_url",
                        "image_url": {"url": _png_data_url(image)}})
    return content


def _window(history: list[dict], history_turns: int) -> list[dict]:
    """The recent exchanges, older turns keeping their text and losing images.

    The window ends on the current user message and holds the last
    ``history_turns`` complete exchanges before it (an odd count of messages,
    so it never opens on an orphan assistant turn). Images from earlier turns
    are dropped: they were decidable only in the moment, and a dozen turns
    of pictures exceed any server's context.
    """
    if history_turns <= 0:
        recent = history[-1:]
    else:
        recent = history[-(2 * history_turns + 1):]
    stripped = []
    for i, message in enumerate(recent):
        if (message["role"] == "user" and isinstance(message["content"], list)
                and i < len(recent) - 1):
            text = next((c["text"] for c in message["content"]
                         if c["type"] == "text"), "")
            stripped.append({"role": "user", "content": text})
        else:
            stripped.append(message)
    return stripped


def run_episode(args, task: dict, seed: int) -> dict:
    from embodiedbench.agent.courier.loop import parse_reply
    from embodiedbench.agent.courier.model_io import ModelClient
    from embodiedbench.training.vagen_courier_env import CourierGymEnv

    client = ModelClient(args.base_url.rstrip("/") + "/chat/completions",
                         args.model, max_tokens=args.max_tokens,
                         max_requeries=args.max_requeries,
                         max_tokens_ceiling=max(args.max_tokens * 4, 8192),
                         api_key=args.api_key)
    env = CourierGymEnv(env_config=dict(task))
    loop = asyncio.new_event_loop()
    transcript: list[dict] = []
    earnings_at: dict[int, float] = {}
    infra_errors = 0
    try:
        obs, _ = loop.run_until_complete(env.reset(seed=seed))
        system = loop.run_until_complete(env.system_prompt())["obs_str"]
        session = env.session
        history: list[dict] = []
        done = False
        for turn in range(task["max_turns"]):
            if done:
                break
            history.append({"role": "user", "content": _content(obs)})
            messages = [{"role": "system", "content": system},
                        *_window(history, args.history_turns)]
            started = time.time()
            try:
                reply, parsed, rejected = client.act(
                    messages,
                    lambda text: parse_reply(text, set(session.allowed),
                                             tools_by_name=session._tools_by_name))
            except Exception as error:  # noqa: BLE001 - transport, not model
                transcript.append({"turn": turn + 1, "status": "infra_error",
                                   "error": f"{type(error).__name__}: {error}"})
                infra_errors += 1
                break
            if parsed is None:
                reply = reply or "(no parseable action)"
            history.append({"role": "assistant", "content": reply})
            obs, _reward, done, _info = loop.run_until_complete(env.step(reply))
            # The trainer's validation logs earnings at turns 20/40/60 (and
            # 80/100 under the older 100-turn readout); record the same
            # checkpoints so a shorter horizon can be read off this run.
            if (turn + 1) in EARNINGS_CHECKPOINTS:
                earnings_at[turn + 1] = float(
                    env.world.summary().get("earnings") or 0.0)
            log = session.run.turns[-1] if session.run.turns else None
            transcript.append({
                "turn": turn + 1,
                "status": getattr(log, "status", ""),
                "action": getattr(log, "action", ""),
                "error": getattr(log, "error", ""),
                "images": len(history[-2]["content"]) - 1,
                "latency_s": round(time.time() - started, 1),
                "requeries": len(rejected),
            })
        summary = env.world.summary()
        if infra_errors:
            termination = "infra_error"
        else:
            termination = (session.run.termination_reason
                           or ("out_of_turns" if not session.finished else ""))
    finally:
        try:
            loop.run_until_complete(env.close())
        finally:
            loop.close()

    final_earnings = float(summary.get("earnings") or 0.0)
    # A shift that ended before a checkpoint (stuck, shift over) keeps its
    # final earnings from then on, exactly as the trainer's metric does.
    checkpoints = {
        f"earnings_at_{k}": (earnings_at[k] if k in earnings_at
                             else final_earnings if k >= len(transcript)
                             else None)
        for k in EARNINGS_CHECKPOINTS if k <= task["max_turns"]
    }
    return {
        "seed": seed,
        "earnings": final_earnings,
        **checkpoints,
        "delivered": int(summary.get("delivered") or 0),
        "late": int(summary.get("late") or 0),
        "red_crossings": int(summary.get("red_crossings") or 0),
        "turns": len(transcript),
        "termination": termination,
        "infra_errors": infra_errors,
        "transcript": transcript if args.transcripts else None,
    }


def cmd_run(args) -> int:
    # The phone's map is rasterised from SVG by cairosvg. Missing, the adapter
    # would go on telling the courier to read the route off a map that never
    # arrives -- a defect that once ran undetected for a whole training run.
    try:
        import cairosvg  # noqa: F401
    except Exception as error:  # noqa: BLE001
        print(f"cairosvg is not importable ({error}); install it (and libcairo) "
              "before evaluating -- without it the phone's map is never sent")
        return 2
    # The prompt builder logs a per-observation size line at WARNING so the
    # trainer's logger cannot swallow it; here it is one line per turn of noise.
    logging.getLogger("courier.prompts").setLevel(logging.ERROR)

    task = load_task_config()
    if getattr(args, "max_turns", None):
        task["max_turns"] = int(args.max_turns)
    if args.seed_list:
        seeds = [int(s) for s in args.seed_list.split(",")]
    else:
        seeds = list(SPLITS[args.split])
    if args.limit:
        seeds = seeds[:args.limit]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or args.model.replace("/", "_")
    episodes_path = out_dir / f"{tag}.episodes.jsonl"
    summary_path = out_dir / f"{tag}.json"

    print(f"model={args.model}  endpoint={args.base_url}")
    print(f"split={args.split}  episodes={len(seeds)}  workers={args.workers}")
    print(f"task from {VAL_YAML.relative_to(REPO)}: "
          f"{task.get('difficulty')}/{task.get('queue_depth')} "
          f"x {task['max_turns']} turns, max_images={task.get('max_images', 5)}, "
          f"image_max_side={task.get('image_max_side')}")

    # With --transcripts every turn of every episode (the action, the
    # harness's verdict, the images sent, the latency) goes to its own file,
    # so a number can be read back to the turns that produced it.
    transcripts_path = out_dir / f"{tag}.transcripts.jsonl"
    rows: list[dict] = []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool, \
            episodes_path.open("w") as sink, \
            (transcripts_path.open("w") if args.transcripts else open(os.devnull, "w")) as turns_sink:
        futures = {pool.submit(run_episode, args, task, s): s for s in seeds}
        for fut in cf.as_completed(futures):
            seed = futures[fut]
            try:
                row = fut.result()
            except Exception as error:  # noqa: BLE001
                row = {"seed": seed, "earnings": 0.0, "delivered": 0,
                       "crashed": f"{type(error).__name__}: {error}"}
            rows.append(row)
            sink.write(json.dumps(
                {k: v for k, v in row.items() if k != "transcript"}) + "\n")
            sink.flush()
            if args.transcripts and row.get("transcript") is not None:
                turns_sink.write(json.dumps(row) + "\n")
                turns_sink.flush()
            print(f"  seed {seed}: earnings={row.get('earnings', 0):.2f} "
                  f"delivered={row.get('delivered', 0)} "
                  f"{row.get('termination', row.get('crashed', ''))}")

    # An episode the infrastructure ended is not a score. It is reported,
    # kept out of every mean, and makes the run's exit status non-zero, so a
    # dead endpoint can never read as a model that earns nothing.
    clean = [r for r in rows
             if not r.get("crashed") and not r.get("infra_errors")]
    crashed = [r["seed"] for r in rows if r.get("crashed")]
    infra = [r["seed"] for r in rows if r.get("infra_errors")]
    earnings = [r["earnings"] for r in clean]
    ci = stats.bootstrap_ci(earnings) if earnings else (0.0, 0.0)
    summary = {
        "model": args.model, "tag": tag, "split": args.split,
        "episodes": len(rows),
        "scored_episodes": len(clean),
        "earnings_mean": round(stats.mean(earnings), 4) if earnings else None,
        "earnings_ci95": [round(ci[0], 4), round(ci[1], 4)],
        # Running earnings at the trainer's checkpoints, averaged over the
        # same clean shifts: ``earnings_at_60`` is the 60-turn number.
        "earnings_at_mean": {
            f"earnings_at_{k}": round(stats.mean(
                [r[f"earnings_at_{k}"] for r in clean]), 4)
            for k in EARNINGS_CHECKPOINTS
            if clean and all(r.get(f"earnings_at_{k}") is not None for r in clean)
        },
        "deliveries_mean": (round(stats.mean(
            [r.get("delivered", 0) for r in clean]), 4) if clean else None),
        "zero_delivery_rate": (round(sum(
            1 for r in clean if not r.get("delivered")) / len(clean), 4)
            if clean else None),
        "scores_by_seed": {r["seed"]: r["earnings"] for r in clean},
        "crashed_seeds": crashed,
        "infra_error_seeds": infra,
        "task": {k: v for k, v in task.items() if not str(k).endswith("_root")},
        "finished_at": int(time.time()),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print()
    if earnings:
        print(f"earnings: {summary['earnings_mean']:.3f} "
              f"[{ci[0]:.3f}, {ci[1]:.3f}]  "
              f"deliveries/shift: {summary['deliveries_mean']:.2f}  "
              f"zero-rate: {summary['zero_delivery_rate']:.0%}  "
              f"(n={len(clean)})")
    if crashed or infra:
        print(f"WARNING: {len(crashed)} crashed episode(s), {len(infra)} ended "
              f"by infrastructure -- excluded from the mean; this run's number "
              f"is not clean, fix and rerun")
    print(f"wrote {summary_path}" + (f" and {transcripts_path}" if args.transcripts else ""))
    return 1 if (crashed or infra) else 0


def cmd_compare(args) -> int:
    a = json.loads(Path(args.a).read_text())
    b = json.loads(Path(args.b).read_text())
    result = stats.paired_compare(
        {int(k): v for k, v in a["scores_by_seed"].items()},
        {int(k): v for k, v in b["scores_by_seed"].items()})
    print(f"A = {a.get('tag')}   B = {b.get('tag')}   "
          f"(paired over {result.get('n')} shared seeds)")
    for key, value in result.items():
        print(f"  {key}: {value}")
    if result.get("n"):
        verdict = ("B > A, and the interval excludes zero"
                   if result["ci_excludes_zero"]
                   and result["mean_diff_b_minus_a"] > 0 else
                   "A > B, and the interval excludes zero"
                   if result["ci_excludes_zero"] else
                   "not distinguishable from noise on these seeds")
        print(f"  verdict: {verdict}")
    return 0
