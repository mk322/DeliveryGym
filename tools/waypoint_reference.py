#!/usr/bin/env python3
"""The waypoint benchmark's reference policies -- its upper bound and its
floor -- on the held-out shifts, built exactly as the evaluator builds them
and written in the evaluator's own results shape, so ``embodiedbench.eval
compare`` can pair either with a model's run seed by seed.

    ALBUMS_DIR=/data/albums python tools/waypoint_reference.py --out results/ceiling.json
    ALBUMS_DIR=/data/albums python tools/waypoint_reference.py --objective floor --out results/floor.json
    python -m embodiedbench.eval compare results/qwen3-vl-4b.json results/ceiling.json

The **ceiling** (``--objective best``, the default) is the privileged
graph-reading courier: it reads the road graph, waits at red and learns
barriers by walking into them (``embodiedbench/tasks/courier_router.py``).
It is not a policy: it sees the survey a policy never sees and spends no
turn on the phone. ``--objective moves`` (``FewestMovesCourier``) plans in
turns, the thing a 60-turn shift rations; ``--objective metres``
(``ShortestPathCourier``) plans in distance; ``best`` keeps the better of
the two per shift.

The **floor** (``--objective floor``) is the text-only courier
(``embodiedbench/tasks/courier_oracle.py``): it reads what the world says
-- the order slip, the phone's range, the street sign and the door numbers
underfoot -- and never a coordinate or a picture. If it delivers, the words
are sufficient; a vision policy that earns less than it is not reading the
pictures for anything the words do not already say.

Each shift is the val yaml's task through ``CourierGymEnv`` -- endless
tier, one order at a time, the 60-turn horizon -- and every reference gets
exactly as many *turns* as the benchmark gives a model (a phone check costs
the floor a turn as it costs a model one), with the running earnings
recorded at the same checkpoints (20/40/60), so the files read on one
scale.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from embodiedbench.eval import stats  # noqa: E402
from embodiedbench.eval.run import EARNINGS_CHECKPOINTS, SPLITS, load_task_config  # noqa: E402
from embodiedbench.tasks.courier_oracle import ObservationOnlyCourier  # noqa: E402
from embodiedbench.tasks.courier_router import FewestMovesCourier, ShortestPathCourier  # noqa: E402

CEILING_COURIERS = {"moves": FewestMovesCourier, "metres": ShortestPathCourier}
COURIERS = {**CEILING_COURIERS, "floor": ObservationOnlyCourier}
#: ``best`` runs both ceiling planners on the same shift and keeps the
#: better: the two differ only in which barriers they happen to walk into
#: and which order the dispatcher hands them next, so the better of the two
#: is the tighter bound and the one reported.
OBJECTIVES = ("best", *sorted(COURIERS))
CEILING_MODEL = "graph-ceiling"
FLOOR_MODEL = "text-floor"
#: The couriers' own loop cap; the turn budget is what stops them.
UNBOUNDED_STEPS = 1_000_000


def model_name(objective: str) -> str:
    return FLOOR_MODEL if objective == "floor" else CEILING_MODEL


def run_shift(world: Any, seed: int, cap: int, *, objective: str = "moves") -> dict[str, Any]:
    """One shift of a reference courier on a reset world: ``cap`` turns at
    most, the running earnings noted at every checkpoint on the way."""
    courier = COURIERS[objective](world, max_steps=UNBOUNDED_STEPS)
    marks = sorted({k for k in EARNINGS_CHECKPOINTS if k <= cap} | {cap})
    earnings_at: dict[int, float] = {}
    result = None
    for mark in marks:
        courier.stop_at_turns = mark
        if world.turns < mark and not (world.finished or world.shift_over):
            result = courier.run(seed)
        earnings_at[mark] = float(world.summary().get("earnings") or 0.0)
    summary = world.summary()
    final = float(summary.get("earnings") or 0.0)
    turns = int(summary.get("turns", 0) or 0)
    return {
        "seed": seed,
        "earnings": round(final, 2),
        **{f"earnings_at_{k}": round(earnings_at.get(k, final), 2)
           for k in EARNINGS_CHECKPOINTS if k <= cap},
        "delivered": int(summary.get("delivered") or 0),
        "on_time": int(summary.get("on_time") or 0),
        "late": int(summary.get("late") or 0),
        "red_crossings": int(summary.get("red_crossings") or 0),
        "turns": turns,
        "walked_m": round(world.walked_cm / 100.0, 1),
        "sim_seconds": round(float(world.sim_seconds), 1),
        "termination": ("shift_over" if world.shift_over
                        else "finished" if world.finished
                        else "out_of_turns" if turns >= cap
                        else "stopped"),
        "trace": list(result.trace[-1:]) if result is not None else [],
        "infra_errors": 0,
    }


def summarise(rows: list[dict[str, Any]], *, task: dict, split: str, cap: int,
              objective: str = "best", tag: str | None = None) -> dict[str, Any]:
    """The evaluator's summary shape over a reference courier's rows."""
    earnings = [float(r["earnings"]) for r in rows]
    low, high = stats.bootstrap_ci(earnings) if earnings else (0.0, 0.0)
    if objective == "floor":
        policy = ("ObservationOnlyCourier: reads only what the world says -- the "
                  "order slip, the phone's range, the street sign and the door "
                  "numbers -- never a coordinate or a picture; the benchmark's "
                  "floor, not a policy under test")
    else:
        if objective == "best":
            courier = ("the better per shift of FewestMovesCourier (plans in turns) "
                       "and ShortestPathCourier (plans in metres)")
        else:
            courier = (f"{COURIERS[objective].__name__}: plans in "
                       f"{'turns' if objective == 'moves' else 'metres'}")
        policy = (f"{courier}; reads the road graph, waits at red, "
                  "learns barriers by walking into them; the benchmark's "
                  "upper bound, not a policy under test")
    name = model_name(objective)
    return {
        "model": name, "tag": tag or f"{name}-{objective}", "split": split,
        "objective": objective,
        "policy": policy,
        "episodes": len(rows), "scored_episodes": len(rows),
        "earnings_mean": round(stats.mean(earnings), 4) if rows else None,
        "earnings_ci95": [round(low, 4), round(high, 4)],
        "earnings_at_mean": {
            f"earnings_at_{k}": round(stats.mean(
                [float(r[f"earnings_at_{k}"]) for r in rows]), 4)
            for k in EARNINGS_CHECKPOINTS
            if rows and all(r.get(f"earnings_at_{k}") is not None for r in rows)
        },
        "deliveries_mean": round(stats.mean([r["delivered"] for r in rows]), 4) if rows else None,
        "on_time_mean": round(stats.mean([r["on_time"] for r in rows]), 4) if rows else None,
        "zero_delivery_rate": (round(sum(1 for r in rows if not r["delivered"]) / len(rows), 4)
                               if rows else None),
        "red_crossings_mean": round(stats.mean([r["red_crossings"] for r in rows]), 3) if rows else None,
        "scores_by_seed": {str(r["seed"]): r["earnings"] for r in rows},
        "crashed_seeds": [], "infra_error_seeds": [],
        "action_cap": cap,
        "task": {k: v for k, v in task.items() if not str(k).endswith("_root")},
        "finished_at": int(time.time()),
    }


def run(split: str, cap: int | None, seeds: list[int] | None,
        *, objective: str = "best") -> tuple[dict, list[dict]]:
    from embodiedbench.training.vagen_courier_env import CourierGymEnv

    task = load_task_config()
    cap = int(task["max_turns"]) if cap is None else cap
    planners = sorted(CEILING_COURIERS) if objective == "best" else [objective]
    rows = []
    loop = asyncio.new_event_loop()
    try:
        for seed in (seeds or list(SPLITS[split])):
            candidates = []
            for planner in planners:
                env = CourierGymEnv(env_config=dict(task))
                loop.run_until_complete(env.reset(seed=seed))
                try:
                    row = run_shift(env.world, seed, cap, objective=planner)
                finally:
                    try:
                        loop.run_until_complete(env.close())
                    except Exception:  # noqa: BLE001 - the adapter is done with
                        pass
                row["planner"] = planner
                candidates.append(row)
            # the better shift; a tie goes to the turn planner (the bound's
            # own objective), which sorts after "metres"
            rows.append(max(candidates, key=lambda r: (r["earnings"], r["planner"])))
            print(f"  seed {seed}: earnings={rows[-1]['earnings']:.2f} "
                  f"delivered={rows[-1]['delivered']} turns={rows[-1]['turns']} "
                  f"{rows[-1]['termination']}", flush=True)
    finally:
        loop.close()
    return summarise(rows, task=task, split=split, cap=cap, objective=objective), rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--split", choices=sorted(SPLITS), default="val")
    parser.add_argument("--max-turns", type=int, default=None,
                        help="turn budget (default: the val yaml's max_turns)")
    parser.add_argument("--seed-list", help="comma-separated explicit seeds")
    parser.add_argument("--objective", choices=OBJECTIVES, default="best",
                        help="best: the better per shift of the turn planner and the "
                             "metres planner (the ceiling); moves or metres alone; "
                             "floor: the text-only courier")
    parser.add_argument("--out", type=Path, required=True,
                        help="summary json; the rows go beside it as .episodes.jsonl")
    args = parser.parse_args(argv)
    seeds = [int(s) for s in args.seed_list.split(",")] if args.seed_list else None
    summary, rows = run(args.split, args.max_turns, seeds, objective=args.objective)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    rows_path = args.out.with_name(args.out.stem + ".episodes.jsonl")
    with rows_path.open("w") as sink:
        for row in rows:
            sink.write(json.dumps(row) + "\n")
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("scores_by_seed", "task")}, indent=1))
    print(f"wrote {args.out} and {rows_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
