"""Turn evaluation failures into order-sampling weights.

The adaptive curriculum's diagnostic half. Episodes append one JSON line of
failure taxonomy each (``vagen_courier_env._log_episode``); this reads the
validation lines, measures where the policy is weak, and writes a profile the
dispatcher's ``order_bias`` consumes. The mapping is deliberately explicit --
four failure rates, four order classes -- because a curriculum whose reasoning
cannot be read is a curriculum whose bugs cannot be found:

    late deliveries        → weight len_long        (deadline pressure lives
                                                     in the long orders)
    red-light crossings    → weight signalled       (routes that pass lamps)
    detour ratio high      → weight junction_dense  (routes with real choices)
    zero-delivery shifts   → weight len_short       (competence has to start
                                                     somewhere; a policy that
                                                     never delivers gets no
                                                     gradient from harder)

Weights are 1.0 + gain * rate, capped, so a policy with no weakness gets the
benchmark distribution back -- the curriculum anneals itself away as the
failures disappear. This is the ADR/PLR family (reweight where the learner is
weak) with the signal taken from held-out greedy evaluation rather than from
training reward, so the knob the optimiser could game is not the knob that
steers.

    python tools/adaptive_profile.py --log EP.jsonl --out profile.json
    python tools/adaptive_profile.py --log EP.jsonl --out profile.json \
        --watch 600        # rebuild every 10 min, for a sidecar loop
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Validation seeds live in [VAL_SEED_MIN, VAL_SEED_END); every training pool
# jumps over that window (experiments/README.md), so the two halves of one
# episode log separate on the seed alone. `seed >= 1000` on its own filed the
# 1200-4199 training seeds as validation and drove the curriculum off the
# (adaptively skewed) training rollouts it exists to ignore.
VAL_SEED_MIN = 1000
VAL_SEED_END = 1200
# How many of the most recent validation episodes constitute "now".
WINDOW = 128
GAIN = 2.0          # weight = 1 + GAIN * rate, so rate 1.0 → weight 3.0
CAP = 3.0
DETOUR_BAD = 1.6    # walking 60% further than optimal counts as lost


# E4b's three restraints (the E4 post-mortem in one line: an abrupt
# distribution step starved the reward signal and the entropy term took the
# gradient). EMA_KEEP smooths every weight toward its target instead of
# jumping; the freeze gate holds the previous bias whenever recent training
# earnings fall below FREEZE_FRACTION of their historical peak -- adaptation
# may steer a healthy run, never chase a sick one downhill.
EMA_KEEP = 0.7
FREEZE_FRACTION = 0.7
TRAIN_WINDOW = 200

# ── modes: four ways to decide where the dispatcher should lean ─────────────
#
# The evolving environment is the core claim, so it gets a bake-off rather
# than one design. Every mode writes the same artefact (an order-class bias
# the dispatcher consumes identically); they differ only in the *signal*:
#
#   fail      failure rates on held-out validation (the e4b design, verbatim)
#   frontier  learnability: GRPO rolls each seed GROUP times, so a seed whose
#             group's earnings VARY is one the policy sometimes solves --
#             the learning frontier of PLR/ACCEL. Classes present in
#             high-variance shifts get the weight; solved (variance ~0,
#             earning) and hopeless (variance ~0, broke) both anneal away.
#             Signal comes from training groups, weights only reweight order
#             classes, so the optimiser cannot game its own curriculum
#             directly.
#   hard      fail's signal at double dose: faster EMA, higher cap, looser
#             freeze. The "was e4b just too gentle?" arm.
#   ladder    self-paced difficulty: no failure attribution at all, just a
#             gate sequence -- short orders until the policy delivers, then
#             mid, long, junction-dense, signalled, each unlocked by the
#             delivery rate crossing a threshold. The control that says
#             whether *reacting* to the policy matters or any schedule works.
MODES = ("fail", "frontier", "hard", "ladder", "city", "city_need", "city_ladder", "skill_battery")
FRONTIER_GAIN = 2.0
HARD = {"gain": 3.0, "cap": 4.0, "ema_keep": 0.4, "freeze_fraction": 0.5}
LADDER_STEPS = (          # (unlock when delivery rate >=, then favour class)
    (0.00, "len_short"),
    (0.35, "len_mid"),
    (0.55, "len_long"),
    (0.70, "junction_dense"),
    (0.80, "signalled"),
)
LADDER_WEIGHT = 2.5

# ── city-level curriculum (the RQ2xRQ3 arm) ────────────────────────────────
#
#   city         learnability over CITIES: group recent training rollouts by
#                seed (GRPO replays one seed GROUP times), credit each
#                group's earnings-std to the city the shift ran in, and
#                weight cities by mean std. Solved cities (variance ~0,
#                earning) and hopeless ones (variance ~0, broke) both fade;
#                the dispatcher sends the courier where learning is live.
#   city_ladder  the non-responsive control: a fixed small-to-large unlock
#                schedule gated only by held-out delivery rate. If this does
#                as well as `city`, responsiveness is not the ingredient.
#   skill_battery  the skill-gated arm: while held-out phone_alive_rate is
#                below the bar, lean the order mix toward long, signalled
#                shifts (they drain the battery, so battery crises -- and
#                practice at charge_phone() -- happen more). At mastery the
#                lean decays and the benchmark distribution returns.
CITY_GAIN = 2.0
CITY_CAP = 3.0
CITY_LADDER_ORDER = (
    "small-city-11", "small-city-13", "small-city-15",
    "medium-city-20", "medium-city-22",
    "large-city-26", "large-city-28", "citycore-paris",
)
CITY_LADDER_GATES = (0.0, 0.30, 0.40, 0.50, 0.58, 0.66, 0.72, 0.78)
CITY_LADDER_WEIGHT = 2.5
SKILL_ALIVE_BAR = 0.90     # below: lean into battery-draining shifts
SKILL_ALIVE_MASTERY = 0.95 # above: the lean decays away
SKILL_WEIGHT = 2.0


def build(log_path: Path, val_seed_min: int = VAL_SEED_MIN,
          previous: dict | None = None, mode: str = "fail") -> dict:
    if mode not in MODES:
        raise ValueError(f"mode is one of {MODES}, not {mode!r}")
    rows = []
    train_rows = []
    with log_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue  # a torn line from a concurrent writer; skip it
            seed = int(row.get("seed", -1))
            if val_seed_min <= seed < VAL_SEED_END:
                rows.append(row)
            else:
                train_rows.append(row)
    rows = rows[-WINDOW:]
    train_rows = train_rows[-max(TRAIN_WINDOW,
                             400 if mode == "frontier" else 0,
                             800 if mode in ("city", "city_need") else 0):]
    n = len(rows)
    if n == 0:
        return {"bias": {}, "episodes": 0, "mode": mode,
                "note": "no validation episodes yet; baseline distribution"}

    delivered = [int(r.get("delivered") or 0) for r in rows]
    late = sum(int(r.get("late") or 0) for r in rows)
    finished = sum(delivered) + late  # late ones still arrived
    zero_rate = sum(1 for d in delivered if d == 0) / n
    late_rate = late / finished if finished else 0.0
    red_rate = sum(1 for r in rows if int(r.get("red_crossings") or 0) > 0) / n
    detours = [float(r["detour"]) for r in rows if r.get("detour")]
    detour_rate = (sum(1 for d in detours if d >= DETOUR_BAD) / len(detours)
                   if detours else 0.0)

    gain = HARD["gain"] if mode == "hard" else GAIN
    cap = HARD["cap"] if mode == "hard" else CAP
    ema_keep = HARD["ema_keep"] if mode == "hard" else EMA_KEEP
    freeze_fraction = (HARD["freeze_fraction"] if mode == "hard"
                       else FREEZE_FRACTION)

    def w(rate: float) -> float:
        return round(min(cap, 1.0 + gain * rate), 3)

    delivery_rate = sum(1 for d in delivered if d > 0) / n

    city_weights: dict = {}
    if mode == "ladder":
        # Self-paced gate sequence: favour the hardest class the policy has
        # unlocked. No failure attribution -- this is the control arm that
        # asks whether *reacting* matters or any schedule works.
        unlocked = [cls for gate, cls in LADDER_STEPS if delivery_rate >= gate]
        bias = {unlocked[-1]: LADDER_WEIGHT} if unlocked else {}
    elif mode == "city":
        groups: dict[int, list[dict]] = {}
        for r in train_rows:
            groups.setdefault(int(r.get("seed", -1)), []).append(r)
        city_var: dict[str, list[float]] = {}
        for members in groups.values():
            if len(members) < 2:
                continue
            city = str(members[0].get("city") or "")
            if not city:
                continue
            money = [float(m.get("earnings") or 0.0) for m in members]
            mean = sum(money) / len(money)
            std = (sum((x - mean) ** 2 for x in money) / len(money)) ** 0.5
            city_var.setdefault(city, []).append(std)
        scores = {c: sum(v) / len(v) for c, v in city_var.items() if v}
        top = max(scores.values()) if scores else 0.0
        bias = {}
        city_weights = ({c: round(min(CITY_CAP, 1.0 + CITY_GAIN * sc / top), 3)
                         for c, sc in scores.items()} if top > 0 else {})
    elif mode == "city_need":
        # The obvious heuristic the learnability mode must beat: upweight
        # the cities where mean earnings are LOWEST. Need-based targeting
        # cannot tell "hard but learnable" from "hopeless", which is
        # exactly the distinction PLR-style variance targeting exists to
        # draw -- so this arm is the ablation that shows whether the
        # distinction matters here.
        sums: dict[str, list[float]] = {}
        for r in train_rows:
            c = str(r.get("city") or "")
            if c:
                sums.setdefault(c, []).append(float(r.get("earnings") or 0.0))
        means = {c: sum(v) / len(v) for c, v in sums.items() if v}
        top = max(means.values()) if means else 0.0
        bias = {}
        city_weights = ({c: round(min(CITY_CAP, 1.0 + CITY_GAIN * (1.0 - m / top)), 3)
                         for c, m in means.items()} if top > 0 else {})
    elif mode == "city_ladder":
        unlocked = [c for gate, c in zip(CITY_LADDER_GATES, CITY_LADDER_ORDER)
                    if delivery_rate >= gate]
        bias = {}
        city_weights = ({c: 1.0 for c in unlocked} if unlocked else {})
        if unlocked:
            city_weights[unlocked[-1]] = CITY_LADDER_WEIGHT
    elif mode == "skill_battery":
        alive = [r.get("phone_alive") for r in rows
                 if r.get("phone_alive") is not None]
        alive_rate = (sum(1 for a in alive if a) / len(alive)) if alive else 1.0
        if alive and alive_rate < SKILL_ALIVE_BAR:
            bias = {"len_long": SKILL_WEIGHT, "signalled": SKILL_WEIGHT}
        elif alive and alive_rate < SKILL_ALIVE_MASTERY:
            bias = dict((previous or {}).get("bias") or {})   # hold
        else:
            bias = {}                                          # decay via EMA
    elif mode == "frontier":
        # Learnability. GRPO rolls each seed GROUP times, so the recent
        # training rows group by seed; a group whose earnings VARY is a shift
        # the policy sometimes solves -- the frontier. Credit that variance
        # to the order classes the shift contained; classes the policy has
        # solved (variance ~0) or cannot touch (variance ~0) both fade.
        groups: dict[int, list[dict]] = {}
        for r in train_rows:
            groups.setdefault(int(r.get("seed", -1)), []).append(r)
        cls_var: dict[str, list[float]] = {}
        for members in groups.values():
            if len(members) < 2:
                continue
            money = [float(m.get("earnings") or 0.0) for m in members]
            mean = sum(money) / len(money)
            std = (sum((x - mean) ** 2 for x in money) / len(money)) ** 0.5
            classes = {}
            for m in members:
                for cls, cnt in (m.get("order_classes") or {}).items():
                    classes[cls] = classes.get(cls, 0) + int(cnt)
            for cls in classes:
                cls_var.setdefault(cls, []).append(std)
        scores = {cls: sum(v) / len(v) for cls, v in cls_var.items() if v}
        top = max(scores.values()) if scores else 0.0
        bias = ({cls: round(min(cap, 1.0 + FRONTIER_GAIN * s / top), 3)
                 for cls, s in scores.items() if s > 0} if top > 0 else {})
    else:  # fail / hard: held-out failure rates, e4b's mapping verbatim
        bias = {}
        if late_rate > 0.05:
            bias["len_long"] = w(late_rate)
        if red_rate > 0.05:
            bias["signalled"] = w(red_rate)
        if detour_rate > 0.05:
            bias["junction_dense"] = w(detour_rate)
        if zero_rate > 0.25:
            # Overrides the "harder" weights when the policy cannot deliver
            # at all: no deadline practice for a courier who never arrives.
            bias = {"len_short": w(zero_rate)}

    prev_bias = dict((previous or {}).get("bias") or {})
    peak = float((previous or {}).get("train_earnings_peak") or 0.0)
    # The freeze gate watches training earnings -- except for the city
    # modes, where a curriculum doing its JOB (steering shifts into the
    # poorest cities) depresses training earnings by construction and
    # would freeze itself permanently. Held-out earnings are mode-neutral
    # health: they only fall when the policy is actually getting worse.
    health_rows = (rows if mode in ("city", "city_need", "city_ladder")
                   else train_rows[-TRAIN_WINDOW:])
    recent = (sum(float(r.get("earnings") or 0.0) for r in health_rows)
              / len(health_rows)) if health_rows else 0.0
    peak = max(peak, recent)
    frozen = bool(peak and recent < freeze_fraction * peak)
    if frozen:
        bias = prev_bias
    else:
        # EMA toward the target: absent keys decay toward 1.0 (no bias).
        keys = set(prev_bias) | set(bias)
        bias = {k: round(ema_keep * prev_bias.get(k, 1.0)
                         + (1 - ema_keep) * bias.get(k, 1.0), 3)
                for k in keys}
        bias = {k: v for k, v in bias.items() if abs(v - 1.0) > 0.02}

    prev_cities = dict((previous or {}).get("city_weights") or {})
    if mode in ("city", "city_need", "city_ladder") and not frozen:
        keys = set(prev_cities) | set(city_weights)
        city_weights = {k: round(ema_keep * prev_cities.get(k, 1.0)
                                 + (1 - ema_keep) * city_weights.get(k, 1.0), 3)
                        for k in keys}
        city_weights = {k: v for k, v in city_weights.items()
                        if abs(v - 1.0) > 0.02 or mode == "city_ladder"}
    elif mode in ("city", "city_need", "city_ladder"):
        city_weights = prev_cities
    else:
        city_weights = {}

    return {
        "bias": bias,
        "city_weights": city_weights,
        "mode": mode,
        "episodes": n,
        "rates": {"zero_delivery": round(zero_rate, 3),
                  "late": round(late_rate, 3),
                  "red_light": round(red_rate, 3),
                  "detour": round(detour_rate, 3),
                  "delivery": round(delivery_rate, 3)},
        "train_earnings_recent": round(recent, 2),
        "train_earnings_peak": round(peak, 2),
        "frozen": frozen,
        "built_at": int(time.time()),
    }


def write_atomic(profile: dict, out: Path) -> None:
    # rename(2) is atomic on the same filesystem, and the env's reader caches
    # on mtime -- a half-written JSON must never be observable.
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(profile, indent=2) + "\n")
    os.replace(tmp, out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--log", required=True, help="episode taxonomy jsonl")
    ap.add_argument("--out", required=True, help="profile json to write")
    ap.add_argument("--watch", type=int, default=0,
                    help="rebuild every N seconds instead of once")
    ap.add_argument("--val-seed-min", type=int, default=VAL_SEED_MIN)
    ap.add_argument("--mode", choices=MODES, default="fail",
                    help="curriculum signal: fail (e4b), frontier "
                         "(learnability), hard (double dose), ladder "
                         "(self-paced schedule)")
    args = ap.parse_args()

    log, out = Path(args.log), Path(args.out)

    while True:
        if log.exists():
            previous = None
            if out.exists():
                try:
                    previous = json.loads(out.read_text())
                except ValueError:
                    previous = None
            profile = build(log, args.val_seed_min, previous=previous,
                            mode=args.mode)
            write_atomic(profile, out)
            print(f"{time.strftime('%H:%M:%S')} mode={args.mode} "
                  f"episodes={profile['episodes']} "
                  f"bias={profile['bias']} "
                  f"city_weights={profile.get('city_weights')} "
                  f"frozen={profile.get('frozen')} "
                  f"train_recent={profile.get('train_earnings_recent')} "
                  f"rates={profile.get('rates')}",
                  flush=True)
        elif not args.watch:
            print(f"no episode log at {log}; nothing to build", file=sys.stderr)
            return 1
        if not args.watch:
            return 0
        time.sleep(args.watch)


if __name__ == "__main__":
    raise SystemExit(main())
