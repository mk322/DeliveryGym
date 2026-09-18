"""Turn episode telemetry into the two numbers a live run is judged by.

    python -m embodiedbench.runtime.live.report /path/to/nav_telemetry

**How fast is it really.** Not the engine's tick rate in a microbenchmark --
the acceleration a *training run* actually received, which is the sim seconds
the couriers walked divided by the wall seconds the fleet was alive. Every gap
between the two is somewhere the fleet was not walking: queueing behind another
episode, waiting on the policy, booting.

**How much of the failure is the simulator's.** ``stuck`` and ``walk_timeout``
arrive at the policy wearing the same clothes as a bad action -- they are
refusals, they charge time, and the return goes down. That is defensible only
while the rate is low and *known*. Unknown, it is a systematic bias with the
policy's name on it: the gradient learns to avoid edges the navmesh cannot
walk, which is a fact about the map, not about being a good courier.

Everything here is read-only over the JSONL the envs append to. It answers on
partial runs, because the interesting moment to ask is usually mid-run.

Throughput is reported TWICE, on purpose. A single figure changes meaning as
the directory fills up: over one rollout burst it says how fast the world
walks, and over several training steps it also counts the optimizer time
between them, when the fleet is idle by design. Measured on the development workstation the same
run read 6.25x at eight episodes and 3.45x at twelve, with nothing having got
slower -- which is exactly the trap. So:

* ``sim_seconds_per_active_second`` -- how fast the world walks, divided by
  the wall clock during which some episode was actually open.
* ``sim_seconds_per_wall_second`` -- what a training step costs end to end,
  divided by the whole window, with ``idle_share_of_wall`` naming the gap.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator


def load(directory: Path) -> Iterator[dict[str, Any]]:
    """Every record under a telemetry directory, tolerating a live writer.

    The last line of a file being appended to right now can be short; that is
    normal, not corruption, and it costs one episode out of the report rather
    than the report itself.
    """
    for path in sorted(directory.glob("episodes-*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def _spread(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "p50": None, "max": None}
    ordered = sorted(values)
    return {"n": len(ordered),
            "p50": round(ordered[len(ordered) // 2], 1),
            "max": round(ordered[-1], 1)}


def _union_seconds(intervals: list[tuple[float, float]]) -> float:
    """Total length of the union of half-open intervals.

    Episodes on different instances overlap, so summing their durations
    overcounts; taking the outer range undercounts the idleness between
    bursts. The union is the thing that means "the fleet was working".
    """
    if not intervals:
        return 0.0
    total = 0.0
    current_start, current_end = None, None
    for start, end in sorted(intervals):
        if current_end is None or start > current_end:
            if current_end is not None:
                total += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    if current_end is not None:
        total += current_end - current_start
    return total


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"episodes": 0}

    outcomes: Counter[str] = Counter()
    walk_sim_seconds = 0.0
    hops = recoveries = degraded = busy_waits = stranded = 0
    busy_seconds = wall_seconds = 0.0
    pose_errors: dict[str, list[float]] = {"arrived": [], "failed": []}
    breaches = 0
    engine_seconds = graph_seconds = 0.0
    workers: Counter[int] = Counter()
    action_spaces: Counter[str] = Counter()
    delivered = 0

    for record in records:
        counters = record.get("counters", {})
        # Per record, not a constant: arrive_cm is a config axis and a
        # directory can hold episodes from more than one setting.
        arrive_cm = float((record.get("config") or {}).get("arrive_cm") or 0.0)
        action_spaces[str((record.get("config") or {}).get("action_space")
                          or "street")] += 1
        outcomes.update(counters.get("outcomes", {}))
        hops += int(counters.get("hops", 0))
        recoveries += int(counters.get("recoveries", 0))
        stranded += sum(1 for h in record.get("hops", [])
                        if h.get("recovery") == "reopen_failed")
        degraded += 1 if counters.get("degraded") else 0
        busy_waits += int(counters.get("busy_waits", 0))
        busy_seconds += float(counters.get("busy_wait_seconds", 0.0))
        wall_seconds += float(record.get("wall_seconds") or 0.0)
        workers[record.get("pid", -1)] += 1
        for hop in record.get("hops", []):
            if "sim_seconds" in hop:
                walk_sim_seconds += float(hop["sim_seconds"])
            if "pose_error_cm" in hop:
                # Split by outcome, because mixing them raises false alarms.
                # An arrived hop must land inside arrive_cm -- that is the
                # contract, and a breach is a real defect. A stuck hop leaves
                # the pawn wherever it stalled, which is by DESIGN and is
                # followed by a respawn. Reported together, four stuck hops
                # turn a clean "max 120.0 cm, exactly the contract" into
                # "max 12.2 m", which reads like the contract broke.
                error_cm = float(hop["pose_error_cm"])
                arrived = hop.get("outcome") == "arrived"
                pose_errors["arrived" if arrived else "failed"].append(error_cm)
                # A rounding-width tolerance, no more: the service declares
                # arrival at a chunk boundary INSIDE the radius, so a landing
                # outside it is not a near miss, it is the invariant failing.
                if arrived and arrive_cm and error_cm > arrive_cm + 0.5:
                    breaches += 1
            # Only arrived hops: a refused hop is charged a floor, not a
            # travel price, so including them would compare two different
            # things and flatter the ratio.
            if hop.get("outcome") == "arrived" and "graph_seconds" in hop:
                engine_seconds += float(hop.get("sim_seconds", 0.0))
                graph_seconds += float(hop["graph_seconds"])
        summary = record.get("summary") or {}
        if summary.get("delivered"):
            delivered += int(summary["delivered"])

    # The wall clock the fleet was actually alive: earliest OPEN to latest
    # close. Not the sum of episode durations -- episodes on different
    # instances overlap, and summing them would report an acceleration the
    # run never had. And emphatically not the range of CLOSING times, which
    # was the first thing tried here and is wrong in the same direction:
    # concurrent episodes finish in a burst, so that range is a fraction of
    # the time they took and it inflated the headline number by roughly an
    # order of magnitude.
    closes = [r["closed_at"] for r in records if r.get("closed_at")]
    opens = [r["closed_at"] - r["wall_seconds"] for r in records
             if r.get("closed_at") and r.get("wall_seconds")]
    span_is_exact = len(opens) == len(closes) and bool(opens)
    # The union of the episode intervals: wall clock during which SOME
    # episode was open somewhere on the fleet. The difference between this
    # and the total span is the time no episode was running at all, which in
    # a training run is the optimizer -- idle by design, not by fault.
    active = _union_seconds(
        [(c - w, c) for c, w in
         ((r.get("closed_at"), r.get("wall_seconds")) for r in records)
         if c and w]) if span_is_exact else 0.0
    if span_is_exact:
        span = max(closes) - min(opens)
    elif len(closes) > 1:
        # Older records predate the open stamp. Report the number, and say
        # what it is: an upper bound, not a measurement.
        span = max(closes) - min(closes)
    else:
        span = 0.0

    total_walks = sum(outcomes.values())
    sim_failures = outcomes.get("stuck", 0) + outcomes.get("timeout", 0) \
        + outcomes.get("walk_timeout", 0)

    return {
        "episodes": len(records),
        "worker_processes": len(workers),
        "episodes_per_worker": dict(sorted(workers.items())),
        # Which question these episodes were asked. First, and a count rather
        # than a name, because the coordinate action space exists to be
        # compared against the street one and the failure mode of that
        # comparison is reading two arms out of one directory as though they
        # were one run. More than one entry here means the numbers below are
        # a blend and every one of them is meaningless.
        "action_spaces": dict(action_spaces),
        "hops": hops,
        "outcomes": dict(outcomes),
        # THE bias number: what fraction of walks failed for reasons the
        # policy could not have avoided, and which the policy is charged for.
        "sim_failure_rate": (round(sim_failures / total_walks, 4)
                             if total_walks else None),
        "recoveries": recoveries,
        # The one branch where pose error stops being bounded: a respawn that
        # failed leaves the pawn where it stalled, and every later walk in
        # that episode starts from the wrong place. Should be zero.
        "stranded_after_failed_respawn": stranded,
        "episodes_degraded_to_album": degraded,
        "delivered": delivered,
        "walk_sim_seconds": round(walk_sim_seconds, 1),
        "wall_span_seconds": round(span, 1),
        # False when some episode lacks an open stamp: the span then covers
        # only the closing burst, so every rate derived from it is an upper
        # bound rather than a measurement.
        "wall_span_exact": span_is_exact,
        # THE throughput number: sim seconds walked per wall second, across
        # the whole fleet. Above 1.0 means the world moved faster than the
        # clock on the wall; it is bounded by how much of the wall clock goes
        # to the policy rather than the engine.
        "sim_seconds_per_wall_second": (round(walk_sim_seconds / span, 3)
                                        if span > 0 else None),
        # Two numbers, both true, answering different questions. The one
        # above divides by the WHOLE window, so over several training steps
        # it also counts the optimizer time between rollouts and answers
        # "what does a training step cost in wall clock". The one below
        # divides by the time some episode was actually open and answers
        # "how fast does the world walk". Reporting only one of them makes
        # the figure change meaning silently as a directory fills up --
        # measured here, 6.25x over one rollout burst became 3.45x over
        # three steps of the same run, with nothing having got slower.
        "rollout_active_seconds": round(active, 1),
        "sim_seconds_per_active_second": (round(walk_sim_seconds / active, 3)
                                          if active > 0 else None),
        "idle_share_of_wall": (round(1.0 - active / span, 4)
                               if span > 0 and active > 0 else None),
        "busy_waits": busy_waits,
        "busy_wait_seconds": round(busy_seconds, 1),
        # Queueing as a share of the run: the honest measure of "is the fleet
        # too small for this rollout width".
        "busy_share_of_wall": (round(busy_seconds / span, 4)
                               if span > 0 else None),
        # How much more the engine charges for the same hop than the graph
        # price every budget in the task is still computed from. 1.0 means the
        # two agree; above 1.0, the live env is quietly running a harder
        # version of the benchmark than the offline one.
        "clock_inflation": (round(engine_seconds / graph_seconds, 4)
                            if graph_seconds > 0 else None),
        "engine_seconds_arrived": round(engine_seconds, 1),
        "graph_seconds_arrived": round(graph_seconds, 1),
        "pose_error_cm": {
            outcome: _spread(values) for outcome, values in pose_errors.items()
        },
        # Any arrived hop outside arrive_cm is a contract breach, not a
        # tolerance. Should always be zero.
        "arrivals_outside_contract": breaches,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory", type=Path)
    parser.add_argument("--json", action="store_true",
                        help="machine-readable, for a dashboard")
    args = parser.parse_args(argv)

    records = list(load(args.directory))
    report = summarize(records)
    if args.json:
        json.dump(report, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    if not report["episodes"]:
        print(f"no episode records under {args.directory}")
        return 1
    print(f"episodes            {report['episodes']} "
          f"across {report['worker_processes']} worker processes")
    spaces = report["action_spaces"]
    print(f"action space        {spaces}"
          + ("   ** MIXED: every number below blends two different tasks **"
             if len(spaces) > 1 else ""))
    print(f"hops                {report['hops']}  {report['outcomes']}")
    print(f"sim failure rate    {report['sim_failure_rate']}   "
          f"(stuck/timeout charged to the policy)")
    print(f"recoveries          {report['recoveries']}   "
          f"stranded {report['stranded_after_failed_respawn']}   "
          f"degraded episodes {report['episodes_degraded_to_album']}")
    print(f"world speed         {report['sim_seconds_per_active_second']} "
          f"sim-sec per wall-sec WHILE ROLLING  "
          f"({report['walk_sim_seconds']}s walked / "
          f"{report['rollout_active_seconds']}s with an episode open)")
    # Built before the f-string, not inside it: a nested multi-line string in
    # an f-string expression is a 3.12 feature, and the evaluation environment
    # is 3.11, where this file would not even import.
    span_note = ("" if report["wall_span_exact"] else
                 "  [UPPER BOUND: some episodes have no open stamp, so the "
                 "span covers only the closing burst]")
    print(f"end-to-end          {report['sim_seconds_per_wall_second']} "
          f"sim-sec per wall-sec over the whole window{span_note}  "
          f"  [idle {report['idle_share_of_wall']} of it -- the optimizer]")
    print(f"queueing            {report['busy_waits']} waits, "
          f"{report['busy_wait_seconds']}s "
          f"({report['busy_share_of_wall']} of wall)")
    print(f"clock inflation     {report['clock_inflation']}x  "
          f"(engine {report['engine_seconds_arrived']}s vs graph "
          f"{report['graph_seconds_arrived']}s on arrived hops)")
    arrived_pose = report["pose_error_cm"]["arrived"]
    failed_pose = report["pose_error_cm"]["failed"]
    breach = report["arrivals_outside_contract"]
    print(f"pose error cm       arrived p50 {arrived_pose['p50']} "
          f"max {arrived_pose['max']}"
          f"{'' if not breach else f'  ** {breach} OUTSIDE arrive_cm **'}")
    if failed_pose["n"]:
        print(f"                    failed  p50 {failed_pose['p50']} "
              f"max {failed_pose['max']}  "
              f"(stalled where it stopped, by design; respawned after)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
