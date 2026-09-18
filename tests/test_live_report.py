"""The throughput number has to be honest, including about not knowing.

Both cases here are mistakes that were actually made and caught by reading
output rather than by a test, which is why they are now tests.
"""

from __future__ import annotations

import json
from pathlib import Path

from embodiedbench.runtime.live.report import load, summarize


def _episode(pid: int, closed_at: float, wall_seconds: float | None,
             hops: list[dict]) -> dict:
    outcomes: dict[str, int] = {}
    for hop in hops:
        outcomes[hop["outcome"]] = outcomes.get(hop["outcome"], 0) + 1
    record = {
        "schema": "nav-live-telemetry/v1",
        "pid": pid,
        "closed_at": closed_at,
        "counters": {"hops": len(hops), "outcomes": outcomes,
                     "recoveries": 0, "degraded": False,
                     "busy_waits": 0, "busy_wait_seconds": 0.0},
        "hops": hops,
        "summary": {},
    }
    if wall_seconds is not None:
        record["wall_seconds"] = wall_seconds
    return record


def _hop(outcome: str = "arrived", sim: float = 100.0,
         graph: float = 100.0) -> dict:
    return {"outcome": outcome, "sim_seconds": sim, "graph_seconds": graph,
            "ticks": 1, "pose_error_cm": 10.0}


def test_throughput_spans_open_to_close_not_the_closing_burst():
    """Six concurrent episodes, each a hundred seconds, all finishing together.

    Measuring the range of closing times says five seconds and reports 120x.
    The fleet was alive for a hundred and five.
    """
    records = [_episode(1, 1000.0 + i, 100.0, [_hop()]) for i in range(6)]
    report = summarize(records)
    assert report["wall_span_exact"] is True
    assert report["wall_span_seconds"] == 105.0
    assert report["sim_seconds_per_wall_second"] == round(600 / 105, 3)


def test_a_span_without_open_stamps_is_flagged_not_quietly_wrong():
    """Older records have no open time. The number still prints -- labelled."""
    records = [_episode(1, 1000.0 + i, None, [_hop()]) for i in range(6)]
    report = summarize(records)
    assert report["wall_span_exact"] is False
    assert report["wall_span_seconds"] == 5.0


def test_refused_hops_stay_out_of_the_clock_ratio():
    """A refusal is charged a floor, not a travel price.

    Including one compares two different things and flatters the ratio that
    exists to expose engine-priced time spent against graph-priced budgets.
    """
    records = [_episode(1, 1000.0, 10.0, [
        _hop("arrived", sim=120.0, graph=100.0),
        _hop("stuck", sim=5.0, graph=100.0),
    ])]
    report = summarize(records)
    assert report["clock_inflation"] == 1.2
    assert report["sim_failure_rate"] == 0.5


def test_a_torn_final_line_costs_one_episode_not_the_report(tmp_path: Path):
    """The interesting moment to ask is mid-run, while a writer is appending."""
    path = tmp_path / "episodes-1.jsonl"
    good = _episode(1, 1000.0, 10.0, [_hop()])
    path.write_text(json.dumps(good) + "\n" + '{"pid": 2, "coun')
    report = summarize(list(load(tmp_path)))
    assert report["episodes"] == 1


def test_the_two_throughput_numbers_separate_walking_from_optimising():
    """One figure would change meaning as the directory fills up.

    Two rollout bursts of ten seconds each, a hundred seconds apart: the
    fleet walked for twenty seconds of wall clock and the run took a hundred
    and ten. Reporting only the first makes a training step look four times
    cheaper than it is; reporting only the second makes the engine look four
    times slower than it is. Measured on the development workstation, the same run read 6.25x at
    eight episodes and 3.45x at twelve with nothing having got slower.
    """
    records = [
        _episode(1, 1010.0, 10.0, [_hop(sim=40.0)]),
        _episode(2, 1010.0, 10.0, [_hop(sim=40.0)]),   # concurrent with it
        _episode(1, 1110.0, 10.0, [_hop(sim=40.0)]),
        _episode(2, 1110.0, 10.0, [_hop(sim=40.0)]),
    ]
    report = summarize(records)
    assert report["wall_span_seconds"] == 110.0
    # Two overlapping pairs, ten seconds each -- not forty.
    assert report["rollout_active_seconds"] == 20.0
    assert report["sim_seconds_per_active_second"] == round(160 / 20, 3)
    assert report["sim_seconds_per_wall_second"] == round(160 / 110, 3)
    assert report["idle_share_of_wall"] == round(1 - 20 / 110, 4)


def test_pose_error_is_split_by_outcome_so_a_stall_is_not_a_breach():
    """Mixing them turns a clean contract into a false alarm.

    An arrived hop must land inside arrive_cm — a breach there is a real
    defect. A stuck hop leaves the pawn where it stalled, by design, and a
    respawn follows. Measured on the development workstation: 170 arrived hops maxed at exactly
    120.0 cm with zero outside the radius, while four stuck hops sat between
    415 and 1220 cm. Reported together that is "max 12.2 m", which reads like
    the contract broke when it held perfectly.
    """
    record = _episode(1, 1000.0, 10.0, [
        _hop("arrived"), _hop("arrived"), _hop("stuck"),
    ])
    record["config"] = {"arrive_cm": 120.0}
    record["hops"][0]["pose_error_cm"] = 83.0
    record["hops"][1]["pose_error_cm"] = 120.0
    record["hops"][2]["pose_error_cm"] = 1219.8
    report = summarize([record])
    assert report["pose_error_cm"]["arrived"] == {"n": 2, "p50": 120.0, "max": 120.0}
    assert report["pose_error_cm"]["failed"]["max"] == 1219.8
    assert report["arrivals_outside_contract"] == 0


def test_an_arrival_outside_the_radius_is_counted_as_a_breach():
    """The one case where a big pose error IS the invariant failing."""
    record = _episode(1, 1000.0, 10.0, [_hop("arrived")])
    record["config"] = {"arrive_cm": 120.0}
    record["hops"][0]["pose_error_cm"] = 400.0
    assert summarize([record])["arrivals_outside_contract"] == 1
