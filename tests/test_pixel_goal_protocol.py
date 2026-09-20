"""The any-point comparison protocol's split and its results file.

The split is the fairness rule of the track: a number is a number on the
held-out scenarios, and those are fixed in ``tools/pixel_goal_order_pool.py``
beside the resolver that draws them. These tests pin the rule to the pool
that ships: the protocol seeds draw each admissible scenario once, the dev
and held-out seeds partition them, and the results tool refuses a group that
is not exactly the split it claims to be.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import pixel_goal_results as results
from tools.pixel_goal_order_pool import (
    PROTOCOL_CONSTRAINTS,
    PROTOCOL_SEEDS,
    PROTOCOL_SPLITS,
    candidate_scenarios,
    load_validated_delivery_pool,
    protocol_seeds,
    resolve_delivery_scenario,
    scenario_identity,
)

POOL_PATH = Path(__file__).resolve().parent.parent / "configs" / "pixel_goal" / \
    "paris_trusted_pedestrian_pool_v3.json"


@pytest.fixture(scope="module")
def pool():
    return load_validated_delivery_pool(POOL_PATH)


class TestSplit:
    def test_dev_and_heldout_partition_the_protocol(self):
        dev, heldout = protocol_seeds("dev"), protocol_seeds("heldout")
        assert not set(dev) & set(heldout)
        assert sorted(dev + heldout) == sorted(PROTOCOL_SEEDS)
        assert protocol_seeds("protocol") == PROTOCOL_SEEDS
        assert len(set(PROTOCOL_SEEDS)) == len(PROTOCOL_SEEDS)

    def test_unknown_split_is_refused_by_name(self):
        with pytest.raises(ValueError, match="unknown protocol split 'test'"):
            protocol_seeds("test")

    def test_protocol_seeds_draw_every_admissible_scenario_once(self, pool):
        candidates = candidate_scenarios(pool, constraints=PROTOCOL_CONSTRAINTS)
        assert len(candidates) == 16
        drawn = {}
        for seed in PROTOCOL_SEEDS:
            scenario = resolve_delivery_scenario(
                pool, mode="random", seed=seed, constraints=PROTOCOL_CONSTRAINTS)
            assert scenario.candidate_count == len(candidates)
            drawn[seed] = scenario.scenario_id
        assert len(set(drawn.values())) == len(candidates)
        expected = {
            scenario_identity(
                pool, spawn=spawn, pickup=pickup, dropoff=dropoff,
                constraints=PROTOCOL_CONSTRAINTS)
            for spawn, pickup, dropoff, _approach, _delivery in candidates
        }
        assert set(drawn.values()) == expected

    def test_protocol_seeds_are_the_smallest_that_do(self, pool):
        # No smaller seed draws a scenario that a listed seed already draws
        # -- the list is a property of the pool, not a choice.
        seen: dict[str, int] = {}
        for seed in range(max(PROTOCOL_SEEDS) + 1):
            scenario = resolve_delivery_scenario(
                pool, mode="random", seed=seed, constraints=PROTOCOL_CONSTRAINTS)
            seen.setdefault(scenario.scenario_id, seed)
        assert sorted(seen.values()) == sorted(PROTOCOL_SEEDS)

    def test_the_split_holds_out_one_order_and_no_door(self, pool):
        def order(seed):
            scenario = resolve_delivery_scenario(
                pool, mode="random", seed=seed, constraints=PROTOCOL_CONSTRAINTS)
            return scenario.pickup.number, scenario.dropoff.number

        dev_orders = {order(seed) for seed in protocol_seeds("dev")}
        heldout_orders = {order(seed) for seed in protocol_seeds("heldout")}
        assert dev_orders == {(14, 11), (11, 14), (11, 16)}
        assert heldout_orders == {(14, 11), (11, 14), (11, 16), (16, 11)}
        assert heldout_orders - dev_orders == {(16, 11)}
        # every door in the held-out set is a door of a dev scenario: the
        # pool's four doors cannot hold one out, and the docs say so
        doors = lambda orders: {door for pair in orders for door in pair}
        assert doors(heldout_orders) <= doors(dev_orders)


def _report(directory: Path, seed: int, **summary) -> None:
    run = directory / f"seed-{seed}"
    run.mkdir(parents=True)
    body = {
        "model": "qwen3-vl-4b",
        "turns": summary.pop("turns", 10),
        "termination": summary.pop("termination", "delivered"),
        "pixel_views": ["front", "left", "right", "rear"],
        "pedestrian_routing": True,
        "dropoff_tolerance_cm": 300.0,
        "scenario": {
            "seed": seed, "scenario_id": f"scenario-{seed:016x}",
            "spawn": {"id": "spawn-a"}, "pickup": {"id": "stop-b"},
            "dropoff": {"id": "stop-c"},
        },
        "summary": {
            "delivered": 0, "earnings": 0.0, "late": 0, "walked_m": 12.0,
            "rejected_actions": 3, "sim_seconds": 100.0, **summary,
        },
    }
    (run / "delivery_report.json").write_text(json.dumps(body))


class TestResultsFile:
    def test_a_group_is_summed_and_every_row_names_its_scenario(self, tmp_path):
        runs = tmp_path / "runs"
        _report(runs, 3, delivered=1, earnings=3.58, walked_m=120.0, turns=20)
        _report(runs, 5)
        doc = results.collect([("four_view", runs)], describe={"four_view": "the harness"},
                              date="2026-09-12")
        assert doc["groups"] == {"four_view": "the harness"}
        assert doc["splits"]["heldout"] == list(PROTOCOL_SPLITS["heldout"])
        summary = doc["summary"]["four_view"]
        assert summary == {
            "runs": 2, "delivered": 1, "earnings_total": 3.58, "earnings_mean": 1.79,
            "late": 0, "mean_walked_m": 66.0, "mean_turns": 15.0,
            "mean_rejected_actions": 3.0, "seeds": [3, 5],
        }
        row = doc["episodes"][0]
        assert row["seed"] == 3 and row["scenario_id"] == "scenario-0000000000000003"
        assert row["pixel_views"] == ["front", "left", "right", "rear"]
        assert row["pedestrian_routing"] is True
        assert row["dropoff_tolerance_cm"] == 300.0
        assert row["termination"] == "delivered"

    def test_a_split_must_be_complete_and_nothing_more(self, tmp_path):
        runs = tmp_path / "runs"
        for seed in PROTOCOL_SPLITS["heldout"][:-1]:
            _report(runs, seed)
        with pytest.raises(ValueError, match="the 'heldout' split is"):
            results.collect([("g", runs)], split="heldout")
        _report(runs, PROTOCOL_SPLITS["heldout"][-1])
        assert results.collect([("g", runs)], split="heldout")["summary"]["g"]["runs"] == 10
        _report(runs, 0)
        with pytest.raises(ValueError, match="has seeds"):
            results.collect([("g", runs)], split="heldout")

    def test_an_attempt_set_aside_beside_a_run_is_not_a_row(self, tmp_path):
        runs = tmp_path / "runs"
        _report(runs, 3, delivered=1, earnings=3.58)
        # a stuck first attempt kept for the record as seed-3.<why>-<when>
        aside = runs / "seed-3.stale-read-1013"
        aside.mkdir()
        (aside / "delivery_report.json").write_text((runs / "seed-3" / "delivery_report.json").read_text())
        _report(runs, 5)
        doc = results.collect([("g", runs)])
        assert doc["summary"]["g"]["runs"] == 2 and doc["summary"]["g"]["seeds"] == [3, 5]

    def test_a_report_whose_seed_disagrees_with_its_directory_is_refused(self, tmp_path):
        runs = tmp_path / "runs"
        _report(runs, 3)
        path = runs / "seed-3" / "delivery_report.json"
        body = json.loads(path.read_text())
        body["scenario"]["seed"] = 4
        path.write_text(json.dumps(body))
        with pytest.raises(ValueError, match="directory says seed 3"):
            results.collect([("g", runs)])

    def test_the_command_line_writes_the_file(self, tmp_path):
        runs = tmp_path / "runs"
        for seed in PROTOCOL_SPLITS["dev"]:
            _report(runs, seed)
        out = tmp_path / "out" / "results.json"
        assert results.main([
            "--out", str(out), "--group", f"dev_group={runs}", "--split", "dev",
            "--describe", "dev_group=six development scenarios", "--date", "2026-09-12",
        ]) == 0
        doc = json.loads(out.read_text())
        assert doc["split"] == "dev"
        assert doc["summary"]["dev_group"]["seeds"] == sorted(PROTOCOL_SPLITS["dev"])
        assert doc["date"] == "2026-09-12"
