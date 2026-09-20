"""The waypoint reference policies: ``tools/waypoint_reference.py`` writes
the evaluator's results shape for the upper bound and the floor, both run
to the benchmark's turn budget, so a model's run pairs with either seed by
seed."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from embodiedbench.compiler.road_network import build_road_network
from embodiedbench.eval import stats
from embodiedbench.eval.run import EARNINGS_CHECKPOINTS
from embodiedbench.runtime.city.courier_env import CourierEnv
from embodiedbench.tasks.courier_oracle import ObservationOnlyCourier
from embodiedbench.tasks.courier_router import FewestMovesCourier, ShortestPathCourier
from tools import waypoint_reference as reference

MAPS = Path(__file__).resolve().parent.parent / "vendor" / "vagen" / "vagen" / "envs" / \
    "deliverybench" / "maps" / "citycore-paris"


def _row(seed: int, earnings: float, delivered: int) -> dict:
    return {
        "seed": seed, "earnings": earnings,
        "earnings_at_20": earnings / 3, "earnings_at_40": 2 * earnings / 3,
        "earnings_at_60": earnings, "delivered": delivered, "on_time": delivered,
        "late": 0, "red_crossings": 0, "turns": 60, "walked_m": 1000.0,
        "sim_seconds": 1500.0, "termination": "out_of_turns", "trace": [],
        "infra_errors": 0,
    }


class TestSummary:
    def test_the_summary_is_the_evaluators_shape(self):
        rows = [_row(1000, 15.0, 3), _row(1001, 12.0, 2), _row(1002, 9.0, 2)]
        task = {"difficulty": "endless", "queue_depth": 1, "max_turns": 60,
                "album_root": "/somewhere"}
        summary = reference.summarise(rows, task=task, split="val", cap=60)
        # what compare reads, and what the evaluator writes
        for key in ("model", "tag", "split", "scored_episodes", "earnings_mean",
                    "earnings_ci95", "earnings_at_mean", "deliveries_mean",
                    "zero_delivery_rate", "scores_by_seed", "crashed_seeds",
                    "infra_error_seeds", "task", "finished_at"):
            assert key in summary, key
        assert summary["model"] == "graph-ceiling"
        assert summary["tag"] == "graph-ceiling-best" and summary["objective"] == "best"
        assert "better per shift" in summary["policy"]
        assert summary["scored_episodes"] == 3
        assert summary["earnings_mean"] == 12.0
        assert summary["earnings_at_mean"] == {
            "earnings_at_20": 4.0, "earnings_at_40": 8.0, "earnings_at_60": 12.0}
        assert summary["deliveries_mean"] == round(7 / 3, 4)
        assert summary["zero_delivery_rate"] == 0.0
        assert summary["scores_by_seed"] == {"1000": 15.0, "1001": 12.0, "1002": 9.0}
        assert summary["crashed_seeds"] == [] and summary["infra_error_seeds"] == []
        assert summary["action_cap"] == 60
        # the albums' root never lands in a results file
        assert "album_root" not in summary["task"]
        assert "not a policy" in summary["policy"]
        assert "FewestMovesCourier" in summary["policy"]
        moves = reference.summarise(rows, task=task, split="val", cap=60, objective="moves")
        assert moves["tag"] == "graph-ceiling-moves" and moves["policy"].startswith("FewestMovesCourier")
        metres = reference.summarise(rows, task=task, split="val", cap=60, objective="metres")
        assert metres["tag"] == "graph-ceiling-metres" and "ShortestPathCourier" in metres["policy"]

    def test_the_floor_is_named_as_what_it_is(self):
        rows = [_row(1000, 3.0, 1)]
        floor = reference.summarise(rows, task={"max_turns": 60}, split="val", cap=60,
                                    objective="floor")
        assert floor["model"] == "text-floor" and floor["tag"] == "text-floor-floor"
        assert floor["policy"].startswith("ObservationOnlyCourier")
        assert "reads only what the world says" in floor["policy"]
        assert "floor" in floor["policy"] and "not a policy" in floor["policy"]


@pytest.fixture(scope="module")
def paris():
    if not MAPS.is_dir():
        pytest.skip("the compiled Paris map is not checked out")
    return build_road_network(MAPS, map_name="citycore-paris")


class TestFewestMoves:
    """Planning in turns never costs more turns than planning in metres, on
    the world's own block rule, without albums (the graph alone)."""

    def _run(self, paris, courier_cls, seed, cap=60):
        env = CourierEnv(paris, seed=seed, difficulty="endless", stride="block")
        env.reset()
        result = courier_cls(env, max_steps=cap).run(seed)
        return env, result

    def test_never_more_turns_per_delivery_than_the_metres_planner(self, paris):
        worse = 0
        for seed in (1000, 1001, 1002, 1003):
            _env_m, metres = self._run(paris, ShortestPathCourier, seed)
            _env_t, turns = self._run(paris, FewestMovesCourier, seed)
            assert turns.delivered >= metres.delivered, seed
            assert turns.earnings >= metres.earnings - 1e-9, seed
            worse += turns.delivered < metres.delivered
        assert worse == 0

    def test_the_plan_is_the_block_rule_not_a_guess(self, paris):
        # Every move the planner commits is a candidate the world offers at
        # the node it stands on, and the walk lands where the plan said.
        env = CourierEnv(paris, seed=1000, difficulty="endless", stride="block")
        env.reset()
        courier = FewestMovesCourier(env, max_steps=60)
        order = env.active_order()
        step = courier._next_step(order.target.kerb_node)
        assert step is not None
        k, toward = step
        assert any(row["k"] == k and row["node"] == toward for row in env.candidates())
        chain = env.block_chain(env.node_id, toward)
        outcome = env.walk_to(*env.street_at(k))
        assert outcome.ok
        assert env.node_id in {b for _a, b in chain}

    def test_the_planner_is_deterministic(self, paris):
        a = self._run(paris, FewestMovesCourier, 1001)[1]
        b = self._run(paris, FewestMovesCourier, 1001)[1]
        assert (a.earnings, a.delivered, a.turns) == (b.earnings, b.delivered, b.turns)

    def test_the_bound_pairs_with_a_model_run_like_compare_does(self):
        bound = reference.summarise([_row(1000, 15.0, 3), _row(1001, 12.0, 2)],
                                    task={"max_turns": 60}, split="val", cap=60)
        model = {"scores_by_seed": {"1000": 6.0, "1001": 7.0, "1002": 1.0}}
        result = stats.paired_compare(
            {int(k): v for k, v in model["scores_by_seed"].items()},
            {int(k): v for k, v in bound["scores_by_seed"].items()})
        assert result["n"] == 2
        assert result["mean_diff_b_minus_a"] == 7.0


class TestTheTurnBudget:
    """A reference courier stops at the world's turn count, not at its own
    step count: for the floor a step is a walk and a phone check, two turns,
    and a floor allowed 60 steps was taking 110-130 turns of the model's 60."""

    def test_the_floor_stops_at_the_turn_budget_and_resumes_without_re_asking(self, paris):
        env = CourierEnv(paris, seed=1000, difficulty="endless", stride="block")
        env.reset()
        courier = ObservationOnlyCourier(env, max_steps=10_000, stop_at_turns=20)
        courier.run(1000)
        # exactly the budget: the check is before every action, so a step
        # that would run over stops between its walk and its phone check
        assert env.turns == 20
        # a resumed run past its budget spends nothing: it does not begin by
        # asking the phone again about the address it is already working on
        courier.run(1000)
        assert env.turns == 20
        courier.stop_at_turns = 40
        courier.run(1000)
        assert env.turns == 40

    def test_the_ceiling_stops_at_the_turn_budget_too(self, paris):
        env = CourierEnv(paris, seed=1000, difficulty="endless", stride="block")
        env.reset()
        FewestMovesCourier(env, max_steps=10_000, stop_at_turns=30).run(1000)
        assert env.turns == 30

    def test_run_shift_gives_every_reference_the_same_turns(self, paris):
        rows = {}
        for objective in ("floor", "moves"):
            env = CourierEnv(paris, seed=1002, difficulty="endless", stride="block")
            env.reset()
            rows[objective] = reference.run_shift(env, 1002, 60, objective=objective)
        for objective, row in rows.items():
            assert row["turns"] == 60, (objective, row["turns"])
            assert row["earnings_at_20"] <= row["earnings_at_40"] <= row["earnings_at_60"] == row["earnings"]
            assert row["termination"] in ("out_of_turns", "shift_over", "finished", "stopped")


class TestTheFloorReadsTheWorldAsItSpeaksNow:
    """The two rules the released floor table was withdrawn for: the door
    numbers on the location line, and the route the phone now draws
    instead of speaking."""

    def test_the_door_numbers_are_read_in_every_form_the_line_takes(self):
        class Speaks:
            def __init__(self, text):
                self.text = text

            def location_text(self):
                return self.text

        def read(text):
            courier = ObservationOnlyCourier.__new__(ObservationOnlyCourier)
            courier.env = Speaks(text)
            return courier.read_here()

        assert read("You are on Rue de Grenelle, outside numbers 28, 30, 35. "
                    "This is the street on the slip; the slip says 54.") == ("Rue de Grenelle", (28, 30, 35))
        assert read("You are on Rue de Grenelle, outside number 3. The numbers fell "
                    "as you walked here.") == ("Rue de Grenelle", (3,))
        assert read("You are on Rue Monge, outside number 6-8.") == ("Rue Monge", (6, 8))
        assert read("You are on Rue Monge, between doors.") == ("Rue Monge", None)

    def test_the_door_is_tried_when_its_number_is_read_not_when_it_is_near_in_number(self, paris):
        """Standing outside 28, 30 and 35 with 32 on the slip is not standing
        at 32: the door is a walk away and the try would be refused (and
        cost a turn). Outside 30 with 30 on the slip, it is tried."""
        def standing(slip_number):
            env = CourierEnv(paris, seed=1000, difficulty="endless", stride="block")
            env.reset()
            tries = []
            env.collect = lambda: tries.append("collect") or type("O", (), {"ok": False, "message": "no"})()
            courier = ObservationOnlyCourier(env, max_steps=1)
            street, numbers = courier.read_here()
            assert numbers is not None
            # the courier already knows the job (no phone call at the start
            # of the run), and the slip says what the test says it says
            courier._target_text = env.active_order().target.text
            courier.target_street, courier.target_number = street, slip_number
            courier.target_distance_m = 100.0
            courier.run(1000)
            return numbers, tries

        numbers, tries = standing(999)
        assert 999 not in numbers and tries == []
        _numbers, tries = standing(numbers[0])
        assert tries == ["collect"]

    def test_the_floor_never_asks_for_the_way(self, paris):
        """Where a place is, it may read (the pin's bearing); the way there,
        never -- tests/test_city_pipeline.py forbids the route calls by name.
        The phone's route is a picture now, so the courier does not call
        navigate() either: nothing it could read would come back."""
        import inspect
        source = inspect.getsource(ObservationOnlyCourier)
        assert "navigate(" not in source and "route_legs" not in source
        env = CourierEnv(paris, seed=1000, difficulty="endless", stride="block")
        env.reset()
        calls = []
        env.navigate = lambda: calls.append("navigate")
        ObservationOnlyCourier(env, max_steps=40).run(1000)
        assert calls == []


def _albums() -> str | None:
    base = os.environ.get("ALBUMS_DIR") or "/data/albums"
    if (Path(base) / "paris_streets_pavement" / "citycore-paris").is_dir():
        return base
    return None


class TestLiveShift:
    def test_one_held_out_shift_runs_at_the_benchmark_horizon(self, tmp_path, monkeypatch):
        base = _albums()
        if base is None:
            pytest.skip("albums not mounted (set ALBUMS_DIR)")
        monkeypatch.setenv("ALBUMS_DIR", base)
        pytest.importorskip("cairosvg")
        out = tmp_path / "ceiling.json"
        assert reference.main(["--seed-list", "1000", "--out", str(out)]) == 0
        summary = json.loads(out.read_text())
        rows = [json.loads(line) for line in
                (tmp_path / "ceiling.episodes.jsonl").read_text().splitlines()]
        assert summary["scored_episodes"] == 1 and rows[0]["seed"] == 1000
        assert summary["action_cap"] == summary["task"]["max_turns"] == 60
        assert rows[0]["turns"] <= 60
        for k in EARNINGS_CHECKPOINTS:
            if k <= 60:
                assert f"earnings_at_{k}" in rows[0]
        # money only ever accumulates along the checkpoints
        assert rows[0]["earnings_at_20"] <= rows[0]["earnings_at_40"] <= rows[0]["earnings_at_60"]
        assert rows[0]["earnings_at_60"] == rows[0]["earnings"]
        assert rows[0]["delivered"] >= 1  # the graph reader always delivers on seed 1000
        assert rows[0]["planner"] in ("moves", "metres")

    def test_best_is_never_below_either_planner(self, tmp_path, monkeypatch):
        base = _albums()
        if base is None:
            pytest.skip("albums not mounted (set ALBUMS_DIR)")
        monkeypatch.setenv("ALBUMS_DIR", base)
        pytest.importorskip("cairosvg")
        scores = {}
        for objective in ("best", "moves", "metres"):
            out = tmp_path / f"{objective}.json"
            assert reference.main(["--seed-list", "1005", "--objective", objective, "--out", str(out)]) == 0
            scores[objective] = json.loads(out.read_text())["scores_by_seed"]["1005"]
        assert scores["best"] == max(scores["moves"], scores["metres"])

    def test_the_floor_runs_at_the_benchmark_horizon(self, tmp_path, monkeypatch):
        base = _albums()
        if base is None:
            pytest.skip("albums not mounted (set ALBUMS_DIR)")
        monkeypatch.setenv("ALBUMS_DIR", base)
        pytest.importorskip("cairosvg")
        out = tmp_path / "floor.json"
        assert reference.main(["--seed-list", "1000", "--objective", "floor", "--out", str(out)]) == 0
        summary = json.loads(out.read_text())
        rows = [json.loads(line) for line in
                (tmp_path / "floor.episodes.jsonl").read_text().splitlines()]
        assert summary["model"] == "text-floor" and summary["action_cap"] == 60
        assert rows[0]["turns"] == 60 and rows[0]["planner"] == "floor"
        assert rows[0]["earnings_at_60"] == rows[0]["earnings"]
