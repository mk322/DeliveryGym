"""The evaluator's load-bearing pieces: the stats, the task source, the seeds."""

from __future__ import annotations

import pytest

from embodiedbench.eval import stats
from embodiedbench.eval.run import SPLITS, load_task_config


class TestStats:
    def test_ci_brackets_the_mean_and_is_deterministic(self):
        xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        lo, hi = stats.bootstrap_ci(xs)
        assert lo < stats.mean(xs) < hi
        assert (lo, hi) == stats.bootstrap_ci(xs)  # seeded

    def test_paired_compare_detects_a_real_shift(self):
        a = {s: float(s % 7) for s in range(64)}
        b = {s: v + 1.0 for s, v in a.items()}   # uniformly one point better
        r = stats.paired_compare(a, b)
        assert r["n"] == 64
        assert r["mean_diff_b_minus_a"] == pytest.approx(1.0)
        assert r["ci_excludes_zero"] and r["p_sign_test"] < 1e-6

    def test_paired_compare_calls_noise_noise(self):
        import random
        rng = random.Random(3)
        a = {s: rng.gauss(5, 2) for s in range(64)}
        b = {s: v + rng.gauss(0, 0.01) for s, v in a.items()}
        r = stats.paired_compare(a, b)
        assert not r["ci_excludes_zero"]

    def test_disjoint_seed_sets_are_reported(self):
        r = stats.paired_compare({1: 1.0, 2: 2.0}, {2: 2.0, 3: 3.0})
        assert r["n"] == 1 and r["dropped_a_only"] == 1 and r["dropped_b_only"] == 1


class TestProtocol:
    def test_val_split_is_the_immutable_64(self):
        assert list(SPLITS["val"]) == list(range(1000, 1064))

    def test_trainprobe_matches_the_yaml_probe(self):
        assert list(SPLITS["trainprobe"]) == list(range(0, 32))

    def test_task_config_comes_from_the_val_yaml(self):
        task = load_task_config()
        # The four facts every reported number depends on.
        assert task["difficulty"] == "endless"
        assert task["queue_depth"] == 1
        # 60 turns, the training horizon, since 2026-09-13; money is
        # photographed at 20/40/60 along the way.
        assert task["max_turns"] == 60
        assert task["reward_basis"] == "earnings"
        assert float(task.get("progress_weight", 0)) == 0.0  # never shaped
