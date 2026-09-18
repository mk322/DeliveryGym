"""The statistics a benchmark number is not allowed to ship without.

Two operations, both deliberately boring:

* a bootstrap confidence interval on a mean, because 64 episodes of a
  high-variance task put a standard error of several points on the mean, and
  a leaderboard number without an interval invites reading noise as ranking;
* a *paired* comparison of two runs over the same seeds, because the episodes
  are paired by construction and the paired test looks only at the seeds that
  changed -- several times more sensitive than comparing two means, at no
  extra compute. Comparing unpaired means on this benchmark once made a real
  5-point gain and pure noise indistinguishable.

Everything is seeded, so the same two result files produce the same interval
every time -- a report that changes between runs of the *analysis* is its own
kind of bug.
"""

from __future__ import annotations

import random
from typing import Sequence

BOOTSTRAP_N = 10_000
_SEED = 20260821  # fixed: the analysis must be as reproducible as the data


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def bootstrap_ci(xs: Sequence[float], level: float = 0.95,
                 n: int = BOOTSTRAP_N) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of ``xs``."""
    if len(xs) < 2:
        return (mean(xs), mean(xs))
    rng = random.Random(_SEED)
    k = len(xs)
    means = sorted(
        sum(rng.choices(xs, k=k)) / k for _ in range(n))
    lo = means[int((1 - level) / 2 * n)]
    hi = means[int((1 + level) / 2 * n) - 1]
    return (lo, hi)


def paired_compare(a: dict[int, float], b: dict[int, float],
                   level: float = 0.95) -> dict:
    """Compare two runs episode-by-episode over their shared seeds.

    ``a`` and ``b`` map seed -> score. Only seeds present in both are used,
    and the count of dropped seeds is reported rather than hidden -- a
    comparison that silently used 40 of 64 seeds is a different comparison.
    """
    shared = sorted(set(a) & set(b))
    diffs = [b[s] - a[s] for s in shared]
    if not diffs:
        return {"n": 0, "note": "no shared seeds"}

    rng = random.Random(_SEED)
    k = len(diffs)
    boot = sorted(sum(rng.choices(diffs, k=k)) / k for _ in range(BOOTSTRAP_N))
    lo = boot[int((1 - level) / 2 * BOOTSTRAP_N)]
    hi = boot[int((1 + level) / 2 * BOOTSTRAP_N) - 1]

    # Sign test p-value (two-sided, exact binomial), scipy-free on purpose:
    # the analysis must run in the leanest environment anyone evaluates from.
    wins = sum(1 for d in diffs if d > 0)
    losses = sum(1 for d in diffs if d < 0)
    m = wins + losses
    if m:
        from math import comb
        tail = sum(comb(m, i) for i in range(0, min(wins, losses) + 1))
        p_sign = min(1.0, 2.0 * tail / (2 ** m))
    else:
        p_sign = 1.0

    return {
        "n": len(shared),
        "dropped_a_only": len(set(a) - set(b)),
        "dropped_b_only": len(set(b) - set(a)),
        "mean_a": round(mean([a[s] for s in shared]), 4),
        "mean_b": round(mean([b[s] for s in shared]), 4),
        "mean_diff_b_minus_a": round(mean(diffs), 4),
        "diff_ci": (round(lo, 4), round(hi, 4)),
        "ci_excludes_zero": bool(lo > 0 or hi < 0),
        "wins_b": wins, "losses_b": losses, "ties": len(diffs) - m,
        "p_sign_test": round(p_sign, 6),
    }
