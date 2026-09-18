"""
benchmark.py — trajectory metrics for DeliveryBench rollouts.

Reads a rollout run directory (the log.jsonl written by rollout_qwen.py) and
reports, as a percentage over trajectories (rollouts):

  1. called_navigation : trajectories that successfully called the NAVIGATE tool
  2. arrived_pickup     : trajectories that successfully completed a PICKUP
                          (a PICKUP can only succeed at the pickup dock, so this
                          is the "reached the pickup point" signal)
  3. delivered          : trajectories with >= 1 completed delivery

Each metric is a per-trajectory boolean (did it happen at least once during the
rollout), then averaged across trajectories.

Usage:
    # latest run under vagen/envs/deliverybench/outputs/
    python -m vagen.envs.deliverybench.benchmark

    # explicit run dir
    python -m vagen.envs.deliverybench.benchmark <run_dir>

Programmatic:
    from vagen.envs.deliverybench.benchmark import compute_metrics
    m = compute_metrics("vagen/envs/deliverybench/outputs/rollout_XXXX")
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List

_PKG_OUTPUTS = Path(__file__).parent / "outputs"
_ACTION_NAME_RE = re.compile(r"^\s*([A-Za-z_]+)\s*\(")


# ──────────────────────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────────────────────

def _latest_run_dir() -> Path:
    runs = sorted(
        (p for p in _PKG_OUTPUTS.glob("rollout_*") if p.is_dir()),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    if not runs:
        raise FileNotFoundError(f"no rollout_* dirs under {_PKG_OUTPUTS}")
    return runs[0]


def load_run(run_dir: Path) -> Dict[int, Dict[str, Any]]:
    """Group a run's log.jsonl into {rollout_id: {steps, end, seed}}."""
    log_path = run_dir / "log.jsonl"
    if not log_path.exists():
        raise FileNotFoundError(log_path)
    traj: Dict[int, Dict[str, Any]] = defaultdict(lambda: {"steps": [], "end": {}, "seed": None})
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        ev, rid = rec.get("event"), rec.get("rollout_id")
        if rid is None:
            continue
        if ev == "step":
            traj[rid]["steps"].append(rec)
        elif ev == "rollout_end":
            traj[rid]["end"] = rec
        elif ev == "rollout_start":
            traj[rid]["seed"] = rec.get("seed")
    return dict(traj)


# ──────────────────────────────────────────────────────────────────────────────
# Per-trajectory checks
# ──────────────────────────────────────────────────────────────────────────────

def _action_name(action: str) -> str:
    m = _ACTION_NAME_RE.match(action or "")
    return m.group(1).upper() if m else ""


def _step_ok(rec: Dict[str, Any]) -> bool:
    """A step whose action parsed and executed without an env error."""
    return bool(rec.get("action_valid")) and not rec.get("action_error")


def _did_action(traj: Dict[str, Any], name: str) -> bool:
    return any(_action_name(s.get("action")) == name and _step_ok(s) for s in traj["steps"])


def called_navigation(traj: Dict[str, Any]) -> bool:
    return _did_action(traj, "NAVIGATE")


def arrived_pickup(traj: Dict[str, Any]) -> bool:
    return _did_action(traj, "PICKUP")


def delivered(traj: Dict[str, Any]) -> bool:
    end = traj.get("end") or {}
    if end.get("deliveries") is not None:
        return int(end["deliveries"]) > 0
    return any(int(s.get("deliveries") or 0) > 0 for s in traj["steps"])


# name -> per-trajectory predicate
CHECKS: Dict[str, Callable[[Dict[str, Any]], bool]] = {
    "called_navigation": called_navigation,
    "arrived_pickup": arrived_pickup,
    "delivered": delivered,
}


# ──────────────────────────────────────────────────────────────────────────────
# Aggregation
# ──────────────────────────────────────────────────────────────────────────────

def _selected_checks(metric_names: List[str] | None) -> Dict[str, Callable]:
    """Resolve a list of metric names to their check functions (all if None)."""
    if not metric_names:
        return dict(CHECKS)
    unknown = [m for m in metric_names if m not in CHECKS]
    if unknown:
        raise ValueError(f"unknown metrics {unknown}; available: {sorted(CHECKS)}")
    return {m: CHECKS[m] for m in metric_names}


def compute_metrics(run_dir: str | Path, metric_names: List[str] | None = None) -> Dict[str, Any]:
    """Compute per-trajectory metric percentages for a run directory.

    metric_names: subset of CHECKS to compute (default: all).
    """
    run_dir = Path(run_dir)
    checks = _selected_checks(metric_names)
    traj = load_run(run_dir)
    n = len(traj)
    counts = {name: sum(1 for t in traj.values() if fn(t)) for name, fn in checks.items()}
    pct = {name: (100.0 * c / n if n else 0.0) for name, c in counts.items()}
    # per-trajectory breakdown (handy for debugging which seeds passed)
    per_traj = {
        rid: {name: fn(t) for name, fn in checks.items()}
        for rid, t in sorted(traj.items())
    }
    return {
        "run_dir": str(run_dir),
        "n_trajectories": n,
        "metrics": list(checks.keys()),
        "counts": counts,
        "pct": pct,
        "per_trajectory": per_traj,
    }


_LABELS = {
    "called_navigation": "called NAVIGATE",
    "arrived_pickup": "arrived pickup",
    "delivered": "delivered",
}


def format_report(metrics: Dict[str, Any]) -> str:
    n = metrics["n_trajectories"]
    lines = [
        f"n_trajectories: {n}",
        f"  {'metric':<20}{'count':>8}{'percent':>10}",
        "  " + "-" * 38,
    ]
    for name in metrics["counts"]:
        c = metrics["counts"][name]
        p = metrics["pct"][name]
        lines.append(f"  {_LABELS.get(name, name):<20}{c:>8}{p:>9.1f}%")
    return "\n".join(lines)


def report(run_dir: str | Path | None = None) -> Dict[str, Any]:
    rd = Path(run_dir) if run_dir else _latest_run_dir()
    metrics = compute_metrics(rd)
    print(format_report(metrics))
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="DeliveryBench rollout trajectory metrics")
    ap.add_argument("run_dir", nargs="?", default=None,
                    help="rollout run dir (default: latest under outputs/)")
    args = ap.parse_args()
    report(args.run_dir)


if __name__ == "__main__":
    main()
