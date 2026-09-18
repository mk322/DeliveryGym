#!/usr/bin/env python3
"""Collect any-point delivery reports into one results file, one group per
harness version or model, in one results document.

    python tools/pixel_goal_results.py --out results.json \\
        --group four_view_heldout=results/anypoint/heldout \\
        --describe four_view_heldout="Qwen3-VL-4B, the four-view harness" \\
        --split heldout

A group is a directory of ``seed-N/delivery_report.json`` runs, the shape
``tools/run_pixel_goal_protocol.sh`` leaves behind. Every row carries the
scenario the run actually resolved (spawn, pickup, drop-off, scenario id),
the harness the report declares (views, routing, door tolerance) and the
``CourierEnv`` summary, so a number is never read against the wrong rules.
``--split`` checks that each group's seeds are exactly the named split of
the comparison protocol -- a results file for the held-out set cannot be
built from a run that skipped a seed or added one.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.pixel_goal_order_pool import PROTOCOL_SPLITS, protocol_seeds  # noqa: E402

PROTOCOL_FLAGS = (
    "--route-profile validated_pool --order-mode random --min-delivery-m 30 "
    "--max-delivery-m 80 --min-route-turns 1 --require-marked-crossing"
)
SEED_DIR = re.compile(r"seed-\d+")


def _label_value(text: str) -> tuple[str, str]:
    label, sep, value = text.partition("=")
    if not sep or not label or not value:
        raise argparse.ArgumentTypeError(f"expected label=value, got {text!r}")
    return label, value


def episode_row(label: str, report_path: Path) -> dict[str, Any]:
    """One results row from one ``delivery_report.json``."""
    report = json.loads(report_path.read_text())
    summary = report["summary"]
    scenario = report.get("scenario") or {}
    seed_text = report_path.parent.name.split("-", 1)[1]
    scenario_seed = scenario.get("seed")
    if scenario_seed is not None and int(scenario_seed) != int(seed_text):
        raise ValueError(
            f"{report_path}: directory says seed {seed_text}, the report's "
            f"scenario says seed {scenario_seed}")
    return {
        "group": label,
        "seed": int(seed_text),
        "model": report.get("model"),
        "scenario_id": scenario.get("scenario_id"),
        "spawn": (scenario.get("spawn") or {}).get("id"),
        "pickup": (scenario.get("pickup") or {}).get("id"),
        "dropoff": (scenario.get("dropoff") or {}).get("id"),
        "pixel_views": report.get("pixel_views"),
        "pedestrian_routing": report.get("pedestrian_routing"),
        "dropoff_tolerance_cm": report.get("dropoff_tolerance_cm"),
        "delivered": summary["delivered"],
        "earnings": summary["earnings"],
        "late": summary.get("late"),
        "walked_m": summary.get("walked_m"),
        "turns": report.get("turns"),
        "rejected_actions": summary.get("rejected_actions"),
        "sim_seconds": summary.get("sim_seconds"),
        "termination": report.get("termination"),
    }


def group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The group's number: runs, deliveries, dollars per run, and the means."""
    return {
        "runs": len(rows),
        "delivered": sum(int(row["delivered"]) for row in rows),
        "earnings_total": round(sum(float(row["earnings"]) for row in rows), 2),
        "earnings_mean": round(statistics.mean(float(row["earnings"]) for row in rows), 3),
        "late": sum(int(bool(row.get("late"))) for row in rows),
        "mean_walked_m": round(statistics.mean(float(row["walked_m"]) for row in rows), 1),
        "mean_turns": round(statistics.mean(float(row["turns"]) for row in rows), 1),
        "mean_rejected_actions": round(
            statistics.mean(float(row["rejected_actions"]) for row in rows), 1),
        "seeds": sorted(int(row["seed"]) for row in rows),
    }


def collect(
    groups: list[tuple[str, Path]],
    *,
    split: str | None = None,
    describe: dict[str, str] | None = None,
    date: str | None = None,
    pool: str = "configs/pixel_goal/paris_trusted_pedestrian_pool_v3.json",
) -> dict[str, Any]:
    """The results document for the groups; ``split`` pins each group's seeds."""
    rows: list[dict[str, Any]] = []
    for label, directory in groups:
        # only a run directory named exactly seed-N counts; an attempt set
        # aside as seed-N.<why>-<when> beside it is kept for the record and
        # is not a row
        reports = sorted((p for p in directory.glob("seed-*/delivery_report.json")
                          if SEED_DIR.fullmatch(p.parent.name)),
                         key=lambda p: int(p.parent.name.split("-", 1)[1]))
        if not reports:
            raise ValueError(f"{directory}: no seed-N/delivery_report.json under it")
        group_rows = [episode_row(label, path) for path in reports]
        if split is not None:
            expected = sorted(protocol_seeds(split))
            found = sorted(row["seed"] for row in group_rows)
            if found != expected:
                raise ValueError(
                    f"group {label!r} has seeds {found}, the {split!r} split is "
                    f"{expected}")
        rows.extend(group_rows)
    labels = list(dict.fromkeys(row["group"] for row in rows))
    return {
        "date": date or _dt.date.today().isoformat(),
        "pool": pool,
        "protocol": PROTOCOL_FLAGS,
        "runner": "tools/run_pixel_goal_front_rear_delivery.sh",
        "splits": {name: list(seeds) for name, seeds in PROTOCOL_SPLITS.items()},
        "split": split,
        "groups": {label: (describe or {}).get(label, "") for label in labels},
        "summary": {
            label: group_summary([row for row in rows if row["group"] == label])
            for label in labels
        },
        "episodes": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--group", action="append", required=True, type=_label_value,
                        metavar="LABEL=DIR", help="a directory of seed-N runs")
    parser.add_argument("--describe", action="append", default=[], type=_label_value,
                        metavar="LABEL=TEXT", help="what the group is")
    parser.add_argument("--split", choices=sorted(PROTOCOL_SPLITS),
                        help="require every group to hold exactly this split's seeds")
    parser.add_argument("--date", help="ISO date to stamp (default: today)")
    args = parser.parse_args(argv)
    doc = collect(
        [(label, Path(directory)) for label, directory in args.group],
        split=args.split, describe=dict(args.describe), date=args.date)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps(doc["summary"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
