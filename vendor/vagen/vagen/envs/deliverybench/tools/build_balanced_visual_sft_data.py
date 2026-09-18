"""Build balanced multi-city visual DeliveryBench SFT datasets.

The generator drives the real visual-route delivery workflow with oracle
actions, then samples the exported policy-visible decision rows. MOVE rows are
balanced across primitive directions, while non-MOVE workflow rows are kept so
the SFT target remains the rollout-style JSON response rather than action-only
classification.

Example:
    PYTHONPATH=. python -m vagen.envs.deliverybench.tools.build_balanced_visual_sft_data \
        --output-dir outputs/deliverybench_sft/visual_workflow_multicity_balanced
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import dataclasses
import io
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from ..deliverybench_env import DeliveryBenchEnvConfig, VISUAL_ROUTE_FOLLOWING_CONFIG
from .generate_visual_sft_data import _run_oracle_episode


TRAIN_MAPS = (
    "small-city-11",
    "small-city-13",
    "medium-city-18",
    "medium-city-20",
    "large-city-26",
    "large-city-28",
)
EVAL_MAPS = ("small-city-15", "medium-city-22", "large-city-30")
MOVE_DIRECTIONS = ("forward", "backward", "left", "right")
WORKFLOW_STAGES = ("view_orders", "accept", "navigate_pickup", "pickup", "navigate_dropoff", "dropoff")
MOVE_ACTION_RE = re.compile(r'^MOVE\(direction=["\'](forward|backward|left|right)["\']\)$')
MOVE_TO_ACTION_RE = re.compile(r"^MOVE_TO\((\d+)\)$")


def parse_csv(text: str) -> List[str]:
    values = [part.strip() for part in str(text or "").split(",") if part.strip()]
    if not values:
        raise ValueError("expected at least one comma-separated value")
    return values


def move_direction(action: str) -> Optional[str]:
    match = MOVE_ACTION_RE.match(str(action or ""))
    return match.group(1) if match else None


def row_move_bucket(row: Mapping[str, Any]) -> Optional[str]:
    """Balancing bucket for a locomotion row.

    MOVE rows bucket by their literal direction. MOVE_TO rows (waypoint-marks
    datasets) bucket by the oracle's ground-truth relative direction
    (gt_rel_direction, recorded at generation time) — the mark index itself is
    position-dependent and meaningless as a class, while the relative
    direction is exactly the skew that needs balancing.
    """
    action = str(row.get("action", ""))
    direction = move_direction(action)
    if direction:
        return direction
    if MOVE_TO_ACTION_RE.match(action):
        rel = str(row.get("gt_rel_direction", "") or "")
        return rel if rel in MOVE_DIRECTIONS else None
    return None


def make_env_config(
    *,
    map_name: str,
    max_steps: int,
    feasible_order_step_budget: int,
    enable_fpv: bool,
    waypoint_marks: bool = False,
    fpv_dir_override: Optional[str] = None,
) -> DeliveryBenchEnvConfig:
    """Create the delivery-like visual SFT config for one city map."""
    cfg = dataclasses.replace(
        VISUAL_ROUTE_FOLLOWING_CONFIG,
        map_name=str(map_name),
        max_steps=int(max_steps),
        enable_fpv=bool(enable_fpv),
        enable_waypoint_marks=bool(waypoint_marks),
        fixed_spawn_position=None,
        max_orders_in_pool=1,
        enable_feasible_orders=True,
        enable_infeasible_orders=False,
        feasible_order_step_budget=int(feasible_order_step_budget),
        feasible_order_non_move_actions=5,
    )
    if fpv_dir_override:
        cfg = dataclasses.replace(cfg, fpv_dir=str(fpv_dir_override))
    elif waypoint_marks and enable_fpv:
        # The per-map FPV manifest lives inside the dataset subdirectory, not
        # the map root the env defaults to — resolve it so the reset-time
        # album-coverage audit sees the photos. Scoped to marks mode so the
        # classic datagen path stays byte-identical.
        from .fpv_waypoint_marks import default_manifest
        cfg = dataclasses.replace(cfg, fpv_dir=str(default_manifest(map_name).parent))
    return cfg


def parse_fpv_dir_overrides(text: Optional[str]) -> Dict[str, str]:
    """Parse --fpv-dir-overrides "map=path,map=path" into a dict."""
    out: Dict[str, str] = {}
    for part in str(text or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"--fpv-dir-overrides entry needs map=path: {part!r}")
        name, path = part.split("=", 1)
        out[name.strip()] = path.strip()
    return out


async def collect_candidates(
    *,
    split: str,
    maps: Sequence[str],
    output_dir: Path,
    seed_start: int,
    seed_stride: int,
    seeds_per_map: int,
    max_steps: int,
    feasible_order_step_budget: int,
    enable_fpv: bool,
    verbose_env: bool,
    log_every: int,
    waypoint_marks: bool = False,
    fpv_dir_overrides: Optional[Mapping[str, str]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Run oracle episodes and return exported rows plus recoverable failures."""
    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    candidate_root = output_dir / "candidates" / split

    for map_index, map_name in enumerate(maps):
        cfg = make_env_config(
            map_name=map_name,
            max_steps=max_steps,
            feasible_order_step_budget=feasible_order_step_budget,
            enable_fpv=enable_fpv,
            waypoint_marks=waypoint_marks,
            fpv_dir_override=(fpv_dir_overrides or {}).get(map_name),
        )
        image_dir = candidate_root / map_name / "images"
        for offset in range(int(seeds_per_map)):
            seed = int(seed_start) + map_index * int(seed_stride) + offset
            try:
                if verbose_env:
                    episode_rows = await _run_oracle_episode(
                        seed=seed,
                        env_config=cfg,
                        image_dir=image_dir,
                        move_repeat=1,
                        include_non_move=True,
                        waypoint_marks=waypoint_marks,
                    )
                else:
                    with contextlib.redirect_stdout(io.StringIO()):
                        episode_rows = await _run_oracle_episode(
                            seed=seed,
                            env_config=cfg,
                            image_dir=image_dir,
                            move_repeat=1,
                            include_non_move=True,
                            waypoint_marks=waypoint_marks,
                        )
            except Exception as exc:  # noqa: BLE001 - written to manifest for audit.
                failures.append(
                    {
                        "split": split,
                        "map_name": map_name,
                        "seed": seed,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                continue

            for row in episode_rows:
                row["split"] = split
                row["map_name"] = map_name
                row["source_seed"] = seed
                row["image_policy"] = "all"
            rows.extend(episode_rows)

            if log_every > 0 and (offset + 1) % int(log_every) == 0:
                print(
                    f"[{split}] {map_name}: seeds={offset + 1}/{seeds_per_map}, "
                    f"rows={len(rows)}, failures={len(failures)}",
                    flush=True,
                )

    return rows, failures


def _target_counts(total: int, workflow_fraction: float, include_workflow: bool) -> Tuple[Dict[str, int], int]:
    total = int(total)
    if total <= 0:
        raise ValueError("target total must be positive")
    workflow_target = int(round(total * float(workflow_fraction))) if include_workflow else 0
    workflow_target = max(0, min(workflow_target, total))
    move_total = total - workflow_target
    per_direction = move_total // len(MOVE_DIRECTIONS)
    remainder = move_total % len(MOVE_DIRECTIONS)
    move_targets = {direction: per_direction for direction in MOVE_DIRECTIONS}
    for direction in MOVE_DIRECTIONS[:remainder]:
        move_targets[direction] += 1
    return move_targets, workflow_target


def _sample_group(
    rows: Sequence[Dict[str, Any]],
    *,
    count: int,
    rng: random.Random,
    allow_repeat: bool,
) -> List[Dict[str, Any]]:
    rows = list(rows)
    if count <= 0:
        return []
    if len(rows) >= count:
        return [dict(row) for row in rng.sample(rows, count)]
    if not allow_repeat:
        raise RuntimeError(f"need {count} rows, only found {len(rows)}")
    if not rows:
        raise RuntimeError(f"need {count} rows, found none")
    selected = [dict(row) for row in rows]
    while len(selected) < count:
        repeated = dict(rng.choice(rows))
        repeated["selected_repeat"] = int(repeated.get("selected_repeat", 0) or 0) + 1
        selected.append(repeated)
    return selected


def _round_robin_sample(
    groups: Mapping[str, Sequence[Dict[str, Any]]],
    *,
    count: int,
    rng: random.Random,
    allow_repeat: bool,
) -> List[Dict[str, Any]]:
    if count <= 0:
        return []

    shuffled = {key: list(value) for key, value in groups.items()}
    for values in shuffled.values():
        rng.shuffle(values)

    selected: List[Dict[str, Any]] = []
    round_index = 0
    group_keys = [key for key in WORKFLOW_STAGES if shuffled.get(key)] + [
        key for key in shuffled if key not in WORKFLOW_STAGES and shuffled.get(key)
    ]
    while len(selected) < count:
        progressed = False
        for key in group_keys:
            values = shuffled[key]
            if round_index < len(values):
                selected.append(dict(values[round_index]))
                progressed = True
                if len(selected) >= count:
                    break
        if not progressed:
            break
        round_index += 1

    if len(selected) >= count:
        return selected[:count]
    if not allow_repeat:
        raise RuntimeError(f"need {count} workflow rows, only found {len(selected)}")
    if not selected:
        raise RuntimeError(f"need {count} workflow rows, found none")
    while len(selected) < count:
        repeated = dict(rng.choice(selected))
        repeated["selected_repeat"] = int(repeated.get("selected_repeat", 0) or 0) + 1
        selected.append(repeated)
    return selected


def sample_balanced_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    target_total: int,
    workflow_fraction: float,
    include_workflow: bool,
    allow_repeat: bool,
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Select rows with balanced MOVE directions plus rollout workflow rows."""
    rng = random.Random(int(seed))
    move_targets, workflow_target = _target_counts(target_total, workflow_fraction, include_workflow)

    by_direction: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_stage: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        direction = row_move_bucket(row)
        if direction:
            by_direction[direction].append(dict(row))
        elif str(row.get("stage", "")) in WORKFLOW_STAGES:
            by_stage[str(row.get("stage", ""))].append(dict(row))

    selected: List[Dict[str, Any]] = []
    for direction, target in move_targets.items():
        selected.extend(
            _sample_group(
                by_direction.get(direction, []),
                count=target,
                rng=rng,
                allow_repeat=allow_repeat,
            )
        )

    if include_workflow and workflow_target > 0:
        selected.extend(
            _round_robin_sample(
                by_stage,
                count=workflow_target,
                rng=rng,
                allow_repeat=allow_repeat,
            )
        )

    rng.shuffle(selected)
    materialized: List[Dict[str, Any]] = []
    for sample_id, row in enumerate(selected):
        row = dict(row)
        row["sample_id"] = sample_id
        row["selected_repeat"] = int(row.get("selected_repeat", 0) or 0)
        materialized.append(row)

    stats = {
        "target_total": int(target_total),
        "selected_total": len(materialized),
        "include_workflow": bool(include_workflow),
        "workflow_fraction": float(workflow_fraction),
        "move_targets": move_targets,
        "workflow_target": int(workflow_target),
        "available_direction_counts": {
            direction: len(by_direction.get(direction, [])) for direction in MOVE_DIRECTIONS
        },
        "available_stage_counts": {stage: len(values) for stage, values in sorted(by_stage.items())},
        "selected_direction_counts": direction_counts(materialized),
        "selected_stage_counts": stage_counts(materialized),
        "selected_map_counts": map_counts(materialized),
        "selected_repeated_rows": sum(1 for row in materialized if int(row.get("selected_repeat", 0) or 0) > 0),
    }
    return materialized, stats


def direction_counts(rows: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        direction = row_move_bucket(row)
        if direction:
            counts[direction] += 1
    return {direction: counts.get(direction, 0) for direction in MOVE_DIRECTIONS}


def stage_counts(rows: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    return dict(Counter(str(row.get("stage", "")) for row in rows))


def map_counts(rows: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    return dict(Counter(str(row.get("map_name", "")) for row in rows))


def image_counts(rows: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    return dict(Counter(len(row.get("images", []) or []) for row in rows))


def write_split(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(rows)).to_parquet(path, index=False)


async def build_dataset(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_maps = parse_csv(args.train_maps)
    eval_maps = parse_csv(args.eval_maps)

    if args.train_candidates_parquet or args.eval_candidates_parquet:
        if not (args.train_candidates_parquet and args.eval_candidates_parquet):
            raise ValueError("--train-candidates-parquet and --eval-candidates-parquet must be provided together")
        train_candidates = pd.read_parquet(Path(args.train_candidates_parquet)).to_dict("records")
        eval_candidates = pd.read_parquet(Path(args.eval_candidates_parquet)).to_dict("records")
        train_failures: List[Dict[str, Any]] = []
        eval_failures: List[Dict[str, Any]] = []
    else:
        train_candidates, train_failures = await collect_candidates(
            split="train",
            maps=train_maps,
            output_dir=output_dir,
            seed_start=args.train_seed_start,
            seed_stride=args.seed_stride,
            seeds_per_map=args.seeds_per_map,
            max_steps=args.max_steps,
            feasible_order_step_budget=args.feasible_order_step_budget,
            enable_fpv=args.enable_fpv,
            verbose_env=args.verbose_env,
            log_every=args.log_every,
            waypoint_marks=args.waypoint_marks,
            fpv_dir_overrides=parse_fpv_dir_overrides(args.fpv_dir_overrides),
        )
        eval_candidates, eval_failures = await collect_candidates(
            split="eval",
            maps=eval_maps,
            output_dir=output_dir,
            seed_start=args.eval_seed_start,
            seed_stride=args.seed_stride,
            seeds_per_map=args.eval_seeds_per_map or args.seeds_per_map,
            max_steps=args.max_steps,
            feasible_order_step_budget=args.feasible_order_step_budget,
            enable_fpv=args.enable_fpv,
            verbose_env=args.verbose_env,
            log_every=args.log_every,
            waypoint_marks=args.waypoint_marks,
            fpv_dir_overrides=parse_fpv_dir_overrides(args.fpv_dir_overrides),
        )

    train_candidates_path = output_dir / "candidates_train.parquet"
    eval_candidates_path = output_dir / "candidates_eval.parquet"
    write_split(train_candidates_path, train_candidates)
    write_split(eval_candidates_path, eval_candidates)

    train_rows, train_stats = sample_balanced_rows(
        train_candidates,
        target_total=args.train_total,
        workflow_fraction=args.workflow_fraction,
        include_workflow=not args.move_only,
        allow_repeat=args.allow_repeat_underfilled,
        seed=args.sample_seed,
    )
    eval_rows, eval_stats = sample_balanced_rows(
        eval_candidates,
        target_total=args.eval_total,
        workflow_fraction=args.workflow_fraction,
        include_workflow=not args.move_only,
        allow_repeat=args.allow_repeat_underfilled,
        seed=args.sample_seed + 1,
    )

    train_path = output_dir / "train.parquet"
    eval_path = output_dir / "eval.parquet"
    write_split(train_path, train_rows)
    write_split(eval_path, eval_rows)

    manifest = {
        "train_parquet": str(train_path),
        "eval_parquet": str(eval_path),
        "train_candidates_parquet": str(train_candidates_path),
        "eval_candidates_parquet": str(eval_candidates_path),
        "train_maps": train_maps,
        "eval_maps": eval_maps,
        "assistant_target_format": "full_json",
        "image_policy": "all",
        "enable_fpv": bool(args.enable_fpv),
        "waypoint_marks": bool(args.waypoint_marks),
        "notes": [
            "Oracle actions drive the full delivery workflow.",
            "Policy prompts do not include oracle_next_move/oracle_next_action.",
            "MOVE rows are direction-balanced; workflow rows preserve rollout-format reasoning/action/future_plan targets.",
        ] + ([
            "waypoint_marks mode: locomotion rows are MOVE_TO(k) over numbered FPV markers;",
            "balancing buckets use gt_rel_direction (oracle relative direction), not the mark index.",
        ] if args.waypoint_marks else []),
        "config": {
            "max_steps": int(args.max_steps),
            "feasible_order_step_budget": int(args.feasible_order_step_budget),
            "seeds_per_map": int(args.seeds_per_map),
            "eval_seeds_per_map": int(args.eval_seeds_per_map or args.seeds_per_map),
            "train_seed_start": int(args.train_seed_start),
            "eval_seed_start": int(args.eval_seed_start),
            "seed_stride": int(args.seed_stride),
            "workflow_fraction": float(args.workflow_fraction),
            "move_only": bool(args.move_only),
            "waypoint_marks": bool(args.waypoint_marks),
            "fpv_dir_overrides": str(args.fpv_dir_overrides or ""),
            "allow_repeat_underfilled": bool(args.allow_repeat_underfilled),
            "verbose_env": bool(args.verbose_env),
            "train_candidates_parquet": str(args.train_candidates_parquet or ""),
            "eval_candidates_parquet": str(args.eval_candidates_parquet or ""),
        },
        "candidate_stats": {
            "train_rows": len(train_candidates),
            "eval_rows": len(eval_candidates),
            "train_direction_counts": direction_counts(train_candidates),
            "eval_direction_counts": direction_counts(eval_candidates),
            "train_stage_counts": stage_counts(train_candidates),
            "eval_stage_counts": stage_counts(eval_candidates),
            "train_map_counts": map_counts(train_candidates),
            "eval_map_counts": map_counts(eval_candidates),
            "train_image_counts": image_counts(train_candidates),
            "eval_image_counts": image_counts(eval_candidates),
        },
        "selected_stats": {
            "train": train_stats,
            "eval": eval_stats,
        },
        "failures": {
            "train_count": len(train_failures),
            "eval_count": len(eval_failures),
            "train_examples": train_failures[:20],
            "eval_examples": eval_failures[:20],
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="outputs/deliverybench_sft/visual_workflow_multicity_balanced",
        help="directory for train/eval parquet, manifest, and images",
    )
    parser.add_argument("--train-maps", default=",".join(TRAIN_MAPS))
    parser.add_argument("--eval-maps", default=",".join(EVAL_MAPS))
    parser.add_argument("--train-total", type=int, default=2000)
    parser.add_argument("--eval-total", type=int, default=100)
    parser.add_argument(
        "--workflow-fraction",
        type=float,
        default=0.20,
        help="fraction of selected rows reserved for non-MOVE workflow actions",
    )
    parser.add_argument("--seeds-per-map", type=int, default=128)
    parser.add_argument("--eval-seeds-per-map", type=int, default=32)
    parser.add_argument("--train-seed-start", type=int, default=1000)
    parser.add_argument("--eval-seed-start", type=int, default=9000)
    parser.add_argument("--seed-stride", type=int, default=1000)
    parser.add_argument("--sample-seed", type=int, default=17)
    parser.add_argument(
        "--train-candidates-parquet",
        default=None,
        help="reuse an existing full-workflow train candidate parquet instead of rerunning env episodes",
    )
    parser.add_argument(
        "--eval-candidates-parquet",
        default=None,
        help="reuse an existing full-workflow eval candidate parquet instead of rerunning env episodes",
    )
    parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument("--feasible-order-step-budget", type=int, default=20)
    parser.add_argument(
        "--enable-fpv",
        action="store_true",
        help="include FPV images if the map provides them; default is map-image-only",
    )
    parser.add_argument(
        "--waypoint-marks",
        action="store_true",
        help="Set-of-Marks mode: enable_waypoint_marks env flag, oracle emits "
        "MOVE_TO(k) over the numbered FPV markers, and locomotion balancing "
        "uses gt_rel_direction buckets. Combine with --enable-fpv for the "
        "marked-FPV observation channel.",
    )
    parser.add_argument(
        "--fpv-dir-overrides",
        default=None,
        help='per-map FPV dataset dir overrides, "map=path,map=path" — for '
        "maps whose default manifest points at a moved/renamed image tree",
    )
    parser.add_argument(
        "--move-only",
        action="store_true",
        help="sample only MOVE rows; not recommended for rollout-aligned SFT",
    )
    parser.add_argument(
        "--allow-repeat-underfilled",
        action="store_true",
        help="repeat rare rows if candidate generation cannot satisfy a target count",
    )
    parser.add_argument(
        "--verbose-env",
        action="store_true",
        help="show raw DeliveryBench oracle episode logs",
    )
    parser.add_argument("--log-every", type=int, default=25)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    manifest = asyncio.run(build_dataset(args))
    print(json.dumps(manifest["selected_stats"], indent=2, ensure_ascii=False))
    print(f"wrote {manifest['train_parquet']}")
    print(f"wrote {manifest['eval_parquet']}")


if __name__ == "__main__":
    main()
