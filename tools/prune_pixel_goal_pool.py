#!/usr/bin/env python3
"""Turn the engine's verdicts on a pool's legs into the next pool version.

    python tools/prune_pixel_goal_pool.py \\
        --pool configs/pixel_goal/paris_trusted_pedestrian_pool_v2.json \\
        --verdicts results/certify/verdicts.jsonl \\
        --out configs/pixel_goal/paris_trusted_pedestrian_pool_v3.json

``tools/certify_pixel_goal_pool_live.py`` judged every leg the harness could
plan (``candidate_legs``) from its start node with the engine's own path
check. The next pool version keeps the legs the engine accepted as straight
walks (a controller path no longer than the chord by more than
``--max-path-excess-cm``), only the nodes those legs join both ways -- the
largest strongly connected component, so a pawn can leave every node it
can reach -- and only the edges some kept leg walks over. The graph hash is
recomputed and the pool records the verdict file it was built on. Every
certified stop and spawn must be in the component and every candidate leg
must have been judged; otherwise this tool stops and says which are
missing rather than write a pool that is only partly certified. The pool's
audit, region and stops are unchanged.
"""
from __future__ import annotations

import argparse
import collections
import datetime as _dt
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.certify_pixel_goal_pool_live import (  # noqa: E402
    PLACEMENT_TOLERANCE_CM, STALE, leg_id)
from tools.pixel_goal_courier_backend import LEG_NAVMESH_ADJUSTMENT_CM  # noqa: E402
from tools.pixel_goal_order_pool import (  # noqa: E402
    CERTIFIED_LEGS_KEY, ENGINE_CERTIFICATION_KEY, ENGINE_CERTIFICATION_METHOD,
    LEG_EPSILON_CM, MAX_LEG_CM, MIN_LEG_CM, ValidatedDeliveryPool, candidate_legs,
    load_validated_delivery_pool, strongly_connected_components)

JUDGED = ("accepted", "rejected")
#: An accepted leg whose controller path exceeds its chord by more than this
#: bent round something and is not the straight walk that was certified.
DEFAULT_MAX_PATH_EXCESS_CM = 150.0


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_verdicts(path: Path) -> dict[str, dict[str, Any]]:
    """The last verdict per leg; a stale-snapshot refusal is not one."""
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("verdict") == "rejected" and row.get("reason") == STALE:
            continue
        rows[row["leg"]] = row          # a later re-judgement replaces an earlier one
    return rows


def sort_verdicts(
    candidates: dict[tuple[str, str], tuple[str, ...]],
    verdicts: dict[str, dict[str, Any]], *, max_path_excess_cm: float,
) -> tuple[set[tuple[str, str]], dict[tuple[str, str], str], list[str]]:
    """(accepted legs, refused legs -> reason, unjudged leg ids)."""
    accepted: set[tuple[str, str]] = set()
    refused: dict[tuple[str, str], str] = {}
    unjudged: list[str] = []
    for (start, end) in sorted(candidates):
        row = verdicts.get(leg_id(start, end))
        if row is None or row.get("verdict") not in JUDGED:
            unjudged.append(leg_id(start, end))
            continue
        if row["verdict"] == "rejected":
            refused[(start, end)] = str(row.get("reason", "rejected"))
            continue
        excess = float(row.get("controller_path_length_cm") or 0.0) - float(row["length_cm"])
        if excess > max_path_excess_cm:
            refused[(start, end)] = "controller_path_not_straight"
            continue
        accepted.add((start, end))
    return accepted, refused, unjudged


def build(
    doc: dict[str, Any], pool: ValidatedDeliveryPool,
    candidates: dict[tuple[str, str], tuple[str, ...]],
    accepted: set[tuple[str, str]],
) -> tuple[dict[str, Any], list[tuple[str, str]], list[str], list[str]]:
    """The next pool document: (document, kept legs, removed edges, removed
    nodes). Stops and spawns outside the component stop the tool."""
    node_ids = [node["id"] for node in doc["nodes"]]
    # The nodes a pawn may stop on are the largest strongly connected
    # component of the accepted legs: reachable and leavable. A node
    # outside it that accepted legs of the component walk over -- the
    # middle of a marked crossing the engine crosses end to end but accepts
    # no leg from -- stays as a pass-through node; a node on no kept leg
    # goes, with every edge no kept leg walks.
    components = strongly_connected_components(node_ids, sorted(accepted))
    component = components[0] if components else set()
    protected = {s["node_id"] for s in doc["spawns"]} | {s["handover_node_id"] for s in doc["stops"]}
    outside = sorted(protected - component)
    if outside:
        raise SystemExit(
            "the accepted legs do not join these certified stops or spawns both ways: "
            f"{outside}")
    legs = sorted(leg for leg in accepted if leg[0] in component and leg[1] in component)
    covered: set[frozenset[str]] = set()
    walked_over: set[str] = set()
    for leg in legs:
        chain = candidates[leg]
        walked_over.update(chain)
        for x, y in zip(chain, chain[1:]):
            covered.add(frozenset((x, y)))
    pass_through = sorted(walked_over - component)
    kept_nodes = component | set(pass_through)
    edges = [e for e in doc["edges"] if frozenset((e["a"], e["b"])) in covered]
    removed_edges = sorted(
        (e["a"], e["b"]) for e in doc["edges"] if frozenset((e["a"], e["b"])) not in covered)
    removed_nodes = sorted(node for node in node_ids if node not in kept_nodes)
    used = {e["a"] for e in edges} | {e["b"] for e in edges}
    stranded = sorted(kept_nodes - used)
    if stranded:
        raise SystemExit(f"nodes joined by legs but by no kept edge: {stranded[:5]}")
    out = dict(doc)
    out["edges"] = sorted(edges, key=lambda e: (e["a"], e["b"]))
    out["nodes"] = sorted((n for n in doc["nodes"] if n["id"] in kept_nodes), key=lambda n: n["id"])
    out[CERTIFIED_LEGS_KEY] = [list(leg) for leg in legs]
    out["_pass_through_nodes"] = pass_through
    return out, legs, removed_edges, removed_nodes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--verdicts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-path-excess-cm", type=float, default=DEFAULT_MAX_PATH_EXCESS_CM)
    parser.add_argument("--certified-at", default=_dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    parser.add_argument("--allow-unjudged", action="store_true",
                        help="build with some legs unjudged (left out); says which")
    args = parser.parse_args(argv)

    pool = load_validated_delivery_pool(args.pool)
    doc = json.loads(args.pool.read_text())
    candidates = candidate_legs(pool)
    verdicts = read_verdicts(args.verdicts)
    accepted, refused, unjudged = sort_verdicts(
        candidates, verdicts, max_path_excess_cm=args.max_path_excess_cm)
    if unjudged and not args.allow_unjudged:
        print(f"{len(unjudged)} of {len(candidates)} legs have no verdict; the first: {unjudged[:8]}")
        return 2
    built, legs, removed_edges, removed_nodes = build(doc, pool, candidates, accepted)
    pass_through = built.pop("_pass_through_nodes")
    built["version"] = int(doc["version"]) + 1
    built["pedestrian_graph"] = dict(doc["pedestrian_graph"])
    built["pedestrian_graph"]["graph_sha256"] = _canonical_sha256(
        {key: built[key] for key in ("nodes", "edges", "spawns", "stops")})
    built[ENGINE_CERTIFICATION_KEY] = {
        "method": ENGINE_CERTIFICATION_METHOD,
        "certified_at": args.certified_at,
        "source_pool_sha256": pool.sha256,
        "verdicts_sha256": hashlib.sha256(args.verdicts.read_bytes()).hexdigest(),
        "legs": len(candidates),
        "accepted": len(legs),
        "refused": len(refused),
        "unjudged": len(unjudged),
        "removed_edges": [list(pair) for pair in removed_edges],
        "removed_nodes": removed_nodes,
        "pass_through_nodes": pass_through,
        "placement_tolerance_cm": PLACEMENT_TOLERANCE_CM,
        "min_leg_cm": MIN_LEG_CM,
        "navmesh_adjustment_cm": LEG_NAVMESH_ADJUSTMENT_CM,
        "leg_epsilon_cm": LEG_EPSILON_CM,
        "max_leg_cm": MAX_LEG_CM,
        "max_path_excess_cm": args.max_path_excess_cm,
    }
    args.out.write_text(json.dumps(built, indent=2, sort_keys=True) + "\n")
    # the loader is the judge of what was written: connectivity, hashes, keys
    loaded = load_validated_delivery_pool(args.out)
    reasons = collections.Counter(refused.values())
    print(json.dumps({
        "out": str(args.out), "version": loaded.version,
        "legs": {"candidates": len(candidates), "accepted_by_engine": len(accepted),
                 "kept": len(legs), "refused": len(refused), "unjudged": len(unjudged),
                 "refusal_reasons": dict(reasons)},
        "edges": {"before": len(doc["edges"]), "after": len(loaded.edges), "removed": len(removed_edges)},
        "nodes": {"before": len(doc["nodes"]), "after": len(loaded.nodes), "removed": removed_nodes,
                  "pass_through": pass_through},
    }, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
