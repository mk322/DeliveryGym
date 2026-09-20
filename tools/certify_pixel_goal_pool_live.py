#!/usr/bin/env python3
"""Certify a pedestrian pool's legs against the engine's own path check.

The v2 pool's edges were certified against the Recast surface under them.
The engine judges a walk by something else -- whether the controller path
from where the pawn stands to the point it was given crosses a road
polygon outside a marked crossing, and what the ray it was aimed by hit --
and its verdict is on a *leg*: one straight walk from one pool node to
another, judged from the node it starts on, in that direction. The same
paving can be walkable one way and refused the other, and a long leg can be
refused where every shorter leg along it is accepted. So this runner asks
the engine about every leg the harness could plan (``candidate_legs``: a
chord of 3-10 m whose chain of pool nodes stays within 40 cm of it), the
way the harness walks it at run time:

    teleport the pawn onto the start node (SPEAR ``K2_TeleportTo``), face
    the end node, capture, project the end node into the picture, and
    resolve that pixel with an acceptance radius wider than the map -- the
    engine resolves and judges without moving.

Every verdict is written to ``verdicts.jsonl`` as it is made (a re-run picks
up where it stopped). ``tools/prune_pixel_goal_pool.py`` turns the verdicts
into the next pool version: the accepted legs, and only the nodes they join
both ways. Runs through the launcher like a delivery:

    PIXEL_GOAL_RUNNER_PY=tools/certify_pixel_goal_pool_live.py \\
    PIXEL_GOAL_OUTPUT=results/certify bash tools/run_pixel_goal_front_rear_delivery.sh \\
      --route-profile validated_pool --order-mode random --seed 0 \\
      --min-delivery-m 30 --max-delivery-m 80 --min-route-turns 1 --require-marked-crossing \\
      --pool configs/pixel_goal/paris_trusted_pedestrian_pool_v2.json
"""
from __future__ import annotations

import argparse
import base64
import collections
import json
import math
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from embodiedbench.runtime.pixel_goal import (  # noqa: E402
    LivePixelGoalRuntime, PixelGoalConfig, PixelGoalRejected, project_world_point_to_pixel)
from embodiedbench.schemas.runtime import NavPixelGoalAction  # noqa: E402
from tools.pixel_goal_courier_backend import LEG_NAVMESH_ADJUSTMENT_CM  # noqa: E402
from tools.pixel_goal_order_pool import (  # noqa: E402
    LEG_EPSILON_CM, MAX_LEG_CM, MIN_LEG_CM, ValidatedDeliveryPool, candidate_legs,
    load_validated_delivery_pool)

REPO_ROOT = Path(__file__).resolve().parent.parent
#: The pool whose legs are judged: the source of the next version.
DEFAULT_POOL = REPO_ROOT / "configs" / "pixel_goal" / "paris_trusted_pedestrian_pool_v2.json"
WIDTH, HEIGHT, HFOV = 640, 360, 90.0
#: The pawn's position is this far above its feet.
PAWN_ABOVE_FEET_CM = 88.0
#: A teleport that lands further than this from the node is retried once and
#: then recorded as a failed placement.
PLACEMENT_TOLERANCE_CM = 15.0
#: The pawn is left where it is when it already stands this close to the
#: node (a resolve without a walk does not move it).
PLACEMENT_KEEP_CM = 2.0
#: A refusal for this reason is the tooling's, not the engine's verdict on
#: the leg (the pawn moved between the capture and the resolve): judged
#: again, up to three times, and never counted as done.
STALE = "camera_snapshot_stale"


def node_heights(audit_path: Path, pool: ValidatedDeliveryPool) -> dict[str, float]:
    """Ground height per node: the audit's projected grid point for grid
    nodes; the nearest grid point's for crossing and entrance nodes."""
    audit = json.loads(audit_path.read_text())
    grid = audit["pedestrian_recast_grid"]
    z_of = {gid: float(pt["projected_cm"][2])
            for gid, pt in zip(grid["grid_ids"], grid["points"]) if pt.get("projected")}
    heights: dict[str, float] = {}
    grid_nodes = [(n, z_of[n.id]) for n in pool.nodes if n.id in z_of]
    for node in pool.nodes:
        if node.id in z_of:
            heights[node.id] = z_of[node.id]
        else:
            nearest = min(grid_nodes, key=lambda item: math.dist(
                (item[0].x_cm, item[0].y_cm), (node.x_cm, node.y_cm)))
            heights[node.id] = nearest[1]
    return heights


def leg_id(start: str, end: str) -> str:
    return f"{start}->{end}"


def ordered_legs(pool: ValidatedDeliveryPool) -> list[dict[str, Any]]:
    """The candidate legs in judging order: by start node, so the pawn is
    teleported once per node, then by length."""
    nodes = pool.nodes_by_id
    rows = []
    for (start, end), chain in candidate_legs(pool).items():
        rows.append({
            "leg": leg_id(start, end), "start": start, "end": end, "chain": list(chain),
            "length_cm": round(math.dist(nodes[start].position, nodes[end].position), 1)})
    rows.sort(key=lambda row: (row["start"], row["length_cm"], row["end"]))
    return rows


def read_done(path: Path) -> dict[str, dict[str, Any]]:
    """The legs already judged, by id; a stale-snapshot refusal is not a
    verdict and is judged again."""
    done: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return done
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("verdict") == "rejected" and row.get("reason") == STALE:
            continue
        done[row["leg"]] = row
    return done


def main() -> int:
    from tools.run_pixel_goal_1b_closed_loop import _prepare_paris_loop
    from tools.run_pixel_goal_1b_poc import (
        AttachedParisGameSession, _reset_paris_poc_trial, _wait_for_paris_poc,
        _warm_up_paris_capture)
    from tools.run_pixel_goal_m1a import SpearPixelGoalEndpoint
    from tools.run_pixel_goal_full_delivery import AGENT_TAG, DELIVERY_REGION
    from tools.run_pixel_goal_front_rear_probe import build_front_rear_setup_request

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--spear-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--navmesh-timeout-s", type=float, default=240.0)
    parser.add_argument("--capture-warmup-s", type=float, default=20.0)
    parser.add_argument("--limit", type=int, default=None, help="judge only the first N legs")
    parser.add_argument("--shard", default="1/1", metavar="K/N",
                        help="judge the K-th of N equal shares of the legs (a second engine)")
    parser.add_argument("--save-images", action="store_true",
                        help="keep the front picture of every refused check")
    args, _unknown = parser.parse_known_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    verdicts_path = out / "verdicts.jsonl"
    done = read_done(verdicts_path)

    pool = load_validated_delivery_pool(args.pool)
    audit_path = args.pool.parent / pool.pedestrian_graph.audit_path
    heights = node_heights(audit_path, pool)
    legs = ordered_legs(pool)
    shard, shards = (int(part) for part in args.shard.split("/"))
    if shards > 1:
        legs = [leg for index, leg in enumerate(legs) if index % shards == shard - 1]
    if args.limit:
        legs = legs[:args.limit]
    todo = [leg for leg in legs if leg["leg"] not in done]
    print(f"certify: {len(legs)} legs (shard {args.shard}), {len(done)} already judged, "
          f"{len(todo)} to do; legs {MIN_LEG_CM:.0f}-{MAX_LEG_CM:.0f} cm, chain within "
          f"{LEG_EPSILON_CM:.0f} cm", flush=True)
    (out / "legs.json").write_text(json.dumps(
        {"pool": str(args.pool), "pool_sha256": pool.sha256, "count": len(legs),
         "min_leg_cm": MIN_LEG_CM, "max_leg_cm": MAX_LEG_CM, "leg_epsilon_cm": LEG_EPSILON_CM,
         "navmesh_adjustment_cm": LEG_NAVMESH_ADJUSTMENT_CM,
         "placement_tolerance_cm": PLACEMENT_TOLERANCE_CM, "shard": args.shard},
        indent=1))

    spawn = pool.spawns[0]
    spawn_node = pool.nodes_by_id[spawn.node_id]
    region = replace(
        DELIVERY_REGION, name=pool.region.name,
        agent_spawn_cm=(spawn_node.x_cm, spawn_node.y_cm, spawn.z_cm),
        agent_yaw_deg=spawn.yaw_deg,
        nav_bounds_center_cm=pool.region.nav_bounds_center_cm,
        nav_bounds_extent_cm=pool.region.nav_bounds_extent_cm)
    setup_request = build_front_rear_setup_request(region)

    session = AttachedParisGameSession(args.spear_config)
    session.begin_play()
    try:
        endpoint = SpearPixelGoalEndpoint(session, out / "events.jsonl")
        _prepare_paris_loop(
            endpoint, setup_request, readiness_timeout_s=args.navmesh_timeout_s,
            capture_warmup_s=args.capture_warmup_s, wait_fn=_wait_for_paris_poc,
            warm_up_fn=_warm_up_paris_capture)
        _reset_paris_poc_trial(endpoint, args.navmesh_timeout_s)
        resolver = LivePixelGoalRuntime(endpoint, PixelGoalConfig(
            max_navmesh_adjustment_cm=LEG_NAVMESH_ADJUSTMENT_CM, acceptance_radius_cm=100000.0,
            movement_timeout_sim_s=10.0, execution_timeout_s=60.0))
        instance, game = session._instance, session._game
        unreal = game.unreal_service

        def feet() -> list[float]:
            return endpoint.call("PixelGoal_GetParisPocStatusJson", {})["agent_feet_position_cm"]

        def pawn():
            return unreal.find_actor_by_tag(AGENT_TAG, "AActor", as_unreal_object=True)

        def set_yaw(yaw_deg: float) -> None:
            with instance.begin_frame():
                pawn().K2_SetActorRotation(
                    NewRotation={"Pitch": 0.0, "Yaw": float(yaw_deg), "Roll": 0.0},
                    bTeleportPhysics=True)
            with instance.end_frame():
                pass

        def place(x: float, y: float, z_feet: float, yaw_deg: float) -> dict[str, Any]:
            """The pawn on the node: left there when it already stands on it
            (the previous check did not move it), else teleported, twice
            if the first landing is off."""
            f = feet()
            err = math.dist((x, y), (f[0], f[1]))
            if err <= PLACEMENT_KEEP_CM:
                return {"feet": f, "placement_error_cm": round(err, 1), "attempts": 0}
            for attempt in (1, 2):
                with instance.begin_frame():
                    pawn().K2_TeleportTo(
                        DestLocation={"X": float(x), "Y": float(y), "Z": float(z_feet + PAWN_ABOVE_FEET_CM)},
                        DestRotation={"Pitch": 0.0, "Yaw": float(yaw_deg), "Roll": 0.0})
                with instance.end_frame():
                    pass
                time.sleep(0.4)
                f = feet()
                err = math.dist((x, y), (f[0], f[1]))
                if err <= PLACEMENT_TOLERANCE_CM:
                    break
            return {"feet": f, "placement_error_cm": round(err, 1), "attempts": attempt}

        def capture():
            pair = resolver.capture_view_pair(
                agent_tag=AGENT_TAG, width_px=WIDTH, height_px=HEIGHT, fov_degrees=HFOV)
            for name, _req, resp in reversed(endpoint.calls):
                if name == "PixelGoal_CaptureViewPairJson":
                    break
            front = next(v for v in resp["views"] if v["camera_view"] == "front")
            return pair, front["camera_location_cm"], front["camera_rotation_degrees"][1]

        def judge(leg: dict[str, Any]) -> dict[str, Any]:
            a, b = pool.nodes_by_id[leg["start"]], pool.nodes_by_id[leg["end"]]
            yaw = math.degrees(math.atan2(b.y_cm - a.y_cm, b.x_cm - a.x_cm))
            row: dict[str, Any] = dict(leg)
            row["placement"] = place(a.x_cm, a.y_cm, heights[leg["start"]], yaw)
            if row["placement"]["placement_error_cm"] > PLACEMENT_TOLERANCE_CM:
                row["verdict"] = "placement_failed"
                return row
            here = feet()
            aim = (b.x_cm, b.y_cm)
            row["aim_cm"] = [round(aim[0], 1), round(aim[1], 1)]
            for attempt in range(3):
                set_yaw(yaw)
                time.sleep(0.4 + 0.3 * attempt)
                pair, cam, cam_yaw = capture()
                uv = project_world_point_to_pixel(tuple(cam), cam_yaw, (aim[0], aim[1], here[2]),
                                                  width_px=WIDTH, height_px=HEIGHT, hfov_deg=HFOV)
                if uv is None or not (0.0 <= uv[0] <= 1.0 and 0.0 <= uv[1] <= 1.0):
                    row["verdict"] = "outside_picture"
                    row["uv"] = uv
                    return row
                row["uv"] = [round(uv[0], 4), round(uv[1], 4)]
                row["attempt"] = attempt + 1
                try:
                    started, result = resolver.execute(
                        NavPixelGoalAction(target={"u_norm": uv[0], "v_norm": uv[1]}), pair.front)
                    req = started.request
                    row.pop("reason", None)      # a retry after a stale snapshot succeeded
                    row.pop("audit", None)
                    row.update(verdict="accepted",
                               controller_path_length_cm=req.controller_path_length_cm,
                               projected_cm=[req.projected_target.x_cm, req.projected_target.y_cm],
                               raw_hit_cm=[req.raw_world_hit.x_cm, req.raw_world_hit.y_cm, req.raw_world_hit.z_cm],
                               moved_cm=result.distance_travelled_cm)
                    for name, _req, resp in reversed(endpoint.calls):
                        if name == "PixelGoal_ResolveAndMoveJson":
                            row["raw_hit_actor"] = resp.get("raw_hit_actor")
                            row["direct_path_legal"] = resp.get("direct_path_legal")
                            break
                    break
                except PixelGoalRejected as rejected:
                    row.update(verdict="rejected", reason=rejected.reason,
                               audit={k: v for k, v in rejected.audit.items() if "points" not in k})
                    if rejected.reason == STALE:
                        continue
                    if args.save_images:
                        _, _, b64 = pair.front.rgb_data_url.partition(",")
                        (out / f"refused-{leg['leg'].replace(':', '_')}.jpg").write_bytes(base64.b64decode(b64))
                    break
                except Exception as error:  # noqa: BLE001 - recorded, the run goes on
                    row.update(verdict="error", error=repr(error), trace=traceback.format_exc()[-400:])
                    break
            row["feet_after"] = feet()
            return row

        started_at = time.perf_counter()
        counts: collections.Counter = collections.Counter(r["verdict"] for r in done.values())
        with verdicts_path.open("a") as sink:
            for index, leg in enumerate(todo, 1):
                row = judge(leg)
                row["seconds"] = round(time.perf_counter() - started_at, 1)
                sink.write(json.dumps(row) + "\n")
                sink.flush()
                counts[row["verdict"]] += 1
                if index % 25 == 0 or row["verdict"] != "accepted":
                    rate = index / max(time.perf_counter() - started_at, 1e-6)
                    print(f"[{index}/{len(todo)}] {row['leg']} {row['length_cm']} cm -> "
                          f"{row['verdict']} {row.get('reason', '')} | {dict(counts)} | "
                          f"{rate:.2f}/s, {((len(todo) - index) / max(rate, 1e-6)) / 3600:.1f} h left",
                          flush=True)
        print("certify done:", dict(counts), flush=True)
        (out / "summary.json").write_text(json.dumps(
            {"pool": str(args.pool), "pool_sha256": pool.sha256, "legs": len(legs),
             "verdicts": dict(counts), "seconds": round(time.perf_counter() - started_at, 1)},
            indent=1))
    finally:
        try:
            session.end_play()
        except Exception:  # noqa: BLE001
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
