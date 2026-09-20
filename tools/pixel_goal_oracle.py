#!/usr/bin/env python3
"""The any-point harness oracle: a policy-shaped script that always points at
the certified route, so a scenario's harness can be shown to carry a courier
to both doors -- and so the money a walkable scenario is worth can be read.

It is not a policy under test. It reads the run's own engine events (the
capture poses the harness recorded) and the certified pool, projects the
next point of the certified route into one of the four photographs, and
answers through the relay endpoint exactly as a served model would:

    python tools/relay_policy_server.py --dir /tmp/relay --port 8601 --model harness-oracle &
    QWEN_ENDPOINT=http://127.0.0.1:8601/v1/chat/completions QWEN_MODEL_NAME=harness-oracle \\
      PIXEL_GOAL_OUTPUT=results/oracle/seed-9 bash tools/run_pixel_goal_front_rear_delivery.sh \\
      --route-profile validated_pool --order-mode random --seed 9 \\
      --min-delivery-m 30 --max-delivery-m 80 --min-route-turns 1 --require-marked-crossing &
    python tools/pixel_goal_oracle.py --relay-dir /tmp/relay --events results/oracle/seed-9/events.jsonl --seed 9

Its reports go through the same validator as a model's, and are grouped
apart in the results file (``harness_oracle_*``): the number they give is
the ceiling the harness allows, never a benchmark row.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from embodiedbench.runtime.pixel_goal import project_world_point_to_pixel  # noqa: E402
from tools.pixel_goal_order_pool import (  # noqa: E402
    PROTOCOL_CONSTRAINTS, ResolvedDeliveryScenario, ValidatedDeliveryPool,
    load_validated_delivery_pool, nearest_pool_node, plan_pool_legs,
    resolve_delivery_scenario, shortest_pool_path)

DEFAULT_POOL = Path(__file__).resolve().parent.parent / "configs" / "pixel_goal" / \
    "paris_trusted_pedestrian_pool_v3.json"
#: The picture's bottom edge is about 2.9 m from the camera: a route point
#: nearer than this projects to the very bottom of the frame, often onto the
#: kerb or a doorstep, so the oracle prefers points further along.
NEAR_CM = 320.0
#: The route point the oracle aims for, when the route is long enough.
REACH_CM = 700.0
#: Pavement points past the goal, along the last edge, tried when the goal
#: itself is under the picture's edge: the harness snaps a pavement pixel to
#: the nearest certified node, so pointing past the door still brings the
#: courier to it.
PAST_GOAL_CM = (150.0, 300.0, 450.0)
#: Where a point may land in a picture to be aimed at.
U_RANGE = (0.05, 0.95)
V_RANGE = (0.55, 0.93)
#: The end of a certified leg from the node the courier stands on may sit
#: lower in the picture: the engine certified that very pixel from that
#: very node (a 3 m leg's end projects at v = 0.94-0.97), and refusing it
#: for being low sent the oracle to an uncertified point round a corner,
#: whose ray met a wall 57 times on one held-out shift.
LEG_V_RANGE = (0.55, 0.975)
#: The camera sits this far above the pawn's position; feet are 88 cm below it.
GROUND_BELOW_POSE_CM = 88.0
#: How long a prompt is held while the events file catches up with the
#: observation's second capture pair (the file is appended a call at a time).
CAPTURE_GRACE_S = 20.0


def captures_from_events(events_path: Path) -> list[dict[str, Any]]:
    """Every capture-pair response the harness recorded, in order."""
    captures = []
    if not events_path.exists():
        return captures
    with events_path.open() as stream:
        for line in stream:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("function") == "PixelGoal_CaptureViewPairJson":
                captures.append(event["response"])
    return captures


def _front_yaw(capture: dict[str, Any]) -> float | None:
    for item in capture.get("views", ()):
        if item.get("camera_view") == "front":
            return float(item["camera_rotation_degrees"][1])
    return None


def cameras_of(captures: list[dict[str, Any]]) -> tuple[dict[str, tuple[list[float], float]], dict | None]:
    """view name -> (camera location, yaw) from the observation's two pairs:
    pair A at the facing (front, rear), pair B a quarter turn to the right
    (right, left); and the pose pair A was taken from.

    The two are the last capture and the one before it only when the last
    is B: its front camera a quarter turn to the right of the previous
    capture's, from the same spot. The harness captures pair A again while
    a capture settles (four times on one turn of held-out seed 14), and the
    events file can be read between A and B, so a last capture that is not
    a B pair is an observation not yet complete: nothing is returned and
    the caller reads again. Pairing the last two blindly gave the right and
    left views the front's yaw and sent the courier's pixels into a wall.
    """
    if len(captures) < 2:
        return {}, None
    a, b = captures[-2], captures[-1]
    yaw_a, yaw_b = _front_yaw(a), _front_yaw(b)
    if yaw_a is None or yaw_b is None:
        return {}, None
    quarter = (yaw_b - yaw_a - 90.0 + 180.0) % 360.0 - 180.0
    same_spot = (abs(a["pose"]["x_cm"] - b["pose"]["x_cm"]) < 1.0
                 and abs(a["pose"]["y_cm"] - b["pose"]["y_cm"]) < 1.0)
    if abs(quarter) > 2.0 or not same_spot:
        return {}, None
    out: dict[str, tuple[list[float], float]] = {}
    for item in a["views"]:
        out["front" if item["camera_view"] == "front" else "rear"] = (
            item["camera_location_cm"], item["camera_rotation_degrees"][1])
    for item in b["views"]:
        out["right" if item["camera_view"] == "front" else "left"] = (
            item["camera_location_cm"], item["camera_rotation_degrees"][1])
    return out, a["pose"]


def current_turn_text(prompt: str) -> str:
    """The relay's prompt file holds the whole conversation; only the last
    turn's text says where the courier is now."""
    return prompt.rsplit("### your notes", 1)[-1]


def leg_candidates(pool: ValidatedDeliveryPool, start: str, goal: str) -> list[tuple[str, tuple[float, float], float]]:
    """The ends of the certified legs out of ``start`` that bring ``goal``
    nearer, as (label, point, metres of certified route left from the end),
    the end with the least left first. The harness's own first move from
    ``start`` is among them (it is the one that leaves the least), so it
    comes first unless a longer leg leaves as little."""
    here = plan_pool_legs(pool, start, goal)
    if here is None:
        return []
    remaining_here = here[0].length_cm
    rows: list[tuple[float, str, tuple[float, float]]] = []
    for s, end in pool.certified_legs:
        if s != start:
            continue
        onward = plan_pool_legs(pool, end, goal)
        if onward is None:
            continue
        remaining = onward[0].length_cm
        if remaining < remaining_here - 100.0:
            rows.append((remaining, end, pool.nodes_by_id[end].position))
    rows.sort()
    return [(f"leg:{end}", xy, remaining) for remaining, end, xy in rows]


def choose(pool: ValidatedDeliveryPool, scenario: ResolvedDeliveryScenario,
           prompt: str, cameras: dict[str, tuple[list[float], float]],
           pose: dict | None, state: dict[str, Any]) -> str:
    """The reply for one turn: a door action at the door, otherwise a pixel on
    the certified route in whichever picture holds it best."""
    current = current_turn_text(prompt)
    where = re.search(r"### where you are\n(.*)\n", current)
    where_line = where.group(1) if where else ""
    collecting = "collect from" in current.split("### where you are")[0]
    number = scenario.pickup.number if collecting else scenario.dropoff.number
    if f"outside number {number}" in where_line:
        return "THOUGHT: at the door.\n```\n" + ("collect()" if collecting else "hand_over()") + "\n```"
    if not cameras or pose is None:
        return "THOUGHT: no capture yet.\n```\nnavigate()\n```"
    here = (pose["x_cm"], pose["y_cm"])
    start = nearest_pool_node(pool, here, radius_cm=400.0, stops_only=True)
    goal = scenario.pickup.handover_node_id if collecting else scenario.dropoff.handover_node_id
    if start is None:
        return "THOUGHT: lost.\n```\nnavigate()\n```"
    path = shortest_pool_path(pool, start[0].id, goal)
    if path is None:
        return "THOUGHT: no certified route from here.\n```\nnavigate()\n```"
    nodes = [pool.nodes_by_id[n] for n in path.node_ids]
    refused = "That did not work" in current
    state["streak"] = state.get("streak", 0) + 1 if refused else 0
    candidates: list[tuple[str, tuple[float, float]]] = []
    # First the ends of certified legs out of the node the courier stands
    # on: a leg's end was certified by resolving exactly that pixel from
    # exactly that node, so it resolves again, where a route node picked
    # by distance can land its ray on a kerb or a wall round a corner.
    # Every certified leg from here that brings the goal nearer, the one
    # that brings it nearest first -- not only the harness's own first
    # move: at a corner node that move (3 m, 53 degrees off the facing)
    # was below the bottom edge of every one of the four pictures, while a
    # 6 m leg the same way was in the right-hand one.
    seen: set[str] = set()
    if pool.certified_legs is not None:
        for label, xy, _remaining in leg_candidates(pool, start[0].id, goal):
            seen.add(label[4:])
            candidates.append((label, xy))
        planned = plan_pool_legs(pool, start[0].id, goal)
        if planned is not None:
            for move in planned[1][1:]:
                if move.aim not in seen:
                    seen.add(move.aim)
                    candidates.append((f"plan:{move.aim}", tuple(move.aim_cm)))
    cum, along = 0.0, []
    for prev, node in zip(nodes, nodes[1:]):
        cum += math.dist(prev.position, node.position)
        along.append((cum, node.id, (node.x_cm, node.y_cm)))
    far = sorted((c for c in along if c[0] >= NEAR_CM), key=lambda c: abs(c[0] - REACH_CM))
    near = [c for c in along if c[0] < NEAR_CM]
    candidates.extend((nid, xy) for _c, nid, xy in far if nid not in seen)
    if len(nodes) >= 2:
        gx, gy = nodes[-1].x_cm, nodes[-1].y_cm
        px, py = nodes[-2].x_cm, nodes[-2].y_cm
        norm = math.hypot(gx - px, gy - py) or 1.0
        dx, dy = (gx - px) / norm, (gy - py) / norm
        for k in PAST_GOAL_CM:
            candidates.append((f"{nodes[-1].id}+{k:.0f}", (gx + dx * k, gy + dy * k)))
    candidates.extend((nid, xy) for _c, nid, xy in reversed(near))
    ground_z = pose["z_cm"] - GROUND_BELOW_POSE_CM
    options = []
    for label, (x, y) in candidates:
        views = []
        v_range = LEG_V_RANGE if label.startswith("leg:") else V_RANGE
        for view, (loc, yaw) in cameras.items():
            uv = project_world_point_to_pixel(tuple(loc), yaw, (x, y, ground_z))
            if uv is None or not (U_RANGE[0] <= uv[0] <= U_RANGE[1] and v_range[0] <= uv[1] <= v_range[1]):
                continue
            views.append((abs(uv[0] - 0.5), view, uv))
        if views:
            views.sort()
            options.append((label, views[0][1], views[0][2]))
    if not options:
        return "THOUGHT: no route point in view.\n```\nnavigate()\n```"
    # after a refusal, the next candidate rather than the same pixel again;
    # round the list rather than stopping at its end, so a refusal of every
    # candidate is not four identical pixels and a run ended as stuck
    label, view, (u, v) = options[state["streak"] % len(options)]
    return (f"THOUGHT: route point {label} in the {view} view.\n```\n"
            f'walk_to_pixel(view="{view}", u={u:.3f}, v={v:.3f})\n```')


def serve(relay_dir: Path, events_path: Path, pool: ValidatedDeliveryPool,
          scenario: ResolvedDeliveryScenario, *, max_turns: int, timeout_s: float,
          poll_s: float = 0.5) -> int:
    """Answer relay turns as they appear until the run ends or the cap is hit."""
    state: dict[str, Any] = {}
    answered: set[str] = set()
    waiting_since: dict[str, float] = {}
    deadline = time.time() + timeout_s
    while time.time() < deadline and len(answered) < max_turns:
        pending_file = relay_dir / "pending"
        pending = pending_file.read_text().strip() if pending_file.exists() else ""
        turn_dir = relay_dir / pending if pending else None
        if (turn_dir is not None and pending not in answered
                and (turn_dir / "prompt.txt").exists() and not (turn_dir / "answer.txt").exists()):
            prompt = (turn_dir / "prompt.txt").read_text()
            cameras, pose = cameras_of(captures_from_events(events_path))
            try:
                reply = choose(pool, scenario, prompt, cameras, pose, state)
            except Exception as error:  # noqa: BLE001 - the turn must be answered
                reply = f"THOUGHT: oracle error {error!r}.\n```\nnavigate()\n```"
            if reply.startswith("THOUGHT: no capture yet."):
                # The prompt is there but the events file does not yet end
                # with the observation's second pair: the harness is still
                # writing it. Read again for a while before giving the turn
                # away to a phone check.
                since = waiting_since.setdefault(pending, time.time())
                if time.time() - since < CAPTURE_GRACE_S:
                    time.sleep(poll_s)
                    continue
            (turn_dir / "answer.txt").write_text(reply)
            answered.add(pending)
            print(pending, reply.splitlines()[-2], flush=True)
        time.sleep(poll_s)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--relay-dir", type=Path, required=True,
                        help="the relay endpoint's --dir")
    parser.add_argument("--events", type=Path, required=True,
                        help="the run's events.jsonl (appears once the engine is up)")
    parser.add_argument("--seed", type=int, required=True, help="the protocol seed of the run")
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--max-turns", type=int, default=60)
    parser.add_argument("--timeout-s", type=float, default=3600.0)
    args = parser.parse_args(argv)
    pool = load_validated_delivery_pool(args.pool)
    scenario = resolve_delivery_scenario(
        pool, mode="random", seed=args.seed, constraints=PROTOCOL_CONSTRAINTS)
    print(f"oracle: seed {args.seed} {scenario.pickup.text} -> {scenario.dropoff.text} "
          f"({scenario.scenario_id})", flush=True)
    return serve(args.relay_dir, args.events, pool, scenario,
                 max_turns=args.max_turns, timeout_s=args.timeout_s)


if __name__ == "__main__":
    sys.exit(main())
