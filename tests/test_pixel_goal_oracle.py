"""The any-point harness oracle: a policy-shaped script that points at the
certified route, answering through the relay like a served model would."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tools import pixel_goal_oracle as oracle
from tools.pixel_goal_order_pool import (
    PROTOCOL_CONSTRAINTS, load_validated_delivery_pool, resolve_delivery_scenario)

POOL_PATH = Path(__file__).resolve().parent.parent / "configs" / "pixel_goal" / \
    "paris_trusted_pedestrian_pool_v3.json"


@pytest.fixture(scope="module")
def pool():
    return load_validated_delivery_pool(POOL_PATH)


@pytest.fixture(scope="module")
def seed9(pool):
    return resolve_delivery_scenario(pool, mode="random", seed=9, constraints=PROTOCOL_CONSTRAINTS)


def cameras_at(pool, node_id, yaw_deg):
    """Four cameras the harness would have recorded standing on ``node_id``
    facing ``yaw_deg``: 148 cm above the feet, pair B a quarter turn right."""
    node = pool.nodes_by_id[node_id]
    loc = [node.x_cm, node.y_cm, 20.0 + 148.0]
    return ({"front": (loc, yaw_deg), "rear": (loc, (yaw_deg + 180.0) % 360.0),
             "right": (loc, (yaw_deg + 90.0) % 360.0), "left": (loc, (yaw_deg + 270.0) % 360.0)},
            {"x_cm": node.x_cm, "y_cm": node.y_cm, "z_cm": 20.0 + 88.0, "yaw_deg": yaw_deg})


def prompt(where: str, *, collecting: bool, refused: bool = False) -> str:
    job = "collect from 11 Rue Oberkampf" if collecting else "hand over at 16 Rue Oberkampf"
    result = "That did not work: That point is on the carriageway." if refused else "You walk 7 m."
    return ("===== system =====\n...\n### your notes\n(none)\n### job\n" + job +
            "\n### where you are\n" + where + "\n### last result\n" + result + "\n")


class TestChoose:
    def test_at_the_door_it_does_the_door_action(self, pool, seed9):
        cams, pose = cameras_at(pool, seed9.spawn.node_id, seed9.spawn.yaw_deg)
        reply = oracle.choose(pool, seed9, prompt("outside number 11 Rue Oberkampf", collecting=True),
                              cams, pose, {})
        assert reply.endswith("collect()\n```")
        reply = oracle.choose(pool, seed9, prompt("outside number 16 Rue Oberkampf", collecting=False),
                              cams, pose, {})
        assert reply.endswith("hand_over()\n```")

    def test_away_from_the_door_it_points_along_the_certified_route(self, pool, seed9):
        # standing on the spawn node, facing along the approach: the route's
        # 7 m point is in some picture and the reply names it
        spawn = pool.nodes_by_id[seed9.spawn.node_id]
        nxt = pool.nodes_by_id[seed9.approach.node_ids[min(3, len(seed9.approach.node_ids) - 1)]]
        yaw = math.degrees(math.atan2(nxt.y_cm - spawn.y_cm, nxt.x_cm - spawn.x_cm))
        cams, pose = cameras_at(pool, seed9.spawn.node_id, yaw)
        reply = oracle.choose(pool, seed9, prompt("on Rue Oberkampf", collecting=True), cams, pose, {})
        assert "walk_to_pixel(" in reply and "route point" in reply
        u = float(reply.split("u=")[1].split(",")[0]); v = float(reply.split("v=")[1].split(")")[0])
        assert 0.05 <= u <= 0.95 and 0.55 <= v <= 0.93

    def test_its_first_choice_is_a_certified_leg_out_of_its_node(self, pool, seed9):
        """The end of a certified leg from the node it stands on resolves
        again (that pixel from that node is what was certified), so it comes
        before any route node picked by distance -- even 3 m away, low in
        the picture."""
        from tools.pixel_goal_order_pool import plan_pool_legs
        spawn = pool.nodes_by_id[seed9.spawn.node_id]
        _path, plan = plan_pool_legs(pool, seed9.spawn.node_id, seed9.pickup.handover_node_id)
        first = plan[0]
        # face the first leg's end so it sits at the bottom middle of the front view
        yaw = math.degrees(math.atan2(first.aim_cm[1] - spawn.y_cm, first.aim_cm[0] - spawn.x_cm))
        cams, pose = cameras_at(pool, seed9.spawn.node_id, yaw)
        reply = oracle.choose(pool, seed9, prompt("on Rue Oberkampf", collecting=True), cams, pose, {})
        assert "route point leg:" in reply
        chosen = reply.split("route point leg:")[1].split(" in the")[0]
        assert (seed9.spawn.node_id, chosen) in pool.certified_legs
        # the leg chosen leaves no more certified route than the harness's
        # own first move would
        left = {label[4:]: value for label, _xy, value in
                oracle.leg_candidates(pool, seed9.spawn.node_id, seed9.pickup.handover_node_id)}
        assert left[chosen] <= left[first.aim] + 1e-6
        v = float(reply.split("v=")[1].split(")")[0])
        assert oracle.V_RANGE[0] <= v <= oracle.LEG_V_RANGE[1]
        # every leg candidate it lists is a certified leg from that node
        state = {"streak": 0}
        labels = set()
        for streak in range(6):
            state["streak"] = streak
            reply = oracle.choose(pool, seed9, prompt("on Rue Oberkampf", collecting=True, refused=streak > 0),
                                  cams, pose, dict(state))
            labels.add(reply.split("route point ")[1].split(" in the")[0] if "route point " in reply else reply)
        for label in labels:
            if label.startswith("leg:"):
                assert (seed9.spawn.node_id, label[4:]) in pool.certified_legs

    def test_every_certified_leg_that_brings_the_goal_nearer_is_a_candidate(self, pool, seed9):
        from tools.pixel_goal_order_pool import plan_pool_legs
        start, goal = seed9.spawn.node_id, seed9.pickup.handover_node_id
        rows = oracle.leg_candidates(pool, start, goal)
        assert rows, "no certified leg out of the spawn brings the pickup nearer"
        assert all((start, label[4:]) in pool.certified_legs for label, _xy, _left in rows)
        left = [row[2] for row in rows]
        assert left == sorted(left)
        here = plan_pool_legs(pool, start, goal)[0].length_cm
        assert all(value < here - 100.0 for value in left)
        # the harness's own first move is the one that leaves the least
        first = plan_pool_legs(pool, start, goal)[1][0].aim
        assert rows[0][0] == f"leg:{first}" or rows[0][2] <= plan_pool_legs(pool, first, goal)[0].length_cm + 1e-6

    def test_after_a_refusal_it_moves_to_the_next_candidate(self, pool, seed9):
        spawn = pool.nodes_by_id[seed9.spawn.node_id]
        nxt = pool.nodes_by_id[seed9.approach.node_ids[min(3, len(seed9.approach.node_ids) - 1)]]
        yaw = math.degrees(math.atan2(nxt.y_cm - spawn.y_cm, nxt.x_cm - spawn.x_cm))
        cams, pose = cameras_at(pool, seed9.spawn.node_id, yaw)
        state = {}
        first = oracle.choose(pool, seed9, prompt("on Rue Oberkampf", collecting=True), cams, pose, state)
        second = oracle.choose(pool, seed9, prompt("on Rue Oberkampf", collecting=True, refused=True),
                               cams, pose, state)
        assert state["streak"] == 1
        assert first != second

    def test_without_a_capture_it_asks_the_phone(self, pool, seed9):
        reply = oracle.choose(pool, seed9, prompt("on Rue Oberkampf", collecting=True), {}, None, {})
        assert reply.endswith("navigate()\n```")


class TestRelayLoop:
    def test_serve_answers_a_pending_turn_and_stops_at_the_cap(self, tmp_path, pool, seed9):
        relay = tmp_path / "relay"; (relay / "turn-0001").mkdir(parents=True)
        (relay / "turn-0001" / "prompt.txt").write_text(
            prompt("outside number 11 Rue Oberkampf", collecting=True))
        (relay / "pending").write_text("turn-0001")
        events = tmp_path / "events.jsonl"; events.write_text("")
        assert oracle.serve(relay, events, pool, seed9, max_turns=1, timeout_s=5.0, poll_s=0.01) == 0
        assert (relay / "turn-0001" / "answer.txt").read_text().endswith("collect()\n```")

    def test_captures_are_read_from_the_events_file(self, tmp_path):
        events = tmp_path / "events.jsonl"
        rows = [{"function": "PixelGoal_SetupParisPocJson", "response": {}},
                {"function": "PixelGoal_CaptureViewPairJson", "response": {"pose": {"x_cm": 1.0}, "views": []}},
                {"function": "PixelGoal_CaptureViewPairJson", "response": {"pose": {"x_cm": 2.0}, "views": []}}]
        events.write_text("\n".join(json.dumps(r) for r in rows) + "\nnot json\n")
        captures = oracle.captures_from_events(events)
        assert [c["pose"]["x_cm"] for c in captures] == [1.0, 2.0]
        cams, pose = oracle.cameras_of(captures)
        assert cams == {} and pose is None

    def test_the_observation_is_its_last_two_pairs_only_when_the_last_is_the_quarter_turn(self):
        def pair(yaw, x=0.0):
            return {"pose": {"x_cm": x, "y_cm": 0.0, "z_cm": 100.0, "yaw_deg": yaw},
                    "views": [{"camera_view": "front", "camera_location_cm": [x, 0.0, 160.0],
                               "camera_rotation_degrees": [0.0, yaw, 0.0]},
                              {"camera_view": "rear", "camera_location_cm": [x, 0.0, 160.0],
                               "camera_rotation_degrees": [0.0, (yaw + 180.0) % 360.0, 0.0]}]}
        a, b = pair(-53.1), pair(36.9)
        cams, pose = oracle.cameras_of([a, a, a, b])
        assert set(cams) == {"front", "rear", "right", "left"}
        assert cams["front"][1] == -53.1 and cams["right"][1] == 36.9 and pose["yaw_deg"] == -53.1
        # the file read between pair A and pair B: four A pairs, no B yet
        assert oracle.cameras_of([a, a, a]) == ({}, None)
        # a leg's capture after the observation is not a pair B either
        assert oracle.cameras_of([a, b, pair(10.0)]) == ({}, None)
        # a quarter turn from another spot is a different observation
        assert oracle.cameras_of([a, pair(36.9, x=300.0)]) == ({}, None)

    def test_serve_holds_a_turn_until_the_observation_is_complete(self, tmp_path, pool, seed9, monkeypatch):
        relay = tmp_path / "relay"; (relay / "turn-0001").mkdir(parents=True)
        (relay / "turn-0001" / "prompt.txt").write_text(prompt("on Rue Oberkampf", collecting=True))
        (relay / "pending").write_text("turn-0001")
        events = tmp_path / "events.jsonl"; events.write_text("")
        monkeypatch.setattr(oracle, "CAPTURE_GRACE_S", 0.3)
        started = __import__("time").time()
        assert oracle.serve(relay, events, pool, seed9, max_turns=1, timeout_s=5.0, poll_s=0.01) == 0
        # it waited the grace, then gave the turn to a phone check
        assert __import__("time").time() - started >= 0.3
        assert (relay / "turn-0001" / "answer.txt").read_text().endswith("navigate()\n```")
