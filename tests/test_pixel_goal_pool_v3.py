"""A v3 pool carries the engine's verdict on its legs, and the loader holds
it to the pool it sits in: every leg joins two pool nodes, the legs join
every node both ways, and the block's counts match the table."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from tools.pixel_goal_order_pool import (
    ENGINE_CERTIFICATION_METHOD, LEG_EPSILON_CM, MAX_LEG_CM, MIN_LEG_CM, candidate_legs,
    certified_leg_chains, certified_moves, leg_chain, load_validated_delivery_pool,
    nearest_pool_node, strongly_connected_components)

V2 = Path(__file__).resolve().parent.parent / "configs" / "pixel_goal" / \
    "paris_trusted_pedestrian_pool_v2.json"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


@pytest.fixture(scope="module")
def v2():
    return load_validated_delivery_pool(V2)


@pytest.fixture(scope="module")
def legs(v2):
    return candidate_legs(v2)


def _v3_document(legs, **block) -> dict:
    doc = json.loads(V2.read_text())
    doc["version"] = 3
    doc["certified_legs"] = [list(leg) for leg in sorted(legs)]
    certification = {
        "method": ENGINE_CERTIFICATION_METHOD,
        "certified_at": "2026-09-14T12:00:00Z",
        "source_pool_sha256": _sha("v2"),
        "verdicts_sha256": _sha("verdicts"),
        "legs": len(legs), "accepted": len(legs), "refused": 0, "unjudged": 0,
        "removed_edges": [], "removed_nodes": [], "pass_through_nodes": [],
        "placement_tolerance_cm": 15.0, "min_leg_cm": 300.0, "navmesh_adjustment_cm": 100.0,
        "leg_epsilon_cm": 40.0, "max_leg_cm": 1000.0, "max_path_excess_cm": 150.0,
    }
    certification.update(block)
    doc["engine_certification"] = certification
    return doc


def _write(tmp_path: Path, doc: dict) -> Path:
    # the pool names its audit and its buildings source relatively (configs/
    # pixel_goal/audits, ../../vendor/...): mirror that layout with links
    repo = V2.parent.parent.parent
    out_dir = tmp_path / "configs" / "pixel_goal"
    out_dir.mkdir(parents=True)
    (out_dir / "audits").symlink_to(V2.parent / "audits")
    (tmp_path / "vendor").symlink_to(repo / "vendor")
    path = out_dir / "pool_v3.json"
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    return path


class TestTheLegTable:
    def test_a_v3_pool_loads_with_its_legs_and_certification(self, tmp_path, legs):
        pool = load_validated_delivery_pool(_write(tmp_path, _v3_document(legs)))
        assert pool.version == 3
        assert pool.certified_legs == frozenset(legs)
        cert = pool.engine_certification
        assert cert is not None and cert.method == ENGINE_CERTIFICATION_METHOD
        assert cert.legs == cert.accepted == len(legs) and cert.refused == 0
        assert cert.removed_edges == () and cert.removed_nodes == ()
        assert cert.max_path_excess_cm == 150.0
        assert pool.profile == "paris-ue-recast-pedestrian-deliveries/v3"

    def test_the_v2_pool_has_no_legs(self, v2):
        assert v2.certified_legs is None and v2.engine_certification is None

    def test_the_table_and_the_block_come_together(self, tmp_path, legs):
        doc = _v3_document(legs)
        del doc["engine_certification"]
        with pytest.raises(ValueError, match="order pool"):
            load_validated_delivery_pool(_write(tmp_path, doc))

    def test_a_leg_must_join_two_pool_nodes(self, tmp_path, legs):
        doc = _v3_document(legs)
        doc["certified_legs"][0] = ["recast-grid--999-999", doc["certified_legs"][0][1]]
        with pytest.raises(ValueError, match="does not join two distinct pool nodes"):
            load_validated_delivery_pool(_write(tmp_path, doc))

    def test_a_leg_listed_twice_is_refused(self, tmp_path, legs):
        doc = _v3_document(legs)
        doc["certified_legs"].append(doc["certified_legs"][0])
        with pytest.raises(ValueError, match="twice"):
            load_validated_delivery_pool(_write(tmp_path, doc))

    def test_the_count_must_match_the_block(self, tmp_path, legs):
        with pytest.raises(ValueError, match="were accepted"):
            load_validated_delivery_pool(_write(tmp_path, _v3_document(legs, accepted=len(legs) - 1)))

    def test_every_node_must_be_joined_both_ways(self, tmp_path, v2, legs):
        # drop every leg out of one node: it can be reached but not left
        victim = v2.spawns[0].node_id
        kept = {leg for leg in legs if leg[0] != victim}
        with pytest.raises(ValueError, match="both ways"):
            load_validated_delivery_pool(_write(tmp_path, _v3_document(kept)))

    def test_a_removed_edge_or_node_the_pool_still_has_is_refused(self, tmp_path, v2, legs):
        edge = v2.edges[0]
        with pytest.raises(ValueError, match="removed edges the pool still has"):
            load_validated_delivery_pool(_write(tmp_path / "edge", _v3_document(
                legs, removed_edges=[[edge.a, edge.b]])))
        with pytest.raises(ValueError, match="removed nodes the pool still has"):
            load_validated_delivery_pool(_write(tmp_path / "node", _v3_document(
                legs, removed_nodes=[v2.nodes[0].id])))

    def test_an_untrusted_method_is_refused(self, tmp_path, legs):
        with pytest.raises(ValueError, match="not trusted"):
            load_validated_delivery_pool(_write(tmp_path, _v3_document(
                legs, method="engine_resolve_only_straight_runs_v1")))

    def test_a_pass_through_node_is_walked_over_and_never_stopped_on(self, tmp_path, legs):
        # the middle of crossing 92: drop every leg that starts or ends on
        # it, keep the legs that cross over it, and say so in the block
        middle = "crosswalk:PR_Crossswalk_92:100"
        kept = {leg for leg in legs if middle not in leg}
        assert len(kept) < len(legs)
        pool = load_validated_delivery_pool(_write(tmp_path / "pass", _v3_document(
            kept, accepted=len(kept), pass_through_nodes=[middle])))
        assert pool.pass_through_nodes == frozenset({middle})
        assert middle in pool.nodes_by_id
        # the crossing is still one leg end to end, walked over the middle
        ends = ("crosswalk:PR_Crossswalk_92:000", "crosswalk:PR_Crossswalk_92:172")
        assert ends in pool.certified_legs and ends[::-1] in pool.certified_legs
        assert middle in certified_leg_chains(pool)[ends]
        # never a stop, never a destination
        assert all(stop != middle for _start, stop in certified_moves(pool))
        node = pool.nodes_by_id[middle]
        # the crossing's samples are about 2.5 m apart, so the next stop is
        # within 3 m of a point beside the middle
        found = nearest_pool_node(pool, (node.x_cm + 5.0, node.y_cm), radius_cm=300.0, stops_only=True)
        assert found is not None and found[0].id != middle
        assert nearest_pool_node(pool, (node.x_cm + 5.0, node.y_cm), radius_cm=300.0)[0].id == middle

    def test_a_pass_through_node_may_start_no_leg(self, tmp_path, legs):
        middle = "crosswalk:PR_Crossswalk_92:100"
        with pytest.raises(ValueError, match="pass-through"):
            load_validated_delivery_pool(_write(tmp_path / "a", _v3_document(
                legs, pass_through_nodes=[middle])))
        with pytest.raises(ValueError, match="distinct pool nodes"):
            load_validated_delivery_pool(_write(tmp_path / "b", _v3_document(
                legs, pass_through_nodes=["recast-grid--999-999"])))


class TestCandidateLegs:
    """What the engine is asked about: every straight walk of 3-10 m between
    two pool nodes whose chain of nodes stays within 40 cm of the line."""

    def test_the_set_the_engine_judged(self, legs):
        # the count the certification run judged on pool v2; a change here
        # is a change of the leg rule, and the pool must be judged again
        assert len(legs) == 11154

    def test_every_leg_is_a_straight_chain_of_edges(self, v2, legs):
        nodes = v2.nodes_by_id
        edges = {frozenset((e.a, e.b)) for e in v2.edges}
        for (start, end), chain in list(legs.items())[::97]:
            assert chain[0] == start and chain[-1] == end
            a, b = nodes[start].position, nodes[end].position
            chord = math.dist(a, b)
            assert MIN_LEG_CM <= chord <= MAX_LEG_CM
            for x, y in zip(chain, chain[1:]):
                assert frozenset((x, y)) in edges
            ux, uy = (b[0] - a[0]) / chord, (b[1] - a[1]) / chord
            for node_id in chain[1:-1]:
                dx, dy = nodes[node_id].x_cm - a[0], nodes[node_id].y_cm - a[1]
                assert abs(dx * uy - dy * ux) <= LEG_EPSILON_CM
                assert -1e-6 <= dx * ux + dy * uy <= chord + 1e-6

    def test_the_marked_crossing_is_a_leg_both_ways(self, v2, legs):
        a, b = "crosswalk:PR_Crossswalk_92:000", "crosswalk:PR_Crossswalk_92:172"
        assert legs[(a, b)] == (a, "crosswalk:PR_Crossswalk_92:050", "crosswalk:PR_Crossswalk_92:100",
                                "crosswalk:PR_Crossswalk_92:150", b)
        assert legs[(b, a)] == legs[(a, b)][::-1]
        assert leg_chain(v2, a, b) == legs[(a, b)]

    def test_a_pair_the_chain_cannot_follow_straight_is_not_a_leg(self, v2, legs):
        # a node and one three rows over and one along: no chain of pavement
        # nodes stays within 40 cm of that line
        assert ("recast-grid--193-9", "recast-grid--190-12") not in legs
        assert leg_chain(v2, "recast-grid--193-9", "recast-grid--190-12") is None
        # too short, too long, or the same node: never a leg
        assert leg_chain(v2, "recast-grid--193-9", "recast-grid--193-10") is None
        assert leg_chain(v2, "recast-grid--193-9", "recast-grid--193-9") is None

    def test_every_door_and_crossing_node_has_legs_both_ways(self, v2, legs):
        for node in v2.nodes:
            if node.role == "recast_grid":
                continue
            assert any(start == node.id for start, _end in legs), node.id
            assert any(end == node.id for _start, end in legs), node.id

    def test_the_legs_join_every_node_both_ways(self, v2, legs):
        components = strongly_connected_components([n.id for n in v2.nodes], list(legs))
        assert len(components) == 1 and len(components[0]) == len(v2.nodes)


class TestMoves:
    def test_a_move_inside_a_leg_aims_at_the_shortest_leg_through_its_stop(self, tmp_path, legs):
        pool = load_validated_delivery_pool(_write(tmp_path, _v3_document(legs)))
        chains = certified_leg_chains(pool)
        moves = certified_moves(pool)
        nodes = pool.nodes_by_id
        for (start, end), chain in list(chains.items())[::211]:
            assert moves[(start, end)] == end
            for stop in chain[1:-1]:
                aim = moves[(start, stop)]
                assert stop in chains[(start, aim)]
                # nothing shorter through the stop
                length = math.dist(nodes[start].position, nodes[aim].position)
                for (s2, e2), c2 in chains.items():
                    if s2 == start and stop in c2[1:] and e2 != stop:
                        assert math.dist(nodes[s2].position, nodes[e2].position) >= length - 1e-9

    def test_components_of_a_small_graph(self):
        components = strongly_connected_components(
            ["a", "b", "c", "d"], [("a", "b"), ("b", "a"), ("b", "c"), ("c", "d")])
        assert components == [{"a", "b"}, {"c"}, {"d"}]
