"""The verdict-to-pool step: accepted legs make the table, a node the legs
do not join both ways goes, an edge no kept leg walks goes, the result
loads as a v3 pool, and an unjudged leg stops the tool."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import prune_pixel_goal_pool as prune
from tools.certify_pixel_goal_pool_live import STALE, leg_id
from tools.pixel_goal_order_pool import (
    PROTOCOL_CONSTRAINTS, candidate_legs, candidate_scenarios, load_validated_delivery_pool)

V2 = Path(__file__).resolve().parent.parent / "configs" / "pixel_goal" / \
    "paris_trusted_pedestrian_pool_v2.json"


@pytest.fixture(scope="module")
def pool():
    return load_validated_delivery_pool(V2)


@pytest.fixture(scope="module")
def legs(pool):
    return candidate_legs(pool)


def _layout(tmp_path: Path) -> Path:
    repo = V2.parent.parent.parent
    out_dir = tmp_path / "configs" / "pixel_goal"
    out_dir.mkdir(parents=True)
    (out_dir / "audits").symlink_to(V2.parent / "audits")
    (tmp_path / "vendor").symlink_to(repo / "vendor")
    return out_dir


def _verdicts(tmp_path: Path, pool, legs, *, rejected=(), skip=(), bent=(), stale=()) -> Path:
    nodes = pool.nodes_by_id
    path = tmp_path / "verdicts.jsonl"
    with path.open("w") as sink:
        for (start, end), chain in legs.items():
            if (start, end) in skip:
                continue
            length = round(((nodes[start].x_cm - nodes[end].x_cm) ** 2
                            + (nodes[start].y_cm - nodes[end].y_cm) ** 2) ** 0.5, 1)
            row = {"leg": leg_id(start, end), "start": start, "end": end, "chain": list(chain),
                   "length_cm": length}
            if (start, end) in rejected:
                row.update(verdict="rejected", reason="controller_path_enters_unmarked_road")
            elif (start, end) in stale:
                row.update(verdict="rejected", reason=STALE)
            else:
                row.update(verdict="accepted", controller_path_length_cm=length + (
                    400.0 if (start, end) in bent else 20.0))
            sink.write(json.dumps(row) + "\n")
    return path


def _triples(pool):
    return sorted((spawn.id, pickup.id, dropoff.id)
                  for spawn, pickup, dropoff, _a, _d in candidate_scenarios(
                      pool, constraints=PROTOCOL_CONSTRAINTS))


def test_every_leg_accepted_makes_a_v3_pool_with_the_whole_table(tmp_path, pool, legs):
    verdicts = _verdicts(tmp_path, pool, legs)
    out = _layout(tmp_path) / "pool_v3.json"
    assert prune.main(["--pool", str(V2), "--verdicts", str(verdicts), "--out", str(out),
                       "--certified-at", "2026-09-14T12:00:00Z"]) == 0
    v3 = load_validated_delivery_pool(out)
    assert v3.version == 3 and v3.certified_legs == frozenset(legs)
    cert = v3.engine_certification
    assert cert.legs == cert.accepted == len(legs) and cert.refused == 0 and cert.unjudged == 0
    assert cert.source_pool_sha256 == pool.sha256 and cert.certified_at == "2026-09-14T12:00:00Z"
    assert len(v3.nodes) == len(pool.nodes) and cert.removed_nodes == ()
    # an edge no leg walks over (a stub too short to be part of any straight
    # 3 m walk) is not a way of the pool any more
    covered = {frozenset(pair) for chain in legs.values() for pair in zip(chain, chain[1:])}
    assert len(v3.edges) == len(covered) < len(pool.edges)
    assert len(cert.removed_edges) == len(pool.edges) - len(covered)
    # the protocol's sixteen scenarios are the same orders from the same spawns
    assert _triples(v3) == _triples(pool)


def test_a_node_the_legs_do_not_join_both_ways_is_walked_over_or_goes(tmp_path, pool, legs):
    # the middle of crossing 92: every leg from or to it refused, the legs
    # across it accepted -- the engine crosses end to end but will not aim
    # at the middle. It stays as a pass-through node: on the chains, never
    # a stop.
    middle = "crosswalk:PR_Crossswalk_92:100"
    rejected = {leg for leg in legs if middle in leg}
    # and a pavement node nothing accepted walks over any more: it goes,
    # with its edges
    island = "recast-grid--202-6"
    rejected |= {leg for leg in legs if island in legs[leg]}
    verdicts = _verdicts(tmp_path, pool, legs, rejected=rejected)
    out = _layout(tmp_path) / "pool_v3.json"
    assert prune.main(["--pool", str(V2), "--verdicts", str(verdicts), "--out", str(out)]) == 0
    v3 = load_validated_delivery_pool(out)
    cert = v3.engine_certification
    assert cert.refused == len(rejected)
    assert middle in v3.nodes_by_id and cert.pass_through_nodes == (middle,)
    assert all(middle not in leg for leg in v3.certified_legs)
    assert any(middle in legs[leg] for leg in v3.certified_legs)
    assert island not in v3.nodes_by_id and island in cert.removed_nodes
    assert all(island not in (e.a, e.b) for e in v3.edges)
    assert v3.certified_legs == frozenset(leg for leg in legs if leg not in rejected)
    stops = {n.id for n in v3.nodes} - {middle}
    assert stops == {node for leg in v3.certified_legs for node in leg}
    assert all(e.a in v3.nodes_by_id and e.b in v3.nodes_by_id for e in v3.edges)


def test_losing_a_stop_or_spawn_stops_the_tool(tmp_path, pool, legs):
    stop_node = pool.stops[0].handover_node_id
    rejected = {leg for leg in legs if leg[0] == stop_node}
    verdicts = _verdicts(tmp_path, pool, legs, rejected=rejected)
    out = _layout(tmp_path) / "pool_v3.json"
    with pytest.raises(SystemExit, match="both ways"):
        prune.main(["--pool", str(V2), "--verdicts", str(verdicts), "--out", str(out)])
    assert not out.exists()


def test_an_unjudged_leg_stops_the_tool_unless_allowed(tmp_path, pool, legs):
    missing = next(iter(legs))
    verdicts = _verdicts(tmp_path, pool, legs, skip={missing})
    out = _layout(tmp_path) / "pool_v3.json"
    assert prune.main(["--pool", str(V2), "--verdicts", str(verdicts), "--out", str(out)]) == 2
    assert not out.exists()
    assert prune.main(["--pool", str(V2), "--verdicts", str(verdicts), "--out", str(out),
                       "--allow-unjudged"]) == 0
    v3 = load_validated_delivery_pool(out)
    assert missing not in v3.certified_legs and v3.engine_certification.unjudged == 1


def test_a_stale_snapshot_refusal_is_not_a_verdict(tmp_path, pool, legs):
    victim = next(iter(legs))
    verdicts = _verdicts(tmp_path, pool, legs, stale={victim})
    out = _layout(tmp_path) / "pool_v3.json"
    assert prune.main(["--pool", str(V2), "--verdicts", str(verdicts), "--out", str(out)]) == 2


def test_an_accepted_leg_the_controller_bends_is_not_a_straight_walk(tmp_path, pool, legs):
    victim = next(iter(legs))
    verdicts = _verdicts(tmp_path, pool, legs, bent={victim})
    out = _layout(tmp_path) / "pool_v3.json"
    assert prune.main(["--pool", str(V2), "--verdicts", str(verdicts), "--out", str(out)]) == 0
    v3 = load_validated_delivery_pool(out)
    assert victim not in v3.certified_legs
    assert v3.engine_certification.refused == 1
    # a wider allowance keeps it
    assert prune.main(["--pool", str(V2), "--verdicts", str(verdicts), "--out", str(out),
                       "--max-path-excess-cm", "500"]) == 0
    assert victim in load_validated_delivery_pool(out).certified_legs


def test_a_later_verdict_on_the_same_leg_replaces_the_earlier(tmp_path, pool, legs):
    victim = next(iter(legs))
    verdicts = _verdicts(tmp_path, pool, legs, rejected={victim})
    with verdicts.open("a") as sink:
        row = json.loads(verdicts.read_text().splitlines()[0])
        row.update(verdict="accepted", controller_path_length_cm=row["length_cm"] + 10.0)
        row.pop("reason", None)
        sink.write(json.dumps(row) + "\n")
    rows = prune.read_verdicts(verdicts)
    assert rows[leg_id(*victim)]["verdict"] == "accepted"
