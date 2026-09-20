"""Asset library, placement engine, and overlay compiler.

These three turn "this map cannot host this task" into "this map can", so the
tests are mostly about the ways that could go wrong quietly: an asset placed
inside a building, three bus stops on one corner, a derived map that silently
edits its source, or an overlay that claims success while leaving the
requirement unmet.

Placement is checked by geometry rather than by trusting the engine's report --
a placement is verified reachable by matching a real graph node, and verified
clear by testing the building footprints directly.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from embodiedbench.compiler.asset_library import (
    BUILDING_LIKE,
    POINT_LIKE,
    AssetLibrary,
    AssetTemplate,
    classify,
    extract_templates,
)
from embodiedbench.compiler.overlay import (
    affordance_inventory,
    compile_overlay,
    report_deficit,
    source_digests,
    source_unchanged,
)
from embodiedbench.compiler.placement import (
    Candidate,
    farthest_point_order,
    inside_building,
    load_building_footprints,
    place_assets,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MAPS = REPO_ROOT / "vendor" / "vagen" / "vagen" / "envs" / "deliverybench" / "maps"

pytestmark = pytest.mark.skipif(not MAPS.exists(), reason="vendored maps not present")


# ─────────────────────────────────────────────────────────────────────────────
# Asset library
# ─────────────────────────────────────────────────────────────────────────────


def test_classification_matches_the_engine_vocabulary():
    """A template the engine would drop is worse than no template."""
    for name in BUILDING_LIKE:
        assert classify(name, "") == "building"
    for name in POINT_LIKE:
        assert classify(name, "") == "point"
    assert classify("", "BP_Building_01_C") == "building"
    # The engine drops these, so we must too.
    assert classify("pedestrian_light", "RT_BP_street_light_ped") is None
    assert classify("boutique", "SM_Shop_A") is None


def test_extraction_finds_the_types_a_map_actually_has():
    templates = {t.poi_type for t in extract_templates("small-city-11")}
    assert {"bus_station", "charging_station", "restaurant", "store", "building"} <= templates


def test_paris_lacks_the_types_the_library_must_supply():
    """The gap that motivates the whole overlay path."""
    paris = {t.poi_type for t in extract_templates("citycore-paris")}
    assert "bus_station" not in paris
    assert "charging_station" not in paris
    assert {"building", "restaurant", "store"} <= paris


def test_every_template_carries_provenance():
    library = AssetLibrary.build(["small-city-11", "large-city-26"])
    assert library.templates
    for template in library.templates:
        assert template.source_map, template.poi_type
        assert len(template.source_sha256) == 64
        assert template.observed_count >= 1
        assert template.instance_name


def test_library_prefers_the_better_evidenced_template():
    """Between two maps offering a type, the one seen more often should win."""
    library = AssetLibrary.build(["small-city-11", "large-city-26"])
    charging = library.template_for("charging_station")
    assert charging is not None
    solo = extract_templates("small-city-11")
    solo_count = next(
        (t.observed_count for t in solo if t.poi_type == "charging_station"), 0
    )
    assert charging.observed_count >= solo_count


def test_library_round_trips(tmp_path):
    library = AssetLibrary.build(["small-city-11"])
    path = library.save(tmp_path / "lib.json")
    restored = AssetLibrary.load(path)
    assert restored.types() == library.types()
    assert restored.to_dict() == library.to_dict()


def test_library_reports_what_it_cannot_supply():
    library = AssetLibrary.build(["citycore-paris"])
    missing = library.missing({"bus_station": 2, "restaurant": 1})
    assert "bus_station" in missing
    assert "restaurant" not in missing


def test_instantiated_node_matches_the_engine_shape():
    """An instantiated asset must be indistinguishable from an authored one."""
    template = AssetLibrary.build(["small-city-11"]).template_for("bus_station")
    assert template is not None
    node = template.instantiate("EB_TEST_0", 1234.0, -567.0, yaw_deg=90.0)
    assert node["instance_name"] == template.instance_name
    properties = node["properties"]
    assert properties["poi_type"] == "bus_station"
    assert properties["location"] == {"x": 1234.0, "y": -567.0, "z": 0.0}
    assert properties["orientation"]["yaw"] == 90.0
    # And the engine would keep it.
    assert classify(properties["poi_type"], node["instance_name"]) is not None


# ─────────────────────────────────────────────────────────────────────────────
# Placement rules
# ─────────────────────────────────────────────────────────────────────────────


def test_building_footprints_load_and_reject_interior_points():
    boxes = load_building_footprints(MAPS / "small-city-11")
    assert boxes, "small-city-11 should have building footprints"
    min_x, min_y, max_x, max_y = boxes[0]
    centre_x, centre_y = (min_x + max_x) / 2, (min_y + max_y) / 2
    assert inside_building(centre_x, centre_y, boxes)
    assert not inside_building(min_x - 1e6, min_y - 1e6, boxes)


def test_farthest_point_order_spreads_rather_than_clusters():
    """Two tight clusters: the first picks must come from different clusters."""
    candidates = (
        [Candidate(x_cm=float(i), y_cm=0.0, node_id=f"a{i}") for i in range(20)]
        + [Candidate(x_cm=100000.0 + i, y_cm=0.0, node_id=f"b{i}") for i in range(20)]
    )
    ordered = farthest_point_order(candidates, seed=0)
    first_two = ordered[:2]
    assert abs(first_two[0].x_cm - first_two[1].x_cm) > 50000.0


def test_farthest_point_order_is_deterministic():
    candidates = [Candidate(x_cm=float(i), y_cm=float(i % 7), node_id=f"n{i}") for i in range(40)]
    a = [c.node_id for c in farthest_point_order(candidates, seed=3)]
    b = [c.node_id for c in farthest_point_order(candidates, seed=3)]
    assert a == b


def test_seed_controls_the_layout():
    """Across seeds the layout varies.

    Asserted over a range rather than on one pair: only the first pick is drawn
    from the seed, so two particular seeds can legitimately collide (3 and 4 both
    draw index 15 out of 40). The property is that the seed controls the layout,
    not that every adjacent pair differs.
    """
    candidates = [Candidate(x_cm=float(i), y_cm=float(i % 7), node_id=f"n{i}") for i in range(40)]
    orderings = {
        tuple(c.node_id for c in farthest_point_order(candidates, seed=s)) for s in range(8)
    }
    assert len(orderings) >= 5, f"only {len(orderings)} distinct layouts across 8 seeds"


def test_empty_candidate_list_is_handled():
    assert farthest_point_order([], seed=0) == []


# ─────────────────────────────────────────────────────────────────────────────
# Placement against a real map
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def paris_env():
    """A loaded Paris map, shared across placement tests (loading is slow)."""
    import asyncio
    import dataclasses

    from embodiedbench.baseline.compat import apply_map_compatibility_patches
    from embodiedbench.baseline.determinism import apply_deterministic_patches
    from embodiedbench.baseline.replay import load_vendor_env_module

    apply_deterministic_patches()
    apply_map_compatibility_patches()
    module = load_vendor_env_module()

    async def load():
        config = dataclasses.asdict(module.PRESETS["nav"])
        config.update(map_name="citycore-paris", render_mode="text", max_steps=8)
        env = module.DeliveryBench(config)
        await env.reset(seed=0)
        return env

    env = asyncio.run(load())
    city_map = env._env.dms[0].city_map
    world = json.loads((MAPS / "citycore-paris" / "progen_world_enriched.json").read_text())
    yield city_map, world.get("nodes", [])
    asyncio.run(env.close())


def test_placements_are_reachable_graph_nodes(paris_env):
    """Reachability is the whole point of anchoring to the graph."""
    city_map, nodes = paris_env
    result = place_assets(
        city_map=city_map, world_nodes=nodes, map_dir=MAPS / "citycore-paris",
        requirements={"charging_station": 5, "bus_station": 3}, seed=0,
    )
    assert result.satisfied, result.reasons
    graph = {
        (round(float(n.position.x), 1), round(float(n.position.y), 1))
        for n in city_map.waypoint_graph.adjacency_list
    }
    for placement in result.placements:
        key = (round(placement.x_cm, 1), round(placement.y_cm, 1))
        assert key in graph, f"{placement.poi_type} at {key} is not a graph node"


def test_placements_avoid_building_interiors(paris_env):
    city_map, nodes = paris_env
    boxes = load_building_footprints(MAPS / "citycore-paris")
    result = place_assets(
        city_map=city_map, world_nodes=nodes, map_dir=MAPS / "citycore-paris",
        requirements={"charging_station": 6}, seed=1,
    )
    for placement in result.placements:
        assert not inside_building(placement.x_cm, placement.y_cm, boxes)


def test_same_type_placements_are_spaced_apart(paris_env):
    city_map, nodes = paris_env
    result = place_assets(
        city_map=city_map, world_nodes=nodes, map_dir=MAPS / "citycore-paris",
        requirements={"bus_station": 4}, seed=0,
    )
    points = [(p.x_cm, p.y_cm) for p in result.placements if p.poi_type == "bus_station"]
    assert len(points) == 4
    closest = min(
        math.hypot(a[0] - b[0], a[1] - b[1])
        for i, a in enumerate(points) for b in points[i + 1:]
    )
    # The declared bus_station spacing is 120 m; allow the stated fallback.
    assert closest >= 60.0 * 100.0, f"closest pair only {closest / 100:.0f} m apart"


def test_placement_is_deterministic(paris_env):
    city_map, nodes = paris_env
    kwargs = dict(
        city_map=city_map, world_nodes=nodes, map_dir=MAPS / "citycore-paris",
        requirements={"charging_station": 4, "bus_station": 2}, seed=7,
    )
    a = place_assets(**kwargs).to_dict()["placements"]
    b = place_assets(**kwargs).to_dict()["placements"]
    assert a == b


def test_different_seeds_give_different_layouts(paris_env):
    city_map, nodes = paris_env
    base = dict(
        city_map=city_map, world_nodes=nodes, map_dir=MAPS / "citycore-paris",
        requirements={"charging_station": 4},
    )
    assert (
        place_assets(**base, seed=1).to_dict()["placements"]
        != place_assets(**base, seed=2).to_dict()["placements"]
    )


def test_every_placement_records_its_rule_and_seed(paris_env):
    """design plan §5.2: record every placement, rule and seed."""
    city_map, nodes = paris_env
    result = place_assets(
        city_map=city_map, world_nodes=nodes, map_dir=MAPS / "citycore-paris",
        requirements={"charging_station": 3}, seed=11,
    )
    for placement in result.placements:
        assert placement.rule
        assert placement.seed == 11
        assert placement.node_id


def test_impossible_demand_reports_shortfall_rather_than_lying(paris_env):
    """A map has finitely many nodes; asking for more must fail honestly."""
    city_map, nodes = paris_env
    result = place_assets(
        city_map=city_map, world_nodes=nodes, map_dir=MAPS / "citycore-paris",
        requirements={"charging_station": 100000}, seed=0,
    )
    assert not result.satisfied
    assert result.shortfall.get("charging_station", 0) > 0
    assert any("placed" in reason for reason in result.reasons)


def test_zero_requirement_places_nothing(paris_env):
    city_map, nodes = paris_env
    result = place_assets(
        city_map=city_map, world_nodes=nodes, map_dir=MAPS / "citycore-paris",
        requirements={"charging_station": 0}, seed=0,
    )
    assert result.placements == []
    assert result.satisfied


# ─────────────────────────────────────────────────────────────────────────────
# Overlay compiler
# ─────────────────────────────────────────────────────────────────────────────


def test_affordance_inventory_matches_the_engine_view():
    counts = affordance_inventory("citycore-paris")
    assert counts.get("building", 0) > 400
    assert counts.get("restaurant", 0) == 18
    assert "bus_station" not in counts
    # pedestrian_light is dropped by the engine, so it must not be counted.
    assert "pedestrian_light" not in counts


def test_deficit_report_precedes_any_modification():
    report = report_deficit("citycore-paris", {"bus_station": 2, "restaurant": 1})
    assert report.deficit == {"bus_station": 2}
    assert not report.already_satisfied
    assert report.unsupplyable == []


def test_deficit_is_empty_when_the_map_already_qualifies():
    report = report_deficit("citycore-paris", {"restaurant": 1, "building": 1})
    assert report.already_satisfied


def test_overlay_reports_when_the_library_cannot_supply_a_type(tmp_path):
    empty = AssetLibrary()
    result = compile_overlay(
        "citycore-paris", {"bus_station": 2}, library=empty, derived_root=tmp_path
    )
    assert result.status == "fail"
    assert any("no template" in reason for reason in result.reasons)
    assert result.derived_dir is None


def test_overlay_short_circuits_when_nothing_is_needed(tmp_path):
    result = compile_overlay(
        "citycore-paris", {"restaurant": 1}, derived_root=tmp_path
    )
    assert result.status == "already_satisfied"
    assert result.derived_dir is None


def test_overlay_materialises_and_leaves_the_source_untouched(tmp_path):
    """The rule that matters most: a derived map, never an edited one."""
    before = source_digests("citycore-paris")
    result = compile_overlay(
        "citycore-paris",
        {"bus_station": 2, "charging_station": 1, "restaurant": 1, "building": 1},
        seed=0,
        derived_root=tmp_path,
    )
    assert result.status == "materialised", result.reasons
    assert source_unchanged("citycore-paris", before), "the overlay modified its source map"

    derived = result.derived_dir
    assert (derived / "roads.json").exists()
    assert (derived / "progen_world_enriched.json").exists()
    assert (derived / "overlay.json").exists()

    # The derived world must contain the new assets.
    world = json.loads((derived / "progen_world_enriched.json").read_text())
    types = [
        (n.get("properties") or {}).get("poi_type") for n in world["nodes"]
    ]
    assert types.count("bus_station") >= 2
    assert types.count("charging_station") >= 1


def test_overlay_records_a_removal_manifest(tmp_path):
    """design plan §5.2.6: an inverse manifest so the source stays recoverable."""
    result = compile_overlay(
        "citycore-paris", {"bus_station": 2, "restaurant": 1, "building": 1},
        seed=0, derived_root=tmp_path,
    )
    assert result.overlay is not None
    removal = result.overlay.removal
    assert removal.data_layers
    assert len(removal.spawned_actor_ids) == len(result.placement.placements)
    world = json.loads((result.derived_dir / "progen_world_enriched.json").read_text())
    present = {n["id"] for n in world["nodes"]}
    for actor_id in removal.spawned_actor_ids:
        assert actor_id in present, f"{actor_id} is named for removal but is not in the map"


def test_overlay_distinguishes_reuse_from_spawn(tmp_path):
    """reuse_then_spawn: existing assets are reused, only the gap is spawned."""
    result = compile_overlay(
        "citycore-paris",
        {"bus_station": 2, "restaurant": 1, "building": 1},
        seed=0, derived_root=tmp_path,
    )
    placements = result.overlay.placements
    reused = [p for p in placements if p.reused_entity_id]
    spawned = [p for p in placements if p.spawned_asset_path]
    assert reused, "Paris already has restaurants and buildings; those must be reused"
    assert spawned, "Paris has no bus stops; those must be spawned"
    for placement in reused:
        assert placement.rule == "reuse_authored_asset"


def test_overlay_is_deterministic(tmp_path):
    requirements = {"bus_station": 2, "charging_station": 2, "restaurant": 1, "building": 1}
    a = compile_overlay("citycore-paris", requirements, seed=5, derived_root=tmp_path / "a")
    b = compile_overlay("citycore-paris", requirements, seed=5, derived_root=tmp_path / "b")
    assert a.overlay.content_hash() == b.overlay.content_hash()
    assert (
        json.loads((a.derived_dir / "progen_world_enriched.json").read_text())
        == json.loads((b.derived_dir / "progen_world_enriched.json").read_text())
    )


def test_overlay_spec_validates_against_its_schema(tmp_path):
    from embodiedbench.schemas.environment import OverlaySpec

    result = compile_overlay(
        "citycore-paris", {"bus_station": 2, "restaurant": 1, "building": 1},
        seed=0, derived_root=tmp_path,
    )
    payload = json.loads((result.derived_dir / "overlay.json").read_text())
    restored = OverlaySpec.from_dict(payload)
    assert restored.to_dict() == payload


# ─────────────────────────────────────────────────────────────────────────────
# The vendored checkout and every source map must stay pristine
# ─────────────────────────────────────────────────────────────────────────────


def test_vendor_checkout_has_no_local_modifications():
    """The given maps and the vendored engine are inputs, never targets.

    Checked with git rather than by inspection so it covers files no test knows
    about. Stray writes have happened: the engine resolves its ``outputs/`` and
    ``log/`` directories against the process cwd, so running a tool from inside
    the checkout deposits artefacts there.
    """
    import subprocess

    vendor = REPO_ROOT / "vendor" / "vagen"
    if not (vendor / ".git").exists():
        pytest.skip("vendored checkout is not a git repo")
    proc = subprocess.run(
        ["git", "-C", str(vendor), "status", "--porcelain"],
        capture_output=True, text=True, check=False,
    )
    dirty = [line for line in proc.stdout.splitlines() if line.strip()]
    # The verl submodule pointer is allowed to move, and only that.
    #
    # VAGEN records verl at a commit on main, but .gitmodules names the
    # vagen-lite branch and that is the branch its trainer needs: main's
    # trainer/ppo/reward.py has no compute_reward, so vagen/ray_trainer.py
    # cannot import against it. Putting the submodule on the branch the
    # metadata already names is required setup, not a stray write, and this
    # guard exists to catch stray writes -- the engine drops outputs/ and log/
    # into its own checkout when run from there.
    dirty = [line for line in dirty if line.strip() != "M verl"]
    # The declared patches are the one sanctioned way to write to vendor/:
    # they live in embodiedbench/training/vagen/patches/ precisely because
    # vendor/ is gitignored and an edit made there directly would be invisible
    # to review. Each patch module names its target; a modification of that
    # file is the patch mechanism working, and any other modification is still
    # the stray write this guard exists to catch.
    import importlib
    import pkgutil

    import embodiedbench.training.vagen.patches as patches_pkg

    patched: set[str] = set()
    for module_info in pkgutil.iter_modules(patches_pkg.__path__):
        module = importlib.import_module(
            f"{patches_pkg.__name__}.{module_info.name}")
        target = getattr(module, "TARGET", None)
        if target is not None:
            patched.add(str(Path(target).relative_to(vendor)))
    dirty = [line for line in dirty
             if line.split(maxsplit=1)[-1] not in patched]
    assert dirty == [], f"vendor/ was written to: {dirty[:5]}"


def test_every_source_map_matches_its_pinned_digest():
    """Byte-level proof that no map was edited in place."""
    import json as _json

    manifest_path = REPO_ROOT / "BASELINE_MANIFEST.json"
    if not manifest_path.exists():
        pytest.skip("no baseline manifest")
    manifest = _json.loads(manifest_path.read_text())
    entry = next(
        (e for e in manifest["entries"] if e["id"] == "vagen_paris_export"), None
    )
    if entry is None:
        pytest.skip("Paris export not pinned")
    from embodiedbench.artifacts.hashing import digest_tree

    pinned = next(d for d in entry["digests"] if d["level"] == "content")
    observed = digest_tree(MAPS / "citycore-paris", level="content")
    assert observed.digest == pinned["digest"], "the Paris source map changed on disk"


def test_derived_maps_are_written_outside_the_source_tree(tmp_path):
    """A derivation must never land next to the map it derives from."""
    result = compile_overlay(
        "citycore-paris", {"bus_station": 1, "restaurant": 1, "building": 1},
        seed=0, derived_root=tmp_path,
    )
    assert result.derived_dir is not None
    assert not str(result.derived_dir).startswith(str(MAPS)), result.derived_dir
    assert tmp_path in result.derived_dir.parents
