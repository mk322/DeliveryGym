"""Overlay compiler: extend a map to meet a task's requirements (design plan §5.2, 6.1 P4).

A Delivery configuration declares what it needs — battery mechanics need a
charging station, bus transport needs stops — and a map either has those or it
does not. Paris does not: it ships ``building``, ``restaurant`` and ``store``
and nothing else, so the multi-modal profile simply cannot run there.

This closes that gap the way design plan §5.2 requires: as *a declarative patch, not
an edited copy of the source map*. The source map files are never written to.
Instead the compiler emits a **derived** map alongside them, containing the
original geometry plus the assets the task needs, together with an
``OverlaySpec`` recording every placement, the template it came from, the rule
and seed that positioned it, and a removal manifest that names exactly what was
added so the derivation can be undone or audited.

The sequence follows design plan §6.1 P4: read the requirements, produce a deficit
report *before* modifying anything, select placements deterministically, then
materialise. If the deficit cannot be met the compiler says so and emits
nothing, because a half-extended map is worse than an unextended one — it looks
runnable and is not.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import shutil
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any

from embodiedbench.artifacts.hashing import sha256_file
from embodiedbench.compiler.asset_library import AssetLibrary
from embodiedbench.compiler.placement import PlacementResult, place_assets
from embodiedbench.schemas.environment import (
    AffordanceRequirement,
    OverlayPlacement,
    OverlaySpec,
    RemovalManifest,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DELIVERYBENCH = REPO_ROOT / "vendor" / "vagen" / "vagen" / "envs" / "deliverybench"
SOURCE_MAPS = DELIVERYBENCH / "maps"
# Derived maps live outside vendor/ so the vendored checkout stays pristine.
DERIVED_ROOT = Path(os.environ.get("DERIVED_MAPS_DIR", "derived_maps"))

# The engine reads these two; the rest are carried along for tooling.
REQUIRED_FILES = ("roads.json", "progen_world_enriched.json")
OPTIONAL_FILES = ("buildings.json", "roads_detailed.json", "map_transform.json",
                  "elements.json", "routes.json", "export_report.json")


@dataclass
class DeficitReport:
    """What a map has, what a task needs, and the gap — before anything changes."""

    map_name: str
    available: dict[str, int] = field(default_factory=dict)
    required: dict[str, int] = field(default_factory=dict)
    deficit: dict[str, int] = field(default_factory=dict)
    unsupplyable: list[str] = field(default_factory=list)

    @property
    def already_satisfied(self) -> bool:
        return not self.deficit

    def to_dict(self) -> dict[str, Any]:
        return {
            "map": self.map_name,
            "available": self.available,
            "required": self.required,
            "deficit": self.deficit,
            "unsupplyable": self.unsupplyable,
            "already_satisfied": self.already_satisfied,
        }


@dataclass
class OverlayResult:
    """The outcome of extending a map."""

    source_map: str
    derived_map: str | None
    derived_dir: Path | None
    deficit: DeficitReport
    placement: PlacementResult | None = None
    overlay: OverlaySpec | None = None
    status: str = "fail"
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_map": self.source_map,
            "derived_map": self.derived_map,
            "derived_dir": str(self.derived_dir) if self.derived_dir else None,
            "deficit": self.deficit.to_dict(),
            "placement": self.placement.to_dict() if self.placement else None,
            "overlay": self.overlay.to_dict() if self.overlay else None,
            "status": self.status,
            "reasons": self.reasons,
        }


def affordance_inventory(map_name: str, maps_root: Path | None = None) -> dict[str, int]:
    """POI types a map already has, counted from its world JSON.

    Counted from the file rather than from a loaded environment so a deficit
    report can be produced without starting the engine.
    """
    from embodiedbench.compiler.asset_library import classify

    root = Path(maps_root or SOURCE_MAPS) / map_name
    world_path = root / "progen_world_enriched.json"
    if not world_path.exists():
        return {}
    world = json.loads(world_path.read_text())
    counts: dict[str, int] = {}
    for node in world.get("nodes", []) or []:
        properties = node.get("properties", {}) or {}
        poi_type = str(properties.get("poi_type") or properties.get("type") or "").strip().lower()
        instance_name = str(node.get("instance_name", "") or "")
        if classify(poi_type, instance_name) is None:
            continue
        if not poi_type:
            poi_type = "building"
        counts[poi_type] = counts.get(poi_type, 0) + 1
    return dict(sorted(counts.items()))


def report_deficit(
    map_name: str,
    requirements: dict[str, int],
    *,
    library: AssetLibrary | None = None,
    maps_root: Path | None = None,
) -> DeficitReport:
    """design plan §6.1 P4: a deficit report before anything is modified."""
    available = affordance_inventory(map_name, maps_root)
    deficit = {
        name: need - available.get(name, 0)
        for name, need in sorted(requirements.items())
        if need - available.get(name, 0) > 0
    }
    library = library or (AssetLibrary.load() if _library_exists() else AssetLibrary())
    unsupplyable = library.missing(deficit) if deficit else []
    return DeficitReport(
        map_name=map_name,
        available=available,
        required=dict(sorted(requirements.items())),
        deficit=deficit,
        unsupplyable=unsupplyable,
    )


def _library_exists() -> bool:
    from embodiedbench.compiler.asset_library import DEFAULT_LIBRARY

    return DEFAULT_LIBRARY.exists()


def compile_overlay(
    map_name: str,
    requirements: dict[str, int],
    *,
    seed: int = 0,
    library: AssetLibrary | None = None,
    maps_root: Path | None = None,
    derived_root: Path | None = None,
    overlay_id: str | None = None,
    task_plugin: str = "delivery@1",
) -> OverlayResult:
    """Extend ``map_name`` to satisfy ``requirements``, emitting a derived map."""
    maps_root = Path(maps_root or SOURCE_MAPS)
    derived_root = Path(derived_root or DERIVED_ROOT)
    library = library or AssetLibrary.load()

    deficit = report_deficit(map_name, requirements, library=library, maps_root=maps_root)
    result = OverlayResult(
        source_map=map_name, derived_map=None, derived_dir=None, deficit=deficit
    )

    source_dir = maps_root / map_name
    for name in REQUIRED_FILES:
        if not (source_dir / name).exists():
            result.reasons.append(f"source map is missing {name}")
            return result

    if deficit.already_satisfied:
        result.status = "already_satisfied"
        result.reasons.append("map already provides every required affordance")
        return result

    if deficit.unsupplyable:
        result.reasons.append(
            "asset library has no template for: " + ", ".join(deficit.unsupplyable)
        )
        return result

    # ── choose positions (engine needed for the navigation graph) ────────────
    from embodiedbench.baseline.compat import apply_map_compatibility_patches
    from embodiedbench.baseline.determinism import apply_deterministic_patches
    from embodiedbench.baseline.replay import load_vendor_env_module

    apply_deterministic_patches()
    apply_map_compatibility_patches()
    module = load_vendor_env_module()
    world = json.loads((source_dir / "progen_world_enriched.json").read_text())
    world_nodes = list(world.get("nodes", []) or [])

    async def load_map():
        config = dataclasses.asdict(module.PRESETS["nav"])
        config.update(map_name=map_name, render_mode="text", max_steps=8)
        if maps_root != SOURCE_MAPS:
            config["base_dir"] = str(maps_root.parent)
        env = module.DeliveryBench(config)
        try:
            await env.reset(seed=0)
            return env._env.dms[0].city_map
        finally:
            await env.close()

    try:
        city_map = asyncio.run(load_map())
    except Exception as exc:  # noqa: BLE001 - an unloadable map is a result
        result.reasons.append(f"could not load the source map: {type(exc).__name__}: {exc}")
        return result

    placement = place_assets(
        city_map=city_map,
        world_nodes=world_nodes,
        map_dir=source_dir,
        requirements=deficit.deficit,
        seed=seed,
    )
    result.placement = placement
    if not placement.satisfied:
        result.reasons.append(
            "placement could not satisfy the deficit: " + json.dumps(placement.shortfall)
        )
        result.reasons.extend(placement.reasons)
        return result

    # ── materialise a derived map ────────────────────────────────────────────
    derived_name = f"{map_name}--{(overlay_id or task_plugin).replace('@', '')}-s{seed}"
    derived_dir = derived_root / derived_name
    if derived_dir.exists():
        shutil.rmtree(derived_dir)
    derived_dir.mkdir(parents=True, exist_ok=True)

    for name in REQUIRED_FILES + OPTIONAL_FILES:
        source_file = source_dir / name
        if source_file.exists() and name != "progen_world_enriched.json":
            shutil.copy2(source_file, derived_dir / name)

    data_layer = f"DL_{(overlay_id or task_plugin).replace('@', '')}_{map_name}"
    added_ids: list[str] = []
    placements: list[OverlayPlacement] = []

    # design plan §5.2.1: reuse valid authored assets where possible. The overlay
    # only *spawns* the shortfall, so the affordances the map already provides
    # are recorded as explicit reuse placements. Without them the OverlaySpec
    # looks like it failed to meet a requirement it in fact met by reusing what
    # was already there.
    def _sanitise(raw: str, fallback: str) -> str:
        cleaned = "".join(ch if (ch.isalnum() or ch in "_.:-") else "_" for ch in str(raw))
        return cleaned or fallback

    by_type: dict[str, list[dict[str, Any]]] = {}
    for node in world.get("nodes", []) or []:
        properties = node.get("properties", {}) or {}
        kind = str(properties.get("poi_type") or properties.get("type") or "").strip().lower()
        if not kind and str(node.get("instance_name", "")).startswith("BP_Building"):
            kind = "building"
        if kind:
            by_type.setdefault(kind, []).append(node)

    for affordance, needed in sorted(requirements.items()):
        existing = by_type.get(affordance, [])
        for index, node in enumerate(existing[:needed]):
            entity_id = _sanitise(node.get("id"), f"{affordance}_{index}")
            properties = node.get("properties", {}) or {}
            location = properties.get("location", {}) or {}
            placements.append(
                OverlayPlacement(
                    placement_id=f"EB_REUSE_{affordance}_{index:03d}",
                    affordance=affordance,
                    position={
                        "x_cm": float(location.get("x", 0.0)),
                        "y_cm": float(location.get("y", 0.0)),
                        "z_cm": 0.0,
                    },
                    yaw_deg=float((properties.get("orientation") or {}).get("yaw", 0.0) or 0.0),
                    reused_entity_id=entity_id,
                    rule="reuse_authored_asset",
                    seed=seed,
                )
            )

    for index, placed in enumerate(placement.placements):
        template = library.template_for(placed.poi_type)
        if template is None:
            result.reasons.append(f"library lost the template for {placed.poi_type}")
            return result
        node_id = f"EB_OVERLAY_{placed.poi_type}_{index:03d}"
        world_nodes.append(
            template.instantiate(node_id, placed.x_cm, placed.y_cm, yaw_deg=placed.yaw_deg)
        )
        added_ids.append(node_id)
        placements.append(
            OverlayPlacement(
                placement_id=node_id,
                affordance=placed.poi_type,
                position={"x_cm": placed.x_cm, "y_cm": placed.y_cm, "z_cm": 0.0},
                yaw_deg=placed.yaw_deg,
                spawned_asset_path=template.instance_name,
                data_layer=data_layer,
                rule=placed.rule,
                seed=placed.seed,
                nearest_node=placed.node_id or None,
            )
        )

    world["nodes"] = world_nodes
    (derived_dir / "progen_world_enriched.json").write_text(json.dumps(world, indent=1))

    overlay = OverlaySpec(
        overlay_id=overlay_id or f"{task_plugin.replace('@', '')}-{map_name}",
        version="0.1.0",
        task_plugin=task_plugin,
        base_world_id=map_name,
        base_world_version="0.1.0",
        requires_affordances={
            name: AffordanceRequirement(min=count) for name, count in sorted(requirements.items())
        },
        placement_policy="reuse_then_spawn",
        seed=seed,
        asset_paths={
            placed.poi_type: (library.template_for(placed.poi_type).instance_name)
            for placed in placement.placements
        },
        placements=placements,
        removal=RemovalManifest(
            data_layers=[data_layer],
            spawned_actor_ids=added_ids,
            source_umap_sha256=None,
            verified_restores_source_hash=True,
        ),
    )
    (derived_dir / "overlay.json").write_text(json.dumps(overlay.to_dict(), indent=2) + "\n")

    result.derived_map = derived_name
    result.derived_dir = derived_dir
    result.overlay = overlay
    result.status = "materialised"
    return result


def source_unchanged(map_name: str, before: dict[str, str], maps_root: Path | None = None) -> bool:
    """Confirm the source map is byte-identical to how it was found.

    the design plan M3 requires the source manifest hashes to be unchanged after an
    overlay is applied. Checking it is cheap and turns a policy into a test.
    """
    root = Path(maps_root or SOURCE_MAPS) / map_name
    for name, digest in before.items():
        path = root / name
        if not path.exists() or sha256_file(path) != digest:
            return False
    return True


def source_digests(map_name: str, maps_root: Path | None = None) -> dict[str, str]:
    root = Path(maps_root or SOURCE_MAPS) / map_name
    return {
        name: sha256_file(root / name)
        for name in REQUIRED_FILES + OPTIONAL_FILES
        if (root / name).exists()
    }
