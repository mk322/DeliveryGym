"""A versioned library of placeable asset templates, extracted from real maps.

Paris ships ``building``, ``restaurant`` and ``store`` and nothing else. It has
no charging station, bus stop, hospital, rest area or car rental, so a Delivery
task needing any of those simply cannot run there — which is what the EnvSpec
affordance inventory reported. The procgen maps do have them.

Rather than invent assets, this harvests the templates that already work: for
each POI type it records the ``instance_name`` the engine recognises, whether
the type is point-like or building-like, its footprint, and the orientation
convention, together with where it came from. A template is therefore a
*citation*, not a guess, and every asset later placed into a map can be traced
to the map it was learned from.

Everything is rule-based. Classification follows the engine's own vocabulary
(``import_pois`` keeps ``poi_type`` in its building-like or point-like set, or
an ``instance_name`` starting with ``BP_Building``), so a template that this
module emits is by construction one the runtime will accept. No model is
involved and no heuristic guesses a type.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from embodiedbench.artifacts.hashing import sha256_file

REPO_ROOT = Path(__file__).resolve().parents[2]
DELIVERYBENCH = REPO_ROOT / "vendor" / "vagen" / "vagen" / "envs" / "deliverybench"
DEFAULT_LIBRARY = REPO_ROOT / "assets" / "asset_library.json"

LIBRARY_SCHEMA = "embodiedbench/asset_library/v0.1"

# The engine's own vocabulary (vlm_delivery/map/map.py::import_pois). A type
# outside these sets is silently dropped at load, so emitting a template for one
# would produce an asset the runtime ignores.
BUILDING_LIKE = frozenset(
    {"restaurant", "store", "rest_area", "hospital", "car_rental", "customer", "building"}
)
POINT_LIKE = frozenset({"charging_station", "bus_station"})


@dataclass
class AssetTemplate:
    """One placeable asset type, learned from a map that already uses it."""

    poi_type: str
    instance_name: str
    kind: str  # "point" | "building"
    bbox_cm: list[float] = field(default_factory=list)  # [x, y, z] when building-like
    default_yaw_deg: float = 0.0
    observed_yaws: list[float] = field(default_factory=list)
    source_map: str = ""
    source_sha256: str = ""
    observed_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        out = {
            "poi_type": self.poi_type,
            "instance_name": self.instance_name,
            "kind": self.kind,
            "default_yaw_deg": self.default_yaw_deg,
            "observed_yaws": self.observed_yaws,
            "source_map": self.source_map,
            "source_sha256": self.source_sha256,
            "observed_count": self.observed_count,
        }
        if self.bbox_cm:
            out["bbox_cm"] = self.bbox_cm
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AssetTemplate":
        return cls(
            poi_type=data["poi_type"],
            instance_name=data["instance_name"],
            kind=data["kind"],
            bbox_cm=list(data.get("bbox_cm") or []),
            default_yaw_deg=float(data.get("default_yaw_deg", 0.0)),
            observed_yaws=list(data.get("observed_yaws") or []),
            source_map=data.get("source_map", ""),
            source_sha256=data.get("source_sha256", ""),
            observed_count=int(data.get("observed_count", 0)),
        )

    def instantiate(
        self, node_id: str, x_cm: float, y_cm: float, *, yaw_deg: float | None = None
    ) -> dict[str, Any]:
        """Build a world-JSON node for this template at a position.

        The shape mirrors what the engine reads, so a placed asset is
        indistinguishable from an authored one at load time.
        """
        properties: dict[str, Any] = {
            "poi_type": self.poi_type,
            "location": {"x": float(x_cm), "y": float(y_cm), "z": 0.0},
            "orientation": {
                "pitch": 0.0,
                "yaw": float(self.default_yaw_deg if yaw_deg is None else yaw_deg),
                "roll": 0.0,
            },
        }
        if self.bbox_cm:
            properties["bbox"] = {
                "x": self.bbox_cm[0],
                "y": self.bbox_cm[1],
                "z": self.bbox_cm[2] if len(self.bbox_cm) > 2 else 1000.0,
            }
        return {"id": node_id, "instance_name": self.instance_name, "properties": properties}


def classify(poi_type: str, instance_name: str) -> str | None:
    """The engine's classification, reproduced exactly. None means it is dropped."""
    normalized = (poi_type or "").strip().lower()
    if normalized in BUILDING_LIKE:
        return "building"
    if normalized in POINT_LIKE:
        return "point"
    if instance_name.startswith("BP_Building"):
        return "building"
    return None


def extract_templates(map_name: str, maps_root: Path | None = None) -> list[AssetTemplate]:
    """Learn one template per (poi_type, instance_name) present in a map."""
    maps_root = maps_root or (DELIVERYBENCH / "maps")
    world_path = Path(maps_root) / map_name / "progen_world_enriched.json"
    if not world_path.exists():
        return []
    digest = sha256_file(world_path)
    world = json.loads(world_path.read_text())

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for node in world.get("nodes", []) or []:
        properties = node.get("properties", {}) or {}
        poi_type = str(properties.get("poi_type") or properties.get("type") or "").strip().lower()
        instance_name = str(node.get("instance_name", "") or "")
        kind = classify(poi_type, instance_name)
        if kind is None:
            continue
        if not poi_type:
            poi_type = "building"
        grouped.setdefault((poi_type, instance_name), []).append(node)

    templates: list[AssetTemplate] = []
    for (poi_type, instance_name), nodes in sorted(grouped.items()):
        kind = classify(poi_type, instance_name) or "building"
        yaws = []
        boxes: list[list[float]] = []
        for node in nodes:
            properties = node.get("properties", {}) or {}
            orientation = properties.get("orientation", {}) or {}
            try:
                yaws.append(round(float(orientation.get("yaw", 0.0) or 0.0) % 360.0, 1))
            except (TypeError, ValueError):
                pass
            bbox = properties.get("bbox") or {}
            if bbox:
                try:
                    boxes.append(
                        [float(bbox.get("x", 0.0)), float(bbox.get("y", 0.0)),
                         float(bbox.get("z", 1000.0))]
                    )
                except (TypeError, ValueError):
                    pass
        median_box: list[float] = []
        if boxes:
            median_box = [
                round(statistics.median(axis), 2) for axis in zip(*boxes)
            ]
        templates.append(
            AssetTemplate(
                poi_type=poi_type,
                instance_name=instance_name,
                kind=kind,
                bbox_cm=median_box,
                default_yaw_deg=(statistics.mode(yaws) if yaws else 0.0),
                observed_yaws=sorted(set(yaws))[:8],
                source_map=map_name,
                source_sha256=digest,
                observed_count=len(nodes),
            )
        )
    return templates


@dataclass
class AssetLibrary:
    """Templates keyed by POI type, with provenance."""

    templates: list[AssetTemplate] = field(default_factory=list)
    built_from: list[str] = field(default_factory=list)

    # ── construction ─────────────────────────────────────────────────────────

    @classmethod
    def build(cls, map_names: list[str], maps_root: Path | None = None) -> "AssetLibrary":
        """Harvest templates from several maps, preferring the best-evidenced.

        When two maps supply the same POI type, the template observed more often
        wins: a type seen ten times carries a more trustworthy footprint and
        orientation convention than one seen once.
        """
        best: dict[str, AssetTemplate] = {}
        used: list[str] = []
        for map_name in sorted(map_names):
            templates = extract_templates(map_name, maps_root)
            if templates:
                used.append(map_name)
            for template in templates:
                existing = best.get(template.poi_type)
                if existing is None or template.observed_count > existing.observed_count:
                    best[template.poi_type] = template
        return cls(templates=[best[k] for k in sorted(best)], built_from=used)

    # ── persistence ──────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": LIBRARY_SCHEMA,
            "built_from": self.built_from,
            "types": [t.poi_type for t in self.templates],
            "templates": [t.to_dict() for t in self.templates],
        }

    def save(self, path: Path | None = None) -> Path:
        path = Path(path or DEFAULT_LIBRARY)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "AssetLibrary":
        path = Path(path or DEFAULT_LIBRARY)
        data = json.loads(path.read_text())
        if data.get("schema") != LIBRARY_SCHEMA:
            raise ValueError(f"{path}: expected {LIBRARY_SCHEMA}, got {data.get('schema')!r}")
        return cls(
            templates=[AssetTemplate.from_dict(t) for t in data["templates"]],
            built_from=list(data.get("built_from") or []),
        )

    # ── lookup ───────────────────────────────────────────────────────────────

    def types(self) -> list[str]:
        return [t.poi_type for t in self.templates]

    def template_for(self, poi_type: str) -> AssetTemplate | None:
        for template in self.templates:
            if template.poi_type == poi_type:
                return template
        return None

    def missing(self, required: dict[str, int]) -> list[str]:
        """Required types this library cannot supply."""
        available = set(self.types())
        return sorted(name for name in required if name not in available)
