"""Turn a pipeline verdict into a validated ``EnvSpec``.

The pipeline measures; this states the result in the standardized contract the
task layer consumes. Keeping the two apart means the measurements can grow new
fields without the published contract shifting under existing consumers.

Album detection deserves a note. Whether an environment supports vision is not
what the config says -- Paris had ``enable_fpv`` available and no album at all,
and a procgen map pointed at a 12-row stub manifest while a 665-row one sat in a
subdirectory. So support is decided by finding a manifest and counting the
waypoints in it, and the count is published so a consumer can judge coverage
rather than trust a boolean.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from embodiedbench.schemas.env_spec import (
    AffordanceInventory,
    EnvSpec,
    GraphRepairSummary,
    GraphSummary,
    NavigationStyle,
    ObservationSupport,
    QualityFlag,
    SolvabilityEvidence,
)
from embodiedbench.compiler.pipeline import GraphAnalysis, decide_point_navigation
from embodiedbench.schemas.environment import CertificationGrade, NavigationMode

REPO_ROOT = Path(__file__).resolve().parents[2]
DELIVERYBENCH = REPO_ROOT / "vendor" / "vagen" / "vagen" / "envs" / "deliverybench"


def album_search_roots(map_name: str) -> list[Path]:
    """Where an album for this map might live, in priority order.

    Albums we bake cannot go inside vendor/ -- that checkout is an input and is
    kept byte-identical -- so a freshly rendered album lives outside it. EB_ALBUM_ROOT
    points at that location and is searched first, falling back to the albums
    that shipped with the vendored maps.
    """
    import os

    roots: list[Path] = []
    external = os.environ.get("EB_ALBUM_ROOT")
    if external:
        roots.append(Path(external) / map_name)
    roots.append(DELIVERYBENCH / "deliverybench_fpv" / map_name)
    return roots


def find_album(map_name: str, album_root: Path | None = None) -> dict[str, Any]:
    """Locate a cached FPV album for a map and measure what it actually covers.

    Searches the album directory and one level of subdirectories, because the
    real manifest is not always at the root: ``small-city-11`` keeps a 12-row
    stub at the top and its 665-row manifest inside
    ``main_base_floor_road_full_1280x960/``. The largest manifest wins, since a
    stub is never the intended album.
    """
    if album_root is not None:
        roots = [Path(album_root)]
    else:
        roots = album_search_roots(map_name)
    root = next((r for r in roots if r.exists()), None)
    if root is None:
        return {
            "found": False,
            "reason": f"no album directory for {map_name!r} under {[str(r) for r in roots]}",
        }

    candidates = [root / "manifest.jsonl"]
    for child in sorted(root.iterdir()):
        if child.is_dir():
            candidates.append(child / "manifest.jsonl")

    best: dict[str, Any] = {"found": False, "reason": "no manifest.jsonl under the album"}
    for manifest in candidates:
        if not manifest.exists():
            continue
        positions: set[tuple[float, float]] = set()
        headings: set[float] = set()
        sample_entries: list[dict[str, Any]] = []
        rows = 0
        for line in manifest.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if entry.get("status") != "ok":
                continue
            rows += 1
            if len(sample_entries) < 40:
                sample_entries.append(entry)
            try:
                positions.add((round(float(entry["x_cm"]), 1), round(float(entry["y_cm"]), 1)))
                headings.add(float(entry["yaw"]))
            except (KeyError, TypeError, ValueError):
                continue
        if not rows or len(positions) <= best.get("waypoints", 0):
            continue

        # A manifest is a statement of intent, not evidence. Eight of the nine
        # procgen "albums" in this checkout carry a full manifest and zero image
        # files, so counting rows reported vision support for maps that have no
        # pixels at all. Resolve a sample of rows to real files before claiming
        # an album exists.
        resolved, sampled = 0, 0
        for entry in sample_entries:
            name = Path(entry["image_path"]).name if entry.get("image_path") else None
            waypoint = str(entry.get("waypoint_id") or "")
            if not name or "_" not in waypoint:
                continue
            kind, number = waypoint.rsplit("_", 1)
            try:
                directory = f"{kind}_{int(number):03d}"
            except ValueError:
                directory = waypoint
            sampled += 1
            for candidate in (
                manifest.parent / "images" / directory / name,
                Path(entry["image_path"]),
                manifest.parent.parent / "images" / directory / name,
                root / "images" / directory / name,
            ):
                if candidate.exists():
                    resolved += 1
                    break
        present = (resolved / sampled) if sampled else 0.0
        if present < 0.5:
            best = {
                "found": False,
                "reason": (
                    f"manifest {manifest.name} lists {rows} rows but only "
                    f"{resolved}/{sampled} sampled images exist on disk; a manifest "
                    "without images is not an album"
                ),
                "manifest_rows": rows,
                "images_present_fraction": round(present, 3),
            }
            continue

        best = {
            "found": True,
            "root": str(root),
            "manifest": str(manifest.relative_to(root.parent)),
            "rows": rows,
            "waypoints": len(positions),
            "headings": len(headings),
            "images_present_fraction": round(present, 3),
        }
    return best


def build_env_spec(
    result: dict[str, Any],
    *,
    album_root: Path | None = None,
    compiler_version: str = "0.1.0",
) -> EnvSpec:
    """Build the published contract from a ``compile_map`` verdict."""
    map_name = result["map"]
    analysis = result.get("analysis") or {}
    unusable = bool(result.get("unusable"))

    flags = [
        QualityFlag(
            code=flag["code"],
            count=int(flag.get("count", 0)),
            detail=str(flag.get("detail", ""))[:2000],
            stage=flag.get("stage", "graph"),
        )
        for flag in result.get("quality_findings", [])
    ]

    validation = result.get("validation") or {}
    solvability = None
    episodes = validation.get("episodes") or []
    if episodes and not unusable:
        solvability = SolvabilityEvidence(
            episodes=len(episodes),
            delivered_episodes=int(validation.get("delivered_episodes", 0)),
            solvability_rate=float(validation.get("solvability_rate", 0.0)),
            mean_steps=float(validation.get("mean_steps", 0.0)),
        )

    repair_raw = result.get("graph_repair") or {}
    repair = GraphRepairSummary(
        applied=bool(repair_raw),
        converged=any("fixpoint" in note for note in repair_raw.get("notes", [])),
        passes_note="; ".join(repair_raw.get("notes", []))[:500],
        edges_before=int(repair_raw.get("edges_before", 0)),
        edges_after=int(repair_raw.get("edges_after", 0)),
        edges_split=int(repair_raw.get("edges_split", 0)),
        skipped_nodes_recovered=int(repair_raw.get("skipped_nodes_recovered", 0)),
        mean_degree_before=float(repair_raw.get("mean_degree_before", 0.0)),
        mean_degree_after=float(repair_raw.get("mean_degree_after", 0.0)),
        longest_edge_before_m=float(repair_raw.get("longest_edge_before_m", 0.0)),
        longest_edge_after_m=float(repair_raw.get("longest_edge_after_m", 0.0)),
    )

    album = find_album(map_name, album_root)
    channels = ["text"]
    if album.get("found"):
        channels.append("rgb")
    observation = ObservationSupport(
        channels=channels,
        has_cached_album=bool(album.get("found")),
        album_waypoints=int(album.get("waypoints", 0)),
        album_headings=int(album.get("headings", 0)),
        album_coverage_fraction=(
            min(1.0, album.get("waypoints", 0) / analysis["node_count"])
            if album.get("found") and analysis.get("node_count")
            else None
        ),
    )

    navigation = result.get("navigation") or {}
    if unusable or not navigation:
        # An unusable environment still gets a well-formed spec, because the
        # task layer must be able to read "no" from the same contract it reads
        # "yes" from rather than special-casing a missing object.
        failure = result.get("failure") or {}
        return EnvSpec(
            env_id=f"{map_name}-env",
            map_name=map_name,
            compiler_version=compiler_version,
            navigation_style=NavigationStyle.GRAPH,
            navigation_modes=[NavigationMode.NAV_WAYPOINT],
            enabled_actions=["MOVE_TO"],
            enable_waypoint_marks=True,
            navigation_rationale="environment is unusable; no navigation was decided",
            graph=GraphSummary(
                node_count=analysis.get("node_count", 0),
                edge_count=analysis.get("edge_count", 0),
                mean_degree=analysis.get("mean_degree", 0.0),
                max_degree=max(
                    (int(k) for k in (analysis.get("degree_histogram") or {})), default=0
                ),
                cardinal_fraction=analysis.get("cardinal_fraction", 0.0),
                largest_component_fraction=analysis.get("largest_component_fraction", 0.0),
            ),
            graph_repair=repair,
            observation=observation,
            grade=CertificationGrade.FAIL,
            quality_flags=flags,
            usable=False,
            failure_code=failure.get("code", "unknown_load_failure"),
            failure_explanation=failure.get("explanation", "")[:2000] or "unspecified",
            thresholds=result.get("thresholds", {}),
        )

    style = (
        NavigationStyle.GRAPH
        if navigation["mode"] == "graph"
        else NavigationStyle.CARDINAL_AND_GRAPH
    )
    # nav_waypoint is unconditional -- a one-hop step to a named neighbour needs
    # no geometry, which is why design plan §9 makes it the production path. The two
    # point modes are added when the map's geometry defines them; which runtimes
    # can actually *serve* them is a separate fact, recorded alongside.
    point_nav = decide_point_navigation(_graph_analysis_from_dict(analysis))
    modes = [NavigationMode.NAV_WAYPOINT]
    if point_nav.definable:
        modes += [NavigationMode.NAV_POINT_3D, NavigationMode.NAV_POINT_2D_DEPTH]

    edge_lengths = analysis.get("edge_length_m") or {}
    graph = GraphSummary(
        node_count=analysis.get("node_count", 0),
        edge_count=analysis.get("edge_count", 0),
        mean_degree=analysis.get("mean_degree", 0.0),
        max_degree=max((int(k) for k in (analysis.get("degree_histogram") or {})), default=0),
        dock_nodes=analysis.get("dock_nodes", 0),
        junction_nodes=analysis.get("junction_nodes", 0),
        cardinal_fraction=analysis.get("cardinal_fraction", 0.0),
        largest_component_fraction=analysis.get("largest_component_fraction", 0.0),
        component_count=analysis.get("component_count", 1),
        median_edge_m=edge_lengths.get("p50", 0.0),
        longest_edge_m=edge_lengths.get("max", 0.0),
        long_edge_threshold_m=analysis.get("long_edge_threshold_m", 0.0),
    )

    return EnvSpec(
        env_id=f"{map_name}-env",
        map_name=map_name,
        compiler_version=compiler_version,
        navigation_style=style,
        navigation_modes=modes,
        enabled_actions=list(navigation["enabled_actions"]),
        enable_waypoint_marks=bool(navigation.get("enable_waypoint_marks", True)),
        navigation_rationale=navigation["rationale"],
        graph=graph,
        graph_repair=repair,
        affordances=AffordanceInventory(counts=result.get("affordances", {})),
        observation=observation,
        grade=CertificationGrade(result["grade"].upper() if result["grade"] in ("a", "b", "c") else result["grade"]),
        quality_flags=flags,
        solvability=solvability,
        usable=True,
        world_bundle_sha256=result.get("world_bundle_sha256"),
        thresholds=result.get("thresholds", {}),
        # EnvSpec 0.1.0 has no dedicated field for point-mode reach and lattice,
        # and runtime_config is the schema's declared free-form channel. Putting
        # it here keeps a runtime able to read the reach it must enforce without
        # a version bump that would invalidate every spec already written.
        runtime_config={
            **(result.get("env_config") or {}),
            "point_navigation": point_nav.to_dict(),
        },
    )


def _graph_analysis_from_dict(analysis: dict[str, Any]) -> GraphAnalysis:
    """Rebuild the analysis dataclass from its serialized form.

    ``compile_map`` publishes its analysis as a dict, but the point-mode rule is
    written against the dataclass so it can also be called directly on a fresh
    analysis. Reconstructing is cheaper than duplicating the rule.
    """
    return GraphAnalysis(
        node_count=int(analysis.get("node_count", 0)),
        edge_count=int(analysis.get("edge_count", 0)),
        edge_length_m=dict(analysis.get("edge_length_m") or {}),
    )
