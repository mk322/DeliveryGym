"""Are the albums all here, and does the environment still run on them?

Deliberately not a unit test. The test suite checks logic and passes on a
machine with no albums at all, because most of it builds its own frames in a
temporary directory. What it cannot catch is the failure a *copy* has: an album
that is missing, half-transferred, or — worst and quietest — present but
detached from the compiled graph.

That last one is the reason this exists. An album is baked against a specific
compiled network, and anything that renames a node orphans every frame while
every manifest row still says ``status: ok``. It happened once, a weld-tolerance
change was enough, and it went unnoticed until a policy reported that 86% of its
candidates had no picture to look at. Coverage is measured here, not assumed.

Run it after installing the albums on a new machine (docs/INSTALL.md), and
after anything that touches the compiler:

    ALBUMS_DIR=/data/albums python -m embodiedbench.tools.migration_check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
DEFAULT_MAPS = REPO / "vendor" / "vagen" / "vagen" / "envs" / "deliverybench" / "maps"


def _default_albums() -> Path:
    """Where this machine keeps the albums: ALBUMS_DIR, else the runtime's fallback."""
    from embodiedbench.training.vagen_courier_env import _albums_base
    return _albums_base()
# Below this, the album and the graph have come apart and nothing else is worth
# reporting. Full coverage is 1.0; the street album measured 1.0 when it was
# baked, so anything under this is a transfer or a recompile, not wear.
MIN_STREET_COVERAGE = 0.95


def check(maps: Path, albums: Path) -> tuple[list[str], dict[str, Any]]:
    """Every complaint, and the numbers behind them."""
    problems: list[str] = []
    report: dict[str, Any] = {}

    map_dir = maps / "citycore-paris"
    if not map_dir.exists():
        return [f"no map at {map_dir} -- copy vendor/.../deliverybench/maps/"], report

    from embodiedbench.compiler.road_network import build_road_network
    from embodiedbench.runtime.city.courier_env import CourierEnv

    net = build_road_network(map_dir, map_name="citycore-paris")
    report["graph"] = {
        "nodes": len(net.nodes), "edges": len(net.edges()),
        "streets": len(net.streets), "addresses": len(net.addresses),
        "components": net.components()[:3],
        "signalised": len(net.signalised_nodes()),
    }
    if len(net.components()) != 1:
        problems.append(
            f"the street graph is in {len(net.components())} pieces; a courier "
            "cannot walk between them")

    # The five albums a walking courier reads. The pavement pair is what the
    # benchmark's embodiment actually sees; a bundle without it used to pass
    # this check and then serve carriageway frames to a pedestrian.
    roots = {
        "streets": albums / "paris_streets_v2" / "citycore-paris",
        "pavement": albums / "paris_streets_pavement" / "citycore-paris",
        "signals": albums / "paris_lamps_real" / "citycore-paris",
        "obstacles": albums / "paris_obstacles" / "citycore-paris",
        "pavement_obstacles": albums / "paris_obstacles_pavement" / "citycore-paris",
    }
    for name, root in roots.items():
        if not root.exists():
            problems.append(f"{name} album missing at {root}")
    report["albums"] = {
        name: {"present": root.exists(),
               "frames": sum(1 for _ in (root / "images").rglob("*.png"))
               if (root / "images").exists() else 0}
        for name, root in roots.items()
    }

    present = {name: root for name, root in roots.items() if root.exists()}
    env = CourierEnv(
        net, seed=0, difficulty="pair", stride="block",
        album_root=present.get("streets"),
        pavement_album_root=present.get("pavement"),
        signal_album_root=present.get("signals"),
        obstacle_album_root=present.get("obstacles"),
        pavement_obstacle_album_root=present.get("pavement_obstacles"),
    )
    env.reset()

    coverage = env.album_coverage()
    report["coverage"] = coverage
    fraction = coverage.get("fraction", 0.0)
    if fraction < MIN_STREET_COVERAGE:
        problems.append(
            f"only {fraction:.1%} of walkable directions have a photograph. The "
            "album and the compiled graph have come apart -- either the copy is "
            "incomplete or the graph was rebuilt after the bake")

    visible = env.visible_signals
    report["signals"] = {
        "visibility_sidecar": visible is not None,
        "legible_approaches": len(visible) if visible else 0,
    }
    if visible is None and roots["signals"].exists():
        problems.append(
            "the signal album has no signal_visibility.json, so crossing on red "
            "is not charged. Regenerate it: python -m "
            "embodiedbench.compiler.signal_legibility <album> --write-sidecar")

    # A live episode, because a world that builds and cannot be walked is not a
    # world. The privileged router is used on purpose: it cannot get lost, so a
    # failure here is the environment's.
    from embodiedbench.tasks.courier_router import run_shortest_path_courier

    result = run_shortest_path_courier(env, 0, max_steps=9000)
    report["episode"] = result.to_dict()
    if result.delivered < result.issued:
        problems.append(
            f"the shortest-path router delivered only {result.delivered} of "
            f"{result.issued}. It knows the map, so this is the environment")

    return problems, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--maps", type=Path, default=DEFAULT_MAPS)
    parser.add_argument("--albums", type=Path, default=None,
                        help="album base directory (default: $ALBUMS_DIR)")
    parser.add_argument("--json", action="store_true", help="report and nothing else")
    args = parser.parse_args(argv)

    problems, report = check(args.maps, args.albums or _default_albums())
    if args.json:
        print(json.dumps({"ok": not problems, "problems": problems, **report}, indent=1))
        return 1 if problems else 0

    graph = report.get("graph", {})
    if graph:
        print(f"graph      {graph['nodes']} nodes, {graph['edges']} edges, "
              f"{graph['streets']} streets, {graph['addresses']} addresses, "
              f"{graph['signalised']} signalised")
    for name, info in report.get("albums", {}).items():
        mark = "ok " if info["present"] else "MISSING"
        print(f"album      {name:10} {mark} {info['frames']} frames")
    coverage = report.get("coverage", {})
    if coverage:
        print(f"coverage   {coverage.get('fraction', 0):.1%} of walkable directions "
              f"have a photograph ({coverage.get('with_frame', 0)}"
              f"/{coverage.get('directed_edges', 0)})")
    signals = report.get("signals", {})
    if signals:
        print(f"signals    {signals['legible_approaches']} approaches with a readable lamp")
    episode = report.get("episode", {})
    if episode:
        print(f"episode    delivered {episode['delivered']}/{episode['issued']} "
              f"in {episode['turns']} turns, {episode['sim_minutes']} min")

    if problems:
        print("\nPROBLEMS")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("\nthe install is complete: the world builds, the pictures are there, "
          "and an episode runs end to end")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
