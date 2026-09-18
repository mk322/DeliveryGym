"""Things in the way, which the text never mentions.

Before this the environment had exactly one mechanic a courier needed its eyes
for -- the pedestrian light -- and lights only happen at signalised junctions.
Between them the pavement was guaranteed clear, so *walking* required no
perception at all: a policy that read the street list and ignored every
photograph walked as well as one that studied them. That is the wrong shape for
an embodied benchmark, because walking is what a courier spends its shift doing.

An obstacle is the missing half. The rule is the same one the signal follows and
it is the only rule that makes vision worth anything:

    the world puts it in the picture, and nowhere else

``navigate()`` does not know about it -- a map app cannot see a skip on the
pavement -- ``look()`` does not describe it, the street list does not flag it,
and no observation field carries its type. The single copy of the fact is the
frame attached to the street it sits on. A courier that looks routes around it;
a courier that does not walks into it and pays.

Two kinds, following the vendored DeliveryBench schema
(``vlm_delivery/utils/hazards.py``) so a sidecar written for one runtime loads
in the other:

``road_block``       impassable. The edge cannot be walked in either direction;
                    the courier has to find another way round. Placement never
                    cuts the network -- see ``bridges`` -- so a way round always
                    exists and no order is ever made undeliverable.
``slow_pedestrian``  passable, slowly. The edge still works, it just costs
                    ``SLOW_SECONDS`` more, which is a real reason to prefer a
                    clear street when one is a similar length.

    A note on the second name, because it is inherited rather than descriptive.
    CityCore Paris ships no character meshes, so what the album actually shows
    at a ``slow_pedestrian`` site is the pavement congested by street furniture
    -- a stand, a chalkboard, bins -- which is a thing you get past slowly. The
    mechanic is the vendored one ("passable at a time cost") and the name is
    kept so the two runtimes' sidecars stay interchangeable; the render is
    honest about being furniture rather than people.

**Placement is rule-based and generalises.** Nothing here reads a hand-authored
list. A *site* is an edge chosen by a hash of the map name and the edge's two
node ids -- so the same map always offers the same sites, and any map at all
offers some. Which sites are *live*, and of which kind, is a second hash that
includes the episode seed. Two consequences matter: the album can be baked once
per map because the sites do not move, and every seed still gets a different
arrangement out of them.

**And it is gated on being visible.** Exactly as for the signal, an obstacle the
courier cannot see is not charged for. The album declares, in
``obstacle_visibility.json``, which approaches actually show one; an obstacle
whose two approaches are not both in that list is inert -- not drawn, not
blocking, not costing. Charging for information the environment withholds is the
one thing this codebase has already had to fix twice, and it is not being
reintroduced with a new mechanic.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

SLOW_PEDESTRIAN = "slow_pedestrian"
ROAD_BLOCK = "road_block"
OBSTACLE_TYPES = (SLOW_PEDESTRIAN, ROAD_BLOCK)

# The sidecar an album writes to declare which approaches show an obstacle.
OBSTACLE_VISIBILITY_FILE = "obstacle_visibility.json"

# What fraction of the map's obstructable edges are obstacle *sites*: places the
# album has a frame for. Everything downstream is drawn from these, so this is
# the number that sets the render bill -- 4 frames per site (two directions by
# two kinds). 20% of Paris's 359 obstructable edges is 60 sites and 240 frames,
# about four minutes of rendering. Being stingier has a cost of its own: too few
# sites and the map is small enough to memorise, at which point a policy stops
# needing to look and starts needing to remember, which is a different task.
SITE_FRACTION = 0.20
# How many of a map's sites are live in any one episode. Not 1.0, because an
# obstacle that is always at the same corner stops being something to look for
# and becomes something to memorise, and a policy that memorises the map is not
# the policy this benchmark is trying to measure.
ACTIVE_RATE = 0.45
# Of the live ones, how many are the impassable kind. Set by measurement, not
# taste: the Paris carriageway graph averages degree 2.3, so shutting an edge is
# expensive and shutting many is crippling. Across ten SHIFT seeds the walk a
# perfect courier needs runs 1.21x longer at this share (worst seed 1.82x) and
# 1.31x at 0.55 with a worst seed of 2.58x. The clock is derived from that same
# obstacle-aware optimum, so neither number tightens the deadline -- but the
# second one turns a delivery round into an orienteering exercise, which is a
# different task from the one the tier claims to set.
BLOCK_SHARE = 0.40
# What squeezing past a congested pavement costs. Set against the alternative:
# the median edge is 18 m, about 13 s of walking, so a detour of two or three
# edges is 40-80 s. A delay much below that would never be worth avoiding and
# the type would be decoration.
SLOW_SECONDS = 40.0
# Walking up to a barrier, seeing it will not do, and coming back. This is what
# a courier who did not look at the photograph pays, and it is the whole
# measurable value of looking: the sighted courier spends 0 s here.
BLOCKED_SECONDS = 45.0


def site_key(a: str, b: str) -> str:
    """The undirected edge key. An obstacle sits on the street, not on a heading."""
    return f"{a}|{b}" if a <= b else f"{b}|{a}"


def approach_key(a: str, b: str) -> str:
    """The directed key an album frame and the visibility sidecar are stored under."""
    return f"{a}|{b}"


def _unit(*parts: Any) -> float:
    """A stable pseudo-random number in [0, 1) for the given parts.

    ``hash()`` is salted per process and ``random.Random`` needs seeding
    discipline the callers would have to share; a digest of the strings needs
    neither and gives the same answer on every machine and every run, which is
    what "deterministic per (map, seed)" has to mean to be worth saying.
    """
    payload = "\x00".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(1 << 64)


def bridges(neighbours: Mapping[str, Iterable[str]]) -> set[str]:
    """Edges whose removal would split the graph, as ``site_key``s.

    A ``road_block`` on one of these would strand part of the city -- and with
    it any address on the far side, turning a seed into an episode nobody can
    finish. The whole point of the difficulty ladder is that every rung stays
    solvable, so these are excluded from the site pool rather than handled as a
    failure later.

    Iterative Tarjan, because the Paris graph is 2000 nodes deep in places and
    the recursive form overflows the interpreter's stack on it.
    """
    order: dict[str, int] = {}
    low: dict[str, int] = {}
    found: set[str] = set()
    counter = 0
    for root in sorted(neighbours):
        if root in order:
            continue
        # (node, parent, iterator over its neighbours)
        stack: list[tuple[str, str | None, Any]] = [(root, None, iter(sorted(neighbours[root])))]
        order[root] = low[root] = counter
        counter += 1
        while stack:
            node, parent, children = stack[-1]
            child = next(children, None)
            if child is None:
                stack.pop()
                if stack:
                    up = stack[-1][0]
                    low[up] = min(low[up], low[node])
                    if low[node] > order[up]:
                        found.add(site_key(up, node))
                continue
            if child == parent:
                continue
            if child in order:
                low[node] = min(low[node], order[child])
                continue
            order[child] = low[child] = counter
            counter += 1
            stack.append((child, node, iter(sorted(neighbours[child]))))
    return found


def obstacle_sites(neighbours: Mapping[str, Iterable[str]], map_name: str, *,
                   fraction: float = SITE_FRACTION) -> list[tuple[str, str]]:
    """Where this map can have an obstacle at all. Seed-free, so the album is too.

    Every edge that is not a bridge is a candidate; a hash of the map name and
    the edge picks ``fraction`` of them. Because the seed is not in the hash, a
    site is a property of the map: the renderer can bake frames for exactly this
    list and every episode of every seed will find its obstacles already
    photographed.
    """
    cut = bridges(neighbours)
    out: list[tuple[str, str]] = []
    for node in sorted(neighbours):
        for other in sorted(neighbours[node]):
            if node >= other:
                continue
            key = site_key(node, other)
            if key in cut:
                continue
            if _unit(map_name, "site", key) < fraction:
                out.append((node, other))
    return out


@dataclass
class ObstacleField:
    """Which of a map's obstacle sites are live this episode, and what is on them.

    Directed lookups are answered from an undirected store, because a barrier is
    a physical object: if it blocks the way from one end it blocks it from the
    other. That is also why ``in_effect`` requires *both* approaches to be
    photographed -- serving an edge that is impassable from the north and clear
    from the south would be a world the pictures contradict.
    """

    map_name: str = ""
    seed: int = 0
    by_site: dict[str, str] = field(default_factory=dict)
    # Approaches the album can actually show an obstacle on. ``None`` means the
    # album made no claim, and then nothing is in effect -- the same default as
    # having no album at all, and for the same reason.
    visible: set[str] | None = None

    def __len__(self) -> int:
        return len(self.by_site)

    @classmethod
    def generate(cls, neighbours: Mapping[str, Iterable[str]], map_name: str, seed: int, *,
                 sites: Iterable[tuple[str, str]] | None = None,
                 visible: set[str] | None = None,
                 active_rate: float = ACTIVE_RATE,
                 block_share: float = BLOCK_SHARE) -> "ObstacleField":
        """The obstacles for one episode: deterministic in ``(map_name, seed)``."""
        if sites is None:
            sites = obstacle_sites(neighbours, map_name)
        by_site: dict[str, str] = {}
        for a, b in sites:
            key = site_key(a, b)
            if _unit(map_name, seed, "live", key) >= active_rate:
                continue
            kind = ROAD_BLOCK if _unit(map_name, seed, "kind", key) < block_share else SLOW_PEDESTRIAN
            by_site[key] = kind
        # Individually safe is not collectively safe. Every site is a non-bridge
        # of the intact graph, but three sites can be the three edges of one
        # loop, and shutting all three strands whatever the loop reached. On
        # Paris that happened on 1 of the first 4 SHIFT seeds tested: an address
        # was left with no route to it at all, which is an unsolvable episode
        # rather than a hard one.
        #
        # So the blocks are accepted one at a time against the graph as it
        # already stands, in sorted order for determinism, and any that would
        # cut it is demoted to congestion instead of dropped -- the site was
        # chosen to have something on it, and a photograph of it exists.
        blocked = sorted(k for k, kind in by_site.items() if kind == ROAD_BLOCK)
        remaining = {n: set(v) for n, v in neighbours.items()}
        for key in blocked:
            a, b = key.split("|")
            if key in bridges(remaining):
                by_site[key] = SLOW_PEDESTRIAN
                continue
            remaining[a].discard(b)
            remaining[b].discard(a)
        return cls(map_name=map_name, seed=seed, by_site=by_site, visible=visible)

    # ── lookups ──────────────────────────────────────────────────────────────

    def type_on(self, a: str, b: str) -> str | None:
        """What is standing on the edge between ``a`` and ``b``, ignoring visibility."""
        return self.by_site.get(site_key(a, b))

    def is_visible(self, a: str, b: str) -> bool:
        """Does the album show the obstacle on this edge, from both ends?"""
        if self.visible is None:
            return False
        return approach_key(a, b) in self.visible and approach_key(b, a) in self.visible

    def in_effect(self, a: str, b: str) -> str | None:
        """The obstacle the courier can see and is therefore accountable for."""
        kind = self.type_on(a, b)
        if kind is None or not self.is_visible(a, b):
            return None
        return kind

    def blocks(self, a: str, b: str) -> bool:
        return self.in_effect(a, b) == ROAD_BLOCK

    def delay_seconds(self, a: str, b: str) -> float:
        return SLOW_SECONDS if self.in_effect(a, b) == SLOW_PEDESTRIAN else 0.0

    # ── serialisation ────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "map": self.map_name, "seed": self.seed,
            "obstacles": [
                {"site": key, "type": kind, "in_effect": self.in_effect(*key.split("|"))}
                for key, kind in sorted(self.by_site.items())
            ],
        }

    def counts(self) -> dict[str, int]:
        """How many of each kind are placed, and how many actually bite."""
        out = {"placed": len(self.by_site), "in_effect": 0}
        for kind in OBSTACLE_TYPES:
            out[kind] = sum(1 for k in self.by_site.values() if k == kind)
            out[f"{kind}_in_effect"] = 0
        for key, kind in self.by_site.items():
            a, b = key.split("|")
            if self.in_effect(a, b):
                out["in_effect"] += 1
                out[f"{kind}_in_effect"] += 1
        return out


def load_visibility(album_root: Path | None) -> set[str] | None:
    """The approaches an obstacle album says it can show. ``None`` if it says nothing.

    Silence is not consent, for the third time in this codebase: an album that
    does not declare what it shows has not been checked, and guessing produces a
    penalty for an invisible obstacle.
    """
    if album_root is None:
        return None
    path = Path(album_root) / OBSTACLE_VISIBILITY_FILE
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return {str(k) for k in data.get("visible", [])}
