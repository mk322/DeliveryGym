"""A courier that already knows the way. The ceiling, not a baseline.

``ObservationOnlyCourier`` establishes the floor: if a policy that may read only
what the world says to it can deliver, then the words are sufficient and the
task is solvable. It says nothing about what a *good* policy would spend, and
that turned out to matter, because "how many turns does one delivery cost" is a
question about the environment that the reference courier cannot answer. It
navigates by asking the phone for a range after every single move -- half of
every episode it has ever run is phone lookups -- so quoting its turn count as
the cost of a delivery measures its habit rather than the city.

This is the other end. It is **privileged and says so**: it reads the road
network directly through ``route_nodes``, which no policy under evaluation may
do. It is not a baseline, it is not scored against, and it never appears in a
results table beside a model. What it is for is bracketing:

    reference courier   what the observation alone is enough for
    this                what the map costs, with the navigation given away

A number quoted between the two is a claim about a policy. A number outside them
is a bug in the measurement.

It still obeys everything the world enforces -- it waits at red lights, it walks
into barriers it was not told about and goes round them, it pays for every tool
it calls. Only the route is free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from embodiedbench.runtime.city.courier_env import ARRIVAL_TOLERANCE_CM, HANDLING_SECONDS, CourierEnv


@dataclass
class RouterResult:
    seed: int
    delivered: int = 0
    issued: int = 0
    on_time: int = 0
    turns: int = 0
    sim_seconds: float = 0.0
    walked_m: float = 0.0
    earnings: float = 0.0
    trace: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed, "delivered": self.delivered, "issued": self.issued,
            "on_time": self.on_time, "turns": self.turns,
            "sim_minutes": round(self.sim_seconds / 60.0, 1),
            "walked_m": round(self.walked_m, 1),
            "earnings": round(self.earnings, 2),
            "turns_per_delivery": (
                round(self.turns / self.delivered, 1) if self.delivered else None
            ),
        }


class ShortestPathCourier:
    """Walks the shortest path to whatever job is in hand, and waits at red.

    Deliberately no cleverer than that. It does not re-sequence a queue, so on
    the deep tiers it is a *lower* bound on what a good planner would earn -- the
    ceiling it establishes is on navigation cost, not on strategy.
    """

    def __init__(self, env: CourierEnv, *, max_steps: int = 4000,
                 stop_at_turns: int | None = None):
        self.env = env
        self.max_steps = max_steps
        #: Stop once the world has counted this many turns -- the budget a
        #: model is scored on (every action is a turn), so a bound run to it
        #: reads on the model's scale. ``max_steps`` caps the courier's own
        #: loop and is the only limit when this is ``None``.
        self.stop_at_turns = stop_at_turns
        # Barriers, learned the way anyone learns them: by walking into one. The
        # route is free; what is standing in it is not, and pretending otherwise
        # would make this a ceiling on a different world.
        self.blocked: set[tuple[str, str]] = set()

    def run(self, seed: int) -> RouterResult:
        env = self.env
        result = RouterResult(seed=seed)
        for _ in range(self.max_steps):
            if env.finished or env.shift_over:
                break
            if self.stop_at_turns is not None and env.turns >= self.stop_at_turns:
                break
            order = env.active_order()
            if order is None:
                break
            if math.dist(env.position(), order.target.kerb) <= ARRIVAL_TOLERANCE_CM:
                outcome = env.hand_over() if order.picked_up else env.collect()
                if not outcome.ok:
                    result.trace.append(f"at the door and refused: {outcome.message[:60]}")
                    break
                continue
            step = self._next_step(order.target.kerb_node)
            if step is None:
                result.trace.append(f"no route from {env.node_id}")
                break
            k, toward = step
            if env.signal_is_visible(env.node_id, toward) and env.light_here(k) == "red":
                env.wait()
                continue
            here = env.node_id
            outcome = env.walk_to(*env.street_at(k))
            # A barrier, found the way anyone finds one. It is remembered
            # here, in the courier, because there is nowhere else to put it:
            # ``route_nodes`` is the survey and the survey does not learn, so
            # every route from now on will still name this street and this
            # policy will still have to steer round it. At block stride the
            # barrier can be several hops down the street and the walk that
            # met it still counts as a move (the metres before it were
            # walked): the courier is left standing at the junction before
            # it, and what it saw there is the hop the world records in
            # ``witnessed_blocks``. Remembering the first hop of a refused
            # walk instead sent it round by another street into the same
            # barrier, and not looking after a successful-but-cut-short walk
            # cost a refused turn at the barrier on the next move.
            seen = getattr(env, "witnessed_blocks", None) or set()
            hops = set(seen) if seen else ({(here, toward)} if not outcome.ok else set())
            for a, b in hops:
                self.blocked.add((a, b))
                self.blocked.add((b, a))
        summary = env.summary()
        result.delivered = summary.get("delivered", 0)
        result.issued = summary.get("orders_issued", len(env.orders))
        result.on_time = summary.get("on_time", 0)
        result.turns = summary.get("turns", 0)
        result.sim_seconds = env.sim_seconds
        result.walked_m = env.walked_cm / 100.0
        result.earnings = summary.get("earnings", 0.0)
        return result

    def _next_step(self, goal: str | None) -> tuple[int, str] | None:
        """The numbered street that starts the best path this courier knows of.

        Its own search, not the phone's. ``route_nodes`` is the survey and the
        survey does not learn -- with ``report_blocked`` gone there is no way to
        tell it about a barrier, so it will name the same shut street on every
        call for the rest of the shift. A ceiling that asked it each turn and
        then guessed greedily when the answer was unwalkable spent 235 turns
        walking into barriers over six seeds and delivered 4 of 12.

        So the privilege is used properly: shortest path over the graph *minus
        the edges this courier has walked into*. That is what a competent agent
        does -- remember, and replan -- and it is the right ceiling to measure a
        model against, because a model can do exactly this from the photographs
        without ever walking into anything.
        """
        import heapq

        env = self.env
        if goal is None or goal not in env.network.nodes:
            return None
        start = env.node_id
        best: dict[str, float] = {start: 0.0}
        came: dict[str, str] = {}
        seen: set[str] = set()
        queue = [(0.0, start)]
        while queue:
            cost, node = heapq.heappop(queue)
            if node == goal:
                break
            if node in seen:
                continue
            seen.add(node)
            here = env.network.nodes[node].position
            for neighbour in env.network.nodes[node].neighbours:
                if neighbour in seen or (node, neighbour) in self.blocked:
                    continue
                step = cost + math.dist(here, env.network.nodes[neighbour].position)
                if step < best.get(neighbour, float("inf")):
                    best[neighbour] = step
                    came[neighbour] = node
                    heapq.heappush(queue, (step, neighbour))
        if goal not in came and goal != start:
            return None
        path = [goal]
        while path[-1] != start:
            path.append(came[path[-1]])
        path.reverse()
        if len(path) < 2:
            return None
        row = {r["node"]: r for r in env.candidates()}.get(path[1])
        return (row["k"], row["node"]) if row is not None else None


class FewestMovesCourier(ShortestPathCourier):
    """The same privileged courier, planning in the currency the benchmark
    actually rations: turns.

    ``ShortestPathCourier`` minimises metres. Under a turn cap that binds
    before the clock does -- the 60-turn benchmark uses about 2000 of the
    shift's 3600 seconds -- metres are the wrong objective: one ``walk_to``
    at block stride runs to the next corner however long the block, so a
    route of three long blocks beats one of six short ones by three turns
    at the same distance. This courier plans over the moves a ``walk_to``
    really makes, using the world's own block rule (``block_chain``) and
    its own stopping rules (a door of a live job, a crossing whose light is
    in the album), and takes the route with the fewest moves that still
    keeps the job's deadline, ties broken by metres -- a route of fewer,
    longer blocks is worth nothing if it turns the full fee into the late
    half. When no route keeps the deadline it takes the earliest arrival.
    It is the upper bound the benchmark is read against; it is still not a
    policy.
    """

    #: Kept back from the deadline for the unknowables on a route: a red
    #: light or two (15 s a wait) and the world's rounding.
    DEADLINE_RESERVE_S = 45.0
    #: Pareto labels kept per node in the two-criterion search.
    MAX_LABELS = 8

    def _metres_to(self, start: str, goal: str) -> float | None:
        """Shortest walking distance over the graph minus known barriers."""
        import heapq

        nodes = self.env.network.nodes
        best = {start: 0.0}
        queue = [(0.0, start)]
        while queue:
            cost, node = heapq.heappop(queue)
            if node == goal:
                return cost
            if cost > best.get(node, float("inf")):
                continue
            for neighbour in nodes[node].neighbours:
                if (node, neighbour) in self.blocked:
                    continue
                step = cost + math.dist(nodes[node].position, nodes[neighbour].position)
                if step < best.get(neighbour, float("inf")):
                    best[neighbour] = step
                    heapq.heappush(queue, (step, neighbour))
        return None

    def _metres_budget(self, goal: str) -> float | None:
        """How far the courier may walk to ``goal`` and still hand this job
        over on time; ``None`` when no job is in hand."""
        env = self.env
        order = env.active_order()
        if order is None or order.issued_at_s is None:
            return None
        remaining = order.due_at() - env.sim_seconds - self.DEADLINE_RESERVE_S
        reserve = HANDLING_SECONDS               # the hand-over
        if not order.picked_up:
            reserve += HANDLING_SECONDS          # the collection, then the delivery leg
            leg = self._metres_to(order.pickup.kerb_node, order.dropoff.kerb_node)
            if leg is not None:
                reserve += leg / env.travel_speed_cm_s()
        return max(0.0, remaining - reserve) * env.travel_speed_cm_s()

    def _next_step(self, goal: str | None) -> tuple[int, str] | None:
        import heapq

        env = self.env
        if goal is None or goal not in env.network.nodes:
            return None
        start = env.node_id
        if start == goal:
            return None
        nodes = env.network.nodes
        neighbours = getattr(env, "_neighbours", None) or {
            n: sorted(node.neighbours) for n, node in nodes.items()}
        signalised = getattr(env, "signalised", set()) or set()
        budget_cm = self._metres_budget(goal)

        def move_from(node: str, toward: str) -> tuple[str, float] | None:
            """Where one ``walk_to`` from ``node`` along ``toward`` stops, and
            how far it walks; ``None`` if the first hop is a known barrier."""
            if (node, toward) in self.blocked:
                return None
            chain = env.block_chain(node, toward)
            metres = 0.0
            end = node
            for index, (a, b) in enumerate(chain):
                if (a, b) in self.blocked:
                    break                      # the walk is refused at this hop
                metres += math.dist(nodes[a].position, nodes[b].position)
                end = b
                if b == goal or env._standing_at_a_door(b):
                    break                      # the stride stops at a live job's door
                following = chain[index + 1] if index + 1 < len(chain) else None
                if (following is not None and b in signalised
                        and env.signal_in_album(b, following[1])):
                    break                      # and at a crossing whose light it can see
            if end == node:
                return None
            return end, metres

        # Two criteria, moves and metres, so a node keeps every label that no
        # other label beats on both: the fewest-moves route to the goal may
        # pass a node by a longer road than the fewest-metres one does, and a
        # single best label per node would lose one of them.
        labels: dict[str, list[tuple[int, float]]] = {start: [(0, 0.0)]}
        arrivals: list[tuple[int, float, str | None]] = []
        queue: list[tuple[int, float, str, str | None]] = [(0, 0.0, start, None)]
        while queue:
            moves, metres, node, first = heapq.heappop(queue)
            if node == goal:
                arrivals.append((moves, metres, first))
                continue
            if any(m <= moves and d <= metres and (m, d) != (moves, metres)
                   for m, d in labels.get(node, [])):
                continue
            for toward in neighbours.get(node, []):
                step = move_from(node, toward)
                if step is None:
                    continue
                end, length = step
                cost = (moves + 1, metres + length)
                kept = labels.setdefault(end, [])
                if any(m <= cost[0] and d <= cost[1] for m, d in kept):
                    continue
                kept[:] = [(m, d) for m, d in kept if not (cost[0] <= m and cost[1] <= d)]
                kept.append(cost)
                if len(kept) > self.MAX_LABELS:
                    kept.sort()
                    del kept[self.MAX_LABELS:]
                    if cost not in kept:
                        continue
                heapq.heappush(queue, (cost[0], cost[1], end, toward if first is None else first))
        if not arrivals:
            return None
        on_time = [a for a in arrivals if budget_cm is None or a[1] <= budget_cm]
        moves, metres, toward = min(on_time) if on_time else min(arrivals, key=lambda a: (a[1], a[0]))
        if toward is None:
            return None
        row = {r["node"]: r for r in env.candidates()}.get(toward)
        return (row["k"], row["node"]) if row is not None else None


def run_shortest_path_courier(env: CourierEnv, seed: int,
                              *, max_steps: int = 4000) -> RouterResult:
    env.reset()
    return ShortestPathCourier(env, max_steps=max_steps).run(seed)


def run_fewest_moves_courier(env: CourierEnv, seed: int,
                             *, max_steps: int = 4000) -> RouterResult:
    env.reset()
    return FewestMovesCourier(env, max_steps=max_steps).run(seed)
