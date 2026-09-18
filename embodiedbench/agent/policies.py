"""Policies that produce actions from observations.

M1 needs one deterministic policy so the walking skeleton has something to drive
it. It is a program, not an agent: no model, no sampling, no wall-clock.

The scripted courier reads privileged state through the runtime handle to plan
its route. That is legitimate for a harness/oracle policy (the design plan reserves
privileged access for evaluators and harness tooling) and it must never be used
as a benchmark policy — ``uses_privileged_state`` says so in a way the harness
records into trajectory metadata rather than leaving to a comment.
"""

from __future__ import annotations

import json
from typing import Any, Protocol

from embodiedbench.runtime.text.vagen_adapter import POSITIONAL_KEY
from embodiedbench.schemas.runtime import ActionEnvelope, Observation, TaskAction


class Policy(Protocol):
    """What the harness needs from anything that chooses actions."""

    name: str
    uses_privileged_state: bool

    def act(self, observation: Observation, runtime: Any, step_index: int) -> ActionEnvelope | None:
        """Return the next action, or None to end the episode."""
        ...


class ScriptedCourierPolicy:
    """A deterministic full-delivery program: view, accept, route, pick up, drop off."""

    name = "scripted_courier"
    uses_privileged_state = True

    def __init__(self, *, order_index: int = 0, max_moves_per_leg: int = 120):
        self.order_index = order_index
        self.max_moves_per_leg = max_moves_per_leg
        self._phase = "view"
        self._moves = 0
        self.raw_outputs: list[str] = []

    def reset(self) -> None:
        self._phase = "view"
        self._moves = 0
        self.raw_outputs = []

    def _emit(self, episode_id: str, step_index: int, name: str, **arguments: Any) -> ActionEnvelope:
        action = TaskAction(name=name, arguments=arguments)
        envelope = ActionEnvelope(episode_id=episode_id, step_index=step_index, action=action)
        # A trajectory records raw model output alongside the parsed action
        # (design plan §10.2, 10.4). A scripted policy has no model text, so it
        # records the JSON it would have emitted rather than leaving the field
        # empty and making the trajectory schema untested on that path.
        self.raw_outputs.append(json.dumps({"action": {"name": name, "arguments": arguments}}))
        return envelope

    def act(self, observation: Observation, runtime: Any, step_index: int) -> ActionEnvelope | None:
        episode_id = observation.episode_id
        dm = runtime._dm()
        if dm is None:
            return None

        if self._phase == "view":
            self._phase = "accept"
            return self._emit(episode_id, step_index, "VIEW_ORDERS")

        if self._phase == "accept":
            self._phase = "to_pickup"
            self._moves = 0
            # ACCEPT_ORDER takes a positional index in the vendored grammar.
            return self._emit(
                episode_id, step_index, "ACCEPT_ORDER", **{POSITIONAL_KEY: [self.order_index]}
            )

        active = list(getattr(dm, "active_orders", []) or [])
        if not active:
            return None
        order = active[0]

        if self._phase == "to_pickup":
            direction = _direction_toward(dm, order.pickup_node)
            if direction is None:
                self._phase = "pickup"
            elif self._moves >= self.max_moves_per_leg:
                return None
            else:
                self._moves += 1
                return self._emit(episode_id, step_index, "MOVE", direction=direction)

        if self._phase == "pickup":
            self._phase = "to_dropoff"
            self._moves = 0
            return self._emit(episode_id, step_index, "PICKUP", orders=[self.order_index])

        if self._phase == "to_dropoff":
            direction = _direction_toward(dm, order.dropoff_node)
            if direction is None:
                self._phase = "dropoff"
            elif self._moves >= self.max_moves_per_leg:
                return None
            else:
                self._moves += 1
                return self._emit(episode_id, step_index, "MOVE", direction=direction)

        if self._phase == "dropoff":
            self._phase = "done"
            return self._emit(episode_id, step_index, "DROP_OFF", oid=self.order_index)

        return None


def _direction_toward(dm: Any, target_node: Any) -> str | None:
    """One facing-relative step along the shortest path, or None if arrived."""
    from vagen.envs.deliverybench.vlm_delivery.actions.move import available_moves

    city_map = dm.city_map
    current = city_map.nearest_waypoint(float(dm.x), float(dm.y))
    if current is target_node:
        return None
    path, _cost = city_map.waypoint_graph.shortest_path_nodes(current, target_node)
    if not path or len(path) < 2:
        return None
    next_node = path[1]
    for direction, candidate in available_moves(dm).items():
        if candidate and candidate.get("node") is next_node:
            return direction
    return None
