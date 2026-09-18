from __future__ import annotations
from typing import Any, Dict, List, Optional
import math

# POI kind constants (assigned by Map._build_addresses from progen_world_enriched.json)
_POI_CHARGING = "charging_station"
_POI_REST     = "rest_area"
_POI_STORE    = "store"

# Tolerance in cm — matches PICKUP handler (600cm)
_PROXIMITY_CM = 600.0


def _dist_cm(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x1 - x2, y1 - y2)


def _near_poi_kind(dm, kind: str) -> bool:
    """Return True if dm is within _PROXIMITY_CM of any POI with the given type."""
    for poi in dm.city_map.poi_meta:
        node = poi.get("node")
        if node is None or getattr(node, "type", None) != kind:
            continue
        dock = poi.get("dock_node")
        if dock is None:
            continue
        px, py = float(dock.position.x), float(dock.position.y)
        if _dist_cm(dm.x, dm.y, px, py) <= _PROXIMITY_CM:
            return True
    return False


def _near_any_pickup(dm: Any) -> bool:
    """Return True if dm is within tolerance of any accepted order's pickup door."""
    for order in dm.active_orders:
        if order.id in dm.carrying: 
            continue
        node = getattr(order, "pickup_node", None)
        if node is None:
            continue
        px, py = float(node.position.x), float(node.position.y)
        if _dist_cm(dm.x, dm.y, px, py) <= _PROXIMITY_CM:
            return True
    return False


def _near_any_dropoff(dm: Any) -> bool:
    """Return True if dm is within tolerance of any carrying order's dropoff door."""
    for order in dm.active_orders:
        if order.id not in dm.carrying:
            continue
        node = getattr(order, "dropoff_node", None)
        if node is None:
            continue
        dx, dy = float(node.position.x), float(node.position.y)
        if _dist_cm(dm.x, dm.y, dx, dy) <= _PROXIMITY_CM:
            return True
    return False


def get_valid_actions(dm: Any, config: Optional[Dict] = None) -> List[str]:
    """
    Return the list of actions that are executable given the agent's
    current state and position.  This is applied ON TOP OF the static
    ``enabled_actions`` whitelist (stage-level gating).
    """
    if config is None:
        config = getattr(dm, "cfg", {}) or {}

    valid: List[str] = []

    # -- always available --
    valid.append("VIEW_ORDERS")
    valid.append("WAIT")

    # -- movement (blocked while rescued or on a bus) --
    if not dm.is_rescued and not dm._bus_ctx:
        valid.append("MOVE")
        valid.append("PASSBY")

    # -- order lifecycle --
    om = dm._order_manager
    if om is not None and om.list_orders():
        valid.append("ACCEPT_ORDER")

    if dm.active_orders and _near_any_pickup(dm):
        valid.append("PICKUP")

    if dm.carrying and _near_any_dropoff(dm):
        valid.append("DROP_OFF")

    # -- battery / charging --
    if config.get("enable_battery", True):
        esc = dm.e_scooter
        if esc and esc.battery_pct < 100 and _near_poi_kind(dm, _POI_CHARGING):
            valid.append("CHARGE")
        if dm.inventory.get("escooter_battery_pack", 0) > 0:
            valid.append("USE_BATTERY_PACK")

    # -- walking energy / rest --
    if config.get("enable_walking_energy", True):
        if dm.energy_pct < 100 and _near_poi_kind(dm, _POI_REST):
            valid.append("REST")
        if dm.inventory.get("energy_drink", 0) > 0:
            valid.append("USE_ENERGY_DRINK")

    # -- store purchases --
    if _near_poi_kind(dm, _POI_STORE):
        valid.append("BUY")

    # -- transport switching --
    if config.get("enable_battery", True) or config.get("enable_advanced_transport", True):
        if dm.e_scooter is not None or dm.car is not None:
            valid.append("SWITCH")

    # -- bag management --
    if config.get("enable_bag_compartments", True) and dm.carrying:
        valid.append("PLACE_FOOD_IN_BAG")
        valid.append("VIEW_BAG")

    if config.get("enable_food_temperature", True):
        if dm.inventory.get("ice_pack", 0) > 0:
            valid.append("USE_ICE_PACK")
        if dm.inventory.get("heat_pack", 0) > 0:
            valid.append("USE_HEAT_PACK")

    # -- advanced transport (location-dependent) --
    if config.get("enable_advanced_transport", True):
        if dm.car is None and _near_poi_kind(dm, "car_rental"):
            valid.append("RENT_CAR")
        if dm.car is not None and _near_poi_kind(dm, "car_rental"):
            valid.append("RETURN_CAR")
        if dm._bus_manager:
            valid.append("VIEW_BUS_SCHEDULE")
            if _near_poi_kind(dm, "bus_station"):
                valid.append("BOARD_BUS")

    # -- navigation (now a regular action; transport mode is a parameter,
    #    gated inside the handler by enable_battery / enable_advanced_transport) --
    if not dm.is_rescued and not dm._bus_ctx:
        valid.append("NAVIGATE")

    # -- intersect with stage-level whitelist (augmented by mechanic flags) --
    from .action_space import effective_enabled_actions
    ea = effective_enabled_actions(config)
    if ea is not None:
        allowed = set(ea)
        valid = [a for a in valid if a in allowed]

    return valid


def get_valid_tools(dm: Any, config: Optional[Dict] = None) -> List[str]:
    """Deprecated: NAVIGATE is now a regular action (see get_valid_actions).

    There are no query-only tools, so this always returns an empty list.
    """
    return []


def format_valid_actions(actions: List[str], tools: Optional[List[str]] = None) -> str:
    """Format action and tool lists as observation text blocks.

    When *tools* is provided (including an empty list), a separate
    ``### available_tools`` section is appended after ``### available_actions``.
    Omitting *tools* (default ``None``) preserves the original single-section output.
    """
    parts: List[str] = []

    if not actions:
        parts.append("### available_actions\nNo actions available.")
    else:
        lines = ["### available_actions", "Actions you can take right now:"]
        for a in actions:
            lines.append(f"  - {a}")
        parts.append("\n".join(lines))

    if tools is not None:
        if not tools:
            parts.append("### available_tools\nNo tools available.")
        else:
            lines = [
                "### available_tools",
                "Tools are query-only and do not mutate position, clock, energy, scooter "
                "battery, orders, bus state, reward, training, lifecycle, or task generation:",
            ]
            for t in tools:
                lines.append(f"  - {t}")
            parts.append("\n".join(lines))

    return "\n".join(parts)
