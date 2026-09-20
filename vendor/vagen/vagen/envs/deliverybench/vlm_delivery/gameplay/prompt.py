# gameplay/prompt.py
# -*- coding: utf-8 -*-

"""
Prompt.py
- English system prompt for the DeliveryMan VLM agent, written in paragraphs (no bullet lists).
"""

from typing import Any, Dict, Optional

from .action_space import get_action_spec, get_tool_spec, get_output_examples

_SYSTEM_PROMPT_TEMPLATE = """\
You are a food-delivery courier in a simulated city. Your primary goal is to earn as much money as possible through delivering food to customers within two hours. Do your best to complete each delivery and don't let your energy drop to 0.

**Action space:**
{action_api_spec}
You can refer to the following concrete action examples:
{output_examples}

{workflow_rules}

**Output:**
Make your decision based on your observation while following the rules. 
Return ONLY a valid JSON object with the following three keys:
{{
"reasoning_and_reflection": "<=60 tokens: state the relevant observation and why the chosen action follows the rules>",
"action": "Your next action as a single-line function call, strictly following the Action space specification",
"future_plan": "<=40 tokens: a concise next-step plan in natural language"
}}
Do not include prose, code fences, or text outside the JSON.
"""


def _fmt(v: float) -> str:
    """Format a number: drop trailing zeros but keep at least one decimal."""
    return f"{v:g}"


def _extract_prompt_values(cfg: Dict[str, Any]) -> Dict[str, str]:
    speeds = cfg.get("avg_speed_cm_s", {})
    energy = cfg.get("energy_pct_decay_per_m_by_mode", {})
    car_defs = cfg.get("rent_car_defaults", {})
    es_defs = cfg.get("escooter_defaults", {})

    return {
        "walk_speed":               _fmt(speeds.get("walk", 200) / 100),
        "walk_energy":              _fmt(energy.get("walk", 0.08)),
        "scooter_speed":            _fmt(speeds.get("e-scooter", 600) / 100),
        "scooter_energy":           _fmt(energy.get("e-scooter", 0.01)),
        "scooter_batt":             _fmt(cfg.get("scooter_batt_decay_pct_per_m", 0.04)),
        "drag_speed":               _fmt(speeds.get("drag_scooter", 150) / 100),
        "drag_energy":              _fmt(energy.get("drag_scooter", 0.1)),
        "car_speed":                _fmt(speeds.get("car", 1200) / 100),
        "car_energy":               _fmt(energy.get("car", 0.008)),
        "car_rate_per_min":         _fmt(car_defs.get("rate_per_min", 1.0)),
        "bus_speed":                _fmt(speeds.get("bus", 1000) / 100),
        "bus_energy":               _fmt(energy.get("bus", 0.006)),
        "charge_price_per_pct":     _fmt(cfg.get("charge_price_per_percent", 0.05)),
        "charge_rate_pct_per_min":  _fmt(es_defs.get("charge_rate_pct_per_min", 7.5)),
    }


def _workflow_rules(cfg: Optional[Dict[str, Any]]) -> str:
    cfg = cfg or {}
    enabled = cfg.get("enabled_actions")
    if enabled is not None and "NAVIGATE" not in set(enabled):
        return ""
    return """\
**Delivery workflow and navigation discipline:**
When you have no active orders, call `VIEW_ORDERS()` and compare the available orders before accepting. Prefer orders that are reachable quickly from your current position, pay well, and have enough time left. If several orders share a nearby pickup or route, you may accept multiple ids with `ACCEPT_ORDER([id1, id2, ...])`; otherwise accept one good order first and finish it.

When `active_orders` shows a known Pickup or Dropoff address, treat that address as your destination. After accepting an order, call NAVIGATE with the actual Pickup address string shown in `active_orders` before moving toward the pickup; for example, if the observation says `Pickup : 146 Church Ave`, call `NAVIGATE(target="146 Church Ave")`. Do not write placeholder text such as `<exact Pickup address>`, and do not guess the route from street names or the map alone. After `PICKUP` succeeds, call NAVIGATE with the actual Dropoff address string shown in `active_orders` before moving toward the customer. Use the exact address string shown in `active_orders` whenever possible, rather than a generic place name.

After NAVIGATE succeeds, follow the refreshed `next_move` exactly until it says `you have arrived`. Translate `next_move` directly to MOVE. When `next_move` says `you have arrived`, stop moving: call `PICKUP` if you are at the pickup and the order is ready, or call `DROP_OFF` if you are carrying the order at the dropoff.
"""


def get_system_prompt(cfg: Optional[Dict[str, Any]] = None) -> str:
    """Build the system prompt, filling transport stats from *cfg*.

    If *cfg* is ``None``, the default values hard-coded in
    ``_extract_prompt_values`` are used (they mirror
    ``game_mechanics_config.json``).
    """
    vals = _extract_prompt_values(cfg or {})
    vals["action_api_spec"] = get_action_spec(cfg)
    vals["tool_api_spec"] = get_tool_spec(cfg)
    vals["output_examples"] = get_output_examples(cfg)
    vals["workflow_rules"] = _workflow_rules(cfg)
    prompt = _SYSTEM_PROMPT_TEMPLATE.format(**vals)
    
    if cfg:
        remove_kws = []
        if not cfg.get("enable_battery", True):
            remove_kws += ["battery", "CHARGE", "charging_station", "USE_BATTERY_PACK", "drag"]
        if not cfg.get("enable_walking_energy", True):
            remove_kws += ["energy_drink", "USE_ENERGY_DRINK", "hospitalized", "energy per meter", "REST"]
        if not cfg.get("enable_advanced_transport", True):
            remove_kws += ["RENT_CAR", "RETURN_CAR", "BOARD_BUS", "bus transportation", "VIEW_BUS", "Bus speed"]
            if not cfg.get("enable_battery", True):
                # When both transport and battery are off, SWITCH has no purpose.
                remove_kws += ["Use SWITCH to switch to different transport"]
        if not cfg.get("enable_multi_agent", True):
            remove_kws += ["VIEW_HELP", "POST_HELP", "ACCEPT_HELP", "TEMP_BOX", "SAY(", "REPORT_HELP", "helping others"]
        if not cfg.get("enable_bag_compartments", True):
            remove_kws += ["PLACE_FOOD_IN_BAG", "insulated bag", "arrange them into"]
        if not cfg.get("enable_food_temperature", True):
            remove_kws += ["ice/heat packs", "USE_ICE_PACK", "USE_HEAT_PACK", "temperature requirements"]
        if not cfg.get("enable_delivery_methods", True):
            remove_kws += ["hand_to_customer", "required method",
                           "egocentric (first-person) view to locate", "leave the order at the door"]
        if remove_kws:
            lines = prompt.split('\n')
            lines = [l for l in lines if not any(kw in l for kw in remove_kws)]
            prompt = '\n'.join(lines)

    return prompt


# Keep a module-level constant for backward compatibility with code that
# imports SYSTEM_PROMPT directly (uses default config values).
SYSTEM_PROMPT = get_system_prompt()
