# gameplay/action_space.py
# -*- coding: utf-8 -*-

from __future__ import annotations
import ast
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from ..utils.util import get_tol

# Arrival tolerance (cm) for being "at the pickup/dropoff door". Order doors are
# building entrances offset from the street waypoint graph (order.py: pickup_node
# = door_node), so the nearest STEP_TO-reachable waypoint can sit several metres
# from the door. A feasibility sweep of small-city-11 found door→nearest-waypoint
# distances up to 8.16 m (≈71% of doors exceed the old 6 m), making most orders
# impossible to PICKUP/DROP_OFF via STEP_TO when MOVE is disabled. 10 m covers
# 100% of measured doors with margin while keeping waypoint navigation as the
# task. Override per-config via cfg["tolerance_cm"]["door"].
_DOOR_TOL_CM_DEFAULT = 1000.0


def _door_tol_cm(dm: Any) -> float:
    return get_tol(getattr(dm, "cfg", {}) or {}, "door", _DOOR_TOL_CM_DEFAULT)
_RESOURCE_ACTIONS = {"CHARGE", "REST", "BUY", "USE_BATTERY_PACK", "USE_ENERGY_DRINK", "SWITCH"}
_PACKING_ACTIONS = {"PLACE_FOOD_IN_BAG", "USE_ICE_PACK", "USE_HEAT_PACK", "VIEW_BAG"}
_TRANSPORT_ACTIONS = {"RENT_CAR", "RETURN_CAR", "BOARD_BUS", "VIEW_BUS_SCHEDULE"}
_CORE_ACTIONS = {"VIEW_ORDERS", "ACCEPT_ORDER", "MOVE", "PICKUP", "DROP_OFF", "WAIT"}
# NAVIGATE is now a regular action (not a query-only tool), so there are no
# tool-only names. Kept as an (empty) set for backward compatibility.
_TOOL_NAMES: set = set()

# ─────────────────────────────────────────────────────────────────────────────
# Mechanic → action vocabulary coupling.
#
# When a mechanic flag is on, the listed actions are automatically added to the
# effective `enabled_actions` whitelist. This prevents the common silent-bug
# where someone sets `enable_battery=true` in a config but forgets to add
# CHARGE to enabled_actions — the agent then never sees CHARGE and the
# mechanic is untestable. See effective_enabled_actions() below.
# ─────────────────────────────────────────────────────────────────────────────
_MECHANIC_ACTIONS: Dict[str, set] = {
    "enable_battery":            {"CHARGE", "USE_BATTERY_PACK", "SWITCH", "BUY"},
    "enable_walking_energy":     {"REST", "USE_ENERGY_DRINK", "BUY"},
    "enable_food_temperature":   {"USE_ICE_PACK", "USE_HEAT_PACK", "BUY"},
    "enable_bag_compartments":   {"PLACE_FOOD_IN_BAG", "VIEW_BAG"},
    "enable_advanced_transport": {"RENT_CAR", "RETURN_CAR", "BOARD_BUS",
                                  "VIEW_BUS_SCHEDULE", "SWITCH"},
    # Pluggable static hazards: obstacles need BYPASS()/PASSBY() to get past a
    # blocked edge; traffic lights use ordinary WAIT(minutes=...) to wait for
    # the next phase change.
    "enable_obstacles":          {"PASSBY"},
    "enable_traffic_lights":     {"WAIT"},
    # Marked-waypoint navigation: MOVE_TO(k) steps to any numbered one-hop
    # waypoint (FPV markers + per-step candidate text). MOVE stays available
    # alongside it.
    "enable_waypoint_marks":     {"MOVE_TO"},
}


def effective_enabled_actions(config: Optional[Dict[str, Any]]) -> Optional[List[str]]:
    """Return the stage's effective action whitelist, augmented by mechanic flags.

    If `enabled_actions` is None, returns None (no whitelist filtering).
    Otherwise returns the explicit list ∪ actions required by every enabled
    mechanic flag.
    """
    if not config:
        return None
    ea = config.get("enabled_actions")
    if ea is None:
        return None
    augmented = set(ea)
    for flag, required in _MECHANIC_ACTIONS.items():
        if config.get(flag, False):
            augmented |= required
    return sorted(augmented)


__all__ = [
    "ACTION_API_SPEC",
    "_TOOL_NAMES",
    "get_tool_spec",
    "sanitize_model_text",
    "parse_action",
    "action_to_text",
]

# =========================================================
# Global switch: single-agent or multi-agent
# True  = enable multi-agent commands (default)
# False = hide all multi-agent–only commands
# =========================================================
IS_MULTI_AGENT = (os.getenv("DELIVERYBENCH_MULTI_AGENT", "1") == "1")


# =========================================================
# 1) Multi-agent command definitions
#     We use canonical targets to ensure full coverage.
#     Any feature whose canonical target is in this set
#     will be hidden entirely when IS_MULTI_AGENT = False.
# =========================================================
_MULTI_AGENT_CANON_TARGETS = {
    "VIEW_HELP_BOARD",
    "POST_HELP",
    "ACCEPT_HELP",
    "EDIT_HELP",
    "PLACE_TEMP_BOX",
    "TAKE_FROM_TEMP_BOX",
    "REPORT_HELP_FINISHED",
    "SAY",
}

# =========================================================
# 2) Helper for filtering API specs & examples (multi-agent)
#    - _filter_multi_agent_lines(...) removes lines that mention
#      any canonical multi-agent target when IS_MULTI_AGENT=False.
# =========================================================
def _filter_multi_agent_lines(text: str) -> str:
    """
    Remove lines that contain multi-agent commands when IS_MULTI_AGENT=False.
    This is based on canonical command names to ensure lossless filtering.
    """
    if IS_MULTI_AGENT:
        return text

    filtered = []
    for line in text.splitlines():
        up = line.upper()
        # hide line if it references any multi-agent canonical target
        if any(name in up for name in _MULTI_AGENT_CANON_TARGETS):
            continue
        filtered.append(line)
    return "\n".join(filtered)


# =========================================================
# 3) Action API spec & output examples for the model
#    - ACTION_API_SPEC: the command cheat-sheet shown to the model.
#    - OUTPUT_EXAMPLES: example calls that demonstrate valid syntax.
#    * Both are filtered by _filter_multi_agent_lines depending on IS_MULTI_AGENT.
# =========================================================
_ACTION_API_SPEC_ACTIONS: str = _filter_multi_agent_lines(r"""
COMMANDS (UPPERCASE):
- VIEW_ORDERS()     # view all available orders
- VIEW_BAG()        # view your bag
- ACCEPT_ORDER(order_id) or ACCEPT_ORDER([order_id, ...])
- MOVE(direction="forward"|"left"|"right"|"backward")  # step to the adjacent waypoint in that direction relative to your facing; your facing turns to match (right=+90, backward=180, left=-90)
- PASSBY()  # alias BYPASS(). Step forward to the next waypoint at 1.5x the time/energy of MOVE(forward); facing unchanged.
- PICKUP(orders=[12, 18])
- PLACE_FOOD_IN_BAG(bag_cmd="order 12: 1,2 -> A; 3 -> B")
- CHARGE(target_pct=100)
- WAIT(minutes=NN) or WAIT("charge_done")
- REST(target_pct=100)
- BUY(item_id="energy_drink", qty=1)
- USE_BATTERY_PACK()
- USE_ENERGY_DRINK()
- USE_ICE_PACK(comp="A")
- USE_HEAT_PACK(comp="B")
- SWITCH(to="walk"|"e-scooter"|"car")
- RENT_CAR()
- RETURN_CAR()
- VIEW_HELP_BOARD()
- POST_HELP(kind="HELP_PICKUP"|"HELP_DELIVERY"|"HELP_BUY"|"HELP_CHARGE", bounty=5.0, ttl_s=300, payload={...})
- ACCEPT_HELP(req_id=123)
- EDIT_HELP(req_id=123, new_bounty=6.0, new_ttl_min=10)
- PLACE_TEMP_BOX(req_id=123, location=(110.0m, 445.0m), content={...})
- TAKE_FROM_TEMP_BOX(req_id=123)
- REPORT_HELP_FINISHED(req_id=123)
- DROP_OFF(oid=<int>, method="leave_at_door|knock|call|hand_to_customer")
- SAY("text")
- SAY(to="agent_id", text="text")
- BOARD_BUS(bus_id="bus_id", target_stop_id="target_stop_id")
- VIEW_BUS_SCHEDULE()
""")

# MOVE_TO spec bullet — only shown when enable_waypoint_marks is on (see
# get_action_spec). Kept out of _ACTION_API_SPEC_ACTIONS so stages without
# the flag get a byte-identical spec.
_MOVE_TO_SPEC_LINE = (
    '- MOVE_TO(k)  # marked-waypoint step: every adjacent waypoint you can '
    'reach in one step is shown as a numbered marker in the first-person '
    'views and listed with the same numbers in the observation text; pick '
    'one number to walk to that waypoint. Your facing turns toward it. Also '
    'accepts the waypoint id from the list, e.g. MOVE_TO("dock_94").'
)

# NAVIGATE is now an action; its bullet is built dynamically by
# _navigate_line() so the advertised transport modes track the config.
# This block is kept (empty) only for backward-compatible references.
_ACTION_API_SPEC_TOOLS: str = ""

# Backward-compatible concatenation — any code that imports ACTION_API_SPEC still works.
_FULL_ACTION_API_SPEC: str = _ACTION_API_SPEC_ACTIONS + _ACTION_API_SPEC_TOOLS

_FULL_OUTPUT_EXAMPLES: str = _filter_multi_agent_lines(r"""
VIEW_ORDERS()
VIEW_BAG()
MOVE(direction="forward")
MOVE(direction="left")
MOVE(direction="right")
PASSBY()
PLACE_FOOD_IN_BAG(bag_cmd="order 12: 1,2 -> A; 3 -> B")
CHARGE(target_pct=100)
WAIT(minutes=3)
WAIT("charge_done")
BUY(item="energy_drink", qty=1)
SWITCH(to="e-scooter")
RENT_CAR()
VIEW_HELP_BOARD()
POST_HELP(kind="HELP_PICKUP", bounty=5.0, ttl_s=600, payload={"order_id":21})
PLACE_TEMP_BOX(req_id=77, location=(110.0m, 445.0m), content={"inventory":{"energy_drink":1}})
TAKE_FROM_TEMP_BOX(req_id=77)
REPORT_HELP_FINISHED(req_id=77)
SAY("Hi all")
BOARD_BUS(bus_id="bus_id", target_stop_id="target_stop_id")
VIEW_BUS_SCHEDULE()
NAVIGATE(target="restaurant 1")
""")

ACTION_API_SPEC: str = _FULL_ACTION_API_SPEC
OUTPUT_EXAMPLES: str = _filter_multi_agent_lines(_FULL_OUTPUT_EXAMPLES)


def _filter_action_lines(text: str, disabled_targets: set) -> str:
    if not disabled_targets:
        return text
    filtered = []
    for line in text.splitlines():
        stripped = line.strip().lstrip("- ")
        cmd_name = stripped.split("(")[0].split()[0].upper() if stripped else ""
        if cmd_name in disabled_targets:
            continue
        filtered.append(line)
    return "\n".join(filtered)

def _navigate_line(config) -> str:
    """Build the NAVIGATE action bullet, advertising only the modes enabled
    by the current config (walk always; e-scooter with battery; bus with
    advanced transport)."""
    config = config or {}
    modes = ["walk"]
    if config.get("enable_battery", True):
        modes.append("e-scooter")
    if config.get("enable_advanced_transport", True):
        modes.append("bus")
    mode_str = "|".join(f'"{m}"' for m in modes)
    if str(config.get("task_mode", "")).lower() == "visual_route_following":
        line = (
            f'- NAVIGATE(target="<address or waypoint>", mode={mode_str})'
            '  # Plan a route to target and highlight the active route on the '
            '2D city map. Use it once whenever you choose a new destination; '
            'do not repeat NAVIGATE every turn. In visual route-following mode, '
            'NAVIGATE does not provide a textual next-step hint; follow the map '
            'route image and observation state. mode defaults to "walk".'
        )
        if "bus" in modes:
            line += ' For mode="bus": optional access_mode/egress_mode = "auto"|"walk"|"scooter"; also reports wait time + fare.'
        return line
    obstacle_clause = ""
    if config.get("enable_obstacles", False):
        obstacle_clause = (
            " If next_move is move forward but the FRONT first-person view shows "
            "a road block or obstacle on that forward route, use the PASSBY() "
            "action."
        )
    line = (
        f'- NAVIGATE(target="<address or waypoint>", mode={mode_str})'
        '  # Plan a route to target (an order pickup/dropoff address, place name, '
        'or waypoint id). It highlights the active route on the 2D city map as a '
        'blue line from your current location toward the destination, and '
        're-draws it from your new position after every move so the blue line '
        'continues to show the route. Small arrows on longer blue segments show '
        'route direction when present, but they may be absent on short segments '
        'or near the final step. The orange arrow attached to your marker shows '
        'only your current facing direction; compare it with the nearby blue '
        'line segment to choose forward/left/right/backward. It also '
        'reports the remaining distance '
        'and estimated travel time' +
        ('/energy' ) + ('/battery' if 'e-scooter' in modes else '') +
        '. Use it once whenever you choose a new destination; do not repeat '
        'NAVIGATE every turn. After a successful NAVIGATE, later observations '
        'keep the [navigation] block and refresh next_move until you arrive or '
        'choose a different destination. Treat next_move as authoritative and '
        'translate it directly to MOVE: move forward -> MOVE(direction="forward"), '
        'move backward -> MOVE(direction="backward"), turn left -> '
        'MOVE(direction="left"), turn right -> MOVE(direction="right"). Do not '
        'override next_move based on street numbers or your own guess.' +
        obstacle_clause +
        ' Only otherwise deviate if the move is '
        'impossible or the direction is blocked/action failed. mode defaults '
        'to "walk".'
    )
    if "bus" in modes:
        line += ' For mode="bus": optional access_mode/egress_mode = "auto"|"walk"|"scooter"; also reports wait time + fare.'
    return line


def get_action_spec(config) -> str:
    """Return the actions section of the API spec, filtered by config.

    NAVIGATE is included as a regular action (its modes track the config).
    """
    if config is None:
        config = {}
    disabled = set()
    if not config.get("enable_battery", True):
        disabled |= {"CHARGE", "USE_BATTERY_PACK"}
    if not config.get("enable_walking_energy", True):
        disabled |= {"REST", "USE_ENERGY_DRINK"}
    if not config.get("enable_food_temperature", True):
        disabled |= {"USE_ICE_PACK", "USE_HEAT_PACK"}
    if not config.get("enable_bag_compartments", True):
        disabled |= {"PLACE_FOOD_IN_BAG", "VIEW_BAG"}
    if not config.get("enable_advanced_transport", True):
        disabled |= {"RENT_CAR", "RETURN_CAR", "BOARD_BUS", "VIEW_BUS_SCHEDULE"}
        # SWITCH is still needed for walk ↔ e-scooter when battery is enabled.
        if not config.get("enable_battery", True):
            disabled |= {"SWITCH"}
    if not config.get("enable_multi_agent", True):
        disabled |= _MULTI_AGENT_CANON_TARGETS
    if (not config.get("enable_battery", True)
            and not config.get("enable_walking_energy", True)
            and not config.get("enable_food_temperature", True)):
        disabled |= {"BUY"}
    ea = effective_enabled_actions(config)
    if ea is not None:
        all_actions = set(_CANON_RAW.values()) - _TOOL_NAMES
        disabled |= (all_actions - set(ea))
    spec = _filter_action_lines(_ACTION_API_SPEC_ACTIONS, disabled)
    if not config.get("enable_delivery_methods", True):
        spec = spec.replace(
            'DROP_OFF(oid=<int>, method="leave_at_door|knock|call|hand_to_customer")',
            'DROP_OFF(oid=<int>)',
        )
    # When obstacles are active, BYPASS/PASSBY is the *only* way past a blocked
    # edge, so the spec must say what it is for. The hazard stays vision-only —
    # the per-step text never names the obstacle — but the agent has to know the
    # action exists for this purpose. Scoped to the flag so obstacle-free stages
    # keep the neutral wording. (PASSBY is auto-added to the whitelist by
    # _MECHANIC_ACTIONS when enable_obstacles is on.)
    if config.get("enable_obstacles", False):
        spec = spec.replace(
            '- PASSBY()  # alias BYPASS(). Step forward to the next waypoint at 1.5x the time/energy of MOVE(forward); facing unchanged.',
            '- PASSBY()  # alias BYPASS(). Detour one step around an obstacle blocking '
            'the path straight ahead: it reaches the same next waypoint as '
            'MOVE(direction="forward") but costs ~1.5x the time/energy, so choose it '
            'instead of MOVE(forward) whenever the way directly ahead is blocked. On a '
            'clear edge it is just a costlier forward step.',
        )
    # Marked-waypoint navigation (enable_waypoint_marks): advertise MOVE_TO
    # right under the MOVE bullet so the two locomotion styles read side by
    # side. Inserted here rather than kept in the static spec so stages
    # without the flag keep a byte-identical action list.
    if config.get("enable_waypoint_marks", False):
        rebuilt: List[str] = []
        inserted = False
        for ln in spec.splitlines():
            rebuilt.append(ln)
            if not inserted and ln.strip().startswith("- MOVE(direction="):
                rebuilt.append(_MOVE_TO_SPEC_LINE)
                inserted = True
        spec = "\n".join(rebuilt)
        if not inserted:
            spec = spec.rstrip("\n") + "\n" + _MOVE_TO_SPEC_LINE
    # NAVIGATE is an action; append it (gated by enabled_actions) so it shows
    # in the action_api_spec alongside the primitive commands.
    if (ea is None) or ("NAVIGATE" in set(ea)):
        spec = spec.rstrip("\n") + "\n" + _navigate_line(config)
    return spec


def get_tool_spec(config) -> str:
    """Deprecated: NAVIGATE is now a regular action (see get_action_spec).

    There are no query-only tools, so this returns an empty string. Kept for
    backward compatibility with callers that still request a tool spec.
    """
    return ""


def get_output_examples(config) -> str:
    if config is None:
        config = {}
    disabled = set()
    if not config.get("enable_battery", True):
        disabled |= {"CHARGE", "USE_BATTERY_PACK"}
    if not config.get("enable_walking_energy", True):
        disabled |= {"REST", "USE_ENERGY_DRINK"}
    if not config.get("enable_food_temperature", True):
        disabled |= {"USE_ICE_PACK", "USE_HEAT_PACK"}
    if not config.get("enable_bag_compartments", True):
        disabled |= {"PLACE_FOOD_IN_BAG", "VIEW_BAG"}
    if not config.get("enable_advanced_transport", True):
        disabled |= {"RENT_CAR", "RETURN_CAR", "BOARD_BUS", "VIEW_BUS_SCHEDULE"}
        if not config.get("enable_battery", True):
            disabled |= {"SWITCH"}
    if not config.get("enable_multi_agent", True):
        disabled |= _MULTI_AGENT_CANON_TARGETS
    if (not config.get("enable_battery", True)
            and not config.get("enable_walking_energy", True)
            and not config.get("enable_food_temperature", True)):
        disabled |= {"BUY"}
    ea = effective_enabled_actions(config)
    if ea is not None:
        all_actions = set(_CANON_RAW.values())
        disabled |= (all_actions - set(ea))
    examples = _filter_action_lines(_FULL_OUTPUT_EXAMPLES, disabled)
    # Gated like the spec bullet: only stages with waypoint marks see MOVE_TO.
    if config.get("enable_waypoint_marks", False):
        examples = examples.rstrip("\n") + '\nMOVE_TO(2)\nMOVE_TO("dock_94")'
    return examples
# =========================================================
# 4) Canonical mapping from raw names to ACTION names
#    - _CANON_RAW maps many aliases (MOVE_TO, PLACE_IN_BAG, ...) to
#      a smaller set of canonical names.
#    - _CANON is filtered by IS_MULTI_AGENT (multi-agent commands removed).
# =========================================================
_CANON_RAW = {
    "MOVE": "MOVE",
    "MOVE_TO": "MOVE_TO",
    "PASSBY": "PASSBY", "BYPASS": "PASSBY",

    "VIEW_ORDERS": "VIEW_ORDERS",
    "ACCEPT_ORDER": "ACCEPT_ORDER",

    "PICKUP": "PICKUP",
    "PLACE_FOOD_IN_BAG": "PLACE_FOOD_IN_BAG",
    "PLACE_IN_BAG": "PLACE_FOOD_IN_BAG",
    "PLACE_BAG": "PLACE_FOOD_IN_BAG",
    "BAG": "PLACE_FOOD_IN_BAG",

    "CHARGE": "CHARGE", "CHARGE_ESCOOTER": "CHARGE",
    "WAIT": "WAIT", "REST": "REST",
    "BUY": "BUY",
    "USE_BATTERY_PACK": "USE_BATTERY_PACK",
    "USE_ENERGY_DRINK": "USE_ENERGY_DRINK",

    "USE_ICE_PACK": "USE_ICE_PACK",
    "USE_HEAT_PACK": "USE_HEAT_PACK",

    "SWITCH": "SWITCH", "SWITCH_TRANSPORT": "SWITCH",
    "RENT_CAR": "RENT_CAR",
    "RETURN_CAR": "RETURN_CAR",

    "VIEW_HELP_BOARD": "VIEW_HELP_BOARD",
    "POST_HELP": "POST_HELP", "POST_HELP_REQUEST": "POST_HELP",
    "ACCEPT_HELP": "ACCEPT_HELP", "ACCEPT_HELP_REQUEST": "ACCEPT_HELP",
    "EDIT_HELP": "EDIT_HELP", "EDIT_HELP_REQUEST": "EDIT_HELP",
    "PLACE_TEMP_BOX": "PLACE_TEMP_BOX", "DROP_TEMP_BOX": "PLACE_TEMP_BOX",
    "TAKE_FROM_TEMP_BOX": "TAKE_FROM_TEMP_BOX", "TAKE_TEMP_BOX": "TAKE_FROM_TEMP_BOX",
    "REPORT_HELP_FINISHED": "REPORT_HELP_FINISHED", "REPORT_DONE": "REPORT_HELP_FINISHED",

    "DROP_OFF": "DROP_OFF", "DELIVER": "DROP_OFF",

    "SAY": "SAY",

    "BOARD_BUS": "BOARD_BUS",
    "VIEW_BUS_SCHEDULE": "VIEW_BUS_SCHEDULE",

    "NAVIGATE": "NAVIGATE",
}

# Generate final CANON based on global mode
_CANON = {
    k: v
    for k, v in _CANON_RAW.items()
    if IS_MULTI_AGENT or v not in _MULTI_AGENT_CANON_TARGETS
}

# =========================================================
# 5) Parsing helpers & action dispatcher
#    - sanitize_model_text: normalize raw model output into "NAME(...)" + optional plan.
#    - _convert_m_to_cm_in_args: rewrite 'Xm' numeric tokens to centimeters (safe for string literals).
#    - _ast_value: safe evaluator for a restricted expression AST (literals, containers, +).
#    - _parse_call: parse "NAME(...)" into (canonical_name, pos_args, kw_args).
#    - _orders_at_pickup / _lookup_order: resolve order objects from IDs / positions.
#    - parse_action: high-level dispatcher that turns one model line into a DMAction.
# =========================================================
_CALL_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)\s*$", re.DOTALL)
FENCE_RE = re.compile(r"^\s*```.*?\n(.*?)\n```", re.DOTALL)

def sanitize_model_text(text: str):
    """
    Normalize raw model output into (action_text, language_plan).

    Processing steps:
    1. Strip surrounding ``` fences if present.
    2. Try to parse the content as JSON:
       - If it is a dict, read:
         * future_plan -> language_plan (str)
         * action:
             - if str, use as action
             - if dict, use action["action_call"] if present
         * or fallback to "action_call" at the top level.
    3. If JSON parsing fails or no action is found:
       - Take the first line as the action (e.g., NAME(...)).

    Returns:
        Tuple[str, Optional[str]]:
            action        : cleaned action string, e.g. "MOVE(102.3m, 5.0m)"
            language_plan : cleaned natural-language plan if available, else None
    """
    t = (text or "").strip()

    # 1) Strip fenced code blocks if present
    m = FENCE_RE.search(t)
    if m:
        t = m.group(1).strip()

    action = None
    language_plan = None

    # 2) Try JSON: extract action and language_plan
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            # language_plan / future_plan may be a plain string
            lp = obj.get("future_plan")
            if isinstance(lp, str):
                language_plan = lp.strip()

            # action may be:
            #   - a plain string
            #   - a dict like {"action_call": "..."}
            if "action" in obj:
                act = obj["action"]
                if isinstance(act, str):
                    action = act.strip()
                elif isinstance(act, dict):
                    ac = act.get("action_call")
                    if isinstance(ac, str):
                        action = ac.strip()

            # compatibility: sometimes directly under "action_call"
            if action is None:
                ac = obj.get("action_call")
                if isinstance(ac, str):
                    action = ac.strip()
    except Exception:
        pass

    def _clean(s: str) -> str:
        # Unescape over-escaped quotes (e.g. NAVIGATE(target=\"x\", mode=\"walk\")
        # -> target="x", mode="walk") that survive when the action is scraped
        # from raw text instead of decoded from JSON. Then trim trailing
        # punctuation while keeping English text intact.
        s = s.replace('\\"', '"').replace("\\'", "'")
        return s.strip().rstrip(";，。.,！？!：:")

    # If JSON parse yielded an action, clean and return
    if action:
        return _clean(action), (_clean(language_plan) if language_plan else None)

    # 3) Non-JSON case: treat the first line as the action (e.g., NAME(...))
    first_line = t.splitlines()[0].strip() if t else ""
    return _clean(first_line), (_clean(language_plan) if language_plan else None)


# Replace numeric "<number>m" tokens (not inside string literals) with centimeters
_UNIT_NUM_M = re.compile(r'(?P<num>[-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*m\b')


def _convert_m_to_cm_in_args(s: str) -> str:
    """
    Convert plain numeric 'Xm' units (not inside string literals) into centimeters.

    Example:
        "MOVE(1.2m, 3m, label='3m away')" ->
        "MOVE(120, 300, label='3m away')"
    """
    out: List[str] = []
    i, n = 0, len(s)
    in_str, quote, esc = False, None, False

    while i < n:
        ch = s[i]

        if in_str:
            out.append(ch)
            if esc:
                esc = False
            else:
                if ch == '\\':
                    esc = True
                elif ch == quote:
                    in_str, quote = False, None
            i += 1
            continue

        m = _UNIT_NUM_M.match(s, i)
        if m:
            num_m = float(m.group('num'))
            num_cm = num_m * 100.0
            if abs(num_cm - round(num_cm)) < 1e-9:
                out.append(str(int(round(num_cm))))
            else:
                out.append(f"{num_cm:.6f}".rstrip('0').rstrip('.'))
            i = m.end()
            continue

        if ch in ('"', "'"):
            in_str, quote = True, ch

        out.append(ch)
        i += 1

    return ''.join(out)


def _ast_value(node: ast.AST) -> Any:
    """
    Evaluate a restricted AST node into a Python value.

    Supported forms include:
    - Constants (int, float, str, bool, None)
    - Unary +/- on numeric constants
    - dict, list, tuple literals
    - simple names: True / False / None
    - string concatenation with '+'
    """
    if isinstance(node, ast.Constant):
        return node.value

    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, (ast.UAdd, ast.USub))
        and isinstance(node.operand, ast.Constant)
    ):
        v = node.operand.value
        if isinstance(v, (int, float)):
            return +v if isinstance(node.op, ast.UAdd) else -v

    if isinstance(node, ast.Dict):
        return {
            _ast_value(k): _ast_value(v)
            for k, v in zip(node.keys, node.values)
        }

    if isinstance(node, ast.List):
        return [_ast_value(e) for e in node.elts]

    if isinstance(node, ast.Tuple):
        return tuple(_ast_value(e) for e in node.elts)

    if isinstance(node, ast.Name) and node.id in ("True", "False", "None"):
        return {"True": True, "False": False, "None": None}[node.id]

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _ast_value(node.left)
        right = _ast_value(node.right)
        if isinstance(left, str) and isinstance(right, str):
            return left + right

    # Common failure: the model copies the spec enumeration like
    # direction="forward"|"left"|"right"|"backward" instead of choosing one
    # value. `|` parses as bitwise-OR -> BinOp(BitOr). Return an actionable
    # message rather than dumping the raw AST into the agent's next prompt.
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        raise ValueError(
            'argument is an enumeration of options (a|b|c); choose exactly ONE '
            'value (e.g. direction="forward"), not the whole option list.'
        )

    raise ValueError(
        "unsupported argument expression; pass a single literal value "
        "(a string, number, or list)."
    )


def _parse_call(text: str) -> Tuple[str, List[Any], Dict[str, Any]]:
    """
    Parse a single-line function-call-like string into (name, args, kwargs).

    This expects a pattern like: "ACTION_NAME(arg1, arg2, key=value)" and returns:
        - canonicalized action name
        - list of positional arguments
        - dict of keyword arguments

    Args:
        text (str):
            A function-call-like string such as:
            "MOVE(102.3m, 885.5m, pace='accel')"

    Returns:
        Tuple[str, List[Any], Dict[str, Any]]:
            name (str)       : canonical action name, e.g. "MOVE"
            pos  (List[Any]) : positional arguments,
                               e.g. [10230.0, 88550.0] (meters converted to centimeters)
            kw   (Dict[str, Any]): keyword arguments,
                               e.g. {"pace": "accel"}

    Raises:
        ValueError:
            If the input is not a valid function-call-like string.

    Examples:
        >>> _parse_call("MOVE(102.3m, 885.5m)")
        ("MOVE", [10230.0, 88550.0], {})

        >>> _parse_call("ACCEPT_ORDER(12)")
        ("ACCEPT_ORDER", [12], {})

        >>> _parse_call("MOVE(100, 200, pace='accel')")
        ("MOVE", [100.0, 200.0], {"pace": "accel"})

        >>> _parse_call("BUY(items=[{'item_id':'energy_drink','qty':2}])")
        ("BUY", [], {"items": [{"item_id": "energy_drink", "qty": 2}]})
    """
    # Match function-call form: ACTION_NAME(...)
    m = _CALL_RE.match(text)
    if not m:
        raise ValueError("Output is not a function call like NAME(...).")

    # Canonicalize action name
    raw_name = m.group(1).strip()
    name = _CANON.get(raw_name.upper(), raw_name.upper())

    # Extract argument source string
    args_src = m.group(2).strip()

    # No arguments
    if args_src == "":
        return name, [], {}

    # Convert numeric 'Xm' units to centimeters
    args_src = _convert_m_to_cm_in_args(args_src)

    # Wrap into a fake call for AST parsing
    fake = f"__F__({args_src})"
    tree = ast.parse(fake, mode="eval")
    if not isinstance(tree.body, ast.Call):
        raise ValueError("Not a call.")

    call = tree.body

    # Positional arguments
    pos = [_ast_value(a) for a in call.args]

    # Keyword arguments
    kw = {
        kw.arg: _ast_value(kw.value)
        for kw in call.keywords
        if kw.arg is not None
    }

    return name, pos, kw


def _orders_at_pickup(dm, tol_cm: float = 300.0) -> List[Any]:
    """
    Collect active orders whose pickup node is within tol_cm of the agent.

    Args:
        dm:
            DeliveryMan-like instance providing active_orders and _is_at_xy.
        tol_cm (float):
            Distance threshold in centimeters for "at pickup door".

    Returns:
        List[Any]: orders that are not yet picked up and whose pickup node
        is close enough to the agent.
    """
    res = []
    for o in getattr(dm, "active_orders", []) or []:
        if getattr(o, "has_picked_up", False):
            continue
        node = getattr(o, "pickup_node", None)
        if node is None:
            continue
        try:
            px, py = float(node.position.x), float(node.position.y)
        except Exception:
            continue
        if dm._is_at_xy(px, py, tol_cm=tol_cm):
            res.append(o)
    return res


def _lookup_order(dm, oid: int) -> Any:
    """
    Find an order object by id, searching multiple sources:

    Search order-manager and agent state in the following order:
    1) dm._order_manager.get(oid) if available.
    2) dm._order_manager._orders pool (fallback scan).
    3) dm.active_orders.
    4) dm.help_orders (helper-attached orders).

    Args:
        dm:
            DeliveryMan-like instance.
        oid (int):
            Order ID.

    Returns:
        Any: the matched order object if found, otherwise None.
    """
    om = getattr(dm, "_order_manager", None)
    oid = int(oid)

    # Primary path: strongly prefer OM.get(int)
    get = getattr(om, "get", None) if om is not None else None
    if callable(get):
        order = get(oid)
        if order is not None:
            return order

    # Fallback: scan OM internals if present
    if om is not None:
        if hasattr(om, "get"):
            o = getattr(om, "get")(oid)  # type: ignore
            if o is not None:
                return o

        pool = list(getattr(om, "_orders", []) or [])
        for o in pool:
            if getattr(o, "id", None) == oid:
                return o

    # Fallback: check active orders bound to the agent
    for o in getattr(dm, "active_orders", []) or []:
        if getattr(o, "id", None) == oid:
            return o

    # Also look up attached helper orders
    o = (getattr(dm, "help_orders", {}) or {}).get(int(oid))
    if o is not None:
        return o

    return None

def parse_action(model_text: str, dm: Any):
    """
    Parse one line of VLM output into a DMAction.

    The function:
    1. Normalizes the raw model text via `sanitize_model_text`, which returns:
       - `text`: a string that should look like NAME(...)
       - `language_plan`: optional natural-language plan (may be None)
    2. Parses the call via `_parse_call` into (name, pos_args, kw_args).
    3. Maps the parsed name/arguments into a concrete DMAction.

    On success:
        returns (DMAction, language_plan)
    On failure:
        raises ValueError (the caller is responsible for logging/reporting).
    """
    from ..entities.delivery_man import DMAction, DMActionKind  # lazy import to avoid cycles

    text, language_plan = sanitize_model_text(model_text)
    name, pos, kw = _parse_call(text)
    _cfg = getattr(dm, 'cfg', None) or {}
    _ea = effective_enabled_actions(_cfg)
    if _ea is not None and name not in _ea:
        raise ValueError(f"Action '{name}' is disabled in current stage config.")

    # -------------------------------------------------------------------------
    # MOVE — direction-based step relative to the agent's facing.
    #   MOVE(direction="forward"|"left"|"right"|"backward")
    #   MOVE("forward")
    # -------------------------------------------------------------------------
    if name == "MOVE":
        direction = None
        if len(pos) >= 1 and isinstance(pos[0], str):
            direction = pos[0]
        if "direction" in kw and isinstance(kw["direction"], str):
            direction = kw["direction"]
        elif "dir" in kw and isinstance(kw["dir"], str):
            direction = kw["dir"]

        norm = (direction or "").strip().lower()
        _aliases = {
            "forward": "forward", "front": "forward", "ahead": "forward", "f": "forward",
            "backward": "backward", "back": "backward", "behind": "backward",
            "reverse": "backward", "b": "backward",
            "left": "left", "l": "left",
            "right": "right", "r": "right",
        }
        norm = _aliases.get(norm)
        if norm is None:
            raise ValueError(
                'MOVE needs direction="forward"|"left"|"right"|"backward", '
                'e.g. MOVE(direction="forward").'
            )
        return DMAction(DMActionKind.MOVE, data={"direction": norm}), language_plan

    # -------------------------------------------------------------------------
    # MOVE_TO — marked-waypoint step (enable_waypoint_marks only).
    #   MOVE_TO(3)            # numbered FPV / candidate-list mark
    #   MOVE_TO("dock_94")    # waypoint id from the candidate list
    # Membership validation (is k one of this step's marks?) happens in the
    # handler, which re-enumerates the candidates from the live graph.
    # -------------------------------------------------------------------------
    if name == "MOVE_TO":
        if not _cfg.get("enable_waypoint_marks", False):
            raise ValueError(
                "MOVE_TO is not available in this configuration; use "
                'MOVE(direction=...) instead.'
            )
        target: Any = pos[0] if len(pos) >= 1 else None
        for _key in ("mark", "k", "index", "target", "waypoint", "id"):
            if _key in kw:
                target = kw[_key]
                break
        if isinstance(target, float) and float(target).is_integer():
            target = int(target)
        if isinstance(target, bool) or not isinstance(target, (int, str)):
            raise ValueError(
                "MOVE_TO needs a mark number or waypoint id, e.g. MOVE_TO(2) "
                'or MOVE_TO("dock_94").'
            )
        return DMAction(DMActionKind.MOVE_TO, data={"target": target}), language_plan

    # -------------------------------------------------------------------------
    # PASSBY — forward step at 1.5x time/energy cost
    # -------------------------------------------------------------------------
    if name == "PASSBY":
        return DMAction(DMActionKind.PASSBY, data={}), language_plan

    # -------------------------------------------------------------------------
    # VIEW_ORDERS
    # -------------------------------------------------------------------------
    if name == "VIEW_ORDERS":
        return DMAction(DMActionKind.VIEW_ORDERS, data={}), language_plan

    # -------------------------------------------------------------------------
    # ACCEPT_ORDER
    #   Supports:
    #       ACCEPT_ORDER(12)
    #       ACCEPT_ORDER([12, 18])
    #       ACCEPT_ORDER(ids=[...])
    #       ACCEPT_ORDER(order_id=12)
    # -------------------------------------------------------------------------
    if name == "ACCEPT_ORDER":
        oids: List[int] = []
        oid: Optional[int] = None

        if len(pos) >= 1:
            arg0 = pos[0]
            if isinstance(arg0, (list, tuple)):
                oids = [int(v) for v in arg0]
            else:
                oid = int(arg0)
        elif "ids" in kw and isinstance(kw["ids"], (list, tuple)):
            oids = [int(v) for v in kw["ids"]]
        else:
            oid = int(kw.get("order_id") or kw.get("id"))

        if oids:
            return DMAction(DMActionKind.ACCEPT_ORDER, data=dict(oids=oids)), language_plan
        return DMAction(DMActionKind.ACCEPT_ORDER, data=dict(oid=int(oid))), language_plan

    # -------------------------------------------------------------------------
    # PICKUP
    #   If orders=[...] is provided, use that.
    #   Otherwise, collect all ready orders at the pickup door.
    # -------------------------------------------------------------------------
    if name == "PICKUP":
        oids: List[int] = []

        if len(pos) >= 1 and isinstance(pos[0], (list, tuple)):
            oids = [int(v) for v in pos[0]]
        elif "orders" in kw and isinstance(kw["orders"], (list, tuple)):
            oids = [int(v) for v in kw["orders"]]

        door_tol = _door_tol_cm(dm)
        orders: List[Any] = []
        if oids:
            for oid in oids:
                o = _lookup_order(dm, int(oid))
                if o is not None:
                    orders.append(o)
        else:
            orders = _orders_at_pickup(dm, tol_cm=door_tol)

        if not orders:
            raise ValueError(
                "PICKUP: no ready orders (MOVE to the pickup door or specify orders=[...])."
            )

        return DMAction(
            DMActionKind.PICKUP,
            data=dict(orders=orders, tol_cm=door_tol),
        ), language_plan

    # -------------------------------------------------------------------------
    # PLACE_FOOD_IN_BAG
    # -------------------------------------------------------------------------
    if name == "PLACE_FOOD_IN_BAG":
        bag_cmd = (
            pos[0]
            if (len(pos) >= 1 and isinstance(pos[0], str))
            else kw.get("bag_cmd") or kw.get("cmd") or ""
        ).strip()
        if not bag_cmd:
            raise ValueError("PLACE_FOOD_IN_BAG needs bag_cmd.")
        return DMAction(
            DMActionKind.PLACE_FOOD_IN_BAG,
            data=dict(bag_cmd=bag_cmd),
        ), language_plan

    # -------------------------------------------------------------------------
    # CHARGE
    # -------------------------------------------------------------------------
    if name == "CHARGE":
        target = float(pos[0]) if len(pos) >= 1 else float(kw.get("target_pct", 100.0))
        return DMAction(
            DMActionKind.CHARGE_ESCOOTER,
            data=dict(target_pct=target),
        ), language_plan

    # -------------------------------------------------------------------------
    # WAIT
    #   Supports explicit "until charge_done" or fixed minutes.
    # -------------------------------------------------------------------------
    if name == "WAIT":
        if (
            len(pos) >= 1
            and isinstance(pos[0], str)
            and pos[0].strip().lower() in ("charge_done", "done", "charge")
        ):
            return DMAction(
                DMActionKind.WAIT,
                data=dict(until="charge_done"),
            ), language_plan

        if "until" in kw and str(kw["until"]).lower() == "charge_done":
            return DMAction(
                DMActionKind.WAIT,
                data=dict(until="charge_done"),
            ), language_plan

        if "minutes" in kw:
            minutes = float(kw["minutes"])
        elif "min" in kw:
            minutes = float(kw["min"])
        elif "seconds" in kw:
            minutes = float(kw["seconds"]) / 60.0
        elif "duration_s" in kw:
            minutes = float(kw["duration_s"]) / 60.0
        elif len(pos) >= 1:
            minutes = float(pos[0])
        else:
            minutes = 0.0

        return DMAction(
            DMActionKind.WAIT,
            data=dict(duration_s=minutes * 60.0),
        ), language_plan

    # -------------------------------------------------------------------------
    # REST
    # -------------------------------------------------------------------------
    if name == "REST":
        tgt = float(pos[0]) if len(pos) >= 1 else float(kw.get("target_pct", 100.0))
        return DMAction(
            DMActionKind.REST,
            data=dict(target_pct=tgt),
        ), language_plan

    # -------------------------------------------------------------------------
    # BUY
    #
    # Two main forms:
    #   1) Single item:
    #       BUY(item_id="energy_drink", qty=2)
    #       BUY(name="...", qty=2)
    #       BUY(item="...", qty=2)
    #
    #   2) Multiple items:
    #       BUY(items=[{"item_id":"energy_drink","qty":2},
    #                  {"name":"escooter_battery_pack","qty":1}])
    # -------------------------------------------------------------------------
    if name == "BUY":
        data: Dict[str, Any] = {}

        # First, normalize items=[...]
        items_param = kw.get("items", None)
        if (
            len(pos) >= 1
            and isinstance(pos[0], (list, tuple))
            and all(isinstance(e, dict) for e in pos[0])
        ):
            items_param = pos[0]

        if items_param is not None:
            if not isinstance(items_param, (list, tuple)):
                raise ValueError("BUY(items=...) must be a list of dicts.")
            normalized: List[Dict[str, Any]] = []
            for e in items_param:
                if not isinstance(e, dict):
                    raise ValueError("Each element in BUY(items=...) must be a dict.")
                iid = e.get("item_id") or e.get("name") or e.get("item")
                try:
                    q = int(e.get("qty", 1))
                except Exception:
                    q = 0
                if iid and q > 0:
                    normalized.append({"item_id": str(iid), "qty": q})
            if not normalized:
                raise ValueError("BUY(items=...) has no valid entries.")
            data["items"] = normalized

        # Then, optionally add a single-item spec on top
        single_item = None
        single_qty = 1

        if len(pos) >= 1 and isinstance(pos[0], str):
            single_item = pos[0]
            if len(pos) >= 2:
                try:
                    single_qty = int(pos[1])
                except Exception:
                    single_qty = 1
        else:
            single_item = kw.get("item") or kw.get("item_id") or kw.get("name")
            try:
                single_qty = int(kw.get("qty", 1))
            except Exception:
                single_qty = 1

        if single_item:
            if single_qty <= 0:
                raise ValueError("BUY qty must be > 0.")
            if "items" in data:
                data["items"].append(
                    {"item_id": str(single_item), "qty": int(single_qty)}
                )
            else:
                data.update(
                    dict(item_id=str(single_item), qty=int(single_qty))
                )

        if not data:
            raise ValueError(
                "BUY needs item_id/name + qty, or items=[{item_id/name, qty}, ...]."
            )

        return DMAction(DMActionKind.BUY, data=data), language_plan

    # -------------------------------------------------------------------------
    # USE_BATTERY_PACK / USE_ENERGY_DRINK
    # -------------------------------------------------------------------------
    if name == "USE_BATTERY_PACK":
        return DMAction(
            DMActionKind.USE_BATTERY_PACK,
            data=dict(item_id="escooter_battery_pack"),
        ), language_plan

    if name == "USE_ENERGY_DRINK":
        return DMAction(
            DMActionKind.USE_ENERGY_DRINK,
            data=dict(item_id="energy_drink"),
        ), language_plan

    # -------------------------------------------------------------------------
    # SWITCH (transport mode)
    # -------------------------------------------------------------------------
    if name == "SWITCH":
        to = (str(pos[0]) if len(pos) >= 1 else str(kw.get("to") or "")).strip()
        if not to:
            raise ValueError("SWITCH needs target 'to'.")
        return DMAction(
            DMActionKind.SWITCH_TRANSPORT,
            data=dict(to=to),
        ), language_plan

    # -------------------------------------------------------------------------
    # RENT_CAR / RETURN_CAR
    # -------------------------------------------------------------------------
    if name == "RENT_CAR":
        return DMAction(DMActionKind.RENT_CAR, data={}), language_plan

    if name == "RETURN_CAR":
        return DMAction(DMActionKind.RETURN_CAR, data={}), language_plan

    # -------------------------------------------------------------------------
    # Multi-agent help workflow:
    #   VIEW_HELP_BOARD
    #   POST_HELP
    #   ACCEPT_HELP
    #   EDIT_HELP
    #   PLACE_TEMP_BOX
    #   TAKE_FROM_TEMP_BOX
    #   REPORT_HELP_FINISHED
    # -------------------------------------------------------------------------
    if name == "VIEW_HELP_BOARD":
        return DMAction(DMActionKind.VIEW_HELP_BOARD, data={}), language_plan

    if name == "POST_HELP":
        kind = str(kw.get("kind") or (pos[0] if len(pos) >= 1 else "")).strip().upper()
        if not kind:
            raise ValueError("POST_HELP needs kind.")

        bounty = float(kw.get("bounty", 0.0))
        ttl_s = float(kw.get("ttl_s", 0.0))
        payload = kw.get("payload", {})

        if len(pos) >= 2 and isinstance(pos[1], (int, float)):
            bounty = float(pos[1])
        if len(pos) >= 3 and isinstance(pos[2], (int, float)):
            ttl_s = float(pos[2])
        if len(pos) >= 4 and isinstance(pos[3], (dict, list, tuple, str)):
            payload = pos[3]
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except Exception:
                    payload = {"raw": payload}

        return DMAction(
            DMActionKind.POST_HELP_REQUEST,
            data=dict(
                help_type=kind,
                bounty=bounty,
                ttl_s=ttl_s,
                payload=payload,
            ),
        ), language_plan

    if name == "ACCEPT_HELP":
        req_id = int(pos[0]) if len(pos) >= 1 else int(kw.get("req_id"))
        return DMAction(
            DMActionKind.ACCEPT_HELP_REQUEST,
            data=dict(req_id=req_id),
        ), language_plan

    if name == "EDIT_HELP":
        req_id = int(kw.get("req_id", pos[0] if len(pos) >= 1 else -1))
        if req_id == -1:
            raise ValueError("EDIT_HELP needs req_id.")

        data: Dict[str, Any] = dict(req_id=req_id)

        if len(pos) >= 2:
            data["new_bounty"] = float(pos[1])
        if "new_bounty" in kw:
            data["new_bounty"] = float(kw["new_bounty"])

        ttl_min: Optional[float] = None
        if len(pos) >= 3:
            ttl_min = float(pos[2])
        if "new_ttl_min" in kw:
            ttl_min = float(kw["new_ttl_min"])
        if ttl_min is not None:
            data["new_ttl_s"] = float(ttl_min) * 60.0

        return DMAction(
            DMActionKind.EDIT_HELP_REQUEST,
            data=data,
        ), language_plan

    if name == "PLACE_TEMP_BOX":
        req_id = int(kw.get("req_id", pos[0] if len(pos) >= 1 else -1))
        if req_id == -1:
            raise ValueError("PLACE_TEMP_BOX needs req_id.")

        location = kw.get("location", None)
        if len(pos) >= 2 and isinstance(pos[1], (list, tuple)):
            location = tuple(pos[1])

        content = kw.get("content", {})
        if len(pos) >= 3 and isinstance(pos[2], (dict, str)):
            content = pos[2]
        if isinstance(content, str):
            content = json.loads(content)

        data = dict(
            req_id=req_id,
            location_xy=tuple(location) if location else None,
            content=content,
        )
        return DMAction(DMActionKind.PLACE_TEMP_BOX, data=data), language_plan

    if name == "TAKE_FROM_TEMP_BOX":
        req_id = int(kw.get("req_id", pos[0] if len(pos) >= 1 else -1))
        if req_id == -1:
            raise ValueError("TAKE_FROM_TEMP_BOX needs req_id.")
        return DMAction(
            DMActionKind.TAKE_FROM_TEMP_BOX,
            data=dict(req_id=req_id),
        ), language_plan

    if name == "REPORT_HELP_FINISHED":
        req_id = int(kw.get("req_id", pos[0] if len(pos) >= 1 else -1))
        if req_id == -1:
            raise ValueError("REPORT_HELP_FINISHED needs req_id.")
        return DMAction(
            DMActionKind.REPORT_HELP_FINISHED,
            data=dict(req_id=req_id),
        ), language_plan

    # -------------------------------------------------------------------------
    # DROP_OFF
    #   DROP_OFF(12, "leave_at_door")
    #   DROP_OFF(oid=12, method="knock")
    # -------------------------------------------------------------------------
    if name == "DROP_OFF":
        if len(pos) >= 1 and isinstance(pos[0], (int, float)):
            oid = int(pos[0])
        else:
            oid = int(kw.get("oid"))

        method = None
        if len(pos) >= 2 and isinstance(pos[1], str):
            method = pos[1]
        else:
            method = kw.get("method")
        if method is None:
            _cfg = getattr(dm, 'cfg', None) or {}
            if not _cfg.get("enable_delivery_methods", True):
                method = "leave_at_door"  # auto-default for Stage 1
            else:
                raise ValueError("DROP_OFF needs 'method'.")

        method_norm = (
            str(method)
            .strip()
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )
        if method_norm not in {
            "leave_at_door",
            "knock",
            "call",
            "hand_to_customer",
        }:
            raise ValueError(
                "DROP_OFF.method must be one of "
                "leave_at_door|knock|call|hand_to_customer"
            )

        return DMAction(
            DMActionKind.DROP_OFF,
            data={"oid": int(oid), "method": method_norm, "tol_cm": _door_tol_cm(dm)},
        ), language_plan

    # -------------------------------------------------------------------------
    # VIEW_BAG
    # -------------------------------------------------------------------------
    if name == "VIEW_BAG":
        return DMAction(DMActionKind.VIEW_BAG, data={}), language_plan

    # -------------------------------------------------------------------------
    # USE_ICE_PACK / USE_HEAT_PACK
    #   USE_ICE_PACK("A") / USE_ICE_PACK(comp="A")
    #   USE_HEAT_PACK("A") / USE_HEAT_PACK(comp="A")
    # -------------------------------------------------------------------------
    if name == "USE_ICE_PACK":
        raw = (pos[0] if len(pos) >= 1 else kw.get("comp"))
        s = (str(raw).strip().upper() if raw is not None else "")
        if not s or not ("A" <= s[0] <= "Z"):
            raise ValueError("USE_ICE_PACK needs comp like 'A'/'B'/...")
        return DMAction(
            DMActionKind.USE_ICE_PACK,
            data=dict(comp=s[0]),
        ), language_plan

    if name == "USE_HEAT_PACK":
        raw = (pos[0] if len(pos) >= 1 else kw.get("comp"))
        s = (str(raw).strip().upper() if raw is not None else "")
        if not s or not ("A" <= s[0] <= "Z"):
            raise ValueError("USE_HEAT_PACK needs comp like 'A'/'B'/...")
        return DMAction(
            DMActionKind.USE_HEAT_PACK,
            data=dict(comp=s[0]),
        ), language_plan

    # -------------------------------------------------------------------------
    # SAY
    #
    # Supports:
    #   SAY("hello")                        -> broadcast
    #   SAY(text="hello")                   -> broadcast
    #   SAY("hello", to="7")                -> direct message
    #   SAY(to="7", text="hello")           -> direct message
    #   SAY(text="hi", to="ALL"|"*")        -> broadcast
    # -------------------------------------------------------------------------
    if name == "SAY":
        if len(pos) >= 1 and isinstance(pos[0], str):
            text = pos[0]
        else:
            text = kw.get("text") or ""

        to = kw.get("to", None)

        # Normalize numeric IDs: SAY(to=7, ...) -> "7"
        if isinstance(to, (int, float)):
            to = str(int(to))

        if to is not None:
            t = str(to).strip()
            # Normalize broadcast markers
            if t == "" or t.upper() in ("ALL", "*"):
                to = None
            else:
                to = t

        text = str(text).strip()
        if not text:
            raise ValueError(
                'SAY needs non-empty "text" '
                '(e.g., SAY("hello") or SAY(text="hello")).'
            )

        return DMAction(
            DMActionKind.SAY,
            data={"text": text, "to": to},
        ), language_plan

    # -------------------------------------------------------------------------
    # Bus-related actions
    # -------------------------------------------------------------------------
    if name == "BOARD_BUS":
        bus_id = kw.get("bus_id")
        target_stop_id = kw.get("target_stop_id")
        return DMAction(
            DMActionKind.BOARD_BUS,
            data={"bus_id": bus_id, "target_stop_id": target_stop_id},
        ), language_plan

    if name == "VIEW_BUS_SCHEDULE":
        return DMAction(DMActionKind.VIEW_BUS_SCHEDULE, data={}), language_plan

    # -------------------------------------------------------------------------
    # NAVIGATE — unified query-only navigation tool (route map + cost estimate).
    #   NAVIGATE(target="restaurant 1")                      # walk route, map image
    #   NAVIGATE("restaurant 1", "e-scooter")                # positional mode
    #   NAVIGATE(target="...", mode="bus", access_mode="scooter", egress_mode="walk")
    # mode is "walk" (default) | "e-scooter" | "bus".
    # -------------------------------------------------------------------------
    if name == "NAVIGATE":
        target_token: Optional[str] = None
        if len(pos) >= 1 and isinstance(pos[0], str):
            target_token = pos[0]
        if "target" in kw and isinstance(kw["target"], str):
            target_token = kw["target"]
        elif "to" in kw and isinstance(kw["to"], str):
            target_token = kw["to"]

        if not target_token or not isinstance(target_token, str):
            raise ValueError(
                'NAVIGATE needs a target, e.g. '
                'NAVIGATE(target="restaurant 1", mode="walk").'
            )

        data: Dict[str, Any] = {"target": target_token.strip()}
        mode = kw.get("mode")
        if mode is None and len(pos) >= 2 and isinstance(pos[1], str):
            mode = pos[1]
        if isinstance(mode, str):
            data["mode"] = mode.strip().lower()
        am = kw.get("access_mode")
        em = kw.get("egress_mode")
        if isinstance(am, str):
            data["access_mode"] = am.strip().lower()
        if isinstance(em, str):
            data["egress_mode"] = em.strip().lower()
        return DMAction(DMActionKind.NAVIGATE, data=data), language_plan

    # -------------------------------------------------------------------------
    # Fallback
    # -------------------------------------------------------------------------
    raise ValueError(f"Unrecognized or malformed action: {str(model_text)[:120]}")


# =========================================================
# 6) Human-readable summaries for DMAction
#    - _fmt_xy_cm / _order_brief: small formatting helpers.
#    - action_to_text: render a DMAction into a concise English description.
# =========================================================
def _fmt_xy_cm(xy: Optional[Tuple[float, float]]) -> str:
    """Format (x_cm, y_cm) as '(x m, y m)'."""
    if not xy:
        return "(N/A, N/A)"
    x, y = float(xy[0]), float(xy[1])
    return f"({x/100.0:.2f} m, {y/100.0:.2f} m)"


def _order_brief(o: Any) -> str:
    """Return a short textual description of an order."""
    oid = getattr(o, "id", None)
    pu = getattr(o, "pickup_road_name", None) or ""
    do = getattr(o, "dropoff_road_name", None) or ""
    core = f"#{oid}" if oid is not None else "(unknown)"
    if pu or do:
        return f"order {core} ({pu} → {do})"
    return f"order {core}"


def action_to_text(action: Any, dm: Optional[Any] = None) -> str:
    """
    Render a DMAction into a concise English description.

    The function is defensive and should not raise; if the action kind
    is unknown, it returns a generic message.
    """
    kind = getattr(action, "kind", None)
    data: Dict[str, Any] = getattr(action, "data", {}) or {}

    # Helper: format a list of orders as "#id1, #id2, ..."
    def _order_ids_text(orders: List[Any]) -> str:
        ids = [
            getattr(o, "id", None)
            for o in orders
            if getattr(o, "id", None) is not None
        ]
        return ", ".join(f"#{i}" for i in ids) if ids else "(none)"

    # -------------------------------------------------------------------------
    # MOVE (direction-based)
    # -------------------------------------------------------------------------
    if str(kind).endswith("MOVE"):
        return f"Move {data.get('direction', '?')}."

    # -------------------------------------------------------------------------
    # MOVE_TO (marked-waypoint step)
    # -------------------------------------------------------------------------
    if str(kind).endswith("MOVE_TO"):
        t = data.get("target")
        if isinstance(t, str) and not t.strip().isdigit():
            return f"Move to waypoint {t}."
        return f"Move to waypoint mark {t}."

    # -------------------------------------------------------------------------
    # PASSBY
    # -------------------------------------------------------------------------
    if str(kind).endswith("PASSBY"):
        return "Pass by (move forward at 1.5x time/energy cost)."

    # -------------------------------------------------------------------------
    # VIEW_ORDERS
    # -------------------------------------------------------------------------
    if str(kind).endswith("VIEW_ORDERS"):
        return "View available orders."

    # -------------------------------------------------------------------------
    # ACCEPT_ORDER
    # -------------------------------------------------------------------------
    if str(kind).endswith("ACCEPT_ORDER"):
        data = data or {}
        if "oids" in data and data["oids"]:
            ids = ", ".join(f"#{int(i)}" for i in data["oids"])
            return f"Accept orders {ids}."
        if "oid" in data:
            return f"Accept order #{int(data['oid'])}."
        o = data.get("order")  # legacy support
        if o is not None:
            return f"Accept {_order_brief(o)}."
        return "Accept the order."

    # -------------------------------------------------------------------------
    # PICKUP
    # -------------------------------------------------------------------------
    if str(kind).endswith("PICKUP"):
        orders = list(data.get("orders") or [])
        if orders:
            place = getattr(orders[0], "pickup_road_name", None)
            suffix = f" at {place}" if place else ""
            return f"Pick up orders {_order_ids_text(orders)}{suffix}."
        return "Pick up ready orders."

    # -------------------------------------------------------------------------
    # PLACE_FOOD_IN_BAG
    # -------------------------------------------------------------------------
    if str(kind).endswith("PLACE_FOOD_IN_BAG"):
        cmd = (data.get("bag_cmd") or "").strip()
        return (
            f'Place items into the insulated bag using: "{cmd}".'
            if cmd
            else "Place items into the insulated bag."
        )

    # -------------------------------------------------------------------------
    # CHARGE_ESCOOTER
    # -------------------------------------------------------------------------
    if str(kind).endswith("CHARGE_ESCOOTER"):
        tgt = data.get("target_pct")
        return (
            f"Charge the e-scooter to {float(tgt):.0f}%."
            if tgt is not None
            else "Charge the e-scooter."
        )

    # -------------------------------------------------------------------------
    # WAIT
    # -------------------------------------------------------------------------
    if str(kind).endswith("WAIT"):
        if data.get("until") == "charge_done":
            return "Wait until charging completes."
        secs = data.get("duration_s")
        if secs:
            mins = float(secs) / 60.0
            return f"Wait for {mins:.1f} min."
        return "Wait."

    # -------------------------------------------------------------------------
    # REST
    # -------------------------------------------------------------------------
    if str(kind).endswith("REST"):
        tgt = data.get("target_pct")
        return (
            f"Rest until energy reaches {float(tgt):.0f}%."
            if tgt is not None
            else "Rest."
        )

    # -------------------------------------------------------------------------
    # BUY
    # -------------------------------------------------------------------------
    if str(kind).endswith("BUY"):
        if "items" in data and data["items"]:
            parts = []
            for it in data["items"]:
                iid = it.get("item_id") or it.get("name") or "item"
                q = int(it.get("qty", 1))
                parts.append(f"{q} × {iid}")
            return "Buy " + ", ".join(parts) + "."
        item = data.get("item_id") or data.get("name") or "item"
        qty = int(data.get("qty", 1))
        return f"Buy {qty} × {item}."

    # -------------------------------------------------------------------------
    # USE_BATTERY_PACK / USE_ENERGY_DRINK
    # -------------------------------------------------------------------------
    if str(kind).endswith("USE_BATTERY_PACK"):
        return "Use a scooter battery pack (recharge to 100%)."

    if str(kind).endswith("USE_ENERGY_DRINK"):
        return "Use an energy drink (+50% energy)."

    # -------------------------------------------------------------------------
    # SWITCH_TRANSPORT
    # -------------------------------------------------------------------------
    if str(kind).endswith("SWITCH_TRANSPORT"):
        to = str(data.get("to") or "").strip() or "target mode"
        return f"Switch to {to}."

    # -------------------------------------------------------------------------
    # RENT_CAR / RETURN_CAR
    # -------------------------------------------------------------------------
    if str(kind).endswith("RENT_CAR"):
        return "Rent a car."

    if str(kind).endswith("RETURN_CAR"):
        return "Return the rental car."

    # -------------------------------------------------------------------------
    # VIEW_HELP_BOARD
    # -------------------------------------------------------------------------
    if str(kind).endswith("VIEW_HELP_BOARD"):
        return "View help board."

    # -------------------------------------------------------------------------
    # POST_HELP_REQUEST
    # -------------------------------------------------------------------------
    if str(kind).endswith("POST_HELP_REQUEST"):
        kind_s = str(data.get("help_type") or "").upper()
        bounty = float(data.get("bounty", 0.0))
        ttl_s = float(data.get("ttl_s", 0.0))
        payload = dict(data.get("payload") or {})
        mins = int(round(ttl_s / 60.0)) if ttl_s else 0

        def _fmt_payload():
            if kind_s == "HELP_DELIVERY":
                oid = payload.get("order_id")
                provide_xy = payload.get("provide_xy")
                brief = ""
                if dm is not None and oid is not None:
                    o = _lookup_order(dm, int(oid))
                    if o is not None:
                        brief = f" {_order_brief(o)}"
                return (
                    f"handoff at {_fmt_xy_cm(provide_xy)} "
                    f"for order #{oid}{brief}"
                )
            if kind_s == "HELP_BUY":
                inv = payload.get("buy_list") or {}
                if isinstance(inv, list):
                    pairs = inv
                else:
                    pairs = list(inv.items())
                parts = [f"{k}×{v}" for (k, v) in pairs]
                deliver_xy = payload.get("deliver_xy")
                return (
                    f"shopping list [{', '.join(parts)}], "
                    f"deliver to {_fmt_xy_cm(deliver_xy)}"
                )
            if kind_s == "HELP_CHARGE":
                provide_xy = payload.get("provide_xy")
                deliver_xy = payload.get("deliver_xy")
                target = payload.get(
                    "want_charge_pct",
                    payload.get("target_pct", 100.0),
                )
                return (
                    f"pick at {_fmt_xy_cm(provide_xy)}, "
                    f"charge to {float(target):.0f}%, "
                    f"deliver to {_fmt_xy_cm(deliver_xy)}"
                )
            return "details provided"

        return (
            f"Post help request ({kind_s}), bounty ${bounty:.2f}, "
            f"TTL {mins} min: {_fmt_payload()}."
        )

    # -------------------------------------------------------------------------
    # ACCEPT_HELP_REQUEST
    # -------------------------------------------------------------------------
    if str(kind).endswith("ACCEPT_HELP_REQUEST"):
        req_id = data.get("req_id")
        desc = f"Accept help request #{req_id}"
        try:
            from .comms import get_comms

            comms = get_comms()
            if comms and req_id is not None:
                det = comms.get_request_detail(req_id=int(req_id))
                if det:
                    who = det.get("publisher_id")
                    k = det.get("kind")
                    if who or k:
                        extra = []
                        if k:
                            extra.append(str(k))
                        if who:
                            extra.append(f"from {who}")
                        desc += f" ({', '.join(extra)})"
        except Exception:
            pass
        return desc + "."

    # -------------------------------------------------------------------------
    # EDIT_HELP_REQUEST
    # -------------------------------------------------------------------------
    if str(kind).endswith("EDIT_HELP_REQUEST"):
        req_id = data.get("req_id")
        parts = []
        if "new_bounty" in data:
            parts.append(f"bounty -> ${float(data['new_bounty']):.2f}")
        if "new_ttl_s" in data:
            parts.append(
                f"TTL -> {int(round(float(data['new_ttl_s'])/60.0))} min"
            )
        tail = f" ({', '.join(parts)})" if parts else ""
        return f"Edit help request #{req_id}{tail}."

    # -------------------------------------------------------------------------
    # PLACE_TEMP_BOX
    # -------------------------------------------------------------------------
    if str(kind).endswith("PLACE_TEMP_BOX"):
        req_id = data.get("req_id")
        loc = data.get("location_xy")
        content = dict(data.get("content") or {})

        parts = []
        inv = content.get("inventory") or {}
        if inv:
            inv_s = ", ".join(f"{k}×{v}" for k, v in inv.items())
            parts.append(f"inventory[{inv_s}]")
        if "food" in content:
            parts.append("all food items")
        if "escooter" in content:
            parts.append("e-scooter")

        stuff = "; ".join(parts) if parts else "items"
        loc_s = _fmt_xy_cm(tuple(loc)) if loc else "current location"
        return (
            f"Place a temp box for request #{req_id} "
            f"at {loc_s} containing {stuff}."
        )

    # -------------------------------------------------------------------------
    # TAKE_FROM_TEMP_BOX
    # -------------------------------------------------------------------------
    if str(kind).endswith("TAKE_FROM_TEMP_BOX"):
        req_id = data.get("req_id")
        return f"Take items from the temp box for request #{req_id}."

    # -------------------------------------------------------------------------
    # REPORT_HELP_FINISHED
    # -------------------------------------------------------------------------
    if str(kind).endswith("REPORT_HELP_FINISHED"):
        req_id = data.get("req_id")
        return f"Report help finished for request #{req_id}."

    # -------------------------------------------------------------------------
    # DROP_OFF
    # -------------------------------------------------------------------------
    if str(kind).endswith("DROP_OFF"):
        oid = data.get("oid")
        method = data.get("method")
        return f"Drop off order #{oid} via {method}."

    # -------------------------------------------------------------------------
    # VIEW_BAG
    # -------------------------------------------------------------------------
    if str(kind).endswith("VIEW_BAG"):
        return "View insulated bag layout."

    # -------------------------------------------------------------------------
    # USE_ICE_PACK / USE_HEAT_PACK
    # -------------------------------------------------------------------------
    if str(kind).endswith("USE_ICE_PACK"):
        c = (data.get("comp") or "?")
        return f"Use an ice pack on compartment {c}."

    if str(kind).endswith("USE_HEAT_PACK"):
        c = (data.get("comp") or "?")
        return f"Use a heat pack on compartment {c}."

    # -------------------------------------------------------------------------
    # SAY
    # -------------------------------------------------------------------------
    if str(kind).endswith("SAY"):
        txt = str(data.get("text") or "")
        to = data.get("to", None)
        # Optionally truncate long messages for logging
        show = txt if len(txt) <= 80 else (txt[:77] + "...")
        if to:
            return f'Send chat to {to}: "{show}"'
        return f'Send broadcast chat: "{show}"'

    # -------------------------------------------------------------------------
    # Bus-related actions
    # -------------------------------------------------------------------------
    if str(kind).endswith("BOARD_BUS"):
        bus_id = data.get("bus_id")
        target_stop_id = data.get("target_stop_id")
        return f"Board bus {bus_id} to {target_stop_id}."

    if str(kind).endswith("VIEW_BUS_SCHEDULE"):
        return "View bus schedule."

    # Unified NAVIGATE tool (mode parameter).
    if str(kind).endswith("NAVIGATE"):
        tgt = data.get("target") or "?"
        mode = data.get("mode") or "walk"
        return f"Show the {mode} route to {tgt} on the map."

    # -------------------------------------------------------------------------
    # Fallback
    # -------------------------------------------------------------------------
    return "Execute action."
