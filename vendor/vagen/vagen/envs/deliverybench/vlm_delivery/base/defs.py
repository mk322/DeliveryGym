# base/defs.py (shared action / transport types for DeliveryMan)
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Callable, TYPE_CHECKING

# Only imported for type checking; this does NOT run at runtime,
# so it will not create a circular import.
if TYPE_CHECKING:
    from ..entities.delivery_man import DeliveryMan


# ===== Transport Modes =====
class TransportMode(str, Enum):
    """
    High-level transport mode used by the delivery agent.

    Values are serialized as strings and also used in prompts / logs.
    """
    WALK          = "walk"
    SCOOTER       = "e-scooter"
    DRAG_SCOOTER  = "drag_scooter"
    CAR           = "car"
    BUS           = "bus"


# Common item IDs used in the environment / inventory system.
ITEM_ESC_BATTERY_PACK = "escooter_battery_pack"
ITEM_ENERGY_DRINK     = "energy_drink"
ITEM_ICE_PACK         = "ice_pack"
ITEM_HEAT_PACK        = "heat_pack"


class DeliveryMethod(str, Enum):
    """
    Delivery method used when completing a drop-off.

    These values are also exposed to the model (e.g., in DROP_OFF.method).
    """
    LEAVE_AT_DOOR     = "leave_at_door"     # Leave the order at the door
    KNOCK             = "knock"             # Knock on the door
    CALL              = "call"              # Call the customer
    HAND_TO_CUSTOMER  = "hand_to_customer"  # Hand directly to the customer


# String set of valid delivery methods for quick validation.
VALID_DELIVERY_METHODS = {
    DeliveryMethod.LEAVE_AT_DOOR.value,
    DeliveryMethod.KNOCK.value,
    DeliveryMethod.CALL.value,
    DeliveryMethod.HAND_TO_CUSTOMER.value,
}


# ===== Actions =====
class DMActionKind(str, Enum):
    """
    Canonical action kinds that the DeliveryMan can execute.

    These enums define the contract between:
      - the planner / VLM outputs,
      - the environment scheduler,
      - and the DeliveryMan implementation.
    """
    MOVE                 = "move"   # direction-based step (forward|left|right|backward) on the waypoint graph
    MOVE_TO              = "move_to"  # marked-waypoint step MOVE_TO(k)/MOVE_TO("id") (enable_waypoint_marks)
    ACCEPT_ORDER         = "accept_order"
    VIEW_ORDERS          = "view_orders"
    VIEW_BAG             = "view_bag"
    PICKUP               = "pickup"
    PLACE_FOOD_IN_BAG    = "place_food_in_bag"
    CHARGE_ESCOOTER      = "charge_escooter"
    WAIT                 = "wait"
    REST                 = "rest"
    BUY                  = "buy"
    USE_BATTERY_PACK     = "use_battery_pack"
    USE_ENERGY_DRINK     = "use_energy_drink"
    USE_ICE_PACK         = "use_ice_pack"
    USE_HEAT_PACK        = "use_heat_pack"
    VIEW_HELP_BOARD      = "view_help_board"
    POST_HELP_REQUEST    = "post_help_request"
    ACCEPT_HELP_REQUEST  = "accept_help_request"
    EDIT_HELP_REQUEST    = "edit_help_request"
    SWITCH_TRANSPORT     = "switch_transport"
    RENT_CAR             = "rent_car"
    RETURN_CAR           = "return_car"
    PLACE_TEMP_BOX       = "place_temp_box"       # publisher/helper drops a temp box
    TAKE_FROM_TEMP_BOX   = "take_from_temp_box"   # helper picks up from temp box
    REPORT_HELP_FINISHED = "report_help_finished" # helper reports completion
    DROP_OFF             = "drop_off"
    SAY                  = "say"
    BOARD_BUS            = "board_bus"
    VIEW_BUS_SCHEDULE    = "view_bus_schedule"
    NAVIGATE             = "navigate"  # unified query-only nav tool (mode=walk|e-scooter|bus)
    PASSBY               = "passby"   # move forward at 1.5x time/energy cost


# Single source of truth for query-only tool action kinds.
# Used to populate info["is_tool"] in deliverybench_env.step().
TOOL_ACTION_KINDS: frozenset = frozenset({
    DMActionKind.NAVIGATE,
})


@dataclass
class DMAction:
    """
    One scheduled action for a DeliveryMan.

    Attributes:
        kind:
            The high-level action kind (MOVE_TO, PICKUP, ...).

        data:
            A free-form dictionary of action parameters, interpreted by
            the DeliveryMan implementation and environment. Examples:
              - MOVE_TO: {"tx": x_cm, "ty": y_cm, "use_route": True, ...}
              - BUY: {"item_id": "energy_drink", "qty": 2}
              - DROP_OFF: {"oid": 12, "method": "leave_at_door"}

        on_done:
            Optional callback invoked after the action completes
            (successfully or not), receiving the DeliveryMan instance.
            This is used for higher-level scheduling / logging hooks.
    """
    kind: DMActionKind
    data: Dict[str, Any] = field(default_factory=dict)
    on_done: Optional[Callable[["DeliveryMan"], None]] = None