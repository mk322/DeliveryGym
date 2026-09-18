# actions/post_help_request.py
# -*- coding: utf-8 -*-

from typing import Any, Dict, List
from ..base.defs import DMAction
from ..gameplay.comms import get_comms, HelpType


def handle_post_help_request(dm: Any, act: DMAction, _allow_interrupt: bool) -> None:
    """
    Posts a help request of a specific type, with validation of required fields.

    Constraints:
    - HELP_DELIVERY:
        * Requires payload.order_id.
        * Requires payload.provide_xy.
        * Must NOT provide deliver_xy.
    - HELP_PICKUP:
        * Requires payload.order_id.
        * Requires payload.deliver_xy.
        * payload.provide_xy is ignored (should not be passed to comms).
    - HELP_BUY:
        * Requires a non-empty payload.buy_list.
        * Requires payload.deliver_xy.
    - HELP_CHARGE:
        * Requires both payload.provide_xy and payload.deliver_xy.
        * payload.target_pct (or want_charge_pct) is optional (default 100).

    No coordinate defaults are applied; missing or invalid coordinates cause failure.
    """
    dm.vlm_clear_ephemeral()
    comms = get_comms()
    if not comms:
        dm.vlm_add_error("post_help_request failed: no comms")
        dm._finish_action(success=False)
        return

    help_type = act.data.get("help_type")
    if isinstance(help_type, str):
        help_type = HelpType[help_type]
    bounty = float(act.data.get("bounty", 0.0))
    ttl_s = float(act.data.get("ttl_s", 0.0))
    payload: Dict[str, Any] = dict(act.data.get("payload") or {})

    def _as_xy(xy):
        """Validate and convert a (x, y) coordinate tuple to float, or None if invalid."""
        if not xy or len(xy) != 2:
            return None
        x, y = xy
        return (float(x), float(y))

    kwargs: Dict[str, Any] = dict(
        publisher_id=str(dm.agent_id),
        kind=help_type,
        reward=bounty,
        time_limit_s=ttl_s,
    )

    if help_type == HelpType.HELP_DELIVERY:
        # Requires: order_id and provide_xy; deliver_xy is not allowed.
        if "order_id" not in payload:
            dm.vlm_add_error(
                "post_help_request failed: HELP_DELIVERY needs payload.order_id"
            )
            dm._finish_action(success=False)
            return
        if "provide_xy" not in payload or _as_xy(payload.get("provide_xy")) is None:
            dm.vlm_add_error(
                "post_help_request failed: HELP_DELIVERY needs payload.provide_xy"
            )
            dm._finish_action(success=False)
            return

        oid = int(payload["order_id"])
        order_obj = next(
            (
                o
                for o in dm.active_orders
                if int(getattr(o, "id", -1)) == oid
            ),
            None,
        )
        if order_obj is None:
            dm.vlm_add_error(
                "post_help_request failed: order_ref not found on publisher"
            )
            dm._finish_action(success=False)
            return

        kwargs["order_id"] = oid
        kwargs["provide_xy"] = _as_xy(payload["provide_xy"])
        kwargs["order_ref"] = order_obj  # pass the order instance into Comms

    elif help_type == HelpType.HELP_PICKUP:
        # Requires: order_id and deliver_xy; provide_xy is ignored (not passed to comms).
        if "order_id" not in payload:
            dm.vlm_add_error(
                "post_help_request failed: HELP_PICKUP needs payload.order_id"
            )
            dm._finish_action(success=False)
            return
        if "deliver_xy" not in payload or _as_xy(payload.get("deliver_xy")) is None:
            dm.vlm_add_error(
                "post_help_request failed: HELP_PICKUP needs payload.deliver_xy"
            )
            dm._finish_action(success=False)
            return

        oid = int(payload["order_id"])
        order_obj = next(
            (
                o
                for o in dm.active_orders
                if int(getattr(o, "id", -1)) == oid
            ),
            None,
        )
        if order_obj is None:
            dm.vlm_add_error(
                "post_help_request failed: order_ref not found on publisher"
            )
            dm._finish_action(success=False)
            return
        if getattr(order_obj, "has_picked_up", False):
            dm.vlm_add_error(
                "post_help_request failed: order already picked up; "
                "use HELP_DELIVERY instead"
            )
            dm._finish_action(success=False)
            return

        kwargs["order_id"] = oid
        kwargs["deliver_xy"] = _as_xy(payload["deliver_xy"])
        kwargs["order_ref"] = order_obj  # pass to Comms; do not include provide_xy

    elif help_type == HelpType.HELP_BUY:
        # Requires: non-empty buy_list and deliver_xy.
        raw = list(payload.get("buy_list") or [])
        buy_items: Dict[str, int] = {}
        for item_id, qty in raw:
            q = int(qty)
            if q > 0:
                buy_items[str(item_id)] = q
        if not buy_items:
            dm.vlm_add_error(
                "post_help_request failed: HELP_BUY needs non-empty payload.buy_list"
            )
            dm._finish_action(success=False)
            return
        if "deliver_xy" not in payload or _as_xy(payload.get("deliver_xy")) is None:
            dm.vlm_add_error(
                "post_help_request failed: HELP_BUY needs payload.deliver_xy"
            )
            dm._finish_action(success=False)
            return

        kwargs["buy_items"] = buy_items
        kwargs["deliver_xy"] = _as_xy(payload["deliver_xy"])

    elif help_type == HelpType.HELP_CHARGE:
        # Requires: provide_xy and deliver_xy; target_pct is optional.
        if "provide_xy" not in payload or _as_xy(payload.get("provide_xy")) is None:
            dm.vlm_add_error(
                "post_help_request failed: HELP_CHARGE needs payload.provide_xy"
            )
            dm._finish_action(success=False)
            return
        if "deliver_xy" not in payload or _as_xy(payload.get("deliver_xy")) is None:
            dm.vlm_add_error(
                "post_help_request failed: HELP_CHARGE needs payload.deliver_xy"
            )
            dm._finish_action(success=False)
            return

        target = float(
            payload.get("want_charge_pct", payload.get("target_pct", 100.0))
        )
        kwargs["target_pct"] = max(0.0, min(100.0, target))
        kwargs["provide_xy"] = _as_xy(payload["provide_xy"])
        kwargs["deliver_xy"] = _as_xy(payload["deliver_xy"])

    else:
        dm.vlm_add_error(
            f"post_help_request failed: unsupported help_type={help_type}"
        )
        dm._finish_action(success=False)
        return

    ok, msg, rid = comms.post_request(**kwargs)
    if not ok:
        dm.vlm_add_error(f"post_help_request failed: {msg}")
        dm._finish_action(success=False)
        return

    def _fmt(xy):
        return f"({xy[0] / 100.0:.2f}m,{xy[1] / 100.0:.2f}m)" if xy else "N/A"

    dm._log(
        f"posted help request #{rid} ({help_type.name}) "
        f"bounty=${bounty:.2f} provide={_fmt(kwargs.get('provide_xy'))}"
    )
    if dm._recorder:
        dm._recorder.inc("help_posted", 1)
    dm._finish_action(success=True)