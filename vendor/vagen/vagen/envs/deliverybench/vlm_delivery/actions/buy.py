# actions/buy.py
# -*- coding: utf-8 -*-

from typing import Any, Dict, Optional, List
from ..base.defs import DMAction
from ..utils.util import nearest_poi_xy, get_tol


def handle_buy(dm: Any, act: DMAction, _allow_interrupt: bool) -> None:
    """
    Handles purchasing one or multiple items.

    Supports two formats:
    1) Single item:
       BUY(item_id="energy_drink", qty=2)
       BUY(name="energy_drink", qty=2)

    2) Multiple items:
       BUY(items=[{"item_id": "energy_drink", "qty": 2},
                  {"name": "escooter_battery_pack", "qty": 1}])

    Rules:
    - If both single-item fields and 'items' are provided, quantities are merged.
    - Entries with qty <= 0 are ignored.
    - Action is successful if at least one item is purchased.
    - Failed entries are logged.
    """
    dm.vlm_clear_ephemeral()

    # Location and dependency checks
    if nearest_poi_xy(dm, "store", tol_cm=get_tol(dm.cfg, "nearby")) is None:
        dm.vlm_add_error("buy failed: not in a store")
        dm._finish_action(success=False)
        return

    if not dm._store_manager:
        dm.vlm_add_error("buy failed: no store manager")
        dm._finish_action(success=False)
        return

    # Merge purchase intentions: item_id -> qty
    purchases: Dict[str, int] = {}

    def _merge(iid: Optional[str], qty: Any):
        """Merge a single item entry into the purchase dictionary."""
        if iid is None:
            return
        sid = str(iid).strip()
        try:
            q = int(qty)
        except Exception:
            q = 0
        if sid and q > 0:
            purchases[sid] = purchases.get(sid, 0) + q

    # Multiple-item format: list/tuple of dicts containing item_id/name + qty
    if "items" in act.data:
        raw_items = act.data.get("items")
        if not isinstance(raw_items, (list, tuple)):
            dm.vlm_add_error(
                "buy failed: 'items' must be a list of dicts "
                "with {'item_id'/'name', 'qty'}"
            )
            dm._finish_action(success=False)
            return

        for entry in raw_items:
            if not isinstance(entry, dict):
                dm.vlm_add_error(
                    "buy failed: each element in 'items' must be a dict "
                    "containing item_id/name and qty"
                )
                dm._finish_action(success=False)
                return
            _merge(entry.get("item_id") or entry.get("name"), entry.get("qty", 1))

    # Single-item format: item_id/name + qty
    if "item_id" in act.data or "name" in act.data:
        _merge(act.data.get("item_id") or act.data.get("name"), act.data.get("qty", 1))

    if not purchases:
        dm.vlm_add_error(
            "buy failed: provide item_id+qty or "
            "items=[{item_id/name, qty}, ...]"
        )
        dm._finish_action(success=False)
        return

    # Process each purchase without modifying the store manager interface
    total_cost = 0.0
    bought_lines: List[str] = []
    failed_lines: List[str] = []

    for iid, q in purchases.items():
        ok, msg, cost = dm._store_manager.purchase(dm, item_id=iid, qty=int(q))
        if ok:
            total_cost += float(cost or 0.0)
            bought_lines.append(f"{q} x {iid}")
        else:
            failed_lines.append(f"{iid} ({msg})")

    # Final result handling
    if bought_lines:
        dm._log(f"bought {', '.join(bought_lines)} for ${total_cost:.2f}")

        if failed_lines:
            dm._log("buy partial fails: " + "; ".join(failed_lines))

        if dm._recorder:
            dm._recorder.on_purchase(
                dm.clock.now_sim(),
                items=", ".join(bought_lines),
                cost=float(total_cost),
            )

        dm._finish_action(success=True)

    else:
        dm.vlm_add_error(
            "buy failed: "
            + ("; ".join(failed_lines) if failed_lines else "unknown reason")
        )
        dm._finish_action(success=False)