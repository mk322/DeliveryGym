# entities/temp_box.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class TempBox:
    """
    Handoff container for help tasks.

    The box can hold:
      - inventory: generic item counts
      - food_by_order: mapping from order_id to a list of food-like objects
      - escooter: shared e-scooter object for CHARGE tasks

    Temperature relaxes towards ambient over virtual time, but only for items
    that expose temp_c / serving_temp_c attributes.
    """
    box_id: int
    req_id: int
    owner_id: str                  # who placed (for logs)
    role: str                      # 'publisher' or 'helper'
    xy: Tuple[float, float]
    created_sim: float

    # unified content
    inventory: Dict[str, int] = field(default_factory=dict)
    food_by_order: Dict[int, List[Any]] = field(default_factory=dict)  # {order_id: [FoodItem,...]}
    escooter: Optional[Any] = None

    last_thermal_sim: Optional[float] = None
    k_food_per_s: float = 1.0 / 1200.0

    # --- temperature relax helper ---
    def _relax_temp(self, t0: float, amb: float, k_per_s: float, dt: float) -> float:
        dt = max(0.0, float(dt))
        k = max(0.0, float(k_per_s))
        return float(amb) + (float(t0) - float(amb)) * math.exp(-k * dt)

    # --- tick only foods, ignore inventory/escooter ---
    def thermal_tick(
        self,
        now_sim: float,
        ambient_c: float,
        k_food_per_s: Optional[float] = None,
    ) -> None:
        now = float(now_sim)
        last = float(self.last_thermal_sim) if self.last_thermal_sim is not None else now
        dt = max(0.0, now - last)
        if dt <= 1e-9:
            self.last_thermal_sim = now
            return

        k = float(k_food_per_s if k_food_per_s is not None else self.k_food_per_s)
        amb = float(ambient_c)

        for items in (self.food_by_order or {}).values():
            for it in (items or []):
                # only update when the item exposes a usable temperature field
                base = getattr(it, "temp_c", None)
                if base is None:
                    base = getattr(it, "serving_temp_c", None)
                if base is None:
                    continue
                new_t = self._relax_temp(float(base), amb, k, dt)
                try:
                    setattr(it, "temp_c", float(new_t))
                    setattr(it, "last_temp_update_sim", now)
                except Exception:
                    # object not writable; ignore
                    pass

        self.last_thermal_sim = now

    def merge_payload(self, payload: Dict[str, Any]) -> None:
        """
        Merge incoming content into the box.

        Caller is responsible for:
          - deducting items from the agent's own inventory
          - ensuring payload structure is correct

        Supported keys:
          - "inventory": {item_id: count}
          - "food_by_order": {order_id: [food_item, ...]}
          - short form: {"order_id": int, "food_items": [food_item, ...]}
          - "escooter": e-scooter object to hand over
        """
        if not payload:
            return

        # inventory
        inv = payload.get("inventory") or {}
        for k, v in inv.items():
            self.inventory[k] = int(self.inventory.get(k, 0)) + int(v)

        # food items (support short form {order_id, food_items})
        fbo = payload.get("food_by_order") or {}
        if "order_id" in payload and "food_items" in payload:
            fbo = {int(payload["order_id"]): list(payload["food_items"])}

        for oid, items in fbo.items():
            oid = int(oid)
            cur = self.food_by_order.get(oid, [])
            cur += list(items or [])
            self.food_by_order[oid] = cur

        # e-scooter handover
        if "escooter" in payload:
            self.escooter = payload["escooter"]