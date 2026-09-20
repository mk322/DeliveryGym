# entities/store.py
# -*- coding: utf-8 -*-

import json
import os
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class StoreItem:
    """Single store entry in the catalog."""
    id: str
    name: str
    price: float
    function: str = ""   # Optional semantic tag or usage description


class StoreManager:
    """
    Store manager for simple in-sim purchases.

    Responsibilities:
      - Load a catalog of items from JSON.
      - Provide lookup utilities (list, get, price).
      - Handle purchase flow: balance check, payment, and inventory update.
    """

    def __init__(self, json_path: Optional[str] = None):
        self.items: Dict[str, StoreItem] = {}
        self.last_cost: float = 0.0
        self.json_path: Optional[str] = None

        if json_path:
            self.load_items(json_path)

    # ---------- Catalog loading ----------

    def load_items(self, json_path: str) -> None:
        """
        Load store items from a JSON file.

        Supported formats:
          1) A dict with an "items" key:  {"items": [ ... ]}
          2) A top-level list:            [ ... ]

        Each record must contain at least:
          - "id":   unique string id
          - "price": numeric price

        Optional fields:
          - "name":     display name (defaults to id)
          - "function": free-form description / tag
        """
        self.json_path = os.path.abspath(json_path)
        with open(self.json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        data = raw.get("items", raw) if isinstance(raw, dict) else raw
        if not isinstance(data, list):
            raise ValueError("store json must be a list or a dict with an 'items' list")

        self.items.clear()
        for rec in data:
            if not isinstance(rec, dict):
                continue
            _id = str(rec["id"])
            _name = str(rec.get("name", _id))
            _price = float(rec["price"])
            _func = str(rec.get("function", ""))  # Optional function/usage field
            self.items[_id] = StoreItem(id=_id, name=_name, price=_price, function=_func)

    # Legacy alias to keep compatibility with older code that calls _load()
    def _load(self) -> None:
        """
        Backward-compatible alias for load_items().

        Requires json_path to be set already (either via __init__ or a previous
        call to load_items).
        """
        if not self.json_path:
            raise RuntimeError("no json_path set; call load_items(path) first or pass it to __init__")
        self.load_items(self.json_path)

    # ---------- Query utilities ----------

    def list_items(self) -> List[StoreItem]:
        """Return all items in the current catalog as a list."""
        return list(self.items.values())

    def get_item(self, item_id: str) -> Optional[StoreItem]:
        """Look up a single item by its id."""
        return self.items.get(str(item_id))

    def get_price(self, item_id: str) -> float:
        """Return the price for the given item id, or 0.0 if unknown."""
        it = self.get_item(item_id)
        return float(it.price) if it else 0.0

    # ---------- Payment / purchase ----------

    def can_afford(self, dm, item_or_id) -> bool:
        """
        Check if the delivery agent has enough balance to buy the given item.

        dm         : any object with an 'earnings_total' attribute.
        item_or_id : StoreItem instance or item id string.
        """
        itm = item_or_id if isinstance(item_or_id, StoreItem) else self.get_item(str(item_or_id))
        return bool(itm and float(getattr(dm, "earnings_total", 0.0)) >= float(itm.price))

    def purchase(self, dm, item_id: str, *, qty: int = 1) -> Tuple[bool, str, float]:
        """
        Charge the delivery agent and add items to dm.inventory.

        Parameters
        ----------
        dm : object
            Must provide:
              - dm.earnings_total : current balance (float)
              - dm.inventory      : dict-like, will be created if missing
        item_id : str
            Store item id to purchase.
        qty : int, optional
            Quantity to purchase, minimum 1.

        Returns
        -------
        (ok, msg, total_cost) : Tuple[bool, str, float]
            ok          : purchase success flag
            msg         : status message ("ok" or error description)
            total_cost  : total cost charged (0.0 on failure)
        """
        itm = self.get_item(item_id)
        if not itm:
            return False, f"unknown item '{item_id}'", 0.0

        qty = max(1, int(qty))
        unit = float(itm.price)
        cost = unit * qty

        funds = float(getattr(dm, "earnings_total", 0.0))
        if funds < cost:
            return False, f"insufficient funds (${funds:.2f} < ${cost:.2f})", 0.0

        # Deduct payment
        dm.earnings_total = funds - cost
        self.last_cost = cost

        # Update inventory (create if missing)
        inv = getattr(dm, "inventory", None)
        if inv is None:
            dm.inventory = {}
            inv = dm.inventory
        inv[item_id] = int(inv.get(item_id, 0)) + qty

        return True, "ok", cost

    # ---------- Text rendering ----------

    def to_text(self, *, title: str = "Store Catalog") -> str:
        """
        Render the current catalog as a human-readable text table.

        Columns:
          - id
          - name
          - price
          - function (optional description / tag)
        """
        if not self.items:
            return f"{title}\n(empty)"

        items = list(self.items.values())

        # Compute simple column widths for alignment
        idw = max(len("id"),   max(len(it.id)   for it in items))
        namew = max(len("name"), max(len(it.name) for it in items))

        header = f"{'id'.ljust(idw)}  {'name'.ljust(namew)}  price   function"
        lines = [title, header, "-" * len(header)]

        for it in sorted(items, key=lambda x: x.id):
            price_str = f"${it.price:.2f}"
            func_str = it.function or ""
            lines.append(f"{it.id.ljust(idw)}  {it.name.ljust(namew)}  {price_str:<7} {func_str}")

        return "\n".join(lines)