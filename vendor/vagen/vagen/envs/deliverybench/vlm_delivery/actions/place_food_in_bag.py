# actions/place_food_in_bag.py
# -*- coding: utf-8 -*-

import re
import copy
from typing import Any, Dict
from ..base.defs import DMAction
from ..entities.insulated_bag import InsulatedBag


def handle_place_food_in_bag(dm: Any, act: DMAction, _allow_interrupt: bool) -> None:
    """
    Places pending food items (from multiple orders) into the insulated bag
    according to a bag_cmd specification.

    Key behavior:
    - Existing items in the bag are moved first (e.g., A.2).
    - Raw numeric references are remapped to unique global indices.
    - Before inserting items, stable identifier fields (order_id/oid/item_id)
      are added to ensure later stable deletion at drop-off.
    """

    # ---------- helpers ----------
    def _extract_local_indices(cmd: str, max_n: int):
        """Extract bare numeric references 1..max_n, skipping patterns like A.2 or B.10."""
        nums = set()
        for m in re.finditer(r'(?<![A-Za-z]\.)\b([1-9]\d*)\b', cmd):
            i = int(m.group(1))
            if 1 <= i <= max_n:
                nums.add(i)
        return nums

    def _remap_cmd_indices(cmd: str, base: int, count: int):
        """Rewrite bare indices 1..count → base..base+count-1, without touching A.2-like segments."""
        mapping = {i: base + (i - 1) for i in range(1, count + 1)}

        def repl(m):
            i = int(m.group(1))
            return str(mapping.get(i, i))

        new_cmd = re.sub(r'(?<![A-Za-z]\.)\b([1-9]\d*)\b', repl, cmd)
        return new_cmd, mapping

    def _next_global_base(bag, need: int):
        """
        Computes the next global base index for newly inserted items.
        Prefer bag.max_index() if available; otherwise use _bag_index_counter.
        """
        if hasattr(bag, "max_index") and callable(getattr(bag, "max_index")):
            return int(bag.max_index()) + 1

        if not hasattr(dm, "_bag_index_counter"):
            approx = 0
            if hasattr(bag, "size") and callable(getattr(bag, "size")):
                approx = int(bag.size())
            dm._bag_index_counter = approx

        base = dm._bag_index_counter + 1
        dm._bag_index_counter += max(int(need), 1)
        return base

    # ---------- main ----------
    dm.vlm_clear_ephemeral()
    spec_text = (act.data.get("bag_cmd") or "").strip()

    if not spec_text:
        dm.vlm_add_error("place_food_in_bag failed: need bag_cmd")
        dm._finish_action(success=False)
        return

    if not dm._pending_food_by_order:
        dm._finish_action(success=True)
        return

    if not dm.insulated_bag:
        dm.insulated_bag = InsulatedBag()

    # Parse "order <id>: <cmd>" blocks
    tokens = re.split(r'(?i)order\s+(\d+)\s*:\s*', spec_text)
    per_order_cmd: Dict[int, str] = {}

    if len(tokens) >= 3:
        # tokens: [prefix, oid1, tail1, oid2, tail2, ...]
        for i in range(1, len(tokens), 2):
            try:
                oid = int(tokens[i])
                tail = tokens[i + 1].strip()
            except Exception:
                continue
            if tail:
                per_order_cmd[oid] = tail
    else:
        # If only a single pending order, allow bag_cmd without prefixing "order <id>:"
        if len(dm._pending_food_by_order) == 1:
            only_oid = next(iter(dm._pending_food_by_order.keys()))
            per_order_cmd[int(only_oid)] = spec_text
        else:
            dm.vlm_add_error(
                "place_food_in_bag failed: multiple pending orders; "
                "prefix each block with 'order <id>:'"
            )
            dm._force_place_food_now = True
            dm.vlm_ephemeral["bag_hint"] = dm._build_bag_place_hint()
            dm._finish_action(success=False)
            return

    hit_oids = [oid for oid in per_order_cmd.keys()
                if oid in dm._pending_food_by_order]

    if not hit_oids:
        dm.vlm_add_error(
            "place_food_in_bag failed: no matching pending orders for provided bag_cmd"
        )
        dm._force_place_food_now = True
        dm.vlm_ephemeral["bag_hint"] = dm._build_bag_place_hint()
        dm._finish_action(success=False)
        return

    # Transaction: snapshot state for rollback
    bag_before = copy.deepcopy(dm.insulated_bag)
    pending_before = {k: list(v) for k, v in dm._pending_food_by_order.items()}

    try:
        tmp_bag = copy.deepcopy(dm.insulated_bag)
        actually_placed_by_oid: Dict[int, list] = {}

        for oid in hit_oids:
            items = list(dm._pending_food_by_order.get(int(oid)) or [])
            if not items:
                continue

            order_cmd = per_order_cmd[int(oid)]
            n = len(items)

            # (1) Move existing items inside the bag (e.g., A.2, B.1).
            tmp_bag.move_items(order_cmd)

            # (2) Process raw-number references (new items).
            local_refs = _extract_local_indices(order_cmd, n)
            if not local_refs:
                continue

            base = _next_global_base(tmp_bag, need=len(local_refs))
            new_cmd, idx_map = _remap_cmd_indices(order_cmd, base, n)

            # Prepare item mapping (global indices → items).
            items_map: Dict[int, Any] = {}
            picked: list = []
            oid_int = int(oid)

            for i in sorted(local_refs):
                it = items[i - 1]

                # Ensure stable identifiers: order_id / oid
                if getattr(it, "order_id", None) != oid_int:
                    try:
                        setattr(it, "order_id", oid_int)
                    except Exception:
                        pass

                if getattr(it, "oid", None) != oid_int:
                    try:
                        setattr(it, "oid", oid_int)
                    except Exception:
                        pass

                # Ensure item_id exists if neither id nor item_id is present
                if (getattr(it, "id", None) is None) and (
                    getattr(it, "item_id", None) is None
                ):
                    try:
                        setattr(it, "item_id", i)
                    except Exception:
                        pass

                # Optional debug key (not used for removal)
                try:
                    iid_val = getattr(
                        it,
                        "id",
                        getattr(it, "item_id", getattr(it, "name", None)),
                    )
                    setattr(it, "_bag_key", (oid_int, iid_val))
                except Exception:
                    pass

                items_map[idx_map[i]] = it
                picked.append(it)

            # Add items to bag using rewritten numeric references
            tmp_bag.add_items(new_cmd, items_map)
            actually_placed_by_oid[int(oid)] = picked

        # Commit transaction
        dm.insulated_bag = tmp_bag

        # Update pending items for each order
        for oid in hit_oids:
            placed_list = actually_placed_by_oid.get(oid, [])
            if not placed_list:
                continue

            old_pending = list(dm._pending_food_by_order.get(int(oid)) or [])
            new_pending = [x for x in old_pending if x not in placed_list]

            if new_pending:
                dm._pending_food_by_order[int(oid)] = new_pending
            else:
                dm._pending_food_by_order.pop(int(oid), None)

    except Exception as e:
        # Rollback on failure
        dm.insulated_bag = bag_before
        dm._pending_food_by_order = pending_before
        dm.vlm_add_error(f"place_food_in_bag failed: {e}")
        dm._force_place_food_now = True
        dm.vlm_ephemeral["bag_hint"] = dm._build_bag_place_hint()
        dm._finish_action(success=False)
        return

    # Finalization: update hints
    if dm._pending_food_by_order:
        dm._force_place_food_now = True
        dm.vlm_ephemeral["bag_hint"] = dm._build_bag_place_hint()
    else:
        dm._force_place_food_now = False
        dm.vlm_ephemeral.pop("bag_hint", None)

    dm._log(f"placed pending food into bag for orders {hit_oids}")
    dm._finish_action(success=True)