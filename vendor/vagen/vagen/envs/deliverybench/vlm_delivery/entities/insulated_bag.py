# entities/insulated_bag.py
# -*- coding: utf-8 -*-

"""
Insulated_bag.py

Multi-compartment insulated bag model with:
- one lightweight "air/bag" node per compartment that exchanges heat with items
- simple in-bag odor mixing
- optional motion damage accumulation for fragile items
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional


def _letters(n: int) -> List[str]:
    """
    Generate compartment labels ["A", "B", ...] for a given count.
    """
    n = max(1, min(26, int(n)))
    return [chr(ord('A') + i) for i in range(n)]


def _describe_item(obj: Any, name: str) -> str:
    parts = [name]
    if hasattr(obj, "temp_c") and isinstance(getattr(obj, "temp_c"), (int, float)):
        serv = getattr(obj, "serving_temp_c", None)
        if isinstance(serv, (int, float)):
            parts.append(f"({obj.temp_c:.1f}°C, serve={serv:.0f}°C)")
        else:
            parts.append(f"({obj.temp_c:.1f}°C)")
    tags: List[str] = []
    serv = getattr(obj, "serving_temp_c", None)
    if isinstance(serv, (int, float)):
        if serv >= 45: tags.append("HOT")
        elif serv <= 5: tags.append("FROZEN")
        elif serv <= 15: tags.append("COLD")
    odor = getattr(obj, "odor", "")
    if odor and str(odor).strip() and str(odor).strip().lower() not in ("none", "no"):
        tags.append("STINKY")
    if bool(getattr(obj, "motion_sensitive", False)):
        tags.append("FRAGILE")
    if tags:
        parts.append("[" + "][".join(tags) + "]")
    return " ".join(parts)


@dataclass
class IcePack:
    """
    Simple cold pack model used inside the insulated bag.
    """
    name: str = "Ice Pack"
    temp_c: float = -8.0
    heat_capacity: float = 8.0
    odor_contamination: float = 0.0
    motion_sensitive: bool = False


@dataclass
class HeatPack:
    """
    Simple heat pack model used inside the insulated bag.
    """
    name: str = "Heat Pack"
    temp_c: float = 60.0
    heat_capacity: float = 5.0
    odor_contamination: float = 0.0
    motion_sensitive: bool = False


@dataclass
class InsulatedBag:
    """
    Insulated delivery bag with multiple logical compartments.

    Each compartment keeps:
    - a list of items (food, thermal packs, misc objects)
    - an internal "air" temperature node
    """
    num_compartments: int = 4

    # Environment / exchange parameters
    ambient_temp_c: float = 23.0        # Initial air temperature for each compartment
    exchange_tau_min: float = 10.0      # Time constant (minutes) for air↔item heat exchange
    air_heat_capacity: float = 0.4      # Effective heat capacity of the air/bag node (relative)
    odor_mix_tau_min: float = 1.5       # Time constant (minutes) for odor mixing
    tau_external_min: Optional[float] = 12.0      # Time constant (minutes) for compartment-air↔ambient relaxation; None = closed bag

    # Runtime state
    _comps: List[List[Any]] = field(default_factory=list, repr=False)   # Items per compartment
    _comp_temp_c: List[float] = field(default_factory=list, repr=False) # Air temperature per compartment
    _move_count: int = field(default=0, repr=False)

    def __post_init__(self):
        """Initialize compartment lists and initial air temperatures."""
        n = int(self.num_compartments)
        self._comps = [[] for _ in range(n)]
        self._comp_temp_c = [float(self.ambient_temp_c) for _ in range(n)]

    # ---------------- basic helpers ----------------
    @property
    def labels(self) -> List[str]:
        """Return the list of compartment labels (e.g., ['A', 'B', 'C', ...])."""
        return _letters(self.num_compartments)

    def _idx(self, label: str) -> int:
        """
        Convert a compartment label (e.g., 'A') to its index.

        Raises:
            ValueError: if the label is invalid.
        """
        lab = str(label).strip().upper()
        if lab not in self.labels:
            raise ValueError(f"invalid compartment: {lab}")
        return self.labels.index(lab)

    # ---------------- display ----------------
    def list_items(self) -> str:
        """
        Render a human-readable summary of all compartments and items.

        Format:
            A: (Ta °C)
              A.1 ItemName (Ti °C)
              A.2 ...
            B: (Ta °C)
            ...

        Returns:
            Multi-line string, or "(empty)" if there is no content.
        """
        out: List[str] = []
        for i, lab in enumerate(self.labels):
            tc = self._comp_temp_c[i]
            out.append(f"{lab}: ({tc:.1f}°C)")
            for j, obj in enumerate(self._comps[i], start=1):
                name = (
                    getattr(obj, "name", None)
                    or getattr(obj, "title", None)
                    or getattr(obj, "label", None)
                    or str(obj)
                )
                out.append(f"  {lab}.{j} {_describe_item(obj, name)}")
        return "\n".join(out) if out else "(empty)"

    # ---------------- internal add/remove ----------------
    def _append_to_comp(self, i: int, item: Any) -> None:
        """
        Append an item to compartment index i.

        The air temperature is left unchanged; items are expected to already
        carry a correct temp_c when they are placed into the bag.
        """
        self._comps[i].append(item)

    def _pop_from_comp(self, i: int, k: int) -> Any:
        """
        Remove the k-th item (1-based) from compartment index i.

        If the compartment becomes empty, its air temperature is reset
        to ambient_temp_c.
        """
        item = self._comps[i].pop(k - 1)
        if not self._comps[i]:
            self._comp_temp_c[i] = float(self.ambient_temp_c)
        return item

    # ---------------- scripted moves/adds ----------------
    def move_items(self, spec: str) -> None:
        """
        Move existing items between compartments according to a spec string.

        Example:
            "A.1, A.3 -> B; C.2 -> A"

        Only moves items that are already in the bag; it does not create items.
        """
        if not spec:
            return
        for clause in spec.split(";"):
            clause = clause.strip()
            if not clause or "->" not in clause:
                continue
            left, right = clause.split("->", 1)
            dst = right.strip().upper()
            ptrs = []
            for tok in left.split(","):
                tok = tok.strip()
                if "." in tok:
                    lab, k = tok.split(".", 1)
                    ptrs.append((lab.strip().upper(), int(k)))
            if not ptrs:
                continue
            # Group by source compartment and pop in descending index order
            by_lab: Dict[str, List[int]] = {}
            for lab, k in ptrs:
                by_lab.setdefault(lab, []).append(k)
            for lab, ks in by_lab.items():
                ks.sort(reverse=True)
                src_i = self._idx(lab)
                dst_i = self._idx(dst)
                for k in ks:
                    if not (1 <= k <= len(self._comps[src_i])):
                        raise IndexError(f"{lab}.{k} not found")
                    item = self._pop_from_comp(src_i, k)
                    self._append_to_comp(dst_i, item)

    def add_items(self, spec: str, items_by_number: Dict[int, Any]) -> None:
        """
        Add numbered items into compartments according to a spec string.

        Example:
            spec: "1,2 -> A; 3 -> B"
            items_by_number: {1: item1, 2: item2, 3: item3, ...}

        Typically used when placing freshly picked-up items into the bag.
        """
        if not spec:
            return
        for clause in spec.split(";"):
            clause = clause.strip()
            if not clause or "->" not in clause:
                continue
            left, right = clause.split("->", 1)
            dst = right.strip().upper()
            dst_i = self._idx(dst)

            nums: List[int] = []
            for tok in re.split(r"[,\s]+", left.strip()):
                if not tok or "." in tok:
                    continue
                try:
                    nums.append(int(tok))
                except Exception:
                    raise ValueError(f"bad item reference {tok!r} in bag_cmd")
            for n in nums:
                if n not in items_by_number:
                    raise KeyError(f"item #{n} not found")
                self._append_to_comp(dst_i, items_by_number[n])

    def add_misc_item(self, dst_label: str, item: Any) -> None:
        """
        Insert an arbitrary item into a compartment.

        For IcePack / HeatPack items, at most one instance of that class
        is kept per compartment. If removing old packs empties the compartment,
        its air temperature is reset to ambient_temp_c.
        """
        i = self._idx(dst_label)

        # Enforce "at most one pack of the same class" per compartment
        if isinstance(item, (IcePack, HeatPack)):
            cls = item.__class__
            remaining = [obj for obj in self._comps[i] if not isinstance(obj, cls)]
            if len(remaining) != len(self._comps[i]):
                self._comps[i] = remaining
                if not self._comps[i]:
                    self._comp_temp_c[i] = float(self.ambient_temp_c)

        self._append_to_comp(i, item)

    # ---------------- virtual time: temperature ----------------
    def tick_temperatures(self, delta_s: float) -> None:
        if not getattr(self, '_enable_temperature', True):
            return
        """
        Advance temperatures for a virtual time step.

        Models heat exchange only between compartment air and items; the bag
        does not exchange heat with the external environment.
        """
        if delta_s <= 0:
            return
        alpha = delta_s / max(1e-6, self.exchange_tau_min * 60.0)
        if alpha <= 0:
            return
        if alpha > 0.5:
            alpha = 0.5  # stability guard

        Cab = float(self.air_heat_capacity)

        for i, comp in enumerate(self._comps):
            if not comp:
                continue

            Ta0 = float(self._comp_temp_c[i])

            # Snapshot old temperatures for a synchronous update
            Ci_list: List[float] = []
            Ti0_list: List[float] = []
            for it in comp:
                Ci = float(getattr(it, "heat_capacity", 1.0) or 1.0)
                Ti0 = float(getattr(it, "temp_c", Ta0))
                Ci_list.append(Ci)
                Ti0_list.append(Ti0)
            if not Ti0_list:
                continue

            # Update air node first (energy-conserving within the bag)
            S = 0.0
            for Ci, Ti0 in zip(Ci_list, Ti0_list):
                S += Ci * (Ti0 - Ta0)
            Ta_new = Ta0 + alpha * (S / max(1e-9, Cab))

            if self.tau_external_min is not None and self.tau_external_min > 0:
                alpha_ext = delta_s / max(1e-6, float(self.tau_external_min) * 60.0)
                if alpha_ext > 0.5:
                    alpha_ext = 0.5
                Ta_new = Ta_new + alpha_ext * (float(self.ambient_temp_c) - Ta_new)

            # Thermal packs are active sources: they hold the compartment air at pack temp
            packs = [it for it in comp if isinstance(it, (IcePack, HeatPack))]
            if packs:
                Ta_new = sum(float(p.temp_c) for p in packs) / len(packs)

            self._comp_temp_c[i] = Ta_new

            # Then update all items using the previous air temperature Ta0
            for it, Ti0 in zip(comp, Ti0_list):
                Ti_new = Ti0 + alpha * (Ta0 - Ti0)
                it.temp_c = Ti_new

    def _item_key(self, it):
        """
        Build a stable key (order_id, item_id) for an item.

        Tries multiple attribute names:
        - order_id / oid / order.id / order.oid for the order
        - id / item_id / name for the item itself
        """
        # Order id
        oid = getattr(it, "order_id", getattr(it, "oid", None))
        if oid is None:
            ord_obj = getattr(it, "order", None)
            if ord_obj is not None:
                oid = getattr(ord_obj, "id", getattr(ord_obj, "oid", None))
        # Item id (fall back to name if needed)
        iid = getattr(it, "id", getattr(it, "item_id", getattr(it, "name", None)))
        return (oid, iid)

    def remove_items(self, items: List[Any]) -> None:
        """
        Remove items from the bag using multiple matching strategies.

        A given entry in `items` can be:
            - an object (matched by identity, equality, and stable key), or
            - a key tuple (order_id, item_id).

        Matching rules:
            1) Same object identity (id).
            2) Equality via __eq__ (if implemented).
            3) Same stable key (order_id, item_id).
        """
        if not items:
            return

        # Identity set for object-based removal
        obj_id_set = {id(x) for x in items if not isinstance(x, tuple)}

        # Stable key set
        key_set = set()
        for it in items:
            if isinstance(it, tuple) and len(it) >= 2:
                key_set.add((it[0], it[1]))
            else:
                key_set.add(self._item_key(it))

        # Filter each compartment in place
        for i, comp in enumerate(self._comps):
            if not comp:
                continue
            kept = []
            for x in comp:
                same_obj = (id(x) in obj_id_set)
                same_key = (self._item_key(x) in key_set)
                same_eq  = (x in items) if not same_obj else True
                if same_obj or same_key or same_eq:
                    continue
                kept.append(x)
            self._comps[i] = kept
            if not kept:
                self._comp_temp_c[i] = float(self.ambient_temp_c)

    def tick_odor(self, delta_s: float) -> None:
        if not getattr(self, '_enable_smell', True):
            return
        """
        Advance odor contamination in each compartment.

        Items expose an `odor_contamination` attribute (0..1). Within each
        compartment, levels monotonically approach the maximum odor present.
        """
        if delta_s <= 0:
            return
        tau_s = max(1e-6, float(self.odor_mix_tau_min) * 60.0)
        alpha = delta_s / tau_s
        if alpha <= 0:
            return
        if alpha > 0.5:
            alpha = 0.5  # stability guard

        for comp in self._comps:
            if not comp:
                continue

            levels = []
            for it in comp:
                try:
                    levels.append(float(getattr(it, "odor_contamination", 0.0)))
                except Exception:
                    levels.append(0.0)

            if not levels:
                continue

            target = max(levels)
            if target <= 0.0:
                # No strong odor source in this compartment
                continue

            # Move each item's level toward the target, clamped to [0, 1]
            for it, oi in zip(comp, levels):
                new_oi = oi + alpha * (target - oi)
                if new_oi < oi:
                    new_oi = oi
                if new_oi > 1.0:
                    new_oi = 1.0
                try:
                    setattr(it, "odor_contamination", float(new_oi))
                except Exception:
                    pass

    def bump_motion_damage_shared(self, every_n: int = 4, inc: int = 1) -> int:
        if not getattr(self, '_enable_fragility', True):
            return 0
        self._move_count += 1
        if every_n <= 0 or (self._move_count % int(every_n) != 0):
            return 0

        changed = 0
        for comp in self._comps:
            if len(comp) <= 1:
                continue
            for it in comp:
                try:
                    if not bool(getattr(it, "motion_sensitive", False)):
                        continue
                    curr = int(getattr(it, "damage_level", 0) or 0)
                    newv = min(curr + int(inc), 3)
                    if newv != curr:
                        setattr(it, "damage_level", newv)
                        changed += 1
                except Exception:
                    continue
        return changed

    # ---------------- motion damage (called externally on bumps) ----------------
    def bump_motion_damage(self, inc: int = 1) -> int:
        if not getattr(self, '_enable_fragility', True):
            return
        """
        Apply motion-induced damage to all motion-sensitive items.

        Items with `motion_sensitive=True` have `damage_level` increased by
        `inc` (default 1), capped at 3. Items without `damage_level` are
        treated as if they start from 0.

        Returns:
            Number of items whose damage_level was increased.
        """
        if inc <= 0:
            return 0

        changed = 0
        for comp in self._comps:
            if not comp:
                continue
            for it in comp:
                try:
                    if bool(getattr(it, "motion_sensitive", False)):
                        curr = int(getattr(it, "damage_level", 0) or 0)
                        newv = curr + int(inc)
                        if newv > 3:
                            newv = 3
                        if newv != curr:
                            setattr(it, "damage_level", newv)
                            changed += 1
                except Exception:
                    # Ignore objects that do not support these attributes
                    continue
        return changed