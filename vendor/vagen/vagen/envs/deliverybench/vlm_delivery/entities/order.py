# entities/order.py
# -*- coding: utf-8 -*-

import time
import random
import math
from dataclasses import dataclass, field
from threading import Lock, RLock
from typing import List, Optional, Dict, Any, Tuple, Iterable, Union

from ..base.types import Vector
from ..base.timer import VirtualClock

# ==================== Tunables ====================

PAY_PER_KM: float = 10.0
AVG_SPEED_MPS: float = 1.6
TIME_MULTIPLIER: float = 1.25
EARNING_JITTER: Tuple[float, float] = (0.75, 1.35)
TIME_JITTER: Tuple[float, float] = (0.85, 1.25)

HANDOFF_RADIUS_MIN_CM: float = 500.0   # 5 m
HANDOFF_RADIUS_MAX_CM: float = 1000.0  # 10 m



# ============================== FoodItem ==============================

def _describe_order_item(it: Any) -> str:
    name = getattr(it, "name", None) or str(it)
    parts = [name]
    serv = getattr(it, "serving_temp_c", None)
    if isinstance(serv, (int, float)):
        parts.append(f"(serve={serv:.0f}°C)")
    tags: List[str] = []
    if isinstance(serv, (int, float)):
        if serv >= 45: tags.append("HOT")
        elif serv <= 5: tags.append("FROZEN")
        elif serv <= 15: tags.append("COLD")
    odor = getattr(it, "odor", "")
    if odor and str(odor).strip() and str(odor).strip().lower() not in ("none", "no"):
        tags.append("STINKY")
    if bool(getattr(it, "motion_sensitive", False)):
        tags.append("FRAGILE")
    if tags:
        parts.append("[" + "][".join(tags) + "]")
    return " ".join(parts)


@dataclass
class FoodItem:
    name: str
    category: str = ""
    odor: str = ""
    motion_sensitive: bool = False
    damage_level: int = 0
    nonthermal_time_sensitive: bool = False
    prep_time_s: int = 0

    # Thermal fields (optional; fall back to dataclass defaults if not provided in menu)
    serving_temp_c: float = 60.0
    safe_min_c: float = 50.0
    safe_max_c: float = 70.0
    heat_capacity: float = 1.0

    # Runtime (virtual time)
    temp_c: float = 0.0
    prepared_at_sim: float = 0.0
    picked_at_sim: float = 0.0
    delivered_at_sim: float = 0.0

    odor_contamination: float = 0.0

    def __str__(self) -> str:
        return self.name


# ============================== Order ==============================
@dataclass
class Order:
    """
    - On init: bind nearest door_node via Map._pick_meta and cache road names.
    - Routing via Map.shortest_path_nodes(start_node, end_node) -> (path, dist_cm).
    - Pricing by distance only; ETA from distance/avg speed.
    """
    city_map: Any
    pickup_address: Vector
    delivery_address: Vector
    items: List[FoodItem] = field(default_factory=list)
    special_note: str = ""

    # Routing / pricing
    path_nodes: List[Any] = field(default_factory=list)
    distance_cm: float = 0.0
    eta_s: float = 0.0
    time_limit_s: float = 0.0
    earnings: float = 0.0

    # Node / road names
    pickup_node: Any = None
    dropoff_node: Any = None
    pickup_road_name: str = ""
    dropoff_road_name: str = ""
    road_name: str = ""

    # Building boundary
    dropoff_building_box: Optional[Dict[str, Any]] = None

    # Handoff
    handoff_address: Optional[Vector] = None

    # State
    id: int = field(init=False, default=-1)
    is_accepted: bool = False
    has_picked_up: bool = False
    has_delivered: bool = False
    start_time: float = field(init=False, default_factory=time.time)
    delivery_time: float = 0.0
    delivery_men: List[Any] = field(default_factory=list)

    is_shared: bool = False
    meeting_point: Optional[Vector] = None

    _lock: RLock = field(init=False, default_factory=RLock)

    # Preparation timeline (virtual time; set on accept)
    prep_started_sim: Optional[float] = None
    prep_ready_sim: Optional[float] = None
    prep_longest_s: float = 0.0

    sim_started_s: float = 0.0
    sim_elapsed_active_s: float = 0.0
    sim_delivered_s: Optional[float] = None

    allowed_delivery_methods: Optional[List[str]] = None

    # Scoring jitters
    earning_mult: float = 1.0
    time_mult: float = 1.0

    # Per-order RNG (injected by OrderManager)
    rng: Optional[random.Random] = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        self._bind_nodes_initial()

    # ---------------- Bind nearest nodes + road names ----------------
    def _bind_nodes_initial(self):
        sx, sy = float(self.pickup_address.x), float(self.pickup_address.y)
        tx, ty = float(self.delivery_address.x), float(self.delivery_address.y)

        if not hasattr(self.city_map, "_pick_meta"):
            raise AttributeError("Map missing _pick_meta(kind, (x,y))")

        pu_meta = self.city_map._pick_meta("restaurant", (sx, sy))
        do_meta = self.city_map._pick_meta("building", (tx, ty))

        # Anchor pickup/dropoff at the POI's DOCK waypoint — the node the agent
        # can actually stand on (the road-side access point in the waypoint
        # graph) — not the building door on the perimeter. The door can sit
        # several metres off the road, farther than the pickup tolerance, so
        # using it made some orders impossible to pick up once the agent had
        # reached the only reachable waypoint. With the dock anchor, "standing on
        # the dock" always satisfies PICKUP/DROP_OFF.
        self.pickup_node = pu_meta.get("dock_node") or pu_meta.get("door_node")
        self.dropoff_node = do_meta.get("dock_node") or do_meta.get("door_node")
        self.pickup_road_name = str(pu_meta.get("road_name", "") or "")
        self.dropoff_road_name = str(do_meta.get("road_name", "") or "")

        self.dropoff_building_box = do_meta.get("building_box")

        if self.pickup_node is None or self.dropoff_node is None:
            raise RuntimeError("Failed to bind pickup or dropoff node")

        self.road_name = (
            self.pickup_road_name
            if self.pickup_road_name == self.dropoff_road_name
            else f"{self.pickup_road_name} -> {self.dropoff_road_name}"
        )

    def _sample_handoff_position(self) -> Optional[Vector]:
        """Sample a handoff point near dropoff_node (outside building)."""
        _rng = self.rng or random
        if self.dropoff_node is None:
            return None

        center_x = float(self.dropoff_node.position.x)
        center_y = float(self.dropoff_node.position.y)

        max_attempts = 50
        for _ in range(max_attempts):
            angle = _rng.uniform(0, 2 * math.pi)
            radius = _rng.uniform(HANDOFF_RADIUS_MIN_CM, HANDOFF_RADIUS_MAX_CM)
            handoff_x = center_x + radius * math.cos(angle)
            handoff_y = center_y + radius * math.sin(angle)
            handoff_pos = Vector(handoff_x, handoff_y)
            if not self._is_point_inside_building(handoff_pos):
                return handoff_pos

        # Fallback: push along door direction
        if self.dropoff_building_box:
            building_center_x = self.dropoff_building_box.get("x", center_x)
            building_center_y = self.dropoff_building_box.get("y", center_y)
            dx = center_x - building_center_x
            dy = center_y - building_center_y
            length = math.hypot(dx, dy)
            if length > 0:
                dx /= length
                dy /= length
                off = HANDOFF_RADIUS_MAX_CM
                return Vector(center_x + dx * off, center_y + dy * off)
        return None

    def _is_point_inside_building(self, point: Vector) -> bool:
        if self.dropoff_building_box is None:
            return False
        b = self.dropoff_building_box
        bx = b.get("x", 0.0)
        by = b.get("y", 0.0)
        bw = b.get("w", 0.0)
        bh = b.get("h", 0.0)
        byaw = b.get("yaw", 0.0)
        if bw <= 0 or bh <= 0:
            return False

        # to local frame
        local_x = point.x - bx
        local_y = point.y - by
        yaw_rad = math.radians(-byaw)
        cos_yaw = math.cos(yaw_rad)
        sin_yaw = math.sin(yaw_rad)
        rx = local_x * cos_yaw - local_y * sin_yaw
        ry = local_x * sin_yaw + local_y * cos_yaw

        return abs(rx) <= bw / 2.0 and abs(ry) <= bh / 2.0

    # ---------------- Routing ----------------
    def plan_route(self) -> List[Any]:
        if not hasattr(self.city_map, "shortest_path_nodes"):
            raise AttributeError("Map missing shortest_path_nodes(start, target)")
        res = self.city_map.shortest_path_nodes(self.pickup_node, self.dropoff_node)
        if not (isinstance(res, tuple) and len(res) == 2):
            raise TypeError("shortest_path_nodes must return (path_nodes, dist_cm)")
        path, dist_cm = res
        if not isinstance(path, list) or not isinstance(dist_cm, (int, float)):
            raise TypeError("shortest_path_nodes returns wrong types")
        if not path or dist_cm <= 0:
            raise RuntimeError("Invalid path or distance")
        self.path_nodes = path
        self.distance_cm = float(dist_cm)
        return self.path_nodes

    # ---------------- Pricing & deadline ----------------
    def price_and_deadline(self):
        dist_m = self.distance_cm / 100.0
        self.eta_s = (dist_m / AVG_SPEED_MPS) if dist_m > 0 else 0.0
        dist_km = dist_m / 1000.0

        _rng = self.rng or random
        if getattr(self, '_disable_jitter', False):
            self.earning_mult = 1.0
            self.time_mult = 1.0
        else:
            self.earning_mult = _rng.uniform(*EARNING_JITTER)
            self.time_mult = _rng.uniform(*TIME_JITTER)
        

        base = PAY_PER_KM * dist_km
        self.earnings = base * self.earning_mult

        raw_limit = self.eta_s * TIME_MULTIPLIER * self.time_mult
        prep_longest = float(getattr(self, "prep_longest_s", 0.0) or 0.0)
        min_limit = prep_longest + 60.0
        self.time_limit_s = max(raw_limit, min_limit)

    # ---------------- One-shot compute ----------------
    def compute(self) -> List[Any]:
        self.plan_route()
        self.price_and_deadline()
        return self.path_nodes

    def compute_with_map(self, city_map: Any = None, tol_cm: float = 50.0) -> List[Any]:
        return self.compute()

    # ---------------- Text output ----------------
    def _fmt_time(self, seconds: float) -> str:
        minutes = int((seconds + 59) // 60)
        return f"{minutes} min"

    def to_text(self) -> str:
        with self._lock:
            def _fmt_min(sec: float) -> str:
                s = max(0.0, float(sec))
                minutes = int((s + 59) // 60)
                return f"{minutes} min"

            px_m = float(self.pickup_node.position.x) / 100.0
            py_m = float(self.pickup_node.position.y) / 100.0
            dx_m = float(self.dropoff_node.position.x) / 100.0
            dy_m = float(self.dropoff_node.position.y) / 100.0

            tl = float(getattr(self, "time_limit_s", 0.0) or 0.0)
            earnings = float(getattr(self, "earnings", 0.0) or 0.0)

            lines = []

            # Address / waypoint id annotations (populated by
            # Map._build_addresses / _build_waypoint_graph; empty strings
            # if the underlying map is the legacy single-graph version).
            pu_addr = getattr(self.pickup_node, "address", "") or ""
            do_addr = getattr(self.dropoff_node, "address", "") or ""

            spent = float(getattr(self, "sim_elapsed_active_s", 0.0) or 0.0)
            left = tl - spent
            if tl > 0:
                time_line = f"  Time Left : {_fmt_min(left)}" if left >= 0 else f"  OVERTIME  : {_fmt_min(-left)} — Deliver ASAP!"
            else:
                time_line = "  Time Left : N/A"

            if bool(getattr(self, "has_delivered", False)):
                status = "Delivered"
            else:
                picked = bool(getattr(self, "has_picked_up", False))
                ready = False
                if hasattr(self, "is_ready_for_pickup"):
                    try:
                        ready = bool(self.is_ready_for_pickup(self.sim_started_s + self.sim_elapsed_active_s))
                    except Exception:
                        ready = False

                if not picked:
                    if ready:
                        status = "Ready for pickup"
                    else:
                        remain_prep = 0.0
                        if hasattr(self, "remaining_prep_s"):
                            try:
                                remain_prep = float(self.remaining_prep_s(self.sim_started_s + self.sim_elapsed_active_s))
                            except Exception:
                                remain_prep = 0.0
                        status = f"Food is still being prepared (~{_fmt_min(remain_prep)})"
                else:
                    status = "Picked up, waiting for delivery"

            pickup_disp = pu_addr or self.pickup_road_name or f"({px_m:.2f}m, {py_m:.2f}m)"
            dropoff_disp = do_addr or self.dropoff_road_name or f"({dx_m:.2f}m, {dy_m:.2f}m)"
            lines.extend([
                f"[Order #{self.id}]",
                f"  Pickup : {pickup_disp}",
                f"  Dropoff: {dropoff_disp}",
                time_line,
                f"  $$     : ${earnings:.2f}",
                f"  Status : {status}",
            ])
            if self.items:
                lines.append("  Items  :")
                for i, it in enumerate(self.items, start=1):
                    lines.append(f"    {i}. {_describe_order_item(it)}")
            if self.special_note:
                lines.append(f"  Note   : {self.special_note}")
            return "\n".join(lines)

    def __str__(self) -> str:
        return self.to_text()

    # ---------------- Viewer hints ----------------
    def to_active_order_hint(self) -> Dict[str, Any]:
        with self._lock:
            px, py = float(self.pickup_address.x), float(self.pickup_address.y)
            dx, dy = float(self.delivery_address.x), float(self.delivery_address.y)
            requires_handoff = 'hand_to_customer' in (self.allowed_delivery_methods or [])
            hint = dict(
                id=self.id,
                pickup_xy=(px, py),
                dropoff_xy=(dx, dy),
                distance_cm=self.distance_cm,
                eta_s=self.eta_s,
                earnings=self.earnings,
                road_name=self.road_name,
                requires_handoff=requires_handoff,
            )
            if requires_handoff and self.handoff_address:
                hx, hy = float(self.handoff_address.x), float(self.handoff_address.y)
                hint["handoff_xy"] = (hx, hy)
            return hint

    # ---------------- Prep timeline helpers ----------------
    def active_now(self) -> float:
        return float(self.sim_started_s) + float(self.sim_elapsed_active_s or 0.0)

    def ready_at(self) -> float:
        return float(self.sim_started_s) + float(self.prep_longest_s or 0.0)

    def is_ready_for_pickup(self, now_sim: Optional[float] = None) -> bool:
        with self._lock:
            if self.prep_started_sim is None:
                return False
            now = self.active_now() if now_sim is None else float(now_sim)
            return now >= self.ready_at()

    def remaining_prep_s(self, now_sim: Optional[float] = None) -> float:
        with self._lock:
            if self.prep_started_sim is None:
                return float(self.prep_longest_s or 0.0)
            now = self.active_now() if now_sim is None else float(now_sim)
            return max(0.0, self.ready_at() - now)

    # ---------------- Agent scoring (unchanged) ----------------
    def score_for_agent(
        self,
        agent_xy: Tuple[float, float],
        active_seeds_xy: List[Tuple[float, float]],
        *,
        include_aux: bool = False,
    ) -> Tuple[float, float, float]:
        ax, ay = float(agent_xy[0]), float(agent_xy[1])

        def _route_len_cm(sx: float, sy: float, tx: float, ty: float, snap: float = 120.0) -> float:
            try:
                poly = self.city_map.route_xy_to_xy(sx, sy, tx, ty, snap_cm=snap)
                if len(poly) >= 2:
                    L = 0.0
                    for i in range(len(poly) - 1):
                        x0, y0 = poly[i]; x1, y1 = poly[i + 1]
                        L += math.hypot(x1 - x0, y1 - y0)
                    return float(L)
            except Exception:
                pass
            return float("inf")

        anchors: List[Tuple[float, float]] = [(float(x), float(y)) for (x, y) in (active_seeds_xy or [])]
        if not anchors:
            anchors = [(ax, ay)]

        K = 6
        SNAP = 120.0

        def _mindist_to_anchors(px: float, py: float) -> float:
            cand = sorted(anchors, key=lambda q: (q[0]-px)**2 + (q[1]-py)**2)[:K]
            best = float("inf")
            for (tx, ty) in cand:
                if math.hypot(tx - px, ty - py) >= best:
                    continue
                L = _route_len_cm(px, py, tx, ty, snap=SNAP)
                if L < best:
                    best = L
            if not math.isfinite(best):
                L = _route_len_cm(px, py, ax, ay, snap=SNAP)
                return L if math.isfinite(L) else math.hypot(px-ax, py-ay)
            return best

        if self.pickup_node is not None:
            px, py = float(self.pickup_node.position.x), float(self.pickup_node.position.y)
        else:
            px, py = float(self.pickup_address.x), float(self.pickup_address.y)

        if self.dropoff_node is not None:
            dx, dy = float(self.dropoff_node.position.x), float(self.dropoff_node.position.y)
        else:
            dx, dy = float(self.delivery_address.x), float(self.delivery_address.y)

        d_pu = _mindist_to_anchors(px, py)
        d_do = _mindist_to_anchors(dx, dy)
        net_dist_cm = 0.5 * (d_pu + d_do)
        return float(net_dist_cm), float(self.earning_mult), float(self.time_mult)


# =========================== OrderManager ===========================

class OrderManager:
    """
    Fixed-size order pool.
    - Sample pickup from restaurants and dropoff from buildings in world_nodes.
    - Use provided JSON menu (no internal defaults).
    - Each manager has its own ID counter and deterministic RNG derived from (seed, order_index).
    """

    def __init__(
        self,
        capacity: int = 10,
        menu: Optional[List[Dict[str, Any]]] = None,
        clock: Optional[VirtualClock] = None,
        special_notes_map: Optional[Dict[str, List[str]]] = None,
        note_prob: float = 0.2,
        seed: Optional[int] = None,
    ):
        assert capacity > 0
        self.capacity = int(capacity)
        self._orders: List[Order] = []
        self._menu: List[Dict[str, Any]] = list(menu) if menu is not None else []
        self._clock = clock if clock is not None else VirtualClock()
        self._lock: RLock = RLock()
        self._special_notes_map: Dict[str, List[str]] = dict(special_notes_map or {})
        self._note_prob: float = float(note_prob)

        self._seed: Optional[int] = int(seed) if seed is not None else None
        self._id_counter: int = 0
        self._id_lock: Lock = Lock()
        

    # ---------- Basic access ----------
    def __len__(self):
        with self._lock:
            return len(self._orders)

    def __iter__(self):
        with self._lock:
            return iter(list(self._orders))

    def list_orders(self) -> List[Order]:
        with self._lock:
            return list(self._orders)

    def get(self, order_id: int) -> Optional[Order]:
        with self._lock:
            for o in self._orders:
                if o.id == order_id:
                    return o
            return None

    # ---------- World node filters ----------
    @staticmethod
    def _is_restaurant(node: Dict[str, Any]) -> bool:
        props = node.get("properties", {})
        poi = str(props.get("poi_type") or props.get("type") or "").lower()
        return poi == "restaurant"

    @staticmethod
    def _is_building(node: Dict[str, Any]) -> bool:
        if str(node.get("instance_name", "")).startswith("BP_Building"):
            return True
        props = node.get("properties", {})
        t = str(props.get("type") or "").lower()
        return t == "building"

    @staticmethod
    def _xy_from_props(node: Dict[str, Any]) -> Tuple[float, float]:
        props = node.get("properties", {})
        loc = props.get("location", {})
        return float(loc.get("x", 0.0)), float(loc.get("y", 0.0))

    @staticmethod
    def _collect_candidates(world_nodes: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        if not world_nodes:
            raise RuntimeError("world_nodes is empty")
        pickups = [n for n in world_nodes if OrderManager._is_restaurant(n)]
        dropoffs = [n for n in world_nodes if OrderManager._is_building(n)]
        if not pickups or not dropoffs:
            raise RuntimeError("No valid pickup or dropoff nodes found")
        return pickups, dropoffs

    # ---------- Local ID & RNG ----------
    def _next_id(self) -> int:
        with self._id_lock:
            i = self._id_counter
            self._id_counter += 1
            return i

    @staticmethod
    def _rng_for(seed: Optional[int], index: int) -> random.Random:
        # 64-bit mix; deterministic across processes
        base = seed if seed is not None else 0x13A5B6C7D8E9F123
        key = (base ^ ((index + 0x9E3779B97F4A7C15) * 0xBF58476D1CE4E5B9)) & ((1 << 64) - 1)
        return random.Random(key)

    # ---------- Menu items ----------
    def _spawn_items_with_rng(self, rng: random.Random, min_n: int = 1, max_n: int = 4) -> List[FoodItem]:
        if not isinstance(self._menu, list) or len(self._menu) == 0:
            raise ValueError("Menu is empty. Provide a non-empty JSON menu.")
        n = rng.randint(min_n, max_n)
        picks = rng.sample(self._menu, k=min(n, len(self._menu)))
        items: List[FoodItem] = []
        for t in picks:
            if not isinstance(t, dict):
                raise TypeError("Menu must be a list of JSON dicts")
            name = str(t.get("name", "Item"))
            category = str(t.get("category", "") or "")
            odor = str(t.get("odor", ""))
            motion_sensitive = bool(t.get("motion_sensitive", False))
            nonthermal_time_sensitive = bool(t.get("nonthermal_time_sensitive", False))
            prep_time_s = int(t.get("prep_time_s", 0))
            serving_temp_c = float(t.get("serving_temp_c", FoodItem.serving_temp_c))
            safe_min_c     = float(t.get("safe_min_c", FoodItem.safe_min_c))
            safe_max_c     = float(t.get("safe_max_c", FoodItem.safe_max_c))
            heat_capacity  = float(t.get("heat_capacity", FoodItem.heat_capacity))
            odor_contamination = 1.0 if odor == "strong" else 0.0

            items.append(FoodItem(
                name=name,
                category=category,
                odor=odor,
                odor_contamination=odor_contamination,
                motion_sensitive=motion_sensitive,
                nonthermal_time_sensitive=nonthermal_time_sensitive,
                prep_time_s=prep_time_s,
                serving_temp_c=serving_temp_c,
                safe_min_c=safe_min_c,
                safe_max_c=safe_max_c,
                heat_capacity=heat_capacity,
                temp_c=float("nan"),
            ))
        return items

    # ---------- Spawn ONE order ----------
    def _spawn_one_order(self, city_map: Any, world_nodes: List[Dict[str, Any]], _ue=None) -> Order:
        pickups, dropoffs = self._collect_candidates(world_nodes)

        new_id = self._next_id()
        local_rng = self._rng_for(self._seed, new_id)

        pu_node = local_rng.choice(pickups)
        do_node = local_rng.choice(dropoffs)
        px, py = self._xy_from_props(pu_node)
        dx, dy = self._xy_from_props(do_node)

        items = self._spawn_items_with_rng(local_rng, 1, 4)

        order = Order(
            city_map=city_map,
            pickup_address=Vector(px, py),
            delivery_address=Vector(dx, dy),
            items=items
        )
        order.id = new_id
        order.rng = local_rng
        order._show_temperature = getattr(self, '_show_temperature', True)
        order._disable_prep_time = getattr(self, '_disable_prep_time', False)
        order._disable_jitter = getattr(self, '_disable_jitter', False)
        order.compute()

        longest = max((int(getattr(it, "prep_time_s", 0) or 0) for it in items), default=0)
        order.prep_longest_s = 0.0 if getattr(order, '_disable_prep_time', False) else float(longest)

        min_limit = order.prep_longest_s + 60.0
        if order.time_limit_s < min_limit:
            order.time_limit_s = min_limit

        if self._special_notes_map and local_rng.random() < self._note_prob:
            phrase, methods = local_rng.choice(list(self._special_notes_map.items()))
            order.special_note = phrase
            order.allowed_delivery_methods = list(methods or [])
        else:
            order.special_note = ""
            order.allowed_delivery_methods = []

        if 'hand_to_customer' in (order.allowed_delivery_methods or []):
            handoff_address = order._sample_handoff_position()
            if handoff_address is not None:
                order.handoff_address = handoff_address
            else:
                try:
                    order.allowed_delivery_methods.remove('hand_to_customer')
                except ValueError:
                    pass

        if 'hand_to_customer' in (order.allowed_delivery_methods or []) and _ue:
            _ue.spawn_customer(order.id, order.handoff_address.x, order.handoff_address.y)

        return order

    # ---------- Fill to capacity ----------
    def fill_pool(self, city_map: Any, world_nodes: List[Dict[str, Any]], _ue=None):
        self._city_map = city_map        
        self._world_nodes = world_nodes  
        with self._lock:
            while len(self._orders) < self.capacity:
                self._orders.append(self._spawn_one_order(city_map, world_nodes, _ue))

    # ---------- Complete & backfill ----------
    def complete_order(self, order_id: int, city_map: Any, world_nodes: List[Dict[str, Any]], _ue=None):
        with self._lock:
            target = None
            for o in self._orders:
                if o.id == order_id:
                    target = o
                    break
            if target is None:
                return
            with target._lock:
                target.has_delivered = True
            self._orders = [o for o in self._orders if o.id != order_id]
            self.fill_pool(city_map, world_nodes, _ue)

    def remove_order(self, order_ids: Union[int, Iterable[int]], city_map: Any, world_nodes: List[Dict[str, Any]], _ue=None):
        if isinstance(order_ids, int):
            ids = {int(order_ids)}
        else:
            ids = {int(i) for i in order_ids}
        if not ids:
            return
        with self._lock:
            self._orders = [o for o in self._orders if int(o.id) not in ids]
            self.fill_pool(city_map, world_nodes, _ue)

    # ---------- Accept (no removal here) ----------
    def accept_order(self, order_ids: Union[int, Iterable[int]]):
        def _accept_one(oid: int) -> bool:
            with self._lock:
                o = self.get(int(oid))
                if o is None:
                    return False
                with o._lock:
                    if getattr(o, "is_accepted", False):
                        return False
                    o.is_accepted = True
                    now = float(self._clock.now_sim() if hasattr(self, "_clock") and self._clock else time.time())
                    o.sim_started_s = now
                    o.sim_elapsed_active_s = 0.0
                    o.sim_delivered_s = None
                    longest = 0.0 if getattr(o, '_disable_prep_time', False) else float(getattr(o, "prep_longest_s", 0.0) or 0.0)
                    o.prep_started_sim = now
                    o.prep_ready_sim = now + longest
                    return True

        if isinstance(order_ids, int):
            return _accept_one(int(order_ids))

        accepted_ids: List[int] = []
        failed_ids: List[int] = []
        seen = set()

        for oid in order_ids:
            i = int(oid)
            if i in seen:
                continue
            seen.add(i)
            if _accept_one(i):
                accepted_ids.append(i)
            else:
                failed_ids.append(i)

        return accepted_ids, failed_ids

    def recompute_all(self):
        with self._lock:
            snapshot = list(self._orders)
        for o in snapshot:
            with o._lock:
                o.compute()

    # ---------- Text dump ----------
    def orders_text(self) -> str:
        with self._lock:
            snapshot = list(self._orders)
        if not snapshot:
            return "No orders."
        lines: List[str] = []
        lines.append(f"Orders ({len(snapshot)}):")
        lines.append("These are open orders available for you to accept:")
        for o in snapshot:
            lines.append(o.to_text())
            lines.append("-" * 60)
        return "\n".join(lines)

    # ---------- MapViewer hints ----------
    def to_active_order_hints(self) -> List[Dict[str, Any]]:
        with self._lock:
            snapshot = list(self._orders)
        hints: List[Dict[str, Any]] = []
        for o in snapshot:
            h = o.to_active_order_hint()
            h["pickup_road"] = o.pickup_road_name
            h["dropoff_road"] = o.dropoff_road_name
            hints.append(h)
        return hints

    def status_text(self) -> str:
        return "Active"

    # ---------- Ranking / scoring (unchanged) ----------
    def rank_orders_for_agent(
        self,
        agent_xy: Tuple[float, float],
        active_seeds_xy: List[Tuple[float, float]],
        *,
        include_aux: bool = False,
        weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        score_high: float = 5.0,
        score_low: float = 1.0,
    ) -> List[Dict[str, Any]]:
        with self._lock:
            snapshot = list(self._orders)

        rows: List[Dict[str, Any]] = []
        for o in snapshot:
            try:
                dist_cm, earn_m, time_m = o.score_for_agent(agent_xy, active_seeds_xy, include_aux=include_aux)
            except Exception:
                dist_cm, earn_m, time_m = float("inf"), float("-inf"), float("-inf")
            rows.append({
                "id": o.id,
                "dist_cm": float(dist_cm),
                "earn_mult": float(earn_m),
                "time_mult": float(time_m),
            })
        if not rows:
            return []

        n = len(rows)

        rows_sorted = sorted(rows, key=lambda r: (r["dist_cm"], r["id"]))
        dist_rank = {r["id"]: i+1 for i, r in enumerate(rows_sorted)}

        rows_sorted = sorted(rows, key=lambda r: (-r["earn_mult"], r["id"]))
        earn_rank = {r["id"]: i+1 for i, r in enumerate(rows_sorted)}

        rows_sorted = sorted(rows, key=lambda r: (-r["time_mult"], r["id"]))
        time_rank = {r["id"]: i+1 for i, r in enumerate(rows_sorted)}

        wd, we, wt = weights
        wsum = max(1e-9, wd + we + wt)

        def rank_to_unit(rank: int) -> float:
            return 1.0 if n == 1 else (1.0 - (rank - 1) / (n - 1))

        items: List[Dict[str, Any]] = []
        for r in rows:
            s_d = rank_to_unit(dist_rank[r["id"]])
            s_e = rank_to_unit(earn_rank[r["id"]])
            s_t = rank_to_unit(time_rank[r["id"]])
            unit = (wd*s_d + we*s_e + wt*s_t) / wsum
            rel = score_low + (score_high - score_low) * unit
            items.append({**r, "rel_score": float(rel)})

        items.sort(key=lambda x: x["rel_score"], reverse=True)
        for i, it in enumerate(items):
            it["rank"] = i + 1
        return items

    def relative_scores_for(
        self,
        agent_xy: Tuple[float, float],
        active_seeds_xy: List[Tuple[float, float]],
        order_ids: Iterable[int],
        *,
        include_aux: bool = False,
        weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        score_high: float = 5.0,
        score_low: float = 1.0,
    ) -> List[float]:
        board = self.rank_orders_for_agent(
            agent_xy, active_seeds_xy,
            include_aux=include_aux,
            weights=weights,
            score_high=score_high,
            score_low=score_low
        )
        s = {it["id"]: float(it["rel_score"]) for it in board}
        return [s.get(int(oid), float("nan")) for oid in order_ids]