#!/usr/bin/env python3
"""
Human-in-the-loop DeliveryBench validator.

This is a Tkinter UI for driving the current VAGEN DeliveryBench text
environment with exactly the same action strings an agent would emit.

Usage:
    python3 VAGEN/tools/deliverybench_hil_tk.py --map-name medium-city-22 --seed 42

Headless smoke check:
    python3 VAGEN/tools/deliverybench_hil_tk.py --smoke
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import tkinter as tk
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from datetime import datetime
from PIL import Image, ImageTk


THIS_FILE = Path(__file__).resolve()
VAGEN_ROOT = THIS_FILE.parents[1]
DEFAULT_BASE_DIR = VAGEN_ROOT / "vagen" / "envs" / "deliverybench"

if str(VAGEN_ROOT) not in sys.path:
    sys.path.insert(0, str(VAGEN_ROOT))


@dataclass(frozen=True)
class ActionField:
    key: str
    label: str
    default: str = ""
    choices: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ActionSpec:
    name: str
    fields: Tuple[ActionField, ...]
    builder: Callable[[Dict[str, str]], str]
    hint: str = ""


def _clean(text: str) -> str:
    return str(text or "").strip()


def _strip_m(text: str) -> str:
    value = _clean(text)
    if value.lower().endswith("m"):
        value = value[:-1].strip()
    return value


def _quote(text: str) -> str:
    return json.dumps(_clean(text))


def _list_or_scalar(text: str) -> str:
    value = _clean(text)
    if value.startswith("[") or value.startswith("("):
        return value
    if "," in value:
        return "[" + value + "]"
    return value


def _orders_arg(text: str) -> str:
    value = _clean(text)
    if not value:
        return ""
    if value.startswith("[") or value.startswith("("):
        return value
    return "[" + value + "]"


def _jsonish(text: str, default: str = "{}") -> str:
    return _clean(text) or default


def _build_no_args(name: str) -> Callable[[Dict[str, str]], str]:
    return lambda _values: f"{name}()"


def _build_move(values: Dict[str, str]) -> str:
    x_m = _strip_m(values.get("x_m", ""))
    y_m = _strip_m(values.get("y_m", ""))
    pace = _clean(values.get("pace", ""))
    if pace and pace != "normal":
        return f'MOVE({x_m}m, {y_m}m, pace="{pace}")'
    return f"MOVE({x_m}m, {y_m}m)"


def _build_accept(values: Dict[str, str]) -> str:
    return f"ACCEPT_ORDER({_list_or_scalar(values.get('order_ids', ''))})"


def _build_pickup(values: Dict[str, str]) -> str:
    order_ids = _orders_arg(values.get("order_ids", ""))
    return f"PICKUP(orders={order_ids})" if order_ids else "PICKUP()"


def _build_place_food(values: Dict[str, str]) -> str:
    return f"PLACE_FOOD_IN_BAG(bag_cmd={_quote(values.get('bag_cmd', ''))})"


def _build_charge(values: Dict[str, str]) -> str:
    return f"CHARGE(target_pct={_clean(values.get('target_pct', '100'))})"


def _build_wait(values: Dict[str, str]) -> str:
    raw = _clean(values.get("minutes_or_charge_done", ""))
    if raw.lower() in {"charge_done", '"charge_done"', "'charge_done'"}:
        return 'WAIT("charge_done")'
    return f"WAIT(minutes={raw})"


def _build_rest(values: Dict[str, str]) -> str:
    return f"REST(target_pct={_clean(values.get('target_pct', '100'))})"


def _build_buy(values: Dict[str, str]) -> str:
    item_id = _clean(values.get("item_id", ""))
    qty = _clean(values.get("qty", "1"))
    return f"BUY(item_id={_quote(item_id)}, qty={qty})"


def _build_use_pack(name: str) -> Callable[[Dict[str, str]], str]:
    def _inner(values: Dict[str, str]) -> str:
        comp = _clean(values.get("comp", "A")).upper()
        return f'{name}(comp="{comp}")'

    return _inner


def _build_switch(values: Dict[str, str]) -> str:
    return f'SWITCH(to="{_clean(values.get("to", "walk"))}")'


def _build_post_help(values: Dict[str, str]) -> str:
    kind = _clean(values.get("kind", "HELP_PICKUP"))
    bounty = _clean(values.get("bounty", "5.0"))
    ttl_s = _clean(values.get("ttl_s", "600"))
    payload = _jsonish(values.get("payload", "{}"))
    return f'POST_HELP(kind="{kind}", bounty={bounty}, ttl_s={ttl_s}, payload={payload})'


def _build_accept_help(values: Dict[str, str]) -> str:
    return f"ACCEPT_HELP(req_id={_clean(values.get('req_id', ''))})"


def _build_edit_help(values: Dict[str, str]) -> str:
    req_id = _clean(values.get("req_id", ""))
    bounty = _clean(values.get("new_bounty", ""))
    ttl_min = _clean(values.get("new_ttl_min", ""))
    parts = [f"req_id={req_id}"]
    if bounty:
        parts.append(f"new_bounty={bounty}")
    if ttl_min:
        parts.append(f"new_ttl_min={ttl_min}")
    return "EDIT_HELP(" + ", ".join(parts) + ")"


def _build_place_temp_box(values: Dict[str, str]) -> str:
    req_id = _clean(values.get("req_id", ""))
    location = _clean(values.get("location", ""))
    content = _jsonish(values.get("content", "{}"))
    parts = [f"req_id={req_id}"]
    if location:
        parts.append(f"location={location}")
    parts.append(f"content={content}")
    return "PLACE_TEMP_BOX(" + ", ".join(parts) + ")"


def _build_help_id(name: str) -> Callable[[Dict[str, str]], str]:
    return lambda values: f"{name}(req_id={_clean(values.get('req_id', ''))})"


def _build_dropoff(values: Dict[str, str]) -> str:
    oid = _clean(values.get("oid", ""))
    method = _clean(values.get("method", "leave_at_door"))
    return f'DROP_OFF(oid={oid}, method="{method}")'


def _build_say(values: Dict[str, str]) -> str:
    text = _clean(values.get("text", ""))
    to = _clean(values.get("to", ""))
    if to:
        return f"SAY(to={_quote(to)}, text={_quote(text)})"
    return f"SAY({_quote(text)})"


def _build_board_bus(values: Dict[str, str]) -> str:
    bus_id = _clean(values.get("bus_id", ""))
    target_stop_id = _clean(values.get("target_stop_id", ""))
    return f"BOARD_BUS(bus_id={_quote(bus_id)}, target_stop_id={_quote(target_stop_id)})"


def _build_turn(values: Dict[str, str]) -> str:
    angle = _clean(values.get("angle", "60"))
    direction = _clean(values.get("direction", "left"))
    return f'TURN_AROUND(angle={angle}, direction="{direction}")'


ACTION_SPECS: Dict[str, ActionSpec] = {
    "VIEW_ORDERS": ActionSpec("VIEW_ORDERS", (), _build_no_args("VIEW_ORDERS"), "Show the current open order pool."),
    "VIEW_BAG": ActionSpec("VIEW_BAG", (), _build_no_args("VIEW_BAG"), "Inspect the insulated bag."),
    "ACCEPT_ORDER": ActionSpec(
        "ACCEPT_ORDER",
        (ActionField("order_ids", "order id(s)", ""),),
        _build_accept,
        "Use one id, a comma list, or [1, 2].",
    ),
    "MOVE": ActionSpec(
        "MOVE",
        (
            ActionField("x_m", "x meters", ""),
            ActionField("y_m", "y meters", ""),
            ActionField("pace", "pace", "normal", ("normal", "accel", "decel")),
        ),
        _build_move,
        "Click the map or a POI row to fill x/y.",
    ),
    "PICKUP": ActionSpec(
        "PICKUP",
        (ActionField("order_ids", "order id(s)", ""),),
        _build_pickup,
        "Leave blank to pick up all ready orders at this location.",
    ),
    "PLACE_FOOD_IN_BAG": ActionSpec(
        "PLACE_FOOD_IN_BAG",
        (ActionField("bag_cmd", "bag command", "order 0: 1 -> A"),),
        _build_place_food,
        "Example: order 12: 1,2 -> A; 3 -> B",
    ),
    "CHARGE": ActionSpec("CHARGE", (ActionField("target_pct", "target pct", "100"),), _build_charge),
    "WAIT": ActionSpec(
        "WAIT",
        (ActionField("minutes_or_charge_done", "minutes / charge_done", "1"),),
        _build_wait,
        'Use a number of minutes or "charge_done".',
    ),
    "REST": ActionSpec("REST", (ActionField("target_pct", "target pct", "100"),), _build_rest),
    "BUY": ActionSpec(
        "BUY",
        (ActionField("item_id", "item id", "energy_drink"), ActionField("qty", "qty", "1")),
        _build_buy,
    ),
    "USE_BATTERY_PACK": ActionSpec("USE_BATTERY_PACK", (), _build_no_args("USE_BATTERY_PACK")),
    "USE_ENERGY_DRINK": ActionSpec("USE_ENERGY_DRINK", (), _build_no_args("USE_ENERGY_DRINK")),
    "USE_ICE_PACK": ActionSpec("USE_ICE_PACK", (ActionField("comp", "bag comp", "A"),), _build_use_pack("USE_ICE_PACK")),
    "USE_HEAT_PACK": ActionSpec("USE_HEAT_PACK", (ActionField("comp", "bag comp", "A"),), _build_use_pack("USE_HEAT_PACK")),
    "SWITCH": ActionSpec(
        "SWITCH",
        (ActionField("to", "mode", "walk", ("walk", "e-scooter", "car", "drag_scooter")),),
        _build_switch,
    ),
    "RENT_CAR": ActionSpec("RENT_CAR", (), _build_no_args("RENT_CAR")),
    "RETURN_CAR": ActionSpec("RETURN_CAR", (), _build_no_args("RETURN_CAR")),
    "VIEW_HELP_BOARD": ActionSpec("VIEW_HELP_BOARD", (), _build_no_args("VIEW_HELP_BOARD")),
    "POST_HELP": ActionSpec(
        "POST_HELP",
        (
            ActionField("kind", "kind", "HELP_PICKUP", ("HELP_PICKUP", "HELP_DELIVERY", "HELP_BUY", "HELP_CHARGE")),
            ActionField("bounty", "bounty", "5.0"),
            ActionField("ttl_s", "ttl seconds", "600"),
            ActionField("payload", "payload", "{}"),
        ),
        _build_post_help,
    ),
    "ACCEPT_HELP": ActionSpec("ACCEPT_HELP", (ActionField("req_id", "req id", ""),), _build_accept_help),
    "EDIT_HELP": ActionSpec(
        "EDIT_HELP",
        (
            ActionField("req_id", "req id", ""),
            ActionField("new_bounty", "new bounty", ""),
            ActionField("new_ttl_min", "new ttl min", ""),
        ),
        _build_edit_help,
    ),
    "PLACE_TEMP_BOX": ActionSpec(
        "PLACE_TEMP_BOX",
        (
            ActionField("req_id", "req id", ""),
            ActionField("location", "location tuple", ""),
            ActionField("content", "content", "{}"),
        ),
        _build_place_temp_box,
        "Click the map to fill MOVE; copy that coordinate tuple here if needed.",
    ),
    "TAKE_FROM_TEMP_BOX": ActionSpec("TAKE_FROM_TEMP_BOX", (ActionField("req_id", "req id", ""),), _build_help_id("TAKE_FROM_TEMP_BOX")),
    "REPORT_HELP_FINISHED": ActionSpec("REPORT_HELP_FINISHED", (ActionField("req_id", "req id", ""),), _build_help_id("REPORT_HELP_FINISHED")),
    "DROP_OFF": ActionSpec(
        "DROP_OFF",
        (
            ActionField("oid", "order id", ""),
            ActionField("method", "method", "leave_at_door", ("leave_at_door", "knock", "call", "hand_to_customer")),
        ),
        _build_dropoff,
    ),
    "SAY": ActionSpec("SAY", (ActionField("text", "text", ""), ActionField("to", "to", "")), _build_say),
    "BOARD_BUS": ActionSpec(
        "BOARD_BUS",
        (ActionField("bus_id", "bus id", "bus_1"), ActionField("target_stop_id", "target stop id", "")),
        _build_board_bus,
    ),
    "VIEW_BUS_SCHEDULE": ActionSpec("VIEW_BUS_SCHEDULE", (), _build_no_args("VIEW_BUS_SCHEDULE")),
    "TURN_AROUND": ActionSpec(
        "TURN_AROUND",
        (
            ActionField("angle", "angle", "60"),
            ActionField("direction", "direction", "left", ("left", "right")),
        ),
        _build_turn,
    ),
    "STEP_FORWARD": ActionSpec("STEP_FORWARD", (), _build_no_args("STEP_FORWARD")),
    "STEP_TO": ActionSpec(
        "STEP_TO",
        (ActionField("id", "waypoint id", "int_0"),),
        lambda values: f'STEP_TO(id="{_clean(values.get("id", ""))}")',
        "Move ONE adjacent waypoint. Use ids like int_5 or dock_12.",
    ),
    "NAVIGATE_WALK": ActionSpec(
        "NAVIGATE_WALK",
        (ActionField("target", "waypoint / address", ""),),
        lambda v: f'NAVIGATE_WALK(target={_quote(v.get("target", ""))})',
        "Query-only: walking directions + time/energy estimate. Does NOT move.",
    ),
    "NAVIGATE_ESCOOTER": ActionSpec(
        "NAVIGATE_ESCOOTER",
        (ActionField("target", "waypoint / address", ""),),
        lambda v: f'NAVIGATE_ESCOOTER(target={_quote(v.get("target", ""))})',
        "Query-only: e-scooter directions + time/energy/battery estimate.",
    ),
    "NAVIGATE_BUS": ActionSpec(
        "NAVIGATE_BUS",
        (
            ActionField("target", "waypoint / address", ""),
            ActionField("access_mode", "access", "auto", ("auto", "walk", "scooter")),
            ActionField("egress_mode", "egress", "auto", ("auto", "walk", "scooter")),
        ),
        lambda v: f'NAVIGATE_BUS(target={_quote(v.get("target", ""))}, access_mode="{_clean(v.get("access_mode", "auto"))}", egress_mode="{_clean(v.get("egress_mode", "auto"))}")',
        "Query-only: bus-assisted route.",
    ),
    "VISUAL_NAVIGATE_WALK": ActionSpec(
        "VISUAL_NAVIGATE_WALK",
        (ActionField("target", "waypoint / address", ""),),
        lambda v: f'VISUAL_NAVIGATE_WALK(target={_quote(v.get("target", ""))})',
        "Query-only: walking route as image.",
    ),
    "VISUAL_NAVIGATE_ESCOOTER": ActionSpec(
        "VISUAL_NAVIGATE_ESCOOTER",
        (ActionField("target", "waypoint / address", ""),),
        lambda v: f'VISUAL_NAVIGATE_ESCOOTER(target={_quote(v.get("target", ""))})',
        "Query-only: e-scooter route as image.",
    ),
    "VISUAL_NAVIGATE_BUS": ActionSpec(
        "VISUAL_NAVIGATE_BUS",
        (
            ActionField("target", "waypoint / address", ""),
            ActionField("access_mode", "access", "auto", ("auto", "walk", "scooter")),
            ActionField("egress_mode", "egress", "auto", ("auto", "walk", "scooter")),
        ),
        lambda v: f'VISUAL_NAVIGATE_BUS(target={_quote(v.get("target", ""))}, access_mode="{_clean(v.get("access_mode", "auto"))}", egress_mode="{_clean(v.get("egress_mode", "auto"))}")',
        "Query-only: top bus route as image.",
    ),
}


POI_COLORS = {
    "restaurant": "#e74c3c",
    "store": "#3498db",
    "rest_area": "#9b59b6",
    "hospital": "#e91e63",
    "car_rental": "#1abc9c",
    "charging_station": "#27ae60",
    "bus_station": "#f39c12",
    "customer": "#2ecc71",
    "building": "#7f8c8d",
}


def _fmt_float(value: Any, digits: int = 2) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return ""


def _fmt_m_from_cm(value: Any, digits: int = 2) -> str:
    try:
        return f"{float(value) / 100.0:.{digits}f}"
    except Exception:
        return ""


def _fmt_xy_m(x_cm: Any, y_cm: Any, digits: int = 2) -> str:
    return f"({_fmt_m_from_cm(x_cm, digits)}m, {_fmt_m_from_cm(y_cm, digits)}m)"


def _node_xy(node: Any) -> Optional[Tuple[float, float]]:
    if node is None:
        return None
    pos = getattr(node, "position", None)
    if pos is not None:
        try:
            return float(pos.x), float(pos.y)
        except Exception:
            return None
    try:
        return float(node["x"]), float(node["y"])
    except Exception:
        return None


def _order_node_xy(order: Any, attr_name: str) -> Optional[Tuple[float, float]]:
    xy = _node_xy(getattr(order, attr_name, None))
    if xy is not None:
        return xy
    fallback_attr = "pickup_address" if attr_name == "pickup_node" else "delivery_address"
    return _node_xy(getattr(order, fallback_attr, None))


def _safe_attr(obj: Any, name: str, default: Any = "") -> Any:
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


def _sim_time_text(seconds: Any) -> str:
    try:
        total = int(round(float(seconds)))
    except Exception:
        return "n/a"
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


def _discover_maps(base_dir: Path) -> List[str]:
    maps_dir = base_dir / "maps"
    if not maps_dir.exists():
        return []
    names = []
    for path in maps_dir.iterdir():
        if path.is_dir() and (path / "roads.json").exists() and (path / "progen_world_enriched.json").exists():
            names.append(path.name)
    return sorted(names)


def _split_observation_sections(text: str) -> List[Tuple[str, str]]:
    sections: List[Tuple[str, List[str]]] = []
    title = "observation"
    body: List[str] = []
    for line in (text or "").splitlines():
        if line.startswith("### "):
            if body or sections:
                sections.append((title, body))
            title = line[4:].strip() or "section"
            body = []
        else:
            body.append(line)
    sections.append((title, body))
    return [(name, "\n".join(lines).strip()) for name, lines in sections if name or lines]


def _short(text: Any, max_len: int = 90) -> str:
    value = str(text or "").replace("\n", " ").strip()
    if len(value) <= max_len:
        return value
    return value[: max_len - 1] + "..."


class DeliveryBenchHilApp:
    def __init__(self, root: tk.Tk, args: argparse.Namespace):
        self.root = root
        self.args = args
        self.base_dir = Path(args.base_dir).resolve()
        self.env: Any = None
        self.last_obs: Dict[str, Any] = {}
        self.last_info: Dict[str, Any] = {}
        self.last_reward: float = 0.0
        self.last_done: bool = False
        self.last_truncated: bool = False
        
        self.last_nav_text = ""
        self.history: List[Tuple[int, str, float, str]] = []

        self.map_markers: List[Dict[str, Any]] = []
        self.poi_records: Dict[str, Dict[str, Any]] = {}
        self.order_records: Dict[str, Any] = {}
        self.view_bounds: Optional[Tuple[float, float, float, float]] = None
        self._drag_start: Optional[Tuple[int, int, Tuple[float, float, float, float]]] = None
        self._dragged = False
        self._last_nav_target_id = None 

        self.action_vars: Dict[str, tk.StringVar] = {}
        self.agent_yaw = 0

        self.root.title("DeliveryBench Human Validator")
        self.root.geometry("1500x950")
        self.root.minsize(1100, 720)

        self._build_ui()
        self._populate_action("VIEW_ORDERS")
        self.reset_env()
        

    # ------------------------------------------------------------------
    # UI layout
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        self._build_topbar()
        self._build_main_area()
        self._build_action_bar()

    def _build_topbar(self) -> None:
        top = ttk.Frame(self.root, padding=(8, 6))
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(12, weight=1)

        ttk.Label(top, text="Map").grid(row=0, column=0, padx=(0, 4))
        self.map_var = tk.StringVar(value=self.args.map_name)
        maps = _discover_maps(self.base_dir)
        self.map_combo = ttk.Combobox(top, textvariable=self.map_var, values=maps, width=20, state="readonly")
        self.map_combo.grid(row=0, column=1, padx=(0, 10))

        ttk.Label(top, text="Seed").grid(row=0, column=2, padx=(0, 4))
        self.seed_var = tk.StringVar(value=str(self.args.seed))
        ttk.Entry(top, textvariable=self.seed_var, width=9).grid(row=0, column=3, padx=(0, 10))

        ttk.Label(top, text="Max steps").grid(row=0, column=4, padx=(0, 4))
        self.max_steps_var = tk.StringVar(value=str(self.args.max_steps))
        ttk.Entry(top, textvariable=self.max_steps_var, width=8).grid(row=0, column=5, padx=(0, 10))

        self.reset_btn = ttk.Button(top, text="Reset", command=self.reset_env)
        self.reset_btn.grid(row=0, column=6, padx=(0, 6))

        ttk.Button(top, text="Fit Map", command=self.fit_map).grid(row=0, column=7, padx=(0, 6))
        ttk.Button(top, text="Zoom +", command=lambda: self.zoom_map(0.8)).grid(row=0, column=8, padx=(0, 4))
        ttk.Button(top, text="Zoom -", command=lambda: self.zoom_map(1.25)).grid(row=0, column=9, padx=(0, 12))
        ttk.Button(top, text="Export Trajectory", command=self.export_trajectory).grid(row=0, column=10, padx=(0, 12))

        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(top, textvariable=self.status_var).grid(row=0, column=12, sticky="e")

    def _build_main_area(self) -> None:
        pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        pane.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 6))


        left = ttk.Frame(pane)
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)
        pane.add(left, weight=3)

        self.canvas = tk.Canvas(left, bg="#f7fbff", highlightthickness=1,
                                highlightbackground="#ccd4dd")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", lambda _event: self.draw_map())
        self.canvas.bind("<ButtonPress-1>", self._on_canvas_press)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Button-4>", lambda event: self._zoom_at_event(event, 0.8))
        self.canvas.bind("<Button-5>", lambda event: self._zoom_at_event(event, 1.25))

        self.map_help_var = tk.StringVar(
            value="[human aid — agent does not see this map] Click marker to fill MOVE; drag pans; wheel zooms."
        )
        ttk.Label(left, textvariable=self.map_help_var).grid(row=1, column=0, sticky="ew", pady=(4, 0))

    
        right_pane = ttk.PanedWindow(pane, orient=tk.VERTICAL)
        pane.add(right_pane, weight=2)

        obs_frame = ttk.LabelFrame(right_pane, text="Agent View (exactly what the agent sees)", padding=4)
        self.feedback_var = tk.StringVar(value="(no action executed yet)")
        self.feedback_label = ttk.Label(obs_frame, textvariable=self.feedback_var,
                                        font=("TkFixedFont", 10, "bold"),
                                        foreground="#7a3b00", wraplength=700,
                                        justify="left")
        self.feedback_label.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        obs_frame.rowconfigure(0, weight=0)
        obs_frame.rowconfigure(1, weight=1) 
        obs_frame.columnconfigure(0, weight=1)
        right_pane.add(obs_frame, weight=3)
        

        self.obs_text = ScrolledText(obs_frame, wrap=tk.WORD, font=("TkFixedFont", 10))
        self.obs_text.grid(row=1, column=0, sticky="nsew")
        self.obs_text.configure(state=tk.DISABLED)
        self.obs_text.tag_configure("header", font=("TkFixedFont", 10, "bold"),
                                    foreground="#1a4f8a", spacing1=8, spacing3=2)
        self.obs_text.tag_configure("feedback", foreground="#7a3b00",
                                    font=("TkFixedFont", 10, "bold"), spacing1=10)
        self.obs_text.tag_configure("error", foreground="#b00020")

        aux = ttk.Notebook(right_pane)
        right_pane.add(aux, weight=1)

        self.state_tab = ttk.Frame(aux, padding=6)
        self.poi_tab = ttk.Frame(aux, padding=6)
        self.history_tab = ttk.Frame(aux, padding=6)
        self.fpv_tab = ttk.Frame(aux, padding=6)
        self.nav_tab = ttk.Frame(aux, padding=6)
        aux.add(self.state_tab, text="Orders")
        aux.add(self.poi_tab, text="POIs")
        aux.add(self.history_tab, text="History")
        aux.add(self.fpv_tab, text="FPV")
        aux.add(self.nav_tab, text="Navigation")

        self._build_state_tab()
        self._build_poi_tab()
        self._build_history_tab()
        self._build_fpv_tab()
        self._build_navigation_tab()

    def _build_state_tab(self) -> None:
        self.state_tab.columnconfigure(0, weight=1)
        self.state_tab.rowconfigure(1, weight=1)

        quick_frame = ttk.LabelFrame(self.state_tab, text="Order Shortcuts", padding=6)
        quick_frame.grid(row=0, column=0, sticky="ew", pady=(6, 0))
        ttk.Button(quick_frame, text="Accept", command=self.fill_accept_selected_order).grid(row=0, column=0, padx=(0, 4))
        ttk.Button(quick_frame, text="Move Pickup", command=lambda: self.fill_order_endpoint("pickup")).grid(
            row=0, column=1, padx=(0, 4)
        )
        ttk.Button(quick_frame, text="Move Dropoff", command=lambda: self.fill_order_endpoint("dropoff")).grid(
            row=0, column=2, padx=(0, 4)
        )
        ttk.Button(quick_frame, text="Pickup", command=self.fill_pickup_selected_order).grid(row=0, column=3, padx=(0, 4))
        ttk.Button(quick_frame, text="Drop Off", command=self.fill_dropoff_selected_order).grid(row=0, column=4, padx=(0, 4))

        orders_frame = ttk.LabelFrame(self.state_tab, text="Orders", padding=6)
        orders_frame.grid(row=1, column=0, sticky="nsew", pady=(6, 0))
        orders_frame.rowconfigure(0, weight=1)
        orders_frame.columnconfigure(0, weight=1)
        columns = ("id", "status", "pickup", "dropoff", "eta", "pay")
        self.orders_tree = ttk.Treeview(orders_frame, columns=columns, show="headings", height=12, selectmode="browse")
        for col, text, width in [
            ("id", "ID", 45),
            ("status", "Status", 110),
            ("pickup", "Pickup", 145),
            ("dropoff", "Dropoff", 145),
            ("eta", "Time", 85),
            ("pay", "Pay", 70),
        ]:
            self.orders_tree.heading(col, text=text)
            self.orders_tree.column(col, width=width, anchor="w", stretch=col in {"pickup", "dropoff"})
        yscroll = ttk.Scrollbar(orders_frame, orient=tk.VERTICAL, command=self.orders_tree.yview)
        self.orders_tree.configure(yscrollcommand=yscroll.set)
        self.orders_tree.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        self.orders_tree.bind("<Double-1>", lambda _event: self.fill_accept_selected_order())

       
    def _build_poi_tab(self) -> None:
        self.poi_tab.rowconfigure(2, weight=1)
        self.poi_tab.columnconfigure(0, weight=1)

        ttk.Label(self.poi_tab, text="Filter").grid(row=0, column=0, sticky="w")
        self.poi_filter_var = tk.StringVar(value="")
        filter_entry = ttk.Entry(self.poi_tab, textvariable=self.poi_filter_var)
        filter_entry.grid(row=1, column=0, sticky="ew", pady=(2, 6))
        self.poi_filter_var.trace_add("write", lambda *_args: self.refresh_poi_table())

        columns = ("name", "type", "xy", "road")
        self.poi_tree = ttk.Treeview(self.poi_tab, columns=columns, show="headings", selectmode="browse")
        for col, text, width in [
            ("name", "Name", 135),
            ("type", "Type", 105),
            ("xy", "Move target", 120),
            ("road", "Road", 160),
        ]:
            self.poi_tree.heading(col, text=text)
            self.poi_tree.column(col, width=width, anchor="w", stretch=col == "road")
        yscroll = ttk.Scrollbar(self.poi_tab, orient=tk.VERTICAL, command=self.poi_tree.yview)
        self.poi_tree.configure(yscrollcommand=yscroll.set)
        self.poi_tree.grid(row=2, column=0, sticky="nsew")
        yscroll.grid(row=2, column=1, sticky="ns")
        self.poi_tree.bind("<Double-1>", lambda _event: self.fill_move_from_selected_poi())

        ttk.Button(self.poi_tab, text="Move To Selected POI", command=self.fill_move_from_selected_poi).grid(
            row=3, column=0, sticky="ew", pady=(6, 0)
        )

    def _build_obs_tab(self) -> None:
        self.obs_tab.rowconfigure(0, weight=1)
        self.obs_tab.columnconfigure(0, weight=1)
        self.obs_notebook = ttk.Notebook(self.obs_tab)
        self.obs_notebook.grid(row=0, column=0, sticky="nsew")

    def _build_history_tab(self) -> None:
        self.history_tab.rowconfigure(0, weight=1)
        self.history_tab.columnconfigure(0, weight=1)
        self.history_text = ScrolledText(self.history_tab, wrap=tk.WORD)
        self.history_text.grid(row=0, column=0, sticky="nsew")
        self.history_text.configure(state=tk.DISABLED)
    def _build_fpv_tab(self) -> None:
        self.fpv_tab.rowconfigure(0, weight=1)
        self.fpv_tab.rowconfigure(1, weight=1)
        self.fpv_tab.columnconfigure(0, weight=1)
        self.fpv_tab.columnconfigure(1, weight=1)
        
        self.fpv_status_var = tk.StringVar(value="Agent facing: N (yaw 0°)")
        status_frame = ttk.Frame(self.fpv_tab)
        status_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 4))
        ttk.Label(status_frame, textvariable=self.fpv_status_var, 
                font=("TkDefaultFont", 11, "bold")).pack()
        
        self.fpv_labels = {}
        self.fpv_frames = {}
        self.fpv_images = {}
        
        yaw_layout = [
            (1, 0, 90,  "North (yaw 90°)"),    
            (1, 1, 0,   "East (yaw 0°)"),     
            (2, 0, 180, "West (yaw 180°)"),    
            (2, 1, 270, "South (yaw 270°)"),  
        ]
        
        for row, col, yaw, name in yaw_layout:
            frame = ttk.LabelFrame(self.fpv_tab, text=name, padding=4)
            frame.grid(row=row, column=col, sticky="nsew", padx=2, pady=2)
            frame.rowconfigure(0, weight=1)
            frame.columnconfigure(0, weight=1)
            lbl = ttk.Label(frame, text="(no image)", anchor="center")
            lbl.grid(row=0, column=0, sticky="nsew")
            self.fpv_labels[yaw] = lbl
            self.fpv_frames[yaw] = frame
        
        self.fpv_tab.rowconfigure(1, weight=1)
        self.fpv_tab.rowconfigure(2, weight=1)
        
    def _build_navigation_tab(self) -> None:
        self.nav_tab.rowconfigure(0, weight=1)
        self.nav_tab.columnconfigure(0, weight=1)
        self.nav_text_widget = ScrolledText(self.nav_tab, wrap=tk.WORD,
                                            font=("TkFixedFont", 10))
        self.nav_text_widget.grid(row=0, column=0, sticky="nsew")
        self.nav_text_widget.configure(state=tk.DISABLED)
        
    def refresh_navigation(self) -> None:
        """Persist the last NAVIGATE result so it survives subsequent steps."""
        dm = self._dm()
        if dm is None:
            self._set_text(self.nav_text_widget, "(no environment)")
            return
        
        eph = getattr(dm, 'vlm_ephemeral', None) or {}
        for key in ('navigation_route', 'navigation_walk', 'navigation_escooter',
                    'navigation_bus', 'visual_navigation_walk',
                    'visual_navigation_escooter', 'visual_navigation_bus'):
            if key in eph:
                self.last_nav_text = f"=== {key} ===\n{eph[key]}\n\n(captured after last action)"
                break  
        display = self.last_nav_text or "(No NAVIGATE results yet. Use NAVIGATE_WALK(target=\"dock_X\") to plan.)"
        self._set_text(self.nav_text_widget, display)

    def refresh_fpv(self) -> None:
        dm = self._dm()
        if dm is None or self.env is None:
            return
        city_map = getattr(self.env, "map", None)
        if city_map is None or not hasattr(city_map, "nearest_waypoint"):
            return
        try:
            wp = city_map.nearest_waypoint(dm.x, dm.y)
        except Exception:
            wp = None
        if wp is None:
            for lbl in self.fpv_labels.values():
                lbl.configure(image='', text="(no waypoint)")
            return
        
        wp_id = getattr(wp, 'waypoint_id', None) or getattr(wp, 'id', None) or ''
        
        # int_25 -> int_025 (folder uses 3-digit zero-padding)
        if "_" in wp_id:
            prefix, num = wp_id.split("_", 1)
            try:
                folder = f"{prefix}_{int(num):03d}"
            except ValueError:
                folder = wp_id
        else:
            folder = wp_id
        
        map_name = self.map_var.get()
        base = self.base_dir / "fpv_images" / "deliverybench_fpv" / map_name / "images" / folder
        
        if not base.exists():
            for lbl in self.fpv_labels.values():
                lbl.configure(image='', text=f"(no images for {folder})")
            return
        
        target_size = (320, 200)
        for yaw in [0, 90, 180, 270]:
            img_path = base / f"yaw_{yaw:03d}.png"
            if img_path.exists():
                try:
                    img = Image.open(img_path)
                    img.thumbnail(target_size)
                    photo = ImageTk.PhotoImage(img)
                    self.fpv_images[yaw] = photo  
                    self.fpv_labels[yaw].configure(image=photo, text='')
                except Exception as exc:
                    self.fpv_labels[yaw].configure(image='', text=f"(error: {exc})")
            else:
                self.fpv_labels[yaw].configure(image='', text='(no image)')
        direction_names = {90: "N", 0: "E", 270: "S", 180: "W"}  
        self.fpv_status_var.set(f"Agent facing: {direction_names[self.agent_yaw]} (yaw {self.agent_yaw}°)")

        for yaw, frame in self.fpv_frames.items():
            if yaw == self.agent_yaw:
                frame.configure(text=f"[*] {direction_names[yaw]} (yaw {yaw}°) -- CURRENT VIEW")
            else:
                frame.configure(text=f"{direction_names[yaw]} (yaw {yaw}°)")
                
    def _build_action_bar(self) -> None:
        frame = ttk.LabelFrame(self.root, text="Action Builder", padding=8)
        frame.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 8))
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(3, weight=3)

        ttk.Label(frame, text="Action").grid(row=0, column=0, sticky="w", padx=(0, 4))
        self.action_name_var = tk.StringVar(value="VIEW_ORDERS")
        self.action_combo = ttk.Combobox(
            frame,
            textvariable=self.action_name_var,
            values=list(ACTION_SPECS.keys()),
            state="readonly",
            width=24,
        )
        self.action_combo.grid(row=0, column=1, sticky="w", padx=(0, 10))
        self.action_combo.bind("<<ComboboxSelected>>", lambda _event: self._populate_action(self.action_name_var.get()))

        self.param_frame = ttk.Frame(frame)
        self.param_frame.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(6, 4))

        ttk.Label(frame, text="Action text").grid(row=2, column=0, sticky="w", padx=(0, 4))
        self.action_text_var = tk.StringVar(value="VIEW_ORDERS()")
        self.action_entry = ttk.Entry(frame, textvariable=self.action_text_var)
        self.action_entry.grid(row=2, column=1, columnspan=3, sticky="ew", padx=(0, 8))
        ttk.Button(frame, text="Execute", command=self.execute_action).grid(row=2, column=4, padx=(0, 4))
        ttk.Button(frame, text="Clear", command=lambda: self.action_text_var.set("")).grid(row=2, column=5)
        self.root.bind("<Return>", lambda _e: self.execute_action())
        
        self.action_hint_var = tk.StringVar(value="")
        ttk.Label(frame, textvariable=self.action_hint_var).grid(row=3, column=0, columnspan=6, sticky="w", pady=(4, 0))

    # ------------------------------------------------------------------
    # Environment lifecycle
    # ------------------------------------------------------------------
    def _make_env(self) -> Any:
        from vagen.envs.deliverybench.vlm_delivery.gym_like_interface import DeliveryBenchGymEnvText

        return DeliveryBenchGymEnvText(
            base_dir=str(self.base_dir),
            map_name=self.map_var.get(),
            time_scale=float(self.args.time_scale),
            max_steps=int(self.max_steps_var.get() or self.args.max_steps),
            enable_map_images=False,
            map_renderer="pil",
            enable_vlm=False,
        )

    def reset_env(self) -> None:
        try:
            if self.env is not None:
                self.env.close()
            self.env = self._make_env()
            seed = int(self.seed_var.get() or "0")
            self.last_obs, self.last_info = self.env.reset(seed=seed)
            self.last_reward = 0.0
            self.last_done = False
            self.last_truncated = False
            self.history.clear()
            self.status_var.set(f"Reset {self.map_var.get()} with seed={seed}.")
            self._load_poi_records()
            self.fit_map()
            self.feedback_var.set("(environment reset)")
            self.refresh_all()
        except Exception as exc:
            self.status_var.set(f"Reset failed: {exc}")
            traceback.print_exc()
            messagebox.showerror("Reset failed", str(exc))

    def export_trajectory(self) -> None:
        """Save current episode history as JSON."""
        
        
        if not self.history:
            messagebox.showinfo("Nothing to export", "No actions in history yet.")
            return
        
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = VAGEN_ROOT / f"trajectory_{self.map_var.get()}_seed{self.seed_var.get()}_{ts}.json"
        
        dm = self._dm()
        data = {
            "metadata": {
                "map": self.map_var.get(),
                "seed": int(self.seed_var.get() or 0),
                "exported_at": ts,
                "total_steps": len(self.history),
                "final_reward": sum(r for _, _, r, _ in self.history),
                "agent_final_position": {
                    "x_cm": float(getattr(dm, "x", 0.0)) if dm else None,
                    "y_cm": float(getattr(dm, "y", 0.0)) if dm else None,
                } if dm else None,
                "completed_orders": len(getattr(dm, "completed_orders", []) or []) if dm else 0,
            },
            "steps": [
                {
                    "step": step,
                    "action": action,
                    "reward": reward,
                    "error": err or None,
                }
                for step, action, reward, err in self.history
            ],
        }
        
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        self.status_var.set(f"Trajectory saved: {out_path.name}")
        messagebox.showinfo("Exported", f"Saved {len(self.history)} steps to:\n{out_path}")
        
    def execute_action(self) -> None:
        if self.env is None:
            self.reset_env()
        action = _clean(self.action_text_var.get())
        if not action:
            messagebox.showwarning("Empty action", "Enter an action before executing.")
            return

        dm_before = self._dm()
        prev_x = float(getattr(dm_before, 'x', 0.0)) if dm_before else 0.0
        prev_y = float(getattr(dm_before, 'y', 0.0)) if dm_before else 0.0
        try:
            obs, reward, terminated, truncated, info = self.env.step(action)
            import re
            nav_match = re.match(r'NAVIGATE\w*\s*\(.*target\s*=\s*["\']([^"\']+)["\']', action, re.IGNORECASE)
            if nav_match:
                self._last_nav_target_id = nav_match.group(1)
            self.last_obs = obs
            self.last_reward = float(reward)
            self.last_done = bool(terminated)
            self.last_truncated = bool(truncated)
            self.last_info = dict(info or {})
            dm_after = self._dm()
            if dm_after is not None:
                new_x = float(dm_after.x)
                new_y = float(dm_after.y)
            
                action_upper = action.upper().strip()
                if action_upper.startswith(("STEP_TO", "MOVE")) and (abs(new_x - prev_x) > 100 or abs(new_y - prev_y) > 100):
                    self.agent_yaw = self._compute_yaw_from_move(prev_x, prev_y, new_x, new_y)

            step_no = int(self.last_info.get("elapsed_steps", len(self.history) + 1))
            err = _clean(self.last_info.get("error", ""))
            self.history.append((step_no, action, self.last_reward, err))
            status = f"Step {step_no}: reward={self.last_reward:.3f}"
            if err:
                status += f" | error: {err}"
            if terminated or truncated:
                status += " | episode finished"
            self.status_var.set(status)
            fb = f"step {step_no} | {action} | reward {self.last_reward:+.3f}"
            if terminated or truncated:
                fb += " | EPISODE FINISHED"
            if err:
                fb += f"\nERROR: {err}"
            self.feedback_var.set(fb)
            self.refresh_all()
        except Exception as exc:
            self.status_var.set(f"Action failed: {exc}")
            traceback.print_exc()
            messagebox.showerror("Action failed", str(exc))

    def refresh_all(self) -> None:
        self.refresh_agent_state()
        self.refresh_orders()
        self.refresh_poi_table()
        self.refresh_observation()
        self.refresh_history()
        self.draw_map()
        self._refresh_action_choices()
        self.refresh_fpv()
        self.refresh_navigation()

    # ------------------------------------------------------------------
    # Action builder
    # ------------------------------------------------------------------
    def _populate_action(self, name: str, values: Optional[Dict[str, str]] = None) -> None:
        spec = ACTION_SPECS.get(name)
        if spec is None:
            return
        self.action_name_var.set(name)
        for child in self.param_frame.winfo_children():
            child.destroy()
        self.action_vars.clear()

        if not spec.fields:
            ttk.Label(self.param_frame, text="No parameters.").grid(row=0, column=0, sticky="w")
        else:
            for idx, field in enumerate(spec.fields):
                ttk.Label(self.param_frame, text=field.label).grid(row=0, column=idx * 2, sticky="w", padx=(0, 4))
                var = tk.StringVar(value=(values or {}).get(field.key, field.default))
                self.action_vars[field.key] = var
                if field.choices:
                    widget = ttk.Combobox(self.param_frame, textvariable=var, values=field.choices, width=18)
                else:
                    widget = ttk.Entry(self.param_frame, textvariable=var, width=24)
                widget.grid(row=0, column=idx * 2 + 1, sticky="ew", padx=(0, 10))
                var.trace_add("write", lambda *_args: self.update_action_preview())
                widget.bind("<Return>", lambda _e: self.execute_action())  
                self.param_frame.columnconfigure(idx * 2 + 1, weight=1)
                
                
        self.action_hint_var.set(spec.hint)
        self.update_action_preview()

    def update_action_preview(self) -> None:
        spec = ACTION_SPECS.get(self.action_name_var.get())
        if spec is None:
            return
        values = {key: var.get() for key, var in self.action_vars.items()}
        try:
            self.action_text_var.set(spec.builder(values))
        except Exception as exc:
            self.action_hint_var.set(f"Could not build action yet: {exc}")

    def set_action(self, name: str, values: Optional[Dict[str, str]] = None) -> None:
        self._populate_action(name, values or {})

    def fill_move_to_xy(self, x_cm: float, y_cm: float, label: str = "") -> None:
        x_m = f"{float(x_cm) / 100.0:.2f}"
        y_m = f"{float(y_cm) / 100.0:.2f}"
        self.set_action("MOVE", {"x_m": x_m, "y_m": y_m, "pace": "normal"})
        tuple_text = f"({float(x_cm):.1f}, {float(y_cm):.1f})"
        self.map_help_var.set(f"Selected {label or 'map point'} at {x_m}m, {y_m}m. MOVE filled. Raw cm tuple: {tuple_text}")

    def fill_accept_selected_order(self) -> None:
        order = self._selected_order()
        if order is None:
            return
        self.set_action("ACCEPT_ORDER", {"order_ids": str(_safe_attr(order, "id", ""))})

    def fill_pickup_selected_order(self) -> None:
        order = self._selected_order()
        if order is None:
            return
        self.set_action("PICKUP", {"order_ids": str(_safe_attr(order, "id", ""))})

    def fill_dropoff_selected_order(self) -> None:
        order = self._selected_order()
        if order is None:
            return
        self.set_action("DROP_OFF", {"oid": str(_safe_attr(order, "id", "")), "method": "leave_at_door"})

    def fill_order_endpoint(self, endpoint: str) -> None:
        order = self._selected_order()
        if order is None:
            return
        attr = "pickup_node" if endpoint == "pickup" else "dropoff_node"
        xy = _order_node_xy(order, attr)
        if xy is None:
            return
        self.fill_move_to_xy(xy[0], xy[1], f"{endpoint} for order #{_safe_attr(order, 'id', '?')}")

    def fill_move_from_selected_poi(self) -> None:
        iid = self.poi_tree.focus()
        rec = self.poi_records.get(iid)
        if not rec:
            return
        self.fill_move_to_xy(rec["x"], rec["y"], rec["name"])

    # ------------------------------------------------------------------
    # State panels
    # ------------------------------------------------------------------
    def _dm(self) -> Any:
        if self.env is not None and getattr(self.env, "dms", None):
            return self.env.dms[0]
        return None

    def refresh_agent_state(self) -> None:
        dm = self._dm()
        if dm is None:
            return
        clock = getattr(dm, "clock", None)
        sim_now = clock.now_sim() if clock is not None and hasattr(clock, "now_sim") else None
        scooter = getattr(dm, "e_scooter", None)
        scooter_text = "none"
        if scooter is not None:
            scooter_text = f"{_fmt_float(getattr(scooter, 'battery_pct', ''), 1)}% battery"
        inventory = getattr(dm, "inventory", {}) or {}
        inv_text = ", ".join(f"{k}:{v}" for k, v in inventory.items()) if inventory else "empty"
        active = len(getattr(dm, "active_orders", []) or [])
        completed = len(getattr(dm, "completed_orders", []) or [])
        carrying = getattr(dm, "carrying", []) or []
       
        lines = [
            f"reward: {self.last_reward:.3f}",
            f"terminated: {self.last_done}",
            f"truncated: {self.last_truncated}",
        ]
        if self.last_info:
            lines.append("info: " + _short(self.last_info, 400))
        if getattr(dm, "vlm_errors", None):
            lines.append("\nerror memory:\n" + str(dm.vlm_errors))
        if getattr(dm, "vlm_ephemeral", None):
            lines.append("\nephemeral:")
            for key, value in (dm.vlm_ephemeral or {}).items():
                lines.append(f"[{key}]\n{value}")
        if getattr(dm, "vlm_last_actions", None):
            lines.append("\nlast successful actions:")
            lines.extend(f"- {x}" for x in list(dm.vlm_last_actions))
        
    def _orders(self) -> List[Any]:
        if self.env is None:
            return []
        om = getattr(self.env, "om", None)
        pool = []
        if om is not None:
            if hasattr(om, "list_orders"):
                pool = list(om.list_orders())
            else:
                pool = list(getattr(om, "_orders", []) or [])
        dm = self._dm()
        active = []
        if dm is not None:
            for attr in ("active_orders", "orders", "accepted_orders"):
                val = getattr(dm, attr, None)
                if val:
                    active = list(val)
                    break
        seen, out = set(), []
        for o in active + pool:          
            if id(o) in seen:
                continue
            seen.add(id(o))
            out.append(o)
        return out

    def refresh_orders(self) -> None:
        selected_id = None
        old = self.orders_tree.focus()
        if old:
            values = self.orders_tree.item(old, "values")
            selected_id = values[0] if values else None

        self.orders_tree.delete(*self.orders_tree.get_children())
        self.order_records.clear()
        for order in self._orders():
            oid = str(_safe_attr(order, "id", ""))
            status = self._order_status(order)
            pickup = self._endpoint_text(order, "pickup_node", "pickup_road_name")
            dropoff = self._endpoint_text(order, "dropoff_node", "dropoff_road_name")
            time_text = self._order_time_text(order)
            pay = f"${_fmt_float(_safe_attr(order, 'earnings', 0.0), 2)}"
            iid = self.orders_tree.insert("", "end", values=(oid, status, pickup, dropoff, time_text, pay))
            self.order_records[iid] = order
            if selected_id is not None and oid == selected_id:
                self.orders_tree.selection_set(iid)
                self.orders_tree.focus(iid)

    def _selected_order(self) -> Any:
        selected = self.orders_tree.selection()
        iid = selected[0] if selected else self.orders_tree.focus()
        order = self.order_records.get(iid)
        if order is None:
            messagebox.showinfo("No order selected", "Select an order first.")
        return order

    def _order_status(self, order: Any) -> str:
        if bool(_safe_attr(order, "has_delivered", False)):
            return "delivered"
        if bool(_safe_attr(order, "has_picked_up", False)):
            return "picked up"
        if bool(_safe_attr(order, "is_accepted", False)):
            try:
                ready = bool(order.is_ready_for_pickup())
            except Exception:
                ready = False
            return "ready" if ready else "accepted"
        return "open"

    def _endpoint_text(self, order: Any, node_attr: str, road_attr: str) -> str:
        xy = _order_node_xy(order, node_attr)
        road = _safe_attr(order, road_attr, "")
        if xy:
            return f"{_fmt_xy_m(xy[0], xy[1], 1)} {road}"
        return str(road)
    
    def _compute_yaw_from_move(self, from_x, from_y, to_x, to_y) -> int:
            """Snap movement direction to nearest cardinal yaw (0/90/180/270)."""
            dx = to_x - from_x
            dy = to_y - from_y
            if abs(dx) < 100 and abs(dy) < 100:
                return self.agent_yaw
            if abs(dx) > abs(dy):
                return 0 if dx > 0 else 180
            else:
                return 90 if dy > 0 else 270

    def _order_time_text(self, order: Any) -> str:
        if not bool(_safe_attr(order, "is_accepted", False)):
            return _sim_time_text(_safe_attr(order, "time_limit_s", 0.0))
        try:
            left = float(_safe_attr(order, "time_limit_s", 0.0)) - float(_safe_attr(order, "sim_elapsed_active_s", 0.0))
            return _sim_time_text(left)
        except Exception:
            return ""

    def _load_poi_records(self) -> None:
        self.poi_records.clear()
        city_map = getattr(self.env, "map", None)
        if city_map is None:
            return
        for idx, meta in enumerate(getattr(city_map, "poi_meta", []) or []):
            node = meta.get("node")
            if node is None:
                continue
            anchor = meta.get("door_node") or meta.get("dock_node") or node
            xy = _node_xy(anchor)
            if xy is None:
                continue
            poi_type = str(getattr(node, "type", "") or "").lower()
            name = str(getattr(node, "display_name", "") or getattr(node, "name", "") or f"{poi_type} {idx}")
            iid = f"poi-{idx}"
            self.poi_records[iid] = {
                "name": name,
                "type": poi_type,
                "x": xy[0],
                "y": xy[1],
                "road": str(meta.get("road_name") or ""),
                "meta": meta,
            }

    def refresh_poi_table(self) -> None:
        self.poi_tree.delete(*self.poi_tree.get_children())
        flt = self.poi_filter_var.get().strip().lower() if hasattr(self, "poi_filter_var") else ""
        for iid, rec in sorted(self.poi_records.items(), key=lambda item: (item[1]["type"], item[1]["name"])):
            hay = f"{rec['name']} {rec['type']} {rec['road']}".lower()
            if flt and flt not in hay:
                continue
            xy_text = _fmt_xy_m(rec["x"], rec["y"], 1)
            self.poi_tree.insert("", "end", iid=iid, values=(rec["name"], rec["type"], xy_text, rec["road"]))

    def refresh_observation(self) -> None:
        dm = self._dm()
        text = ""
        if dm is not None and hasattr(dm, "build_state_observation"):
            try:
                text = dm.build_state_observation()
            except Exception as exc:
                text = f"Could not build observation: {exc}"

        self.obs_text.configure(state=tk.NORMAL)
        self.obs_text.delete("1.0", tk.END)
        for line in text.splitlines(keepends=True):
            if line.startswith("### "):
                self.obs_text.insert(tk.END, line, "header")
            else:
                self.obs_text.insert(tk.END, line)

        # ---- feedback block ----
        fb_info = getattr(self, "last_info", None) or {}
        fb_reward = getattr(self, "last_reward", 0.0)
        fb_done = getattr(self, "last_done", False)
        fb_trunc = getattr(self, "last_truncated", False)
        self.obs_text.insert(
            tk.END,
            "\n" + "=" * 12 + " feedback " + "=" * 12 + "\n"
            f"step {fb_info.get('elapsed_steps', '?')} | "
            f"reward {fb_reward:+.3f} | "
            f"terminated={fb_done} truncated={fb_trunc}\n",
            "feedback",
        )
        err = _clean(fb_info.get("error", ""))
        if err:
            self.obs_text.insert(tk.END, f"error: {err}\n", "error")
        self.obs_text.configure(state=tk.DISABLED)
            
    def _refresh_action_choices(self) -> None:
        """Filter the action dropdown to currently-valid actions + query tools."""
        from vagen.envs.deliverybench.vlm_delivery.gameplay.valid_actions import (
            get_valid_actions, get_valid_tools,
        )

        dm = self._dm()
        if dm is None:
            valid_actions = list(ACTION_SPECS.keys())
        else:
            try:
                valid = set(get_valid_actions(dm, dm.cfg))
                valid |= set(get_valid_tools(dm, dm.cfg))
            except Exception:
                valid = set(ACTION_SPECS.keys())
            valid_actions = [name for name in ACTION_SPECS.keys() if name in valid]
            if not valid_actions:
                valid_actions = list(ACTION_SPECS.keys())

        self.action_combo['values'] = valid_actions
        if self.action_name_var.get() not in valid_actions and valid_actions:
            self.action_name_var.set(valid_actions[0])
            self._populate_action(valid_actions[0])
            
    def refresh_history(self) -> None:
        lines = []
        for step, action, reward, err in self.history:
            tail = f" | error={err}" if err else ""
            lines.append(f"{step:04d} | reward={reward:+.3f} | {action}{tail}")
        self._set_text(self.history_text, "\n".join(lines))

    def _set_text(self, widget: ScrolledText, text: str) -> None:
        widget.configure(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        widget.insert("1.0", text)
        widget.configure(state=tk.DISABLED)

    # ------------------------------------------------------------------
    # Map drawing and interaction
    # ------------------------------------------------------------------
    def fit_map(self) -> None:
        bounds = self._compute_bounds()
        if bounds is not None:
            self.view_bounds = bounds
        self.draw_map()

    def zoom_map(self, factor: float) -> None:
        if self.view_bounds is None:
            return
        xmin, xmax, ymin, ymax = self.view_bounds
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        w = (xmax - xmin) * factor
        h = (ymax - ymin) * factor
        self.view_bounds = (cx - w / 2.0, cx + w / 2.0, cy - h / 2.0, cy + h / 2.0)
        self.draw_map()

    def _zoom_at_event(self, event: tk.Event, factor: float) -> None:
        if self.view_bounds is None:
            return
        before = self.canvas_to_world(event.x, event.y)
        xmin, xmax, ymin, ymax = self.view_bounds
        w = (xmax - xmin) * factor
        h = (ymax - ymin) * factor
        rx = (before[0] - xmin) / max(1e-9, xmax - xmin)
        ry = (before[1] - ymin) / max(1e-9, ymax - ymin)
        new_xmin = before[0] - rx * w
        new_ymin = before[1] - ry * h
        self.view_bounds = (new_xmin, new_xmin + w, new_ymin, new_ymin + h)
        self.draw_map()

    def _on_mousewheel(self, event: tk.Event) -> None:
        factor = 0.8 if event.delta > 0 else 1.25
        self._zoom_at_event(event, factor)

    def _on_canvas_press(self, event: tk.Event) -> None:
        if self.view_bounds is None:
            return
        self._drag_start = (event.x, event.y, self.view_bounds)
        self._dragged = False

    def _on_canvas_drag(self, event: tk.Event) -> None:
        if self._drag_start is None:
            return
        sx, sy, bounds = self._drag_start
        dx = event.x - sx
        dy = event.y - sy
        if abs(dx) + abs(dy) < 4:
            return
        self._dragged = True
        xmin, xmax, ymin, ymax = bounds
        scale = self._current_scale()
        if scale <= 0:
            return
        world_dx = -dx / scale
        world_dy = dy / scale
        self.view_bounds = (xmin + world_dx, xmax + world_dx, ymin + world_dy, ymax + world_dy)
        self.draw_map()

    def _on_canvas_release(self, event: tk.Event) -> None:
        if not self._dragged:
            marker = self._nearest_marker(event.x, event.y)
            if marker is not None:
                self.fill_move_to_xy(marker["x"], marker["y"], marker["label"])
            else:
                x_cm, y_cm = self.canvas_to_world(event.x, event.y)
                self.fill_move_to_xy(x_cm, y_cm, "map point")
        self._drag_start = None
        self._dragged = False

    def _compute_bounds(self) -> Optional[Tuple[float, float, float, float]]:
        if self.env is None:
            return None
        city_map = getattr(self.env, "map", None)
        xs: List[float] = []
        ys: List[float] = []
        if city_map is not None:
            for node in getattr(city_map, "nodes", []) or []:
                xy = _node_xy(node)
                if xy:
                    xs.append(xy[0])
                    ys.append(xy[1])
            for meta in getattr(city_map, "poi_meta", []) or []:
                for key in ("node", "door_node", "dock_node"):
                    xy = _node_xy(meta.get(key))
                    if xy:
                        xs.append(xy[0])
                        ys.append(xy[1])
                box = meta.get("building_box")
                if box:
                    bx = float(box.get("x", 0.0))
                    by = float(box.get("y", 0.0))
                    bw = float(box.get("w", 0.0))
                    bh = float(box.get("h", 0.0))
                    xs.extend([bx - bw / 2.0, bx + bw / 2.0])
                    ys.extend([by - bh / 2.0, by + bh / 2.0])
        for order in self._orders():
            for attr in ("pickup_node", "dropoff_node"):
                xy = _order_node_xy(order, attr)
                if xy:
                    xs.append(xy[0])
                    ys.append(xy[1])
        dm = self._dm()
        if dm is not None:
            xs.append(float(getattr(dm, "x", 0.0)))
            ys.append(float(getattr(dm, "y", 0.0)))
        if not xs or not ys:
            return None
        pad = 2500.0
        return (min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad)

    def _current_scale(self) -> float:
        if self.view_bounds is None:
            return 1.0
        xmin, xmax, ymin, ymax = self.view_bounds
        w = max(1, self.canvas.winfo_width())
        h = max(1, self.canvas.winfo_height())
        pad = 24
        draw_w = max(1, w - pad * 2)
        draw_h = max(1, h - pad * 2)
        return min(draw_w / max(1.0, xmax - xmin), draw_h / max(1.0, ymax - ymin))

    def world_to_canvas(self, x: float, y: float) -> Tuple[float, float]:
        if self.view_bounds is None:
            return 0.0, 0.0
        xmin, xmax, ymin, ymax = self.view_bounds
        w = max(1, self.canvas.winfo_width())
        h = max(1, self.canvas.winfo_height())
        pad = 24
        scale = self._current_scale()
        span_x = xmax - xmin
        span_y = ymax - ymin
        used_w = span_x * scale
        used_h = span_y * scale
        draw_w = max(1, w - pad * 2)
        draw_h = max(1, h - pad * 2)
        off_x = pad + (draw_w - used_w) / 2.0
        off_y = pad + (draw_h - used_h) / 2.0
        px = off_x + (float(x) - xmin) * scale
        py = off_y + (ymax - float(y)) * scale
        return px, py

    def canvas_to_world(self, px: float, py: float) -> Tuple[float, float]:
        if self.view_bounds is None:
            return 0.0, 0.0
        xmin, xmax, _ymin, ymax = self.view_bounds
        w = max(1, self.canvas.winfo_width())
        h = max(1, self.canvas.winfo_height())
        pad = 24
        scale = self._current_scale()
        used_w = (xmax - xmin) * scale
        used_h = (ymax - _ymin) * scale
        draw_w = max(1, w - pad * 2)
        draw_h = max(1, h - pad * 2)
        off_x = pad + (draw_w - used_w) / 2.0
        off_y = pad + (draw_h - used_h) / 2.0
        x = xmin + (float(px) - off_x) / scale
        y = ymax - (float(py) - off_y) / scale
        return x, y

    def draw_map(self) -> None:
        self.canvas.delete("all")
        self.map_markers.clear()
        if self.env is None or self.view_bounds is None:
            self.canvas.create_text(20, 20, anchor="nw", text="No environment loaded.", fill="#333333")
            return
        city_map = getattr(self.env, "map", None)
        if city_map is None:
            return

        self._draw_buildings(city_map)
        self._draw_edges(city_map)
        self._draw_pois(city_map)
        self._draw_orders()
        self._draw_agent()
        self._draw_legend()
        self._draw_waypoints(city_map) 
        self._draw_navigation_route()

    def _draw_edges(self, city_map: Any) -> None:
        road_label_candidates: Dict[str, tuple] = {}
        for edge in getattr(city_map, "edges", []) or []:
            a = getattr(edge, "node1", None)
            b = getattr(edge, "node2", None)
            a_xy = _node_xy(a)
            b_xy = _node_xy(b)
            if not a_xy or not b_xy:
                continue
            meta = {}
            try:
                meta = city_map._get_edge_meta(a, b) or {}
            except Exception:
                meta = {}
            kind = str(meta.get("kind") or "")
            if kind.startswith("aux_"):
                color = "#d3d8de"
                width = 1
            elif kind == "crosswalk":
                color = "#8f9aa8"
                width = 2
            elif kind == "drive":
                color = "#b8a272"
                width = 2
            else:
                color = "#9ea7b3"
                width = 3
            x1, y1 = self.world_to_canvas(*a_xy)
            x2, y2 = self.world_to_canvas(*b_xy)
            self.canvas.create_line(x1, y1, x2, y2, fill=color, width=width)
            self.canvas.create_line(x1, y1, x2, y2, fill=color, width=width)

            road_name = str(meta.get("name") or "")
            if road_name and kind not in ("crosswalk",) and not kind.startswith("aux_"):
                seg_len = (x2 - x1) ** 2 + (y2 - y1) ** 2
                prev = road_label_candidates.get(road_name)
                if prev is None or seg_len > prev[0]:
                    road_label_candidates[road_name] = (seg_len, (x1 + x2) / 2, (y1 + y2) / 2)

        for road_name, (_l, mx, my) in road_label_candidates.items():
            self.canvas.create_text(mx, my - 6, text=road_name,
                                    fill="#6b7480", font=("TkDefaultFont", 7, "italic"))
            
    def _draw_buildings(self, city_map: Any) -> None:
        for meta in getattr(city_map, "poi_meta", []) or []:
            box = meta.get("building_box")
            if not box:
                continue
            poi_type = str(box.get("poi_type", "building")).lower()
            fill = POI_COLORS.get(poi_type, "#b0bec5")
            pts = self._rotated_box_points(box)
            if not pts:
                continue
            flat: List[float] = []
            for x, y in pts:
                px, py = self.world_to_canvas(x, y)
                flat.extend([px, py])
            self.canvas.create_polygon(flat, fill=fill, outline="#4f5963", stipple="gray75")

    

    def _draw_navigation_route(self) -> None:
        """Draw the planned route from current to last NAVIGATE target."""
        if not self._last_nav_target_id:
            return
        
        city_map = getattr(self.env, "map", None)
        dm = self._dm()
        if not city_map or not dm:
            return
        
        try:
            current_wp = city_map.nearest_waypoint(dm.x, dm.y)
            target_wp = city_map.resolve_waypoint(self._last_nav_target_id)
        except Exception:
            return
        
        if not current_wp or not target_wp:
            return
        
        try:
            path, _ = city_map.waypoint_graph.shortest_path_nodes(current_wp, target_wp)
        except Exception:
            return
        
        if not path or len(path) < 2:
            return
    
        points = []
        for node in path:
            xy = _node_xy(node)
            if xy:
                px, py = self.world_to_canvas(*xy)
                points.extend([px, py])
        
        if len(points) >= 4:
            self.canvas.create_line(*points, fill="#ff00ff", width=3, 
                                    dash=(6, 4), arrow=tk.LAST)
        
    
    
    
    def _rotated_box_points(self, box: Dict[str, Any]) -> List[Tuple[float, float]]:
        cx = float(box.get("x", 0.0))
        cy = float(box.get("y", 0.0))
        w = float(box.get("w", 0.0))
        h = float(box.get("h", 0.0))
        yaw = math.radians(float(box.get("yaw", 0.0)))
        if w <= 0 or h <= 0:
            return []
        raw = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
        out = []
        c = math.cos(yaw)
        s = math.sin(yaw)
        for x, y in raw:
            out.append((cx + x * c - y * s, cy + x * s + y * c))
        return out

    def _draw_pois(self, city_map: Any) -> None:
        scale = self._current_scale()
        show_labels = scale > 0.004
        for rec in self.poi_records.values():
            poi_type = rec["type"]
            color = POI_COLORS.get(poi_type, "#7f8c8d")
            x, y = rec["x"], rec["y"]
            px, py = self.world_to_canvas(x, y)
            radius = 5 if poi_type != "building" else 3
            self.canvas.create_oval(px - radius, py - radius, px + radius, py + radius, fill=color, outline="#ffffff")
            label = f"{rec['name']} {_fmt_xy_m(x, y, 1)}"
            self.map_markers.append({"x": x, "y": y, "px": px, "py": py, "r": radius + 8, "label": label})
            if show_labels and poi_type != "building":
                self.canvas.create_text(px + 7, py - 7, anchor="sw", text=rec["name"], fill="#1f2d3d", font=("TkDefaultFont", 8))

    def _draw_orders(self) -> None:
        groups: Dict[tuple, Dict[str, Any]] = {}
        for order in self._orders():
            oid = _safe_attr(order, "id", "?")
            accepted = bool(_safe_attr(order, "is_accepted", False))
            if bool(_safe_attr(order, "has_delivered", False)):
                continue
            for endpoint, attr in [("pickup", "pickup_node"), ("dropoff", "dropoff_node")]:
                xy = _order_node_xy(order, attr)
                if xy is None:
                    continue
                key = (endpoint, round(xy[0], 1), round(xy[1], 1))
                g = groups.setdefault(key, {"xy": xy, "endpoint": endpoint,
                                            "ids": [], "accepted": False})
                g["ids"].append(str(oid))
                g["accepted"] = g["accepted"] or accepted

        for g in groups.values():
            px, py = self.world_to_canvas(*g["xy"])
            endpoint = g["endpoint"]
            label_prefix = "P" if endpoint == "pickup" else "D"
            if g["accepted"]:
                base = "#fd9488" if endpoint == "pickup" else "#2ecc71"
                r, fill = 9, base
                txt_color, font = "#111111", ("TkDefaultFont", 9, "bold")
                self.canvas.create_oval(px - 15, py - 15, px + 15, py + 15,
                                        outline=base, width=2)
            else:
                r, fill = 5, "#c9ced4"
                txt_color, font = "#9aa1a8", ("TkDefaultFont", 7)

            if endpoint == "pickup":
                self.canvas.create_polygon(px, py - r, px + r, py + r, px - r, py + r,
                                        fill=fill, outline="#ffffff")
            else:
                self.canvas.create_rectangle(px - r, py - r, px + r, py + r,
                                            fill=fill, outline="#ffffff")
            text = label_prefix + ",".join(g["ids"])     # 例如 "P0,1,3,10"
            self.canvas.create_text(px + r + 2, py, anchor="w", text=text,
                                    fill=txt_color, font=font)
            self.map_markers.append({
                "x": g["xy"][0], "y": g["xy"][1], "px": px, "py": py,
                "r": r + 8, "label": f"{endpoint} for orders {','.join(g['ids'])}",
            })

    def _draw_agent(self) -> None:
        dm = self._dm()
        if dm is None:
            return
        x = float(getattr(dm, "x", 0.0))
        y = float(getattr(dm, "y", 0.0))
        px, py = self.world_to_canvas(x, y)
        r = 10
        self.canvas.create_oval(px - r, py - r, px + r, py + r, fill="#000000", outline="#ffffff", width=2)
        self.canvas.create_text(px + 13, py - 2, anchor="w", text="agent", fill="#000000", font=("TkDefaultFont", 9, "bold"))

    def _draw_legend(self) -> None:
        x = 12
        y = 12
        items = [
            ("agent", "#000000"),
            ("pickup", "#fd9488"),
            ("dropoff", "#2ecc71"),
            ("restaurant", POI_COLORS["restaurant"]),
            ("store", POI_COLORS["store"]),
            ("charge", POI_COLORS["charging_station"]),
        ]
        self.canvas.create_rectangle(x - 6, y - 6, x + 150, y + len(items) * 18 + 6, fill="#ffffff", outline="#ccd4dd")
        for idx, (label, color) in enumerate(items):
            yy = y + idx * 18
            self.canvas.create_rectangle(x, yy, x + 10, yy + 10, fill=color, outline="#ffffff")
            self.canvas.create_text(x + 16, yy + 5, anchor="w", text=label, fill="#222222", font=("TkDefaultFont", 8))
    def _draw_waypoints(self, city_map: Any) -> None:
        """Label waypoints relevant to STEP_TO planning: current + adjacents + intersections + order endpoints."""
        dm = self._dm()
        if dm is None or not hasattr(city_map, "nearest_waypoint"):
            return
        try:
            current_wp = city_map.nearest_waypoint(dm.x, dm.y)
        except Exception:
            current_wp = None
        
        adjacent_ids = set()
        if current_wp is not None:
            try:
                for adj_info in (city_map.adjacents(current_wp) or []):
                    adj_node = adj_info.get("node")
                    if adj_node:
                        adj_id = getattr(adj_node, "waypoint_id", "") or ""
                        if adj_id:
                            adjacent_ids.add(adj_id)
            except Exception:
                pass
        
        current_id = getattr(current_wp, "waypoint_id", "") if current_wp else ""

        pickup_ids = set()
        dropoff_ids = set()
        for order in self._orders():
            if not bool(_safe_attr(order, "is_accepted", False)):
                continue
            if bool(_safe_attr(order, "has_delivered", False)):
                continue
            pn = getattr(order, "pickup_node", None)
            if pn:
                pid = getattr(pn, "waypoint_id", "") or ""
                if pid:
                    pickup_ids.add(pid)
            dn = getattr(order, "dropoff_node", None)
            if dn:
                did = getattr(dn, "waypoint_id", "") or ""
                if did:
                    dropoff_ids.add(did)
        
        seen_ids = set()
        nodes_to_draw = []
        if current_wp is not None:
            nodes_to_draw.append((current_wp, "current"))

        if current_wp is not None:
            for adj_info in (city_map.adjacents(current_wp) or []):
                adj_node = adj_info.get("node")
                if adj_node and adj_node is not current_wp:
                    nodes_to_draw.append((adj_node, "adjacent"))

        for order in self._orders():
            if not bool(_safe_attr(order, "is_accepted", False)):
                continue
            if bool(_safe_attr(order, "has_delivered", False)):
                continue
            for attr, role in [("pickup_node", "pickup"), ("dropoff_node", "dropoff")]:
                n = getattr(order, attr, None)
                if n:
                    nodes_to_draw.append((n, role))
        seen_int_ids = set()
        for node in (getattr(city_map, "nodes", []) or []):
            wp_id = getattr(node, "waypoint_id", "") or ""
            if not wp_id.startswith("int_") or wp_id in seen_int_ids:
                continue
            seen_int_ids.add(wp_id)
            nodes_to_draw.append((node, "intersection"))
        priority = {"current": 0, "pickup": 1, "dropoff": 2, "adjacent": 3, "intersection": 4}
        best = {}
        for node, role in nodes_to_draw:
            cur = best.get(id(node))
            if cur is None or priority[role] < priority[cur[1]]:
                best[id(node)] = (node, role)

        for node, role in best.values():
            xy = _node_xy(node)
            if not xy:
                continue
            px, py = self.world_to_canvas(*xy)
            
            if role == "current":
                color, r, font_w, font_sz = "#cc0000", 6, "bold", 9
            elif role == "pickup":
                color, r, font_w, font_sz = "#ff6600", 6, "bold", 9
            elif role == "dropoff":
                color, r, font_w, font_sz = "#009900", 6, "bold", 9
            elif role == "adjacent":
                color, r, font_w, font_sz = "#0066cc", 5, "bold", 9
            else:  # intersection
                color, r, font_w, font_sz = "#5599dd", 3, "normal", 7
            
            wp_id = getattr(node, "waypoint_id", "") or ""
            self.canvas.create_oval(px - r, py - r, px + r, py + r,
                                    fill=color, outline="#ffffff", width=1)
            self.canvas.create_text(px + 8, py - 8, anchor="sw",
                                    text=wp_id, fill=color,
                                    font=("TkDefaultFont", font_sz, font_w))
        
    def _nearest_marker(self, px: float, py: float) -> Optional[Dict[str, Any]]:
        best = None
        best_d2 = float("inf")
        for marker in self.map_markers:
            dx = float(marker["px"]) - float(px)
            dy = float(marker["py"]) - float(py)
            d2 = dx * dx + dy * dy
            limit = float(marker.get("r", 12))
            if d2 <= limit * limit and d2 < best_d2:
                best = marker
                best_d2 = d2
        return best


def run_smoke(args: argparse.Namespace) -> None:
    from vagen.envs.deliverybench.vlm_delivery.gym_like_interface import DeliveryBenchGymEnvText

    base_dir = Path(args.base_dir).resolve()
    env = DeliveryBenchGymEnvText(
        base_dir=str(base_dir),
        map_name=args.map_name,
        time_scale=float(args.time_scale),
        max_steps=int(args.max_steps),
        enable_map_images=False,
        map_renderer="pil",
        enable_vlm=False,
    )
    obs, info = env.reset(seed=int(args.seed))
    step_obs, reward, terminated, truncated, step_info = env.step(args.smoke_action)
    dm = env.dms[0]
    print("smoke: ok")
    print(f"map: {args.map_name}")
    print(f"seed: {args.seed}")
    print(f"initial_state: {obs.get('state')}")
    print(f"agent_xy_cm: ({float(dm.x):.2f}, {float(dm.y):.2f})")
    print(f"action: {args.smoke_action}")
    print(f"reward: {reward}")
    print(f"terminated: {terminated}")
    print(f"truncated: {truncated}")
    print(f"info: {step_info}")
    print(f"next_state: {step_obs.get('state')}")
    print(f"run_dir: {info.get('run_dir')}")
    env.close()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tkinter human validator for VAGEN DeliveryBench.")
    parser.add_argument("--base-dir", default=str(DEFAULT_BASE_DIR), help="DeliveryBench package directory containing maps/ and vlm_delivery/.")
    parser.add_argument("--map-name", default="medium-city-22", help="Map directory name under maps/.")
    parser.add_argument("--seed", type=int, default=42, help="Reset seed.")
    parser.add_argument("--max-steps", type=int, default=100, help="Episode max steps.")
    parser.add_argument("--time-scale", type=float, default=1.0, help="Simulation time scale.")
    parser.add_argument("--smoke", action="store_true", help="Run a headless reset/step smoke check instead of opening Tk.")
    parser.add_argument("--smoke-action", default="VIEW_ORDERS()", help="Action to run in --smoke mode.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.smoke:
        run_smoke(args)
        return
    root = tk.Tk()
    DeliveryBenchHilApp(root, args)
    root.mainloop()


if __name__ == "__main__":
    main()
