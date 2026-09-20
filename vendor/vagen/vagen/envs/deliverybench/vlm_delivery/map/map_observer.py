# map/map_observer.py
# -*- coding: utf-8 -*-

from typing import Dict, Any, List, Optional, Tuple, Callable
import math, random

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QPushButton, QWidget, QHBoxLayout, QDialog, QVBoxLayout, QScrollArea,
    QGridLayout, QFrame, QLabel, QTabWidget,
    QGraphicsRectItem, QGraphicsPathItem
)
from PyQt5.QtGui import QFont, QPainterPath, QImage, QPixmap

import numpy as np
try:
    import cv2  # Only used for color space conversion; code still runs without it
except Exception:
    cv2 = None

import pyqtgraph as pg

from .map_canvas_base import MapCanvasBase, _node_xy, UI_AGENT_SIZE
from ..base.timer import VirtualClock  # Virtual time (for charging/rest timers)

# ===== Visual parameters =====
AGENT_FILL   = "#E0E0E0"
AGENT_BORDER = "#000000"
AGENT_TEXT   = "#000000"
AGENT_Z      = 20
DEFAULT_TICK_MS = 100

BADGE_OFFSET_X_CM = 420.0
BADGE_OFFSET_Y_CM = -420.0
BADGE_SIZE        = 16

BATT_BAR_W_CM     = 620.0
BATT_BAR_H_CM     = 150.0
BATT_OFFSET_X_CM  = 460.0
BATT_OFFSET_Y_CM  = 220.0

ENERGY_BAR_W_CM    = BATT_BAR_W_CM
ENERGY_BAR_H_CM    = BATT_BAR_H_CM
ENERGY_OFFSET_X_CM = BATT_OFFSET_X_CM
ENERGY_OFFSET_Y_CM = BATT_OFFSET_Y_CM + BATT_BAR_H_CM + 220.0

LABEL_TEXT_PT      = 12
LABEL_MARGIN_X_CM  = 120.0

Z_LABEL = AGENT_Z + 100
Z_BADGE = AGENT_Z + 40
Z_BATT  = AGENT_Z + 50
Z_ENERGY= AGENT_Z + 55
Z_LIGHT = AGENT_Z + 60
Z_STATUS= AGENT_Z + 70
Z_RESCUE= AGENT_Z + 90

COLOR_BADGE_SCOOTER = "#33cc66"  # e-scooter badge
COLOR_BADGE_CAR     = "#ff3333"  # car badge

COLOR_BAR_BG        = "#999999"
COLOR_BAR_FG        = "#33cc66"  # Scooter battery fill
COLOR_ENERGY_BAR_BG = "#bfbfbf"
COLOR_ENERGY_BAR_FG = "#ffb347"
COLOR_LIGHTNING     = "#33cc66"
COLOR_STROKE        = "#222222"
COLOR_RESCUE_CROSS  = "#ff3333"


# ---------------------- Orders main window (enhanced) ----------------------
class OrdersDialog(QDialog):
    CARD_STYLE = """
        QFrame { background:#FAFAFA; border:1px solid #222; border-radius:8px; }
        QLabel { color:#111; }
    """
    def __init__(self, parent=None, order_manager=None, agents: List[Any]=None, clock: Optional[VirtualClock]=None):
        super().__init__(parent)
        self.setWindowTitle("Orders")
        self.resize(1000, 680)

        self._om = order_manager
        self._agents: List[Any] = list(agents or [])
        self._clock = clock if clock is not None else VirtualClock()

        self._suppress_refresh = False

        top = QWidget(self)
        hb  = QHBoxLayout(top); hb.setContentsMargins(0,0,0,0); hb.setSpacing(8)
        self.btn_refresh = QPushButton("Refresh", top)
        hb.addStretch(1); hb.addWidget(self.btn_refresh)

        self.tabs = QTabWidget(self); self.tabs.setDocumentMode(True)

        # Pool tab
        self.page_pool = QWidget(self.tabs)
        self.pool_scroll = QScrollArea(self.page_pool); self.pool_scroll.setWidgetResizable(True)
        self.pool_content = QWidget(self.pool_scroll); self.pool_scroll.setWidget(self.pool_content)
        v_pool = QVBoxLayout(self.page_pool); v_pool.setContentsMargins(0,0,0,0); v_pool.addWidget(self.pool_scroll)
        self.tabs.addTab(self.page_pool, "Order Pool")

        # Current tab
        self.page_current = QWidget(self.tabs)
        v_cur = QVBoxLayout(self.page_current); v_cur.setContentsMargins(0,0,0,0)
        self.cur_agent_tabs = QTabWidget(self.page_current); self.cur_agent_tabs.setDocumentMode(True)
        v_cur.addWidget(self.cur_agent_tabs)
        self.tabs.addTab(self.page_current, "Current (Accepted)")

        # Completed tab
        self.page_done = QWidget(self.tabs)
        v_done = QVBoxLayout(self.page_done); v_done.setContentsMargins(0,0,0,0)
        self.done_agent_tabs = QTabWidget(self.page_done); self.done_agent_tabs.setDocumentMode(True)
        v_done.addWidget(self.done_agent_tabs)
        self.tabs.addTab(self.page_done, "Completed")

        root = QVBoxLayout(self); root.addWidget(top); root.addWidget(self.tabs)

        self.btn_refresh.clicked.connect(self.refresh_all)
        self.tabs.currentChanged.connect(self._on_any_tab_changed)
        self.cur_agent_tabs.currentChanged.connect(self._on_any_tab_changed)
        self.done_agent_tabs.currentChanged.connect(self._on_any_tab_changed)

        self.refresh_all()

    def _ue_actor_name(self, dm, fallback_id: str) -> str:
        """
        Resolve the UE actor name for a DeliveryMan:
        - Prefer dm._ue_actor_name or dm.ue_actor_name if already recorded
        - Otherwise use GEN_DELIVERY_MAN_{dm.agent_id}
        - Fall back to the provided fallback_id (usually the viewer's agent id)
        """
        return (
            getattr(dm, "_ue_actor_name", None)
            or getattr(dm, "ue_actor_name", None)
            or (f"GEN_DELIVERY_MAN_{getattr(dm, 'agent_id', fallback_id)}" if getattr(dm, "agent_id", None) is not None else fallback_id)
        )

    def set_sources(self, order_manager=None, agents: List[Any]=None, clock: Optional[VirtualClock]=None):
        if order_manager is not None: self._om = order_manager
        if agents is not None: self._agents = list(agents)
        if clock is not None: self._clock = clock
        self.refresh_all()

    def _reset_scroll_content(self, scroll: QScrollArea) -> QWidget:
        old = scroll.takeWidget()
        content = QWidget(); scroll.setWidget(content)
        if old is not None: old.deleteLater()
        return content

    def _text_card(self, text: str, parent: QWidget) -> QFrame:
        fr = QFrame(parent); fr.setStyleSheet(self.CARD_STYLE)
        v = QVBoxLayout(fr); v.setContentsMargins(10,8,10,8); v.setSpacing(6)
        lab = QLabel(fr)
        lab.setText(text); lab.setTextFormat(Qt.PlainText); lab.setWordWrap(True)
        f = lab.font(); f.setFamily("Consolas"); f.setPointSize(9); lab.setFont(f)
        lab.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
        v.addWidget(lab, alignment=Qt.AlignLeft | Qt.AlignTop)
        return fr

    def _fmt_min(self, seconds: float) -> str:
        minutes = int((float(seconds) + 59) // 60)
        return f"{minutes} min"

    def _status_of(self, o) -> str:
        if getattr(o, "has_delivered", False): return "Delivered"
        if getattr(o, "has_picked_up", False): return "Delivering"
        if getattr(o, "is_accepted", False):   return "Accepted"
        return "New"

    def _on_any_tab_changed(self, *_):
        if self._suppress_refresh: return
        self.refresh_all()

    def refresh_all(self):
        self.setUpdatesEnabled(False)
        self._suppress_refresh = True
        main_idx = self.tabs.currentIndex()
        cur_idx  = self.cur_agent_tabs.currentIndex()
        done_idx = self.done_agent_tabs.currentIndex()
        try:
            self._build_pool_tab()
            self._build_current_tabs()
            self._build_completed_tabs()
        finally:
            if 0 <= main_idx < self.tabs.count(): self.tabs.setCurrentIndex(main_idx)
            if 0 <= cur_idx < self.cur_agent_tabs.count(): self.cur_agent_tabs.setCurrentIndex(cur_idx)
            if 0 <= done_idx < self.done_agent_tabs.count(): self.done_agent_tabs.setCurrentIndex(done_idx)
            self._suppress_refresh = False
            self.setUpdatesEnabled(True)

    # ---------- Text formatting helpers ----------
    def _order_text_with_items(self, o, *, now_sim: float) -> str:
        dist_m = float(getattr(o, "distance_cm", 0.0)) / 100.0
        dist_km = dist_m / 1000.0
        base = [
            f"[Order #{getattr(o,'id','?')}]",
            f"  Pickup : ({o.pickup_address.x:.1f}, {o.pickup_address.y:.1f})  | road: {getattr(o,'pickup_road_name','')}",
            f"  Dropoff: ({o.delivery_address.x:.1f}, {o.delivery_address.y:.1f}) | road: {getattr(o,'dropoff_road_name','')}",
            f"  Path   : {len(getattr(o,'path_nodes',[]) or [])} nodes",
            f"  Dist   : {dist_m:.1f} m  ({dist_km:.3f} km)",
        ]
        tl = float(getattr(o, "time_limit_s", 0.0))
        base.append(f"  Limit  : {self._fmt_min(tl)}")
        spent = float(getattr(o, "sim_elapsed_active_s", 0.0) or 0.0)
        base.append(f"  Spent  : {self._fmt_min(spent)}")
        if hasattr(o, "earnings"):
            base.append(f"  $$     : ${float(o.earnings):.2f}")

        if hasattr(o, "is_ready_for_pickup") and not o.is_ready_for_pickup(now_sim):
            remain = float(o.remaining_prep_s(now_sim))
            base.append(f"  Prep   : ready in ~{self._fmt_min(remain)} (virtual)")

        items = list(getattr(o, "items", []) or [])
        if items:
            base.append("  Items  :")
            for it in items:
                name = getattr(it, "name", str(it))

                # Item temperature
                t = float(getattr(it, "temp_c", float('nan')))
                t_str = f"{t:.1f}°C" if (t == t) else "N/A"

                # Odor contamination level (0..1 -> percentage)
                oc = getattr(it, "odor_contamination", None)
                try:
                    oc_val = max(0.0, min(1.0, float(oc)))
                    contam_str = f"{oc_val*100:.0f}%"
                except (TypeError, ValueError):
                    contam_str = "N/A"

                damage_level = int(getattr(it, "damage_level", 0))
                damage_str = str(damage_level) if (0 <= damage_level <= 3) else "N/A"

                base.append(f"    - {name}  (T={t_str}, damage={damage_str}, odor contam={contam_str})")

        note = getattr(o, "special_note", "")
        if note:
            base.append(f"  Note   : {note}")
        return "\n".join(base)

    # ---------- Build pages ----------
    def _build_pool_tab(self):
        self.pool_content = self._reset_scroll_content(self.pool_scroll)
        lay = QGridLayout(self.pool_content); lay.setContentsMargins(10,10,10,10); lay.setSpacing(12)
        orders = self._om.list_orders() if (self._om and hasattr(self._om, "list_orders")) else []
        col = row = 0; cols = 2
        now_sim = self._clock.now_sim()
        for o in orders:
            lay.addWidget(self._text_card(self._order_text_with_items(o, now_sim=now_sim), self.pool_content), row, col)
            col += 1
            if col >= cols: col = 0; row += 1

    def _build_current_tabs(self):
        while self.cur_agent_tabs.count():
            w = self.cur_agent_tabs.widget(0); self.cur_agent_tabs.removeTab(0); w.deleteLater()
        for dm in self._agents:
            agent_id = str(getattr(dm, "_viewer_agent_id", getattr(dm, "name", "?")))
            page = QWidget(self.cur_agent_tabs)
            sa = QScrollArea(page); sa.setWidgetResizable(True)
            content = QWidget(sa); sa.setWidget(content)
            v = QVBoxLayout(page); v.setContentsMargins(0,0,0,0); v.addWidget(sa)
            lay = QGridLayout(content); lay.setContentsMargins(10,10,10,10); lay.setSpacing(12)

            now_sim = self._clock.now_sim()
            active_orders = getattr(dm, "active_orders", None)
            if isinstance(active_orders, list) and active_orders:
                col = row = 0; cols = 2
                for o in active_orders:
                    txt = f"Agent {agent_id} — Current\n" + self._order_text_with_items(o, now_sim=now_sim) + f"\n  Status : {self._status_of(o)}"
                    lay.addWidget(self._text_card(txt, content), row, col)
                    col += 1
                    if col >= cols: col = 0; row += 1
            else:
                o = getattr(dm, "current_order", None)
                if o is None or getattr(o, "has_delivered", False):
                    lay.addWidget(self._text_card(f"Agent {agent_id} — Current\nNo current order.", content), 0, 0)
                else:
                    txt = f"Agent {agent_id} — Current\n" + self._order_text_with_items(o, now_sim=now_sim) + f"\n  Status : {self._status_of(o)}"
                    lay.addWidget(self._text_card(txt, content), 0, 0)

            self.cur_agent_tabs.addTab(page, f"Agent {agent_id}")

    def _build_completed_tabs(self):
        while self.done_agent_tabs.count():
            w = self.done_agent_tabs.widget(0); self.done_agent_tabs.removeTab(0); w.deleteLater()
        for dm in self._agents:
            agent_id = str(getattr(dm, "_viewer_agent_id", getattr(dm, "name", "?")))
            page = QWidget(self.done_agent_tabs)
            sa = QScrollArea(page); sa.setWidgetResizable(True)
            content = QWidget(sa); sa.setWidget(content)
            v = QVBoxLayout(page); v.setContentsMargins(0,0,0,0); v.addWidget(sa)
            lay = QGridLayout(content); lay.setContentsMargins(10,10,10,10); lay.setSpacing(12)

            hist = list(getattr(dm, "completed_orders", []) or [])
            if not hist:
                lay.addWidget(self._text_card("No completed orders", content), 0, 0)
            else:
                col = row = 0; cols = 2
                for rec in hist:
                    txt = (
                        f"[Order #{rec.get('id','?')}]\n"
                        f"  {rec.get('pickup','')} -> {rec.get('dropoff','')}\n"
                        f"  Time   : {self._fmt_min(rec.get('duration_s',0.0))}\n"
                        f"  $$     : ${rec.get('paid_total', 0.0):.2f}\n"
                        f"  Base   : ${rec.get('earnings', 0.0):.2f}\n"
                        f"  Extra  : ${rec.get('bonus_extra', 0.0):+.2f}\n"
                        f"  Stars  : {rec.get('rating', 0)}\n"
                    )
                    lay.addWidget(self._text_card(txt, content), row, col)
                    col += 1
                    if col >= cols: col = 0; row += 1

            self.done_agent_tabs.addTab(page, f"Agent {agent_id}")


# ---------------------- Requests window (Comms request board) ----------------------
class RequestsDialog(QDialog):
    CARD_STYLE = """
        QFrame { background:#FAFAFA; border:1px solid #222; border-radius:8px; }
        QLabel { color:#111; }
    """
    def __init__(self, parent=None, comms=None, agents: List[Any]=None, clock: Optional[VirtualClock]=None):
        super().__init__(parent)
        self.setWindowTitle("Requests")
        self.resize(1000, 680)

        self._comms = comms
        self._agents: List[Any] = list(agents or [])
        self._clock = clock if clock is not None else VirtualClock()

        top = QWidget(self)
        hb  = QHBoxLayout(top); hb.setContentsMargins(0,0,0,0); hb.setSpacing(8)
        self.btn_refresh = QPushButton("Refresh", top)
        hb.addStretch(1); hb.addWidget(self.btn_refresh)

        self.tabs = QTabWidget(self); self.tabs.setDocumentMode(True)

        # Board (unaccepted requests)
        self.page_board = QWidget(self.tabs)
        self.board_scroll = QScrollArea(self.page_board); self.board_scroll.setWidgetResizable(True)
        self.board_content = QWidget(self.board_scroll); self.board_scroll.setWidget(self.board_content)
        v_board = QVBoxLayout(self.page_board); v_board.setContentsMargins(0,0,0,0); v_board.addWidget(self.board_scroll)
        self.tabs.addTab(self.page_board, "Request Board")

        # Active (accepted / ongoing)
        self.page_active = QWidget(self.tabs)
        self.active_scroll = QScrollArea(self.page_active); self.active_scroll.setWidgetResizable(True)
        self.active_content = QWidget(self.active_scroll); self.active_scroll.setWidget(self.active_content)
        v_act = QVBoxLayout(self.page_active); v_act.setContentsMargins(0,0,0,0); v_act.addWidget(self.active_scroll)
        self.tabs.addTab(self.page_active, "Active (Accepted)")

        # Completed
        self.page_done = QWidget(self.tabs)
        self.done_scroll = QScrollArea(self.page_done); self.done_scroll.setWidgetResizable(True)
        self.done_content = QWidget(self.done_scroll); self.done_scroll.setWidget(self.done_content)
        v_done = QVBoxLayout(self.page_done); v_done.setContentsMargins(0,0,0,0); v_done.addWidget(self.done_scroll)
        self.tabs.addTab(self.page_done, "Completed")

        root = QVBoxLayout(self); root.addWidget(top); root.addWidget(self.tabs)

        self.btn_refresh.clicked.connect(self.refresh_all)
        self.tabs.currentChanged.connect(self.refresh_all)
        self.refresh_all()

    # --- External injection of data sources ---
    def set_sources(self, comms=None, agents: List[Any]=None, clock: Optional[VirtualClock]=None):
        if comms is not None: self._comms = comms
        if agents is not None: self._agents = list(agents or [])
        if clock is not None: self._clock = clock
        self.refresh_all()

    # --- Small helpers ---
    def _reset_scroll_content(self, scroll: QScrollArea) -> QWidget:
        old = scroll.takeWidget()
        content = QWidget(); scroll.setWidget(content)
        if old is not None: old.deleteLater()
        return content

    def _text_card(self, text: str, parent: QWidget) -> QFrame:
        fr = QFrame(parent); fr.setStyleSheet(self.CARD_STYLE)
        v = QVBoxLayout(fr); v.setContentsMargins(10,8,10,8); v.setSpacing(6)
        lab = QLabel(fr)
        lab.setText(text); lab.setTextFormat(Qt.PlainText); lab.setWordWrap(True)
        f = lab.font(); f.setFamily("Consolas"); f.setPointSize(9); lab.setFont(f)
        lab.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
        v.addWidget(lab, alignment=Qt.AlignLeft | Qt.AlignTop)
        return fr

    def _fmt_min(self, seconds: float) -> str:
        minutes = int((float(seconds) + 59) // 60)
        return f"{minutes} min"

    def _agent_label(self, agent_id: Optional[str]) -> str:
        if not agent_id:
            return "N/A"
        aid = str(agent_id)
        for dm in self._agents:
            if str(getattr(dm, "agent_id", "")) == aid:
                return str(getattr(dm, "_viewer_agent_id", getattr(dm, "name", aid)))
        return aid

    def _fmt_xy(self, xy) -> str:
        if isinstance(xy, (list, tuple)) and len(xy) >= 2:
            return f"({float(xy[0]):.1f}, {float(xy[1]):.1f})"
        return "(N/A)"

    def _fmt_board_card(self, rec: Dict[str, Any]) -> str:
        rid   = rec.get("id", "?")
        pub   = self._agent_label(rec.get("publisher"))
        kind  = str(rec.get("kind", ""))
        brief = str(rec.get("brief", "") or "")
        xy    = self._fmt_xy(rec.get("location_xy"))
        reward= float(rec.get("reward", 0.0) or 0.0)
        left  = float(rec.get("time_left_s", 0.0) or 0.0)
        lines = [
            f"[Req #{rid}]",
            f"  Kind    : {kind}",
            f"  From    : {pub}",
            f"  Where   : {xy}",
            f"  $$      : ${reward:.2f}",
            f"  TimeLeft: {self._fmt_min(left)}",
        ]
        if brief:
            lines.append(f"  Brief   : {brief}")
        return "\n".join(lines)

    def _fmt_active_card(self, req_obj: Any, now_sim: float) -> str:
        rid = getattr(req_obj, "req_id", "?")
        pub = self._agent_label(getattr(req_obj, "publisher_id", None))
        helper = self._agent_label(getattr(req_obj, "accepted_by", None))
        kind = getattr(getattr(req_obj, "kind", ""), "value", str(getattr(req_obj, "kind", "")))
        reward = float(getattr(req_obj, "reward", 0.0) or 0.0)
        limit  = float(getattr(req_obj, "time_limit_s", 0.0) or 0.0)
        created= float(getattr(req_obj, "created_sim", 0.0) or 0.0)
        left   = max(0.0, (created + limit) - now_sim)
        brief  = str(getattr(req_obj, "brief", "") or "")
        xy     = self._fmt_xy(getattr(req_obj, "location_xy", None))
        lines = [
            f"[Req #{rid}]",
            f"  Kind   : {kind}",
            f"  From   : {pub}",
            f"  Helper : {helper}",
            f"  Where  : {xy}",
            f"  $$     : ${reward:.2f}",
            f"  Limit  : {self._fmt_min(limit)}   Left: {self._fmt_min(left)}",
        ]
        if brief: lines.append(f"  Brief  : {brief}")
        if getattr(req_obj, "order_id", None) is not None:
            lines.append(f"  Order  : #{int(req_obj.order_id)}")
        if getattr(req_obj, "target_pct", None) is not None:
            lines.append(f"  Target : {float(req_obj.target_pct):.0f}%")
        bi = getattr(req_obj, "buy_items", None)
        if bi:
            parts = ", ".join(f"{k} x{int(v)}" for k, v in bi.items())
            lines.append(f"  Buy    : {parts}")
        return "\n".join(lines)

    def _fmt_completed_card(self, rec_obj: Any) -> str:
        rid = getattr(rec_obj, "req_id", None)
        if rid is None and isinstance(rec_obj, dict):
            rid = rec_obj.get("req_id", rec_obj.get("id", "?"))
        def _get(field, default=""):
            if isinstance(rec_obj, dict):
                return rec_obj.get(field, default)
            return getattr(rec_obj, field, default)
        kind  = _get("kind", "")
        if hasattr(kind, "value"): kind = kind.value
        pub   = self._agent_label(_get("publisher_id", None))
        helper= self._agent_label(_get("accepted_by", None))
        reward= float(_get("reward", 0.0) or 0.0)
        brief = str(_get("brief", "") or "")
        xy    = self._fmt_xy(_get("location_xy", None))
        lines = [
            f"[Req #{rid}]",
            f"  Kind   : {kind}",
            f"  From   : {pub}",
            f"  Helper : {helper}",
            f"  Where  : {xy}",
            f"  $$     : ${reward:.2f}",
            f"  Status : Completed",
        ]
        if brief:
            lines.append(f"  Brief  : {brief}")
        return "\n".join(lines)

    # --- Build pages ---
    def _build_board_tab(self):
        self.board_content = self._reset_scroll_content(self.board_scroll)
        lay = QGridLayout(self.board_content); lay.setContentsMargins(10,10,10,10); lay.setSpacing(12)

        rows = self._comms.list_board() if (self._comms and hasattr(self._comms, "list_board")) else []
        if not rows:
            lay.addWidget(self._text_card("No requests on board.", self.board_content), 0, 0)
            return

        col = row = 0; cols = 2
        for r in rows:
            lay.addWidget(self._text_card(self._fmt_board_card(r), self.board_content), row, col)
            col += 1
            if col >= cols: col = 0; row += 1

    def _build_active_tab(self):
        self.active_content = self._reset_scroll_content(self.active_scroll)
        lay = QGridLayout(self.active_content); lay.setContentsMargins(10,10,10,10); lay.setSpacing(12)

        active_map: Dict[int, Any] = {}
        if self._comms:
            if hasattr(self._comms, "list_active"):
                rows = self._comms.list_active()
                if isinstance(rows, dict):
                    active_map = rows
                elif isinstance(rows, list):
                    for obj in rows:
                        rid = getattr(obj, "req_id", None)
                        if rid is None and isinstance(obj, dict):
                            rid = obj.get("req_id") or obj.get("id")
                        if rid is not None:
                            active_map[int(rid)] = obj
            else:
                active_map = dict(getattr(self._comms, "_active", {}) or {})

        if not active_map:
            lay.addWidget(self._text_card("No active requests.", self.active_content), 0, 0)
            return

        now_sim = self._clock.now_sim()
        col = row = 0; cols = 2
        for rid in sorted(active_map.keys()):
            obj = active_map[rid]
            lay.addWidget(self._text_card(self._fmt_active_card(obj, now_sim), self.active_content), row, col)
            col += 1
            if col >= cols: col = 0; row += 1

    def _build_completed_tab(self):
        self.done_content = self._reset_scroll_content(self.done_scroll)
        lay = QGridLayout(self.done_content); lay.setContentsMargins(10,10,10,10); lay.setSpacing(12)

        completed = []
        if self._comms:
            if hasattr(self._comms, "list_completed"):
                completed = list(self._comms.list_completed() or [])
            elif hasattr(self._comms, "_completed"):
                completed = list(getattr(self._comms, "_completed", []) or [])

        if not completed:
            lay.addWidget(self._text_card("No completed requests.", self.done_content), 0, 0)
            return

        col = row = 0; cols = 2
        for obj in completed:
            lay.addWidget(self._text_card(self._fmt_completed_card(obj), self.done_content), row, col)
            col += 1
            if col >= cols: col = 0; row += 1

    def refresh_all(self):
        self.setUpdatesEnabled(False)
        try:
            self._build_board_tab()
            self._build_active_tab()
            self._build_completed_tab()
        finally:
            self.setUpdatesEnabled(True)


# ---------------------- Agent Bag window ----------------------
class AgentBagDialog(QDialog):
    CARD_STYLE = """
        QFrame { background:#FAFAFA; border:1px solid #222; border-radius:8px; }
        QLabel { color:#111; }
    """
    def __init__(self, parent=None, agents: List[Any]=None):
        super().__init__(parent)
        self.setWindowTitle("Agent Bags")
        self.resize(900, 560)
        self._agents: List[Any] = list(agents or [])

        # Top bar
        top = QWidget(self)
        hb  = QHBoxLayout(top); hb.setContentsMargins(4,4,4,4); hb.setSpacing(8)
        self.btn_refresh = QPushButton("Refresh", top)
        hb.addStretch(1); hb.addWidget(self.btn_refresh)

        self.tabs = QTabWidget(self); self.tabs.setDocumentMode(True)

        root = QVBoxLayout(self)
        root.addWidget(top); root.addWidget(self.tabs)

        self.btn_refresh.clicked.connect(self.refresh_all)
        self.refresh_all()

    def set_agents(self, agents: List[Any]):
        self._agents = list(agents or [])
        self.refresh_all()

    def _text_card(self, text: str, parent: QWidget) -> QFrame:
        fr = QFrame(parent); fr.setStyleSheet(self.CARD_STYLE)
        v = QVBoxLayout(fr); v.setContentsMargins(10,8,10,8); v.setSpacing(6)
        lab = QLabel(fr)
        lab.setText(text); lab.setTextFormat(Qt.PlainText); lab.setWordWrap(True)
        f = lab.font(); f.setFamily("Consolas"); f.setPointSize(9); lab.setFont(f)
        lab.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
        v.addWidget(lab, alignment=Qt.AlignLeft | Qt.AlignTop)
        return fr

    def _fmt_slot_text(self, label: str, items: List[Any]) -> str:
        lines = [f"  {label}:"]
        if not items:
            lines.append("    (empty)")
            return "\n".join(lines)
        for it in items:
            name = getattr(it, "name", str(it))
            t = float(getattr(it, "temp_c", float("nan")))
            t_str = f"{t:.1f}°C" if (t == t) else "N/A"
            oid = getattr(it, "order_id", getattr(it, "source_order_id", None))
            src = f" (order #{oid})" if oid is not None else ""
            lines.append(f"    - {name}{src}  T={t_str}")
        return "\n".join(lines)

    def _bag_text(self, dm) -> str:
        bag = getattr(dm, "insulated_bag", None)
        if not bag:
            return "No insulated bag."
        slots = bag.get_slots() if hasattr(bag, "get_slots") else getattr(bag, "slots", None)
        if isinstance(slots, dict):
            lines = [f"Agent {getattr(dm,'_viewer_agent_id',getattr(dm,'name','?'))} — Bag"]
            labels = [str(x) for x in getattr(bag, "labels", slots.keys())]
            for lab in labels:
                items = list(slots.get(lab, []) or [])
                lines.append(self._fmt_slot_text(str(lab), items))
            return "\n".join(lines)
        text = bag.list_items()
        if not text:
            text = "(empty)"
        head = f"Agent {getattr(dm,'_viewer_agent_id',getattr(dm,'name','?'))} — Bag\n"
        return head + text

    def refresh_all(self):
        keep_idx = self.tabs.currentIndex()
        while self.tabs.count():
            w = self.tabs.widget(0); self.tabs.removeTab(0); w.deleteLater()
        for dm in self._agents:
            page = QWidget(self.tabs)
            sa = QScrollArea(page); sa.setWidgetResizable(True)
            content = QWidget(sa); sa.setWidget(content)
            v = QVBoxLayout(page); v.setContentsMargins(0,0,0,0); v.addWidget(sa)
            lay = QVBoxLayout(content); lay.setContentsMargins(10,10,10,10); lay.setSpacing(12)
            lay.addWidget(self._text_card(self._bag_text(dm), content))
            aid = str(getattr(dm, "_viewer_agent_id", getattr(dm, "name", "?")))
            self.tabs.addTab(page, f"Agent {aid}")
        if 0 <= keep_idx < self.tabs.count():
            self.tabs.setCurrentIndex(keep_idx)


# ---------------------- SimWorld camera window ----------------------
class CamerasDialog(QDialog):
    """
    Show per-agent first-person camera views (camera_id = agent_id).
    Periodically pulls frames via communicator.get_camera_observation(...) and updates smoothly.
    """
    def __init__(self, parent=None, agents: List[Any]=None, viewmode: str = "lit", poll_ms: int = 200):
        super().__init__(parent)
        self.setWindowTitle("SimWorld - Agent Cameras")
        self.resize(1100, 720)
        self._agents: List[Any] = list(agents or [])
        self._viewmode = str(viewmode)
        self._poll_ms = int(max(30, poll_ms))
        self._labels: Dict[str, QLabel] = {}
        self._pix_last: Dict[str, QPixmap] = {}
        self._comm = self._resolve_comm()

        top = QWidget(self)
        hb = QHBoxLayout(top); hb.setContentsMargins(6, 6, 6, 6); hb.setSpacing(8)
        self.btn_refresh = QPushButton("Refresh Layout", top)
        hb.addStretch(1); hb.addWidget(self.btn_refresh)

        self.scroll = QScrollArea(self); self.scroll.setWidgetResizable(True)
        self.content = QWidget(self.scroll); self.scroll.setWidget(self.content)
        self.grid = QGridLayout(self.content); self.grid.setContentsMargins(10,10,10,10); self.grid.setSpacing(10)

        root = QVBoxLayout(self); root.addWidget(top); root.addWidget(self.scroll)

        self.btn_refresh.clicked.connect(self._rebuild_grid)

        self._timer = QTimer(self)
        self._timer.setInterval(self._poll_ms)
        self._timer.timeout.connect(self._update_all)
        self._rebuild_grid()
        self._timer.start()

    def _resolve_comm(self):
        for dm in self._agents:
            ue = getattr(dm, "_ue", None)
            if ue is not None:
                return ue
        return None

    def set_agents(self, agents: List[Any]):
        self._agents = list(agents or [])
        self._comm = self._resolve_comm()
        self._rebuild_grid()

    # ---- UI construction and updates ----
    def _rebuild_grid(self):
        while self.grid.count():
            it = self.grid.takeAt(0)
            w = it.widget()
            if w: w.deleteLater()
        self._labels.clear()
        self._pix_last.clear()

        cols = 3
        row = col = 0
        for idx, dm in enumerate(self._agents):
            aid = str(getattr(dm, "_viewer_agent_id", getattr(dm, "name", "?")))
            box = QFrame(self.content)
            box.setStyleSheet("QFrame{background:#FAFAFA;border:1px solid #222;border-radius:6px;}")
            v = QVBoxLayout(box); v.setContentsMargins(8,8,8,8); v.setSpacing(6)

            title = QLabel(box); title.setText(f"Agent {aid}")
            tf = QFont(); tf.setPointSize(10); tf.setBold(True); title.setFont(tf)
            v.addWidget(title, alignment=Qt.AlignLeft)

            lab = QLabel(box); lab.setAlignment(Qt.AlignCenter)
            lab.setFixedSize(320, 180)
            lab.setStyleSheet("QLabel{background:#000; color:#ddd;}")
            lab.setText("loading...")
            v.addWidget(lab)

            self._labels[aid] = lab
            self.grid.addWidget(box, row, col)

            col += 1
            if col >= cols:
                col = 0; row += 1

    def _camera_id_of(self, aid: str, idx: int) -> int:
        try:
            return int(aid)
        except Exception:
            return idx

    def _np_to_qpixmap(self, img: np.ndarray) -> Optional[QPixmap]:
        if img is None or not isinstance(img, np.ndarray): return None
        h, w = img.shape[:2]
        if img.ndim == 2:
            qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
        elif img.shape[2] == 3:
            if cv2 is not None:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                rgb = img[:, :, ::-1].copy()
            qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        elif img.shape[2] == 4:
            qimg = QImage(img.data, w, h, w * 4, QImage.Format_RGBA8888)
        else:
            return None
        return QPixmap.fromImage(qimg.copy())

    def _update_all(self):
        if self._comm is None or not self._labels:
            return
        for idx, dm in enumerate(self._agents):
            aid = str(getattr(dm, "_viewer_agent_id", getattr(dm, "name", "?")))
            lab = self._labels.get(aid)
            if lab is None:
                continue
            pix = None
            try:
                img = self._comm.get_camera_observation(self._camera_id_of(aid, idx), viewmode=self._viewmode)
                pix = self._np_to_qpixmap(img)
            except Exception:
                pix = None
            if pix is not None:
                scaled = pix.scaled(lab.width(), lab.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                lab.setPixmap(scaled)
                self._pix_last[aid] = scaled
            else:
                last = self._pix_last.get(aid)
                if last is not None:
                    lab.setPixmap(last)


# ---------------------- Agents status window ----------------------
class AgentsDialog(QDialog):
    CARD_STYLE = """
        QFrame { background:#FAFAFA; border:1px solid #222; border-radius:8px; }
        QLabel { color:#111; }
    """
    def __init__(self, parent=None, agents: List[Any]=None):
        super().__init__(parent)
        self.setWindowTitle("Agents")
        self.resize(900, 560)
        self._agents: List[Any] = list(agents or [])
        self._suppress_refresh = False

        top = QWidget(self)
        hb  = QHBoxLayout(top); hb.setContentsMargins(0,0,0,0); hb.setSpacing(8)
        self.btn_refresh = QPushButton("Refresh", top)
        hb.addStretch(1); hb.addWidget(self.btn_refresh)

        self.tabs = QTabWidget(self); self.tabs.setDocumentMode(True)

        root = QVBoxLayout(self)
        root.addWidget(top); root.addWidget(self.tabs)

        self.tabs.currentChanged.connect(self._on_tab_changed)
        self.btn_refresh.clicked.connect(self.refresh_all)
        self.refresh_all()

    def _on_tab_changed(self, *_):
        if self._suppress_refresh: return
        self.refresh_all()

    def set_agents(self, agents: List[Any]):
        self._agents = list(agents or [])
        self.refresh_all()

    def _text_card(self, text: str, parent: QWidget) -> QFrame:
        fr = QFrame(parent); fr.setStyleSheet(self.CARD_STYLE)
        v = QVBoxLayout(fr); v.setContentsMargins(10,8,10,8); v.setSpacing(6)
        lab = QLabel(fr)
        lab.setText(text); lab.setTextFormat(Qt.PlainText); lab.setWordWrap(True)
        f = lab.font(); f.setFamily("Consolas"); f.setPointSize(9); lab.setFont(f)
        lab.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
        v.addWidget(lab, alignment=Qt.AlignLeft | Qt.AlignTop)
        return fr

    def _fmt_agent(self, dm) -> str:
        aid   = str(getattr(dm, "_viewer_agent_id", getattr(dm, "name", "?")))
        x     = float(getattr(dm, "x", 0.0)); y = float(getattr(dm, "y", 0.0))
        mode  = getattr(dm, "mode", None)
        modev = (mode.value if getattr(mode, "value", None) else str(mode or ""))
        spd   = float(getattr(dm, "get_current_speed_for_viewer", lambda: getattr(dm, "speed_cm_s", 0.0))())
        enpct = float(getattr(dm, "energy_pct", 0.0))
        earn  = float(getattr(dm, "earnings_total", 0.0))
        cur   = getattr(dm, "current_order", None)
        curid = getattr(cur, "id", None)
        qlen  = len(getattr(dm, "_queue", []) or [])
        completed = getattr(dm, "completed_orders", []) or []
        active = getattr(dm, "active_orders", None)
        active_n = (len(active) if isinstance(active, list) else (1 if cur else 0))
        carrying = getattr(dm, "carrying", []) or []

        lines = [
            f"Agent {aid}",
            f"  Mode     : {modev}",
            f"  Position : ({x:.1f}, {y:.1f})",
            f"  Speed    : {spd:.1f} cm/s",
            f"  Energy   : {enpct:.0f}%",
            f"  Earnings : ${earn:.2f}",
            f"  Busy     : {bool(getattr(dm, 'is_busy', lambda: False)())}    Queue: {qlen}",
            f"  Active Orders : {active_n}    Current: #{curid if curid is not None else 'None'}",
            f"  Carrying IDs  : {carrying}",
            f"  Completed     : {len(completed)}",
        ]
        if modev == "e-scooter" and getattr(dm, "e_scooter", None):
            lines.append(f"  Scooter Batt : {dm.e_scooter.battery_pct:.0f}%")
        if getattr(dm, "is_rescued", False):
            lines.append("  Status  : IN HOSPITAL RESCUE")
        return "\n".join(lines)

    def refresh_all(self):
        keep_idx = self.tabs.currentIndex()
        self._suppress_refresh = True
        while self.tabs.count():
            w = self.tabs.widget(0); self.tabs.removeTab(0); w.deleteLater()
        for dm in self._agents:
            aid = str(getattr(dm, "_viewer_agent_id", getattr(dm, "name", "?")))
            page = QWidget(self.tabs)
            from PyQt5.QtWidgets import QVBoxLayout, QScrollArea, QWidget as _Qw
            sa = QScrollArea(page); sa.setWidgetResizable(True)
            content = _Qw(sa); sa.setWidget(content)
            v = QVBoxLayout(page); v.setContentsMargins(0,0,0,0); v.addWidget(sa)
            lay = QVBoxLayout(content); lay.setContentsMargins(10,10,10,10); lay.setSpacing(12)
            lay.addWidget(self._text_card(self._fmt_agent(dm), content))
            self.tabs.addTab(page, f"Agent {aid}")
        if 0 <= keep_idx < self.tabs.count():
            self.tabs.setCurrentIndex(keep_idx)
        self._suppress_refresh = False


# ---------------------- Main observer ----------------------
class MapObserver(MapCanvasBase):
    def __init__(self, title: str = "Map Observer", clock: Optional[VirtualClock] = None):
        super().__init__(title=title)

        self._agents: Dict[str, Dict[str, Any]] = {}
        self._dm_registry: List[Any] = []
        self._order_manager = None
        self._comms = None  # CommsSystem source
        self._bus_manager = None  # BusManager source

        # Virtual time (used for non-movement events)
        self.clock: VirtualClock = clock if clock is not None else VirtualClock()
        self._last_sim_ts: float = self.clock.now_sim()

        # UI tick (does not drive movement; only for refreshing / pulling positions from UE)
        self._anim_timer = QTimer(self)
        self._anim_timer.setInterval(DEFAULT_TICK_MS)
        self._anim_timer.timeout.connect(self._on_tick)
        self._anim_timer.start()

        # Top button bar
        btn_bar = QWidget(self)
        hb = QHBoxLayout(btn_bar); hb.setContentsMargins(4, 4, 4, 4); hb.setSpacing(8)
        self.btn_show_orders = QPushButton("Show Orders", btn_bar)
        self.btn_show_orders.clicked.connect(self._on_show_orders_clicked)
        hb.addWidget(self.btn_show_orders)
        self.btn_show_agents = QPushButton("Show Agents", btn_bar)
        self.btn_show_agents.clicked.connect(self._on_show_agents_clicked)
        hb.addWidget(self.btn_show_agents)
        self.btn_show_cameras = QPushButton("Show SimWorld", btn_bar)
        self.btn_show_cameras.clicked.connect(self._on_show_cameras_clicked)
        hb.addWidget(self.btn_show_cameras)
        self.btn_show_bag = QPushButton("Show Agent Bag", btn_bar)
        self.btn_show_bag.clicked.connect(self._on_show_bag_clicked)
        hb.addWidget(self.btn_show_bag)
        self.btn_show_requests = QPushButton("Show Requests", btn_bar)
        self.btn_show_requests.clicked.connect(self._on_show_requests_clicked)
        hb.addWidget(self.btn_show_requests)
        hb.addStretch(1)
        self.vbox.addWidget(btn_bar)

        # Dialog handles
        self._orders_dialog: Optional[OrdersDialog] = None
        self._agents_dialog: Optional[AgentsDialog] = None
        self._cameras_dialog: Optional[CamerasDialog] = None
        self._bags_dialog: Optional[AgentBagDialog] = None
        self._requests_dialog: Optional[RequestsDialog] = None

        # Overlay caches
        self._escooter_badge_items: Dict[str, Optional[pg.ScatterPlotItem]] = {}
        self._car_badge_items: Dict[str, Optional[pg.ScatterPlotItem]] = {}
        self._batt_items: Dict[str, Tuple[QGraphicsRectItem, QGraphicsRectItem]] = {}
        self._energy_items: Dict[str, Tuple[QGraphicsRectItem, QGraphicsRectItem]] = {}
        self._status_text_items: Dict[str, Optional[pg.TextItem]] = {}
        self._light_items: Dict[str, Optional[QGraphicsPathItem]] = {}
        self._rescue_items: Dict[str, Optional[QGraphicsPathItem]] = {}
        self._es_text_items: Dict[str, Optional[pg.TextItem]] = {}  # "ES ..." label near parked scooter
        self._escooter_badge_items_inline: Dict[str, Optional[pg.ScatterPlotItem]] = {}
        self._escooter_badge_items_parked: Dict[str, List[pg.ScatterPlotItem]] = {}

        # Bus overlays
        self._bus_items: Dict[str, Optional[pg.ScatterPlotItem]] = {}  # bus markers
        self._bus_text_items: Dict[str, Optional[pg.TextItem]] = {}    # bus status labels

    # ---------- Top buttons ----------
    def _on_show_orders_clicked(self):
        if self._orders_dialog is None:
            self._orders_dialog = OrdersDialog(self, order_manager=self._order_manager, agents=self._dm_registry, clock=self.clock)
        self._orders_dialog.set_sources(order_manager=self._order_manager, agents=self._dm_registry, clock=self.clock)
        self._orders_dialog.show(); self._orders_dialog.raise_(); self._orders_dialog.activateWindow()

    def _on_show_agents_clicked(self):
        if self._agents_dialog is None:
            self._agents_dialog = AgentsDialog(self, agents=self._dm_registry)
        else:
            self._agents_dialog.set_agents(self._dm_registry)
        self._agents_dialog.show(); self._agents_dialog.raise_(); self._agents_dialog.activateWindow()

    def _on_show_cameras_clicked(self):
        if self._cameras_dialog is None:
            self._cameras_dialog = CamerasDialog(self, agents=self._dm_registry, viewmode="lit", poll_ms=200)
        else:
            self._cameras_dialog.set_agents(self._dm_registry)
        self._cameras_dialog.show(); self._cameras_dialog.raise_(); self._cameras_dialog.activateWindow()

    def _on_show_bag_clicked(self):
        if self._bags_dialog is None:
            self._bags_dialog = AgentBagDialog(self, agents=self._dm_registry)
        else:
            self._bags_dialog.set_agents(self._dm_registry)
        self._bags_dialog.show(); self._bags_dialog.raise_(); self._bags_dialog.activateWindow()

    def _on_show_requests_clicked(self):
        if self._requests_dialog is None:
            self._requests_dialog = RequestsDialog(self, comms=self._comms, agents=self._dm_registry, clock=self.clock)
        else:
            self._requests_dialog.set_sources(comms=self._comms, agents=self._dm_registry, clock=self.clock)
        self._requests_dialog.show(); self._requests_dialog.raise_(); self._requests_dialog.activateWindow()

    # ---------- Helper: get ViewBox ----------
    def _vb(self):
        try:
            return self.plot.getViewBox()
        except Exception:
            return getattr(self.plot, 'vb', None)

    # --- escooter/assist_scooter selection (minimal-change version) ---
    def _own_scooter(self, dm):
        return getattr(dm, "e_scooter", None)

    def _assist_scooter(self, dm):
        return getattr(dm, "assist_scooter", None)

    def _is_parked(self, s) -> bool:
        return bool(s and getattr(s, "park_xy", None))

    def _pick_inline_scooter(self, dm):
        """
        Choose the scooter that should be rendered inline with the agent:
        - If an assist scooter exists and is not parked -> use assist scooter (you are towing it)
        - Else if own scooter exists with with_owner=True and is not parked -> use own scooter
        - Else -> None (no scooter is merged into the agent marker)
        """
        s_a = self._assist_scooter(dm)
        if s_a is not None and not self._is_parked(s_a):
            return s_a

        s_o = self._own_scooter(dm)
        if s_o is not None and bool(getattr(s_o, "with_owner", False)) and not self._is_parked(s_o):
            return s_o

        return None

    def _pick_parked_scooters(self, dm) -> List[Any]:
        """
        List of scooters that should be anchored on the ground (may contain multiple):
        - Only own parked scooter is shown; parked assist scooters are rendered by their owner.
        """
        out = []
        s_o = self._own_scooter(dm)
        if self._is_parked(s_o):
            out.append(s_o)

        # To additionally show parked assist scooters here, uncomment:
        # s_a = self._assist_scooter(dm)
        # if self._is_parked(s_a): out.append(s_a)

        return out

    def _pick_all_parked_scooters(self, dm) -> List[Any]:
        out = []
        s_o = self._own_scooter(dm)
        if self._is_parked(s_o): out.append(s_o)
        s_a = self._assist_scooter(dm)
        if self._is_parked(s_a): out.append(s_a)
        return out

    def _escooter_ctx(self, rec, dm):
        """
        Return ("inline" | "parked" | None, scooter)
        - "inline": _pick_inline_scooter(dm)
        - "parked": when no inline scooter, but own scooter is parked
        - None    : nothing should be rendered here
        """
        inline = self._pick_inline_scooter(dm)
        if inline is not None:
            return ("inline", inline)

        parked_list = self._pick_parked_scooters(dm)
        if parked_list:
            return ("parked", parked_list[0])  # kept for backward compatibility

        return (None, None)

    # ---------- Wiring & registration ----------
    def attach_order_manager(self, om): self._order_manager = om
    def attach_comms(self, comms): self._comms = comms
    def attach_bus_manager(self, bus_manager): self._bus_manager = bus_manager

    def register_delivery_man(self, dm_obj: Any):
        if dm_obj not in self._dm_registry:
            self._dm_registry.append(dm_obj)
        aid = str(getattr(dm_obj, "_viewer_agent_id", getattr(dm_obj, "name", "?")))
        rec = self._agents.get(aid)
        if rec is not None:
            rec['dm'] = dm_obj
        if getattr(self, "_agents_dialog", None) is not None:
            self._agents_dialog.set_agents(self._dm_registry)
        if getattr(self, "_orders_dialog", None) is not None:
            self._orders_dialog.set_sources(order_manager=self._order_manager, agents=self._dm_registry, clock=self.clock)
        if getattr(self, "_bags_dialog", None) is not None:
            self._bags_dialog.set_agents(self._dm_registry)
        if getattr(self, "_requests_dialog", None) is not None:
            self._requests_dialog.set_sources(comms=self._comms, agents=self._dm_registry, clock=self.clock)

    # ---------- Agent visuals ----------
    def add_agent(self, agent_id: str, x: float, y: float,
                  speed_cm_s: float = 300.0, label_text: Optional[str] = None,
                  on_anim_done: Optional[Callable[..., None]] = None):
        aid = str(agent_id)
        label_text = label_text if (label_text is not None) else aid

        dot = pg.ScatterPlotItem(
            pos=[(x, y)], size=UI_AGENT_SIZE,
            brush=pg.mkBrush(AGENT_FILL),
            pen=pg.mkPen(AGENT_BORDER, width=1.6),
            symbol="o", antialias=True
        )
        dot.setZValue(AGENT_Z)

        lbl = pg.TextItem(text=str(label_text), color=AGENT_TEXT)
        lbl.setAnchor((0.5, 0.5))
        lbl.setFont(self._label_font)
        lbl.setPos(x, y)
        lbl.setZValue(Z_LABEL)

        if aid in self._agents:
            self.plot.removeItem(self._agents[aid]['dot'])
            self.plot.removeItem(self._agents[aid]['lbl'])

        self.plot.addItem(dot); self.plot.addItem(lbl)
        self._agents[aid] = dict(
            xy=(float(x), float(y)),
            speed=float(speed_cm_s),
            dot=dot, lbl=lbl,
            route=[],             # used only for arrival detection and path highlight
            dest=None,            # (tx, ty)
            on_done=on_anim_done,
            dm=None,
            allow_interrupt=True
        )

        self._refresh_overlays(aid)

    def remove_agent(self, agent_id: str):
        aid = str(agent_id)
        rec = self._agents.pop(aid, None)
        if rec:
            for key in ['dot', 'lbl']:
                self.plot.removeItem(rec[key])
        self._remove_escooter_badge(aid)
        self._remove_car_badge(aid)
        self._remove_batt_box(aid)
        self._remove_energy_box(aid)
        self._remove_status_text(aid)
        self._remove_lightning(aid)
        self._remove_rescue(aid)
        self._remove_es_text(aid)

    def set_agent_xy(self, agent_id: str, x: float, y: float):
        aid = str(agent_id)
        rec = self._agents.get(aid)
        if not rec:
            self.add_agent(agent_id, x, y); return
        rec['xy'] = (float(x), float(y))
        rec['dot'].setData(pos=[(x, y)])
        rec['lbl'].setPos(x, y)
        rec['lbl'].setZValue(Z_LABEL)
        self._refresh_overlays(aid)

    def set_agent_label(self, agent_id: str, text: str):
        rec = self._agents.get(str(agent_id))
        if rec:
            rec['lbl'].setText(str(text))
            rec['lbl'].setZValue(Z_LABEL)

    def set_speed(self, agent_id: str, speed_cm_s: float):
        rec = self._agents.get(str(agent_id))
        if rec: rec['speed'] = float(speed_cm_s)

    # ---------- Routing & movement ----------
    def go_to_xy(self, agent_id: str, route: List[Tuple[float, float]], *,
             arrive_tolerance_cm: float = 300.0,
             allow_interrupt: bool = True, show_path_ms: int = 2000):
        rec = self._agents.get(str(agent_id))
        if not rec or not route:
            return

        self.clear_path_highlight()
        if len(route) >= 2:
            self.highlight_path(route)
        if show_path_ms and show_path_ms > 0:
            QTimer.singleShot(int(show_path_ms), self.clear_path_highlight)

        rec['route'] = list(route)
        rec['dest'] = route[-1]
        rec['dest_tol'] = float(arrive_tolerance_cm)   # record tolerance for arrival detection
        rec['allow_interrupt'] = bool(allow_interrupt)

        sx, sy = route[0]
        self.set_agent_xy(str(agent_id), sx, sy)

    def is_busy(self, agent_id: str) -> bool:
        rec = self._agents.get(str(agent_id))
        return bool(rec and rec.get('dest') is not None)

    def _scooter_with_me(self, dm, s) -> bool:
        if not s:
            return False

        # Assist scooter: as long as it is not parked, it is considered "with me" (being towed)
        if s is getattr(dm, "assist_scooter", None):
            return not getattr(s, "park_xy", None)

        # Own scooter: with_owner=True and not parked means "with me"
        if s is getattr(dm, "e_scooter", None):
            if getattr(s, "with_owner", None) is not None:
                return bool(getattr(s, "with_owner")) and not getattr(s, "park_xy", None)
            # Backward compatibility: when with_owner is missing, treat non-parked as "with me"
            return not getattr(s, "park_xy", None)

        return False

    # ---------- Tick ----------
    def _on_tick(self):
        now_sim = self.clock.now_sim()
        _ = now_sim - self._last_sim_ts
        self._last_sim_ts = now_sim

        # 1) Legacy mode: if agent has no own time loop thread, UI thread can drive time events.
        # for aid, rec in list(self._agents.items()):
        #     dm = rec.get('dm')
        #     if dm is not None and getattr(dm, "_loop_thread", None) is None and hasattr(dm, "poll_time_events"):
        #         dm.poll_time_events()

        # 2) Pull position from UE, sync state, and handle interruption/arrival
        for aid, rec in list(self._agents.items()):
            dm = rec.get('dm')
            if dm is None or not getattr(dm, "_ue", None):
                continue

            # Resolve UE actor name
            ue_name = str(dm.agent_id)

            # Pull position (supports dict or tuple return from UE)
            try:
                posdir = dm._ue.get_position_and_direction(ue_name)
            except Exception:
                posdir = {}

            if isinstance(posdir, dict):
                posyaw = posdir.get(ue_name, None)
                if isinstance(posyaw, (tuple, list)) and len(posyaw) >= 2:
                    pos, _yaw = posyaw[0], posyaw[1]
                else:
                    pos, _yaw = (None, None)
            elif isinstance(posdir, (tuple, list)) and len(posdir) >= 2:
                pos, _yaw = posdir[0], posdir[1]
            else:
                pos, _yaw = (None, None)

            if not pos:
                continue

            px_old, py_old = rec['xy']
            px_new, py_new = float(pos.x), float(pos.y)
            moved_cm = math.hypot(px_new - px_old, py_new - py_old)
            if moved_cm > 0.0:
                try:
                    dm.on_move_consumed(moved_cm)
                except Exception:
                    pass

            self.set_agent_xy(aid, px_new, py_new)
            try:
                dm.x, dm.y = px_new, py_new
            except Exception:
                pass

            # Interrupt
            if rec.get('allow_interrupt', True):
                try:
                    if dm._interrupt_move_flag:
                        dm._interrupt_move_flag = False
                        try:
                            dm._ue.stop_go_to(ue_name)
                        except Exception:
                            pass
                        rec['route'] = []
                        rec['dest']  = None
                        try:
                            cb = rec.get('on_done')
                            if callable(cb):
                                cb(aid, px_new, py_new, "blocked")
                        except Exception:
                            pass
                        continue
                except Exception:
                    pass

            # Arrival (using per-move tolerance)
            dest = rec.get('dest')
            if dest is not None:
                tx, ty = dest
                tol = float(rec.get('dest_tol', 300.0))
                if math.hypot(px_new - tx, py_new - ty) <= tol:
                    rec['route'] = []
                    rec['dest']  = None
                    try:
                        cb = rec.get('on_done')
                        if callable(cb):
                            cb(aid, tx, ty, "move")
                    except Exception:
                        pass

        # 3) Refresh overlays
        for aid in list(self._agents.keys()):
            self._update_escooter_badge(aid)
            self._update_batt_box(aid)
            self._update_energy_box(aid)
            self._update_status_text(aid)
            self._update_lightning(aid)
            self._update_es_text(aid)
            self._update_rescue(aid)
            self._update_car_badge(aid)

        # 4) Update buses
        if self._bus_manager:
            self._bus_manager.update_all_buses()
            self._update_all_buses()

    # ---------- Overlay aggregation ----------
    def _refresh_overlays(self, aid: str):
        self._update_escooter_badge(aid)
        self._update_batt_box(aid)
        self._update_energy_box(aid)
        self._update_status_text(aid)
        self._update_lightning(aid)
        self._update_rescue(aid)
        self._update_car_badge(aid)

    # ---------- Overlay: e-scooter badge ----------
    def _remove_escooter_badge(self, aid: str):
        it = self._escooter_badge_items_inline.pop(aid, None)
        if it:
            try: self.plot.removeItem(it)
            except Exception: pass

        lst = self._escooter_badge_items_parked.pop(aid, None)
        if lst:
            for it in lst:
                try: self.plot.removeItem(it)
                except Exception: pass

    def _update_escooter_badge(self, aid: str):
        rec = self._agents.get(aid)
        if not rec: return
        dm = rec.get('dm')
        if dm is None:
            self._remove_escooter_badge(aid); return

        # ① Inline scooter merged with the agent marker
        inline = self._pick_inline_scooter(dm)
        if inline is not None:
            x, y = rec['xy']
            pos_inline = (x + BADGE_OFFSET_X_CM, y + BADGE_OFFSET_Y_CM)
            it = self._escooter_badge_items_inline.get(aid)
            if it is None:
                it = pg.ScatterPlotItem(
                    pos=[pos_inline], size=BADGE_SIZE,
                    brush=pg.mkBrush(COLOR_BADGE_SCOOTER),
                    pen=pg.mkPen("#222", width=1.2), symbol="s", antialias=True
                )
                it.setZValue(Z_BADGE); self.plot.addItem(it)
                self._escooter_badge_items_inline[aid] = it
            else:
                it.setData(pos=[pos_inline])
                it.setBrush(pg.mkBrush(COLOR_BADGE_SCOOTER))
        else:
            it = self._escooter_badge_items_inline.pop(aid, None)
            if it:
                try: self.plot.removeItem(it)
                except Exception: pass

        # ② Parked scooters (currently only own scooter)
        parked = self._pick_parked_scooters(dm)
        old_list = self._escooter_badge_items_parked.get(aid, [])
        # Shrink or expand to match count
        if len(old_list) > len(parked):
            # Remove extras
            for it in old_list[len(parked):]:
                try: self.plot.removeItem(it)
                except Exception: pass
            old_list = old_list[:len(parked)]
        elif len(old_list) < len(parked):
            # Add more
            for _ in range(len(parked) - len(old_list)):
                it = pg.ScatterPlotItem(
                    pos=[(0,0)], size=BADGE_SIZE,
                    brush=pg.mkBrush(COLOR_BADGE_SCOOTER),
                    pen=pg.mkPen("#222", width=1.2), symbol="s", antialias=True
                )
                it.setZValue(Z_BADGE); self.plot.addItem(it)
                old_list.append(it)

        # Position each parked scooter badge
        for it, s in zip(old_list, parked):
            pos_park = tuple(s.park_xy)
            it.setData(pos=[pos_park])
            it.setBrush(pg.mkBrush(COLOR_BADGE_SCOOTER))

        if parked:
            self._escooter_badge_items_parked[aid] = old_list
        else:
            if old_list:
                for it in old_list:
                    try: self.plot.removeItem(it)
                    except Exception: pass
            self._escooter_badge_items_parked.pop(aid, None)

    # ---------- Overlay: car badge ----------
    def _remove_car_badge(self, aid: str):
        it = self._car_badge_items.pop(aid, None)
        if it:
            try: self.plot.removeItem(it)
            except Exception: pass

    def _update_car_badge(self, aid: str):
        rec = self._agents.get(aid)
        if not rec: return
        dm = rec.get('dm')
        if dm is None or not getattr(dm, "car", None):
            self._remove_car_badge(aid); return

        car = dm.car
        if getattr(car, "park_xy", None):
            anchor_xy = tuple(car.park_xy)
        else:
            x, y = rec['xy']
            anchor_xy = (x + BADGE_OFFSET_X_CM, y + BADGE_OFFSET_Y_CM)

        it = self._car_badge_items.get(aid)
        if it is None:
            it = pg.ScatterPlotItem(
                pos=[anchor_xy], size=BADGE_SIZE,
                brush=pg.mkBrush(COLOR_BADGE_CAR),
                pen=pg.mkPen("#222", width=1.2),
                symbol="s", antialias=True
            )
            it.setZValue(Z_BADGE)
            self.plot.addItem(it)
            self._car_badge_items[aid] = it
        else:
            it.setData(pos=[anchor_xy])
            it.setBrush(pg.mkBrush(COLOR_BADGE_CAR))

    # ---------- Overlay: scooter battery bar (attached to agent or parked scooter) ----------
    def _remove_batt_box(self, aid: str):
        pair = self._batt_items.pop(aid, None)
        if pair:
            bg, fg = pair
            try: self.plot.removeItem(bg)
            except Exception: pass
            try: self.plot.removeItem(fg)
            except Exception: pass

    def _batt_anchor_xy(self, rec, dm) -> Tuple[float, float]:
        base_x, base_y = rec['xy']
        state, scoot = self._escooter_ctx(rec, dm)
        if state == "parked" and scoot and getattr(scoot, "park_xy", None):
            base_x, base_y = scoot.park_xy
        # Otherwise (inline / None) anchor at agent position
        x0 = base_x + BATT_OFFSET_X_CM
        y0 = base_y + BATT_OFFSET_Y_CM
        return x0, y0

    def _update_batt_box(self, aid: str):
        rec = self._agents.get(aid)
        if not rec: return
        dm = rec.get('dm')
        if dm is None:
            self._remove_batt_box(aid); return

        # Select scooter whose battery should be shown: inline preferred, otherwise first parked own scooter
        inline = self._pick_inline_scooter(dm)
        parked_list = self._pick_parked_scooters(dm)
        scoot_ui = inline if inline is not None else (parked_list[0] if parked_list else None)
        if scoot_ui is None:
            self._remove_batt_box(aid); return

        # Anchor: inline -> agent; parked -> scooter park_xy
        if scoot_ui is inline:
            base_xy = tuple(rec['xy'])
        else:
            base_xy = tuple(scoot_ui.park_xy)

        try:
            pct01 = max(0.0, min(1.0, float(getattr(scoot_ui, "battery_pct", 0.0)) / 100.0))
        except Exception:
            pct01 = 0.0

        x0 = base_xy[0] + BATT_OFFSET_X_CM
        y0 = base_xy[1] + BATT_OFFSET_Y_CM
        w, h = BATT_BAR_W_CM, BATT_BAR_H_CM
        vb = self._vb()

        pair = self._batt_items.get(aid)
        if pair is None:
            bg = QGraphicsRectItem(x0, y0, w, h)
            fg = QGraphicsRectItem(x0, y0, w * pct01, h)
            if vb is not None:
                bg.setParentItem(vb); fg.setParentItem(vb)
            bg.setBrush(pg.mkBrush(COLOR_BAR_BG))
            bg.setPen(pg.mkPen(COLOR_STROKE, width=1.0))
            bg.setZValue(Z_BATT)
            fg.setBrush(pg.mkBrush(COLOR_BAR_FG))
            fg.setPen(pg.mkPen(COLOR_STROKE, width=0.0))
            fg.setZValue(Z_BATT + 1)
            self.plot.addItem(bg); self.plot.addItem(fg)
            self._batt_items[aid] = (bg, fg)
        else:
            bg, fg = pair
            bg.setRect(x0, y0, w, h)
            fg.setRect(x0, y0, w * pct01, h)

    # ---------- Overlay: energy (stamina) bar (always attached to agent) ----------
    def _remove_energy_box(self, aid: str):
        pair = self._energy_items.pop(aid, None)
        if pair:
            bg, fg = pair
            try: self.plot.removeItem(bg)
            except Exception: pass
            try: self.plot.removeItem(fg)
            except Exception: pass

    def _update_energy_box(self, aid: str):
        rec = self._agents.get(aid)
        if not rec: return
        dm = rec.get('dm')
        if dm is None:
            self._remove_energy_box(aid); return

        try:
            pct01 = max(0.0, min(1.0, float(dm.energy_pct) / 100.0))
        except Exception:
            pct01 = 0.0

        base_x, base_y = rec['xy']
        x0 = base_x + ENERGY_OFFSET_X_CM
        y0 = base_y + ENERGY_OFFSET_Y_CM
        w = ENERGY_BAR_W_CM
        h = ENERGY_BAR_H_CM

        vb = self._vb()

        pair = self._energy_items.get(aid)
        if pair is None:
            bg = QGraphicsRectItem(x0, y0, w, h)
            fg = QGraphicsRectItem(x0, y0, w * pct01, h)
            if vb is not None:
                bg.setParentItem(vb); fg.setParentItem(vb)
            bg.setBrush(pg.mkBrush(COLOR_ENERGY_BAR_BG))
            bg.setPen(pg.mkPen(COLOR_STROKE, width=1.0))
            bg.setZValue(Z_ENERGY)
            fg.setBrush(pg.mkBrush(COLOR_ENERGY_BAR_FG))
            fg.setPen(pg.mkPen(COLOR_STROKE, width=0.0))
            fg.setZValue(Z_ENERGY + 1)
            self.plot.addItem(bg); self.plot.addItem(fg)
            self._energy_items[aid] = (bg, fg)
        else:
            bg, fg = pair
            bg.setRect(x0, y0, w, h)
            fg.setRect(x0, y0, w * pct01, h)

    # ---------- Overlay: combined text on the right (same row as stamina bar) ----------
    def _remove_status_text(self, aid: str):
        it = self._status_text_items.pop(aid, None)
        if it:
            try: self.plot.removeItem(it)
            except Exception: pass

    def _scooter_is_parked(self, dm) -> bool:
        scoot = getattr(dm, "e_scooter", None)
        return bool(scoot and getattr(scoot, "park_xy", None))

    def _update_status_text(self, aid: str):
        rec = self._agents.get(aid)
        if not rec:
            return
        dm = rec.get('dm')
        if dm is None:
            self._remove_status_text(aid)
            return

        # --- DeliveryMan energy (with rest target if any) ---
        try:
            dm_cur = int(round(float(getattr(dm, "energy_pct", 0.0))))
        except Exception:
            dm_cur = 0
        dm_txt = f"DM {dm_cur}%"
        rp = None
        try:
            rp = dm.resting_progress()
        except Exception:
            rp = None
        if rp:
            try:
                dm_tgt = int(round(float(rp.get("target_pct", dm_cur))))
            except Exception:
                dm_tgt = dm_cur
            if dm_tgt != dm_cur:
                dm_txt = f"DM {dm_cur}% \u2192 {dm_tgt}%"

        # When inline scooter exists, append ES battery info on same line
        inline = self._pick_inline_scooter(dm)
        combined = dm_txt
        if inline is not None:
            try:
                es_cur = int(round(float(getattr(inline, "battery_pct", 0.0))))
            except Exception:
                es_cur = 0
            es_txt = f"ES {es_cur}%"
            # Only own scooter shows charging target
            if inline is getattr(dm, "e_scooter", None):
                cp = None
                try:
                    cp = dm.charging_progress()
                except Exception:
                    cp = None
                if isinstance(cp, dict):
                    try:
                        es_tgt = int(round(float(cp.get("target_pct", es_cur))))
                    except Exception:
                        es_tgt = es_cur
                    if es_tgt != es_cur:
                        es_txt = f"ES {es_cur}% \u2192 {es_tgt}%"
            combined = f"{dm_txt} | {es_txt}"

        # Position text to the right of the energy bar, anchored to the agent
        base_x, base_y = rec['xy']
        x0 = base_x + ENERGY_OFFSET_X_CM + ENERGY_BAR_W_CM + LABEL_MARGIN_X_CM
        y0 = base_y + ENERGY_OFFSET_Y_CM + ENERGY_BAR_H_CM * 0.5

        it = self._status_text_items.get(aid)
        if it is None:
            it = pg.TextItem(text=combined, color=(0, 0, 0))
            vb = self._vb()
            if vb is not None:
                it.setParentItem(vb)
            f = QFont(); f.setPointSize(LABEL_TEXT_PT); it.setFont(f)
            it.setZValue(Z_STATUS)
            it.setPos(x0, y0)
            self.plot.addItem(it)
            self._status_text_items[aid] = it
        else:
            it.setText(combined)
            it.setPos(x0, y0)
            it.setZValue(Z_STATUS)

    # ---------- Overlay: charging lightning icon ----------
    def _remove_lightning(self, aid: str):
        it = self._light_items.pop(aid, None)
        if it:
            try: self.plot.removeItem(it)
            except Exception: pass

    def _update_lightning(self, aid: str):
        rec = self._agents.get(aid)
        if not rec: return
        dm = rec.get('dm')
        if dm is None:
            self._remove_lightning(aid); return

        try:
            cp = dm.charging_progress()
        except Exception:
            cp = None
        if not (isinstance(cp, dict) and cp.get("xy") is not None):
            self._remove_lightning(aid); return

        cx, cy = tuple(cp["xy"])

        # Only consider scooters this observer is responsible for: inline + own parked
        inline = self._pick_inline_scooter(dm)
        parked_list = self._pick_parked_scooters(dm)

        candidates = []
        if inline is not None:
            candidates.append(("inline", inline))
        for s in parked_list:
            candidates.append(("parked", s))

        if not candidates:
            self._remove_lightning(aid); return

        # Pick scooter closest to the charging point
        best = None; best_d = float("inf")
        for kind, s in candidates:
            if kind == "inline":
                sxy = tuple(rec['xy'])
            else:  # parked
                sxy = tuple(s.park_xy)
            d = math.hypot(sxy[0]-cx, sxy[1]-cy)
            if d < best_d:
                best_d, best = d, s

        if best is None:
            self._remove_lightning(aid); return

        # Draw lightning icon to the right of the corresponding battery bar
        x_bar = cx + BATT_OFFSET_X_CM
        y_bar = cy + BATT_OFFSET_Y_CM

        w_bar = BATT_BAR_W_CM; h_bar = BATT_BAR_H_CM
        margin = 0.20 * h_bar; size = 0.80 * h_bar; bolt_w = 0.60 * size
        cx_draw = x_bar + w_bar + margin + bolt_w * 0.5
        cy_draw = y_bar + h_bar * 0.5

        def _bolt_path(cx0: float, cy0: float, w: float, h: float) -> QPainterPath:
            pts = [(-0.45,-0.30),(0.25,-0.30),(-0.05,0.05),(0.45,0.05),(-0.25,0.40),(0.00,0.00)]
            path = QPainterPath()
            x0 = cx0 + pts[0][0] * w
            y0 = cy0 + pts[0][1] * h
            path.moveTo(x0, y0)
            for (ux, uy) in pts[1:]:
                path.lineTo(cx0 + ux * w, cy0 + uy * h)
            path.closeSubpath()
            return path

        path = _bolt_path(cx_draw, cy_draw, bolt_w, size)
        it = self._light_items.get(aid)
        if it is None:
            it = QGraphicsPathItem(path)
            vb = self._vb()
            if vb is not None: it.setParentItem(vb)
            it.setBrush(pg.mkBrush(COLOR_LIGHTNING))
            it.setPen(pg.mkPen(COLOR_STROKE, width=0.0))
            it.setZValue(Z_LIGHT)
            self.plot.addItem(it)
            self._light_items[aid] = it
        else:
            it.setPath(path); it.setZValue(Z_LIGHT)

    # ---------- Overlay: ES text label near parked scooter ----------
    def _remove_es_text(self, aid: str):
        it = self._es_text_items.pop(aid, None)
        if it:
            try:
                self.plot.removeItem(it)
            except Exception:
                pass

    def _update_es_text(self, aid: str):
        rec = self._agents.get(aid)
        if not rec:
            return
        dm = rec.get('dm')
        if dm is None:
            self._remove_es_text(aid); return

        # Only consider parked scooters we are responsible for (own scooters)
        parked_list = self._pick_parked_scooters(dm)
        if not parked_list:
            self._remove_es_text(aid); return

        # Prefer own scooter if multiple are present
        own = getattr(dm, "e_scooter", None)
        chosen = own if (own in parked_list) else parked_list[0]

        # Safely read park_xy
        try:
            sx, sy = map(float, chosen.park_xy)
            if not (math.isfinite(sx) and math.isfinite(sy)):
                raise ValueError
        except Exception:
            self._remove_es_text(aid); return

        # Current battery percentage
        try:
            es_cur = int(round(float(getattr(chosen, "battery_pct", 0.0))))
        except Exception:
            es_cur = 0
        es_txt = f"ES {es_cur}%"

        # If the scooter is near the charging point, show "-> target%"
        try:
            cp = dm.charging_progress()
        except Exception:
            cp = None
        if isinstance(cp, dict) and isinstance(cp.get("xy"), (tuple, list)):
            try:
                cx, cy = map(float, cp["xy"])
                SAME_TOL = 320.0
                if math.hypot(sx - cx, sy - cy) <= SAME_TOL:
                    es_tgt = int(round(float(cp.get("target_pct", es_cur))))
                    if es_tgt != es_cur:
                        es_txt = f"ES {es_cur}% \u2192 {es_tgt}%"
            except Exception:
                pass

        # Anchor text near the parked scooter (aligned with its battery bar)
        x_bar = sx + BATT_OFFSET_X_CM
        y_bar = sy + BATT_OFFSET_Y_CM
        w_bar = BATT_BAR_W_CM
        h_bar = BATT_BAR_H_CM

        x_txt = x_bar + w_bar + LABEL_MARGIN_X_CM * 0.6
        y_txt = y_bar + h_bar * 0.5

        # Avoid overlapping with the DM status text line
        ax, ay = rec['xy']
        dm_x_txt = ax + ENERGY_OFFSET_X_CM + ENERGY_BAR_W_CM + LABEL_MARGIN_X_CM
        dm_y_txt = ay + ENERGY_OFFSET_Y_CM + ENERGY_BAR_H_CM * 0.5
        if abs(x_txt - dm_x_txt) < 300.0 and abs(y_txt - dm_y_txt) < 240.0:
            y_txt = y_bar + h_bar + 220.0

        it = self._es_text_items.get(aid)
        if it is None:
            it = pg.TextItem(text=es_txt, color=(0, 0, 0))
            vb = self._vb()
            if vb is not None: it.setParentItem(vb)
            f = QFont(); f.setPointSize(LABEL_TEXT_PT); it.setFont(f)
            it.setZValue(Z_STATUS)
            it.setPos(x_txt, y_txt)
            self.plot.addItem(it)
            self._es_text_items[aid] = it
        else:
            it.setText(es_txt)
            it.setPos(x_txt, y_txt)
            it.setZValue(Z_STATUS)

    # ---------- Overlay: hospital rescue indicator ----------
    def _remove_rescue(self, aid: str):
        it = self._rescue_items.pop(aid, None)
        if it:
            try: self.plot.removeItem(it)
            except Exception: pass

    def _update_rescue(self, aid: str):
        rec = self._agents.get(aid)
        if not rec: return
        dm = rec.get('dm')
        if dm is None:
            self._remove_rescue(aid); return

        rp = None
        try:
            rp = dm.rescue_progress()
        except Exception:
            rp = None

        if not rp:
            self._remove_rescue(aid); return

        x, y = rec['xy']
        radius = 480.0

        path = QPainterPath()
        path.addEllipse(x - radius, y - radius, radius*2, radius*2)
        path.moveTo(x - radius*0.6, y - radius*0.6)
        path.lineTo(x + radius*0.6, y + radius*0.6)
        path.moveTo(x + radius*0.6, y - radius*0.6)
        path.lineTo(x - radius*0.6, y + radius*0.6)

        it = self._rescue_items.get(aid)
        if it is None:
            it = QGraphicsPathItem(path)
            vb = self._vb()
            if vb is not None:
                it.setParentItem(vb)
            it.setBrush(pg.mkBrush(0, 0, 0, 0))
            it.setPen(pg.mkPen(COLOR_RESCUE_CROSS, width=4.0))
            it.setZValue(Z_RESCUE)
            self.plot.addItem(it)
            self._rescue_items[aid] = it
        else:
            it.setPath(path)
            it.setZValue(Z_RESCUE)

    # ---------- Misc helpers ----------
    def random_xy_on_roads(self) -> Optional[Tuple[float, float]]:
        if not self._map or not getattr(self._map, "nodes", None):
            return None
        road_nodes = [n for n in self._map.nodes if getattr(n, "type", "") in ("normal", "intersection")]
        if not road_nodes: return None
        x, y = _node_xy(random.choice(road_nodes))
        return (x, y)

    def log_action(self, msg: str, *, also_print: bool = True, max_lines: int = 200):
        return  # Disable Qt-side logging to improve performance

        try:
            cur = self.info.toPlainText() if hasattr(self, "info") else ""
        except Exception:
            cur = ""
        new_text = (cur + ("\n" if cur else "") + msg)
        lines = new_text.splitlines()
        if len(lines) > max_lines: lines = lines[-max_lines:]
        new_text = "\n".join(lines)
        try:
            self.info.setPlainText(new_text); self.info.moveCursor(self.info.textCursor().End)
        except Exception:
            pass
        if also_print: print(msg)

    # ---------- Bus overlays ----------
    def _update_all_buses(self):
        """Update all bus markers and labels."""
        if not self._bus_manager:
            return

        # Fetch all buses
        all_buses = list(self._bus_manager.buses.values())
        current_bus_ids = set(bus.id for bus in all_buses)

        # Remove buses that no longer exist
        for bus_id in list(self._bus_items.keys()):
            if bus_id not in current_bus_ids:
                self._remove_bus(bus_id)

        # Update or create markers for all current buses
        for bus in all_buses:
            self._update_bus(bus)

    def _remove_bus(self, bus_id: str):
        """Remove a bus marker and its label."""
        # Remove bus marker
        bus_item = self._bus_items.pop(bus_id, None)
        if bus_item:
            try:
                self.plot.removeItem(bus_item)
            except Exception:
                pass

        # Remove bus label
        bus_text = self._bus_text_items.pop(bus_id, None)
        if bus_text:
            try:
                self.plot.removeItem(bus_text)
            except Exception:
                pass

    def _update_bus(self, bus):
        """Update a single bus marker and its status label."""
        bus_id = bus.id

        # Bus marker
        bus_item = self._bus_items.get(bus_id)
        if bus_item is None:
            # Create new marker
            bus_item = pg.ScatterPlotItem(
                pos=[(bus.x, bus.y)], size=20,
                brush=pg.mkBrush("#FF6B35"),  # orange
                pen=pg.mkPen("#000000", width=2),
                symbol="s", antialias=True  # square for bus
            )
            bus_item.setZValue(AGENT_Z + 10)  # slightly above agents
            self.plot.addItem(bus_item)
            self._bus_items[bus_id] = bus_item
        else:
            # Update position
            bus_item.setData(pos=[(bus.x, bus.y)])

        # Bus status text
        status_text = f"Bus {bus_id}"
        if bus.state.value == "stopped":
            current_stop = bus.get_current_stop()
            if current_stop:
                status_text += f" @ {current_stop.name or current_stop.id}"
        elif bus.state.value == "moving":
            next_stop = bus.get_next_stop()
            if next_stop:
                status_text += f" → {next_stop.name or next_stop.id}"

        bus_text = self._bus_text_items.get(bus_id)
        if bus_text is None:
            # Create new label
            bus_text = pg.TextItem(text=status_text, color=(0, 0, 0))
            bus_text.setAnchor((0.5, 1.5))  # label above the bus marker
            bus_text.setFont(self._label_font)
            bus_text.setPos(bus.x, bus.y)
            bus_text.setZValue(AGENT_Z + 20)
            self.plot.addItem(bus_text)
            self._bus_text_items[bus_id] = bus_text
        else:
            # Update label content and position
            bus_text.setText(status_text)
            bus_text.setPos(bus.x, bus.y)
            bus_text.setZValue(AGENT_Z + 20)