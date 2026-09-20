# gameplay/comms.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock, RLock
from typing import Dict, Any, List, Optional, Tuple

from ..base.timer import VirtualClock
from .help import HelpType, HelpRequest
from ..entities.temp_box import TempBox


@dataclass
class ChatMessage:
    """Lightweight record for a chat message stored in the inbox."""
    msg_id: int
    ts_sim: float
    from_agent: str
    to_agent: Optional[str]   # None = broadcast
    kind: str                 # "direct" | "broadcast"
    text: str


# ===== Comms System =====
class CommsSystem:
    """
    Central coordination system for help-board, TempBox handoff,
    settlements, timers, scooter charging, and agent-to-agent chat.

    - Help board: post / accept / modify requests
    - TempBox: one per role (publisher/helper) per request; used for handoff
    - Settlement: escrow bounty, upfront costs, 5-minute grace (half pay), refunds
    - Timing: DELIVERY/CHARGE start when publisher places items into TempBox;
              BUY/PICKUP start on accept
    - Thinking pause: pause_timers_for / resume_timers_for (per helper)
    """

    def __init__(
        self,
        clock: Optional[VirtualClock] = None,
        ambient_temp_c: float = 22.0,
        k_food_per_s: float = 1.0 / 1200.0,
    ):
        self.clock = clock if clock is not None else VirtualClock()
        self._ambient_temp_c = ambient_temp_c
        self.k_food_per_s = k_food_per_s
        self._lock = RLock()  # re-entrant to avoid deadlocks

        # help board pools
        self._next_req_id = 1
        self._board: Dict[int, HelpRequest] = {}      # open
        self._active: Dict[int, HelpRequest] = {}     # accepted but not finished
        self._completed: Dict[int, HelpRequest] = {}  # finished

        # temp boxes
        self._next_box_id = 1
        self._boxes: Dict[int, TempBox] = {}
        self._role_index: Dict[Tuple[int, str], int] = {}  # (req_id, role) -> box_id

        # agents registry
        self._agents: Dict[str, Any] = {}

        # publisher notifications
        self._msgs_for_publisher: Dict[str, List[Dict[str, Any]]] = {}

        # scooters / charging
        self._scooters_by_owner: Dict[str, Any] = {}
        self._charging_busy: Dict[Tuple[int, int], Dict[str, Any]] = {}

        # chat
        self._next_msg_id = 1
        self._chat_inbox: Dict[str, List[ChatMessage]] = {}
        self._chat_max_per_agent = 200  # cap inbox to avoid unbounded growth

        # distance tolerance (cm)
        self._tol_cm_default = 300.0

        # price fallbacks
        self._DEFAULT_PRICE_TABLE = {
            "energy_drink": 2.0,
            "escooter_battery_pack": 10.0,
        }
        self._DEFAULT_CHARGE_PRICE_PER_PERCENT = 0.5

    # ----- Agent registry -----
    def register_agent(self, dm: Any) -> None:
        """Register an agent so it can post/accept help, chat, etc."""
        with self._lock:
            self._agents[str(dm.agent_id)] = dm

    # ----- Logging / geometry helpers -----
    def _log_agent(self, agent_id: str, text: str) -> None:
        """Log a message via the agent's logger if available, otherwise stdout."""
        dm = self._agents.get(str(agent_id))
        if dm is not None and hasattr(dm, "_log"):
            dm._log(text)
        else:
            print(f"[Comms] agent {agent_id}: {text}")

    def _within(
        self,
        xy: Optional[Tuple[float, float]],
        target_xy: Optional[Tuple[float, float]],
        tol_cm: float,
    ) -> bool:
        """Return True if xy is within tol_cm of target_xy (in centimeters)."""
        if xy is None or target_xy is None:
            return False
        dx = float(xy[0]) - float(target_xy[0])
        dy = float(xy[1]) - float(target_xy[1])
        return math.hypot(dx, dy) <= float(tol_cm)

    # ----- Timer helpers -----
    def _start_timer_if_needed(self, req: HelpRequest, reason: str) -> None:
        """Start the request timer once, recording a short note."""
        if req.timer_started_sim is None:
            req.timer_started_sim = float(self.clock.now_sim())
            req.start_note = str(reason)

    def _effective_elapsed_s(self, req: HelpRequest, now: Optional[float] = None) -> float:
        """Effective elapsed time minus paused intervals."""
        if req.timer_started_sim is None:
            return 0.0
        now = self.clock.now_sim() if now is None else float(now)
        paused = float(req.timer_paused_accum_s or 0.0)
        if req.timer_paused_at is not None:
            paused += max(0.0, now - float(req.timer_paused_at))
        return max(0.0, now - float(req.timer_started_sim) - paused)

    def _time_left_s(self, req: HelpRequest, now: Optional[float] = None) -> Optional[float]:
        """Time left before deadline; None if timer has not started."""
        if req.timer_started_sim is None:
            return None
        now = self.clock.now_sim() if now is None else float(now)
        used = self._effective_elapsed_s(req, now)
        return max(0.0, float(req.time_limit_s) - used)

    def pause_timers_for(self, helper_id: str) -> None:
        """Pause timers for all active requests accepted by the given helper."""
        with self._lock:
            now = self.clock.now_sim()
            for r in self._active.values():
                if (
                    r.accepted_by == str(helper_id)
                    and r.timer_started_sim is not None
                    and r.timer_paused_at is None
                ):
                    r.timer_paused_at = float(now)

    def resume_timers_for(self, helper_id: str) -> None:
        """Resume timers for all active requests accepted by the given helper."""
        with self._lock:
            now = self.clock.now_sim()
            for r in self._active.values():
                if r.accepted_by == str(helper_id) and r.timer_paused_at is not None:
                    r.timer_paused_accum_s += max(0.0, float(now) - float(r.timer_paused_at))
                    r.timer_paused_at = None

    def get_ambient_temp_c(self) -> float:
        return self._ambient_temp_c

    def get_k_food_per_s(self) -> float:
        return self.k_food_per_s

    # ----- Pricing helpers (upfront estimation) -----
    def _any_store_manager(self) -> Optional[Any]:
        """Return any available store manager from registered agents."""
        for dm in self._agents.values():
            sm = getattr(dm, "_store_manager", None)
            if sm is not None:
                return sm
        return None

    def _estimate_buy_cost(self, buy_items: Optional[Dict[str, int]]) -> float:
        """Estimate cost for HELP_BUY items using store manager or fallback table."""
        if not buy_items:
            return 0.0
        sm = self._any_store_manager()
        total = 0.0
        for item_id, qty in buy_items.items():
            q = int(qty or 0)
            if q <= 0:
                continue
            unit = 0.0
            if sm:
                if hasattr(sm, "get_price"):
                    try:
                        unit = float(sm.get_price(item_id))
                    except Exception:
                        unit = 0.0
                elif hasattr(sm, "get_unit_price"):
                    try:
                        unit = float(sm.get_unit_price(item_id))
                    except Exception:
                        unit = 0.0
            if unit <= 0.0:
                unit = float(self._DEFAULT_PRICE_TABLE.get(str(item_id), 0.0))
            total += unit * q
        return float(total)

    def _estimate_upfront_cost(self, req: HelpRequest) -> float:
        """Estimate upfront cost needed for a request (BUY / CHARGE)."""
        if req.kind == HelpType.HELP_BUY:
            return self._estimate_buy_cost(req.buy_items)
        if req.kind == HelpType.HELP_CHARGE:
            tgt = int(req.target_pct or 100)
            return float(tgt) * float(self._DEFAULT_CHARGE_PRICE_PER_PERCENT)
        return 0.0

    # ----- Listing: requests -----
    def list_open_requests(self) -> List[HelpRequest]:
        with self._lock:
            return list(self._board.values())

    def list_my_posts_open(self, publisher_id: str) -> List[HelpRequest]:
        with self._lock:
            return [r for r in self._board.values() if r.publisher_id == str(publisher_id)]

    def list_my_posts_active(self, publisher_id: str) -> List[HelpRequest]:
        with self._lock:
            return [r for r in self._active.values() if r.publisher_id == str(publisher_id)]

    def list_my_helps(self, helper_id: str) -> List[HelpRequest]:
        with self._lock:
            return [r for r in self._active.values() if r.accepted_by == str(helper_id)]

    def list_my_helps_completed(self, helper_id: str) -> List[HelpRequest]:
        with self._lock:
            return [r for r in self._completed.values() if r.accepted_by == str(helper_id)]

    # ----- Post / Accept / Modify requests -----
    def _auto_brief(self, kind: HelpType, order_id, buy_items, target_pct) -> str:
        """Generate a concise default description when the caller does not provide one."""
        if kind == HelpType.HELP_DELIVERY:
            return f"Deliver order #{order_id}."
        if kind == HelpType.HELP_PICKUP:
            return f"Pick up order #{order_id}."
        if kind == HelpType.HELP_BUY:
            return "Buy items."
        if kind == HelpType.HELP_CHARGE:
            tgt = int(target_pct or 100)
            return f"Charge e-scooter to {tgt}%."
        return "Help request."

    def post_request(
        self,
        publisher_id: str,
        kind: HelpType,
        *,
        reward: float,
        time_limit_s: float,
        order_id: Optional[int] = None,
        buy_items: Optional[Dict[str, int]] = None,
        target_pct: Optional[float] = None,
        brief: Optional[str] = None,
        location_xy: Optional[Tuple[float, float]] = None,
        provide_xy: Optional[Tuple[float, float]] = None,
        deliver_xy: Optional[Tuple[float, float]] = None,
        order_ref: Optional[Any] = None,
    ) -> Tuple[bool, str, Optional[int]]:
        """Post a help request to the board and return (ok, reason, req_id)."""
        now = self.clock.now_sim()
        with self._lock:
            rid = self._next_req_id
            self._next_req_id += 1

            if deliver_xy is None:
                # backward compatibility: reuse location_xy if no explicit deliver_xy
                deliver_xy = location_xy

            req = HelpRequest(
                req_id=rid,
                publisher_id=str(publisher_id),
                kind=HelpType(kind),
                reward=float(reward),
                time_limit_s=float(time_limit_s),
                created_sim=float(now),
                order_id=int(order_id) if order_id is not None else None,
                buy_items=dict(buy_items) if buy_items else None,
                target_pct=float(target_pct) if target_pct is not None else None,
                brief=str(brief) if brief else self._auto_brief(kind, order_id, buy_items, target_pct),
                provide_xy=(tuple(provide_xy) if provide_xy else None),
                deliver_xy=(tuple(deliver_xy) if deliver_xy else None),
                order_ref=order_ref,
            )
            self._board[rid] = req
            return True, "posted", rid

    def accept_request(self, req_id: int, helper_id: str) -> Tuple[bool, str]:
        """Accept an open request and move it to the active pool."""
        with self._lock:
            req = self._board.get(int(req_id))
            if not req:
                return False, "not_on_board"
            pub = self._agents.get(str(req.publisher_id))
            hel = self._agents.get(str(helper_id))
            if pub is None or hel is None:
                return False, "agent_not_registered"
            if pub.agent_id == hel.agent_id:
                return False, "cannot_accept_own_request"

            # upfront estimation (BUY/CHARGE) and total needed: escrow bounty + upfront
            upfront = self._estimate_upfront_cost(req)
            need_total = float(req.reward) + float(upfront)

            # funds check (publisher)
            pub_balance = float(getattr(pub, "earnings_total", 0.0))
            if pub_balance < need_total - 1e-9:
                shortfall = need_total - pub_balance
                # notify both sides; do not accept
                self._log_agent(
                    str(helper_id),
                    f"[Help #{req.req_id}] Accept failed: publisher (agent {req.publisher_id}) "
                    f"insufficient funds. need=${need_total:.2f} "
                    f"(escrow=${req.reward:.2f} + upfront=${upfront:.2f}), "
                    f"has=${pub_balance:.2f}, shortfall=${shortfall:.2f}.",
                )
                self._log_agent(
                    str(req.publisher_id),
                    (
                        f"[Help #{req.req_id}] Insufficient funds; the request cannot be accepted. "
                        f"Required pre-hold=${need_total:.2f} "
                        f"(escrow=${req.reward:.2f} + upfront=${upfront:.2f}), "
                        f"current balance=${pub_balance:.2f}, shortfall=${shortfall:.2f}."
                    ),
                )
                return False, "publisher_insufficient_funds"

            # deduct/transfer: publisher pays escrow + upfront; helper receives upfront immediately
            pub_before = float(pub.earnings_total)
            hel_before = float(hel.earnings_total)

            pub.earnings_total -= need_total
            hel.earnings_total += float(upfront)

            pub_after = float(pub.earnings_total)
            hel_after = float(hel.earnings_total)

            # bookkeeping
            req.escrow_reward = float(req.reward)
            req.upfront_paid_to_helper = float(upfront)
            req.accepted_by = str(helper_id)

            # logs (money only)
            self._log_agent(
                str(req.publisher_id),
                (
                    f"[Help #{req.req_id}] Funds held: escrow=${req.reward:.2f} + upfront=${upfront:.2f} "
                    f"= total ${need_total:.2f}. Balance: ${pub_before:.2f} -> ${pub_after:.2f}."
                ),
            )
            self._log_agent(
                str(helper_id),
                (
                    f"[Help #{req.req_id}] Upfront received=${upfront:.2f}. "
                    f"Balance: ${hel_before:.2f} -> ${hel_after:.2f}."
                ),
            )

            if upfront > 1e-9:
                pub_rec = getattr(pub, "_recorder", None)
                if pub_rec and hasattr(pub_rec, "note_advance_out"):
                    category = (
                        "buy"
                        if req.kind == HelpType.HELP_BUY
                        else ("charge" if req.kind == HelpType.HELP_CHARGE else None)
                    )
                    if category:
                        pub_rec.note_advance_out(
                            ts_sim=self.clock.now_sim(),
                            req_id=req.req_id,
                            category=category,
                            amount=float(upfront),
                        )

            # BUY / PICKUP: timer starts on accept (no extra settlement log here)
            if req.kind in (HelpType.HELP_BUY, HelpType.HELP_PICKUP):
                self._start_timer_if_needed(req, reason="start_on_accept")

            # move to active
            self._board.pop(int(req_id), None)
            self._active[int(req_id)] = req
            return True, "accepted"

    def modify_request(
        self,
        publisher_id: str,
        req_id: int,
        *,
        reward: Optional[float] = None,
        time_limit_s: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """
        Modify reward or TTL.
        - If still on board: reward and TTL can be changed.
        - If already active: only TTL can be changed.
        """
        with self._lock:
            req = self._board.get(int(req_id))
            if req and req.publisher_id == str(publisher_id):
                if reward is not None:
                    req.reward = float(reward)
                if time_limit_s is not None:
                    req.time_limit_s = float(time_limit_s)
                return True, "modified"
            req = self._active.get(int(req_id))
            if req and req.publisher_id == str(publisher_id):
                if reward is not None:
                    return False, "cannot_modify_reward_after_accept"
                if time_limit_s is not None:
                    req.time_limit_s = float(time_limit_s)
                    return True, "modified_ttl"
                return False, "no_change"
            return False, "not_found_or_not_owner"

    # ----- Scooter registry -----
    def register_scooter(self, owner_id: str, scooter: Any) -> Any:
        """Register or reuse the canonical scooter instance for this owner."""
        with self._lock:
            key = str(owner_id)
            canon = self._scooters_by_owner.get(key)
            if canon is None:
                self._scooters_by_owner[key] = scooter
                return scooter
            return canon  # already have a canonical instance

    def get_scooter_by_owner(self, owner_id: str) -> Optional[Any]:
        with self._lock:
            return self._scooters_by_owner.get(str(owner_id))

    def _canonicalize_escooter(self, sc: Optional[Any]) -> Optional[Any]:
        """Ensure we always store/return the canonical instance for the scooter's owner."""
        if sc is None:
            return None
        owner = getattr(sc, "owner_id", None)
        if not owner:
            return sc
        with self._lock:
            key = str(owner)
            canon = self._scooters_by_owner.get(key)
            if canon is None:
                # first time seeing this scooter -> remember this instance as canonical
                self._scooters_by_owner[key] = sc
                return sc
            return canon

    # ----- TempBox core -----
    def _get_or_create_box(self, req_id: int, role: str, xy: Tuple[float, float]) -> TempBox:
        """Return TempBox for (req_id, role), creating or updating coordinates."""
        key = (int(req_id), str(role))
        box_id = self._role_index.get(key)
        if box_id is None:
            box_id = self._next_box_id
            self._next_box_id += 1
            box = TempBox(
                box_id=box_id,
                req_id=int(req_id),
                owner_id="",
                role=str(role),
                xy=tuple(xy),
                created_sim=float(self.clock.now_sim()),
            )
            self._boxes[box_id] = box
            self._role_index[key] = box_id
        else:
            box = self._boxes[box_id]
            box.xy = tuple(xy)  # update to latest location
        return box

    def place_temp_box(
        self,
        req_id: int,
        by_agent: str,
        location_xy: Tuple[float, float],
        content: Dict[str, Any],
    ) -> Tuple[bool, str]:
        """Place or update a TempBox for this request and role with given payload."""
        with self._lock:
            req = self._active.get(int(req_id)) or self._board.get(int(req_id))
            if not req:
                return False, "request_not_found"

            role = "publisher" if str(by_agent) == str(req.publisher_id) else "helper"

            if "escooter" in (content or {}):
                content = dict(content)  # shallow copy
                content["escooter"] = self._canonicalize_escooter(content["escooter"])

            box = self._get_or_create_box(int(req_id), role, tuple(location_xy))
            box.owner_id = str(by_agent)

            try:
                box.thermal_tick(
                    self.clock.now_sim(),
                    self.get_ambient_temp_c(),
                    self.get_k_food_per_s(),
                )
            except Exception:
                pass

            box.merge_payload(content or {})

            if role == "helper":
                # save a snapshot of helper's last drop for later validation / settlement
                snap_sc = self._canonicalize_escooter(box.escooter)
                req.last_helper_drop = {
                    "ts_sim": self.clock.now_sim(),
                    "xy": tuple(location_xy),
                    "at_destination": bool(
                        self._within(tuple(location_xy), req.deliver_xy, self._tol_cm_default)
                    ),
                    "inventory": dict(box.inventory),
                    "food_by_order_counts": {
                        int(k): len(v or []) for k, v in (box.food_by_order or {}).items()
                    },
                    "has_escooter": (snap_sc is not None),
                    "scooter_battery_pct": (
                        float(getattr(snap_sc, "battery_pct", 0.0))
                        if snap_sc is not None
                        else None
                    ),
                }

            # trigger timer:
            # DELIVERY -> start when publisher places food (has food content or compatible short form)
            # CHARGE   -> start when publisher places scooter
            by_publisher = str(by_agent) == str(req.publisher_id)
            if by_publisher:
                if req.kind == HelpType.HELP_DELIVERY and (
                    content.get("food_by_order")
                    or ("order_id" in content and "food_items" in content)
                    or ("food" in content)  # compatibility with short form
                ):
                    self._start_timer_if_needed(req, reason="publisher_placed_food")
                elif req.kind == HelpType.HELP_CHARGE and ("escooter" in content):
                    self._start_timer_if_needed(req, reason="publisher_placed_escooter")

            return True, "placed"

    def take_from_temp_box(self, req_id: int, by_agent: str):
        """
        Take payload from the opposite role's box:
        - helper takes from publisher_box
        - publisher takes from helper_box
        """
        with self._lock:
            req = (
                self._active.get(int(req_id))
                or self._board.get(int(req_id))
                or self._completed.get(int(req_id))
            )
            if not req:
                return False, "request_not_found", None

            target_role = "publisher" if str(by_agent) != str(req.publisher_id) else "helper"
            key = (int(req_id), target_role)
            box_id = self._role_index.get(key)
            if box_id is None or box_id not in self._boxes:
                return False, "box_not_found", None

            box = self._boxes[box_id]

            try:
                box.thermal_tick(
                    self.clock.now_sim(),
                    self.get_ambient_temp_c(),
                    self.get_k_food_per_s(),
                )
            except Exception:
                pass

            sc = self._canonicalize_escooter(box.escooter)
            payload = dict(
                inventory=dict(box.inventory),
                food_by_order={int(k): list(v) for k, v in box.food_by_order.items()},
                escooter=sc,
                xy=box.xy,
            )
            # clear content (box stays)
            box.inventory.clear()
            box.food_by_order.clear()
            box.escooter = None
            return True, "ok", payload

    def get_temp_box_info(self, req_id: int) -> Dict[str, Any]:
        """Return lightweight TempBox status for UI / copying into prompts."""
        with self._lock:
            out = {"publisher_box": {}, "helper_box": {}}
            for role in ("publisher", "helper"):
                key = (int(req_id), role)
                bid = self._role_index.get(key)
                if bid and bid in self._boxes:
                    b = self._boxes[bid]
                    has_content = bool(
                        b.escooter or b.inventory or any(b.food_by_order.values())
                    )
                    out[f"{role}_box"] = {
                        "xy": b.xy,
                        "has_content": has_content,
                        "ready": has_content,  # minimal heuristic: any content => ready
                        "box_id": bid,
                    }
            return out

    # ----- TempBox helpers -----
    def _box_has_content(self, box: TempBox) -> bool:
        """Return True if box contains any inventory, food, or scooter."""
        if box is None:
            return False
        if box.escooter is not None:
            return True
        if box.inventory:
            # any positive quantity counts; non-positive values are treated as empty
            for _, v in box.inventory.items():
                if int(v or 0) > 0:
                    return True
        # any non-empty list in food_by_order counts
        return any(bool(items) for items in (box.food_by_order or {}).values())

    def _summarize_box(self, box: TempBox) -> Dict[str, Any]:
        """Small summary for UI / API without exposing raw objects."""
        if box is None:
            return {"has_content": False}
        return {
            "box_id": int(box.box_id),
            "xy": tuple(box.xy),
            "has_content": self._box_has_content(box),
            "inventory": dict(box.inventory),
            # only expose counts to avoid leaking object references
            "food_by_order_counts": {
                int(k): len(v or []) for k, v in (box.food_by_order or {}).items()
            },
            "has_escooter": (box.escooter is not None),
            "created_sim": float(box.created_sim),
        }

    # ----- Retrieval listing -----
    def list_retrievable_tempboxes(
        self,
        agent_id: str,
        role: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return all TempBoxes from which this agent can retrieve items.

        Includes active and completed requests.

        - role=None: return both publisher and helper perspectives
        - role='publisher': only requests posted by the agent where helper_box has content
        - role='helper':   only requests accepted by the agent where publisher_box has content

        Each entry has:
          - req_id, kind, brief, state ('active' | 'completed')
          - take_from ('helper_box' | 'publisher_box')
          - can_take_as ('publisher' | 'helper')
          - xy, summary (from _summarize_box)
        """
        agent_id = str(agent_id)
        out: List[Dict[str, Any]] = []
        with self._lock:
            def _get_box(req_id: int, role_name: str) -> Optional[TempBox]:
                bid = self._role_index.get((int(req_id), str(role_name)))
                if bid and bid in self._boxes:
                    return self._boxes[bid]
                return None

            def _consider(req: HelpRequest, state_label: str) -> None:
                # view A: as publisher, can take from helper_box
                if role in (None, "publisher") and req.publisher_id == agent_id:
                    hb = _get_box(req.req_id, "helper")
                    if hb and self._box_has_content(hb):
                        out.append({
                            "req_id": int(req.req_id),
                            "kind": req.kind.value,
                            "brief": req.brief,
                            "state": state_label,
                            "take_from": "helper_box",
                            "can_take_as": "publisher",
                            "xy": tuple(hb.xy),
                            "summary": self._summarize_box(hb),
                        })
                # view B: as helper, can take from publisher_box
                if role in (None, "helper") and req.accepted_by == agent_id:
                    pb = _get_box(req.req_id, "publisher")
                    if pb and self._box_has_content(pb):
                        out.append({
                            "req_id": int(req.req_id),
                            "kind": req.kind.value,
                            "brief": req.brief,
                            "state": state_label,
                            "take_from": "publisher_box",
                            "can_take_as": "helper",
                            "xy": tuple(pb.xy),
                            "summary": self._summarize_box(pb),
                        })

            # scan both active and completed in one pass
            for r in self._active.values():
                _consider(r, "active")
            for r in self._completed.values():
                _consider(r, "completed")

            # sort by box creation time (oldest first)
            out.sort(key=lambda e: float(e["summary"].get("created_sim", 0.0)))
            return out

    # ----- Pickables (human-readable) -----
    def _fmt_xy_text(self, xy: Optional[Tuple[float, float]]) -> str:
        if not xy:
            return "N/A"
        x, y = xy
        return f"({x/100.0:.2f}m, {y/100.0:.2f}m)"

    def _fmt_pickable_things(self, summary: Dict[str, Any]) -> str:
        """Make a compact, human-friendly 'things=' string from box summary."""
        parts: List[str] = []
        inv = summary.get("inventory") or {}

        # inventory: kxv
        inv_pairs = [f"{k}x{int(v)}" for k, v in inv.items() if int(v or 0) > 0]
        if inv_pairs:
            parts.append("inventory: " + ", ".join(inv_pairs))

        # food orders: #id, #id2 ...
        fbo_counts = summary.get("food_by_order_counts") or {}
        if fbo_counts:
            parts.append(
                "food orders: " + ", ".join(f"#{int(oid)}" for oid in fbo_counts.keys())
            )

        # e-scooter
        if bool(summary.get("has_escooter")):
            parts.append("e-scooter")

        return "; ".join(parts) if parts else "(unknown)"

    def _fmt_pickable_line(self, rec: Dict[str, Any]) -> str:
        """One-line, human-friendly description of a retrievable TempBox."""
        rid = int(rec.get("req_id", 0))
        role = str(rec.get("can_take_as", ""))  # 'helper' / 'publisher'
        xy = rec.get("xy")
        things = self._fmt_pickable_things(rec.get("summary") or {})
        return f"[Help #{rid}] role={role}, box={self._fmt_xy_text(xy)}, things={things}"

    def pickables_lines_for(
        self,
        agent_id: str,
        *,
        role: Optional[str] = None,             # None / 'helper' / 'publisher'
        include_completed: bool = True,
    ) -> List[str]:
        """Return one-line descriptions of what this agent can retrieve now."""
        recs = self.list_retrievable_tempboxes(str(agent_id), role=role)
        if not include_completed:
            recs = [r for r in recs if r.get("state") == "active"]
        return [self._fmt_pickable_line(r) for r in recs]

    def pickables_text_for(
        self,
        agent_id: str,
        *,
        bullet: str = "- ",
        role: Optional[str] = None,
        include_completed: bool = True,
        none_text: str = "(none)",
    ) -> str:
        """Return a block of text describing retrievable items for prompts."""
        lines = self.pickables_lines_for(
            agent_id,
            role=role,
            include_completed=include_completed,
        )
        if not lines:
            return f"{none_text}"
        return "\n".join(bullet + s for s in lines)

    # ----- Finish / Settlement -----
    def push_helper_delivered(
        self,
        req_id: int,
        by_agent: str,
        order_id: int,
        at_xy: Tuple[float, float],
    ) -> Tuple[bool, str]:
        """
        Record that the helper has reached destination and dropped off.

        The publisher will auto-settle in their own loop; finalization still
        requires REPORT_HELP_FINISHED (via report_help_finished).
        """
        with self._lock:
            req = self._active.get(int(req_id))
            if not req:
                return False, "not_active"
            if str(req.accepted_by) != str(by_agent):
                return False, "not_helper"
            self._msgs_for_publisher.setdefault(str(req.publisher_id), []).append({
                "type": "HELP_DELIVERY_DONE",
                "req_id": int(req.req_id),
                "order_id": int(order_id),
                "at_xy": tuple(at_xy),
            })
            return True, "ok"

    def pop_msgs_for_publisher(self, publisher_id: str) -> List[Dict[str, Any]]:
        """Pop and clear queued notifications for this publisher."""
        with self._lock:
            arr = self._msgs_for_publisher.pop(str(publisher_id), [])
            return list(arr or [])

    def report_help_finished(
        self,
        req_id: int,
        by_agent: str,
        at_xy: Tuple[float, float],
    ):
        """
        Final settlement for a help request.

        Validates drop evidence, computes on-time / grace / fail,
        settles escrow and upfront, and moves request to completed.
        """
        with self._lock:
            req = self._active.get(int(req_id))
            if not req:
                return False, "not_active", None
            if str(req.accepted_by) != str(by_agent):
                return False, "not_helper", None

            now = self.clock.now_sim()

            # effective elapsed time
            if req.timer_started_sim is None:
                elapsed = 0.0
            else:
                elapsed = self._effective_elapsed_s(req, now)

            # status: on time / within 5-min grace / fail
            limit = float(req.time_limit_s or 0.0)
            grace = 300.0  # 5 minutes
            if req.timer_started_sim is None:
                status = "success"
            elif elapsed <= limit + 1e-9:
                status = "success"
            elif elapsed <= limit + grace + 1e-9:
                status = "partial"
            else:
                status = "fail"

            pub = self._agents.get(str(req.publisher_id))
            hel = self._agents.get(str(req.accepted_by or ""))
            if pub is None or hel is None:
                return False, "agent_not_found", None

            # Validation: use snapshot of helper's last drop (so publisher taking items later won't break proofs)
            need_drop_kinds = (HelpType.HELP_BUY, HelpType.HELP_PICKUP, HelpType.HELP_CHARGE)
            if req.kind in need_drop_kinds:
                snap = getattr(req, "last_helper_drop", None)
                if not snap:
                    return False, "drop_proof_missing", {
                        "hint": "No drop snapshot found. Place a TempBox at the destination before calling REPORT_HELP_FINISHED.",
                    }
                if not snap.get("at_destination"):
                    return False, "drop_wrong_location", {
                        "hint": "The last TempBox placement was not at the destination. Move to the drop-off point and place again.",
                    }

                # BUY: verify requested items present at drop-time
                if req.kind == HelpType.HELP_BUY and req.buy_items:
                    missing: Dict[str, int] = {}
                    inv = snap.get("inventory") or {}
                    for item_id, need_q in req.buy_items.items():
                        have_q = int(inv.get(item_id, 0))
                        lack = int(need_q or 0) - have_q
                        if lack > 0:
                            missing[item_id] = lack
                    if missing:
                        return False, "buy_items_incomplete", {
                            "hint": "One or more items were missing at the drop. Top up the missing quantities and place again.",
                            "missing": missing,
                        }

                # PICKUP: must contain food; if order_id specified, that order must be there
                if req.kind == HelpType.HELP_PICKUP:
                    fbo_counts = snap.get("food_by_order_counts") or {}
                    has_any_food = any(int(c or 0) > 0 for c in fbo_counts.values())
                    ok_food = has_any_food
                    if req.order_id is not None:
                        ok_food = ok_food and bool(fbo_counts.get(int(req.order_id)))
                    if not ok_food:
                        return False, "pickup_food_missing", {
                            "hint": "No food (or target order) in the drop snapshot. Place the food at the destination and try again.",
                        }

                # CHARGE: scooter must be present and meet target battery at drop-time
                if req.kind == HelpType.HELP_CHARGE:
                    if not bool(snap.get("has_escooter")):
                        return False, "scooter_missing", {
                            "hint": "No e-scooter in the drop snapshot. Place the scooter at the destination before finishing.",
                        }
                    tgt_pct = float(req.target_pct or 100.0)
                    cur_pct = snap.get("scooter_battery_pct")
                    if cur_pct is None or float(cur_pct) + 1e-6 < tgt_pct:
                        return False, "undercharged", {
                            "hint": "Battery below the required target in the drop snapshot. Charge to the target and place again.",
                            "target_pct": tgt_pct,
                            "current_pct": float(cur_pct or 0.0),
                        }

            if req.kind == HelpType.HELP_DELIVERY:
                order_id = int(req.order_id) if req.order_id is not None else -1
                completed_set = getattr(hel, "help_orders_completed", set()) or set()
                if order_id >= 0 and order_id not in completed_set:
                    return False, "delivery_proof_missing", {
                        "hint": "Please DROP_OFF the order at the drop-off location first, then call REPORT_HELP_FINISHED.",
                        "expected_order_id": order_id,
                    }

            # snapshot before settlement for logging
            pub_before = float(getattr(pub, "earnings_total", 0.0))
            hel_before = float(getattr(hel, "earnings_total", 0.0))

            # escrow settlement
            escrow = float(req.escrow_reward or 0.0)
            if status == "success":
                hel.earnings_total += escrow
            elif status == "partial":
                hel.earnings_total += escrow * 0.5
                pub.earnings_total += escrow * 0.5
            else:  # fail
                pub.earnings_total += escrow

            # upfront: success/partial -> helper keeps; fail -> claw back to publisher
            upfront = float(req.upfront_paid_to_helper or 0.0)
            if status == "fail" and upfront > 0.0:
                hel.earnings_total -= upfront
                pub.earnings_total += upfront

            # snapshot after settlement for logging
            pub_after = float(getattr(pub, "earnings_total", 0.0))
            hel_after = float(getattr(hel, "earnings_total", 0.0))

            # logs (money only), to both sides
            if status == "success":
                self._log_agent(
                    str(req.publisher_id),
                    (
                        f"[Help #{req.req_id}] Settlement: success. Escrow ${escrow:.2f} "
                        f"paid to helper (agent {req.accepted_by}). "
                        f"Your balance: ${pub_before:.2f} -> ${pub_after:.2f}."
                    ),
                )
                self._log_agent(
                    str(req.accepted_by or ""),
                    (
                        f"[Help #{req.req_id}] Settlement: success. Received escrow=${escrow:.2f}; "
                        f"upfront=${upfront:.2f} kept. Your balance: ${hel_before:.2f} -> "
                        f"${hel_after:.2f}."
                    ),
                )
            elif status == "partial":
                self._log_agent(
                    str(req.publisher_id),
                    (
                        f"[Help #{req.req_id}] Settlement: partial. 50% of escrow "
                        f"(${escrow*0.5:.2f}) returned to you. "
                        f"Your balance: ${pub_before:.2f} -> ${pub_after:.2f}."
                    ),
                )
                self._log_agent(
                    str(req.accepted_by or ""),
                    (
                        f"[Help #{req.req_id}] Settlement: partial. Received 50% of escrow "
                        f"(${escrow*0.5:.2f}); upfront=${upfront:.2f} kept. "
                        f"Your balance: ${hel_before:.2f} -> ${hel_after:.2f}."
                    ),
                )
            else:  # fail
                self._log_agent(
                    str(req.publisher_id),
                    (
                        f"[Help #{req.req_id}] Settlement: fail. Full escrow (${escrow:.2f}) returned; "
                        f"upfront (${upfront:.2f}) clawed back from helper. "
                        f"Your balance: ${pub_before:.2f} -> ${pub_after:.2f}."
                    ),
                )
                self._log_agent(
                    str(req.accepted_by or ""),
                    (
                        f"[Help #{req.req_id}] Settlement: fail. Escrow ${escrow:.2f} returned to "
                        f"publisher; upfront ${upfront:.2f} clawed back. "
                        f"Your balance: ${hel_before:.2f} -> ${hel_after:.2f}."
                    ),
                )

            ts = now
            pub_rec = getattr(pub, "_recorder", None)
            hel_rec = getattr(hel, "_recorder", None)

            # simple counters: how many helps given / received
            if status in ("success", "partial"):
                if hel_rec and hasattr(hel_rec, "inc"):
                    hel_rec.inc("help_given", 1)
                if pub_rec and hasattr(pub_rec, "inc"):
                    pub_rec.inc("help_received", 1)

            # help income: escrow actually paid to helper (success 100%, grace 50%, fail 0)
            escrow_paid = (
                escrow if status == "success" else (escrow * 0.5 if status == "partial" else 0.0)
            )
            if hel_rec and hasattr(hel_rec, "on_help_income") and escrow_paid > 1e-9:
                hel_rec.on_help_income(
                    ts_sim=ts,
                    req_id=int(req.req_id),
                    kind=req.kind.value,
                    amount=float(escrow_paid),
                )
            if pub_rec and hasattr(pub_rec, "on_help_expense") and escrow_paid > 1e-9:
                pub_rec.on_help_expense(
                    ts_sim=ts,
                    req_id=int(req.req_id),
                    kind=req.kind.value,
                    amount=float(escrow_paid),
                )

            if (
                status == "fail"
                and upfront > 1e-9
                and pub_rec
                and hasattr(pub_rec, "note_advance_return")
            ):
                pub_rec.note_advance_return(
                    ts_sim=ts,
                    req_id=int(req.req_id),
                    amount=float(upfront),
                )

            # finalize
            req.completed = True
            req.result = status
            self._active.pop(int(req_id), None)
            self._completed[int(req_id)] = req

            # notify publisher
            self._msgs_for_publisher.setdefault(str(req.publisher_id), []).append({
                "type": "HELP_FINISHED",
                "req_id": int(req.req_id),
                "status": status,
                "at_xy": tuple(at_xy),
            })
            return True, "ok", {"status": status}

    # ----- Query helpers -----
    def get_request(self, req_id: int) -> Optional[HelpRequest]:
        """Return the HelpRequest by id, searching open, active, then completed."""
        with self._lock:
            return (
                self._board.get(int(req_id))
                or self._active.get(int(req_id))
                or self._completed.get(int(req_id))
            )

    def get_request_detail(self, req_id: int) -> Dict[str, Any]:
        """Return a plain dict with key details of a request."""
        with self._lock:
            r = (
                self._board.get(int(req_id))
                or self._active.get(int(req_id))
                or self._completed.get(int(req_id))
            )
            if not r:
                return {}
            return {
                "id": int(r.req_id),
                "req_id": int(r.req_id),
                "publisher_id": str(r.publisher_id),
                "accepted_by": r.accepted_by,
                "kind": r.kind.value,
                "reward": float(r.reward),
                "time_limit_s": float(r.time_limit_s),
                "order_id": r.order_id,
                "buy_items": dict(r.buy_items) if r.buy_items else None,
                "target_pct": r.target_pct,
                "provide_xy": tuple(r.provide_xy) if r.provide_xy else None,
                "deliver_xy": tuple(r.deliver_xy) if r.deliver_xy else None,
                "order_ref": r.order_ref,
                "completed": bool(r.completed),
                "result": r.result,
                "timer_started_sim": r.timer_started_sim,
                "escrow_reward": r.escrow_reward,
                "upfront_paid_to_helper": r.upfront_paid_to_helper,
            }

    # ----- Board text -----
    def board_to_text(
        self,
        *,
        include_active: bool = False,
        include_completed: bool = False,
        max_items: int = 50,
        exclude_publisher: Optional[str] = None,
        viewer_id: Optional[str] = None,
    ) -> str:
        """
        Build help-board text in English for agents.

        - By default lists open requests only.
        - exclude_publisher: hide requests posted by this agent.
        NOTE: snapshot inside lock, render outside to avoid re-entrant locking.
        """

        def _one(req: HelpRequest) -> str:
            # use neutral perspective; to_text_for already embeds agent IDs and timer rules
            return req.to_text_for(viewer_id or "", self, view_as=None)

        # snapshots under lock
        with self._lock:
            opens = list(self._board.values())
            act = list(self._active.values()) if include_active else []
            fin = list(self._completed.values()) if include_completed else []

        # filter & render outside lock
        if exclude_publisher:
            opens = [r for r in opens if r.publisher_id != str(exclude_publisher)]
            act = [r for r in act if r.publisher_id != str(exclude_publisher)]
            fin = [r for r in fin if r.publisher_id != str(exclude_publisher)]

        lines: List[str] = []
        if opens:
            lines.append("=== OPEN REQUESTS ===")
            for r in opens[:max_items]:
                lines.append(_one(r))
        if include_active and act:
            lines.append("=== ACTIVE REQUESTS ===")
            for r in act[:max_items]:
                lines.append(_one(r))
        if include_completed and fin:
            lines.append("=== COMPLETED REQUESTS ===")
            for r in fin[:max_items]:
                status = r.result or "unknown"
                lines.append(_one(r) + f" [completed: {status}]")

        return "\n".join(lines) if lines else "(empty help board)"

    # ----- Charging station helpers -----
    def _station_key(self, xy: Tuple[float, float]) -> Tuple[int, int]:
        """Quantize float coordinates into integer centimeters for station keys."""
        x, y = float(xy[0]), float(xy[1])
        return int(round(x)), int(round(y))

    def reserve_charging_spot(
        self,
        xy: Tuple[float, float],
        agent_id: str,
    ) -> Tuple[bool, str, Tuple[int, int]]:
        """
        Try to reserve this charging location.

        Returns (True, "", key) on success; if taken by another agent, returns
        (False, message, key).
        """
        key = self._station_key(xy)
        cur = self._charging_busy.get(key)
        if cur is None or str(cur.get("agent_id")) == str(agent_id):
            self._charging_busy[key] = {
                "agent_id": str(agent_id),
                "xy": (float(xy[0]), float(xy[1])),
            }
            return True, "", key
        return (
            False,
            f"charging station at ({xy[0]/100.0:.2f}m,{xy[1]/100.0:.2f}m) "
            f"is occupied by agent {cur.get('agent_id')}",
            key,
        )

    def release_charging_spot(
        self,
        xy_or_key,
        agent_id: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """
        Release a reserved charging spot.

        If agent_id is provided, the caller must match the current reservation.
        """
        if (
            isinstance(xy_or_key, tuple)
            and len(xy_or_key) == 2
            and isinstance(xy_or_key[0], int)
        ):
            key = xy_or_key
        else:
            key = self._station_key(xy_or_key)
        cur = self._charging_busy.get(key)
        if cur is None:
            return True, "already free"
        if agent_id is not None and str(cur.get("agent_id")) != str(agent_id):
            return False, f"occupied by another agent ({cur.get('agent_id')})"
        self._charging_busy.pop(key, None)
        return True, ""

    def charging_spot_status(self, xy: Tuple[float, float]) -> Optional[Dict[str, Any]]:
        """Return reservation info for this location, or None if free."""
        return self._charging_busy.get(self._station_key(xy))

    # ----- Chat -----
    def send_chat(
        self,
        from_agent: str,
        text: str,
        *,
        to_agent: Optional[str] = None,
        broadcast: bool = False,
    ) -> Tuple[bool, str, Optional[int]]:
        """
        Send a chat message.

        - For direct messages: broadcast=False, to_agent must be set.
        - For broadcast: broadcast=True and to_agent=None send to all other registered agents.
        """
        with self._lock:
            txt = (text or "").strip()
            if not txt:
                return False, "empty_text", None
            if not broadcast and not to_agent:
                return False, "no_target", None

            msg_id = self._next_msg_id
            self._next_msg_id += 1
            ts = float(self.clock.now_sim())

            def _push(aid: str) -> None:
                arr = self._chat_inbox.setdefault(str(aid), [])
                arr.append(
                    ChatMessage(
                        msg_id=msg_id,
                        ts_sim=ts,
                        from_agent=str(from_agent),
                        to_agent=str(to_agent) if not broadcast else None,
                        kind=("broadcast" if broadcast else "direct"),
                        text=txt,
                    )
                )
                if len(arr) > int(self._chat_max_per_agent):
                    # keep only the most recent N messages
                    del arr[:-int(self._chat_max_per_agent)]

            if broadcast:
                for aid in self._agents.keys():
                    if str(aid) != str(from_agent):
                        _push(aid)
                # for sender, only log; do not insert back into own inbox
                return True, "sent", msg_id

            # direct chat: ensure recipient exists
            if str(to_agent) not in self._agents:
                return False, "agent_not_registered", None

            _push(str(to_agent))
            return True, "sent", msg_id

    def pop_chat(
        self,
        agent_id: str,
        max_items: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Pop and clear this agent's inbox.

        If max_items is set and smaller than the inbox, the remaining messages
        are kept for next retrieval.
        """
        with self._lock:
            msgs = self._chat_inbox.pop(str(agent_id), [])
            if max_items is not None and max_items >= 0 and len(msgs) > max_items:
                remain = msgs[max_items:]
                msgs = msgs[:max_items]
                # put back leftover messages
                if remain:
                    self._chat_inbox[str(agent_id)] = remain
            # convert to plain dicts (do not expose dataclass instances)
            return [
                {
                    "msg_id": m.msg_id,
                    "ts_sim": m.ts_sim,
                    "from": m.from_agent,
                    "to": m.to_agent,
                    "kind": m.kind,
                    "text": m.text,
                }
                for m in msgs
            ]


# ==== Module-level singleton helpers ====
_comms_singleton_lock = Lock()
_comms_singleton: Optional[CommsSystem] = None


def reset_comms() -> None:
    """Tear down the global CommsSystem so the next init_comms creates a fresh one.

    Must be called at the start of every episode reset to prevent stale
    scooter / help-request state from leaking across episodes.
    """
    global _comms_singleton
    with _comms_singleton_lock:
        _comms_singleton = None


def init_comms(
    clock: Optional[VirtualClock] = None,
    ambient_temp_c: float = 20.0,
    k_food_per_s: float = 1.0 / 1200.0,
) -> CommsSystem:
    """Create the global CommsSystem once (or return existing instance)."""
    global _comms_singleton
    with _comms_singleton_lock:
        if _comms_singleton is None:
            _comms_singleton = CommsSystem(
                clock=clock,
                ambient_temp_c=ambient_temp_c,
                k_food_per_s=k_food_per_s,
            )
    return _comms_singleton


def set_comms(instance: Optional[CommsSystem]) -> None:
    """Explicitly set or replace the global CommsSystem (mainly for tests)."""
    global _comms_singleton
    with _comms_singleton_lock:
        _comms_singleton = instance


def get_comms() -> Optional[CommsSystem]:
    """Get the global CommsSystem (may be None if not initialized)."""
    return _comms_singleton


def is_comms_ready() -> bool:
    """Return True if the global CommsSystem has been initialized."""
    return _comms_singleton is not None


__all__ = [
    "HelpType",
    "TempBox",
    "HelpRequest",
    "CommsSystem",
    "init_comms",
    "reset_comms",
    "get_comms",
    "set_comms",
    "is_comms_ready",
]