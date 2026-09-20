# gameplay/help.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from .comms import CommsSystem

class HelpType(str, Enum):
    HELP_PICKUP   = "HELP_PICKUP"    # helper picks up at restaurant -> drops at deliver_xy (TempBox)
    HELP_DELIVERY = "HELP_DELIVERY"  # helper delivers an existing order (publisher prepares/places food)
    HELP_BUY      = "HELP_BUY"       # helper buys items -> drops at deliver_xy (TempBox)
    HELP_CHARGE   = "HELP_CHARGE"    # helper charges e-scooter to target_pct -> drops at deliver_xy (TempBox)


@dataclass
class HelpRequest:
    req_id: int
    publisher_id: str
    kind: HelpType
    reward: float
    time_limit_s: float
    created_sim: float

    # task params
    order_id: Optional[int] = None
    buy_items: Optional[Dict[str, int]] = None
    target_pct: Optional[float] = None

    # handover/finish locations
    provide_xy: Optional[Tuple[float, float]] = None
    deliver_xy: Optional[Tuple[float, float]] = None

    brief: str = ""

    # runtime state
    accepted_by: Optional[str] = None
    completed: bool = False
    order_ref: Optional[Any] = None

    # ==== Timer (starts when the task becomes "ready") ====
    timer_started_sim: Optional[float] = None      # None = not started
    timer_paused_accum_s: float = 0.0
    timer_paused_at: Optional[float] = None
    start_note: Optional[str] = None

    # ==== Funds (escrow and upfront) ====
    escrow_reward: float = 0.0
    upfront_paid_to_helper: float = 0.0

    # ==== Result ====
    result: Optional[str] = None                   # 'success' | 'partial' | 'fail'
    last_helper_drop: Optional[Dict[str, Any]] = None

    # ---- Text rendering (English only) ----
    def to_text_for(
        self,
        agent_id: str,
        comms: "CommsSystem",
        now: Optional[float] = None,
        view_as: Optional[str] = None,  # "helper" / "publisher" / None(auto)
    ) -> str:
        """
        Human-readable instruction block for a given agent:
        - Clear next steps (where to go / what to do / when to REPORT).
        - Shows TempBox state (placed? what/where?).
        - Explicitly states timer rules and both agent IDs.
        - Always includes actionable coordinates for pickup / handover / dropoff.
        """
        now = comms.clock.now_sim() if now is None else float(now)

        def _fmt_xy(xy):
            return f"({xy[0]/100.0:.2f}m, {xy[1]/100.0:.2f}m)" if xy else "N/A"

        def _fmt_place(name: Optional[str], xy: Optional[Tuple[float, float]]) -> str:
            xy_txt = _fmt_xy(xy)
            if name and xy:
                return f"{name} @ {xy_txt}"
            if xy:
                return xy_txt
            if name:
                return f"{name} @ N/A"
            return "N/A"

        def _fmt_buy_list(buy_items: Optional[Dict[str, int]]) -> str:
            if not buy_items:
                return "the required items"
            parts = [f"{k} x{int(v)}" for k, v in buy_items.items()]
            return ", ".join(parts)

        # time text (shows overdue explicitly)
        if self.timer_started_sim is None:
            head_time = "time left — not started"
        else:
            elapsed = comms._effective_elapsed_s(self, now)
            remaining = float(self.time_limit_s) - float(elapsed)
            if remaining >= 0:
                mins = int(math.ceil(remaining / 60.0))
                head_time = f"time left {mins} min"
            else:
                over = int(math.ceil((-remaining) / 60.0))
                head_time = f"time left 0 min (overdue by {over} min)"

        reward = f"${self.reward:.2f}"
        head = f"[Help #{self.req_id}] {self.kind.value.replace('HELP_', '').title()} — reward {reward}, {head_time}."

        # roles
        am_helper = (self.accepted_by == str(agent_id))
        am_publisher = (self.publisher_id == str(agent_id))
        if view_as == "helper":
            am_helper, am_publisher = True, False
        elif view_as == "publisher":
            am_helper, am_publisher = False, True

        helper_label = self.accepted_by if self.accepted_by else "TBD"
        role_line = f"Participants: help-seeker (agent {self.publisher_id}), helper (agent {helper_label})."

        # timer rule line
        when_line = ""
        if self.kind in (HelpType.HELP_DELIVERY, HelpType.HELP_CHARGE):
            if self.timer_started_sim is None:
                thing = "order food" if self.kind == HelpType.HELP_DELIVERY else "e-scooter"
                when_line = (
                    f"Note: the timer starts when the help-seeker (agent {self.publisher_id}) "
                    f"places the {thing} into the TempBox (handover ready). Not started yet."
                )
            else:
                when_line = "Note: timer has started (counting from the handover-ready moment)."
        else:
            # BUY / PICKUP
            if self.accepted_by:
                when_line = "Note: this task requires no handover from the help-seeker; timer runs from acceptance."
            else:
                when_line = "Note: once accepted, the timer runs immediately."

        # TempBox overview
        info = comms.get_temp_box_info(int(self.req_id)) or {}
        pub_box = info.get("publisher_box", {}) or {}
        hel_box = info.get("helper_box", {}) or {}

        # ---- Canonical locations (always compute name + xy) ----
        # Provide / Handover point
        provide_xy = self.provide_xy
        provide_line = _fmt_place(None, provide_xy)

        # Destination / Dropoff
        dst_name, dst_xy = None, self.deliver_xy
        if self.kind == HelpType.HELP_DELIVERY and self.order_ref is not None:
            try:
                dst_name = getattr(self.order_ref, "dropoff_road_name", None)
                dn = getattr(self.order_ref, "dropoff_node", None)
                if dn is not None:
                    x = float(getattr(dn.position, "x", 0.0))
                    y = float(getattr(dn.position, "y", 0.0))
                    dst_xy = (x, y)
            except Exception:
                pass
        dst_line = _fmt_place(dst_name, dst_xy)

        # Pickup (only meaningful for HELP_PICKUP / part of HELP_DELIVERY when helper picks from publisher box)
        pickup_name, pickup_xy = None, None
        try:
            if self.order_ref is not None:
                pickup_name = getattr(self.order_ref, "pickup_road_name", None)
                pn = getattr(self.order_ref, "pickup_node", None)
                if pn is not None:
                    x = float(getattr(pn.position, "x", 0.0))
                    y = float(getattr(pn.position, "y", 0.0))
                    pickup_xy = (x, y)
        except Exception:
            pass
        # fallback: use provide_xy as pickup when appropriate
        if pickup_xy is None and self.provide_xy is not None:
            pickup_xy = self.provide_xy
        pickup_line = _fmt_place(pickup_name, pickup_xy if pickup_xy else None)
        if pickup_line == "N/A":
            pickup_line = "the restaurant @ N/A"

        _items_suffix, _note_suffix = "", ""
        try:
            if self.order_ref is not None:
                # items -> " | items: A, B, C"
                _names = [getattr(it, "name", str(it)) for it in (getattr(self.order_ref, "items", []) or [])]
                if _names:
                    _items_suffix = " | items: " + ", ".join(_names)
                # note  -> " | note: xxx"
                _note_txt = (getattr(self.order_ref, "special_note", "") or "").strip()
                if _note_txt:
                    _note_suffix = " | note: " + _note_txt
        except Exception:
            pass

        plan = ""
        plan_intro = f"{role_line} {when_line} "

        if self.kind == HelpType.HELP_BUY:
            buy_txt = _fmt_buy_list(self.buy_items)
            if am_helper:
                if hel_box.get("has_content") and hel_box.get("ready"):
                    plan = (
                        f"Your TempBox at {_fmt_xy(hel_box.get('xy'))} already has items. "
                        f"If the list is complete ({buy_txt}), REPORT_HELP_FINISHED. "
                        f"Otherwise buy missing items and update the box."
                    )
                else:
                    plan = (
                        f"Go to a store and buy {buy_txt}. "
                        f"Then deliver to {dst_line}. "
                        f"On arrival, PLACE_TEMP_BOX with the purchased items and REPORT_HELP_FINISHED."
                    )
            elif am_publisher:
                if hel_box.get("has_content"):
                    plan = (
                        f"Helper has dropped the purchased items at {_fmt_xy(hel_box.get('xy'))}. "
                        f"You can TAKE_FROM_TEMP_BOX when convenient."
                    )
                else:
                    who = f" (accepted by agent {self.accepted_by})" if self.accepted_by else " (waiting for a helper)"
                    plan = (
                        f"Helper will buy {buy_txt} and drop at {dst_line}{who}. "
                        f"No action required from you until they place the TempBox."
                    )
            else:
                plan = (
                    f"Buy {buy_txt} and deliver to {dst_line}. "
                    f"PLACE_TEMP_BOX on arrival, then REPORT_HELP_FINISHED."
                )

        elif self.kind == HelpType.HELP_CHARGE:
            tgt = int(self.target_pct if self.target_pct is not None else 100)
            if am_helper:
                if pub_box.get("has_content"):
                    plan = (
                        f"Pick the e-scooter from the help-seeker's TempBox at {_fmt_xy(pub_box.get('xy'))}. "
                        f"Charge to {tgt}% at a charging station, then deliver to {dst_line}. "
                        f"PLACE_TEMP_BOX(req_id={self.req_id}, content={{'escooter':''}}) and REPORT_HELP_FINISHED."
                    )
                elif pub_box.get("xy"):
                    if self.timer_started_sim is not None:
                        sc = comms.get_scooter_by_owner(self.publisher_id)
                        cur = int(round(float(getattr(sc, "battery_pct", 0.0)))) if sc is not None else None
                        if sc and cur is not None and cur >= tgt:
                            loc_txt = (
                                _fmt_xy(getattr(sc, "park_xy", None))
                                if getattr(sc, "park_xy", None)
                                else "your current location"
                            )
                            plan = (
                                f"You have taken the e-scooter. It's charged to {cur}% and parked at {loc_txt}. "
                                f"Deliver to {dst_line}, PLACE_TEMP_BOX(req_id={self.req_id}, content={{'escooter':''}}) "
                                f"and REPORT_HELP_FINISHED."
                            )
                        else:
                            cur_txt = f" (current ~{cur}%)" if cur is not None else ""
                            plan = (
                                f"You have taken the e-scooter from the handover point {_fmt_xy(pub_box.get('xy'))}. "
                                f"Go charge to {tgt}%{cur_txt}, then deliver to {dst_line}. "
                                f"PLACE_TEMP_BOX(req_id={self.req_id}, content={{'escooter':''}}) and REPORT_HELP_FINISHED."
                            )
                    else:
                        plan = (
                            f"Handover TempBox set at {_fmt_xy(pub_box.get('xy'))} but empty. "
                            f"Wait for the help-seeker to place the scooter, then charge to {tgt}%, "
                            f"deliver to {dst_line}, PLACE_TEMP_BOX and REPORT_HELP_FINISHED."
                        )
                else:
                    plan = (
                        f"Help-seeker hasn't placed the scooter yet. Handover point: {provide_line}. "
                        f"Once picked, charge to {tgt}% at a charging station, deliver to {dst_line}, "
                        f"PLACE_TEMP_BOX and REPORT_HELP_FINISHED."
                    )
            elif am_publisher:
                if pub_box.get("has_content"):
                    plan = (
                        f"You placed the scooter at {_fmt_xy(pub_box.get('xy'))}. "
                        f"Waiting for the helper to charge it to {tgt}% and drop at {dst_line}."
                    )
                else:
                    who = f" (accepted by agent {self.accepted_by})" if self.accepted_by else " (on board)"
                    plan = (
                        f"Please place your e-scooter at the handover point {provide_line}{who} for the helper. "
                        f"Target charge {tgt}%, drop at {dst_line}."
                    )
            else:
                plan = (
                    f"Charge the e-scooter to {tgt}%, deliver to {dst_line}. "
                    f"PLACE_TEMP_BOX, then REPORT_HELP_FINISHED."
                )

        elif self.kind == HelpType.HELP_DELIVERY:
            if am_helper:
                if pub_box.get("has_content"):
                    plan = (
                        f"Pick up the food from the help-seeker's TempBox at {_fmt_xy(pub_box.get('xy'))}"
                        f"{_items_suffix}{_note_suffix}. "
                        f"Deliver to {dst_line}. After delivery, REPORT_HELP_FINISHED."
                    )
                elif pub_box.get("xy"):
                    if self.timer_started_sim is not None:
                        plan = (
                            f"You have taken the food from {_fmt_xy(pub_box.get('xy'))}"
                            f"{_items_suffix}{_note_suffix}. "
                            f"Deliver to {dst_line}. After delivery, REPORT_HELP_FINISHED."
                        )
                    else:
                        plan = (
                            f"TempBox set at {_fmt_xy(pub_box.get('xy'))} but empty. "
                            f"Wait for the help-seeker to place the food{_items_suffix}{_note_suffix}, "
                            f"then deliver to {dst_line}. After delivery, REPORT_HELP_FINISHED."
                        )
                else:
                    plan = (
                        f"Help-seeker hasn't placed the food yet. Handover point: {provide_line}. "
                        f"{_items_suffix}{_note_suffix}, deliver to {dst_line}. "
                        f"After delivery, REPORT_HELP_FINISHED."
                    )
            elif am_publisher:
                need_pickup_first = False
                try:
                    if self.order_ref is not None:
                        has_picked = bool(getattr(self.order_ref, "has_picked_up", False))
                        has_delivered = bool(getattr(self.order_ref, "has_delivered", False))
                        need_pickup_first = (not has_picked) and (not has_delivered)
                except Exception:
                    pass

                if pub_box.get("has_content"):
                    plan = (
                        f"You placed the food at {_fmt_xy(pub_box.get('xy'))}. "
                        f"Waiting for the helper to deliver to {dst_line}."
                    )
                else:
                    prefix = ""
                    if need_pickup_first and self.order_id is not None:
                        prefix = (
                            f"First PICK UP order #{int(self.order_id)} at the restaurant, then "
                        )
                    who = f" (accepted by agent {self.accepted_by})" if self.accepted_by else " (on board)"
                    plan = (
                        f"{prefix}place the food at the handover point {provide_line}{who}. "
                        f"The helper will take it and deliver to {dst_line}."
                    )
            else:
                plan = (
                    f"Handover at {provide_line}. "
                    f"Deliver to {dst_line}. REPORT_HELP_FINISHED after dropoff."
                )

        elif self.kind == HelpType.HELP_PICKUP:
            if am_helper:
                if pub_box.get("has_content"):
                    # unusual: publisher pre-placed items in a box
                    plan = (
                        f"Items available at {_fmt_xy(pub_box.get('xy'))}"
                        f"{_items_suffix}{_note_suffix}. "
                        f"Go to {dst_line}, PLACE_TEMP_BOX(req_id={self.req_id}, content={{'food':''}}) "
                        f"and REPORT_HELP_FINISHED."
                    )
                else:
                    oid_txt = (
                        f"order #{int(self.order_id)}"
                        if self.order_id is not None
                        else "the order"
                    )
                    plan = (
                        f"Pick up {oid_txt} at {pickup_line}"
                        f"{_items_suffix}{_note_suffix}. "
                        f"Then go to {dst_line}, PLACE_TEMP_BOX(req_id={self.req_id}, content={{'food':''}}) "
                        f"and REPORT_HELP_FINISHED."
                    )
            elif am_publisher:
                who = f" (accepted by agent {self.accepted_by})" if self.accepted_by else " (on board)"
                plan = (
                    f"Helper will pick up the order at {pickup_line} and head to {dst_line}{who}. "
                    f"They will PLACE_TEMP_BOX at the destination and REPORT_HELP_FINISHED. "
                    f"You can TAKE_FROM_TEMP_BOX afterwards."
                )
            else:
                plan = (
                    f"Pick up the order at {pickup_line}. "
                    f"Go to {dst_line}, PLACE_TEMP_BOX(req_id={self.req_id}, content={{'food':''}}), "
                    f"then REPORT_HELP_FINISHED."
                )
        else:
            plan = f"Destination: {dst_line}."

        return f"{head} {plan_intro}{plan}"