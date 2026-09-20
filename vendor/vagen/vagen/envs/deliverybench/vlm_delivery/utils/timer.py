# utils/timer.py
# -*- coding: utf-8 -*-

import os
import time
import math
from typing import Any, Dict

from ..gameplay.comms import get_comms
from ..base.defs import DMActionKind, TransportMode
from .ctx import ctx_mark_pause, ctx_mark_resume
from .util import is_at_xy, fmt_xy_m

# Flag indicating whether we are running in a multi-agent setup.
IS_MULTI_AGENT = (os.getenv("DELIVERYBENCH_MULTI_AGENT", "1") == "1")


def agent_timers_pause(agent: Any) -> None:
    """
    Pause all internal timers for a given agent.

    This marks the current simulation time in each context so that elapsed
    time during the pause is not counted.
    """
    if agent._timers_paused:
        return

    now = agent.clock.now_sim()

    # Generic pause handling for all time-based contexts.
    ctx_mark_pause(agent._charge_ctx,   now)
    ctx_mark_pause(agent._rest_ctx,     now)
    ctx_mark_pause(agent._wait_ctx,     now)
    ctx_mark_pause(agent._hospital_ctx, now)

    # Notify the comms system, if present.
    comms = get_comms()
    if comms:
        comms.pause_timers_for(str(agent.agent_id))

    agent._timers_paused = True


def agent_timers_resume(agent: Any) -> None:
    """
    Resume internal timers for a given agent.

    If the agent is currently waiting for a VLM call to complete,
    timers remain paused to avoid counting stalled time.
    """
    if not agent._timers_paused:
        return
    if getattr(agent, "_waiting_vlm", False):
        return

    now = agent.clock.now_sim()

    # Resume all time-based contexts.
    ctx_mark_resume(agent._charge_ctx,   now)
    ctx_mark_resume(agent._rest_ctx,     now)
    ctx_mark_resume(agent._wait_ctx,     now)
    ctx_mark_resume(agent._hospital_ctx, now)

    # Rental context uses its own 'last_tick_sim' bookkeeping.
    if agent._rental_ctx is not None:
        agent._rental_ctx["last_tick_sim"] = now

    # Notify the comms system, if present.
    comms = get_comms()
    if comms:
        comms.resume_timers_for(str(agent.agent_id))

    # Update agent's internal bookkeeping for active-time tracking.
    agent._timers_paused = False
    agent._orders_last_tick_sim = now
    agent._life_last_tick_sim = now

    # Apply charging compensation after resuming, based on the stored context.
    agent._advance_charge_to_now()


def handle_poll_time_events(dm: Any) -> None:
    """
    Mirror the original DeliveryMan.poll_time_events(self) logic.

    This helper keeps the logic in one place (with self -> dm) so that
    DeliveryMan can delegate to it. The only differences from the original
    single-agent version are:
      - multi-agent compatibility via IS_MULTI_AGENT + get_comms()
      - charging-spot release in all exit paths of charging.
    """
    now = dm.clock.now_sim()

    # Try auto drop-off when reaching a suitable location.
    dm._auto_try_dropoff()

    # === Active orders: elapsed time tracking ===
    if dm._orders_last_tick_sim is None:
        dm._orders_last_tick_sim = now
    if not dm._timers_paused:
        delta = max(0.0, now - dm._orders_last_tick_sim)
        if delta > 0:
            for o in dm.active_orders:
                if getattr(o, "is_accepted", False) and not getattr(o, "has_delivered", False):
                    cur = float(getattr(o, "sim_elapsed_active_s", 0.0) or 0.0)
                    o.sim_elapsed_active_s = cur + delta
        dm._orders_last_tick_sim = now

    if dm._life_last_tick_sim is None:
        dm._life_last_tick_sim = now

    # === Recorder bookkeeping and lifecycle termination ===
    rec = dm._recorder
    if rec:
        if rec.started_sim_s is None:
            rec.start(now_sim=now)
        if not dm._timers_paused:
            delta = max(0.0, now - dm._life_last_tick_sim)
            rec.tick_active(delta)
            # Track active time by current transport mode.
            try:
                rec.tick_transport(getattr(dm.mode, "value", str(dm.mode)), delta)
            except Exception:
                pass
            dm._life_last_tick_sim = now

        # If lifecycle limits are reached, stop the run and export a report.
        if rec.should_end():
            stop_reason = "unknown"
            stop_message = "Lifecycle reached. Stopping this run."

            # Simulation-time limit.
            sim_time_end = (rec.lifecycle_s > 0) and (rec.active_elapsed_s >= rec.lifecycle_s)

            # Real-time limit.
            realtime_end = False
            elapsed_realtime_hours = 0.0
            if rec.realtime_stop_hours > 0 and rec.realtime_start_ts is not None:
                current_realtime = time.time()
                elapsed_realtime_hours = (current_realtime - rec.realtime_start_ts) / 3600.0
                realtime_end = elapsed_realtime_hours >= rec.realtime_stop_hours

            # VLM call limit.
            vlm_call_end = (rec.vlm_call_limit > 0) and (rec.counters.vlm_calls >= rec.vlm_call_limit)

            if sim_time_end and realtime_end and vlm_call_end:
                stop_reason = "all_limits_reached"
                stop_message = (
                    f"All limits reached: simulation time ({rec.active_elapsed_s/3600:.2f}h), "
                    f"real time ({elapsed_realtime_hours:.2f}h), and VLM calls "
                    f"({rec.counters.vlm_calls}). Stopping this run."
                )
                dm.logger.info("Agent " + dm.agent_id + ": " + stop_message)
            elif sim_time_end and realtime_end:
                stop_reason = "both_times_reached"
                stop_message = (
                    f"Both simulation time ({rec.active_elapsed_s/3600:.2f}h) and real time "
                    f"({elapsed_realtime_hours:.2f}h) reached. Stopping this run."
                )
                dm.logger.info("Agent " + dm.agent_id + ": " + stop_message)
            elif sim_time_end and vlm_call_end:
                stop_reason = "sim_time_and_vlm_reached"
                stop_message = (
                    f"Simulation time ({rec.active_elapsed_s/3600:.2f}h) and VLM calls "
                    f"({rec.counters.vlm_calls}) reached. Stopping this run."
                )
                dm.logger.info("Agent " + dm.agent_id + ": " + stop_message)
            elif realtime_end and vlm_call_end:
                stop_reason = "realtime_and_vlm_reached"
                stop_message = (
                    f"Real time ({elapsed_realtime_hours:.2f}h) and VLM calls "
                    f"({rec.counters.vlm_calls}) reached. Stopping this run."
                )
                dm.logger.info("Agent " + dm.agent_id + ": " + stop_message)
            elif sim_time_end:
                stop_reason = "sim_time_reached"
                stop_message = (
                    f"Simulation time reached ({rec.active_elapsed_s/3600:.2f}h). Stopping this run."
                )
                dm.logger.info("Agent " + dm.agent_id + ": " + stop_message)
            elif realtime_end:
                stop_reason = "realtime_reached"
                stop_message = (
                    f"Real time reached ({elapsed_realtime_hours:.2f}h). Stopping this run."
                )
                dm.logger.info("Agent " + dm.agent_id + ": " + stop_message)
            elif vlm_call_end:
                stop_reason = "vlm_call_limit_reached"
                stop_message = (
                    f"VLM call limit reached ({rec.counters.vlm_calls}). Stopping this run."
                )
                dm.logger.info("Agent " + dm.agent_id + ": " + stop_message)

            rec.mark_end(now_sim=now)
            # Close any ongoing charging/rental sessions to avoid leaking accounting.
            rec.finish_charging(end_ts=now, reason="lifecycle_end")
            rec.finish_rental(end_ts=now)

            # If there is an active charging context, release the spot (multi-agent) and clear it.
            if dm._charge_ctx:
                if IS_MULTI_AGENT:
                    comms = get_comms()
                    if comms and hasattr(comms, "release_charging_spot"):
                        key = dm._charge_ctx.get("station_key") or dm._charge_ctx.get("station_xy")
                        comms.release_charging_spot(key, agent_id=str(dm.agent_id))
                dm._charge_ctx = None

            dm._interrupt_and_stop("lifecycle_ended", stop_message)
            try:
                path = rec.export(dm)
                dm._log(f"run report exported to {path}")
            except Exception as e:
                dm._log(f"run report export failed: {e}")
            dm._lifecycle_done = True
            # Do not advance additional state within the same tick once lifecycle ends.
            return

    # === MOVE arrival / blocked ===
    if dm._move_ctx is not None:
        if dm._interrupt_move_flag:
            dm._move_ctx["blocked"] = 1.0
            dm._interrupt_move_flag = False

        tx = float(dm._move_ctx["tx"])
        ty = float(dm._move_ctx["ty"])
        tol = float(dm._move_ctx["tol"])

        if dm._move_ctx.get("blocked", 0.0) == 1.0:
            dm._move_ctx = None
            if dm._current and dm._current.kind == DMActionKind.MOVE:
                dm._finish_action(success=False)

        elif is_at_xy(dm, tx, ty, tol_cm=tol):
            dm._move_ctx = None
            if dm._current and dm._current.kind == DMActionKind.MOVE:
                dm._finish_action(success=True)

        else:
            # Detect stagnation if the agent is not making progress.
            current_pos = (float(dm.x), float(dm.y))
            last_pos = dm._move_ctx.get("last_position", current_pos)
            last_pos_time = dm._move_ctx.get("last_position_time", now)

            position_change = math.hypot(
                current_pos[0] - last_pos[0],
                current_pos[1] - last_pos[1],
            )
            position_change_threshold = 50.0  # 50 cm movement threshold

            if position_change > position_change_threshold:
                # Position moved significantly; reset stagnation timer.
                dm._move_ctx["last_position"] = current_pos
                dm._move_ctx["last_position_time"] = now
                dm._move_ctx["stagnant_time"] = 0.0
            else:
                # Position did not change much; accumulate stagnation time.
                time_delta = now - last_pos_time
                dm._move_ctx["stagnant_time"] += time_delta
                dm._move_ctx["last_position_time"] = now

                # If stagnation persists beyond the threshold, treat as failure.
                stagnant_threshold = dm._move_ctx.get("stagnant_threshold", 60.0)
                if dm._move_ctx["stagnant_time"] >= stagnant_threshold:
                    dm._log(
                        "move_to failed: position stagnant for "
                        f"{dm._move_ctx['stagnant_time']:.1f}s "
                        f"(threshold: {stagnant_threshold}s)"
                    )
                    dm._move_ctx = None
                    if dm._current and dm._current.kind == DMActionKind.MOVE:
                        dm.vlm_add_error(
                            "move_to failed: cannot move to "
                            f"{fmt_xy_m(tx, ty)}, change a place or choose a new action"
                        )
                        dm._finish_action(success=False)

    # === WAIT (pause-safe; duration-based) ===
    if dm._current and dm._current.kind == DMActionKind.WAIT and dm._wait_ctx:
        ctx = dm._wait_ctx

        # Special case: waiting for a charging session to finish is handled
        # by the CHARGE branch below; here we simply skip updating.
        if str(ctx.get("until") or "").lower() == "charge_done":
            pass
        else:
            if dm._timers_paused:
                ctx["was_paused"] = True
            else:
                # Lazy initialization on the first frame.
                if "last_update_sim" not in ctx:
                    ctx["last_update_sim"] = now
                    # Backward-compatible support for 'end_sim' -> duration_s.
                    dur = float(ctx.get("duration_s", 0.0))
                    if "end_sim" in ctx:
                        dur = max(dur, float(ctx["end_sim"]) - float(now))
                    ctx["duration_s"] = max(0.0, dur)
                    ctx.setdefault("elapsed_active_s", 0.0)

                # When resuming from pause, discard the paused interval.
                if ctx.pop("was_paused", False):
                    ctx["last_update_sim"] = now

                # Advance active waiting time.
                delta_s = max(0.0, float(now - ctx["last_update_sim"]))
                ctx["last_update_sim"] = now
                ctx["elapsed_active_s"] = float(ctx.get("elapsed_active_s", 0.0)) + delta_s

                # Finish once the requested duration has been reached.
                if ctx["elapsed_active_s"] + 1e-6 >= float(ctx.get("duration_s", 0.0)):
                    dm._wait_ctx = None
                    dm._finish_action(success=True)

    # === CHARGE (e-scooter charging progress; pause-safe, rate-based) ===
    if dm._charge_ctx and dm._charge_ctx.get("scooter_ref"):
        ctx = dm._charge_ctx
        sc = ctx["scooter_ref"]  # The scooter currently being charged.

        def _release_charging_spot_if_needed() -> None:
            """Release a reserved charging spot in multi-agent setups (if any)."""
            if not IS_MULTI_AGENT:
                return
            comms = get_comms()
            if not (comms and hasattr(comms, "release_charging_spot")):
                return
            key = ctx.get("station_key") or ctx.get("station_xy")
            if key is not None:
                comms.release_charging_spot(key, agent_id=str(dm.agent_id))

        # When timers are paused, do not progress charging or billing.
        if dm._timers_paused:
            ctx["was_paused"] = True
        else:
            # Lazy initialization on the first frame of charging.
            if "last_update_sim" not in ctx:
                ctx["last_update_sim"] = now
                ctx.setdefault("elapsed_active_s", 0.0)
                ctx.setdefault("start_pct", float(ctx.get("start_pct", getattr(sc, "battery_pct", 0.0))))
                ctx.setdefault("target_pct", float(ctx.get("target_pct", 100.0)))
                ctx.setdefault("paid_pct", float(ctx.get("paid_pct", ctx["start_pct"])))
                ctx.setdefault("price_per_pct", float(ctx.get("price_per_pct", dm.charge_price_per_pct)))
                ctx.setdefault(
                    "park_xy_start",
                    tuple(getattr(sc, "park_xy", (dm.x, dm.y)) or (dm.x, dm.y)),
                )

            # When resuming from pause, discard the paused interval.
            if ctx.pop("was_paused", False):
                ctx["last_update_sim"] = now

            # Advance elapsed charging time (only when not paused).
            delta_s = max(0.0, float(now - ctx["last_update_sim"]))
            ctx["last_update_sim"] = now
            ctx["elapsed_active_s"] = float(ctx.get("elapsed_active_s", 0.0)) + delta_s

            # Charging rate is specified as percentage per minute; convert to per second.
            rate_per_min = float(getattr(sc, "charge_rate_pct_per_min", 0.0) or 0.0)
            rate_pct_per_s = rate_per_min / 60.0

            p0 = float(ctx["start_pct"])
            pt = float(ctx["target_pct"])
            paid_pct = float(ctx.get("paid_pct", p0))
            price_per_pct = float(ctx.get("price_per_pct", dm.charge_price_per_pct))

            cur_should = p0 + rate_pct_per_s * ctx["elapsed_active_s"]
            lo, hi = (p0, pt) if p0 <= pt else (pt, p0)
            cur_should = min(max(cur_should, lo), hi)

            add_pct_need = max(0.0, cur_should - paid_pct)
            max_afford_pct = float("inf") if price_per_pct <= 0.0 else max(0.0, dm.earnings_total) / price_per_pct
            add_pct_can = min(add_pct_need, max_afford_pct)

            # Apply payment and update scooter charge.
            if add_pct_can > 1e-9:
                cost = add_pct_can * price_per_pct
                dm.earnings_total = max(0.0, dm.earnings_total - cost)
                paid_pct = paid_pct + add_pct_can
                ctx["paid_pct"] = paid_pct
                sc.charge_to(min(100.0, max(0.0, paid_pct)))

                if dm._recorder:
                    dm._recorder.accrue_charging(
                        ts_sim=now,
                        which=str(ctx.get("which", "own")),
                        delta_pct=float(add_pct_can),
                        cost=float(cost),
                        req_id=(int(ctx.get("req_id")) if ctx.get("req_id") is not None else None),
                        start_ts=float(ctx.get("start_sim", now)),
                    )

            finished_by_target = (paid_pct + 1e-6) >= pt
            out_of_money = (add_pct_need > 1e-9) and (add_pct_can + 1e-9 < add_pct_need)

            rec = dm._recorder

            if finished_by_target:
                # Target percentage reached: finalize charging.
                sc.charge_to(min(100.0, max(0.0, pt)))
                if rec:
                    rec.finish_charging(end_ts=now, reason="finished", target_pct=pt)

                # Release the charging spot (multi-agent) before clearing context.
                _release_charging_spot_if_needed()
                dm._charge_ctx = None

                px, py = (sc.park_xy if getattr(sc, "park_xy", None) else (dm.x, dm.y))
                loc = fmt_xy_m(px, py)
                which = ctx.get("which", "own")
                dm._log(f"charging finished ({which}): {p0:.0f}% -> {pt:.0f}% at {loc}")
                dm.vlm_ephemeral["scooter_ready"] = (
                    f"{'Assisting scooter' if which == 'assist' else 'Your scooter'} charged to {pt:.0f}%. "
                    f"It's parked at {loc}. You can come here to retrieve it."
                )
                # If we were in a WAIT('charge_done') action, mark it as finished.
                if dm._current and dm._current.kind == DMActionKind.WAIT and dm._wait_ctx:
                    dm._wait_ctx = None
                    dm._finish_action(success=True)

            elif out_of_money:
                # Charging stopped due to insufficient funds.
                if rec:
                    rec.finish_charging(end_ts=now, reason="no_money", target_pct=pt)
                    rec.inc("charge_insufficient", 1)

                _release_charging_spot_if_needed()
                dm._charge_ctx = None

                px, py = (sc.park_xy if getattr(sc, "park_xy", None) else (dm.x, dm.y))
                loc = fmt_xy_m(px, py)
                which = ctx.get("which", "own")
                dm._log(
                    f"charging interrupted ({which}) for insufficient funds at "
                    f"{paid_pct:.0f}% (target {pt:.0f}%)"
                )
                dm.vlm_ephemeral["charging_interrupted"] = (
                    f"Charging was interrupted due to insufficient funds at {paid_pct:.0f}%. "
                    f"The scooter is parked at {loc}. Earn more money, then CHARGE_ESCOOTER again."
                )
                # If we were in a WAIT('charge_done') action, also finish it.
                if dm._current and dm._current.kind == DMActionKind.WAIT and dm._wait_ctx:
                    dm._wait_ctx = None
                    dm._finish_action(success=True)

            elif (
                getattr(sc, "park_xy", None) is None
                or (ctx.get("park_xy_start") and tuple(sc.park_xy or ()) != tuple(ctx["park_xy_start"]))
            ):
                # Charging stopped because the scooter was moved away from the original spot.
                if rec:
                    rec.finish_charging(end_ts=now, reason="moved", target_pct=pt)

                _release_charging_spot_if_needed()
                dm._charge_ctx = None

                px, py = (sc.park_xy if getattr(sc, "park_xy", None) else (dm.x, dm.y))
                loc = fmt_xy_m(px, py)
                which = ctx.get("which", "own")
                dm._log(
                    f"charging interrupted ({which}) at {paid_pct:.0f}% (scooter moved)"
                )
                dm.vlm_ephemeral["charging_interrupted"] = (
                    f"Charging was interrupted at {paid_pct:.0f}% because the scooter was moved."
                )
                # If we were in a WAIT('charge_done') action, also finish it.
                if dm._current and dm._current.kind == DMActionKind.WAIT and dm._wait_ctx:
                    dm._wait_ctx = None
                    dm._finish_action(success=True)

    # === REST ===
    if dm._rest_ctx and not dm._timers_paused:
        t0, t1 = dm._rest_ctx["start_sim"], dm._rest_ctx["end_sim"]
        e0, et = dm._rest_ctx["start_pct"], dm._rest_ctx["target_pct"]
        if t1 <= t0:
            cur = et
        else:
            r = max(0.0, min(1.0, (now - t0) / (t1 - t0)))
            cur = e0 + (et - e0) * r
        dm.energy_pct = float(cur)
        if now >= t1:
            dm.energy_pct = float(et)
            dm._log(f"rest finished: {e0:.0f}% -> {et:.0f}%")
            dm._rest_ctx = None
            if dm._current and dm._current.kind == DMActionKind.REST:
                dm._finish_action(success=True)

    # === HOSPITAL ===
    if dm._hospital_ctx and now >= dm._hospital_ctx["end_sim"] and not dm._timers_paused:
        dm.rescue()
        dm._hospital_ctx = None
        dm._log("rescue finished: full energy at Hospital")
        if not getattr(dm, "manual_step", False):
            dm.kickstart()

    # === Rental billing ===
    if dm._rental_ctx and not dm._timers_paused:
        dt = max(0.0, now - float(dm._rental_ctx["last_tick_sim"]))
        if dt > 0:
            rate = float(dm._rental_ctx["rate_per_min"])
            cost = rate * (dt / 60.0)
            old_balance = float(dm.earnings_total)
            if dm.earnings_total - cost <= 0.0:
                in_car = (dm.mode == TransportMode.CAR)
                dm.car = None
                dm._rental_ctx = None
                dm.earnings_total = max(0.0, dm.earnings_total - cost)
                dm._interrupt_and_stop(
                    "car_rental_ended",
                    "Your car rental has ended (insufficient funds). You may SWITCH_TRANSPORT(to='walk'), "
                    "RENT_CAR(...) again, or choose another mode.",
                )
                dm._log("rental ended (no money) -> interrupt; waiting for decision")
                charge_amount = min(cost, max(0.0, old_balance))
                if dm._recorder and charge_amount > 1e-12:
                    dm._recorder.accrue_rental(
                        dt_s=float(dt),
                        cost=float(charge_amount),
                        start_ts=float(dm._rental_ctx.get("start_sim", now)),
                    )
                if dm._recorder:
                    dm._recorder.finish_rental(end_ts=now)
            else:
                dm.earnings_total -= cost
                dm._rental_ctx["last_tick_sim"] = now
                if dm._recorder and cost > 1e-12:
                    dm._recorder.accrue_rental(
                        dt_s=float(dt),
                        cost=float(cost),
                        start_ts=float(dm._rental_ctx.get("start_sim", now)),
                    )

    # === Insulated bag: temperature and odor ===
    if not dm._timers_paused:
        if dm._last_bag_tick_sim is None:
            dm._last_bag_tick_sim = now
        else:
            dt = max(0.0, now - dm._last_bag_tick_sim)
            if dt > 0:
                if dm.insulated_bag:
                    dm.insulated_bag.tick_temperatures(dt)
                    dm.insulated_bag.tick_odor(dt)

                # Bare-food Newton cooling: food carried in inventory but NOT
                # inside the insulated bag cools toward ambient.
                # This decouples the thermal system from the bag — food spoils
                # on its own; the bag (+ heat packs) is the tool to prevent it.
                if getattr(dm, "_enable_food_temperature",
                           dm.cfg.get("enable_food_temperature", True)):
                    bag_item_ids: set = set()
                    if dm.insulated_bag:
                        for comp in dm.insulated_bag._comps:
                            for it in comp:
                                bag_item_ids.add(id(it))
                    k = float(getattr(dm, "k_food_per_s", 1.0 / 1800.0))
                    ambient = float(getattr(dm, "ambient_temp_c", 22.0))
                    decay = math.exp(-k * dt)
                    # carried orders live in dm.active_orders, not the pending pool
                    order_lookup = {int(o.id): o for o in (getattr(dm, "active_orders", []) or [])}
                    for oid in list(getattr(dm, "carrying", [])):
                        order = order_lookup.get(int(oid))
                        if order is None:
                            continue
                        for it in (getattr(order, "items", []) or []):
                            if id(it) in bag_item_ids:
                                continue
                            tc = getattr(it, "temp_c", float("nan"))
                            if not math.isnan(tc):
                                it.temp_c = ambient + (tc - ambient) * decay

                # Bare-food odor mixing: items outside the bag share one space
                if getattr(dm, "_enable_food_smell",
                           dm.cfg.get("enable_food_smell", True)):
                    bag_item_ids = set()
                    if dm.insulated_bag:
                        for comp in dm.insulated_bag._comps:
                            for it in comp:
                                bag_item_ids.add(id(it))
                    tau_min = float(getattr(dm.insulated_bag, "odor_mix_tau_min", 1.5)
                                    if dm.insulated_bag else 1.5)
                    alpha = min(0.5, dt / max(1e-6, tau_min * 60.0))
                    order_lookup = {int(o.id): o for o in (getattr(dm, "active_orders", []) or [])}
                    bare = [it for oid in list(getattr(dm, "carrying", []))
                            if (order := order_lookup.get(int(oid))) is not None
                            for it in (getattr(order, "items", []) or [])
                            if id(it) not in bag_item_ids]
                    levels = [float(getattr(it, "odor_contamination", 0.0) or 0.0)
                              for it in bare]
                    target = max(levels, default=0.0)
                    if target > 0.0:
                        for it, oi in zip(bare, levels):
                            setattr(it, "odor_contamination",
                                    min(1.0, oi + alpha * (target - oi)))

                dm._last_bag_tick_sim = now

    # === Comms inbox ===
    comms = get_comms()
    if comms:
        inbox = comms.pop_chat(str(dm.agent_id), max_items=20)
        if inbox:
            # Render as simple text lines, newest at the bottom.
            lines = []
            for m in inbox:
                ts = float(m.get("ts_sim", 0.0))
                src = str(m.get("from", ""))
                kind = m.get("kind", "direct")
                txt = str(m.get("text", ""))
                if kind == "broadcast":
                    lines.append(f"[broadcast] from {src}: {txt}")
                else:
                    lines.append(f"from {src}: {txt}")
            dm.vlm_ephemeral["chat_inbox"] = "\n".join(lines[-20:])

    # Refresh hints for nearby POIs (including charging hints for assist/own scooters).
    dm._refresh_poi_hints_nearby()

    # === Bus riding state update ===
    if dm._bus_ctx and dm.mode == TransportMode.BUS:
        dm._update_bus_riding(now)