# utils/util.py
# -*- coding: utf-8 -*-

"""
Utility functions for image conversion, filename sanitization, spatial checks,
POI queries, formatting helpers, and DeliveryMan logging/introspection tools.

All functionality is preserved exactly; only comments and organization
have been made clearer and more structured.
"""

import numpy as np
from PIL import Image
from io import BytesIO
import re
from typing import Any, Dict, Tuple, Optional, List
import math


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def ensure_png_bytes(img) -> bytes:
    """
    Convert input into PNG bytes.

    Supported inputs:
        • bytes / bytearray → returned as-is
        • numpy.ndarray → converted to PNG bytes
          Supports grayscale, BGR, BGRA, and float arrays.

    Returns:
        PNG-encoded bytes.
    """
    # Already bytes → return directly
    if isinstance(img, (bytes, bytearray)):
        return bytes(img)

    # Otherwise must be ndarray
    arr = img

    # Normalize dtype to uint8
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            max_val = float(arr.max()) if arr.size else 1.0
            if max_val <= 1.0:
                arr = (np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8)
            else:
                arr = np.clip(arr, 0.0, 255.0).round().astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

    # Determine image mode
    if arr.ndim == 2:
        mode = "L"
        out = arr
    elif arr.ndim == 3 and arr.shape[2] == 3:
        # Convert BGR → RGB
        out = arr[:, :, ::-1].copy()
        mode = "RGB"
    elif arr.ndim == 3 and arr.shape[2] == 4:
        # Convert BGRA → RGBA
        b, g, r, a = arr.transpose(2, 0, 1)
        out = np.dstack([r, g, b, a])
        mode = "RGBA"
    else:
        raise ValueError(f"Unsupported ndarray shape: {arr.shape}")

    # Encode PNG
    out = np.ascontiguousarray(out)
    bio = BytesIO()
    Image.fromarray(out, mode=mode).save(bio, format="PNG")
    return bio.getvalue()


# ---------------------------------------------------------------------------
# Filename / config helpers
# ---------------------------------------------------------------------------

def sanitize_filename(s: str) -> str:
    """
    Sanitize a string for safe filesystem usage:
      • Replace forbidden characters with '-'
      • Collapse whitespace into '-'
    """
    s = re.sub(r'[\\/:*?"<>|]+', '-', s)
    s = re.sub(r'\s+', '-', s).strip('-')
    return s


def get_tol(cfg: Dict[str, Any], key: str, fallback: float = 500.0) -> float:
    """Fetch tolerance (in cm) from config, falling back to default."""
    return float(cfg.get("tolerance_cm", {}).get(key, fallback))


def pace_scale(pace_state: str, pace_scales: Dict[str, float]) -> float:
    """Return multiplicative speed factor for the given pace state."""
    return float(pace_scales.get(pace_state, 1.0))


# ---------------------------------------------------------------------------
# Spatial helpers
# ---------------------------------------------------------------------------

def xy_of_node(node: Any) -> Optional[Tuple[float, float]]:
    """
    Extract (x, y) coordinates (in cm) from a city_map node object.
    """
    return float(node.position.x), float(node.position.y)


def is_at_xy(agent: Any, x: float, y: float, tol_cm: Optional[float] = None) -> bool:
    """
    Return True if the agent is within `tol_cm` of (x, y) in cm.
    If tol_cm is None, use config's "nearby" tolerance.
    """
    tol_cm = float(tol_cm) if tol_cm is not None else float(get_tol(agent.cfg, "nearby"))
    return math.hypot(agent.x - x, agent.y - y) <= tol_cm


def nearest_poi_xy(agent: Any, kind: str, tol_cm: Optional[float] = None) -> Optional[Tuple[float, float]]:
    """
    Find the nearest POI of the given type within `tol_cm` (cm).

    Anchor is the POI's dock waypoint — the reachable point on the road
    graph — so an agent standing on the dock counts as "at" the POI even
    when the door coordinate is several meters away across the sidewalk.
    Returns:
        (x, y) of the dock in cm, or None if no POI is close enough.
    """
    tol = float(tol_cm) if tol_cm is not None else float("inf")
    best_xy = None
    best_d = float("inf")

    for meta in getattr(agent.city_map, "poi_meta", []):
        node = meta.get("node")
        if node is None or getattr(node, "type", "") != kind:
            continue
        dock = meta.get("dock_node")
        if dock is None:
            continue
        xy = xy_of_node(dock)
        if not xy:
            continue

        d = math.hypot(agent.x - xy[0], agent.y - xy[1])
        if d < best_d:
            best_d, best_xy = d, xy

    return best_xy if best_xy is not None and best_d <= tol else None


def closest_poi_xy(agent: Any, kind: str) -> Optional[Tuple[float, float]]:
    """
    Find the closest POI of a given type, regardless of distance.
    Anchor is the dock waypoint (see nearest_poi_xy).
    """
    best_xy = None
    best_d = float("inf")

    for meta in getattr(agent.city_map, "poi_meta", []):
        node = meta.get("node")
        if node is None or getattr(node, "type", "") != kind:
            continue
        dock = meta.get("dock_node")
        if dock is None:
            continue
        xy = xy_of_node(dock)
        if not xy:
            continue

        d = math.hypot(agent.x - xy[0], agent.y - xy[1])
        if d < best_d:
            best_d, best_xy = d, xy

    return best_xy


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt_xy_m(x_cm: float, y_cm: float) -> str:
    """Format cm coordinates into '(xx.xx m, yy.yy m)'."""
    return f"({x_cm / 100.0:.2f}m, {y_cm / 100.0:.2f}m)"


def fmt_xy_m_opt(xy: Optional[Tuple[float, float]]) -> str:
    """Like fmt_xy_m, but returns 'N/A' if xy is None."""
    if not xy:
        return "N/A"
    x, y = xy
    return fmt_xy_m(float(x), float(y))


def remaining_range_m(agent: Any) -> Optional[float]:
    """
    Estimate remaining travel distance (in meters) for the agent's e-scooter.
    Returns None if there is no e-scooter.
    """
    if not getattr(agent, "e_scooter", None):
        return None
    return float(agent.e_scooter.battery_pct) / max(1e-9, float(agent.scooter_batt_decay_pct_per_m))


def fmt_time(sim_seconds: float) -> str:
    """Format simulation seconds as 'HH:MM:SS'."""
    hours = int(sim_seconds // 3600)
    minutes = int((sim_seconds % 3600) // 60)
    seconds = int(sim_seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


# ---------------------------------------------------------------------------
# DeliveryMan logging helpers
# ---------------------------------------------------------------------------

def dm_log(dm: Any, text: str) -> None:
    """
    Unified DeliveryMan logging helper:
      • If viewer exists with log_action → forward message there.
      • Forward to dm.logger if available.
      • Else print to stdout.
    """
    viewer = getattr(dm, "_viewer", None)
    if viewer is not None and hasattr(viewer, "log_action"):
        viewer_id = getattr(dm, "_viewer_agent_id", None)
        prefix = f"[Agent {viewer_id or dm.name}] "
        viewer.log_action(prefix + text, also_print=False)

        logger = getattr(dm, "logger", None)
        if logger is not None:
            logger.info(f"[Agent {dm.agent_id}] {text}")
    else:
        print(f"[DeliveryMan {dm.name}] {text}")


def dm_to_text(dm: Any) -> str:
    """
    Human-readable summary of DeliveryMan state.

    Includes:
      • position, mode, speed, pace, energy, earnings
      • active/helping orders
      • scooter/car status (battery, location, etc.)
    """
    from .transport import remaining_range_m  # avoid cyclic import

    active_ids = [getattr(o, "id", None) for o in dm.active_orders]
    mode_str = "towing" if dm.towing_scooter else dm.mode.value

    lines: List[str] = [
        f"[DeliveryMan {dm.name}]",
        f"  Position : {fmt_xy_m(dm.x, dm.y)}",
        f"  Mode     : {mode_str}",
        f"  Speed    : {dm.get_current_speed_for_viewer():.1f} cm/s",
        f"  Pace     : {dm.pace_state} (×{dm._pace_scale():.2f})",
        f"  Energy   : {dm.energy_pct:.0f}%",
        f"  Earnings : ${dm.earnings_total:.2f}",
        f"  Active Orders : {active_ids}",
        f"  Helping Orders: {list(dm.help_orders.keys())}",
        f"  Carrying : {dm.carrying}",
        f"  Queue    : {len(dm._queue)} action(s), Busy: {dm.is_busy()}",
    ]

    if dm.e_scooter:
        rng_m = remaining_range_m(dm)
        rng_km = (rng_m / 1000.0) if rng_m is not None else None
        lines.append(
            "  Scooter  : "
            f"state={dm.e_scooter.state.value}, "
            f"battery={dm.e_scooter.battery_pct:.0f}% "
            f"({dm.e_scooter.charge_rate_pct_per_min:.1f}%/min), "
            f"avg_speed={dm.e_scooter.avg_speed_cm_s:.0f} cm/s, "
            f"park_xy={fmt_xy_m_opt(dm.e_scooter.park_xy)}, "
            f"remaining={'{:.1f} km'.format(rng_km) if rng_km is not None else 'N/A'}"
        )

    if dm.assist_scooter:
        s = dm.assist_scooter
        lines.append(
            "  AssistScooter : "
            f"owner={getattr(s, 'owner_id','?')}, "
            f"battery={s.battery_pct:.0f}% "
            f"({s.charge_rate_pct_per_min:.1f}%/min), "
            f"park_xy={fmt_xy_m_opt(s.park_xy)}"
        )

    if dm.car:
        lines.append(
            "  Car      : "
            f"state={dm.car.state.value}, "
            f"rate=${dm.car.rate_per_min:.2f}/min, "
            f"park_xy={fmt_xy_m_opt(dm.car.park_xy)}, "
            f"rental={'on' if dm._rental_ctx else 'off'}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Interrupt / rescue / bus utilities
# ---------------------------------------------------------------------------

def interrupt_and_stop_agent(dm: Any, reason: str, hint: Optional[str] = None) -> None:
    """
    Generic interrupt handler:
      • Mark interrupt reason
      • Stop movement in UE
      • Optionally attach an ephemeral VLM hint
      • Log & record stats
    """
    dm._interrupt_reason = str(reason)
    dm._interrupt_move_flag = True

    # Stop movement in UE
    if getattr(dm, "_ue", None) and hasattr(dm._ue, "delivery_man_stop"):
        try:
            dm._ue.delivery_man_stop(str(dm.agent_id))
        except Exception:
            pass

    # Attach ephemeral hint
    if hint and getattr(dm, "vlm_ephemeral", None) is not None:
        dm.vlm_ephemeral[str(reason)] = str(hint)

    dm._log(f"interrupt: {reason} -> stop moving & wait for decision")

    rec = getattr(dm, "_recorder", None)
    if rec:
        if reason == "escooter_depleted":
            rec.inc("scooter_depleted", 1)
        if reason == "car_rental_ended":
            rec.inc("rent_insufficient", 1)


def trigger_hospital_if_needed(dm: Any) -> None:
    """
    Trigger the hospital/rescue flow if the DeliveryMan is fully exhausted.

    Includes:
      • Mark rescue
      • Deduct hospital fee
      • Teleport to hospital
      • Clear queues and stop current action
      • Set hospital duration context
      • Record inactivity
    """
    if getattr(dm, "is_rescued", False) or getattr(dm, "_hospital_ctx", None) is not None:
        return

    dm.is_rescued = True

    # Deduct hospital rescue fee
    fee = float(getattr(dm, "hospital_rescue_fee", 0.0))
    deduct = 0.0
    if fee > 0:
        deduct = min(fee, max(0.0, float(dm.earnings_total)))
        if deduct > 0:
            dm.earnings_total -= deduct
            dm._log(f"hospital rescue fee charged: ${deduct:.2f} (balance ${dm.earnings_total:.2f})")

        rec = getattr(dm, "_recorder", None)
        if rec:
            rec.on_hospital_fee(dm.clock.now_sim(), fee=deduct)
            rec.inc("hospital_rescue", 1)

    hospital_mins = int(getattr(dm, "hospital_duration_s", 900) / 60)
    dm.vlm_add_error(
        f"You ran out of energy and were rescued to the hospital. "
        f"Hospital fee: ${fee:.2f}. Recovery time: {hospital_mins} mins. You are now at the hospital with full energy."
    )

    # Teleport to nearest hospital
    hxy = nearest_poi_xy(dm, "hospital") or (dm.x, dm.y)
    if getattr(dm, "_ue", None) and hasattr(dm._ue, "teleport_xy"):
        dm._ue.teleport_xy(str(dm.agent_id), float(hxy[0]), float(hxy[1]))
    dm.x, dm.y = float(hxy[0]), float(hxy[1])

    # Update viewer
    if getattr(dm, "_viewer", None) and getattr(dm, "_viewer_agent_id", None):
        if hasattr(dm._viewer, "set_agent_xy"):
            dm._viewer.set_agent_xy(dm._viewer_agent_id, dm.x, dm.y)

    # Clear actions
    if hasattr(dm, "_queue"):
        dm._queue.clear()
    if hasattr(dm, "_current"):
        dm._current = None

    now_sim = dm.clock.now_sim()
    duration_s = float(getattr(dm, "hospital_duration_s", 0.0))
    dm._hospital_ctx = dict(
        start_sim=now_sim,
        end_sim=now_sim + duration_s,
    )

    rec = getattr(dm, "_recorder", None)
    if rec:
        rec.tick_inactive("hospital", duration_s)

    # In manual-step (text-env) mode, resolve the hospital immediately:
    # advance the clock past the recovery duration and run poll_time_events
    # so the agent is ready for its next action within the same step.
    if getattr(dm, "manual_step", False) and duration_s > 0:
        dm.clock.advance(duration_s)
        dm.poll_time_events()


def update_bus_riding(dm: Any, now: float) -> None:
    """
    Update DeliveryMan state while riding a bus.

    Includes:
      • Follow the bus position
      • Auto-alight when reaching target stop
    """
    if not getattr(dm, "_bus_ctx", None) or not getattr(dm, "_bus_manager", None):
        return

    bus_id = dm._bus_ctx.get("bus_id")
    bus = dm._bus_manager.get_bus(bus_id)
    if not bus:
        dm._bus_ctx = None
        return

    # Sync agent position with bus
    dm.x = bus.x
    dm.y = bus.y

    # Check stop arrival
    target_stop_id = dm._bus_ctx.get("target_stop")
    if target_stop_id and bus.is_at_stop():
        current_stop = bus.get_current_stop()
        if current_stop and current_stop.id == target_stop_id:
            bus.alight_passenger(str(dm.agent_id))
            dm._log(f"arrived at target stop {target_stop_id}, auto alighting")

            dm.set_mode(dm._bus_ctx.get("transport_mode"))

            if getattr(dm, "_ue", None) and hasattr(dm._ue, "teleport_xy"):
                dm._ue.teleport_xy(str(dm.agent_id), float(dm.x), float(dm.y))

            dm._bus_ctx = None
            dm._finish_action(success=True)