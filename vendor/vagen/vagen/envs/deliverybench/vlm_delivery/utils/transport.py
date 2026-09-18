# utils/transport.py
# -*- coding: utf-8 -*-

"""
Transport- and energy-related utilities for DeliveryMan agents.

This module implements helper functions that mirror the agent's
internal transport behavior, including:

- Determining towing state
- Calculating effective travel speed
- Changing transport modes with validation
- Applying personal energy consumption
- Applying vehicle (e-scooter) battery consumption
- Unified distance-based consumption handler

All functions operate on a generic `agent` object with the expected fields.
"""

from typing import Any, Optional

from ..base.defs import *
from ..entities.escooter import ScooterState


# ----------------------------------------------------------------------
# Towing state utilities
# ----------------------------------------------------------------------

def transport_recalc_towing(agent: Any) -> None:
    """
    Recompute whether the agent is currently towing an e-scooter.

    Towing is true when:
    - The agent is explicitly in DRAG_SCOOTER mode, or
    - The agent owns an e-scooter that is depleted, still associated
      with the owner, and not parked elsewhere.
    """
    if agent.mode == TransportMode.DRAG_SCOOTER:
        agent.towing_scooter = True
    elif (
        agent.e_scooter
        and getattr(agent.e_scooter, "with_owner", True)
        and agent.e_scooter.state == ScooterState.DEPLETED
        and not agent.e_scooter.park_xy
    ):
        agent.towing_scooter = True
    else:
        agent.towing_scooter = False


def transport_get_current_speed_for_viewer(agent: Any) -> float:
    """
    Compute the current movement speed used by the viewer UI.

    This includes:
    - Refreshing towing state
    - Applying agent speed, pace scaling, and global time scaling
    """
    transport_recalc_towing(agent)
    ts = float(getattr(agent.clock, "time_scale", 1.0) or 1.0)
    return float(agent.speed_cm_s) * agent._pace_scale() * ts


# ----------------------------------------------------------------------
# Transport mode management
# ----------------------------------------------------------------------

def transport_set_mode(
    agent: Any,
    mode: TransportMode,
    *,
    override_speed_cm_s: Optional[float] = None,
) -> None:
    """
    Set the agent's transport mode.

    This includes:
    - Validation of mode transitions
    - Automatic fallback from SCOOTER → DRAG_SCOOTER when riding is not allowed
    - Checking e-scooter ownership, battery, and proximity constraints
    - Selecting appropriate base speed from vehicle or on-foot tables
    """
    mode = TransportMode(mode)

    # Assisted scooter → cannot ride, only drag
    if mode == TransportMode.SCOOTER and agent.assist_scooter is not None:
        mode = TransportMode.DRAG_SCOOTER

    # Riding validation
    if mode == TransportMode.SCOOTER:
        if not agent.e_scooter:
            mode = TransportMode.DRAG_SCOOTER
        else:
            owner_ok = (getattr(agent.e_scooter, "owner_id", None) == str(agent.agent_id))
            usable   = agent.e_scooter.state not in (ScooterState.DEPLETED, ScooterState.PARKED)
            with_me  = bool(getattr(agent.e_scooter, "with_owner", True))
            if not (owner_ok and usable and with_me):
                mode = TransportMode.DRAG_SCOOTER

    agent.mode = mode
    agent.pace_state = "normal"

    # Determine base speed according to mode
    if agent.mode == TransportMode.SCOOTER and agent.e_scooter:
        base = float(agent.e_scooter.avg_speed_cm_s)
        if override_speed_cm_s is not None:
            base = agent.e_scooter.clamp_speed(float(override_speed_cm_s))
            agent.e_scooter.avg_speed_cm_s = base
        agent.speed_cm_s = base

    elif agent.mode == TransportMode.CAR and agent.car:
        base = float(agent.car.avg_speed_cm_s)
        if override_speed_cm_s is not None:
            base = float(override_speed_cm_s)
        agent.speed_cm_s = base

    else:
        base = float(agent.avg_speed_by_mode.get(agent.mode))
        if override_speed_cm_s is not None:
            base = float(override_speed_cm_s)
        agent.speed_cm_s = base

    transport_recalc_towing(agent)


# ----------------------------------------------------------------------
# Personal energy consumption
# ----------------------------------------------------------------------

def _consume_personal_energy_by_distance(agent: Any, distance_m: float) -> None:
    """
    Deduct the agent's personal energy based on traveled distance.

    Applies to all transport modes. Energy cost is computed per meter,
    scaled by the current pace. Consumption is tracked in the recorder
    if available.
    """
    cost_per_m = float(agent.energy_cost_by_mode.get(agent.mode, 0.0)) * agent._pace_scale()
    if cost_per_m <= 0.0:
        return

    before = float(agent.energy_pct)
    delta = cost_per_m * max(0.0, float(distance_m))
    after = max(0.0, before - delta)
    agent.energy_pct = after

    consumed = max(0.0, before - after)
    if consumed > 0.0 and getattr(agent, "_recorder", None):
        agent._recorder.energy_personal_consumed_pct += float(consumed)

    if agent.energy_pct <= 0.0:
        agent._trigger_hospital_if_needed()


# ----------------------------------------------------------------------
# Vehicle (e-scooter) energy consumption
# ----------------------------------------------------------------------

def _consume_vehicle_by_distance(agent: Any, distance_m: float) -> None:
    """
    Deduct e-scooter battery based on traveled distance.

    Only applies when:
    - The agent is in SCOOTER mode,
    - The e-scooter is present, associated with the owner,
    - And is traveling with the agent.
    """
    if (
        agent.mode == TransportMode.SCOOTER
        and agent.e_scooter
        and getattr(agent.e_scooter, "with_owner", True)
    ):
        pace = agent._pace_scale()
        delta_pct_req = max(0.0, float(distance_m)) * agent.scooter_batt_decay_pct_per_m * pace

        before = float(agent.e_scooter.battery_pct)
        if delta_pct_req <= 0.0:
            return
        if before <= 0.0:
            agent.e_scooter.state = ScooterState.DEPLETED
            transport_set_mode(agent, TransportMode.WALK)
            return

        agent.e_scooter.consume_pct(delta_pct_req)
        after = float(agent.e_scooter.battery_pct)
        consumed = max(0.0, before - after)

        if consumed > 0.0 and getattr(agent, "_recorder", None):
            agent._recorder.scooter_batt_consumed_pct += float(consumed)

        # Trigger fallback behavior when battery becomes depleted
        if after <= 0.0 and before > 0.0:
            agent.e_scooter.state = ScooterState.DEPLETED
            agent._interrupt_and_stop(
                "escooter_depleted",
                (
                    "Your e-scooter battery is depleted. You are now walking "
                    "and your scooter is following you. Walk to a "
                    "charging_station and CHARGE(target_pct=80) to ride "
                    "again, or SWITCH(to='walk') to leave the scooter behind."
                ),
            )
            transport_set_mode(agent, TransportMode.WALK)

    # Car and bus currently do not simulate fuel/energy usage.


# ----------------------------------------------------------------------
# Unified distance-based consumption
# ----------------------------------------------------------------------

def transport_consume_by_distance(agent: Any, distance_cm: float) -> None:
    """
    Apply both personal and vehicle energy consumption based on traveled distance.

    Converts centimeters → meters, refreshes towing state, then applies:
    - Personal energy deduction
    - Vehicle energy deduction (e-scooter only)
    """
    distance_m = max(0.0, float(distance_cm) / 100.0)
    if distance_m <= 0.0:
        return

    transport_recalc_towing(agent)

    cfg = getattr(agent, "cfg", {}) or {}
    if cfg.get("enable_walking_energy", True):
        _consume_personal_energy_by_distance(agent, distance_m)
    if cfg.get("enable_battery", True):
        _consume_vehicle_by_distance(agent, distance_m)


def transport_on_move_consumed(agent: Any, distance_cm: float) -> None:
    """
    Entry point used by movement updates.

    Applies energy consumption whenever the agent moves a positive distance.
    """
    if distance_cm <= 0.0:
        return
    transport_consume_by_distance(agent, distance_cm)