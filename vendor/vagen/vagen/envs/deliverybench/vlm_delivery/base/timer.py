# base/timer.py
# -*- coding: utf-8 -*-

# Pure Python virtual time utilities, decoupled from any UI framework.
# Used to keep viewer / UE / agent logic on a shared "simulation time" axis.
# - VirtualClock: global simulation clock with adjustable time_scale.
# - TimeCursor: per-consumer cursor for convenient dt_sim queries.

import time
from dataclasses import dataclass


class VirtualClock:
    """
    Global simulation clock.

    Simulation time is defined as:
        now_sim = base_sim + (monotonic() - base_real) * time_scale

    Changing time_scale keeps now_sim continuous.
    """

    def __init__(self, time_scale: float = 1.0):
        self._time_scale = float(time_scale)
        self._base_real = time.monotonic()
        self._base_sim = 0.0  # start at 0

    @property
    def time_scale(self) -> float:
        """Current multiplier from real seconds to simulated seconds."""
        return self._time_scale

    def set_time_scale(self, scale: float):
        """
        Set a new time_scale while keeping now_sim continuous.

        scale <= 0 is clamped to a small positive value.
        """
        scale = float(scale)
        if scale <= 0:
            scale = 1e-9

        now_r = time.monotonic()
        now_s = self.now_sim()

        self._base_real = now_r
        self._base_sim = now_s
        self._time_scale = scale

    def now_sim(self) -> float:
        """Return current simulation time in seconds."""
        return self._base_sim + (time.monotonic() - self._base_real) * self._time_scale

    def advance(self, sim_seconds: float) -> float:
        """
        Advance simulation time by `sim_seconds` without sleeping.

        This is useful for text-only / instant-transition environments where
        actions (MOVE / WAIT / CHARGE / REST) should complete immediately while
        still progressing the simulation clock deterministically.

        Returns:
            The new simulation time (seconds).
        """
        dt = max(0.0, float(sim_seconds))
        now_r = time.monotonic()
        now_s = self.now_sim()
        self._base_real = now_r
        self._base_sim = now_s + dt
        return self._base_sim

    def sim_to_real_duration(self, sim_seconds: float) -> float:
        """
        Convert a duration in simulated seconds to the corresponding
        real-time duration (in seconds), given the current time_scale.
        """
        return float(sim_seconds) / max(1e-9, self._time_scale)

    def make_cursor(self) -> "TimeCursor":
        """Create a per-consumer cursor bound to this clock."""
        return TimeCursor(self)


@dataclass
class TimeCursor:
    """
    Per-consumer helper for advancing time on a VirtualClock.

    Typical usage:
        cursor = clock.make_cursor()
        ...
        dt = cursor.dt()  # simulated seconds since last call
    """
    clock: VirtualClock
    _last_sim: float = None

    def __post_init__(self):
        self._last_sim = self.clock.now_sim()

    def dt(self) -> float:
        """Return simulated seconds since the last call and advance the cursor."""
        now_s = self.clock.now_sim()
        dt_s = max(0.0, now_s - self._last_sim)
        self._last_sim = now_s
        return dt_s

    def now(self) -> float:
        """Return current simulation time in seconds, via the underlying clock."""
        return self.clock.now_sim()