# utils/hazards.py
# -*- coding: utf-8 -*-

"""
Pluggable static hazards for DeliveryBench: road obstacles + traffic lights.

These are **default-off** safety / social-navigation mechanics (see
``OBSTACLE_TRAFFIC_DESIGN.md``). This module is the **single runtime source of
truth** for both mechanics — the env attaches an ``ObstacleField`` /
``TrafficController`` onto the DeliveryMan at reset and the MOVE / PASSBY / WAIT
handlers consult them. There is no second runtime traffic-light path; the
world-JSON helpers in ``traffic_lights.py`` and the node generator in
``utils/pedestrian_lights.py`` are **data-generation only** (they build the 3D
renders), they do not decide red/green at runtime.

Alignment with the sampled FPV images
--------------------------------------
The traffic-light controller is built **from the FPV manifest** so the runtime
signalised set, the crossing axis of each captured face, and the red/green
choice all match the images that were actually rendered
(``deliverybench_fpv/<map>/manifest.jsonl`` rows with
``render_kind == "traffic_light"``). A hand-authored ``traffic_lights.json``
sidecar is still accepted as a fallback / override.

Positions are keyed by ``(round(x_cm, 1), round(y_cm, 1))`` — the same stable
position join the FPV lookup uses, since capture ``waypoint_id``s have drifted
from the runtime graph.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

# Obstacle types that behave identically for v1 (the type only selects the FPV
# sprite + a future penalty weight). "others" is reserved / not yet captured.
OBSTACLE_TYPES = ("slow_pedestrian", "road_block")

# Canonical crossing axes. The runtime uses NS / EW everywhere; the FPV manifest
# and the 2D crossing renderer use the verbose ``south-north`` / ``east-west``
# spelling. ``normalize_axis`` bridges the two so there is one convention.
AXIS_NS = "NS"
AXIS_EW = "EW"


def _pkey(x_cm: float, y_cm: float) -> Tuple[float, float]:
    return (round(float(x_cm), 1), round(float(y_cm), 1))


def normalize_axis(axis: Any) -> str:
    """Map any spelling of a crossing axis to ``"NS"`` or ``"EW"``.

    Accepts ``south-north`` / ``north-south`` / ``vertical`` / ``sn`` / ``ns``
    (→ NS) and ``east-west`` / ``west-east`` / ``horizontal`` / ``ew``
    (→ EW). Unknown values fall back to NS.
    """
    a = str(axis or "").strip().lower().replace("_", "-")
    if a in {"ns", "sn", "south-north", "north-south", "vertical", "v"}:
        return AXIS_NS
    if a in {"ew", "we", "east-west", "west-east", "horizontal", "h"}:
        return AXIS_EW
    return AXIS_NS


class ObstacleField:
    """Directed obstacle lookup: an obstacle sits on the edge ``src -> dst`` and
    blocks a forward MOVE onto that edge (the agent must BYPASS instead)."""

    def __init__(self, obstacles: List[Dict[str, Any]]):
        self._by_edge: Dict[Tuple[Tuple[float, float], Tuple[float, float]], str] = {}
        for o in obstacles or []:
            try:
                sk = _pkey(o["src_x_cm"], o["src_y_cm"])
                dk = _pkey(o["dst_x_cm"], o["dst_y_cm"])
            except (KeyError, TypeError):
                continue
            self._by_edge[(sk, dk)] = str(o.get("type", "road_block"))

    def __len__(self) -> int:
        return len(self._by_edge)

    def obstacle_on(self, src_x: float, src_y: float, dst_x: float, dst_y: float) -> Optional[str]:
        """Return the obstacle type on edge ``src -> dst`` (directed), or None."""
        return self._by_edge.get((_pkey(src_x, src_y), _pkey(dst_x, dst_y)))

    @classmethod
    def load(cls, path: Path) -> "ObstacleField":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data.get("obstacles", []))


class TrafficController:
    """Global minute-parity traffic signals at signalised intersections.

    odd minute  -> ``odd_minute_red_axis`` is RED, the other axis GREEN
    even minute -> flipped

    The relevant light is chosen by the agent's travel *axis* (NS vs EW). On the
    cardinal-grid maps the axis classification is exact, so at any intersection
    the two perpendicular crossing directions always show opposite colours (one
    red, one green) — exactly the "front is red, right is green" case the FPV
    must reflect.

    The signalised set is normally built ``from_manifest`` so it matches the
    rendered FPV faces; a hand-authored sidecar (``load``) is also accepted.
    """

    def __init__(
        self,
        signalised: List[Dict[str, Any]],
        odd_minute_red_axis: str = "NS",
        minute_period_s: float = 60.0,
    ):
        self._sig: set = set()
        # pkey -> {yaw_deg: axis} for the captured faces (informational; eval
        # uses the travel bearing's axis, which is exact on the grid maps).
        self._faces_by_pos: Dict[Tuple[float, float], Dict[float, str]] = {}
        for s in signalised or []:
            try:
                pk = _pkey(s["x_cm"], s["y_cm"])
            except (KeyError, TypeError):
                continue
            self._sig.add(pk)
            yaw = s.get("yaw")
            axis = s.get("axis") or s.get("signal_axis")
            if yaw is not None and axis is not None:
                self._faces_by_pos.setdefault(pk, {})[round(float(yaw), 1)] = normalize_axis(axis)
        self._odd_red = normalize_axis(odd_minute_red_axis)
        self._period = float(minute_period_s) or 60.0

    def __len__(self) -> int:
        return len(self._sig)

    def is_signalised(self, x_cm: float, y_cm: float) -> bool:
        return _pkey(x_cm, y_cm) in self._sig

    def face_axis(self, x_cm: float, y_cm: float, yaw_deg: float) -> Optional[str]:
        """The captured crossing axis for a (position, yaw) face, if rendered."""
        return self._faces_by_pos.get(_pkey(x_cm, y_cm), {}).get(round(float(yaw_deg), 1))

    @staticmethod
    def axis_of(bearing_deg: float) -> str:
        """Classify a compass bearing into the NS or EW crossing axis."""
        b = float(bearing_deg) % 180.0
        return AXIS_NS if (b < 45.0 or b >= 135.0) else AXIS_EW

    def _red_axis_at(self, sim_time_s: float) -> str:
        minute = int(float(sim_time_s) // self._period)
        if minute % 2 == 1:                       # odd minute
            return self._odd_red
        return AXIS_EW if self._odd_red == AXIS_NS else AXIS_NS

    def light(self, axis: str, sim_time_s: float) -> str:
        """'red' or 'green' for the given travel axis at the given sim time."""
        return "red" if normalize_axis(axis) == self._red_axis_at(sim_time_s) else "green"

    def light_for_bearing(self, bearing_deg: float, sim_time_s: float) -> str:
        return self.light(self.axis_of(bearing_deg), sim_time_s)

    def seconds_to_next_minute(self, sim_time_s: float) -> float:
        """Seconds to advance to reach the start of the next wall-minute.

        Always strictly positive — landing exactly on a boundary advances a full
        period (so WAIT('traffic_light') is deterministic and never a no-op)."""
        rem = float(sim_time_s) % self._period
        return self._period - rem if rem > 1e-9 else self._period

    # -- constructors ---------------------------------------------------------
    @classmethod
    def from_manifest(
        cls,
        rows: Iterable[Mapping[str, Any]],
        odd_minute_red_axis: str = "NS",
        minute_period_s: float = 60.0,
    ) -> "TrafficController":
        """Build the signalised set from FPV manifest rows.

        Any row with ``render_kind == "traffic_light"`` (or a ``signal_state``
        field) marks its ``(x_cm, y_cm)`` as signalised and records the captured
        face axis per yaw. This is what aligns the runtime to the images.
        """
        signalised: List[Dict[str, Any]] = []
        for r in rows or []:
            if not isinstance(r, Mapping):
                continue
            is_light = (
                str(r.get("render_kind") or "").lower() == "traffic_light"
                or r.get("signal_state") is not None
                or (str(r.get("variant") or "").lower() == "light")
            )
            if not is_light:
                continue
            try:
                signalised.append(
                    {
                        "x_cm": float(r["x_cm"]),
                        "y_cm": float(r["y_cm"]),
                        "yaw": r.get("yaw"),
                        "axis": r.get("signal_axis") or r.get("axis"),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
        return cls(
            signalised,
            odd_minute_red_axis=odd_minute_red_axis,
            minute_period_s=minute_period_s,
        )

    @classmethod
    def load_manifest(
        cls,
        path: Path,
        odd_minute_red_axis: str = "NS",
        minute_period_s: float = 60.0,
    ) -> "TrafficController":
        rows: List[Mapping[str, Any]] = []
        with Path(path).open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return cls.from_manifest(
            rows,
            odd_minute_red_axis=odd_minute_red_axis,
            minute_period_s=minute_period_s,
        )

    @classmethod
    def load(cls, path: Path) -> "TrafficController":
        """Load a hand-authored ``traffic_lights.json`` sidecar (override)."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            data.get("signalised_intersections", []),
            odd_minute_red_axis=data.get("odd_minute_red_axis", "NS"),
            minute_period_s=data.get("minute_period_s", 60.0),
        )
