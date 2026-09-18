# map/map_exportor.py
# -*- coding: utf-8 -*-

from typing import Any, Dict, List, Optional, Tuple

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

from .map_debug_viewer import MapDebugViewer


# ------------------ Order metadata conversion ------------------
class _Pos:
    def __init__(self, x: float, y: float):
        self.x = float(x)
        self.y = float(y)


class _NodeStub:
    def __init__(self, x: float, y: float):
        self.position = _Pos(x, y)
        self.type = "normal"


def _xy_from(obj: Any):
    """
    Try to extract (x, y) coordinates from a few common object shapes:
    * node-like: obj.position.x / obj.position.y
    * address-like: obj.x / obj.y
    * dict-like: obj["x"] / obj["y"]
    """
    if obj is None:
        return None
    try:
        # node-like
        return float(obj.position.x), float(obj.position.y)
    except Exception:
        pass
    try:
        # address-like
        return float(obj.x), float(obj.y)
    except Exception:
        pass
    try:
        # dict-like
        return float(obj["x"]), float(obj["y"])
    except Exception:
        return None


def _order_meta_from(orders: List[Any]) -> List[Dict[str, Any]]:
    """
    Build a lightweight list of order metadata records that MapDebugViewer
    understands. Each record may contain:
        - "id": order id
        - "pickup_node": node-like object with .position.{x,y}
        - "dropoff_node": node-like object with .position.{x,y}

    When pickup/dropoff nodes are missing, fall back to address fields or
    plain coordinate dicts.
    """
    metas: List[Dict[str, Any]] = []
    for o in (orders or []):
        try:
            oid = getattr(o, "id", None)
            pu_node = getattr(o, "pickup_node", None)
            do_node = getattr(o, "dropoff_node", None)

            # Fallback: use pickup_address / delivery_address if node is missing
            if pu_node is None:
                pu_xy = _xy_from(getattr(o, "pickup_address", None))
                if pu_xy:
                    pu_node = _NodeStub(*pu_xy)
            if do_node is None:
                do_xy = _xy_from(getattr(o, "delivery_address", None))
                if do_xy:
                    do_node = _NodeStub(*do_xy)

            # Fallback: plain dict orders with pickup_xy / dropoff_xy
            if pu_node is None and isinstance(o, dict) and "pickup_xy" in o:
                x, y = o["pickup_xy"]
                pu_node = _NodeStub(x, y)
            if do_node is None and isinstance(o, dict) and "dropoff_xy" in o:
                x, y = o["dropoff_xy"]
                do_node = _NodeStub(x, y)

            meta: Dict[str, Any] = {"id": oid}
            if pu_node is not None:
                meta["pickup_node"] = pu_node
            if do_node is not None:
                meta["dropoff_node"] = do_node
            metas.append(meta)
        except Exception:
            # Ignore malformed orders instead of failing the whole batch
            continue
    return metas


# ------------------ Single-threaded off-screen exporter ------------------
class MapExportor:
    """
    Single-threaded, off-screen exporter that renders two PNGs (global / local).

    Typical usage:
        1) exportor = MapExportor(map_obj=city_map,
                                  world_json_path=...,
                                  show_road_names=False)
        2) exportor.prepare_base()   # called once to build static bases
        3) Each decision step:
           g_bytes, l_bytes = exportor.export(agent_xy=(x, y), orders=orders)

    Notes:
        * Completely decoupled from any visible UI: internally creates a
          hidden MapDebugViewer that never shows on screen.
        * Static base layers are drawn only once in prepare_base().
          export() updates dynamic layers (agent / reachable set / order markers)
          and performs the export.
        * Visual appearance matches MapDebugViewer (line width / font size /
          antialiasing / axes / road names, etc.).
    """
    def __init__(
        self,
        *,
        map_obj: Any,
        world_json_path: Optional[str] = None,
        show_road_names: bool = False,
        viewer: Optional[MapDebugViewer] = None,
    ):
        app = QApplication.instance()
        if app is None:
            raise RuntimeError(
                "QApplication not initialized. Create it in main() first and run its event loop."
            )

        self.map = map_obj
        self.world_json_path = world_json_path
        self.show_road_names_default = bool(show_road_names)

        if viewer is None:
            # Truly off-screen: the internal viewer never appears on screen
            self.viewer = MapDebugViewer(title="Exporter (headless)")
            self.viewer.setAttribute(Qt.WA_DontShowOnScreen, True)
            self.viewer.hide()
        else:
            self.viewer = viewer

        self._base_ready = False

    # ---- Called once to build static base layers ----
    def prepare_base(self):
        """
        Build the static export base (roads / nodes / POIs / bus routes / buildings)
        in the internal MapDebugViewer. This is usually called once.
        """
        self.viewer.prepare_export_base(map_obj=self.map, world_json_path=self.world_json_path)
        self._base_ready = True

    # ---- Per-decision call: output two PNGs (bytes) ----
    def export(self, *, agent_xy: Tuple[float, float], orders: List[Any]) -> Tuple[bytes, bytes]:
        """
        Export global and local PNGs for the given agent state and orders.

        Args:
            agent_xy: Current agent position (x, y) in centimeters.
            orders:   List of order objects or dicts used to derive PU/DO metadata.

        Returns:
            (global_bytes, local_bytes): PNG data for global and local views.
        """
        if not self._base_ready:
            raise RuntimeError(
                "MapExportor.export: export base not prepared. Call prepare_base() first."
            )

        ax, ay = float(agent_xy[0]), float(agent_xy[1])

        # Temporarily attach order_meta to the map (used by dynamic layers).
        # The original value is restored after export to avoid side effects.
        had_attr = hasattr(self.map, "order_meta")
        prev_val = getattr(self.map, "order_meta", None)
        setattr(self.map, "order_meta", _order_meta_from(orders))

        try:
            # Keep behavior consistent with existing reachable-set logic
            get_reach = getattr(self.map, "get_reachable_set_xy", None)
            reachable = (
                get_reach(ax, ay)
                if callable(get_reach)
                else {"next_hop": [], "next_intersections": []}
            )

            # Update agent coordinates and export two images.
            # Only dynamic layers are updated; static bases are reused.
            self.viewer.set_agent_xy(ax, ay)
            return self.viewer.generate_images(
                reachable,
                show_road_names=self.show_road_names_default,
                agent_xy=(ax, ay),
            )

        finally:
            # Restore map.order_meta to avoid leaking temporary state
            if had_attr:
                setattr(self.map, "order_meta", prev_val)
            else:
                try:
                    delattr(self.map, "order_meta")
                except Exception:
                    pass