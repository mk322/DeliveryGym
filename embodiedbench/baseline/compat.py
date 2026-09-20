"""Compatibility patches for maps the vendored engine was not exercised on.

The vendored DeliveryBench engine was developed against procgen maps, all of
which ship a full complement of sidecar files and a populated ``bus_routes``
list. CityCore Paris does not, and design plan §3.2 already records why:
``progen_world_enriched.json`` for Paris "has 568 nodes and an empty
``bus_routes`` list".

Where the engine reacts to a missing feature by crashing rather than by having
none of that feature, that is a defect, and design plan §3.3 lists exactly this class
of Paris gap as work to be done. Following ADR-0003 these are fixed as recorded
runtime patches in our layer; ``vendor/`` is never edited, and each patch
carries the sha256 of the source it replaced so a vendor bump surfaces as a
decision instead of a silent revert.

These are deliberately *not* in ``determinism.py``. A determinism patch changes
which of several correct answers you get; a compatibility patch is the
difference between running and raising. Keeping them separate means a report can
say which kind was active.
"""

from __future__ import annotations

import inspect
from typing import Any

from embodiedbench.artifacts.hashing import sha256_bytes
from embodiedbench.baseline.determinism import PatchRecord, PatchSet

EMPTY_BUS_ROUTES_REASON = (
    "vlm_delivery/entities/bus_manager.py init_bus_system() ends with "
    "`self.create_bus(f'bus_{i+1}', list(self.routes.keys())[0])`, indexing the "
    "first route without checking that any route exists. CityCore Paris ships an "
    "empty `bus_routes` list (design plan §3.2), so every Paris reset raises "
    "IndexError before the environment can be used at all. A map with no bus "
    "routes should have no buses, not fail to load. The patch skips bus creation "
    "when there are no routes and is otherwise identical; on any map that does "
    "define routes the behaviour is unchanged. Tracked as PARIS-F1."
)


def _patched_init_bus_system(self, world_data: dict[str, Any]) -> None:
    """``init_bus_system`` that tolerates a map with no bus routes."""
    self._birth_sim_time = float(self.clock.now_sim())
    self.load_routes_from_world_data(world_data)
    route_ids = list(self.routes.keys())
    if not route_ids:
        # No routes on this map: no buses. The bus manager stays valid and
        # empty, so NAVIGATE(mode="bus") reports no service rather than
        # dereferencing a bus that was never created.
        return
    for index in range(self.num_buses):
        self.create_bus(f"bus_{index + 1}", route_ids[0])


def bootstrap_vendor_import() -> dict[str, Any]:
    """Make the vendored engine importable from any working directory.

    ``vlm_delivery/utils/global_logger.py`` ends with a module-level
    ``_global_logger = GlobalLogger()``, whose constructor calls
    ``os.makedirs("../../log")``. The path is relative to the process cwd, so
    merely *importing* the engine fails with PermissionError unless the cwd's
    grandparent happens to be writable. Running the compiler from a repository
    root is enough to break it.

    Because the failure happens during import, it cannot be fixed by patching
    the function afterwards -- there is no afterwards. Instead the import is
    performed from a scratch directory chosen so that ``../../log`` lands
    somewhere writable, and the previous cwd is restored immediately. The chdir
    is scoped to the import and nothing else.
    """
    import os
    import tempfile

    # Per-user, because a fixed /tmp name on a shared machine belongs to
    # whoever imported first: sixteen tests failed here with EACCES on a
    # directory another user owned.
    scratch_root = os.environ.get("EB_SCRATCH_DIR") or os.path.join(
        tempfile.gettempdir(),
        f"embodiedbench-vendor-{os.environ.get('USER', os.getuid())}"
    )
    # <root>/log is what "../../log" resolves to from <root>/a/b.
    workdir = os.path.join(scratch_root, "a", "b")
    os.makedirs(workdir, exist_ok=True)
    os.makedirs(os.path.join(scratch_root, "log"), exist_ok=True)

    previous = os.getcwd()
    os.chdir(workdir)
    try:
        from vagen.envs.deliverybench.vlm_delivery.utils import global_logger  # noqa: F401
    finally:
        os.chdir(previous)
    return {"scratch_root": scratch_root, "import_cwd": workdir, "restored_cwd": previous}


LOG_FOLDER_REASON = (
    "vlm_delivery/utils/global_logger.py::_setup_logger defaults log_folder to the "
    "relative path '../../log' and calls os.makedirs on it, so the engine can only "
    "run from a directory whose grandparent is writable. Running the compiler from "
    "a repository root raises PermissionError on '/home/log' before any map is "
    "loaded, which makes the whole map->env pipeline depend on the caller's cwd. "
    "The patch resolves the log directory to EB_LOG_DIR when set, and otherwise "
    "falls back to a temporary directory when the requested path is not writable. "
    "Logging is a side effect; it must never decide whether a map compiles. "
    "Tracked as STRESS-F1."
)


def _patched_setup_logger(self, log_folder: str = "../../log", *args: Any, **kwargs: Any):
    """``_setup_logger`` that never fails because of an unwritable log path."""
    import os
    import tempfile

    override = os.environ.get("EB_LOG_DIR")
    candidates = [override, log_folder, os.path.join(tempfile.gettempdir(), "embodiedbench-logs")]
    chosen = None
    for candidate in candidates:
        if not candidate:
            continue
        try:
            os.makedirs(candidate, exist_ok=True)
            if os.access(candidate, os.W_OK):
                chosen = candidate
                break
        except OSError:
            continue
    if chosen is None:
        chosen = tempfile.mkdtemp(prefix="embodiedbench-logs-")
    return _ORIGINAL_SETUP_LOGGER(self, chosen, *args, **kwargs)


_ORIGINAL_SETUP_LOGGER: Any = None


def apply_map_compatibility_patches() -> PatchSet:
    """Apply every map-compatibility patch. Idempotent."""
    global _ORIGINAL_SETUP_LOGGER
    from embodiedbench.baseline.replay import _ensure_vendor_on_path

    _ensure_vendor_on_path()
    # Must happen before any vendored import: the logger module cannot be
    # imported at all from an unsuitable cwd (see bootstrap_vendor_import).
    bootstrap_vendor_import()
    from vagen.envs.deliverybench.vlm_delivery.entities import bus_manager as bus_module
    from vagen.envs.deliverybench.vlm_delivery.utils import global_logger as logger_module

    if getattr(bus_module, "_embodiedbench_compat_applied", False):
        return getattr(bus_module, "_embodiedbench_compat_patchset")

    patches = PatchSet()

    logger_original = logger_module.GlobalLogger._setup_logger
    logger_source = inspect.getsource(logger_original)
    logger_record = PatchRecord(
        target="vlm_delivery.utils.global_logger.GlobalLogger._setup_logger",
        reason=LOG_FOLDER_REASON,
        original_source_sha256=sha256_bytes(logger_source.encode()),
        applied=False,
    )
    if "../../log" not in logger_source:
        logger_record.skipped_reason = (
            "vendored _setup_logger no longer hardcodes a relative '../../log' path; "
            "re-review before applying"
        )
    else:
        _ORIGINAL_SETUP_LOGGER = logger_original
        logger_module.GlobalLogger._setup_logger = _patched_setup_logger
        logger_record.applied = True
    patches.records.append(logger_record)
    original = bus_module.BusManager.init_bus_system
    source = inspect.getsource(original)
    record = PatchRecord(
        target="vlm_delivery.entities.bus_manager.BusManager.init_bus_system",
        reason=EMPTY_BUS_ROUTES_REASON,
        original_source_sha256=sha256_bytes(source.encode()),
        applied=False,
    )
    if "list(self.routes.keys())[0]" not in source:
        record.skipped_reason = (
            "vendored init_bus_system no longer indexes the first route unguarded; "
            "re-review this patch against the new implementation before applying"
        )
    else:
        bus_module.BusManager.init_bus_system = _patched_init_bus_system
        record.applied = True
    patches.records.append(record)

    bus_module._embodiedbench_compat_applied = True
    bus_module._embodiedbench_compat_patchset = patches
    return patches


MISSING_MOVEMENT_HELPERS_REASON = (
    "vlm_delivery/utils/traffic_lights.py calls movement_direction() and "
    "movement_axis() from _select_edge_light and edge_signal_check, and defines "
    "neither -- the module imports only `typing`. Every call raises NameError, so "
    "the whole runtime half of the signal system is dead code in this checkout. "
    "The functions are reconstructed here from two things the file itself fixes: "
    "its own _DIR_DEG table {north:0, east:90, south:180, west:270}, and "
    "signal_state_for_axis, whose axis spellings ('south-north', 'north-south', "
    "'vertical', 'sn', 'ns') say exactly what movement_axis must return. Tracked "
    "as ROSE-F1; if the upstream implementations arrive, "
    "test_movement_helpers_match_the_axis_convention catches any disagreement."
)


def _compass_bearing(from_node, to_node) -> float:
    """Compass bearing a->b: 0 = north (+Y), increasing clockwise.

    Matches map.py's _bearing_deg, which is atan2(dx, dy) rather than the
    mathematical atan2(dy, dx). The two differ by a reflection *and* a quarter
    turn, so mixing them silently mirrors every heading.
    """
    import math

    dx = float(to_node.position.x) - float(from_node.position.x)
    dy = float(to_node.position.y) - float(from_node.position.y)
    return math.degrees(math.atan2(dx, dy)) % 360.0


def movement_direction(from_node, to_node) -> str:
    """Cardinal direction of travel, in the vendored compass convention."""
    bearing = _compass_bearing(from_node, to_node)
    return ("north", "east", "south", "west")[int((bearing + 45.0) % 360.0 // 90.0)]


def movement_axis(from_node, to_node) -> str:
    """Crossing axis of travel: what signal_state_for_axis expects."""
    return "north-south" if movement_direction(from_node, to_node) in ("north", "south") else "east-west"


def apply_traffic_light_patches() -> PatchSet:
    """Install the two helpers rose's traffic_lights.py calls but never defines."""
    from vagen.envs.deliverybench.vlm_delivery.utils import traffic_lights as module

    patches = PatchSet()
    if not hasattr(module, "math"):
        # The module's only import is `typing`, yet _dist_point_to_segment calls
        # math.hypot. Three symbols the file uses and never defines -- math,
        # movement_direction, movement_axis -- says this copy was truncated
        # rather than merely buggy, so it is repaired here and the repair is
        # recorded instead of vendor being edited.
        import math as _math

        module.math = _math
        patches.records.append(PatchRecord(
            target="vlm_delivery.utils.traffic_lights.math",
            reason=(
                "traffic_lights.py calls math.hypot in _dist_point_to_segment but "
                "imports only `typing`, so every edge_signal_check raised NameError. "
                "Tracked as ROSE-F1."
            ),
            original_source_sha256=sha256_bytes(
                inspect.getsource(module._dist_point_to_segment).encode()
            ),
            applied=True,
        ))
    if not hasattr(module, "movement_direction"):
        module.movement_direction = movement_direction
        module.movement_axis = movement_axis
        patches.records.append(PatchRecord(
            target="vlm_delivery.utils.traffic_lights.movement_direction",
            reason=MISSING_MOVEMENT_HELPERS_REASON,
            original_source_sha256=sha256_bytes(
                inspect.getsource(module.edge_signal_check).encode()
            ),
            applied=True,
        ))
    return patches
