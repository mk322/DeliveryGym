"""Structural extraction and hashing of authoritative simulator state.

design plan §7.2 requires text/cached/live runtimes to agree on task state, economy,
inventory, clock, events, and reward components — not on pixels. Every such
comparison needs one canonical answer to "what is this environment's state right
now", stable across processes and runtimes.

Enumerating field names by hand would rot on contact with a 31k-line vendored
engine. Instead this walks the object graph and keeps what can affect a
transition: public attributes with JSON-representable values. It drops the
things that are per-process identity rather than state — loggers, locks, RNG
objects, sockets, executors, bound methods, and anything whose repr embeds a
memory address.

Non-finite floats are preserved as tagged sentinels rather than rejected, so an
``inf`` deadline shows up as a state difference instead of crashing the digest.
"""

from __future__ import annotations

import enum
import math
from collections import deque
from dataclasses import dataclass, field, is_dataclass
from typing import Any, Callable, Iterable

from embodiedbench.artifacts.hashing import canonical_json, sha256_bytes

# Attribute names never treated as state: process identity, not simulation.
DEFAULT_ATTR_DENYLIST = frozenset(
    {
        "logger",
        "_logger",
        "vlm_prompt",
        "_vlm_client",
        "_vlm_executor",
        "_recorder",
        "_lock",
        "_rng",
        "_ue",
        "comms",
        "_comms",
        "_action_handlers",
        "_realtime_start_ts",
    }
)

# Type names dropped wherever they appear in the graph.
DEFAULT_TYPE_DENYLIST = frozenset(
    {
        "Logger",
        "LoggerAdapter",
        "RLock",
        "Lock",
        "Condition",
        "Event",
        "Thread",
        "ThreadPoolExecutor",
        "Random",
        "Generator",
        "socket",
        "Popen",
        "module",
    }
)

MAX_DEPTH = 12
MAX_SEQUENCE = 4096


@dataclass
class ExtractionPolicy:
    """What counts as state for a given digest.

    ``documented_exclusions`` maps an attribute name to the written reason it is
    not authoritative state. Everything excluded that way is reported alongside
    the digest, so a milestone report shows exactly what a hash does *not*
    cover. Widening ``attr_denylist`` silently would let a real nondeterminism
    be hidden by the same mechanism that hides an inert field.
    """

    attr_denylist: frozenset[str] = DEFAULT_ATTR_DENYLIST
    type_denylist: frozenset[str] = DEFAULT_TYPE_DENYLIST
    include_private: bool = False
    max_depth: int = MAX_DEPTH
    max_sequence: int = MAX_SEQUENCE
    # Attribute names that are always kept even when private.
    private_allowlist: frozenset[str] = frozenset()
    documented_exclusions: dict[str, str] = field(default_factory=dict)

    def excluded(self, name: str) -> bool:
        return name in self.attr_denylist or name in self.documented_exclusions

    def exclusion_report(self) -> list[dict[str, str]]:
        return [
            {"attribute": name, "reason": reason}
            for name, reason in sorted(self.documented_exclusions.items())
        ]


@dataclass
class ExtractionStats:
    """Diagnostics about one extraction, for auditing what a digest covers."""

    nodes: int = 0
    truncated_sequences: int = 0
    depth_limited: int = 0
    dropped_types: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": self.nodes,
            "truncated_sequences": self.truncated_sequences,
            "depth_limited": self.depth_limited,
            "dropped_types": dict(sorted(self.dropped_types.items())),
        }


def _encode_float(value: float) -> Any:
    if math.isnan(value):
        return {"__float__": "nan"}
    if math.isinf(value):
        return {"__float__": "inf" if value > 0 else "-inf"}
    return value


def extract_state(
    obj: Any,
    *,
    policy: ExtractionPolicy | None = None,
    stats: ExtractionStats | None = None,
) -> Any:
    """Recursively reduce ``obj`` to a JSON-representable state tree."""
    policy = policy or ExtractionPolicy()
    stats = stats if stats is not None else ExtractionStats()
    return _extract(obj, policy, stats, depth=0, seen=set())


def _extract(
    obj: Any, policy: ExtractionPolicy, stats: ExtractionStats, depth: int, seen: set[int]
) -> Any:
    stats.nodes += 1

    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return _encode_float(obj)
    if isinstance(obj, bytes):
        return {"__bytes_sha256__": sha256_bytes(obj)}
    if isinstance(obj, enum.Enum):
        return {"__enum__": f"{type(obj).__name__}.{obj.name}", "value": _extract(
            obj.value, policy, stats, depth + 1, seen
        )}

    type_name = type(obj).__name__
    if type_name in policy.type_denylist:
        stats.dropped_types[type_name] = stats.dropped_types.get(type_name, 0) + 1
        return {"__dropped__": type_name}

    if callable(obj) and not isinstance(obj, type):
        stats.dropped_types["callable"] = stats.dropped_types.get("callable", 0) + 1
        return {"__dropped__": "callable"}

    if depth >= policy.max_depth:
        stats.depth_limited += 1
        return {"__depth_limited__": type_name}

    # Cycle guard. Repeated references are normal in this object graph (orders
    # point back at the map), so a revisit is reported rather than inlined.
    ident = id(obj)
    if ident in seen:
        return {"__cycle__": type_name}
    seen = seen | {ident}

    if isinstance(obj, dict):
        items = list(obj.items())
        truncated = len(items) > policy.max_sequence
        if truncated:
            stats.truncated_sequences += 1
            items = items[: policy.max_sequence]
        # Exclusions apply to names wherever they appear, not only to object
        # attributes: configuration lives in nested dicts, so a field like
        # cfg["traffic_lights"]["visible_signal_views"] is a dict key rather than
        # an attribute and would otherwise slip past a documented exclusion.
        out = {
            str(k): _extract(v, policy, stats, depth + 1, seen)
            for k, v in sorted(items, key=lambda kv: str(kv[0]))
            if not policy.excluded(str(k))
        }
        if truncated:
            out["__truncated_at__"] = policy.max_sequence
        return out

    if isinstance(obj, (list, tuple, deque)):
        items = list(obj)
        if len(items) > policy.max_sequence:
            stats.truncated_sequences += 1
            items = items[: policy.max_sequence]
        return [_extract(v, policy, stats, depth + 1, seen) for v in items]

    if isinstance(obj, (set, frozenset)):
        # Sets have no stable iteration order; sort by canonical encoding.
        encoded = [_extract(v, policy, stats, depth + 1, seen) for v in obj]
        return sorted(encoded, key=lambda v: canonical_json(v))

    if is_dataclass(obj) and not isinstance(obj, type):
        source = {f.name: getattr(obj, f.name, None) for f in obj.__dataclass_fields__.values()}
    elif hasattr(obj, "__dict__"):
        source = vars(obj)
    elif hasattr(obj, "__slots__"):
        source = {name: getattr(obj, name, None) for name in obj.__slots__}
    else:
        # Opaque leaf (a C extension object, say). Its repr may embed an address,
        # so record only the type.
        stats.dropped_types[type_name] = stats.dropped_types.get(type_name, 0) + 1
        return {"__opaque__": type_name}

    out = {"__type__": type_name}
    for name, value in sorted(source.items()):
        if policy.excluded(name):
            continue
        if name.startswith("_") and not policy.include_private:
            if name not in policy.private_allowlist:
                continue
        out[name] = _extract(value, policy, stats, depth + 1, seen)
    return out


def state_digest(
    obj: Any,
    *,
    policy: ExtractionPolicy | None = None,
    stats: ExtractionStats | None = None,
) -> str:
    """sha256 of the canonical encoding of ``obj``'s extracted state."""
    return sha256_bytes(canonical_json(extract_state(obj, policy=policy, stats=stats)))


def digest_of(value: Any) -> str:
    """sha256 of an already-JSON-safe value."""
    return sha256_bytes(canonical_json(value))


def diff_state(left: Any, right: Any, *, path: str = "", limit: int = 40) -> list[str]:
    """Human-readable list of the first ``limit`` differences between two trees.

    Used to explain a failed conformance assertion instead of printing two
    64-character hashes that differ.
    """
    out: list[str] = []

    def walk(a: Any, b: Any, p: str) -> None:
        if len(out) >= limit:
            return
        if type(a) is not type(b):
            out.append(f"{p or '<root>'}: type {type(a).__name__} != {type(b).__name__}")
            return
        if isinstance(a, dict):
            for key in sorted(set(a) | set(b)):
                if key not in a:
                    out.append(f"{p}.{key}: missing on left (right={b[key]!r})")
                elif key not in b:
                    out.append(f"{p}.{key}: missing on right (left={a[key]!r})")
                else:
                    walk(a[key], b[key], f"{p}.{key}")
                if len(out) >= limit:
                    return
        elif isinstance(a, list):
            if len(a) != len(b):
                out.append(f"{p}: length {len(a)} != {len(b)}")
            for i, (x, y) in enumerate(zip(a, b)):
                walk(x, y, f"{p}[{i}]")
                if len(out) >= limit:
                    return
        elif a != b:
            out.append(f"{p or '<root>'}: {a!r} != {b!r}")

    walk(left, right, path)
    return out
