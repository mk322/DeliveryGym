"""Contract tests for authoritative-state extraction and digesting.

This is the mechanism every conformance claim rests on (design plan §7.2, M5, M10),
so its failure modes matter more than its happy path: it must not silently drop
real state, must not be fooled by per-process identity, and must explain a
mismatch rather than only report two different hashes.
"""

from __future__ import annotations

import enum
import threading
from collections import deque
from dataclasses import dataclass

import pytest

from embodiedbench.artifacts.state_digest import (
    ExtractionPolicy,
    ExtractionStats,
    diff_state,
    digest_of,
    extract_state,
    state_digest,
)


class Mode(enum.Enum):
    WALK = "walk"
    SCOOTER = "scooter"


@dataclass
class Order:
    order_id: str
    earnings: float
    picked_up: bool = False


class Courier:
    def __init__(self):
        self.x = 1.5
        self.y = -2.25
        self.mode = Mode.SCOOTER
        self.orders = [Order("o1", 12.5)]
        self.inventory = {"bag": ["burger"]}
        self._private = "hidden"
        self.logger = threading.Lock()
        self.history = deque([1, 2, 3])


def test_equivalent_objects_digest_equally():
    assert state_digest(Courier()) == state_digest(Courier())


def test_mutation_changes_digest():
    a, b = Courier(), Courier()
    b.orders[0].picked_up = True
    assert state_digest(a) != state_digest(b)


def test_float_change_changes_digest():
    a, b = Courier(), Courier()
    b.x = 1.5000001
    assert state_digest(a) != state_digest(b)


def test_private_attributes_excluded_by_default():
    tree = extract_state(Courier())
    assert "_private" not in tree
    assert extract_state(Courier(), policy=ExtractionPolicy(include_private=True))["_private"]


def test_denylisted_attribute_names_are_removed():
    """`logger` is a denylisted name, so it does not appear at all."""
    assert "logger" not in extract_state(Courier())


def test_lock_typed_values_are_dropped_and_counted():
    """A lock reaching the walker under any other name is dropped, not digested."""
    stats = ExtractionStats()
    tree = extract_state({"guard": threading.RLock()}, stats=stats)
    assert "__dropped__" in tree["guard"]
    assert sum(stats.dropped_types.values()) == 1


def test_deque_treated_as_sequence_not_opaque():
    """A deque holds real state; digesting it as an opaque leaf would hide it."""
    a, b = Courier(), Courier()
    b.history.append(4)
    assert state_digest(a) != state_digest(b)
    assert extract_state(Courier())["history"] == [1, 2, 3]


def test_enum_encoded_by_name_and_value():
    tree = extract_state(Courier())
    assert tree["mode"]["__enum__"] == "Mode.SCOOTER"
    a, b = Courier(), Courier()
    b.mode = Mode.WALK
    assert state_digest(a) != state_digest(b)


def test_sets_digest_order_independently():
    assert digest_of(extract_state({"s": {3, 1, 2}})) == digest_of(extract_state({"s": {2, 3, 1}}))


def test_dict_key_order_does_not_matter():
    assert state_digest({"a": 1, "b": 2}) == state_digest({"b": 2, "a": 1})


def test_cycles_do_not_recurse_forever():
    node = Courier()
    node.self_ref = node
    tree = extract_state(node)
    assert tree["self_ref"] == {"__cycle__": "Courier"}


def test_non_finite_floats_are_tagged_not_rejected():
    """An inf deadline must show as a state difference, not crash the digest."""
    tree = extract_state({"deadline": float("inf"), "slack": float("nan")})
    assert tree["deadline"] == {"__float__": "inf"}
    assert tree["slack"] == {"__float__": "nan"}
    assert state_digest({"d": float("inf")}) != state_digest({"d": float("-inf")})


def test_depth_limit_is_reported_not_silent():
    deep = current = {}
    for _ in range(40):
        current["next"] = {}
        current = current["next"]
    stats = ExtractionStats()
    extract_state(deep, stats=stats)
    assert stats.depth_limited > 0


def test_long_sequences_truncate_visibly():
    stats = ExtractionStats()
    policy = ExtractionPolicy(max_sequence=10)
    extract_state({"xs": list(range(50))}, policy=policy, stats=stats)
    assert stats.truncated_sequences == 1


def test_documented_exclusion_removes_field_and_is_reportable():
    policy = ExtractionPolicy(documented_exclusions={"x": "wall-clock, never read"})
    assert "x" not in extract_state(Courier(), policy=policy)
    assert policy.exclusion_report() == [{"attribute": "x", "reason": "wall-clock, never read"}]


def test_exclusion_actually_masks_a_difference():
    a, b = Courier(), Courier()
    b.x = 999.0
    policy = ExtractionPolicy(documented_exclusions={"x": "excluded"})
    assert state_digest(a, policy=policy) == state_digest(b, policy=policy)
    assert state_digest(a) != state_digest(b)


def test_diff_state_names_the_differing_path():
    a, b = extract_state(Courier()), extract_state(Courier())
    b["orders"][0]["earnings"] = 99.0
    diffs = diff_state(a, b)
    assert any("orders[0].earnings" in d for d in diffs)
    assert any("12.5" in d and "99.0" in d for d in diffs)


def test_diff_state_reports_length_and_missing_keys():
    assert any("length" in d for d in diff_state({"xs": [1, 2]}, {"xs": [1]}))
    assert any("missing on right" in d for d in diff_state({"a": 1, "b": 2}, {"a": 1}))


def test_diff_state_empty_for_equal_trees():
    assert diff_state(extract_state(Courier()), extract_state(Courier())) == []


def test_bytes_digested_by_content():
    assert state_digest({"b": b"abc"}) == state_digest({"b": b"abc"})
    assert state_digest({"b": b"abc"}) != state_digest({"b": b"abd"})
