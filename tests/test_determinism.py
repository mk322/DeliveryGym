"""Tests for the M0-F1 determinism patch.

M0-F1: the vendored Dijkstra used ``id(node)`` — a CPython memory address — as
its priority-queue tie-breaker, so which of several equal-cost routes it
returned depended on where objects happened to be allocated.

The tests that matter here are about the *key*, not about wiring: it must be
total, address-independent, and stable across processes. The end-to-end proof
that replay converges lives in the M0 replay gate, which needs the vendored
engine and is therefore not a unit test.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from embodiedbench.baseline.determinism import stable_node_key

REPO_ROOT = Path(__file__).resolve().parents[1]


class Vector:
    def __init__(self, x, y, z=0.0):
        self.x, self.y, self.z = x, y, z


class Node:
    def __init__(self, x, y, waypoint_id=None):
        self.position = Vector(x, y)
        if waypoint_id is not None:
            self.waypoint_id = waypoint_id


def test_key_is_independent_of_object_identity():
    """The whole point: two distinct objects at the same place sort the same."""
    assert stable_node_key(Node(1.0, 2.0, "int_1")) == stable_node_key(Node(1.0, 2.0, "int_1"))


def test_key_distinguishes_different_nodes():
    assert stable_node_key(Node(1.0, 2.0, "int_1")) != stable_node_key(Node(1.0, 2.0, "int_2"))
    assert stable_node_key(Node(1.0, 2.0)) != stable_node_key(Node(1.0, 3.0))


def test_key_is_totally_ordered_across_mixed_nodes():
    """Sorting must not raise: a missing waypoint id has to compare against a str."""
    nodes = [Node(3.0, 1.0), Node(1.0, 2.0, "int_9"), Node(2.0, 0.0, "dock_1"), Node(0.0, 0.0)]
    keys = sorted(stable_node_key(n) for n in nodes)
    assert len(keys) == 4
    assert keys == sorted(keys)


def test_key_ordering_does_not_depend_on_allocation_order():
    forward = [Node(float(i), 0.0, f"n_{i}") for i in range(20)]
    backward = [Node(float(i), 0.0, f"n_{i}") for i in reversed(range(20))]
    assert sorted(stable_node_key(n) for n in forward) == sorted(
        stable_node_key(n) for n in backward
    )


def test_key_tolerates_missing_position():
    class Bare:
        waypoint_id = "wp_1"

    assert stable_node_key(Bare()) == ("wp_1", ())


def test_key_quantizes_representation_noise():
    """Rounding must not merge genuinely distinct nodes."""
    assert stable_node_key(Node(1.0, 2.0)) == stable_node_key(Node(1.0 + 1e-9, 2.0))
    assert stable_node_key(Node(1.0, 2.0)) != stable_node_key(Node(1.01, 2.0))


def test_key_is_stable_across_processes():
    """A fix that only works under a fixed PYTHONHASHSEED would not be a fix."""
    code = (
        "from embodiedbench.baseline.determinism import stable_node_key\n"
        "class V:\n"
        "    def __init__(s,x,y,z): s.x,s.y,s.z=x,y,z\n"
        "class N:\n"
        "    def __init__(s): s.position=V(12.5,-3.25,0.0); s.waypoint_id='int_42'\n"
        "print(stable_node_key(N()))\n"
    )
    outputs = set()
    for _ in range(3):
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            check=True,
        )
        outputs.add(proc.stdout.strip())
    assert len(outputs) == 1, outputs


@pytest.mark.skipif(
    not (REPO_ROOT / "vendor" / "vagen").exists(),
    reason="vendored VAGEN checkout not present",
)
def test_patch_reports_what_it_replaced():
    from embodiedbench.baseline.determinism import apply_deterministic_patches

    patches = apply_deterministic_patches().to_dict()
    targets = {p["target"] for p in patches["patches"]}
    assert targets == {
        "vlm_delivery.base.graph.Graph.shortest_path_nodes",
        "vlm_delivery.base.graph.Graph.shortest_path_xy_to_node",
    }
    for patch in patches["patches"]:
        # Either it was applied, or it explains why it refused to be.
        assert patch["applied"] or patch.get("skipped_reason")
        assert len(patch["original_source_sha256"]) == 64
        assert patch["reason"]


@pytest.mark.skipif(
    not (REPO_ROOT / "vendor" / "vagen").exists(),
    reason="vendored VAGEN checkout not present",
)
def test_patch_is_idempotent():
    from embodiedbench.baseline.determinism import apply_deterministic_patches

    first = apply_deterministic_patches().to_dict()
    second = apply_deterministic_patches().to_dict()
    assert first == second
