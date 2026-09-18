"""A fleet of N must actually be used by N workers.

The bug this pins had no symptom of its own. Six instances reported healthy,
six GPUs were spent, and one instance did all the work while the other five sat
idle -- because a Track B lease picks the least-loaded member, load counters
live in the worker process, and a trainer runs its env workers as separate Ray
actor processes. Every pool is cold at the moment it picks, so "least loaded"
is a tie among all of them, and a tie-break on instance id resolves the same
way in every process.

There is no cross-process coordination available here and none is wanted: the
statefile has a single writer by design, and a lock held across worker
processes is its own incident. Rotating the tie-break by pid is enough --
deterministic inside a worker, spread between them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from embodiedbench.runtime.live.pool import RenderPool


class _Client:
    """Stands in for UERenderClient: identity is all these tests need."""

    def __init__(self, base_url: str, *, instance_id: str = "", **_: object):
        self.base_url = base_url
        self.instance_id = instance_id or base_url

    def healthz(self):  # pragma: no cover - never reached in these tests
        raise AssertionError("no health probe expected")


@pytest.fixture()
def endpoints(tmp_path: Path) -> Path:
    path = tmp_path / "endpoints.json"
    path.write_text(json.dumps({
        "version": 0,
        "instances": [
            {"id": f"ue-{i}", "base_url": f"http://127.0.0.1:{18800 + i}",
             "map_name": "citycore-paris"}
            for i in range(6)
        ],
    }))
    return path


def _pool(endpoints: Path, pid: int, monkeypatch) -> RenderPool:
    monkeypatch.setattr("embodiedbench.runtime.live.pool.os.getpid",
                        lambda: pid)
    return RenderPool(endpoints, client_factory=_Client)


def test_cold_pools_in_different_processes_lease_different_instances(
        endpoints, monkeypatch):
    """Six workers, six instances, six different leases.

    Before the rotation this returned ue-0 six times, which is the whole
    failure: five instances never receive an episode, and the only externally
    visible consequence is that the extra GPUs bought nothing.
    """
    leased = []
    for pid in range(6):
        pool = _pool(endpoints, pid, monkeypatch)
        with pool.lease_embodied(f"episode-{pid}") as client:
            leased.append(client.instance_id)
    assert len(set(leased)) == 6, f"cold pools collided on {leased}"


def test_a_lease_can_skip_an_instance_that_said_busy(endpoints, monkeypatch):
    """An instance is the authority on whether it is free; the pool is not.

    A lease is a local belief. When the instance itself answers busy, the
    caller must be able to go elsewhere rather than retry the same one --
    otherwise a process that guessed wrong waits out its whole budget against
    an instance that is working for somebody else.
    """
    pool = _pool(endpoints, 0, monkeypatch)
    with pool.lease_embodied("first") as first:
        pass
    with pool.lease_embodied("second", exclude={first.instance_id}) as second:
        assert second.instance_id != first.instance_id


def test_the_rotation_is_stable_within_one_process(endpoints, monkeypatch):
    """Same worker, same choice: reproducibility inside a process is the half
    of this that must NOT be traded away for spread between them."""
    pool_a = _pool(endpoints, 3, monkeypatch)
    with pool_a.lease_embodied("a") as client_a:
        first = client_a.instance_id
    pool_b = _pool(endpoints, 3, monkeypatch)
    with pool_b.lease_embodied("b") as client_b:
        assert client_b.instance_id == first
