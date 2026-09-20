"""Many UE instances behind one ``render``: least-loaded dispatch, quarantine.

The multiplexing decision comes from the spec (section 3): renders are
self-contained and the service restores the level between requests, so there
is nothing to lease per *render* episode. A batch goes to whichever healthy
instance has the least in flight, ties broken round-robin, and more envs than
instances is the normal case, not a degraded one. The one exception is
Track B (spec 3b): an *embodied* episode is stateful on its instance, so
``lease_embodied`` takes an instance out of render dispatch exclusively for
the episode's duration.

Failure handling is the spec's two-strikes rule (section 5): an instance
failing health twice in a row is quarantined and readmitted on a successful
probe. A transport failure during a render counts as a strike too -- it is
the same evidence a probe would have gathered, arriving earlier -- and the
batch fails over to another instance within the same call, so a dying
instance costs latency, not an episode. The pool never kills a process; the
fleet owns lifecycle, and the nav side's whole authority over an instance is
declining to send it work.

Quarantine is persisted to a small statefile beside the endpoints file, so a
restarted trainer does not spend its first batches rediscovering a dead
instance. **No file lock, and that is a documented decision, not an
oversight**: today one trainer process runs per host (verl's env workers
share this pool in-process), so the statefile has a single writer. The write
is atomic (temp file + rename) so a reader never sees a torn file; if two
trainer processes ever share a host, the worst case is one relearning a
quarantine the other already knew, which costs two probes. A lock buys
nothing until then, and file locks held across NFS are their own incident.
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

from .client import (
    RenderServiceError,
    ServiceBusy,
    ServiceUnreachable,
    UERenderClient,
)
from .protocol import RenderBatch, RenderResult

# The env var the endpoints file is found under when no explicit path is given.
ENDPOINTS_ENV = "EB_UE_ENDPOINTS"
# Strikes in a row before an instance stops being offered work.
QUARANTINE_STRIKES = 2
# When every admitted instance answers busy, wait this long between retry
# sweeps -- up to three retries, then the batch is given up as ServiceBusy.
# Busy is a load signal, not a health verdict (spec section 3, normative): it
# is never a strike, so a saturated fleet stays fully admitted throughout.
BUSY_BACKOFF_S = (0.5, 1.0, 2.0)


class NoHealthyInstance(RenderServiceError):
    """Every instance is quarantined or failed for this batch.

    Still a ``RenderServiceError``, so a caller that degrades to album mode on
    "the render backend is unavailable" needs exactly one except clause.
    """

    code = "no_healthy_instance"


class EndpointsError(ValueError):
    """The endpoints file is missing or malformed. Refused loudly at
    construction: a pool that silently starts empty renders nothing and looks
    like a network problem."""


@dataclass
class _Member:
    """One instance and everything the pool believes about it."""

    id: str
    base_url: str
    map_name: str
    gpu_uuid: str
    client: UERenderClient
    strikes: int = 0
    quarantined: bool = False
    in_flight: int = 0
    dispatched: int = 0     # lifetime batches, the round-robin tiebreak
    quarantined_at: float = 0.0
    # The embodied episodes holding seats on this instance. Track B endpoints
    # are stateful, and an instance serves ``seats`` couriers at once; a member
    # with ANY lease is withdrawn from /render dispatch entirely, because the
    # service answers /render busy while any courier is alive. In-process state
    # only: leases die with the process that took them, and a fleet restart
    # clears the service side anyway.
    leases: set[str] = field(default_factory=set)
    # How many couriers this instance will carry, straight from the fleet's
    # endpoints.json (``--max-episodes``). Defaults to 1, so an endpoints file
    # written before seats existed keeps the old one-episode-per-instance shape.
    seats: int = 1

    @property
    def leased_to(self) -> str | None:
        """The historical single-lease view, kept for state dumps and tests."""
        return next(iter(sorted(self.leases)), None)

    def state(self) -> dict[str, Any]:
        return {"id": self.id, "base_url": self.base_url,
                "quarantined": self.quarantined, "strikes": self.strikes,
                "in_flight": self.in_flight, "dispatched": self.dispatched,
                "leased_to": self.leased_to,
                "leases": sorted(self.leases), "seats": self.seats}


_SHARED_POOLS: dict[str, "RenderPool"] = {}
_SHARED_LOCK = threading.Lock()


def shared_pool(endpoints_path: str | Path | None = None, **kwargs: Any) -> "RenderPool":
    """One pool per endpoints file per process.

    Track B leases are exclusive per instance, and the bookkeeping that makes
    them exclusive lives in the pool object. A trainer that builds a fresh
    pool per environment therefore has as many private views of the fleet as
    it has environments, and two of them will hand the same instance to two
    episodes -- which the service answers with 503 busy. Measured on the development workstation:
    four concurrent GRPO episodes, one Paris instance, ServiceBusy killed the
    run at the first rollout.
    """
    key = str(endpoints_path or os.environ.get(ENDPOINTS_ENV) or "")
    with _SHARED_LOCK:
        pool = _SHARED_POOLS.get(key)
        if pool is None:
            pool = RenderPool(endpoints_path, **kwargs)
            _SHARED_POOLS[key] = pool
        return pool


class RenderPool:
    """Loads endpoints.json, dispatches batches, quarantines the dying."""

    def __init__(
        self,
        endpoints_path: str | Path | None = None,
        *,
        client_factory: Callable[..., UERenderClient] = UERenderClient,
        render_timeout_s: float | None = None,
        lease_timeout_s: float = 600.0,
        lease_poll_s: float = 2.0,
    ):
        #: How long an embodied episode waits for a free instance before the
        #: fleet is declared oversubscribed-beyond-patience. A trainer running
        #: more concurrent episodes than instances parks here by design.
        self.lease_timeout_s = float(lease_timeout_s)
        self.lease_poll_s = float(lease_poll_s)
        #: Leases are handed out from several env threads in one trainer
        #: process; pick-and-mark must be atomic or two episodes take the
        #: same instance and the second meets 503 busy.
        self._lease_lock = threading.Lock()
        raw = endpoints_path or os.environ.get(ENDPOINTS_ENV)
        if not raw:
            raise EndpointsError(
                f"no endpoints file: pass a path or set {ENDPOINTS_ENV}. A pool "
                "with no instances is not a fallback, it is a typo.")
        self.endpoints_path = Path(raw)
        if not self.endpoints_path.exists():
            raise EndpointsError(f"endpoints file does not exist: {self.endpoints_path}")
        try:
            data = json.loads(self.endpoints_path.read_text())
        except ValueError as error:
            raise EndpointsError(
                f"endpoints file is not JSON: {self.endpoints_path} ({error})") from None
        instances = data.get("instances") or []
        if not instances:
            raise EndpointsError(f"endpoints file lists no instances: {self.endpoints_path}")

        kwargs: dict[str, Any] = {}
        if render_timeout_s is not None:
            kwargs["render_timeout_s"] = render_timeout_s
        self.members: list[_Member] = []
        for row in instances:
            base_url = str(row["base_url"])
            member_id = str(row.get("id") or base_url)
            try:
                seats = int(row.get("max_episodes") or 1)
            except (TypeError, ValueError):
                seats = 1
            self.members.append(_Member(
                id=member_id, base_url=base_url,
                map_name=str(row.get("map_name") or ""),
                gpu_uuid=str(row.get("gpu_uuid") or ""),
                client=client_factory(base_url, instance_id=member_id, **kwargs),
                seats=max(1, seats),
            ))

        self._index_of = {m.id: i for i, m in enumerate(self.members)}
        # Deterministic per process, arbitrary across processes -- which is
        # exactly the property needed: reproducible within a worker, spread
        # between them.
        self._rotation = os.getpid() % max(1, len(self.members))
        # Beside the endpoints file, as the spec says, so whoever looks at the
        # fleet's config sees the nav side's opinion of it in the same place.
        self.statefile = self.endpoints_path.with_name(
            self.endpoints_path.stem + ".quarantine.json")
        self._load_quarantine()
        # An instance attribute so a test can shrink the waits without
        # patching a module constant out from under a concurrent test.
        self.busy_backoff_s: tuple[float, ...] = BUSY_BACKOFF_S

    # ── dispatch ─────────────────────────────────────────────────────────────

    def render(self, batch: RenderBatch) -> tuple[RenderResult, ...]:
        """Send one batch to the best instance, failing over on transport death.

        Per-*item* failures come back in the results untouched -- they are the
        caller's information, not evidence against the instance. Only "no HTTP
        conversation happened" and "the engine is down" are strikes.

        ``busy`` is neither: a saturated instance is healthy, just loaded, so
        it costs no strike -- the batch fails over to the next healthy
        instance, and when every one of them is busy the pool sleeps through
        ``busy_backoff_s`` (0.5 s, 1 s, 2 s) and sweeps again, up to three
        retries. Still busy after that, the batch is given up as
        ``ServiceBusy`` for the caller to skip -- its cache miss remains, so
        the next lookup retries -- rather than mis-reported as fleet death.
        """
        tried: set[str] = set()
        busy: set[str] = set()
        backoffs = iter(self.busy_backoff_s)
        last_busy: ServiceBusy | None = None
        while True:
            member = self._pick(tried | busy)
            if member is None:
                if busy:
                    delay = next(backoffs, None)
                    if delay is None:
                        raise ServiceBusy(
                            f"every instance busy for batch of "
                            f"{len(batch.requests)} after "
                            f"{len(self.busy_backoff_s)} backed-off retries"
                        ) from last_busy
                    time.sleep(delay)
                    busy.clear()
                    continue
                if not self._readmit_one(tried):
                    raise NoHealthyInstance(
                        f"no healthy instance for batch of {len(batch.requests)} "
                        f"({len(self.members)} configured, "
                        f"{sum(m.quarantined for m in self.members)} quarantined)")
                continue
            member.in_flight += 1
            member.dispatched += 1
            try:
                results = member.client.render(batch)
            except ServiceBusy as error:
                # No strike: strikes are for the dead, and this instance just
                # answered. Leave its count where it was and move on.
                last_busy = error
                busy.add(member.id)
                continue
            except ServiceUnreachable:
                self._strike(member)
                tried.add(member.id)
                continue
            except RenderServiceError as error:
                if error.code == "engine_down":
                    self._strike(member)
                    tried.add(member.id)
                    continue
                raise
            finally:
                member.in_flight -= 1
            member.strikes = 0
            return results

    def _pick(self, exclude: set[str], *, for_lease: bool = False) -> _Member | None:
        """The best instance, by two different definitions of "free".

        For /render, free means *no courier at all*: the service answers
        /render busy while any embodied episode is alive, so one lease
        withdraws the whole instance from dispatch.

        For a lease, free means *a seat left*. Seats are ordered ahead of
        every other criterion so the pool fills instances breadth-first --
        four episodes over four instances rather than four stacked on the
        first. They cost nearly nothing to stack (96 pawns tick at 1.01x the
        cost of one) but they do share one GPU's render throughput, so
        spreading first is strictly better whenever there is somewhere to
        spread to.
        """
        if for_lease:
            candidates = [m for m in self.members
                          if not m.quarantined and len(m.leases) < m.seats
                          and m.id not in exclude]
            key = lambda m: (len(m.leases), m.in_flight, m.dispatched,  # noqa: E731
                             self._rotated(m))
        else:
            candidates = [m for m in self.members
                          if not m.quarantined and not m.leases
                          and m.id not in exclude]
            key = lambda m: (m.in_flight, m.dispatched, self._rotated(m))  # noqa: E731
        if not candidates:
            return None
        return min(candidates, key=key)

    def _rotated(self, member: _Member) -> int:
        """Tie-break position, rotated by process.

        Load counters are per-process, and a trainer runs its env workers in
        separate Ray actor processes. Every one of those pools is therefore
        COLD -- in_flight and dispatched are zero everywhere -- so a tie-break
        on ``id`` makes every process independently choose the same instance.
        Measured consequence with six instances up: one instance takes every
        episode, the other five never receive one, and the five extra GPUs buy
        nothing. Rotating the tie-break by pid spreads cold pools across the
        fleet without any cross-process coordination, which there is none of.
        """
        index = self._index_of[member.id]
        return (index - self._rotation) % len(self.members)

    # ── embodied leases (Track B) ────────────────────────────────────────────

    @contextlib.contextmanager
    def lease_embodied(self, episode_id: str,
                       exclude: set[str] | None = None) -> Iterator[UERenderClient]:
        """One instance, exclusively, for one embodied episode.

        Track B endpoints are stateful (spec 3b): one active embodied episode
        per instance, ``busy`` otherwise. So an embodied episode does not
        multiplex at request granularity the way renders do -- it takes the
        least-loaded healthy instance out of /render dispatch for its whole
        duration and gives it back on episode end or error, which the
        ``with`` block guarantees.

        The lease yields the member's *client*: the episode speaks to its one
        instance directly, and the pool's job shrinks to bookkeeping. A
        ``busy`` from an embodied endpoint follows the same no-strike rule as
        a busy render -- it reaches the caller as ``ServiceBusy`` without
        passing through the pool, so it cannot be counted against the
        instance's health by construction.
        """
        deadline = time.monotonic() + self.lease_timeout_s
        # Instances this episode already found busy: a lease is a local
        # belief about an instance, and the instance itself is the
        # authority. Being told 'busy' is how the belief gets corrected.
        skip = set(exclude or ())
        while True:
            with self._lease_lock:
                member = self._pick(skip, for_lease=True)
                if member is not None:
                    member.leases.add(episode_id)
            if member is None:
                # Same last resort as render: probe the quarantined before
                # declaring the fleet dead.
                with self._lease_lock:
                    self._readmit_one(skip)
                    member = self._pick(skip, for_lease=True)
                    if member is not None:
                        member.leases.add(episode_id)
            if member is not None:
                break
            leased = sum(len(m.leases) for m in self.members)
            # All healthy instances merely LEASED is the normal shape of a
            # trainer driving more concurrent episodes than the fleet has
            # instances -- wait for one to free rather than failing the
            # episode. A fleet with nothing leased and nothing pickable is
            # actually dead, and waiting would only delay the report.
            if leased == 0 or time.monotonic() >= deadline:
                raise NoHealthyInstance(
                    f"no seat free to lease for embodied episode {episode_id} "
                    f"({len(self.members)} instances configured, "
                    f"{sum(m.seats for m in self.members)} seats, "
                    f"{sum(m.quarantined for m in self.members)} quarantined, "
                    f"{leased} leased)")
            time.sleep(self.lease_poll_s)
        try:
            yield member.client
        except (ServiceUnreachable, RenderServiceError) as error:
            # The pool learns from a LEASED episode too.
            #
            # /render strikes an instance the moment it is unreachable, but a
            # lease hands the caller the member's client and shrinks the pool
            # to bookkeeping -- so an instance that died under an embodied
            # episode was struck nowhere, and the next lease picked it again.
            # Twice today a training run ended on `ServiceUnreachable: ue-0
            # /walk unreachable`, with the pool still holding the corpse as
            # its healthiest member.
            #
            # Only for the shapes that mean "this instance is gone". A busy is
            # not one (it just answered), and neither is a bad_request about
            # an episode id -- that is a live service disagreeing with us.
            if isinstance(error, ServiceUnreachable) or getattr(
                    error, "code", None) in ("engine_down", "render_failed"):
                self._strike(member)
            raise
        finally:
            with self._lease_lock:
                member.leases.discard(episode_id)

    # ── health ───────────────────────────────────────────────────────────────

    def check_health(self) -> dict[str, str]:
        """Probe every instance; apply strikes, quarantine, and readmission.

        Returns ``{instance_id: "ok" | "degraded" | "quarantined"}`` so a
        caller can log the fleet's shape in one line.
        """
        out: dict[str, str] = {}
        for member in self.members:
            if self._probe(member):
                out[member.id] = "ok"
            else:
                out[member.id] = "quarantined" if member.quarantined else "degraded"
        self._save_quarantine()
        return out

    def _probe(self, member: _Member) -> bool:
        try:
            member.client.healthz()
        except RenderServiceError:
            self._strike(member)
            return False
        member.strikes = 0
        if member.quarantined:
            member.quarantined = False
            member.quarantined_at = 0.0
        return True

    def _readmit_one(self, exclude: set[str]) -> bool:
        """When nothing is admitted, probe the quarantined before giving up.

        Readmit-on-probe is what makes quarantine a pause rather than a
        verdict: an instance the fleet restarted comes back the moment it
        answers, without anyone editing a file.
        """
        changed = False
        for member in self.members:
            if member.quarantined and member.id not in exclude:
                if self._probe(member):
                    changed = True
        if changed:
            self._save_quarantine()
        return changed

    def _strike(self, member: _Member) -> None:
        member.strikes += 1
        if member.strikes >= QUARANTINE_STRIKES and not member.quarantined:
            member.quarantined = True
            member.quarantined_at = time.time()
        self._save_quarantine()

    # ── the statefile ────────────────────────────────────────────────────────

    def _load_quarantine(self) -> None:
        if not self.statefile.exists():
            return
        try:
            data = json.loads(self.statefile.read_text())
        except ValueError:
            # A torn or hand-mangled statefile must not take the pool down;
            # the worst it recorded was pessimism, and probes rebuild that.
            return
        quarantined = set(data.get("quarantined") or [])
        strikes = data.get("strikes") or {}
        for member in self.members:
            if member.id in quarantined:
                member.quarantined = True
            member.strikes = int(strikes.get(member.id, 0))

    def _save_quarantine(self) -> None:
        payload = {
            "version": 0,
            "quarantined": sorted(m.id for m in self.members if m.quarantined),
            "strikes": {m.id: m.strikes for m in self.members if m.strikes},
        }
        # Atomic on POSIX: a reader sees the old file or the new one, never a
        # prefix. This is the whole of the "no lock needed for v0" story --
        # single writer per host, torn reads impossible by construction.
        temporary = self.statefile.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, self.statefile)

    # ── reporting ────────────────────────────────────────────────────────────

    def state(self) -> list[dict[str, Any]]:
        return [m.state() for m in self.members]
