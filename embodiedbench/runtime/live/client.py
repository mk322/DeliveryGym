"""One HTTP client per UE render instance.

Deliberately thin. The client's whole job is to speak nav-render/v0 to one
``base_url`` and to translate the two ways that can fail into a taxonomy the
pool can act on:

* the *service* answered with an error -- a non-200 carrying
  ``{"error": {"code", "message"}}`` -- which becomes the exception class for
  that code, so a caller can tell "you sent garbage" (its own bug, do not
  retry) from "the engine is down" (the instance's problem, fail over);
* the *transport* failed -- connection refused, timeout -- which becomes
  ``ServiceUnreachable`` after exactly one reconnect attempt.

One reconnect and no more, on purpose: retries hide a dying instance from the
pool, and the pool's quarantine is the mechanism that is supposed to see it.
A client that retried five times would turn "instance ue-2 is dead" into
"renders are mysteriously slow", which is the harder bug to find.

stdlib urllib only. The trainer imports this in every rollout worker, and a
requests/httpx dependency for two endpoints is a supply chain for a GET.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from .protocol import (
    PROTOCOL,
    EpisodeEndRequest,
    EpisodeRequest,
    EpisodeResponse,
    Healthz,
    ObserveRequest,
    ObserveViewsRequest,
    ObserveViewsResponse,
    ProtocolViolation,
    RenderBatch,
    RenderResponse,
    RenderResult,
    WalkPixelRequest,
    WalkPixelResponse,
    WalkRequest,
    WalkResponse,
    WireError,
)

# Long enough for a cold instance to settle a big batch, short enough that a
# hung engine is a failure rather than a stall. Overridable per client.
DEFAULT_RENDER_TIMEOUT_S = 120.0
DEFAULT_HEALTH_TIMEOUT_S = 5.0


class RenderServiceError(RuntimeError):
    """Any failure talking to a render service. ``code`` says which."""

    code = "unknown"


class BadRequestError(RenderServiceError):
    """The service rejected the request as malformed. This is the caller's
    bug; failing over to another instance would send the same garbage."""

    code = "bad_request"


class EngineDownError(RenderServiceError):
    """The service is up but its UE instance is not."""

    code = "engine_down"


class MapMismatchError(RenderServiceError):
    """The instance is serving a different map than the request assumes.

    Not retryable anywhere: a pool whose endpoints file mixes maps is
    misconfigured, and rendering Paris poses against another city would
    produce frames that look plausible and mean nothing.
    """

    code = "map_mismatch"


class RenderFailedError(RenderServiceError):
    """The whole batch failed inside the engine. (A *single* bad item is not
    this -- it comes back as a ``failed`` result in a 200 response.)"""

    code = "render_failed"


class ServiceBusy(RenderServiceError):
    """The service refused the batch under load. Another instance may not.

    Busy is TRANSIENT by spec (section 3, normative): it is a load signal, not
    a health verdict. Callers must not count it as a quarantine strike and
    must not degrade an episode over it -- the pool fails over and backs off,
    and an env that still sees this skips the batch, leaving the cache miss
    for the next lookup to retry.
    """

    code = "busy"


class ServiceUnreachable(RenderServiceError):
    """No HTTP conversation happened at all, even after one reconnect."""

    code = "unreachable"


_BY_CODE: dict[str, type[RenderServiceError]] = {
    cls.code: cls
    for cls in (BadRequestError, EngineDownError, MapMismatchError,
                RenderFailedError, ServiceBusy)
}


def error_for(code: str, message: str) -> RenderServiceError:
    """The exception for a wire error code. Unknown codes stay errors --
    a service speaking codes this client does not know is a version skew,
    not a success."""
    cls = _BY_CODE.get(code, RenderServiceError)
    out = cls(f"{code}: {message}")
    if cls is RenderServiceError:
        out.code = code  # keep the wire's own word for the report
    return out


class UERenderClient:
    """nav-render/v0 over HTTP against one instance's ``base_url``."""

    def __init__(
        self,
        base_url: str,
        *,
        instance_id: str = "",
        render_timeout_s: float = DEFAULT_RENDER_TIMEOUT_S,
        health_timeout_s: float = DEFAULT_HEALTH_TIMEOUT_S,
    ):
        self.base_url = base_url.rstrip("/")
        self.instance_id = instance_id or self.base_url
        self.render_timeout_s = float(render_timeout_s)
        self.health_timeout_s = float(health_timeout_s)

    # ── the Track A endpoints ────────────────────────────────────────────────

    def healthz(self) -> Healthz:
        data = self._request("GET", "/healthz", timeout_s=self.health_timeout_s)
        return Healthz.from_dict(data)

    def render(self, batch: RenderBatch) -> tuple[RenderResult, ...]:
        data = self._request("POST", "/render", body=batch.to_dict(),
                             timeout_s=self.render_timeout_s)
        return RenderResponse.from_dict(data).results

    # ── the Track B endpoints (spec 3b, stateful) ────────────────────────────
    #
    # Same taxonomy, same plumbing: a stateful endpoint that answers busy --
    # someone else's episode holds the instance -- raises the same ServiceBusy
    # a saturated /render does, and the caller decides what an episode does
    # about it. The walk timeout on the wire is *sim* time; the HTTP timeout
    # here stays the render timeout, because a lockstep engine faster than
    # wall clock finishes a 120 s walk well inside it and a hung engine should
    # be a failure, not a stall.

    def episode(self, request: EpisodeRequest) -> EpisodeResponse:
        data = self._request("POST", "/episode", body=request.to_dict(),
                             timeout_s=self.render_timeout_s)
        return EpisodeResponse.from_dict(data)

    def walk(self, request: WalkRequest) -> WalkResponse:
        # NOT retried. Every other endpoint here is idempotent by contract
        # -- /render is self-contained, /observe is a read, /episode and
        # /episode_end are declared idempotent for the same id -- but a walk
        # MOVES the pawn. A transport timeout on a walk the engine actually
        # completed, retried, finds the pawn already standing on the target
        # and answers arrived with ticks=0 and sim_seconds=0.0. The hop then
        # costs nothing: the courier travelled for free and the clock never
        # heard about it. Surfacing the transport failure is honest; a silent
        # free hop is not.
        data = self._request("POST", "/walk", body=request.to_dict(),
                             retry=False,
                             timeout_s=self.render_timeout_s)
        return WalkResponse.from_dict(data)

    def walk_pixel(self, request: WalkPixelRequest) -> WalkPixelResponse:
        # NOT retried, for the same reason as ``walk``: a transport timeout
        # on a call that actually moved the pawn must not be replayed against
        # a fresh spawn.
        data = self._request("POST", "/walk_pixel", body=request.to_dict(),
                             retry=False,
                             timeout_s=self.render_timeout_s)
        return WalkPixelResponse.from_dict(data)

    def observe(self, request: ObserveRequest) -> RenderResult:
        data = self._request("POST", "/observe", body=request.to_dict(),
                             timeout_s=self.render_timeout_s)
        return RenderResult.from_dict(data)

    def observe_views(self, request: ObserveViewsRequest) -> ObserveViewsResponse:
        data = self._request("POST", "/observe_views", body=request.to_dict(),
                             timeout_s=self.render_timeout_s)
        return ObserveViewsResponse.from_dict(data)

    def episode_end(self, request: EpisodeEndRequest) -> bool:
        data = self._request("POST", "/episode_end", body=request.to_dict(),
                             timeout_s=self.render_timeout_s)
        return bool(data.get("ok"))

    # ── plumbing ─────────────────────────────────────────────────────────────

    def _request(self, method: str, path: str, *, body: dict[str, Any] | None = None,
                 timeout_s: float, retry: bool = True) -> dict[str, Any]:
        payload = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + path, data=payload, method=method,
            headers={"Content-Type": "application/json"} if payload else {},
        )
        last: Exception | None = None
        # Two passes: the original attempt and one reconnect. A service
        # restarting between batches produces exactly one refused connection,
        # and that one is not worth a failover; a second is.
        for _ in range(2 if retry else 1):
            try:
                with urllib.request.urlopen(request, timeout=timeout_s) as response:
                    return self._parse(response.read())
            except urllib.error.HTTPError as error:
                # The service answered; this is a protocol error, not a
                # transport one, and a reconnect would just be told again.
                raise self._wire_error(error) from None
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as error:
                last = error
        raise ServiceUnreachable(
            f"{self.instance_id}: {self.base_url}{path} unreachable"
            + (f" after one reconnect attempt ({last})" if retry
               else f" ({last}); not retried -- this endpoint is not idempotent"))

    @staticmethod
    def _parse(raw: bytes) -> dict[str, Any]:
        try:
            data = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as error:
            raise ProtocolViolation(f"response is not JSON: {error}") from None
        if not isinstance(data, dict):
            raise ProtocolViolation(f"response is {type(data).__name__}, not an object")
        return data

    def _wire_error(self, error: urllib.error.HTTPError) -> RenderServiceError:
        try:
            wire = WireError.from_dict(self._parse(error.read()))
        except (ProtocolViolation, OSError):
            if error.code == 503:
                # 503 means busy even when the body is not parseable -- a
                # saturated service (or a proxy in front of it) may not manage
                # a spec-shaped body, and treating its overload as a generic
                # failure is exactly the mis-map the busy taxonomy exists to
                # prevent.
                return ServiceBusy(
                    f"{self.instance_id}: HTTP 503 without a {PROTOCOL} error "
                    "body; treating as busy")
            return RenderServiceError(
                f"{self.instance_id}: HTTP {error.code} with a body that is "
                f"not a {PROTOCOL} error")
        return error_for(wire.code, wire.message)
