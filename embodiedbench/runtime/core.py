"""The common runtime API (design plan §7, Contract B).

    class EmbodiedRuntime(Protocol):
        capabilities: RuntimeCapabilities
        def reset(self, instance: EpisodeSpec) -> tuple[Observation, ResetInfo]: ...
        def step(self, action: ActionEnvelope) -> StepResult: ...
        def close(self) -> None: ...

Snapshot/restore is an optional extension advertised through
``capabilities.supports_snapshot``, not a mandatory method — design plan §7 makes it
evaluator/harness-only in every v1 track and never agent-visible, so it lives on
a separate protocol that the agent-facing one does not inherit.

``authoritative_state_digest`` is not in the design plan's sketch but is required to
implement design plan §7.2: conformance compares "task state and inventory, economy
and constraint state, event sequence, reward components" across runtimes, and
that comparison needs one canonical answer per runtime rather than an ad-hoc
reach into each implementation's internals.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from embodiedbench.schemas.episode import EpisodeSpec
from embodiedbench.schemas.runtime import (
    ActionEnvelope,
    Observation,
    ResetInfo,
    RuntimeCapabilities,
    StepResult,
)


class RuntimeError_(Exception):
    """Base for runtime-layer failures. Named to avoid shadowing the builtin."""


class SnapshotUnsupported(RuntimeError_):
    """Raised when snapshot/restore is used on a runtime that does not offer it."""


class EpisodeNotStarted(RuntimeError_):
    """``step`` was called before ``reset``."""


class StepIndexMismatch(RuntimeError_):
    """An action arrived with the wrong step index.

    design plan §7.3 requires a monotonic step index and an idempotency key in the
    service transport. Enforcing monotonicity in-process too means a harness bug
    surfaces as an error rather than as a silently reordered trajectory.
    """


@runtime_checkable
class EmbodiedRuntime(Protocol):
    """What every runtime mode implements."""

    capabilities: RuntimeCapabilities

    def reset(self, instance: EpisodeSpec) -> tuple[Observation, ResetInfo]: ...

    def step(self, action: ActionEnvelope) -> StepResult: ...

    def close(self) -> None: ...

    def authoritative_state_digest(self) -> str:
        """Canonical digest of the state design plan §7.2 compares across runtimes."""
        ...


@runtime_checkable
class SnapshotCapable(Protocol):
    """Optional extension (design plan §7). Evaluator/harness-only in v1."""

    def snapshot(self) -> dict[str, Any]: ...

    def restore(self, snapshot: dict[str, Any]) -> None: ...
