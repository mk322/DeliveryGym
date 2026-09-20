"""Schema foundation: versioning, strictness, typed errors, and the registry.

Three decisions here shape every contract in this package.

**Unknown fields are rejected.** the design plan M1 requires unknown-field tests to
pass for every contract. A silently ignored field is how a producer and a
consumer come to disagree about what an episode meant, which is exactly the
drift design plan §7.2's conformance suite exists to catch. ``extra="forbid"``
makes that a load-time error instead of a silent divergence.

**Versions are compared, not just carried.** Each model declares a
``SCHEMA_ID`` and ``SCHEMA_VERSION``. Loading a payload whose major version
differs raises ``VersionError``, because design plan §5.4 defines major as "changed
transitions, action meanings, coordinate frames, task semantics, or scores" —
a payload we cannot interpret. A newer minor is accepted, since minor is
"backward-compatible fields or new optional capabilities".

**Errors are typed.** the design plan M1 requires invalid versions, coordinate frames,
capabilities, and action payloads to "fail with typed errors". A caller must be
able to distinguish "you sent an unsupported frame" from "you sent malformed
JSON" without string-matching a message.
"""

from __future__ import annotations

import re
from typing import Any, ClassVar, Iterator

from pydantic import BaseModel, ConfigDict, ValidationError

from embodiedbench.artifacts.hashing import canonical_json, sha256_bytes

_SEMVER = re.compile(r"^(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)$")


# ─────────────────────────────────────────────────────────────────────────────
# Typed errors
# ─────────────────────────────────────────────────────────────────────────────


class SchemaError(Exception):
    """Base for every schema-layer failure."""


class VersionError(SchemaError):
    """A payload's schema id or major version is not one we can interpret."""

    def __init__(self, expected_id: str, expected: str, observed_id: str, observed: str):
        self.expected_id = expected_id
        self.expected = expected
        self.observed_id = observed_id
        self.observed = observed
        super().__init__(
            f"cannot load {observed_id!r} v{observed} as {expected_id!r} v{expected}"
        )


class FrameError(SchemaError):
    """A coordinate frame is unknown, or a transform crosses frames illegally."""


class CapabilityError(SchemaError):
    """An action or option was requested that the runtime/embodiment does not advertise."""


class PayloadError(SchemaError):
    """A payload failed structural validation (wrong type, unknown field, bad value)."""

    def __init__(self, model: str, errors: Any):
        self.model = model
        self.errors = errors
        super().__init__(f"{model}: {errors}")


# ─────────────────────────────────────────────────────────────────────────────
# Versions
# ─────────────────────────────────────────────────────────────────────────────


class SchemaVersion:
    """A semantic version with design plan §5.4's compatibility rule."""

    __slots__ = ("major", "minor", "patch")

    def __init__(self, major: int, minor: int, patch: int):
        self.major, self.minor, self.patch = major, minor, patch

    @classmethod
    def parse(cls, text: str) -> "SchemaVersion":
        match = _SEMVER.match(text or "")
        if not match:
            raise SchemaError(f"not a semantic version: {text!r}")
        return cls(int(match["major"]), int(match["minor"]), int(match["patch"]))

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SchemaVersion) and str(self) == str(other)

    def __hash__(self) -> int:
        return hash(str(self))

    def can_read(self, payload: "SchemaVersion") -> bool:
        """Whether a reader at this version can interpret ``payload``.

        Major must match exactly. While the whole package is ``0.x``, the design plan M1
        says there is no backward-compatibility promise, so a differing *minor*
        is also rejected at major 0 — pinning exact revisions is the stated
        contract, and silently accepting 0.2 data into a 0.1 reader would break
        it. From 1.0 on, a newer minor is readable.
        """
        if self.major != payload.major:
            return False
        if self.major == 0:
            return self.minor == payload.minor
        return payload.minor <= self.minor


# ─────────────────────────────────────────────────────────────────────────────
# Base model
# ─────────────────────────────────────────────────────────────────────────────

_REGISTRY: dict[str, type["SchemaModel"]] = {}


class SchemaModel(BaseModel):
    """Base for every versioned protocol object."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=False,
        validate_assignment=True,
        str_strip_whitespace=True,
        arbitrary_types_allowed=False,
    )

    SCHEMA_ID: ClassVar[str] = "embodiedbench/unnamed"
    SCHEMA_VERSION: ClassVar[str] = "0.1.0"
    # Whether the serialized form carries its schema id/version. Envelope-level
    # artifacts do; small value objects nested inside them do not, so the wire
    # form does not repeat a version on every vector.
    VERSIONED_ENVELOPE: ClassVar[bool] = False

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls.SCHEMA_ID != "embodiedbench/unnamed":
            existing = _REGISTRY.get(cls.SCHEMA_ID)
            if existing is not None and existing is not cls:
                raise SchemaError(
                    f"duplicate SCHEMA_ID {cls.SCHEMA_ID!r}: {existing.__name__} and {cls.__name__}"
                )
            _REGISTRY[cls.SCHEMA_ID] = cls

    # ── serialization ────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize, adding schema id and version for envelope-level artifacts."""
        data = self.model_dump(mode="json", exclude_none=True)
        if self.VERSIONED_ENVELOPE:
            return {"schema": self.SCHEMA_ID, "schema_version": self.SCHEMA_VERSION, **data}
        return data

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SchemaModel":
        """Validate a payload, checking schema identity before structure.

        Version is checked first so a payload from an incompatible revision
        reports ``VersionError`` rather than a pile of field errors caused by a
        rename we already know about.
        """
        if not isinstance(payload, dict):
            raise PayloadError(cls.__name__, f"expected an object, got {type(payload).__name__}")
        body = dict(payload)
        if cls.VERSIONED_ENVELOPE:
            observed_id = body.pop("schema", None)
            observed_version = body.pop("schema_version", None)
            if observed_id is None or observed_version is None:
                raise VersionError(
                    cls.SCHEMA_ID, cls.SCHEMA_VERSION, str(observed_id), str(observed_version)
                )
            if observed_id != cls.SCHEMA_ID:
                raise VersionError(
                    cls.SCHEMA_ID, cls.SCHEMA_VERSION, observed_id, str(observed_version)
                )
            reader = SchemaVersion.parse(cls.SCHEMA_VERSION)
            try:
                payload_version = SchemaVersion.parse(str(observed_version))
            except SchemaError:
                raise VersionError(
                    cls.SCHEMA_ID, cls.SCHEMA_VERSION, observed_id, str(observed_version)
                ) from None
            if not reader.can_read(payload_version):
                raise VersionError(
                    cls.SCHEMA_ID, cls.SCHEMA_VERSION, observed_id, str(payload_version)
                )
        try:
            return cls.model_validate(body)
        except ValidationError as exc:
            raise PayloadError(cls.__name__, exc.errors()) from exc

    # ── identity ─────────────────────────────────────────────────────────────

    def content_hash(self) -> str:
        """sha256 of the canonical serialization, for artifact pinning."""
        return sha256_bytes(canonical_json(self.to_dict()))

    def round_trip(self) -> "SchemaModel":
        return type(self).from_dict(self.to_dict())


def schema_registry() -> dict[str, type[SchemaModel]]:
    """Every registered schema id, so tests can enumerate rather than list."""
    return dict(_REGISTRY)


def versioned_envelopes() -> Iterator[type[SchemaModel]]:
    for model in _REGISTRY.values():
        if model.VERSIONED_ENVELOPE:
            yield model
